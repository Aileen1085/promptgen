"""Capture genuine decoder logits for the explicitly authorized AMOS liver case.

Each method runs in a fresh process. Existing signed evaluators own every model
operation, prompt, threshold, AutoExpand decision, and full-volume metric.
No CT or GT voxels are exported. A partial output must be inspected manually.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
METRICS = ('dice', 'mean_slice_dice', 'iou', 'precision', 'recall', 'specificity',
           'pred_gt_ratio', 'mean_slice_nsd_1mm', 'mean_slice_nsd_2mm', 'mean_slice_nsd_3mm')
PARENT_SHA = '5e2bdbf2fa340c69b9d35bd079ed3d04332d95f9b1eea5871111c3ee55cf8777'
PROTOCOL_SHA = '0df2ca0724a5c8a86e2d28264875f1556f84f2cdaded53d41d5847055d78541a'
OWNER_SHA = 'dcf768949161021469138e02b5ba2894b09e30f98ae59caf928d27ec4cb89c67'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def path(value):
    p = Path(value)
    return p if p.is_absolute() else PROJECT / p


def new_json(filename, value):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def array(value):
    if hasattr(value, 'detach'):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value)


def continuous_logits(value):
    value = array(value)
    if value.ndim == 4 and value.shape[1] == 1:
        value = value[:, 0]
    if (value.ndim != 3 or value.dtype.kind != 'f' or not np.isfinite(value).all()
            or np.isin(value, [0., 1.]).all()):
        raise ValueError('expected true finite continuous floating ROI logits, not binary surrogate')
    return value.astype(np.float32, copy=True)


@contextmanager
def observe_v10(namespace):
    """Observe tuple[0] BEFORE the task substitutes binary metric logits."""
    observed = {}
    decode = namespace['_dynamic_adaptive_decode']
    score = namespace['restore_and_score_full_volume']
    def decoded(*args, **kwargs):
        result = decode(*args, **kwargs)
        observed.update(logits_roi=continuous_logits(result[0]),
                        roi_prediction=array(result[1]).astype(bool, copy=True),
                        relative_bounds=list(result[2]), threshold_audit=dict(result[4]),
                        patch_geometry=result[5])
        return result
    def scored(*args, **kwargs):
        result = score(*args, **kwargs)
        observed.update(prediction_dhw=array(result[1]).astype(np.uint8, copy=True),
                        frame_indices=np.asarray(kwargs['frame_indices'], np.int64).copy(),
                        spacing_dhw_mm=list(kwargs['spacing_dhw']))
        return result
    namespace['_dynamic_adaptive_decode'] = decoded
    namespace['restore_and_score_full_volume'] = scored
    try:
        yield observed
    finally:
        namespace['_dynamic_adaptive_decode'] = decode
        namespace['restore_and_score_full_volume'] = score


class NativeFrames:
    """Only the final AutoExpand stage; original forward/reverse last-write wins."""
    def __init__(self):
        self.frames = {}

    def begin(self):
        self.frames.clear()

    def record(self, frame, mask):
        values = array(mask)
        if values.ndim != 2 or values.dtype.kind != 'f' or not np.isfinite(values).all():
            raise ValueError('invalid propagated native floating mask logits')
        self.frames[int(frame)] = values.astype(np.float32, copy=True)

    def finish(self, prediction, left_top):
        if set(self.frames) != set(range(len(prediction))):
            raise ValueError('native capture missing true ROI frame')
        left, top = left_top
        height, width = prediction.shape[1:]
        logits = np.stack([self.frames[i][top:top+height, left:left+width]
                           for i in range(len(prediction))])
        if logits.shape != prediction.shape or not np.array_equal(logits > 0, prediction):
            raise ValueError('native saved-logit threshold/full prediction mismatch')
        return continuous_logits(logits)


@contextmanager
def observe_native(base, builder, assets):
    observed = {}; frames = NativeFrames(); model_hooks = []
    predict, overlay, build = base.predict_roi, base.overlay, builder.build_sam2_video_predictor
    def built(_config, checkpoint, **kwargs):
        model = build(assets['config'], checkpoint, **kwargs)
        if model.image_size != assets['image_size']:
            raise ValueError('native SAM2 image size mismatch')
        propagate = model.propagate_in_video
        def propagated(*args, **kw):
            for result in propagate(*args, **kw):
                frames.record(result[0], result[2][0, 0])
                yield result
        model_hooks.append((model, propagate))
        model.propagate_in_video = propagated
        return model
    def predicted(*args, **kwargs):
        frames.begin()
        result = predict(*args, **kwargs)
        observed.update(logits_roi=frames.finish(result[0], result[1]['square_pad_left_top']),
                        roi_prediction=array(result[0]).astype(bool, copy=True))
        return result
    def overlaid(_file, _ct, _target, _fg, pred):
        observed['prediction_dhw'] = np.asarray(pred, np.uint8).copy()
    base.predict_roi, base.overlay = predicted, overlaid
    builder.build_sam2_video_predictor = built
    try:
        yield observed
    finally:
        base.predict_roi, base.overlay = predict, overlay
        builder.build_sam2_video_predictor = build
        for model, propagate in reversed(model_hooks):
            model.propagate_in_video = propagate


def plane_coords(mask, plane, selected):
    coords = np.argwhere(mask)
    if plane == 'coronal':
        return np.column_stack((coords[:, 0], coords[:, 1], np.full(len(coords), selected))).astype(np.int64)
    if plane != 'sagittal':
        raise ValueError('unknown prompt plane')
    return np.column_stack((coords[:, 0], np.full(len(coords), selected), coords[:, 1])).astype(np.int64)


def native_points(stage):
    _, (h0, _), (w0, _) = stage['bounds_dhw']
    fg, bg = [], []
    for record in stage['point_provenance']:
        fg.extend((record['z'], y+h0, x+w0) for x, y in record['fg_sampled_xy'])
        bg.extend((record['z'], y+h0, x+w0) for y, x in record['bg_original_yx'])
    return np.asarray(fg, np.int64).reshape(-1, 3), np.asarray(bg, np.int64).reshape(-1, 3)


def place_roi(roi, shape, frame_indices, bounds):
    full = np.zeros(shape, np.uint8)
    (h0, h1), (w0, w1) = bounds[1:]
    indices = np.asarray(frame_indices, np.int64)
    if roi.shape != (len(indices), h1-h0, w1-w0):
        raise ValueError('ROI shape/mapping mismatch')
    full[indices, h0:h1, w0:w1] = roi
    return full


def restore_v10_prediction(roi, shape, frame_indices, geometry):
    # PyTorch nearest interpolation uses floor(out_index * input_size / output_size).
    def nearest(values, height, width):
        rows = np.arange(height) * values.shape[1] // height
        cols = np.arange(width) * values.shape[2] // width
        return values[:, rows[:, None], cols]
    if geometry is not None:
        left, right, top, bottom = geometry.bounds
        patch = nearest(roi, bottom-top, right-left)
        roi = np.zeros((len(roi), geometry.full_side, geometry.full_side), np.uint8)
        roi[:, top:bottom, left:right] = patch
    side = max(shape[1:]); top, left = (side-shape[1])//2, (side-shape[2])//2
    square = nearest(roi, side, side)
    raw = square[:, top:top+shape[1], left:left+shape[2]]
    return place_roi(raw, shape, frame_indices, ((0, shape[0]), (0, shape[1]), (0, shape[2])))


def validate_actual_inputs(selection, prompt_path, prompt_sha, ct_events):
    if (path(prompt_path).resolve() != path(selection['prompt_package_path']).resolve()
            or prompt_sha != selection['prompt_package_sha256']):
        raise ValueError('actual prompt package differs from selection')
    if not ct_events or any(event['mode'] != 'read_only_hit'
            or path(event['path']).resolve() != path(selection['ct_cache_path']).resolve()
            or path(event['image']).resolve() != path(selection['image_path']).resolve() for event in ct_events):
        raise ValueError('actual CT cache differs from selected readonly shared CT')


def compare_metrics(row, reference):
    differences = {}
    for name in METRICS:
        delta = abs(float(row[name])-float(reference[name]))
        if not np.isfinite(delta) or delta > 1e-6:
            raise ValueError('reference metric mismatch {}: {}'.format(name, delta))
        differences[name] = delta
    for name in ('mask_threshold_used', 'roi_initial_frames', 'roi_reserved_frames',
                 'roi_final_frames', 'roi_auto_expand_iterations'):
        if name in reference and row.get(name) != reference[name]:
            raise ValueError('reference protocol mismatch '+name)
    return differences


def save_capture(out, metadata, arrays):
    required = {'prediction_dhw', 'logits_roi', 'frame_indices', 'bounds_dhw', 'prompt_fg_dhw', 'prompt_bg_dhw'}
    allowed = required | {p+'_'+r+'_dhw' for p in ('coronal', 'sagittal') for r in ('fg', 'bg')}
    if not required <= set(arrays) or set(arrays)-allowed:
        raise ValueError('missing or unexpected tensor export (CT/GT prohibited)')
    continuous_logits(arrays['logits_roi'])
    out = Path(out)
    if (out/'capture.json').exists() or (out/'capture.npz').exists():
        raise FileExistsError('capture already exists; no overwrite or partial resume')
    with (out/'capture.npz').open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    info = dict(metadata, capture_npz_sha256=sha(out/'capture.npz'),
                arrays={k: dict(shape=list(v.shape), dtype=str(v.dtype), finite=bool(np.isfinite(v).all()))
                        for k, v in arrays.items()})
    new_json(out/'capture.json', info)
    return info


def resource_binding(gpu):
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.free',
                             '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
    rows = {int(p[0]): p for p in ([s.strip() for s in line.split(',')]
                                  for line in result.stdout.splitlines())}
    actual = rows[gpu]
    if int(actual[2]) < 30720:
        raise SystemExit(75)
    os.environ['CUDA_VISIBLE_DEVICES'] = actual[1]
    os.environ['V9_SHARED_CACHE_WRITE_POLICY'] = 'read_only'
    import torch
    uuid = 'GPU-'+str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')
    if uuid != actual[1] or torch.ones(1, device='cuda:0').item() != 1:
        raise ValueError('physical/CUDA UUID or allocation probe mismatch')
    torch.set_num_threads(2)
    return dict(gpu=gpu, gpu_uuid=uuid, actual_start_free_mib=int(actual[2]),
                minimum_free_mib=30720, cuda_probe=True, visible_cuda_ordinal=0,
                frame_batch=4, actual_frame_batch=4, pid=os.getpid(), time_unix=time.time())


def capture_v10(selection, out, resource):
    from tools import amos_prompt_ablation_eval as controller
    from tools.amos_prompt_ablation_native_manifest_eval import native_manifest_payload
    from tools import amos_prompt_reuse_owner_runtime as runtime
    from tools.amos_prompt_ablation_prompts import _validated_inputs
    if sha(PROJECT/'tools/amos_prompt_reuse_owner_runtime.py') != OWNER_SHA:
        raise ValueError('owner runtime SHA changed')
    parent = path(selection['parent_output']); protocol = read(parent/'protocol.json')
    if sha(parent/'protocol.json') != PROTOCOL_SHA or sha(parent/'merged_metrics.json') != PARENT_SHA:
        raise ValueError('signed parent source changed')
    for domain in ('sources', 'assets', 'training'):
        controller.verify_inventory(protocol[domain])
    task, ref = selection['task'], selection['ref']
    prepared = out/'prepared'
    new_json(prepared/'manifest.json', native_manifest_payload(tasks=[task], expected_task_count=1))
    new_json(prepared/'refs.json', dict(per_case_class=[ref]))
    job = dict(name='single_case_amos_liver_capture', case_id=task['case_id'], pending=[task], task_count=1,
               control=False, run=str(out), prepared=str(prepared), frame_batch=4, actual_frame_batch=4,
               gpu=resource['gpu'], gpu_uuid=resource['gpu_uuid'], minimum_free_mib=30720)
    paths = controller.runtime_paths(parent, out, protocol)
    session = runtime.initialize(job, paths)
    new_json(out/'runtime_input_audit.json', dict(paths=paths, actual=session.input_audit))
    with observe_v10(session.namespace) as observed:
        record = runtime.evaluate(session, task, 'fg_scribble_bg_scribble')
    _, package, groups, cache_path, cache_digest = session.raw_cache
    validate_actual_inputs(selection, cache_path, cache_digest, record['ct_cache_access'])
    shape, slices, fg, native, _points, _planes = _validated_inputs(package, groups)
    prompts = {p+'_'+r+'_dhw': plane_coords(fg[p] if r == 'fg' else native[p][1], p, slices[p])
               for p in ('coronal', 'sagittal') for r in ('fg', 'bg')}
    if (slices['coronal'], slices['sagittal']) != (selection['selected_coronal_w'], selection['selected_sagittal_h']):
        raise ValueError('selected v10 prompt plane changed')
    final = session.finish()
    if (final['epoch'], final['prompt_parameters_exact'], final['decoder_parameters_exact'],
            final['successful_native_completion']) != (490, 418, 212, True):
        raise ValueError('missing normal E490 completion proof')
    compare_metrics(record['row'], selection['v10_2_reference'])
    if record['row']['mask_threshold_used'] != .7:
        raise ValueError('selected v10 threshold changed')
    new_json(out/'restoration_final.json', final)
    new_json(out/'actual_record.json', record)
    indices = observed['frame_indices']
    if not np.array_equal(indices, record['roi']['final']):
        raise ValueError('observed v10 frame mapping changed')
    geometry = observed['patch_geometry']
    full = restore_v10_prediction(observed['roi_prediction'], shape, indices, geometry)
    if not np.array_equal(full, observed['prediction_dhw']):
        raise ValueError('v10 actual ROI/full-volume mapping mismatch')
    mapping = dict(grid='actual_decoder_output', axes='ZXY', native_output_shape=list(observed['logits_roi'].shape),
                   relative_reserved_bounds=observed['relative_bounds'], full_shape_dhw=list(shape),
                   final_frame_indices=indices.tolist(),
                   padding='centered square; pad_top=(side-H)//2; pad_left=(side-W)//2',
                   full_volume_postprocess='native threshold/connected processing; nearest resize to padded raw square then unpad',
                   outside_roi_logits='unpredicted; no fabricated logits',
                   raw_threshold_does_not_replace_native_postprocessed_mask=True)
    if geometry is not None:
        mapping['patch_geometry'] = dict(bounds_lrtb=list(geometry.bounds), full_side=int(geometry.full_side),
                                        input_side=int(geometry.input_side))
    bounds = np.array([[int(indices.min()), int(indices.max())+1], [0, shape[1]], [0, shape[2]]], np.int64)
    arrays = dict(prediction_dhw=observed['prediction_dhw'], logits_roi=observed['logits_roi'],
                  frame_indices=indices, bounds_dhw=bounds, **prompts,
                  prompt_fg_dhw=np.concatenate([prompts[p+'_fg_dhw'] for p in ('coronal', 'sagittal')]),
                  prompt_bg_dhw=np.concatenate([prompts[p+'_bg_dhw'] for p in ('coronal', 'sagittal')]))
    return arrays, dict(metrics={m: record['row'][m] for m in METRICS}, threshold_probability=.7,
                        logit_grid_mapping=mapping, model_proofs=dict(record=record, normal_finish=final),
                        actual_prompt_package=dict(path=cache_path, sha256=cache_digest))


def capture_native(selection, out, resource):
    from tools import native_ct13_eval as native
    assets = native.asset_spec('sam2'); code = PROJECT/assets['code_root']
    sys.path.insert(0, str(code))
    import sam2.build_sam as builder
    if not Path(builder.__file__).resolve().is_relative_to(code.resolve()):
        raise ValueError('wrong native SAM2 source')
    base = native.base_module(); base.load_task = native.CaseLoader(base)
    task = dict(selection['task'], prompt_cache_name=selection['ref']['prompt_cache_name'])
    job = dict(tasks=[task], output=str(out), memory_mode='on', dense_prior='none',
               background_mode='scribble', background_cache_dir=str(out/'background_metadata'),
               max_points=2, max_background_points=2, point_spacing_mm=3.)
    new_json(out/'job.json', job)
    files = [PROJECT/'tools/native_ct13_eval.py', PROJECT/'tools/native_sam2_physical_eval.py',
             PROJECT/'infer/native_sam2_physical_prompts.py', PROJECT/'utils/metrics.py',
             Path(builder.__file__), code/'sam2/sam2_video_predictor.py', code/'sam2'/assets['config'],
             PROJECT/assets['checkpoint']]
    before = {str(p): sha(p) for p in files}
    new_json(out/'native_assets.json', dict(assets=assets, files=before, protocol=native.PROTOCOL))
    args = base.parse_args(['--job', str(out/'job.json'), '--checkpoint', assets['checkpoint'],
                            '--expected-uuid', resource['gpu_uuid']])
    with observe_native(base, builder, assets) as observed:
        base.run(args)
    rows = read(out/'complete_metrics.json')['per_case_class']
    native.validate_rows(rows, [task])
    audit = read(out/'exact_prompt_audit.json')['per_case_class'][0]
    if audit['promptgen_loaded'] or not audit['complete_frame_prediction'] or audit['dense_prior'] != 'none':
        raise ValueError('native model/prediction proof mismatch')
    if any(stage['effective_memory_slots'] != 7 for stage in audit['stages']):
        raise ValueError('native memory settings changed')
    package = native.CACHE/'metadata/shared_adaptive_prompt_v1/amos'/task['prompt_cache_name']
    actual_ct = native.CACHE/'_case_images_v1'/audit['case_image_name']
    validate_actual_inputs(selection, package, audit['prompt_cache_sha256'],
        [dict(path=str(actual_ct), image=base.load_task.current[2],
              mode='read_only_hit' if audit['case_cache_hit'] else 'current_case_ram_miss')])
    if path(base.load_task.label_name).resolve() != path(selection['label_path']).resolve():
        raise ValueError('actual native label differs from selection')
    stage = audit['stages'][-1]; bounds = np.asarray(stage['bounds_dhw'], np.int64)
    indices = np.arange(*bounds[0], dtype=np.int64)
    placed = place_roi(observed['roi_prediction'], observed['prediction_dhw'].shape, indices, bounds)
    if not np.array_equal(placed, observed['prediction_dhw']):
        raise ValueError('native full-volume saved mask mismatch')
    compare_metrics(rows[0], selection['native_sam2_reference'])
    if {str(p): sha(p) for p in files} != before:
        raise ValueError('native source/weights changed during inference')
    fg, bg = native_points(stage)
    arrays = dict(prediction_dhw=observed['prediction_dhw'], logits_roi=observed['logits_roi'],
                  bounds_dhw=bounds, frame_indices=indices, prompt_fg_dhw=fg, prompt_bg_dhw=bg)
    return arrays, dict(metrics={m: rows[0][m] for m in METRICS}, threshold_probability=.5,
                        logit_grid_mapping=dict(grid='canonical_native_roi_unpadded', axes='ZXY',
                            crop_after_original_square_padding=True, final_frame_indices=indices.tolist(),
                            bounds_dhw=bounds.tolist(), overwrite_order='original forward then reverse; last visit wins',
                            outside_roi_logits='unpredicted; no fabricated logits'),
                        model_proofs=dict(assets=assets, file_sha256=before, audit=audit,
                                          promptgen_loaded=False, reference_differences=compare_metrics(rows[0], selection['native_sam2_reference'])))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, help='new selection output root containing selection.json')
    parser.add_argument('--method', choices=('v10_2', 'native_sam2'), required=True)
    parser.add_argument('--gpu', type=int, choices=(2, 4, 5, 6), required=True)
    parser.add_argument('--frame-batch', type=int, choices=(4,), default=4)
    args = parser.parse_args(argv)
    sys.path.insert(0, str(PROJECT))
    root = path(args.root).resolve(); selection = read(root/'selection.json')
    task = selection['task']
    if (selection.get('schema') != 'amos_liver_selection_v1'
            or selection.get('logit_export_authorized_for_this_single_case') is not True
            or (task['case_id'], task['global_class_id'], task['local_class_id']) != ('amos_0167', 123, 6)
            or selection['selected_variant'] != 'fg_scribble_bg_scribble'):
        raise ValueError('capture is restricted to frozen authorized single-case selection')
    ct = path(selection['ct_cache_path']); package = path(selection['prompt_package_path'])
    if not ct.is_file() or not package.is_file():
        raise ValueError('existing readonly shared CT/prompt cache required')
    signed = [ct, package, path(selection['image_path']), path(selection['label_path']), root/'selection.json']
    before = {str(p): sha(p) for p in signed}
    if selection.get('prompt_package_sha256', before[str(package)]) != before[str(package)]:
        raise ValueError('frozen prompt package hash mismatch')
    import nibabel as nib
    label = nib.as_closest_canonical(nib.load(str(path(selection['label_path']))))
    image = nib.as_closest_canonical(nib.load(str(path(selection['image_path']))))
    shape = tuple(label.shape[i] for i in (2, 0, 1))
    affine = label.affine[:, [2, 0, 1, 3]]
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    if (list(shape) != selection['shape_dhw'] or label.shape != image.shape
            or not np.allclose(label.affine, image.affine, atol=1e-5, rtol=0)
            or np.load(ct, mmap_mode='r').shape != shape):
        raise ValueError('canonical CT/label/shared cache grid mismatch')
    out = root/args.method
    out.mkdir(exist_ok=False)
    new_json(out/'input_signatures.json', before)
    resource = resource_binding(args.gpu)
    new_json(out/'resource_policy_audit.json', resource)
    arrays, detail = (capture_v10 if args.method == 'v10_2' else capture_native)(selection, out, resource)
    if arrays['prediction_dhw'].shape != shape:
        raise ValueError('saved full-volume mask shape mismatch')
    for name, values in arrays.items():
        if (name.endswith('_dhw') and 'prompt' in name) or name.startswith(('coronal_', 'sagittal_')):
            if values.shape[1:] != (3,) or np.any(values < 0) or np.any(values >= np.asarray(shape)):
                raise ValueError('saved prompt coordinate outside canonical volume')
    if {str(p): sha(p) for p in signed} != before:
        raise ValueError('readonly source/cache input changed during capture')
    metadata = dict(schema='amos_liver_capture_v1', method=args.method, case_id=task['case_id'],
                    global_class_id=123, label_id=6, ct_cache_path=str(ct),
                    image_path=str(path(selection['image_path'])), label_path=str(path(selection['label_path'])),
                    shape_dhw=list(shape), spacing_dhw_mm=spacing.tolist(), affine_dhw_ras=affine.tolist(),
                    canonical_axes='ZXY', selected_coronal_w=selection['selected_coronal_w'],
                    selected_sagittal_h=selection['selected_sagittal_h'],
                    prompt_coordinate_units='integer canonical DHW voxel centers',
                    provenance=dict(input_sha256=before, capture_source_sha256=sha(__file__),
                                    selection_sha256=before[str(root/'selection.json')], resources=resource), **detail)
    info = save_capture(out, metadata, arrays)
    print('CAPTURE_COMPLETE', args.method, info['capture_npz_sha256'], flush=True)


if __name__ == '__main__':
    main()
