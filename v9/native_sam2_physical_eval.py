"""Original SAM2 video baseline with read-only, physically mapped scribble prompts.

No learned prompt adapter, text encoder, GT-defined crop, or large cache writes.
The Gaussian prior is evidence at real scribble pixels, NOT an estimated GT mask.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from infer.native_sam2_physical_prompts import (expand_bounds_3d, gaussian_prior,
    map_voxels, pad_points, pad_square, prompt_z_bounds, reorient_grid,
    sample_scribble_points, prompt_bounds_3d, annotation_order)

CACHE = 'v10/.cache/totalseg_sam2_promptgen_v10_prompt_roi_full_v1'
PREVIOUS = 'output/ct13_full_val_v92e360_v102e490_20261005'
SOURCES = ('magic', 'msd_task07', 'msd_task10', 'msd_task08')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.{}.tmp'.format(os.getpid()))
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(str(tmp), str(path))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def key(row):
    return str(row['source_name']), str(row['case_id']), int(row['global_class_id'])


def prepare(output, full=False):
    """Only JSON references: the previous exact prompt packages stay read-only."""
    previous = ROOT / PREVIOUS
    refs = {}
    for file in sorted((previous / 'prepared').glob('*/refs.json')):
        for row in read(file)['per_case_class']:
            refs[key(row)] = row
    tasks = [r for r in read(previous / 'manifest.json')['tasks'] if r['source_name'] in SOURCES]
    tasks.sort(key=key)
    if not full:
        selected = {}; [selected.setdefault((r['source_name'], r['global_class_id']), r) for r in tasks]
        tasks = list(selected.values())
    split = read(ROOT / 'configs/ct13_v10_2_split_20260928.json')
    for task in tasks:
        dataset = split['datasets'][task['source_name']]
        assert task['case_id'] not in {r['case_id'] for r in dataset['train']}
        assert task['case_id'] in {r['case_id'] for r in dataset['val']}
        task['prompt_cache_name'] = refs[key(task)]['prompt_cache_name']
    write(output / 'manifest.json', {'tasks': tasks, 'train_overlap': 0,
        'scope': 'full135' if full else 'one_fixed_val_case_per_class12'})
    jobs = []
    for source in SOURCES:
        for arm in ('none', 'gaussian'):
            name = '{}_{}'.format(source, arm)
            job = {'name': name, 'dense_prior': arm, 'output': str(output / name),
                   'tasks': [r for r in tasks if r['source_name'] == source]}
            path = output / 'job_specs' / (name + '.json'); write(path, job)
            jobs.append(str(path))
    write(output / 'jobs.json', {'jobs': jobs})
    print('Prepared {} held-out tasks, {} paired-arm jobs'.format(len(tasks), len(jobs)), flush=True)


def canonical(path):
    import nibabel as nib
    image = nib.as_closest_canonical(nib.load(str(path)))
    affine = np.eye(4); affine[:3, :3] = image.affine[:3, [2, 0, 1]]
    affine[:3, 3] = image.affine[:3, 3]
    return image, affine


def load_task(task, split):
    """Canonical target and exact cache-grid reorientation; target never selects ROI."""
    source = task['source_name']
    entry = next(r for r in split['datasets'][source]['val'] if r['case_id'] == task['case_id'])
    image_path = entry.get('image', entry.get('image_path'))
    label_path = entry.get('label', entry.get('label_path'))
    image, target_affine = canonical(image_path)
    target_image, label_affine = canonical(label_path)
    assert np.allclose(label_affine, target_affine, atol=1e-4), 'target/image physical grids disagree'
    target_values = np.asarray(target_image.dataobj).transpose(2, 0, 1)
    target = (target_values > 0 if source == 'magic' else target_values == int(task['local_class_id']))
    package_path = ROOT / CACHE / 'metadata/shared_adaptive_prompt_v1' / source / task['prompt_cache_name']
    with np.load(package_path, allow_pickle=False) as package:
        shape = tuple(int(v) for v in package['shape_dhw'])
        supports = [np.unpackbits(package[name].astype(np.uint8), count=int(np.prod(shape)),
                    bitorder='little').reshape(shape).astype(bool) for name in ('coronal_bits', 'sagittal_bits')]
        bg = np.asarray(package['background_points_dhw'], dtype=np.float64).reshape(-1, 3)
        case_name = str(package['case_image_name'].item())
    ct = np.load(ROOT / CACHE / '_case_images_v1' / case_name, mmap_mode='r')
    assert ct.shape == shape, 'prompt/cache source grid mismatch'
    source_affine = target_affine.copy()
    if source == 'magic':
        import SimpleITK as sitk
        reader = sitk.ImageFileReader(); reader.SetFileName(task['image']); reader.ReadImageInformation()
        lps = np.eye(4)
        lps[:3, :3] = (np.asarray(reader.GetDirection()).reshape(3, 3) @ np.diag(reader.GetSpacing()))[:, [2, 1, 0]]
        lps[:3, 3] = reader.GetOrigin()
        source_affine = np.diag([-1., -1., 1., 1.]) @ lps
    target_shape = target.shape
    ct = reorient_grid(ct, source_affine, target_affine, target_shape)
    transformed = [reorient_grid(s, source_affine, target_affine, target_shape) for s in supports]
    foreground = transformed[0] | transformed[1]
    bg_mapped, bg_error = map_voxels(bg, source_affine, target_affine)
    fg_coords = np.argwhere(supports[0] | supports[1])
    fg_mapped, fg_error = map_voxels(fg_coords, source_affine, target_affine)
    bg = np.rint(bg_mapped).astype(int)
    assert max(fg_error, bg_error) < 1e-3, 'physical roundtrip failed'
    mapped_mask = np.zeros(target_shape, bool); mapped_mask[tuple(np.rint(fg_mapped).astype(int).T)] = True
    assert np.array_equal(mapped_mask, foreground), 'array and coordinate mapping disagree'
    assert len(bg) == 4 and len(supports) == 2, 'expected original 2 FG planes and 4 BG points'
    assert ((bg >= 0) & (bg < np.asarray(target_shape))).all(), 'background outside image'
    spacing = np.linalg.norm(target_affine[:3, :3], axis=0)
    audit = {'prompt_cache_sha256': sha(package_path), 'case_image_name': case_name,
        'source_shape_dhw': shape, 'target_shape_dhw': target_shape,
        'source_affine_dhw_ras': source_affine.tolist(), 'target_affine_dhw_ras': target_affine.tolist(),
        'physical_roundtrip_error_mm': max(fg_error, bg_error),
        'original_plane_voxels': [int(s.sum()) for s in supports],
        'mapped_plane_voxels': [int(s.sum()) for s in transformed],
        'fg_voxels': int(foreground.sum()), 'fg_in_gt_fraction': float(target[foreground].mean()),
        'bg_points_dhw': bg.tolist(), 'bg_in_gt_count': int(target[tuple(bg.T)].sum()),
        'spacing_dhw_mm': spacing.tolist(), 'gt_used_for_input': False}
    return ct, target, foreground, bg, spacing, audit


class LazyFrames:
    def __init__(self, ct, bounds, image_size, low, high):
        self.ct = ct; self.start, self.end = bounds; self.image_size = image_size
        self.low = low; self.scale = max(high - low, 1e-6)

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, index):
        import torch
        import torch.nn.functional as F
        image = np.clip((np.asarray(self.ct[self.start + index], dtype=np.float32) - self.low) / self.scale, 0., 1.)
        image, _ = pad_square(image)
        tensor = torch.from_numpy(image.copy())[None, None]
        tensor = F.interpolate(tensor, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)[0]
        tensor = tensor.repeat(3, 1, 1)
        return (tensor - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]


def predict_roi(model, ct, foreground, bg, spacing, bounds, args):
    import torch
    import sam2.sam2_video_predictor as video_module
    start, end = bounds; height, width = ct.shape[1:]; side = max(height, width)
    offset = ((side - width) // 2, (side - height) // 2)
    low, high = np.percentile(ct[start:end], (1., 99.))
    frames = LazyFrames(ct, bounds, model.image_size, float(low), float(high))
    original_loader = video_module.load_video_frames
    video_module.load_video_frames = lambda **kwargs: (frames, side, side)
    try:
        state = model.init_state(video_path='read_only_shared_CT', offload_video_to_cpu=True, offload_state_to_cpu=True)
    finally:
        video_module.load_video_frames = original_loader
    evidence = {}; provenance = []; counters = {'point_encoder_calls': 0, 'mask_encoder_calls': 0}
    encoder = model.sam_prompt_encoder; original_encoder = encoder.forward
    def encoded(*a, **kw):
        if kw.get('points') is not None: counters['point_encoder_calls'] += 1
        if kw.get('masks') is not None: counters['mask_encoder_calls'] += 1
        return original_encoder(*a, **kw)
    encoder.forward = encoded
    original_step = model._run_single_frame_inference
    annotated, background_only = annotation_order(foreground, bg, bounds)
    def step(self, **kwargs):
        frame = kwargs['frame_idx']
        if kwargs['point_inputs'] is not None and not kwargs['run_mem_encoder']:
            # A negative-only frame cannot define the identity of a new object.
            # Use the native memory of the already registered positive frames.
            if frame + start in background_only:
                kwargs['is_init_cond_frame'] = False
        if kwargs['point_inputs'] is not None and not kwargs['run_mem_encoder'] and args.dense_prior == 'gaussian':
            kwargs['prev_sam_mask_logits'] = torch.from_numpy(evidence[frame])[None, None].to(self.device)
        return original_step(**kwargs)
    model._run_single_frame_inference = types.MethodType(step, model)
    try:
        for z in annotated:
            if background_only and z == background_only[0]:
                model.propagate_in_video_preflight(state)
            sampled = sample_scribble_points(foreground[z], spacing[1:], args.max_points)
            negatives = bg[bg[:, 0] == z, 1:]
            points = np.concatenate((sampled, negatives[:, ::-1].astype(np.float32)))
            labels = np.r_[np.ones(len(sampled), np.int32), np.zeros(len(negatives), np.int32)]
            if len(sampled):
                assert foreground[z][tuple(sampled[:, ::-1].astype(int).T)].all(), 'point outside original scribble'
            frame = z - start
            evidence[frame] = gaussian_prior(foreground[z], negatives, spacing[1:], args.sigma_mm, args.amplitude)
            provenance.append({'z': z, 'fg_sampled_xy': sampled.tolist(), 'bg_original_yx': negatives.tolist()})
            model.add_new_points_or_box(state, frame_idx=frame, obj_id=1,
                points=pad_points(points, offset), labels=labels, clear_old_points=True)
        anchor = int(np.argmax(foreground[start:end].sum(axis=(1, 2))))
        result = np.zeros((end - start, height, width), bool); visited = set()
        for reverse in (False, True):
            if reverse and anchor == 0: continue
            for frame, _, masks in model.propagate_in_video(state, start_frame_idx=anchor, reverse=reverse):
                pred = (masks[0, 0] > 0).detach().cpu().numpy()
                left, top = offset; result[frame] = pred[top:top + height, left:left + width]
                visited.add(frame)
        assert len(visited) == end - start, 'missing true ROI frames'
        assert counters['point_encoder_calls'] >= len(annotated)
        assert counters['mask_encoder_calls'] == (len(annotated) if args.dense_prior == 'gaussian' else 0)
        return result, {'bounds_z': list(bounds), 'annotated_frames': annotated, 'native_encoder': counters,
            'background_only_frames_memory_conditioned': background_only,
            'roi_frames_predicted': len(visited), 'point_provenance': provenance,
            'image_percentile_1_99': [float(low), float(high)], 'square_pad_left_top': list(offset)}
    finally:
        model._run_single_frame_inference = original_step; encoder.forward = original_encoder
        del state


def metric_module():
    spec = importlib.util.spec_from_file_location('_native_complete_mask_metrics', ROOT / 'utils/metrics.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    # Exact same four-neighbour boundary and directed arithmetic averaging;
    # cKDTree avoids materialising quadratic torch.cdist matrices on CPU.
    from scipy.ndimage import binary_erosion
    from scipy.spatial import cKDTree
    def nsd(pred, target, tolerance=1., spacing=None):
        p = pred.numpy().astype(bool); t = target.numpy().astype(bool)
        if not p.any() and not t.any(): return 1.
        if not p.any() or not t.any(): return 0.
        scale = np.asarray(spacing if spacing is not None else (1., 1.))
        ps = np.argwhere(p & ~binary_erosion(p)) * scale
        ts = np.argwhere(t & ~binary_erosion(t)) * scale
        return float(.5 * ((cKDTree(ts).query(ps)[0] <= tolerance).mean() + (cKDTree(ps).query(ts)[0] <= tolerance).mean()))
    module.nsd_score = nsd
    return module


def overlay(path, ct, target, fg, pred):
    from PIL import Image
    from scipy.ndimage import binary_erosion
    z = int(np.argmax(fg.sum(axis=(1, 2))))
    lo, hi = np.percentile(ct[z], (1, 99)); gray = np.clip((ct[z] - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = np.repeat((gray[..., None] * 255).astype(np.uint8), 3, axis=2)
    rgb[target[z] & ~binary_erosion(target[z])] = [255, 80, 80]
    rgb[pred[z] & ~binary_erosion(pred[z])] = [80, 200, 255]
    rgb[fg[z]] = [70, 255, 70]
    Image.fromarray(rgb).save(path)


def run(args):
    import torch
    sys.path.insert(0, str(ROOT / 'sam2'))
    from sam2.build_sam import build_sam2_video_predictor
    job = read(args.job); output = Path(job['output']); output.mkdir(parents=True, exist_ok=True)
    args.dense_prior = job.get('dense_prior', args.dense_prior)
    gpu_uuid = str(torch.cuda.get_device_properties(0).uuid).replace('GPU-', '')
    if args.expected_uuid and gpu_uuid != args.expected_uuid.replace('GPU-', ''):
        raise ValueError('CUDA/physical GPU UUID mismatch')
    model = build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_l.yaml', str(ROOT / args.checkpoint),
        device='cuda', hydra_overrides_extra=['++model.use_mask_input_as_output_without_sam=false', '++model.fill_hole_area=0'])
    assert not model.use_mask_input_as_output_without_sam
    # The optional native CUDA connected-components extension is unavailable on
    # this server. Explicitly disable the otherwise skipped tiny-hole operation.
    model.fill_hole_area = 0
    scorer = metric_module(); rows = []; audits = []
    split = read(ROOT / 'configs/ct13_v10_2_split_20260928.json')
    for index, task in enumerate(job['tasks']):
        started = time.time()
        ct, target, fg, bg, spacing, audit = load_task(task, split)
        if args.xy_crop == 'prompt':
            initial = prompt_bounds_3d(fg, bg, spacing)
        else:
            initial = (prompt_z_bounds(fg, bg, spacing[0]), (0, ct.shape[1]), (0, ct.shape[2]))
        bounds = initial; stages = []
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            for _ in range(8):
                (_, _), (h0, h1), (w0, w1) = bounds
                local_bg = bg - np.array([0, h0, w0])
                roi_prediction, stage = predict_roi(model, ct[:, h0:h1, w0:w1], fg[:, h0:h1, w0:w1],
                    local_bg, spacing, bounds[0], args)
                stage['bounds_dhw'] = [list(b) for b in bounds]
                stages.append(stage)
                enlarged = expand_bounds_3d(roi_prediction, fg, bounds, initial)
                if enlarged == bounds: break
                bounds = enlarged
            else: raise RuntimeError('AutoExpand did not converge within bounded iterations')
        prediction = np.zeros(target.shape, bool)
        prediction[tuple(slice(start, end) for start, end in bounds)] = roi_prediction
        metrics = scorer.binary_prompt_metrics_from_masks(torch.from_numpy(prediction), torch.from_numpy(target), tuple(spacing))
        supported = fg.any(axis=(1, 2))
        prompt_metrics = scorer.binary_prompt_metrics_from_masks(torch.from_numpy(prediction[supported]),
            torch.from_numpy(target[supported]), tuple(spacing))
        values = {k: float(v) for k, v in metrics.items()}
        assert all(np.isfinite(v) for v in values.values())
        rows.append(dict(task, **values, seconds=time.time() - started))
        audits.append(dict(task, **audit, stages=stages, prompt_frame_dice=float(prompt_metrics['dice']),
            complete_frame_prediction=True, dense_prior=args.dense_prior, native_mask_threshold_logits=0.,
            xy_crop=args.xy_crop, native_fill_hole_area=0, semantic_input=False,
            promptgen_loaded=any('prompt_generator' in name or 'prompt_token_generator' in name for name in sys.modules)))
        assert not audits[-1]['promptgen_loaded']
        write(output / 'all_metrics_live.json', {'per_case_class': rows})
        write(output / 'exact_prompt_audit.json', {'per_case_class': audits})
        overlay(output / ('overlay_{:03d}.png'.format(index)), ct, target, fg, prediction)
        print('DONE {}/{} {} Dice={:.4f} prompt-frame={:.4f} elapsed={:.1f}s'.format(
            index + 1, len(job['tasks']), key(task), values['dice'], prompt_metrics['dice'], time.time() - started), flush=True)
        del ct, target, fg, prediction, roi_prediction
        torch.cuda.empty_cache()
    write(output / 'complete_metrics.json', {'model': 'native_sam2.1_hiera_l',
        'checkpoint': args.checkpoint, 'checkpoint_sha256': sha(ROOT / args.checkpoint),
        'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in
            (Path(__file__), ROOT / 'infer/native_sam2_physical_prompts.py')},
        'dense_prior': args.dense_prior, 'xy_crop': args.xy_crop, 'sigma_mm': args.sigma_mm, 'amplitude': args.amplitude,
        'max_points_per_supported_slice': args.max_points, 'gpu_uuid': gpu_uuid,
        'metric_protocol': 'complete_mask; axial_union_nonempty; physical_axial_NSD1/2/3',
        'per_case_class': rows, 'summary': scorer.aggregate_promptgen_binary_metrics(rows)})


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--output', default='output/native_sam2_physical_pilot_20261006')
    parser.add_argument('--checkpoint', default='sam2/checkpoints/sam2.1_hiera_large.pt')
    parser.add_argument('--dense-prior', choices=('none', 'gaussian'), default='gaussian')
    parser.add_argument('--xy-crop', choices=('full', 'prompt'), default='prompt')
    parser.add_argument('--sigma-mm', type=float, default=2.)
    parser.add_argument('--amplitude', type=float, default=4.)
    parser.add_argument('--max-points', type=int, default=16)
    parser.add_argument('--expected-uuid')
    return parser.parse_args(argv)


if __name__ == '__main__':
    arguments = parse_args()
    if arguments.prepare:
        prepare(ROOT / arguments.output, arguments.full)
    elif arguments.job:
        run(arguments)
    else:
        raise SystemExit('--job or --prepare is required')
