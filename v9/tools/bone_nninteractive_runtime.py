"""Read-only Bone prompt evaluation through the official nnInteractive session.

The queue helper and native interaction methods keep their original semantics.
Only the user-level ``_predict`` entry is audited; autozoom's internal network
passes remain native model computation within that single interaction round.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import subprocess

from tools.ct13_full_validation_eval import METRICS


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _key(row):
    return str(row['source_name']), str(row['case_id']), int(row['global_class_id'])


def _check_task(task):
    if (task['source_name'] != 'bone' or int(task['local_class_id']) <= 0 or
            int(task['global_class_id']) != 10000 + int(task['local_class_id'])):
        raise ValueError('Bone global bookkeeping ID must map to the original local label ID')


def _verify_files(root, files):
    for name, signature in files.items():
        if _sha(Path(root) / name) != signature:
            raise ValueError(f'frozen source/model signature changed: {name}')


def _gpu_resource(job):
    """Check physical identity/free memory, then verify the CUDA-visible UUID."""
    gpu = int(job['gpu'])
    minimum = int(job['minimum_free_mib'])
    if gpu not in (2, 4, 5, 6) or minimum < 30720:
        raise ValueError('only physical GPU 2/4/5/6 with >=30720 MiB is permitted')
    output = subprocess.check_output(
        ['nvidia-smi', '-i', str(gpu), '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'],
        text=True).strip().split(',')
    physical_uuid, available = output[0].strip(), int(output[1].strip())
    if physical_uuid != job['gpu_uuid']:
        raise ValueError('physical GPU UUID changed')
    if available < minimum:
        raise SystemExit(75)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('expected exactly one CUDA-visible GPU')
    torch.cuda.set_device(0)
    probe = torch.zeros(1, device='cuda:0')
    torch.cuda.synchronize()
    del probe
    # CUDA driver UUID checks also work on Torch releases whose device-property
    # object does not expose uuid. The driver respects CUDA_VISIBLE_DEVICES.
    import ctypes
    import uuid
    driver = ctypes.CDLL('libcuda.so.1')
    device = ctypes.c_int()
    raw_uuid = (ctypes.c_ubyte * 16)()
    uuid_function = getattr(driver, 'cuDeviceGetUuid_v2', driver.cuDeviceGetUuid)
    for status in (driver.cuInit(0), driver.cuDeviceGet(ctypes.byref(device), 0),
                   uuid_function(ctypes.byref(raw_uuid), device)):
        if status != 0:
            raise RuntimeError(f'CUDA UUID query failed with status {status}')
    cuda_uuid = 'GPU-' + str(uuid.UUID(bytes=bytes(raw_uuid)))
    if cuda_uuid.lower() != physical_uuid.lower():
        raise ValueError('CUDA-visible UUID does not match the assigned physical GPU')
    return dict(gpu=gpu, gpu_uuid=physical_uuid, cuda_uuid=physical_uuid,
                actual_start_free_mib=available, minimum_free_mib=minimum)


def validate_prompt(image, prompt):
    import numpy as np

    shape = tuple(image.shape)
    if len(shape) != 3:
        raise ValueError('case image must have three dimensions')
    planes = []
    for name, axis in (('coronal', 2), ('sagittal', 1)):
        scribble = np.asarray(prompt[name])
        if scribble.shape != shape or not np.all((scribble == 0) | (scribble == 1)):
            raise ValueError(f'{name} must be a binary scribble matching the case shape')
        support = np.argwhere(scribble > 0)
        positions = np.unique(support[:, axis])
        if len(positions) != 1:
            raise ValueError(f'{name} must occupy exactly one nonempty plane')
        position = int(positions[0])
        if int(prompt.get(name + '_slice', position)) != position:
            raise ValueError(f'{name} plane metadata mismatch')
        planes.append(position)
    points = np.asarray(prompt['background_points_dhw'])
    if points.shape != (4, 3) or not np.isfinite(points).all() or not np.array_equal(points, np.rint(points)):
        raise ValueError('exactly four integer background points are required')
    if np.any(points < 0) or np.any(points >= np.asarray(shape)):
        raise ValueError('background point outside the case volume')
    if list(prompt['background_point_planes']) != ['coronal', 'coronal', 'sagittal', 'sagittal']:
        raise ValueError('background point plane order must be coronal, coronal, sagittal, sagittal')
    if not np.all(points[:2, 2] == planes[0]) or not np.all(points[2:, 1] == planes[1]):
        raise ValueError('background points must remain on their frozen prompt planes')


class SingleRoundPredictor:
    """Reuse preprocessing for one identical image object, reset each class."""

    def __init__(self, session, apply_prompt):
        self.session = session
        self.apply_prompt = apply_prompt
        self._image = None
        self._preprocess_calls = 0
        self._reset_calls = 0

    def _reset(self):
        self.session.reset_interactions()
        self._reset_calls += 1

    def predict(self, image, prompt):
        import numpy as np
        import torch

        validate_prompt(image, prompt)
        session = self.session
        reused = image is self._image
        if not reused:
            self._image = None
            session.set_image(np.asarray(image, dtype=np.float32)[None])
            session._finish_preprocessing_and_initialize_interactions()
            self._preprocess_calls += 1
            self._image = image
        self._reset()
        buffer = torch.zeros(tuple(image.shape), dtype=torch.uint8, device='cpu')
        session.set_target_buffer(buffer)
        events = []
        additions = 0
        triggers = 0
        # Preserve whether each method originally lived on the instance or its
        # class, so failure cannot leave a wrapper or bound-method override.
        originals = {}

        def replace(name, method):
            originals[name] = (name in session.__dict__, session.__dict__.get(name))
            setattr(session, name, method)

        native_predict = session._predict
        native_point = session.add_point_interaction
        native_scribble = session.add_scribble_interaction

        def point(coordinates, include_interaction, run_prediction=True):
            nonlocal additions
            expected = np.asarray(prompt['background_points_dhw'])
            if (additions >= 4 or include_interaction or run_prediction or
                    not np.array_equal(np.asarray(coordinates), expected[additions])):
                raise ValueError('unexpected background point, order, role or prediction flag')
            events.append(dict(kind='point', role='background', coordinates=list(map(int, coordinates)),
                               plane=prompt['background_point_planes'][additions], run_prediction=False))
            additions += 1
            return native_point(coordinates, include_interaction=include_interaction, run_prediction=run_prediction)

        def scribble(scribble_image, include_interaction, run_prediction=True):
            nonlocal additions
            name = 'coronal' if additions == 4 else 'sagittal'
            if (additions not in (4, 5) or not include_interaction or
                    bool(run_prediction) != (additions == 5) or
                    not np.array_equal(np.asarray(scribble_image), np.asarray(prompt[name]))):
                raise ValueError('unexpected foreground scribble, order, role or prediction flag')
            events.append(dict(kind='scribble', role='foreground', plane=name,
                               run_prediction=bool(run_prediction), voxels=int(np.count_nonzero(scribble_image))))
            additions += 1
            return native_scribble(scribble_image, include_interaction=include_interaction, run_prediction=run_prediction)

        def predict_once(*args, **kwargs):
            nonlocal triggers
            if additions != 6:
                raise ValueError('early prediction before the complete six-prompt bundle')
            if triggers:
                raise ValueError('a second prediction is forbidden; exactly one round is required')
            triggers += 1
            events.append(dict(kind='predict', prompt_addition_count=additions, prediction_trigger_count=triggers))
            return native_predict(*args, **kwargs)

        def unexpected(*_args, **_kwargs):
            raise ValueError('unexpected prompt interaction type')

        try:
            replace('_predict', predict_once)
            replace('add_point_interaction', point)
            replace('add_scribble_interaction', scribble)
            for name in ('add_bbox_interaction', 'add_lasso_interaction', 'add_initial_seg_interaction'):
                if callable(getattr(session, name, None)):
                    replace(name, unexpected)
            self.apply_prompt(session, prompt['coronal'], prompt['sagittal'], prompt['background_points_dhw'])
            if additions != 6 or triggers != 1:
                raise ValueError('missing prompt input or actual single prediction trigger')
            prediction = buffer.detach().clone().float().cpu()
        finally:
            for name, (was_instance, value) in originals.items():
                if was_instance:
                    setattr(session, name, value)
                else:
                    delattr(session, name)
            self._reset()
        audit = dict(prediction_trigger_count=triggers, prompt_addition_count=additions,
                     foreground_scribbles=2, background_points=4, events=events,
                     case_image_reused=reused, preprocess_calls=self._preprocess_calls,
                     reset_interactions_calls=self._reset_calls)
        return prediction, audit


def _source_provenance(root, config, obj):
    source_path = Path(inspect.getsourcefile(obj)).resolve()
    expected = {(root / name).resolve(): (name, signature) for name, signature in config['sources'].items()}
    if source_path not in expected:
        raise ValueError(f'imported native source is outside the frozen bundle: {source_path.name}')
    name, signature = expected[source_path]
    if _sha(source_path) != signature:
        raise ValueError(f'imported native source signature mismatch: {name}')
    return dict(path=name, sha256=signature, module=obj.__module__, name=obj.__qualname__)


def _model_provenance(root, config, session, source_proofs):
    model_dir = (root / config['model_dir']).resolve()
    expected = {(root / name).resolve(): signature for name, signature in config['model_files'].items()}
    checkpoints = list(model_dir.glob('fold_*/checkpoint_final.pth'))
    if len(checkpoints) != 1:
        raise ValueError('native restore requires exactly one checkpoint_final fold')
    checkpoint = checkpoints[0]
    for path in [checkpoint, *(model_dir / name for name in
                               ('plans.json', 'dataset.json', 'inference_session_class.json'))]:
        if path.resolve() not in expected:
            raise ValueError(f'native restore file lacks frozen signature: {path.name}')
    read_json = lambda name: json.loads((model_dir / name).read_text(encoding='utf-8'))
    plans = read_json('plans.json')
    return dict(model_dir=config['model_dir'], files=dict(config['model_files']),
                autozoom=bool(session.do_autozoom),
                checkpoint_path=os.path.relpath(checkpoint, root).replace(os.sep, '/'),
                checkpoint_sha256=expected[checkpoint.resolve()],
                restore='initialize_from_trained_model_folder; default checkpoint_final.pth; sole fold',
                state_dict_restore='native load_state_dict(parameters); strict=True default',
                trainer_name=str(session.trainer_name),
                plans={key: plans.get(key) for key in ('dataset_name', 'plans_name', 'transpose_forward', 'transpose_backward')},
                session_metadata=read_json('inference_session_class.json'),
                dataset_metadata=read_json('dataset.json'), **source_proofs)


def validate_record(record, task, config, job):
    """Lightweight record checks; does not import Torch or initialize CUDA."""
    _check_task(task)
    if record['task'] != task:
        raise ValueError('record task differs from frozen original task')
    row = record['row']
    if (_key(row) != _key(task) or int(row['local_class_id']) != int(task['local_class_id']) or
            int(row['label_id']) != int(task['local_class_id'])):
        raise ValueError('record lost the original Bone label ID')
    if any(name not in row or not math.isfinite(float(row[name])) for name in METRICS):
        raise ValueError('record has missing or nonfinite full-mask metrics')
    audit = record['interaction']
    if any(audit.get(name) != count for name, count in
           (('prediction_trigger_count', 1), ('prompt_addition_count', 6),
            ('foreground_scribbles', 2), ('background_points', 4))):
        raise ValueError('record is not a single complete interaction round')
    events = audit.get('events', [])
    if (len(events) != 7 or [event.get('kind') for event in events] != ['point'] * 4 + ['scribble'] * 2 + ['predict'] or
            [event.get('run_prediction') for event in events[:6]] != [False] * 5 + [True] or
            [event.get('role') for event in events[:6]] != ['background'] * 4 + ['foreground'] * 2 or
            [event.get('plane') for event in events[:6]] != ['coronal'] * 2 + ['sagittal'] * 2 + ['coronal', 'sagittal'] or
            events[-1].get('prompt_addition_count') != 6 or events[-1].get('prediction_trigger_count') != 1):
        raise ValueError('record lacks six ordered actual additions before its sole native prediction')
    resource = record['resource']
    if (int(resource['gpu']) != int(job['gpu']) or resource['gpu_uuid'] != job['gpu_uuid'] or
            resource['cuda_uuid'] != job['gpu_uuid'] or int(resource['minimum_free_mib']) != int(job['minimum_free_mib']) or
            int(resource['actual_start_free_mib']) < int(job['minimum_free_mib'])):
        raise ValueError('record resource identity/free-memory proof mismatch')
    name = record['prompt_cache_name']
    if config['prompt_hashes'].get(name) != record['prompt_sha256']:
        raise ValueError('record frozen prompt signature mismatch')
    refs = [ref for ref in config['prompt_refs'] if _key(ref) == _key(task)]
    if len(refs) != 1 or refs[0]['prompt_cache_name'] != name:
        raise ValueError('record prompt reference does not match its frozen task')
    model = record['model']
    if model.get('autozoom') is not True or model.get('files') != config['model_files'] or model.get('model_dir') != config['model_dir']:
        raise ValueError('record native model restore signature/autozoom mismatch')
    if config['model_files'].get(model.get('checkpoint_path')) != model.get('checkpoint_sha256'):
        raise ValueError('record native checkpoint signature mismatch')
    for kind in ('session_source', 'prompt_helper_source'):
        source = model.get(kind, {})
        if source.get('path') not in config['sources'] or config['sources'][source['path']] != source.get('sha256'):
            raise ValueError('record imported native source is outside the frozen bundle')
    proof = record['metric_execution']
    if (proof.get('provider_sha256') != _sha(Path(__file__).with_name('bone_fast_metrics.py')) or
            proof.get('stats', {}).get('comparison_failures', 0) or
            any(proof.get('metrics', {}).get(name) != row[name] for name in METRICS)):
        raise ValueError('record is not bound to the exact full-mask metric provider output')
    if proof.get('comparisons', 0) < 1 or proof.get('max_error') is None or not 0 <= float(proof['max_error']) <= 1e-6:
        raise ValueError('record lacks a successful native metric equality check')
    geometry = record['geometry']
    if (geometry.get('prediction_shape_dhw') != geometry.get('shape_dhw') or
            len(geometry.get('shape_dhw', [])) != 3 or len(geometry.get('spacing_dhw', [])) != 3 or
            any(not math.isfinite(float(v)) or float(v) <= 0 for v in geometry['spacing_dhw'])):
        raise ValueError('record full-volume geometry mismatch')


def run_case(root, job, config, commit):
    """Load one official model/case, score each frozen task and commit scalars."""
    from tools import ct13_full_validation_eval as common

    root = Path(root).resolve()
    tasks = job['tasks']
    if not tasks:
        raise ValueError('case worker has no tasks')
    for task in tasks:
        _check_task(task)
    if (len({_key(task) for task in tasks}) != len(tasks) or
            any(str(task['case_id']) != str(job['case_id']) for task in tasks) or
            len({(task['image'], task['label']) for task in tasks}) != 1):
        raise ValueError('case worker must contain disjoint classes of one image/label pair')
    _verify_files(root, config['sources'])
    _verify_files(root, config['model_files'])
    refs = config.get('prompt_refs')
    if refs is None:
        refs = common.read(root / config['reference_output'] / 'refs.json')['per_case_class']
        config = {**config, 'prompt_refs': refs}
    reference_index = {_key(ref): ref for ref in refs}
    if len(reference_index) != len(refs):
        raise ValueError('duplicate frozen prompt references')
    cache_root = root / config.get('cache_root', common.CACHE)
    prompt_paths = []
    for task in tasks:
        ref = reference_index[_key(task)]
        name = ref['prompt_cache_name']
        if Path(name).name != name:
            raise ValueError('prompt reference must be a cache filename')
        path = cache_root / 'metadata/shared_adaptive_prompt_v1/bone' / name
        if name not in config['prompt_hashes'] or _sha(path) != config['prompt_hashes'][name]:
            raise ValueError(f'frozen prompt signature changed: {name}')
        prompt_paths.append(path)

    resource = _gpu_resource(job)
    os.environ['V9_SHARED_CACHE_WRITE_POLICY'] = 'read_only'
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    protocol, evaluator = common.nn_modules(job['gpu'])
    import numpy as np
    import torch
    import adaptive_prompt_cache
    import multisource_split_protocol as cache
    from tools.bone_fast_metrics import install_provider

    torch.set_num_threads(2)
    protocol.EIGHT_SOURCES = (*protocol.EIGHT_SOURCES, 'bone') if 'bone' not in protocol.EIGHT_SOURCES else protocol.EIGHT_SOURCES
    evaluator.patch_nninteractive_trainer_for_current_nnunet()
    session = evaluator.nnInteractiveInferenceSession(
        device=torch.device('cuda:0'), use_torch_compile=False, verbose=False,
        torch_n_threads=2, do_autozoom=True, use_pinned_memory=True)
    source_proofs = dict(session_source=_source_provenance(root, config, type(session)),
                         prompt_helper_source=_source_provenance(root, config, evaluator.apply_two_plane_prompt))
    session.initialize_from_trained_model_folder(str(root / config['model_dir']))
    model = _model_provenance(root, config, session, source_proofs)
    first_task = tasks[0]
    native_task = protocol.AdaptiveValidationTask(
        source_name='bone', case_id=str(first_task['case_id']),
        global_class_id=int(first_task['global_class_id']), local_class_id=int(first_task['local_class_id']),
        class_name=first_task['class_name'], image_path=root / first_task['image'],
        label_path=root / first_task['label'], category=first_task.get('category'))
    case_path = cache.shared_case_cache_path(native_task.image_path, cache_root, 'bone')
    image = cache._valid_case_cache(case_path)
    image_cache_source = 'cache_hit'
    if image is None:
        image = evaluator.load_raw_axial_volume(str(native_task.image_path), canonical=True).astype(np.float32, copy=False)
        image_cache_source = 'source_memory_only'
    integer_label = evaluator.load_raw_axial_volume(str(native_task.label_path), canonical=True).astype(np.int32, copy=False)
    native_shape, spacing = evaluator.load_axial_shape_spacing_dhw(str(native_task.image_path), canonical=True)
    if tuple(image.shape) != tuple(native_shape) or tuple(integer_label.shape) != tuple(image.shape):
        raise ValueError('native image/cache/label geometry mismatch')
    compare_first = [True]
    metric, tracker = install_provider(common.metric_module(), compare_native=lambda: compare_first[0], workers=2, cache_entries=0)
    predictor = SingleRoundPredictor(session, evaluator.apply_two_plane_prompt)
    for task, path in zip(tasks, prompt_paths):
        if _sha(path) != config['prompt_hashes'][path.name]:
            raise ValueError(f'frozen prompt signature changed: {path.name}')
        prompt = adaptive_prompt_cache._read(path)
        if prompt is None or _sha(path) != config['prompt_hashes'][path.name]:
            raise ValueError(f'missing, invalid or changed frozen prompt package: {path.name}')
        if prompt.get('case_image_name') != case_path.name:
            raise ValueError('frozen prompt package refers to a different shared case CT')
        native_task = protocol.AdaptiveValidationTask(
            source_name='bone', case_id=str(task['case_id']), global_class_id=int(task['global_class_id']),
            local_class_id=int(task['local_class_id']), class_name=task['class_name'],
            image_path=root / task['image'], label_path=root / task['label'], category=task.get('category'))
        target = protocol.load_task_target(native_task, source_loader=lambda _path: integer_label)
        if target.shape != image.shape or not np.any(target):
            raise ValueError('empty or mismatched original-label Bone target')
        with torch.inference_mode():
            prediction, audit = predictor.predict(image, prompt)
            if tuple(prediction.shape) != tuple(target.shape):
                raise ValueError('native prediction must cover the complete original volume')
            values = metric(prediction, torch.from_numpy(target.astype(np.float32, copy=False)), spacing_dhw=spacing)
        stats = tracker.snapshot()
        metrics = {name: float(values[name]) for name in METRICS}
        row = dict(source_name='bone', case_id=str(task['case_id']), global_class_id=int(task['global_class_id']),
                   local_class_id=int(task['local_class_id']), label_id=int(task['local_class_id']),
                   class_name=task['class_name'], gt_voxels=int(target.sum()), pred_voxels=int(prediction.sum().item()), **metrics)
        proof = dict(comparisons=stats['comparisons'], max_error=stats['comparison_max_abs_error'],
                     native_compared=stats['last_native_compared'],
                     provider_sha256=stats['provider_source_sha256'], native_source_sha256=stats['native_source_sha256'],
                     metrics=metrics, stats=stats)
        record = dict(task=dict(task), row=row, interaction=audit,
                      prompt_cache_name=path.name, prompt_sha256=config['prompt_hashes'][path.name],
                      resource=dict(resource), model=model, metric_execution=proof,
                      geometry=dict(shape_dhw=list(map(int, image.shape)), prediction_shape_dhw=list(map(int, prediction.shape)),
                                    spacing_dhw=list(map(float, spacing)), canonical=True, axial_first=True,
                                    image_cache_source=image_cache_source, case_image_cache_name=case_path.name))
        validate_record(record, task, config, job)
        commit(record)
        print(f"Bone nnInteractive {task['case_id']} label={task['local_class_id']} one_round=1 Dice={metrics['dice']:.6f}", flush=True)
        compare_first[0] = False
        del prediction, target, prompt
