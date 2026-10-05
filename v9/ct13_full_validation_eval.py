"""Complete held-out CT13 evaluation; historical model entries remain unchanged.

Only lightweight manifests, prompt references and metric JSONs are written.
Model workers run in separate processes to isolate legacy import/CLI hooks.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
METRICS = ('dice', 'mean_slice_dice', 'iou', 'precision', 'recall', 'specificity',
           'pred_gt_ratio', 'mean_slice_nsd_1mm', 'mean_slice_nsd_2mm', 'mean_slice_nsd_3mm')
NEW_SOURCES = ('lndb', 'msd_task06', 'covid19_20', 'kits23', 'lnq2023_lite')
SPLIT = 'configs/ct13_v10_2_split_20260928.json'
CACHE = 'v10/.cache/totalseg_sam2_promptgen_v10_prompt_roi_full_v1'
WEIGHTS = {
    'v9_2': 'output/sam2_v9_2_ct13_semantic_e380_400_20260928/20260928_200027/epoch360_dice0.7817.pth',
    'v10_2': 'v10/output/v10_2_ct13_e470_zero_delta_e510_20261001/20261001_215853/epoch490.pth',
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temp, path)


def key(row):
    return (str(row.get('source_name', row.get('dataset'))), str(row['case_id']),
            int(row.get('global_class_id', row.get('class_id'))))


def manifest_payload(rows):
    return {'manifest_kind': 'independent_eight_source_validation_v1',
            'scope': 'ct13_all_previously_split_heldout_cases', 'expected_task_count': len(rows),
            'per_source_task_counts': dict(Counter(key(r)[0] for r in rows)), 'tasks': rows}


def build_manifest(split):
    catalogue = {int(r['global_class_id']): r for r in split['class_catalog']}
    rows = []
    for source in split['source_order']:
        dataset = split['datasets'][source]
        train_ids = {str(r['case_id']) for r in dataset['train']}
        val_ids = [str(r['case_id']) for r in dataset['val']]
        if train_ids & set(val_ids):
            raise ValueError(f'train/validation overlap in {source}')
        if len(val_ids) != len(set(val_ids)):
            raise ValueError(f'duplicate validation cases in {source}')
        for entry in dataset['val']:
            image = str(entry.get('image', entry.get('image_path')))
            label = str(entry.get('label', entry.get('label_path')))
            if source == 'magic':
                category, case = str(entry['case_id']).split('/', 1)
                data = Path(image).parents[3]
                image = str(data / 'MAGIC-CT' / category / 'scans' / f'{case}.nrrd')
                label = str(data / 'MAGIC-CT' / category / 'segmentations' / f'{case}.seg.nrrd')
            for class_id in entry['classes']:
                meta = catalogue[int(class_id)]
                if bool(meta.get('not_applicable', False)) or int(class_id) in (40, 41):
                    continue
                if str(meta['dataset']) != source:
                    raise ValueError(f'class source mismatch: {source}/{class_id}')
                rows.append({'source_name': source, 'case_id': str(entry['case_id']),
                             'global_class_id': int(class_id), 'local_class_id': int(meta['local_class_id']),
                             'class_name': str(meta['class_name']), 'image': image, 'label': label,
                             'category': entry.get('category'), 'split_role': 'validation'})
    if len({key(r) for r in rows}) != len(rows):
        raise ValueError('duplicate validation tasks')
    return manifest_payload(rows)


def case_shard(rows, count, index):
    if count < 1 or not 0 <= index < count:
        raise ValueError('invalid shard')
    grouped = defaultdict(list)
    for row in rows:
        grouped[key(row)[:2]].append(row)
    parts = [[] for _ in range(count)]
    loads = [0] * count
    for _, values in sorted(grouped.items(), key=lambda pair: (-len(pair[1]), pair[0])):
        selected = min(range(count), key=lambda i: (loads[i], i))
        parts[selected].extend(values)
        loads[selected] += len(values)
    return sorted(parts[index], key=key)


def validate_rows(rows, expected):
    keys = [key(r) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('duplicate metric tasks')
    if set(keys) != {key(r) for r in expected}:
        raise ValueError('metric task coverage mismatch')
    for row in rows:
        if any(m not in row or not math.isfinite(float(row[m])) for m in METRICS):
            raise ValueError(f'missing/nonfinite metric in {key(row)}')


def nn_modules(gpu):
    sys.argv = [sys.argv[0], '--gpu', str(gpu)]
    folder = ROOT / 'nninteractive_test' / 'infer'
    sys.path.insert(0, str(folder))
    protocol = importlib.import_module('eight_source_adaptive_protocol')
    protocol.EIGHT_SOURCES = tuple(read(ROOT / SPLIT)['source_order'])
    evaluator = importlib.import_module('infer_eight_source_adaptive_nninteractive')
    return protocol, evaluator


def prepare(job):
    import fcntl
    import numpy as np
    protocol, evaluator = nn_modules(job['gpu'])
    prepared = Path(job['prepared'])
    prepared.mkdir(parents=True, exist_ok=True)
    with (prepared / 'prepare.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (prepared / 'refs.json').is_file():
            return
        tasks = protocol.load_independent_validation_tasks(prepared / 'manifest.json')
        args = evaluator.parse_args()
        args.shared_case_cache_root = str(ROOT / CACHE)
        adaptive = protocol.load_adaptive_module(ROOT / 'v9_2_adaptive_scribble.py')
        from multisource_split_protocol import shared_case_cache_path
        refs = []
        integer_label = None
        previous = None
        for task in tasks:
            if task.source_name not in ('totalseg', 'magic'):
                if previous != task.label_path:
                    integer_label = evaluator.load_raw_axial_volume(str(task.label_path), canonical=True).astype(np.int32)
                    previous = task.label_path
                target = protocol.load_task_target(task, source_loader=lambda _path: integer_label)
            else:
                target = protocol.load_task_target(task,
                    source_loader=lambda path: evaluator.load_raw_axial_volume(path, canonical=True),
                    magic_target_loader=lambda path, reference: evaluator.read_magic_target(path, reference_image_path=reference))
            if not np.any(target):
                raise ValueError(f'empty GT in full held-out manifest: {task}')
            path = evaluator.adaptive_prompt_cache_path(ROOT / CACHE, task, protocol.target_source_path(task),
                min_radius=1, max_radius=20, width_fraction=.15,
                point_seed=evaluator._stable_seed(2026, task.source_name, task.case_id, task.global_class_id),
                background_points_per_plane=2, negative_point_mode='bbox_outside')
            case_name = shared_case_cache_path(task.image_path, ROOT / CACHE, task.source_name).name
            evaluator.load_or_create_adaptive_prompt_package(path,
                lambda: evaluator._build_prompt_package(args, task, target, case_name, adaptive))
            refs.append({'source_name': task.source_name, 'case_id': task.case_id,
                         'global_class_id': int(task.global_class_id), 'prompt_protocol': evaluator.PROMPT_PROTOCOL,
                         'prompt_cache_name': path.name})
            print(f'Prepared {task.source_name}/{task.case_id}/{task.global_class_id} {len(refs)}/{len(tasks)}', flush=True)
        write(prepared / 'refs.json', {'per_case_class': refs})


def metric_module():
    spec = importlib.util.spec_from_file_location('_ct13_unified_metrics', ROOT / 'utils' / 'metrics.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_model(job):
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(job['gpu'])
    os.environ['V9_SHARED_CACHE_WRITE_POLICY'] = 'read_only'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    prepared = Path(job['prepared'])
    rows = read(prepared / 'manifest.json')['tasks']
    run = Path(job['run'])
    run.mkdir(parents=True, exist_ok=True)
    import torch
    torch.set_num_threads(2)
    source_order = tuple(read(ROOT / SPLIT)['source_order'])
    if job['model'] == 'nninteractive':
        protocol, ev = nn_modules(job['gpu'])
        import multisource_split_protocol as cache
        def readonly(image, root, source, loader, **_kwargs):
            path = cache.shared_case_cache_path(image, root, source)
            cached = cache._valid_case_cache(path)
            return (cached, 'cache_hit', path) if cached is not None else (loader(str(image)), 'source_memory_only', path)
        ev.load_or_create_shared_case_ct = readonly
        args = ev.parse_args()
        args.task_manifest = str(prepared / 'manifest.json')
        args.model_dir = str(ROOT.parent / 'nninteractive_test_amos/checkpoints/nnInteractive_v1.0')
        args.shared_case_cache_root = str(ROOT / CACHE)
        args.adaptive_scribble_module = str(ROOT / 'v9_2_adaptive_scribble.py')
        args.out_dir = str(run.parent); args.run_name = run.name
        args.source = job['source']; args.torch_threads = 2
        ev.compute_binary_metrics = metric_module().binary_prompt_metrics_from_masks
        ev.run_inference(args)
        output_rows = read(run / 'metrics.json')['per_case_class']
    else:
        import v9_2_extended_protocol as protocol
        protocol.SOURCE_ORDER = source_order
        split = read(ROOT / SPLIT)
        common = ['--independent-task-manifest', str(prepared / 'manifest.json'),
            '--nninteractive-metrics', str(prepared / 'refs.json'), '--shared-adaptive-prompt-root', str(ROOT / CACHE),
            '--source', job['source'], '--eval-out-dir', str(run.parent), '--run-name', run.name,
            '--gpu', str(job['gpu']), '--sam2-frame-batch-size', str(job['frame_batch']),
            '--base-split-json', str(ROOT / SPLIT)]
        output_rows = []
        def save_row(metric, task, **extra):
            row = {**task, **{m: float(metric[m]) for m in METRICS}, **extra}
            output_rows.append(row)
            write(run / 'all_metrics_live.json', {'per_case_class': output_rows})
            print(f"CT13 {job['model']} {key(task)} {len(output_rows)}/{len(rows)} Dice={row['dice']:.5f} Axial={row['mean_slice_dice']:.5f}", flush=True)
        if job['model'] == 'v10_2':
            sys.path.insert(0, str(ROOT / 'v10'))
            sys.argv = [sys.argv[0], *common, '--weights', str(ROOT / WEIGHTS['v10_2']), '--background-mode', 'point']
            import v10.finetune_totalseg_amos_magic_v10_2_joint as entry
            import v10_2_data as data
            from ct13_training_entry import install_v10_protocol
            install_v10_protocol(entry, data, split)
            original_text = entry.conditioning_text_from_metadata
            from finetune_multisource_sam2_v9_2_precision_mixed_semantic_eval import _legacy_text
            entry.conditioning_text_from_metadata = lambda r: _legacy_text(r) if r['dataset'] in ('totalseg', 'amos') else original_text(r)
            import v10.infer.infer_amos_sam2_promptgen_v10 as scorer
            scorer.binary_prompt_metrics_from_masks = metric_module().binary_prompt_metrics_from_masks
            original_score = scorer.restore_and_score_full_volume
            def score(*args, **kwargs):
                result = original_score(*args, **kwargs)
                save_row(result[0], sorted(rows, key=key)[len(output_rows)])
                return result
            scorer.restore_and_score_full_volume = score
            import v10.infer.infer_eight_source_adaptive_v10_2 as ev
            ev.main()
            output_rows = read(run / 'validation_only_metrics.json')['per_case_class']
        else:
            sys.argv = [sys.argv[0], *common, '--checkpoint-dir', str((ROOT / WEIGHTS['v9_2']).parent),
                '--v9-2-weights', str(ROOT / WEIGHTS['v9_2']), '--allow-uniform-weight', '--exact-background-mode', 'point']
            import finetune_multisource_sam2_v9_2_precision as original
            from ct13_training_entry import install_v9_protocol
            install_v9_protocol(original, protocol, split)
            from v9_family_shared_cache_entry import install_multidataset_factories
            install_multidataset_factories(original.multi)
            import finetune_multisource_sam2_v9_2_precision_mixed_semantic_eval as mixed
            sys.modules['finetune_multisource_sam2_v9_2_precision'] = mixed
            import v9_totalseg_runtime as runtime
            unified = metric_module()
            def binary(prediction, target, spacing=None, **_kwargs):
                import numpy as np
                values = unified.binary_prompt_metrics_from_masks(torch.from_numpy(np.asarray(prediction)),
                    torch.from_numpy(np.asarray(target)), spacing_dhw=spacing)
                return {**values, 'volume_nsd_1mm': float('nan'), 'volume_hd95_mm': float('nan')}
            runtime.base.binary_volume_metrics_from_prediction = binary
            import infer.v9_2_dynamic_validation_patch as dynamic
            installer = dynamic.install_dynamic_validation_patch
            def install(rt, audit_rows, **settings):
                installer(rt, audit_rows, **settings)
                adaptive = rt._evaluate_adaptive_prompt_roi_task
                def final(*args, **kwargs):
                    metric = adaptive(*args, **kwargs)
                    current_key = tuple(prompt_audit['current']['loaded_task_keys'][-1])
                    task = next(r for r in rows if key(r) == current_key)
                    if int(args[5]) != int(task['global_class_id']):
                        raise RuntimeError('metric task/class alignment mismatch')
                    save_row(metric, task, mask_threshold_used=audit_rows[-1]['threshold'])
                    return metric
                rt._evaluate_adaptive_prompt_roi_task = final
            dynamic.install_dynamic_validation_patch = install
            import infer.infer_eight_source_adaptive_v9_2 as ev
            prompt_audit = {}
            exact_factory = ev._install_exact_prompt_factory
            def bind_exact_audit(entry, tasks, index, root, audit, mode):
                prompt_audit['current'] = audit
                return exact_factory(entry, tasks, index, root, audit, mode)
            ev._install_exact_prompt_factory = bind_exact_audit
            ev.checkpoint_plan = lambda _p: {s: {'path': str(ROOT / WEIGHTS['v9_2']), 'epoch': 360, 'selection': 'user_explicit'} for s in source_order}
            append = ev._append_flag
            def append_training_context(arguments, flag, *values):
                append(arguments, flag, *( (192,) if flag in ('--max-frames', '--validation-window-frames') else values))
            ev._append_flag = append_training_context
            ev.main()
    validate_rows(output_rows, rows)
    write(run / 'complete_metrics.json', {'model': job['model'], 'checkpoint': WEIGHTS.get(job['model']),
        'metric_protocol': 'final_complete_mask; axial_union_nonempty; NSD_axial_physical_mm', 'per_case_class': output_rows})


def worker(job):
    command = [sys.executable, str(Path(__file__).resolve())]
    for mode in ('prepare', 'model'):
        subprocess.run([*command, mode, '--job', str(job)], check=True, cwd=ROOT)


def create_jobs(out):
    split = read(ROOT / SPLIT)
    manifest = build_manifest(split)
    for row in manifest['tasks']:
        if not Path(row['image']).is_file() or not Path(row['label']).exists():
            raise FileNotFoundError(f'missing files for {key(row)}')
    write(out / 'manifest.json', manifest)
    jobs = []
    limits = {'totalseg': 24, 'amos': 12, 'magic': 7, 'topcow2024_cta': 6}
    for source in split['source_order']:
        source_rows = [r for r in manifest['tasks'] if r['source_name'] == source]
        count = min(limits.get(source, 3), len({r['case_id'] for r in source_rows}))
        for i in range(count):
            selected = case_shard(source_rows, count, i)
            prepared = out / 'prepared' / f'{source}_s{i:02d}of{count}'
            write(prepared / 'manifest.json', manifest_payload(selected))
            models = ['v9_2', 'v10_2'] + (['nninteractive'] if source in NEW_SOURCES else [])
            for model in models:
                name = f'{model}_{source}_s{i:02d}of{count}'
                jobs.append({'name': name, 'model': model, 'source': source, 'prepared': str(prepared),
                             'task_count': len(selected), 'frame_batch': 16, 'attempt': 0})
    # Interleave sources and methods so long TotalSeg shards do not block others.
    jobs.sort(key=lambda j: (int(j['name'].split('_s')[-1].split('of')[0]), j['task_count'], j['model']))
    write(out / 'jobs.json', jobs)
    write(out / 'input_audit.json', {'split': SPLIT, 'split_sha256': hashlib.sha256((ROOT / SPLIT).read_bytes()).hexdigest(),
        'counts': manifest['per_source_task_counts'], 'train_validation_overlap': 0,
        'checkpoints': {m: {'path': p, 'sha256': hashlib.sha256((ROOT / p).read_bytes()).hexdigest()} for m, p in WEIGHTS.items()},
        'semantics': 'TotalSeg/AMOS legacy; other sources metadata-expanded', 'cache': 'CT read-only; lightweight shared prompts',
        'metrics': list(METRICS), 'v9_prompt_context_frames': 192})
    return jobs


def merge(out):
    jobs = read(out / 'jobs.json')
    manifest = read(out / 'manifest.json')['tasks']
    result = {}
    for model in ('v9_2', 'v10_2', 'nninteractive'):
        rows = []
        for job in jobs:
            if job['model'] == model:
                rows.extend(read(out / 'jobs' / job['name'] / 'complete_metrics.json')['per_case_class'])
        expected = [r for r in manifest if model != 'nninteractive' or r['source_name'] in NEW_SOURCES]
        validate_rows(rows, expected)
        by_source = {}
        for source in sorted({key(r)[0] for r in rows}):
            selected = [r for r in rows if key(r)[0] == source]
            by_source[source] = {'tasks': len(selected), **{m: sum(float(r[m]) for r in selected)/len(selected) for m in METRICS}}
        result[model] = {'task_count': len(rows), 'per_source': by_source, 'per_case_class': rows}
    write(out / 'merged_metrics.json', result)


def launch(out, gpu_ids):
    import fcntl
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'launcher.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    jobs = read(out / 'jobs.json') if (out / 'jobs.json').is_file() else create_jobs(out)
    pending = [j for j in jobs if not (out / 'jobs' / j['name'] / 'complete_metrics.json').is_file()]
    running = {}
    failed = []
    while pending or running:
        for gpu, (proc, job, handle) in list(running.items()):
            if proc.poll() is None:
                continue
            handle.close()
            del running[gpu]
            if proc.returncode:
                tail = Path(job['log']).read_text(errors='replace')[-10000:]
                if ('out of memory' in tail.lower() or 'OutOfMemoryError' in tail) and job['frame_batch'] > 1:
                    job['frame_batch'] = max(1, job['frame_batch']//2)
                    job['attempt'] += 1
                    pending.insert(0, job)
                    print(f"OOM retry {job['name']} frame_batch={job['frame_batch']}", flush=True)
                else:
                    failed.append({'name': job['name'], 'returncode': proc.returncode, 'log': job['log']})
                    print(f"FAILED {job['name']} {job['log']}", flush=True)
            else:
                print(f"DONE {job['name']} tasks={job['task_count']}", flush=True)
        query = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free,uuid', '--format=csv,noheader,nounits'], text=True)
        available = {int(line.split(',')[0]): (int(line.split(',')[1]), line.split(',')[2].strip()) for line in query.splitlines()}
        for gpu in gpu_ids:
            if gpu in running or not pending or available[gpu][0] < 45000:
                continue
            job = pending.pop(0)
            job.update(gpu=gpu, gpu_uuid=available[gpu][1], run=str(out / 'jobs' / job['name']))
            jobfile = out / 'job_specs' / f"{job['name']}.json"
            job['log'] = str(out / 'logs' / f"{job['name']}_a{job['attempt']}.log")
            write(jobfile, job)
            Path(job['log']).parent.mkdir(parents=True, exist_ok=True)
            handle = Path(job['log']).open('w')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
            proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'worker', '--job', str(jobfile)], cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            running[gpu] = (proc, job, handle)
            print(f"START gpu={gpu} uuid={job['gpu_uuid']} pid={proc.pid} {job['name']} tasks={job['task_count']}", flush=True)
        write(out / 'status.json', {'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'pending': len(pending),
            'running': {str(g): {'name': j['name'], 'pid': p.pid, 'log': j['log']} for g, (p, j, h) in running.items()}, 'failed': failed})
        if pending or running:
            time.sleep(10)
    if failed:
        raise RuntimeError(f'{len(failed)} jobs failed; see status.json, successful outputs retained')
    merge(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'model', 'worker', 'launch', 'manifest', 'merge'))
    parser.add_argument('--job')
    parser.add_argument('--out', default='output/ct13_full_val_v92e360_v102e490_20261005')
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6')
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.mode in ('prepare', 'model'):
        (prepare if args.mode == 'prepare' else run_model)(read(args.job))
    elif args.mode == 'worker':
        worker(args.job)
    else:
        out = (ROOT / args.out).resolve()
        if args.mode == 'launch':
            launch(out, [int(g) for g in args.gpus.split(',')])
        elif args.mode == 'manifest':
            create_jobs(out)
        else:
            merge(out)


if __name__ == '__main__':
    main()
