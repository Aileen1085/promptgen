"""Isolated CT13 full-val v10.2 evaluation with user-selected source checkpoints."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('_ct13_source_best_base', ROOT / 'tools/ct13_full_validation_eval.py')
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
DEFAULT_OUT = 'output/v102_source_best_ct13_full6094_20261007'
REF_OUT = 'output/ct13_full_val_v92e360_v102e490_20261005'
CHECKPOINTS = {
    'A_E10': 'v10/output/v10_2_ct13_e490_decoder_frozen_control_10_20261005/20261005_155255/epoch010.pth',
    'hard_E15': 'v10/output/v10_2_ct13_hard_source_20_20261006/20261006_143104/epoch015.pth',
    'hard_E20': 'v10/output/v10_2_ct13_hard_source_20_20261006/20261006_143104/epoch020.pth',
}
EPOCHS = {'A_E10': 10, 'hard_E15': 15, 'hard_E20': 20}
SOURCE_SELECTION = {
    'totalseg': 'A_E10', 'amos': 'A_E10', 'magic': 'A_E10',
    'msd_task07': 'hard_E20', 'msd_task10': 'hard_E15', 'msd_task08': 'A_E10',
    'parse2022': 'hard_E20', 'topcow2024_cta': 'A_E10', 'lndb': 'hard_E15',
    'msd_task06': 'A_E10', 'covid19_20': 'hard_E15', 'kits23': 'hard_E15', 'lnq2023_lite': 'hard_E15',
}
EXPECTED = {'totalseg': 4034, 'amos': 1485, 'magic': 35, 'msd_task07': 40,
            'msd_task10': 20, 'msd_task08': 40, 'parse2022': 20, 'topcow2024_cta': 277,
            'lndb': 24, 'msd_task06': 13, 'covid19_20': 24, 'kits23': 58, 'lnq2023_lite': 24}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def validate_job_selection(job):
    expected = CHECKPOINTS[SOURCE_SELECTION[job['source']]]
    if job.get('checkpoint') != expected:
        raise ValueError(f"source checkpoint mismatch: {job['source']}")


def select_refs(rows, index):
    result = []
    for row in rows:
        ref = index.get(base.key(row))
        if not ref or not ref.get('prompt_cache_name'):
            raise ValueError(f'missing prompt reference: {base.key(row)}')
        result.append(ref)
    return result


def create_jobs(out):
    if (out / 'jobs.json').exists():
        raise FileExistsError('output already prepared; use launch/status, do not overwrite')
    manifest = base.build_manifest(base.read(ROOT / base.SPLIT))
    if manifest['per_source_task_counts'] != EXPECTED or manifest['expected_task_count'] != 6094:
        raise ValueError('unexpected full-val scope')
    refs = {}
    for path in sorted((ROOT / REF_OUT / 'prepared').glob('*/refs.json')):
        for row in base.read(path)['per_case_class']:
            k = base.key(row)
            if k in refs and refs[k] != row:
                raise ValueError(f'conflicting prompt reference: {k}')
            refs[k] = row
    select_refs(manifest['tasks'], refs)
    for row in manifest['tasks']:
        if not Path(row['image']).is_file() or not Path(row['label']).exists():
            raise FileNotFoundError(f'source files missing: {base.key(row)}')
    cache_root = ROOT / base.CACHE
    cache_names = {p.name for p in cache_root.rglob('*.npz')}
    missing = {r['prompt_cache_name'] for r in select_refs(manifest['tasks'], refs)} - cache_names
    if missing:
        raise FileNotFoundError(f'missing readonly prompt caches: {sorted(missing)[:5]}')
    weights = {name: {'path': path, 'sha256': sha(ROOT / path), 'expected_epoch': EPOCHS[name]}
               for name, path in CHECKPOINTS.items()}
    jobs = []
    for source in SOURCE_SELECTION:
        rows = [r for r in manifest['tasks'] if r['source_name'] == source]
        count = min({'totalseg': 64, 'amos': 24}.get(source, 4), len({r['case_id'] for r in rows}))
        for i in range(count):
            selected = base.case_shard(rows, count, i)
            name = f'v10_2_{source}_s{i:02d}of{count}'
            prepared = out / 'prepared' / name
            base.write(prepared / 'manifest.json', base.manifest_payload(selected))
            base.write(prepared / 'refs.json', {'per_case_class': select_refs(selected, refs)})
            jobs.append({'name': name, 'source': source, 'model': 'v10_2',
                         'checkpoint': CHECKPOINTS[SOURCE_SELECTION[source]],
                         'selection': SOURCE_SELECTION[source], 'task_count': len(selected),
                         'prepared': str(prepared), 'frame_batch': 48, 'attempt': 0})
    jobs.sort(key=lambda j: (int(j['name'].split('_s')[-1].split('of')[0]), j['task_count'], j['source']))
    base.write(out / 'manifest.json', manifest)
    base.write(out / 'jobs.json', jobs)
    base.write(out / 'input_audit.json', {
        'scope': 'full_heldout_6094', 'counts': EXPECTED, 'train_validation_overlap': 0,
        'split': base.SPLIT, 'split_sha256': sha(ROOT / base.SPLIT), 'checkpoints': weights,
        'source_selection': SOURCE_SELECTION, 'wrapper_sha256': sha(__file__),
        'base_sha256': sha(ROOT / 'tools/ct13_full_validation_eval.py'),
        'semantics': 'TotalSeg/AMOS legacy; all other sources expanded',
        'prompt': 'cached adaptive foreground scribble; two background points per plane',
        'dynamic_threshold': [0.35, 0.75], 'fallback': 0.60,
        'cache': 'readonly case CT and existing prompt caches; no volume export',
        'metrics': list(base.METRICS), 'reference_output': REF_OUT})
    print(f'PREPARED {len(jobs)} shards / 6094 tasks', flush=True)
    return jobs


def verify_inputs(out):
    audit = base.read(out / 'input_audit.json')
    if audit['wrapper_sha256'] != sha(__file__) or audit['base_sha256'] != sha(ROOT / 'tools/ct13_full_validation_eval.py'):
        raise ValueError('changed code signature; do not resume blindly')
    if audit['split_sha256'] != sha(ROOT / base.SPLIT):
        raise ValueError('changed split')
    for item in audit['checkpoints'].values():
        if item['sha256'] != sha(ROOT / item['path']):
            raise ValueError('changed checkpoint signature')
    for job in base.read(out / 'jobs.json'):
        validate_job_selection(job)


def run_model(job):
    validate_job_selection(job)
    base.WEIGHTS = {'v10_2': job['checkpoint']}
    base.run_model(job)


def worker(job_path):
    job = base.read(job_path)
    validate_job_selection(job)
    if not (Path(job['prepared']) / 'refs.json').is_file():
        raise ValueError('readonly prompt references not prepared')
    subprocess.run([sys.executable, str(Path(__file__).resolve()), 'model', '--job', str(job_path)], check=True, cwd=ROOT)


def merge(out):
    verify_inputs(out)
    rows = []
    for job in base.read(out / 'jobs.json'):
        run = out / 'jobs' / job['name']
        payload = base.read(run / 'complete_metrics.json')
        if payload['checkpoint'] != job['checkpoint']:
            raise ValueError('result checkpoint mismatch')
        audit = base.read(run / 'exact_prompt_audit.json')
        selected = base.read(Path(job['prepared']) / 'manifest.json')['tasks']
        base.validate_rows(payload['per_case_class'], selected)
        if {tuple(k) for k in audit['loaded_task_keys']} != {base.key(r) for r in selected}:
            raise ValueError('prompt audit task mismatch')
        if audit['dynamic_threshold_summary']['count'] != len(selected):
            raise ValueError('dynamic threshold coverage mismatch')
        if Path(audit['checkpoint']).resolve() != (ROOT / job['checkpoint']).resolve():
            raise ValueError('prompt audit checkpoint mismatch')
        for r in payload['per_case_class']:
            if not 0.35 <= float(r['mask_threshold_used']) <= 0.75:
                raise ValueError('threshold outside protocol')
        rows.extend(payload['per_case_class'])
    base.validate_rows(rows, base.read(out / 'manifest.json')['tasks'])
    counts = dict(Counter(base.key(r)[0] for r in rows))
    if counts != EXPECTED:
        raise ValueError('merged source coverage mismatch')
    per_source = {}
    for source in SOURCE_SELECTION:
        selected = [r for r in rows if base.key(r)[0] == source]
        per_source[source] = {'tasks': len(selected), 'checkpoint': CHECKPOINTS[SOURCE_SELECTION[source]],
                              **{m: sum(float(r[m]) for r in selected) / len(selected) for m in base.METRICS}}
    base.write(out / 'merged_metrics.json', {'model': 'v10_2_source_selected', 'task_count': len(rows),
        'per_source': per_source, 'per_case_class': rows, 'source_selection': SOURCE_SELECTION})
    print('MERGED 6094 tasks, all ten metrics finite', flush=True)


def status(out):
    jobs = base.read(out / 'jobs.json')
    rows = []
    complete = 0
    for job in jobs:
        run = out / 'jobs' / job['name']
        path = run / 'complete_metrics.json'
        if path.is_file():
            complete += 1
        else:
            path = run / 'all_metrics_live.json'
        if path.is_file():
            rows.extend(base.read(path)['per_case_class'])
    print({'completed_tasks': len({base.key(r) for r in rows}), 'total_tasks': 6094,
           'complete_shards': complete, 'total_shards': len(jobs),
           'per_source': dict(Counter(base.key(r)[0] for r in rows)),
           'scheduler': base.read(out / 'status.json') if (out / 'status.json').exists() else None})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('manifest', 'launch', 'worker', 'model', 'merge', 'status'))
    parser.add_argument('--out', default=DEFAULT_OUT)
    parser.add_argument('--gpus', default='2,3,4,5,6')
    parser.add_argument('--job')
    args = parser.parse_args()
    os.chdir(ROOT)
    out = (ROOT / args.out).resolve()
    if args.mode == 'manifest':
        create_jobs(out)
    elif args.mode == 'worker':
        worker(args.job)
    elif args.mode == 'model':
        run_model(base.read(args.job))
    elif args.mode == 'status':
        status(out)
    elif args.mode == 'merge':
        merge(out)
    else:
        verify_inputs(out)
        gpu_ids = [int(g) for g in args.gpus.split(',')]
        if set(gpu_ids) - {2, 3, 4, 5, 6}:
            raise ValueError('only GPUs2/3/4/5/6 permitted; active training0/1 and GPU7 excluded')
        base.__file__ = str(Path(__file__).resolve())
        base.merge = merge
        base.launch(out, gpu_ids)


if __name__ == '__main__':
    main()
