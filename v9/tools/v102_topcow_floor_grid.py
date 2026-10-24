"""Frozen E490 TopCoW full-val dynamic-floor confirmations; readonly shared inputs."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import v102_floor_grid_recovery as recovery

g = recovery.grid
b = g.b
SOURCE = 'topcow2024_cta'
FLOORS = (.55, .60, .65, .70)
OUT = 'output/v102_topcow277_floor_grid_e490_20261024'
HISTORY = 'output/ct13_full_val_v92e360_v102e490_20261005'


def select_tasks(rows, split):
    part = sorted([r for r in rows if r['source_name'] == SOURCE], key=b.key)
    ds = split['datasets'][SOURCE]
    cases = {r['case_id'] for r in part}
    if (len(part) != 277 or len({b.key(r) for r in part}) != 277 or len(cases) != 25
            or cases != {r['case_id'] for r in ds['val']}
            or cases & {r['case_id'] for r in ds['train']}):
        raise ValueError('requires complete held-out TopCoW25 CT/277 tasks, train0')
    return part


def make_jobs(rows):
    rows = sorted(rows, key=b.key)
    jobs = []
    for floor in FLOORS:
        parts = [rows[:1]] + [b.case_shard(rows[1:], 8, i) for i in range(8)]
        for i, tasks in enumerate(parts):
            if not tasks:
                continue
            name = SOURCE+'__floor%02d__' % round(floor*100)+('smoke' if i == 0 else 's%02d' % i)
            jobs.append(dict(name=name, model='v10_2', variant='baseline', source=SOURCE,
                floor=floor, tasks=tasks, task_count=len(tasks), frame_batch=16, attempt=0, smoke=i == 0))
    return sorted(jobs, key=lambda j: (not j['smoke'], j['floor'], j['name']))


def freeze_settings(args, floor):
    if (floor not in FLOORS or args.dynamic_threshold_min != .35
            or args.dynamic_threshold_max != .75 or args.dynamic_threshold_fallback != .60
            or args.prompt_logit_neighborhood_radius != 2
            or getattr(args, 'eval_memory_group_size', 1) != 1):
        raise ValueError('unexpected floor or original custom-parser settings')
    args.dynamic_threshold_min = floor
    return args


def validate_prompt_audit(a, tasks, source, floor):
    settings = a['dynamic_threshold']
    keys = [tuple(k) for k in a['loaded_task_keys']]
    thresholds = a['dynamic_threshold_rows']
    threshold_keys = [(r['source_name'], r['case_id'], int(r['class_id'])) for r in thresholds]
    expected = {b.key(r) for r in tasks}
    if (source != SOURCE or floor not in FLOORS or settings['minimum'] != floor
            or settings['maximum'] != .75 or settings['fallback'] != .60
            or a['background_mode'] != 'point' or a['sources'] != [SOURCE]
            or len(keys) != len(set(keys)) or set(keys) != expected
            or len(threshold_keys) != len(set(threshold_keys)) or set(threshold_keys) != expected
            or any(not g.threshold_valid(r, floor) for r in thresholds)):
        raise ValueError('TopCoW prompt/task/floor/fallback audit mismatch')


def baseline():
    rows = b.read(ROOT/HISTORY/'merged_metrics.json')['v10_2']['per_case_class']
    rows = select_tasks(rows, b.read(ROOT/b.SPLIT))
    b.validate_rows(rows, rows)
    return rows


def signature(out):
    value = g.previous.diagnostic.signature(out)
    for path in ('tools/v102_floor_grid.py', 'tools/v102_floor_grid_recovery.py',
                 'tools/v102_floor60_confirmation.py', 'tools/v102_topcow_floor_grid.py'):
        value['code'][path] = g.amos.sha(ROOT/path)
    value.update(floors=list(FLOORS), jobs=g.amos.sha(out/'jobs.json'),
        baseline=g.amos.sha(ROOT/HISTORY/'merged_metrics.json'),
        prepared={str(p.relative_to(out)):g.amos.sha(p) for p in sorted((out/'prepared').glob('*/*.json'))},
        protocol='TopCoW25CT277 fullval; E490 live; floors.55/.60/.65/.70; max.75 fallback.60 unclamped; '
                 'logit radius2 memory1 every AutoExpand stage; expanded semantics; exact readonly CT/FG/BG; fullmask10')
    return value


def verify(out):
    if signature(out) != b.read(out/'protocol.json'):
        raise ValueError('changed frozen TopCoW assets; independent recovery required')


def prepare(out):
    if out.exists():
        raise FileExistsError(out)
    rows = select_tasks(b.read(ROOT/HISTORY/'manifest.json')['tasks'], b.read(ROOT/b.SPLIT))
    b.validate_rows(baseline(), rows)
    refs = []
    for p in sorted((ROOT/HISTORY/'prepared').glob(SOURCE+'_s*of6/refs.json')):
        refs += b.read(p)['per_case_class']
    index = {b.key(r):r for r in refs}
    if len(index) != len(refs) or set(index) != {b.key(r) for r in rows}:
        raise ValueError('historical prompt reference scope mismatch')
    for ref in refs:
        g.amos.sha(ROOT/b.CACHE/'metadata/shared_adaptive_prompt_v1'/SOURCE/ref['prompt_cache_name'])
    b.write(out/'manifest.json', b.manifest_payload(rows))
    b.write(out/'refs.json', dict(per_case_class=[index[b.key(r)] for r in rows]))
    jobs = []
    for j in make_jobs(rows):
        tasks = j.pop('tasks'); prepared = out/'prepared'/j['name']; j['prepared'] = str(prepared)
        b.write(prepared/'manifest.json', b.manifest_payload(tasks))
        b.write(prepared/'refs.json', dict(per_case_class=[index[b.key(r)] for r in tasks]))
        jobs.append(j)
    b.write(out/'jobs.json', jobs)
    b.write(out/'protocol.json', signature(out))
    print('PREPARED36 shards; four real smokes pending; new1108 tasks', flush=True)


def summarize(rows, old, sixty):
    return g.summarize(rows, old, sixty)


def merge(out):
    verify(out)
    expected = select_tasks(b.read(out/'manifest.json')['tasks'], b.read(ROOT/b.SPLIT))
    by_floor = {f:[] for f in FLOORS}
    for j in b.read(out/'jobs.json'):
        spec = b.read(out/'job_specs'/(j['name']+'.json'))
        tasks = g.verify_job(spec); run = Path(spec['run'])
        a = b.read(run/'confirmation_audit.json')
        if (a['memory_group'] != 1 or a['dynamic_floor'] != j['floor'] or not a['actual_decode_stages']
                or any(s['memory_group'] != 1 or s['threshold_min'] != j['floor'] for s in a['actual_decode_stages'])):
            raise ValueError('stage-specific TopCoW settings mismatch')
        rows = b.read(run/'complete_metrics.json')['per_case_class']
        b.validate_rows(rows, tasks); by_floor[j['floor']] += rows
    for rows in by_floor.values():
        b.validate_rows(rows, expected)
    old = baseline()
    result = {format(f, '.2f'):summarize(rows, old, by_floor[.60]) for f,rows in by_floor.items()}
    b.write(out/'merged_metrics.json', dict(floors=result, baseline035=g.summarize(old, old, by_floor[.60]),
        new_predictions=1108, readonly_baseline_tasks=277,
        selection='all four frozen candidates; development validation, not unbiased test'))
    print('MERGED four floors each277; readonly baseline277', flush=True)


def configure():
    # All bindings are process-local; no historical file or protocol is changed.
    g.FLOORS = FLOORS; g.OUT = OUT; g.freeze_settings = freeze_settings
    g.validate_prompt_audit = validate_prompt_audit; g.signature = signature
    g.verify = verify; g.merge = merge; g.configure = configure
    g.amos.verify = verify; g.amos.merge = merge; g.amos.__file__ = __file__


def status(out):
    jobs = b.read(out/'jobs.json')
    done = [j for j in jobs if (out/'jobs'/j['name']/'confirmation_audit.json').exists()]
    return dict(jobs=len(jobs), audited=len(done), audited_tasks=sum(j['task_count'] for j in done),
        expected_new=1108,
        per_floor={str(f):sum(j['task_count'] for j in done if j['floor'] == f) for f in FLOORS},
        phase=b.read(out/'queue_phase.json') if (out/'queue_phase.json').exists() else None,
        scheduler=b.read(out/'status.json') if (out/'status.json').exists() else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare','worker','launch','status','merge'))
    parser.add_argument('--out', default=OUT); parser.add_argument('--job')
    parser.add_argument('--gpus', default='0,2,3,4,5,6')
    args = parser.parse_args(); out = (ROOT/args.out).resolve(); configure()
    if args.mode == 'prepare':
        prepare(out)
    elif args.mode == 'worker':
        g.run(b.read(args.job))
    elif args.mode == 'merge':
        merge(out)
    elif args.mode == 'status':
        import json
        print(json.dumps(status(out), indent=2))
    else:
        ids = [int(x) for x in args.gpus.split(',')]
        if len(ids) != len(set(ids)) or not ids or not set(ids) <= {0,2,3,4,5,6}:
            raise ValueError('invalid GPU list')
        g.amos.launch(out, ids)


if __name__ == '__main__':
    main()
