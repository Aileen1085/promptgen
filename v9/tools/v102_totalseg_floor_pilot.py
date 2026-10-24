"""E490 TotalSeg stratified202-task pilot; native full-mask floor grid."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import v102_topcow_floor_grid as core

g, b = core.g, core.b
SOURCE = 'totalseg'
FLOORS = (.45, .50, .55, .60, .65, .70, .75)
SEED = 'totalseg-floor-pilot-20261024-v1'
N = 202
OUT = 'output/v102_totalseg202_floor_grid_e490_20261024'
HISTORY = core.HISTORY


def class_quotas(counts, target):
    total = sum(counts.values())
    if not counts or not len(counts) <= target <= total or min(counts.values()) < 1:
        raise ValueError('infeasible stratified quota')
    quota = {c:max(1, target*n//total) for c,n in counts.items()}
    while sum(quota.values()) > target:
        c = max((c for c in counts if quota[c] > 1),
                key=lambda c:(quota[c]*total-target*counts[c], -c))
        quota[c] -= 1
    while sum(quota.values()) < target:
        c = max((c for c in counts if quota[c] < counts[c]),
                key=lambda c:(target*counts[c]-quota[c]*total, -c))
        quota[c] += 1
    return quota


def select_subset(rows, split):
    full = sorted([r for r in rows if b.key(r)[0] == SOURCE], key=b.key)
    ds = split['datasets'][SOURCE]
    cases = {r['case_id'] for r in full}
    counts = Counter(b.key(r)[2] for r in full)
    if (len(full) != 4034 or len({b.key(r) for r in full}) != 4034 or len(counts) != 115
            or {40,41} & set(counts) or cases != {r['case_id'] for r in ds['val']}
            or cases & {r['case_id'] for r in ds['train']}):
        raise ValueError('requires exact TotalSeg4034 held-out tasks/115 classes/train0')
    quota = class_quotas(counts, N)
    grouped = defaultdict(list)
    for r in full:
        grouped[b.key(r)[2]].append(r)
    def rank(r):
        encoded = json.dumps([SEED, *b.key(r)], separators=(',', ':')).encode()
        return hashlib.sha256(encoded).hexdigest(), b.key(r)
    selected = [r for c in sorted(grouped) for r in sorted(grouped[c], key=rank)[:quota[c]]]
    return sorted(selected, key=b.key)


def make_jobs(rows):
    if len(rows) != N or len({b.key(r) for r in rows}) != N:
        raise ValueError('pilot jobs require202 unique tasks')
    rows = sorted(rows, key=b.key); jobs = []
    for floor in FLOORS:
        parts = [rows[:1]] + [b.case_shard(rows[1:], 12, i) for i in range(12)]
        for i,tasks in enumerate(parts):
            if not tasks: continue
            name = SOURCE+'__floor%02d__' % round(floor*100)+('smoke' if i==0 else 's%02d' % i)
            jobs.append(dict(name=name, model='v10_2', variant='baseline', source=SOURCE,
                floor=floor, tasks=tasks, task_count=len(tasks), frame_batch=16, attempt=0, smoke=i==0))
    return sorted(jobs, key=lambda j:(not j['smoke'], j['floor'], j['name']))


def freeze_settings(args, floor):
    if (floor not in FLOORS or args.dynamic_threshold_min != .35
            or args.dynamic_threshold_max != .75 or args.dynamic_threshold_fallback != .60
            or args.prompt_logit_neighborhood_radius != 2 or getattr(args,'eval_memory_group_size',1) != 1):
        raise ValueError('unexpected original parser settings')
    args.dynamic_threshold_min = floor
    return args


def validate_prompt_audit(a, tasks, source, floor):
    keys = [tuple(k) for k in a['loaded_task_keys']]
    threshold_rows = a['dynamic_threshold_rows']; settings = a['dynamic_threshold']
    threshold_keys = [(r['source_name'],r['case_id'],int(r['class_id'])) for r in threshold_rows]
    expected = {b.key(r) for r in tasks}
    if (source != SOURCE or floor not in FLOORS or a['sources'] != [SOURCE]
            or a['background_mode'] != 'point' or settings['minimum'] != floor
            or settings['maximum'] != .75 or settings['fallback'] != .60
            or len(keys) != len(set(keys)) or set(keys) != expected
            or len(threshold_keys) != len(set(threshold_keys)) or set(threshold_keys) != expected
            or any(not g.threshold_valid(r,floor) for r in threshold_rows)):
        raise ValueError('TotalSeg pilot exact prompt/floor/fallback coverage mismatch')


def baseline():
    rows = b.read(ROOT/HISTORY/'merged_metrics.json')['v10_2']['per_case_class']
    rows = select_subset(rows, b.read(ROOT/b.SPLIT)); b.validate_rows(rows, rows)
    return rows


def payload(rows):
    value = b.manifest_payload(rows)
    value['scope'] = 'ct13_totalseg_stratified202_tasks_pilot_v1'
    return value


def signature(out):
    value = core.signature(out)
    value['code']['tools/v102_totalseg_floor_pilot.py'] = g.amos.sha(Path(__file__))
    value.update(floors=list(FLOORS), selection=g.amos.sha(out/'selection.json'),
        protocol='TotalSeg202/4034 stratified task pilot; E490 live; floors.45-.75step.05; '
            'max.75 fallback.60 unclamped; logit radius2 memory1 every stage; OLD semantics; '
            'exact readonly CT/FG/BGpoint; fullmask10; fixed hash selection; preliminary val')
    return value


def verify(out):
    if signature(out) != b.read(out/'protocol.json'):
        raise ValueError('changed signed pilot assets; independent recovery required')


def prepare(out):
    if out.exists(): raise FileExistsError(out)
    split = b.read(ROOT/b.SPLIT); full = b.read(ROOT/HISTORY/'manifest.json')['tasks']
    rows = select_subset(full,split)
    current = select_subset(b.build_manifest(split)['tasks'],split)
    if [b.key(r) for r in rows] != [b.key(r) for r in current]:
        raise ValueError('historical and current held-out task selection differ')
    b.validate_rows(baseline(), rows)
    refs = [r for p in sorted((ROOT/HISTORY/'prepared').glob('totalseg_*/refs.json'))
            for r in b.read(p)['per_case_class']]
    index = {b.key(r):r for r in refs}
    expected_full = {b.key(r) for r in full if b.key(r)[0] == SOURCE}
    if len(index) != len(refs) or set(index) != expected_full:
        raise ValueError('historical full prompt reference scope mismatch')
    refs = [index[b.key(r)] for r in rows]
    for ref in refs:
        g.amos.sha(ROOT/b.CACHE/'metadata/shared_adaptive_prompt_v1'/SOURCE/ref['prompt_cache_name'])
    counts = Counter(b.key(r)[2] for r in full if b.key(r)[0] == SOURCE)
    b.write(out/'selection.json',dict(seed=SEED,method='class proportional minimum1/largest deficit; SHA256 task rank',
        original_tasks=4034,selected_tasks=N,classes=115,full_class_counts=dict(counts),
        class_quotas=class_quotas(counts,N),selected_keys=[list(b.key(r)) for r in rows],
        selected_cases=sorted({r['case_id'] for r in rows}),train_overlap=0,
        selected_without_model_metrics=True))
    b.write(out/'manifest.json',payload(rows)); b.write(out/'refs.json',dict(per_case_class=refs))
    jobs = []
    for j in make_jobs(rows):
        tasks = j.pop('tasks'); prepared = out/'prepared'/j['name']; j['prepared'] = str(prepared)
        b.write(prepared/'manifest.json',payload(tasks))
        b.write(prepared/'refs.json',dict(per_case_class=[index[b.key(r)] for r in tasks])); jobs.append(j)
    b.write(out/'jobs.json',jobs); b.write(out/'protocol.json',signature(out))
    print('PREPARED',len(jobs),'shards;202 tasks/115 classes; seven real smokes pending',flush=True)


def summarize(rows, old, sixty):
    return g.summarize(rows,old,sixty)


def merge(out):
    verify(out); expected = b.read(out/'manifest.json')['tasks']
    chosen = select_subset(b.read(ROOT/HISTORY/'manifest.json')['tasks'],b.read(ROOT/b.SPLIT))
    if [b.key(r) for r in expected] != [b.key(r) for r in chosen]:
        raise ValueError('pilot selection changed')
    by_floor = {f:[] for f in FLOORS}
    for j in b.read(out/'jobs.json'):
        spec = b.read(out/'job_specs'/(j['name']+'.json')); tasks = g.verify_job(spec); run = Path(spec['run'])
        a = b.read(run/'confirmation_audit.json')
        if (a['memory_group'] != 1 or a['dynamic_floor'] != j['floor'] or not a['actual_decode_stages']
                or any(s['memory_group'] != 1 or s['threshold_min'] != j['floor'] for s in a['actual_decode_stages'])):
            raise ValueError('actual memory/floor stage mismatch')
        rows = b.read(run/'complete_metrics.json')['per_case_class']; b.validate_rows(rows,tasks)
        by_floor[j['floor']] += rows
    for rows in by_floor.values(): b.validate_rows(rows,expected)
    old = baseline()
    result = {format(f,'.2f'):summarize(rows,old,by_floor[.60]) for f,rows in by_floor.items()}
    b.write(out/'merged_metrics.json',dict(floors=result,baseline035=summarize(old,old,by_floor[.60]),
        new_predictions=N*len(FLOORS),readonly_baseline_tasks=N,
        selection='frozen202/4034 class-stratified pilot; development val, not full or unbiased test'))
    print('MERGED seven floors each202; readonly paired baseline202',flush=True)


def configure():
    g.FLOORS=FLOORS; g.OUT=OUT; g.freeze_settings=freeze_settings; g.validate_prompt_audit=validate_prompt_audit
    g.signature=signature; g.verify=verify; g.merge=merge; g.configure=configure
    g.amos.verify=verify; g.amos.merge=merge; g.amos.__file__=__file__


def status(out):
    jobs=b.read(out/'jobs.json'); done=[j for j in jobs if (out/'jobs'/j['name']/'confirmation_audit.json').exists()]
    return dict(jobs=len(jobs),audited=len(done),audited_tasks=sum(j['task_count'] for j in done),expected_new=N*len(FLOORS),
        per_floor={str(f):sum(j['task_count'] for j in done if j['floor']==f) for f in FLOORS},
        phase=b.read(out/'queue_phase.json') if (out/'queue_phase.json').exists() else None,
        scheduler=b.read(out/'status.json') if (out/'status.json').exists() else None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','worker','launch','status','merge'))
    parser.add_argument('--out',default=OUT); parser.add_argument('--job'); parser.add_argument('--gpus',default='0,2,3,4,5,6')
    a=parser.parse_args(); out=(ROOT/a.out).resolve(); configure()
    if a.mode=='prepare': prepare(out)
    elif a.mode=='worker': g.run(b.read(a.job))
    elif a.mode=='merge': merge(out)
    elif a.mode=='status': print(json.dumps(status(out),indent=2))
    else:
        ids=[int(x) for x in a.gpus.split(',')]
        if not ids or len(ids)!=len(set(ids)) or not set(ids)<={0,2,3,4,5,6}: raise ValueError('invalid GPU list')
        g.amos.launch(out,ids)


if __name__=='__main__': main()
