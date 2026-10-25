"""PAM AMOS100CT/1485 tasks with the unchanged nine-source probability fusion."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import pam_nine_source_probability as q

n = q.n
DEFAULT = 'output/pam_probability_amos_full1485_20261025'
PILOT = ROOT/'output/pam_amos_soft_fusion_pilot119_20261023/merged_metrics.json'
PROTOCOL = dict(q.PROTOCOL, scope='AMOS_complete_val_100CT_1485_train0',
                smoke='first_whole_case_then24_whole_case_shards',
                pilot_control='all119_overlap_ten_metrics_atol1e-6_boxes_and_propagation_exact')
_native_assets = q.assets_signature


def validate_scope(tasks, split):
    q.p.validate_scope(tasks, split, 1485, 100)


def make_jobs(tasks):
    tasks = sorted(tasks, key=n.key)
    first = tasks[0]['case_id']
    parts = [[t for t in tasks if t['case_id'] == first]]
    parts += n.case_parts([t for t in tasks if t['case_id'] != first], 24)
    return [dict(name=f'amos_prob_{i:03d}', tasks=part, smoke=i == 0)
            for i, part in enumerate(parts)]


def validate_pilot_control(rows, reference):
    old = {n.key(r): r for r in reference if r['method'] == 'dual_prob'}
    for row in rows:
        before = old.get(n.key(row))
        if before is None: continue
        if row['method'] != 'dual_prob' or any(abs(row[m]-before[m]) > 1e-6 for m in n.METRICS):
            raise ValueError('pilot overlap metric mismatch')
        if row['prompt_audit']['submission_sha256'] != before['prompt_audit']['submission_sha256']:
            raise ValueError('pilot submitted boxes changed')
        for a, b in zip(row['propagation_audit'], before['propagation_audit'], strict=True):
            for key in ('guide_slices', 'box_decoder_calls', 'prop_decoder_calls', 'guiding_pixels'):
                if a[key] != b[key]: raise ValueError('pilot native propagation changed '+key)


def guarded_writer(original, reference):
    def write(path, value):
        if Path(path).name == 'complete_metrics.json':
            validate_pilot_control(value['per_case_class'], reference)
        return original(path, value)
    return write


def assets_signature():
    data = _native_assets()
    data['files'].update({str(p.relative_to(ROOT)): n.sha(p) for p in (Path(__file__), PILOT)})
    data['protocol'] = PROTOCOL
    return data


def prepare(out):
    if out.exists(): raise ValueError('output exists; do not overwrite')
    tasks = sorted([r for r in n.read(q.MANIFEST)['tasks'] if r['source_name'] == 'amos'], key=n.key)
    validate_scope(tasks, n.read(q.SPLIT))
    assets = assets_signature()
    n.write(out/'protocol.json', assets)
    n.write(out/'manifest.json', dict(tasks=tasks, expected=1485, per_source_counts={'amos':1485},
                                    cases={'amos':100}, train_overlap=0))
    metas = []
    for job in make_jobs(tasks):
        job['output'] = str(out/'jobs'/job['name'])
        job['signature'] = q.signature(job, assets)
        path = out/'job_specs'/(job['name']+'.json'); n.write(path, job)
        metas.append(dict(name=job['name'], path=str(path), tasks=len(job['tasks']), smoke=job['smoke']))
    n.write(out/'jobs.json', dict(jobs=metas, total_case_class=1485, total_method_rows=1485))
    print('PREPARED', len(metas), 'whole-case shards /1485 tasks /100CT', flush=True)


def worker(job, uuid):
    reference = n.read(PILOT)['per_case_class']
    original = n.write
    n.write = guarded_writer(original, reference)
    try: q.worker(job, uuid)
    finally: n.write = original


def progress(out):
    complete = count = 0
    metas = n.read(out/'jobs.json')['jobs']
    for meta in metas:
        folder = out/'jobs'/meta['name']; file = folder/'complete_metrics.json'
        if file.exists(): complete += 1
        else: file = folder/'all_metrics_live.json'
        if file.exists(): count += len(n.read(file)['per_case_class'])
    return dict(completed_jobs=complete, total_jobs=len(metas), completed_tasks=count,
                per_source={'amos':count}, expected=1485)


def merge(out):
    q.merge(out)
    rows = n.read(out/'merged_metrics.json')['per_case_class']
    reference = n.read(PILOT)['per_case_class']
    validate_pilot_control(rows, reference)
    overlap = {n.key(r) for r in rows} & {n.key(r) for r in reference}
    if len(overlap) != 119: raise ValueError('not all119 pilot overlap keys checked')
    print('MERGED1485 /100CT /119pilot controls passed', flush=True)


def configure():
    q.COUNTS = {'amos':1485}; q.PROTOCOL = PROTOCOL
    q.validate_scope = validate_scope; q.assets_signature = assets_signature
    q.p.assets_signature = assets_signature; q.p.progress = progress
    q.p.merge = merge; q.p.__file__ = __file__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare','worker','launch','status','merge'))
    parser.add_argument('--output', default=DEFAULT); parser.add_argument('--job'); parser.add_argument('--uuid')
    args = parser.parse_args(); configure(); out = ROOT/args.output
    if args.action == 'prepare': prepare(out)
    elif args.action == 'worker': worker(args.job, args.uuid)
    elif args.action == 'launch': q.p.schedule(out)
    elif args.action == 'status': print(json.dumps(progress(out), indent=2))
    else: merge(out)


if __name__ == '__main__': main()
