"""Readonly3493 reuse and isolated541-task resource-only recovery, E490 floor.50."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import v102_totalseg_best_floor_full as f
b,g=f.b,f.g
OLD=f.OUT
OUT='output/v102_totalseg4034_best_floor_e490_20261025_oom_recovery'
FILES=('complete_metrics.json','all_metrics_live.json','exact_prompt_audit.json',
       'intervention_audit.json','confirmation_audit.json')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_drained(state,processes):
    if state.get('pending',1) or state.get('running') or processes:
        raise ValueError('original/recovery processes or pending jobs still active')


def process_matches(args,pid,self_pid):
    return str(pid)!=str(self_pid) and any(Path(x).name in (
        'v102_totalseg_best_floor_full.py','v102_totalseg_oom_recovery.py') for x in args
        ) and any(x in ('worker','launch') for x in args)


def related_processes():
    found=[]
    for q in Path('/proc').iterdir():
        if not q.name.isdigit():continue
        try:
            args=(q/'cmdline').read_bytes().decode(errors='replace').split('\0')
            if process_matches(args,q.name,os.getpid()):found.append(q.name)
        except OSError:pass
    return found


def partition(jobs,completed,expected=4034):
    names={j['name'] for j in jobs}
    if len(jobs)!=58 or len(names)!=58 or not set(completed)<=names or sum(j['task_count'] for j in jobs)!=expected:
        raise ValueError('invalid exact recovery partition')
    reused=sum(j['task_count'] for j in jobs if j['name'] in completed)
    return dict(reused_shards=len(completed),pending_shards=len(jobs)-len(completed),
                reused_tasks=reused,new_tasks=expected-reused)


def verify_files(files):
    for name,digest in files.items():
        if sha(name)!=digest:raise ValueError('readonly reuse changed: '+name)


def validate_stage(a):
    if (a['dynamic_floor']!=.50 or a['memory_group']!=1 or not a['actual_decode_stages']
            or any(s['memory_group']!=1 or s['threshold_min']!=.50 for s in a['actual_decode_stages'])):
        raise ValueError('recovery actual memory/floor mismatch')


def signature(out):
    value=f.signature(out)
    value['code']['tools/v102_totalseg_oom_recovery.py']=sha(__file__)
    value['readonly_reuse']=sha(out/'reuse_audit.json')
    return value


def verify(out):
    if signature(out)!=b.read(out/'protocol.json'):raise ValueError('changed recovery signature')
    f.verify(ROOT/OLD)
    verify_files(b.read(out/'reuse_audit.json')['files'])


def prepare(out):
    if out.exists():raise FileExistsError(out)
    f.configure();old=ROOT/OLD;f.verify(old)
    state=b.read(old/'status.json');require_drained(state,related_processes())
    expected=f.full_tasks(b.read(old/'manifest.json')['tasks'],b.read(ROOT/b.SPLIT))
    jobs=b.read(old/'jobs.json');completed=set();rows=[];files={}
    for j in jobs:
        q=old/'jobs'/j['name']
        if not (q/'complete_metrics.json').exists():continue
        specfile=old/'job_specs'/(j['name']+'.json');spec=b.read(specfile)
        tasks=g.verify_job(spec);validate_stage(b.read(q/'confirmation_audit.json'))
        part=b.read(q/'complete_metrics.json')['per_case_class'];b.validate_rows(part,tasks)
        rows+=part;completed.add(j['name'])
        for path in [specfile]+[q/n for n in FILES]:files[str(path)]=sha(path)
    info=partition(jobs,completed)
    if info!=dict(reused_shards=52,pending_shards=6,reused_tasks=3493,new_tasks=541):
        raise ValueError('original completion changed; inspect before new recovery')
    b.validate_rows(rows,[r for j in jobs if j['name'] in completed
        for r in b.read(Path(j['prepared'])/'manifest.json')['tasks']])
    failed={j['name'] for j in state['failed']}
    if failed!={j['name'] for j in jobs if j['name'] not in completed}:raise ValueError('failed/pending mismatch')
    for n in ('protocol.json','jobs.json','manifest.json','winner_selection.json'):files[str(old/n)]=sha(old/n)
    f.prepare(out)
    for j in jobs:
        if j['name'] not in completed:continue
        (out/'jobs').mkdir(exist_ok=True);(out/'job_specs').mkdir(exist_ok=True)
        (out/'jobs'/j['name']).symlink_to(old/'jobs'/j['name'],target_is_directory=True)
        (out/'job_specs'/(j['name']+'.json')).symlink_to(old/'job_specs'/(j['name']+'.json'))
    b.write(out/'reuse_audit.json',dict(info,files=files,original_output=OLD,
        completed_names=sorted(completed),failed_names=sorted(failed),partial_metrics_reused=False,
        scope='same4034 keys; readonly complete3493 plus541 new; unchanged E490 live/floor.50 protocol'))
    b.write(out/'protocol.json',signature(out));verify(out)
    print('RECOVERY_PREPARED readonly3493;pending541;old logs/partials untouched',flush=True)


def merge(out):
    verify(out);expected=f.full_tasks(b.read(out/'manifest.json')['tasks'],b.read(ROOT/b.SPLIT));rows=[]
    for j in b.read(out/'jobs.json'):
        if j['floor']!=.5:raise ValueError('winner changed')
        spec=b.read(out/'job_specs'/(j['name']+'.json'));tasks=g.verify_job(spec);q=Path(spec['run'])
        validate_stage(b.read(q/'confirmation_audit.json'))
        part=b.read(q/'complete_metrics.json')['per_case_class'];b.validate_rows(part,tasks);rows+=part
    b.validate_rows(rows,expected)
    old=f.full_tasks(b.read(ROOT/f.HISTORY/'merged_metrics.json')['v10_2']['per_case_class'],b.read(ROOT/b.SPLIT))
    value=f.summarize(rows,old);value.update(chosen_floor=.5,scope='TotalSeg4034/57CT/115classes/train0',
        winner_selection=b.read(out/'winner_selection.json'),recovery=b.read(out/'reuse_audit.json'),
        selection='pilot-tuned full-val confirmation, not unbiased test')
    b.write(out/'merged_metrics.json',value);print('MERGED4034 with readonly3493+new541',flush=True)


def configure():
    f.configure();g.signature=signature;g.verify=verify;g.merge=merge;g.configure=configure
    g.amos.verify=verify;g.amos.merge=merge;g.amos.__file__=__file__


def gpu_ids(text):
    ids=[int(x) for x in text.split(',') if x]
    if not ids or len(ids)!=len(set(ids)) or not set(ids)<={0,2,3,4,5,6}:raise ValueError('invalid GPU list')
    return ids


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('prepare','worker','launch','status','merge'))
    p.add_argument('--out',default=OUT);p.add_argument('--job');p.add_argument('--gpus',default='2,4,6')
    a=p.parse_args();out=(ROOT/a.out).resolve()
    if a.mode=='prepare':prepare(out);return
    configure();verify(out)
    if a.mode=='worker':g.run(b.read(a.job))
    elif a.mode=='status':print(json.dumps(f.status(out),indent=2))
    elif a.mode=='merge':merge(out)
    else:
        require_drained(b.read(ROOT/OLD/'status.json'),related_processes())
        g.amos.launch(out,gpu_ids(a.gpus))


if __name__=='__main__':main()
