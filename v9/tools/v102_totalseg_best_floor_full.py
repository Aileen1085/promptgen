"""Frozen pilot-Dice winner, then native E490 TotalSeg full4034 confirmation."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import v102_totalseg_floor_pilot as p
g,b=p.g,p.b
CANDIDATES=(.35,)+p.FLOORS
OUT='output/v102_totalseg4034_best_floor_e490_20261024'
PILOT=p.OUT
HISTORY=p.HISTORY


def require_complete(s):
    q=s.get('scheduler') or {}
    if (s.get('jobs')!=91 or s.get('audited')!=91 or s.get('audited_tasks')!=1414
            or s.get('expected_new')!=1414 or q.get('pending',1)!=0
            or q.get('running') or q.get('failed')):
        raise ValueError('all pilot shards/tasks must complete without pending/running/failed')


def choose_winner(result,expected):
    if (len(expected)!=202 or len({b.key(r)[2] for r in expected})!=115
            or result.get('new_predictions')!=1414 or result.get('readonly_baseline_tasks')!=202
            or set(result.get('floors',{}))!={format(f,'.2f') for f in p.FLOORS}):
        raise ValueError('incomplete pilot candidate scope')
    parts={.35:result['baseline035'],**{float(f):v for f,v in result['floors'].items()}}
    scores={}
    for f,s in parts.items():
        rows=s['per_case_class'];b.validate_rows(rows,expected)
        if s['tasks']!=202:raise ValueError('incomplete candidate')
        for m in b.METRICS:
            value=math.fsum(float(r[m]) for r in rows)/202
            if not math.isfinite(float(s['mean'][m])) or abs(value-float(s['mean'][m]))>1e-12:
                raise ValueError('candidate mean mismatch')
        scores[format(f,'.2f')]=math.fsum(float(r['dice']) for r in rows)/202
    chosen=min(scores,key=lambda f:(-scores[f],float(f)))
    return dict(chosen_floor=float(chosen),candidate_dice=scores,
        rule='maximum task-mean complete-volume Dice on fixed202 keys; exact tie smaller floor',
        pilot_tasks=202,weight_epoch=490,weight_mode='live',selection_uses_full_results=False)


def checked_pilot():
    path=ROOT/PILOT;p.verify(path);require_complete(p.status(path))
    data=b.read(path/'merged_metrics.json');expected=b.read(path/'manifest.json')['tasks']
    chosen=p.select_subset(b.read(ROOT/HISTORY/'manifest.json')['tasks'],b.read(ROOT/b.SPLIT))
    if [b.key(r) for r in expected]!=[b.key(r) for r in chosen]:raise ValueError('pilot keys changed')
    value=choose_winner(data,expected)
    value['pilot_assets']={n:g.amos.sha(path/n) for n in ('merged_metrics.json','protocol.json','selection.json','manifest.json')}
    value['pilot_output']=PILOT
    return value


def full_tasks(rows,split):
    rows=sorted([r for r in rows if b.key(r)[0]=='totalseg'],key=b.key)
    ds=split['datasets']['totalseg'];cases={r['case_id'] for r in rows};cs={b.key(r)[2] for r in rows}
    if (len(rows)!=4034 or len({b.key(r) for r in rows})!=4034 or len(cases)!=57 or len(cs)!=115
            or cs & {40,41} or cases!={r['case_id'] for r in ds['val']}
            or cases & {r['case_id'] for r in ds['train']}):raise ValueError('requires full TotalSeg4034/57CT/115classes/train0')
    return rows


def make_jobs(rows,floor):
    if floor not in CANDIDATES or len(rows)!=4034 or len({b.key(r) for r in rows})!=4034:
        raise ValueError('invalid full tasks/floor')
    rows=sorted(rows,key=b.key);parts=[rows[:1]]+[b.case_shard(rows[1:],64,i) for i in range(64)]
    return [dict(name='totalseg__floor%02d__'%round(floor*100)+('smoke' if i==0 else 's%02d'%i),
        model='v10_2',variant='baseline',source='totalseg',floor=floor,tasks=tasks,
        task_count=len(tasks),frame_batch=16,attempt=0,smoke=i==0) for i,tasks in enumerate(parts) if tasks]


def freeze_settings(args,floor):
    if (floor not in CANDIDATES or args.dynamic_threshold_min!=.35 or args.dynamic_threshold_max!=.75
            or args.dynamic_threshold_fallback!=.60 or args.prompt_logit_neighborhood_radius!=2
            or getattr(args,'eval_memory_group_size',1)!=1):raise ValueError('changed native inference settings')
    args.dynamic_threshold_min=floor
    return args


def validate_prompt_audit(a,tasks,source,floor):
    st=a['dynamic_threshold'];keys=[tuple(k) for k in a['loaded_task_keys']]
    ts=a['dynamic_threshold_rows'];tk=[(r['source_name'],r['case_id'],int(r['class_id'])) for r in ts]
    expected={b.key(r) for r in tasks}
    if (source!='totalseg' or floor not in CANDIDATES or st['minimum']!=floor or st['maximum']!=.75
            or st['fallback']!=.60 or a['background_mode']!='point' or a['sources']!=['totalseg']
            or len(keys)!=len(set(keys)) or set(keys)!=expected or len(tk)!=len(set(tk)) or set(tk)!=expected
            or any(not g.threshold_valid(r,floor) for r in ts)):raise ValueError('full prompt/floor/task audit mismatch')


def signature(out):
    value=p.core.signature(out)
    for name in ('tools/v102_totalseg_floor_pilot.py','tools/v102_totalseg_best_floor_full.py'):
        value['code'][name]=g.amos.sha(ROOT/name)
    winner=b.read(out/'winner_selection.json')
    value.update(winner=g.amos.sha(out/'winner_selection.json'),floors=[winner['chosen_floor']],
        pilot={n:g.amos.sha(ROOT/PILOT/n) for n in winner['pilot_assets']},
        protocol='one frozen pilot task-Dice winner; TotalSeg4034 fullval/57CT; E490 live; OLD semantics; '
            'native memory1/radius2/max.75/fallback.60 unclamped/all AutoExpand stages; readonly CT/prompts; fullmask10')
    return value


def verify(out):
    if signature(out)!=b.read(out/'protocol.json'):raise ValueError('changed signed full assets; independent recovery required')


def prepare(out):
    if out.exists():raise FileExistsError(out)
    winner=checked_pilot();split=b.read(ROOT/b.SPLIT)
    tasks=full_tasks(b.read(ROOT/HISTORY/'manifest.json')['tasks'],split)
    if [b.key(r) for r in tasks]!=[b.key(r) for r in full_tasks(b.build_manifest(split)['tasks'],split)]:
        raise ValueError('full current/historical keys differ')
    refs=[r for q in sorted((ROOT/HISTORY/'prepared').glob('totalseg_*/refs.json')) for r in b.read(q)['per_case_class']]
    index={b.key(r):r for r in refs}
    if len(index)!=len(refs) or set(index)!={b.key(r) for r in tasks}:raise ValueError('full prompt reference scope')
    b.write(out/'winner_selection.json',winner);b.write(out/'manifest.json',b.manifest_payload(tasks))
    b.write(out/'refs.json',dict(per_case_class=[index[b.key(r)] for r in tasks]));jobs=[]
    for j in make_jobs(tasks,winner['chosen_floor']):
        part=j.pop('tasks');q=out/'prepared'/j['name'];j['prepared']=str(q)
        b.write(q/'manifest.json',b.manifest_payload(part));b.write(q/'refs.json',dict(per_case_class=[index[b.key(r)] for r in part]));jobs.append(j)
    b.write(out/'jobs.json',jobs);b.write(out/'protocol.json',signature(out))
    print('PREPARED',len(jobs),'shards;4034 full tasks; chosen floor',winner['chosen_floor'],'real smoke pending',flush=True)


def summarize(rows,old):
    result=g.previous.diagnostic.magic.summarize(rows,old)
    result['fallback_count']=sum(str(r.get('dynamic_threshold_source','')).startswith('fallback') for r in rows)
    return result


def merge(out):
    verify(out);split=b.read(ROOT/b.SPLIT);expected=full_tasks(b.read(out/'manifest.json')['tasks'],split)
    floor=b.read(out/'winner_selection.json')['chosen_floor'];rows=[]
    for j in b.read(out/'jobs.json'):
        if j['floor']!=floor:raise ValueError('job winner mismatch')
        spec=b.read(out/'job_specs'/(j['name']+'.json'));tasks=g.verify_job(spec);q=Path(spec['run'])
        a=b.read(q/'confirmation_audit.json')
        if (a['dynamic_floor']!=floor or a['memory_group']!=1 or not a['actual_decode_stages']
                or any(s['memory_group']!=1 or s['threshold_min']!=floor for s in a['actual_decode_stages'])):
            raise ValueError('full actual memory/floor mismatch')
        part=b.read(q/'complete_metrics.json')['per_case_class'];b.validate_rows(part,tasks);rows+=part
    b.validate_rows(rows,expected)
    old=full_tasks(b.read(ROOT/HISTORY/'merged_metrics.json')['v10_2']['per_case_class'],split)
    result=summarize(rows,old);result.update(chosen_floor=floor,scope='TotalSeg4034 fullval/57CT/115classes/train0',
        winner_selection=b.read(out/'winner_selection.json'),selection='pilot-tuned full-val confirmation, not unbiased test')
    b.write(out/'merged_metrics.json',result);print('MERGED4034 tasks at frozen winner',floor,flush=True)


def configure():
    g.FLOORS=CANDIDATES;g.OUT=OUT;g.freeze_settings=freeze_settings;g.validate_prompt_audit=validate_prompt_audit
    g.signature=signature;g.verify=verify;g.merge=merge;g.configure=configure
    g.amos.verify=verify;g.amos.merge=merge;g.amos.__file__=__file__


def status(out):
    js=b.read(out/'jobs.json');done=[j for j in js if (out/'jobs'/j['name']/'confirmation_audit.json').exists()]
    return dict(jobs=len(js),audited=len(done),audited_tasks=sum(j['task_count'] for j in done),expected_new=4034,
        chosen_floor=b.read(out/'winner_selection.json')['chosen_floor'],
        scheduler=b.read(out/'status.json') if (out/'status.json').exists() else None)


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('mode',choices=('prepare','worker','launch','status','merge'))
    a.add_argument('--out',default=OUT);a.add_argument('--job');a.add_argument('--gpus',default='0,2,3,4,5,6')
    args=a.parse_args();out=(ROOT/args.out).resolve();configure()
    if args.mode=='prepare':prepare(out)
    elif args.mode=='worker':g.run(b.read(args.job))
    elif args.mode=='merge':merge(out)
    elif args.mode=='status':print(json.dumps(status(out),indent=2))
    else:
        ids=[int(s) for s in args.gpus.split(',')]
        if not ids or len(ids)!=len(set(ids)) or not set(ids)<={0,2,3,4,5,6}:raise ValueError('invalid GPU list')
        g.amos.launch(out,ids)


if __name__=='__main__':main()
