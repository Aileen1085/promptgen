"""Native v9.2 E360 probability-floor pilot on the frozen TotalSeg202 tasks."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import v102_totalseg_floor_pilot as p
from tools import amos_module_sensitivity as amos
b=amos.base_module()
FLOORS=p.FLOORS
OLD=p.OUT
HISTORY=p.HISTORY
OUT='output/v92_totalseg202_floor_grid_e360_20261025'


def make_jobs(rows):
    jobs=p.make_jobs(rows)
    for j in jobs:j.update(model='v9_2',name='v9_2__'+j['name'])
    return jobs


def freeze_settings(args,floor):
    if (floor not in FLOORS or args.dynamic_threshold_min!=.35 or args.dynamic_threshold_max!=.75
            or args.dynamic_threshold_fallback!=.60 or args.prompt_logit_neighborhood_radius!=2):
        raise ValueError('changed native v9.2 parser settings')
    args.dynamic_threshold_min=floor
    return args


def stage_record(args,floor,frames):
    if (args.dynamic_threshold_min!=floor or args.dynamic_threshold_max!=.75
            or args.dynamic_threshold_fallback!=.60 or args.prompt_logit_neighborhood_radius!=2
            or args.max_frames!=192 or args.validation_window_frames!=192 or args.model_input_size!=1024):
        raise ValueError('changed actual v9.2 context/scale/calibration settings')
    return dict(frames=int(frames),floor=floor,context=192,input_size=1024,
                method='prompt_probability_midpoint',memory='native_v92_no_cross_frame_memory')


def validate_prompt_audit(a,tasks,floor):
    s=a['dynamic_threshold'];keys=[tuple(k) for k in a['loaded_task_keys']];ts=a['dynamic_threshold_rows']
    if (a['background_mode']!='point' or a['sources']!=['totalseg']
            or s['method']!='prompt_probability_midpoint' or s['minimum']!=floor or s['maximum']!=.75
            or s['fallback']!=.60 or s['prompt_logit_neighborhood_radius']!=2
            or len(keys)!=len(set(keys)) or set(keys)!={b.key(t) for t in tasks}
            or Counter(int(t['class_id']) for t in ts)!=Counter(b.key(t)[2] for t in tasks)
            or any(not p.g.threshold_valid(t,floor) for t in ts)):
        raise ValueError('native probability prompt/floor/task audit mismatch')


def attach_threshold_sources(rows,thresholds):
    if len(rows)!=len(thresholds):raise ValueError('baseline threshold count mismatch')
    result=[]
    for row,t in zip(rows,thresholds):
        if b.key(row)[2]!=int(t['class_id']) or abs(row['mask_threshold_used']-t['threshold'])>1e-7:
            raise ValueError('baseline threshold/class association mismatch')
        result.append(dict(row,dynamic_threshold_source=t['source']))
    return result


def baseline():
    rows=p.select_subset(b.read(ROOT/HISTORY/'merged_metrics.json')['v9_2']['per_case_class'],b.read(ROOT/b.SPLIT))
    originals=[]
    for q in sorted((ROOT/HISTORY/'jobs').glob('v9_2_totalseg_*')):
        originals+=attach_threshold_sources(b.read(q/'complete_metrics.json')['per_case_class'],
            b.read(q/'exact_prompt_audit.json')['dynamic_threshold_rows'])
    index={b.key(t):t for t in originals}
    if len(index)!=4034 or len(originals)!=4034:raise ValueError('historical v92 baseline scope')
    for row in rows:
        ref=index[b.key(row)]
        if any(float(row[m])!=float(ref[m]) for m in b.METRICS):raise ValueError('baseline metrics differ')
    return [dict(row,dynamic_threshold_source=index[b.key(row)]['dynamic_threshold_source']) for row in rows]


def signature(out):
    files=list(ROOT.glob('*.py'))
    for folder in ('model','infer'):files+=list((ROOT/folder).glob('*.py'))
    files += [Path(__file__),ROOT/'tools/ct13_full_validation_eval.py',ROOT/'tools/amos_module_sensitivity.py',
              ROOT/'tools/v102_totalseg_floor_pilot.py',ROOT/'tools/v102_topcow_floor_grid.py',ROOT/'utils/metrics.py']
    refs=b.read(out/'refs.json')['per_case_class'];weight=b.WEIGHTS['v9_2']
    return dict(code={str(f.relative_to(ROOT)):amos.sha(f) for f in sorted(set(files))},
        weight=dict(path=weight,sha256=amos.sha(ROOT/weight),epoch=360,mode='live'),split=amos.sha(ROOT/b.SPLIT),
        assets={n:amos.sha(out/n) for n in ('manifest.json','refs.json','selection.json','jobs.json')},
        prepared={str(f.relative_to(out)):amos.sha(f) for f in sorted((out/'prepared').glob('*/*.json'))},
        readonly202={n:amos.sha(ROOT/OLD/n) for n in ('protocol.json','manifest.json','refs.json','selection.json')},
        baseline=amos.sha(ROOT/HISTORY/'merged_metrics.json'),
        baseline_audits={str(f.relative_to(ROOT)):amos.sha(f) for q in sorted((ROOT/HISTORY/'jobs').glob('v9_2_totalseg_*'))
            for f in (q/'complete_metrics.json',q/'exact_prompt_audit.json')},
        prompts={r['prompt_cache_name']:amos.sha(ROOT/b.CACHE/'metadata/shared_adaptive_prompt_v1/totalseg'/r['prompt_cache_name']) for r in refs},
        protocol='v9.2 E360 live native probability midpoint/radius2; context192/input1024/old semantics; '
            'same202 FG scribble/BGpoint readonly; floors.45-.75; max.75 fallback.60 unclamped; every AutoExpand; fullmask10')


def verify(out):
    if signature(out)!=b.read(out/'protocol.json'):raise ValueError('changed signed v92 pilot assets; isolated recovery required')


def prepare(out):
    if out.exists():raise FileExistsError(out)
    old=ROOT/OLD;p.verify(old);rows=p.select_subset(b.read(ROOT/HISTORY/'manifest.json')['tasks'],b.read(ROOT/b.SPLIT))
    frozen=b.read(old/'manifest.json')['tasks']
    if [b.key(t) for t in rows]!=[b.key(t) for t in frozen]:raise ValueError('frozen202 selection differs')
    b.validate_rows(baseline(),rows)
    for n in ('selection.json','manifest.json','refs.json'):b.write(out/n,b.read(old/n))
    refs=b.read(out/'refs.json')['per_case_class'];index={b.key(t):t for t in refs}
    if len(index)!=202 or set(index)!={b.key(t) for t in rows}:raise ValueError('prompt refs mismatch')
    jobs=[]
    for j in make_jobs(rows):
        ts=j.pop('tasks');q=out/'prepared'/j['name'];j['prepared']=str(q)
        b.write(q/'manifest.json',p.payload(ts));b.write(q/'refs.json',dict(per_case_class=[index[b.key(t)] for t in ts]));jobs.append(j)
    b.write(out/'jobs.json',jobs);b.write(out/'protocol.json',signature(out))
    print('PREPARED91 shards/1414 new; native v92; seven real smokes pending',flush=True)


def verify_job(job):
    q=Path(job['run']);tasks=b.read(Path(job['prepared'])/'manifest.json')['tasks'];a=b.read(q/'exact_prompt_audit.json')
    validate_prompt_audit(a,tasks,job['floor']);i=b.read(q/'intervention_audit.json');c=b.read(q/'confirmation_audit.json')
    if (not i['guards_passed'] or i['variant']!='baseline' or i['checkpoint_epoch']!=360 or i['weight_mode']!='live'
            or i['prompt_parameters_exact']<=0 or i['decoder_parameters_exact']<=0 or i['completed_tasks']!=len(tasks)
            or i['memory_disabled'] or not c['stages']):raise ValueError('v92 restore/stage audit mismatch')
    if any(s['floor']!=job['floor'] or s['context']!=192 or s['input_size']!=1024
           or s['method']!='prompt_probability_midpoint' or not p.g.threshold_valid(s,job['floor']) for s in c['stages']):
        raise ValueError('actual stage protocol mismatch')
    last={tuple(s['key']):s for s in c['stages']}
    rows=b.read(q/'complete_metrics.json')['per_case_class'];b.validate_rows(rows,tasks)
    if set(last)!={b.key(t) for t in tasks}:raise ValueError('actual stage task coverage')
    for t in rows:
        if abs(t['mask_threshold_used']-last[b.key(t)]['threshold'])>1e-7:raise ValueError('final threshold/task mismatch')
    return tasks


def run(job):
    configure();out=Path(job['run']).parents[1];verify(out)
    signed=next(j for j in b.read(out/'jobs.json') if j['name']==job['name'])
    if (job['model'],job['variant'],job['floor'])!=('v9_2','baseline',signed['floor']):raise ValueError('worker protocol mismatch')
    os.environ['CUDA_VISIBLE_DEVICES']=str(job['gpu']);sys.argv=[__file__,'--gpu',str(job['gpu'])]
    import infer.infer_eight_source_adaptive_v9_2 as ev
    import infer.totalseg_dynamic_runtime as tr
    extract=ev._extract_custom_args;factory=ev._install_exact_prompt_factory;calibrate=tr.calibrate_prompt_logits;context={};stages=[]
    def parse():
        a,remaining=extract();return freeze_settings(a,job['floor']),remaining
    def bind(entry,tasks,index,root,audit,mode):
        context['audit']=audit;return factory(entry,tasks,index,root,audit,mode)
    def stage(logits,bundle,args):
        record=stage_record(args,job['floor'],logits.shape[0]);record['key']=context['audit']['loaded_task_keys'][-1]
        result=calibrate(logits,bundle,args);record.update(threshold=float(result[1]['threshold']),source=result[1]['source']);stages.append(record);return result
    ev._extract_custom_args=parse;ev._install_exact_prompt_factory=bind;tr.calibrate_prompt_logits=stage
    amos.run(job)
    b.write(Path(job['run'])/'confirmation_audit.json',dict(floor=job['floor'],stages=stages,memory='native_v92'))
    verify_job(job);print('V92_FLOOR_ACTUAL_STAGES_OK',len(stages),flush=True)


def winner(parts,expected):
    scores={}
    for f,rows in parts.items():
        b.validate_rows(rows,expected);scores[format(f,'.2f')]=math.fsum(float(t['dice']) for t in rows)/len(rows)
    chosen=min(scores,key=lambda f:(-scores[f],float(f)))
    return dict(chosen_floor=float(chosen),task_mean_dice=scores,rule='maximum task-mean complete-volume Dice; exact tie smaller floor; includes0.35 reference')


def merge(out):
    verify(out);expected=b.read(out/'manifest.json')['tasks'];parts={f:[] for f in FLOORS};fallback={f:0 for f in FLOORS}
    for j in b.read(out/'jobs.json'):
        spec=b.read(out/'job_specs'/(j['name']+'.json'));verify_job(spec);q=Path(spec['run']);parts[j['floor']]+=b.read(q/'complete_metrics.json')['per_case_class']
        fallback[j['floor']]+=b.read(q/'exact_prompt_audit.json')['dynamic_threshold_summary']['fallback_count']
    old=baseline();result={}
    for f,rows in parts.items():
        b.validate_rows(rows,expected);s=p.summarize(rows,old,parts[.60]);s['fallback_count']=fallback[f]
        groups={c:[t['dice'] for t in rows if b.key(t)[2]==c] for c in {b.key(t)[2] for t in rows}}
        s['class_macro_dice']=math.fsum(math.fsum(v)/len(v) for v in groups.values())/len(groups);result[format(f,'.2f')]=s
    chosen=winner({.35:old,**parts},expected)
    b.write(out/'merged_metrics.json',dict(floors=result,baseline035=p.summarize(old,old,parts[.60]),winner=chosen,
        new_predictions=1414,readonly_baseline_tasks=202,selection='same frozen202 preliminary val; not full4034 or unbiased test'))
    print('MERGED1414 v92; winner',chosen['chosen_floor'],flush=True)


def configure():
    amos.verify=verify;amos.merge=merge;amos.__file__=__file__


def gpu_ids(text):
    ids=[int(x) for x in text.split(',') if x]
    if not ids or len(ids)!=len(set(ids)) or not set(ids)<={0,2,3,4,5,6}:raise ValueError('invalid GPU list')
    return ids


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('prepare','worker','launch','status','merge'))
    parser.add_argument('--out',default=OUT);parser.add_argument('--job');parser.add_argument('--gpus',default='2,4,5,6')
    a=parser.parse_args();out=(ROOT/a.out).resolve();configure()
    if a.mode=='prepare':prepare(out)
    elif a.mode=='worker':run(b.read(a.job))
    elif a.mode=='launch':verify(out);amos.launch(out,gpu_ids(a.gpus))
    elif a.mode=='merge':merge(out)
    else:
        verify(out);jobs=b.read(out/'jobs.json');done=[j for j in jobs if (out/'jobs'/j['name']/'confirmation_audit.json').exists()]
        print(json.dumps(dict(jobs=len(jobs),audited=len(done),audited_tasks=sum(j['task_count'] for j in done),expected=1414,
            scheduler=b.read(out/'status.json') if (out/'status.json').exists() else None),indent=2))


if __name__=='__main__':main()
