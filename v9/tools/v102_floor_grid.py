"""End-to-end shared dynamic-floor grid, fixed E490 AMOS/MAGIC full val."""
import argparse
import json
import math
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import v102_floor60_confirmation as previous
b=previous.b
amos=previous.amos
FLOORS=(.50,.55,.65,.70)
OUT='output/v102_dynamic_floor_grid_amos1485_magic35_e490_20261023'


def freeze_settings(args,floor):
    if (floor not in FLOORS or args.dynamic_threshold_min!=.35 or args.dynamic_threshold_max!=.75
            or args.dynamic_threshold_fallback!=.60 or args.prompt_logit_neighborhood_radius!=2
            or args.eval_memory_group_size!=1):
        raise ValueError('unexpected floor or original inference configuration')
    args.dynamic_threshold_min=floor
    return args


def make_jobs(rows):
    jobs=[]
    for floor in FLOORS:
        for source,count in (('amos',24),('magic',7)):
            part=sorted([r for r in rows if r['source_name']==source],key=b.key)
            if not part:raise ValueError('missing source')
            parts=[part[:1]]+[b.case_shard(part[1:],count,i) for i in range(count)]
            for i,tasks in enumerate(parts):
                if not tasks:continue
                name=source+'__floor%02d__'%round(floor*100)+('smoke' if i==0 else 's%02d'%i)
                jobs.append(dict(name=name,model='v10_2',variant='baseline',source=source,floor=floor,
                    tasks=tasks,task_count=len(tasks),frame_batch=16,attempt=0,smoke=i==0))
    return sorted(jobs,key=lambda j:(not j['smoke'],j['source']=='amos',j['floor'],j['name']))


def validate_prompt_audit(a,tasks,source,floor):
    settings=a['dynamic_threshold'];keys=[tuple(k) for k in a['loaded_task_keys']]
    thresholds=a['dynamic_threshold_rows']
    if (floor not in FLOORS or settings['minimum']!=floor or settings['maximum']!=.75
            or settings['fallback']!=.60 or a['background_mode']!='point' or a['sources']!=[source]
            or len(keys)!=len(set(keys)) or set(keys)!={b.key(r) for r in tasks}
            or len(thresholds)!=len(tasks)
            or any(not floor<=float(r['threshold'])<=.75 for r in thresholds)):
        raise ValueError('grid floor/prompt/threshold coverage audit mismatch')
    if source=='magic' and a['prompt_coordinate_transform']!='raw_magic_nrrd_to_preprocessed_nifti_swap_flip_xy_v1':
        raise ValueError('MAGIC alignment mismatch')


def verify_job(job):
    run=Path(job['run']);tasks=b.read(Path(job['prepared'])/'manifest.json')['tasks']
    validate_prompt_audit(b.read(run/'exact_prompt_audit.json'),tasks,job['source'],job['floor'])
    i=b.read(run/'intervention_audit.json')
    if (not i['guards_passed'] or i['checkpoint_epoch']!=490 or i['weight_mode']!='live'
            or i['prompt_parameters_exact']<=0 or i['decoder_parameters_exact']<=0
            or i['historical_memory_calls']['memory_encoder']<=0):raise ValueError('restoration/memory audit')
    return tasks


def reused():
    previous.verify(ROOT/previous.OUT)
    sixty=b.read(ROOT/previous.OUT/'merged_metrics.json')
    result={'.35':{},'.60':{}}
    for source,path in previous.BASELINES.items():
        result['.35'][source]=b.read(ROOT/path/'merged_metrics.json')['v10_2']['baseline']['per_case_class']
        result['.60'][source]=sixty[source]['per_case_class']
        b.validate_rows(result['.60'][source],result['.35'][source])
    for by_source in result.values():
        previous.select_tasks(sum(by_source.values(),[]),b.read(ROOT/b.SPLIT))
    return result


def signature(out):
    value=previous.signature(out)
    value['code'][str(Path(__file__).relative_to(ROOT))]=amos.sha(Path(__file__))
    value['new_floors']=list(FLOORS)
    value['floor60_result']=amos.sha(ROOT/previous.OUT/'merged_metrics.json')
    value['floor60_protocol']=amos.sha(ROOT/previous.OUT/'protocol.json')
    value['jobs']=amos.sha(out/'jobs.json')
    value['prepared']={str(p.relative_to(out)):amos.sha(p) for p in sorted((out/'prepared').glob('*/*.json'))}
    value['protocol']='E490 live full AMOS1485 MAGIC35; new floor .50/.55/.65/.70; reuse signed full .35/.60; all AutoExpand stages; max.75 fallback.60 logit/radius2 memory1; original readonly CT/prompts/semantics; fullmask10'
    return value


def verify(out):
    if signature(out)!=b.read(out/'protocol.json'):
        raise ValueError('changed code/weights/prompts/reuse/grid; independent recovery required')


def prepare(out):
    if out.exists():raise FileExistsError(out)
    old=ROOT/previous.OUT;reused()
    manifest=b.read(old/'manifest.json');rows=previous.select_tasks(manifest['tasks'],b.read(ROOT/b.SPLIT))
    refs=b.read(old/'refs.json');index={b.key(r):r for r in refs['per_case_class']}
    if len(index)!=len(rows) or set(index)!={b.key(r) for r in rows}:raise ValueError('reference scope')
    b.write(out/'manifest.json',manifest);b.write(out/'refs.json',refs)
    jobs=[]
    for j in make_jobs(rows):
        tasks=j.pop('tasks');prepared=out/'prepared'/j['name'];j['prepared']=str(prepared)
        b.write(prepared/'manifest.json',b.manifest_payload(tasks))
        b.write(prepared/'refs.json',dict(per_case_class=[index[b.key(t)] for t in tasks]));jobs.append(j)
    b.write(out/'jobs.json',jobs)
    b.write(out/'reuse_audit.json',dict(mode='readonly_no_reprediction',
        sources={'.35':{s:dict(path=p+'/merged_metrics.json',sha256=amos.sha(ROOT/p/'merged_metrics.json')) for s,p in previous.BASELINES.items()},
                 '.60':dict(path=previous.OUT+'/merged_metrics.json',sha256=amos.sha(old/'merged_metrics.json'))},
        expected_each=1520,new_predictions=6080,all_metric_rows=9120))
    b.write(out/'protocol.json',signature(out))
    print('PREPARED',len(jobs),'jobs; new6080 tasks; reuse3040 rows',flush=True)


def configure():
    amos.verify=verify;amos.merge=merge;amos.__file__=__file__


def run(job):
    configure();out=Path(job['run']).parents[1];verify(out);floor=job['floor']
    original=next(j for j in b.read(out/'jobs.json') if j['name']==job['name'])
    if original['floor']!=floor:raise ValueError('worker floor differs from signed grid')
    os.environ['CUDA_VISIBLE_DEVICES']=str(job['gpu']);sys.path.insert(0,str(ROOT/'v10'))
    sys.argv=[__file__,'--gpu',str(job['gpu'])]
    import v10.infer.infer_eight_source_adaptive_v10_2 as ev
    import v10_sam2_memory_roi as memory
    native_args=ev._extract_args
    ev._extract_args=lambda:freeze_settings(native_args(),floor)
    native_decode=memory.memory_sliding_window_decode;stages=[]
    def decode(adapter,video,bundle,tokens,anchor,args,device,training):
        if (training or args.eval_memory_group_size!=1 or args.dynamic_threshold_min!=floor
                or args.dynamic_threshold_max!=.75 or args.prompt_logit_neighborhood_radius!=2):
            raise ValueError('actual memory/threshold settings differ from grid')
        stages.append(dict(frames=int(video.shape[0]),memory_group=1,threshold_min=floor))
        return native_decode(adapter,video,bundle,tokens,anchor,args,device,training)
    memory.memory_sliding_window_decode=decode
    amos.run(job)
    if not stages:raise ValueError('no actual decoder stages')
    tasks=verify_job(job)
    b.write(Path(job['run'])/'confirmation_audit.json',dict(dynamic_floor=floor,dynamic_max=.75,
        task_count=len(tasks),all_stages_use_new_settings=True,actual_decode_stages=stages,memory_group=1))


def summarize(rows,baseline,floor60):
    result=previous.diagnostic.magic.summarize(rows,baseline)
    result['delta_from_floor60']=previous.diagnostic.magic.summarize(rows,floor60)['delta_from_baseline']
    return result


def merge(out):
    verify(out);expected=previous.select_tasks(b.read(out/'manifest.json')['tasks'],b.read(ROOT/b.SPLIT))
    by_floor={floor:[] for floor in FLOORS}
    for j in b.read(out/'jobs.json'):
        run=out/'jobs'/j['name'];spec=b.read(out/'job_specs'/(j['name']+'.json'));tasks=verify_job(spec)
        a=b.read(run/'confirmation_audit.json')
        if (a['memory_group']!=1 or a['dynamic_floor']!=j['floor'] or not a['actual_decode_stages']
            or any(s['memory_group']!=1 or s['threshold_min']!=j['floor'] for s in a['actual_decode_stages'])):
            raise ValueError('stage-specific grid audit mismatch')
        rows=b.read(run/'complete_metrics.json')['per_case_class'];b.validate_rows(rows,tasks);by_floor[j['floor']]+=rows
    for rows in by_floor.values():b.validate_rows(rows,expected)
    reuse=reused();all_rows={'.35':reuse['.35'],'.60':reuse['.60']}
    for floor,rows in by_floor.items():all_rows['%.2f'%floor]={s:[r for r in rows if r['source_name']==s] for s in ('amos','magic')}
    result={name:{s:summarize(rows,reuse['.35'][s],reuse['.60'][s]) for s,rows in sources.items()} for name,sources in all_rows.items()}
    b.write(out/'merged_metrics.json',dict(floors=result,new_predictions=6080,readonly_reused_metric_rows=3040,
        total_metric_rows=9120,selection='report all frozen candidates; development val, not unbiased test'))
    print('MERGED six shared floors each1520 tasks; paired .35/.60',flush=True)


def status(out):
    jobs=b.read(out/'jobs.json');done=[j for j in jobs if (out/'jobs'/j['name']/'confirmation_audit.json').exists()]
    return dict(jobs=len(jobs),audited=len(done),audited_tasks=sum(j['task_count'] for j in done),expected_new=6080,
        per_floor={str(f):sum(j['task_count'] for j in done if j['floor']==f) for f in FLOORS},
        phase=b.read(out/'queue_phase.json') if (out/'queue_phase.json').exists() else None,
        scheduler=b.read(out/'status.json') if (out/'status.json').exists() else None)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('prepare','worker','launch','status','merge'))
    p.add_argument('--out',default=OUT);p.add_argument('--job');p.add_argument('--gpus',default='0,2,3,4,5,6')
    args=p.parse_args();os.chdir(ROOT);out=(ROOT/args.out).resolve()
    if args.mode=='prepare':prepare(out)
    elif args.mode=='worker':run(b.read(args.job))
    elif args.mode=='merge':merge(out)
    elif args.mode=='status':print(json.dumps(status(out),indent=2))
    else:
        gpus=[int(x) for x in args.gpus.split(',')]
        if len(set(gpus))!=len(gpus) or not set(gpus)<={0,2,3,4,5,6}:raise ValueError('invalid GPU list')
        configure();amos.launch(out,gpus)

if __name__=='__main__':main()
