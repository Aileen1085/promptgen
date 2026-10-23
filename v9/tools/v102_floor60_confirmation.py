"""Frozen E490 full-val confirmation: only dynamic threshold minimum .35 -> .60."""
import argparse
import json
import math
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import v102_inference_calibration as diagnostic
amos=diagnostic.amos
b=amos.base_module()
OUT='output/v102_floor60_amos1485_magic35_e490_20261023'
BASELINES={'amos':'output/amos_module_sensitivity_v92e360_v102e490_20261022',
           'magic':'output/magic_module_sensitivity_v92e360_v102e490_20261023'}


def freeze_settings(args):
    if (args.dynamic_threshold_min!=.35 or args.dynamic_threshold_max!=.75 or
            args.dynamic_threshold_fallback!=.60 or args.prompt_logit_neighborhood_radius!=2):
        raise ValueError('unexpected original calibration protocol')
    args.dynamic_threshold_min=.60
    return args


def floor_rows(baseline,fixed,floor):
    index={b.key(r):r for r in fixed}
    if len(index)!=len(fixed) or set(index)!={b.key(r) for r in baseline}:
        raise ValueError('floor derivation task coverage mismatch')
    return [dict(r if float(r['threshold'])>=floor else index[b.key(r)]) for r in baseline]


def select_tasks(rows,split,expected=None):
    expected=expected or {'amos':(100,1485),'magic':(35,35)}
    selected=[]
    for source,(case_count,task_count) in expected.items():
        part=sorted([r for r in rows if r['source_name']==source],key=b.key)
        ds=split['datasets'][source]; cases={r['case_id'] for r in part}
        if (len(part)!=task_count or len(cases)!=case_count or len({b.key(r) for r in part})!=task_count
                or cases!={r['case_id'] for r in ds['val']}
                or cases&{r['case_id'] for r in ds['train']}):
            raise ValueError('requires complete held-out cases/tasks and train overlap zero')
        selected+=part
    return sorted(selected,key=b.key)


def signature(out):
    result=diagnostic.signature(out)
    result['code'][str(Path(__file__).relative_to(ROOT))]=amos.sha(Path(__file__))
    result['diagnostic_result']=amos.sha(ROOT/diagnostic.OUT/'merged_metrics.json')
    result['baseline_results']={s:amos.sha(ROOT/p/'merged_metrics.json') for s,p in BASELINES.items()}
    result['protocol']='Frozen E490 live AMOS1485 MAGIC35; logit midpoint min .60 max .75 fallback .60 radius2; actual memory1; new threshold used at EVERY AutoExpand stage; fullmask10; original exact prompts/semantics; readonly CT; no mask export'
    return result


def verify(out):
    if signature(out)!=b.read(out/'protocol.json'):
        raise ValueError('changed source/weights/prompts/baselines; independent recovery required')


def prepare(out):
    if out.exists():raise FileExistsError(out)
    rows=select_tasks(b.read(ROOT/amos.HISTORY/'manifest.json')['tasks'],b.read(ROOT/b.SPLIT))
    refs={}
    for source,count in (('amos',12),('magic',7)):
        for path in sorted((ROOT/amos.HISTORY/'prepared').glob(source+'_s*of%d/refs.json'%count)):
            for ref in b.read(path)['per_case_class']:
                if b.key(ref) in refs:raise ValueError('duplicate prompt reference')
                refs[b.key(ref)]=ref
    selected_refs=[dict(refs[b.key(r)],source_name=r['source_name']) for r in rows]
    for ref in selected_refs:
        amos.sha(ROOT/b.CACHE/'metadata/shared_adaptive_prompt_v1'/ref['source_name']/ref['prompt_cache_name'])
    b.write(out/'manifest.json',b.manifest_payload(rows))
    b.write(out/'refs.json',dict(per_case_class=selected_refs))
    jobs=[]
    for source,count in (('amos',24),('magic',7)):
        part=[r for r in rows if r['source_name']==source]
        parts=[part[:1]]+[b.case_shard(part[1:],count,i) for i in range(count)]
        for i,tasks in enumerate(parts):
            if not tasks:continue
            name=source+'__floor60__'+('smoke' if i==0 else 's%02d'%i)
            prepared=out/'prepared'/name
            b.write(prepared/'manifest.json',b.manifest_payload(tasks))
            b.write(prepared/'refs.json',dict(per_case_class=[refs[b.key(r)] for r in tasks]))
            jobs.append(dict(name=name,model='v10_2',variant='baseline',source=source,prepared=str(prepared),
                task_count=len(tasks),frame_batch=16,attempt=0,smoke=i==0))
    jobs.sort(key=lambda j:(not j['smoke'],j['source']=='amos',j['name']))
    b.write(out/'jobs.json',jobs);b.write(out/'protocol.json',signature(out))
    print('PREPARED',len(jobs),'jobs /1520 tasks; frozen dynamic floor60 memory1',flush=True)


def verify_job(job):
    run=Path(job['run']);a=b.read(run/'exact_prompt_audit.json')
    tasks=b.read(Path(job['prepared'])/'manifest.json')['tasks']
    settings=a['dynamic_threshold'];keys=[tuple(k) for k in a['loaded_task_keys']]
    if (settings['minimum']!=.60 or settings['maximum']!=.75 or settings['fallback']!=.60
            or a['background_mode']!='point' or a['sources']!=[job['source']]
            or len(keys)!=len(set(keys)) or set(keys)!={b.key(r) for r in tasks}
            or any(not .60<=float(r['threshold'])<=.75 for r in a['dynamic_threshold_rows'])):
        raise ValueError('frozen threshold/prompt scope audit mismatch')
    if job['source']=='magic' and a['prompt_coordinate_transform']!='raw_magic_nrrd_to_preprocessed_nifti_swap_flip_xy_v1':
        raise ValueError('MAGIC alignment mismatch')
    return dict(dynamic_floor=.60,dynamic_max=.75,task_count=len(keys),all_stages_use_new_settings=True)


def configure():
    amos.verify=verify;amos.merge=merge;amos.__file__=__file__


def run(job):
    configure();out=Path(job['run']).parents[1];verify(out)
    os.environ['CUDA_VISIBLE_DEVICES']=str(job['gpu']);sys.path.insert(0,str(ROOT/'v10'))
    sys.argv=[__file__,'--gpu',str(job['gpu'])]
    import v10.infer.infer_eight_source_adaptive_v10_2 as ev
    import v10_sam2_memory_roi as memory
    native_args=ev._extract_args
    def extract():return freeze_settings(native_args())
    ev._extract_args=extract
    native_decode=memory.memory_sliding_window_decode;stages=[]
    def decode(adapter,video,bundle,tokens,anchor,args,device,training):
        if (training or args.eval_memory_group_size!=1 or args.dynamic_threshold_min!=.60
                or args.dynamic_threshold_max!=.75 or args.prompt_logit_neighborhood_radius!=2):
            raise ValueError('actual decoder settings differ from frozen protocol')
        stages.append(dict(frames=int(video.shape[0]),memory_group=1,threshold_min=.60))
        return native_decode(adapter,video,bundle,tokens,anchor,args,device,training)
    memory.memory_sliding_window_decode=decode
    amos.run(job)
    if not stages:raise ValueError('actual memory decoder not called')
    audit=verify_job(job);audit.update(actual_decode_stages=stages,memory_group=1)
    b.write(Path(job['run'])/'confirmation_audit.json',audit)


def merge(out):
    verify(out);expected=select_tasks(b.read(out/'manifest.json')['tasks'],b.read(ROOT/b.SPLIT));rows=[]
    for j in b.read(out/'jobs.json'):
        run=out/'jobs'/j['name'];spec=b.read(out/'job_specs'/(j['name']+'.json'))
        audit=b.read(run/'confirmation_audit.json')
        if (audit['memory_group']!=1 or any(x['memory_group']!=1 for x in audit['actual_decode_stages'])
                or not b.read(run/'intervention_audit.json')['guards_passed']):raise ValueError('invalid confirmation audit')
        verify_job(spec);rows+=b.read(run/'complete_metrics.json')['per_case_class']
    b.validate_rows(rows,expected);result={}
    for source,path in BASELINES.items():
        baseline=b.read(ROOT/path/'merged_metrics.json')['v10_2']['baseline']['per_case_class']
        part=[r for r in rows if r['source_name']==source]
        result[source]=diagnostic.magic.summarize(part,baseline)
    b.write(out/'merged_metrics.json',result);print('MERGED1520 full-val tasks; paired historical baseline',flush=True)


def diagnostic_summary():
    diagnostic.verify(ROOT/diagnostic.OUT)
    original=b.read(ROOT/diagnostic.OUT/'merged_metrics.json');result={}
    for group,sources in original.items():
        result[group]={}
        for source,policies in sources.items():
            baseline=policies['baseline']['per_case_class']
            result[group][source]={name:{k:v for k,v in p.items() if k!='per_case_class'} for name,p in policies.items()}
            for floor in (.60,.65,.70):
                rows=floor_rows(baseline,policies['fixed_%.2f'%floor]['per_case_class'],floor)
                summary=diagnostic.magic.summarize(rows,baseline)
                result[group][source]['dynamic_floor_%.2f'%floor]={k:v for k,v in summary.items() if k!='per_case_class'}
    b.write(ROOT/diagnostic.OUT/'derived_prompt_only_floor_summary.json',result)
    for group,sources in result.items():
        for source,policies in sources.items():
            for policy,summary in policies.items():
                print(json.dumps(dict(group=group,source=source,policy=policy,**summary)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('prepare','worker','launch','merge','status','diagnostic-summary'))
    p.add_argument('--out',default=OUT);p.add_argument('--job');p.add_argument('--gpus',default='0,2,3,4,5,6')
    args=p.parse_args();os.chdir(ROOT);out=(ROOT/args.out).resolve()
    if args.mode=='prepare':prepare(out)
    elif args.mode=='worker':run(b.read(args.job))
    elif args.mode=='merge':merge(out)
    elif args.mode=='diagnostic-summary':diagnostic_summary()
    elif args.mode=='launch':
        gpus=[int(x) for x in args.gpus.split(',')]
        if len(set(gpus))!=len(gpus) or not set(gpus)<={0,2,3,4,5,6}:raise ValueError('invalid GPU list')
        configure();amos.launch(out,gpus)
    else:
        jobs=b.read(out/'jobs.json')
        print(json.dumps(dict(jobs=len(jobs),audited=sum((out/'jobs'/j['name']/'confirmation_audit.json').exists() for j in jobs),
            phase=b.read(out/'queue_phase.json') if (out/'queue_phase.json').exists() else None,
            scheduler=b.read(out/'status.json') if (out/'status.json').exists() else None),indent=2))


if __name__=='__main__':main()
