"""Fixed E360/E490 Bone770 full-mask dynamic-floor grid; independent entry."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools import bone_floor_protocol as p
from tools import amos_module_sensitivity as amos
b=amos.base_module()
OUT='output/bone_v92e360_v102e490_floor_grid_20261026'
DATA=ROOT.parent/'data/bone'
BASE_SPLIT='configs/ct13_v10_2_split_20260928.json'
DATA_HASHES={'manifest.json':'39550b42384e2049993b148bbbcc6a59f939e6293790749a7f0120b4ebf618dd',
 'tasks.json':'bc1a3a0f1f67959d055d49b6bbe7cda8b1769a84756ef6231c7a3d7cfce23f73',
 'pairing_validation.json':'223b3b67fc4a40fdedf03d32f1a1e53d9e79a4c4d8f64f84dac1f1527368bddb'}


def gpu_ids(text):
    ids=[int(x) for x in text.split(',') if x]
    if not ids or len(ids)!=len(set(ids)) or not set(ids)<={0,2,3,4,5,6}:raise ValueError('invalid GPU selection')
    return ids


def attempt_paths(out,name,attempt):
    if not name or Path(name).name!=name or '/' in name or '\\' in name or not isinstance(attempt,int) or attempt<0:raise ValueError('unsafe attempt identity')
    tag=name+'_a'+str(attempt)
    return dict(run=str(out/'jobs'/tag),spec=str(out/'job_specs'/(tag+'.json')),log=str(out/'logs'/(tag+'.log')))


def validate_job_spec(job,signed,out):
    for field in ('name','model','variant','source','floor','prepared','task_count','smoke'):
        if job.get(field)!=signed[field]:raise ValueError('worker unsigned field: '+field)
    if job.get('gpu') not in (0,2,3,4,5,6) or not isinstance(job.get('gpu_uuid'),str):raise ValueError('prohibited worker GPU')
    attempt=job.get('attempt');paths=attempt_paths(out,job['name'],attempt)
    if job.get('frame_batch') not in (16,8,4,2,1) or any(job.get(k)!=paths[k] for k in ('run','log')):raise ValueError('worker attempt/batch paths mismatch')
    oom=job.get('oom_retries',0)
    if not isinstance(oom,int) or not 0<=oom<=attempt or job['frame_batch']!=max(1,16//(2**oom)):raise ValueError('invalid OOM frame-batch history')


def gpu_snapshot():
    query=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,uuid','--format=csv,noheader,nounits'],text=True)
    return {int(a.strip()):dict(free=int(f.strip()),uuid=u.strip()) for a,f,u in (line.split(',') for line in query.splitlines())}


def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')


def freeze_settings(model,args,floor):
    if model not in p.MODELS or floor not in p.FLOORS or args.dynamic_threshold_min!=.35 or args.dynamic_threshold_max!=.75 or args.dynamic_threshold_fallback!=.60 or args.prompt_logit_neighborhood_radius!=2:raise ValueError('changed native floor parser settings')
    args.dynamic_threshold_min=floor
    return args


def stage_record(model,args,key,frames):
    floor=float(args.dynamic_threshold_min)
    if floor not in p.FLOORS or args.dynamic_threshold_max!=.75 or args.dynamic_threshold_fallback!=.60 or args.prompt_logit_neighborhood_radius!=2 or args.model_input_size!=1024:raise ValueError('changed actual stage settings')
    r=dict(key=list(key),floor=floor,frames=int(frames),input_size=1024,
        method='prompt_probability_midpoint' if model=='v9_2' else 'prompt_logit_midpoint')
    if model=='v9_2':
        if args.max_frames!=192 or args.validation_window_frames!=192:raise ValueError('v92 context changed')
        r.update(context=192,memory='native_v92_no_cross_frame_memory')
    elif model=='v10_2':
        if args.eval_memory_group_size!=1:raise ValueError('v102 memory changed')
        r['memory_group']=1
    else:raise ValueError('unknown model')
    return r


def install_readonly_ct(cls):
    import numpy as np
    def read(self,path):
        key=str(Path(path).resolve())
        if getattr(self,'_bone_case_key',None)==key:return self._bone_case_array
        cache=self._load_case_image_cache(self._case_image_cache_path(path))
        if cache is None:
            cache=np.asarray(self._load_array(path),dtype=np.float32)
            if cache.ndim==4 and cache.shape[0]==1:cache=cache[0]
            if cache.ndim!=3:raise ValueError('CT must be a 3D case volume')
        self._bone_case_key=key;self._bone_case_array=cache
        return cache
    def forbid(*_args,**_kwargs):raise RuntimeError('Bone grid prohibits CT cache writes')
    cls._get_or_create_case_image=read;cls._save_case_image_cache=staticmethod(forbid)


def validate_stages(stages,tasks,model,floor):
    if not stages or {tuple(s['key']) for s in stages}!={p.key(t) for t in tasks}:raise ValueError('actual stage task coverage')
    for s in stages:
        if s['floor']!=floor or s['input_size']!=1024 or not p.threshold_valid(s,floor):raise ValueError('actual stage floor/calibration mismatch')
        if model=='v9_2':
            if s['method']!='prompt_probability_midpoint' or s.get('context')!=192 or s.get('memory')!='native_v92_no_cross_frame_memory':raise ValueError('actual native v92 stage mismatch')
        elif model=='v10_2':
            if s['method']!='prompt_logit_midpoint' or s.get('memory_group')!=1:raise ValueError('actual native v102 stage mismatch')
        else:raise ValueError('unknown model')


def validate_prompt_metadata(meta,expected_shape):
    from collections import Counter
    import numpy as np
    shape=np.asarray(meta['shape_dhw']).reshape(3);slices=np.asarray(meta['slices']).reshape(2)
    points=np.asarray(meta['background_points_dhw']);planes=[str(x) for x in meta['background_point_planes']]
    if (not np.array_equal(shape,expected_shape) or points.shape!=(4,3)
        or Counter(planes)!={'coronal':2,'sagittal':2} or np.any(points<0) or np.any(points>=shape)
        or not 0<=slices[0]<shape[2] or not 0<=slices[1]<shape[1]):raise ValueError('prompt budget/shape mismatch')
    for point,plane in zip(points,planes):
        axis,index=(2,0) if plane=='coronal' else (1,1)
        if point[axis]!=slices[index]:raise ValueError('background point not on frozen prompt plane')
    return dict(shape_dhw=shape.tolist(),slices=slices.tolist(),background_points=points.tolist(),
        background_points_per_plane=2,foreground_planes=['coronal','sagittal'])


def dataset_signature(rehash=False):
    assets={n:amos.sha(DATA/'paired_v1'/n) for n in DATA_HASHES}
    if assets!=DATA_HASHES:raise ValueError('audited Bone input manifest changed')
    m=b.read(DATA/'paired_v1/manifest.json');files={}
    if not m['test_ready'] or m['ready_case_count']!=20 or m['ready_task_count']!=770:raise ValueError('Bone paired dataset not ready')
    for c in m['cases']:
        for path_field,hash_field in (('image','image_sha256'),('label_nifti','label_nifti_sha256')):
            path=DATA/c[path_field];s=path.stat()
            if rehash and amos.sha(path)!=c[hash_field]:raise ValueError('Bone source bytes changed')
            files[c[path_field]]=dict(size=s.st_size,mtime_ns=s.st_mtime_ns,sha256=c[hash_field])
    return dict(assets=assets,files=files)


def initial_signature(out):
    files=list(ROOT.glob('*.py'))
    for folder in ('model','infer','v10','v10/model','v10/infer'):files+=list((ROOT/folder).glob('*.py'))
    files += [Path(__file__),ROOT/'tools/bone_floor_protocol.py',ROOT/'tools/amos_module_sensitivity.py',ROOT/'tools/ct13_full_validation_eval.py',ROOT/'utils/metrics.py']
    prompt_root=ROOT.parent/'nninteractive_test_amos'
    if not (prompt_root/'infer/infer_3d_point_common.py').is_file():prompt_root=ROOT/'nninteractive_test'
    for folder in (prompt_root/'infer',prompt_root/'utils',ROOT/'utils'):files+=list(folder.glob('*.py'))
    return dict(code={os.path.relpath(f,ROOT):amos.sha(f) for f in sorted(set(files))},
        weights={m:dict(path=w,sha256=amos.sha(ROOT/w),epoch=360 if m=='v9_2' else 490,mode='live') for m,w in b.WEIGHTS.items()},
        dataset=dataset_signature(),assets={n:amos.sha(out/n) for n in ('bone_split.json','manifest.json','jobs.json')},
        prepared_manifests={str(f.relative_to(out)):amos.sha(f) for f in sorted((out/'prepared').glob('*/manifest.json'))},
        floors=list(p.FLOORS),predictions=13860,protocol='Bone20/47actual/770; fullmask10; same exact two-plane adaptive FG and2BG/plane; native live v92/v102; every AutoExpand floor; max.75/fallback.60 native bypass; readonly CT; no masks/logits exported')


def signature(out):
    refs=b.read(out/'refs.json')['per_case_class']
    return dict(initial=initial_signature(out),refs=amos.sha(out/'refs.json'),prompt_budget=amos.sha(out/'prompt_budget_audit.json'),
        prompts={r['prompt_cache_name']:amos.sha(ROOT/b.CACHE/'metadata/shared_adaptive_prompt_v1/bone'/r['prompt_cache_name']) for r in refs},
        prepared_refs={str(f.relative_to(out)):amos.sha(f) for f in sorted((out/'prepared').glob('*/refs.json'))})


def verify(out):
    if signature(out)!=b.read(out/'protocol.json'):raise ValueError('changed signed Bone code/data/weight/prompt assets; independent recovery required')


def configure(out):
    b.SPLIT=str((out/'bone_split.json').relative_to(ROOT))
    amos.verify=verify;amos.merge=merge;amos.__file__=__file__


def prompt_worker(job):
    out=Path(job['out']);configure(out)
    if initial_signature(out)!=b.read(out/'preparation_protocol.json'):raise ValueError('changed preparation assets')
    b.prepare(job)


def prompt_worker_argv(jobfile):
    return [str(ROOT.parent/'conda-envs/nninteractive_eval/bin/python'),__file__,'prompt-worker','--job',str(jobfile)]


def prepare(out):
    if out.exists():raise FileExistsError('output exists; do not overwrite/resume preparation blindly')
    if not Path(prompt_worker_argv('unused')[0]).is_file():raise FileNotFoundError('existing nninteractive_eval Python required for shared prompt utilities')
    dataset_signature(rehash=True);m=b.read(DATA/'paired_v1/manifest.json')
    tasks=p.make_tasks(m,DATA)
    if len(tasks)!=770 or len({t['case_id'] for t in tasks})!=20 or len({t['local_class_id'] for t in tasks})!=47:raise ValueError('Bone770/20/47 scope mismatch')
    out.mkdir(parents=True);split=p.augment_split(b.read(ROOT/BASE_SPLIT),tasks,m['classes'])
    b.write(out/'bone_split.json',split);b.write(out/'manifest.json',b.manifest_payload(tasks));configure(out)
    jobs=[]
    for j in p.make_jobs(tasks):
        rows=j.pop('tasks');q=out/'prepared'/j['name'];j['prepared']=str(q)
        b.write(q/'manifest.json',b.manifest_payload(rows));jobs.append(j)
    b.write(out/'jobs.json',jobs);b.write(out/'preparation_protocol.json',initial_signature(out))
    cpu=[]
    for case in sorted({t['case_id'] for t in tasks}):
        q=out/'prompt_prepared'/case;b.write(q/'manifest.json',b.manifest_payload([t for t in tasks if t['case_id']==case]))
        jf=out/'prompt_jobs'/(case+'.json');b.write(jf,dict(gpu=2,prepared=str(q),out=str(out)));cpu.append(jf)
    b.write(out/'preparation_status.json',dict(phase='prompt_preparation',cases=20,done=0,expected_predictions=13860))
    def one(jf):
        log=out/'prompt_logs'/(jf.stem+'.log');log.parent.mkdir(exist_ok=True)
        with log.open('x') as h:
            env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
            proc=subprocess.run(prompt_worker_argv(jf),cwd=ROOT,env=env,stdout=h,stderr=subprocess.STDOUT)
        if proc.returncode:raise RuntimeError('prompt preparation failed: '+str(log))
        return jf.stem
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            from concurrent.futures import as_completed
            futures=[pool.submit(one,jf) for jf in cpu]
            for n,f in enumerate(as_completed(futures),1):
                case=f.result();b.write(out/'preparation_status.json',dict(phase='prompt_preparation',cases=20,done=n));print('PROMPTS_CASE_DONE',case,n,'/20',flush=True)
        refs=[r for q in sorted((out/'prompt_prepared').glob('*/refs.json')) for r in b.read(q)['per_case_class']]
        index={p.key(r):r for r in refs}
        if len(refs)!=770 or len(index)!=770 or set(index)!={p.key(t) for t in tasks}:raise ValueError('exact prompt scope mismatch')
        b.write(out/'refs.json',dict(per_case_class=[index[p.key(t)] for t in tasks]))
        import numpy as np
        cases={c['case_id']:c for c in m['cases']};budgets=[]
        for ref in refs:
            path=ROOT/b.CACHE/'metadata/shared_adaptive_prompt_v1/bone'/ref['prompt_cache_name']
            with np.load(path,allow_pickle=False) as meta:record=validate_prompt_metadata(meta,cases[ref['case_id']]['shape_dhw'])
            budgets.append(dict(key=list(p.key(ref)),prompt_sha256=amos.sha(path),**record))
        b.write(out/'prompt_budget_audit.json',dict(tasks=770,per_case_class=budgets))
        for j in jobs:
            q=Path(j['prepared']);rows=b.read(q/'manifest.json')['tasks']
            b.write(q/'refs.json',dict(per_case_class=[index[p.key(t)] for t in rows]))
        if initial_signature(out)!=b.read(out/'preparation_protocol.json'):raise ValueError('changed assets during CPU preparation')
        b.write(out/'protocol.json',signature(out));b.write(out/'preparation_status.json',dict(phase='prepared',cases=20,done=20,real_smokes_passed=0))
        print('PREPARED',len(jobs),'jobs;13860 new predictions;18 real smokes pending',flush=True)
    except Exception:
        b.write(out/'preparation_error.json',dict(error=traceback.format_exc()));raise


def install_source_hooks(out):
    configure(out);order=tuple(b.read(out/'bone_split.json')['source_order'])
    import ct13_training_entry as ct
    import v9_2_extended_protocol as protocol
    ct.SOURCE_ORDER=order;protocol.SOURCE_ORDER=order
    native=protocol.conditioning_text_from_metadata
    def text(row):return row['semantic_text'] if row.get('dataset')=='bone' else native(row)
    protocol.conditioning_text_from_metadata=text


def verify_job(job):
    out=Path(job['run']).parents[1];signed=next(j for j in b.read(out/'jobs.json') if j['name']==job['name']);validate_job_spec(job,signed,out)
    run=Path(job['run']);tasks=b.read(Path(job['prepared'])/'manifest.json')['tasks']
    a=b.read(run/'exact_prompt_audit.json');i=b.read(run/'intervention_audit.json');c=b.read(run/'confirmation_audit.json')
    expected={p.key(t) for t in tasks};keys=[tuple(k) for k in a['loaded_task_keys']]
    method='prompt_probability_midpoint' if job['model']=='v9_2' else 'prompt_logit_midpoint'
    settings=a['dynamic_threshold']
    if (a['sources']!=['bone'] or a['background_mode']!='point' or len(keys)!=len(set(keys)) or set(keys)!=expected
        or settings['method']!=method or settings['minimum']!=job['floor'] or settings['maximum']!=.75
        or settings['fallback']!=.60 or settings['prompt_logit_neighborhood_radius']!=2):raise ValueError('exact Bone prompt settings/coverage mismatch')
    if (not i['guards_passed'] or i['variant']!='baseline' or i['weight_mode']!='live'
        or i['checkpoint_epoch']!=(360 if job['model']=='v9_2' else 490) or i['prompt_parameters_exact']<=0 or i['decoder_parameters_exact']<=0):raise ValueError('checkpoint restoration audit mismatch')
    if job['model']=='v10_2' and i['historical_memory_calls'].get('memory_encoder',0)<=0:raise ValueError('memory encoder not exercised')
    validate_stages(c['stages'],tasks,job['model'],job['floor'])
    rows=b.read(run/'complete_metrics.json')['per_case_class'];p.validate_rows(rows,tasks)
    live=b.read(run/'all_metrics_live.json')['per_case_class'];p.validate_rows(live,tasks);li={p.key(r):r for r in live}
    last={tuple(s['key']):s for s in c['stages']}
    for r in rows:
        if any(float(r[m])!=float(li[p.key(r)][m]) for m in p.METRICS) or abs(r['mask_threshold_used']-last[p.key(r)]['threshold'])>1e-7:raise ValueError('metric/actual threshold alignment mismatch')
    return tasks


def run(job):
    out=Path(job['run']).parents[1];verify(out);install_source_hooks(out)
    signed=next(j for j in b.read(out/'jobs.json') if j['name']==job['name'])
    validate_job_spec(job,signed,out);gpu=gpu_snapshot()[job['gpu']]
    if gpu['uuid']!=job['gpu_uuid']:raise ValueError('physical GPU UUID changed')
    if gpu['free']<45000:
        print('GPU_RESOURCE_WAIT: free memory changed before model load',flush=True);raise SystemExit(75)
    os.environ['CUDA_VISIBLE_DEVICES']=str(job['gpu']);os.environ['V9_SHARED_CACHE_WRITE_POLICY']='read_only';sys.argv=[__file__,'--gpu',str(job['gpu'])]
    model=job['model'];floor=job['floor'];stages=[];context={}
    if model=='v9_2':
        import infer.infer_eight_source_adaptive_v9_2 as ev
        import infer.totalseg_dynamic_runtime as tr
        native_parse=ev._extract_custom_args;native_bind=ev._install_exact_prompt_factory;native_calibrate=tr.calibrate_prompt_logits
        def parse():
            args,remaining=native_parse();return freeze_settings(model,args,floor),remaining
        def bind(entry,tasks,index,root,audit,mode):context['audit']=audit;return native_bind(entry,tasks,index,root,audit,mode)
        def calibrate(logits,bundle,args):
            r=stage_record(model,args,context['audit']['loaded_task_keys'][-1],logits.shape[0]);result=native_calibrate(logits,bundle,args)
            r.update(threshold=float(result[1]['threshold']),source=result[1]['source']);stages.append(r);return result
        ev._extract_custom_args=parse;ev._install_exact_prompt_factory=bind;tr.calibrate_prompt_logits=calibrate
    else:
        sys.path.insert(0,str(ROOT/'v10'))
        import v10.finetune_totalseg_amos_magic_v10_2_joint as entry
        native_text=entry.conditioning_text_from_metadata
        entry.conditioning_text_from_metadata=lambda r:r['semantic_text'] if r.get('dataset')=='bone' else native_text(r)
        import v10.infer.infer_eight_source_adaptive_v10_2 as ev
        import v10_sam2_memory_roi as memory
        import v10_data
        import v10.infer.infer_amos_sam2_promptgen_v10 as scorer
        install_readonly_ct(v10_data.SAM2CachedTotalSegV10Dataset)
        native_parse=ev._extract_args;ev._extract_args=lambda:freeze_settings(model,native_parse(),floor)
        native_bind=ev._install_exact_prompts_and_reserved_roi
        def bind(*args,**kwargs):context['audit']=kwargs['audit'];return native_bind(*args,**kwargs)
        ev._install_exact_prompts_and_reserved_roi=bind
        native_decode=memory.memory_sliding_window_decode;native_post=scorer.postprocess_v10_logits_from_prompts
        def decode(adapter,video,bundle,tokens,anchor,args,device,training):
            if training:raise ValueError('training path prohibited')
            context['stage']=stage_record(model,args,context['audit']['loaded_task_keys'][-1],video.shape[0])
            return native_decode(adapter,video,bundle,tokens,anchor,args,device,training)
        def post(*args,**kwargs):
            if kwargs['minimum']!=floor or kwargs['maximum']!=.75 or kwargs['fallback']!=.60:raise ValueError('postprocess floor differs')
            result=native_post(*args,**kwargs);r=dict(context['stage']);r.update(threshold=float(result[1]['threshold']),source=result[1]['source']);stages.append(r);return result
        memory.memory_sliding_window_decode=decode;scorer.postprocess_v10_logits_from_prompts=post
    amos.run(job)
    tasks=b.read(Path(job['prepared'])/'manifest.json')['tasks'];validate_stages(stages,tasks,model,floor)
    b.write(Path(job['run'])/'confirmation_audit.json',dict(model=model,floor=floor,stages=stages,CT_cache='existing shared readonly or one-case RAM',bone_semantics='frozen explicit source catalogue text',prompt_budget='same2FGplanes+2BGpoints/plane'))
    verify_job(job);verify(out);print('BONE_REAL_TASK_AUDITS_PASS',model,floor,len(tasks),flush=True)


def metric_rows(job):
    verify_job(job);q=Path(job['run']);tasks=b.read(Path(job['prepared'])/'manifest.json')['tasks'];index={p.key(t):t for t in tasks}
    last={tuple(s['key']):s for s in b.read(q/'confirmation_audit.json')['stages']}
    return [dict(r,label_id=index[p.key(r)]['local_class_id'],dynamic_threshold_source=last[p.key(r)]['source']) for r in b.read(q/'complete_metrics.json')['per_case_class']]


def merge(out):
    verify(out);dataset_signature(rehash=True);expected=b.read(out/'manifest.json')['tasks'];parts={m:{f:[] for f in p.FLOORS} for m in p.MODELS}
    for j in b.read(out/'jobs.json'):
        job=b.read(out/'job_specs'/(j['name']+'.json'));parts[j['model']][j['floor']]+=metric_rows(job)
    result={}
    for model,values in parts.items():
        for rows in values.values():p.validate_rows(rows,expected)
        result[model]=dict(floors={format(f,'.2f'):p.summarize(rows,values[.35],values[.60]) for f,rows in values.items()},winner=p.winner(values))
    b.write(out/'merged_metrics.json',dict(models=result,task_count_per_configuration=770,predictions=13860,
        scope='all20 Bone cases/47actual classes; development floor selection on these same data, not an unbiased heldout threshold test'))
    print('BONE_MERGED13860',json.dumps({m:r['winner'] for m,r in result.items()}),flush=True)


def launch(out,ids):
    configure(out);verify(out)
    if (out/'status.json').exists() or (out/'job_specs').exists():raise ValueError('queue already started; inspect processes/partials before any recovery')
    import fcntl
    lock=(out/'bone_schedule.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    jobs=b.read(out/'jobs.json')
    for smoke in (True,False):
        phase='smoke' if smoke else 'full';pending=[dict(j,oom_retries=0) for j in jobs if j['smoke']==smoke]
        running={};failed=[];done=0;b.write(out/'queue_phase.json',dict(phase=phase))
        while pending or running:
            for gpu,(proc,job,handle) in list(running.items()):
                if proc.poll() is None:continue
                handle.close();del running[gpu]
                if proc.returncode:
                    tail=Path(job['log']).read_text(errors='replace')[-10000:]
                    oom='out of memory' in tail.lower() or 'OutOfMemoryError' in tail
                    if proc.returncode==75 or (oom and job['frame_batch']>1):
                        job['attempt']+=1
                        if oom:job['oom_retries']+=1;job['frame_batch']=max(1,job['frame_batch']//2)
                        pending.insert(0,job);print('RESOURCE_RETRY' if proc.returncode==75 else 'OOM_RETRY',job['name'],'next_attempt',job['attempt'],'batch',job['frame_batch'],flush=True)
                    else:failed.append(dict(name=job['name'],log=job['log'],returncode=proc.returncode))
                else:
                    verify_job(job);write_new(out/'job_specs'/(job['name']+'.json'),job);done+=1
                    print('DONE',job['name'],'tasks',job['task_count'],'attempt',job['attempt'],flush=True)
            available=gpu_snapshot()
            for gpu in ids:
                if failed or gpu in running or not pending or available[gpu]['free']<45000:continue
                job=pending.pop(0);paths=attempt_paths(out,job['name'],job['attempt'])
                job.update(gpu=gpu,gpu_uuid=available[gpu]['uuid'],run=paths['run'],log=paths['log'])
                signed=next(j for j in jobs if j['name']==job['name']);validate_job_spec(job,signed,out)
                if Path(job['run']).exists():raise ValueError('attempt output already exists; no overwrite')
                write_new(paths['spec'],job);Path(job['log']).parent.mkdir(parents=True,exist_ok=True);handle=Path(job['log']).open('x')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
                proc=subprocess.Popen([sys.executable,__file__,'worker','--job',paths['spec']],cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
                running[gpu]=(proc,job,handle);print('START',phase,'gpu',gpu,'pid',proc.pid,job['name'],'tasks',job['task_count'],flush=True)
            b.write(out/'status.json',dict(time=time.strftime('%Y-%m-%d %H:%M:%S'),phase=phase,pending=len(pending),done_in_phase=done,
                running={str(g):dict(name=j['name'],pid=pr.pid,log=j['log'],attempt=j['attempt']) for g,(pr,j,h) in running.items()},failed=failed))
            if failed and not running:raise RuntimeError('terminal worker failure; pending retained, inspect status/logs')
            if pending or running:time.sleep(10)
        for j in jobs:
            if j['smoke']==smoke:verify_job(b.read(out/'job_specs'/(j['name']+'.json')))
        print('PHASE_PASSED',phase,flush=True)
    merge(out)


def status(out):
    result=dict(output=str(out),preparation=b.read(out/'preparation_status.json') if (out/'preparation_status.json').exists() else None,
        signed=(out/'protocol.json').exists(),expected_predictions=13860)
    if result['signed']:
        verify(out);jobs=b.read(out/'jobs.json');done=[]
        for j in jobs:
            q=out/'job_specs'/(j['name']+'.json')
            if q.exists():verify_job(b.read(q));done.append(j)
        result.update(jobs=len(jobs),audited=len(done),audited_tasks=sum(j['task_count'] for j in done),
            queue=b.read(out/'status.json') if (out/'status.json').exists() else None,
            phase=b.read(out/'queue_phase.json') if (out/'queue_phase.json').exists() else None)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('prepare','prompt-worker','worker','launch','status','merge'))
    parser.add_argument('--out',default=OUT);parser.add_argument('--job');parser.add_argument('--gpus',default='0,2,3,4,5,6')
    a=parser.parse_args();out=(ROOT/a.out).resolve()
    if a.mode=='prepare':prepare(out)
    elif a.mode=='prompt-worker':prompt_worker(b.read(a.job))
    elif a.mode=='worker':run(b.read(a.job))
    elif a.mode=='launch':launch(out,gpu_ids(a.gpus))
    elif a.mode=='merge':configure(out);merge(out)
    else:configure(out);print(json.dumps(status(out),indent=2))


if __name__=='__main__':main()
