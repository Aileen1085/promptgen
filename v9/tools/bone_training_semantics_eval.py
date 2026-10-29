"""Fresh Bone20/47/770 training-aligned semantic evaluation at two fixed floors.

The historical native inference is reused only in a private function namespace.
Preparation, provenance, dispatch, class proofs and final aggregation are owned here.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import copy
from contextlib import redirect_stderr,redirect_stdout
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import types

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools import bone_floor_protocol as p

OUT='output/bone_v92e360_v102e490_training_semantics_bestfloor_20261029'
OLD='output/bone_v92e360_v102e490_floor_grid_20261028_cpugpu_postdrain'
TMUX='bone_v92_v102_training_semantics'
PIPELINE='output/bone_training_semantics_pipeline_20261029.log'
ENTRY='bone_training_semantics_eval.py'
TRAINING='configs/ct13_v10_2_split_20260928.json'
TRAINING_SHA='99b2eee8b4ae549537922614b0570665c6fa90387e21e04c69c04ed0652de022'
NN_REFERENCE='output/bone_nninteractive_single_round_20261028/merged_metrics.json'
NN_SHA='885357d422c93f6ea7feb37abdf67a1334cc7be0c500ef107a63a4a0408727b1'
NATIVE_SHA='b1cedcb405e26df9c07f451d1eac01c9cccd2dd53f5f87fb3fa70c674eca12d2'
REFERENCE_HASHES={
    'manifest.json':'3643ac8ae04bee3d9404f9987965d0a863067f16300445577381e31183355c00',
    'refs.json':'81f0d3f1c15fca1f56c9a363098d6f82a600d7c65174a055627e5405c8d45b78',
    'prompt_budget_audit.json':'7b1ef9e7404485d351b3b0eb530bffc8a5da6c5a650108f7b1ee86a3cd8c9869',
    'merged_metrics.json':'7554dd78885c7caf13327abd1c51a4ff34537bd70430869da9bb936dafb0a60e',
}
ALLOWED=(2,4,5,6)
FLOORS={'v9_2':.35,'v10_2':.75}
PREPARED_ASSETS=('bone_split.json','manifest.json','refs.json','prompt_budget_audit.json',
                 'jobs.json','semantic_alignment_audit.json','preparation_audit.json')
SOURCE_FILES=(
    'tools/bone_training_semantics.py','tools/bone_training_semantics_runtime.py',
    'tools/bone_training_semantics_eval.py','tools/bone_shared_floor_grid.py',
    'tools/bone_shared_floor_runtime.py','tools/bone_shared_floor_restore_runtime.py',
    'tools/bone_shared_floor_probe_runtime.py','tools/bone_metric_speed_runtime.py',
    'tools/bone_shared_logits.py','tools/bone_fast_metrics.py','tools/bone_floor_protocol.py',
    'tools/bone_floor_grid.py','tools/bone_floor_memory_wait_recovery.py',
    'tools/bone_floor_recovery.py','tools/amos_module_sensitivity.py',
    'tools/ct13_full_validation_eval.py',
)
_ENGINE=None
_NATIVE_CACHE_METHODS={}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):digest.update(chunk)
    return digest.hexdigest()


def atomic_new(path,value):
    """Publish exclusively after fsync; an existing class is never overwritten."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp_'+str(os.getpid())+'_'+str(time.time_ns()))
    with temp.open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.link(temp,path)
    temp.unlink()  # exactly one explicitly named temporary file


def write_status(path,value):
    path=Path(path);temp=path.with_name(path.name+'.tmp_'+str(os.getpid()))
    with temp.open('w',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def validate_scope(tasks):
    keys=[p.key(t) for t in tasks]
    if (len(keys)!=770 or len(set(keys))!=770 or {k[0] for k in keys}!={'bone'}
            or len({k[1] for k in keys})!=20 or len({k[2] for k in keys})!=47):
        raise ValueError('fixed Bone20/47/770 scope changed')
    for task in tasks:
        if int(task['local_class_id'])!=p.key(task)[2]-10000 or int(task['local_class_id'])<=0:
            raise ValueError('original Bone label identity changed')


def case_jobs(tasks,history,strict=True):
    if strict:validate_scope(tasks)
    grouped=defaultdict(list)
    for t in sorted(tasks,key=p.key):grouped[p.key(t)[:2]].append(t)
    if sum(len(r) for r in grouped.values())!=len({p.key(t) for t in tasks}):raise ValueError('duplicate task')
    result=[]
    for (_,case),rows in sorted(grouped.items()):
        for model in p.MODELS:
            prior=copy.deepcopy(history.get((model,case),{}))
            policy={k:prior[k] for k in ('frame_batch','minimum_free_mib','last_failed_gpu','avoid_gpu_until','not_before') if k in prior}
            batch=min(2,policy.get('frame_batch',2));minimum=max(30720,policy.get('minimum_free_mib',30720))
            if batch not in (1,2) or minimum not in (30720,45000):raise ValueError('invalid conservative case resource history')
            avoid={str(policy['last_failed_gpu']):policy.get('avoid_gpu_until',0)} if policy.get('last_failed_gpu') is not None else {}
            result.append(dict(name='semantic_'+model+'_'+case,model=model,case_id=case,source='bone',variant='baseline',
                pending=[dict(task=copy.deepcopy(t),floors=[FLOORS[model]]) for t in rows],task_count=len(rows),
                frame_batch=batch,minimum_free_mib=minimum,parent_history=[policy] if policy else [],
                attempt=0,oom_retries=0,not_before=policy.get('not_before',0.),avoid_gpus=avoid,
                avoid_gpu_until=policy.get('avoid_gpu_until',0.),last_failed_gpu=policy.get('last_failed_gpu'),control=False))
    return result


def control_jobs(tasks,smoke_key):
    matches=[t for t in tasks if p.key(t)==tuple(smoke_key)]
    if len(matches)!=1:raise ValueError('original smoke key missing or ambiguous')
    jobs=case_jobs(matches,{},strict=False)
    for j in jobs:j.update(name='semantic_control_'+j['model'],control=True)
    return jobs


def retry(job,code,text,now):
    # Exit75 is always a same-batch resource retry, even if the log mentions OOM.
    if code==0:return None
    oom=code!=75 and ('cuda out of memory' in text.lower() or 'OutOfMemoryError' in text)
    if code!=75 and not oom:return None
    value=copy.deepcopy(job);value.update(attempt=job['attempt']+1,not_before=now+20)
    if oom:
        value['oom_retries']=job['oom_retries']+1
        if job['frame_batch']==2:value['frame_batch']=1
        else:value.update(minimum_free_mib=45000,not_before=now+60,last_failed_gpu=job['gpu'],avoid_gpu_until=now+600,
                          avoid_gpus={**job.get('avoid_gpus',{}),str(job['gpu']):now+600})
    return value


def attempt_outcome(code,text,remaining):
    resource=code==75 or 'cuda out of memory' in text.lower() or 'OutOfMemoryError' in text
    if code and not resource:return 'fatal'
    if not remaining:return 'done' if code==0 else 'fatal'
    return 'retry' if resource else 'fatal'


def eligible(job,gpu,available,now):
    return (gpu in ALLOWED and gpu in available and now>=job['not_before']
            and available[gpu]['free']>=job['minimum_free_mib']
            and now>=job.get('avoid_gpus',{}).get(str(gpu),0)
            and not (job.get('last_failed_gpu')==gpu and now<job.get('avoid_gpu_until',0)))


def require_merge_state(state,live):
    if live or state.get('phase')!='full' or state.get('pending') or state.get('running') or state.get('failed'):
        raise ValueError('merge requires naturally drained successful full queue')


def related():
    if not Path('/proc').exists():return []
    result=[]
    for f in Path('/proc').iterdir():
        if not f.name.isdigit() or int(f.name)==os.getpid():continue
        try:args=(f/'cmdline').read_bytes().decode(errors='replace').split('\0')
        except OSError:continue
        if (any(Path(a).name.startswith('bone_') and a.endswith('.py') for a in args)
                and any(a in ('worker','run','wait-launch','launch','controls','prepare') for a in args)):
            result.append(dict(pid=int(f.name),args=args))
    return result


def preflight(out,*,processes=None,tmux_exists=None,pipeline=None):
    if Path(out).exists():raise FileExistsError('fresh output already exists; duplicate launch refused')
    processes=related() if processes is None else processes
    if processes:raise ValueError('existing Bone process; duplicate/overlapping launch refused')
    if tmux_exists is None:
        proc=subprocess.run(['tmux','has-session','-t',TMUX],capture_output=True,text=True)
        if proc.returncode not in (0,1):raise RuntimeError('tmux preflight failed: '+proc.stderr)
        tmux_exists=proc.returncode==0
    # A dedicated launcher sets this only inside its newly created empty session.
    if tmux_exists and os.environ.get('BONE_SEMANTICS_OWN_SESSION')!=TMUX:raise ValueError('dedicated tmux session already exists')
    if pipeline is not None and Path(pipeline).exists():raise FileExistsError('pipeline log already exists')


def checkpoint_metadata(model,checkpoint):
    args=checkpoint.get('args',{})
    args=vars(args) if hasattr(args,'__dict__') else args
    candidates={k:TRAINING for k,v in args.items() if isinstance(v,str) and v.replace('\\','/').endswith('/'+TRAINING)}
    candidates.update({k:TRAINING for k,v in args.items() if v==TRAINING})
    if not candidates:raise ValueError('actual checkpoint args do not reference frozen training split')
    modality='v9_1_sat_modality' if model=='v9_2' else 'v10_1_sat_modality'
    semantic=args.get('prompt_semantic_enabled',True)
    if (checkpoint.get('epoch')!={'v9_2':360,'v10_2':490}[model] or semantic is not True
            or args.get('prompt_semantic_dim')!=128 or str(args.get(modality,'')).lower()!='ct'
            or args.get('prompt_semantic_template','{}')!='{}'):
        raise ValueError('actual checkpoint epoch/SAT128/CT semantics changed')
    def portable(v):
        if not isinstance(v,str):return v
        path=v.replace('\\','/')
        for marker in ('/models/','/configs/'):
            if marker in path:return marker[1:]+path.split(marker,1)[1]
        return v
    semantic_args={k:portable(v) for k,v in args.items() if 'semantic' in k or '_sat_' in k}
    return dict(epoch=checkpoint['epoch'],catalog_args=candidates,semantic_enabled=bool(semantic),
                semantic_dim=128,modality='ct',semantic_args=semantic_args)


def training_checkpoint_audit():
    import torch
    result={}
    for model,path in engine().b.WEIGHTS.items():
        checkpoint=torch.load(ROOT/path,map_location='cpu',weights_only=False)
        result[model]=checkpoint_metadata(model,checkpoint)
        del checkpoint
    return result


def source_paths():
    files=set(ROOT/n for n in SOURCE_FILES)
    for folder in ('','model','infer','utils','v10','v10/model','v10/infer'):
        files.update((ROOT/folder).glob('*.py'))
    files.update((ROOT/'tools').glob('bone_*.py'))
    for pattern in ('*.py','*.yaml','*.yml'):files.update((ROOT/'sam2/sam2').rglob(pattern))
    # Freeze the actual native prompt utilities, which can be in the sibling project.
    prompt_root=ROOT.parent/'nninteractive_test_amos'
    if not (prompt_root/'infer/infer_3d_point_common.py').is_file():prompt_root=ROOT/'nninteractive_test'
    for folder in ('infer','utils'):files.update((prompt_root/folder).glob('*.py'))
    return sorted(files)


def native_case_cache_paths(image,cache_root):
    """Execute the original path functions without importing Torch or any dataset."""
    root_key=str(ROOT)
    if root_key not in _NATIVE_CACHE_METHODS:
        def function(path,name):
            tree=ast.parse((ROOT/path).read_text(encoding='utf-8'))
            matches=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name]
            if len(matches)!=1:raise ValueError('native case-cache method anchor changed: '+path+'/'+name)
            node=copy.deepcopy(matches[0]);node.decorator_list=[]
            namespace=dict(os=os,hashlib=hashlib,Path=Path)
            exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),__file__,'exec'),namespace)
            return namespace[name]
        def version(path):
            tree=ast.parse((ROOT/path).read_text(encoding='utf-8'))
            matches=[n.value for n in ast.walk(tree) if isinstance(n,ast.Assign)
                     and any(isinstance(t,ast.Name) and t.id=='CASE_IMAGE_CACHE_VERSION' for t in n.targets)]
            if len(matches)!=1:raise ValueError('native CT cache version anchor changed')
            return ast.literal_eval(matches[0])
        _NATIVE_CACHE_METHODS[root_key]={
            'v9_2':(function('v9_2_shared_cache.py','_case_image_cache_path'),function('v9_full_roi_data.py','_source_signature'),version('v9_2_shared_cache.py')),
            'v10_2':(function('v10/v10_2_data.py','_case_image_cache_path'),function('v10/v10_data.py','_source_signature'),version('v10/v10_data.py')),
        }
    paths={}
    for model,(method,source_signature,cache_version) in _NATIVE_CACHE_METHODS[root_key].items():
        owner=types.SimpleNamespace(use_cache=True,cache_dir=str(cache_root),shared_case_cache_dir=str(cache_root),
            v9_2_shared_cache_dir=str(cache_root),source_name='bone',v9_2_shared_cache_source='bone',
            CASE_IMAGE_CACHE_VERSION=cache_version,_source_signature=source_signature)
        value=method(owner,str(image))
        if value is None:raise ValueError('native shared case CT path unavailable')
        paths[model]=Path(value)
    return paths


def case_cache_signature(tasks,refs,cache_root,reader=None,path_resolver=None):
    """Hash only the20 selected, shared case CT paths; absence is a frozen state."""
    validate_scope(tasks);expected={p.key(t) for t in tasks};indexed={p.key(r):r for r in refs}
    if len(refs)!=770 or len(indexed)!=770 or set(indexed)!=expected:raise ValueError('shared case CT reference coverage changed')
    cache_root=Path(cache_root)
    if reader is None:
        import numpy as np
        def reader(path):
            with np.load(path,allow_pickle=False) as data:return str(data['case_image_name'].item())
    by_case={};path_resolver=path_resolver or native_case_cache_paths
    for t in sorted(tasks,key=p.key):
        ref=indexed[p.key(t)];prompt_name=ref['prompt_cache_name']
        if Path(prompt_name).name!=prompt_name:raise ValueError('unsafe prompt package path')
        name=reader(cache_root/'metadata/shared_adaptive_prompt_v1/bone'/prompt_name)
        if not isinstance(name,str) or Path(name).name!=name or not name.endswith('.npy') or '/' in name or '\\' in name:
            raise ValueError('unsafe shared case CT reference')
        case=t['case_id']
        entry=(name,t['image'])
        if case in by_case and by_case[case]!=entry:raise ValueError('class prompt references do not share one CT per case')
        by_case[case]=entry
    if len(by_case)!=20 or len({v[0] for v in by_case.values()})!=20:raise ValueError('twenty distinct shared case CT paths required')
    result={}
    for case,(name,image) in sorted(by_case.items()):
        path=cache_root/'_case_images_v1'/name;value=dict(cache_name=name,path='_case_images_v1/'+name,state='absent')
        source_image=Path(image) if Path(image).is_absolute() else ROOT/image
        native=path_resolver(source_image,cache_root)
        if set(native)!=set(FLOORS) or any(Path(p)!=path for p in native.values()):raise ValueError('frozen prompt CT reference differs from native model cache path')
        if path.exists():
            stat=path.stat();value.update(state='present',size=stat.st_size,mtime_ns=stat.st_mtime_ns,sha256=sha(path))
        result[case]=value
    return result


def snapshot(out,rehash=False):
    e=engine();reference={n:sha(ROOT/OLD/n) for n in REFERENCE_HASHES}
    if reference!=REFERENCE_HASHES:raise ValueError('frozen old reference manifest/prompts/metrics changed')
    if sha(ROOT/NN_REFERENCE)!=NN_SHA:raise ValueError('frozen same770 nnInteractive reference changed')
    if sha(ROOT/TRAINING)!=TRAINING_SHA:raise ValueError('actual training catalog SHA changed')
    files=source_paths()
    refs=read(Path(out)/'refs.json')['per_case_class']
    prompts={r['prompt_cache_name']:sha(ROOT/e.b.CACHE/'metadata/shared_adaptive_prompt_v1/bone'/r['prompt_cache_name']) for r in refs}
    weights={m:dict(path=w,sha256=sha(ROOT/w),epoch=360 if m=='v9_2' else 490,mode='live') for m,w in e.b.WEIGHTS.items()}
    model_assets=[ROOT/'sam2/checkpoints/sam2.1_hiera_large.pt',ROOT/'models/sat-medical-text/text_encoder.pth']
    model_assets+=sorted(f for f in (ROOT/'models/biolord-2023-c').rglob('*') if f.is_file())
    weights['native_model_assets']={str(f.relative_to(ROOT)):sha(f) for f in model_assets}
    dataset=e.g.dataset_signature(rehash=rehash)
    dataset['shared_case_ct']=case_cache_signature(read(Path(out)/'manifest.json')['tasks'],refs,ROOT/e.b.CACHE)
    return dict(sources={os.path.relpath(f,ROOT):sha(f) for f in files},weights=weights,
        training=dict(path=TRAINING,sha256=TRAINING_SHA,checkpoint_audit=read(Path(out)/'preparation_audit.json')['checkpoints']),
        dataset=dataset,prompts=prompts,
        reference={**reference,'bone_split.json':sha(ROOT/OLD/'bone_split.json'),'protocol.json':sha(ROOT/OLD/'protocol.json'),NN_REFERENCE:NN_SHA})


def signature(out):
    out=Path(out)
    return dict(frozen=snapshot(out),assets={n:sha(out/n) for n in PREPARED_ASSETS},
        prepared={str(f.relative_to(out)):sha(f) for f in sorted((out/'prepared').glob('*/*.json'))},
        policy=dict(floors=FLOORS,models=list(p.MODELS),groups=40,task_count_per_model=770,final_rows=1540,
            controls=2,controls_in_final=False,weights='live',max_threshold=.75,fallback_threshold=.60,
            physical_gpus=list(ALLOWED),one_worker_per_card=True,frame_batch_cap=2,
            ordinary_free_mib=30720,batch1_oom_free_mib=45000,oom_cooldown_seconds=60,avoid_card_seconds=600,
            semantic='training builder final text; positive actual native consumption',
            metric='ten finite complete-mask metrics; original axial union; native control <=1e-6',
            old_predictions_reused=False,disk_tensors=False))


def validate_prepared(out):
    from tools.bone_training_semantics import validate_alignment
    out=Path(out);old=read(ROOT/OLD/'bone_split.json');split=read(out/'bone_split.json')
    original=[r for r in old['class_catalog'] if r.get('dataset')=='bone']
    aligned=[r for r in split['class_catalog'] if r.get('dataset')=='bone']
    validate_alignment(original,read(ROOT/TRAINING)['class_catalog'],aligned,read(out/'semantic_alignment_audit.json'))
    expected=copy.deepcopy(old);expected['class_catalog']=[next(r for r in aligned if r['global_class_id']==t['global_class_id']) if t.get('dataset')=='bone' else t for t in old['class_catalog']]
    if split!=expected:raise ValueError('non-semantic split content changed')
    tasks=read(out/'manifest.json')['tasks'];validate_scope(tasks)
    for name in ('manifest.json','refs.json','prompt_budget_audit.json'):
        if sha(out/name)!=REFERENCE_HASHES[name]:raise ValueError('frozen task/prompt bundle changed')
    jobs=read(out/'jobs.json');audit=read(out/'preparation_audit.json')
    canonical=case_jobs(tasks,{(x['model'],x['case_id']):x['policy'] for x in audit['resource_history']})
    controls=control_jobs(tasks,audit['smoke_key'])
    for collection in (canonical,controls):
        for j in collection:j['prepared']=str(out/'prepared'/j['name'])
    if jobs!=dict(cases=canonical,controls=controls):raise ValueError('signed case/control jobs changed')
    refs=read(out/'refs.json')
    for j in jobs['cases']+jobs['controls']:
        validate_worker_inputs(j,j,read(Path(j['prepared'])/'manifest.json'),read(Path(j['prepared'])/'refs.json'),refs)


def verify(out):
    out=Path(out)
    if signature(out)!=read(out/'protocol.json'):raise ValueError('changed signed source/weight/training/data/prompt/semantic protocol')
    validate_prepared(out);bind_context(out)


def history_and_smoke(old,tasks):
    """Read only conservative resource fields, never any previous prediction."""
    history={};smokes=set();old=Path(old)
    # Original smoke tasks are recorded in the historical per-floor jobs.
    for folder in [old,ROOT/'output/bone_v92e360_v102e490_floor_grid_20261026']:
        if not (folder/'jobs.json').is_file():continue
        catalog=read(folder/'jobs.json');values=catalog if isinstance(catalog,list) else catalog.get('controls',[])
        for j in values:
            if not (j.get('smoke') or j.get('control')):continue
            if 'pending' in j:rows=[x['task'] for x in j['pending']]
            else:rows=read(Path(j['prepared'])/'manifest.json')['tasks']
            if len(rows)==1:smokes.add(p.key(rows[0]))
    if len(smokes)!=1:raise ValueError('cannot uniquely recover original smoke key')
    specs=sorted((old/'job_specs').glob('*.json'))
    for f in specs:
        j=read(f)
        if j.get('control') or j.get('model') not in FLOORS or j.get('case_id') is None:continue
        k=j['model'],j['case_id'];prior=history.setdefault(k,dict(frame_batch=2,minimum_free_mib=30720))
        prior['frame_batch']=min(prior['frame_batch'],j['frame_batch'],2)
        prior['minimum_free_mib']=max(prior['minimum_free_mib'],j.get('minimum_free_mib',30720))
    return history,next(iter(smokes))


def verify_old_reference():
    """Audit the frozen native chain once before signing any fresh preparation."""
    from tools.bone_gpu_postdrain_grid import engine as old_engine
    reference=old_engine();reference.verify(ROOT/OLD);reference.check_controls(ROOT/OLD)
    return dict(native_chain_verified=True,protocol_sha256=sha(ROOT/OLD/'protocol.json'),
                frozen_reference_sha256=copy.deepcopy(REFERENCE_HASHES),old_predictions_reused=False)


def prepare(out,*,already_checked=False):
    out=Path(out)
    if not already_checked:preflight(out,pipeline=ROOT/PIPELINE)
    if out!=ROOT/OUT or out.exists():raise FileExistsError('fresh dedicated output required')
    e=engine()
    for name,expected in REFERENCE_HASHES.items():
        if sha(ROOT/OLD/name)!=expected:raise ValueError('frozen reference changed: '+name)
    if sha(ROOT/TRAINING)!=TRAINING_SHA:raise ValueError('frozen actual training catalog changed')
    reference_audit=verify_old_reference()
    e.g.dataset_signature(rehash=True)
    old=read(ROOT/OLD/'bone_split.json');tasks=read(ROOT/OLD/'manifest.json')['tasks'];validate_scope(tasks)
    from tools.bone_training_semantics import align_catalog,validate_alignment
    original=[r for r in old['class_catalog'] if r.get('dataset')=='bone'];training=read(ROOT/TRAINING)['class_catalog']
    aligned,audit=align_catalog(original,training);validate_alignment(original,training,aligned,audit)
    by_gid={int(r['global_class_id']):r for r in aligned}
    split=copy.deepcopy(old);split['class_catalog']=[by_gid[int(r['global_class_id'])] if r.get('dataset')=='bone' else r for r in split['class_catalog']]
    history,smoke=history_and_smoke(ROOT/OLD,tasks);jobs=case_jobs(tasks,history);controls=control_jobs(tasks,smoke)
    checkpoints=training_checkpoint_audit();out.mkdir(parents=True,exist_ok=False)
    atomic_new(out/'bone_split.json',split);atomic_new(out/'semantic_alignment_audit.json',audit)
    for name in ('manifest.json','refs.json','prompt_budget_audit.json'):
        # Small immutable JSON metadata only; retain byte-identical signatures.
        with (out/name).open('xb') as f:
            f.write((ROOT/OLD/name).read_bytes());f.flush();os.fsync(f.fileno())
    refs={p.key(r):r for r in read(out/'refs.json')['per_case_class']}
    for j in jobs+controls:
        prepared=out/'prepared'/j['name'];rows=[x['task'] for x in j['pending']]
        atomic_new(prepared/'manifest.json',e.g.bone_payload(rows))
        atomic_new(prepared/'refs.json',dict(per_case_class=[refs[p.key(r)] for r in rows]));j['prepared']=str(prepared)
    atomic_new(out/'jobs.json',dict(cases=jobs,controls=controls))
    atomic_new(out/'preparation_audit.json',dict(checkpoints=checkpoints,smoke_key=list(smoke),
        resource_history=[dict(model=m,case_id=c,policy=v) for (m,c),v in sorted(history.items())],
        fresh_predictions=True,old_predictions_reused=False,controls_in_final=False,reference_verification=reference_audit))
    atomic_new(out/'protocol.json',signature(out));verify(out)
    print('SEMANTICS_PREPARED',len(jobs),'groups',1540,'fresh rows plus 2 controls',flush=True)


def validate_attempt(job,canonical,out):
    for field in ('name','model','control','parent_history','variant','source','case_id','prepared'):
        if job.get(field)!=canonical.get(field):raise ValueError('unsigned case/model binding: '+field)
    if (type(job.get('gpu')) is not int or job['gpu'] not in ALLOWED or not isinstance(job.get('gpu_uuid'),str)
            or not job['gpu_uuid'].startswith('GPU-') or job.get('frame_batch') not in (1,2)
            or job['frame_batch']>canonical['frame_batch'] or job.get('minimum_free_mib') not in (30720,45000)
            or job['minimum_free_mib']<canonical['minimum_free_mib']):raise ValueError('GPU/batch/resource binding changed')
    attempt=job.get('attempt');oom=job.get('oom_retries')
    if type(attempt) is not int or attempt<0 or type(oom) is not int or not 0<=oom<=attempt:raise ValueError('invalid attempt/OOM history')
    tag=canonical['name']+'_a'+str(attempt)
    if any(job.get(k)!=str(Path(out)/folder/(tag+suffix)) for k,folder,suffix in [('run','runs',''),('spec','job_specs','.json'),('log','logs','.log')]):
        raise ValueError('attempt path outside exclusive output')


def validate_worker_inputs(spec,canonical,manifest,refs,frozen_refs):
    expected={p.key(x['task']):x for x in canonical['pending']};current=spec['pending'];keys=[p.key(x['task']) for x in current]
    if (not current or len(keys)!=len(set(keys)) or spec['task_count']!=len(keys)
            or any(expected.get(k)!=x for k,x in zip(keys,current))):raise ValueError('unsigned worker task/floor subset')
    if manifest['tasks']!=[x['task'] for x in current] or manifest['expected_task_count']!=len(current):raise ValueError('prepared manifest differs')
    by_key={p.key(r):r for r in frozen_refs['per_case_class']}
    if refs['per_case_class']!=[by_key[k] for k in keys]:raise ValueError('frozen prompt reference differs')


def marker_paths(out,run,marker):
    record=marker.with_name(marker.name.removesuffix('.done.json')+'.json')
    return (record,run/'restoration_start.json',run/'resource_policy_audit.json',Path(out)/'job_specs'/(run.name+'.json'),
            run/'prepared/manifest.json',run/'prepared/refs.json',run/'prepared_input_audit.json',Path(out)/'protocol.json')


def require_marker_files(info,marker,out,canonical):
    marker=Path(marker);out=Path(out);run=marker.parents[1];prefix=canonical['name']+'_a'
    if run.parent!=out/'runs' or not run.name.startswith(prefix) or not run.name[len(prefix):].isdigit():raise ValueError('marker outside signed attempt')
    paths=marker_paths(out,run,marker)
    if info['record']!=str(paths[0]) or set(info['files'])!={str(f) for f in paths}:raise ValueError('marker omitted/misbound mandatory class proof')
    return run


def validate_class(record,expected,control=None):
    if record['model'] not in FLOORS or expected['floors']!=[FLOORS[record['model']]]:raise ValueError('changed fixed floor')
    if record['task']!=expected['task']:raise ValueError('changed original task identity')
    result=engine().native_validate_class(record,expected)
    from tools.bone_metric_speed_runtime import validate_metric_record
    from tools.bone_training_semantics_runtime import validate_semantic_record
    validate_metric_record(record)
    if control is None:control=record.get('semantic_execution',{}).get('control',False)
    validate_semantic_record(record,read(ROOT/OUT/'bone_split.json')['class_catalog'],control=control)
    for value in result:
        if int(value['row'].get('label_id',-1))!=int(expected['task']['local_class_id']):raise ValueError('original metric label ID lost')
    return result


def class_records(out,control=False):
    out=Path(out);jobs={j['name']:j for j in read(out/'jobs.json')['controls' if control else 'cases']};found={}
    for marker in sorted((out/'runs').glob('*/classes/*.done.json')):
        info=read(marker)
        if info.get('job_name') not in jobs:
            # Markers for the other declared phase are expected; unknown markers fail.
            all_jobs=read(out/'jobs.json');all_names={j['name'] for rows in all_jobs.values() for j in rows}
            if info.get('job_name') not in all_names:raise ValueError('unknown committed class job')
            continue
        if info.get('protocol_sha')!=sha(out/'protocol.json'):raise ValueError('class protocol SHA mismatch')
        canonical=jobs[info['job_name']];run=require_marker_files(info,marker,out,canonical)
        for path,expected_sha in info['files'].items():
            if sha(path)!=expected_sha:raise ValueError('changed immutable class proof: '+path)
        record=read(info['record']);spec=read(out/'job_specs'/(run.name+'.json'));validate_attempt(spec,canonical,out)
        if record['model']!=canonical['model']:raise ValueError('class model binding changed')
        validate_worker_inputs(spec,canonical,read(run/'prepared/manifest.json'),read(run/'prepared/refs.json'),read(out/'refs.json'))
        prepared=read(run/'prepared_input_audit.json')
        if prepared!=dict(canonical_prepared=canonical['prepared'],files={str(run/'prepared'/n):sha(run/'prepared'/n) for n in ('manifest.json','refs.json')}):raise ValueError('prepared proof mismatch')
        if record['restoration']!=read(run/'restoration_start.json'):raise ValueError('restoration proof mismatch')
        resource=read(run/'resource_policy_audit.json')
        if (any(resource.get(k)!=spec[k] for k in ('gpu','gpu_uuid','frame_batch','minimum_free_mib','parent_history'))
                or resource['actual_start_free_mib']<spec['minimum_free_mib'] or resource['new_oom_retries']!=spec['oom_retries']):raise ValueError('actual resource proof mismatch')
        matches=[x for x in spec['pending'] if x['task']==record['task']]
        if len(matches)!=1:raise ValueError('class not in exact signed pending subset')
        validate_class(record,matches[0],control=canonical['control']);identity=record['model'],p.key(record['task'])
        if identity in found:raise ValueError('duplicate immutable committed class')
        found[identity]=dict(record=record,marker=str(marker),marker_sha=sha(marker),spec=spec)
    return found


def validate_native_twins(record):
    probes=record.get('native_batch_probe',{})
    if set(map(str,probes))!={'1','2','4'}:raise ValueError('missing original v10 native batch1/2/4 twins')
    for v in probes.values():
        if (not v.get('exact_same_batch') or not v.get('raw_sha') or v['raw_sha']!=v.get('native_raw_sha')
                or not v.get('encoder_shas') or v['encoder_shas']!=v.get('native_encoder_shas')):raise ValueError('native batch twin mismatch')


def check_controls(out):
    from tools.bone_metric_speed_runtime import validate_metric_record
    from tools.bone_training_semantics_runtime import validate_semantic_record
    out=Path(out);jobs=read(out/'jobs.json')['controls'];records=class_records(out,True);catalog=read(out/'bone_split.json')['class_catalog']
    if len(jobs)!=2 or len(records)!=2:raise ValueError('two fresh GPU controls required')
    for j in jobs:
        item=records.get((j['model'],p.key(j['pending'][0]['task'])))
        if item is None:raise ValueError('fresh original-key control missing')
        r=item['record'];validate_metric_record(r,True);validate_semantic_record(r,catalog,control=True)
        if j['model']=='v10_2':validate_native_twins(r)
    return records


def bind_context(out):
    e=engine();e.b.SPLIT=str(Path(out).relative_to(ROOT)/'bone_split.json')
    e.g.amos.verify=verify;e.g.amos.__file__=__file__


def engine():
    global _ENGINE
    if _ENGINE is not None:return _ENGINE
    path=ROOT/'tools/bone_shared_floor_grid.py';source=path.read_bytes()
    if hashlib.sha256(source).hexdigest()!=NATIVE_SHA:raise ValueError('frozen reviewed native worker source changed')
    value=types.ModuleType('bone_training_semantics_private_native');value.__file__=str(path)
    exec(compile(source,str(path),'exec'),value.__dict__)
    value.native_validate_class=value.validate_class
    # Clone the native driver namespace; its source hooks must resolve our context.
    native_g=value.g;private_g=types.ModuleType('bone_training_semantics_private_driver');private_g.__dict__.update(native_g.__dict__)
    private_g.configure=bind_context
    private_g.install_source_hooks=types.FunctionType(native_g.install_source_hooks.__code__,dict(native_g.install_source_hooks.__globals__,configure=bind_context))
    value.g=private_g;value.OUT=OUT;value.__file__=__file__;value.ALLOWED=ALLOWED
    for name in ('verify','validate_attempt','validate_worker_inputs','class_records','validate_class','atomic_new'):
        setattr(value,name,globals()[name])
    value._context=bind_context
    tree=ast.parse(source);run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    imports=[n for n in ast.walk(run) if isinstance(n,ast.ImportFrom) and n.module=='tools.bone_shared_floor_runtime']
    if len(imports)!=1:raise ValueError('native runtime import anchor changed')
    imports[0].module='tools.bone_training_semantics_runtime';imports[0].names.append(ast.alias(name='configure_semantics'))
    constructors=[(i,n) for i,n in enumerate(run.body) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='GridSession']
    if len(constructors)!=1:raise ValueError('native session constructor anchor changed')
    i,_=constructors[0]
    run.body.insert(i,ast.parse("configure_semantics(b.read(out/'bone_split.json')['class_catalog'],control=bool(job['control']))").body[0])
    run.body.insert(i+2,ast.parse("session.shared_control=bool(job['control'])").body[0])
    # Bind protocol.json into the original immutable marker's file map.
    anchors=[n for n in ast.walk(run) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='files' for t in n.targets)]
    if len(anchors)!=1 or not isinstance(anchors[0].value,ast.Dict):raise ValueError('native marker proof anchor changed')
    anchors[0].value.keys.append(ast.parse("str(out/'protocol.json')",mode='eval').body)
    anchors[0].value.values.append(ast.parse("sha(out/'protocol.json')",mode='eval').body)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[run],type_ignores=[])),__file__,'exec'),value.__dict__)
    value.native_run=value.run
    _ENGINE=value;value.g.amos.verify=verify
    return value


def worker(job):
    os.environ['V9_SHARED_CACHE_WRITE_POLICY']='read_only'
    engine().native_run(job)


def make_attempt(out,job,gpu,available,completed):
    pending=[x for x in job['pending'] if (job['model'],p.key(x['task'])) not in completed]
    if not pending:return None
    out=Path(out);tag=job['name']+'_a'+str(job['attempt'])
    paths=dict(run=str(out/'runs'/tag),spec=str(out/'job_specs'/(tag+'.json')),log=str(out/'logs'/(tag+'.log')))
    if any(Path(v).exists() for v in paths.values()):raise FileExistsError('exclusive attempt already exists')
    value=dict(job,pending=pending,task_count=len(pending),gpu=gpu,gpu_uuid=available[gpu]['uuid'],**paths)
    canonical=next(j for j in read(out/'jobs.json')['controls' if job['control'] else 'cases'] if j['name']==job['name'])
    validate_attempt(value,canonical,out);atomic_new(value['spec'],value);Path(value['log']).parent.mkdir(parents=True,exist_ok=True)
    return value


class Backend:
    def __init__(self,out,control):self.out,self.control=Path(out),control
    def records(self):return class_records(self.out,self.control)
    def available(self):return engine().g.gpu_snapshot()
    def now(self):return time.time()
    def sleep(self,seconds):time.sleep(seconds)
    def attempt(self,j,g,a,completed):return make_attempt(self.out,j,g,a,completed)
    def log(self,j):return Path(j['log']).read_text(encoding='utf-8',errors='replace')[-30000:]
    def status(self,state):write_status(self.out/'status.json',state)
    def emit(self,*args):print(*args,flush=True)
    def complete(self,j):
        value=read(Path(j['run'])/'worker_complete.json')
        if value.get('classes')!=j['task_count'] or not value.get('carrier_only'):raise ValueError('missing complete worker proof')
    def spawn(self,j):
        handle=Path(j['log']).open('x',encoding='utf-8')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(j['gpu']),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                 PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',V9_SHARED_CACHE_WRITE_POLICY='read_only')
        try:proc=subprocess.Popen([sys.executable,__file__,'worker','--job',j['spec']],cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
        except BaseException:handle.close();raise
        return proc,handle


def schedule_queue(jobs,control,backend):
    """One worker/card. Every fatal path stops dispatch and drains started peers."""
    pending=copy.deepcopy(jobs);active={};failed=[];committed={};cause=None;status_writable=True
    while pending or active:
        stage='class_records'
        try:
            if not failed:committed=backend.records()
            for gpu,(proc,j,handle) in list(active.items()):
                stage='worker_poll'
                if proc.poll() is None:continue
                del active[gpu];stage='log_close';handle.close()
                if failed:continue
                stage='completed_proofs';done=backend.records()
                remaining=[x for x in j['pending'] if (j['model'],p.key(x['task'])) not in done]
                stage='worker_log';text=backend.log(j);outcome=attempt_outcome(proc.returncode,text,bool(remaining))
                if outcome=='fatal':failed.append(dict(name=j['name'],returncode=proc.returncode,log=j['log']))
                elif outcome=='retry':pending.append(retry(j,proc.returncode,text,backend.now()))
                elif proc.returncode==0:stage='worker_complete';backend.complete(j)
            if not failed:
                stage='class_records';committed=backend.records();stage='gpu_snapshot';available=backend.available()
                for gpu in ALLOWED:
                    if gpu in active:continue
                    selected=next((j for j in pending if eligible(j,gpu,available,backend.now())),None)
                    if selected is None:continue
                    pending.remove(selected)
                    if all((selected['model'],p.key(x['task'])) in committed for x in selected['pending']):continue
                    stage='make_attempt';j=backend.attempt(selected,gpu,available,committed)
                    if j is None:continue
                    stage='worker_spawn';proc,handle=backend.spawn(j);active[gpu]=(proc,j,handle)
                    stage='pipeline_log';backend.emit('SEMANTICS_START',gpu,proc.pid,j['name'],j['task_count'],'batch',j['frame_batch'])
        except BaseException as error:
            # KeyboardInterrupt/SystemExit in the controller also prohibit new work.
            failed.append(dict(stage=stage,error=repr(error)));cause=cause or error
        state=dict(time=backend.now(),phase='controls' if control else 'full',pending=len(pending),
            running={str(g):dict(pid=pr.pid,name=j['name'],spec=j['spec'],log=j['log']) for g,(pr,j,h) in active.items()},
            failed=copy.deepcopy(failed),committed_classes=len(committed))
        if status_writable:
            try:backend.status(state)
            except BaseException as error:
                failed.append(dict(stage='status_write',error=repr(error)));cause=cause or error;status_writable=False
        if failed and not active:raise RuntimeError('fatal queue error; started peers drained naturally; evidence retained: '+repr(failed)) from cause
        if pending or active:
            try:backend.sleep(10)
            except BaseException as error:
                failed.append(dict(stage='controller_wait',error=repr(error)));cause=cause or error
    return state


def schedule(out,control):
    verify(out)
    if not control:check_controls(out)
    return schedule_queue(read(Path(out)/'jobs.json')['controls' if control else 'cases'],control,Backend(out,control))


def summarize(rows):
    rows=sorted(rows,key=p.key);p.validate_rows(rows,rows);n=len(rows)
    if not n:raise ValueError('no complete rows')
    by_class=defaultdict(list)
    for r in rows:by_class[p.key(r)[2]].append(float(r['dice']))
    return dict(task_count=n,mean={m:math.fsum(float(r[m]) for r in rows)/n for m in p.METRICS},
        class_macro_dice=math.fsum(math.fsum(v)/len(v) for _,v in sorted(by_class.items()))/len(by_class),
        fallback_count=sum(r['dynamic_threshold_source'] in p.NATIVE_FALLBACK_SOURCES for r in rows),
        per_case_class=copy.deepcopy(rows))


def paired_delta(rows,reference):
    p.validate_rows(reference,rows);old={p.key(r):r for r in reference}
    return {m:math.fsum(float(r[m])-float(old[p.key(r)][m]) for r in sorted(rows,key=p.key))/len(rows) for m in p.METRICS}


def merge(out):
    out=Path(out)
    if (out/'merged_metrics.json').exists():raise FileExistsError('once-only merged result already exists')
    require_merge_state(read(out/'status.json'),related());verify(out);snapshot(out,rehash=True);check_controls(out)
    tasks=read(out/'manifest.json')['tasks'];records=class_records(out)
    if len(records)!=1540:raise ValueError('1540 fresh complete classes required; controls excluded')
    old=read(ROOT/OLD/'merged_metrics.json');result={}
    for model,floor in FLOORS.items():
        rows=[records[(model,p.key(t))]['record']['results'][0]['row'] for t in tasks];p.validate_rows(rows,tasks)
        reference=old['models'][model]['floors'][format(floor,'.2f')]['per_case_class']
        summary=summarize(rows);summary['delta_from_old_same_floor']=paired_delta(rows,reference)
        summary['floor']=floor
        summary['actual_frame_batches']={str(b):sum(v['spec']['frame_batch']==b for (m,k),v in records.items() if m==model) for b in (1,2)}
        result[model]=summary
    value=dict(models=result,rows=1540,predictions=1540,task_count_per_model=770,controls_in_final=False,
        protocol_sha=sha(out/'protocol.json'),scope='Bone20/47/770 fixed floors; fresh training-aligned semantics',
        caveat='Paired old-semantic deltas may also reflect actual frame-batch differences; no new floor selection.')
    # nnInteractive reference is read only and must share the exact frozen770 keys.
    nn=ROOT/NN_REFERENCE
    if nn.is_file():
        reference=read(nn)
        rows=reference.get('per_case_class',reference.get('models',{}).get('nninteractive',{}).get('per_case_class',[]))
        p.validate_rows(rows,tasks);value['nninteractive_reference_sha256']=sha(nn)
        for model in FLOORS:value['models'][model]['delta_from_nninteractive']=paired_delta(value['models'][model]['per_case_class'],rows)
    else:raise FileNotFoundError('same770 nnInteractive reference required before final merge: '+str(nn))
    verify(out);require_merge_state(read(out/'status.json'),related());atomic_new(out/'merged_metrics.json',value)
    print('SEMANTICS_MERGED1540',flush=True);return value


def parser():
    value=argparse.ArgumentParser(description=__doc__)
    value.add_argument('mode',choices=('prepare','run','worker','status','merge'));value.add_argument('--job')
    return value


def run_pipeline(out):
    pipeline=ROOT/PIPELINE;preflight(out,pipeline=pipeline);pipeline.parent.mkdir(parents=True,exist_ok=True)
    # The runner owns this pathname. A launcher may use a different bootstrap log.
    with pipeline.open('x',encoding='utf-8',buffering=1) as log,redirect_stdout(log),redirect_stderr(log):
        try:
            prepare(out,already_checked=True)
            schedule(out,True);check_controls(out);print('SEMANTICS_TWO_NATIVE_CONTROLS_PASS',flush=True)
            schedule(out,False);merge(out);print('SEMANTICS_PIPELINE_RC=0',flush=True)
        except BaseException:
            traceback.print_exc();print('SEMANTICS_PIPELINE_RC=1',flush=True);raise


def main():
    args=parser().parse_args();out=ROOT/OUT
    if args.mode=='worker':
        if not args.job:raise ValueError('--job required')
        worker(read(args.job))
    elif args.mode=='prepare':prepare(out)
    elif args.mode=='run':run_pipeline(out)
    elif args.mode=='merge':merge(out)
    else:
        verify(out);records=class_records(out)
        print(json.dumps(dict(queue=read(out/'status.json'),committed_rows=len(records),controls=len(class_records(out,True))),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
