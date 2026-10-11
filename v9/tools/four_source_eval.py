"""Independent 30-patient external evaluation; no CT or prediction-volume writes."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('fumpe', 'graz', 'pengwin', 'adrenal')
METRICS = ('dice','mean_slice_dice','iou','precision','recall','specificity',
           'pred_gt_ratio','mean_slice_nsd_1mm','mean_slice_nsd_2mm','mean_slice_nsd_3mm')
WEIGHT = 'v10/output/v10_2_ct13_e470_zero_delta_e510_20261001/20261001_215853/epoch490.pth'
CACHE = 'v10/.cache/totalseg_sam2_promptgen_v10_prompt_roi_full_v1'
OUT = 'output/four_source_v102e490_nninteractive_30case_20261011'
NAMES = {'fumpe': ['pulmonary_embolism'], 'graz': ['aortic_true_lumen','aortic_false_lumen'],
         'pengwin': ['sacrum_fragments','left_hip_bone_fragments','right_hip_bone_fragments','femur_fragments'],
         'adrenal': ['adrenocortical_carcinoma_tumor']}
TEXT = {'pulmonary_embolism': 'pulmonary embolism lesion inside the pulmonary artery on CT angiography',
        'aortic_true_lumen': 'true lumen of the aorta in aortic dissection on CT angiography',
        'aortic_false_lumen': 'false lumen of the aorta in aortic dissection on CT angiography',
        'sacrum_fragments': 'all sacral bone fragments of a fractured sacrum on CT',
        'left_hip_bone_fragments': 'all left hip bone fragments of a fractured pelvis on CT',
        'right_hip_bone_fragments': 'all right hip bone fragments of a fractured pelvis on CT',
        'femur_fragments': 'all femoral bone fragments on CT',
        'adrenocortical_carcinoma_tumor': 'adrenocortical carcinoma tumor mass on contrast enhanced CT, not normal adrenal tissue'}
CLASSES = {}; _id = 2001
for _source in SOURCES:
    CLASSES[_source] = []
    for _value, _name in enumerate(NAMES[_source],1):
        _morph = 'tubular' if _source == 'graz' else 'compact'
        CLASSES[_source].append(dict(dataset=_source,global_class_id=_id,local_class_id=_value,
            label_value=_value,class_name=_name,semantic_text=TEXT[_name],semantic_key='target.'+_name,
            parent_semantic_key=None,equivalence_group='target.'+_name,morphology_group=_morph))
        _id += 1


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(temp,path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def task_key(r):
    return str(r['source_name']),str(r['case_id']),int(r.get('global_class_id',r.get('class_id')))


def bind_protocol(path,payload):
    if Path(path).is_file():
        if read(path)!=payload:raise ValueError('protocol/code/checkpoint changed; use a new output series')
    else:write(path,payload)


def shard_cases(rows,count):
    if count<1:raise ValueError('positive shard count required')
    groups=defaultdict(list)
    for r in rows:groups[task_key(r)[:2]].append(r)
    parts=[[] for _ in range(count)]
    for k,values in sorted(groups.items(),key=lambda p:(-len(p[1]),p[0])):
        owner=min(range(count),key=lambda i:(len(parts[i]),i));parts[owner].extend(values)
    return [sorted(part,key=task_key) for part in parts]


def audit_prompt(package,target):
    import numpy as np
    shape=target.shape
    counts=[]
    for plane in ('coronal','sagittal'):
        fg=np.asarray(package[plane])>0
        if fg.shape!=shape or not fg.any() or np.any(fg & (target==0)):
            raise ValueError('foreground prompt is empty/misaligned/outside GT')
        counts.append(int(fg.sum()))
    pts=np.asarray(package['background_points_dhw'],dtype=int).reshape(-1,3)
    if np.any(pts<0) or np.any(pts>=np.asarray(shape)) or np.any(target[tuple(pts.T)]):
        raise ValueError('background prompt is out of grid or inside GT')
    return dict(foreground_inside_gt=True,background_outside_gt=True,foreground_counts=counts,
                background_count=len(pts),shape_dhw=list(shape))


def validate_results(rows,expected):
    keys=[task_key(r) for r in rows]
    if len(keys)!=len(set(keys)) or set(keys)!={task_key(r) for r in expected}:
        raise ValueError('result task coverage/uniqueness mismatch')
    for r in rows:
        if any(m not in r or not math.isfinite(float(r[m])) for m in METRICS):
            raise ValueError('nonfinite/missing metric')


def validate_model_rows(rows,expected,model):
    validate_results(rows,expected)
    for row in rows:
        if not row.get('prompt_sha256'):raise ValueError('missing prompt signature')
        if model=='nninteractive':
            if row.get('prediction_trigger_count')!=1:raise ValueError('missing/invalid final prediction trigger audit')
        elif model=='v10_2':
            threshold=float(row.get('mask_threshold_used',float('nan')))
            if not .35<=threshold<=.75 or not row.get('dynamic_threshold_source'):
                raise ValueError('missing/invalid dynamic threshold audit')
        else:raise ValueError('unknown model')


def metric_path():
    return ROOT/'utils/metrics.py'


def metric_module():
    spec=importlib.util.spec_from_file_location('_four_source_unified_metrics',metric_path())
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def setup():
    os.chdir(ROOT)
    for p in (ROOT,ROOT/'tools',ROOT/'third_party/four_source_eval_deps'):
        if str(p) not in sys.path:sys.path.insert(0,str(p))


def nn_modules(gpu='0'):
    folder=ROOT.parent/'nninteractive_test_amos/infer'
    if not (folder/'infer_3d_point_common.py').is_file():folder=ROOT/'nninteractive_test/infer'
    for p in (folder.parent,folder):sys.path.insert(0,str(p))
    sys.argv=[sys.argv[0],'--gpu',str(gpu),'--shared-case-cache-root',str(ROOT/CACHE)]
    protocol=importlib.import_module('eight_source_adaptive_protocol');protocol.EIGHT_SOURCES=SOURCES
    ev=importlib.import_module('infer_eight_source_adaptive_nninteractive')
    return protocol,ev


class InMemoryPairs:
    """One canonical case in RAM; lightweight reference files are not NIfTI exports."""
    def __init__(self,out):
        self.out=Path(out);self.current=None;self.pair=None;self.original=None

    def load(self,path,*args,**kwargs):
        p=Path(path).resolve()
        if p.parent.parent != (self.out/'references').resolve():
            return self.original(path,*args,**kwargs)
        record=read(p);identity=(record['source_name'],record['case_id'])
        if identity!=self.current:
            from four_source_data import load_pair
            self.pair=load_pair(record);self.current=identity
        return self.pair[0] if p.name=='image.nii.gz' else self.pair[1]

    def install(self):
        import nibabel as nib
        self.original=nib.load;nib.load=self.load


def manifest(rows):
    return dict(manifest_kind='independent_eight_source_validation_v1',expected_task_count=len(rows),
                per_source_task_counts=dict(Counter(r['source_name'] for r in rows)),tasks=rows)


def freeze(out,data_root):
    from four_source_data import discover,select_cases
    frozen=out/'selected_cases.json'
    if frozen.is_file():return read(frozen)
    candidates=discover(data_root)
    selected={s:select_cases(candidates[s],count=30,seed=20261011) for s in SOURCES}
    write(frozen,selected)
    return selected


def prepare(out,data_root):
    import numpy as np
    from four_source_data import load_pair
    selected=freeze(out,data_root);protocol,ev=nn_modules()
    adaptive=protocol.load_adaptive_module(ROOT/'v9_2_adaptive_scribble.py');args=ev.parse_args()
    args.adaptive_min_radius=1;args.adaptive_max_radius=20;args.adaptive_width_fraction=.15
    args.background_points_per_plane=2;args.negative_point_mode='bbox_outside';args.point_seed=2026
    rows=[];refs=[];geometry=[];na=[];datasets={}
    progress=read(out/'preparation_progress.json') if (out/'preparation_progress.json').exists() else {}
    for source in SOURCES:
        datasets[source]={'train':[],'val':[]}
        for rec in selected[source]:
            directory=out/'references'/f"{source}_{rec['case_id']}"
            image_path=directory/'image.nii.gz';label_path=directory/'label.nii.gz'
            # These are <10 KiB JSON references.  Source files are never converted/copied.
            for p in (image_path,label_path):
                if p.exists() and read(p)!=rec:raise ValueError('frozen case reference changed')
                if not p.exists():write(p,rec)
            image,label,audit=load_pair(rec)
            target_full=np.asarray(label.dataobj).transpose(2,0,1)
            geometry.append(dict(source=source,case_id=rec['case_id'],**audit))
            classes=[]
            for meta in CLASSES[source]:
                cid=int(meta['global_class_id']);local=int(meta['local_class_id'])
                target=(target_full==local).astype(np.uint8)
                if not target.any():
                    na.append(dict(source_name=source,case_id=rec['case_id'],global_class_id=cid,reason='GT_empty'));continue
                classes.append(cid)
                row=dict(source_name=source,case_id=rec['case_id'],global_class_id=cid,local_class_id=local,
                         class_name=meta['class_name'],image=str(image_path),label=str(label_path),split_role='validation')
                task=protocol.AdaptiveValidationTask(source,rec['case_id'],cid,local,meta['class_name'],image_path,label_path)
                path=ev.adaptive_prompt_cache_path(ROOT/CACHE,task,label_path,min_radius=1,max_radius=20,
                    width_fraction=.15,point_seed=ev._stable_seed(2026,source,rec['case_id'],cid),
                    background_points_per_plane=2,negative_point_mode='bbox_outside')
                package,cache_status=ev.load_or_create_adaptive_prompt_package(path,
                    lambda:ev._build_prompt_package(args,task,target,'memory_only',adaptive))
                pa=audit_prompt(package,target)
                if pa['background_count']!=4:raise ValueError('exactly two background points per plane required')
                rows.append(row);refs.append(dict(**row,prompt_cache_name=path.name,prompt_sha256=sha(path),
                    prompt_protocol=ev.PROMPT_PROTOCOL,prompt_audit=pa))
            datasets[source]['val'].append(dict(case_id=rec['case_id'],image=str(image_path),label=str(label_path),classes=classes,modality='ct'))
            write(out/'preparation_progress.json',dict(completed_cases=len(geometry),total_cases=120,last_case=[source,rec['case_id']]))
            print(f"PREPARE {source}/{rec['case_id']} cases={len(geometry)}/120 tasks={len(rows)}",flush=True)
            del image,label,target_full
    split=dict(source_order=list(SOURCES),class_catalog=[r for s in SOURCES for r in CLASSES[s]],datasets=datasets)
    write(out/'split.json',split);write(out/'manifest.json',manifest(rows));write(out/'refs.json',{'per_case_class':refs})
    write(out/'geometry_audit.json',dict(per_case=geometry,not_applicable=na,patient_counts={s:len(selected[s]) for s in SOURCES}))
    jobs=[]
    for source in SOURCES:
        parts=shard_cases([r for r in rows if r['source_name']==source],6)
        for i,part in enumerate(parts):
            if not part:continue
            for model in ('v10_2','nninteractive'):
                name=f'{model}_{source}_s{i:02}'
                jobs.append(dict(name=name,model=model,source=source,tasks=part,out=str(out),frame_batch=16,attempt=0))
    write(out/'jobs.json',jobs)


def fingerprint(out):
    code=['tools/four_source_eval.py','tools/four_source_data.py','utils/metrics.py','v9_2_adaptive_scribble.py',
          'v10/infer/infer_eight_source_adaptive_v10_2.py','nninteractive_test/infer/prompt_protocol.py']
    bundle=ROOT.parent/'nninteractive_test_amos'
    nn_assets=[bundle/'checkpoints/nnInteractive_v1.0'/p for p in (
        'fold_0/checkpoint_final.pth','plans.json','dataset.json','inference_session_class.json')]
    nn_code=[bundle/'infer'/p for p in ('infer_eight_source_adaptive_nninteractive.py',
        'eight_source_adaptive_protocol.py','adaptive_prompt_cache.py')]
    return dict(code={p:sha(ROOT/p) for p in code},checkpoint=WEIGHT,checkpoint_sha=sha(ROOT/WEIGHT),
        nn_assets={str(p.relative_to(ROOT.parent)):sha(p) for p in nn_assets+nn_code},
        selected_sha=sha(out/'selected_cases.json'),manifest_sha=sha(out/'manifest.json'),
        prompt_refs_sha=sha(out/'refs.json'),metrics=list(METRICS),threshold=[.35,.75,.60],context=192,
        foreground='adaptive_coronal_1_sagittal_1',background='2_points_per_plane',ct_cache='read_only_or_memory',
        semantic_text=TEXT,selection_seed=20261011,cases_per_source=30)


def prompt_refs(out,rows):
    index={task_key(r):r for r in read(out/'refs.json')['per_case_class']}
    selected=[index[task_key(r)] for r in rows]
    for r in selected:
        p=ROOT/CACHE/'metadata/shared_adaptive_prompt_v1'/r['source_name']/r['prompt_cache_name']
        if sha(p)!=r['prompt_sha256']:raise ValueError('shared prompt SHA changed')
    return selected


def run_nn(job,rows,attempt):
    import numpy as np
    import nibabel as nib
    import torch
    protocol,ev=nn_modules(job['gpu']);ev.patch_nninteractive_trainer_for_current_nnunet()
    from adaptive_prompt_cache import _read
    session=ev.nnInteractiveInferenceSession(device=torch.device('cuda:0'),use_torch_compile=False,
        verbose=False,torch_n_threads=2,do_autozoom=True,use_pinned_memory=True)
    session.initialize_from_trained_model_folder(str(ROOT.parent/'nninteractive_test_amos/checkpoints/nnInteractive_v1.0'))
    binary_prompt_metrics_from_masks=metric_module().binary_prompt_metrics_from_masks
    refs={task_key(r):r for r in prompt_refs(Path(job['out']),rows)};done=[];current=None;image=None
    for row in rows:
        if current!=row['image']:
            ct=nib.load(row['image']);image=np.asarray(ct.dataobj,dtype=np.float32).transpose(2,0,1)
            spacing=tuple(float(ct.header.get_zooms()[i]) for i in (2,0,1))
            session.set_image(image[None]);current=row['image']
        label=np.asarray(nib.load(row['label']).dataobj).transpose(2,0,1)
        target=(label==int(row['local_class_id'])).astype(np.uint8)
        ref=refs[task_key(row)];p=ROOT/CACHE/'metadata/shared_adaptive_prompt_v1'/row['source_name']/ref['prompt_cache_name']
        package=_read(p);audit_prompt(package,target)
        buffer=torch.zeros(image.shape,dtype=torch.uint8,device='cpu');session.set_target_buffer(buffer)
        # Count requested prediction triggers, not internal sliding patch forwards.
        calls=[]
        class TriggerAudit:
            def __getattr__(self,name):
                fn=getattr(session,name)
                if name not in ('add_scribble_interaction','add_point_interaction'):return fn
                def call(*args,**kwargs):
                    calls.append(bool(kwargs.get('run_prediction',True)));return fn(*args,**kwargs)
                return call
        ev.apply_two_plane_prompt(TriggerAudit(),package['coronal'],package['sagittal'],package['background_points_dhw'])
        if sum(calls)!=1:raise RuntimeError('nnInteractive expected one final prediction trigger')
        metric=binary_prompt_metrics_from_masks(buffer,torch.from_numpy(target),spacing_dhw=spacing)
        result={**row,**{m:float(metric[m]) for m in METRICS},'prediction_trigger_count':1,
                'prompt_sha256':ref['prompt_sha256'],'prompt_cache_name':ref['prompt_cache_name']}
        done.append(result);write(attempt/'all_metrics_live.json',{'per_case_class':done})
        print(f"NN {task_key(row)} {len(done)}/{len(rows)} Dice={result['dice']:.5f}",flush=True)
        session.reset_interactions();torch.cuda.empty_cache()
    return done


def run_v10(job,rows,attempt):
    import numpy as np
    import torch
    import v10.finetune_totalseg_amos_magic_v10_2_joint as entry
    import v10_2_data as data
    data.EIGHT_SOURCE_ORDER=SOURCES;entry.EIGHT_SOURCE_ORDER=SOURCES
    entry.extended_source_ratios=lambda _args:{s:.25 for s in SOURCES}
    original_bind=entry._bind_v10_2
    def bind(args):
        args.max_frames=192;args.validation_window_frames=192
        return original_bind(args)
    entry._bind_v10_2=bind
    # No raw CT writes, and reuse the one in-memory case across every class.
    import nibabel as nib
    data.SingleFileMulticlassV102Dataset._get_or_create_case_image=lambda self,p:np.asarray(nib.load(p).dataobj,dtype=np.float32).transpose(2,0,1)
    data.SingleFileMulticlassV102Dataset._load_array=lambda self,p:np.asarray(nib.load(p).dataobj).transpose(2,0,1)
    binary_prompt_metrics_from_masks=metric_module().binary_prompt_metrics_from_masks
    import v10.infer.infer_amos_sam2_promptgen_v10 as scorer
    scorer.binary_prompt_metrics_from_masks=binary_prompt_metrics_from_masks
    original_score=scorer.restore_and_score_full_volume;live=[];dynamic={}
    def score(*args,**kwargs):
        result=original_score(*args,**kwargs);row=rows[len(live)]
        ref=next(r for r in refs if task_key(r)==task_key(row))
        live.append({**row,**{m:float(result[0][m]) for m in METRICS},
            'mask_threshold_used':float(kwargs['threshold']),
            'dynamic_threshold_source':str(dynamic['source']),
            'prompt_sha256':ref['prompt_sha256'],'prompt_cache_name':ref['prompt_cache_name']})
        write(attempt/'all_metrics_live.json',{'per_case_class':live})
        print(f"V10 {task_key(row)} {len(live)}/{len(rows)} Dice={result[0]['dice']:.5f}",flush=True)
        return result
    scorer.restore_and_score_full_volume=score
    out=Path(job['out']);write(attempt/'manifest.json',manifest(rows))
    refs=prompt_refs(out,rows);write(attempt/'refs.json',{'per_case_class':refs})
    sys.argv=[sys.argv[0],'--independent-task-manifest',str(attempt/'manifest.json'),
        '--nninteractive-metrics',str(attempt/'refs.json'),'--shared-adaptive-prompt-root',str(ROOT/CACHE),
        '--weights',str(ROOT/WEIGHT),'--source',job['source'],'--background-mode','point',
        '--eval-out-dir',str(attempt),'--run-name','result','--gpu',str(job['gpu']),
        '--sam2-frame-batch-size',str(job['frame_batch']),'--base-split-json',str(out/'split.json')]
    import v10.infer.infer_eight_source_adaptive_v10_2 as ev
    original_decode=ev._dynamic_adaptive_decode
    def decode(*args,**kwargs):
        result=original_decode(*args,**kwargs);dynamic.clear();dynamic.update(result[4]);return result
    ev._dynamic_adaptive_decode=decode
    ev.main()
    result=read(attempt/'result/validation_only_metrics.json')['per_case_class']
    index={task_key(r):r for r in refs}
    for r in result:
        ref=index[task_key(r)];r['prompt_sha256']=ref['prompt_sha256'];r['prompt_cache_name']=ref['prompt_cache_name']
    return result


def worker(job):
    out=Path(job['out']);bind_protocol(out/'protocol.json',fingerprint(out))
    os.environ['CUDA_VISIBLE_DEVICES']=str(job['gpu']);os.environ['V9_SHARED_CACHE_WRITE_POLICY']='read_only'
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
    import torch
    torch.set_num_threads(2);visible='GPU-'+str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')
    if visible!=job['gpu_uuid']:raise RuntimeError('physical GPU/UUID/CUDA mapping changed')
    bridge=InMemoryPairs(out);bridge.install()
    run=out/'jobs'/job['name'];prior=[]
    for p in sorted(run.glob('attempt_*/all_metrics_live.json')):
        for r in read(p)['per_case_class']:
            if task_key(r) not in {task_key(v) for v in prior}:prior.append(r)
    completed={task_key(r) for r in prior};rows=sorted([r for r in job['tasks'] if task_key(r) not in completed],key=task_key)
    attempt=run/f"attempt_{job['attempt']:02}";attempt.mkdir(parents=True,exist_ok=True)
    if rows:
        results=(run_v10 if job['model']=='v10_2' else run_nn)(job,rows,attempt)
        write(attempt/'all_metrics_live.json',{'per_case_class':results});prior.extend(results)
    refs={task_key(r):r for r in prompt_refs(out,job['tasks'])}
    for row in prior:
        ref=refs[task_key(row)];row['prompt_sha256']=ref['prompt_sha256'];row['prompt_cache_name']=ref['prompt_cache_name']
    validate_model_rows(prior,job['tasks'],job['model']);write(run/'complete_metrics.json',{'model':job['model'],'per_case_class':prior})


def status(out):
    jobs=read(out/'jobs.json');counts={};done=0;partial=0
    for model in ('v10_2','nninteractive'):
        count=0
        for j in jobs:
            if j['model']!=model:continue
            run=out/'jobs'/j['name'];p=run/'complete_metrics.json'
            if p.is_file():count+=len(read(p)['per_case_class']);done+=1
            else:
                keys=set()
                for q in run.glob('attempt_*/all_metrics_live.json'):keys.update(task_key(r) for r in read(q)['per_case_class'])
                count+=len(keys);partial+=len(keys)
        counts[model]=count
    return dict(completed_jobs=done,total_jobs=len(jobs),task_progress=counts,expected_per_model=len(read(out/'manifest.json')['tasks']),
                scheduler=read(out/'status.json') if (out/'status.json').exists() else None)


def merge(out):
    rows=read(out/'manifest.json')['tasks'];jobs=read(out/'jobs.json');result={}
    for model in ('v10_2','nninteractive'):
        combined=[]
        for j in jobs:
            if j['model']==model:combined.extend(read(out/'jobs'/j['name']/'complete_metrics.json')['per_case_class'])
        validate_model_rows(combined,rows,model)
        result[model]={'per_case_class':combined,'task_count':len(combined),'per_source':{}}
        for source in SOURCES:
            values=[r for r in combined if r['source_name']==source]
            result[model]['per_source'][source]={'tasks':len(values),**{m:sum(float(r[m]) for r in values)/len(values) for m in METRICS}}
    a={task_key(r):r['prompt_sha256'] for r in result['v10_2']['per_case_class']}
    b={task_key(r):r['prompt_sha256'] for r in result['nninteractive']['per_case_class']}
    if a!=b:raise ValueError('two methods did not use identical prompts')
    write(out/'merged_metrics.json',result);return result


def schedule(out,gpus):
    import fcntl
    lock=(out/'scheduler.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    bind_protocol(out/'protocol.json',fingerprint(out))
    pending=[j for j in read(out/'jobs.json') if not (out/'jobs'/j['name']/'complete_metrics.json').exists()]
    running={};failed=[]
    while pending or running:
        for gpu,(p,j,h) in list(running.items()):
            if p.poll() is None:continue
            h.close();del running[gpu]
            if p.returncode:
                tail=Path(j['log']).read_text(errors='replace')[-14000:]
                if 'out of memory' in tail.lower() and j['attempt']<5:
                    if j['model']=='v10_2':j['frame_batch']=max(1,j['frame_batch']//2)
                    j['attempt']+=1;pending.append(j)
                else:failed.append({'job':j['name'],'log':j['log'],'returncode':p.returncode})
            else:print('DONE',j['name'],flush=True)
        query=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
        inventory={int(l.split(',')[0]):(l.split(',')[1].strip(),int(l.split(',')[2])) for l in query.splitlines()}
        for gpu in gpus:
            if gpu in running or not pending or inventory[gpu][1]<45000:continue
            j=pending.pop(0);j.update(gpu=gpu,gpu_uuid=inventory[gpu][0])
            run=out/'jobs'/j['name']
            existing=[int(p.name.split('_')[-1]) for p in run.glob('attempt_*')]
            if existing:j['attempt']=max(j['attempt'],max(existing)+1)
            j['log']=str(out/'logs'/f"{j['name']}_a{j['attempt']:02}.log")
            write(out/'job_specs'/f"{j['name']}.json",j);Path(j['log']).parent.mkdir(parents=True,exist_ok=True)
            h=Path(j['log']).open('w');exe=ROOT.parent/'conda-envs'/('al' if j['model']=='v10_2' else 'nninteractive_eval')/'bin/python'
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
            p=subprocess.Popen([str(exe),str(Path(__file__).resolve()),'worker','--job',str(out/'job_specs'/f"{j['name']}.json")],
                cwd=ROOT,env=env,stdout=h,stderr=subprocess.STDOUT)
            running[gpu]=(p,j,h);print(f"START {j['name']} GPU{gpu} {j['gpu_uuid']} pid={p.pid}",flush=True)
        write(out/'status.json',dict(time=time.strftime('%Y-%m-%d %H:%M:%S'),pending=len(pending),
            running={str(g):dict(name=j['name'],pid=p.pid,log=j['log']) for g,(p,j,h) in running.items()},failed=failed))
        if pending or running:time.sleep(10)
    if failed:raise RuntimeError('worker failures; evidence preserved in status.json')
    merge(out)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('prepare','worker','schedule','status','merge'))
    p.add_argument('--out',default=OUT);p.add_argument('--data-root',default='../data');p.add_argument('--job');p.add_argument('--gpus',default='0,3,4,5')
    args=p.parse_args();setup();out=(ROOT/args.out).resolve()
    if args.mode=='prepare':prepare(out,(ROOT/args.data_root).resolve())
    elif args.mode=='worker':worker(read(args.job))
    elif args.mode=='schedule':schedule(out,[int(v) for v in args.gpus.split(',')])
    elif args.mode=='status':print(json.dumps(status(out),ensure_ascii=False,indent=2))
    else:merge(out)


if __name__=='__main__':main()
