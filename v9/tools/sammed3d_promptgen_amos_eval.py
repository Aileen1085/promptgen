"""E230 SAM-Med3D/PromptGen, full AMOS, immutable historical FG/BG prompts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import MethodType

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ct13_full_validation_eval as base

ROOT=base.ROOT
WEIGHT='sammed3d/output/amos_turbo_20261009/epoch230.pth'
WEIGHT_SHA='e815061263af44203d9699bbfb14388323d448858e44718fbbba7f1f70837b6a'
CORE='third_party/SAM_Med3D_official_20261007/checkpoints/sam_med3d_turbo.pth'
CORE_SHA='899a46d04d3b70f723282ceb489149373558bf0aaba389a346f5ab57da5cdd3c'
DONOR='v10/output/v10_2_ct13_e490_decoder_frozen_control_10_20261005/20261005_155255/epoch010.pth'
DONOR_SHA='d3eefcbb8cbe99e30e718c2e8ef842dc3c1957272e0e8120a1fa98a7e44e0ac9'
ARCH='sammed3d_turbo_promptgen_v102_amos_v1'
REFERENCE='output/ct13_full_val_v92e360_v102e490_20261005'
OUT='output/sammed3d_promptgen_e230_amos_full1485_20261021'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def validate_scope(rows,split,expected=1485,cases=100):
    val={r['case_id'] for r in split['datasets']['amos']['val']}
    train={r['case_id'] for r in split['datasets']['amos']['train']}
    keys=[base.key(r) for r in rows]
    if (len(keys)!=expected or len(set(keys))!=expected or len({k[1] for k in keys})!=cases
            or any(k[0]!='amos' or k[1] not in val or k[1] in train for k in keys)):
        raise ValueError('wrong AMOS held-out scope or duplicate tasks')


def validate_checkpoint(s,split_sha):
    for k,v in dict(epoch=230,architecture=ARCH,split_sha256=split_sha,
                    turbo_sha256=CORE_SHA,promptgen_donor_sha256=DONOR_SHA).items():
        if s.get(k)!=v:raise ValueError('checkpoint identity mismatch: '+k)


def append_exact_background(foreground_cube_points,background_cube_points):
    import numpy as np
    fg=np.asarray(foreground_cube_points,float).reshape(-1,3)
    bg=np.asarray(background_cube_points,float).reshape(-1,3)
    if not len(fg) or len(bg)!=4 or not np.isfinite(fg).all() or not np.isfinite(bg).all():
        raise ValueError('exact two-plane FG and four finite BG points required')
    return np.concatenate((fg,bg)),np.r_[np.ones(len(fg),np.int64),np.zeros(4,np.int64)]


def protocol(out):
    files=[Path(__file__),ROOT/'tools/ct13_full_validation_eval.py',
           ROOT/'tools/native_ct13_eval.py',ROOT/'tools/native_sam2_physical_eval.py',
           ROOT/'utils/metrics.py',ROOT/'v10/infer/infer_amos_sam2_promptgen_v10.py']
    for folder in ('sammed3d','model','v10/model'):
        files+=sorted((ROOT/folder).glob('*.py'))
    return dict(model='SAM-Med3D Turbo + PromptGen',epoch=230,weight=WEIGHT,
        weights={p:sha(ROOT/p) for p in (WEIGHT,CORE,DONOR)},
        code={str(p.relative_to(ROOT)):sha(p) for p in files},split_sha=sha(ROOT/base.SPLIT),
        manifest_sha=sha(out/'manifest.json'),refs_sha=sha(out/'refs.json'),
        input='training physical fit128 XYZ, native normalization + PG three medical channels',
        prompts='exact cached FG coronal/sagittal scribble, exact 4 BG points; no resimulation',
        native_points='FG FPS4 per plane; ALL four cached BG points, no BG FPS reduction',
        semantic='training SAT128, legacy AMOS class names',
        threshold='prompt dynamic 0.35-0.75 fallback0.60, minimum logit separation0.50',
        roi='training prompt8mm20%, FG-connected expand50 cap100, no GT bbox',
        metrics=list(base.METRICS),metric_scope='final full mask; axial union, single-empty0',
        cache='read-only shared CT and prompt cache; missing CT RAM only; no mask export')


def verify(out):
    if protocol(out)!=base.read(out/'protocol.json'):
        raise ValueError('code/checkpoint/prompt protocol changed; refusing unsafe resume')


def prepare(out):
    import numpy as np
    if out.exists():raise FileExistsError('output exists; use launch/status, not prepare')
    for p,v in ((WEIGHT,WEIGHT_SHA),(CORE,CORE_SHA),(DONOR,DONOR_SHA)):
        if sha(ROOT/p)!=v:raise ValueError('weight hash mismatch: '+p)
    split=base.read(ROOT/base.SPLIT)
    rows=sorted([r for r in base.read(ROOT/REFERENCE/'manifest.json')['tasks']
                 if r['source_name']=='amos'],key=base.key)
    validate_scope(rows,split)
    refs={}
    for p in sorted((ROOT/REFERENCE/'prepared').glob('amos_s*of12/refs.json')):
        for r in base.read(p)['per_case_class']:
            k=base.key(r)
            if k in refs:raise ValueError('duplicate historical prompt ref')
            path=ROOT/base.CACHE/'metadata/shared_adaptive_prompt_v1/amos'/r['prompt_cache_name']
            with np.load(path,allow_pickle=False) as z:
                if np.asarray(z['background_points_dhw']).reshape(-1,3).shape!=(4,3):
                    raise ValueError('historical BG budget mismatch')
            refs[k]=dict(r,prompt_sha256=sha(path))
    if set(refs)!={base.key(r) for r in rows}:raise ValueError('historical prompt coverage mismatch')
    base.write(out/'manifest.json',base.manifest_payload(rows))
    base.write(out/'refs.json',dict(per_case_class=[refs[base.key(r)] for r in rows]))
    first=rows[0]['case_id'];parts=[[r for r in rows if r['case_id']==first]]
    remaining=[r for r in rows if r['case_id']!=first]
    parts += [base.case_shard(remaining,24,i) for i in range(24)]
    jobs=[]
    for i,part in enumerate(parts):
        name='amos_smoke' if i==0 else 'amos_s%02d'%i
        prepared=out/'prepared'/name
        base.write(prepared/'manifest.json',base.manifest_payload(part))
        base.write(prepared/'refs.json',dict(per_case_class=[refs[base.key(r)] for r in part]))
        jobs.append(dict(name=name,model='sammed3d_promptgen',source='amos',prepared=str(prepared),
                         task_count=len(part),frame_batch=1,attempt=0))
    base.write(out/'jobs.json',jobs);base.write(out/'protocol.json',protocol(out))
    print('PREPARED100 CT/1485 tasks/25 whole-case jobs',flush=True)


class ExactLoader:
    """Read a single shared CT/label once per current case; only lightweight prompts vary."""
    def __init__(self,refs):
        self.refs={base.key(r):r for r in refs};self.current=None

    def __call__(self,task,data,native,physical):
        import numpy as np
        ref=self.refs[base.key(task)]
        path=ROOT/base.CACHE/'metadata/shared_adaptive_prompt_v1/amos'/ref['prompt_cache_name']
        if sha(path)!=ref['prompt_sha256']:raise ValueError('prompt cache signature changed')
        with np.load(path,allow_pickle=False) as z:
            shape=tuple(map(int,z['shape_dhw']))
            support=[np.unpackbits(z[k],count=int(np.prod(shape)),bitorder='little').reshape(shape).astype(bool)
                     for k in ('coronal_bits','sagittal_bits')]
            bg=np.asarray(z['background_points_dhw'],dtype=np.int32).reshape(-1,3)
            case_name=str(z['case_image_name'].item())
        if self.current!=task['case_id']:
            image,aff=physical.canonical(task['image']);label,laff=physical.canonical(task['label'])
            native.assert_compatible_grid(aff,laff,shape,image_qform=image.get_qform(),label_qform=label.get_qform())
            if shape!=tuple(image.shape[i] for i in (2,0,1)):raise ValueError('cached CT shape mismatch')
            ctpath=ROOT/base.CACHE/'_case_images_v1'/case_name
            self.ct_hit=ctpath.is_file()
            self.ct=np.load(ctpath,mmap_mode='r') if self.ct_hit else np.asarray(image.dataobj,dtype=np.float32).transpose(2,0,1)
            self.labels=np.asarray(label.dataobj).transpose(2,0,1)
            self.spacing=tuple(float(v) for v in np.linalg.norm(aff[:3,:3],axis=0))
            self.current=task['case_id'];self.case_name=case_name
        if case_name!=self.case_name or self.ct.shape!=shape or self.labels.shape!=shape:
            raise ValueError('CT/GT/prompt grids disagree')
        truth=self.labels==int(task['local_class_id']);fg=support[0]|support[1]
        if (not truth.any() or not fg.any() or not truth[fg].all() or len(bg)!=4
                or np.any(bg<0) or np.any(bg>=np.asarray(shape)) or truth[tuple(bg.T)].any()):
            raise ValueError('raw FG/BG alignment or budget failed')
        slices=native.canonical_plane_slices(support)
        if not ((bg[:,2]==slices[0]).sum()>=2 and (bg[:,1]==slices[1]).sum()>=2):
            raise ValueError('two background points per plane not present')
        audit=dict(prompt_cache_sha256=ref['prompt_sha256'],case_image_name=case_name,
                   case_cache_hit=self.ct_hit,fg_voxels=int(fg.sum()),fg_in_gt_fraction=1.,
                   bg_points_dhw=bg.tolist(),bg_in_gt_count=0,background_mode='point',
                   gt_used_for_roi=False,raw_prompt_reused=True,plane_slices=list(slices))
        return data.RawCase(self.ct,truth,fg,bg,self.spacing,(2,1),slices,audit)


def load_model(out):
    import torch
    sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'sammed3d'))
    import backend,factory,data,training,geometry
    from tools import native_ct13_eval as native
    state=torch.load(ROOT/WEIGHT,map_location='cpu',weights_only=True)
    validate_checkpoint(state,sha(ROOT/base.SPLIT))
    prompt=factory.load_v102_promptgen(ROOT/DONOR,'cuda:0')
    model=backend.SAMMedPromptGen(backend.load_core(ROOT/'third_party/SAM_Med3D_official_20261007',ROOT/CORE,'cuda:0'),prompt).cuda()
    expected={n for n,p in prompt.named_parameters() if p.requires_grad}|set(dict(prompt.named_buffers()))
    if expected!=set(state['prompt_generator']):raise ValueError('learned PromptGen checkpoint membership mismatch')
    result=prompt.load_state_dict(state['prompt_generator'],strict=False)
    if result.unexpected_keys:raise ValueError('unexpected PromptGen keys')
    for part,key in ((model.bridge,'feature_bridge'),(model.adapter,'prompt_adapter')):
        part.load_state_dict(state[key],strict=True)
    for part,key in ((prompt,'prompt_generator'),(model.bridge,'feature_bridge'),(model.adapter,'prompt_adapter')):
        for k,v in state[key].items():
            if not torch.equal(part.state_dict()[k].detach().cpu(),v):raise ValueError('tensor restore mismatch: '+key+'/'+k)
    prompt.set_morphology_routing_epoch(prompt.source_epoch+230)
    prompt._v10_forced_background_mode='point'
    prompt._one_point_per_group=MethodType(lambda self,source,fallback=None:(source>0).to(source.dtype),prompt)
    original=data.cube_prompts
    def exact_points(fg,bg,slices,g,cap=4):
        groups,planes,coords,pts,labels=original(fg,bg,slices,g,cap)
        points,roles=append_exact_background(pts[labels==1],geometry.to_cube(bg,g))
        if ((points<-.5)|(points>127.5)).any():raise ValueError('native prompt point outside cube')
        return groups,planes,coords,points,roles
    data.cube_prompts=exact_points
    model.requires_grad_(False);model.eval()
    audit=dict(epoch=230,checkpoint_sha=WEIGHT_SHA,strict_effective_modules_exact=True,
               learned_prompt_tensors=len(expected),bridge_tensors=len(state['feature_bridge']),
               adapter_tensors=len(state['prompt_adapter']),routing_epoch=prompt.source_epoch+230,
               background_mode='point',all_four_background_points=True,turbo_frozen=True)
    return model,data,training,geometry,native,audit


def predict(model,raw,data,training,geometry):
    import numpy as np
    import torch
    from v10.infer.infer_amos_sam2_promptgen_v10 import postprocess_v10_logits_from_prompts
    initial=geometry.prompt_bounds(raw.foreground,raw.background,raw.spacing);bounds=initial;stages=[]
    for iteration in range(8):
        packed=data.pack_cube(raw,bounds,'cuda:0')
        if int((packed['labels']==0).sum())!=4:raise ValueError('background point reduction')
        with torch.autocast('cuda',dtype=torch.bfloat16):result=training.forward(model,packed,raw.audit['class_name'])
        logits=result['logits'][0,0].float()
        if not torch.isfinite(logits).all():raise FloatingPointError('nonfinite SAM-Med3D logits')
        probability=geometry.restore_full(logits.sigmoid().cpu().numpy(),packed['geometry'],raw.ct.shape)
        fg=raw.foreground;bg=np.zeros(raw.ct.shape,bool);bg[tuple(raw.background.T)]=True
        native_logits=torch.logit(torch.from_numpy(probability).clamp(1e-6,1-1e-6))[:,None]
        pred,threshold=postprocess_v10_logits_from_prompts(native_logits,fg,bg,
            fallback=.60,minimum=.35,maximum=.75,min_logit_separation=.50)
        grown=geometry.expand_bounds(pred[tuple(slice(a,b) for a,b in bounds)],fg,bounds,initial)
        q=packed['points'][0].cpu().numpy()[:,[2,0,1]]
        physical_bg=geometry.from_cube(q[-4:],packed['geometry'])
        error=float(np.abs(physical_bg-raw.background).max())
        if error>1e-4:raise ValueError('background coordinate roundtrip error')
        stages.append(dict(bounds=[list(b) for b in bounds],threshold=threshold,
                           background_points_encoded=4,foreground_points_encoded=int((packed['labels']==1).sum()),
                           bg_roundtrip_voxels=error,all_prompts_before_forward=True,
                           decoder_calls=1,gt_used_for_roi=False))
        del packed,result,logits,native_logits,probability,bg
        if grown==bounds:return pred,stages
        bounds=grown
    raise RuntimeError('bounded AutoExpand did not converge')


def worker(job):
    import numpy as np
    import torch
    torch.set_num_threads(2)
    out=Path(job['run']).parents[1];verify(out)
    uuid=str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')
    if uuid!=job['gpu_uuid'].removeprefix('GPU-') or torch.ones(1,device='cuda:0').item()!=1:
        raise ValueError('physical GPU UUID/CUDA tensor mismatch')
    run=Path(job['run']);expected=base.read(Path(job['prepared'])/'manifest.json')['tasks']
    refs=base.read(Path(job['prepared'])/'refs.json')['per_case_class'];loader=ExactLoader(refs)
    rows=base.read(run/'all_metrics_live.json')['per_case_class'] if (run/'all_metrics_live.json').exists() else []
    done={base.key(r) for r in rows};base.validate_rows(rows,[t for t in expected if base.key(t) in done])
    model,data,training,geometry,native,model_audit=load_model(out)
    model_audit['gpu_uuid']=job['gpu_uuid'];base.write(run/'model_audit.json',model_audit)
    physical=native.base_module();scorer=physical.metric_module()
    with torch.inference_mode():
        for task in expected:
            if base.key(task) in done:continue
            start=time.time();raw=loader(task,data,native,physical);raw.audit['class_name']=task['class_name']
            pred,stages=predict(model,raw,data,training,geometry)
            values=scorer.binary_prompt_metrics_from_masks(torch.from_numpy(pred),torch.from_numpy(raw.truth),spacing_dhw=raw.spacing)
            row=dict(task,**{m:float(values[m]) for m in base.METRICS},seconds=time.time()-start,
                     prompt_audit=raw.audit,stages=stages,epoch=230,weight_sha256=WEIGHT_SHA)
            rows.append(row);done.add(base.key(task));base.validate_rows(rows,[t for t in expected if base.key(t) in done])
            base.write(run/'all_metrics_live.json',dict(per_case_class=rows))
            base.write(run/'exact_prompt_audit.json',dict(per_case_class=[dict(task_key=list(base.key(r)),
                prompt_audit=r['prompt_audit'],stages=r['stages']) for r in rows]))
            print('SAMMED_PG',job['name'],len(rows),'/',len(expected),task['case_id'],task['class_name'],
                  'dice',row['dice'],'seconds',row['seconds'],flush=True)
            del raw,pred,values
    base.validate_rows(rows,expected)
    base.write(run/'complete_metrics.json',dict(per_case_class=rows,epoch=230,weight_sha256=WEIGHT_SHA))


def merge(out):
    verify(out);rows=[]
    for j in base.read(out/'jobs.json'):
        run=out/'jobs'/j['name'];a=base.read(run/'model_audit.json')
        if not a['strict_effective_modules_exact'] or a['epoch']!=230:raise ValueError('model audit mismatch')
        part=base.read(run/'complete_metrics.json')['per_case_class']
        base.validate_rows(part,base.read(Path(j['prepared'])/'manifest.json')['tasks'])
        rows+=part
    base.validate_rows(rows,base.read(out/'manifest.json')['tasks']);validate_scope(rows,base.read(ROOT/base.SPLIT))
    for r in rows:
        if r['weight_sha256']!=WEIGHT_SHA or len(r['prompt_audit']['bg_points_dhw'])!=4:raise ValueError('result identity mismatch')
        for s in r['stages']:
            if s['background_points_encoded']!=4 or s['gt_used_for_roi'] or s['bg_roundtrip_voxels']>1e-4:
                raise ValueError('prompt/ROI audit mismatch')
    payload=dict(tasks=len(rows),cases=100,epoch=230,weight=WEIGHT,protocol=base.read(out/'protocol.json'),
                 mean={m:sum(r[m] for r in rows)/len(rows) for m in base.METRICS},per_case_class=rows)
    base.write(out/'merged_metrics.json',payload);print('MERGED1485',json.dumps(payload['mean']),flush=True)


def launch(out,gpus):
    import fcntl
    import subprocess
    verify(out)
    lock=(out/'sammed_schedule.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    first=base.read(out/'jobs.json')[0]
    if not (out/'jobs'/first['name']/'complete_metrics.json').exists():
        q=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
        available=[(int(x[0]),x[1].strip()) for x in (l.split(',') for l in q.splitlines()) if int(x[0]) in gpus and int(x[2])>=45000]
        if not available:raise ValueError('smoke requires45GB free')
        gpu,uuid=available[0];first.update(gpu=gpu,gpu_uuid=uuid,run=str(out/'jobs'/first['name']))
        spec=out/'job_specs'/first['name'];base.write(spec,first)
        with (out/'smoke.log').open('a') as log:
            subprocess.run([sys.executable,__file__,'worker','--job',str(spec)],cwd=ROOT,
                stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),check=True)
        print('WHOLE_CASE_SMOKE_PASSED; releasing full queue',flush=True)
    base.__file__=__file__;base.merge=merge;base.launch(out,gpus)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','worker','launch','merge','status'])
    p.add_argument('--out',default=OUT);p.add_argument('--job');p.add_argument('--gpus',default='0,1,2,3,4,5,6,7')
    a=p.parse_args();os.chdir(ROOT);out=(ROOT/a.out).resolve()
    if a.mode=='prepare':prepare(out)
    elif a.mode=='worker':worker(base.read(a.job))
    elif a.mode=='launch':launch(out,list(map(int,a.gpus.split(','))))
    elif a.mode=='merge':merge(out)
    else:
        complete=0;count=0
        for j in base.read(out/'jobs.json'):
            run=out/'jobs'/j['name'];f=run/'complete_metrics.json'
            if f.exists():complete+=1
            else:f=run/'all_metrics_live.json'
            if f.exists():count+=len(base.read(f)['per_case_class'])
        print(json.dumps(dict(completed_jobs=complete,tasks=count,total=1485,
            scheduler=base.read(out/'status.json') if (out/'status.json').exists() else 'smoke/prelaunch'),indent=2))


if __name__=='__main__':main()
