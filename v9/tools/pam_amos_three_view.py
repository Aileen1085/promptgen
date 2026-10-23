"""PAM full AMOS: independent coronal/sagittal boxes and frozen OR fusion.

Official model source is isolated. Missing tutorial Box2Mask helper is implemented
explicitly; no GT mask, PromptGen, semantic encoder, CT writes or mask exports.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import native_ct13_eval as n
REV='5f9f6005ea9f012e6bd93bc29a5ab5cf32cd66b3'
ASSETS=ROOT/'third_party/PAM_official_20261010'
SOURCE=ASSETS/('source_'+REV[:12])
DEFAULT='output/pam_amos_three_view_full1485_20261023'
METHODS=('coronal','sagittal','dual_or')
ORDERS={'coronal':(2,0,1),'sagittal':(1,0,2)}
PROTOCOL=dict(upstream_revision=REV,methods=METHODS,input_size=224,
    box='selected_cached_slice_2d_GT_bbox_exclusive',box_window='box_CT_percentile_0.5_99.5',
    box_crop='exact_rectangle_resize224_no_GT_mask',prop_crop_scale=1.5,
    neighbor_mm=20,neighbor_cap=40,lower_frontier_min_pixels=100,upper_frontier_min_pixels=0,
    prop_vote='any_positive_then_3D_initial_mask_component',prop_guiding='mean_vote_gt_0.5',
    propagation_window='initial_Box2Mask_prediction_CT_percentiles',normalize='full_slice_before_crop',fusion='binary_OR_frozen',
    prompts_submitted_once=True,internal_propagations=2,dual_extra_calls=0,
    query_batch=4,precision='float32',metrics=list(n.METRICS),no_CT_writes=True)

def orient(array,spacing,plane):
    order=ORDERS[plane]
    return array.transpose(order),tuple(np.asarray(spacing)[list(order)])

def restore(array,plane):
    return array.transpose(tuple(np.argsort(ORDERS[plane])))

def slice_box(gt,plane,index):
    frame=orient(gt,(1,1,1),plane)[0][index]
    y,x=np.where(frame)
    if not len(x):raise ValueError('cached guiding slice has empty target')
    return int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)

def box_crop(image,box):
    x1,y1,x2,y2=map(int,box)
    if not (0<=x1<x2<=image.shape[1] and 0<=y1<y2<=image.shape[0]):
        raise ValueError('box outside selected image')
    return image[y1:y2,x1:x2],(x1,y1,x2,y2)

def paste(mask,shape,bounds):
    x1,y1,x2,y2=bounds
    if mask.shape!=(y2-y1,x2-x1):raise ValueError('paste shape mismatch')
    out=np.zeros(shape,bool);out[y1:y2,x1:x2]=mask
    return out

def strip_state(state):
    out={}
    for name,value in state.items():
        name=name.removeprefix('module.')
        if name in out:raise ValueError('checkpoint prefix collision')
        out[name]=value
    return out

def normalize_u8(image):
    return ((image-image.min())/(image.max()-image.min()+1e-8)*255).astype(np.uint8)

def guide_binary(votes,count):
    return votes/(count+1e-8)>.5

def validate_scope(tasks,split,expected_tasks=1485,expected_cases=100):
    ds=split['datasets']['amos'];val={r['case_id'] for r in ds['val']};train={r['case_id'] for r in ds['train']}
    if len(tasks)!=expected_tasks or len({t['case_id'] for t in tasks})!=expected_cases:
        raise ValueError('not complete requested AMOS validation scope')
    if len({n.key(t) for t in tasks})!=len(tasks):raise ValueError('duplicate task')
    if any(t['source_name']!='amos' or t['case_id'] not in val or t['case_id'] in train for t in tasks):
        raise ValueError('validation membership/training overlap')

def finish_votes(votes,counts,initial):
    from scipy.ndimage import label
    mask=(votes/(counts[:,None,None]+1e-8))>0
    labels,_=label(mask)
    keep=np.unique(labels[initial]);keep=keep[keep!=0]
    return np.isin(labels,keep)

def predict_three(ct,spacing,payload,predictor):
    # Both boxes must exist before the first call; predictor never receives GT.
    if tuple(p['plane'] for p in payload)!=('coronal','sagittal'):
        raise ValueError('two immutable orthogonal prompts required')
    rows={};audits=[]
    for p in payload:
        stack,sp=orient(ct,spacing,p['plane'])
        pred,audit=predictor(stack,sp,p)
        rows[p['plane']]=restore(pred,p['plane']);audits.append(audit)
    rows['dual_or']=rows['coronal']|rows['sagittal']
    return rows,audits

class Models:
    def __init__(self):
        import torch
        # Force official fork ahead of installed dynamic_network_architectures.
        sys.path.insert(0,str(SOURCE))
        spec=importlib.util.spec_from_file_location('_isolated_official_PAM',SOURCE/'model/PAM.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        args=SimpleNamespace(conv_dim=2,input_channels=1,n_stages=6,max_channels=512,
            num_classes=2,deep_supervision=True,n_attn_stage=4,from_scratch_ratio=0.)
        self.box=module.Box2SegNet(args)
        self.prop=module.PAM(conv_dim=2,input_channels=1,n_stages=6,max_channels=512,
            num_classes=2,deep_supervision=True,n_attn_stage=4,from_scratch_ratio=0.)
        self.audit={}
        for name,model,file in [('box',self.box,'box2mask.pth'),('prop',self.prop,'propmask.pth')]:
            path=ASSETS/'checkpoints'/file
            checkpoint=torch.load(path,map_location='cpu',weights_only=True)
            state=strip_state(checkpoint['model_state_dict'])
            model.load_state_dict(state,strict=True)
            self.audit[name]=dict(sha256=n.sha(path),tensor_keys=len(state),epoch=checkpoint['epoch'],strict=True)
            model.eval().cuda()
        self.torch=torch

    def tensor(self,image):
        import cv2
        # Receive already full-slice normalized uint8, just like official A.Resize.
        value=cv2.resize(image,(224,224),interpolation=cv2.INTER_LINEAR)
        return self.torch.from_numpy(value.astype(np.float32)/127.5-1)[None].cuda()

    def output(self,result,shape):
        import cv2
        torch=self.torch
        predictions=result['predictions']
        if not isinstance(predictions,(tuple,list)):predictions=[predictions]
        size=predictions[0].shape[-2:]
        value=torch.stack([torch.nn.functional.interpolate(p,size=size,mode='bilinear',align_corners=True)
                          for p in predictions]).mean(0)
        masks=value.argmax(1).cpu().numpy().astype(np.uint8)
        return [cv2.resize(m,tuple(shape[::-1]),interpolation=cv2.INTER_NEAREST).astype(bool) for m in masks]

    def __call__(self,stack,spacing,prompt):
        import cv2
        torch=self.torch;z=int(prompt['slice']);box=prompt['box']
        crop,bounds=box_crop(stack[z],box)
        lo,hi=np.percentile(crop,(.5,99.5));center=int((lo+hi)/2);width=max(int(hi-lo),1)
        def window(image):return np.clip(image,center-width//2,center+width//2)
        with torch.inference_mode():
            result=self.box({'query_xs':self.tensor(normalize_u8(window(stack[z]))[bounds[1]:bounds[3],bounds[0]:bounds[2]])[None,None]})
            guiding=paste(self.output(result,crop.shape)[0],stack.shape[1:],bounds)
            initial=np.zeros(stack.shape,bool);initial[z]=guiding
            votes=initial.astype(np.int16);counts=np.zeros(len(stack),np.int16);counts[z]=1
            calls=0;guides=[];seen=set()
            if guiding.any():
                lo,hi=np.percentile(stack[z][guiding],(.5,99.5));center=int((lo+hi)/2);width=max(int(hi-lo),1)
            radius=min(int(20/float(spacing[0])),40)
            if radius<1:raise ValueError('native propagation physical neighborhood is empty')
            queue=[(z,votes[z]/(counts[z]+1e-8))]
            while queue:
                guide,mask=queue.pop(0)
                if guide in seen:continue
                seen.add(guide)
                if not mask.any():continue
                ys,xs=np.where(mask);cy=int((ys.min()+ys.max())/2);cx=int((xs.min()+xs.max())/2)
                side=max(int(1.5*max(int(ys.max()-ys.min()),int(xs.max()-xs.min()))),16)
                x1=max(cx-side//2,0);y1=max(cy-side//2,0)
                x2=min(x1+side,stack.shape[2]);y2=min(y1+side,stack.shape[1])
                def pad(image):return np.pad(image[y1:y2,x1:x2],((0,side-(y2-y1)),(0,side-(x2-x1))))
                gx=self.tensor(pad(normalize_u8(window(stack[guide]))))
                gm=cv2.resize(pad(mask).astype(np.float32),(224,224),interpolation=cv2.INTER_NEAREST)
                gm=torch.from_numpy((gm>.5).astype(np.float32)).cuda()[None]
                neighbors=[i for i in range(max(guide-radius,0),min(guide+radius+1,len(stack))) if i!=guide]
                for off in range(0,len(neighbors),4):
                    ids=neighbors[off:off+4]
                    q=torch.stack([self.tensor(pad(normalize_u8(window(stack[i])))) for i in ids])[None]
                    result=self.prop({'guiding_x':gx[None], 'guiding_prp':gm,'adjacent_x':q})
                    predicted=self.output(result,(side,side));calls+=1
                    for i,pred in zip(ids,predicted):
                        votes[i,y1:y2,x1:x2]+=pred[:y2-y1,:x2-x1];counts[i]+=1
                guides.append(guide)
                covered=np.flatnonzero(counts)
                # Freeze both frontier guides before propagating either one.
                if len(covered) and not queue:
                    lower,upper=int(covered.min()),int(covered.max())
                    if lower>0 and votes[lower].sum()>100 and lower not in seen:queue.append((lower,votes[lower]/(counts[lower]+1e-8)))
                    if upper<len(stack)-1 and votes[upper].sum()>0 and upper not in seen:queue.append((upper,votes[upper]/(counts[upper]+1e-8)))
            pred=finish_votes(votes,counts,initial)
        return pred,dict(plane=prompt['plane'],slice=z,box=list(box),box_decoder_calls=1,
            prop_decoder_calls=calls,guiding_pixels=int(guiding.sum()),guide_slices=guides,
            covered_slices=int(np.count_nonzero(counts)),stack_spacing_mm=list(spacing),
            native_neighbor_radius=radius,gt_mask_supplied=False)

class Loader:
    def __init__(self):self.current=None
    def __call__(self,task):
        base=n.base_module();package=n.CACHE/'metadata/shared_adaptive_prompt_v1/amos'/task['prompt_cache_name']
        with np.load(package,allow_pickle=False) as p:
            shape=tuple(map(int,p['shape_dhw']));name=str(p['case_image_name'].item())
            supports=[np.unpackbits(p[k].astype(np.uint8),count=int(np.prod(shape)),bitorder='little').reshape(shape).astype(bool)
                      for k in ('coronal_bits','sagittal_bits')]
        if self.current!=task['case_id']:
            image,affine=base.canonical(task['image']);label,la=base.canonical(task['label'])
            canonical_shape=tuple(image.shape[i] for i in (2,0,1))
            if canonical_shape!=shape or tuple(label.shape[i] for i in (2,0,1))!=shape:raise ValueError('CT/GT/cache size mismatch')
            error=n.assert_compatible_grid(affine,la,shape,image_qform=image.get_qform(),label_qform=label.get_qform())
            path=n.CACHE/'_case_images_v1'/name
            self.ct=np.load(path,mmap_mode='r') if path.is_file() else np.asarray(image.dataobj,dtype=np.float32).transpose(2,0,1)
            self.label=np.asarray(label.dataobj).transpose(2,0,1);self.spacing=np.linalg.norm(affine[:3,:3],axis=0)
            self.grid_error=error;self.current=task['case_id'];self.hit=path.is_file()
        gt=n.select_target(self.label,task)
        if self.ct.shape!=gt.shape or not gt.any():raise ValueError('empty/misaligned task')
        if not all(s.any() and gt[s].all() for s in supports):raise ValueError('cached foreground alignment')
        slices=n.canonical_plane_slices(supports)
        payload=tuple(dict(plane=plane,slice=int(index),box=slice_box(gt,plane,index)) for plane,index in zip(('coronal','sagittal'),slices))
        audit=dict(prompt_cache_sha256=n.sha(package),case_cache_hit=self.hit,case_image_name=name,
            shape_dhw=list(gt.shape),spacing_dhw_mm=self.spacing.tolist(),grid_error_mm=self.grid_error,
            boxes=payload,gt_used_only_for_selected_2D_box=True,three_D_GT_ROI=False,
            submission_sha256=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest())
        return self.ct,gt,self.spacing,payload,audit

def assets_signature():
    paths=[Path(__file__),ROOT/'tools/native_ct13_eval.py',ROOT/'tools/native_sam2_physical_eval.py',ROOT/'utils/metrics.py']
    paths+=sorted(SOURCE.rglob('*.py'))
    if len(list(SOURCE.rglob('*.py')))!=22:raise ValueError('incomplete official Python sources')
    return dict(files={str(p.relative_to(ROOT)):n.sha(p) for p in paths},
        weights={f:n.sha(ASSETS/'checkpoints'/f) for f in ('box2mask.pth','propmask.pth')},protocol=PROTOCOL)

def signature(job):
    return hashlib.sha256(json.dumps(dict(assets=assets_signature(),tasks=job['tasks'],
        prompts=[n.sha(n.CACHE/'metadata/shared_adaptive_prompt_v1/amos'/t['prompt_cache_name']) for t in job['tasks']]),sort_keys=True).encode()).hexdigest()

def prepare(out):
    manifest=n.read(ROOT/'output/native_sam2_medsam2_ct13_full6094_20261007_gridfix/manifest.json')
    tasks=sorted([r for r in manifest['tasks'] if r['source_name']=='amos'],key=n.key)
    validate_scope(tasks,n.read(ROOT/'configs/ct13_v10_2_split_20260928.json'))
    if (out/'jobs.json').exists():raise ValueError('already prepared; do not overwrite')
    assets=assets_signature();n.write(out/'protocol.json',assets)
    n.write(out/'manifest.json',dict(tasks=tasks,expected_per_method=1485,CT=100,train_overlap=0))
    # One whole first case is a genuine all-method smoke and is reused, not rerun.
    first=tasks[0]['case_id'];smoke=[r for r in tasks if r['case_id']==first]
    parts=[smoke]+n.case_parts([r for r in tasks if r['case_id']!=first],24)
    metas=[]
    for i,part in enumerate(parts):
        job=dict(name='amos_{:03d}'.format(i),tasks=part,output=str(out/'jobs'/'amos_{:03d}'.format(i)))
        job['signature']=signature(job);path=out/'job_specs'/(job['name']+'.json');n.write(path,job)
        metas.append(dict(name=job['name'],path=str(path),tasks=len(part),smoke=(i==0)))
    n.write(out/'jobs.json',dict(jobs=metas,total_case_class=1485,total_method_rows=4455))
    print('PREPARED',len(metas),'whole-case shards',flush=True)

def worker(jobpath,uuid):
    import torch
    torch.set_num_threads(2)
    if str(torch.cuda.get_device_properties(0).uuid).replace('GPU-','')!=uuid.replace('GPU-',''):raise ValueError('CUDA/physical UUID mismatch')
    job=n.read(jobpath);out=Path(job['output']);sig=signature(job)
    if sig!=job['signature']:raise ValueError('changed-code/assets/prompt signature')
    rows=n.read(out/'all_metrics_live.json')['per_case_class'] if (out/'all_metrics_live.json').exists() else []
    if any(r.get('signature')!=sig for r in rows):raise ValueError('partial result signature mismatch')
    for method in METHODS:n.validate_rows([r for r in rows if r['method']==method],[t for t in job['tasks'] if n.key(t) in {n.key(r) for r in rows}])
    done={n.key(r) for r in rows}
    models=Models();loader=Loader();scorer=n.base_module().metric_module()
    n.write(out/'restoration_audit.json',models.audit)
    for t in job['tasks']:
        if n.key(t) in done:continue
        start=time.time();ct,gt,spacing,payload,audit=loader(t)
        predictions,pa=predict_three(ct,spacing,payload,models)
        for method,pred in predictions.items():
            metrics=scorer.binary_prompt_metrics_from_masks(torch.from_numpy(np.ascontiguousarray(pred)),torch.from_numpy(gt),tuple(spacing))
            rows.append(dict(t,method=method,**{m:float(metrics[m]) for m in n.METRICS},prompt_audit=audit,
                propagation_audit=pa,seconds=time.time()-start,gpu_uuid=uuid,signature=sig))
        n.write(out/'all_metrics_live.json',dict(per_case_class=rows,signature=sig))
        print('DONE',job['name'],len(rows)//3,'/',len(job['tasks']),t['case_id'],t['class_name'],
            'Dice',[round(r['dice'],5) for r in rows[-3:]],'seconds',round(time.time()-start,1),flush=True)
    for method in METHODS:n.validate_rows([r for r in rows if r['method']==method],job['tasks'])
    n.write(out/'complete_metrics.json',dict(per_case_class=rows,signature=sig))

def progress(out):
    counts=Counter();complete=0
    for m in n.read(out/'jobs.json')['jobs']:
        folder=out/'jobs'/m['name'];p=folder/'complete_metrics.json'
        if p.exists():complete+=1
        else:p=folder/'all_metrics_live.json'
        if p.exists():counts.update(r['method'] for r in n.read(p)['per_case_class'])
    return dict(completed_jobs=complete,total_jobs=25,method_rows=dict(counts),expected_each=1485)

def merge(out):
    rows=[]
    for meta in n.read(out/'jobs.json')['jobs']:
        job=n.read(meta['path']);d=n.read(Path(job['output'])/'complete_metrics.json')
        if d['signature']!=signature(job):raise ValueError('merge signature mismatch')
        rows+=d['per_case_class']
    tasks=n.read(out/'manifest.json')['tasks']
    validate_scope(tasks,n.read(ROOT/'configs/ct13_v10_2_split_20260928.json'))
    for method in METHODS:n.validate_rows([r for r in rows if r['method']==method],tasks)
    n.write(out/'merged_metrics.json',dict(per_case_class=rows,protocol=PROTOCOL,
        summary={method:{m:float(np.mean([r[m] for r in rows if r['method']==method])) for m in n.METRICS} for method in METHODS}))

def schedule(out):
    import fcntl
    lock=(out/'scheduler.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if assets_signature()!=n.read(out/'protocol.json'):raise ValueError('frozen assets changed')
    metas=n.read(out/'jobs.json')['jobs'];active={}
    while True:
        done={m['name'] for m in metas if (out/'jobs'/m['name']/'complete_metrics.json').exists()}
        for name,item in list(active.items()):
            rc=item['p'].poll()
            if rc is None:continue
            item['log'].close();del active[name]
            if rc:
                n.write(out/'failure.json',dict(job=name,returncode=rc,log=str(item['path'])))
                for v in active.values():v['p'].wait();v['log'].close()
                raise RuntimeError('PAM worker failed; preserve evidence: '+name)
        pending=[m for m in metas if m['name'] not in done and m['name'] not in active]
        smoke_done=metas[0]['name'] in done
        if not smoke_done:pending=[m for m in pending if m['smoke']]
        if len(done)==len(metas) and not active:
            merge(out);n.write(out/'status.json',dict(state='complete',**progress(out)));return
        inv=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
        busy={v['uuid'] for v in active.values()}
        for line in inv.strip().splitlines():
            index,uuid,free=[s.strip() for s in line.split(',')]
            if int(index) not in (0,2,3,4,5,6) or uuid in busy or int(free)<24000 or not pending:continue
            meta=pending.pop(0);path=out/'logs'/(meta['name']+'.log');path.parent.mkdir(exist_ok=True)
            if path.exists():path=path.with_name(path.stem+'.'+str(int(time.time()))+'.log')
            log=path.open('w');env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='2')
            proc=subprocess.Popen([sys.executable,str(Path(__file__)),'worker','--job',meta['path'],'--uuid',uuid],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            active[meta['name']]=dict(p=proc,log=log,path=path,uuid=uuid)
            print('START',meta['name'],'physicalGPU',index,uuid,'PID',proc.pid,'smoke',meta['smoke'],flush=True)
        n.write(out/'status.json',dict(state='full' if smoke_done else 'smoke',time=time.strftime('%Y-%m-%d %H:%M:%S'),
            scheduler_pid=os.getpid(),active={k:dict(pid=v['p'].pid,uuid=v['uuid']) for k,v in active.items()},**progress(out)))
        time.sleep(20)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','worker','launch','status','merge','assets'))
    p.add_argument('--output',default=DEFAULT);p.add_argument('--job');p.add_argument('--uuid');a=p.parse_args();out=ROOT/a.output
    if a.action=='prepare':prepare(out)
    elif a.action=='worker':worker(a.job,a.uuid)
    elif a.action=='launch':schedule(out)
    elif a.action=='status':print(json.dumps(progress(out),indent=2))
    elif a.action=='merge':merge(out)
    elif a.action=='assets':print(json.dumps(Models().audit,indent=2))

if __name__=='__main__':main()
