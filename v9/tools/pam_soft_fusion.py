"""Independent PAM AMOS119 soft-fusion pilot; no volume exports or extra forwards."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import pam_amos_three_view as p
n=p.n
DEFAULT='output/pam_amos_soft_fusion_pilot119_20261023'
METHODS=('coronal','sagittal','dual_or','coronal_prob','sagittal_prob',
         'coronal_margin','sagittal_margin','dual_prob','dual_margin')
PROTOCOL=dict(p.PROTOCOL,methods=list(METHODS),scope='first8_sorted_AMOS_val_CT_119_tasks',
    fusion='per_voxel_query_mean_then_equal_valid_view_mean',
    probabilities='softmax_before_bilinear_native_crop_restore',
    margins='foreground_minus_background_before_bilinear_native_crop_restore',
    coverage='actual_native_query_crop_per_voxel; one_view_use_it; neither_background',
    thresholds={'probability':.5,'margin':0.},ties='background',
    soft_components='touch_learned_initial_mask_union_for_dual',
    propagation='unchanged_hard_masks_frontier_and_guiding',no_soft_exports=True)


class SoftVolume:
    def __init__(self,shape):
        self.prob=np.zeros(shape,np.float32);self.margin=np.zeros(shape,np.float32)
        self.count=np.zeros(shape,np.uint16)

    def add(self,z,bounds,prob,margin):
        x1,y1,x2,y2=bounds;shape=(y2-y1,x2-x1)
        if prob.shape!=shape or margin.shape!=shape:raise ValueError('soft crop shape')
        if not np.isfinite(prob).all() or not np.isfinite(margin).all() or np.any((prob<0)|(prob>1)):
            raise ValueError('nonfinite/invalid soft output')
        target=(z,slice(y1,y2),slice(x1,x2))
        if np.any(self.count[target]==65535):raise ValueError('coverage count overflow')
        self.prob[target]+=prob;self.margin[target]+=margin;self.count[target]+=1

    def means(self):
        valid=self.count>0
        np.divide(self.prob,self.count,out=self.prob,where=valid)
        np.divide(self.margin,self.count,out=self.margin,where=valid)
        return self.prob,self.margin,valid


def fuse(a,b):
    if any(x.shape!=a[0].shape for x in (*a,*b)):raise ValueError('canonical shape mismatch')
    count=a[2].astype(np.uint8)+b[2].astype(np.uint8);valid=count>0
    result=[]
    for i in (0,1):
        value=np.where(a[2],a[i],0)+np.where(b[2],b[i],0)
        np.divide(value,count,out=value,where=valid)
        result.append(value)
    return *result,valid


def seed_components(mask,seed):
    from scipy.ndimage import label
    labels,_=label(mask);keep=np.unique(labels[seed]);keep=keep[keep!=0]
    return np.isin(labels,keep)


def select_pilot(tasks):
    cases=sorted({t['case_id'] for t in tasks})[:8]
    return sorted([t for t in tasks if t['case_id'] in cases],key=n.key)


class SoftModels(p.Models):
    def output(self,result,shape):
        import cv2
        torch=self.torch;predictions=result['predictions']
        if not isinstance(predictions,(tuple,list)):predictions=[predictions]
        size=predictions[0].shape[-2:]
        value=torch.stack([torch.nn.functional.interpolate(v,size=size,mode='bilinear',align_corners=True)
                           for v in predictions]).mean(0)
        if value.shape[1]!=2 or not torch.isfinite(value).all():raise ValueError('invalid PAM logits')
        hard=value.argmax(1).cpu().numpy().astype(np.uint8)
        prob=value.softmax(1)[:,1].cpu().numpy()
        margin=(value[:,1]-value[:,0]).cpu().numpy()
        resize=lambda x,mode:cv2.resize(x,tuple(shape[::-1]),interpolation=mode)
        return [(resize(h,cv2.INTER_NEAREST).astype(bool),resize(q,cv2.INTER_LINEAR),resize(l,cv2.INTER_LINEAR))
                for h,q,l in zip(hard,prob,margin)]

    def __call__(self,stack,spacing,prompt):
        # Native hard-guided propagation is unchanged. Soft observations never feed back.
        import cv2
        torch=self.torch;z=int(prompt['slice']);box=prompt['box']
        crop,bounds=p.box_crop(stack[z],box);soft=SoftVolume(stack.shape)
        lo,hi=np.percentile(crop,(.5,99.5));center=int((lo+hi)/2);width=max(int(hi-lo),1)
        def window(image):return np.clip(image,center-width//2,center+width//2)
        with torch.inference_mode():
            result=self.box({'query_xs':self.tensor(p.normalize_u8(window(stack[z]))[bounds[1]:bounds[3],bounds[0]:bounds[2]])[None,None]})
            hard,q,l=self.output(result,crop.shape)[0];soft.add(z,bounds,q,l)
            guiding=p.paste(hard,stack.shape[1:],bounds)
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
                gx=self.tensor(pad(p.normalize_u8(window(stack[guide]))))
                gm=cv2.resize(pad(mask).astype(np.float32),(224,224),interpolation=cv2.INTER_NEAREST)
                gm=torch.from_numpy((gm>.5).astype(np.float32)).cuda()[None]
                neighbors=[i for i in range(max(guide-radius,0),min(guide+radius+1,len(stack))) if i!=guide]
                for off in range(0,len(neighbors),4):
                    ids=neighbors[off:off+4]
                    query=torch.stack([self.tensor(pad(p.normalize_u8(window(stack[i])))) for i in ids])[None]
                    result=self.prop({'guiding_x':gx[None],'guiding_prp':gm,'adjacent_x':query})
                    predicted=self.output(result,(side,side));calls+=1
                    for i,(pred,q,l) in zip(ids,predicted):
                        votes[i,y1:y2,x1:x2]+=pred[:y2-y1,:x2-x1];counts[i]+=1
                        soft.add(i,(x1,y1,x2,y2),q[:y2-y1,:x2-x1],l[:y2-y1,:x2-x1])
                guides.append(guide);covered=np.flatnonzero(counts)
                if len(covered) and not queue:
                    lower,upper=int(covered.min()),int(covered.max())
                    if lower>0 and votes[lower].sum()>100 and lower not in seen:queue.append((lower,votes[lower]/(counts[lower]+1e-8)))
                    if upper<len(stack)-1 and votes[upper].sum()>0 and upper not in seen:queue.append((upper,votes[upper]/(counts[upper]+1e-8)))
            hard=p.finish_votes(votes,counts,initial)
        audit=dict(plane=prompt['plane'],slice=z,box=list(box),box_decoder_calls=1,
            prop_decoder_calls=calls,guiding_pixels=int(guiding.sum()),guide_slices=guides,
            covered_slices=int(np.count_nonzero(counts)),stack_spacing_mm=list(spacing),
            native_neighbor_radius=radius,gt_mask_supplied=False,
            soft_observed_voxels=int(np.count_nonzero(soft.count)),soft_max_observations=int(soft.count.max()))
        return hard,soft.means(),initial,audit


def predict(ct,spacing,payload,models):
    if tuple(v['plane'] for v in payload)!=('coronal','sagittal'):raise ValueError('both boxes required before inference')
    rows={};fields=[];seeds=[];audits=[]
    for prompt in payload:
        plane=prompt['plane'];stack,sp=p.orient(ct,spacing,plane)
        hard,soft,initial,audit=models(stack,sp,prompt)
        fields.append(tuple(p.restore(v,plane) for v in soft));seeds.append(p.restore(initial,plane))
        rows[plane]=p.restore(hard,plane);audits.append(audit)
        for j,name,cutoff in ((0,'prob',.5),(1,'margin',0.)):
            rows[plane+'_'+name]=seed_components((fields[-1][j]>cutoff)&fields[-1][2],seeds[-1])
    rows['dual_or']=rows['coronal']|rows['sagittal']
    q,l,valid=fuse(*fields);seed=seeds[0]|seeds[1]
    rows['dual_prob']=seed_components((q>.5)&valid,seed)
    rows['dual_margin']=seed_components((l>0)&valid,seed)
    return rows,audits


def assets_signature():
    data=p.assets_signature();data['files']['tools/pam_soft_fusion.py']=n.sha(Path(__file__))
    data['protocol']=PROTOCOL
    return data


def signature(job):
    data=dict(assets=assets_signature(),tasks=job['tasks'],
        prompts=[n.sha(n.CACHE/'metadata/shared_adaptive_prompt_v1/amos'/t['prompt_cache_name']) for t in job['tasks']])
    return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()


def prepare(out):
    original=ROOT/p.DEFAULT;tasks=select_pilot(n.read(original/'manifest.json')['tasks'])
    p.validate_scope(tasks,n.read(ROOT/'configs/ct13_v10_2_split_20260928.json'),119,8)
    if (out/'jobs.json').exists():raise ValueError('already prepared')
    n.write(out/'protocol.json',assets_signature())
    n.write(out/'manifest.json',dict(tasks=tasks,expected_per_method=119,CT=8,train_overlap=0))
    n.write(out/'control_source.json',dict(path=str(original/'merged_metrics.json'),sha256=n.sha(original/'merged_metrics.json')))
    metas=[]
    for i,case in enumerate(sorted({t['case_id'] for t in tasks})):
        part=[t for t in tasks if t['case_id']==case];name='amos_{:03d}'.format(i)
        job=dict(name=name,tasks=part,output=str(out/'jobs'/name));job['signature']=signature(job)
        path=out/'job_specs'/(name+'.json');n.write(path,job)
        metas.append(dict(name=name,path=str(path),tasks=len(part),smoke=(i==0)))
    n.write(out/'jobs.json',dict(jobs=metas,total_case_class=119,total_method_rows=119*len(METHODS)))
    print('PREPARED 8 cases /119 tasks /9 methods',flush=True)


def worker(jobpath,uuid):
    import torch
    torch.set_num_threads(2)
    if str(torch.cuda.get_device_properties(0).uuid).replace('GPU-','')!=uuid.replace('GPU-',''):raise ValueError('CUDA/physical UUID mismatch')
    job=n.read(jobpath);out=Path(job['output']);sig=signature(job)
    if sig!=job['signature']:raise ValueError('changed code/assets/prompts')
    rows=n.read(out/'all_metrics_live.json')['per_case_class'] if (out/'all_metrics_live.json').exists() else []
    if any(r.get('signature')!=sig for r in rows):raise ValueError('partial signature mismatch')
    done={n.key(r) for r in rows}
    for method in METHODS:n.validate_rows([r for r in rows if r['method']==method],[t for t in job['tasks'] if n.key(t) in done])
    ref=n.read(out.parents[1]/'control_source.json')
    if n.sha(Path(ref['path']))!=ref['sha256']:raise ValueError('original control results changed')
    old={(r['method'],n.key(r)):r for r in n.read(ref['path'])['per_case_class']}
    models=SoftModels();loader=p.Loader();scorer=n.base_module().metric_module()
    n.write(out/'restoration_audit.json',models.audit)
    for task in job['tasks']:
        if n.key(task) in done:continue
        start=time.time();ct,gt,spacing,payload,audit=loader(task)
        predictions,pa=predict(ct,spacing,payload,models);new=[]
        for method in METHODS:
            metrics=scorer.binary_prompt_metrics_from_masks(torch.from_numpy(np.ascontiguousarray(predictions[method])),torch.from_numpy(gt),tuple(spacing))
            values={m:float(metrics[m]) for m in n.METRICS}
            if not all(np.isfinite(v) for v in values.values()):raise ValueError('nonfinite metrics')
            error=0.
            if method in p.METHODS:
                control=old[(method,n.key(task))];error=max(abs(values[m]-control[m]) for m in n.METRICS)
                if error>1e-6:raise ValueError('native hard-control mismatch: '+str(error))
                for before,after in zip(control['propagation_audit'],pa):
                    for k in ('guide_slices','box_decoder_calls','prop_decoder_calls','guiding_pixels'):
                        if before[k]!=after[k]:raise ValueError('native propagation changed '+k)
            new.append(dict(task,method=method,**values,prompt_audit=audit,propagation_audit=pa,
                native_control_max_error=error,fusion_extra_calls=0,seconds=time.time()-start,gpu_uuid=uuid,signature=sig))
        rows+=new;n.write(out/'all_metrics_live.json',dict(per_case_class=rows,signature=sig))
        print('DONE',job['name'],len(rows)//len(METHODS),'/',len(job['tasks']),task['case_id'],task['class_name'],
              'dual_prob/margin',[round(new[METHODS.index(m)]['dice'],5) for m in ('dual_prob','dual_margin')],flush=True)
    for method in METHODS:n.validate_rows([r for r in rows if r['method']==method],job['tasks'])
    n.write(out/'complete_metrics.json',dict(per_case_class=rows,signature=sig))


def progress(out):
    counts=Counter();metas=n.read(out/'jobs.json')['jobs'];complete=0
    for meta in metas:
        folder=out/'jobs'/meta['name'];file=folder/'complete_metrics.json'
        if file.exists():complete+=1
        else:file=folder/'all_metrics_live.json'
        if file.exists():counts.update(r['method'] for r in n.read(file)['per_case_class'])
    return dict(completed_jobs=complete,total_jobs=len(metas),method_rows=dict(counts),expected_each=119)


def merge(out):
    import math
    rows=[];tasks=n.read(out/'manifest.json')['tasks']
    p.validate_scope(tasks,n.read(ROOT/'configs/ct13_v10_2_split_20260928.json'),119,8)
    for meta in n.read(out/'jobs.json')['jobs']:
        job=n.read(meta['path']);data=n.read(Path(job['output'])/'complete_metrics.json')
        if data['signature']!=signature(job) or any(r['signature']!=data['signature'] for r in data['per_case_class']):raise ValueError('merge signature')
        rows+=data['per_case_class']
    for method in METHODS:n.validate_rows([r for r in rows if r['method']==method],tasks)
    n.write(out/'merged_metrics.json',dict(per_case_class=rows,protocol=PROTOCOL,
        summary={method:{m:math.fsum(r[m] for r in rows if r['method']==method)/len(tasks) for m in n.METRICS} for method in METHODS}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','worker','launch','status','merge'))
    parser.add_argument('--output',default=DEFAULT);parser.add_argument('--job');parser.add_argument('--uuid')
    args=parser.parse_args();out=ROOT/args.output
    # Process-local scheduler bindings; old signed entry and completed files untouched.
    p.assets_signature_original=p.assets_signature
    original_assets=p.assets_signature
    def signed_assets():
        data=original_assets()
        data['files']['tools/pam_amos_three_view.py']=n.sha(ROOT/'tools/pam_amos_three_view.py')
        data['files']['tools/pam_soft_fusion.py']=n.sha(Path(__file__));data['protocol']=PROTOCOL
        return data
    globals()['assets_signature']=signed_assets
    p.assets_signature=signed_assets;p.progress=progress;p.merge=merge;p.__file__=__file__
    if args.action=='prepare':prepare(out)
    elif args.action=='worker':worker(args.job,args.uuid)
    elif args.action=='launch':p.schedule(out)
    elif args.action=='status':print(json.dumps(progress(out),indent=2))
    else:merge(out)


if __name__=='__main__':main()
