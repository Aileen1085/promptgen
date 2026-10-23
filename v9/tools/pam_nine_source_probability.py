"""PAM dual-box probability fusion on nine complete held-out CT sources."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import pam_soft_fusion as soft
p=soft.p;n=soft.n
_pam_assets=p.assets_signature
DEFAULT='output/pam_probability_nine_source_full4548_20261023'
COUNTS=dict(totalseg=4034,magic=35,msd_task07=40,msd_task10=20,msd_task08=40,
            parse2022=20,topcow2024_cta=277,kits23=58,lnq2023_lite=24)
MANIFEST=ROOT/'output/native_sam2_medsam2_ct13_full6094_20261007_gridfix/manifest.json'
SPLIT=ROOT/'configs/ct13_v10_2_split_20260928.json'
PROTOCOL=dict(soft.PROTOCOL,methods=['dual_prob'],scope='nine_sources_complete_val_4548_train0',
    tensor_order='canonical_DHW_ZXY',background='none_box_only',
    magic_transform='existing_raw_NRRD_to_canonical_NIfTI_physical_mapping_once',
    thresholds={'probability':.5},margins='not_used_for_fusion_or_scoring',
    semantic=False,promptgen=False,large_cache_writes=False)


def validate_scope(tasks,split):
    if dict(Counter(t['source_name'] for t in tasks))!=COUNTS:raise ValueError('wrong nine-source scope')
    if len({n.key(t) for t in tasks})!=sum(COUNTS.values()):raise ValueError('duplicate/missing tasks')
    for source in COUNTS:
        d=split['datasets'][source];val={r['case_id'] for r in d['val']};train={r['case_id'] for r in d['train']}
        for t in tasks:
            if t['source_name']!=source:continue
            if t['case_id'] not in val or t['case_id'] in train:raise ValueError('train/val overlap')
            if source=='totalseg' and int(t['global_class_id']) in (40,41):raise ValueError('empty official class')


def make_jobs(tasks):
    smoke=[min((r for r in tasks if r['source_name']==s),key=n.key) for s in sorted(COUNTS)]
    smoke_keys={n.key(t) for t in smoke}
    jobs=[dict(name='smoke_nine_sources',tasks=smoke,smoke=True)]
    for source in sorted(COUNTS):
        rows=[t for t in tasks if t['source_name']==source and n.key(t) not in smoke_keys]
        if not rows:continue
        for i,part in enumerate(n.case_parts(rows,64 if source=='totalseg' else 4)):
            jobs.append(dict(name='{}_{:03d}'.format(source,i),tasks=part,smoke=False))
    return jobs


def probability_mean(a,b):
    if a[0].shape!=b[0].shape or a[1].shape!=a[0].shape or b[1].shape!=b[0].shape:raise ValueError('grid mismatch')
    count=a[1].astype(np.uint8)+b[1].astype(np.uint8);valid=count>0
    mean=np.where(a[1],a[0],0)+np.where(b[1],b[0],0)
    np.divide(mean,count,out=mean,where=valid)
    return mean,valid


def predict(ct,spacing,payload,model):
    if tuple(a['plane'] for a in payload)!=('coronal','sagittal'):raise ValueError('two boxes must be submitted first')
    fields=[];seeds=[];audits=[]
    for prompt in payload:
        plane=prompt['plane'];stack,sp=p.orient(ct,spacing,plane)
        hard,values,initial,audit=model(stack,sp,prompt)
        fields.append((p.restore(values[0],plane),p.restore(values[2],plane)))
        seeds.append(p.restore(initial,plane));audits.append(audit)
        del hard,values
    mean,valid=probability_mean(*fields)
    return soft.seed_components((mean>.5)&valid,seeds[0]|seeds[1]),audits


class Loader:
    """Readonly shared case CT; no background generation or CT cache writes."""
    def __init__(self,split):
        self.split=split;self.base=n.base_module();self.current=None;self.label_name=None

    def __call__(self,task):
        source=task['source_name'];entry=next(x for x in self.split['datasets'][source]['val'] if x['case_id']==task['case_id'])
        image_path=entry.get('image',entry.get('image_path'))
        label_path=n.target_path(dict(task,label=entry.get('label',entry.get('label_path'))))
        package_path=n.CACHE/'metadata/shared_adaptive_prompt_v1'/source/task['prompt_cache_name']
        with np.load(package_path,allow_pickle=False) as package:
            shape=tuple(int(v) for v in package['shape_dhw']);name=str(package['case_image_name'].item())
            supports=[np.unpackbits(package[k].astype(np.uint8),count=int(np.prod(shape)),bitorder='little').reshape(shape).astype(bool)
                      for k in ('coronal_bits','sagittal_bits')]
        current=(source,task['case_id'],str(image_path),name)
        if current!=self.current:
            image,self.affine=self.base.canonical(image_path);self.qform=image.get_qform();self.source_affine=self.affine.copy()
            target_shape=tuple(image.shape[i] for i in (2,0,1))
            if source=='magic':
                import SimpleITK as sitk
                reader=sitk.ImageFileReader();reader.SetFileName(task['image']);reader.ReadImageInformation()
                lps=np.eye(4);lps[:3,:3]=(np.asarray(reader.GetDirection()).reshape(3,3)@np.diag(reader.GetSpacing()))[:,[2,1,0]]
                lps[:3,3]=reader.GetOrigin();self.source_affine=np.diag([-1.,-1.,1.,1.])@lps
            cache=n.CACHE/'_case_images_v1'/name;self.hit=cache.is_file()
            if self.hit:
                raw=np.load(cache,mmap_mode='r')
                if raw.shape!=shape:raise ValueError('cache/source shape mismatch')
                self.ct=self.base.reorient_grid(raw,self.source_affine,self.affine,target_shape)
            else:self.ct=np.asarray(image.dataobj,dtype=np.float32).transpose(2,0,1)
            self.current=current;self.label_name=None
        if self.label_name!=str(label_path):
            label,affine=self.base.canonical(label_path)
            self.grid_error=n.assert_compatible_grid(self.affine,affine,self.ct.shape,image_qform=self.qform,label_qform=label.get_qform())
            self.label=np.asarray(label.dataobj).transpose(2,0,1);self.label_name=str(label_path)
        target=n.select_target(self.label,task)
        if target.shape!=self.ct.shape or not target.any():raise ValueError('empty/misaligned target')
        mapped=[self.base.reorient_grid(s,self.source_affine,self.affine,target.shape) for s in supports]
        foreground=mapped[0]|mapped[1]
        coords,error=self.base.map_voxels(np.argwhere(supports[0]|supports[1]),self.source_affine,self.affine)
        direct=np.zeros(target.shape,bool);direct[tuple(np.rint(coords).astype(int).T)]=True
        if error>=1e-3 or not np.array_equal(direct,foreground) or not foreground.any() or not target[foreground].all():
            raise ValueError('foreground physical alignment failed')
        slices=n.canonical_plane_slices(mapped)
        payload=tuple(dict(plane=plane,slice=int(index),box=p.slice_box(target,plane,index))
                      for plane,index in zip(('coronal','sagittal'),slices))
        spacing=np.linalg.norm(self.affine[:3,:3],axis=0)
        audit=dict(prompt_cache_sha256=n.sha(package_path),case_image_name=name,case_cache_hit=self.hit,
            source_shape_dhw=list(shape),shape_dhw=list(target.shape),spacing_dhw_mm=spacing.tolist(),
            source_affine_dhw_ras=self.source_affine.tolist(),target_affine_dhw_ras=self.affine.tolist(),
            physical_roundtrip_error_mm=float(error),grid_error_mm=self.grid_error,fg_in_gt_fraction=1.,
            canonical_prompt_slices=list(slices),boxes=payload,background='none',
            gt_used_only_for_selected_2D_box=True,three_D_GT_ROI=False,
            magic_mapping_applications=int(source=='magic'),
            submission_sha256=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest())
        return self.ct,target,spacing,payload,audit


def assets_signature():
    data=_pam_assets()
    paths=[ROOT/'tools/pam_amos_three_view.py',ROOT/'tools/pam_soft_fusion.py',Path(__file__),
           ROOT/'infer/native_sam2_physical_prompts.py',SPLIT,MANIFEST]
    data['files'].update({str(f.relative_to(ROOT)):n.sha(f) for f in paths});data['protocol']=PROTOCOL
    return data


def signature(job,assets=None):
    data=dict(assets=assets if assets is not None else assets_signature(),tasks=job['tasks'],
        prompts=[n.sha(n.CACHE/'metadata/shared_adaptive_prompt_v1'/t['source_name']/t['prompt_cache_name']) for t in job['tasks']])
    return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()


def validate_audit(row):
    a=row['prompt_audit'];pa=row['propagation_audit']
    if (a['fg_in_gt_fraction']!=1. or a['physical_roundtrip_error_mm']>=1e-3 or a['three_D_GT_ROI']
            or a['magic_mapping_applications']!=int(row['source_name']=='magic') or len(a['boxes'])!=2
            or row['fusion_extra_calls']!=0 or len(pa)!=2
            or any(x['box_decoder_calls']!=1 or x['gt_mask_supplied'] for x in pa)):
        raise ValueError('physical/box/fusion audit')


def prepare(out):
    if out.exists():raise ValueError('output already exists; do not overwrite')
    tasks=sorted([t for t in n.read(MANIFEST)['tasks'] if t['source_name'] in COUNTS],key=n.key)
    validate_scope(tasks,n.read(SPLIT));assets=assets_signature()
    n.write(out/'protocol.json',assets)
    n.write(out/'manifest.json',dict(tasks=tasks,expected=4548,per_source_counts=COUNTS,train_overlap=0,
        cases={s:len({t['case_id'] for t in tasks if t['source_name']==s}) for s in COUNTS}))
    metas=[]
    for job in make_jobs(tasks):
        job['output']=str(out/'jobs'/job['name']);job['signature']=signature(job,assets)
        path=out/'job_specs'/(job['name']+'.json');n.write(path,job)
        metas.append(dict(name=job['name'],path=str(path),tasks=len(job['tasks']),smoke=job['smoke']))
    n.write(out/'jobs.json',dict(jobs=metas,total_case_class=4548,total_method_rows=4548))
    print('PREPARED',len(metas),'shards /4548 tasks /nine-source smoke9',flush=True)


def worker(jobpath,uuid):
    import torch
    torch.set_num_threads(2)
    actual=str(torch.cuda.get_device_properties(0).uuid).replace('GPU-','')
    if actual!=uuid.replace('GPU-','') or torch.ones(1,device='cuda').item()!=1:raise ValueError('UUID/CUDA mismatch')
    job=n.read(jobpath);out=Path(job['output']);sig=signature(job)
    if sig!=job['signature']:raise ValueError('changed-code/assets/prompts')
    rows=n.read(out/'all_metrics_live.json')['per_case_class'] if (out/'all_metrics_live.json').exists() else []
    if any(x['signature']!=sig for x in rows):raise ValueError('partial signature changed')
    done={n.key(r) for r in rows};n.validate_rows(rows,[t for t in job['tasks'] if n.key(t) in done])
    for r in rows:validate_audit(r)
    models=soft.SoftModels();loader=Loader(n.read(SPLIT));scorer=n.base_module().metric_module()
    n.write(out/'restoration_audit.json',dict(models.audit,gpu_uuid=uuid,cuda_ordinal=0,actual_cuda_uuid=actual,cuda_tensor_pass=True))
    for t in job['tasks']:
        if n.key(t) in done:continue
        start=time.time();ct,gt,spacing,payload,audit=loader(t)
        pred,pa=predict(ct,spacing,payload,models)
        scores=scorer.binary_prompt_metrics_from_masks(torch.from_numpy(np.ascontiguousarray(pred)),torch.from_numpy(gt),tuple(spacing))
        row=dict(t,method='dual_prob',**{m:float(scores[m]) for m in n.METRICS},prompt_audit=audit,
            propagation_audit=pa,fusion_extra_calls=0,gpu_uuid=uuid,seconds=time.time()-start,signature=sig)
        validate_audit(row);rows.append(row);done.add(n.key(t))
        n.validate_rows(rows,[x for x in job['tasks'] if n.key(x) in done])
        n.write(out/'all_metrics_live.json',dict(per_case_class=rows,signature=sig))
        print('DONE',job['name'],len(rows),'/',len(job['tasks']),t['source_name'],t['case_id'],t['class_name'],
              'Dice',round(row['dice'],5),'seconds',round(row['seconds'],1),flush=True)
    n.validate_rows(rows,job['tasks']);n.write(out/'complete_metrics.json',dict(per_case_class=rows,signature=sig))


def progress(out):
    counts=Counter();complete=0;metas=n.read(out/'jobs.json')['jobs']
    for j in metas:
        run=out/'jobs'/j['name'];file=run/'complete_metrics.json'
        if file.exists():complete+=1
        else:file=run/'all_metrics_live.json'
        if file.exists():counts.update(r['source_name'] for r in n.read(file)['per_case_class'])
    return dict(completed_jobs=complete,total_jobs=len(metas),completed_tasks=sum(counts.values()),per_source=dict(counts),expected=4548)


def merge(out):
    assets=assets_signature()
    if n.read(out/'protocol.json')!=assets:raise ValueError('changed frozen assets')
    tasks=n.read(out/'manifest.json')['tasks'];validate_scope(tasks,n.read(SPLIT));rows=[]
    for meta in n.read(out/'jobs.json')['jobs']:
        job=n.read(meta['path']);data=n.read(Path(job['output'])/'complete_metrics.json')
        if data['signature']!=signature(job,assets) or any(r['signature']!=data['signature'] for r in data['per_case_class']):raise ValueError('merge signature')
        n.validate_rows(data['per_case_class'],job['tasks']);rows+=data['per_case_class']
    n.validate_rows(rows,tasks)
    for row in rows:validate_audit(row)
    n.write(out/'merged_metrics.json',dict(per_case_class=rows,protocol=PROTOCOL,
        summary={s:dict(tasks=c,mean={m:math.fsum(r[m] for r in rows if r['source_name']==s)/c for m in n.METRICS}) for s,c in COUNTS.items()}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','worker','launch','status','merge'))
    parser.add_argument('--output',default=DEFAULT);parser.add_argument('--job');parser.add_argument('--uuid')
    args=parser.parse_args();out=ROOT/args.output
    p.assets_signature=assets_signature;p.progress=progress;p.merge=merge;p.__file__=__file__
    if args.action=='prepare':prepare(out)
    elif args.action=='worker':worker(args.job,args.uuid)
    elif args.action=='launch':p.schedule(out)
    elif args.action=='status':print(json.dumps(progress(out),indent=2))
    else:merge(out)


if __name__=='__main__':main()
