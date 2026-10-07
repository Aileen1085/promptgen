"""Readonly VISTA result reuse and physical half-voxel window correction."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools import native_vista3d_ct13 as full
native=full.native;pilot=full.pilot
OLD=native.ROOT/full.DEFAULT
OUTPUT=native.ROOT/'output/native_vista3d_ct13_full6094_fit_fps4_20261007_cellfix'
OLD_ENTRY='0b1c881c399217e40063847be20b438fba0cbd7818c4c342b645c378c4be796b'
OLD_PILOT='0a616f1ec0915629374b1dc826ed559c2c0ee6610828cee5e526fd3954cc2f3d'
OLD_GEOMETRY='6da377c892f09e614043b6693fc425a500a91cceed9cac353bccccb3c77ebbfc'
ORIGINAL_SIGNATURE=full.signature
ORIGINAL_PROGRESS=full.progress


def cell_inside(points,start,size):
    # Coordinates refer to voxel centers; first cell is [-.5,+.5).
    q=np.asarray(points)-np.asarray(start)
    return np.all((q>=-.5)&(q<size-.5),axis=1)


def patch_predict(model,image,points,labels):
    import torch
    from scipy.ndimage import label
    ws,shape,pad=pilot.windows(points,image.shape,128)
    padded=np.pad(image,[(int(a),int(b-n-a)) for a,b,n in zip(pad,shape,image.shape)],mode='constant')
    q=points+pad;outputs=[];used=set();audits=[]
    for bounds in ws:
        start=np.array([a for a,b in bounds]);inside=cell_inside(q,start,128)
        if not inside.any():raise ValueError('empty point-centered patch')
        ids=np.flatnonzero(inside);used.update(ids.tolist())
        co=torch.as_tensor((q[inside]-start)[None],dtype=torch.float32,device='cuda')
        la=torch.as_tensor(labels[inside][None],dtype=torch.long,device='cuda')
        tensor=torch.from_numpy(np.ascontiguousarray(padded[tuple(slice(a,b) for a,b in bounds)])[None,None]).cuda()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            out=model(input_images=tensor,point_coords=co,point_labels=la,class_vector=None,
                      prompt_class=None,prev_mask=None,labels=None,keep_cache=False)
        x=out[0,0].float().cpu().numpy()
        if x.shape!=(128,)*3 or not np.isfinite(x).all():raise ValueError('invalid native VISTA output')
        outputs.append(x);audits.append(dict(bounds=bounds,encoded_point_indices=ids.tolist(),forward_calls=1,
            class_vector=None,prompt_class=None,previous_mask=None,gt_used=False))
        del tensor,co,la,out
    if used!=set(range(len(points))):raise ValueError('unencoded selected point')
    logits=pilot.fuse(outputs,ws,shape)
    logits=logits[tuple(slice(int(a),int(a+n)) for a,n in zip(pad,image.shape))]
    mask=logits>0;components,_=label(mask)
    positive=np.clip(np.rint(points[labels==1]).astype(int),0,np.array(image.shape)-1)
    selected=np.unique(components[tuple(positive.T)]);selected=selected[selected!=0]
    logits[mask&(~np.isin(components,selected))]=-9999.
    return logits,audits


def signature(job):
    return hashlib.sha256((ORIGINAL_SIGNATURE(job)+native.sha(Path(__file__))).encode()).hexdigest()


def prepare():
    assert native.sha(Path(full.__file__))==OLD_ENTRY
    assert native.sha(Path(pilot.__file__))==OLD_PILOT
    assert native.sha(native.ROOT/'infer/native_vista3d_prompts.py')==OLD_GEOMETRY
    assert not OUTPUT.exists(),'inspect existing recovery before preparing'
    manifest=native.read(OLD/'manifest.json');tasks=manifest['tasks']
    full.validate_scope(tasks,native.read(native.ROOT/'configs/ct13_v10_2_split_20260928.json'))
    refs=[];rows=[];pending_jobs=[];complete_jobs=0
    for meta in native.read(OLD/'jobs.json')['jobs']:
        job=native.read(meta['path']);out=Path(job['output'])
        file=out/'complete_metrics.json'
        complete=file.is_file()
        if not complete:file=out/'all_metrics_live.json'
        old=[]
        if file.is_file():
            data=native.read(file);old=data['per_case_class'];sig=ORIGINAL_SIGNATURE(job)
            assert data['signature']==sig
            assert native.read(out/'protocol.json')==dict(signature=sig,protocol=full.PROTOCOL,weight_sha256=pilot.WEIGHT_SHA)
            keys={native.key(r) for r in old}
            native.validate_rows(old,[t for t in job['tasks'] if native.key(t) in keys])
            for r in old:full.validate_audit(r)
            refs.append(dict(file=str(file),count=len(old),files={str(p):native.sha(p) for p in (file,Path(meta['path']),out/'protocol.json')}))
            rows+=old
        done={native.key(r) for r in old};pending=[t for t in job['tasks'] if native.key(t) not in done]
        if not pending:complete_jobs+=1
        else:pending_jobs.append((meta,dict(job,tasks=pending,output=str(OUTPUT/'jobs'/job['name']))))
    keys={native.key(r) for r in rows};assert len(keys)==len(rows)
    native.validate_rows(rows,[t for t in tasks if native.key(t) in keys])
    assert len(rows)+sum(len(j['tasks']) for _,j in pending_jobs)==6094
    native.write(OUTPUT/'manifest.json',manifest);metas=[]
    for meta,job in pending_jobs:
        path=OUTPUT/'job_specs'/(job['name']+'.json');native.write(path,job)
        metas.append(dict(meta,path=str(path),tasks=len(job['tasks'])))
    native.write(OUTPUT/'jobs.json',dict(jobs=metas,total_tasks=6094-len(rows)))
    native.write(OUTPUT/'protocol.json',full.PROTOCOL)
    native.write(OUTPUT/'reuse_audit.json',dict(refs=refs,reused_tasks=len(rows),old_completed_jobs=complete_jobs,
        recovery_sha256=native.sha(Path(__file__)),old_entry_sha256=OLD_ENTRY,old_pilot_sha256=OLD_PILOT,
        reason='voxel-cell membership [-0.5,size-0.5); unchanged physical point coordinates and fit128 inputs'))
    print('PREPARED',len(rows),'readonly reused;',6094-len(rows),'pending;',len(metas),'shards',flush=True)


def reused_rows():
    audit=native.read(OUTPUT/'reuse_audit.json');assert audit['recovery_sha256']==native.sha(Path(__file__))
    assert native.sha(Path(full.__file__))==OLD_ENTRY and native.sha(Path(pilot.__file__))==OLD_PILOT
    rows=[]
    for ref in audit['refs']:
        for path,sha in ref['files'].items():assert native.sha(path)==sha,'readonly reused file changed'
        data=native.read(ref['file'])['per_case_class'];assert len(data)==ref['count'];rows+=data
    assert len(rows)==audit['reused_tasks'];return rows


def progress(out):
    result=ORIGINAL_PROGRESS(out);old=reused_rows()
    result['completed_tasks']+=len(old)
    result['completed_jobs']+=native.read(OUTPUT/'reuse_audit.json')['old_completed_jobs']
    counts=Counter(result['per_source']);counts.update(r['source_name'] for r in old);result['per_source']=dict(counts)
    result['total_tasks']=6094
    return result


def merge(out):
    rows=list(reused_rows())
    for meta in native.read(out/'jobs.json')['jobs']:
        job=native.read(meta['path']);data=native.read(Path(job['output'])/'complete_metrics.json')
        assert data['signature']==signature(job)
        native.validate_rows(data['per_case_class'],job['tasks']);rows+=data['per_case_class']
    native.validate_rows(rows,native.read(out/'manifest.json')['tasks'])
    assert dict(Counter(r['source_name'] for r in rows))==native.COUNTS
    for r in rows:full.validate_audit(r)
    native.write(out/'merged_metrics.json',dict(per_case_class=rows,protocol=full.PROTOCOL,weight_sha256=pilot.WEIGHT_SHA,
        readonly_reuse_audit='reuse_audit.json',summary={m:float(np.mean([r[m] for r in rows])) for m in native.METRICS},
        per_source={s:dict(count=n,**{m:float(np.mean([r[m] for r in rows if r['source_name']==s])) for m in native.METRICS}) for s,n in native.COUNTS.items()}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for a in ('prepare','schedule','status','merge','worker'):p.add_argument('--'+a,action='store_true')
    p.add_argument('--job');p.add_argument('--uuid');a=p.parse_args()
    if a.prepare:prepare();return
    reused_rows();pilot.patch_predict=patch_predict;full.signature=signature;full.progress=progress;full.merge=merge
    if a.status:print(json.dumps(progress(OUTPUT),indent=2))
    elif a.merge:merge(OUTPUT)
    elif a.worker:full.worker(a.job,a.uuid)
    elif a.schedule:
        # Existing scheduler is reused, but child launch must use this signed recovery.
        original=full.subprocess.Popen
        def launch(command,**kwargs):
            command=list(command)
            assert command[1]==str(Path(full.__file__))
            command[1]=str(Path(__file__));return original(command,**kwargs)
        full.subprocess.Popen=launch;full.schedule(OUTPUT,{0,2,3,4,5,6})
    else:p.error('choose action')


if __name__=='__main__':main()
