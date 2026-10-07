"""Native VISTA3D physical scribble pilot; no class head, PromptGen or GT clicks."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.ndimage import affine_transform,label

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from infer.native_vista3d_prompts import geometry,forward_points,inverse_points,native_view,canonical_view,windows,fuse
from infer.native_sammed3d_prompts import scribble_points
from infer.native_sam2_physical_prompts import prompt_bounds_3d,expand_bounds_3d
from tools import native_ct13_eval as native,native_sammed3d_pilot as shared
CODE=ROOT/'vista3d/official_source/vista3d';WEIGHT=ROOT/'vista3d/official_weights/model_monai1.3.pt'
WEIGHT_SHA='889042ab37dbb9f9b2467e4a91654fb3b56482b74b98d315fc89026ab1570af8'
OUTPUT=ROOT/'output/native_vista3d_physical_pilot_20261007'
ARMS={'fixed_fps2':(2,'fps','fixed'),'fixed_fps4':(4,'fps','fixed'),
      'fixed_axis4':(4,'axis','fixed'),'fit_fps4':(4,'fps','fit')}
PROTOCOL=dict(model='VISTA3D research official raw',threshold=.5,semantic=False,promptgen=False,
    previous_mask=None,points_once_per_patch=True,class_vector=None,prompt_class=None,
    patch_size=128,point_spacing_mm=3.,normalization='official_HU_-963.8247715525971_1053.678477684517',
    roi='prompt8mm20pct_connected_expand50_cap100',fusion='unique_windows_overlap_count_average_logits',
    postprocess='positive_selected_point_connected_components',metrics=list(native.METRICS))

def normalize(ct):return np.clip((np.asarray(ct,np.float32)+963.8247715525971)/(1053.678477684517+963.8247715525971),0,1)

def resample(ct,g):
    crop=np.asarray(ct[tuple(slice(int(a),int(a+n)) for a,n in zip(g['starts'],g['shape']))],np.float32)
    return affine_transform(crop,np.diag(np.array(g['pitch'])/g['spacing']),g['offset'],
        output_shape=tuple(g['target']),order=1,mode='nearest',prefilter=False)

def restore(logits,g):
    step=np.array(g['spacing'])/g['pitch'];offset=-np.array(g['offset'])*step
    return affine_transform(np.asarray(logits,np.float32),np.diag(step),offset,
        output_shape=tuple(g['shape']),order=1,mode='nearest',prefilter=False)

def load_model():
    if native.sha(WEIGHT)!=WEIGHT_SHA:raise ValueError('official VISTA weight mismatch')
    spec=importlib.util.spec_from_file_location('_official_vista_loader',ROOT/'vista3d/official_loader.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    return helper.load_official_vista3d(CODE,WEIGHT,'cuda')

def patch_predict(model,image,points,labels):
    import torch
    ws,shape,pad=windows(points,image.shape,128)
    padded=np.pad(image,[(int(a),int(b-n-a)) for a,b,n in zip(pad,shape,image.shape)],mode='constant')
    q=points+pad;outputs=[];used=set();audits=[]
    for bounds in ws:
        start=np.array([a for a,b in bounds]);stop=start+128
        inside=np.all((q>=start)&(q<stop),axis=1)
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
    logits=fuse(outputs,ws,shape)
    logits=logits[tuple(slice(int(a),int(a+n)) for a,n in zip(pad,image.shape))]
    mask=logits>0;components,_=label(mask)
    positive=np.clip(np.rint(points[labels==1]).astype(int),0,np.array(image.shape)-1)
    selected=np.unique(components[tuple(positive.T)]);selected=selected[selected!=0]
    connected=np.isin(components,selected)
    logits[(mask)&(~connected)]=-9999.
    return logits,audits

def predict(model,ct,fg,bg,spacing,slices,arm):
    cap,sampler,mode=ARMS[arm];points,roles=scribble_points(fg,bg,spacing,slices,cap,sampler)
    initial=prompt_bounds_3d(fg,bg,spacing);bounds=initial;stages=[]
    for i in range(8):
        g=geometry(bounds,spacing,mode);q=forward_points(points,g)
        error=float(np.max(abs(inverse_points(q,g)-points)))
        if error>1e-8:raise ValueError('point roundtrip')
        image=resample(ct,g)
        if mode=='fit':
            # Constant physical padding instead of anatomical edge replication.
            valid=np.ones(image.shape,bool)
            for axis,n in enumerate(image.shape):
                raw=inverse_points(np.eye(3)[axis]*np.arange(n)[:,None],g)[:,axis]-g['starts'][axis]
                index=[None]*3;index[axis]=slice(None)
                valid&=((raw>=-.5)&(raw<g['shape'][axis]-.5))[tuple(index)]
            image[~valid]=0
        image,coords=native_view(image,q)
        logits,patches=patch_predict(model,image,coords,roles)
        roi=restore(canonical_view(logits),g)>0
        stages.append(dict(bounds=bounds,geometry=g,roundtrip_voxels=error,point_axis_order='canonical_XYZ',
            selected_points_dhw=points.tolist(),points_native_xyz=coords.tolist(),point_labels=roles.tolist(),
            foreground_points=int((roles==1).sum()),background_points=int((roles==0).sum()),
            patches=patches,gt_used_for_roi=False,encoded_all_selected_points=True))
        expanded=expand_bounds_3d(roi,fg,bounds,initial)
        if expanded==bounds:break
        if i<7:bounds=expanded
    result=np.zeros(ct.shape,bool);result[tuple(slice(a,b) for a,b in bounds)]=roi
    return result,stages

def prepare():
    tasks=shared.select_tasks(native.read(ROOT/'output/native_sam2_medsam2_ct13_full6094_20261007_gridfix/manifest.json')['tasks'])
    split=native.read(ROOT/'configs/ct13_v10_2_split_20260928.json')['datasets']['amos']
    for t in tasks:
        if t['case_id'] not in {r['case_id'] for r in split['val']} or t['case_id'] in {r['case_id'] for r in split['train']}:raise ValueError('val membership')
    manifest=dict(tasks=tasks,arms=ARMS,train_overlap=0,weight_sha256=WEIGHT_SHA,protocol=PROTOCOL)
    if (OUTPUT/'manifest.json').exists():
        if native.read(OUTPUT/'manifest.json')!=json.loads(json.dumps(manifest)):raise ValueError('changed manifest')
    else:native.write(OUTPUT/'manifest.json',manifest)
    print('PREPARED 12 held-out tasks x 4 arms',flush=True)

def signature(tasks):
    files=[Path(__file__),ROOT/'infer/native_vista3d_prompts.py',ROOT/'infer/native_sammed3d_prompts.py',
           ROOT/'infer/native_sam2_physical_prompts.py',ROOT/'tools/native_ct13_eval.py',
           ROOT/'tools/native_sam2_physical_eval.py',ROOT/'tools/native_sammed3d_pilot.py',
           ROOT/'utils/metrics.py',ROOT/'vista3d/official_loader.py',ROOT/'configs/ct13_v10_2_split_20260928.json']
    files+=sorted((CODE/'vista3d').rglob('*.py'))+sorted((CODE/'scripts/utils').rglob('*.py'))
    files+=[native.CACHE/'metadata/shared_adaptive_prompt_v1'/t['source_name']/t['prompt_cache_name'] for t in tasks]
    return hashlib.sha256((''.join(native.sha(f) for f in files)+WEIGHT_SHA+json.dumps(tasks,sort_keys=True)).encode()).hexdigest()

def run(arm,uuid,limit=None):
    import torch
    if str(torch.cuda.get_device_properties(0).uuid).replace('GPU-','')!=uuid.replace('GPU-',''):raise ValueError('CUDA UUID mismatch')
    tasks=native.read(OUTPUT/'manifest.json')['tasks'];out=OUTPUT/arm
    protocol=dict(PROTOCOL,signature=signature(tasks),arm=arm,weight_sha256=WEIGHT_SHA)
    if (out/'protocol.json').exists():
        if native.read(out/'protocol.json')!=protocol:raise ValueError('changed-code resume refused')
    else:native.write(out/'protocol.json',protocol)
    if (out/'complete_metrics.json').exists():print('ALREADY_COMPLETE');return
    rows=native.read(out/'all_metrics_live.json')['per_case_class'] if (out/'all_metrics_live.json').exists() else []
    done={native.key(r) for r in rows};native.validate_rows(rows,[t for t in tasks if native.key(t) in done])
    model=load_model();base=native.base_module();loader=native.CaseLoader(base);scorer=base.metric_module()
    split=native.read(ROOT/'configs/ct13_v10_2_split_20260928.json');case=None;normalized=None;n=0
    for t in tasks:
        if native.key(t) in done:continue
        start=time.time();ct,gt,fg,bg,spacing,audit=loader(t,split,'scribble',str(shared.BG))
        if not audit['background_cache_hit'] or audit['fg_in_gt_fraction']!=1 or audit['bg_in_gt_count']!=0:raise ValueError('prompt alignment/cache')
        if case!=t['case_id']:normalized=normalize(ct);case=t['case_id']
        pred,stages=predict(model,normalized,fg,bg,spacing,audit['canonical_prompt_slices'],arm)
        metric=scorer.binary_prompt_metrics_from_masks(torch.from_numpy(pred),torch.from_numpy(gt),tuple(spacing))
        row=dict(t,**{k:float(metric[k]) for k in native.METRICS},seconds=time.time()-start,prompt_audit=audit,stages=stages,gpu_uuid=uuid)
        rows.append(row);done.add(native.key(t));native.validate_rows(rows,[x for x in tasks if native.key(x) in done])
        native.write(out/'all_metrics_live.json',dict(per_case_class=rows,protocol=protocol))
        print('DONE',arm,len(rows),'/12',t['case_id'],t['class_name'],'Dice',row['dice'],'seconds',row['seconds'],flush=True)
        n+=1
        if limit and n>=limit:break
    if len(rows)==12:native.write(out/'complete_metrics.json',dict(per_case_class=rows,protocol=protocol,
        summary={k:float(np.mean([r[k] for r in rows])) for k in native.METRICS}))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true')
    p.add_argument('--arm',choices=tuple(ARMS));p.add_argument('--uuid');p.add_argument('--limit',type=int);a=p.parse_args()
    if a.prepare:prepare()
    elif a.arm and a.uuid:run(a.arm,a.uuid,a.limit)
    else:p.error('use --prepare or --arm and --uuid')

if __name__=='__main__':main()
