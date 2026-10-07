"""Isolated Turbo pilot: physical 3D scribble points, full canonical metrics."""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
import types
import numpy as np
from scipy.ndimage import affine_transform

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from infer.native_sammed3d_prompts import cube_geometry,to_cube,from_cube,scribble_points

CODE=ROOT/'third_party/SAM_Med3D_official_20261007'
WEIGHT=CODE/'checkpoints/sam_med3d_turbo.pth'
WEIGHT_SHA='899a46d04d3b70f723282ceb489149373558bf0aaba389a346f5ab57da5cdd3c'
OUTPUT=ROOT/'output/native_sammed3d_turbo_pilot_20261007'
BG=ROOT/'output/native_sam2_amos_sparse_bg_pilot_20261007/background_scribbles_light'
ARMS={'fps2':(2,'fps','physical'),'fps4':(4,'fps','physical'),
      'fps8':(8,'fps','physical'),'axis4':(4,'axis','physical'),
      'fps4_stretch':(4,'fps','stretch')}


def select_tasks(tasks):
    names=('liver','spleen','kidney_left','pancreas','aorta','adrenal_gland_left')
    aliases={'left kidney':'kidney_left','left adrenal gland':'adrenal_gland_left'}
    category=lambda t:aliases.get(t['class_name'],t['class_name'])
    rows=[dict(t) for t in tasks if t['source_name']=='amos' and category(t) in names]
    cases=sorted({t['case_id'] for t in rows})[:2]
    rows=sorted([t for t in rows if t['case_id'] in cases],key=lambda t:(t['case_id'],t['class_name']))
    if len(rows)!=12 or any({category(t) for t in rows if t['case_id']==c}!=set(names) for c in cases):
        raise ValueError('pilot needs exactly 2 held-out cases with six specified classes')
    return rows


def resample_cube(ct,bounds,g):
    starts=np.array([a for a,b in bounds]);shape=np.array(g['shape'],int)
    crop=np.asarray(ct[tuple(slice(a,b) for a,b in bounds)],np.float32)
    step=np.array(g['pitch'])/g['spacing']
    offset=(.5-np.array(g['pad']))*step-.5
    # Physical-to-array mapping, with center-consistent coordinates; no GT input.
    return affine_transform(crop,np.diag(step),offset=offset,
        output_shape=(g['size'],)*3,order=1,mode='constant',cval=0.,prefilter=False)


def restore_logits(cube,g):
    step=np.array(g['spacing'])/g['pitch']
    offset=.5*step+np.array(g['pad'])-.5
    return affine_transform(np.asarray(cube,np.float32),np.diag(step),offset=offset,
        output_shape=tuple(np.array(g['shape'],int)),order=1,mode='nearest',prefilter=False)


def read(p):return json.loads(Path(p).read_text())
def write(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+'.{}.tmp'.format(os.getpid()))
    temp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf8');os.replace(temp,p)


def prepare():
    from tools import native_ct13_eval as native
    source=ROOT/'output/native_sam2_medsam2_ct13_full6094_20261007_gridfix/manifest.json'
    rows=select_tasks(read(source)['tasks'])
    split=read(ROOT/'configs/ct13_v10_2_split_20260928.json')
    for t in rows:
        d=split['datasets']['amos']
        assert t['case_id'] in {r['case_id'] for r in d['val']} and t['case_id'] not in {r['case_id'] for r in d['train']}
        assert 'prompt_cache_name' in t
    manifest=dict(tasks=rows,scope='AMOS two cases six classes pilot only',train_overlap=0,arms=ARMS,
                  checkpoint=str(WEIGHT.relative_to(ROOT)),weight_sha256=WEIGHT_SHA)
    if (OUTPUT/'manifest.json').exists():assert read(OUTPUT/'manifest.json')==manifest
    else:write(OUTPUT/'manifest.json',manifest)
    print('PILOT_PREPARED',[(t['case_id'],t['class_name']) for t in rows],flush=True)


def load_model():
    import torch
    from tools.native_ct13_eval import sha
    assert sha(WEIGHT)==WEIGHT_SHA,'official checkpoint SHA mismatch'
    # Namespace package avoids official top-level imports of unused 2D utilities.
    package=types.ModuleType('_native_sammed3d');package.__path__=[str(CODE/'segment_anything')]
    sys.modules['_native_sammed3d']=package
    builder=importlib.import_module('_native_sammed3d.build_sam3D')
    model=builder.sam_model_registry3D['vit_b_ori'](checkpoint=None)
    data=torch.load(WEIGHT,map_location='cpu',weights_only=True)
    state=data.get('model_state_dict',data.get('state_dict',data))
    model.load_state_dict(state,strict=True)
    assert model.image_encoder.img_size==128
    return model.eval().cuda()


def predict(model,ct,fg,bg,spacing,slices,arm):
    import torch
    import torch.nn.functional as F
    from infer.native_sam2_physical_prompts import prompt_bounds_3d,expand_bounds_3d
    cap,method,mode=ARMS[arm]
    points,labels=scribble_points(fg,bg,spacing,slices,cap,method)
    initial=prompt_bounds_3d(fg,bg,spacing);bounds=initial;stages=[]
    for stage_idx in range(8):
        g=cube_geometry(bounds,spacing,128,mode)
        q=to_cube(points,g)
        assert np.all(q>=-.5) and np.all(q<127.5),'3D prompt outside cube'
        assert np.max(abs(from_cube(q,g)-points))<1e-8,'3D point roundtrip'
        cube=resample_cube(ct,bounds,g)
        positive=cube[cube>0]
        if len(positive)<2:raise ValueError('ZNormalization has insufficient positive voxels')
        mean=float(positive.mean());std=max(float(positive.std(ddof=1)),1e-6)
        normalized=(cube-mean)/std
        # Keep physical padding zero rather than treating it as anatomical HU.
        axes=[np.arange(128) for _ in range(3)]
        coords=np.stack(np.meshgrid(*axes,indexing='ij'),-1)
        raw=from_cube(coords,g);start=np.array(g['starts']);shape=np.array(g['shape'])
        valid=np.all((raw>=start-.5)&(raw<start+shape-.5),axis=-1)
        normalized[~valid]=0
        tensor=torch.from_numpy(normalized[None,None].astype(np.float32)).cuda()
        co=torch.as_tensor(q[None],dtype=torch.float32,device='cuda')
        la=torch.as_tensor(labels[None],dtype=torch.long,device='cuda')
        with torch.inference_mode():
            features=model.image_encoder(tensor)
            sparse,dense=model.prompt_encoder(points=(co,la),boxes=None,
                masks=torch.zeros((1,1,32,32,32),device='cuda'))
            low,_=model.mask_decoder(image_embeddings=features,image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,dense_prompt_embeddings=dense,multimask_output=False)
            logits=F.interpolate(low.float(),size=(128,)*3,mode='trilinear',align_corners=False)[0,0].cpu().numpy()
        if not np.isfinite(logits).all():raise ValueError('nonfinite native 3D logits')
        roi=restore_logits(logits,g)>0.
        stages.append(dict(bounds=[list(b) for b in bounds],geometry=g,
            foreground_points=int((labels==1).sum()),background_points=int((labels==0).sum()),
            points_array_order=points.tolist(),points_cube_order=q.tolist(),point_labels=labels.tolist(),
            roundtrip_voxels=float(np.max(abs(from_cube(q,g)-points))),image_mean=mean,image_std=std,
            decoder_calls=1,gt_used_for_roi=False,predicted_shape=list(roi.shape)))
        expanded=expand_bounds_3d(roi,fg,bounds,initial)
        if expanded==bounds:break
        if stage_idx<7:bounds=expanded
        del tensor,features,sparse,dense,low
    result=np.zeros(ct.shape,bool);result[tuple(slice(a,b) for a,b in bounds)]=roi
    return result,stages


def run(arm,uuid):
    import torch
    from tools import native_ct13_eval as native
    assert str(torch.cuda.get_device_properties(0).uuid).replace('GPU-','')==uuid.replace('GPU-','')
    out=OUTPUT/arm
    if (out/'complete_metrics.json').exists():raise ValueError('pilot already complete; no duplicate prediction')
    signature=native.sha(Path(__file__))+'|'+native.sha(ROOT/'infer/native_sammed3d_prompts.py')+'|'+WEIGHT_SHA
    if (out/'protocol.json').exists():assert read(out/'protocol.json')['signature']==signature,'changed-code resume rejected'
    else:write(out/'protocol.json',dict(signature=signature,arm=arm,points_once=True,threshold=.5,
            masks='zero initial previous logits',native_image_size=128,semantic=False,promptgen=False,
            official_source_sha256=native.sha(CODE/'segment_anything/build_sam3D.py')))
    rows=read(out/'all_metrics_live.json')['per_case_class'] if (out/'all_metrics_live.json').exists() else []
    done={native.key(r) for r in rows}
    model=load_model();base=native.base_module();loader=native.CaseLoader(base);scorer=base.metric_module()
    split=read(ROOT/'configs/ct13_v10_2_split_20260928.json');tasks=read(OUTPUT/'manifest.json')['tasks']
    for t in tasks:
        if native.key(t) in done:continue
        began=time.time();ct,target,fg,bg,spacing,audit=loader(t,split,'scribble',str(BG))
        assert audit['background_cache_hit'],'background prompt must reuse existing cache'
        pred,stages=predict(model,ct,fg,bg,spacing,audit['canonical_prompt_slices'],arm)
        metric=scorer.binary_prompt_metrics_from_masks(torch.from_numpy(pred),torch.from_numpy(target),tuple(spacing))
        row=dict(t,**{k:float(metric[k]) for k in native.METRICS},seconds=time.time()-began,
                 prompt_audit=audit,stages=stages,gpu_uuid=uuid)
        rows.append(row);native.validate_rows(rows,[x for x in tasks if native.key(x) in {native.key(r) for r in rows}])
        write(out/'all_metrics_live.json',dict(per_case_class=rows,arm=arm))
        print('DONE',arm,t['case_id'],t['class_name'],'Dice',row['dice'],'seconds',row['seconds'],flush=True)
    native.validate_rows(rows,tasks)
    write(out/'complete_metrics.json',dict(per_case_class=rows,summary={k:float(np.mean([r[k] for r in rows])) for k in native.METRICS}))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true')
    p.add_argument('--arm',choices=tuple(ARMS));p.add_argument('--uuid');a=p.parse_args(argv)
    if a.prepare:prepare()
    elif a.arm and a.uuid:run(a.arm,a.uuid)
    else:p.error('use --prepare or --arm and --uuid')

if __name__=='__main__':main()
