"""Native SegVol physical scribble conversion pilot on twelve held-out AMOS tasks."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.ndimage import affine_transform

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from infer.native_segvol_prompts import geometry,forward_points,inverse_points,native_view,canonical_view
from infer.native_sammed3d_prompts import scribble_points
from infer.native_sam2_physical_prompts import prompt_bounds_3d,expand_bounds_3d
from tools import native_ct13_eval as native,native_sammed3d_pilot as shared
CODE=ROOT/'segvol-3d/vendor/SegVol';WEIGHT=ROOT/'segvol-3d/weights/pytorch_model.bin'
WEIGHT_SHA='500f2758a8f989339b2b2baf09a819169bc87549795193d3cfe505726ac0b399'
OUTPUT=ROOT/'output/native_segvol_physical_pilot_20261007'
ARMS={'official_fps2':(2,'fps','stretch','official'),'official_fps4':(4,'fps','stretch','official'),
      'official_axis4':(4,'axis','stretch','official'),'physical_fps4':(4,'fps','physical','official'),
      'official_fps4_pe_aligned':(4,'fps','stretch','pe_aligned')}

def normalize(ct):
    x=np.asarray(ct,np.float32);f=x[x>x.mean()]
    if f.size<2:f=x.reshape(-1)
    lo,hi=np.percentile(f,(.05,99.95));x=(np.clip(x,lo,hi)-f.mean())/max(float(f.std()),1e-8)
    x-=x.min();x/=max(float(x.max()),1e-8);return x

def resample(ct,g):
    slices=tuple(slice(int(a),int(a+n)) for a,n in zip(g['starts'],g['shape']))
    step=np.array(g['pitch'])/g['spacing'];offset=(.5-np.array(g['pad']))*step-.5
    return affine_transform(np.asarray(ct[slices],np.float32),np.diag(step),offset,
        output_shape=tuple(g['target']),order=1,mode='nearest',prefilter=False)

def restore(logits,g):
    step=np.array(g['spacing'])/g['pitch'];offset=.5*step+np.array(g['pad'])-.5
    return affine_transform(logits,np.diag(step),offset,output_shape=tuple(g['shape']),
        order=1,mode='nearest',prefilter=False)

def load_model():
    import torch,importlib,types
    from argparse import Namespace
    if native.sha(WEIGHT)!=WEIGHT_SHA:raise ValueError('SegVol official weight SHA mismatch')
    package=types.ModuleType('_native_segvol');package.__path__=[str(CODE/'segment_anything_volumetric')]
    sys.modules['_native_segvol']=package
    builder=importlib.import_module('_native_segvol.build_sam')
    model=builder.sam_model_registry['vit'](args=Namespace(spatial_size=(32,256,256),patch_size=(4,16,16)))
    saved=torch.load(WEIGHT,map_location='cpu',weights_only=True);state=saved.get('model',saved)
    cleaned={k.removeprefix('module.').removeprefix('model.'):v for k,v in state.items()}
    if len(cleaned)!=len(state):raise ValueError('checkpoint key collision')
    for name in ('image_encoder','prompt_encoder','mask_decoder'):
        prefix=name+'.';getattr(model,name).load_state_dict({k[len(prefix):]:v for k,v in cleaned.items() if k.startswith(prefix)},strict=True)
    return model.eval().cuda()

def predict(model,ct,fg,bg,spacing,slices,arm):
    import torch
    import torch.nn.functional as F
    cap,sampler,mode,convention=ARMS[arm]
    points,labels=scribble_points(fg,bg,spacing,slices,cap,sampler)
    initial=prompt_bounds_3d(fg,bg,spacing);bounds=initial;stages=[]
    for i in range(8):
        g=geometry(bounds,spacing,mode);q=forward_points(points,g)
        error=float(np.max(abs(inverse_points(q,g)-points)))
        if error>1e-8 or np.any(q<-.5) or np.any(q>=np.array(g['target'])-.5+1e-8):raise ValueError('point geometry')
        image=resample(ct,g)
        if mode=='physical':
            axes=[np.arange(n) for n in g['target']];raw=inverse_points(np.stack(np.meshgrid(*axes,indexing='ij'),-1),g)
            valid=np.all((raw>=np.array(g['starts'])-.5)&(raw<np.array(g['starts'])+g['shape']-.5),axis=-1);image[~valid]=0
        image,coords=native_view(image,q,convention)
        tensor=torch.from_numpy(image[None,None]).cuda()
        co=torch.as_tensor(coords[None],dtype=torch.float32,device='cuda');la=torch.as_tensor(labels[None],device='cuda')
        with torch.inference_mode():
            tokens,_=model.image_encoder(tensor);features=tokens.transpose(1,2).reshape(1,768,8,16,16)
            sparse,dense=model.prompt_encoder(points=(co,la),boxes=None,masks=None,text_embedding=None)
            low,_=model.mask_decoder(image_embeddings=features,text_embedding=None,image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,dense_prompt_embeddings=dense,multimask_output=False)
            logits=F.interpolate(low.float(),size=(32,256,256),mode='trilinear',align_corners=False)[0,0].cpu().numpy()
        if not np.isfinite(logits).all():raise ValueError('nonfinite SegVol logits')
        roi=restore(canonical_view(logits),g)>0
        stages.append(dict(bounds=[list(b) for b in bounds],geometry=g,roundtrip_voxels=error,
            tensor_axis_order='Z_Y_X',point_convention=convention,points_native=coords.tolist(),point_labels=labels.tolist(),
            foreground_points=int((labels==1).sum()),background_points=int((labels==0).sum()),decoder_calls=1,gt_used_for_roi=False))
        expanded=expand_bounds_3d(roi,fg,bounds,initial)
        if expanded==bounds:break
        if i<7:bounds=expanded
    pred=np.zeros(ct.shape,bool);pred[tuple(slice(a,b) for a,b in bounds)]=roi
    return pred,stages

def prepare():
    tasks=shared.select_tasks(native.read(ROOT/'output/native_sam2_medsam2_ct13_full6094_20261007_gridfix/manifest.json')['tasks'])
    split=native.read(ROOT/'configs/ct13_v10_2_split_20260928.json')['datasets']['amos']
    for t in tasks:
        if t['case_id'] not in {r['case_id'] for r in split['val']} or t['case_id'] in {r['case_id'] for r in split['train']}:raise ValueError('val membership')
    data=dict(tasks=tasks,arms=ARMS,train_overlap=0,weight_sha256=WEIGHT_SHA)
    if (OUTPUT/'manifest.json').exists():
        if native.read(OUTPUT/'manifest.json')!=json.loads(json.dumps(data)):raise ValueError('changed manifest')
    else:native.write(OUTPUT/'manifest.json',data)
    print('PREPARED 12 held-out tasks x 5 arms',flush=True)

def signature():
    files=[Path(__file__),ROOT/'infer/native_segvol_prompts.py',ROOT/'infer/native_sammed3d_prompts.py',
        ROOT/'infer/native_sam2_physical_prompts.py',ROOT/'tools/native_ct13_eval.py',ROOT/'utils/metrics.py']
    files+=sorted((CODE/'segment_anything_volumetric').rglob('*.py'))
    return hashlib.sha256((''.join(native.sha(f) for f in files)+WEIGHT_SHA).encode()).hexdigest()

def run(arm,uuid,limit=None):
    import torch
    if str(torch.cuda.get_device_properties(0).uuid).replace('GPU-','')!=uuid.replace('GPU-',''):raise ValueError('GPU mapping')
    tasks=native.read(OUTPUT/'manifest.json')['tasks'];out=OUTPUT/arm
    protocol=dict(signature=signature(),arm=arm,threshold=.5,semantic=False,promptgen=False,masks=None,points_once=True,
        resize='center-consistent linear',roi='prompt8mm20pct_connected50_cap100',weight_sha256=WEIGHT_SHA)
    if (out/'protocol.json').exists():
        if native.read(out/'protocol.json')!=protocol:raise ValueError('changed-code resume')
    else:native.write(out/'protocol.json',protocol)
    if (out/'complete_metrics.json').exists():print('ALREADY_COMPLETE',arm);return
    rows=native.read(out/'all_metrics_live.json')['per_case_class'] if (out/'all_metrics_live.json').exists() else []
    done={native.key(r) for r in rows};native.validate_rows(rows,[t for t in tasks if native.key(t) in done])
    model=load_model();base=native.base_module();loader=native.CaseLoader(base);scorer=base.metric_module()
    split=native.read(ROOT/'configs/ct13_v10_2_split_20260928.json');case=None;ctnorm=None;n=0
    for t in tasks:
        if native.key(t) in done:continue
        began=time.time();ct,gt,fg,bg,spacing,audit=loader(t,split,'scribble',str(shared.BG))
        if not audit['background_cache_hit'] or audit['fg_in_gt_fraction']!=1 or audit['bg_in_gt_count']!=0:raise ValueError('prompt audit')
        if case!=t['case_id']:ctnorm=normalize(ct);case=t['case_id']
        pred,stages=predict(model,ctnorm,fg,bg,spacing,audit['canonical_prompt_slices'],arm)
        metric=scorer.binary_prompt_metrics_from_masks(torch.from_numpy(pred),torch.from_numpy(gt),tuple(spacing))
        row=dict(t,**{k:float(metric[k]) for k in native.METRICS},seconds=time.time()-began,prompt_audit=audit,stages=stages,gpu_uuid=uuid)
        rows.append(row);done.add(native.key(t));native.validate_rows(rows,[x for x in tasks if native.key(x) in done])
        native.write(out/'all_metrics_live.json',dict(per_case_class=rows,protocol=protocol))
        print('DONE',arm,len(rows),'/12',t['case_id'],t['class_name'],'Dice',row['dice'],'seconds',row['seconds'],flush=True)
        n+=1
        if limit and n>=limit:break
    if len(rows)==len(tasks):native.write(out/'complete_metrics.json',dict(per_case_class=rows,protocol=protocol,
        summary={k:float(np.mean([r[k] for r in rows])) for k in native.METRICS}))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true')
    p.add_argument('--arm',choices=tuple(ARMS));p.add_argument('--uuid');p.add_argument('--limit',type=int)
    a=p.parse_args()
    if a.prepare:prepare()
    elif a.arm and a.uuid:run(a.arm,a.uuid,a.limit)
    else:p.error('use --prepare or --arm/--uuid')

if __name__=='__main__':main()
