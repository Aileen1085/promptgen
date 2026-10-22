"""Isolated MAGIC rerun: every calibration seed must have model coverage."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import magic_promptgen_three_backbones as e

OUT='output/magic_promptgen_three_backbones35_20261022_validroi'
ORIGINAL_SIGNATURE=e.signature
ORIGINAL_LOAD_MODEL=e.load_model
ORIGINAL_MERGE=e.merge
PYTHONS=dict(sammed3d='../conda-envs/al/bin/python',
             vista3d='vista3d/.venv-vista/bin/python',segvol='segvol-3d/.venv/bin/python')


def include_prompt_bounds(bounds,foreground,background):
    import numpy as np
    fg=np.asarray(foreground,bool);bg=np.asarray(background,int).reshape(-1,3)
    if not fg.any() or not len(bg) or (bg<0).any() or (bg>=fg.shape).any():
        raise ValueError('empty/invalid foreground or background coordinates')
    points=np.concatenate((np.argwhere(fg),bg))
    lo=points.min(axis=0);hi=points.max(axis=0)+1
    result=tuple((min(int(a),int(lo[i])),max(int(b),int(hi[i]))) for i,(a,b) in enumerate(bounds))
    if len(result)!=3 or any(a<0 or b>fg.shape[i] or a>=b for i,(a,b) in enumerate(result)):
        raise ValueError('invalid prompt ROI bounds')
    return result


def initial_z_bounds(foreground,background):
    import numpy as np
    from tools import segvol_promptgen_amos_eval as sg
    fg=np.asarray(foreground,bool);bg=np.asarray(background,int).reshape(-1,3)
    include_prompt_bounds(((0,len(fg)),(0,fg.shape[1]),(0,fg.shape[2])),fg,bg)
    support=fg.copy();support[tuple(bg.T)]=True
    return sg.initial_z_bounds(support)


def calibration_audit(foreground,background,valid):
    import numpy as np
    fg=np.asarray(foreground,bool);bg=np.asarray(background,bool);valid=np.asarray(valid,bool)
    if fg.shape!=bg.shape or fg.shape!=valid.shape or not fg.any() or not bg.any():
        raise ValueError('nonempty matching calibration masks required')
    fg_out=int((fg&~valid).sum());bg_out=int((bg&~valid).sum())
    if fg_out or bg_out:
        raise ValueError('calibration prompt outside actual model-predicted coverage: FG=%d BG=%d'%(fg_out,bg_out))
    return dict(valid_foreground_count=int((fg&valid).sum()),valid_background_count=int((bg&valid&~fg).sum()),
        foreground_outside_predicted_roi=fg_out,background_outside_predicted_roi=bg_out,
        actual_predicted_voxels=int(valid.sum()),validity_source='explicit prediction coverage, not probability or GT')


def postprocess_valid(probability,foreground,background,valid):
    import numpy as np
    import torch
    from v10.infer.infer_amos_sam2_promptgen_v10 import postprocess_v10_logits_from_prompts
    audit=calibration_audit(foreground,background,valid)
    prob=torch.as_tensor(probability).float().cpu()
    if tuple(prob.shape)!=foreground.shape or not torch.isfinite(prob[torch.from_numpy(valid)]).all():
        raise ValueError('invalid/nonfinite probability on predicted coverage')
    logits=torch.logit(prob.clamp(1e-6,1-1e-6))[:,None]
    pred,threshold=postprocess_v10_logits_from_prompts(logits,foreground&valid,background&valid,
        fallback=.60,minimum=.35,maximum=.75,min_logit_separation=.50)
    pred=np.asarray(pred,bool)&valid
    return pred,threshold,audit


def view_predict(model_name,model,training,dataset,index,entry,task,raw,text,physical):
    import numpy as np
    import torch
    from tools import segvol_promptgen_amos_eval as sg
    from v10_data import SAM2CachedTotalSegV10Dataset
    from v10.infer.infer_amos_sam2_promptgen_v10 import expand_prompt_roi_bounds,prompt_connected_z_boundary_contacts
    ct,gt,fg,bg,spacing,slices,supports,prompt_audit=raw
    groups,planes,coords=sg.raw_prompt_groups(fg,bg,slices,supports)
    square=max(ct.shape[1:]);top=(square-ct.shape[1])//2;left=(square-ct.shape[2])//2
    groups=[SAM2CachedTotalSegV10Dataset._resize_prompt_group_preserve(g,256,pad)
        for g,pad in zip(groups,[(top,square-ct.shape[1]-top),(left,square-ct.shape[2]-left)])]
    bundle=dict(groups=torch.from_numpy(np.stack(groups).astype(np.float32)),plane_ids=torch.tensor(planes),
        slice_coords=torch.tensor(coords,dtype=torch.float32),canonical_depth=len(ct))
    initial=initial_z_bounds(fg,bg);bounds=initial;stages=[]
    bgmask=np.zeros(ct.shape,bool);bgmask[tuple(bg.T)]=True
    for iteration in range(8):
        sg.bind_readonly(dataset,ct,gt,bundle,bounds,int(task['global_class_id']))
        wrapper=dataset
        while True:
            wrapper._atomic_save=lambda *a,**k:None
            if 'dataset' not in vars(wrapper):break
            wrapper=wrapper.dataset
        kw=dict(vista_hw=96,prompt_hw=model.promptgen.work_size) if model_name=='vista3d' else {}
        view=training.read_case_view(dataset,index,int(task['global_class_id']),Path(entry['image']),**kw)
        frames=np.asarray(view.frame_indices)
        if not np.array_equal(frames,np.arange(*bounds)):raise ValueError('complete ROI frames mismatch')
        view.bundle['v10_forced_background_mode']='point'
        depth,stride=(96,48) if model_name=='vista3d' else (32,16)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            probability=training.predict_complete_roi(model,view,text,depth=depth,stride=stride,device='cuda:0')
        valid=np.zeros(ct.shape,bool);valid[frames]=True
        pred,threshold,calibration=postprocess_valid(probability,fg,bgmask,valid)
        start,end=bounds;touch=prompt_connected_z_boundary_contacts(pred[start:end],fg[start:end],band_frames=2)
        grown=expand_prompt_roi_bounds(current_bounds=bounds,initial_bounds=initial,available_bounds=(0,len(ct)),
            step_ratio=.50,max_expand_ratio=1.,expand_start=touch[0],expand_end=touch[1])
        stages.append(dict(z_bounds=list(bounds),depth=depth,stride=stride,complete_roi_frames=True,
            threshold=threshold,calibration=calibration,background_points_encoded=int(len(bg)),
            all_prompts_before_forward=True,gt_used_for_roi=False))
        del view,probability,valid
        if grown==bounds:return pred,stages
        bounds=grown
    raise RuntimeError('bounded AutoExpand failed to converge')


def predict_sammed(model,raw,data,training,geometry):
    import numpy as np
    import torch
    initial=include_prompt_bounds(geometry.prompt_bounds(raw.foreground,raw.background,raw.spacing),raw.foreground,raw.background)
    bounds=initial;stages=[]
    bg=np.zeros(raw.ct.shape,bool);bg[tuple(raw.background.T)]=True
    for iteration in range(8):
        packed=data.pack_cube(raw,bounds,'cuda:0')
        if int((packed['labels']==0).sum())!=4:raise ValueError('background point reduction')
        with torch.autocast('cuda',dtype=torch.bfloat16):result=training.forward(model,packed,raw.audit['class_name'])
        logits=result['logits'][0,0].float()
        if not torch.isfinite(logits).all():raise FloatingPointError('nonfinite SAM-Med3D logits')
        probability=geometry.restore_full(logits.sigmoid().cpu().numpy(),packed['geometry'],raw.ct.shape)
        valid=np.zeros(raw.ct.shape,bool);valid[tuple(slice(a,b) for a,b in bounds)]=True
        pred,threshold,calibration=postprocess_valid(probability,raw.foreground,bg,valid)
        grown=geometry.expand_bounds(pred[tuple(slice(a,b) for a,b in bounds)],raw.foreground,bounds,initial)
        q=packed['points'][0].cpu().numpy()[:,[2,0,1]]
        error=float(np.abs(geometry.from_cube(q[-4:],packed['geometry'])-raw.background).max())
        if error>1e-4:raise ValueError('background coordinate roundtrip error')
        stages.append(dict(bounds=[list(b) for b in bounds],threshold=threshold,calibration=calibration,
            background_points_encoded=int((packed['labels']==0).sum()),foreground_points_encoded=int((packed['labels']==1).sum()),
            bg_roundtrip_voxels=error,all_prompts_before_forward=True,decoder_calls=1,gt_used_for_roi=False))
        del packed,result,logits,probability,valid
        if grown==bounds:return pred,stages
        bounds=grown
    raise RuntimeError('bounded AutoExpand did not converge')


def load_model(model_name,out):
    model,helpers,audit=ORIGINAL_LOAD_MODEL(model_name,out)
    if model_name=='sammed3d':helpers=(SimpleNamespace(predict=predict_sammed),)+helpers[1:]
    audit=dict(audit,calibration_policy='all input prompts inside actual predicted ROI; no padded samples')
    return model,helpers,audit


def signature(out):
    result=ORIGINAL_SIGNATURE(out)
    result['valid_roi_correction']=dict(entry_sha256=e.sha(__file__),
        roi='initial FG plus all four BG; original context/AutoExpand',
        threshold='unchanged logit midpoint .35-.75 fallback.60; only genuinely predicted seeds',
        spatial_scale='unchanged training full-XY; not a resolution experiment')
    return result


def merge(out):
    for job in e.base.read(out/'jobs.json'):
        part=e.base.read(out/'jobs'/job['name']/'complete_metrics.json')['per_case_class']
        for row in part:
            for s in row['stages']:
                a=s['calibration']
                if (a['foreground_outside_predicted_roi'] or a['background_outside_predicted_roi']
                        or a['valid_background_count']!=4 or a['valid_foreground_count']<=0):
                    raise ValueError('invalid calibration coverage in result')
    ORIGINAL_MERGE(out)


def launch(out,gpus):
    e.verify(out)
    for model in e.ASSETS:
        if not (out/'jobs'/(model+'_smoke')/'complete_metrics.json').is_file():
            raise ValueError('three corrected real-task smoke checks required')
    e.base.write(out/'environment_launcher_audit.json',dict(entry_sha256=e.sha(__file__),
        interpreters=PYTHONS,worker_policy='one worker per sufficient-free-memory GPU; UUID checked'))
    original=subprocess.Popen
    def spawn(command,**kwargs):
        job=e.base.read(command[command.index('--job')+1]);command=list(command)
        command[0]=str(ROOT/PYTHONS[job['model']])
        return original(command,**kwargs)
    e.base.subprocess=SimpleNamespace(Popen=spawn,check_output=subprocess.check_output,STDOUT=subprocess.STDOUT)
    e.base.__file__=__file__;e.base.merge=merge;e.base.launch(out,gpus)


def install():
    e.signature=signature;e.load_model=load_model;e.view_predict=view_predict


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','worker','launch','merge','status'])
    p.add_argument('--out',default=OUT);p.add_argument('--job');p.add_argument('--gpus',default='0,2,3,4,5,6')
    a=p.parse_args();os.chdir(ROOT);out=ROOT/a.out;install()
    if a.mode=='prepare':e.prepare(out)
    elif a.mode=='worker':e.worker(e.base.read(a.job))
    elif a.mode=='merge':merge(out)
    elif a.mode=='launch':launch(out,list(map(int,a.gpus.split(','))))
    else:print(e.base.read(out/'status.json') if (out/'status.json').exists() else 'prepared/smoke')


if __name__=='__main__':main()
