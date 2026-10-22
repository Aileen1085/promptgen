"""Fixed AMOS-adapted PromptGen backbones transferred to all 35 MAGIC val tasks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import MethodType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import ct13_full_validation_eval as base

OUT = 'output/magic_promptgen_three_backbones35_20261022_planefix'
REFERENCE = 'output/native_sam2_medsam2_ct13_full6094_20261007_gridfix'
BG_CACHE = 'output/native_sam2_amos_sparse_bg_pilot_20261007/background_scribbles_light'
ASSETS = {
    'sammed3d': ('sammed3d/output/amos_turbo_20261009/epoch230.pth', 230),
    'vista3d': ('vista3d/output/amos_single_e40_e80_plateau_20261003/epoch065.pth', 65),
    'segvol': ('segvol-3d/output/amos_fpn_e40_plateau_20261003/epoch070_fpn_best.pth', 70),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''): h.update(block)
    return h.hexdigest()


def validate_scope(rows, split):
    keys = [base.key(r) for r in rows]
    source = split['datasets']['magic']
    val = {r['case_id'] for r in source['val']}
    train = {r['case_id'] for r in source['train']}
    if (len(keys) != 35 or len(set(keys)) != 35 or len({k[1] for k in keys}) != 35
            or any(k[0] != 'magic' or k[1] not in val or k[1] in train for k in keys)):
        raise ValueError('requires exact 35 unique held-out MAGIC cases/tasks, train overlap0')


def map_background(points, source_affine, target_affine):
    import numpy as np
    p = np.asarray(points, float).reshape(-1, 3)
    transform = np.linalg.solve(np.asarray(target_affine), np.asarray(source_affine))
    mapped = (np.c_[p, np.ones(len(p))] @ transform.T)[:, :3]
    if not np.isfinite(mapped).all() or np.abs(mapped-np.rint(mapped)).max() > 1e-4:
        raise ValueError('nonintegral physical background mapping')
    return np.rint(mapped).astype(np.int32)


def mapped_plane_metadata(raw_slices, source_affine, target_affine):
    """Plane normals come from cached metadata, never inferred from a thin line."""
    import numpy as np
    transform=np.linalg.solve(np.asarray(target_affine),np.asarray(source_affine))
    mapped={}
    for original,(axis,index) in enumerate(zip((2,1),raw_slices)):
        normal=int(np.argmax(np.abs(transform[:3,axis])))
        row=transform[normal,:3].copy();scale=row[axis];row[axis]=0
        coordinate=scale*int(index)+transform[normal,3]
        if (normal not in (1,2) or not np.allclose(row,0,atol=1e-5)
                or abs(abs(scale)-1)>1e-5 or abs(coordinate-round(coordinate))>1e-4):
            raise ValueError('not an axis-preserving physical plane transformation')
        if normal in mapped:raise ValueError('orthogonal planes collapsed')
        mapped[normal]=(int(round(coordinate)),original)
    return (mapped[2][0],mapped[1][0]),(mapped[2][1],mapped[1][1])


def validate_background(bg, truth, slices):
    import numpy as np
    bg = np.asarray(bg)
    if (bg.shape != (4, 3) or (bg < 0).any() or (bg >= truth.shape).any()
            or truth[tuple(bg.T)].any()
            or (bg[:, 2] == slices[0]).sum() < 2 or (bg[:, 1] == slices[1]).sum() < 2):
        raise ValueError('four original BG points/two orthogonal planes/alignment required')


def signature(out):
    files = [Path(__file__), ROOT/base.SPLIT, ROOT/'utils/metrics.py',
             ROOT/'tools/native_ct13_eval.py', ROOT/'tools/native_sam2_physical_eval.py',
             ROOT/'tools/sammed3d_promptgen_amos_eval.py', ROOT/'tools/segvol_promptgen_amos_eval.py',
             ROOT/'tools/vista_promptgen_amos_full.py', ROOT/'v9_2_extended_protocol.py',
             ROOT/'v10/v10_2_data.py', ROOT/'v10/v10_data.py', ROOT/'v9_2_ct_channels.py',
             ROOT/'v10/infer/infer_amos_sam2_promptgen_v10.py']
    for folder in ('sammed3d', 'vista3d', 'segvol-3d', 'model', 'v10/model'):
        files += sorted((ROOT/folder).glob('*.py'))
    from tools import sammed3d_promptgen_amos_eval as sm
    from tools import vista_promptgen_amos_full as vi
    from tools import segvol_promptgen_amos_eval as sg
    weights = [sm.WEIGHT, sm.CORE, sm.DONOR, vi.CHECKPOINT, vi.CONSTRUCTION, vi.CORE,
               str(sg.WEIGHT.relative_to(ROOT)), str(sg.CONSTRUCTION.relative_to(ROOT)), str(sg.OFFICIAL.relative_to(ROOT))]
    result=dict(code={str(p.relative_to(ROOT)):sha(p) for p in files},
        weights={p:sha(ROOT/p) for p in weights}, manifest=sha(out/'manifest.json'),
        prompts={t['prompt_cache_name']:sha(ROOT/base.CACHE/'metadata/shared_adaptive_prompt_v1/magic'/t['prompt_cache_name'])
                 for t in base.read(out/'manifest.json')['tasks']},
        protocol='MAGIC35 rawNRRD-to-model once; cached plane normals; FGadaptive+4BGpoint; expanded semantic; dynamic.35-.75 fallback.60; fullmask10')
    if (out/'reuse_audit.json').exists():
        for path,expected in base.read(out/'reuse_audit.json')['source_files'].items():
            if sha(ROOT/path)!=expected:raise ValueError('readonly reuse source changed')
        result['reuse']={name:sha(out/name) for name in ('reuse_audit.json','reused_metrics.json')}
    return result


def verify(out):
    if signature(out) != base.read(out/'protocol.json'):
        raise ValueError('changed weights/code/prompt signatures; refusing mixed results')


def prepare(out, reuse_from=None):
    if out.exists(): raise FileExistsError('new output required; preserve historical outputs')
    rows = sorted([r for r in base.read(ROOT/REFERENCE/'manifest.json')['tasks'] if r['source_name']=='magic'], key=base.key)
    validate_scope(rows, base.read(ROOT/base.SPLIT))
    base.write(out/'manifest.json', base.manifest_payload(rows))
    reused={model:[] for model in ASSETS}
    if reuse_from:
        import numpy as np
        old=ROOT/reuse_from;oldp=base.read(old/'protocol.json');source_files={}
        if oldp['code']['tools/magic_promptgen_three_backbones.py']!='eaea564ed41cb129a2cd46ce41103bf44315a7c08243fdf05bd3d10c836f34d7':
            raise ValueError('unapproved predecessor entry signature')
        for name,value in oldp['code'].items():
            if name!='tools/magic_promptgen_three_backbones.py' and sha(ROOT/name)!=value:raise ValueError('old dependency changed')
        for name,value in oldp['weights'].items():
            if sha(ROOT/name)!=value:raise ValueError('old checkpoint changed')
        if base.read(old/'manifest.json')['tasks']!=rows:raise ValueError('reuse task scope mismatch')
        from tools import native_ct13_eval as native
        physical=native.base_module()
        for j in base.read(old/'jobs.json'):
            run=old/'jobs'/j['name'];path=run/'complete_metrics.json'
            if not path.exists():path=run/'all_metrics_live.json'
            if not path.exists():continue
            model_audit=run/'model_audit.json'
            if not base.read(model_audit)['strict_effective_modules_exact']:raise ValueError('old model restore audit failed')
            source_files[str(path.relative_to(ROOT))]=sha(path)
            source_files[str(model_audit.relative_to(ROOT))]=sha(model_audit)
            part=base.read(path)['per_case_class']
            expected=base.read(Path(j['prepared'])/'manifest.json')['tasks']
            base.validate_rows(part,expected[:len(part)])
            for row in part:
                a=row['prompt_audit'];p=ROOT/base.CACHE/'metadata/shared_adaptive_prompt_v1/magic'/row['prompt_cache_name']
                if sha(p)!=a['prompt_cache_sha256'] or sha(p)!=oldp['prompts'][row['prompt_cache_name']]:raise ValueError('reused prompt changed')
                with np.load(p) as z:
                    slices,order=mapped_plane_metadata(z['slices'],a['source_affine_dhw_ras'],a['target_affine_dhw_ras'])
                    shape=tuple(map(int,z['shape_dhw']))
                    supports=[np.unpackbits(z[n],count=int(np.prod(shape)),bitorder='little').reshape(shape).astype(bool)
                              for n in ('coronal_bits','sagittal_bits')]
                supports=[physical.reorient_grid(s,np.asarray(a['source_affine_dhw_ras']),np.asarray(a['target_affine_dhw_ras']),tuple(a['target_shape_dhw'])) for s in supports]
                old_order=(0,1) if np.all(np.argwhere(supports[0])[:,2]==slices[0]) else (1,0)
                if tuple(a['plane_slices'])!=slices or any(not np.array_equal(supports[x],supports[y]) for x,y in zip(old_order,order)):
                    continue  # Wrong/ambiguous plane would be re-predicted, never silently reused.
                reused[j['model']].append(row)
        source_files[str((old/'protocol.json').relative_to(ROOT))]=sha(old/'protocol.json')
        base.write(out/'reused_metrics.json',reused)
        base.write(out/'reuse_audit.json',dict(source_files=source_files,
            model_counts={k:len(v) for k,v in reused.items()},criterion='exact plane indices, supports, prompt and checkpoint SHA; old outputs readonly'))
    jobs = []
    # First case is a complete, independently retained real-data smoke for each model.
    for model in ASSETS:
        done={base.key(r) for r in reused[model]};pending=[r for r in rows if base.key(r) not in done]
        smoke=[r for r in pending if r['case_id']=='liver_cyst/51'] or pending[:1]
        rest=[r for r in pending if base.key(r) not in {base.key(t) for t in smoke}]
        parts=[smoke]+[base.case_shard(rest,4,i) for i in range(4)] if rest else [smoke]
        for i, part in enumerate(parts):
            if not part:continue
            name = model+('_smoke' if i==0 else '_s%02d'%i)
            prepared = out/'prepared'/name
            base.write(prepared/'manifest.json', base.manifest_payload(part))
            jobs.append(dict(name=name, model=model, source='magic', prepared=str(prepared),
                             task_count=len(part), frame_batch=1, attempt=0))
    base.write(out/'jobs.json', jobs)
    base.write(out/'protocol.json', signature(out))
    print('PREPARED 3x35 tasks',len(jobs),'remaining jobs; reused',{m:len(v) for m,v in reused.items()},flush=True)


class MagicLoader:
    def __init__(self):
        from tools import native_ct13_eval as native
        self.native = native; self.physical = native.base_module()
        def missing(*a, **k): raise FileNotFoundError('readonly historical BG cache missing')
        self.physical.training_background_scribble_coords = missing
        self.loader = native.CaseLoader(self.physical)

    def __call__(self, task, split):
        import numpy as np
        ct, gt, fg, _unused_bg_scribble, spacing, audit = self.loader(task, split, 'scribble', str(ROOT/BG_CACHE))
        path = ROOT/base.CACHE/'metadata/shared_adaptive_prompt_v1/magic'/task['prompt_cache_name']
        source = np.asarray(audit['source_affine_dhw_ras']); target = np.asarray(audit['target_affine_dhw_ras'])
        with np.load(path, allow_pickle=False) as p:
            bg = map_background(p['background_points_dhw'], source, target)
            shape = tuple(map(int, p['shape_dhw']))
            supports = [np.unpackbits(p[n], count=int(np.prod(shape)), bitorder='little').reshape(shape).astype(bool)
                        for n in ('coronal_bits', 'sagittal_bits')]
            slices,order=mapped_plane_metadata(p['slices'],source,target)
        supports = [self.physical.reorient_grid(s, source, target, gt.shape) for s in supports]
        supports=[supports[i] for i in order]
        if not (np.all(np.argwhere(supports[0])[:,2]==slices[0]) and np.all(np.argwhere(supports[1])[:,1]==slices[1])):
            raise ValueError('mapped support/plane metadata mismatch')
        if not np.array_equal(fg, supports[0] | supports[1]): raise ValueError('FG mapping mismatch')
        validate_background(bg, gt, slices)
        audit.update(background_mode='point', bg_points_dhw=bg.tolist(), bg_in_gt_count=0,
                     original_four_bg_points=True, physical_mapping_applications=1,
                     gt_used_for_roi=False, class_name=task['class_name'], plane_slices=list(slices),
                     canonical_prompt_slices=list(slices),plane_source='cached raw plane metadata')
        return ct, gt, fg, bg, spacing, slices, supports, audit


def load_model(model_name, out):
    import torch
    if model_name == 'sammed3d':
        from tools import sammed3d_promptgen_amos_eval as sm
        model, data, training, geometry, native, audit = sm.load_model(out)
        return model, (sm, data, training, geometry), audit
    if model_name == 'segvol':
        from tools import segvol_promptgen_amos_eval as sg
        train, training = sg.training_modules(); model = sg.load_model(train)
        fields = [('prompt_generator', model.promptgen), ('segvol_feature_bridge', model.feature_bridge),
                  ('segvol_prompt_adapter', model.prompt_adapter)]
    else:
        from tools import vista_promptgen_amos_full as vi
        sys.path.insert(0, str(ROOT/'v10')); sys.path.insert(0, str(ROOT/'vista3d'))
        import train
        import training
        args = train.parser().parse_args(['validate'])
        args.device='cuda:0'; args.vista_source=ROOT/'vista3d/official_source/vista3d'
        args.vista_checkpoint=ROOT/vi.CORE; args.promptgen_checkpoint=ROOT/vi.CONSTRUCTION
        args.depth=96; args.stride=48; args.vista_hw=96
        state=torch.load(ROOT/ASSETS[model_name][0], map_location='cpu', weights_only=True)
        train.require_matching_feature_mode(state, 'single')
        train.require_matching_vista_checkpoint(state, sha(ROOT/vi.CORE))
        model=train._build_model(args)
        fields=[('prompt_generator', model.promptgen), ('vista_feature_bridge', model.feature_bridge),
                ('vista_prompt_adapter', model.prompt_adapter)]
    state=torch.load(ROOT/ASSETS[model_name][0], map_location='cpu', weights_only=True)
    if state['epoch'] != ASSETS[model_name][1]: raise ValueError('checkpoint epoch mismatch')
    for name, module in fields:
        module.load_state_dict(state[name], strict=True)
        if not all(torch.equal(v.detach().cpu(), state[name][k]) for k,v in module.state_dict().items()):
            raise ValueError('strict tensor restoration mismatch: '+name)
    source_epoch = getattr(model.promptgen, 'vista_source_epoch' if model_name=='vista3d' else 'segvol_source_epoch')
    model.promptgen.set_morphology_routing_epoch(source_epoch+state['epoch'])
    model.promptgen._v10_forced_background_mode='point'
    model.promptgen._one_point_per_group=MethodType(lambda self,x,fallback=None:(x>0).to(x.dtype), model.promptgen)
    model.requires_grad_(False); model.eval()
    return model, training, dict(epoch=state['epoch'], strict_effective_modules_exact=True, all_four_background_points=True)


def make_dataset(training, split):
    # Reuse the unmodified backbone's exact training namespace/reader, routing only
    # its source factory to MAGIC. Do not masquerade MAGIC as AMOS in the data reader.
    import v10_2_data
    entries=split['datasets']['magic']['val']
    names={int(r['global_class_id']):r['class_name'] for r in split['class_catalog'] if r['dataset']=='magic'}
    cases=SimpleNamespace(val=[SimpleNamespace(**r) for r in entries], train=[], class_names=names)
    original=v10_2_data.make_source_dataset
    def magic_source(_source, rows, args, total_factory, **kw):
        args.v10_2_class_catalog=split['class_catalog']
        return original('magic', rows, args, total_factory, **kw)
    v10_2_data.make_source_dataset=magic_source
    try: dataset=training.make_amos_dataset(cases, ROOT/base.CACHE, ROOT/'.cache/prompt_planes', training=False)
    finally: v10_2_data.make_source_dataset=original
    return dataset, {r['case_id']:i for i,r in enumerate(entries)}, {r['case_id']:r for r in entries}


def view_predict(model_name, model, training, dataset, index, entry, task, raw, text, physical):
    import numpy as np
    import torch
    from tools import segvol_promptgen_amos_eval as sg
    from v10_data import SAM2CachedTotalSegV10Dataset
    from v10.infer.infer_amos_sam2_promptgen_v10 import postprocess_v10_logits_from_prompts, expand_prompt_roi_bounds, prompt_connected_z_boundary_contacts
    ct,gt,fg,bg,spacing,slices,supports,audit=raw
    groups,planes,coords=sg.raw_prompt_groups(fg,bg,slices,supports)
    square=max(ct.shape[1:]);top=(square-ct.shape[1])//2;left=(square-ct.shape[2])//2
    groups=[SAM2CachedTotalSegV10Dataset._resize_prompt_group_preserve(g,256,pad)
            for g,pad in zip(groups,[(top,square-ct.shape[1]-top),(left,square-ct.shape[2]-left)])]
    bundle=dict(groups=torch.from_numpy(np.stack(groups).astype(np.float32)), plane_ids=torch.tensor(planes),
                slice_coords=torch.tensor(coords,dtype=torch.float32), canonical_depth=ct.shape[0])
    initial=sg.initial_z_bounds(fg);bounds=initial;stages=[]
    for iteration in range(8):
        sg.bind_readonly(dataset,ct,gt,bundle,bounds,int(task['global_class_id']))
        wrapper=dataset
        while True:
            wrapper._atomic_save=lambda *a,**k:None
            if 'dataset' not in vars(wrapper):break
            wrapper=wrapper.dataset
        kw=dict(vista_hw=96,prompt_hw=model.promptgen.work_size) if model_name=='vista3d' else {}
        view=training.read_case_view(dataset,index,int(task['global_class_id']),Path(entry['image']),**kw)
        if not np.array_equal(np.asarray(view.frame_indices),np.arange(*bounds)):
            raise ValueError('complete ROI frames mismatch')
        view.bundle['v10_forced_background_mode']='point'
        depth,stride=(96,48) if model_name=='vista3d' else (32,16)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            prob=training.predict_complete_roi(model,view,text,depth=depth,stride=stride,device='cuda:0')
        if not torch.isfinite(prob).all():raise FloatingPointError('nonfinite probability')
        bgmask=np.zeros(ct.shape,bool);bgmask[tuple(bg.T)]=True
        logits=torch.logit(prob.float().clamp(1e-6,1-1e-6))[:,None]
        pred,threshold=postprocess_v10_logits_from_prompts(logits,fg,bgmask,
            fallback=.60,minimum=.35,maximum=.75,min_logit_separation=.50)
        start,end=bounds;touch=prompt_connected_z_boundary_contacts(pred[start:end],fg[start:end],band_frames=2)
        grown=expand_prompt_roi_bounds(current_bounds=bounds,initial_bounds=initial,available_bounds=(0,len(ct)),
            step_ratio=.50,max_expand_ratio=1.,expand_start=touch[0],expand_end=touch[1])
        stages.append(dict(z_bounds=list(bounds),depth=depth,stride=stride,complete_roi_frames=True,
            threshold=threshold,background_points_encoded=4,all_prompts_before_forward=True,gt_used_for_roi=False))
        del view,prob,logits,bgmask
        if grown==bounds:return pred,stages
        bounds=grown
    raise RuntimeError('bounded AutoExpand failed to converge')


def worker(job):
    os.environ['V9_SHARED_CACHE_WRITE_POLICY']='read_only'
    import numpy as np
    import torch
    torch.set_num_threads(2)
    out=Path(job['run']).parents[1];verify(out)
    if (str(torch.cuda.get_device_properties(0).uuid).replace('GPU-','')!=job['gpu_uuid'].replace('GPU-','')
            or torch.ones(1,device='cuda:0').item()!=1):raise ValueError('physical GPU mismatch')
    run=Path(job['run']);rows_expected=base.read(Path(job['prepared'])/'manifest.json')['tasks']
    loader=MagicLoader();split=base.read(ROOT/base.SPLIT)
    from v9_2_extended_protocol import conditioning_text_from_metadata
    text={int(r['global_class_id']):conditioning_text_from_metadata(r) for r in split['class_catalog']}
    model_name=job['model'];model,helpers,audit=load_model(model_name,out)
    base.write(run/'model_audit.json',audit)
    if model_name!='sammed3d':dataset,indices,entries=make_dataset(helpers,split)
    scorer=loader.physical.metric_module();rows=[]
    if (run/'all_metrics_live.json').exists():raise FileExistsError('partial results need explicit recovery; refusing overwrite')
    with torch.inference_mode():
        for task in rows_expected:
            start=time.time();raw=loader(task,split);ct,gt,fg,bg,spacing,slices,supports,pa=raw
            semantic=text[int(task['global_class_id'])];pa['class_name']=semantic
            if model_name=='sammed3d':
                sm,data,training,geometry=helpers
                case=data.RawCase(ct,gt,fg,bg,spacing,(2,1),slices,pa)
                pred,stages=sm.predict(model,case,data,training,geometry)
            else:
                pred,stages=view_predict(model_name,model,helpers,dataset,indices[task['case_id']],
                    entries[task['case_id']],task,raw,semantic,loader.physical)
            values=scorer.binary_prompt_metrics_from_masks(torch.from_numpy(np.asarray(pred)),torch.from_numpy(gt),spacing_dhw=spacing)
            rows.append(dict(task,**{m:float(values[m]) for m in base.METRICS},prompt_audit=pa,
                stages=stages,semantic_text=semantic,epoch=ASSETS[model_name][1],seconds=time.time()-start))
            base.validate_rows(rows,rows_expected[:len(rows)])
            base.write(run/'all_metrics_live.json',dict(per_case_class=rows))
            base.write(run/'exact_prompt_audit.json',dict(per_case_class=[dict(task_key=list(base.key(r)),
                prompt_audit=r['prompt_audit'],stages=r['stages']) for r in rows]))
            print('MAGIC_PG',model_name,len(rows),'/',len(rows_expected),base.key(task),'dice',rows[-1]['dice'],flush=True)
            del raw,pred,gt,ct,fg,supports
    base.validate_rows(rows,rows_expected)
    base.write(run/'complete_metrics.json',dict(per_case_class=rows,model=model_name,model_audit=audit))


def merge(out):
    verify(out);expected=base.read(out/'manifest.json')['tasks'];result={}
    for model in ASSETS:
        rows=base.read(out/'reused_metrics.json')[model][:] if (out/'reused_metrics.json').exists() else []
        for job in base.read(out/'jobs.json'):
            if job['model']!=model:continue
            p=base.read(out/'jobs'/job['name']/'complete_metrics.json')
            if not p['model_audit']['strict_effective_modules_exact']:raise ValueError('checkpoint audit failed')
            part=p['per_case_class'];base.validate_rows(part,base.read(Path(job['prepared'])/'manifest.json')['tasks']);rows+=part
        base.validate_rows(rows,expected);validate_scope(rows,base.read(ROOT/base.SPLIT))
        for row in rows:
            a=row['prompt_audit'];path=ROOT/base.CACHE/'metadata/shared_adaptive_prompt_v1/magic'/row['prompt_cache_name']
            if (a['physical_mapping_applications']!=1 or a['bg_in_gt_count']!=0 or a['fg_in_gt_fraction']!=1
                    or a['prompt_cache_sha256']!=sha(path) or len(a['bg_points_dhw'])!=4):raise ValueError('prompt mapping/signature audit failed')
            for s in row['stages']:
                if s['gt_used_for_roi'] or s['background_points_encoded']!=4:raise ValueError('ROI/background audit failed')
        result[model]=dict(task_count=35,weight=ASSETS[model][0],epoch=ASSETS[model][1],per_case_class=rows,
            summary={m:sum(r[m] for r in rows)/35 for m in base.METRICS})
    base.write(out/'merged_metrics.json',result);print('MERGED105 MAGIC tasks / ten finite metrics',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','worker','launch','merge','status'])
    p.add_argument('--out',default=OUT);p.add_argument('--job');p.add_argument('--gpus',default='0,3,4,5,6');p.add_argument('--reuse')
    a=p.parse_args();os.chdir(ROOT);out=ROOT/a.out
    if a.mode=='prepare':prepare(out,a.reuse)
    elif a.mode=='worker':worker(base.read(a.job))
    elif a.mode=='merge':merge(out)
    elif a.mode=='launch':
        verify(out)
        for m in ASSETS:
            if not (out/'jobs'/(m+'_smoke')/'complete_metrics.json').exists():
                raise ValueError('three model real-task smoke checks must finish before full launch')
        base.__file__=__file__;base.merge=merge;base.launch(out,list(map(int,a.gpus.split(','))))
    else:
        print(json.dumps(base.read(out/'status.json') if (out/'status.json').exists() else {'state':'prepared/smoke'},indent=2))


if __name__=='__main__':main()
