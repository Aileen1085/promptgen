"""Isolated, resumable full-val SAM2/MedSAM2 physical-prompt baseline.

No PromptGen, semantic encoder, learned adapter, CT-cache writes or mask exports.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
METRICS = ('dice', 'mean_slice_dice', 'iou', 'precision', 'recall', 'specificity',
           'pred_gt_ratio', 'mean_slice_nsd_1mm', 'mean_slice_nsd_2mm', 'mean_slice_nsd_3mm')
COUNTS = dict(totalseg=4034, amos=1485, magic=35, msd_task07=40, msd_task10=20,
              msd_task08=40, parse2022=20, topcow2024_cta=277, lndb=24,
              msd_task06=13, covid19_20=24, kits23=58, lnq2023_lite=24)
PROTOCOL = dict(version='native_ct13_fg2_bg2_v1', fg_cap=2, bg_cap=2,
                point_spacing_mm=3., threshold_probability=.5, dense_prior='none',
                memory_mode='on', background='v9_training_vertical_bg_v1',
                metrics='complete_mask_axial_union_NSD1_2_3', xy_crop='prompt')
PREVIOUS = ROOT / 'output/ct13_full_val_v92e360_v102e490_20261005'
CACHE = ROOT / 'v10/.cache/totalseg_sam2_promptgen_v10_prompt_roi_full_v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.{}.tmp'.format(os.getpid()))
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(str(temp), str(path))


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''): result.update(block)
    return result.hexdigest()


def key(row):
    return str(row['source_name']), str(row['case_id']), int(row['global_class_id'])


def asset_spec(model):
    if model == 'sam2':
        return dict(model=model, code_root='sam2', config='configs/sam2.1/sam2.1_hiera_l.yaml',
                    checkpoint='sam2/checkpoints/sam2.1_hiera_large.pt', image_size=1024)
    if model == 'medsam2':
        official = 'third_party/MedSAM2_official_20261007'
        code = official if (ROOT / official / 'sam2/build_sam.py').is_file() else 'MedSAM2'
        return dict(model=model, code_root=code, config='configs/sam2.1_hiera_t512.yaml',
                    checkpoint='MedSAM2/checkpoints/MedSAM2_latest.pt', image_size=512)
    raise ValueError('unknown native model: ' + str(model))


def target_path(task):
    path = Path(task['label'])
    return path / (task['class_name'] + '.nii.gz') if task['source_name']=='totalseg' else path


def select_target(label, task):
    values = np.asarray(label)
    if task['source_name'] in ('totalseg', 'magic'): return values > 0
    return np.rint(values).astype(np.int32) == int(task['local_class_id'])


def assert_compatible_grid(image_affine, label_affine, shape, tolerance_mm=.1):
    # Some TotalSeg mask qforms round oblique direction cosines differently
    # from the CT sform. Compare physical corner displacement, not coefficients.
    import itertools
    corners = np.asarray(list(itertools.product(*[(0, n-1) for n in shape])), dtype=float)
    homogeneous = np.c_[corners, np.ones(len(corners))]
    delta = homogeneous @ (np.asarray(image_affine)-np.asarray(label_affine)).T
    error = float(np.linalg.norm(delta[:, :3], axis=1).max())
    if error > tolerance_mm:
        raise ValueError('label/CT physical grid mismatch: {:.6f} mm'.format(error))
    return error


def canonical_plane_slices(supports):
    candidates = []
    for support in supports:
        coords = np.argwhere(support)
        if not len(coords): raise ValueError('empty foreground support plane')
        candidates.append({axis:int(coords[0,axis]) for axis in (1,2)
                           if np.all(coords[:,axis]==coords[0,axis])})
    for coro, sag in ((0,1),(1,0)):
        if 2 in candidates[coro] and 1 in candidates[sag]:
            return candidates[coro][2], candidates[sag][1]
    raise ValueError('mapped supports are not two orthogonal canonical planes')


def case_parts(rows, count):
    groups = defaultdict(list)
    for row in rows: groups[key(row)[:2]].append(row)
    count = min(int(count), len(groups))
    if count < 1: raise ValueError('empty shard or invalid shard count')
    parts = [[] for _ in range(count)]
    for _, values in sorted(groups.items(), key=lambda pair:(-len(pair[1]),pair[0])):
        parts[min(range(count), key=lambda i:len(parts[i]))].extend(values)
    return [sorted(p,key=key) for p in parts]


def validate_rows(rows, expected):
    keys = [key(r) for r in rows]
    if len(keys)!=len(set(keys)) or set(keys)!={key(r) for r in expected}:
        raise ValueError('native result duplicate/missing/out-of-scope tasks')
    if any(m not in r or not math.isfinite(float(r[m])) for r in rows for m in METRICS):
        raise ValueError('native result nonfinite/missing metric')


def protocol_signature(job, weight_sha, code_sha):
    payload = dict(protocol=PROTOCOL, model=job['model'], tasks=job['tasks'],
                   weights=weight_sha, code=code_sha)
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def base_module():
    spec = importlib.util.spec_from_file_location('_native_ct13_base', ROOT/'tools/native_sam2_physical_eval.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class CaseLoader:
    """One readonly CT/label per current case; no per-class CT copies on disk."""
    def __init__(self, base):
        self.base=base; self.current=None; self.case=None; self.label_name=None; self.label=None

    def __call__(self, task, split, background_mode='scribble', background_cache_dir=''):
        source=task['source_name']; entry=next(r for r in split['datasets'][source]['val'] if r['case_id']==task['case_id'])
        # MAGIC cache is raw NRRD; official split entries point to model NIfTI.
        image_path=entry.get('image',entry.get('image_path'))
        label_path=target_path(dict(task,label=entry.get('label',entry.get('label_path'))))
        package_path=CACHE/'metadata/shared_adaptive_prompt_v1'/source/task['prompt_cache_name']
        with np.load(package_path,allow_pickle=False) as package:
            shape=tuple(int(v) for v in package['shape_dhw'])
            supports=[np.unpackbits(package[n].astype(np.uint8),count=int(np.prod(shape)),bitorder='little').reshape(shape).astype(bool)
                      for n in ('coronal_bits','sagittal_bits')]
            case_name=str(package['case_image_name'].item())
        current=(source,task['case_id'],str(image_path),case_name)
        if self.current!=current:
            image, target_affine=self.base.canonical(image_path)
            source_affine=target_affine.copy()
            if source=='magic':
                import SimpleITK as sitk
                reader=sitk.ImageFileReader();reader.SetFileName(task['image']);reader.ReadImageInformation()
                lps=np.eye(4);lps[:3,:3]=(np.asarray(reader.GetDirection()).reshape(3,3)@np.diag(reader.GetSpacing()))[:,[2,1,0]]
                lps[:3,3]=reader.GetOrigin();source_affine=np.diag([-1.,-1.,1.,1.])@lps
            case_path=CACHE/'_case_images_v1'/case_name
            hit=case_path.is_file()
            if hit:
                raw=np.load(case_path,mmap_mode='r');assert raw.shape==shape
                ct=self.base.reorient_grid(raw,source_affine,target_affine,tuple(image.shape[i] for i in (2,0,1)))
            else:
                ct=np.asarray(image.dataobj,dtype=np.float32).transpose(2,0,1)
            self.case=(ct,target_affine,source_affine,hit);self.current=current;self.label_name=None;self.label=None
        ct,target_affine,source_affine,hit=self.case
        if self.label_name!=str(label_path):
            image,affine=self.base.canonical(label_path)
            assert_compatible_grid(target_affine,affine,ct.shape)
            self.label=np.asarray(image.dataobj).transpose(2,0,1);self.label_name=str(label_path)
        target=select_target(self.label,task)
        assert target.shape==ct.shape and target.any(),'empty/misaligned manifest target'
        mapped=[self.base.reorient_grid(s,source_affine,target_affine,target.shape) for s in supports]
        foreground=mapped[0]|mapped[1]
        mapped_points,error=self.base.map_voxels(np.argwhere(supports[0]|supports[1]),source_affine,target_affine)
        direct=np.zeros(target.shape,bool);direct[tuple(np.rint(mapped_points).astype(int).T)]=True
        assert error<1e-3 and np.array_equal(direct,foreground),'physical prompt mapping failed'
        assert foreground.any() and target[foreground].all(),'foreground cache/target alignment failed'
        slices=canonical_plane_slices(mapped)
        stat=label_path.stat()
        signature=dict(version='v9_training_vertical_bg_v1',margin_ratio=.08,
                       foreground_package_sha256=sha(package_path),shape=list(target.shape),
                       label_signature=[str(label_path),stat.st_size,stat.st_mtime_ns])
        bg_sig=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
        cache=Path(background_cache_dir)/(bg_sig+'.npz');bg_hit=cache.is_file()
        if bg_hit:
            with np.load(cache,allow_pickle=False) as saved:
                assert str(saved['signature'].item())==bg_sig
                bg=saved['coords_dhw'].astype(np.int32)
        else:
            bg=self.base.training_background_scribble_coords(target,*slices)
            cache.parent.mkdir(parents=True,exist_ok=True)
            temporary=cache.with_name(cache.name+'.{}.tmp.npz'.format(os.getpid()))
            np.savez_compressed(temporary,coords_dhw=bg,signature=np.asarray(bg_sig));os.replace(str(temporary),str(cache))
        assert len(bg) and not target[tuple(bg.T)].any(),'invalid background simulation'
        spacing=np.linalg.norm(target_affine[:3,:3],axis=0)
        audit=dict(prompt_cache_sha256=sha(package_path),case_image_name=case_name,case_cache_hit=hit,
                   source_shape_dhw=shape,target_shape_dhw=list(target.shape),source_affine_dhw_ras=source_affine.tolist(),
                   target_affine_dhw_ras=target_affine.tolist(),physical_roundtrip_error_mm=error,
                   original_plane_voxels=[int(s.sum()) for s in supports],mapped_plane_voxels=[int(s.sum()) for s in mapped],
                   fg_voxels=int(foreground.sum()),fg_in_gt_fraction=float(target[foreground].mean()),
                   bg_points_dhw=bg.tolist(),bg_in_gt_count=0,spacing_dhw_mm=spacing.tolist(),gt_used_for_input=False,
                   background_mode='scribble',background_cache_signature=bg_sig,background_cache_hit=bg_hit,
                   canonical_prompt_slices=list(slices),gt_used_only_for_background_prompt_simulation=True)
        return ct,target,foreground,bg,spacing,audit


def prepare(output):
    output=Path(output); manifest=read(PREVIOUS/'manifest.json'); rows=manifest['tasks']
    assert len(rows)==6094 and dict(Counter(key(r)[0] for r in rows))==COUNTS
    assert len({key(t) for t in rows})==6094
    split=read(ROOT/'configs/ct13_v10_2_split_20260928.json')
    refs={key(t):t for p in (PREVIOUS/'prepared').glob('*/refs.json') for t in read(p)['per_case_class']}
    tasks=[]
    for raw in rows:
        task=dict(raw);source=task['source_name'];dataset=split['datasets'][source]
        assert task['case_id'] not in {t['case_id'] for t in dataset['train']}
        assert task['case_id'] in {t['case_id'] for t in dataset['val']}
        task['prompt_cache_name']=refs[key(task)]['prompt_cache_name']
        assert (CACHE/'metadata/shared_adaptive_prompt_v1'/source/task['prompt_cache_name']).is_file()
        tasks.append(task)
    if (output/'jobs.json').exists():
        assert read(output/'manifest.json')['tasks']==tasks
        return
    output.mkdir(parents=True,exist_ok=True)
    write(output/'manifest.json',dict(tasks=tasks,expected=6094,per_source_counts=COUNTS,train_overlap=0,protocol=PROTOCOL))
    jobs=[]
    for source in COUNTS:
        source_rows=[r for r in tasks if r['source_name']==source]
        count=40 if source=='totalseg' else (20 if source=='amos' else 4)
        parts=case_parts(source_rows,count)
        for index,part in enumerate(parts):
            for model in ('sam2','medsam2'):
                name='{}_{:03d}_{}'.format(source,index,model)
                job=dict(name=name,model=model,tasks=part,output=str(output/'jobs'/name),
                         background_cache_dir=str(ROOT/'output/native_sam2_amos_sparse_bg_pilot_20261007/background_scribbles_light'))
                path=output/'job_specs'/(name+'.json');write(path,job)
                jobs.append(dict(name=name,path=str(path),tasks=len(part),model=model,source=source))
    # Start rare domains first, retain whole-case CT locality; do not starve SAM2.
    jobs.sort(key=lambda j:(j['source']=='totalseg',j['source']=='amos',j['tasks'],j['name']))
    write(output/'jobs.json',dict(jobs=jobs,total_tasks=12188))
    print('PREPARED',len(jobs),'jobs;6094 tasks per model',flush=True)


def collect_attempts(job):
    rows=[];audits=[]
    for attempt in sorted(Path(job['output']).glob('attempt_*')):
        if (attempt/'all_metrics_live.json').exists():rows+=read(attempt/'all_metrics_live.json')['per_case_class']
        if (attempt/'exact_prompt_audit.json').exists():audits+=read(attempt/'exact_prompt_audit.json')['per_case_class']
    if len({key(r) for r in rows})!=len(rows):raise ValueError('duplicate task across retry attempts')
    if len({key(r) for r in audits})!=len(audits):raise ValueError('duplicate audit across retry attempts')
    return rows,audits


def worker(job_path,uuid):
    import torch
    job=read(job_path);out=Path(job['output']);out.mkdir(parents=True,exist_ok=True)
    assets=asset_spec(job['model']);code=ROOT/assets['code_root']
    sys.path.insert(0,str(code));import sam2.build_sam as builder
    assert Path(builder.__file__).resolve().is_relative_to(code.resolve()),'wrong SAM2 code imported'
    config=code/'sam2'/assets['config']
    code_files=[Path(__file__),ROOT/'tools/native_sam2_physical_eval.py',ROOT/'infer/native_sam2_physical_prompts.py',
                ROOT/'utils/metrics.py',Path(builder.__file__),code/'sam2/sam2_video_predictor.py',config]
    code_sha=hashlib.sha256(''.join(sha(p) for p in code_files).encode()).hexdigest()
    weight_sha=sha(ROOT/assets['checkpoint']);signature=protocol_signature(job,weight_sha,code_sha)
    if (out/'protocol.json').exists():assert read(out/'protocol.json')['signature']==signature,'unsafe changed-code resume'
    else:write(out/'protocol.json',dict(signature=signature,assets=assets,weight_sha256=weight_sha,code_sha256=code_sha,protocol=PROTOCOL))
    rows,audits=collect_attempts(job);done={key(r) for r in rows};audit_keys={key(a) for a in audits}
    assert done==audit_keys,'partial metric/audit write; recover explicitly before resume'
    pending=[t for t in job['tasks'] if key(t) not in done]
    if pending:
        attempt=out/('attempt_{:03d}'.format(len(list(out.glob('attempt_*')))+1))
        config_job=dict(tasks=pending,output=str(attempt),memory_mode='on',dense_prior='none',
                        background_mode='scribble',background_cache_dir=job['background_cache_dir'],
                        max_points=2,max_background_points=2,point_spacing_mm=3.)
        path=out/(attempt.name+'.json');write(path,config_job)
        base=base_module();base.load_task=CaseLoader(base);base.overlay=lambda *args:None
        original=builder.build_sam2_video_predictor
        def official(_config,checkpoint,**kwargs):
            model=original(assets['config'],checkpoint,**kwargs)
            assert model.image_size==assets['image_size'],'native configuration/image-size mismatch'
            return model
        builder.build_sam2_video_predictor=official
        args=base.parse_args(['--job',str(path),'--checkpoint',assets['checkpoint'],'--expected-uuid',uuid])
        base.run(args)
        rows,audits=collect_attempts(job)
    validate_rows(rows,job['tasks'])
    assert {key(a) for a in audits}=={key(t) for t in job['tasks']}
    for a in audits:
        assert a['complete_frame_prediction'] and not a['promptgen_loaded'] and not a['semantic_input']
        for stage in a['stages']:
            assert stage['foreground_supported_frames']==stage['foreground_point_frames']
            assert stage['background_support_frames']==stage['background_point_frames']
            assert stage['effective_memory_slots']==7
    summary={m:float(np.mean([r[m] for r in rows])) for m in METRICS}
    write(out/'exact_prompt_audit.json',dict(per_case_class=audits,signature=signature))
    write(out/'complete_metrics.json',dict(per_case_class=rows,summary=summary,model=job['model'],assets=assets,
                                          protocol=PROTOCOL,signature=signature,gpu_uuid=uuid))


def progress(output):
    output=Path(output);counts=Counter();sources=defaultdict(Counter);complete=0
    for meta in read(output/'jobs.json')['jobs']:
        job=read(meta['path']);file=Path(job['output'])/'complete_metrics.json'
        if file.exists():rows=read(file)['per_case_class'];complete+=1
        else:rows,_=collect_attempts(job)
        counts[job['model']]+=len(rows)
        for r in rows:sources[job['model']][r['source_name']]+=1
    return dict(completed_jobs=complete,model_tasks=dict(counts),per_source={k:dict(v) for k,v in sources.items()})


def merge(output):
    output=Path(output);jobs=read(output/'jobs.json')['jobs'];manifest=read(output/'manifest.json')['tasks'];result={}
    for model in ('sam2','medsam2'):
        rows=[]
        for meta in jobs:
            if meta['model']==model: rows+=read(Path(read(meta['path'])['output'])/'complete_metrics.json')['per_case_class']
        validate_rows(rows,manifest);assert dict(Counter(r['source_name'] for r in rows))==COUNTS
        result[model]=dict(per_case_class=rows,summary={m:float(np.mean([r[m] for r in rows])) for m in METRICS},
                          per_source={s:dict(count=COUNTS[s],**{m:float(np.mean([r[m] for r in rows if r['source_name']==s])) for m in METRICS}) for s in COUNTS})
    write(output/'merged_metrics.json',dict(methods=result,protocol=PROTOCOL,task_count_per_model=6094))


def schedule(output,gpu_indices):
    import fcntl
    output=Path(output);lock=(output/'scheduler.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    metas=read(output/'jobs.json')['jobs'];active={};failures=Counter()
    while True:
        done={m['name'] for m in metas if (Path(read(m['path'])['output'])/'complete_metrics.json').exists()}
        for name,item in list(active.items()):
            rc=item['process'].poll()
            if rc is not None:
                item['log'].close();del active[name]
                if rc!=0:
                    failures[name]+=1
                    print('WORKER_FAILED',name,'returncode',rc,flush=True)
                    log=Path(item['log_path']).read_text()
                    oom=any(term in log for term in ('CUDA out of memory','OutOfMemoryError'))
                    if not oom or failures[name]>=3:
                        write(output/'failure.json',dict(job=name,returncode=rc,oom=oom,log=item['log_path']))
                        # Let other already-running jobs finish; do not kill any process.
                        for running in active.values():running['process'].wait();running['log'].close()
                        raise RuntimeError('Native worker failed: '+name)
        pending=[m for m in metas if m['name'] not in done and m['name'] not in active]
        if not pending and not active:
            merge(output);write(output/'status.json',dict(state='complete',**progress(output)));break
        inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
        busy={x['uuid'] for x in active.values()}
        for line in inventory.strip().splitlines():
            index,uuid,free=[s.strip() for s in line.split(',')]
            if int(index) not in gpu_indices or uuid in busy or int(free)<24000 or not pending:continue
            meta=pending.pop(0);log_path=output/'logs'/(meta['name']+'.try{}.log'.format(failures[meta['name']]))
            log_path.parent.mkdir(exist_ok=True);log=log_path.open('w')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='2')
            process=subprocess.Popen([sys.executable,str(Path(__file__)), '--worker','--job',meta['path'],'--uuid',uuid],cwd=str(ROOT),env=env,stdout=log,stderr=subprocess.STDOUT)
            active[meta['name']]=dict(process=process,uuid=uuid,log=log,log_path=str(log_path))
            print('START',meta['name'],'GPU',index,uuid,'PID',process.pid,'tasks',meta['tasks'],flush=True)
        write(output/'status.json',dict(state='running',time=time.strftime('%Y-%m-%d %H:%M:%S'),
              scheduler_pid=os.getpid(),active={k:dict(pid=v['process'].pid,uuid=v['uuid']) for k,v in active.items()},**progress(output)))
        time.sleep(20)


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--worker',action='store_true')
    parser.add_argument('--schedule',action='store_true');parser.add_argument('--merge',action='store_true')
    parser.add_argument('--status',action='store_true');parser.add_argument('--job');parser.add_argument('--uuid')
    parser.add_argument('--output',default='output/native_sam2_medsam2_ct13_full6094_20261007')
    parser.add_argument('--gpus',default='0,2,3,4,5,6')
    return parser.parse_args(argv)


if __name__=='__main__':
    args=parse_args();os.chdir(ROOT)
    if args.prepare:prepare(ROOT/args.output)
    elif args.worker:worker(args.job,args.uuid)
    elif args.schedule:schedule(ROOT/args.output,{int(i) for i in args.gpus.split(',')})
    elif args.merge:merge(ROOT/args.output)
    elif args.status:print(json.dumps(progress(ROOT/args.output),indent=2))
    else:raise SystemExit('Choose prepare/worker/schedule/merge/status')
