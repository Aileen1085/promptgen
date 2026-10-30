"""Fresh fixed-floor AMOS prompt ablations, immutable class commits, and GPU queue.

Only JSON metadata is written. Native runtime owns inference and audited metrics;
this controller never imports a Bone runtime or creates CT/tensor caches.
"""
from __future__ import annotations
import argparse
import ast
from collections import defaultdict
from contextlib import redirect_stderr, redirect_stdout
import copy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import types

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
OUT='output/amos_v102_e490_floor070_prompt_ablations_20261030'
TMUX='amos_v102_floor070_prompt_ablations'
PIPELINE='output/amos_prompt_ablations_pipeline_20261030.log'
HISTORY='output/ct13_full_val_v92e360_v102e490_20261005'
OLD='output/v102_dynamic_floor_grid_amos1485_magic35_e490_20261023_nativeargsfix/merged_metrics.json'
OLD_SHA='7315d43040aff256851bb3be0fbb0a02774bb11ee56497aeb1c4326fcae09e8e'
TRAINING='configs/ct13_v10_2_split_20260928.json'
TRAINING_SHA='99b2eee8b4ae549537922614b0570665c6fa90387e21e04c69c04ed0652de022'
WEIGHTS='v10/output/v10_2_ct13_e470_zero_delta_e510_20261001/20261001_215853/epoch490.pth'
WEIGHT_SHA='2f8dc28e1c77d6196d42675eb271c624b9b94c920699119c4830e44330d92358'
CACHE='v10/.cache/totalseg_sam2_promptgen_v10_prompt_roi_full_v1'
NATIVE='v10/infer/infer_eight_source_adaptive_v10_2.py'
VARIANTS=('baseline','sagittal_only','coronal_only','fg_box_bg_point','fg_scribble_bg_scribble','fg_box_bg_scribble')
ALLOWED=(2,4,5,6)
METRICS=('dice','mean_slice_dice','iou','precision','recall','specificity','pred_gt_ratio',
         'mean_slice_nsd_1mm','mean_slice_nsd_2mm','mean_slice_nsd_3mm')
FALLBACK=frozenset(('fallback_missing_prompt_logits','fallback_insufficient_separation'))
SOURCE_SEEDS=('tools/amos_prompt_ablation_eval.py','tools/amos_prompt_ablation_runtime.py',
              'tools/amos_prompt_ablation_prompts.py',NATIVE,'utils/metrics.py',
              'v10/v10_2_data.py','v10/v10_data.py','v9_2_adaptive_scribble.py','tools/bone_fast_metrics.py')
_CACHE_PATH_METHODS={}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def atomic_new(path,value):
    """Exclusive publication; a failed writer never overwrites earlier evidence."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp_'+str(os.getpid())+'_'+str(time.time_ns()))
    with temp.open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    try:os.link(temp,path)
    finally:temp.unlink()  # one explicitly named temporary metadata file


def write_status(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp_'+str(os.getpid()))
    with temp.open('w',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def key(row):
    return str(row.get('source_name',row.get('dataset'))),str(row['case_id']),int(row.get('global_class_id',row.get('class_id')))


def validate_scope(tasks,split=None,require_labels=True):
    keys=[key(t) for t in tasks]
    if (len(keys)!=1485 or len(set(keys))!=1485 or {k[0] for k in keys}!={'amos'}
            or len({k[1] for k in keys})!=100 or len({k[2] for k in keys})!=15):
        raise ValueError('fixed AMOS100CT/15class/1485 original task keys required')
    if require_labels:
        by_gid={}
        for t in tasks:
            local=int(t['local_class_id'])
            if not 1<=local<=15:raise ValueError('original AMOS local label required')
            ident=key(t)[2]
            if ident in by_gid and by_gid[ident]!=local:raise ValueError('global/local label identity changed')
            by_gid[ident]=local
        if set(by_gid.values())!=set(range(1,16)):raise ValueError('all15 original local labels required')
    if split is not None:
        ds=split['datasets']['amos'];val=[str(r['case_id']) for r in ds['val']];train={str(r['case_id']) for r in ds['train']}
        if len(val)!=100 or len(set(val))!=100 or set(val)!={k[1] for k in keys} or set(val)&train:
            raise ValueError('all and only original100 held-out AMOS CTs required')


def case_jobs(tasks,strict=True):
    if strict:validate_scope(tasks)
    grouped=defaultdict(list);seen=set()
    for t in sorted(tasks,key=key):
        if key(t) in seen:raise ValueError('duplicate task')
        seen.add(key(t));grouped[key(t)[:2]].append(copy.deepcopy(t))
    return [dict(name='amos_'+case,model='v10_2',case_id=case,source='amos',control=False,
                 variants=list(VARIANTS),pending=rows,task_count=len(rows),frame_batch=4,
                 minimum_free_mib=30720,attempt=0,oom_retries=0,not_before=0.,avoid_gpus={})
            for (_,case),rows in sorted(grouped.items())]


def control_jobs(tasks):
    j=case_jobs([sorted(tasks,key=key)[0]],strict=False)[0]
    return [dict(j,name='amos_native_control',control=True)]


def retry(job,code,text,now):
    oom=code!=75 and ('cuda out of memory' in text.lower() or 'OutOfMemoryError' in text)
    if code==0 or (code!=75 and not oom):return None
    value=copy.deepcopy(job);value.update(attempt=job['attempt']+1,not_before=now+20)
    for name in ('run','spec','log','gpu_uuid','actual_frame_batch','control_gate_sha256'):value.pop(name,None)
    if oom:
        value['oom_retries']=job['oom_retries']+1
        if job['frame_batch']>1:value['frame_batch']=job['frame_batch']//2
        else:value.update(minimum_free_mib=45000,not_before=now+60,
                          avoid_gpus={**job.get('avoid_gpus',{}),str(job['gpu']):now+600})
    return value


def eligible(job,gpu,available,now):
    return (gpu in ALLOWED and gpu in available and now>=job['not_before']
            and available[gpu]['free']>=job['minimum_free_mib']
            and now>=job.get('avoid_gpus',{}).get(str(gpu),0))


def require_merge_state(state,live):
    if live or state.get('phase')!='full' or state.get('pending') or state.get('running') or state.get('failed'):
        raise ValueError('merge requires naturally drained successful full queue')


def related():
    """Inspect only; no signals, process restarts, or interference with other jobs."""
    if not Path('/proc').exists():return []
    rows=[]
    for d in Path('/proc').iterdir():
        if not d.name.isdigit() or int(d.name)==os.getpid():continue
        try:args=(d/'cmdline').read_bytes().decode(errors='replace').split('\0')
        except OSError:continue
        if any(Path(a).name=='amos_prompt_ablation_eval.py' for a in args) and any(a in ('worker','run','prepare') for a in args):
            rows.append(dict(pid=int(d.name),args=args))
    return rows


def preflight(out,*,processes=None,tmux_exists=None,pipeline=None):
    if Path(out).exists():raise FileExistsError('fresh output exists; relaunch refused')
    if related() if processes is None else processes:raise ValueError('live ablation process; duplicate launch refused')
    if tmux_exists is None:
        proc=subprocess.run(['tmux','has-session','-t',TMUX],capture_output=True,text=True)
        if proc.returncode not in (0,1):raise RuntimeError('tmux preflight failed: '+proc.stderr)
        tmux_exists=proc.returncode==0
    if tmux_exists and os.environ.get('AMOS_PROMPT_ABLATIONS_OWN_SESSION')!=TMUX:raise ValueError('dedicated tmux session exists')
    if pipeline is not None and Path(pipeline).exists():raise FileExistsError('pipeline log exists')


def file_inventory(paths,root=None):
    root=Path(root or ROOT)
    return {os.path.relpath(p,root).replace('\\','/'):sha(p) for p in sorted(set(map(Path,paths)))}


def verify_inventory(inventory,root=None):
    root=Path(root or ROOT)
    for name,expected in inventory.items():
        if sha(root/name)!=expected:raise ValueError('frozen dependency changed: '+name)


def dependency_sources():
    """Freeze a recursive import closure once; later verification never globs sources."""
    nn=ROOT.parent/'nninteractive_test_amos'
    if not (nn/'infer/infer_3d_point_common.py').exists():nn=ROOT/'nninteractive_test'
    search=(ROOT,ROOT/'v10',ROOT/'infer',ROOT/'sam2',nn,nn/'infer',nn/'utils')
    def resolve(name):
        rel=Path(*name.split('.'))
        for base in search:
            for candidate in (base/rel.with_suffix('.py'),base/rel/'__init__.py'):
                if candidate.is_file():return candidate.resolve()
        return None
    todo=[ROOT/n for n in SOURCE_SEEDS if (ROOT/n).is_file()]
    if not (ROOT/NATIVE).exists():raise FileNotFoundError(ROOT/NATIVE)
    # Native SAM2 Hydra targets and runtime relative imports are dynamic; freeze
    # this bounded package/config inventory during preparation, never at verify.
    for folder in (ROOT/'sam2/sam2',ROOT/'v10/model',ROOT/'model',nn/'infer',nn/'utils'):
        if folder.exists():todo.extend(p for p in folder.rglob('*') if p.suffix in ('.py','.yaml','.yml') and p.is_file())
    found=set()
    while todo:
        path=todo.pop().resolve()
        if path in found:continue
        found.add(path)
        # Importing a submodule executes each existing package initializer.
        # Analyze those initializers too, since they can import further code.
        boundaries=[base.resolve() for base in search if path.is_relative_to(base.resolve())]
        boundary=min(boundaries,key=lambda p:len(p.parts)) if boundaries else None
        for parent in path.parents:
            initializer=parent/'__init__.py'
            if initializer.is_file() and initializer.resolve() not in found:todo.append(initializer)
            if parent==boundary:break
        if path.suffix!='.py':continue
        tree=ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
        for node in ast.walk(tree):
            names=[]
            if isinstance(node,ast.Import):names=[a.name for a in node.names]
            elif isinstance(node,ast.ImportFrom):
                if node.level:
                    base=path.parent
                    for _ in range(node.level-1):base=base.parent
                    mod=base/Path(*(node.module or '').split('.')) if node.module else base
                    for candidate in (mod.with_suffix('.py'),mod/'__init__.py'):
                        if candidate.is_file():todo.append(candidate)
                    for a in node.names:
                        candidate=mod/(a.name+'.py')
                        if candidate.is_file():todo.append(candidate)
                elif node.module:names=[node.module]+[node.module+'.'+a.name for a in node.names]
            elif isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='import_module' and node.args and isinstance(node.args[0],ast.Constant) and isinstance(node.args[0].value,str):names=[node.args[0].value]
            for name in names:
                candidate=resolve(name)
                if candidate is not None:todo.append(candidate)
    return sorted(found)


def native_reference():
    if sha(ROOT/OLD)!=OLD_SHA:raise ValueError('historical prior-selected baseline SHA changed')
    value=read(ROOT/OLD);floors=value['floors'];part=floors.get('0.70',floors.get('.70'))
    if part is None:raise ValueError('historical floor.70 missing')
    rows=part['amos']['per_case_class'];validate_scope(rows,require_labels=False);validate_rows(rows,rows)
    return rows


def prompt_references(tasks):
    expected={key(t) for t in tasks};index={};paths=[]
    for path in sorted((ROOT/HISTORY/'prepared').glob('amos_s*of12/refs.json')):
        paths.append(path)
        for r in read(path)['per_case_class']:
            if key(r) in index:raise ValueError('duplicate historical prompt reference')
            index[key(r)]=r
    if set(index)!=expected:raise ValueError('historical exact1485 prompt keys missing')
    return [index[key(t)] for t in sorted(tasks,key=key)],paths


def native_case_cache_path(image,cache_root):
    """Execute only the actual native path/signature methods, without model imports."""
    source_path=ROOT/'v10/v10_data.py';method_path=ROOT/'v10/v10_2_data.py'
    identity=(str(ROOT),sha(source_path),sha(method_path))
    if identity not in _CACHE_PATH_METHODS:
        tree=ast.parse(source_path.read_text(encoding='utf-8-sig'))
        versions=[n.value for n in ast.walk(tree) if isinstance(n,ast.Assign)
                  and any(isinstance(t,ast.Name) and t.id=='CASE_IMAGE_CACHE_VERSION' for t in n.targets)]
        if len(versions)!=1:raise ValueError('native case CT cache version anchor changed')
        def function(path,name):
            parsed=ast.parse(path.read_text(encoding='utf-8-sig'))
            matches=[n for n in ast.walk(parsed) if isinstance(n,ast.FunctionDef) and n.name==name]
            if len(matches)!=1:raise ValueError('native cache path method anchor changed: '+name)
            node=copy.deepcopy(matches[0]);node.decorator_list=[]
            ns=dict(os=os,Path=Path,hashlib=hashlib)
            # Native annotations require no runtime imports or Python-version binding.
            node.returns=None
            for arg in node.args.posonlyargs+node.args.args+node.args.kwonlyargs:arg.annotation=None
            exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),ns)
            return ns[name]
        _CACHE_PATH_METHODS[identity]=(function(method_path,'_case_image_cache_path'),
                                       function(source_path,'_source_signature'),ast.literal_eval(versions[0]))
    method,signature,version=_CACHE_PATH_METHODS[identity]
    owner=types.SimpleNamespace(use_cache=True,cache_dir=str(cache_root),shared_case_cache_dir=str(cache_root),
                                source_name='amos',CASE_IMAGE_CACHE_VERSION=version,_source_signature=signature)
    value=method(owner,str(image))
    if value is None:raise ValueError('actual native case CT cache path unavailable')
    return Path(value)


def data_and_cache_inventory(tasks,refs):
    """Hash all CT/GT and prompts, plus selected shared CT presence; write no data."""
    import numpy as np
    data_paths=set();prompt_paths=[];cases={};native_paths={};case_files=defaultdict(lambda:dict(dataset=set(),prompts=set()));base=ROOT/CACHE
    by_key={key(r):r for r in refs}
    for t in tasks:
        image=Path(t['image']);label=Path(t['label'])
        current=(image if image.is_absolute() else ROOT/image,label if label.is_absolute() else ROOT/label)
        data_paths.update(current);case_files[t['case_id']]['dataset'].update(os.path.relpath(p,ROOT).replace('\\','/') for p in current)
        name=by_key[key(t)]['prompt_cache_name']
        if Path(name).name!=name or '\\' in name:raise ValueError('unsafe prompt path')
        path=base/'metadata/shared_adaptive_prompt_v1/amos'/name;prompt_paths.append(path)
        case_files[t['case_id']]['prompts'].add(os.path.relpath(path,ROOT).replace('\\','/'))
        with np.load(path,allow_pickle=False) as package:cache_name=str(package['case_image_name'].item())
        if Path(cache_name).name!=cache_name or '\\' in cache_name or not cache_name.endswith('.npy'):raise ValueError('unsafe shared CT reference')
        entry=dict(image=str(image),label=str(label),cache_name=cache_name)
        if t['case_id'] in cases and cases[t['case_id']]!=entry:raise ValueError('classes do not share one case CT')
        if t['case_id'] not in native_paths:
            actual=native_case_cache_path(current[0],base)
            if actual.resolve()!=(base/'_case_images_v1'/cache_name).resolve():raise ValueError('prompt CT reference differs from actual native cache path')
            native_paths[t['case_id']]=actual
        cases[t['case_id']]=entry
    if len(cases)!=100 or len({v['cache_name'] for v in cases.values()})!=100:raise ValueError('100 distinct case CT references required')
    cache={}
    for case,entry in sorted(cases.items()):
        source_image=Path(entry['image'])
        if not source_image.is_absolute():source_image=ROOT/source_image
        path=native_paths[case]
        value=dict(path=os.path.relpath(path,ROOT).replace('\\','/'),image=os.path.relpath(source_image,ROOT).replace('\\','/'),present=path.exists())
        if path.exists():value.update(size=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns,sha256=sha(path))
        cache[case]=value
    stats={os.path.relpath(p,ROOT).replace('\\','/'):dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in set(data_paths)|set(prompt_paths)}
    return dict(dataset=file_inventory(data_paths),prompts=file_inventory(prompt_paths),shared_ct=cache,stats=stats,
                case_files={c:{kind:sorted(paths) for kind,paths in values.items()} for c,values in sorted(case_files.items())})


def verify_cache(cache,rehash=True,case_id=None):
    for case,entry in cache.items():
        path=ROOT/entry['path']
        if path.exists()!=entry['present']:raise ValueError('selected CT cache presence changed')
        if path.exists():
            if path.stat().st_size!=entry['size'] or path.stat().st_mtime_ns!=entry['mtime_ns']:raise ValueError('selected shared CT cache stat changed')
            if (rehash or case==case_id) and sha(path)!=entry['sha256']:raise ValueError('selected shared CT cache changed')


def prepare(out,*,already_checked=False):
    out=Path(out)
    if not already_checked:preflight(out,pipeline=ROOT/PIPELINE)
    if out!=ROOT/OUT or out.exists():raise FileExistsError('fresh dedicated output required')
    if sha(ROOT/TRAINING)!=TRAINING_SHA or sha(ROOT/WEIGHTS)!=WEIGHT_SHA:raise ValueError('training split/E490 changed')
    reference=native_reference();split=read(ROOT/TRAINING)
    historical=read(ROOT/HISTORY/'manifest.json')['tasks'];tasks=sorted([t for t in historical if key(t)[0]=='amos'],key=key)
    validate_scope(tasks,split);validate_rows(reference,tasks)
    # Match every label and image against the frozen original split, beyond case IDs.
    catalog={int(r['global_class_id']):r for r in split['class_catalog']}
    entries={str(r['case_id']):r for r in split['datasets']['amos']['val']}
    for t in tasks:
        e=entries[t['case_id']];meta=catalog[key(t)[2]]
        if (int(t['local_class_id'])!=int(meta['local_class_id']) or t['class_name']!=meta['class_name']
                or key(t)[2] not in map(int,e['classes'])
                or t['image']!=str(e.get('image',e.get('image_path')))
                or t['label']!=str(e.get('label',e.get('label_path')))):raise ValueError('original image/label/class mapping changed')
    refs,ref_paths=prompt_references(tasks);assets=data_and_cache_inventory(tasks,refs)
    sources=dependency_sources()
    for required in SOURCE_SEEDS:
        if not (ROOT/required).is_file():raise FileNotFoundError('missing required dependency '+required)
    model_assets=[ROOT/WEIGHTS,ROOT/'sam2/checkpoints/sam2.1_hiera_large.pt',ROOT/'models/sat-medical-text/text_encoder.pth']
    model_assets.extend(p for p in (ROOT/'models/biolord-2023-c').rglob('*') if p.is_file())
    if len(model_assets)<=3:raise ValueError('semantic tokenizer/model assets absent')
    out.mkdir(parents=True,exist_ok=False)
    atomic_new(out/'manifest.json',dict(tasks=tasks,expected_task_count=1485))
    atomic_new(out/'refs.json',dict(per_case_class=refs))
    index={key(r):r for r in refs};jobs=case_jobs(tasks);controls=control_jobs(tasks)
    for j in jobs+controls:
        prepared=out/'prepared'/j['name'];j['prepared']=str(prepared)
        atomic_new(prepared/'manifest.json',dict(tasks=j['pending'],expected_task_count=j['task_count']))
        atomic_new(prepared/'refs.json',dict(per_case_class=[index[key(t)] for t in j['pending']]))
    atomic_new(out/'jobs.json',dict(cases=jobs,controls=controls))
    immutable=[out/'manifest.json',out/'refs.json',out/'jobs.json']+[out/'prepared'/j['name']/name for j in jobs+controls for name in ('manifest.json','refs.json')]
    protocol=dict(sources=file_inventory(sources),assets=file_inventory(model_assets),training=file_inventory([ROOT/TRAINING]),
                  reference=file_inventory([ROOT/OLD,ROOT/HISTORY/'manifest.json']+ref_paths),data=assets,
                  prepared=file_inventory(immutable),native_source_sha256=sha(ROOT/NATIVE),
                  policy=dict(variants=list(VARIANTS),tasks=1485,cases=100,classes=15,rows=8910,floor=.70,
                    dynamic_max=.75,fallback=.60,weights='E490live',memory_group=1,threshold='logit_midpoint_radius2',
                    semantic='legacy_amos',image_size=1024,patch_scaling=False,interactions=1,
                    controls_in_final=False,physical_gpus=list(ALLOWED),frame_batch_cap=4,initial_free_mib=30720,
                    batch1_oom_free_mib=45000,oom_wait_seconds=60,avoid_last_gpu_seconds=600,
                    metric_definition='native ten complete-mask metrics; axial nonempty union; oneempty0; bothemptyexcluded',
                    disk_tensors=False,selected_floor_prior=True,winner_selection=False))
    atomic_new(out/'protocol.json',protocol)
    atomic_new(out/'protocol_signature.json',dict(sha256=sha(out/'protocol.json')))
    verify(out);print('AMOS_ABLATIONS_PREPARED100CASES1485KEYS8910ROWS',flush=True)


def verify(out,*,rehash=False,case_id=None):
    out=Path(out)
    if read(out/'protocol_signature.json')!=dict(sha256=sha(out/'protocol.json')):raise ValueError('frozen protocol checksum changed')
    protocol=read(out/'protocol.json')
    for field in ('sources','assets','training','reference','prepared'):verify_inventory(protocol[field])
    data=protocol['data']
    for name,expected in data.get('stats',{}).items():
        stat=(ROOT/name).stat()
        if dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns)!=expected:raise ValueError('frozen CT/GT/prompt stat changed: '+name)
    for field in ('dataset','prompts'):
        inventory=data[field]
        if rehash:verify_inventory(inventory)
        elif case_id is not None:
            selected=data['case_files'][case_id][field];verify_inventory({name:inventory[name] for name in selected})
    verify_cache(data['shared_ct'],rehash=rehash,case_id=case_id)
    if protocol['policy']['variants']!=list(VARIANTS) or protocol['policy']['floor']!=.70:raise ValueError('fixed policy changed')
    if protocol['training'].get(TRAINING)!=TRAINING_SHA or protocol['assets'].get(WEIGHTS)!=WEIGHT_SHA or protocol['reference'].get(OLD)!=OLD_SHA:
        raise ValueError('frozen training/pretrained/reference identities changed')
    validate_scope(read(Path(out)/'manifest.json')['tasks'],read(ROOT/TRAINING))
    return protocol


def runtime():return importlib.import_module('tools.amos_prompt_ablation_runtime')


def validate_rows(rows,expected):
    keys=[key(r) for r in rows]
    if len(keys)!=len(set(keys)) or set(keys)!={key(r) for r in expected}:raise ValueError('metric exact key coverage mismatch')
    for r in rows:
        for name in METRICS:
            value=float(r[name])
            if not math.isfinite(value) or value<0 or (name!='pred_gt_ratio' and value>1):raise ValueError('nonfinite/out-of-bounds '+name)


def validate_record(record,task,variant,batch,control=False):
    if record.get('variant')!=variant or record.get('actual_frame_batch')!=batch:raise ValueError('variant/actualbatch changed')
    row=record['row'];validate_rows([row],[task])
    if int(row.get('label_id',task['local_class_id']))!=int(task['local_class_id']):raise ValueError('original label changed')
    threshold=float(row['mask_threshold_used']);source=row['dynamic_threshold_source']
    if not ((source in FALLBACK and threshold==.60) or (source not in FALLBACK and math.isfinite(threshold) and .70<=threshold<=.75)):
        raise ValueError('actual fixed-floor threshold changed')
    runtime().validate_record(record,task,variant,batch,control=control)


def validate_class(records,task,control=False):
    if len(records)!=6 or {r.get('variant') for r in records}!=set(VARIANTS):raise ValueError('all six variants must complete atomically')
    batches={r.get('actual_frame_batch') for r in records}
    if len(batches)!=1 or next(iter(batches)) not in (1,2,4):raise ValueError('all six variants need same actual frame batch')
    batch=next(iter(batches))
    for record in records:validate_record(record,task,record['variant'],batch,control=control)
    return batch


def commit_class(job,task,records,proofs):
    validate_class(records,task,job['control'])
    if len(proofs)<8:raise ValueError('at least eight independently bound class proof files required')
    for path,expected in proofs.items():
        if sha(path)!=expected:raise ValueError('class input proof changed')
    folder=Path(job['run'])/'classes';name=task['case_id']+'__'+str(key(task)[2]);record=folder/(name+'.json');marker=folder/(name+'.done.json')
    atomic_new(record,dict(task=task,records=records,control=job['control'],actual_frame_batch=records[0]['actual_frame_batch']))
    atomic_new(marker,dict(job_name=job['name'],record=str(record),files={**proofs,str(record):sha(record)}))
    return marker


def validate_attempt(job,canonical,out):
    for f in ('name','model','case_id','source','control','variants','prepared'):
        if job.get(f)!=canonical.get(f):raise ValueError('unsigned canonical case binding: '+f)
    if (job.get('gpu') not in ALLOWED or not str(job.get('gpu_uuid','')).startswith('GPU-')
            or job.get('frame_batch') not in (1,2,4) or job.get('actual_frame_batch')!=job['frame_batch']
            or job.get('minimum_free_mib') not in (30720,45000) or type(job.get('attempt')) is not int
            or not 0<=job.get('oom_retries',-1)<=job['attempt']):raise ValueError('invalid attempt resource policy')
    by_key={key(t):t for t in canonical['pending']};current=job['pending']
    if not current or len(current)!=len({key(t) for t in current}) or job['task_count']!=len(current) or any(by_key.get(key(t))!=t for t in current):raise ValueError('unsigned pending class subset')
    tag=canonical['name']+'_a'+str(job['attempt'])
    for f,folder,suffix in (('run','runs',''),('spec','job_specs','.json'),('log','logs','.log')):
        if job.get(f)!=str(Path(out)/folder/(tag+suffix)):raise ValueError('attempt path binding changed')


def mandatory_proofs(out,run):
    return [Path(out)/'protocol.json',Path(out)/'protocol_signature.json',Path(out)/'job_specs'/(run.name+'.json'),
            run/'prepared/manifest.json',run/'prepared/refs.json',run/'resource_policy_audit.json',
            run/'restoration_start.json',run/'prepared_input_audit.json',run/'runtime_input_audit.json']


def class_records(out,control=False):
    out=Path(out);alljobs=read(out/'jobs.json');jobs={j['name']:j for j in alljobs['controls' if control else 'cases']}
    allnames={j['name'] for v in alljobs.values() for j in v};result={}
    for marker in sorted((out/'runs').glob('*/classes/*.done.json')):
        info=read(marker);name=info.get('job_name')
        if name not in allnames:raise ValueError('unknown class marker job')
        if name not in jobs:continue
        run=marker.parents[1];canonical=jobs[name];spec=read(out/'job_specs'/(run.name+'.json'));validate_attempt(spec,canonical,out)
        record_path=marker.with_name(marker.name[:-10]+'.json')
        expected=set(map(str,mandatory_proofs(out,run)+[record_path]))
        if info.get('record')!=str(record_path) or set(info.get('files',{}))!=expected:raise ValueError('marker omitted or misbound mandatory proofs')
        for path,expected_sha in info['files'].items():
            if sha(path)!=expected_sha:raise ValueError('immutable class proof changed: '+path)
        value=read(record_path);task=value['task']
        if task not in spec['pending'] or value['control']!=control:raise ValueError('class not in signed pending/phase')
        batch=validate_class(value['records'],task,control)
        if batch!=spec['frame_batch'] or value['actual_frame_batch']!=batch:raise ValueError('signed actualbatch mismatch')
        manifest=read(run/'prepared/manifest.json');refs=read(run/'prepared/refs.json');frozen={key(r):r for r in read(out/'refs.json')['per_case_class']}
        if manifest!=dict(tasks=spec['pending'],expected_task_count=spec['task_count']) or refs!=dict(per_case_class=[frozen[key(t)] for t in spec['pending']]):raise ValueError('prepared attempt inputs changed')
        if read(run/'prepared_input_audit.json')!={str(run/'prepared'/n):sha(run/'prepared'/n) for n in ('manifest.json','refs.json')}:raise ValueError('prepared audit changed')
        resource=read(run/'resource_policy_audit.json')
        if any(resource.get(f)!=spec[f] for f in ('gpu','gpu_uuid','frame_batch','minimum_free_mib','oom_retries')) or resource['actual_start_free_mib']<spec['minimum_free_mib'] or not resource['cuda_probe']:raise ValueError('actual resource policy audit changed')
        if any(r['restoration']!=read(run/'restoration_start.json') for r in value['records']):raise ValueError('restoration state differs across variants')
        runtime_inputs=read(run/'runtime_input_audit.json')
        protocol=read(out/'protocol.json')
        if runtime_inputs.get('paths')!=runtime_paths(out,run,protocol):raise ValueError('runtime paths/source binding changed')
        for r in value['records']:
            if (r.get('runtime_input_audit')!=runtime_inputs.get('actual') or r.get('native_source_sha256')!=protocol['native_source_sha256']
                    or r.get('resource_binding',{}).get('gpu')!=spec['gpu']
                    or r.get('resource_binding',{}).get('gpu_uuid')!=spec['gpu_uuid']):raise ValueError('actual runtime/spec/resource binding changed')
            validate_ct_cache_access(r,task,protocol['data']['shared_ct'][task['case_id']])
        if key(task) in result:raise ValueError('duplicate committed class')
        result[key(task)]=dict(record=value,marker=str(marker),marker_sha256=sha(marker),spec=spec)
    return result


def validate_ct_cache_access(record,task,entry):
    """Actual runtime reads must reference the signed case CT, including RAM misses."""
    expected_path=(ROOT/entry['path']).resolve();expected_image=(ROOT/entry['image']).resolve()
    task_image=Path(task['image'])
    if not task_image.is_absolute():task_image=ROOT/task_image
    if task_image.resolve()!=expected_image:raise ValueError('frozen native CT image differs from task')
    events=record.get('ct_cache_access',[])
    if not events:raise ValueError('missing actual native CT cache access')
    expected_mode='read_only_hit' if entry['present'] else 'current_case_ram_miss'
    for event in events:
        if not event.get('path') or not event.get('image'):raise ValueError('missing actual CT cache path/image')
        path=Path(event['path']);image=Path(event['image'])
        if not path.is_absolute():path=ROOT/path
        if not image.is_absolute():image=ROOT/image
        if path.resolve()!=expected_path or image.resolve()!=expected_image or event.get('mode')!=expected_mode:
            raise ValueError('actual native CT cache path/image/hit-or-RAM state differs from frozen case')


def verify_worker_complete(job):
    value=read(Path(job['run'])/'native_worker_complete.json')
    out=Path(job['run']).parents[1]
    if (value.get('native_worker_complete') is not True or value.get('classes')!=job['task_count'] or value.get('variants')!=6
            or value.get('spec_sha256')!=sha(job['spec']) or value.get('protocol_sha256')!=sha(out/'protocol.json')):
        raise ValueError('missing native_worker_complete success guard')
    run=Path(job['run']);final=read(run/'restoration_final.json')
    if (value.get('restoration_final_sha256')!=sha(run/'restoration_final.json')
            or final.get('initial_restoration')!=read(run/'restoration_start.json')
            or final.get('successful_native_completion') is not True or final.get('full_tensor_comparison_phase')!='normal_worker_completion'
            or tuple(final.get(f) for f in ('model','epoch','weight_mode','prompt_parameters_exact','decoder_parameters_exact','guards_passed','model_load_count'))!=('v10_2',490,'live',418,212,True,1)):
        raise ValueError('missing normal worker final exact restoration proof')


def complete_case(out,job):
    """A successful final attempt certifies retained whole classes from earlier attempts."""
    out=Path(out);verify_worker_complete(job)
    canonical=next(j for j in read(out/'jobs.json')['controls' if job['control'] else 'cases'] if j['name']==job['name'])
    found=class_records(out,job['control']);chosen={k:v for k,v in found.items() if v['spec']['name']==job['name']}
    if set(chosen)!={key(t) for t in canonical['pending']}:raise ValueError('successful case missing retained complete classes')
    value=dict(job_name=job['name'],final_spec=job['spec'],final_spec_sha256=sha(job['spec']),
               native_worker_complete=str(Path(job['run'])/'native_worker_complete.json'),
               native_worker_complete_sha256=sha(Path(job['run'])/'native_worker_complete.json'),
               final_log=job['log'],final_log_sha256=sha(job['log']),
               contributing_attempts=contributing_attempts(chosen),
               markers={v['marker']:v['marker_sha256'] for v in chosen.values()})
    atomic_new(out/'case_success'/(job['name']+'.json'),value);return value


def verify_case_success(out,canonical,found):
    out=Path(out);proof=read(out/'case_success'/(canonical['name']+'.json'))
    spec=read(proof['final_spec']);validate_attempt(spec,canonical,out);verify_worker_complete(spec)
    if (proof['job_name']!=canonical['name'] or proof['final_spec']!=spec['spec']
            or sha(proof['final_spec'])!=proof['final_spec_sha256']
            or proof['native_worker_complete']!=str(Path(spec['run'])/'native_worker_complete.json')
            or sha(proof['native_worker_complete'])!=proof['native_worker_complete_sha256']
            or proof['final_log']!=spec['log'] or sha(spec['log'])!=proof['final_log_sha256']):raise ValueError('successful final-attempt proof changed')
    chosen={k:v for k,v in found.items() if v['spec']['name']==canonical['name']}
    if set(chosen)!={key(t) for t in canonical['pending']} or proof['markers']!={v['marker']:v['marker_sha256'] for v in chosen.values()}:
        raise ValueError('retained class marker coverage changed')
    if proof.get('contributing_attempts')!=contributing_attempts(chosen):raise ValueError('retained completed attempt/spec/log changed')
    return proof


def contributing_attempts(chosen):
    specs={v['spec']['spec']:v['spec'] for v in chosen.values()}
    return {path:dict(spec_sha256=sha(path),log=j['log'],log_sha256=sha(j['log'])) for path,j in sorted(specs.items())}


def gpu_snapshot():
    p=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    result={}
    for line in p.stdout.splitlines():
        index,uuid,free=[s.strip() for s in line.split(',')]
        if int(index) in ALLOWED:result[int(index)]=dict(uuid=uuid,free=int(free))
    return result


def runtime_paths(out,run,protocol):
    return dict(root=str(ROOT),manifest=str(run/'prepared/manifest.json'),refs=str(run/'prepared/refs.json'),
                base_split_json=str(ROOT/TRAINING),weights=str(ROOT/WEIGHTS),shared_adaptive_prompt_root=str(ROOT/CACHE),
                run_dir=str(run),native_source_sha256=protocol['native_source_sha256'],protocol=str(Path(out)/'protocol.json'))


def worker(job):
    out=Path(job['run']).parents[1];protocol=verify(out,case_id=job['case_id'])
    canonical=next(j for j in read(out/'jobs.json')['controls' if job['control'] else 'cases'] if j['name']==job['name'])
    validate_attempt(job,canonical,out)
    if read(job['spec'])!=job:raise ValueError('worker differs from immutable successful spec')
    if not job['control']:
        check_control_gate(out)
        if sha(out/'control_verification.json')!=job['control_gate_sha256']:raise ValueError('independent gate changed')
    available=gpu_snapshot();actual=available.get(job['gpu'])
    if actual is None or actual['uuid']!=job['gpu_uuid'] or actual['free']<job['minimum_free_mib']:raise SystemExit(75)
    os.environ['CUDA_VISIBLE_DEVICES']=str(job['gpu']);os.environ['V9_SHARED_CACHE_WRITE_POLICY']='read_only'
    import torch
    actual_uuid='GPU-'+str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')
    if actual_uuid!=job['gpu_uuid'] or torch.ones(1,device='cuda:0').item()!=1:raise ValueError('actual CUDA UUID/probe mismatch')
    torch.set_num_threads(2)
    run=Path(job['run']);run.mkdir(parents=True,exist_ok=False)
    frozen={key(r):r for r in read(out/'refs.json')['per_case_class']}
    atomic_new(run/'prepared/manifest.json',dict(tasks=job['pending'],expected_task_count=job['task_count']))
    atomic_new(run/'prepared/refs.json',dict(per_case_class=[frozen[key(t)] for t in job['pending']]))
    atomic_new(run/'prepared_input_audit.json',{str(run/'prepared'/n):sha(run/'prepared'/n) for n in ('manifest.json','refs.json')})
    atomic_new(run/'resource_policy_audit.json',{f:job[f] for f in ('gpu','gpu_uuid','frame_batch','minimum_free_mib','oom_retries')}|
               dict(actual_start_free_mib=actual['free'],cuda_uuid=actual_uuid,cuda_probe=True))
    paths=runtime_paths(out,run,protocol)
    session=runtime().initialize(job,paths)
    atomic_new(run/'runtime_input_audit.json',dict(paths=paths,actual=session.input_audit))
    for task in job['pending']:
        native=runtime().evaluate(session,task,'native_control') if job['control'] else None
        rows=[runtime().evaluate(session,task,v) for v in VARIANTS]
        if not (run/'restoration_start.json').exists():atomic_new(run/'restoration_start.json',rows[0]['restoration'])
        if job['control']:
            proof=runtime().validate_control(native,rows)
            atomic_new(run/'native_control.json',dict(task=task,native=native,variants=rows,proof=proof))
        commit_class(job,task,rows,{str(p):sha(p) for p in mandatory_proofs(out,run)})
    final=session.finish()
    atomic_new(run/'restoration_final.json',final)
    atomic_new(run/'native_worker_complete.json',dict(native_worker_complete=True,classes=job['task_count'],variants=6,
                restoration_final_sha256=sha(run/'restoration_final.json'),
                spec_sha256=sha(job['spec']),protocol_sha256=sha(out/'protocol.json')))
    print('AMOS_ABLATIONS_NATIVE_WORKER_COMPLETE',job['name'],job['task_count'],flush=True)


def check_controls(out):
    out=Path(out);jobs=read(out/'jobs.json')['controls'];committed=class_records(out,True)
    if len(jobs)!=1 or len(committed)!=1:raise ValueError('fresh real-GPU native plus6variant control required')
    item=next(iter(committed.values()));spec=item['spec'];verify_case_success(out,jobs[0],committed)
    path=Path(spec['run'])/'native_control.json';control=read(path)
    validate_record(control['native'],control['task'],'native_control',spec['frame_batch'],control=True)
    runtime().validate_control(control['native'],control['variants'])
    return dict(digest=digest(dict(marker_sha256=item['marker_sha256'],native_control_sha256=sha(path),
                                  worker_complete_sha256=sha(Path(spec['run'])/'native_worker_complete.json'))),
                real_gpu=True,task=control['task'],spec=spec)


def check_control_gate(out):
    out=Path(out);controls=check_controls(out);gate=read(out/'control_verification.json')
    if (gate.get('passed') is not True or gate.get('real_gpu') is not True or not gate.get('reviewer')
            or gate['reviewer'] in ('controller','runtime','cpu_tests')
            or gate.get('protocol_sha256')!=sha(out/'protocol.json') or gate.get('controls_digest')!=controls['digest']):
        raise ValueError('independent real-GPU control artifact gate failed')
    return gate


def wait_control_gate(out):
    control=check_controls(out);print('AMOS_ABLATIONS_AWAIT_INDEPENDENT_CONTROL_REVIEW',json.dumps(control,sort_keys=True),flush=True)
    write_status(Path(out)/'status.json',dict(phase='control_gate',pending=100,running={},failed=[],controls_digest=control['digest']))
    while not (Path(out)/'control_verification.json').exists():time.sleep(10)
    check_control_gate(out);print('AMOS_ABLATIONS_INDEPENDENT_GPU_CONTROL_PASS',flush=True)


def make_attempt(out,job,gpu,available,completed):
    pending=[t for t in job['pending'] if key(t) not in completed]
    if not pending:return None
    out=Path(out);tag=job['name']+'_a'+str(job['attempt'])
    paths=dict(run=str(out/'runs'/tag),spec=str(out/'job_specs'/(tag+'.json')),log=str(out/'logs'/(tag+'.log')))
    if any(Path(p).exists() for p in paths.values()):raise FileExistsError('immutable attempt already exists')
    value=dict(job,pending=pending,task_count=len(pending),gpu=gpu,gpu_uuid=available[gpu]['uuid'],actual_frame_batch=job['frame_batch'],**paths)
    if not value['control']:value['control_gate_sha256']=sha(out/'control_verification.json')
    canonical=next(j for j in read(out/'jobs.json')['controls' if job['control'] else 'cases'] if j['name']==job['name'])
    validate_attempt(value,canonical,out);atomic_new(value['spec'],value)
    return value


class Backend:
    def __init__(self,out,control):self.out,self.control=Path(out),control
    def records(self):return class_records(self.out,self.control)
    def available(self):return gpu_snapshot()
    def now(self):return time.time()
    def sleep(self,s):time.sleep(s)
    def attempt(self,j,g,a,c):return make_attempt(self.out,j,g,a,c)
    def log(self,j):return Path(j['log']).read_text(encoding='utf-8',errors='replace')[-30000:]
    def status(self,state):write_status(self.out/'status.json',state)
    def emit(self,*a):print(*a,flush=True)
    def complete(self,j):complete_case(self.out,j)
    def spawn(self,j):
        Path(j['log']).parent.mkdir(parents=True,exist_ok=True);handle=Path(j['log']).open('x',encoding='utf-8')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(j['gpu']),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                 PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',V9_SHARED_CACHE_WRITE_POLICY='read_only')
        try:process=subprocess.Popen([sys.executable,__file__,'worker','--job',j['spec']],cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
        except BaseException:handle.close();raise
        return process,handle


def schedule_queue(jobs,control,backend):
    """Fatal errors stop dispatch immediately and allow all started peers to drain."""
    pending=copy.deepcopy(jobs);active={};failed=[];committed={};cause=None;status_writable=True
    while pending or active:
        stage='class_records'
        try:
            if not failed:committed=backend.records()
            for gpu,(proc,j,handle) in list(active.items()):
                stage='worker_poll'
                if proc.poll() is None:continue
                del active[gpu];stage='log_close';handle.close()
                if failed:continue
                stage='completed_proofs';done=backend.records();remaining=[t for t in j['pending'] if key(t) not in done]
                stage='worker_log';text=backend.log(j);resource=proc.returncode==75 or 'cuda out of memory' in text.lower() or 'OutOfMemoryError' in text
                if proc.returncode!=0 and not resource:failed.append(dict(name=j['name'],returncode=proc.returncode,log=j['log']))
                elif proc.returncode==0:
                    if remaining:raise ValueError('worker exited0 without all class commits')
                    stage='native_worker_complete';backend.complete(j)
                elif remaining:pending.append(retry(j,proc.returncode,text,backend.now()))
                else:failed.append(dict(name=j['name'],reason='failed worker cannot certify native completion'))
            if not failed:
                stage='class_records';committed=backend.records();stage='gpu_snapshot';available=backend.available()
                for gpu in ALLOWED:
                    if gpu in active:continue
                    selected=next((j for j in pending if eligible(j,gpu,available,backend.now())),None)
                    if selected is None:continue
                    pending.remove(selected)
                    if all(key(t) in committed for t in selected['pending']):continue
                    stage='make_attempt';j=backend.attempt(selected,gpu,available,committed)
                    if j is None:continue
                    stage='worker_spawn';proc,handle=backend.spawn(j);active[gpu]=(proc,j,handle)
                    backend.emit('AMOS_ABLATIONS_START',gpu,proc.pid,j['name'],j['task_count'],'batch',j['frame_batch'])
        except BaseException as error:
            failed.append(dict(stage=stage,error=repr(error)));cause=cause or error
        state=dict(time=backend.now(),phase='controls' if control else 'full',pending=len(pending),
                   running={str(g):dict(pid=p.pid,name=j['name'],spec=j['spec'],log=j['log']) for g,(p,j,h) in active.items()},
                   failed=copy.deepcopy(failed),committed_classes=len(committed))
        if status_writable:
            try:backend.status(state)
            except BaseException as error:failed.append(dict(stage='status_write',error=repr(error)));cause=cause or error;status_writable=False
        if failed and not active:raise RuntimeError('fatal queue error; peers drained naturally; evidence retained: '+repr(failed)) from cause
        if pending or active:
            try:backend.sleep(10)
            except BaseException as error:failed.append(dict(stage='controller_wait',error=repr(error)));cause=cause or error
    return state


def schedule(out,control):
    verify(out,rehash=control)
    if not control:check_control_gate(out)
    return schedule_queue(read(Path(out)/'jobs.json')['controls' if control else 'cases'],control,Backend(out,control))


def summarize(rows):
    rows=sorted(rows,key=key);validate_rows(rows,rows)
    if not rows:raise ValueError('no rows')
    by_class=defaultdict(list)
    for r in rows:by_class[key(r)[2]].append(float(r['dice']))
    return dict(task_count=len(rows),mean={m:math.fsum(float(r[m]) for r in rows)/len(rows) for m in METRICS},
                class_macro_dice=math.fsum(math.fsum(v)/len(v) for _,v in sorted(by_class.items()))/len(by_class),
                fallback_count=sum(r['dynamic_threshold_source'] in FALLBACK for r in rows),per_case_class=copy.deepcopy(rows))


def paired_delta(rows,reference):
    validate_rows(reference,rows);old={key(r):r for r in reference};rows=sorted(rows,key=key)
    if not rows:raise ValueError('no paired rows')
    return {m:math.fsum(float(r[m])-float(old[key(r)][m]) for r in rows)/len(rows) for m in METRICS}


def merge(out):
    out=Path(out)
    if (out/'merged_metrics.json').exists():raise FileExistsError('once-only merge already exists')
    # Ignore this run controller, but every actual worker must have exited.
    require_merge_state(read(out/'status.json'),[p for p in related() if 'worker' in p['args']]);verify(out,rehash=True);check_control_gate(out)
    tasks=read(out/'manifest.json')['tasks'];found=class_records(out)
    if len(found)!=1485 or set(found)!={key(t) for t in tasks}:raise ValueError('exact1485 fresh all-six class commits required')
    for j in read(out/'jobs.json')['cases']:verify_case_success(out,j,found)
    historical=native_reference();by_variant={v:[] for v in VARIANTS}
    for t in sorted(tasks,key=key):
        for r in found[key(t)]['record']['records']:by_variant[r['variant']].append(r['row'])
    result={};fresh=by_variant['baseline'];native={key(r):r for r in historical};baseline={key(r):r for r in fresh}
    for variant,rows in by_variant.items():
        validate_rows(rows,tasks);summary=summarize(rows);summary['delta_from_fresh_baseline']=paired_delta(rows,fresh)
        summary['delta_from_historical_baseline']=paired_delta(rows,historical);summary['floor']=.70
        summary['paired_deltas']=[dict(key=list(key(r)),fresh_baseline={m:float(r[m])-float(baseline[key(r)][m]) for m in METRICS},
                                      historical_baseline={m:float(r[m])-float(native[key(r)][m]) for m in METRICS}) for r in sorted(rows,key=key)]
        summary['actual_frame_batches']={str(b):sum(item['spec']['frame_batch']==b for item in found.values()) for b in (1,2,4)}
        result[variant]=summary
    value=dict(variants=result,rows=8910,task_count_per_variant=1485,controls_in_final=False,
               floor=.70,weights='E490live',winner_selection=False,protocol_sha256=sha(out/'protocol.json'),
               control_verification_sha256=sha(out/'control_verification.json'),historical_reference_sha256=OLD_SHA,
               caveat='Historical baseline deltas can also reflect frame-batch differences; fresh paired baseline uses the same actual batch per class.')
    verify(out);require_merge_state(read(out/'status.json'),[p for p in related() if 'worker' in p['args']]);atomic_new(out/'merged_metrics.json',value)
    print('AMOS_ABLATIONS_MERGED8910',flush=True);return value


def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('prepare','run','worker','status','merge'));p.add_argument('--job');return p


def run_pipeline(out):
    pipeline=ROOT/PIPELINE;preflight(out,pipeline=pipeline);pipeline.parent.mkdir(parents=True,exist_ok=True)
    with pipeline.open('x',encoding='utf-8',buffering=1) as log,redirect_stdout(log),redirect_stderr(log):
        try:
            prepare(out,already_checked=True);schedule(out,True);wait_control_gate(out);schedule(out,False);merge(out)
            print('AMOS_ABLATIONS_PIPELINE_RC=0',flush=True)
        except BaseException:traceback.print_exc();print('AMOS_ABLATIONS_PIPELINE_RC=1',flush=True);raise


def main():
    args=parser().parse_args();out=ROOT/OUT
    if args.mode=='worker':
        if not args.job:raise ValueError('--job required')
        worker(read(args.job))
    elif args.mode=='prepare':prepare(out)
    elif args.mode=='run':run_pipeline(out)
    elif args.mode=='merge':merge(out)
    else:
        verify(out);print(json.dumps(dict(queue=read(out/'status.json'),committed_classes=len(class_records(out)),
                    control_classes=len(class_records(out,True))),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
