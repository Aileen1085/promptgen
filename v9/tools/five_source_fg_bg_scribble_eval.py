"""Fresh five-source paired FG/BG scribble supplement.

All outputs are light JSON. Production records become visible only after normal
whole-case completion, final parameter checks and immutable input rehash.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import quote

ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools import five_source_fg_bg_scribble_runtime as runtime
from tools.five_source_fg_bg_scribble_runtime import SOURCES,VARIANTS,METRICS,key,sha


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.'+path.name+'.',suffix='.tmp',dir=str(path.parent))
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as stream:
            json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
            stream.flush();os.fsync(stream.fileno())
        # A hard link publishes one complete immutable file and fails on collision.
        os.link(temporary,path)
    finally: os.unlink(temporary)


def validate_inventory(inventory):
    if inventory.get('schema') != 'five_source_prompt_type_inventory_v1': raise ValueError('inventory schema mismatch')
    tasks = inventory['tasks']; keys = [key(t) for t in tasks]
    if len(keys) != len(set(keys)) or dict(Counter(k[0] for k in keys)) != {s:c['count'] for s,c in SOURCES.items()}:
        raise ValueError('requires exact frozen173 unique tasks')
    refs = inventory['refs']; refkeys = [key(r) for r in refs]
    if len(refkeys) != len(set(refkeys)) or set(refkeys) != set(keys) or any(not r.get('prompt_cache_name') for r in refs):
        raise ValueError('requires exact frozen173 prompt references')
    if not inventory.get('input_sha256'): raise ValueError('complete source/input SHA inventory required')
    return sorted(tasks,key=key)


def worker_inputs(inventory,source,case):
    all_inputs=inventory['input_sha256']
    if not inventory.get('case_inputs'): return all_inputs
    mapping={tuple(json.loads(name)):files for name,files in inventory['case_inputs'].items()}
    if (source,case) not in mapping: raise ValueError('worker case input scope missing')
    case_files={name for files in mapping.values() for name in files}
    checkpoints={config['checkpoint'] for config in SOURCES.values()}
    selected=(set(all_inputs)-case_files-checkpoints)|set(mapping[source,case])|{SOURCES[source]['checkpoint']}
    if not selected <= set(all_inputs): raise ValueError('worker input scope has unpinned files')
    return {name:all_inputs[name] for name in sorted(selected)}


def verify_inputs(protocol,case=None):
    inventory=protocol['inventory']
    selected=worker_inputs(inventory,*case) if case is not None else inventory['input_sha256']
    for relative,expected in selected.items():
        if Path(relative).is_absolute() or sha(ROOT/relative) != expected: raise ValueError('frozen input SHA mismatch '+relative)
    return selected


def protocol_at(out,case=None):
    path = Path(out)/'protocol.json'; protocol = read(path)
    validate_protocol(protocol);verify_inputs(protocol,case)
    return protocol,sha(path)


def case_name(source,case):
    if source not in SOURCES or not isinstance(case,str) or not case: raise ValueError('invalid source/case')
    return source+'__case_'+quote(case,safe='')


def native_manifest(tasks):
    return dict(manifest_kind='independent_eight_source_validation_v1',scope='ct13_all_previously_split_heldout_cases',
        expected_task_count=len(tasks),per_source_task_counts=dict(Counter(key(t)[0] for t in tasks)),tasks=tasks)


def bind_resources(job,out):
    if shutil.disk_usage(out).free < 3*1024**3: raise SystemExit(75)
    def query():
        text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
        rows={int(p[0]):p for p in ([v.strip() for v in line.split(',')] for line in text.splitlines() if line.strip())}
        gpu=int(job['gpu'])
        if gpu not in range(8) or gpu not in rows or rows[gpu][1] != job['gpu_uuid']:
            raise ValueError('physical GPU identity/UUID mismatch')
        minimum=max(30720,int(job.get('minimum_free_mib',30720)))
        if int(rows[gpu][2]) < minimum: raise SystemExit(75)
        return runtime.resource_snapshot(text,int(job['gpu']),job['gpu_uuid'])
    first=query()
    os.environ['CUDA_VISIBLE_DEVICES']=str(job['gpu']);os.environ['V9_SHARED_CACHE_WRITE_POLICY']='read_only'
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
    import torch
    raw_uuid=str(torch.cuda.get_device_properties(0).uuid)
    visible=raw_uuid if raw_uuid.startswith('GPU-') else 'GPU-'+raw_uuid
    if visible != job['gpu_uuid'] or torch.ones(1,device='cuda:0').item() != 1: raise ValueError('CUDA UUID/probe mismatch')
    torch.set_num_threads(2)
    second=query()
    return dict(first_snapshot=first,second_snapshot=second,gpu=int(job['gpu']),gpu_uuid=visible,
        free_mib_before_load=second['free_mib'],minimum_free_mib=max(30720,int(job.get("minimum_free_mib",30720))),cuda_probe=True,pid=os.getpid())


def compare_control(native,baseline):
    if native['actual_frame_batch'] != baseline['actual_frame_batch']: raise ValueError('control batch mismatch')
    difference=max(abs(float(native['row'][m])-float(baseline['row'][m])) for m in METRICS)
    if not math.isfinite(difference) or difference > 1e-6: raise ValueError('fresh native baseline metric mismatch')
    for name in ('initial','reserved','final','history'):
        if native['roi'].get(name) != baseline['roi'].get(name): raise ValueError('control ROI mismatch')
    if native['stages'] != baseline['stages']: raise ValueError('native baseline stages/consumption differ')
    for name in ('mask_threshold_used','dynamic_threshold_source'):
        if native['row'].get(name) != baseline['row'].get(name): raise ValueError('baseline postprocess differs')
    return dict(passed=True,max_metric_error=difference,roi_equal=True,stages_equal=True,actual_frame_batch=native['actual_frame_batch'])


def record_path(out,task,variant,control=False):
    source,case,class_id=key(task);case_name(source,case)
    prefix=Path(out)/('control_records' if control else 'records')
    return prefix/source/case_name(source,case)/variant/(str(class_id)+'.json')


def require_batch_gate(out,source,batch,digest):
    gate=read(Path(out)/'controls'/source/('batch_'+str(batch))/'control_pass.json')
    if (gate.get('passed'),gate.get('source'),gate.get('actual_frame_batch'),gate.get('protocol_sha256')) != (True,source,batch,digest):
        raise ValueError('matching source/batch native control required')
    return gate


def publish_case(out,run,tasks,records,batch):
    """Expose both conditions in one same-filesystem directory rename."""
    expected={(key(task),variant) for task in tasks for variant in VARIANTS}
    actual=[(key(task),variant) for task,variant,record in records]
    if len(actual) != len(expected) or set(actual) != expected or any(record['actual_frame_batch'] != batch for task,variant,record in records):
        raise ValueError('complete same-batch two-condition case required')
    source,case=key(tasks[0])[:2]
    if any(key(task)[:2] != (source,case) for task in tasks): raise ValueError('single case publication required')
    destination=Path(out)/'records'/source/case_name(source,case)
    if destination.exists(): raise FileExistsError('case already published; no overwrite')
    staging=Path(run)/'case_publication'
    if staging.exists(): raise FileExistsError('case staging already exists')
    for task,variant,record in records:
        atomic_new(staging/variant/(str(key(task)[2])+'.json'),record)
    destination.parent.mkdir(parents=True,exist_ok=True)
    os.rename(staging,destination)


def worker(out,job,control=False):
    out=Path(out);source=job['source'];case=(source,str(job['case_id']))
    protocol,digest=protocol_at(out,case)
    expected=[t for t in protocol['inventory']['tasks'] if key(t)[:2] == (source,str(job['case_id']))]
    if not expected or sorted(job['tasks'],key=key) != sorted(expected,key=key): raise ValueError('worker must own complete frozen case')
    if job.get('frame_batch') not in (1,2,4): raise ValueError('frame batch cap4')
    if not control:
        for s in SOURCES:
            gate=read(out/'controls'/s/'control_pass.json')
            if not gate['passed'] or gate.get('source') != s or gate['protocol_sha256'] != digest: raise ValueError('all five source controls must pass first')
        require_batch_gate(out,source,job['frame_batch'],digest)
        if job['variant'] != 'all': raise ValueError('production must couple both variants')
    variants=('native_control',*VARIANTS) if control else VARIANTS
    control_root=out/('control_batch_'+str(job['frame_batch'])) if control else out
    if any(record_path(control_root,t,v,control).exists() for t in expected for v in variants): raise FileExistsError('complete records already exist; do not overwrite')
    run=out/'jobs'/(case_name(source,job['case_id'])+'__'+('control' if control else job['variant'])+'__b'+str(job['frame_batch'])+'__a'+str(job.get('attempt',0)))
    if run.exists(): raise FileExistsError('attempt exists; use next explicit attempt')
    prepared=run/'prepared';refs={key(r):r for r in protocol['inventory']['refs']}
    atomic_new(prepared/'manifest.json',native_manifest(expected))
    atomic_new(prepared/'refs.json',dict(per_case_class=[refs[key(t)] for t in expected]))
    actual=dict(job,run=str(run/'native'),prepared=str(prepared),control=control,resource_binding=bind_resources(job,out))
    session=runtime.initialize(actual,dict(protocol['inventory']['paths'],root=str(ROOT)))
    records=[];comparisons=[]
    for task in sorted(expected,key=key):
        by_variant={}
        for variant in variants:
            record=session.evaluate(task,variant);by_variant[variant]=record
            records.append((task,variant,record))
        if control: comparisons.append(dict(task_key=list(key(task)),**compare_control(by_variant['native_control'],by_variant['baseline'])))
    finish=session.finish()
    if not finish.get('successful_native_completion'): raise ValueError('native normal finish missing')
    scoped_inputs=verify_inputs(protocol,case)
    published=[(task,variant,dict(record,normal_finish=finish,protocol_sha256=digest,input_sha256=scoped_inputs,control=control)) for task,variant,record in records]
    for task,variant,record in published:
        validate_bound_record(record,task,variant,protocol,digest,control)
    if control:
        for task,variant,record in published: atomic_new(record_path(control_root,task,variant,True),record)
    else: publish_case(out,run,expected,published,job['frame_batch'])
    complete=dict(passed=True,protocol_sha256=digest,source=source,case_id=job['case_id'],normal_finish=finish,
        actual_frame_batch=job['frame_batch'],comparisons=comparisons,records=len(records),controls_excluded_from346=control)
    atomic_new(run/'complete.json',complete)
    if control:
        atomic_new(out/'controls'/source/('batch_'+str(job['frame_batch']))/'control_pass.json',complete)
        if not (out/'controls'/source/'control_pass.json').exists():
            atomic_new(out/'controls'/source/'control_pass.json',complete)
    print('CONTROL_PASS' if control else 'CASE_COMPLETE',source,job['case_id'],len(records),flush=True)
    return complete



PARENT = ROOT/'output/five_source_prompt_types_20261110_v2_casecoupled'
PARENT_HASHES = {
    'protocol.json':'8e6f9e238eea2b410245cb3607988bd35b052bc52ee41273b7c220698303377e',
    'merged_metrics.json':'3272789391b522cf624b89662740192f8c1f526d5e7ea49129183750a802dfb4',
    'independent_final_audit.json':'f2db8cd9fb2a85fd592a4cc5ae1e9310f56b5963a6ca28c6b788d1b4d263eab1',
    'resource_policy_anygpu_20261110.json':'183a35759d1b8ea975d485e6a34726579a4ab7e8ffead112a9043c81f58cd174',
    'independent_anygpu_final_policy_audit.json':'acc2329c2b4940c4417ba19f102d4428339790b0223b02d725265b119016e83a',
}
NEW_INPUTS = tuple(f'{folder}/five_source_fg_bg_scribble_{name}.py' if folder=='tools' else f'{folder}/test_five_source_fg_bg_scribble_{name}.py'
                   for folder in ('tools','tests') for name in ('runtime','eval','queue'))


def case_jobs(tasks, prior_batches):
    grouped=defaultdict(list)
    for task in tasks: grouped[key(task)[:2]].append(task)
    jobs=[];controls=[]
    for (source,case),selected in sorted(grouped.items()):
        batch=prior_batches[source,case]
        if batch not in (1,2,4): raise ValueError('invalid parent actual batch')
        base=dict(source=source,case_id=case,tasks=selected,frame_batch=batch,attempt=0)
        jobs.append(dict(base,name=case_name(source,case)+'__all',variant='all'))
        if not any(c['source']==source for c in controls):
            controls.append(dict(base,name=case_name(source,case)+'__control',variant='control',control=True))
    return jobs,controls


def expected_protocol(parent=PARENT):
    import copy
    import hashlib
    parent=Path(parent).absolute()
    if parent!=PARENT.absolute():raise ValueError('signed parent proof path mismatch')
    for name,expected in PARENT_HASHES.items():
        if sha(parent/name)!=expected:raise ValueError('signed parent proof mismatch '+name)
    parent_protocol=read(parent/'protocol.json');inventory=copy.deepcopy(parent_protocol['inventory'])
    tasks=validate_inventory(inventory)
    prior=read(parent/'merged_metrics.json');audit=read(parent/'independent_final_audit.json')
    if audit.get('passed') is not True or prior.get('task_count')!=519:raise ValueError('complete signed parent required')
    expected={(key(t),v) for t in tasks for v in ('baseline','fg_box_bg_point','fg_box_bg_scribble')}
    rows=prior['per_case_class'];identities=[(key(r),r['variant']) for r in rows]
    if len(identities)!=519 or set(identities)!=expected:raise ValueError('parent key coverage mismatch')
    batches=defaultdict(set)
    for row in rows:batches[key(row)[:2]].add(row['actual_frame_batch'])
    if len(batches)!=119 or any(len(b)!=1 for b in batches.values()):raise ValueError('parent whole-case actual batch mismatch')
    # The full old inventory remains pinned byte-for-byte. Add light proofs only.
    proofs=[parent/name for name in PARENT_HASHES]
    policy_path=parent/'resource_policy_anygpu_20261110.json';policy=read(policy_path)
    canonical={k:v for k,v in policy.items() if k!='policy_sha256'}
    policy_digest=hashlib.sha256(json.dumps(canonical,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8')).hexdigest()
    if (policy.get('schema'),policy.get('protocol_sha256'),policy.get('allowed_physical_gpus'),policy.get('min_free_mib'),policy.get('max_workers'),policy.get('source_path'))!=(
        'five_source_prompt_type_resource_policy_anygpu_v1',PARENT_HASHES['protocol.json'],list(range(8)),30720,4,'tools/five_source_prompt_type_anygpu.py'):
        raise ValueError('parent resource policy scope mismatch')
    if policy.get('policy_sha256')!=policy_digest or policy_digest!='4bc59445d3498b8eb0ad7bc9d5ec5ee1d9cd0ebe6518e16cd1782ebc7b9978e7':
        raise ValueError('parent resource policy canonical digest mismatch')
    if read(parent/'independent_anygpu_final_policy_audit.json').get('passed') is not True:
        raise ValueError('parent independent resource audit failed')
    references={policy['source_path']:policy['source_sha256'],**policy.get('extra_source_sha256',{})}
    for relative,expected_hash in references.items():
        if Path(relative).is_absolute() or '..' in Path(relative).parts or sha(ROOT/relative)!=expected_hash:
            raise ValueError('parent resource source binding mismatch')
        old=inventory['input_sha256'].get(relative)
        if old is not None and old!=expected_hash:raise ValueError('conflicting parent source hash')
        inventory['input_sha256'][relative]=expected_hash
    for path in proofs:
        relative=path.relative_to(ROOT.absolute()).as_posix()
        inventory['input_sha256'][relative]=sha(path)
    for relative in NEW_INPUTS:inventory['input_sha256'][relative]=sha(ROOT/relative)
    protocol=dict(schema='five_source_fg_bg_scribble_protocol_v1',inventory=inventory,source_config=SOURCES,
        variants=list(VARIANTS),production_task_count=346,requested_target_count=173,reference_count=173,
        parent=str(parent.relative_to(ROOT.absolute()).as_posix()),parent_proof_sha256=PARENT_HASHES,
        metric_scope='native_raw_nifti_full_volume',metrics=list(METRICS),controls_excluded=True,
        max_frame_batch=4,oom_retry=[4,2,1],allowed_physical_gpus=list(range(8)),min_free_mib=30720,max_workers=4,
        project_disk_reserve_bytes=3*1024**3,cache_policy='existing_shared_CT_read_only',array_exports=False,
        production_unit='whole_case_both_variants_one_session_same_actual_batch',atomic_publication='case_directory_rename',
        parent_actual_batches={json.dumps(k):next(iter(v)) for k,v in sorted(batches.items())},
        native_settings={source:dict(runtime._native.private_hooks(source).FROZEN_SETTINGS) for source in SOURCES})
    return protocol


def validate_protocol(protocol):
    if protocol!=expected_protocol():raise ValueError('strict signed supplement protocol mismatch')
    validate_inventory(protocol['inventory'])
    return protocol


def prepare(out,parent=PARENT):
    out=Path(out)
    if out.exists():raise FileExistsError('output scope already exists')
    protocol=expected_protocol(parent)
    verify_inputs(protocol)
    inventory=protocol['inventory'];tasks=validate_inventory(inventory)
    batches={tuple(json.loads(k)):v for k,v in protocol['parent_actual_batches'].items()}

    jobs,controls=case_jobs(tasks,batches)
    atomic_new(out/'protocol.json',protocol);atomic_new(out/'manifest.json',native_manifest(tasks))
    atomic_new(out/'jobs.json',jobs);atomic_new(out/'control_jobs.json',controls)
    return dict(production_rows=346,target_rows=173,reference_rows=173,jobs=len(jobs),controls=len(controls),protocol_sha256=sha(out/'protocol.json'))


def validate_binding(binding):
    minimum=max(30720,int(binding.get('minimum_free_mib',30720)))
    for name in ('first_snapshot','second_snapshot'):
        value=binding[name]
        if value.get('gpu')!=binding['gpu'] or value.get('gpu_uuid')!=binding['gpu_uuid'] or value.get('free_mib',0)<minimum:
            raise ValueError('actual repeated resource binding mismatch')
    if binding.get('free_mib_before_load',0)<minimum or binding.get('cuda_probe') is not True:
        raise ValueError('actual resource binding/probe incomplete')


def validate_bound_record(record,task,variant,protocol,digest,control=False):
    if record.get('protocol_sha256')!=digest or record.get('control') is not control:raise ValueError('record protocol/control mismatch')
    if record.get('input_sha256')!=worker_inputs(protocol['inventory'],*key(task)[:2]):raise ValueError('record scoped input mismatch')
    if record.get('native_source_sha256')!=protocol['inventory']['input_sha256']['v10/infer/infer_eight_source_adaptive_v10_2.py']:
        raise ValueError('record native source SHA mismatch')
    if not record.get('normal_finish',{}).get('successful_native_completion'):raise ValueError('normal finish missing')
    runtime.validate_record(record,task,variant,record['actual_frame_batch'],control=control)
    validate_binding(record['resource_binding'])


def status(out):
    out=Path(out);protocol=read(out/'protocol.json');digest=sha(out/'protocol.json')
    validate_protocol(protocol)
    tasks=validate_inventory(protocol['inventory']);counts=Counter();expected_cases=defaultdict(set);observed=defaultdict(set);batches=defaultdict(set)
    expected_paths=set()
    for task in tasks:
        case=key(task)[:2]
        for variant in VARIANTS:
            identity=(key(task)[2],variant);expected_cases[case].add(identity)
            path=record_path(out,task,variant);expected_paths.add(path)
            if not path.exists():continue
            record=read(path);validate_bound_record(record,task,variant,protocol,digest)
            require_batch_gate(out,case[0],record['actual_frame_batch'],digest)
            observed[case].add(identity);batches[case].add(record['actual_frame_batch']);counts[case[0],variant]+=1
    if set((out/'records').rglob('*.json'))-expected_paths:raise ValueError('unexpected raw production records')
    for case,values in observed.items():
        if values!=expected_cases[case] or len(batches[case])!=1:raise ValueError('partial or mixed-batch published case')
    per_source={s:{v:counts[s,v] for v in VARIANTS} for s in SOURCES}
    return dict(completed=sum(counts.values()),paired_rows=sum(counts.values()),expected=346,pending=346-sum(counts.values()),
        target_completed=sum(counts[s,VARIANTS[1]] for s in SOURCES),reference_completed=sum(counts[s,'baseline'] for s in SOURCES),
        expected_targets=173,per_source=per_source,per_source_variant=per_source)


def summarize(records,tasks):
    expected={key(t) for t in tasks};index={}
    if len(expected)!=len(tasks):raise ValueError('duplicate task identity')
    for r in records:
        ident=(key(r['row']),r['variant'])
        if ident in index:raise ValueError('duplicate raw record')
        index[ident]=r
        for m in METRICS:
            value=float(r['row'][m])
            if not math.isfinite(value) or value<0 or (m!='pred_gt_ratio' and value>1):raise ValueError('invalid raw metric')
    if set(index)!={(k,v) for k in expected for v in VARIANTS}:raise ValueError('paired raw key coverage mismatch')
    case_batches=defaultdict(set)
    for (k,v),r in index.items():case_batches[k[:2]].add(r['actual_frame_batch'])
    if any(len(b)!=1 or not b<={1,2,4} for b in case_batches.values()):raise ValueError('whole case must share actual batch')
    result={}
    for source in sorted({k[0] for k in expected}):
        keys=sorted(k for k in expected if k[0]==source);result[source]={}
        for variant in VARIANTS:
            selected=[index[k,variant] for k in keys];groups=defaultdict(list)
            for r in selected:groups[key(r['row'])[2]].append(r)
            result[source][variant]=dict(tasks=len(keys),
                mean={m:math.fsum(float(r['row'][m]) for r in selected)/len(keys) for m in METRICS},
                class_macro={m:math.fsum(math.fsum(float(r['row'][m]) for r in group)/len(group) for group in groups.values())/len(groups) for m in METRICS},
                paired_delta_vs_baseline={m:math.fsum(float(index[k,variant]['row'][m])-float(index[k,'baseline']['row'][m]) for k in keys)/len(keys) for m in METRICS},
                fallback_tasks=sum(str(r['row'].get('dynamic_threshold_source','')).startswith('fallback') for r in selected),
                actual_batches=dict(Counter(str(r['actual_frame_batch']) for r in selected)))
    return result


def raw_records(out,protocol):
    return [read(record_path(out,t,v)) for t in sorted(protocol['inventory']['tasks'],key=key) for v in VARIANTS]


def merge(out):
    out=Path(out)
    if (out/'merged_metrics.json').exists():raise FileExistsError('merged evidence already exists')
    protocol,digest=protocol_at(out)
    if status(out)['completed']!=346:raise ValueError('346 fresh paired rows required')
    records=raw_records(out,protocol)
    result=dict(schema='five_source_fg_bg_scribble_merged_v1',protocol_sha256=digest,task_count=346,target_count=173,
        controls_excluded=True,per_source=summarize(records,protocol['inventory']['tasks']),
        per_case_class=[dict(r['row'],variant=r['variant'],actual_frame_batch=r['actual_frame_batch']) for r in records])
    atomic_new(out/'merged_metrics.json',result);return result


def require_queue_complete(out,phase):
    state=read(Path(out)/('queue_'+phase+'.json'))
    if state.get('protocol_sha256')!=sha(Path(out)/'protocol.json'):raise ValueError('queue protocol SHA mismatch')
    if state.get('phase')!='complete' or state.get('failed') or state.get('pending') or state.get('running'):
        raise ValueError('queue did not naturally drain '+phase)
    return state


def validate_completed_jobs(out,phase,protocol,digest):
    state=require_queue_complete(out,phase)
    tasks=protocol['inventory']['tasks'];prior={tuple(json.loads(k)):v for k,v in protocol['parent_actual_batches'].items()}
    jobs,controls=case_jobs(tasks,prior)
    expected_cases={(j['source'],str(j['case_id'])):j for j in jobs}
    templates={j['source']:j for j in controls}
    completed=state.get('completed',[])
    production={};control_seen=set()
    for job in completed:
        source=job['source'];case=(source,str(job['case_id']));batch=job['frame_batch']
        if case not in expected_cases or batch not in (1,2,4) or batch>prior[case]:raise ValueError('queue completed scope/batch mismatch')
        if sorted(job['tasks'],key=key)!=sorted(expected_cases[case]['tasks'],key=key):raise ValueError('queue completed frozen task mismatch')
        if job.get('control'):
            template=templates[source]
            if job.get('variant')!='control' or str(job['case_id'])!=str(template['case_id']):raise ValueError('queue control case mismatch')
            if (source,batch) in control_seen:raise ValueError('duplicate queue control completion')
            control_seen.add((source,batch))
            gate=require_batch_gate(out,source,batch,digest)
            if str(gate['case_id'])!=case[1]:raise ValueError('queue control gate case mismatch')
            if phase=='controls':
                first=read(Path(out)/'controls'/source/'control_pass.json')
                if first!=gate:raise ValueError('initial source gate differs from completed control')
        else:
            if phase!='full' or job.get('variant')!='all' or case in production:raise ValueError('duplicate/unexpected production completion')
            production[case]=job
    if phase=='controls':
        if len(completed)!=len(SOURCES) or {s for s,b in control_seen}!=set(SOURCES):raise ValueError('exact five source control completions required')
    else:
        if set(production)!=set(expected_cases):raise ValueError('exact complete production case set required')
        for case,job in production.items():
            for task in job['tasks']:
                for variant in VARIANTS:
                    record=read(record_path(out,task,variant))
                    if record['actual_frame_batch']!=job['frame_batch']:raise ValueError('completed queue actual record batch mismatch')
    return state


def audit_used_batch(out,protocol,digest,source,batch):
    out=Path(out);gate=require_batch_gate(out,source,batch,digest)
    tasks=[t for t in protocol['inventory']['tasks'] if key(t)[:2]==(source,str(gate['case_id']))]
    if not tasks or gate.get('records')!=3*len(tasks):raise ValueError('three-variant control scope mismatch')
    errors=[]
    for task in tasks:
        records={}
        for v in ('native_control',*VARIANTS):
            r=read(record_path(out/('control_batch_'+str(batch)),task,v,True))
            validate_bound_record(r,task,v,protocol,digest,True)
            if r['actual_frame_batch']!=batch:raise ValueError('control actual batch mismatch')
            records[v]=r
        errors.append(compare_control(records['native_control'],records['baseline'])['max_metric_error'])
    return dict(source=source,actual_batch=batch,control_records=3*len(tasks),max_ten_metric_error=max(errors))


def audit_controls(out):
    out=Path(out);protocol,digest=protocol_at(out);state=validate_completed_jobs(out,'controls',protocol,digest)
    completed=state.get('completed',[])
    if len(completed)!=5 or {j['source'] for j in completed}!=set(SOURCES):raise ValueError('five source controls required')
    comparisons=[]
    for source in SOURCES:
        gate=read(out/'controls'/source/'control_pass.json')
        if gate.get('source')!=source or gate.get('protocol_sha256')!=digest or gate.get('passed') is not True:raise ValueError('source gate binding mismatch')
        comparisons.append(audit_used_batch(out,protocol,digest,source,gate['actual_frame_batch']))
    value=dict(passed=True,protocol_sha256=digest,controls_excluded=True,independently_verified_source_controls=comparisons)
    atomic_new(out/'independent_control_gate.json',value);return value


def independent_summary(records,tasks):
    """Rebuild audit statistics independently of the merge aggregation helper."""
    task_keys=[key(t) for t in tasks]
    wanted={(k,v) for k in task_keys for v in VARIANTS}
    by_identity={}
    for record in records:
        ident=(key(record['row']),record['variant'])
        if ident in by_identity:raise ValueError('duplicate raw audit record')
        by_identity[ident]=record
        for metric in METRICS:
            value=float(record['row'][metric])
            if not math.isfinite(value):raise ValueError('nonfinite raw audit metric')
    if len(set(task_keys))!=len(task_keys) or set(by_identity)!=wanted:raise ValueError('raw audit identity coverage mismatch')
    output={}
    for source in SOURCES:
        keys=sorted(k for k in task_keys if k[0]==source)
        if not keys:continue
        output[source]={}
        classes=sorted({k[2] for k in keys})
        for variant in VARIANTS:
            means={};macros={};deltas={}
            for metric in METRICS:
                means[metric]=math.fsum(float(by_identity[k,variant]['row'][metric]) for k in keys)/len(keys)
                per_class=[]
                for class_id in classes:
                    members=[k for k in keys if k[2]==class_id]
                    per_class.append(math.fsum(float(by_identity[k,variant]['row'][metric]) for k in members)/len(members))
                macros[metric]=math.fsum(per_class)/len(classes)
                differences=[float(by_identity[k,variant]['row'][metric])-float(by_identity[k,'baseline']['row'][metric]) for k in keys]
                deltas[metric]=math.fsum(differences)/len(keys)
            actual_batches={};fallback=0
            for k in keys:
                record=by_identity[k,variant];batch=str(record['actual_frame_batch'])
                actual_batches[batch]=actual_batches.get(batch,0)+1
                fallback+=str(record['row'].get('dynamic_threshold_source','')).startswith('fallback')
                if record['actual_frame_batch']!=by_identity[k,'baseline']['actual_frame_batch']:raise ValueError('audit pair batch mismatch')
            output[source][variant]=dict(tasks=len(keys),mean=means,class_macro=macros,paired_delta_vs_baseline=deltas,
                fallback_tasks=fallback,actual_batches=actual_batches)
    return output


def compare_numeric_tree(actual,expected):
    if isinstance(expected,dict):
        if not isinstance(actual,dict) or set(actual)!=set(expected):raise ValueError('merged summary field mismatch')
        return max((compare_numeric_tree(actual[k],v) for k,v in expected.items()),default=0.)
    if isinstance(expected,(int,float)):
        if not isinstance(actual,(int,float)) or not math.isfinite(float(actual)):raise ValueError('nonfinite numeric evidence')
        return abs(actual-expected)
    if actual!=expected:raise ValueError('merged evidence mismatch')
    return 0.


def audit_final(out):
    out=Path(out);protocol,digest=protocol_at(out);validate_completed_jobs(out,'full',protocol,digest)
    if status(out)['completed']!=346:raise ValueError('346 paired rows required')
    tasks=protocol['inventory']['tasks'];records=raw_records(out,protocol)
    if len({key(t)[:2] for t in tasks})!=119:raise ValueError('119 case scope required')
    controls=[audit_used_batch(out,protocol,digest,s,b) for s,b in sorted({(key(r['row'])[0],r['actual_frame_batch']) for r in records})]
    merged=read(out/'merged_metrics.json')
    if (merged.get('schema'),merged.get('protocol_sha256'),merged.get('task_count'),merged.get('target_count'),merged.get('controls_excluded'))!=(
        'five_source_fg_bg_scribble_merged_v1',digest,346,173,True):raise ValueError('merged protocol/scope mismatch')
    summary=independent_summary(records,tasks);error=compare_numeric_tree(merged['per_source'],summary)
    saved=merged['per_case_class'];index={(key(r),r['variant']):r for r in saved}
    if len(saved)!=346 or len(index)!=346 or set(index)!={(key(r['row']),r['variant']) for r in records}:raise ValueError('merged raw keys mismatch')
    for r in records:
        expected=dict(r['row'],variant=r['variant'],actual_frame_batch=r['actual_frame_batch'])
        error=max(error,compare_numeric_tree(index[key(r['row']),r['variant']],expected))
    if error>1e-12:raise ValueError('independent raw fsum aggregation mismatch')
    # A second full rehash closes the audit interval before immutable publication.
    verify_inputs(protocol)
    value=dict(passed=True,protocol_sha256=digest,unique_production_rows=346,unique_targets=173,unique_cases=119,
        controls_excluded=True,paired_same_actual_batch=True,maximum_recompute_error=error,per_source=summary,
        full_input_rehash_files=len(protocol['inventory']['input_sha256']),independently_verified_used_batch_controls=controls,
        merged_sha256=sha(out/'merged_metrics.json'))
    atomic_new(out/'independent_final_audit.json',value)
    print('FIVE_SOURCE_FG_BG_SCRIBBLE_FINAL_PASS',flush=True);return value


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','control','worker','status','merge','audit-controls','audit-final'))
    parser.add_argument('--out',required=True,type=Path);parser.add_argument('--parent',type=Path,default=PARENT);parser.add_argument('--job',type=Path)
    args=parser.parse_args(argv);os.chdir(ROOT)
    if args.mode=='prepare':value=prepare(args.out,args.parent)
    elif args.mode in ('worker','control'):value=worker(args.out,read(args.job),args.mode=='control')
    else:value={'status':status,'merge':merge,'audit-controls':audit_controls,'audit-final':audit_final}[args.mode](args.out)
    if args.mode!='merge':print(json.dumps(value,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
