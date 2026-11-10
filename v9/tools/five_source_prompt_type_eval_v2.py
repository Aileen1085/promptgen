"""Additive v2: wait on resource races; publish matched-batch whole cases.

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
from tools import five_source_prompt_type_runtime as runtime
from tools.five_source_prompt_type_runtime import SOURCES,VARIANTS,METRICS,key,sha


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
    validate_inventory(protocol['inventory']);verify_inputs(protocol,case)
    return protocol,sha(path)


def case_name(source,case):
    if source not in SOURCES or not isinstance(case,str) or not case: raise ValueError('invalid source/case')
    return source+'__case_'+quote(case,safe='')


def native_manifest(tasks):
    return dict(manifest_kind='independent_eight_source_validation_v1',scope='ct13_all_previously_split_heldout_cases',
        expected_task_count=len(tasks),per_source_task_counts=dict(Counter(key(t)[0] for t in tasks)),tasks=tasks)


def prepare(out, inventory_path):
    out = Path(out); inventory = read(inventory_path);tasks=validate_inventory(inventory)
    if out.exists(): raise FileExistsError('output scope already exists')
    for source,config in SOURCES.items():
        if config['checkpoint'] not in inventory['input_sha256']: raise ValueError('checkpoint not pinned '+source)
    for name in ('tools/five_source_prompt_type_eval_v2.py','tools/five_source_prompt_type_runtime.py',
                 'tools/amos_prompt_ablation_runtime.py','tools/amos_prompt_ablation_prompts.py','tools/bone_fast_metrics.py',
                 'v10/infer/infer_eight_source_adaptive_v10_2.py'):
        if inventory['input_sha256'].get(name) != sha(ROOT/name): raise ValueError('required source binding missing '+name)
    verify_inputs(dict(inventory=inventory))
    protocol = dict(schema='five_source_prompt_type_protocol_v2',inventory=inventory,source_config=SOURCES,
        variants=list(VARIANTS),production_task_count=519,metric_scope='native_raw_nifti_full_volume',metrics=list(METRICS),
        controls_excluded=True,max_frame_batch=4,oom_retry=[4,2,1],allowed_physical_gpus=[0,3],min_free_mib=30720,
        project_disk_reserve_bytes=3*1024**3,cache_policy='existing_shared_CT_read_only',array_exports=False,
        production_unit='whole_case_all_three_variants_one_session_same_actual_batch',atomic_publication='case_directory_rename')
    atomic_new(out/'protocol.json',protocol)
    atomic_new(out/'manifest.json',native_manifest(tasks))
    jobs,controls=case_jobs(tasks)
    atomic_new(out/'jobs.json',jobs);atomic_new(out/'control_jobs.json',controls)
    return dict(production_rows=519,jobs=len(jobs),controls=len(controls),protocol_sha256=sha(out/'protocol.json'))


def case_jobs(tasks):
    grouped = defaultdict(list)
    for task in tasks: grouped[key(task)[:2]].append(task)
    jobs=[];controls=[]
    for (source,case),selected in sorted(grouped.items()):
        name=case_name(source,case)
        base=dict(source=source,case_id=case,tasks=selected,frame_batch=4,attempt=0)
        jobs.append(dict(base,name=name+'__all',variant='all'))
        if not any(c['source'] == source for c in controls): controls.append(dict(base,name=name+'__control',variant='control',control=True))
    return jobs,controls


def bind_resources(job,out):
    if shutil.disk_usage(out).free < 3*1024**3: raise SystemExit(75)
    def query():
        text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
        rows={int(p[0]):p for p in ([v.strip() for v in line.split(',')] for line in text.splitlines() if line.strip())}
        gpu=int(job['gpu'])
        if gpu not in (0,3) or gpu not in rows or rows[gpu][1] != job['gpu_uuid']:
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
        free_mib_before_load=second['free_mib'],minimum_free_mib=30720,cuda_probe=True,pid=os.getpid())


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
    """Expose all three conditions in one same-filesystem directory rename."""
    expected={(key(task),variant) for task in tasks for variant in VARIANTS}
    actual=[(key(task),variant) for task,variant,record in records]
    if len(actual) != len(expected) or set(actual) != expected or any(record['actual_frame_batch'] != batch for task,variant,record in records):
        raise ValueError('complete same-batch three-condition case required')
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
        if job['variant'] != 'all': raise ValueError('production must couple all three variants')
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
    if control:
        for task,variant,record in published: atomic_new(record_path(control_root,task,variant,True),record)
    else: publish_case(out,run,expected,published,job['frame_batch'])
    complete=dict(passed=True,protocol_sha256=digest,source=source,case_id=job['case_id'],normal_finish=finish,
        actual_frame_batch=job['frame_batch'],comparisons=comparisons,records=len(records),controls_excluded_from519=control)
    atomic_new(run/'complete.json',complete)
    if control:
        atomic_new(out/'controls'/source/('batch_'+str(job['frame_batch']))/'control_pass.json',complete)
        if not (out/'controls'/source/'control_pass.json').exists():
            atomic_new(out/'controls'/source/'control_pass.json',complete)
    print('CONTROL_PASS' if control else 'CASE_COMPLETE',source,job['case_id'],len(records),flush=True)
    return complete


def status(out):
    out=Path(out);protocol=read(out/'protocol.json');digest=sha(out/'protocol.json')
    tasks=validate_inventory(protocol['inventory']);counts=Counter();pending=[]
    expected_cases=defaultdict(set);observed_cases=defaultdict(set);case_batches=defaultdict(set)
    for task in tasks:
        for variant in VARIANTS:
            case=key(task)[:2];identity=(key(task)[2],variant)
            expected_cases[case].add(identity)
            path=record_path(out,task,variant)
            if path.exists():
                record=read(path)
                if record['protocol_sha256'] != digest or record['control']: raise ValueError('mixed/control production record')
                if record['input_sha256'] != worker_inputs(protocol['inventory'],*key(task)[:2]): raise ValueError('record input inventory mismatch')
                expected_native=protocol['inventory']['input_sha256']['v10/infer/infer_eight_source_adaptive_v10_2.py']
                if record['native_source_sha256'] != expected_native: raise ValueError('record native source SHA mismatch')
                runtime.validate_record(record,task,variant,record['actual_frame_batch'])
                require_batch_gate(out,key(task)[0],record['actual_frame_batch'],digest)
                if not record['normal_finish']['successful_native_completion']: raise ValueError('incomplete record')
                observed_cases[case].add(identity);case_batches[case].add(record['actual_frame_batch'])
                counts[(key(task)[0],variant)]+=1
            else: pending.append([*key(task),variant])
    for case,observed in observed_cases.items():
        if observed != expected_cases[case] or len(case_batches[case]) != 1:
            raise ValueError('published case must contain every class and all three conditions at one actual batch')
    return dict(completed=sum(counts.values()),expected=519,pending=len(pending),per_source_variant={s:{v:counts[s,v] for v in VARIANTS} for s in SOURCES})


def merge(out):
    out=Path(out);protocol,digest=protocol_at(out)
    if status(out)['completed'] != 519: raise ValueError('all519 fresh production records required')
    rows=[];summary={}
    for source in SOURCES:
        selected=[t for t in protocol['inventory']['tasks'] if key(t)[0] == source]
        records={v:[read(record_path(out,t,v)) for t in selected] for v in VARIANTS}
        summary[source]={}
        for variant,values in records.items():
            metrics={m:math.fsum(float(r['row'][m]) for r in values)/len(values) for m in METRICS}
            grouped=defaultdict(list)
            for r in values: grouped[key(r['row'])[2]].append(r)
            macro={m:math.fsum(math.fsum(float(r['row'][m]) for r in group)/len(group) for group in grouped.values())/len(grouped) for m in METRICS}
            delta={m:math.fsum(float(a['row'][m])-float(b['row'][m]) for a,b in zip(values,records['baseline']))/len(values) for m in METRICS}
            fallback=sum(str(r['row'].get('dynamic_threshold_source','')).startswith('fallback') for r in values)
            summary[source][variant]=dict(tasks=len(values),mean=metrics,class_macro=macro,paired_delta_vs_baseline=delta,fallback_tasks=fallback,
                actual_batches=dict(Counter(str(r['actual_frame_batch']) for r in values)))
            rows.extend(dict(r['row'],variant=variant,actual_frame_batch=r['actual_frame_batch']) for r in values)
    result=dict(schema='five_source_prompt_type_merged_v2',protocol_sha256=digest,task_count=519,controls_excluded=True,per_source=summary,per_case_class=rows)
    atomic_new(out/'merged_metrics.json',result);return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','control','worker','status','merge'))
    parser.add_argument('--out',required=True,type=Path);parser.add_argument('--inventory',type=Path);parser.add_argument('--job',type=Path)
    args=parser.parse_args(argv);os.chdir(ROOT)
    if args.mode == 'prepare': result=prepare(args.out,args.inventory)
    elif args.mode in ('control','worker'): result=worker(args.out,read(args.job),args.mode == 'control')
    elif args.mode == 'status': result=status(args.out)
    else: result=merge(args.out)
    if args.mode != 'merge': print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__ == '__main__': main()
