"""Independent Bone grid overlapping CPU work through exclusive GPU leases."""
import ast
import functools
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import types

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.bone_metric_speed_grid import engine as prior_engine
from tools.bone_shared_floor_restore_grid import recovery_source

PARENT_SHA='0e96733ba06c4bf01b88a478d0dca4fcadb870f3a630556a0621a2a5c109d977'
NATIVE_SHA='b1cedcb405e26df9c07f451d1eac01c9cccd2dd53f5f87fb3fa70c674eca12d2'
HOST_SLOTS=3
MIN_HOST_BYTES=128*1024**3
PROCESS_NAMES=('bone_floor_memory_wait_recovery.py','bone_shared_floor_grid.py',
               'bone_shared_floor_restore_grid.py','bone_shared_floor_probe_grid.py',
               'bone_metric_speed_grid.py','bone_gpu_pipeline_grid.py')


def handoff_helpers(source=None):
    """Compile guarded handoff functions into their own globals, never ancestors'."""
    source=(ROOT/'tools/bone_metric_speed_grid.py').read_bytes().decode() if source is None else source
    if hashlib.sha256(source.encode()).hexdigest()!=PARENT_SHA:
        raise ValueError('frozen metric parent source changed')
    tree=ast.parse(source)
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name!='engine']
    constants={'CURRENT_SCRIPT':'bone_metric_speed_grid.py','ENTRY':'bone_gpu_pipeline_grid.py',
               'HANDOFF':'output/bone_cpugpu_handoff_20261028.json',
               'STOPPED':'output/bone_cpugpu_handoff_stopped_20261028.json',
               'DRAINED':'output/bone_cpugpu_handoff_drained_20261028.json'}
    replacements={'bone_metric_checkpoint_stop_':'bone_cpugpu_checkpoint_stop_',
                  'output/bone_shared_floor_probe_pipeline_20261027.log':'output/bone_metricfast_pipeline_20261028.log'}
    counts={name:0 for name in replacements};names=0
    for function in functions:
        for node in ast.walk(function):
            if isinstance(node,ast.Constant) and isinstance(node.value,str) and node.value in replacements:
                counts[node.value]+=1;node.value=replacements[node.value]
            if (function.name=='matching_processes' and isinstance(node,ast.Assign)
                    and any(isinstance(t,ast.Name) and t.id=='names' for t in node.targets)):
                if not isinstance(node.value,ast.Set):raise ValueError('parent process-name anchor changed')
                node.value=ast.Set(elts=[ast.Constant(v) for v in PROCESS_NAMES]);names+=1
    if names!=1 or any(count!=1 for count in counts.values()):raise ValueError('parent handoff AST anchor changed')
    helper=types.ModuleType('bone_cpugpu_private_handoff')
    helper.__dict__.update(ROOT=ROOT,prior_engine=prior_engine,os=os,Path=Path,signal=signal,
                           sys=sys,time=time,json=json,**constants)
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions,type_ignores=[])),__file__,'exec'),helper.__dict__)
    return helper


def compile_pipeline_run(value,source=None):
    """Prove the inherited run before changing its single runtime import."""
    if source is None:
        raw=(ROOT/'tools/bone_shared_floor_grid.py').read_bytes()
        if hashlib.sha256(raw).hexdigest()!=NATIVE_SHA:raise ValueError('frozen native runner changed')
        source=recovery_source(raw.decode())
    tree=ast.parse(source)
    runs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run']
    if len(runs)!=1:raise ValueError('parent run AST anchor changed')
    run=runs[0]
    imports=[n for n in ast.walk(run) if isinstance(n,ast.ImportFrom) and n.module=='tools.bone_shared_floor_restore_runtime']
    if len(imports)!=1:raise ValueError('parent run runtime import anchor changed')
    imports[0].module='tools.bone_metric_speed_runtime'
    module=ast.fix_missing_locations(ast.Module(body=[run],type_ignores=[]))
    namespace={};exec(compile(module,__file__,'exec'),namespace)
    inherited=value.run.__code__;candidate=namespace['run'].__code__
    def shape(code):
        return (code.co_code,code.co_names,code.co_varnames,
                tuple(shape(v) if isinstance(v,types.CodeType) else v for v in code.co_consts))
    if shape(inherited)!=shape(candidate):raise ValueError('inherited metric run AST anchor differs')
    imports[0].module='tools.bone_gpu_pipeline_runtime'
    exec(compile(module,__file__,'exec'),value.__dict__)
    return value.run


def configured_run(native):
    @functools.wraps(native)
    def run(job):
        from tools.bone_gpu_pipeline_runtime import configure_job
        configure_job(job)
        return native(job)
    return run


def checked_class_records(value,native):
    def records(out,control=False):
        result=native(out,control)
        from tools.bone_gpu_pipeline_runtime import validate_pipeline_record
        for item in result.values():
            run=Path(item['marker']).parents[1]
            spec=value.b.read(out/'job_specs'/(run.name+'.json'))
            validate_pipeline_record(item['record'],spec)
        return result
    return records


def require_exclusive_leases(records):
    """Check cross-process physical-card intervals; nested leases share an owner."""
    events={}
    for item in records.values():
        proof=item['record']['pipeline_execution']
        for event in [proof['startup_lease'],*proof['native_probe_leases'],*proof['raw_events']]:
            if event['nested']:continue
            existing=events.get(event['id'])
            if existing is not None and existing!=event:raise ValueError('conflicting pipeline lease id')
            events[event['id']]=event
    by_gpu={}
    for event in events.values():
        if event['end_monotonic']<event['start_monotonic']:raise ValueError('invalid pipeline lease interval')
        by_gpu.setdefault(event['gpu'],[]).append(event)
    for values in by_gpu.values():
        values.sort(key=lambda e:e['start_monotonic'])
        for index,event in enumerate(values):
            for prior in values[:index]:
                if prior['pid']!=event['pid'] and prior['end_monotonic']>event['start_monotonic']:
                    raise ValueError('physical GPU pipeline leases overlap across workers')


def host_available_bytes():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):return int(line.split()[1])*1024
    raise ValueError('host available RAM cannot be established')


def resource_eligible(value,job,gpu,slot,available,now,active,host_bytes):
    return (type(gpu) is int and gpu in value.ALLOWED and type(slot) is int and 0<=slot<HOST_SLOTS
            and f'{gpu}:{slot}' not in active and host_bytes>=MIN_HOST_BYTES
            and value.eligible(job,gpu,available,now))


def report_fatal(*parts):
    # A broken pipeline log must not interrupt natural worker drainage.
    try:print('PIPELINE_SCHEDULER_FATAL_DRAIN',*parts,flush=True)
    except OSError:pass


def install_scheduler(value):
    native_validate=value.validate_attempt
    def validate_attempt(job,canonical,out):
        native_validate(job,canonical,out)
        if (type(job.get('slot')) is not int or not 0<=job['slot']<HOST_SLOTS
                or type(job.get('gpu')) is not int or job['gpu'] not in value.ALLOWED):
            raise ValueError('invalid signed physical GPU/host slot')
    value.validate_attempt=validate_attempt

    def make_attempt(out,job,gpu,available,completed,slot=0):
        completed=value.class_records(out,job['control'])
        pending=[x for x in job['pending'] if (job['model'],value.p.key(x['task'])) not in completed]
        if not pending:return None
        tag=job['name']+'_a'+str(job['attempt'])
        run=out/'runs'/tag;spec=out/'job_specs'/(tag+'.json');log=out/'logs'/(tag+'.log')
        if run.exists() or spec.exists() or log.exists():raise ValueError('attempt paths already exist')
        actual=dict(job,pending=pending,task_count=len(pending),gpu=gpu,slot=slot,gpu_uuid=available[gpu]['uuid'],
                    run=str(run),spec=str(spec),log=str(log))
        validate_attempt(actual,job,out)
        value.atomic_new(spec,actual);log.parent.mkdir(parents=True,exist_ok=True)
        return actual
    value.make_attempt=make_attempt

    def schedule(out,control):
        jobs=value.b.read(out/'jobs.json')['controls' if control else 'cases']
        pending=[dict(j) for j in jobs];active={};failed=[];committed={};cause=None;status_writable=True
        while pending or active:
            stage='class_records'
            try:
                if not failed:committed=value.class_records(out,control)
                for key,(proc,job,handle) in list(active.items()):
                    stage='worker_poll'
                    if proc.poll() is None:continue
                    del active[key];handle.close()
                    # A fatal queue error ends validation and dispatch. Existing
                    # peers finish naturally; their files remain available for audit.
                    if failed:continue
                    stage='completed_class_records'
                    done=value.class_records(out,control)
                    remaining=[x for x in job['pending'] if (job['model'],value.p.key(x['task'])) not in done]
                    stage='worker_log'
                    text=Path(job['log']).read_text(errors='replace')[-14000:]
                    outcome=value.attempt_outcome(proc.returncode,text,bool(remaining))
                    if outcome=='fatal':failed.append(dict(name=job['name'],log=job['log'],returncode=proc.returncode,gpu=job['gpu'],slot=job['slot']))
                    elif outcome=='retry':
                        next_job=value.retry(job,proc.returncode,text,time.time());pending.append(next_job)
                        print('SHARED_RESOURCE_REQUEUE',job['name'],next_job['frame_batch'],next_job['minimum_free_mib'],flush=True)
                    else:print('SHARED_CASE_DONE',job['name'],flush=True)
                if not failed:
                    stage='class_records';committed=value.class_records(out,control)
                    stage='gpu_snapshot';available=value.g.gpu_snapshot()
                    # Controls share a card to exercise exclusion; production fills
                    # each physical card's first slot before adding another worker.
                    positions=([(gpu,slot) for gpu in value.ALLOWED for slot in range(HOST_SLOTS)] if control
                               else [(gpu,slot) for slot in range(HOST_SLOTS) for gpu in value.ALLOWED])
                    for gpu,slot in positions:
                        if failed:break
                        stage='host_resources';ram=host_available_bytes()
                        selected=next((j for j in pending if resource_eligible(value,j,gpu,slot,available,time.time(),active,ram)),None)
                        if selected is None:continue
                        pending.remove(selected)
                        if all((selected['model'],value.p.key(x['task'])) in committed for x in selected['pending']):continue
                        stage='make_attempt';job=make_attempt(out,selected,gpu,available,committed,slot=slot)
                        if job is None:continue
                        stage='log_open';handle=Path(job['log']).open('x')
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
                        try:
                            proc=value.subprocess.Popen([sys.executable,value.__file__,'worker','--job',job['spec']],cwd=ROOT,env=env,stdout=handle,stderr=value.subprocess.STDOUT)
                        except Exception as error:
                            handle.close()
                            failed.append(dict(name=job['name'],log=job['log'],gpu=gpu,slot=slot,spawn_error=repr(error)))
                            cause=error;break
                        active[f'{gpu}:{slot}']=(proc,job,handle)
                        print('SHARED_START',gpu,proc.pid,job['name'],job['task_count'],'batch',job['frame_batch'],'host_slot',slot,flush=True)
            except Exception as error:
                failed.append(dict(stage=stage,scheduler_error=repr(error)))
                cause=cause or error
                report_fatal(stage,repr(error))
            if status_writable:
                try:
                    value.b.write(out/'status.json',dict(time=time.strftime('%F %T'),phase='controls' if control else 'full',pending=len(pending),
                        running={key:dict(gpu=job['gpu'],slot=job['slot'],pid=proc.pid,spec=job['spec'],log=job['log'],name=job['name']) for key,(proc,job,handle) in active.items()},
                        failed=failed,committed_classes=len(committed)))
                except Exception as error:
                    failed.append(dict(stage='status_write',scheduler_error=repr(error)))
                    cause=cause or error;status_writable=False
                    report_fatal('status_write',repr(error))
            if failed and not active:raise RuntimeError('pipeline non-resource failure; drained workers and retained evidence: '+repr(failed)) from cause
            if pending or active:time.sleep(10)
    value.schedule=schedule


def engine():
    helpers=handoff_helpers();value=prior_engine();parent=ROOT/value.OUT
    native_run=compile_pipeline_run(value)
    value.OUT='output/bone_v92e360_v102e490_floor_grid_20261028_cpugpu'
    value.__file__=__file__
    value.SOURCES+=('tools/bone_gpu_pipeline_runtime.py','tools/bone_gpu_pipeline_grid.py')
    value.run=configured_run(native_run)
    value.related=lambda:[str(r['pid']) for r in helpers.matching_processes()]
    value.finish_parent=lambda:helpers.finish_parent(value)
    value.collect_parent=helpers.collect_parent
    native_signature=value.signature
    def signature(out):
        prior_engine().verify(parent)
        result=native_signature(out);result['pipeline_parent_protocol']=value.sha(parent/'protocol.json')
        result['gpu_pipeline_policy']=dict(allowed_gpus=list(value.ALLOWED),host_slots_per_gpu=HOST_SLOTS,minimum_host_available_bytes=MIN_HOST_BYTES,
                                           exclusive_gpu_producers=True,ordinary_minimum_free_mib=30720)
        return result
    value.signature=signature
    install_scheduler(value)
    value.class_records=checked_class_records(value,value.class_records)
    native_controls=value.check_controls
    def controls(out):
        records=native_controls(out)
        require_exclusive_leases(records)
        return records
    value.check_controls=controls
    return value


def freeze_parent():return handoff_helpers().freeze_parent()


if __name__=='__main__':
    if len(sys.argv)==2 and sys.argv[1]=='freeze-parent':freeze_parent()
    else:engine().main()
