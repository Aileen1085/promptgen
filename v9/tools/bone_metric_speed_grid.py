"""Independent exact-metric recovery of the frozen Bone shared-stage grid."""
import ast
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.bone_shared_floor_probe_grid import engine as prior_engine
from tools.bone_shared_floor_restore_grid import recovery_source
CURRENT_SCRIPT='bone_shared_floor_probe_grid.py'
ENTRY='bone_metric_speed_grid.py'
HANDOFF='output/bone_metric_handoff_20261028.json'
STOPPED='output/bone_metric_handoff_stopped_20261028.json'
DRAINED='output/bone_metric_handoff_drained_20261028.json'


def combine_completed(serial,records):
    p=prior_engine().p;result={}
    for item in serial:
        identity=(item['model'],item['floor'],p.key(item['row']))
        if identity in result:raise ValueError('duplicate original reused row')
        result[identity]=item['row']
    for item in records.values():
        record=item['record']
        for value in record['results']:
            identity=(record['model'],value['floor'],p.key(value['row']))
            if identity in result:raise ValueError('duplicate complete-class reused row')
            result[identity]=value['row']
    return result


def inherited_history(spec,text,when,wait_status,intentional=False):
    s=prior_engine()
    if intentional:
        if wait_status==0:return dict(spec)  # natural completion can win the signal race
        if wait_status==signal.SIGTERM:return dict(spec,intentional_checkpoint_stop=True)
        # A resource exit may win the same race; use its actual wait status/log,
        # rather than relabeling it as the requested intentional retirement.
    if wait_status==0:return dict(spec)
    code=os.waitstatus_to_exitcode(wait_status)
    oom='cuda out of memory' in text.lower() or 'OutOfMemoryError' in text
    if code!=75 and not oom:raise ValueError('unresolved non-resource parent exit')
    return dict(s.retry(spec,code,text,when),inherited_failure_time=when)


def recorded_resource_exit(name,text,launcher):
    # The frozen native scheduler prints requeue only after an actual resource
    # exit. Require its last event for this group; absence of Traceback is not proof.
    last=None
    for line in launcher.splitlines():
        words=line.split()
        if len(words)>3 and words[0]=='SHARED_START' and words[3]==name:last='start'
        elif len(words)>1 and words[0]=='SHARED_RESOURCE_REQUEUE' and words[1]==name:last='resource'
        elif len(words)>1 and words[0]=='SHARED_CASE_DONE' and words[1]==name:last='done'
    if last!='resource':raise ValueError('parent exit lacks positive resource requeue evidence')
    if 'cuda out of memory' in text.lower() or 'OutOfMemoryError' in text:return 256
    return 75<<8


def validate_scheduler(info,root,uid,frozen=True):
    args=info['args']
    if (info['uid']!=uid or Path(info['cwd'])!=root
            or (info['state']!='T' if frozen else info['state'] in ('T','Z'))
            or len(args)!=3 or (root/args[1]).resolve()!=(root/'tools'/CURRENT_SCRIPT).resolve() or args[2]!='wait-launch'
            or not info.get('starttime')):raise ValueError('frozen own scheduler identity mismatch')


def process_record(pid):
    proc=Path('/proc')/str(pid)
    if not proc.exists():return None
    fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
    record=dict(pid=int(pid),uid=proc.stat().st_uid,state=fields[0],starttime=fields[19],exit_status=int(fields[49]))
    if record['state']!='Z':
        record.update(args=(proc/'cmdline').read_bytes().decode().rstrip('\0').split('\0'),cwd=str((proc/'cwd').resolve()))
    return record


def matching_processes():
    names={'bone_floor_memory_wait_recovery.py','bone_shared_floor_grid.py','bone_shared_floor_restore_grid.py',CURRENT_SCRIPT,ENTRY}
    result=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:
            r=process_record(proc.name)
            if r and r['state']!='Z' and len(r['args'])>=3 and Path(r['args'][1]).name in names and r['args'][2] in ('worker','wait-launch','launch','controls'):result.append(r)
        except (OSError,ValueError):continue
    return result


def _same_process(now,old):
    return now is not None and all(now.get(k)==old.get(k) for k in ('pid','uid','starttime')) and (now['state']=='Z' or all(now.get(k)==old.get(k) for k in ('args','cwd')))


def freeze_parent():
    s=prior_engine();q=s.ROOT/s.OUT;s.verify(q);s.check_controls(q)
    if (ROOT/HANDOFF).exists() or (ROOT/STOPPED).exists():raise FileExistsError('handoff already recorded; do not repeat freeze')
    processes=matching_processes()
    if any(Path(r['args'][1]).name!=CURRENT_SCRIPT for r in processes):raise ValueError('unexpected or duplicate predecessor/new queue')
    schedulers=[r for r in processes if r['args'][2]=='wait-launch']
    if len(schedulers)!=1:raise ValueError('expected one own current scheduler')
    scheduler=schedulers[0]
    validate_scheduler(scheduler,ROOT,os.getuid(),frozen=False)
    def workers():
        values=[];catalog=s.b.read(q/'jobs.json')
        for r in matching_processes():
            if r['args'][2]!='worker':continue
            if ((ROOT/r['args'][1]).resolve()!=(ROOT/'tools'/CURRENT_SCRIPT).resolve()
                    or r['uid']!=os.getuid() or Path(r['cwd'])!=ROOT or len(r['args'])!=5 or r['args'][3]!='--job'):raise ValueError('unexpected worker identity')
            path=Path(r['args'][4])
            if path.parent!=q/'job_specs':raise ValueError('worker spec outside signed parent')
            spec=s.b.read(path);canonical=next(j for j in catalog['cases'] if j['name']==spec['name']);s.validate_attempt(spec,canonical,q)
            r['spec']=str(path);r['baseline_markers']={str(f):s.sha(f) for f in (Path(spec['run'])/'classes').glob('*.done.json')};values.append(r)
        return values
    before=workers()
    s.atomic_new(ROOT/HANDOFF,dict(time=time.strftime('%F %T'),scheduler=scheduler,workers=before,parent_protocol_sha=s.sha(q/'protocol.json'),parent_state=s.b.read(q/'status.json')))
    if not _same_process(process_record(scheduler['pid']),scheduler):raise ValueError('scheduler changed before freeze')
    os.kill(scheduler['pid'],signal.SIGSTOP);time.sleep(.3)
    frozen=process_record(scheduler['pid']);validate_scheduler(frozen,ROOT,os.getuid())
    s.atomic_new(ROOT/STOPPED,dict(scheduler=frozen,workers=workers(),handoff_sha=s.sha(ROOT/HANDOFF)))
    print('OWN_DISPATCH_FROZEN_CHECKPOINT_HANDOFF_READY',flush=True)


def finish_parent(value):
    s=prior_engine();q=s.ROOT/s.OUT;e=s.b.read(ROOT/STOPPED);scheduler=e['scheduler'];workers=e['workers'];exits={};intents={}
    while len(exits)<len(workers):
        frozen=process_record(scheduler['pid'])
        if not _same_process(frozen,scheduler):raise ValueError('frozen parent scheduler changed')
        validate_scheduler(frozen,ROOT,os.getuid())
        changed=[]
        for old in workers:
            if old['spec'] in exits:continue
            now=process_record(old['pid'])
            if not _same_process(now,old):raise ValueError('captured worker disappeared or identity changed before exit audit')
            if now['state']=='Z':
                exits[old['spec']]=dict(worker=old,wait_status=now['exit_status'],intent=intents.get(old['spec']))
                continue
            spec=s.b.read(old['spec']);markers=list((Path(spec['run'])/'classes').glob('*.done.json'))
            fresh=[m for m in markers if str(m) not in old['baseline_markers']]
            if fresh and old['spec'] not in intents:changed.append((old,fresh))
        if changed:
            checked=s.class_records(q)
            for old,fresh in changed:
                allowed={item['marker']:item for item in checked.values()}
                marker=next((m for m in fresh if str(m) in allowed),None)
                if marker is None:raise ValueError('new boundary marker failed full class validation')
                now=process_record(old['pid'])
                if not _same_process(now,old):raise ValueError('worker changed at checked boundary')
                if now['state']=='Z':continue
                intent=ROOT/'output'/('bone_metric_checkpoint_stop_'+str(old['pid'])+'_20261028.json')
                data=dict(worker=old,marker=str(marker),marker_sha=s.sha(marker),files=s.b.read(marker)['files'],time=time.strftime('%F %T'))
                s.atomic_new(intent,data);intents[old['spec']]=dict(path=str(intent),sha=s.sha(intent))
                os.kill(old['pid'],signal.SIGTERM);print('OWN_WORKER_CHECKPOINT_STOP',old['pid'],marker.name,flush=True)
        if len(exits)<len(workers):time.sleep(2)
    for specpath,exitinfo in exits.items():
        spec=s.b.read(specpath);text=Path(spec['log']).read_text(errors='replace')[-14000:]
        inherited_history(spec,text,Path(spec['log']).stat().st_mtime,exitinfo['wait_status'],bool(exitinfo['intent']))
    s.atomic_new(ROOT/DRAINED,dict(exits=exits,stopped_sha=s.sha(ROOT/STOPPED),all_workers_exited=True))
    now=process_record(scheduler['pid'])
    if not _same_process(now,scheduler):raise ValueError('scheduler identity changed before retirement')
    validate_scheduler(now,ROOT,os.getuid());os.kill(scheduler['pid'],signal.SIGTERM);os.kill(scheduler['pid'],signal.SIGCONT)
    for _ in range(60):
        if not value.related():break
        time.sleep(1)
    value.require_handoff([dict(verified=True)],value.related());time.sleep(3);value.require_handoff([dict(verified=True)],value.related())
    return dict(verified=True,checkpoint_handoff=True,evidence={str(ROOT/n):s.sha(ROOT/n) for n in (HANDOFF,STOPPED,DRAINED)})


def collect_parent():
    s=prior_engine();q=s.ROOT/s.OUT;s.verify(q);s.check_controls(q)
    reuse=s.b.read(q/'reuse_audit.json');records=s.class_records(q);completed=combine_completed(reuse['completed'],records)
    exits=s.b.read(ROOT/DRAINED)['exits'];history={}
    pipeline=ROOT/'output/bone_shared_floor_probe_pipeline_20261027.log'
    launcher=pipeline.read_text(errors='replace')
    for canonical in s.b.read(q/'jobs.json')['cases']:
        attempts=sorted((s.b.read(f) for f in (q/'job_specs').glob(canonical['name']+'_a*.json')),key=lambda j:j['attempt'],reverse=True)
        if attempts:
            spec=attempts[0];log=Path(spec['log']);text=log.read_text(errors='replace')[-14000:]
            exitinfo=exits.get(spec['spec'])
            if exitinfo:status=exitinfo['wait_status'];intent=bool(exitinfo['intent'])
            elif (Path(spec['run'])/'worker_complete.json').exists():status=0;intent=False
            elif all((canonical['model'],s.p.key(item['task'])) in records for item in canonical['pending']):status=0;intent=False
            else:status=recorded_resource_exit(canonical['name'],text,launcher);intent=False
            policy=inherited_history(spec,text,log.stat().st_mtime,status,intent)
        else:policy=dict(canonical,never_launched=True)
        for item in canonical['pending']:
            for floor in item['floors']:history[(canonical['model'],floor,s.p.key(item['task']))]=policy
    files={str(f):s.sha(f) for f in q.rglob('*') if f.is_file()}
    files[str(pipeline)]=s.sha(pipeline)
    files.update({str(ROOT/n):s.sha(ROOT/n) for n in (HANDOFF,STOPPED,DRAINED)})
    for info in exits.values():
        if info['intent']:files[info['intent']['path']]=info['intent']['sha']
    return completed,history,reuse['specs'],files,['complete_shared_classes:'+str(len(records))]


def engine():
    value=prior_engine();parent=ROOT/value.OUT
    value.OUT='output/bone_v92e360_v102e490_floor_grid_20261028_metricfast';value.__file__=__file__
    value.SOURCES+=('tools/bone_fast_metrics.py','tools/bone_metric_speed_runtime.py','tools/bone_metric_speed_grid.py')
    source=(ROOT/'tools/bone_shared_floor_grid.py').read_bytes()
    if hashlib.sha256(source).hexdigest()!='b1cedcb405e26df9c07f451d1eac01c9cccd2dd53f5f87fb3fa70c674eca12d2':raise ValueError('frozen native shared runner changed')
    tree=ast.parse(recovery_source(source.decode()));run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    imports=[n for n in ast.walk(run) if isinstance(n,ast.ImportFrom) and n.module=='tools.bone_shared_floor_restore_runtime']
    if len(imports)!=1:raise ValueError('metric runtime import anchor changed')
    imports[0].module='tools.bone_metric_speed_runtime'
    exec(compile(ast.fix_missing_locations(ast.Module(body=[run],type_ignores=[])),__file__,'exec'),value.__dict__)
    value.related=lambda:[str(r['pid']) for r in matching_processes()]
    value.finish_parent=lambda:finish_parent(value);value.collect_parent=collect_parent
    native_signature=value.signature
    def signature(out):
        prior_engine().verify(parent)
        result=native_signature(out);result['metric_parent_protocol']=value.sha(parent/'protocol.json')
        return result
    value.signature=signature
    native_validate=value.validate_class
    def validate(record,expected):
        result=native_validate(record,expected)
        from tools.bone_metric_speed_runtime import validate_metric_record
        validate_metric_record(record)
        return result
    value.validate_class=validate
    native_controls=value.check_controls
    def controls(out):
        records=native_controls(out)
        from tools.bone_metric_speed_runtime import validate_metric_record
        for item in records.values():validate_metric_record(item['record'],True)
        return records
    value.check_controls=controls
    return value


if __name__=='__main__':
    if len(sys.argv)==2 and sys.argv[1]=='freeze-parent':freeze_parent()
    else:engine().main()
