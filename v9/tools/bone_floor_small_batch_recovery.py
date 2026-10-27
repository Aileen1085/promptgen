"""Independent Bone recovery: batch cap4, true30GiB startup guard, readonly whole shards."""
from __future__ import annotations
import argparse
import inspect
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools import bone_floor_resource_recovery as s
r,g,b,p=s.r,s.g,s.b,s.p
OLD=s.OUT
OUT='output/bone_v92e360_v102e490_floor_grid_20261027_smallbatch'
START_FREE_MIB=30720
BATCH_CAP=4
FILES=s.FILES+('resource_policy_audit.json',)
_PARENT_VERIFY=s.verify
_RESTORE_VERIFY_JOB=r.verify_job
_NATIVE_VALIDATE=g.validate_job_spec
_NATIVE_RUN=g.run


def process_matches(args,pid,self_pid):
    return str(pid)!=str(self_pid) and any(Path(x).name in
        ('bone_floor_grid.py','bone_floor_recovery.py','bone_floor_resource_recovery.py',
         'bone_floor_small_batch_recovery.py') for x in args) and any(x in
        ('prepare','prompt-worker','worker','launch','canary','wait-launch') for x in args)


def related_processes():
    found=[]
    for q in Path('/proc').iterdir():
        if not q.name.isdigit():continue
        try:
            if process_matches((q/'cmdline').read_bytes().decode(errors='replace').split('\0'),q.name,os.getpid()):found.append(q.name)
        except OSError:pass
    return found


def cap_settings(settings):
    value=dict(settings)
    value['frame_batch_ceiling']=min(BATCH_CAP,value['frame_batch'])
    value['inherited_oom_retries']=value['oom_retries']
    value['frame_batch']=value['frame_batch_ceiling']
    return value


def resume_settings(latest=None,oom=False):
    if latest and 'frame_batch_ceiling' in latest:
        value={k:latest[k] for k in ('attempt','oom_retries','frame_batch','frame_batch_ceiling','inherited_oom_retries')}
        value['attempt']+=1
        if oom and value['frame_batch']>1:value['oom_retries']+=1;value['frame_batch']//=2
        return value
    return cap_settings(s.resume_settings(latest,oom=oom))


def validate_batch(job,limits):
    fields=('attempt','oom_retries','frame_batch','frame_batch_ceiling','inherited_oom_retries')
    if any(type(job.get(k)) is not int for k in fields):raise ValueError('invalid batch history types')
    ceiling=job['frame_batch_ceiling'];inherited=job['inherited_oom_retries'];oom=job['oom_retries']
    if (ceiling not in (1,2,4) or inherited<0 or oom<inherited or oom>job['attempt']
        or job['frame_batch']!=max(1,ceiling//2**(oom-inherited))):raise ValueError('invalid actual small batch/OOM history')
    if limits:
        s.validate_resume(job,limits)
        if any(job[k]!=limits[k] for k in ('frame_batch_ceiling','inherited_oom_retries')):raise ValueError('changed frozen proactive batch policy')


def validate_job_spec(job,signed,out):
    out=Path(out)
    if out!=(ROOT/OUT).resolve():return _NATIVE_VALIDATE(job,signed,out)
    for field in ('name','model','variant','source','floor','prepared','task_count','smoke'):
        if job.get(field)!=signed[field]:raise ValueError('worker unsigned field: '+field)
    if job.get('gpu') not in (2,4,5,6) or not isinstance(job.get('gpu_uuid'),str):raise ValueError('prohibited worker GPU')
    paths=g.attempt_paths(out,job['name'],job.get('attempt'))
    if any(job.get(k)!=paths[k] for k in ('run','log')):raise ValueError('worker attempt paths mismatch')
    limits=b.read(out/'reuse_audit.json')['resume'].get(job['name'])
    if limits is None and not (job['smoke'] and job['floor']==.35 and job['task_count']==1):raise ValueError('unsigned pending/canary task')
    validate_batch(job,limits)
    if limits is None and (job['frame_batch_ceiling']!=BATCH_CAP or job['inherited_oom_retries']!=0):raise ValueError('canary batch policy changed')


def resource_source(source):
    guards=("gpu['free']<45000", "available[gpu]['free']<45000", "available[i]['free']>=45000")
    if source.count('45000')!=1 or sum(pattern in source for pattern in guards)!=1:
        raise ValueError('expected exactly one native startup-memory guard')
    return source.replace('45000','START_FREE_MIB')


def verify_job(job):
    tasks=_RESTORE_VERIFY_JOB(job)
    if Path(job['run']).parents[1]==(ROOT/OUT).resolve():
        audit=b.read(Path(job['run'])/'resource_policy_audit.json')
        expected=dict(gpu=job['gpu'],gpu_uuid=job['gpu_uuid'],frame_batch=job['frame_batch'],
            frame_batch_ceiling=job['frame_batch_ceiling'],inherited_oom_retries=job['inherited_oom_retries'],
            oom_retries=job['oom_retries'],minimum_free_mib=START_FREE_MIB)
        if any(audit.get(k)!=v for k,v in expected.items()) or audit.get('actual_start_free_mib',0)<START_FREE_MIB:raise ValueError('actual startup resource audit mismatch')
    return tasks


def signature(out):
    return dict(parent=s.signature(out),small_batch_source=s.sha(__file__),
        minimum_free_mib=START_FREE_MIB,default_frame_batch_cap=BATCH_CAP,allowed_gpus=[2,4,5,6],
        policy='only startup memory guard and proactive frame batch cap change; all native inference/scoring unchanged')


def verify(out):
    if signature(out)!=b.read(out/'protocol.json'):raise ValueError('changed small-batch recovery signature')
    _PARENT_VERIFY(ROOT/OLD)
    s.verify_canaries(ROOT/OLD)
    s.verify_files(b.read(out/'reuse_audit.json')['files'])


def configure(out):
    r.configure(out)
    r.verify=verify;g.verify=verify;g.amos.verify=verify
    g.validate_job_spec=validate_job_spec;r.verify_job=verify_job
    namespace=dict(g.__dict__,START_FREE_MIB=START_FREE_MIB)
    exec(compile(resource_source(inspect.getsource(_NATIVE_RUN)),str(_NATIVE_RUN.__code__.co_filename),'exec'),namespace)
    g.run=namespace['run']


def prepare(out):
    if out.exists():raise FileExistsError('new small-batch output already exists')
    old=ROOT/OLD;_PARENT_VERIFY(old);s.verify_canaries(old)
    state=b.read(old/'status.json');s.require_drained(state,related_processes())
    jobs=b.read(old/'jobs.json');prior=b.read(old/'reuse_audit.json');complete=set();files={};resumes={}
    for j in jobs:
        passed=old/'job_specs'/(j['name']+'.json')
        if passed.exists():
            spec=b.read(passed);_RESTORE_VERIFY_JOB(spec);complete.add(j['name'])
            for f in [passed]+[Path(spec['run'])/n for n in s.FILES]:files[str(f)]=s.sha(f)
        else:
            attempts=[b.read(f) for f in (old/'job_specs').glob(j['name']+'_a*.json')]
            latest=max(attempts,key=lambda t:t['attempt']) if attempts else None
            if latest:
                _NATIVE_VALIDATE(latest,j,old);spec=old/'job_specs'/(j['name']+'_a'+str(latest['attempt'])+'.json');files[str(spec)]=s.sha(spec)
                tail=Path(latest['log']).read_text(errors='replace')[-10000:]
                resumes[j['name']]=resume_settings(latest,oom='out of memory' in tail.lower() or 'OutOfMemoryError' in tail)
            elif j['name'] in prior['resume']:resumes[j['name']]=cap_settings(prior['resume'][j['name']])
            else:resumes[j['name']]=resume_settings()
    info=s.partition(jobs,complete);s.validate_state_partition(state,jobs,complete)
    if not info['new_tasks']:raise ValueError('no pending work')
    for n in ('protocol.json','reuse_audit.json','status.json','jobs.json','manifest.json','refs.json','prompt_budget_audit.json','terminal_oom_evidence_20261027_resourcefix.json'):
        files[str(old/n)]=s.sha(old/n)
    for model in p.MODELS:
        marker=old/('canary_'+model+'.json');files[str(marker)]=s.sha(marker);files.update(b.read(marker)['files'])
    for f in (old/'logs').glob('*.log'):files[str(f)]=s.sha(f)
    out.mkdir(parents=True)
    for n in ('bone_split.json','manifest.json','jobs.json','refs.json','prompt_budget_audit.json','prepared','prompt_prepared','preparation_status.json','preparation_protocol.json'):
        (out/n).symlink_to(old/n,target_is_directory=(old/n).is_dir())
    (out/'job_specs').mkdir();(out/'jobs').mkdir()
    for j in jobs:
        if j['name'] not in complete:continue
        passed=old/'job_specs'/(j['name']+'.json');spec=b.read(passed)
        (out/'job_specs'/(j['name']+'.json')).symlink_to(passed)
        (out/'jobs'/j['name']).symlink_to(Path(spec['run']),target_is_directory=True)
    g.write_new(out/'reuse_audit.json',dict(info,original_output=OLD,files=files,completed_names=sorted(complete),resume=resumes,
        partial_metrics_reused=False,additional_canaries=2,canaries_in_final_metrics=False,allowed_gpus=[2,4,5,6],minimum_free_mib=START_FREE_MIB,default_frame_batch_cap=BATCH_CAP))
    g.write_new(out/'protocol.json',signature(out));verify(out)
    print('SMALL_BATCH_PREPARED',info,flush=True)


def run(job):
    out=Path(job['run']).parents[1];configure(out);verify(out)
    signed=next(j for j in b.read(out/'jobs.json') if j['name']==job['name']);validate_job_spec(job,signed,out)
    gpu=g.gpu_snapshot()[job['gpu']]
    if gpu['uuid']!=job['gpu_uuid']:raise ValueError('physical GPU UUID changed')
    if gpu['free']<START_FREE_MIB:raise SystemExit(75)
    g.write_new(Path(job['run'])/'resource_policy_audit.json',dict(gpu=job['gpu'],gpu_uuid=job['gpu_uuid'],
        frame_batch=job['frame_batch'],frame_batch_ceiling=job['frame_batch_ceiling'],inherited_oom_retries=job['inherited_oom_retries'],
        oom_retries=job['oom_retries'],minimum_free_mib=START_FREE_MIB,actual_start_free_mib=gpu['free']))
    r.run(job)


def merge(out):
    configure(out);verify(out);engine(out).verify_canaries(out);r.merge(out)
    result=b.read(out/'merged_metrics.json');result['small_batch_recovery']=b.read(out/'reuse_audit.json');b.write(out/'merged_metrics.json',result)


def engine(out):
    namespace=dict(s.__dict__,__file__=__file__,OUT=OUT,OLD=OLD,START_FREE_MIB=START_FREE_MIB,FILES=FILES,
        configure=configure,verify=verify,prepare=prepare,run=run,merge=merge,related_processes=related_processes,resume_settings=resume_settings)
    for name in ('canary','launch'):
        exec(compile(resource_source(inspect.getsource(getattr(s,name))),__file__,'exec'),namespace)
    exec(compile(inspect.getsource(s.verify_canaries),__file__,'exec'),namespace)
    return SimpleNamespace(**namespace)


def wait_launch(out,ids):
    if out.exists():raise FileExistsError('do not relaunch existing output')
    while True:
        processes=related_processes()
        for pid in processes:
            try:args=(Path('/proc')/pid/'cmdline').read_bytes().decode(errors='replace').split('\0')
            except OSError:continue
            if any(Path(x).name==Path(__file__).name for x in args):raise ValueError('duplicate small-batch queue')
        if not processes:break
        print('WAIT_CURRENT_DRAINING',len(processes),flush=True);time.sleep(30)
    s.require_drained(b.read(ROOT/OLD/'status.json'),[]);time.sleep(5)
    s.require_drained(b.read(ROOT/OLD/'status.json'),related_processes());prepare(out)
    while True:
        try:engine(out).launch(out,ids);return
        except SystemExit as e:
            if e.code!=75 or (out/'status.json').exists():raise
            print('WAIT_SMALL_BATCH_CANARY_MEMORY',flush=True);time.sleep(30)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','worker','launch','wait-launch','status','merge','canary'))
    parser.add_argument('--out',default=OUT);parser.add_argument('--job');parser.add_argument('--gpus',default='2,4,5,6')
    a=parser.parse_args();out=(ROOT/a.out).resolve()
    if out!=(ROOT/OUT).resolve():raise ValueError('small-batch recovery requires its dedicated output')
    if a.mode=='prepare':prepare(out)
    elif a.mode=='worker':run(b.read(a.job))
    elif a.mode=='wait-launch':wait_launch(out,s.gpu_ids(a.gpus))
    elif a.mode=='launch':engine(out).launch(out,s.gpu_ids(a.gpus))
    elif a.mode=='canary':engine(out).canary(out,s.gpu_ids(a.gpus))
    elif a.mode=='merge':merge(out)
    else:configure(out);print(json.dumps(g.status(out),indent=2))


if __name__=='__main__':main()
