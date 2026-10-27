"""Bone whole-shard recovery: memory OOM waits without halting other pending work."""
from __future__ import annotations
import argparse
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools import bone_floor_small_batch_recovery as n
r,g,b,p=n.r,n.g,n.b,n.p
s=n.s
OLD=n.OUT
OUT='output/bone_v92e360_v102e490_floor_grid_20261027_memorywait'
START_FREE_MIB=30720
LARGE_FREE_MIB=45000
FILES=n.FILES
_PARENT_VERIFY=n.verify
_PARENT_JOB=n.verify_job
_PARENT_VALIDATE=n.validate_job_spec
_NATIVE_RUN=n._NATIVE_RUN


def process_matches(args,pid,self_pid):
    return str(pid)!=str(self_pid) and any(Path(x).name in
        ('bone_floor_grid.py','bone_floor_recovery.py','bone_floor_resource_recovery.py',
         'bone_floor_small_batch_recovery.py','bone_floor_memory_wait_recovery.py') for x in args) and any(
        x in ('prepare','prompt-worker','worker','launch','canary','wait-launch') for x in args)


def related_processes():
    found=[]
    for q in Path('/proc').iterdir():
        if not q.name.isdigit():continue
        try:
            if process_matches((q/'cmdline').read_bytes().decode(errors='replace').split('\0'),q.name,os.getpid()):found.append(q.name)
        except OSError:pass
    return found


def memory_defaults(value):
    return dict(dict(minimum_free_mib=START_FREE_MIB,memory_wait_retries=0,not_before=0.,
        avoid_gpu_until=0.,last_failed_gpu=None),**value)


def resume_settings(latest=None,oom=False,now=None):
    now=time.time() if now is None else now
    value=n.resume_settings(latest,oom=oom)
    policy={k:latest[k] for k in ('minimum_free_mib','memory_wait_retries','not_before','avoid_gpu_until','last_failed_gpu') if latest and k in latest}
    result=memory_defaults(value);result.update(policy)
    if oom and latest and latest['frame_batch']==1:
        result['oom_retries']+=1
        result.update(minimum_free_mib=LARGE_FREE_MIB,memory_wait_retries=result['memory_wait_retries']+1,
            last_failed_gpu=latest['gpu'],not_before=now+60,avoid_gpu_until=now+600)
    return result


def retry_job(job,returncode,text,now):
    if returncode==0:return None
    oom='cuda out of memory' in text.lower() or 'OutOfMemoryError' in text
    if returncode!=75 and not oom:return None
    value=dict(job,**resume_settings(job,oom=oom,now=now))
    if not oom or job['frame_batch']>1:value['not_before']=now+20
    return value


def select_pending(pending,gpu,available,now):
    for i,job in enumerate(pending):
        if now<job['not_before'] or available[gpu]['free']<job['minimum_free_mib']:continue
        if job['last_failed_gpu']==gpu and now<job['avoid_gpu_until']:continue
        return pending.pop(i)
    return None


def validate_memory_policy(job,limits):
    minimum=job.get('minimum_free_mib');wait=job.get('memory_wait_retries')
    if minimum not in (START_FREE_MIB,LARGE_FREE_MIB) or type(wait) is not int or wait<0:raise ValueError('invalid memory wait policy')
    if wait and minimum!=LARGE_FREE_MIB:raise ValueError('batch1 OOM must wait for larger memory')
    if wait>job.get('oom_retries',0) or (wait and job.get('last_failed_gpu') is None):raise ValueError('memory wait history has no observed OOM/GPU')
    if job.get('last_failed_gpu') not in (None,2,4,5,6):raise ValueError('invalid failed GPU binding')
    for field in ('not_before','avoid_gpu_until'):
        value=job.get(field)
        if type(value) not in (int,float) or not math.isfinite(value) or value<0:raise ValueError('invalid resource retry deadline')
    if limits and (minimum<limits['minimum_free_mib'] or wait<limits['memory_wait_retries']):raise ValueError('frozen memory requirement was lowered')


def validate_job_spec(job,signed,out):
    out=Path(out)
    if out!=(ROOT/OUT).resolve():return _PARENT_VALIDATE(job,signed,out)
    for field in ('name','model','variant','source','floor','prepared','task_count','smoke'):
        if job.get(field)!=signed[field]:raise ValueError('worker unsigned field: '+field)
    if job.get('gpu') not in (2,4,5,6) or not isinstance(job.get('gpu_uuid'),str):raise ValueError('prohibited GPU')
    paths=g.attempt_paths(out,job['name'],job.get('attempt'))
    if any(job.get(k)!=paths[k] for k in ('run','log')):raise ValueError('worker attempt paths mismatch')
    limits=b.read(out/'reuse_audit.json')['resume'].get(job['name'])
    if limits is None and not (job['smoke'] and job['floor']==.35 and job['task_count']==1):raise ValueError('unsigned canary/pending task')
    n.validate_batch(job,limits);validate_memory_policy(job,limits)
    if limits is None and (job['frame_batch_ceiling']!=4 or job['inherited_oom_retries']!=0):raise ValueError('changed canary batch cap')


def artifact_names(job):
    return FILES if Path(job['run']).parents[1] in ((ROOT/n.OUT).resolve(),(ROOT/OUT).resolve()) else s.FILES


def verify_job(job):
    tasks=_PARENT_JOB(job)
    if Path(job['run']).parents[1]==(ROOT/OUT).resolve():
        audit=b.read(Path(job['run'])/'resource_policy_audit.json')
        expected={k:job[k] for k in ('gpu','gpu_uuid','frame_batch','frame_batch_ceiling','inherited_oom_retries','oom_retries','minimum_free_mib','memory_wait_retries')}
        if any(audit.get(k)!=v for k,v in expected.items()) or audit.get('actual_start_free_mib',0)<job['minimum_free_mib']:raise ValueError('actual memory/startup audit mismatch')
    return tasks


def signature(out):
    return dict(parent=n.signature(out),memory_wait_source=s.sha(__file__),base_free_mib=START_FREE_MIB,
        batch1_oom_free_mib=LARGE_FREE_MIB,allowed_gpus=[2,4,5,6],cooldown_seconds=60,alternate_gpu_seconds=600,
        policy='resource-only OOM requeue; batch1 waits for larger actual memory while unrelated shards continue; scientific inference unchanged')


def verify(out):
    if signature(out)!=b.read(out/'protocol.json'):raise ValueError('changed memory-wait signature')
    _PARENT_VERIFY(ROOT/OLD);n.engine(ROOT/OLD).verify_canaries(ROOT/OLD)
    s.verify_files(b.read(out/'reuse_audit.json')['files'])


def configure(out,minimum=START_FREE_MIB):
    r.configure(out);r.verify=verify;g.verify=verify;g.amos.verify=verify
    r.verify_job=verify_job;g.validate_job_spec=validate_job_spec
    namespace=dict(g.__dict__,START_FREE_MIB=minimum)
    exec(compile(n.resource_source(inspect.getsource(_NATIVE_RUN)),_NATIVE_RUN.__code__.co_filename,'exec'),namespace)
    g.run=namespace['run']


def prepare(out):
    if out.exists():raise FileExistsError('new recovery output exists')
    old=ROOT/OLD;_PARENT_VERIFY(old);n.engine(old).verify_canaries(old)
    state=b.read(old/'status.json');s.require_drained(state,related_processes())
    jobs=b.read(old/'jobs.json');prior=b.read(old/'reuse_audit.json');complete=set();files={};resumes={}
    for j in jobs:
        passed=old/'job_specs'/(j['name']+'.json')
        if passed.exists():
            spec=b.read(passed);_PARENT_JOB(spec);complete.add(j['name'])
            for f in [passed]+[Path(spec['run'])/name for name in artifact_names(spec)]:files[str(f)]=s.sha(f)
        else:
            attempts=[b.read(f) for f in (old/'job_specs').glob(j['name']+'_a*.json')]
            latest=max(attempts,key=lambda a:a['attempt']) if attempts else None
            if latest:
                _PARENT_VALIDATE(latest,j,old);f=old/'job_specs'/(j['name']+'_a'+str(latest['attempt'])+'.json');files[str(f)]=s.sha(f)
                tail=Path(latest['log']).read_text(errors='replace')[-10000:]
                resumes[j['name']]=resume_settings(latest,oom='cuda out of memory' in tail.lower() or 'OutOfMemoryError' in tail)
            elif j['name'] in prior['resume']:resumes[j['name']]=memory_defaults(dict(prior['resume'][j['name']]))
            else:resumes[j['name']]=resume_settings()
    info=s.partition(jobs,complete);s.validate_state_partition(state,jobs,complete)
    if not info['new_tasks']:raise ValueError('no pending work')
    for name in ('protocol.json','reuse_audit.json','status.json','jobs.json','manifest.json','refs.json','prompt_budget_audit.json','terminal_oom_evidence_20261027_smallbatch.json'):
        files[str(old/name)]=s.sha(old/name)
    for model in p.MODELS:
        marker=old/('canary_'+model+'.json');files[str(marker)]=s.sha(marker);files.update(b.read(marker)['files'])
    for f in (old/'logs').glob('*.log'):files[str(f)]=s.sha(f)
    out.mkdir(parents=True)
    for name in ('bone_split.json','manifest.json','jobs.json','refs.json','prompt_budget_audit.json','prepared','prompt_prepared','preparation_status.json','preparation_protocol.json'):
        (out/name).symlink_to(old/name,target_is_directory=(old/name).is_dir())
    (out/'jobs').mkdir();(out/'job_specs').mkdir()
    for j in jobs:
        if j['name'] not in complete:continue
        f=old/'job_specs'/(j['name']+'.json');spec=b.read(f)
        (out/'job_specs'/f.name).symlink_to(f);(out/'jobs'/j['name']).symlink_to(Path(spec['run']),target_is_directory=True)
    g.write_new(out/'reuse_audit.json',dict(info,original_output=OLD,files=files,completed_names=sorted(complete),resume=resumes,
        partial_metrics_reused=False,additional_canaries=2,canaries_in_final_metrics=False,base_free_mib=START_FREE_MIB,batch1_oom_free_mib=LARGE_FREE_MIB))
    g.write_new(out/'protocol.json',signature(out));verify(out);print('MEMORY_WAIT_PREPARED',info,flush=True)


def run(job):
    out=Path(job['run']).parents[1];configure(out,job['minimum_free_mib']);verify(out)
    signed=next(j for j in b.read(out/'jobs.json') if j['name']==job['name']);validate_job_spec(job,signed,out)
    gpu=g.gpu_snapshot()[job['gpu']]
    if gpu['uuid']!=job['gpu_uuid']:raise ValueError('physical GPU changed')
    if gpu['free']<job['minimum_free_mib']:raise SystemExit(75)
    audit={k:job[k] for k in ('gpu','gpu_uuid','frame_batch','frame_batch_ceiling','inherited_oom_retries','oom_retries','minimum_free_mib','memory_wait_retries')}
    g.write_new(Path(job['run'])/'resource_policy_audit.json',dict(audit,actual_start_free_mib=gpu['free']))
    r.run(job)


def engine(out):
    namespace=dict(s.__dict__,__file__=__file__,OUT=OUT,OLD=OLD,FILES=FILES,START_FREE_MIB=START_FREE_MIB,
        configure=configure,verify=verify,related_processes=related_processes,resume_settings=resume_settings)
    exec(compile(n.resource_source(inspect.getsource(s.canary)),__file__,'exec'),namespace)
    exec(compile(inspect.getsource(s.verify_canaries),__file__,'exec'),namespace)
    return SimpleNamespace(**namespace)


def merge(out):
    configure(out);verify(out);engine(out).verify_canaries(out);r.merge(out)
    result=b.read(out/'merged_metrics.json');result['memory_wait_recovery']=b.read(out/'reuse_audit.json');b.write(out/'merged_metrics.json',result)


def launch(out,ids):
    configure(out);verify(out);s.require_drained(b.read(ROOT/OLD/'status.json'),related_processes())
    if (out/'status.json').exists():raise ValueError('do not relaunch started recovery')
    import fcntl
    lock=(out/'bone_schedule.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    engine(out).canary(out,ids);engine(out).verify_canaries(out)
    jobs=b.read(out/'jobs.json');reuse=b.read(out/'reuse_audit.json');complete=set(reuse['completed_names'])
    for smoke in (True,False):
        phase='smoke' if smoke else 'full';pending=[dict(j,**reuse['resume'][j['name']]) for j in s.pending_jobs(jobs,complete,smoke)]
        running={};failed=[];done=sum(j['smoke']==smoke and j['name'] in complete for j in jobs)
        b.write(out/'queue_phase.json',dict(phase=phase,readonly_reused_jobs=done))
        while pending or running:
            for gpu,(proc,job,handle) in list(running.items()):
                if proc.poll() is None:continue
                handle.close();del running[gpu]
                if proc.returncode:
                    text=Path(job['log']).read_text(errors='replace')[-10000:];retry=retry_job(job,proc.returncode,text,time.time())
                    if retry is None:failed.append(dict(name=job['name'],log=job['log'],returncode=proc.returncode))
                    else:pending.append(retry);print('MEMORY_RESOURCE_REQUEUE',job['name'],'batch',retry['frame_batch'],'min_free',retry['minimum_free_mib'],'attempt',retry['attempt'],flush=True)
                else:
                    verify_job(job);g.write_new(out/'job_specs'/(job['name']+'.json'),job);done+=1;print('DONE',job['name'],job['task_count'],flush=True)
            available=g.gpu_snapshot()
            for gpu in ids:
                if failed or gpu in running:continue
                job=select_pending(pending,gpu,available,time.time())
                if job is None:continue
                paths=g.attempt_paths(out,job['name'],job['attempt']);job.update(gpu=gpu,gpu_uuid=available[gpu]['uuid'],run=paths['run'],log=paths['log'])
                validate_job_spec(job,next(j for j in jobs if j['name']==job['name']),out)
                if Path(job['run']).exists():raise ValueError('attempt output already exists')
                g.write_new(paths['spec'],job);Path(paths['log']).parent.mkdir(exist_ok=True);handle=Path(paths['log']).open('x')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
                proc=subprocess.Popen([sys.executable,__file__,'worker','--job',paths['spec']],cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
                running[gpu]=(proc,job,handle);print('START',phase,'gpu',gpu,'pid',proc.pid,job['name'],'batch',job['frame_batch'],'min_free',job['minimum_free_mib'],flush=True)
            b.write(out/'status.json',dict(time=time.strftime('%Y-%m-%d %H:%M:%S'),phase=phase,pending=len(pending),done_in_phase=done,
                running={str(gpu):dict(name=j['name'],pid=pr.pid,log=j['log'],attempt=j['attempt']) for gpu,(pr,j,h) in running.items()},failed=failed,
                memory_wait=[{k:j[k] for k in ('name','minimum_free_mib','not_before','last_failed_gpu','avoid_gpu_until')} for j in pending if j['memory_wait_retries'] or j['not_before']>time.time()]))
            if failed and not running:raise RuntimeError('non-OOM terminal failure; retain all attempts')
            if pending or running:time.sleep(10)
        for j in jobs:
            if j['smoke']==smoke:verify_job(b.read(out/'job_specs'/(j['name']+'.json')))
        print('PHASE_PASSED',phase,flush=True)
    merge(out)


def wait_launch(out,ids):
    if out.exists():raise FileExistsError('new waiter only')
    while True:
        processes=related_processes()
        for pid in processes:
            try:args=(Path('/proc')/pid/'cmdline').read_bytes().decode(errors='replace').split('\0')
            except OSError:continue
            if any(Path(x).name==Path(__file__).name for x in args):raise ValueError('duplicate memory-wait queue')
        if not processes:break
        print('WAIT_OLD_DRAINING',len(processes),flush=True);time.sleep(30)
    s.require_drained(b.read(ROOT/OLD/'status.json'),[]);time.sleep(5);s.require_drained(b.read(ROOT/OLD/'status.json'),related_processes());prepare(out)
    while True:
        try:launch(out,ids);return
        except SystemExit as error:
            if error.code!=75 or (out/'status.json').exists():raise
            print('WAIT_CANARY_MEMORY',flush=True);time.sleep(30)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('prepare','worker','launch','wait-launch','status','merge','canary'))
    parser.add_argument('--out',default=OUT);parser.add_argument('--job');parser.add_argument('--gpus',default='2,4,5,6');a=parser.parse_args();out=(ROOT/a.out).resolve()
    if out!=(ROOT/OUT).resolve():raise ValueError('dedicated output required')
    if a.mode=='prepare':prepare(out)
    elif a.mode=='worker':run(b.read(a.job))
    elif a.mode=='wait-launch':wait_launch(out,s.gpu_ids(a.gpus))
    elif a.mode=='launch':launch(out,s.gpu_ids(a.gpus))
    elif a.mode=='canary':engine(out).canary(out,s.gpu_ids(a.gpus))
    elif a.mode=='merge':merge(out)
    else:configure(out);print(json.dumps(g.status(out),indent=2))


if __name__=='__main__':main()
