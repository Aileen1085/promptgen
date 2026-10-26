"""Resource-only Bone recovery on GPUs2/4/5/6; readonly audited reuse, no partial reuse."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools import bone_floor_recovery as r
g,b,p=r.g,r.b,r.p
OLD=r.OUT
OUT='output/bone_v92e360_v102e490_floor_grid_20261027_resourcefix'
parent_verify=r.verify
sha=r.sha
verify_files=r.verify_files
FILES=('complete_metrics.json','all_metrics_live.json','exact_prompt_audit.json',
       'intervention_audit.json','confirmation_audit.json','recovery_audit.json')


def gpu_ids(value):
    ids=[int(x) for x in value.split(',') if x]
    if not ids or len(ids)!=len(set(ids)) or not set(ids)<={2,4,5,6}:
        raise ValueError('resource recovery only permits GPUs2/4/5/6')
    return ids


def process_matches(args,pid,self_pid):
    return str(pid)!=str(self_pid) and any(Path(x).name in
        ('bone_floor_grid.py','bone_floor_recovery.py','bone_floor_resource_recovery.py') for x in args
        ) and any(x in ('prepare','prompt-worker','worker','launch','canary','wait-launch') for x in args)


def related_processes():
    found=[]
    for q in Path('/proc').iterdir():
        if not q.name.isdigit(): continue
        try:
            if process_matches((q/'cmdline').read_bytes().decode(errors='replace').split('\0'),q.name,os.getpid()):found.append(q.name)
        except OSError: pass
    return found


def require_drained(state,processes):
    if state.get('phase')!='full' or state.get('running') or not state.get('failed') or processes:
        raise ValueError('old full queue must have failed and ALL scheduler/workers exited')


def partition(jobs,complete):
    names={j['name'] for j in jobs}; smoke={j['name'] for j in jobs if j['smoke']}
    if (len(jobs)!=162 or len(names)!=162 or sum(j['task_count'] for j in jobs)!=13860
        or len(smoke)!=18 or not smoke<=set(complete) or not set(complete)<=names):
        raise ValueError('invalid exact162/13860 recovery partition or incomplete18 smokes')
    reused=sum(j['task_count'] for j in jobs if j['name'] in complete)
    return dict(reused_jobs=len(complete),pending_jobs=162-len(complete),reused_tasks=reused,
                new_tasks=13860-reused,total_tasks=13860)


def validate_state_partition(state,jobs,complete):
    partition(jobs,complete); require_drained(state,[])
    full={j['name'] for j in jobs if not j['smoke']}; failed={j['name'] for j in state['failed']}
    done=len(full & set(complete))
    if (not failed<=full or failed & set(complete) or len(failed)!=len(state['failed'])
        or done!=state['done_in_phase'] or done+len(failed)+state['pending']!=len(full)):
        raise ValueError('old completed/failed/pending job accounting mismatch')


def pending_jobs(jobs,complete,smoke):
    return [dict(j) for j in jobs if j['smoke']==smoke and j['name'] not in complete]


def resume_settings(latest=None,oom=False):
    if latest is None:return dict(attempt=0,oom_retries=0,frame_batch=16)
    value=dict(attempt=latest['attempt']+1,oom_retries=latest['oom_retries'],frame_batch=latest['frame_batch'])
    if oom and value['frame_batch']>1:
        value['oom_retries']+=1;value['frame_batch']//=2
    return value


def validate_resume(job,limits):
    if limits and (job['oom_retries']<limits['oom_retries'] or job['attempt']<limits['attempt']
                   or job['frame_batch']>limits['frame_batch']):
        raise ValueError('previous reduced frame batch/attempt history was reset')


def signature(out):
    return dict(validation_only_parent=r.signature(out),resource_recovery_source=sha(__file__),
                allowed_gpus=[2,4,5,6],policy='reuse only fully audited whole shards; preserve prior reduced frame batches; native inference unchanged')


def verify(out):
    if signature(out)!=b.read(out/'protocol.json'):raise ValueError('changed resource recovery signature')
    parent_verify(ROOT/OLD)
    verify_files(b.read(out/'reuse_audit.json')['files'])


def configure(out):
    r.configure(out)
    r.verify=verify;g.verify=verify;g.amos.verify=verify


def prepare(out):
    if out.exists():raise FileExistsError('resource recovery output already exists')
    old=ROOT/OLD;parent_verify(old);state=b.read(old/'status.json')
    require_drained(state,related_processes()); jobs=b.read(old/'jobs.json'); complete=set();files={};resumes={}
    for j in jobs:
        passed=old/'job_specs'/(j['name']+'.json')
        if passed.exists():
            spec=b.read(passed);r.verify_job(spec);complete.add(j['name'])
            for f in [passed]+[Path(spec['run'])/n for n in FILES]:files[str(f)]=sha(f)
        else:
            attempts=[b.read(f) for f in (old/'job_specs').glob(j['name']+'_a*.json')]
            latest=max(attempts,key=lambda v:v['attempt']) if attempts else None
            oom=False
            if latest:
                g.validate_job_spec(latest,j,old)
                f=old/'job_specs'/(j['name']+'_a'+str(latest['attempt'])+'.json');files[str(f)]=sha(f)
                tail=Path(latest['log']).read_text(errors='replace')[-10000:]
                oom='out of memory' in tail.lower() or 'OutOfMemoryError' in tail
            resumes[j['name']]=resume_settings(latest,oom=oom)
    info=partition(jobs,complete);validate_state_partition(state,jobs,complete)
    if not info['new_tasks']:raise ValueError('no recovery work remains')
    for n in ('protocol.json','reuse_audit.json','status.json','jobs.json','manifest.json','refs.json',
              'prompt_budget_audit.json','terminal_oom_evidence_20261027.json'):
        files[str(old/n)]=sha(old/n)
    for f in (old/'logs').glob('*.log'):files[str(f)]=sha(f)
    out.mkdir(parents=True)
    for n in ('bone_split.json','manifest.json','jobs.json','refs.json','prompt_budget_audit.json',
              'prepared','prompt_prepared','preparation_status.json','preparation_protocol.json'):
        (out/n).symlink_to(old/n,target_is_directory=(old/n).is_dir())
    (out/'job_specs').mkdir();(out/'jobs').mkdir()
    for j in jobs:
        if j['name'] not in complete:continue
        passed=old/'job_specs'/(j['name']+'.json');spec=b.read(passed)
        (out/'job_specs'/(j['name']+'.json')).symlink_to(passed)
        (out/'jobs'/j['name']).symlink_to(Path(spec['run']),target_is_directory=True)
    g.write_new(out/'reuse_audit.json',dict(info,original_output=OLD,files=files,
        completed_names=sorted(complete),resume=resumes,partial_metrics_reused=False,
        additional_canaries=2,canaries_in_final_metrics=False,allowed_gpus=[2,4,5,6]))
    g.write_new(out/'protocol.json',signature(out));verify(out)
    print('RESOURCE_RECOVERY_PREPARED',info,'readonly18smokes; fresh2canaries required',flush=True)


def validate_canary_spec(job,out,model):
    if (Path(job['run']).parents[1]!=out or job['model']!=model or job['source']!='bone'
        or job['variant']!='baseline' or job['floor']!=.35 or not job['smoke']
        or job['task_count']!=1 or job['gpu'] not in (2,4,5,6)):
        raise ValueError('canary must be fresh same-floor one-task baseline in new output')


def run(job):
    out=Path(job['run']).parents[1];configure(out);verify(out)
    if job['gpu'] not in (2,4,5,6):raise ValueError('GPU0/3/1/7 prohibited in resource recovery')
    limits=b.read(out/'reuse_audit.json')['resume'].get(job['name'])
    validate_resume(job,limits)
    r.run(job)


def verify_canaries(out):
    for model in p.MODELS:
        marker=b.read(out/('canary_'+model+'.json'));verify_files(marker['files'])
        job=b.read(marker['spec']);validate_canary_spec(job,out,model);r.verify_job(job)
        reference=b.read(out/'job_specs'/(job['name']+'.json'));r.verify_job(reference)
        rows=b.read(Path(job['run'])/'complete_metrics.json')['per_case_class']
        original=b.read(Path(reference['run'])/'complete_metrics.json')['per_case_class']
        if len(rows)!=1 or p.key(rows[0])!=p.key(original[0]) or any(abs(rows[0][m]-original[0][m])>1e-6 for m in p.METRICS):
            raise ValueError('fresh resource canary differs from readonly original smoke')


def canary(out,ids):
    configure(out);verify(out);require_drained(b.read(ROOT/OLD/'status.json'),related_processes())
    for model in p.MODELS:
        marker=out/('canary_'+model+'.json')
        if marker.exists():continue
        available=g.gpu_snapshot();gpu=next((i for i in ids if available[i]['free']>=45000),None)
        if gpu is None:print('CANARY_RESOURCE_WAIT',flush=True);raise SystemExit(75)
        j=next(j for j in b.read(out/'jobs.json') if j['smoke'] and j['model']==model and j['floor']==.35)
        prior=[b.read(f) for f in (out/'job_specs').glob(j['name']+'_a*.json')]
        previous=max(prior,key=lambda v:v['attempt']) if prior else None
        settings=resume_settings(previous)
        if previous:
            tail=Path(previous['log']).read_text(errors='replace')[-10000:]
            if 'out of memory' in tail.lower() or 'OutOfMemoryError' in tail:
                if settings['frame_batch']==1:raise RuntimeError('canary exhausted allowed frame batches')
                settings['oom_retries']+=1;settings['frame_batch']//=2
        paths=g.attempt_paths(out,j['name'],settings['attempt'])
        job=dict(j,**settings);job.update(gpu=gpu,gpu_uuid=available[gpu]['uuid'],run=paths['run'],log=paths['log'])
        validate_canary_spec(job,out,model);g.write_new(paths['spec'],job)
        Path(paths['log']).parent.mkdir(exist_ok=True)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        with Path(paths['log']).open('x') as h:
            proc=subprocess.run([sys.executable,__file__,'worker','--job',paths['spec']],cwd=ROOT,env=env,stdout=h,stderr=subprocess.STDOUT)
        if proc.returncode:
            tail=Path(paths['log']).read_text(errors='replace')[-10000:]
            oom='out of memory' in tail.lower() or 'OutOfMemoryError' in tail
            if proc.returncode==75 or (oom and job['frame_batch']>1):
                print('CANARY_RESOURCE_RETRY',model,flush=True);raise SystemExit(75)
            raise RuntimeError('fresh canary failed; retain '+paths['log'])
        r.verify_job(job)
        files={str(f):sha(f) for f in [Path(paths['spec'])]+[Path(job['run'])/n for n in FILES]}
        g.write_new(marker,dict(spec=paths['spec'],files=files,extra_verification_only=True))
        print('FRESH_RESOURCE_CANARY_PASS',model,flush=True)
    verify_canaries(out)


def merge(out):
    configure(out);verify(out);verify_canaries(out);r.merge(out)
    result=b.read(out/'merged_metrics.json');result['resource_recovery']=b.read(out/'reuse_audit.json')
    b.write(out/'merged_metrics.json',result)


def launch(out,ids):
    configure(out);verify(out);require_drained(b.read(ROOT/OLD/'status.json'),related_processes())
    if (out/'status.json').exists():raise ValueError('do not relaunch started resource queue')
    import fcntl
    lock=(out/'bone_schedule.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    canary(out,ids);verify_canaries(out)
    jobs=b.read(out/'jobs.json');reuse=b.read(out/'reuse_audit.json');complete=set(reuse['completed_names'])
    for smoke in (True,False):
        phase='smoke' if smoke else 'full'
        pending=[dict(j,**reuse['resume'][j['name']]) for j in pending_jobs(jobs,complete,smoke)]
        running={};failed=[];done=sum(j['smoke']==smoke and j['name'] in complete for j in jobs)
        b.write(out/'queue_phase.json',dict(phase=phase,readonly_reused_jobs=done))
        while pending or running:
            for gpu,(proc,job,handle) in list(running.items()):
                if proc.poll() is None:continue
                handle.close();del running[gpu]
                if proc.returncode:
                    tail=Path(job['log']).read_text(errors='replace')[-10000:]
                    oom='out of memory' in tail.lower() or 'OutOfMemoryError' in tail
                    if proc.returncode==75 or (oom and job['frame_batch']>1):
                        job['attempt']+=1
                        if oom:job['oom_retries']+=1;job['frame_batch']=max(1,job['frame_batch']//2)
                        pending.insert(0,job);print('RESOURCE_RETRY' if proc.returncode==75 else 'OOM_RETRY',job['name'],job['attempt'],job['frame_batch'],flush=True)
                    else:failed.append(dict(name=job['name'],log=job['log'],returncode=proc.returncode))
                else:
                    r.verify_job(job);g.write_new(out/'job_specs'/(job['name']+'.json'),job);done+=1
                    print('DONE',job['name'],'tasks',job['task_count'],'attempt',job['attempt'],flush=True)
            available=g.gpu_snapshot()
            for gpu in ids:
                if failed or gpu in running or not pending or available[gpu]['free']<45000:continue
                job=pending.pop(0);paths=g.attempt_paths(out,job['name'],job['attempt'])
                job.update(gpu=gpu,gpu_uuid=available[gpu]['uuid'],run=paths['run'],log=paths['log'])
                signed=next(j for j in jobs if j['name']==job['name']);g.validate_job_spec(job,signed,out)
                if Path(job['run']).exists():raise ValueError('attempt output exists')
                g.write_new(paths['spec'],job);Path(job['log']).parent.mkdir(exist_ok=True);handle=Path(job['log']).open('x')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
                proc=subprocess.Popen([sys.executable,__file__,'worker','--job',paths['spec']],cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
                running[gpu]=(proc,job,handle);print('START',phase,'gpu',gpu,'pid',proc.pid,job['name'],flush=True)
            b.write(out/'status.json',dict(time=time.strftime('%Y-%m-%d %H:%M:%S'),phase=phase,pending=len(pending),done_in_phase=done,
                running={str(gpu):dict(name=j['name'],pid=pr.pid,log=j['log'],attempt=j['attempt']) for gpu,(pr,j,h) in running.items()},failed=failed))
            if failed and not running:raise RuntimeError('terminal resource recovery failure; retain all attempts')
            if pending or running:time.sleep(10)
        for j in jobs:
            if j['smoke']==smoke:r.verify_job(b.read(out/'job_specs'/(j['name']+'.json')))
        print('PHASE_PASSED',phase,'readonly_reuse',sum(j['smoke']==smoke and j['name'] in complete for j in jobs),flush=True)
    merge(out)


def wait_launch(out,ids):
    """Wait without interrupting old tasks; only then freeze reuse and start recovery."""
    if out.exists():raise FileExistsError('wait-launch is only for a new recovery directory')
    previous=None
    while True:
        processes=related_processes()
        for pid in processes:
            try:args=(Path('/proc')/pid/'cmdline').read_bytes().decode(errors='replace').split('\0')
            except FileNotFoundError:continue
            if any(Path(x).name=='bone_floor_resource_recovery.py' for x in args):
                raise ValueError('another resource recovery helper/worker is already active')
        if not processes:break
        if len(processes)!=previous:
            print('WAIT_OLD_DRAINING',len(processes),'no model started; old processes untouched',flush=True);previous=len(processes)
        time.sleep(30)
    require_drained(b.read(ROOT/OLD/'status.json'),[])
    time.sleep(5)
    require_drained(b.read(ROOT/OLD/'status.json'),related_processes())
    prepare(out)
    while True:
        try:launch(out,ids);return
        except SystemExit as error:
            if error.code!=75 or (out/'status.json').exists():raise
            print('WAIT_CANARY_GPU_MEMORY',flush=True);time.sleep(30)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','canary','worker','launch','wait-launch','status','merge'))
    parser.add_argument('--out',default=OUT);parser.add_argument('--job');parser.add_argument('--gpus',default='2,4,5,6')
    a=parser.parse_args();out=(ROOT/a.out).resolve()
    if a.mode=='prepare':prepare(out)
    elif a.mode=='worker':run(b.read(a.job))
    elif a.mode=='launch':launch(out,gpu_ids(a.gpus))
    elif a.mode=='wait-launch':wait_launch(out,gpu_ids(a.gpus))
    elif a.mode=='canary':canary(out,gpu_ids(a.gpus))
    elif a.mode=='merge':merge(out)
    else:
        configure(out);value=g.status(out);reuse=b.read(out/'reuse_audit.json')
        value['readonly_reuse']={k:reuse[k] for k in ('reused_jobs','pending_jobs','reused_tasks','new_tasks','total_tasks')}
        value['resumed_batches']={n:s for n,s in reuse['resume'].items() if s['frame_batch']!=16}
        print(json.dumps(value,indent=2))


if __name__=='__main__':main()
