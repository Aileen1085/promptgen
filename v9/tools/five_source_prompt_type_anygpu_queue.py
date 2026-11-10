"""Resume the existing case-coupled experiment on any eligible physical GPU."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools import five_source_prompt_type_queue_v2 as old
from tools import five_source_prompt_type_anygpu as overlay
from tools import five_source_prompt_type_eval_v2 as e


def ident(job):return job['source'],job['case_id'],job.get('variant','control')


def select_gpu(job,resources,busy,now):
    if now<job.get('not_before',0):return None
    for gpu,row in sorted(resources.items(),key=lambda pair:(-pair[1]['free'],pair[0])):
        if (gpu in range(8) and gpu not in busy and row['free']>=max(30720,job.get('minimum_free_mib',30720))
                and now>=job.get('avoid_gpus',{}).get(str(gpu),0)):
            return gpu
    return None


def restore_pending(templates,completed_ids,attempts,read_log,now):
    latest={}
    for attempt in attempts:
        key=ident(attempt)
        if key not in latest or attempt.get('attempt',0)>latest[key].get('attempt',0):latest[key]=attempt
    pending=[]
    for template in templates:
        key=ident(template)
        if key in completed_ids:continue
        if key not in latest:pending.append(copy.deepcopy(template));continue
        attempt=latest[key]
        retry=old.retry_job(attempt,attempt['returncode'],read_log(attempt),now)
        if retry is None:raise ValueError('unresolved non-retryable attempt: '+repr(key))
        retry.pop('returncode',None);retry.pop('pid',None)
        pending.append(retry)
    return pending


def next_unused_attempt(out,job):
    value=copy.deepcopy(job)
    while (Path(out)/'jobs'/(e.case_name(value['source'],value['case_id'])+'__'+
            ('control' if value.get('control') else value['variant'])+'__b'+str(value['frame_batch'])+
            '__a'+str(value.get('attempt',0)))).exists():
        value['attempt']=value.get('attempt',0)+1
    return value


def completed_job(out,job,digest):
    if job.get('control'):
        path=out/'controls'/job['source']/('batch_'+str(job['frame_batch']))/'control_pass.json'
        if not path.exists():return False
        proof=e.read(path)
        if (proof.get('passed'),proof.get('protocol_sha256'),proof.get('source'),proof.get('case_id'),
                proof.get('actual_frame_batch'))!=(True,digest,job['source'],job['case_id'],job['frame_batch']):
            raise ValueError('existing control certificate mismatch')
        return True
    paths=[e.record_path(out,t,v) for t in job['tasks'] for v in e.VARIANTS]
    present=[p.exists() for p in paths]
    if any(present) and not all(present):raise ValueError('partial published case forbidden')
    if not all(present):return False
    for task in job['tasks']:
        batches=set()
        for variant in e.VARIANTS:
            record=e.read(e.record_path(out,task,variant))
            if record.get('protocol_sha256')!=digest:raise ValueError('existing record protocol mismatch')
            e.runtime.validate_record(record,task,variant,record['actual_frame_batch'])
            if not record['normal_finish']['successful_native_completion']:raise ValueError('normal finish missing')
            batches.add(record['actual_frame_batch'])
        if len(batches)!=1:raise ValueError('case variants have different batches')
    return True


def recovery(out,phase,digest):
    templates=e.read(out/'control_jobs.json')
    jobs=e.read(out/('control_jobs.json' if phase=='controls' else 'jobs.json'))
    previous=out/('queue_'+phase+'.json')
    state=e.read(previous) if previous.exists() else dict(completed=[],attempts=[],running=[],failed=[])
    if state.get('failed'):raise ValueError('non-OOM failure requires independent investigation')
    if previous.exists() and state.get('protocol_sha256')!=digest:raise ValueError('old queue protocol mismatch')
    attempts=list(state.get('attempts',[]))
    if state.get('running'):
        retired=e.read(out/'anygpu_parent_retirement_20261110.json')
        if retired.get('protocol_sha256')!=digest or retired.get('workers_drained') is not True:
            raise ValueError('old workers not independently drained')
        recovered=retired['worker_exits']
        for job in state['running']:
            found=[a for a in recovered if a['pid']==job['pid'] and ident(a)==ident(job)
                   and a['attempt']==job['attempt']]
            if len(found)!=1:raise ValueError('missing actual prior worker exit')
            attempts.append(found[0])
    if phase=='full':
        controls=e.read(out/'queue_controls.json')
        if controls.get('phase')!='complete' or controls.get('running') or controls.get('pending') or controls.get('failed'):
            raise ValueError('source controls not complete')
        for job in jobs:
            gate=e.read(out/'controls'/job['source']/'control_pass.json')
            if gate.get('passed') is not True or gate.get('protocol_sha256')!=digest:raise ValueError('source control mismatch')
            job['frame_batch']=min(job['frame_batch'],gate['actual_frame_batch'])
    completed=[]
    for job in jobs:
        candidates=[a for a in attempts if ident(a)==ident(job) and a.get('returncode')==0]
        candidate=max(candidates,key=lambda a:a.get('attempt',0)) if candidates else job
        if phase=='controls':
            gate_path=out/'controls'/job['source']/'control_pass.json'
            if gate_path.exists():candidate=dict(candidate,frame_batch=e.read(gate_path)['actual_frame_batch'])
        if completed_job(out,candidate,digest):completed.append(candidate)
    pending=restore_pending(jobs,{ident(j) for j in completed},attempts,
        lambda a:Path(a['log']).read_text(errors='replace'),time.time())
    return pending,completed,attempts,templates


def run_phase(out,phase):
    policy=overlay.install(out);digest=policy['protocol_sha256']
    pending,completed,attempts,templates=recovery(out,phase,digest)
    active={};failed=[];status=out/('queue_'+phase+'.json')
    specs=out/'queue_specs';logs=out/'queue_logs';specs.mkdir(exist_ok=True);logs.mkdir(exist_ok=True)
    def save(final=False):
        old.atomic(status,dict(phase=('failed' if failed else 'complete') if final else ('failed_draining' if failed else phase),
            protocol_sha256=digest,resource_policy_sha256=policy['policy_sha256'],coordinator_pid=os.getpid(),
            pending=len(pending),running=[v['job'] for v in active.values()],completed=completed,
            failed=failed,attempts=attempts,updated=time.time()))
    while pending or active:
        now=time.time()
        for gpu,item in list(active.items()):
            rc=item['proc'].poll()
            if rc is None:continue
            item['stream'].close();job=item['job'];attempt=dict(job,returncode=rc);attempts.append(attempt)
            if rc==0:
                try:
                    if not completed_job(out,job,digest):raise ValueError('successful worker has no complete certificate')
                    completed.append(job)
                except Exception as error:failed.append(dict(attempt,certificate_error=repr(error)))
            else:
                retry=old.retry_job(job,rc,Path(job['log']).read_text(errors='replace'),now)
                if retry is None:failed.append(attempt)
                else:
                    retry.pop('pid',None)
                    if job.get('control') and retry['frame_batch']<job['frame_batch']:
                        pending=old.lower_pending_batch(pending,job['source'],job['frame_batch'],retry['frame_batch'])
                    pending.append(retry)
            del active[gpu]
        save()
        if failed:
            if not active:break
            time.sleep(5);continue
        overlay.read_policy(out)
        passed=old.passed_batches(out,digest)
        if phase=='full':pending=old.add_batch_controls(pending,templates,passed,[a['job'] for a in active.values()])
        if shutil.disk_usage(out).free>=3*1024**3 and len(active)<4:
            free=old.resources(list(range(8)))
            for job in list(pending):
                if len(active)>=4:break
                if job.get('control') and (job['source'],job['frame_batch']) in passed:
                    if phase=='controls' and ident(job) not in {ident(j) for j in completed}:completed.append(job)
                    pending.remove(job);continue
                if phase=='full' and not job.get('control') and (job['source'],job['frame_batch']) not in passed:continue
                gpu=select_gpu(job,free,active,now)
                if gpu is None:continue
                second=old.resources([gpu])
                if select_gpu(job,second,active,now)!=gpu or second[gpu]['uuid']!=free[gpu]['uuid']:continue
                if shutil.disk_usage(out).free<3*1024**3:break
                original_job=job;job=next_unused_attempt(out,job)
                name=hashlib.sha256(('\0'.join(ident(job))).encode()).hexdigest()[:20]
                stem='anygpu_'+phase+'_'+name+'_b'+str(job['frame_batch'])+'_a'+str(job.get('attempt',0))
                spec=specs/(stem+'.json');log=logs/(stem+'.log')
                actual=dict(job,gpu=gpu,gpu_uuid=second[gpu]['uuid'],log=str(log))
                with spec.open('x') as f:json.dump(actual,f,indent=2)
                stream=log.open('x')
                proc=subprocess.Popen([sys.executable,'-B',str(ROOT/'tools/five_source_prompt_type_anygpu.py'),
                    'control' if job.get('control') else 'worker','--out',str(out),'--job',str(spec)],
                    cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1'),
                    stdout=stream,stderr=subprocess.STDOUT)
                active[gpu]=dict(proc=proc,stream=stream,job=dict(actual,pid=proc.pid));pending.remove(original_job)
                save()
        if pending or active:time.sleep(5)
    save(final=True)
    if failed:raise RuntimeError('non-OOM failure; new dispatch stopped and workers naturally drained')
    print('ANYGPU_QUEUE_PHASE_COMPLETE',phase,len(completed),flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args(argv)
    import fcntl
    out=args.out.absolute();policy=overlay.install(out)
    if 'tools/five_source_prompt_type_anygpu_queue.py' not in policy['extra_source_sha256']:
        raise ValueError('resource policy must bind this dispatcher source')
    if (out/'queue_full.json').exists() or (out/'independent_control_gate.json').exists():
        raise FileExistsError('one-shot controls handoff only; later-phase recovery requires a new reviewed handoff')
    retired=e.read(out/'anygpu_parent_retirement_20261110.json')
    if not retired.get('workers_drained') or not retired.get('coordinators_retired'):
        raise ValueError('old coordinator retirement proof required')
    for pid in (retired['queue_pid'],retired['pipeline_pid']):
        stat=Path('/proc')/str(pid)/'stat'
        if stat.exists() and stat.read_text().rsplit(')',1)[1].split()[0]!='Z':raise ValueError('old coordinator still alive')
    lock=(out/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    e.atomic_new(out/'anygpu_launch_20261110.json',dict(pid=os.getpid(),time=time.time(),resource_policy=overlay.read_policy(out)))
    run_phase(out,'controls')
    subprocess.run([sys.executable,'-B',str(ROOT/'tools/five_source_prompt_type_anygpu.py'),
        'audit-controls','--out',str(out)],cwd=ROOT,check=True)
    run_phase(out,'full')
    subprocess.run([sys.executable,'-B',str(ROOT/'tools/five_source_prompt_type_anygpu.py'),
        'merge','--out',str(out)],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'-B',str(ROOT/'tools/five_source_prompt_type_anygpu.py'),
        'audit-final','--out',str(out)],cwd=ROOT,check=True)


if __name__=='__main__':main()
