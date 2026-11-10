"""Bounded queue for the isolated five-source prompt experiment; no signals."""
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

ROOT=Path(__file__).resolve().parents[1]
SOURCES={'magic','msd_task07','msd_task10','parse2022','kits23'}


def require_scope(gpus):
    if not gpus or len(gpus)!=len(set(gpus)) or not set(gpus)<={0,3}:
        raise ValueError('distinct permitted physical GPUs required')
    return gpus


def retry_job(job, code, text, now):
    oom=code!=75 and ('cuda out of memory' in text.lower() or 'OutOfMemoryError' in text)
    if code==0 or (code!=75 and not oom):return None
    value=copy.deepcopy(job);value['attempt']=job.get('attempt',0)+1
    value['not_before']=now+20
    for k in ('gpu','gpu_uuid','run','log'):value.pop(k,None)
    if oom:
        if value['frame_batch']>1:value['frame_batch']//=2
        else:
            value.update(minimum_free_mib=45000,not_before=now+60)
            value['avoid_gpus']={**value.get('avoid_gpus',{}),str(job['gpu']):now+600}
    return value


def available_gpu(job, resources, busy, now):
    if now<job.get('not_before',0):return None
    for gpu,row in sorted(resources.items()):
        if (gpu not in busy and row['free']>=max(30720,job.get('minimum_free_mib',30720))
                and now>=job.get('avoid_gpus',{}).get(str(gpu),0)):
            return gpu
    return None


def lower_pending_batch(pending,source,old,new):
    values=copy.deepcopy(pending)
    for j in values:
        if j['source']==source and j['frame_batch']==old and not j.get('control'):
            j['frame_batch']=new
    return values


def add_batch_controls(pending,templates,passed,active):
    values=copy.deepcopy(pending);present={(j['source'],j['frame_batch']) for j in [*values,*active] if j.get('control')}
    for j in pending:
        ident=(j['source'],j['frame_batch'])
        if not j.get('control') and ident not in passed and ident not in present:
            c=copy.deepcopy(next(x for x in templates if x['source']==j['source']))
            c.update(frame_batch=j['frame_batch'],attempt=0,control=True)
            values.insert(0,c);present.add(ident)
    return values


def passed_batches(out,digest):
    found=set()
    for p in (out/'controls').glob('*/batch_*/control_pass.json'):
        v=json.loads(p.read_text())
        if v.get('passed') is not True or v.get('protocol_sha256')!=digest:
            raise ValueError('invalid actual control gate')
        found.add((v['source'],v['actual_frame_batch']))
    return found


def atomic(path,value):
    temp=path.with_name(path.name+'.tmp_'+str(os.getpid()))
    with temp.open('w') as f:json.dump(value,f,indent=2,allow_nan=False)
    os.replace(temp,path)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def resources(gpus):
    s=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],
                     check=True,capture_output=True,text=True)
    result={}
    for line in s.stdout.splitlines():
        index,uuid,free=map(str.strip,line.split(','))
        if int(index) in gpus:result[int(index)]=dict(uuid=uuid,free=int(free))
    return result


def run(out,phase,gpus):
    import fcntl
    require_scope(gpus)
    protocol=out/'protocol.json';protocol_sha=sha(protocol)
    lock=(out/'queue.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=out/('queue_'+phase+'.json')
    if status.exists():raise FileExistsError('phase already dispatched; inspect proofs before recovery')
    if phase=='full':
        state=json.loads((out/'queue_controls.json').read_text())
        if state.get('phase')!='complete' or state.get('failed') or state.get('protocol_sha256')!=protocol_sha:
            raise ValueError('five-source real controls must naturally complete first')
        if {j['source'] for j in state['completed']}!=SOURCES or len(state['completed'])!=5:
            raise ValueError('all five controls required')
    jobs=json.loads((out/('control_jobs.json' if phase=='controls' else 'jobs.json')).read_text())
    templates=json.loads((out/'control_jobs.json').read_text())
    if phase=='full':
        for j in jobs:
            gate=json.loads((out/'controls'/j['source']/'control_pass.json').read_text())
            if gate.get('passed') is not True or gate.get('protocol_sha256')!=protocol_sha:
                raise ValueError('five-source control gate binding failed')
            j['frame_batch']=min(j['frame_batch'],gate['actual_frame_batch'])
    if not isinstance(jobs,list) or not jobs or {j['source'] for j in jobs}!=SOURCES:
        raise ValueError('invalid whole-case jobs')
    pending=copy.deepcopy(jobs);active={};completed=[];failed=[];attempts=[]
    specs=out/'queue_specs';logs=out/'queue_logs';specs.mkdir(exist_ok=True);logs.mkdir(exist_ok=True)
    while pending or active:
        now=time.time()
        for gpu,item in list(active.items()):
            rc=item['proc'].poll()
            if rc is None:continue
            item['stream'].close();job=item['job'];attempt=dict(job,returncode=rc)
            attempts.append(attempt)
            if rc==0:completed.append(job)
            else:
                retry=retry_job(job,rc,Path(job['log']).read_text(errors='replace'),now)
                if retry is None:failed.append(attempt)
                else:
                    if job.get('control') and retry['frame_batch']<job['frame_batch']:
                        pending=lower_pending_batch(pending,job['source'],job['frame_batch'],retry['frame_batch'])
                    pending.append(retry)
            del active[gpu]
        snapshot=dict(phase='failed_draining' if failed else phase,protocol_sha256=protocol_sha,
                      pending=len(pending),running=[x['job'] for x in active.values()],
                      completed=completed,failed=failed,attempts=attempts,updated=time.time())
        atomic(status,snapshot)
        if failed:
            if not active:break
            time.sleep(5);continue
        if sha(protocol)!=protocol_sha:raise ValueError('frozen protocol changed')
        passed=passed_batches(out,protocol_sha)
        if phase=='full':
            pending=add_batch_controls(pending,templates,passed,[a['job'] for a in active.values()])
        if shutil.disk_usage(out).free>=3*1024**3:
            free=resources(gpus)
            for job in list(pending):
                if phase=='full' and not job.get('control') and (job['source'],job['frame_batch']) not in passed:
                    continue
                if job.get('control') and (job['source'],job['frame_batch']) in passed:
                    pending.remove(job)
                    continue
                gpu=available_gpu(job,free,active,now)
                if gpu is None:continue
                second=resources([gpu])
                if available_gpu(job,second,active,now)!=gpu or second[gpu]['uuid']!=free[gpu]['uuid']:continue
                if shutil.disk_usage(out).free<3*1024**3:break
                name=hashlib.sha256((job['source']+'\0'+job['case_id']+'\0'+job.get('variant','control')).encode()).hexdigest()[:20]
                stem=f"{phase}_{name}_b{job['frame_batch']}_a{job.get('attempt',0)}"
                spec=specs/(stem+'.json');log=logs/(stem+'.log')
                actual=dict(job,gpu=gpu,gpu_uuid=second[gpu]['uuid'],log=str(log))
                with spec.open('x') as f:json.dump(actual,f,indent=2)
                stream=log.open('x')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1')
                proc=subprocess.Popen([sys.executable,'-B',str(ROOT/'tools/five_source_prompt_type_eval.py'),
                     'control' if job.get('control') else 'worker','--out',str(out),'--job',str(spec)],
                     cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
                active[gpu]=dict(proc=proc,stream=stream,job=dict(actual,pid=proc.pid))
                pending.remove(job)
                if len(active)>=min(4,len(gpus)):break
        if pending or active:time.sleep(5)
    atomic(status,dict(phase='failed' if failed else 'complete',protocol_sha256=protocol_sha,
                      pending=len(pending),running=[],completed=completed,failed=failed,attempts=attempts,
                      updated=time.time()))
    if failed:raise RuntimeError('non-OOM failure; stopped new dispatch and naturally drained active workers')
    print('QUEUE_PHASE_COMPLETE',phase,len(completed),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('controls','full'))
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--gpus',default='0,3')
    a=p.parse_args();run(a.out.resolve(),a.phase,[int(g) for g in a.gpus.split(',')])


if __name__=='__main__':main()
