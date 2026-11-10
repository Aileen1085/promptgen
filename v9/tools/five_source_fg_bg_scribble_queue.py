"""One-shot whole-case paired supplement, using any eligible physical GPU."""
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
from tools.five_source_prompt_type_queue_v2 import retry_job,add_batch_controls,lower_pending_batch,atomic,resources


def select_gpu(job,rows,busy,now):
    if now<job.get('not_before',0):return None
    for gpu,row in sorted(rows.items(),key=lambda item:(-item[1]['free'],item[0])):
        if (gpu in range(8) and gpu not in busy and row['free']>=max(30720,job.get('minimum_free_mib',30720))
                and now>=job.get('avoid_gpus',{}).get(str(gpu),0)):
            return gpu
    return None


def align_initial_batches(jobs,source_batches):
    result=copy.deepcopy(jobs)
    for job in result:job['frame_batch']=min(job['frame_batch'],source_batches[job['source']])
    return result


def apply_retry_cap(job,retry,caps):
    """Remember a failed fallback control so late OOMs cannot recreate it."""
    result=copy.deepcopy(retry)
    if retry['frame_batch']<job['frame_batch']:
        source=job['source']
        if job.get('control'):caps[source]=min(caps.get(source,4),retry['frame_batch'])
        result['frame_batch']=min(retry['frame_batch'],caps.get(source,4))
    return result


def allocate_attempt(job,history,active):
    """Monotonic numbering across a control case's dynamic batch attempts."""
    result=copy.deepcopy(job)
    identity=lambda j:(j['source'],j['case_id'],j['variant'])
    used=[j.get('attempt',0) for j in [*history,*active] if identity(j)==identity(job)]
    if used:result['attempt']=max(result.get('attempt',0),max(used)+1)
    return result


def validate_job_files(e,protocol,jobs,controls):
    batches={tuple(json.loads(k)):v for k,v in protocol['parent_actual_batches'].items()}
    expected_jobs,expected_controls=e.case_jobs(e.validate_inventory(protocol['inventory']),batches)
    if jobs!=expected_jobs or controls!=expected_controls:
        raise ValueError('job files differ from immutable complete parent case/batch schedule')


def run_phase(out,phase):
    from tools import five_source_fg_bg_scribble_eval as e
    protocol=e.read(out/'protocol.json');digest=e.sha(out/'protocol.json')
    e.validate_protocol(protocol)
    status=out/('queue_'+phase+'.json')
    if status.exists():raise FileExistsError('phase already dispatched; preserve evidence and inspect, never blind restart')
    all_jobs=e.read(out/'jobs.json');templates=e.read(out/'control_jobs.json')
    validate_job_files(e,protocol,all_jobs,templates)
    jobs=copy.deepcopy(templates if phase=='controls' else all_jobs)
    if phase=='full':
        gate=e.read(out/'independent_control_gate.json')
        controls=e.read(out/'queue_controls.json')
        if gate.get('passed') is not True or gate.get('protocol_sha256')!=digest:
            raise ValueError('new independent control gate required')
        if controls.get('phase')!='complete' or controls['pending'] or controls['running'] or controls['failed']:
            raise ValueError('controls not naturally complete')
        batches={s:e.read(out/'controls'/s/'control_pass.json')['actual_frame_batch'] for s in e.SOURCES}
        jobs=align_initial_batches(jobs,batches)
    if not jobs or {j['source'] for j in jobs}!=set(e.SOURCES):raise ValueError('all five sources required')
    pending=copy.deepcopy(jobs);active={};completed=[];failed=[];attempts=[];fallback_caps={}
    specs=out/'queue_specs';logs=out/'queue_logs';specs.mkdir(exist_ok=True);logs.mkdir(exist_ok=True)
    def save(final=False):
        atomic(status,dict(phase=('failed' if failed else 'complete') if final else ('failed_draining' if failed else phase),
            protocol_sha256=digest,coordinator_pid=os.getpid(),pending=len(pending),
            running=[v['job'] for v in active.values()],completed=completed,failed=failed,attempts=attempts,updated=time.time()))
    def passed_batches():
        result=set()
        for path in (out/'controls').glob('*/batch_*/control_pass.json'):
            value=e.read(path)
            if value.get('passed') is not True or value.get('protocol_sha256')!=digest:raise ValueError('actual control certificate binding mismatch')
            result.add((value['source'],value['actual_frame_batch']))
        return result
    def verify_dispatch_sources():
        if e.sha(out/'protocol.json')!=digest:raise ValueError('immutable protocol changed')
        for name in ('tools/five_source_fg_bg_scribble_queue.py','tools/five_source_fg_bg_scribble_eval.py','tools/five_source_fg_bg_scribble_runtime.py'):
            if e.sha(ROOT/name)!=protocol['inventory']['input_sha256'][name]:raise ValueError('signed supplement dispatch source changed')
    while pending or active:
        now=time.time()
        for gpu,item in list(active.items()):
            rc=item['proc'].poll()
            if rc is None:continue
            item['stream'].close();job=item['job'];attempt=dict(job,returncode=rc);attempts.append(attempt)
            if rc==0:
                path=out/'jobs'/(e.case_name(job['source'],job['case_id'])+'__'+('control' if job.get('control') else job['variant'])+'__b'+str(job['frame_batch'])+'__a'+str(job.get('attempt',0)))/'complete.json'
                try:
                    proof=e.read(path)
                    if (proof.get('passed'),proof.get('protocol_sha256'),proof.get('actual_frame_batch'))!=(True,digest,job['frame_batch']):
                        raise ValueError('successful worker lacks complete whole-case proof')
                    if not proof['normal_finish']['successful_native_completion']:raise ValueError('normal finish absent')
                    completed.append(job)
                except Exception as error:failed.append(dict(attempt,certificate_error=repr(error)))
            else:
                try:retry=retry_job(job,rc,Path(job['log']).read_text(errors='replace'),now)
                except Exception as error:retry=None;attempt['log_error']=repr(error)
                if retry is None:failed.append(attempt)
                else:
                    retry=apply_retry_cap(job,retry,fallback_caps)
                    retry.pop('pid',None)
                    if job.get('control') and retry['frame_batch']<job['frame_batch']:
                        pending=lower_pending_batch(pending,job['source'],job['frame_batch'],retry['frame_batch'])
                    pending.append(retry)
            del active[gpu]
        save()
        if failed:
            if not active:break
            time.sleep(5);continue
        try:
            verify_dispatch_sources();passed=passed_batches()
            if phase=='full':pending=add_batch_controls(pending,templates,passed,[v['job'] for v in active.values()])
            if shutil.disk_usage(out).free>=3*1024**3 and len(active)<4:
                free=resources(list(range(8)))
                for job in list(pending):
                    if len(active)>=4:break
                    if job.get('control') and (job['source'],job['frame_batch']) in passed:
                        pending.remove(job);continue
                    if phase=='full' and not job.get('control') and (job['source'],job['frame_batch']) not in passed:continue
                    gpu=select_gpu(job,free,active,now)
                    if gpu is None:continue
                    second=resources([gpu])
                    if select_gpu(job,second,active,now)!=gpu or second[gpu]['uuid']!=free[gpu]['uuid']:continue
                    if shutil.disk_usage(out).free<3*1024**3:break
                    original_job=job
                    job=allocate_attempt(job,attempts,[v['job'] for v in active.values()])
                    ident=job['source']+'\0'+job['case_id']+'\0'+job['variant']
                    stem=phase+'_'+hashlib.sha256(ident.encode()).hexdigest()[:20]+'_b'+str(job['frame_batch'])+'_a'+str(job.get('attempt',0))
                    spec=specs/(stem+'.json');log=logs/(stem+'.log')
                    actual=dict(job,gpu=gpu,gpu_uuid=second[gpu]['uuid'],log=str(log))
                    with spec.open('x') as f:json.dump(actual,f,indent=2)
                    stream=log.open('x')
                    try:
                        process=subprocess.Popen([sys.executable,'-B',str(ROOT/'tools/five_source_fg_bg_scribble_eval.py'),
                            'control' if job.get('control') else 'worker','--out',str(out),'--job',str(spec)],cwd=ROOT,
                            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1'),stdout=stream,stderr=subprocess.STDOUT)
                    except Exception:stream.close();raise
                    active[gpu]=dict(proc=process,stream=stream,job=dict(actual,pid=process.pid));pending.remove(original_job);save()
        except Exception as error:
            failed.append(dict(coordinator_error=repr(error),time=time.time()));save()
        if pending or active:time.sleep(5)
    save(final=True)
    if failed:raise RuntimeError('non-OOM failure; stopped new dispatch and naturally drained all own workers')
    print('FG_BG_SCRIBBLE_QUEUE_COMPLETE',phase,len(completed),flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args(argv)
    import fcntl
    from tools import five_source_fg_bg_scribble_eval as e
    out=args.out.absolute();protocol=e.read(out/'protocol.json')
    e.validate_protocol(protocol)
    if (out/'queue_controls.json').exists() or (out/'queue_full.json').exists():raise FileExistsError('one-shot pipeline already dispatched')
    lock=(out/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    e.atomic_new(out/'pipeline_launch.json',dict(pid=os.getpid(),protocol_sha256=e.sha(out/'protocol.json'),time=time.time()))
    entry=str(ROOT/'tools/five_source_fg_bg_scribble_eval.py')
    run_phase(out,'controls')
    subprocess.run([sys.executable,'-B',entry,'audit-controls','--out',str(out)],cwd=ROOT,check=True)
    run_phase(out,'full')
    subprocess.run([sys.executable,'-B',entry,'merge','--out',str(out)],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'-B',entry,'audit-final','--out',str(out)],cwd=ROOT,check=True)


if __name__=='__main__':main()
