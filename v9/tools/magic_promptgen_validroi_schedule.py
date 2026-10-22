"""Wait for real GPU headroom; gate corrected MAGIC full queue on three smokes."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
MIN_FREE_MIB=24576


def available_gpus(query,allowed,running):
    result=[]
    for line in query.strip().splitlines():
        gpu,free,uuid=[p.strip() for p in line.split(',')]
        gpu=int(gpu)
        if gpu in allowed and gpu not in running and int(free)>=MIN_FREE_MIB:
            result.append((gpu,uuid))
    return result


def run_phase(fix,out,jobs,gpus,phase):
    e=fix.e
    pending=[j.copy() for j in jobs if not (out/'jobs'/j['name']/'complete_metrics.json').is_file()]
    running={};failed=[];last_wait=0
    while pending or running:
        for gpu,(process,job,handle) in list(running.items()):
            if process.poll() is None:continue
            handle.close();del running[gpu]
            if process.returncode:
                failed.append(dict(name=job['name'],returncode=process.returncode,log=job['log']))
                print('FAILED',job['name'],job['log'],flush=True)
            else:print('DONE',job['name'],job['task_count'],flush=True)
        if not failed:
            query=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,uuid',
                '--format=csv,noheader,nounits'],text=True)
            for gpu,uuid in available_gpus(query,set(gpus),set(running)):
                if not pending:break
                job=pending.pop(0);job.update(gpu=gpu,gpu_uuid=uuid,run=str(out/'jobs'/job['name']))
                jobfile=out/'job_specs'/(job['name']+'.json')
                log=out/'logs'/(job['name']+'_validroi.log')
                if log.exists():raise FileExistsError('preserve existing log; explicit recovery required: '+str(log))
                job['log']=str(log);e.base.write(jobfile,job);log.parent.mkdir(parents=True,exist_ok=True)
                handle=log.open('x')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',
                    MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
                process=subprocess.Popen([str(ROOT/fix.PYTHONS[job['model']]),str(Path(fix.__file__).resolve()),
                    'worker','--job',str(jobfile)],cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
                running[gpu]=(process,job,handle)
                print('START',phase,'gpu',gpu,'uuid',uuid,'pid',process.pid,job['name'],flush=True)
        state=dict(time=time.strftime('%Y-%m-%d %H:%M:%S'),phase=phase,pending=len(pending),
            running={str(g):dict(name=j['name'],pid=p.pid,log=j['log']) for g,(p,j,h) in running.items()},
            failed=failed,minimum_free_mib=MIN_FREE_MIB)
        e.base.write(out/'status.json',state)
        if failed and not running:raise RuntimeError('worker failed; successful/partial outputs preserved; no automatic unsafe retry')
        if pending and not running and time.time()-last_wait>=60:
            print('WAIT_GPU_MEMORY',phase,'pending',len(pending),'requires_MiB',MIN_FREE_MIB,flush=True);last_wait=time.time()
        if pending or running:time.sleep(15)
    for job in jobs:
        rows=e.base.read(out/'jobs'/job['name']/'complete_metrics.json')['per_case_class']
        e.base.validate_rows(rows,e.base.read(Path(job['prepared'])/'manifest.json')['tasks'])
        for row in rows:
            for s in row['stages']:
                a=s['calibration']
                if a['valid_background_count']!=4 or a['background_outside_predicted_roi'] or a['foreground_outside_predicted_roi']:
                    raise ValueError('real-task coverage audit failed before next phase')


def main():
    sys.path.insert(0,str(ROOT))
    from tools import magic_promptgen_validroi_eval as fix
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default=fix.OUT);p.add_argument('--gpus',default='0,2,3,4,5,6')
    a=p.parse_args()
    import fcntl
    out=ROOT/a.out;os.chdir(ROOT);fix.install();fix.e.verify(out)
    lock=(out/'validroi_schedule.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    audit=out/'validroi_queue_audit.json'
    payload=dict(launcher_sha256=fix.e.sha(__file__),worker_sha256=fix.e.sha(fix.__file__),
        interpreters=fix.PYTHONS,minimum_free_mib=MIN_FREE_MIB,policy='no killing, no input reductions; smoke before full')
    if audit.exists() and fix.e.base.read(audit)!=payload:raise ValueError('queue code/worker signature changed')
    fix.e.base.write(audit,payload)
    jobs=fix.e.base.read(out/'jobs.json');gpus=list(map(int,a.gpus.split(',')))
    run_phase(fix,out,[j for j in jobs if j['name'].endswith('_smoke')],gpus,'smoke')
    print('ALL_THREE_CORRECTED_SMOKES_PASSED',flush=True)
    run_phase(fix,out,[j for j in jobs if not j['name'].endswith('_smoke')],gpus,'full')
    fix.merge(out)
    fix.e.base.write(out/'status.json',dict(time=time.strftime('%Y-%m-%d %H:%M:%S'),phase='complete',pending=0,running={},failed=[]))


if __name__=='__main__':main()
