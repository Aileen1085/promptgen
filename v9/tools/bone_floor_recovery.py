"""Independent validation-only recovery; reuse frozen Bone770 prompts, no old predictions."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from tools import bone_floor_grid as g
b, p = g.b, g.p
OLD = g.OUT
OUT = 'output/bone_v92e360_v102e490_floor_grid_20261026_validationfix'
native_verify, native_configure = g.verify, g.configure
sha = g.amos.sha


class UnusedTrainingDataset:
    def __len__(self):
        raise RuntimeError('validation-only recovery must not access training data or lazy warmup')
    def __getitem__(self, index):
        raise RuntimeError('validation-only recovery must not access training data or lazy warmup')


def validation_only_factory(original, audit):
    def factory(pairs, args, *pos, **kw):
        if bool(getattr(args, 'validation_only', False)) and kw.get('class_weights') is not None:
            audit['training_dataset_constructions_skipped'] = audit.get('training_dataset_constructions_skipped', 0) + 1
            return UnusedTrainingDataset()
        return original(pairs, args, *pos, **kw)
    return factory


def process_matches(args, pid, self_pid):
    return str(pid) != str(self_pid) and any(Path(x).name in
        ('bone_floor_grid.py', 'bone_floor_recovery.py') for x in args) and any(
        x in ('prepare', 'prompt-worker', 'worker', 'launch') for x in args)


def related_processes():
    found = []
    for q in Path('/proc').iterdir():
        if not q.name.isdigit(): continue
        try:
            if process_matches((q/'cmdline').read_bytes().decode(errors='replace').split('\0'), q.name, os.getpid()):
                found.append(q.name)
        except OSError: pass
    return found


def validate_old_state(state, processes, complete_count):
    if state.get('phase') != 'smoke' or state.get('running') or not state.get('failed') or processes or complete_count:
        raise ValueError('recovery requires exited failed smoke queue with zero complete predictions')


def verify_files(files):
    for name, digest in files.items():
        if sha(name) != digest: raise ValueError('readonly reused file changed: '+name)


def signature(out):
    return dict(native=g.signature(out), recovery_source=sha(__file__),
                reuse_audit=sha(out/'reuse_audit.json'),
                fix='validation_only training-factory sentinel; no dataset/loader access; native validation unchanged')


def verify(out):
    if signature(out) != b.read(out/'protocol.json'): raise ValueError('changed recovery signature')
    native_verify(ROOT/OLD)
    verify_files(b.read(out/'reuse_audit.json')['files'])


def configure(out):
    native_configure(out)
    g.verify = verify
    g.amos.verify = verify
    g.amos.merge = g.merge


def prepare(out):
    if out.exists(): raise FileExistsError('new recovery output must not exist')
    old = ROOT/OLD
    native_verify(old)
    complete = list((old/'jobs').glob('*/complete_metrics.json'))
    validate_old_state(b.read(old/'status.json'), related_processes(), len(complete))
    refs = b.read(old/'refs.json')['per_case_class']
    tasks = b.read(old/'manifest.json')['tasks']
    if len(refs) != 770 or {p.key(r) for r in refs} != {p.key(t) for t in tasks}:
        raise ValueError('readonly prompt coverage changed')
    files = {}
    for name in ('protocol.json', 'preparation_protocol.json', 'bone_split.json', 'manifest.json',
                 'jobs.json', 'refs.json', 'prompt_budget_audit.json', 'failure_evidence_validation_trainquota.json'):
        files[str(old/name)] = sha(old/name)
    for folder in ('prepared', 'prompt_prepared'):
        for path in sorted((old/folder).glob('*/*.json')): files[str(path)] = sha(path)
    for ref in refs:
        path = ROOT/b.CACHE/'metadata/shared_adaptive_prompt_v1/bone'/ref['prompt_cache_name']
        files[str(path)] = sha(path)
    out.mkdir(parents=True)
    for name in ('bone_split.json', 'manifest.json', 'jobs.json', 'refs.json', 'prompt_budget_audit.json',
                 'prepared', 'prompt_prepared', 'preparation_status.json', 'preparation_protocol.json'):
        (out/name).symlink_to(old/name, target_is_directory=(old/name).is_dir())
    g.write_new(out/'reuse_audit.json', dict(original_output=OLD, files=files,
        reused_prompt_tasks=770, reused_predictions=0, new_predictions=13860,
        old_attempts_untouched=True, training_data_added=False,
        scope='same20cases/47actualclasses/770 keys; all18configurations rerun'))
    g.write_new(out/'protocol.json', signature(out)); verify(out)
    print('RECOVERY_PREPARED reused770prompts;zero old predictions;13860 new predictions pending', flush=True)


def run(job):
    out = Path(job['run']).parents[1]; configure(out); verify(out)
    audit = dict(training_dataset_constructions_skipped=0, model=job['model'])
    if job['model'] == 'v9_2':
        import infer.infer_eight_source_adaptive_v9_2 as ev
        native_bind = ev._install_exact_prompt_factory
        def bind(entry, *a, **kw):
            result = native_bind(entry, *a, **kw)
            entry.base.make_dataset = validation_only_factory(entry.base.make_dataset, audit)
            return result
        ev._install_exact_prompt_factory = bind
    g.run(job)
    if audit['training_dataset_constructions_skipped'] != (1 if job['model'] == 'v9_2' else 0):
        raise ValueError('unexpected validation-only training factory count')
    g.write_new(Path(job['run'])/'recovery_audit.json', audit)
    verify_job(job)


def verify_job(job):
    tasks = g.verify_job(job)
    audit = b.read(Path(job['run'])/'recovery_audit.json')
    if audit != dict(training_dataset_constructions_skipped=1 if job['model']=='v9_2' else 0, model=job['model']):
        raise ValueError('recovery training-data bypass audit mismatch')
    return tasks


def merge(out):
    configure(out); verify(out)
    for j in b.read(out/'jobs.json'):
        verify_job(b.read(out/'job_specs'/(j['name']+'.json')))
    g.merge(out)


def launch(out, ids):
    configure(out); verify(out)
    if related_processes(): raise ValueError('related scheduler/worker already active')
    if (out/'status.json').exists() or (out/'job_specs').exists(): raise ValueError('do not relaunch started recovery')
    import fcntl
    lock=(out/'bone_schedule.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    jobs=b.read(out/'jobs.json')
    for smoke in (True, False):
        phase='smoke' if smoke else 'full'; pending=[dict(j,oom_retries=0) for j in jobs if j['smoke']==smoke]
        running={}; failed=[]; done=0; b.write(out/'queue_phase.json',dict(phase=phase))
        while pending or running:
            for gpu,(proc,job,handle) in list(running.items()):
                if proc.poll() is None: continue
                handle.close(); del running[gpu]
                if proc.returncode:
                    tail=Path(job['log']).read_text(errors='replace')[-10000:]
                    oom='out of memory' in tail.lower() or 'OutOfMemoryError' in tail
                    if proc.returncode==75 or (oom and job['frame_batch']>1):
                        job['attempt']+=1
                        if oom: job['oom_retries']+=1; job['frame_batch']=max(1,job['frame_batch']//2)
                        pending.insert(0,job); print('RESOURCE_RETRY' if proc.returncode==75 else 'OOM_RETRY',job['name'],job['attempt'],job['frame_batch'],flush=True)
                    else: failed.append(dict(name=job['name'],log=job['log'],returncode=proc.returncode))
                else:
                    verify_job(job); g.write_new(out/'job_specs'/(job['name']+'.json'),job); done+=1
                    print('DONE',job['name'],'tasks',job['task_count'],'attempt',job['attempt'],flush=True)
            available=g.gpu_snapshot()
            for gpu in ids:
                if failed or gpu in running or not pending or available[gpu]['free']<45000: continue
                job=pending.pop(0); paths=g.attempt_paths(out,job['name'],job['attempt'])
                job.update(gpu=gpu,gpu_uuid=available[gpu]['uuid'],run=paths['run'],log=paths['log'])
                signed=next(j for j in jobs if j['name']==job['name']); g.validate_job_spec(job,signed,out)
                if Path(job['run']).exists(): raise ValueError('attempt output exists')
                g.write_new(paths['spec'],job); Path(job['log']).parent.mkdir(parents=True,exist_ok=True); handle=Path(job['log']).open('x')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
                proc=subprocess.Popen([sys.executable,__file__,'worker','--job',paths['spec']],cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
                running[gpu]=(proc,job,handle); print('START',phase,'gpu',gpu,'pid',proc.pid,job['name'],flush=True)
            b.write(out/'status.json',dict(time=time.strftime('%Y-%m-%d %H:%M:%S'),phase=phase,pending=len(pending),done_in_phase=done,
                running={str(gpu):dict(name=j['name'],pid=pr.pid,log=j['log'],attempt=j['attempt']) for gpu,(pr,j,h) in running.items()},failed=failed))
            if failed and not running: raise RuntimeError('terminal recovery failure; preserve attempts')
            if pending or running: time.sleep(10)
        for j in jobs:
            if j['smoke']==smoke: verify_job(b.read(out/'job_specs'/(j['name']+'.json')))
        print('PHASE_PASSED',phase,flush=True)
    merge(out)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','worker','launch','status','merge'))
    parser.add_argument('--out',default=OUT); parser.add_argument('--job'); parser.add_argument('--gpus',default='0,2,3,4,5,6')
    a=parser.parse_args(); out=(ROOT/a.out).resolve()
    if a.mode=='prepare': prepare(out)
    elif a.mode=='worker': run(b.read(a.job))
    elif a.mode=='launch': launch(out,g.gpu_ids(a.gpus))
    elif a.mode=='merge': merge(out)
    else:
        configure(out); print(json.dumps(g.status(out),indent=2))


if __name__=='__main__': main()
