"""v9.2-only external evaluation reusing the frozen four-source preparation."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import four_source_eval as base

ROOT = base.ROOT
WEIGHT = 'output/sam2_v9_2_ct13_semantic_e380_400_20260928/20260928_200027/epoch360_dice0.7817.pth'
SOURCE = base.OUT
OUT = 'output/four_source_v92e360_30case_20261011'
PREP_FILES = ('selected_cases.json', 'manifest.json', 'refs.json', 'split.json', 'geometry_audit.json')


def link_preparation(source, out):
    source, out = Path(source).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name in PREP_FILES:
        target = out/name
        if target.is_symlink() or target.exists():
            if target.resolve() != source/name:
                raise ValueError('preparation points to a different source')
        else:
            if not (source/name).is_file():
                raise FileNotFoundError(source/name)
            target.symlink_to(source/name)


def validate_rows(rows, expected, refs):
    base.validate_model_rows(rows, expected, 'v10_2')  # same threshold audit contract
    base.validate_prompt_signatures(rows, refs)


def fingerprint(out):
    old = base.WEIGHT
    try:
        base.WEIGHT = WEIGHT
        value = base.fingerprint(out)
    finally:
        base.WEIGHT = old
    value['code']['tools/four_source_v92_eval.py'] = base.sha(Path(__file__))
    value.update(model='v9_2', checkpoint_epoch=360, context_frames=192,
                 reference_source=str((out/'manifest.json').resolve().parent))
    return value


def make_jobs(rows, out, shards=6):
    return [dict(name=f'v9_2_{s}_s{i:02}', model='v9_2', source=s, tasks=part,
                 out=str(out), frame_batch=16, attempt=0)
            for s in base.SOURCES
            for i, part in enumerate(base.shard_cases([r for r in rows if r['source_name']==s], shards)) if part]


def smoke_jobs(rows, out, gpus):
    jobs = [j for j in base.smoke_jobs(rows, out, gpus) if j['model']=='v10_2']
    for j in jobs:
        j.update(model='v9_2', name=j['name'].replace('v10_2', 'v9_2'))
    return jobs


def prepare(out):
    source = (ROOT/SOURCE).resolve()
    link_preparation(source, out)
    rows = base.read(out/'manifest.json')['tasks']
    if len(rows)!=180 or base.manifest(rows)['per_source_task_counts']!={'fumpe':28,'graz':60,'pengwin':62,'adrenal':30}:
        raise ValueError('unexpected frozen four-source coverage')
    base.prompt_refs(out, rows)
    base.bind_protocol(out/'protocol.json', fingerprint(out))
    jobs = make_jobs(rows, out)
    if (out/'jobs.json').exists() and base.read(out/'jobs.json')!=jobs:
        raise ValueError('job plan changed')
    base.write(out/'jobs.json', jobs)
    print('PREPARED 180 tasks; existing case references and prompts only', flush=True)


def run_v9(job, rows, attempt):
    import numpy as np
    import torch
    out = Path(job['out'])
    refs = base.prompt_refs(out, rows)
    index = {base.task_key(r):r for r in refs}
    base.write(attempt/'manifest.json', base.manifest(rows))
    base.write(attempt/'refs.json', {'per_case_class':refs})
    sys.argv = [sys.argv[0], '--independent-task-manifest', str(attempt/'manifest.json'),
        '--nninteractive-metrics', str(attempt/'refs.json'), '--shared-adaptive-prompt-root', str(ROOT/base.CACHE),
        '--source',job['source'],'--eval-out-dir',str(attempt),'--run-name','result',
        '--gpu',str(job['gpu']),'--sam2-frame-batch-size',str(job['frame_batch']),
        '--base-split-json',str(out/'split.json'),'--checkpoint-dir',str((ROOT/WEIGHT).parent),
        '--v9-2-weights',str(ROOT/WEIGHT),'--allow-uniform-weight','--exact-background-mode','point',
        '--data-root',str(ROOT/'data/CT'),'--train-end','1']
    import v9_2_extended_protocol as protocol
    import finetune_multisource_sam2_v9_2_precision as original
    protocol.SOURCE_ORDER = original.SOURCE_ORDER = base.SOURCES
    original.validate_source_ratios = lambda _r: {s:.25 for s in base.SOURCES}
    # Generic single-file factory applies identical canonical preprocessing to new sources.
    from v9_family_shared_cache_entry import install_multidataset_factories
    install_multidataset_factories(original.multi)
    import v9_totalseg_runtime as runtime
    unified = base.metric_module()
    def binary(prediction, target, spacing=None, **_kw):
        values = unified.binary_prompt_metrics_from_masks(torch.from_numpy(np.asarray(prediction)),
            torch.from_numpy(np.asarray(target)), spacing_dhw=spacing)
        return {**values,'volume_nsd_1mm':float('nan'),'volume_hd95_mm':float('nan')}
    runtime.base.binary_volume_metrics_from_prediction = binary
    live=[]; prompt_audit={}
    import infer.v9_2_dynamic_validation_patch as dynamic
    installer = dynamic.install_dynamic_validation_patch
    def install(rt, audit_rows, **settings):
        installer(rt, audit_rows, **settings)
        adaptive = rt._evaluate_adaptive_prompt_roi_task
        def final(*args, **kwargs):
            metric = adaptive(*args, **kwargs)
            key = tuple(prompt_audit['current']['loaded_task_keys'][-1])
            row = next(r for r in rows if base.task_key(r)==key)
            if int(args[5])!=int(row['global_class_id']):
                raise RuntimeError('metric task/class alignment mismatch')
            ref=index[key]; a=audit_rows[-1]
            live.append({**row,**{m:float(metric[m]) for m in base.METRICS},
                'mask_threshold_used':float(a['threshold']),'dynamic_threshold_source':a['source'],
                'prompt_sha256':ref['prompt_sha256'],'prompt_cache_name':ref['prompt_cache_name']})
            base.write(attempt/'all_metrics_live.json', {'per_case_class':live})
            print(f"V92 {key} {len(live)}/{len(rows)} Dice={metric['dice']:.5f}",flush=True)
            return metric
        rt._evaluate_adaptive_prompt_roi_task=final
    dynamic.install_dynamic_validation_patch=install
    import infer.infer_eight_source_adaptive_v9_2 as ev
    exact=ev._install_exact_prompt_factory
    def bind(entry,tasks,idx,root,audit,mode):
        prompt_audit['current']=audit
        return exact(entry,tasks,idx,root,audit,mode)
    ev._install_exact_prompt_factory=bind
    ev.checkpoint_plan=lambda _p:{s:{'path':str(ROOT/WEIGHT),'epoch':360,'selection':'fixed_best'} for s in base.SOURCES}
    append=ev._append_flag
    def context(arguments,flag,*values):
        append(arguments,flag,*((192,) if flag in ('--max-frames','--validation-window-frames') else values))
    ev._append_flag=context
    ev.main()
    audit=base.read(attempt/'result/exact_prompt_audit.json')
    if audit['loaded_task_count']!=len(rows) or audit['dynamic_threshold_summary']['count']!=len(rows):
        raise ValueError('exact prompt/threshold coverage mismatch')
    validate_rows(live,rows,refs)
    return live


def worker(job):
    out=Path(job['out']);base.bind_protocol(out/'protocol.json',fingerprint(out))
    os.environ.update(CUDA_VISIBLE_DEVICES=str(job['gpu']),V9_SHARED_CACHE_WRITE_POLICY='read_only',
                      PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    import torch
    torch.set_num_threads(2)
    uuid='GPU-'+str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')
    if uuid!=job['gpu_uuid']:raise RuntimeError('GPU UUID/CUDA mapping changed')
    base.InMemoryPairs((out/'manifest.json').resolve().parent).install()
    run=out/'jobs'/job['name'];prior={}
    for p in sorted(run.glob('attempt_*/all_metrics_live.json')):
        for r in base.read(p)['per_case_class']:prior[base.task_key(r)]=r
    rows=sorted([r for r in job['tasks'] if base.task_key(r) not in prior],key=base.task_key)
    attempt=run/f"attempt_{job['attempt']:02}";attempt.mkdir(parents=True,exist_ok=True)
    if rows:
        for r in run_v9(job,rows,attempt):prior[base.task_key(r)]=r
    results=list(prior.values());validate_rows(results,job['tasks'],base.prompt_refs(out,job['tasks']))
    base.write(run/'complete_metrics.json',dict(model='v9_2',checkpoint=WEIGHT,per_case_class=results))


def active(out):
    found=[]
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args=p.read_bytes().decode(errors='replace').split('\0')
            if 'worker' in args and '--job' in args and any(Path(a).name==Path(__file__).name for a in args if a):
                if Path(args[args.index('--job')+1]).resolve().parent==(out/'job_specs').resolve():found.append(int(p.parent.name))
        except (OSError,ValueError,IndexError):pass
    return found


def inventory():
    q=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    return {int(v[0]):(v[1].strip(),int(v[2])) for v in (l.split(',') for l in q.splitlines())}


def start(out,job,gpu,inv):
    job.update(gpu=gpu,gpu_uuid=inv[gpu][0])
    run=out/'jobs'/job['name'];existing=[int(p.name.split('_')[-1]) for p in run.glob('attempt_*')]
    if existing:job['attempt']=max(job['attempt'],max(existing)+1)
    spec=out/'job_specs'/f"{job['name']}.json";base.write(spec,job)
    log=out/'logs'/f"{job['name']}_a{job['attempt']:02}.log";log.parent.mkdir(parents=True,exist_ok=True)
    job['log']=str(log);h=log.open('a')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    p=subprocess.Popen([str(ROOT.parent/'conda-envs/al/bin/python'),str(Path(__file__).resolve()),'worker','--job',str(spec)],
        cwd=ROOT,env=env,stdout=h,stderr=subprocess.STDOUT)
    print(f"START {job['name']} GPU{gpu} {inv[gpu][0]} pid={p.pid}",flush=True)
    return p,job,h


def execute(out,gpus,smoke=False):
    import fcntl
    lock=(out/'scheduler.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    base.bind_protocol(out/'protocol.json',fingerprint(out))
    if active(out):raise RuntimeError('existing workers; refuse duplicate launch')
    jobs=smoke_jobs(base.read(out/'manifest.json')['tasks'],out,gpus) if smoke else base.read(out/'jobs.json')
    pending=[j for j in jobs if not (out/'jobs'/j['name']/'complete_metrics.json').exists()]
    running={};failed=[]
    try:
        while pending or running:
            for gpu,(p,j,h) in list(running.items()):
                if p.poll() is None:continue
                h.close();del running[gpu]
                if p.returncode:
                    tail=Path(j['log']).read_text(errors='replace')[-14000:]
                    if not smoke and 'out of memory' in tail.lower() and j['attempt']<5:
                        j['frame_batch']=max(1,j['frame_batch']//2);j['attempt']+=1;pending.append(j)
                    else:failed.append(dict(job=j['name'],log=j['log'],returncode=p.returncode))
                else:print('DONE',j['name'],flush=True)
            inv=inventory()
            for gpu in gpus:
                if gpu in running or not pending or inv[gpu][1]<45000:continue
                j=pending.pop(0);running[gpu]=start(out,j,gpu,inv)
            base.write(out/('smoke_status.json' if smoke else 'status.json'),dict(time=time.strftime('%Y-%m-%d %H:%M:%S'),
                pending=len(pending),running={str(g):dict(name=j['name'],pid=p.pid,log=j['log']) for g,(p,j,h) in running.items()},failed=failed))
            if pending or running:time.sleep(10)
        if failed:raise RuntimeError(f'worker failures: {failed}')
        if smoke:
            for j in jobs:validate_rows(base.read(out/'jobs'/j['name']/'complete_metrics.json')['per_case_class'],j['tasks'],base.prompt_refs(out,j['tasks']))
            base.write(out/'smoke_audit.json',dict(passed=True,jobs=[j['name'] for j in jobs]))
            print('SMOKE PASS',flush=True)
        else:merge(out)
    finally:
        for p,j,h in running.values():
            if p.poll() is None:p.terminate()
        for p,j,h in running.values():p.wait();h.close()


def status(out):
    jobs=base.read(out/'jobs.json');done=0;keys=set()
    for j in jobs:
        run=out/'jobs'/j['name'];p=run/'complete_metrics.json'
        files=[p] if p.exists() else sorted(run.glob('attempt_*/all_metrics_live.json'))
        done+=int(p.exists())
        for f in files:keys.update(base.task_key(r) for r in base.read(f)['per_case_class'])
    return dict(completed_jobs=done,total_jobs=len(jobs),completed_tasks=len(keys),expected_tasks=180,
                scheduler=base.read(out/'status.json') if (out/'status.json').exists() else None)


def merge(out):
    base.bind_protocol(out/'protocol.json',fingerprint(out));rows=[]
    for j in base.read(out/'jobs.json'):rows.extend(base.read(out/'jobs'/j['name']/'complete_metrics.json')['per_case_class'])
    expected=base.read(out/'manifest.json')['tasks'];validate_rows(rows,expected,base.prompt_refs(out,expected))
    result=dict(model='v9_2',checkpoint=WEIGHT,epoch=360,task_count=len(rows),per_case_class=rows,
        per_source={s:dict(tasks=len(v),**{m:sum(r[m] for r in v)/len(v) for m in base.METRICS})
            for s in base.SOURCES for v in [[r for r in rows if r['source_name']==s]]})
    base.write(out/'merged_metrics.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('prepare','worker','smoke','schedule','status','merge'))
    p.add_argument('--out',default=OUT);p.add_argument('--job');p.add_argument('--gpus',default='0,3,4,5')
    args=p.parse_args();base.setup();out=(ROOT/args.out).resolve()
    if args.mode=='prepare':prepare(out)
    elif args.mode=='worker':worker(base.read(args.job))
    elif args.mode=='status':print(json.dumps(status(out),indent=2))
    elif args.mode=='merge':merge(out)
    else:
        if args.mode=='schedule' and not base.read(out/'smoke_audit.json').get('passed'):raise RuntimeError('smoke required')
        execute(out,[int(v) for v in args.gpus.split(',')],smoke=args.mode=='smoke')


if __name__=='__main__':main()
