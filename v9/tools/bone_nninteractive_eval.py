"""Bone20/47/770 nnInteractive baseline: one complete prompt round per result."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
OUT='output/bone_nninteractive_single_round_20261028'
REFERENCE='output/bone_v92e360_v102e490_floor_grid_20261028_cpugpu_postdrain'
MODEL='../nninteractive_test_amos/checkpoints/nnInteractive_v1.0'
CACHE='v10/.cache/totalseg_sam2_promptgen_v10_prompt_roi_full_v1'
ALLOWED=(2,4,5,6)
METRICS=('dice','mean_slice_dice','iou','precision','recall','specificity','pred_gt_ratio','mean_slice_nsd_1mm','mean_slice_nsd_2mm','mean_slice_nsd_3mm')
ASSETS={'manifest.json':'3643ac8ae04bee3d9404f9987965d0a863067f16300445577381e31183355c00',
        'refs.json':'81f0d3f1c15fca1f56c9a363098d6f82a600d7c65174a055627e5405c8d45b78',
        'prompt_budget_audit.json':'7b1ef9e7404485d351b3b0eb530bffc8a5da6c5a650108f7b1ee86a3cd8c9869'}
DATA_HASHES={'manifest.json':'39550b42384e2049993b148bbbcc6a59f939e6293790749a7f0120b4ebf618dd',
             'tasks.json':'bc1a3a0f1f67959d055d49b6bbe7cda8b1769a84756ef6231c7a3d7cfce23f73',
             'pairing_validation.json':'223b3b67fc4a40fdedf03d32f1a1e53d9e79a4c4d8f64f84dac1f1527368bddb'}

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for value in iter(lambda:f.read(1048576),b''):h.update(value)
    return h.hexdigest()

def new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')

def atomic_new(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    new(temporary,value)
    os.link(temporary,path)
    temporary.unlink()

def state_write(out,value):
    p=out/f'status.{os.getpid()}.tmp';new(p,value);os.replace(p,out/'status.json')

def key(row):return str(row.get('source_name',row.get('dataset'))),str(row['case_id']),int(row.get('global_class_id',row.get('class_id')))

def validate_tasks(rows):
    indexed={key(r):r for r in rows}
    if len(rows)!=770 or len(indexed)!=770 or len({k[1] for k in indexed})!=20 or len({k[2] for k in indexed})!=47:raise ValueError('Bone scope requires20cases/47classes/770unique tasks')
    for k,r in indexed.items():
        if k[0]!='bone' or r['split_role']!='validation' or not 1<=int(r['local_class_id'])<=74 or k[2]!=10000+int(r['local_class_id']):raise ValueError('original local label mapping changed')
    return indexed

def plan_jobs(rows):
    validate_tasks(rows);groups=defaultdict(list)
    for r in sorted(rows,key=key):groups[r['case_id']].append(r)
    return [dict(name='bone_nn_'+case,case_id=case,tasks=values,attempt=0,minimum_free_mib=30720,not_before=0.,avoid_gpu_until=0.,last_failed_gpu=None) for case,values in groups.items()]

def validate_interaction(value):
    if any(value.get(k)!=v for k,v in dict(prediction_trigger_count=1,prompt_addition_count=6,foreground_scribbles=2,background_points=4).items()):raise ValueError('actual single-round interaction evidence differs')

def gate_ready(records):
    if not records:return False
    for proof in records.values():
        if proof['metric_execution'].get('native_compared') is not True:continue
        validate_interaction(proof['interaction']);value=proof['metric_execution']
        if value['comparisons']<1 or not math.isfinite(float(value['max_error'])) or value['max_error']>1e-6:raise ValueError('real first prediction metric gate missing')
        return proof
    raise ValueError('no actual native-compared prediction for single-round gate')

def frozen_prompt_hashes(rows,refs,budget,checksum):
    expected=set(validate_tasks(rows));by_ref={key(r):r for r in refs}
    signed={tuple(r['key']):r['prompt_sha256'] for r in budget['per_case_class']}
    if len(refs)!=770 or len(by_ref)!=770 or budget['tasks']!=770 or len(budget['per_case_class'])!=770 or len(signed)!=770 or set(by_ref)!=expected or set(signed)!=expected:raise ValueError('original signed prompt coverage changed')
    result={}
    for k,ref in by_ref.items():
        name=ref['prompt_cache_name']
        if Path(name).name!=name or name in result or checksum(name)!=signed[k]:raise ValueError('frozen original prompt signature changed')
        result[name]=signed[k]
    return result

def merge_rows(rows,tasks):
    expected=validate_tasks(tasks);found={key(r):r for r in rows}
    if len(rows)!=770 or len(found)!=770 or set(found)!=set(expected):raise ValueError('duplicate/missing metric tasks')
    for k,r in found.items():
        if int(r['label_id'])!=int(expected[k]['local_class_id']):raise ValueError('original label ID mismatch')
        if any(n not in r or not math.isfinite(float(r[n])) for n in METRICS):raise ValueError('nonfinite or missing ten metrics')
    means={n:math.fsum(float(found[k][n]) for k in sorted(found))/770 for n in METRICS}
    grouped=defaultdict(list)
    for k,r in found.items():grouped[k[2]].append(float(r['dice']))
    macro=math.fsum(math.fsum(v)/len(v) for v in grouped.values())/47
    return dict(task_count=770,mean=means,class_macro_dice=macro,per_case_class=[found[k] for k in sorted(found)])

def retry_job(job,oom,now):
    return dict(job,attempt=job['attempt']+1,minimum_free_mib=45000 if oom else job['minimum_free_mib'],not_before=now+60.,avoid_gpu_until=now+600.,last_failed_gpu=job['gpu'])

def gpu_snapshot():
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    return {int(a.strip()):dict(uuid=u.strip(),free=int(f.strip())) for a,u,f in (line.split(',') for line in text.splitlines())}

def source_paths():
    paths=[ROOT/n for n in ('tools/bone_nninteractive_runtime.py','tools/bone_nninteractive_eval.py','tools/ct13_full_validation_eval.py','tools/bone_fast_metrics.py','utils/metrics.py')]
    base=ROOT.parent/'nninteractive_test_amos'
    paths+=list((base/'infer').glob('*.py'))+list((base/'nnInteractive').rglob('*.py'))
    return sorted(set(paths))

def dataset_files(rehash=False):
    base=ROOT.parent/'data/bone'
    for name,h in DATA_HASHES.items():
        if sha(base/'paired_v1'/name)!=h:raise ValueError('original paired Bone manifest changed')
    manifest=read(base/'paired_v1/manifest.json');result={}
    for r in manifest['cases']:
        for field,hfield in (('image','image_sha256'),('label_nifti','label_nifti_sha256')):
            p=base/r[field];s=p.stat()
            if rehash and sha(p)!=r[hfield]:raise ValueError('Bone CT/GT bytes changed')
            result[os.path.relpath(p,ROOT)]=dict(size=s.st_size,mtime_ns=s.st_mtime_ns,sha256=r[hfield])
    if len(result)!=40:raise ValueError('expected20CT+20GT')
    return result

def prepare(out):
    if out.exists():raise FileExistsError('do not prepare or relaunch an existing output')
    reference=ROOT/REFERENCE
    for name,h in ASSETS.items():
        if sha(reference/name)!=h:raise ValueError('frozen Bone prompt/task asset changed')
    rows=read(reference/'manifest.json')['tasks'];validate_tasks(rows)
    refs=read(reference/'refs.json')['per_case_class']
    if len(refs)!=770 or {key(r) for r in refs}!={key(r) for r in rows}:raise ValueError('prompt refs coverage mismatch')
    prompt_hashes=frozen_prompt_hashes(rows,refs,read(reference/'prompt_budget_audit.json'),lambda name:sha(ROOT/CACHE/'metadata/shared_adaptive_prompt_v1/bone'/name))
    model_files={os.path.relpath(p,ROOT):sha(p) for p in sorted((ROOT/MODEL).rglob('*')) if p.is_file()}
    if len(model_files)!=4 or not any(p.endswith('checkpoint_final.pth') for p in model_files):raise ValueError('official nnInteractive model files incomplete')
    config=dict(reference_output=REFERENCE,model_dir=MODEL,cache_root=CACHE,prompt_refs=refs,prompt_hashes=prompt_hashes,
                model_files=model_files,sources={os.path.relpath(p,ROOT):sha(p) for p in source_paths()},dataset=dataset_files(True),reference_assets=ASSETS,
                policy=dict(allowed_gpus=list(ALLOWED),minimum_free_mib=30720,one_worker_per_gpu=True,interaction_rounds=1,prediction_triggers=1,autozoom=True))
    out.mkdir(parents=True,exist_ok=False)
    for name in ASSETS:
        with (out/name).open('xb') as f:f.write((reference/name).read_bytes())
    new(out/'jobs.json',plan_jobs(rows));config['jobs_sha256']=sha(out/'jobs.json');new(out/'protocol.json',config)
    print('BONE_NN_PREPARED',len(rows),20,flush=True)

def verify(out):
    config=read(out/'protocol.json')
    for name,h in config['sources'].items():
        if sha(ROOT/name)!=h:raise ValueError('signed nnInteractive source changed: '+name)
    for name,h in config['model_files'].items():
        if sha(ROOT/name)!=h:raise ValueError('official model file changed')
    for name,h in config['reference_assets'].items():
        if sha(out/name)!=h or sha(ROOT/config['reference_output']/name)!=h:raise ValueError('frozen input asset changed')
    if sha(out/'jobs.json')!=config['jobs_sha256'] or dataset_files()!=config['dataset']:raise ValueError('job catalogue or dataset signature changed')
    for name,h in config['prompt_hashes'].items():
        if sha(ROOT/config['cache_root']/'metadata/shared_adaptive_prompt_v1/bone'/name)!=h:raise ValueError('frozen prompt bytes changed')
    return config

def validate_attempt(job,out):
    canonical=next((j for j in read(out/'jobs.json') if j['name']==job['name']),None)
    if canonical is None or job['case_id']!=canonical['case_id'] or job['gpu'] not in ALLOWED or not isinstance(job['attempt'],int) or job['attempt']<0:raise ValueError('invalid signed case attempt')
    expected={key(r):r for r in canonical['tasks']}
    if not job['tasks'] or len({key(r) for r in job['tasks']})!=len(job['tasks']) or any(expected.get(key(r))!=r for r in job['tasks']):raise ValueError('attempt task subset mismatch')
    tag=job['name']+'_a'+str(job['attempt'])
    paths=dict(run=str(out/'runs'/tag),spec=str(out/'job_specs'/(tag+'.json')),log=str(out/'logs'/(tag+'.log')))
    if any(job.get(k)!=p for k,p in paths.items()) or job['minimum_free_mib'] not in (30720,45000):raise ValueError('attempt path/resource policy mismatch')

def class_records(out,config):
    from tools.bone_nninteractive_runtime import validate_record
    expected={key(r):r for r in read(out/'manifest.json')['tasks']};found={}
    for marker in (out/'runs').glob('*/classes/*.done.json'):
        run=marker.parents[1];spec=out/'job_specs'/(run.name+'.json');job=read(spec);validate_attempt(job,out)
        info=read(marker);path=marker.with_name(marker.name.removesuffix('.done.json')+'.json')
        if info!={'record_sha256':sha(path),'spec_sha256':sha(spec),'protocol_sha256':sha(out/'protocol.json')}:raise ValueError('complete class proof SHA mismatch')
        record=read(path);k=key(record['task'])
        if k in found or record['task']!=expected[k] or record['task'] not in job['tasks']:raise ValueError('duplicate or unsigned complete class')
        validate_record(record,expected[k],config,job);validate_interaction(record['interaction'])
        found[k]=record
    return found

def worker(out,job):
    config=verify(out);validate_attempt(job,out)
    available=gpu_snapshot()[job['gpu']]
    if available['uuid']!=job['gpu_uuid']:raise ValueError('physical GPU UUID changed')
    if available['free']<job['minimum_free_mib']:raise SystemExit(75)
    from tools.bone_nninteractive_runtime import run_case,validate_record
    run=Path(job['run']);run.mkdir(parents=True,exist_ok=False)
    def commit(record):
        task=next(t for t in job['tasks'] if key(t)==key(record['task']))
        validate_record(record,task,config,job);validate_interaction(record['interaction'])
        path=run/'classes'/(str(key(task)[2])+'.json');new(path,record)
        atomic_new(path.with_name(path.stem+'.done.json'),dict(record_sha256=sha(path),spec_sha256=sha(job['spec']),protocol_sha256=sha(out/'protocol.json')))
        print('NN_SINGLE_ROUND_CLASS_DONE',key(task),record['row']['dice'],flush=True)
    run_case(ROOT,job,config,commit)
    new(run/'complete.json',dict(tasks=len(job['tasks']),prediction_triggers_per_result=1))

def related():
    rows=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid():continue
        try:args=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')
        except OSError:continue
        if any(Path(x).name=='bone_nninteractive_eval.py' for x in args) and any(x in ('worker','run','launch') for x in args):rows.append(p.name)
    return rows

def schedule(out):
    config=verify(out)
    if related():raise RuntimeError('an nnInteractive Bone controller/worker already exists')
    new(out/'launched.json',dict(pid=os.getpid(),time=time.time()))
    active={}
    try:_schedule_loop(out,config,active)
    except BaseException as error:
        failure=dict(controller_error=repr(error))
        try:state_write(out,dict(time=time.strftime('%F %T'),pending='unknown_after_controller_error',running={str(g):dict(pid=p.pid,spec=j['spec'],log=j['log']) for g,(p,j,h) in active.items()},failed=[failure]))
        except Exception:pass
        failure['drain_errors']=drain_workers(active)
        try:state_write(out,dict(time=time.strftime('%F %T'),pending='unknown_after_controller_error',running={str(g):dict(pid=p.pid,spec=j['spec'],log=j['log']) for g,(p,j,h) in active.items()},failed=[failure]))
        except Exception:pass
        raise

def drain_workers(active):
    errors=[]
    for gpu,(proc,job,handle) in list(active.items()):
        exited=False
        try:proc.wait();exited=True
        except BaseException as error:errors.append(dict(gpu=gpu,stage='wait',error=repr(error)))
        try:handle.close()
        except Exception as error:errors.append(dict(gpu=gpu,stage='log_close',error=repr(error)))
        if exited:del active[gpu]
    return errors

def remaining_after_exit(out,config,job):
    records=class_records(out,config)
    return [t for t in job['tasks'] if key(t) not in records],records

def _schedule_loop(out,config,active):
    pending=read(out/'jobs.json');failed=[];gate=False
    while pending or active:
        records=class_records(out,config)
        for gpu,(proc,job,handle) in list(active.items()):
            if proc.poll() is None:continue
            del active[gpu];handle.close()
            if failed:continue
            remaining,records=remaining_after_exit(out,config,job)
            if not remaining and proc.returncode==0:print('NN_CASE_DONE',job['name'],flush=True);continue
            text=Path(job['log']).read_text(errors='replace')[-22000:]
            oom='CUDA out of memory' in text or 'torch.OutOfMemoryError' in text
            if proc.returncode==75 or oom:
                candidate=retry_job(job,oom,time.time());candidate['tasks']=remaining
                if remaining:pending.append(candidate)
                print('NN_RESOURCE_WAIT',job['name'],proc.returncode,oom,flush=True)
            else:failed.append(dict(name=job['name'],returncode=proc.returncode,log=job['log']))
        proof=gate_ready(records) if not gate else None
        if proof:
            gate=True;new(out/'single_round_gate.json',dict(task=proof['task'],interaction=proof['interaction'],metric_execution=proof['metric_execution']))
            print('NNINTERACTIVE_SINGLE_ROUND_GATE_PASS',flush=True)
        if not failed:
            available=gpu_snapshot();now=time.time()
            for gpu in ALLOWED:
                if gpu in active or (not gate and active):continue
                selected=next((j for j in pending if now>=j['not_before'] and available[gpu]['free']>=j['minimum_free_mib'] and not(j['last_failed_gpu']==gpu and now<j['avoid_gpu_until'])),None)
                if selected is None:continue
                pending.remove(selected);tasks=[t for t in selected['tasks'] if key(t) not in records]
                if not tasks:continue
                tag=selected['name']+'_a'+str(selected['attempt'])
                job=dict(selected,tasks=tasks,gpu=gpu,gpu_uuid=available[gpu]['uuid'],run=str(out/'runs'/tag),spec=str(out/'job_specs'/(tag+'.json')),log=str(out/'logs'/(tag+'.log')))
                validate_attempt(job,out);new(job['spec'],job)
                Path(job['log']).parent.mkdir(parents=True,exist_ok=True);handle=Path(job['log']).open('x')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
                proc=subprocess.Popen([sys.executable,str(Path(__file__)),'worker','--job',job['spec']],cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
                active[gpu]=(proc,job,handle);print('NN_START',gpu,proc.pid,tag,len(tasks),flush=True)
        state_write(out,dict(time=time.strftime('%F %T'),pending=len(pending),running={str(g):dict(pid=p.pid,spec=j['spec'],log=j['log']) for g,(p,j,h) in active.items()},failed=failed,complete_tasks=len(records),single_round_gate=gate))
        if failed and not active:raise RuntimeError('non-resource failure; all peers drained; evidence retained: '+repr(failed))
        if pending or active:time.sleep(10)

def merge(out):
    state=read(out/'status.json')
    if state['pending'] or state['running'] or state['failed'] or related():raise ValueError('merge requires drained successful770 task queue')
    config=verify(out);dataset_files(True);records=class_records(out,config)
    result=merge_rows([r['row'] for r in records.values()],read(out/'manifest.json')['tasks'])
    result.update(model='nnInteractive',scope='Bone20/47/770 single-round prompt comparison',interaction_rounds_per_task=1,
                  actual_prediction_triggers=sum(r['interaction']['prediction_trigger_count'] for r in records.values()),protocol_sha256=sha(out/'protocol.json'))
    new(out/'merged_metrics.json',result);print('BONE_NN_SINGLE_ROUND_MERGED_770',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('prepare','run','launch','worker','status','merge'));p.add_argument('--job');a=p.parse_args();out=ROOT/OUT
    if a.mode=='prepare':prepare(out)
    elif a.mode=='run':prepare(out);schedule(out);merge(out)
    elif a.mode=='launch':schedule(out);merge(out)
    elif a.mode=='worker':worker(out,read(a.job))
    elif a.mode=='merge':merge(out)
    else:
        config=verify(out);records=class_records(out,config)
        print(json.dumps(dict(queue=read(out/'status.json'),audited=len(records),target=770),indent=2))

if __name__=='__main__':main()
