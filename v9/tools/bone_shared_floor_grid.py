"""Case-grouped Bone floor grid with exact native stage reuse and audited recovery."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools import bone_floor_memory_wait_recovery as parent
g,b,p=parent.g,parent.b,parent.p
OUT='output/bone_v92e360_v102e490_shared_floor_grid_20261027'
OLD=parent.OUT
SOURCES=('tools/bone_shared_logits.py','tools/bone_shared_floor_runtime.py','tools/bone_shared_floor_grid.py')
ALLOWED=(2,4,5,6)


def sha(path):return parent.s.sha(path)


def atomic_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp_'+str(os.getpid()))
    with temp.open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.link(temp,path)
    temp.unlink()


def case_jobs(tasks,completed,history):
    grouped=defaultdict(list)
    for task in sorted(tasks,key=p.key):grouped[p.key(task)[:2]].append(task)
    result=[]
    for (_,case),rows in sorted(grouped.items()):
        for model in p.MODELS:
            pending=[];inherited={}
            for task in rows:
                floors=[f for f in p.FLOORS if (model,f,p.key(task)) not in completed]
                if not floors:continue
                pending.append(dict(task=task,floors=floors))
                for floor in floors:
                    h=history.get((model,floor,p.key(task)))
                    if h:inherited[h['name']]=h
            if not pending:continue
            values=list(inherited.values())
            avoid={}
            for value in values:
                gpu=value.get('last_failed_gpu')
                if gpu is not None:avoid[str(gpu)]=max(avoid.get(str(gpu),0.),value.get('avoid_gpu_until',0.))
            result.append(dict(name='shared_'+model+'_'+case,model=model,case_id=case,source='bone',variant='baseline',
                pending=pending,task_count=len(pending),frame_batch=min([2]+[v['frame_batch'] for v in values]),
                minimum_free_mib=max([30720]+[v.get('minimum_free_mib',30720) for v in values]),
                parent_history=values,attempt=0,oom_retries=0,not_before=max([0.]+[v.get('not_before',0.) for v in values]),
                avoid_gpus=avoid,avoid_gpu_until=0.,last_failed_gpu=None,control=False))
    return result


def retry(job,code,text,now):
    if code==0:return None
    oom='cuda out of memory' in text.lower() or 'OutOfMemoryError' in text
    if code!=75 and not oom:return None
    result=dict(job,attempt=job['attempt']+1,not_before=now+20)
    if oom:
        result['oom_retries']=job['oom_retries']+1
        if job['frame_batch']>1:result['frame_batch']=job['frame_batch']//2
        else:result.update(minimum_free_mib=45000,not_before=now+60,last_failed_gpu=job['gpu'],avoid_gpu_until=now+600,
            avoid_gpus={**job.get('avoid_gpus',{}),str(job['gpu']):now+600})
    return result


def attempt_outcome(code,text,remaining):
    resource=code==75 or 'cuda out of memory' in text.lower() or 'OutOfMemoryError' in text
    if code and not resource:return 'fatal'
    if not remaining:return 'done'
    return 'retry' if resource else 'fatal'


def require_handoff(proofs,live):
    if live or not proofs or any(not x.get('verified') for x in proofs):
        raise ValueError('parent handoff not independently drained/audited')


def validate_attempt(job,canonical,out):
    if any(job.get(k)!=canonical[k] for k in ('name','model','control','parent_history','variant','source','case_id','prepared')):
        raise ValueError('unsigned case/model/history binding')
    if (job.get('gpu') not in ALLOWED or not isinstance(job.get('gpu_uuid'),str)
            or job.get('frame_batch') not in (1,2) or job['frame_batch']>canonical['frame_batch']
            or job.get('minimum_free_mib') not in (30720,45000)
            or job['minimum_free_mib']<canonical['minimum_free_mib']):
        raise ValueError('invalid inherited GPU/batch/memory policy')
    attempt=job.get('attempt');oom=job.get('oom_retries')
    if type(attempt) is not int or attempt<0 or type(oom) is not int or not 0<=oom<=attempt:
        raise ValueError('invalid new-group attempt/OOM history')
    tag=canonical['name']+'_a'+str(attempt)
    expected=dict(run=str(out/'runs'/tag),spec=str(out/'job_specs'/(tag+'.json')),log=str(out/'logs'/(tag+'.log')))
    if any(job.get(k)!=v for k,v in expected.items()):raise ValueError('non-independent attempt path')


def artifact_files(job):
    return [Path(job['run'])/name for name in parent.artifact_names(job)]


def reused_rows(job):
    # Legacy v9 metric rows lack threshold source; native helper reconstructs it
    # from the audited final calibration stage and retains original local IDs.
    return g.metric_rows(job)


def parent_limit(job,latest,prior,oom=False):
    if latest is not None:return dict(latest,**parent.resume_settings(latest,oom=oom))
    return dict(job,**prior) if prior else dict(job,minimum_free_mib=30720,never_launched=True)


def eligible(job,gpu,available,now):
    return (now>=job['not_before'] and available[gpu]['free']>=job['minimum_free_mib']
        and now>=job.get('avoid_gpus',{}).get(str(gpu),0.)
        and not (job.get('last_failed_gpu')==gpu and now<job.get('avoid_gpu_until',0.)))


def validate_class(record,expected):
    if p.key(record['task'])!=p.key(expected['task']) or record['model'] not in p.MODELS:
        raise ValueError('class proof identity mismatch')
    results=record['results']
    if len(results)!=len(expected['floors']) or {r['floor'] for r in results}!=set(expected['floors']):
        raise ValueError('incomplete class floor result')
    model=record['model'];proof=record['restoration'];counts=(438,192,360) if model=='v9_2' else (418,212,490)
    if not proof['guards_passed'] or proof['model']!=model or proof['weight_mode']!='live' or (proof['prompt_parameters_exact'],proof['decoder_parameters_exact'],proof['epoch'])!=counts:
        raise ValueError('class restoration audit mismatch')
    if record['training_audit']['training_dataset_constructions_skipped']!=(1 if model=='v9_2' else 0):raise ValueError('training dataset audit')
    for result in results:
        p.validate_rows([result['row']],[expected['task']]);g.validate_stages(result['stages'],[expected['task']],model,result['floor'])
        if result['row']['mask_threshold_used']!=result['stages'][-1]['threshold']:raise ValueError('final threshold mismatch')
        for stage in result['stages']:
            producer=record['raw_producers'][stage['raw_key']]
            if producer['raw_key']!=stage['raw_key'] or producer['frames']!=stage['frames']:raise ValueError('raw-stage producer mismatch')
            if model=='v10_2' and producer['actual_memory_calls']['memory_encoder']<=0:raise ValueError('memory producer not exercised')
    return results


def require_marker_files(info,marker,out,canonical):
    run=marker.parents[1];prefix=canonical['name']+'_a'
    if run.parent!=out/'runs' or not run.name.startswith(prefix) or not run.name[len(prefix):].isdigit():raise ValueError('class marker outside signed attempt')
    record=marker.with_name(marker.name.removesuffix('.done.json')+'.json')
    required={str(f) for f in (record,run/'restoration_start.json',run/'resource_policy_audit.json',out/'job_specs'/(run.name+'.json'),run/'prepared/manifest.json',run/'prepared/refs.json',run/'prepared_input_audit.json')}
    if info['record']!=str(record) or set(info['files'])!=required:raise ValueError('class marker omits/misbinds mandatory proof paths')
    return run


def validate_worker_inputs(spec,canonical,manifest,refs,frozen_refs):
    expected={p.key(x['task']):x for x in canonical['pending']}
    current=spec['pending'];keys=[p.key(x['task']) for x in current]
    if len(keys)!=len(set(keys)) or spec['task_count']!=len(keys) or any(expected.get(k)!=x for k,x in zip(keys,current)):
        raise ValueError('worker prepared task is not signed pending subset')
    if manifest['tasks']!=[x['task'] for x in current] or manifest['expected_task_count']!=len(current):raise ValueError('prepared manifest path/scope changed')
    by_key={p.key(r):r for r in frozen_refs['per_case_class']}
    if refs['per_case_class']!=[by_key[k] for k in keys]:raise ValueError('prepared prompt refs changed')


def require_merge_state(state,live):
    if live or state.get('phase')!='full' or state.get('pending') or state.get('running') or state.get('failed'):
        raise ValueError('final merge requires successful drained full queue')


def expected_class(spec,task):
    found=[x for x in spec['pending'] if p.key(x['task'])==p.key(task)]
    if len(found)!=1 or found[0]['task']!=task:raise ValueError('class proof not in exact attempt task subset')
    return found[0]


def _context(out):
    b.SPLIT=str((out/'bone_split.json').relative_to(ROOT));g.amos.verify=verify


def signature(out):
    return dict(parent_protocol=sha(ROOT/OLD/'protocol.json'),parent_signature=parent.signature(ROOT/OLD),
        sources={n:sha(ROOT/n) for n in SOURCES},
        assets={n:sha(out/n) for n in ('bone_split.json','manifest.json','refs.json','prompt_budget_audit.json','jobs.json','reuse_audit.json')},
        protocol='same18x770 native floors; shared exact raw stages before calibration; independent AutoExpand; CPU RAM only; fullmask10')


def verify(out):
    if signature(out)!=b.read(out/'protocol.json'):raise ValueError('changed shared-grid signature')
    parent.configure(ROOT/OLD);parent.verify(ROOT/OLD);parent.engine(ROOT/OLD).verify_canaries(ROOT/OLD)
    parent.s.verify_files(b.read(out/'reuse_audit.json')['files']);_context(out)


def related():
    found=[]
    for f in Path('/proc').iterdir():
        if not f.name.isdigit() or int(f.name)==os.getpid():continue
        try:args=(f/'cmdline').read_bytes().decode(errors='replace').split('\0')
        except OSError:continue
        if any(Path(x).name in ('bone_floor_memory_wait_recovery.py','bone_shared_floor_grid.py') for x in args) and any(x in ('worker','wait-launch','launch','controls') for x in args):found.append(f.name)
    return found


def finish_parent():
    evidence=b.read(ROOT/'output/bone_shared_handoff_poststop_20261027.json');scheduler=evidence['scheduler'];pid=scheduler['pid']
    while True:
        live=[x for x in related() if int(x)!=pid]
        if not live:break
        print('WAIT_NATIVE_WORKERS_DRAIN',live,flush=True);time.sleep(30)
    proc=Path('/proc')/str(pid)
    if proc.exists():
        state=(proc/'stat').read_text().split();args=(proc/'cmdline').read_bytes().decode().split('\0')
        if state[21]!=scheduler['starttime'] or args!=scheduler['args'] or state[2]!='T' or proc.stat().st_uid!=os.getuid():raise ValueError('frozen scheduler identity/state changed')
        os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
        for _ in range(30):
            if not proc.exists() or (proc/'stat').read_text().split()[2]=='Z':break
            time.sleep(1)
    time.sleep(3)
    require_handoff([dict(verified=True)],related());time.sleep(3);require_handoff([dict(verified=True)],related())
    return dict(verified=True,scheduler=scheduler,all_workers_naturally_exited=True,old_status_not_modified=True)


def collect_parent():
    old=ROOT/OLD;parent.configure(old);parent.verify(old);parent.engine(old).verify_canaries(old)
    completed={};specs={};files={};history={};recovered=[];prior_resume=b.read(old/'reuse_audit.json')['resume']
    for job in b.read(old/'jobs.json'):
        canonical=old/'job_specs'/(job['name']+'.json');chosen=None
        attempts=sorted((b.read(f) for f in (old/'job_specs').glob(job['name']+'_a*.json')),key=lambda j:j['attempt'],reverse=True)
        if canonical.exists():chosen=b.read(canonical);parent.verify_job(chosen);specfile=canonical
        else:
            for attempt in attempts:
                if all(f.exists() for f in artifact_files(attempt)):
                    parent.verify_job(attempt);chosen=attempt;specfile=old/'job_specs'/(job['name']+'_a'+str(attempt['attempt'])+'.json');recovered.append(job['name']);break
        if chosen:
            rows=reused_rows(chosen)
            for row in rows:
                identity=(job['model'],job['floor'],p.key(row))
                if identity in completed:raise ValueError('duplicate readonly result')
                completed[identity]=row
            specs[job['name']]=str(specfile);files[str(specfile)]=sha(specfile)
            for f in artifact_files(chosen):files[str(f)]=sha(f)
        if attempts:
            latest=attempts[0];text=Path(latest['log']).read_text(errors='replace')[-10000:]
            inherited=parent_limit(job,latest,None,oom='cuda out of memory' in text.lower() or 'OutOfMemoryError' in text)
        else:inherited=parent_limit(job,None,prior_resume.get(job['name']))
        for task in b.read(Path(job['prepared'])/'manifest.json')['tasks']:history[(job['model'],job['floor'],p.key(task))]=inherited
    for folder in ('logs','job_specs'):
        for f in (old/folder).iterdir():
            if f.is_file():files[str(f)]=sha(f)
    for name in ('protocol.json','reuse_audit.json','jobs.json','status.json','manifest.json','refs.json','prompt_budget_audit.json','bone_split.json'):
        files[str(old/name)]=sha(old/name)
    for name in ('bone_shared_handoff_20261027.json','bone_shared_handoff_poststop_20261027.json'):files[str(ROOT/'output'/name)]=sha(ROOT/'output'/name)
    return completed,history,specs,files,recovered


def prepare(out,handoff):
    if out.exists():raise FileExistsError('independent shared output already exists')
    require_handoff([handoff],related())
    completed,history,specs,files,recovered=collect_parent();old=ROOT/OLD
    tasks=b.read(old/'manifest.json')['tasks'];jobs=case_jobs(tasks,set(completed),history)
    out.mkdir(parents=True)
    for name in ('bone_split.json','manifest.json','refs.json','prompt_budget_audit.json'):(out/name).symlink_to(old/name)
    refs={p.key(r):r for r in b.read(old/'refs.json')['per_case_class']}
    for job in jobs:
        prepared=out/'prepared'/job['name'];rows=[x['task'] for x in job['pending']]
        b.write(prepared/'manifest.json',g.bone_payload(rows));b.write(prepared/'refs.json',dict(per_case_class=[refs[p.key(t)] for t in rows]));job['prepared']=str(prepared)
    original_jobs=b.read(old/'jobs.json')
    control_task=next(j for j in original_jobs if j['smoke'] and j['model']=='v9_2' and j['floor']==.35)
    task=b.read(Path(control_task['prepared'])/'manifest.json')['tasks'][0]
    controls=[]
    for model in p.MODELS:
        job=dict(name='control_'+model,model=model,source='bone',variant='baseline',case_id=task['case_id'],pending=[dict(task=task,floors=list(p.FLOORS))],task_count=1,frame_batch=2,minimum_free_mib=30720,parent_history=[],attempt=0,oom_retries=0,not_before=0.,avoid_gpu_until=0.,last_failed_gpu=None,control=True)
        prepared=out/'prepared'/job['name'];b.write(prepared/'manifest.json',g.bone_payload([task]));b.write(prepared/'refs.json',dict(per_case_class=[refs[p.key(task)]]));job['prepared']=str(prepared);controls.append(job)
    b.write(out/'jobs.json',dict(cases=jobs,controls=controls))
    serial=[dict(model=m,floor=f,row=r) for (m,f,k),r in completed.items()]
    b.write(out/'reuse_audit.json',dict(files=files,specs=specs,completed=serial,handoff=handoff,
        recovered_drained_attempts=recovered,reused_tasks=len(serial),pending_tasks=13860-len(serial),partial_metrics_reused=False))
    b.write(out/'protocol.json',signature(out));verify(out)
    print('SHARED_GRID_PREPARED',len(serial),'readonly',13860-len(serial),'pending',len(jobs),'case_model_jobs',flush=True)


def class_records(out,control=False):
    canonical=b.read(out/'jobs.json');jobs={j['name']:j for j in canonical['controls' if control else 'cases']};found={}
    for marker in (out/'runs').glob('*/classes/*.done.json'):
        info=b.read(marker)
        if info['job_name'] not in jobs:continue
        if info['protocol_sha']!=sha(out/'protocol.json'):raise ValueError('class proof belongs to another protocol')
        job=jobs[info['job_name']];run=require_marker_files(info,marker,out,job)
        parent.s.verify_files(info['files']);record=b.read(info['record'])
        if record['model']!=job['model']:raise ValueError('class proof model differs from signed job')
        spec=b.read(out/'job_specs'/(run.name+'.json'));validate_attempt(spec,job,out)
        validate_worker_inputs(spec,job,b.read(run/'prepared/manifest.json'),b.read(run/'prepared/refs.json'),b.read(out/'refs.json'))
        prepared=b.read(run/'prepared_input_audit.json')
        if prepared['canonical_prepared']!=spec['prepared'] or prepared['files']!={str(run/'prepared'/name):sha(run/'prepared'/name) for name in ('manifest.json','refs.json')}:raise ValueError('prepared proof mismatch')
        if record['restoration']!=b.read(run/'restoration_start.json'):raise ValueError('class restoration proof differs from restored worker')
        resource=b.read(run/'resource_policy_audit.json')
        if any(resource[k]!=spec[k] for k in ('gpu','gpu_uuid','frame_batch','minimum_free_mib','parent_history')) or resource['actual_start_free_mib']<spec['minimum_free_mib'] or resource['new_oom_retries']!=spec['oom_retries']:raise ValueError('class actual resource audit mismatch')
        expected=expected_class(spec,record['task'])
        validate_class(record,expected)
        identity=(job['model'],p.key(record['task']))
        if identity in found:raise ValueError('duplicate committed shared class')
        found[identity]=dict(record=record,marker=str(marker),marker_sha=sha(marker))
    return found


def run(job):
    out=Path(job['run']).parents[1];out=out.parent if out.name=='runs' else out
    verify(out);catalog=b.read(out/'jobs.json');canonical=next(j for j in catalog['controls' if job['control'] else 'cases'] if j['name']==job['name'])
    if out!=(ROOT/OUT):raise ValueError('dedicated shared output required')
    validate_attempt(job,canonical,out)
    prior=class_records(out,job['control']);expected=[x for x in canonical['pending'] if (job['model'],p.key(x['task'])) not in prior]
    if job['pending']!=expected or job['task_count']!=len(expected):raise ValueError('worker remaining-class coverage mismatch')
    snapshot=g.gpu_snapshot()[job['gpu']]
    if snapshot['uuid']!=job['gpu_uuid']:raise ValueError('physical UUID mismatch')
    if snapshot['free']<job['minimum_free_mib']:raise SystemExit(75)
    run=Path(job['run']);run.mkdir(parents=True,exist_ok=False)
    atomic_new(run/'resource_policy_audit.json',dict(gpu=job['gpu'],gpu_uuid=job['gpu_uuid'],frame_batch=job['frame_batch'],minimum_free_mib=job['minimum_free_mib'],actual_start_free_mib=snapshot['free'],parent_history=canonical['parent_history'],new_oom_retries=job['oom_retries']))
    prepared=run/'prepared';rows=[x['task'] for x in expected];refs={p.key(x):x for x in b.read(out/'refs.json')['per_case_class']}
    b.write(prepared/'manifest.json',g.bone_payload(rows));b.write(prepared/'refs.json',dict(per_case_class=[refs[p.key(t)] for t in rows]))
    validate_worker_inputs(job,canonical,b.read(prepared/'manifest.json'),b.read(prepared/'refs.json'),b.read(out/'refs.json'))
    prepared_hashes={str(prepared/name):sha(prepared/name) for name in ('manifest.json','refs.json')}
    atomic_new(run/'prepared_input_audit.json',dict(canonical_prepared=job['prepared'],files=prepared_hashes))
    job=dict(job,prepared=str(prepared))
    from tools.bone_shared_floor_runtime import GridSession,install_v92,install_v10,install_restoration,tag_carrier
    def save_proof(proof):atomic_new(run/'restoration_start.json',proof)
    def committed(record):
        name=record['task']['case_id']+'__'+str(p.key(record['task'])[2]);f=run/'classes'/(name+'.json')
        validate_class(record,next(x for x in expected if p.key(x['task'])==p.key(record['task'])))
        parent.s.verify_files(prepared_hashes)
        atomic_new(f,record)
        files={str(f):sha(f),str(run/'restoration_start.json'):sha(run/'restoration_start.json'),str(run/'resource_policy_audit.json'):sha(run/'resource_policy_audit.json'),job['spec']:sha(job['spec']),str(run/'prepared_input_audit.json'):sha(run/'prepared_input_audit.json'),**prepared_hashes}
        atomic_new(run/'classes'/(name+'.done.json'),dict(job_name=job['name'],record=str(f),files=files,protocol_sha=sha(out/'protocol.json')))
        print('SHARED_CLASS_DONE',job['model'],p.key(record['task']),len(record['results']),'raw',record['cache_stats']['raw'],flush=True)
    session=GridSession(job['model'],expected,json.dumps(b.read(out/'protocol.json'),sort_keys=True),committed)
    g.install_source_hooks(out);_context(out)
    if job['model']=='v9_2':install_v92(session)
    else:
        sys.path.insert(0,str(ROOT/'v10'))
        import v10_data
        g.install_readonly_ct(v10_data.SAM2CachedTotalSegV10Dataset)
        import v10.finetune_totalseg_amos_magic_v10_2_joint as entry
        original_text=entry.conditioning_text_from_metadata
        entry.conditioning_text_from_metadata=lambda row:row['semantic_text'] if row.get('dataset')=='bone' else original_text(row)
        install_v10(session)
    install_restoration(session,save_proof)
    native_model=b.run_model
    def carrier_model(value):
        native_model(value)
        for name in ('complete_metrics.json','all_metrics_live.json','validation_only_metrics.json','exact_prompt_audit.json'):
            f=run/name
            if f.exists():b.write(f,tag_carrier(b.read(f),expected))
    b.run_model=carrier_model
    g.amos.run(job)
    atomic_new(run/'worker_complete.json',dict(classes=len(expected),memory_calls=session.memory_calls,cache_stats=dict(raw=session.raw_memo.stats,image=session.feature_memo.stats),carrier_only=True))


def make_attempt(out,job,gpu,available,completed):
    completed=class_records(out,job['control'])
    pending=[x for x in job['pending'] if (job['model'],p.key(x['task'])) not in completed]
    if not pending:return None
    run=out/'runs'/(job['name']+'_a'+str(job['attempt']));spec=out/'job_specs'/(job['name']+'_a'+str(job['attempt'])+'.json');log=out/'logs'/(job['name']+'_a'+str(job['attempt'])+'.log')
    if run.exists() or spec.exists() or log.exists():raise ValueError('attempt paths already exist')
    value=dict(job,pending=pending,task_count=len(pending),gpu=gpu,gpu_uuid=available[gpu]['uuid'],run=str(run),spec=str(spec),log=str(log))
    atomic_new(spec,value);log.parent.mkdir(parents=True,exist_ok=True)
    return value


def check_controls(out):
    records=class_records(out,True);reuse_audit=b.read(out/'reuse_audit.json');reuse=reuse_audit['completed']
    originals={(x['model'],x['floor'],p.key(x['row'])):x['row'] for x in reuse}
    for job in b.read(out/'jobs.json')['controls']:
        identity=(job['model'],p.key(job['pending'][0]['task']))
        if identity not in records:raise ValueError('fresh nine-floor control incomplete')
        record=records[identity]['record']
        reference={}
        for floor in p.FLOORS:
            name=job['model']+'_bone_floor'+format(floor,'.2f')+'_smoke'
            spec=b.read(reuse_audit['specs'][name])
            reference[floor]=dict(row=originals[(job['model'],floor,identity[1])],stages=b.read(Path(spec['run'])/'confirmation_audit.json')['stages'])
        compare_control(record,reference)
        if record['cache_stats']['raw']['hits']<8:raise ValueError('shared control did not reuse common raw stages')
    return records


def compare_control(record,original):
    fields=('key','floor','frames','input_size','method','context','memory','memory_group','threshold','source')
    for result in record['results']:
        reference=original[result['floor']]
        if any(abs(result['row'][name]-reference['row'][name])>1e-6 for name in p.METRICS):raise ValueError('shared-grid true control metrics differ')
        current=[{k:s[k] for k in fields if k in s} for s in result['stages']]
        expected=[{k:s[k] for k in fields if k in s} for s in reference['stages']]
        if current!=expected:raise ValueError('shared-grid true control expansion/calibration differs')


def schedule(out,control):
    jobs=b.read(out/'jobs.json')['controls' if control else 'cases'];pending=[dict(j) for j in jobs];active={};failed=[]
    while pending or active:
        committed=class_records(out,control)
        for gpu,(proc,job,handle) in list(active.items()):
            if proc.poll() is None:continue
            handle.close();del active[gpu]
            done=class_records(out,control);remaining=[x for x in job['pending'] if (job['model'],p.key(x['task'])) not in done]
            text=Path(job['log']).read_text(errors='replace')[-14000:];outcome=attempt_outcome(proc.returncode,text,bool(remaining))
            if outcome=='fatal':failed.append(dict(name=job['name'],log=job['log'],returncode=proc.returncode))
            elif outcome=='retry':
                next_job=retry(job,proc.returncode,text,time.time());pending.append(next_job)
                print('SHARED_RESOURCE_REQUEUE',job['name'],next_job['frame_batch'],next_job['minimum_free_mib'],flush=True)
            else:print('SHARED_CASE_DONE',job['name'],flush=True)
        committed=class_records(out,control);available=g.gpu_snapshot()
        for gpu in ALLOWED:
            if gpu in active or failed:continue
            selected=next((j for j in pending if eligible(j,gpu,available,time.time())),None)
            if selected is None:continue
            pending.remove(selected)
            if all((selected['model'],p.key(x['task'])) in committed for x in selected['pending']):continue
            job=make_attempt(out,selected,gpu,available,committed)
            if job is None:continue
            handle=Path(job['log']).open('x')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
            proc=subprocess.Popen([sys.executable,__file__,'worker','--job',job['spec']],cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
            active[gpu]=(proc,job,handle);print('SHARED_START',gpu,proc.pid,job['name'],job['task_count'],'batch',job['frame_batch'],flush=True)
        b.write(out/'status.json',dict(time=time.strftime('%F %T'),phase='controls' if control else 'full',pending=len(pending),running={str(gpu):dict(pid=proc.pid,spec=job['spec'],log=job['log'],name=job['name']) for gpu,(proc,job,h) in active.items()},failed=failed,committed_classes=len(committed)))
        if failed and not active:raise RuntimeError('shared-grid non-resource failure; retain all evidence')
        if pending or active:time.sleep(10)


def merge(out):
    require_merge_state(b.read(out/'status.json'),related())
    verify(out);g.dataset_signature(rehash=True);check_controls(out);all_rows={(x['model'],x['floor'],p.key(x['row'])):x['row'] for x in b.read(out/'reuse_audit.json')['completed']}
    for value in class_records(out).values():
        record=value['record']
        for result in record['results']:
            identity=(record['model'],result['floor'],p.key(result['row']))
            if identity in all_rows:raise ValueError('duplicate reused/new grid result')
            all_rows[identity]=result['row']
    tasks=b.read(out/'manifest.json')['tasks'];result={}
    for model in p.MODELS:
        parts={f:[all_rows[(model,f,p.key(t))] for t in tasks] for f in p.FLOORS}
        result[model]=dict(floors={format(f,'.2f'):p.summarize(rows,parts[.35],parts[.60]) for f,rows in parts.items()},winner=p.winner(parts))
    if len(all_rows)!=13860:raise ValueError('final unique row count mismatch')
    shards=[dict(j,tasks=b.read(Path(j['prepared'])/'manifest.json')['tasks']) for j in b.read(ROOT/OLD/'jobs.json')]
    if len(shards)!=162:raise ValueError('original shard count changed')
    partition=validate_original_partition(shards,all_rows)
    atomic_new(out/'merged_metrics.json',dict(models=result,rows=13860,predictions=13860,task_count_per_configuration=770,
        original_shard_counts=partition,scope='Bone20/47/770 development tuning; shared native stages',protocol_sha=sha(out/'protocol.json')))
    print('SHARED_GRID_MERGED_13860',flush=True)


def validate_original_partition(shards,rows):
    result={}
    for job in shards:
        try:selected=[rows[(job['model'],job['floor'],p.key(t))] for t in job['tasks']]
        except KeyError as error:raise ValueError('missing original shard result key') from error
        p.validate_rows(selected,job['tasks']);result[job['name']]=len(selected)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('wait-launch','worker','status','merge'));parser.add_argument('--job');args=parser.parse_args();out=ROOT/OUT
    if args.mode=='worker':run(b.read(args.job))
    elif args.mode=='wait-launch':
        if out.exists():raise FileExistsError('do not relaunch shared grid')
        handoff=finish_parent();prepare(out,handoff);schedule(out,True);check_controls(out);print('SHARED_TRUE_18_CONTROL_PASS',flush=True);schedule(out,False);merge(out)
    elif args.mode=='merge':merge(out)
    else:
        verify(out);records=class_records(out);print(json.dumps(dict(queue=b.read(out/'status.json'),reused=b.read(out/'reuse_audit.json')['reused_tasks'],new_committed=sum(len(v['record']['results']) for v in records.values())),indent=2))


if __name__=='__main__':main()
