"""AMOS full-val E120 EMA evaluation: no historical memory, frozen exact prompts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ct13_full_validation_eval as base

ROOT = base.ROOT
WEIGHT = 'v10/output/v10_2_ct13_no_memory_e403_converge_20261019/20261019_212018/epoch120.pth'
WEIGHT_SHA = 'ead73a9c3542107949d3e270d58ebdfb90e6f2af0aef4e645a6c9a27281e7a66'
OUT = 'output/amos_no_memory_e120_full1485_20261021'
PREVIOUS = 'output/ct13_full_val_v92e360_v102e490_20261005'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def require_no_memory(args):
    if args.get('v10_memory_enabled') is not False:
        raise ValueError('checkpoint/runtime must explicitly disable memory')


def ema_bindings(shadow, params):
    pairs = []
    if not shadow: raise ValueError('empty EMA')
    for key in sorted(shadow):
        if 'memory' in key or key not in params or tuple(shadow[key].shape) != tuple(params[key].shape):
            raise ValueError('invalid/unmapped EMA tensor: '+key)
        pairs.append((key, params[key], shadow[key]))
    return pairs


def make_parts(rows, count):
    return [base.case_shard(rows, count, i) for i in range(count)]


def frozen_protocol(out):
    return dict(checkpoint=WEIGHT, checkpoint_sha=sha(ROOT/WEIGHT), epoch=120,
        weight_mode='EMA', memory_enabled=False, metrics=list(base.METRICS),
        code={p:sha(ROOT/p) for p in ('tools/amos_no_memory_eval.py',
            'tools/ct13_full_validation_eval.py','v10/v10_2_no_memory_train.py',
            'v10/v10_sam2_memory_roi.py','v10/infer/infer_eight_source_adaptive_v10_2.py','utils/metrics.py')},
        split_sha=sha(ROOT/base.SPLIT), manifest_sha=sha(out/'manifest.json'),
        prompt_refs_sha=sha(out/'refs.json'), cache='shared CT and prompts read-only',
        foreground='adaptive coronal+sagittal scribble',background='2 points per plane',
        threshold=[.35,.75,.60],semantics='AMOS legacy')


def verify_protocol(out):
    if frozen_protocol(out) != base.read(out/'protocol.json'):
        raise ValueError('code/checkpoint/protocol changed; new output required')


def prepare(out):
    if out.exists(): raise ValueError('output already exists; use status/launch, not prepare')
    if sha(ROOT/WEIGHT) != WEIGHT_SHA: raise ValueError('checkpoint SHA mismatch')
    payload = base.build_manifest(base.read(ROOT/base.SPLIT))
    rows = sorted([r for r in payload['tasks'] if r['source_name']=='amos'], key=base.key)
    if len(rows)!=1485 or len({r['case_id'] for r in rows})!=100:
        raise ValueError('expected 1485 tasks / 100 held-out CT cases')
    refs = {}
    for p in sorted((ROOT/PREVIOUS/'prepared').glob('amos_s*of12/refs.json')):
        for r in base.read(p)['per_case_class']:
            k=base.key(r)
            if k in refs: raise ValueError('duplicate historical prompt')
            r=dict(r);r['prompt_sha256']=sha(ROOT/base.CACHE/'metadata/shared_adaptive_prompt_v1/amos'/r['prompt_cache_name'])
            refs[k]=r
    if set(refs)!={base.key(r) for r in rows}: raise ValueError('historical prompt coverage mismatch')
    base.write(out/'manifest.json',base.manifest_payload(rows))
    base.write(out/'refs.json',{'per_case_class':[refs[base.key(r)] for r in rows]})
    first=rows[0]['case_id'];smoke=[r for r in rows if r['case_id']==first]
    parts=[smoke]+make_parts([r for r in rows if r['case_id']!=first],24);jobs=[]
    for i,part in enumerate(parts):
        if not part: continue
        name='amos_smoke' if i==0 else 'amos_s%02d'%i
        prepared=out/'prepared'/name
        base.write(prepared/'manifest.json',base.manifest_payload(part))
        base.write(prepared/'refs.json',{'per_case_class':[refs[base.key(r)] for r in part]})
        jobs.append(dict(name=name,model='v10_2',source='amos',prepared=str(prepared),
            task_count=len(part),frame_batch=48,attempt=0))
    base.write(out/'jobs.json',jobs);base.write(out/'protocol.json',frozen_protocol(out))
    print('PREPARED 100 CT / 1485 tasks / %d whole-case jobs'%len(jobs),flush=True)


def run(job):
    import torch
    out=Path(job['run']).parents[1];verify_protocol(out)
    os.environ['CUDA_VISIBLE_DEVICES']=str(job['gpu'])
    uuid='GPU-'+str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')
    if uuid!=job['gpu_uuid'] or torch.ones(1,device='cuda:0').item()!=1:
        raise ValueError('GPU physical UUID/CUDA mapping mismatch')
    sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'v10'))
    import v10.finetune_totalseg_amos_magic_v10_2_joint as entry
    native_make=entry.make_prompt_v10_2;native_load=entry.load_init_v10_2
    context={};audit={}
    def forbidden(*a,**kw): raise RuntimeError('NO_MEMORY invariant: historical memory invoked')
    def make(args,adapter,device):
        require_no_memory(vars(args));context.update(adapter=adapter,args=args)
        return native_make(args,adapter,device)
    def load(path,prompt,decoder):
        state=native_load(path,prompt,decoder);require_no_memory(state['args'])
        if state['epoch']!=120: raise ValueError('wrong checkpoint epoch')
        shadow=state['ema_state']['shadow']
        params={'prompt.'+k:v for k,v in prompt.named_parameters()}
        params.update({'sam.sam_mask_decoder.'+k:v for k,v in decoder.named_parameters()})
        pairs=ema_bindings(shadow,params)
        if len(pairs)!=233 or state['ema_state']['num_updates']!=24000:
            raise ValueError('unexpected EMA membership/updates')
        with torch.no_grad():
            for name,target,value in pairs:
                if not torch.isfinite(value).all(): raise ValueError('nonfinite EMA: '+name)
                target.copy_(value.to(target))
                if not torch.equal(target.detach().cpu(),value.to(dtype=target.dtype)):
                    raise ValueError('EMA restoration mismatch: '+name)
        sam=context['adapter'].sam
        sam.memory_encoder.forward=forbidden;sam.memory_attention.forward=forbidden
        decoder.prompt_memory_adapter.forward=forbidden
        import v10_sam2_memory_roi as memory
        memory._compact_memory=forbidden
        age=int(state['args'].get('routing_age_offset',0))+int(state['epoch'])
        prompt.set_morphology_routing_epoch(age)
        audit.update(memory_enabled=False,ema_tensors=len(pairs),ema_updates=24000,
            routing_epoch=age,gpu_uuid=uuid,checkpoint_sha=WEIGHT_SHA,
            runtime_context={k:getattr(context['args'],k,None) for k in (
                'roi_image_size','cache_image_size','prompt_full_depth_limit',
                'prompt_depth_chunk_size','prompt_depth_halo','v10_memory_enabled')})
        base.write(Path(job['run'])/'no_memory_runtime_audit.json',audit)
        print('NO_MEMORY EMA_RESTORED',json.dumps(audit),flush=True)
        return state
    entry.make_prompt_v10_2=make;entry.load_init_v10_2=load
    base.WEIGHTS['v10_2']=WEIGHT;base.run_model(job)
    p=Path(job['run']);ea=base.read(p/'exact_prompt_audit.json')
    count=job['task_count']
    if ea.get('loaded_task_count')!=count or ea.get('dynamic_threshold_summary',{}).get('count')!=count:
        raise ValueError('exact-prompt/dynamic threshold coverage mismatch')
    audit.update(completed_tasks=count,memory_guards_passed=True)
    base.write(p/'no_memory_runtime_audit.json',audit)


def merge(out):
    verify_protocol(out);rows=[]
    for j in base.read(out/'jobs.json'):
        p=out/'jobs'/j['name'];a=base.read(p/'no_memory_runtime_audit.json')
        if not a.get('memory_guards_passed') or a['memory_enabled'] or a['ema_tensors']!=233:
            raise ValueError('invalid memory/EMA audit')
        rows.extend(base.read(p/'complete_metrics.json')['per_case_class'])
    base.validate_rows(rows,base.read(out/'manifest.json')['tasks'])
    result=dict(tasks=len(rows),cases=len({r['case_id'] for r in rows}),epoch=120,
        memory_enabled=False,weight_mode='EMA',
        mean={m:sum(float(r[m]) for r in rows)/len(rows) for m in base.METRICS},per_case_class=rows)
    base.write(out/'merged_metrics.json',result);print(json.dumps(result['mean']),flush=True)


def launch(out,gpus):
    import fcntl
    verify_protocol(out)
    # The reused launch lock also covers the smoke phase in this wrapper.
    lock=(out/'no_memory_schedule.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for cmd in Path('/proc').glob('[0-9]*/cmdline'):
        try: args=cmd.read_bytes().decode(errors='replace').split('\0')
        except OSError: continue
        if 'worker' in args and '--job' in args and any(str(out/'job_specs') in a for a in args):
            raise ValueError('existing worker alive; refuse duplicate launch')
    first=base.read(out/'jobs.json')[0]
    if not (out/'jobs'/first['name']/'complete_metrics.json').exists():
        query=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
        inventory={int(r.split(',')[0]):(r.split(',')[1].strip(),int(r.split(',')[2])) for r in query.splitlines()}
        candidates=[g for g in gpus if inventory[g][1]>=45000]
        if not candidates: raise ValueError('smoke requires at least45GB free')
        gpu=candidates[0];first.update(gpu=gpu,gpu_uuid=inventory[gpu][0],run=str(out/'jobs'/first['name']))
        spec=out/'job_specs'/first['name'];base.write(spec,first)
        log=out/'smoke.log'
        with log.open('a') as f:
            subprocess.run([sys.executable,__file__,'worker','--job',str(spec)],cwd=ROOT,
                stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),check=True)
        if not base.read(Path(first['run'])/'no_memory_runtime_audit.json').get('memory_guards_passed'):
            raise ValueError('smoke failed audit')
        print('SMOKE PASSED; releasing full queue',flush=True)
    base.__file__=__file__;base.merge=merge;base.launch(out,gpus)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('prepare','worker','launch','status','merge'))
    p.add_argument('--out',default=OUT);p.add_argument('--job');p.add_argument('--gpus',default='0,1,2,3,4,5,6,7')
    a=p.parse_args();os.chdir(ROOT);out=(ROOT/a.out).resolve()
    if a.mode=='prepare': prepare(out)
    elif a.mode=='worker': run(base.read(a.job))
    elif a.mode=='launch': launch(out,[int(v) for v in a.gpus.split(',')])
    elif a.mode=='merge': merge(out)
    else:
        done=sum((out/'jobs'/j['name']/'complete_metrics.json').exists() for j in base.read(out/'jobs.json'))
        print(json.dumps(dict(completed_jobs=done,total_jobs=len(base.read(out/'jobs.json')),
            scheduler=base.read(out/'status.json') if (out/'status.json').exists() else 'smoke/prelaunch'),indent=2))


if __name__=='__main__': main()
