"""Fixed-checkpoint AMOS module sensitivity; not a retrained ablation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import MethodType

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT='output/amos_module_sensitivity_v92e360_v102e490_20261022'
HISTORY='output/ct13_full_val_v92e360_v102e490_20261005'


def variants(model):
    if model not in ('v9_2','v10_2'):raise ValueError('unknown model')
    return ('baseline','shared','no_3d_adapter','no_semantic') if model=='v9_2' else (
        'baseline','shared','no_3d_adapter','no_memory_adapter','no_both_adapters','no_semantic')


def apply_intervention(prompt,decoder,variant):
    import torch
    if variant not in variants('v10_2'):raise ValueError('unknown intervention')
    calls={};audit=dict(variant=variant,calls=calls,memory_disabled=False,
        adapter_policy='3D: parameter-free token broadcast; memory: identity on memory-fused features')
    def counted(name):calls[name]=calls.get(name,0)+1
    def forbidden(*a,**k):raise RuntimeError('removed module unexpectedly invoked')
    if variant=='shared':
        encoder=prompt.encoder3d
        def shared(self,inputs):
            counted('shared');return self.base_encoder(inputs)
        encoder.forward=MethodType(shared,encoder)
        encoder.router.forward=forbidden
        for expert in encoder.experts:expert.forward=forbidden
    if variant in ('no_3d_adapter','no_both_adapters'):
        def broadcast(self,tokens,frame_coords,sparse_prompt_embeddings=None,**kw):
            if tokens.ndim!=3 or tokens.shape[0]!=1:raise ValueError('one case token set required')
            counted('no_3d_adapter')
            result=tokens.expand(frame_coords.numel(),-1,-1).contiguous()
            if sparse_prompt_embeddings is not None:
                result=torch.cat((sparse_prompt_embeddings,result),dim=1)
            return result
        decoder.prompt_3d_adapter.forward=MethodType(broadcast,decoder.prompt_3d_adapter)
    if variant in ('no_memory_adapter','no_both_adapters'):
        if not hasattr(decoder,'prompt_memory_adapter'):raise ValueError('Memory Adapter not applicable')
        def identity(self,features,*a,**k):counted('no_memory_adapter');return features
        decoder.prompt_memory_adapter.forward=MethodType(identity,decoder.prompt_memory_adapter)
    if variant=='no_semantic':
        def zero(self,semantic_text,semantic_embedding,reference):
            counted('no_semantic');return reference.new_zeros((1,self.semantic_dim))
        prompt._semantic_embedding=MethodType(zero,prompt)
        if hasattr(prompt,'semantic_encoder'):prompt.semantic_encoder.forward=forbidden
    return audit


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def base_module():
    from tools import ct13_full_validation_eval as base
    return base


def signature(out):
    b=base_module()
    files=[Path(__file__),ROOT/'tools/ct13_full_validation_eval.py',ROOT/'utils/metrics.py']
    files+=sorted(ROOT.glob('*.py'))
    for folder in ('model','infer','v10','v10/model','v10/infer'):
        files+=sorted((ROOT/folder).glob('*.py'))
    refs=b.read(out/'refs.json')['per_case_class']
    return dict(code={str(p.relative_to(ROOT)):sha(p) for p in files},
        weights={m:dict(path=p,sha256=sha(ROOT/p)) for m,p in b.WEIGHTS.items()},
        split=sha(ROOT/b.SPLIT),manifest=sha(out/'manifest.json'),refs=sha(out/'refs.json'),
        prompts={r['prompt_cache_name']:sha(ROOT/b.CACHE/'metadata/shared_adaptive_prompt_v1/amos'/r['prompt_cache_name']) for r in refs},
        protocol='inference sensitivity; original fullmask10, dynamic.35-.75, legacy AMOS, readonly caches; live weights not EMA')


def verify(out):
    if signature(out)!=base_module().read(out/'protocol.json'):
        raise ValueError('changed code/weights/split/prompts; independent output required')


def prepare(out):
    b=base_module()
    if out.exists():raise FileExistsError(out)
    rows=sorted([r for r in b.read(ROOT/HISTORY/'manifest.json')['tasks'] if r['source_name']=='amos'],key=b.key)
    split=b.read(ROOT/b.SPLIT)['datasets']['amos']
    if (len(rows)!=1485 or len({r['case_id'] for r in rows})!=100
            or {r['case_id'] for r in rows}!={r['case_id'] for r in split['val']}
            or {r['case_id'] for r in rows}&{r['case_id'] for r in split['train']}):
        raise ValueError('requires all 100 AMOS val CT / 1485 tasks and zero train overlap')
    refs={}
    for p in sorted((ROOT/HISTORY/'prepared').glob('amos_s*of12/refs.json')):
        for row in b.read(p)['per_case_class']:
            if b.key(row) in refs:raise ValueError('duplicate prompt reference')
            refs[b.key(row)]=row
    if set(refs)!={b.key(r) for r in rows}:raise ValueError('prompt reference scope mismatch')
    b.write(out/'manifest.json',b.manifest_payload(rows))
    b.write(out/'refs.json',dict(per_case_class=[refs[b.key(r)] for r in rows]))
    jobs=[]
    for model in ('v9_2','v10_2'):
        for mode in variants(model):
            parts=[rows[:1]]+[b.case_shard(rows[1:],24,i) for i in range(24)]
            for i,part in enumerate(parts):
                if not part:continue
                name=model+'__'+mode+('__smoke' if i==0 else '__s%02d'%i)
                prepared=out/'prepared'/name
                b.write(prepared/'manifest.json',b.manifest_payload(part))
                b.write(prepared/'refs.json',dict(per_case_class=[refs[b.key(r)] for r in part]))
                jobs.append(dict(name=name,model=model,variant=mode,source='amos',prepared=str(prepared),
                    task_count=len(part),frame_batch=16,attempt=0,smoke=i==0))
    jobs.sort(key=lambda j:(not j['smoke'],j['name']))
    b.write(out/'jobs.json',jobs);b.write(out/'protocol.json',signature(out))
    b.write(out/'not_applicable.json',dict(v9_2=dict(no_memory_adapter='N/A: absent in original architecture',
        no_both_adapters='equivalent to no_3d_adapter; no duplicate prediction')))
    print('PREPARED',len(jobs),'jobs; 10 configurations x1485 tasks; exact readonly prompts',flush=True)


def compare_parameters(module,state):
    import torch
    count=0
    for name,p in module.named_parameters():
        if name not in state or not torch.equal(p.detach().cpu(),state[name].to(dtype=p.dtype)):
            raise ValueError('checkpoint parameter not restored exactly: '+name)
        count+=1
    if not count:raise ValueError('empty restoration audit')
    return count


def run(job):
    b=base_module();out=Path(job['run']).parents[1];verify(out)
    os.environ['CUDA_VISIBLE_DEVICES']=str(job['gpu'])
    import torch
    uuid='GPU-'+str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')
    if uuid!=job['gpu_uuid'] or torch.ones(1,device='cuda:0').item()!=1:raise ValueError('GPU UUID/CUDA mismatch')
    context={};model=job['model'];variant=job['variant']
    def loaded(prompt,decoder,state):
        if context:raise ValueError('unexpected repeated model load')
        context.update(prompt=prompt,decoder=decoder,state=state,
            intervention=apply_intervention(prompt,decoder,variant))
    if model=='v10_2':
        sys.path.insert(0,str(ROOT/'v10'))
        import v10.finetune_totalseg_amos_magic_v10_2_joint as entry
        native=entry.load_init_v10_2
        def load(path,prompt,decoder):
            state=native(path,prompt,decoder);loaded(prompt,decoder,state);return state
        entry.load_init_v10_2=load
    else:
        import finetune_multisource_sam2_v9_2_precision as multi
        import finetune_multisource_sam2_v9_2_precision_mixed_semantic_eval
        import finetune_totalseg_sam2_promptgen_v9_joint as joint
        original_init=joint.SAM2MedicalAdapterV9.__init__;holder={}
        def init(self,*a,**k):original_init(self,*a,**k);holder['decoder']=self.sam.sam_mask_decoder
        joint.SAM2MedicalAdapterV9.__init__=init
        native=multi.base.load_prompt_generator_checkpoint
        def load(path,prompt,*a,**k):
            result=native(path,prompt,*a,**k);loaded(prompt,holder['decoder'],result[0]);return result
        multi.base.load_prompt_generator_checkpoint=load
    b.run_model(job)
    if not context:raise ValueError('model load hook not called')
    prompt=context['prompt'];decoder=context['decoder'];state=context['state'];audit=context['intervention']
    if state['epoch']!=dict(v9_2=360,v10_2=490)[model]:raise ValueError('wrong checkpoint epoch')
    audit.update(prompt_parameters_exact=compare_parameters(prompt,state['prompt_generator']),
        decoder_parameters_exact=compare_parameters(decoder,state['mask_decoder' if model=='v10_2' else 'sam2_mask_decoder']),
        checkpoint_epoch=state['epoch'],gpu_uuid=uuid,weight_mode='live',variant=variant)
    required={'shared':['shared'],'no_3d_adapter':['no_3d_adapter'],
        'no_memory_adapter':['no_memory_adapter'],'no_both_adapters':['no_3d_adapter','no_memory_adapter'],
        'no_semantic':['no_semantic'],'baseline':[]}[variant]
    if any(audit['calls'].get(k,0)<=0 for k in required):raise ValueError('intervention not exercised')
    ea=b.read(Path(job['run'])/'exact_prompt_audit.json')
    if (ea['loaded_task_count']!=job['task_count']
            or ea['dynamic_threshold_summary']['count']!=job['task_count']):raise ValueError('prompt/threshold scope mismatch')
    audit.update(completed_tasks=job['task_count'],guards_passed=True)
    b.write(Path(job['run'])/'intervention_audit.json',audit)
    print('INTERVENTION_AUDIT_OK',model,variant,json.dumps(audit),flush=True)


def merge(out):
    b=base_module();verify(out);expected=b.read(out/'manifest.json')['tasks'];result={}
    for model in ('v9_2','v10_2'):
        result[model]={}
        for mode in variants(model):
            rows=[]
            for j in b.read(out/'jobs.json'):
                if (j['model'],j['variant'])!=(model,mode):continue
                run=out/'jobs'/j['name'];a=b.read(run/'intervention_audit.json')
                if not a['guards_passed'] or a['variant']!=mode:raise ValueError('invalid intervention audit')
                rows+=b.read(run/'complete_metrics.json')['per_case_class']
            b.validate_rows(rows,expected)
            result[model][mode]=dict(tasks=len(rows),mean={m:sum(r[m] for r in rows)/len(rows) for m in b.METRICS},per_case_class=rows)
        baseline=result[model]['baseline']['mean']
        for mode,v in result[model].items():v['delta_from_baseline']={m:v['mean'][m]-baseline[m] for m in b.METRICS}
    b.write(out/'merged_metrics.json',result);print('MERGED10 configurations /14850 tasks',flush=True)


def launch(out,gpus):
    import fcntl
    b=base_module();verify(out)
    lock=(out/'sensitivity_schedule.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:args=p.read_bytes().decode(errors='replace').split('\0')
        except OSError:continue
        if 'worker' in args and any(str(out/'job_specs') in x for x in args):raise ValueError('existing worker alive')
    jobs=b.read(out/'jobs.json')
    for j in jobs:
        run=out/'jobs'/j['name']
        if (run/'complete_metrics.json').exists() and not (run/'intervention_audit.json').exists():
            raise ValueError('unverified completed file retained; investigate before resume')
    native_read=b.read;b.__file__=__file__;b.merge=lambda _out:None
    try:
        for smoke in (True,False):
            selected=[j for j in jobs if j['smoke']==smoke]
            b.read=lambda p:native_read(p) if Path(p)!=out/'jobs.json' else selected
            b.write(out/'queue_phase.json',dict(phase='smoke' if smoke else 'full'))
            b.launch(out,gpus)
            for j in selected:
                if not native_read(out/'jobs'/j['name']/'intervention_audit.json')['guards_passed']:
                    raise ValueError('real-task smoke/worker audit failed')
            print('PHASE_PASSED','smoke' if smoke else 'full',flush=True)
    finally:b.read=native_read
    merge(out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','worker','launch','status','merge'])
    p.add_argument('--out',default=OUT);p.add_argument('--job');p.add_argument('--gpus',default='0,2,3,4,5,6')
    a=p.parse_args();os.chdir(ROOT);out=(ROOT/a.out).resolve();b=base_module()
    if a.mode=='prepare':prepare(out)
    elif a.mode=='worker':run(b.read(a.job))
    elif a.mode=='launch':launch(out,[int(x) for x in a.gpus.split(',')])
    elif a.mode=='merge':merge(out)
    else:
        jobs=b.read(out/'jobs.json');print(json.dumps(dict(total_jobs=len(jobs),
            audited_jobs=sum((out/'jobs'/j['name']/'intervention_audit.json').exists() for j in jobs),
            scheduler=b.read(out/'status.json') if (out/'status.json').exists() else None),indent=2))


if __name__=='__main__':main()
