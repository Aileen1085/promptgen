"""Fresh fixed-checkpoint modules, all consuming actual FG/BG scribbles.

Signed native task/ROI/metric code and original interventions are retained.
Only four native prompt-selection anchors differ in an isolated namespace.
Encoder reuse is the signed owner-bound, exact 2 GiB CPU implementation.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib
from pathlib import Path
import types

ROOT = Path(__file__).resolve().parents[1]
MODULE_VARIANTS = ('baseline','shared','no_3d_adapter','no_memory_adapter','no_both_adapters','no_semantic')
PROMPT_VARIANT = 'fg_scribble_bg_scribble'
INTERVENTION = 'tools/amos_module_sensitivity.py'
INTERVENTION_SHA256 = 'c45416c1e2edc267c8bb2e88c4ce6fd7fbc66561b0f3e044c586b5315290c7c7'
NATIVE = 'tools/amos_prompt_ablation_runtime.py'
NATIVE_SHA256 = '5387c458a6331491fd258718b8a2f370a594ec73a38db546702a116900f0b18e'
OWNER = 'tools/amos_prompt_reuse_owner_runtime.py'
OWNER_SHA256 = 'dcf768949161021469138e02b5ba2894b09e30f98ae59caf928d27ec4cb89c67'
SIGNED = {INTERVENTION: INTERVENTION_SHA256, NATIVE: NATIVE_SHA256, OWNER: OWNER_SHA256,
    'tools/amos_prompt_reuse_runtime.py':'bd3f4c1274088edc8fea1cfc0c908d9a4e1b436471a85d2e2400be9f56d9725a',
    'tools/amos_prompt_ablation_restore_runtime.py':'ad1c19d82449c84186160b876adfba14ea40f0faf41fc491a5a2f637dce929e3'}
TRANSFORM_COUNTS = {'prompt_selectors':3,'background_default':1}
_NATIVE = _REUSE = None


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_guard():
    for name, expected in {**SIGNED, INTERVENTION:INTERVENTION_SHA256}.items():
        if sha(ROOT/name) != expected:raise ValueError('signed module source SHA changed: '+name)


def transform_native(source):
    """Reverse every allowed edit and require exact recovery of the signed AST."""
    if hashlib.sha256(source).hexdigest()!=NATIVE_SHA256:raise ValueError('native runtime SHA mismatch')
    original=ast.parse(source);tree=copy.deepcopy(original);changes=[];counts=dict.fromkeys(TRANSFORM_COUNTS,0)
    for parent in ast.walk(tree):
        for field, value in ast.iter_fields(parent):
            candidates=list(enumerate(value)) if isinstance(value,list) else [(None,value)]
            for index,item in candidates:
                if not (isinstance(item,ast.IfExp) and ast.unparse(item) in (
                    "'baseline' if self.variant == 'native_control' else self.variant",
                    "'baseline' if variant == 'native_control' else variant")):continue
                changes.append((parent,field,index,item))
                replacement=ast.Name(id='PROMPT_VARIANT',ctx=ast.Load())
                if index is None:setattr(parent,field,replacement)
                else:value[index]=replacement
                counts['prompt_selectors']+=1
        if isinstance(parent,ast.keyword) and parent.arg=='background_mode' and isinstance(parent.value,ast.Constant) and parent.value.value=='point':
            changes.append((parent,'value',None,parent.value));parent.value=ast.Constant(value='scribble');counts['background_default']+=1
    if counts!=TRANSFORM_COUNTS:raise ValueError('native prompt selection anchors changed')
    transformed=copy.deepcopy(tree)
    for parent,field,index,value in reversed(changes):
        if index is None:setattr(parent,field,value)
        else:getattr(parent,field)[index]=value
    if ast.dump(tree,include_attributes=False)!=ast.dump(original,include_attributes=False):raise ValueError('unexpected native AST mutation')
    return ast.fix_missing_locations(transformed),dict(counts=counts,original_sha256=NATIVE_SHA256,
        original_ast_sha256=hashlib.sha256(ast.dump(original,include_attributes=False).encode()).hexdigest(),
        transformed_ast_sha256=hashlib.sha256(ast.dump(transformed,include_attributes=False).encode()).hexdigest(),reverse_ast_equal=True)


def native_engine(*, source=None):
    global _NATIVE
    source_guard()
    if source is not None or _NATIVE is None:
        tree,proof=transform_native((ROOT/NATIVE).read_bytes() if source is None else source)
        module=types.ModuleType('_amos_bg_scribble_native_private');ns=module.__dict__
        ns['__file__']=str(ROOT/NATIVE)
        exec(compile(tree,str(ROOT/NATIVE),'exec'),ns)
        ns.update(PROMPT_VARIANT=PROMPT_VARIANT,TRANSFORM_PROOF=proof,_MODULE=module)
        if source is None:_NATIVE=ns
        return ns
    return _NATIVE


def normalized_record(record):
    """Only label normalization; prompt/ROI/stages and all tensor proofs stay exact."""
    if record.get('prompt_variant')!=PROMPT_VARIANT:raise ValueError('constant BG-scribble prompt proof missing')
    result=copy.deepcopy(record)
    label='native_control' if record['variant']=='native_control' else PROMPT_VARIANT
    result['variant']=label
    for event in result['metric_execution']['events']:event['variant']=label
    return result


def reuse_engine():
    global _REUSE
    source_guard()
    if _REUSE is None:
        owner=types.ModuleType('_amos_bg_scribble_owner_private');owner.__file__=str(ROOT/OWNER)
        exec(compile((ROOT/OWNER).read_bytes(),str(ROOT/OWNER),'exec'),owner.__dict__)
        ns=owner.engine();native=native_engine()
        restore=importlib.import_module('tools.amos_prompt_ablation_restore_runtime')
        shim=types.ModuleType('_amos_bg_scribble_restore_private');shim.__dict__.update(restore.__dict__)
        # Keep the real dictionary backing Session and the native functions.
        shim._native=native['_MODULE']
        def initialize_prompt(job,paths):
            split_path=native['_resolve'](paths,job,'base_split_json','configs/ct13_v10_2_split_20260928.json')
            import json
            proof=restore._preload_ct13(json.loads(split_path.read_text(encoding='utf-8')))
            session=native['initialize'](job,paths);session.input_audit['ct13_import_order_restore']=proof
            return session
        def evaluate_prompt(session,task,variant):
            label='native_control' if variant=='native_control' else PROMPT_VARIANT
            record=native['evaluate'](session,task,label)
            record.update(variant=variant,module_variant=variant,prompt_variant=PROMPT_VARIANT)
            for event in record['metric_execution']['events']:event['variant']=variant
            return record
        def validate_prompt(record,task,variant,batch,control=False):
            label='native_control' if variant=='native_control' else PROMPT_VARIANT
            return native['validate_record'](normalized_record(record),task,label,batch,control)
        shim.initialize=initialize_prompt;shim.evaluate=evaluate_prompt;shim.validate_record=validate_prompt
        ns['_restore']=shim;_REUSE=ns
    return _REUSE


def method_targets(prompt,decoder):
    encoder=prompt.encoder3d
    values=[('encoder',encoder,'forward'),('base_encoder',encoder.base_encoder,'forward'),
            ('router',encoder.router,'forward'),('adapter3d',decoder.prompt_3d_adapter,'forward'),
            ('memory_adapter',decoder.prompt_memory_adapter,'forward'),('semantic',prompt,'_semantic_embedding')]
    values.extend(('expert_'+str(i),expert,'forward') for i,expert in enumerate(encoder.experts))
    if hasattr(prompt,'semantic_encoder'):values.append(('text_encoder',prompt.semantic_encoder,'forward'))
    if getattr(prompt,'semantic_residual_mlp',None) is not None:values.append(('semantic_residual',prompt.semantic_residual_mlp,'forward'))
    return values


def method_snapshot(prompt,decoder):
    return [(name,obj,method,method in vars(obj),vars(obj).get(method),getattr(obj,method))
            for name,obj,method in method_targets(prompt,decoder)]


def restore_methods(snapshot):
    for _name,obj,method,owned,raw,_call in snapshot:
        if owned:setattr(obj,method,raw)
        elif method in vars(obj):delattr(obj,method)


def method_digest(snapshot):
    return hashlib.sha256(repr([(name,id(obj),method,owned,id(raw) if owned else None,
        id(getattr(call,'__func__',call)),id(getattr(call,'__self__',None))) for name,obj,method,owned,raw,call in snapshot]).encode()).hexdigest()


def apply_intervention(prompt,decoder,variant):
    source_guard()
    return importlib.import_module('tools.amos_module_sensitivity').apply_intervention(prompt,decoder,variant)


def forbidden_names(variant, names):
    return {name for name in names if (variant=='shared' and (name=='router' or name.startswith('expert_')))
            or (variant=='no_semantic' and name in ('text_encoder','semantic_residual'))}


def required_names(variant):
    return {'encoder','adapter3d','memory_adapter','semantic'} | ({'base_encoder'} if variant=='shared' else set())


def observe_methods(prompt,decoder,variant):
    observed={};forbidden={};targets=method_targets(prompt,decoder)
    disabled=forbidden_names(variant,[name for name,_,_ in targets])
    for name,obj,method in targets:
        if name in disabled:forbidden[name]=0
        original=getattr(obj,method)
        def call(*args,_name=name,_original=original,**kwargs):
            observed[_name]=observed.get(_name,0)+1
            if _name in disabled:
                forbidden[_name]+=1
                raise RuntimeError('removed module unexpectedly invoked: '+_name)
            result=_original(*args,**kwargs)
            if variant=='no_semantic' and _name=='semantic':
                value=native_engine()['_array'](result)
                if tuple(value.shape)!=(1,128) or value.any():raise ValueError('noSemantic must return exact final128D zero')
            return result
        setattr(obj,method,call)
    return observed,forbidden


def initialize(job,paths):
    source_guard();session=reuse_engine()['initialize'](job,paths)
    ns=session.namespace;session.module_methods=method_snapshot(ns['prompt'],ns['adapter'].sam.sam_mask_decoder)
    session.input_audit['module_protocol']=dict(prompt_variant=PROMPT_VARIANT,module_variants=list(MODULE_VARIANTS),
        intervention_sha256=INTERVENTION_SHA256,runtime_sha256=sha(__file__),native_transform=native_engine()['TRANSFORM_PROOF'])
    return session


def evaluate(session,task,variant):
    source_guard()
    if variant not in (*MODULE_VARIANTS,'native_control'):raise ValueError('unknown module variant')
    ns=session.namespace;prompt=ns['prompt'];decoder=ns['adapter'].sam.sam_mask_decoder
    before=method_snapshot(prompt,decoder)
    if before!=session.module_methods:raise ValueError('module methods not restored before intervention')
    session.reuse_engine.check_guard();parameters=session.check_parameter_guard()
    # Full checkpoint equality was already established before this snapshot.
    if tuple(session.restoration.get(n) for n in ('model','epoch','weight_mode','prompt_parameters_exact','decoder_parameters_exact','guards_passed','model_load_count'))!=('v10_2',490,'live',418,212,True,1):
        raise ValueError('intervention before exact E490 restoration')
    try:
        audit=apply_intervention(prompt,decoder,'baseline' if variant=='native_control' else variant)
        observed,forbidden=observe_methods(prompt,decoder,variant)
        record=reuse_engine()['evaluate'](session,task,variant)
    finally:
        restore_methods(before)
        if method_snapshot(prompt,decoder)!=before:raise ValueError('exception-safe module method restoration failed')
        session.reuse_engine.check_guard();after=session.check_parameter_guard()
        if parameters!=after:raise ValueError('intervention changed tensor parameters')
    proof=dict(schema='amos_bg_scribble_module_v1',variant=variant,prompt_variant=PROMPT_VARIANT,
        intervention_sha256=INTERVENTION_SHA256,runtime_sha256=sha(__file__),original=audit,
        observed_calls=observed,forbidden_calls=forbidden,methods_restored=True,
        method_guard_before=method_digest(before),method_guard_after=method_digest(method_snapshot(prompt,decoder)),
        parameter_guard_before=parameters,parameter_guard_after=after,owner_encoder_guard_passed=True)
    record['module_execution']=proof
    validate_record(record,task,variant,session.batch,bool(session.job.get('control') or variant=='native_control'))
    return record


def validate_module_proof(proof,variant):
    if (proof.get('schema')!='amos_bg_scribble_module_v1' or proof.get('variant')!=variant
            or proof.get('prompt_variant')!=PROMPT_VARIANT or proof.get('intervention_sha256')!=INTERVENTION_SHA256
            or proof.get('runtime_sha256')!=sha(__file__) or proof.get('methods_restored') is not True
            or proof.get('owner_encoder_guard_passed') is not True
            or not proof.get('parameter_guard_before') or proof['parameter_guard_before']!=proof.get('parameter_guard_after')
            or len(proof.get('method_guard_before',''))!=64 or proof['method_guard_before']!=proof.get('method_guard_after')):
        raise ValueError('module execution/restoration/source proof missing')
    active='baseline' if variant=='native_control' else variant
    calls=proof.get('observed_calls',{})
    if any(type(calls.get(name)) is not int or calls[name]<=0 for name in required_names(active)):
        raise ValueError('required actual module calls absent')
    forbidden=proof.get('forbidden_calls',{})
    expected=forbidden_names(active,calls.keys()|forbidden.keys())
    if active=='shared' and ('router' not in forbidden or not any(k.startswith('expert_') for k in forbidden)):
        raise ValueError('shared forbidden router/expert guards absent')
    if active=='no_semantic' and 'text_encoder' not in forbidden:raise ValueError('semantic encoder forbidden guard absent')
    if set(forbidden)!=expected or any(type(v) is not int or v!=0 for v in forbidden.values()):raise ValueError('forbidden module executed')
    audit=proof.get('original',{})
    if audit.get('variant')!=active or audit.get('memory_disabled') is not False:raise ValueError('original intervention policy changed')
    required={'shared':['shared'],'no_3d_adapter':['no_3d_adapter'],'no_memory_adapter':['no_memory_adapter'],
        'no_both_adapters':['no_3d_adapter','no_memory_adapter'],'no_semantic':['no_semantic']}.get(active,[])
    if any(audit.get('calls',{}).get(name,0)<=0 for name in required):raise ValueError('original intervention did not execute')
    return True


def validate_record(record,task,variant,actual_frame_batch,control=False):
    source_guard()
    if record.get('module_variant')!=variant or record.get('variant')!=variant:raise ValueError('module label mismatch')
    normalized_record(record)
    reuse_engine()['validate_record'](record,task,variant,actual_frame_batch,control)
    validate_module_proof(record.get('module_execution',{}),variant)
    # Removing the learned memory adapter never removes historical SAM2 memory.
    for stage in record['stages']:
        if stage['memory_calls'].get('memory_encoder')!=stage['frames'] or (stage['frames']>1 and stage['memory_calls'].get('memory_attention',0)<=0):
            raise ValueError('historical SAM2 memory execution missing')
    return True


def validate_control(native_record,records):
    if len(records)!=6 or {r['variant'] for r in records}!=set(MODULE_VARIANTS):raise ValueError('six fresh module controls required')
    batch=native_record['actual_frame_batch'];task=native_record['row']
    validate_record(native_record,task,'native_control',batch,True)
    for record in records:validate_record(record,task,record['variant'],batch,True)
    baseline=next(r for r in records if r['variant']=='baseline')
    metrics=native_engine()['METRICS'];error=max(abs(float(native_record['row'][m])-float(baseline['row'][m])) for m in metrics)
    if error>1e-6:raise ValueError('fresh baseline differs from native BGscribble metrics')
    geometry=lambda r:{k:r['roi'][k] for k in ('initial','final','reserved','history')}
    if geometry(native_record)!=geometry(baseline) or native_record['stages']!=baseline['stages']:
        raise ValueError('native BGscribble baseline ROI/stages differ')
    for name in ('mask_threshold_used','dynamic_threshold_source'):
        if native_record['row'][name]!=baseline['row'][name]:raise ValueError('native BGscribble threshold differs')
    left,right=native_record['reuse_execution'],baseline['reuse_execution']
    if left['io_chain_sha256']!=right['io_chain_sha256'] or left['event_count']!=right['event_count']:
        raise ValueError('native-baseline full encoder IO chain differs')
    pairs=lambda p:[(e['input_sha256'],e['output_sha256']) for e in p['events']]
    if pairs(left)!=pairs(right):raise ValueError('native-baseline exact encoder trees differ')
    return dict(passed=True,variants=list(MODULE_VARIANTS),prompt_variant=PROMPT_VARIANT,actual_frame_batch=batch,
        native_control_excluded=True,max_metric_error=error,roi_equal=True,stages_equal=True,
        exact_encoder_reuse_passed=True,encoder_input_output_trees_equal=True,
        actual_native_encoder_twins=sum(r['reuse_execution']['counters']['native_twins'] for r in records),
        actual_baseline_encoder_hits=right['counters']['hits'],production_speedup_claimed=False)


def __getattr__(name):
    if name.startswith('__'):raise AttributeError(name)
    if name=='METRICS':return native_engine()['METRICS']
    raise AttributeError(name)
