"""Native prompt-derived ROI, SAT semantics, and original FG scribbles only.

The signed native task body, learned forwards, preprocessing and metrics remain
unchanged. Audited private bindings isolate this protocol from prompt baselines.
CPU boundary tests are not a substitute for the independent real GPU gate.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import json
import math
from pathlib import Path
import types

import numpy as np
from tools.amos_semantic_fg_scribble_prompts import TARGET, variant_modes, build_variant

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('semantic_fg_scribble',)
CONTROL_VARIANTS = ('baseline', 'semantic_fg_scribble')
PROMPT_VARIANT = 'fg_scribble_bg_scribble'
NATIVE = 'tools/amos_prompt_ablation_runtime.py'
NATIVE_SHA256 = '5387c458a6331491fd258718b8a2f370a594ec73a38db546702a116900f0b18e'
OWNER = 'tools/amos_prompt_reuse_owner_runtime.py'
SIGNED = {
    NATIVE:NATIVE_SHA256,
    OWNER:'dcf768949161021469138e02b5ba2894b09e30f98ae59caf928d27ec4cb89c67',
    'tools/amos_prompt_reuse_runtime.py':'bd3f4c1274088edc8fea1cfc0c908d9a4e1b436471a85d2e2400be9f56d9725a',
    'tools/amos_prompt_ablation_restore_runtime.py':'ad1c19d82449c84186160b876adfba14ea40f0faf41fc491a5a2f637dce929e3',
    'tools/amos_prompt_ablation_prompts.py':'39e2fe5e40dc3f7316ffbee58eb8d5691987fe44b2d48c7d6902af58296df433',
}
# These are complete reviewed source profiles, never independently selectable
# hashes. The local checkout supports metadata/CPU tests only. Production must
# use the unchanged sources that produced the authenticated selected S21 run.
NATIVE_PROGRAM='v10/infer/infer_eight_source_adaptive_v10_2.py'
MEMORY_SOURCE='v10/v10_sam2_memory_roi.py'
SCORER_SOURCE='v10/infer/infer_amos_sam2_promptgen_v10.py'
LOCAL_DEPENDENCIES={
    NATIVE_PROGRAM:'12ad8e8c0d58c8996da56ecf95da0d3ab219000c27e1539c28fb093fafae3407',
    MEMORY_SOURCE:'2402a8addf00423275f0d8538ac51fd560eda7dff6664369f384b477cb331a41',
    SCORER_SOURCE:'5e1092c49c62710aaf60fbf075460e9aba8e0fdfc1b16b94a30653bc68c8a761',
}
S21_DEPENDENCIES={
    NATIVE_PROGRAM:'31bb0252748a3962d420621b53f754c09f123d633cb5a96ad7bcf257502f393b',
    MEMORY_SOURCE:'138dc1651ace2eef934e2613cfdf6e9115ef2737f9c082a8253e904895ffefab',
    SCORER_SOURCE:LOCAL_DEPENDENCIES[SCORER_SOURCE],
}
S21_PROTOCOL='output/amos_v102_e490_floor070_prompt_ablations_20261031_reusefast_ownerfix/protocol.json'
S21_PROTOCOL_SHA256='0df2ca0724a5c8a86e2d28264875f1556f84f2cdaded53d41d5847055d78541a'
CONTROL_EVENT_LIMIT=4096
PRODUCTION_EVENT_LIMIT=64
_NATIVE=None
_REUSE={}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def classify_dependencies(hashes):
    if hashes==LOCAL_DEPENDENCIES:return 'local_metadata_only'
    if hashes==S21_DEPENDENCIES:return 'production_s21'
    raise ValueError('unknown or mixed semantic dependency source profile')


def dependency_profile():
    hashes={name:sha(ROOT/name) for name in LOCAL_DEPENDENCIES}
    return classify_dependencies(hashes),hashes


def validate_s21_protocol(protocol,signature,protocol_sha256):
    if protocol_sha256!=S21_PROTOCOL_SHA256 or signature!={'sha256':S21_PROTOCOL_SHA256}:
        raise ValueError('authenticated S21 protocol SHA/signature changed')
    sources=protocol.get('sources',{})
    if (any(sources.get(name)!=expected for name,expected in S21_DEPENDENCIES.items())
            or protocol.get('native_source_sha256')!=S21_DEPENDENCIES[NATIVE_PROGRAM]):
        raise ValueError('authenticated S21 native/scorer/memory source binding changed')
    return True


def _read_s21_protocol():
    path=ROOT/S21_PROTOCOL;payload=path.read_bytes()
    signature=json.loads(path.with_name('protocol_signature.json').read_text(encoding='utf-8'))
    return json.loads(payload),signature,hashlib.sha256(payload).hexdigest()


def source_guard(*,production=False):
    for name, expected in SIGNED.items():
        if sha(ROOT/name)!=expected:raise ValueError('signed semantic source SHA changed: '+name)
    profile,dependencies=dependency_profile()
    proof=dict(profile=profile,dependencies=dependencies,production=bool(production))
    if production:
        if profile!='production_s21':raise ValueError('production requires the authenticated S21 dependency profile; local metadata profile cannot run inference')
        validate_s21_protocol(*_read_s21_protocol())
        proof['selected_s21_protocol_sha256']=S21_PROTOCOL_SHA256
    return proof


def _ast(node):return ast.dump(node,include_attributes=False)


def transform_native(source):
    """Guard every edit by source SHA, exact anchor counts, and AST reversal."""
    if hashlib.sha256(source).hexdigest()!=NATIVE_SHA256:raise ValueError('native runtime SHA mismatch')
    original=ast.parse(source);tree=copy.deepcopy(original);undo=[];counts={}
    def replace(parent,field,value,index=None,label=''):
        old=getattr(parent,field) if index is None else getattr(parent,field)[index]
        undo.append((parent,field,index,old));counts[label]=counts.get(label,0)+1
        if index is None:setattr(parent,field,value)
        else:getattr(parent,field)[index]=value
    for n in list(ast.walk(tree)):
        for field,value in list(ast.iter_fields(n)):
            for index,item in list(enumerate(value)) if isinstance(value,list) else [(None,value)]:
                if isinstance(item,ast.IfExp) and ast.unparse(item) in (
                    "'baseline' if self.variant == 'native_control' else self.variant",
                    "'baseline' if variant == 'native_control' else variant"):
                    variable='self.variant' if 'self.variant' in ast.unparse(item) else 'variant'
                    replace(n,field,ast.parse(f"PROMPT_VARIANT if {variable} in ('baseline', 'native_control') else {variable}",mode='eval').body,index,'baseline_selectors')
                if isinstance(item,ast.ImportFrom) and item.module=='tools.amos_prompt_ablation_prompts':
                    names={a.name for a in item.names}
                    if names=={'VARIANTS'}:replace(n,field,ast.Pass(),index,'variant_imports')
                    elif names=={'variant_modes'}:replace(n,field,ast.Pass(),index,'mode_imports')
                    elif names=={'variant_modes','roi_indices'}:
                        replace(item,'names',[a for a in item.names if a.name!='variant_modes'],label='validator_mode_imports')
                    elif names=={'build_variant'}:
                        replace(item,'module','tools.amos_semantic_fg_scribble_prompts',label='builder_import')
                if isinstance(item,ast.Import) and len(item.names)==1:
                    name=item.names[0].name
                    if name in ('v10_sam2_memory_roi','v10.infer.infer_amos_sam2_promptgen_v10'):
                        target='memory' if name=='v10_sam2_memory_roi' else 'scorer'
                        replace(n,field,ast.parse(target+' = _private_module("'+target+'")').body[0],index,'private_'+target)
        if isinstance(n,ast.keyword) and n.arg=='background_mode' and isinstance(n.value,ast.Constant) and n.value.value=='point':
            replace(n,'value',ast.Constant(value='scribble'),label='background_default')
        if isinstance(n,ast.Call) and ast.unparse(n)=="sorted([modes['foreground_type_id'], modes['background_type_id']])":
            replace(n,'func',ast.Name(id='expected_types',ctx=ast.Load()),label='record_types_func')
            replace(n,'args',[ast.Name(id='modes',ctx=ast.Load())],label='record_types_args')
        if isinstance(n,ast.Compare) and ast.unparse(n)=="event['role_ids'] != [0, 1]":
            replace(n,'comparators',[ast.parse('expected_roles(modes)',mode='eval').body],label='record_roles')
        if isinstance(n,ast.Tuple) and len(n.elts)==4 and ast.unparse(n)=="(modes['foreground_mode'], modes['background_mode'], 0 if modes['foreground_mode'] == 'scribble' else 2, True)":
            replace(n,'elts',[*n.elts[:-1],ast.parse("session.variant != 'semantic_fg_scribble'",mode='eval').body],label='mixed_background')
        if isinstance(n,ast.Tuple) and len(n.elts)==3 and ast.unparse(n)=="(session.modes()['foreground_mode'], session.modes()['background_mode'], True)":
            replace(n,'elts',[*n.elts[:-1],ast.parse("session.variant != 'semantic_fg_scribble'",mode='eval').body],label='pool_background')
    expected=dict(baseline_selectors=3,variant_imports=2,mode_imports=1,validator_mode_imports=1,
        private_memory=1,private_scorer=2,background_default=1,builder_import=1,
        mixed_background=1,pool_background=1,record_types_func=1,record_types_args=1,record_roles=1)
    if counts!=expected:raise ValueError('native semantic anchors changed: '+repr(counts))
    transformed=copy.deepcopy(tree)
    for p,f,i,v in reversed(undo):
        if i is None:setattr(p,f,v)
        else:getattr(p,f)[i]=v
    if _ast(tree)!=_ast(original):raise ValueError('unexpected native semantic AST mutation')
    return ast.fix_missing_locations(transformed),dict(reverse_ast_equal=True,counts=counts,
        source_sha256=NATIVE_SHA256,original_ast_sha256=hashlib.sha256(_ast(original).encode()).hexdigest(),
        transformed_ast_sha256=hashlib.sha256(_ast(transformed).encode()).hexdigest())


def transform_decode(source,expected_sha256):
    if hashlib.sha256(source).hexdigest()!=expected_sha256:raise ValueError('native decode SHA mismatch')
    original=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_dynamic_adaptive_decode')
    fn=copy.deepcopy(original);undo=[];counts=dict(postprocess_imports=0,memory_imports=0)
    for parent in ast.walk(fn):
        for field,value in ast.iter_fields(parent):
            if not isinstance(value,list):continue
            for i,n in enumerate(value):
                if not isinstance(n,ast.ImportFrom):continue
                names={'postprocess_v10_logits_from_prompts'} if n.module=='v10.infer.infer_amos_sam2_promptgen_v10' else (
                    {'generate_case_3d_tokens','memory_sliding_window_decode'} if n.module=='v10_sam2_memory_roi' else set())
                found=[a for a in n.names if a.name in names]
                if not found:continue
                undo.append((value,i,copy.deepcopy(n)))
                counts['postprocess_imports' if 'postprocess_v10_logits_from_prompts' in names else 'memory_imports']+=len(found)
                kept=[a for a in n.names if a.name not in names]
                value[i]=ast.ImportFrom(module=n.module,names=kept,level=n.level) if kept else ast.Pass()
    if counts!=dict(postprocess_imports=1,memory_imports=2):raise ValueError('native local decode import anchors changed')
    transformed=copy.deepcopy(fn)
    for value,i,n in undo:value[i]=n
    if _ast(fn)!=_ast(original):raise ValueError('native decode reversal failed')
    return ast.fix_missing_locations(ast.Module(body=[transformed],type_ignores=[])),dict(counts,reverse_ast_equal=True,
        source_sha256=expected_sha256,original_function_ast_sha256=hashlib.sha256(_ast(original).encode()).hexdigest())


def _clone_module(module,name):
    """Clone function globals as well as attributes; no old module is patched."""
    private=types.ModuleType(name);ns=private.__dict__;ns.update(module.__dict__)
    for key,value in list(ns.items()):
        if isinstance(value,types.FunctionType) and value.__globals__ is module.__dict__:
            call=types.FunctionType(value.__code__,ns,value.__name__,value.__defaults__,value.__closure__)
            call.__kwdefaults__=value.__kwdefaults__;ns[key]=call
    return private


def native_engine(*,source=None):
    global _NATIVE
    source_guard()
    if source is None and _NATIVE is not None:return _NATIVE
    tree,proof=transform_native((ROOT/NATIVE).read_bytes() if source is None else source)
    module=types.ModuleType('_amos_fg_scribble_native_private');ns=module.__dict__;ns['__file__']=str(ROOT/NATIVE)
    exec(compile(tree,str(ROOT/NATIVE),'exec'),ns)
    ns.update(PROMPT_VARIANT=PROMPT_VARIANT,VARIANTS=(*CONTROL_VARIANTS,PROMPT_VARIANT),variant_modes=variant_modes,
        expected_roles=expected_roles,expected_types=expected_types,
        TRANSFORM_PROOF=proof,_MODULE=module)
    ns['_baseline_consumption']=ns['validate_consumption'];ns['_baseline_types']=ns['validate_output_types']
    ns['_native_validate_record']=ns['validate_record']
    ns['validate_consumption']=validate_consumption;ns['validate_output_types']=validate_output_types
    ns['validate_record']=validate_native_record
    private_modules={}
    def private_module(name):
        if name not in private_modules:
            source_name='v10_sam2_memory_roi' if name=='memory' else 'v10.infer.infer_amos_sam2_promptgen_v10'
            private_modules[name]=_clone_module(importlib.import_module(source_name),'_amos_semantic_'+name)
        return private_modules[name]
    ns['_private_module']=private_module
    original_namespace=ns['NativeProgram'].namespace
    def namespace(self):
        result=original_namespace(self)
        scorer=private_module('scorer');memory=private_module('memory')
        decode_tree,decode_proof=transform_decode(Path(self.filename).read_bytes(),self.source_sha256)
        result['restore_and_score_full_volume']=scorer.restore_and_score_full_volume
        # Late resolution observes the execution hooks installed after model load.
        result['postprocess_v10_logits_from_prompts']=lambda *a,**k:scorer.postprocess_v10_logits_from_prompts(*a,**k)
        result['generate_case_3d_tokens']=lambda *a,**k:memory.generate_case_3d_tokens(*a,**k)
        result['memory_sliding_window_decode']=lambda *a,**k:memory.memory_sliding_window_decode(*a,**k)
        exec(compile(decode_tree,self.filename,'exec'),result)
        result['_semantic_decode_binding']=decode_proof
        return result
    ns['NativeProgram'].namespace=namespace
    if source is None:_NATIVE=ns
    return ns




def bind_native_scoring(session):
    # The untouched native main prefix imports this alias again. Rebind only
    # after that prefix has completed, before any task is allowed to execute.
    scorer=native_engine()['_private_module']('scorer')
    session.namespace['restore_and_score_full_volume']=scorer.restore_and_score_full_volume
    if scorer.restore_and_score_full_volume.__globals__ is not scorer.__dict__:
        raise ValueError('actual scorer does not use private observed globals')


def expected_types(modes):
    return [0] if modes['background_mode']=='none' else sorted([modes['foreground_type_id'],modes['background_type_id']])


def expected_roles(modes):
    return [0] if modes['background_mode']=='none' else [0,1]


def validate_consumption(groups,plane_ids,foreground_mode_id,modes,expected,*,slice_coords=None,expected_slice_coords=None):
    ns=native_engine()
    if modes['background_mode']!='none':
        return ns['_baseline_consumption'](groups,plane_ids,foreground_mode_id,modes,expected,slice_coords=slice_coords,expected_slice_coords=expected_slice_coords)
    a=ns['_array'];g=a(groups);planes=a(plane_ids).reshape(-1)
    if (g.ndim!=4 or g.shape[:2]!=(2,3) or tuple(planes)!=(1,0) or ns['_scalar'](foreground_mode_id)!=0
            or g.shape!=a(expected).shape or not np.array_equal(g,a(expected)) or not np.isfinite(g).all()
            or slice_coords is None or expected_slice_coords is None or not np.array_equal(a(slice_coords),a(expected_slice_coords))):
        raise ValueError('target actual prompt raster/planes/mode/coordinates differ')
    counts=np.count_nonzero(g,axis=(2,3))
    if np.any(counts[:,0]<=0) or np.any(counts[:,1:]):raise ValueError('target foreground absent or BG/box leaked')
    return dict(plane_ids=[1,0],foreground_mode_id=0,groups_sha256=ns['_digest'](groups),
        raster_sha256=hashlib.sha256(g.tobytes()).hexdigest(),channel_nonzero=counts.tolist(),slice_coords=a(slice_coords).reshape(-1).astype(float).tolist())


def validate_output_types(output,modes,*,metadata=True):
    ns=native_engine()
    if modes['background_mode']!='none':return ns['_baseline_types'](output,modes,metadata=metadata)
    types_=ns['_array'](output['prompt_3d_type_ids']).reshape(-1);roles=ns['_array'](output['prompt_3d_role_ids']).reshape(-1)
    if not len(types_) or roles.shape!=types_.shape or set(types_.tolist())!={0} or set(roles.tolist())!={0}:
        raise ValueError('target adapter permits FG role0/type0 only')
    if metadata and (output.get('foreground_prompt_mode'),output.get('background_prompt_mode'),output.get('foreground_channel'),output.get('background_active'))!=('scribble','none',0,False):
        raise ValueError('target output mode or active BG leaked')
    return dict(type_ids=[0],role_ids=[0],type_counts={'0':len(types_)})


def validate_native_record(record,task,variant,batch,control=False):
    ns=native_engine();ns['_native_validate_record'](record,task,variant,batch,control)
    if variant!=TARGET:return True
    p=record['prompt']
    if p['background_mode_id']!=2 or p['background_active'] is not False:raise ValueError('target none conditioning changed')
    for plane,a in p['planes'].items():
        if a['raw_channel_nonzero'][0]<=0 or a['raw_channel_nonzero'][1:]!=[0,0] or a['raster_channel_nonzero'][0]<=0 or a['raster_channel_nonzero'][1:]!=[0,0]:
            raise ValueError('target raw/raster BG or box leakage')
        if a['raw_background_point_count']!=0:raise ValueError('target background point leaked')
    for stage in record['stages']:
        if stage.get('memory_calls',{}).get('memory_attention')!=max(0,int(stage['frames'])-1):raise ValueError('target native memory1 attention coverage missing')
        for event in stage['mixed_calls']:
            if (event.get('foreground'),event.get('background'),event.get('foreground_channel'))!=('scribble','none',0):raise ValueError('target actual mixed mode changed')
        for event in stage['global_pool_calls']:
            if event.get('background_active') is not False or event.get('background_mode')!='none':raise ValueError('target actual global pool BG changed')
        post=stage['threshold_audit'].get('foreground_only_native_postprocess',{})
        if (post.get('foreground_seed_count',0)<=0 or post.get('background_seed_count')!=0 or post.get('native_component_filter') is not True
                or post.get('native_scorer_sha256')!=S21_DEPENDENCIES[SCORER_SOURCE]):raise ValueError('target native foreground filter proof missing')
        for event in stage['prompt_calls']:
            counts=event.get('aligned_channel_nonzero',[])
            if len(counts)!=3 or counts[0]<0 or counts[1:]!=[0,0] or event.get('background_none_mode_id')!=2 or len(event.get('background_none_embedding_sha256',''))!=64:
                raise ValueError('target aligned FG or learned none embedding proof missing')
        if not any(e['aligned_channel_nonzero'][0]>0 for e in stage['prompt_calls']):raise ValueError('target stage contains no aligned foreground')
    return True


def install_postprocess_observer(session):
    scorer=native_engine()['_private_module']('scorer');native=scorer.postprocess_v10_logits_from_prompts
    def observed(logits,foreground_seed,background_seed,**kwargs):
        value=native(logits,foreground_seed,background_seed,**kwargs)
        if session.variant==TARGET:
            fg=int(np.count_nonzero(foreground_seed));bg=int(np.count_nonzero(background_seed))
            if fg<=0 or bg:raise ValueError('target actual native FG/BG seeds differ')
            value[1]['foreground_only_native_postprocess']=dict(foreground_seed_count=fg,background_seed_count=bg,
                native_component_filter=True,native_scorer_sha256=sha(ROOT/SCORER_SOURCE))
        return value
    scorer.postprocess_v10_logits_from_prompts=observed


def _float_array(value):
    return value.detach().float().cpu().numpy() if hasattr(value,'detach') else np.asarray(value,dtype=np.float32)


def embedding_metadata(value,text):
    array=_float_array(value)
    if not isinstance(text,str) or not text.strip() or array.shape!=(1,128) or not np.isfinite(array).all():
        raise ValueError('actual semantic embedding must be finite128 with nonempty legacy text')
    return dict(shape=[1,128],finite=True,sha256=native_engine()['_digest'](array),semantic_text=text)


def install_output_observers(session):
    prompt=session.namespace['prompt'];array=native_engine()['_array'];digest=native_engine()['_digest']
    native_forward=prompt.forward_case_3d_tokens;native_pool=prompt.build_case_3d_tokens_from_global_pool
    def forward(groups,video,*a,**k):
        result=native_forward(groups,video,*a,**k)
        if session.variant=='semantic_fg_scribble':
            aligned=array(result['aligned_prompt'])
            if np.any(aligned[:,1:]):raise ValueError('target aligned BG/box leaked')
            dense=_float_array(result['dense_prompt_embeddings']);sparse=_float_array(result['sparse_prompt_embeddings'])
            if not np.isfinite(dense).all() or not np.isfinite(sparse).all():raise ValueError('nonfinite actual semantic dense/sparse embeddings')
            event=session.active_stage['prompt_calls'][-1]
            event.update(frame_coords=array(k['frame_coords']).reshape(-1).astype(float).tolist(),
                aligned_channel_nonzero=[int(np.count_nonzero(aligned[:,i])) for i in range(aligned.shape[1])],
                background_none_mode_id=int(prompt.BACKGROUND_MODE_IDS['none']),
                background_none_embedding_sha256=digest(_float_array(prompt.background_mode_embedding.weight[2])),
                dense_shape=list(dense.shape),sparse_shape=list(sparse.shape),
                dense_sha256=digest(dense),sparse_sha256=digest(sparse))
        return result
    def pool(features,*a,**k):
        result=native_pool(features,*a,**k)
        if session.variant=='semantic_fg_scribble':session.active_stage['global_pool_calls'][-1].update(pooled_shape=list(features.shape),background_active=bool(k['background_active']),background_mode=k['background_mode'])
        return result
    prompt.forward_case_3d_tokens=forward;prompt.build_case_3d_tokens_from_global_pool=pool


def method_guard(session):
    prompt=session.namespace['prompt'];sam=session.namespace['adapter'].sam;decoder=sam.sam_mask_decoder
    targets=[(prompt,n) for n in ('forward_case_3d_tokens','_mixed_prompt_groups','build_case_3d_tokens_from_global_pool','_semantic_embedding')]
    targets.extend([(prompt.semantic_encoder,'forward'),(prompt.semantic_residual_mlp,'forward'),
        (decoder.prompt_3d_adapter,'forward'),(decoder.prompt_memory_adapter,'forward')])
    values=[]
    for obj,name in targets:
        call=getattr(obj,name);values.append((id(obj),name,id(getattr(call,'__func__',call)),id(getattr(call,'__self__',None))))
    return hashlib.sha256(repr(values).encode()).hexdigest()




def reuse_engine(control=False):
    """Independent owner namespaces: full bounded controls, native production.

    Only trace retention differs. Native encoder inputs, cache keys, owner
    guards, lossless trees and the 2 GiB feature budget remain unchanged.
    """
    if type(control) is not bool:raise ValueError('real boolean control provenance required')
    source_guard()
    if control not in _REUSE:
        owner=types.ModuleType('_amos_fg_scribble_owner_control' if control else '_amos_fg_scribble_owner_production');owner.__file__=str(ROOT/OWNER)
        exec(compile((ROOT/OWNER).read_bytes(),str(ROOT/OWNER),'exec'),owner.__dict__)
        ns=owner.engine();native=native_engine()
        restore=importlib.import_module('tools.amos_prompt_ablation_restore_runtime')
        shim=types.ModuleType('_amos_semantic_restore_private');shim.__dict__.update(restore.__dict__);shim._native=native['_MODULE']
        def initialize_prompt(job,paths):
            split=native['_resolve'](paths,job,'base_split_json','configs/ct13_v10_2_split_20260928.json')
            proof=restore._preload_ct13(json.loads(split.read_text(encoding='utf-8')))
            session=native['initialize'](job,paths);session.input_audit['ct13_import_order_restore']=proof
            return session
        shim.initialize=initialize_prompt;shim.evaluate=native['evaluate'];shim.validate_record=validate_native_record
        shim.validate_control=validate_control
        ns['_restore']=shim
        ns.update(EVENT_LIMIT=CONTROL_EVENT_LIMIT if control else PRODUCTION_EVENT_LIMIT,
            _TRACE_CONTROL=control,_TRACE_LIMIT=CONTROL_EVENT_LIMIT if control else PRODUCTION_EVENT_LIMIT,
            _native_trace_validator=ns['validate_reuse_proof'])
        # These methods resolve only this fresh owner's namespace. Production
        # keeps the signed class; controls stop before an unrecordable call.
        if control:
            exec('''
_NativeTraceEncoderReuse=EncoderReuse
class EncoderReuse(_NativeTraceEncoderReuse):
    def __call__(self,images):
        if self.control is not True:raise ValueError('control encoder requires real control provenance')
        if self.event_count>=_TRACE_LIMIT:raise ValueError('complete control encoder trace exhausted')
        return super().__call__(images)
''',ns)
        exec('''
def validate_reuse_proof(proof,variant,control=False):
    if type(control) is not bool or control is not _TRACE_CONTROL or proof.get('control') is not _TRACE_CONTROL:
        raise ValueError('encoder trace control provenance mismatch')
    if EVENT_LIMIT!=_TRACE_LIMIT or proof.get('event_limit')!=_TRACE_LIMIT:
        raise ValueError('encoder trace event limit differs from signed policy')
    if _TRACE_CONTROL and (type(proof.get('event_count')) is not int or not 0<proof['event_count']<=_TRACE_LIMIT
            or len(proof.get('events',[]))!=proof['event_count'] or proof.get('remaining_event_count')!=0):
        raise ValueError('complete bounded control encoder trace required; exhaustion or truncation')
    return _native_trace_validator(proof,variant,control)
''',ns)
        _REUSE[control]=ns
    return _REUSE[control]


def validate_reuse_proof(proof,variant,control=False):
    return reuse_engine(control=control)['validate_reuse_proof'](proof,variant,control)


def initialize(job,paths):
    source_binding=source_guard(production=True)
    control=job.get('control',False)
    if type(control) is not bool:raise ValueError('real boolean job control provenance required')
    native=native_engine();original=native['_install_execution_hooks']
    def hooks(session):
        session.semantic_verified=False
        bind_native_scoring(session)
        install_postprocess_observer(session)
        return original(session)
    native['_install_execution_hooks']=hooks
    try:session=reuse_engine(control=control)['initialize'](job,paths)
    finally:native['_install_execution_hooks']=original
    prompt=session.namespace['prompt']
    session.semantic_calls={}
    session.semantic_embedding_events=[]
    targets=[('semantic',prompt,'_semantic_embedding'),('sat',prompt.semantic_encoder,'forward'),
             ('residual',prompt.semantic_residual_mlp,'forward')]
    if prompt.semantic_residual_mlp is None:raise ValueError('learned semantic residual missing')
    session.semantic_methods=[]
    for name,obj,method in targets:
        call=getattr(obj,method);session.semantic_methods.append((name,obj,method,call))
        def observed(*a,_name=name,_call=call,**k):
            result=_call(*a,**k);session.semantic_calls[_name]=session.semantic_calls.get(_name,0)+1
            if _name=='semantic':
                text=k.get('semantic_text',a[0] if a else None)
                session.semantic_embedding_events.append(embedding_metadata(result,text))
            return result
        setattr(obj,method,observed)
    install_output_observers(session)
    session.semantic_method_guard=method_guard(session)
    native_finish=session.finish
    def finish():
        if method_guard(session)!=session.semantic_method_guard:raise ValueError('semantic model method binding changed before finish')
        proof=native_finish()
        proof['semantic_method_guard']=dict(before=session.semantic_method_guard,after=method_guard(session),unchanged=True)
        return proof
    session.finish=finish
    session.input_audit['semantic_fg_scribble_protocol']=dict(runtime_sha256=sha(__file__),helper_sha256=sha(ROOT/'tools/amos_semantic_fg_scribble_prompts.py'),
        dependency_source_binding=source_binding,
        native_transform=native['TRANSFORM_PROOF'],decode_transform=session.namespace['_semantic_decode_binding'],
        spatial_prompt_policy='native_cached_foreground_scribble_only',geometry_source='cached_original_foreground_support',full_xy=True,
        component_policy='native_foreground_connected_components',background_none_embedding_retained=True,
        original_preprocessing_order=True,native_threshold_calibration_unchanged=True,
        encoder_trace_policy=dict(control=control,event_limit=CONTROL_EVENT_LIMIT if control else PRODUCTION_EVENT_LIMIT,
            complete_trace_required=control))
    session.reuse_control=control
    return session


def evaluate(session,task,variant):
    if variant not in (*CONTROL_VARIANTS,'native_control'):raise ValueError('unknown semantic protocol variant')
    source_guard();session.semantic_calls.clear();session.semantic_embedding_events.clear()
    before=method_guard(session)
    if before!=session.semantic_method_guard:raise ValueError('semantic model method binding changed before evaluation')
    session.semantic_verified=variant=='semantic_fg_scribble'
    control=session.job.get('control',False)
    if type(control) is not bool or control is not session.reuse_control or (variant=='native_control' and not control):raise ValueError('session control provenance changed')
    # Baseline remains the native BGscribble path; only external labels differ.
    try:record=reuse_engine(control=control)['evaluate'](session,task,variant)
    finally:
        if method_guard(session)!=before:raise ValueError('semantic model method binding changed during evaluation')
    record['semantic_execution']=dict(calls=dict(session.semantic_calls),runtime_sha256=sha(__file__),
        policy='native_cached_foreground_scribble_only' if variant=='semantic_fg_scribble' else PROMPT_VARIANT,verified_semantic=variant=='semantic_fg_scribble',
        method_guard_before=before,method_guard_after=method_guard(session),methods_unchanged=True,
        embeddings=copy.deepcopy(session.semantic_embedding_events))
    validate_record(record,task,variant,session.batch,bool(session.job.get('control') or variant=='native_control'))
    return record




def validate_record(record,task,variant,batch,control=False):
    source_guard();validate_native_record(record,task,variant,batch,control)
    reuse=reuse_engine(control=control);reuse['validate_record'](record,task,variant,batch,control)
    expected_trace=dict(control=control,event_limit=CONTROL_EVENT_LIMIT if control else PRODUCTION_EVENT_LIMIT,complete_trace_required=control)
    if record.get('runtime_input_audit',{}).get('semantic_fg_scribble_protocol',{}).get('encoder_trace_policy')!=expected_trace:raise ValueError('runtime input encoder trace policy changed')
    proof=record.get('semantic_execution',{})
    if (proof.get('runtime_sha256')!=sha(__file__) or proof.get('verified_semantic')!=(variant=='semantic_fg_scribble')
            or any(proof.get('calls',{}).get(k,0)<=0 for k in ('semantic','sat','residual'))
            or proof.get('methods_unchanged') is not True or len(proof.get('method_guard_before',''))!=64
            or proof['method_guard_before']!=proof.get('method_guard_after')):raise ValueError('actual SAT/residual/method guard proof missing')
    embeddings=proof.get('embeddings',[])
    if len(embeddings)!=proof['calls']['semantic'] or any(e.get('shape')!=[1,128] or e.get('finite') is not True
            or len(e.get('sha256',''))!=64 or e.get('semantic_text')!=record.get('semantic_text') for e in embeddings):raise ValueError('actual finite SAT128 metadata proof missing')
    return True


def validate_control(native,records):
    if len(records)!=2 or {r['variant'] for r in records}!=set(CONTROL_VARIANTS):raise ValueError('controls require native BG, replay BG, and target FG-only')
    baseline=next(r for r in records if r['variant']=='baseline');target=next(r for r in records if r['variant']==TARGET)
    if native['actual_frame_batch']!=4 or any(r['actual_frame_batch']!=4 for r in records):raise ValueError('controls require actual batch4')
    metrics=native_engine()['METRICS'];error=max(abs(float(native['row'][m])-float(baseline['row'][m])) for m in metrics)
    geometry=lambda r:{k:r['roi'][k] for k in ('initial','reserved','final','history')}
    if error>1e-6 or geometry(native)!=geometry(baseline) or native['stages']!=baseline['stages']:raise ValueError('native BGscribble replay differs')
    for k in ('mask_threshold_used','dynamic_threshold_source'):
        if native['row'][k]!=baseline['row'][k]:raise ValueError('native baseline postprocess differs')
    for key in ('initial','reserved'):
        if target['roi'][key]!=baseline['roi'][key]:raise ValueError('target initial/reserved original FG ROI differs')
    if target['prompt']['raw_foreground_z_support']!=baseline['prompt']['raw_foreground_z_support']:raise ValueError('target original FG support differs')
    for plane in ('coronal','sagittal'):
        left=baseline['prompt']['planes'][plane];right=target['prompt']['planes'][plane]
        for key in ('slice_index','slice_coord'):
            if left[key]!=right[key]:raise ValueError('target selected plane coordinate differs')
        for key in ('raw_channel_sha256','raster_channel_sha256'):
            if left[key][0]!=right[key][0]:raise ValueError('target original FG bytes differ')
    reuse=reuse_engine(control=True)
    for r in [native,*records]:reuse['validate_reuse_proof'](r['reuse_execution'],r['variant'],True)
    if (native['reuse_execution']['io_chain_sha256']!=baseline['reuse_execution']['io_chain_sha256']
            or native['reuse_execution']['event_count']!=baseline['reuse_execution']['event_count']):raise ValueError('native baseline encoder inputs differ')
    if target['reuse_execution']['counters']['hits']<=0 or target['reuse_execution']['hit_twin_count']!=1:raise ValueError('target requires real encoder hit and native twin')
    return dict(passed=True,max_metric_error=error,roi_equal=True,stages_equal=True,actual_frame_batch=4,
        variants=list(CONTROL_VARIANTS),native_control_excluded=True,exact_encoder_reuse_passed=True,
        target_original_foreground_exact=True,target_initial_reserved_roi_equal=True,target_background_absent=True,
        target_final_autoexpand_native=True)


def engine(control=False):return reuse_engine(control=control)


