"""Full original CT with SAT semantics and exactly zero spatial prompt groups.

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
from tools.amos_semantic_only_prompts import make_empty_bundle, validate_empty_bundle, full_ct_indices, semantic_modes

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('semantic_only',)
CONTROL_VARIANTS = ('baseline', 'semantic_only')
PROMPT_VARIANT = 'fg_scribble_bg_scribble'
NATIVE = 'tools/amos_prompt_ablation_runtime.py'
NATIVE_SHA256 = '5387c458a6331491fd258718b8a2f370a594ec73a38db546702a116900f0b18e'
OWNER = 'tools/amos_prompt_reuse_owner_runtime.py'
SIGNED = {
    NATIVE:NATIVE_SHA256,
    OWNER:'dcf768949161021469138e02b5ba2894b09e30f98ae59caf928d27ec4cb89c67',
    'tools/amos_prompt_reuse_runtime.py':'bd3f4c1274088edc8fea1cfc0c908d9a4e1b436471a85d2e2400be9f56d9725a',
    'tools/amos_prompt_ablation_restore_runtime.py':'ad1c19d82449c84186160b876adfba14ea40f0faf41fc491a5a2f637dce929e3',
    'tools/amos_semantic_only_prompts.py':'474ac73730734732a83c110cd8d5aa04f54ea8e80aaedafcf128495176bd6d09',
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
_NATIVE = _REUSE = None


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
                    replace(n,field,ast.Name(id='PROMPT_VARIANT',ctx=ast.Load()),index,'baseline_selectors')
                if isinstance(item,ast.ImportFrom) and item.module=='tools.amos_prompt_ablation_prompts':
                    names={a.name for a in item.names}
                    if names=={'VARIANTS'}:replace(n,field,ast.Pass(),index,'variant_imports')
                    elif names=={'variant_modes'}:replace(n,field,ast.Pass(),index,'mode_imports')
                    elif names=={'variant_modes','roi_indices'}:
                        replace(item,'names',[a for a in item.names if a.name!='variant_modes'],label='validator_mode_imports')
                if isinstance(item,ast.Import) and len(item.names)==1:
                    name=item.names[0].name
                    if name in ('v10_sam2_memory_roi','v10.infer.infer_amos_sam2_promptgen_v10'):
                        target='memory' if name=='v10_sam2_memory_roi' else 'scorer'
                        replace(n,field,ast.parse(target+' = _private_module("'+target+'")').body[0],index,'private_'+target)
        if isinstance(n,ast.keyword) and n.arg=='background_mode' and isinstance(n.value,ast.Constant) and n.value.value=='point':
            replace(n,'value',ast.Constant(value='scribble'),label='background_default')
        if isinstance(n,ast.FunctionDef) and n.name=='actual_roi':
            replace(n,'body',[ast.parse("if session.variant == 'semantic_only':\n    return full_ct_roi(session, self, target)").body[0],*n.body],label='full_ct_roi')
        if isinstance(n,ast.FunctionDef) and n.name=='modes':
            replace(n,'body',[ast.parse("if self.variant == 'semantic_only':\n    return semantic_modes()").body[0],*n.body],label='semantic_modes')
        if isinstance(n,ast.FunctionDef) and n.name=='converted':
            replace(n,'body',[ast.parse("if self.variant == 'semantic_only':\n    return empty_converted(self, inner, index, class_id)").body[0],*n.body],label='empty_converted')
        if isinstance(n,ast.Tuple) and len(n.elts)==4 and ast.unparse(n)=="(modes['foreground_mode'], modes['background_mode'], 0 if modes['foreground_mode'] == 'scribble' else 2, True)":
            replace(n,'elts',[*n.elts[:-1],ast.parse("session.variant != 'semantic_only'",mode='eval').body],label='mixed_background')
        if isinstance(n,ast.Tuple) and len(n.elts)==3 and ast.unparse(n)=="(session.modes()['foreground_mode'], session.modes()['background_mode'], True)":
            replace(n,'elts',[*n.elts[:-1],ast.parse("session.variant != 'semantic_only'",mode='eval').body],label='pool_background')
    expected=dict(baseline_selectors=3,variant_imports=2,mode_imports=1,validator_mode_imports=1,
        private_memory=1,private_scorer=2,background_default=1,full_ct_roi=1,semantic_modes=1,
        empty_converted=1,mixed_background=1,pool_background=1)
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
    module=types.ModuleType('_amos_semantic_native_private');ns=module.__dict__;ns['__file__']=str(ROOT/NATIVE)
    exec(compile(tree,str(ROOT/NATIVE),'exec'),ns)
    from tools.amos_prompt_ablation_prompts import variant_modes
    ns.update(PROMPT_VARIANT=PROMPT_VARIANT,VARIANTS=CONTROL_VARIANTS,variant_modes=variant_modes,
        semantic_modes=semantic_modes,full_ct_roi=full_ct_roi,empty_converted=empty_converted,
        TRANSFORM_PROOF=proof,_MODULE=module)
    ns['_baseline_consumption']=ns['validate_consumption'];ns['_baseline_types']=ns['validate_output_types']
    ns['_baseline_validate_record']=ns['validate_record']
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


def _ct(session,inner,index):
    image=inner.image_paths[int(index)]
    volume=inner._get_or_create_case_image(image)
    if np.ndim(volume)!=3:raise ValueError('actual canonical CT must be3D')
    shape=tuple(int(n) for n in volume.shape)
    if min(shape)<=0:raise ValueError('empty actual CT')
    return shape


def full_ct_roi(session,inner,_unused_target):
    index=int(getattr(inner,'_v10_active_idx',0));shape=_ct(session,inner,index)
    indices=full_ct_indices(shape)
    session.roi_calls.append(dict(source='actual_readonly_case_CT_shape',depth=shape[0],indices=indices.tolist(),canonical_shape_dhw=list(shape)))
    return indices


def empty_converted(session,inner,index,class_id):
    shape=_ct(session,inner,index);size=int(inner.prompt_plane_size)
    bundle=make_empty_bundle(shape,size);validate_empty_bundle(bundle,shape,size)
    session.expected_groups=bundle['groups'];session.expected_slice_coords=bundle['slice_coords']
    session.prompt_audit=dict(semantic_modes(),canonical_shape_dhw=list(shape),groups_shape=[0,3,size,size],
        groups_sha256=hashlib.sha256(bundle['groups'].tobytes()).hexdigest(),planes={},raw_foreground_z_support=[],
        raw_background_points_per_plane={},geometry_source='actual_readonly_case_CT_shape',prompt_cache_read=False,
        selected_plane_read=False,gt_geometry_read=False,full_xy=True)
    return bundle


def install_zero_groups(session,prompt):
    native=prompt._mixed_prompt_groups
    def mixed(incoming):
        if session.variant!='semantic_only':return native(incoming)
        shape=tuple(incoming.shape)
        if len(shape)!=4 or shape[0]!=0 or shape[1]!=3:raise ValueError('semantic mixed groups must be exactly N0')
        return incoming,'scribble','scribble',0,False
    prompt._mixed_prompt_groups=mixed


def validate_consumption(groups,plane_ids,foreground_mode_id,modes,expected,*,slice_coords=None,expected_slice_coords=None):
    ns=native_engine()
    if modes.get('spatial_prompt_policy')!='none':
        return ns['_baseline_consumption'](groups,plane_ids,foreground_mode_id,modes,expected,
            slice_coords=slice_coords,expected_slice_coords=expected_slice_coords)
    array=ns['_array'];g=array(groups);planes=array(plane_ids)
    if (g.ndim!=4 or g.shape[0:2]!=(0,3) or planes.shape!=(0,) or ns['_scalar'](foreground_mode_id)!=0
            or g.shape!=array(expected).shape or not np.array_equal(g,array(expected))
            or slice_coords is None or expected_slice_coords is None or array(slice_coords).shape!=(0,)
            or array(expected_slice_coords).shape!=(0,)):
        raise ValueError('semantic consumption contains phantom spatial input')
    return dict(plane_ids=[],foreground_mode_id=0,groups_sha256=ns['_digest'](groups),
        raster_sha256=hashlib.sha256(g.tobytes()).hexdigest(),channel_nonzero=[],slice_coords=[])


def validate_output_types(output,modes,*,metadata=True):
    ns=native_engine()
    if modes.get('spatial_prompt_policy')!='none':return ns['_baseline_types'](output,modes,metadata=metadata)
    roles=ns['_array'](output['prompt_3d_role_ids']).reshape(-1);types_=ns['_array'](output['prompt_3d_type_ids']).reshape(-1)
    if not len(roles) or roles.shape!=types_.shape or set(roles.tolist())!={0} or set(types_.tolist())!={0}:
        raise ValueError('semantic role/type IDs must be exactly{0}')
    if metadata and (output.get('foreground_prompt_mode'),output.get('background_prompt_mode'),output.get('foreground_channel'),output.get('background_active'))!=('scribble','scribble',0,False):
        raise ValueError('semantic output has active background/spatial mode')
    return dict(type_ids=[0],role_ids=[0],type_counts={'0':len(types_)})


def postprocess_dispatch(variant,verified,logits,foreground_seed,background_seed,*,native,threshold,probability=None,**kwargs):
    if variant!='semantic_only':return native(logits,foreground_seed,background_seed,**kwargs)
    if not verified or np.any(foreground_seed) or np.any(background_seed):
        raise ValueError('semantic component bypass requires verified empty FG/BG seeds')
    expected=dict(fallback=.60,minimum=.70,maximum=.75,min_logit_separation=.50)
    if kwargs!=expected:raise ValueError('semantic threshold settings changed')
    audit=threshold(logits,foreground_seed,background_seed,**kwargs)
    if audit.get('threshold')!=.60 or audit.get('source')!='fallback_missing_prompt_logits':
        raise ValueError('native missing-prompt fallback must be0.60')
    if probability is None:
        import torch
        prediction=(torch.sigmoid(logits.detach().float())>=float(audit['threshold'])).squeeze(1).cpu().numpy().astype(bool)
    else:prediction=np.asarray(probability(logits)>=audit['threshold']).squeeze(1).astype(bool)
    audit=dict(audit,semantic_component_bypass=True,foreground_seed_count=0,background_seed_count=0,
        raw_threshold_mask_retained=True,threshold_mask_sha256=hashlib.sha256(prediction.tobytes()).hexdigest(),
        threshold_mask_foreground_voxels=int(np.count_nonzero(prediction)))
    return prediction,audit


def bind_native_scoring(session):
    # The untouched native main prefix imports this alias again. Rebind only
    # after that prefix has completed, before any task is allowed to execute.
    scorer=native_engine()['_private_module']('scorer')
    session.namespace['restore_and_score_full_volume']=scorer.restore_and_score_full_volume
    if scorer.restore_and_score_full_volume.__globals__ is not scorer.__dict__:
        raise ValueError('actual scorer does not use private observed globals')


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
        if session.variant=='semantic_only':
            aligned=array(result['aligned_prompt'])
            if np.any(aligned):raise ValueError('semantic native aligned prompt must remain exact zero')
            dense=_float_array(result['dense_prompt_embeddings']);sparse=_float_array(result['sparse_prompt_embeddings'])
            if not np.isfinite(dense).all() or not np.isfinite(sparse).all():raise ValueError('nonfinite actual semantic dense/sparse embeddings')
            event=session.active_stage['prompt_calls'][-1]
            event.update(frame_coords=array(k['frame_coords']).reshape(-1).astype(float).tolist(),
                aligned_prompt_exact_zero=True,dense_shape=list(dense.shape),sparse_shape=list(sparse.shape),
                dense_sha256=digest(dense),sparse_sha256=digest(sparse))
        return result
    def pool(features,*a,**k):
        if session.variant=='semantic_only' and np.any(array(features)):raise ValueError('native zero prompt weighted pool changed')
        result=native_pool(features,*a,**k)
        if session.variant=='semantic_only':session.active_stage['global_pool_calls'][-1].update(pooled_features_exact_zero=True,pooled_shape=list(features.shape))
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


def _install_semantic_post(session):
    scorer=native_engine()['_private_module']('scorer');native=scorer.postprocess_v10_logits_from_prompts
    # Install inside the existing stage observer so it sees the actual audit.
    raw=scorer.__dict__.get('_semantic_original_post',None)
    if raw is not None:raise ValueError('semantic postprocess already installed')
    scorer._semantic_original_post=native
    def dispatch(*a,**k):
        return postprocess_dispatch(session.variant,session.variant=='semantic_only' and session.semantic_verified,*a,
            native=native,threshold=scorer.dynamic_mask_threshold_from_prompt_logits,**k)
    scorer.postprocess_v10_logits_from_prompts=dispatch


def reuse_engine():
    global _REUSE
    source_guard()
    if _REUSE is None:
        owner=types.ModuleType('_amos_semantic_owner_private');owner.__file__=str(ROOT/OWNER)
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
        ns['_restore']=shim;_REUSE=ns
    return _REUSE


def initialize(job,paths):
    source_binding=source_guard(production=True)
    native=native_engine();original=native['_install_execution_hooks']
    def hooks(session):
        session.semantic_verified=False
        bind_native_scoring(session)
        install_zero_groups(session,session.namespace['prompt'])
        _install_semantic_post(session)
        return original(session)
    native['_install_execution_hooks']=hooks
    try:session=reuse_engine()['initialize'](job,paths)
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
    session.input_audit['semantic_only_protocol']=dict(runtime_sha256=sha(__file__),helper_sha256=SIGNED['tools/amos_semantic_only_prompts.py'],
        dependency_source_binding=source_binding,
        native_transform=native['TRANSFORM_PROOF'],decode_transform=session.namespace['_semantic_decode_binding'],
        spatial_prompt_policy='none',geometry_source='actual_readonly_case_CT_shape',full_xy=True,
        component_policy='retain_threshold_mask_only_when_verified_semantic_and_both_seeds_empty',
        original_preprocessing_order=True,zero_weighted_pool_unchanged=True,zero_strength_anchor_unchanged=True)
    return session


def evaluate(session,task,variant):
    if variant not in (*CONTROL_VARIANTS,'native_control'):raise ValueError('unknown semantic protocol variant')
    source_guard();session.semantic_calls.clear();session.semantic_embedding_events.clear()
    before=method_guard(session)
    if before!=session.semantic_method_guard:raise ValueError('semantic model method binding changed before evaluation')
    session.semantic_verified=variant=='semantic_only'
    # Baseline remains the native BGscribble path; only external labels differ.
    try:record=reuse_engine()['evaluate'](session,task,variant)
    finally:
        if method_guard(session)!=before:raise ValueError('semantic model method binding changed during evaluation')
    record['semantic_execution']=dict(calls=dict(session.semantic_calls),runtime_sha256=sha(__file__),
        policy='none' if variant=='semantic_only' else PROMPT_VARIANT,verified_semantic=variant=='semantic_only',
        method_guard_before=before,method_guard_after=method_guard(session),methods_unchanged=True,
        embeddings=copy.deepcopy(session.semantic_embedding_events))
    validate_record(record,task,variant,session.batch,bool(session.job.get('control') or variant=='native_control'))
    return record


def _baseline_record(record,variant):
    result=copy.deepcopy(record);label='native_control' if variant=='native_control' else PROMPT_VARIANT
    result['variant']=label
    for e in result['metric_execution']['events']:e['variant']=label
    return result,label


def validate_native_record(record,task,variant,batch,control=False):
    """Never relax the original positive-prompt baseline validator."""
    if variant!='semantic_only':
        value,label=_baseline_record(record,variant)
        return native_engine()['_baseline_validate_record'](value,task,label,batch,control)
    return validate_semantic_record(record,task,batch,control)


def validate_semantic_record(record,task,batch,control=False):
    ns=native_engine();row=record['row'];prompt=record['prompt'];roi=record['roi'];stages=record['stages']
    if record.get('variant')!='semantic_only' or ns['_key'](row)!=ns['_key'](task) or record.get('actual_frame_batch')!=batch or batch not in (1,2,4):
        raise ValueError('semantic task/variant/batch mismatch')
    if row.get('metric_space')!='raw_nifti_full_volume' or row.get('local_patch_bounds') is not None or row.get('class_name')!=task['class_name']:
        raise ValueError('semantic native full-volume labels/XY changed')
    if 'local_class_id' in task and row.get('label_id',task['local_class_id'])!=task['local_class_id']:
        raise ValueError('semantic original local label changed')
    for name in ns['METRICS']:
        value=float(row[name])
        if not math.isfinite(value) or value<0 or (name!='pred_gt_ratio' and value>1):raise ValueError('invalid semantic metric')
    r=record['restoration']
    if tuple(r.get(k) for k in ('model','epoch','weight_mode','prompt_parameters_exact','decoder_parameters_exact','guards_passed','model_load_count'))!=('v10_2',490,'live',418,212,True,1):
        raise ValueError('semantic checkpoint restoration changed')
    if (r.get('full_tensor_comparison_phase')!='post_load' or r.get('per_evaluation_guard')!='parameter_version_pointer_shape_dtype_device'
            or record.get('parameter_guard_verified') is not True or not record.get('parameter_guard_before')
            or record['parameter_guard_before']!=record.get('parameter_guard_after')):raise ValueError('semantic parameter guard missing')
    shape=prompt['canonical_shape_dhw'];indices=full_ct_indices(shape).tolist();coords=(2*np.arange(shape[0],dtype=np.float32)/max(1,shape[0]-1)-1).tolist()
    if not prompt.get('group_builds') or not roi.get('builds') or not stages:raise ValueError('semantic actual execution missing')
    group_shape=prompt.get('groups_shape',[])
    if (len(group_shape)!=4 or group_shape[:2]!=[0,3] or type(group_shape[2]) is not int
            or group_shape[2]<=0 or group_shape[2]!=group_shape[3]):raise ValueError('semantic audited group layout is not N0')
    for build in prompt['group_builds']:
        if build.get('plane_ids')!=[] or build.get('groups_sha256')!=stages[0]['bundle'].get('groups_sha256'):
            raise ValueError('semantic empty group build differs from actual consumption')
    for k,v in semantic_modes().items():
        if prompt.get(k)!=(list(v) if isinstance(v,tuple) else v):raise ValueError('semantic spatial mode changed: '+k)
    if (prompt.get('planes')!={} or prompt.get('raw_foreground_z_support')!=[] or prompt.get('raw_background_points_per_plane')!={}
            or prompt.get('geometry_source')!='actual_readonly_case_CT_shape' or prompt.get('full_xy') is not True
            or any(prompt.get(k) is not False for k in ('prompt_cache_read','selected_plane_read','gt_geometry_read'))):raise ValueError('semantic phantom geometry')
    if any(roi.get(k)!=indices for k in ('initial','reserved','final')):raise ValueError('semantic must cover every original CT frame')
    for build in roi['builds']:
        if build.get('source')!='actual_readonly_case_CT_shape' or build.get('indices')!=indices or build.get('canonical_shape_dhw')!=shape:raise ValueError('semantic CT geometry proof differs')
    calls=0
    for stage in stages:
        if stage.get('frames')!=shape[0] or stage.get('frame_coords')!=coords or stage.get('settings')!={**ns['FROZEN_SETTINGS'],'sam2_frame_batch_size':batch}:
            raise ValueError('semantic frames/coordinates/settings changed')
        if stage.get('memory_calls',{}).get('memory_encoder')!=shape[0] or (shape[0]>1 and stage['memory_calls'].get('memory_attention',0)<=0):raise ValueError('semantic historical memory coverage missing')
        for e in [stage['bundle'],*stage['prompt_calls']]:
            if (e.get('plane_ids')!=[] or e.get('slice_coords')!=[] or e.get('foreground_mode_id')!=0 or e.get('channel_nonzero')!=[]
                    or e.get('raster_sha256')!=prompt.get('groups_sha256') or e.get('groups_sha256')!=stage['bundle']['groups_sha256']):raise ValueError('semantic actual empty consumption differs')
        if not stage['prompt_calls'] or len(stage['mixed_calls'])!=len(stage['prompt_calls']) or len(stage['global_pool_calls'])!=1 or len(stage['decoder_calls'])!=1 or len(stage['adapter_calls'])!=shape[0]:raise ValueError('semantic actual module coverage missing')
        covered=set()
        for event in stage['prompt_calls']:
            chunk=event.get('frame_coords',[])
            if not chunk or chunk[0] not in coords:raise ValueError('semantic actual chunk coordinates missing')
            start=coords.index(chunk[0])
            if chunk!=coords[start:start+len(chunk)] or len(chunk)!=event.get('frames'):raise ValueError('semantic actual chunk coordinates differ')
            covered.update(range(start,start+len(chunk)))
            if (event.get('aligned_prompt_exact_zero') is not True or event.get('dense_shape',[None])[0]!=len(chunk)
                    or event.get('sparse_shape',[None])[0]!=len(chunk) or len(event.get('dense_sha256',''))!=64
                    or len(event.get('sparse_sha256',''))!=64):raise ValueError('actual semantic dense/perframe embeddings missing')
        if covered!=set(indices) or stage['global_pool_calls'][0].get('pooled_features_exact_zero') is not True:
            raise ValueError('native semantic chunk coverage/zero weighted pool changed')
        if stage['decoder_calls'][0].get('anchor')!=0:raise ValueError('native zero-strength anchor changed')
        for e in [*stage['prompt_calls'],*stage['global_pool_calls'],*stage['decoder_calls'],*stage['adapter_calls']]:
            if e.get('role_ids')!=[0] or e.get('type_ids')!=[0]:raise ValueError('semantic actual roles/types changed')
        if any(e.get('semantic_text')!=record.get('semantic_text') for e in stage['prompt_calls']):raise ValueError('semantic AMOS text changed')
        audit=stage['threshold_audit']
        if audit.get('threshold')!=.6 or audit.get('source')!='fallback_missing_prompt_logits':raise ValueError('semantic fallback changed')
        if (audit.get('semantic_component_bypass') is not True or audit.get('foreground_seed_count')!=0
                or audit.get('background_seed_count')!=0 or audit.get('raw_threshold_mask_retained') is not True):
            raise ValueError('semantic verified empty-seed component bypass proof missing')
        calls+=len(stage['prompt_calls'])
    if calls!=prompt.get('actual_consumption_calls') or row.get('mask_threshold_used')!=.6 or row.get('dynamic_threshold_source')!='fallback_missing_prompt_logits':raise ValueError('semantic final audit differs')
    for name in ('native_source_sha256','task_body_ast_sha256'):
        if len(record.get(name,''))!=64:raise ValueError('semantic source binding missing')
    binding=record['resource_binding']
    if binding.get('gpu') not in (2,4,5,6) or binding.get('minimum_free_mib',0)<30720 or binding.get('free_mib_before_load',0)<binding['minimum_free_mib']:raise ValueError('semantic GPU proof missing')
    execution=record['metric_execution'];events=execution.get('events',[]);tracker=execution.get('tracker',{})
    if execution.get('provider_sha256')!=sha(ROOT/'tools/bone_fast_metrics.py') or execution.get('cache_scope')!='class_local_metadata_only' or tracker.get('cache_tensor_bytes')!=0 or tracker.get('comparison_failures')!=0 or len(events)!=1:raise ValueError('semantic native metric execution missing')
    e=events[0]
    if e.get('key')!=list(ns['_key'](task)) or e.get('variant')!='semantic_only' or len(e.get('input_sha256',''))!=64 or e.get('full_mask_shape')!=shape or any(abs(float(e['metrics'][m])-float(row[m]))>1e-12 for m in ns['METRICS']):raise ValueError('semantic actual metric output differs')
    if control and (e.get('native_compared') is not True or e.get('max_abs_error') is None or float(e['max_abs_error'])>1e-6):raise ValueError('semantic real native metric twin missing')
    return True


def validate_record(record,task,variant,batch,control=False):
    source_guard();validate_native_record(record,task,variant,batch,control)
    reuse=reuse_engine();reuse['validate_record'](record,task,variant,batch,control)
    proof=record.get('semantic_execution',{})
    if (proof.get('runtime_sha256')!=sha(__file__) or proof.get('verified_semantic')!=(variant=='semantic_only')
            or any(proof.get('calls',{}).get(k,0)<=0 for k in ('semantic','sat','residual'))
            or proof.get('methods_unchanged') is not True or len(proof.get('method_guard_before',''))!=64
            or proof['method_guard_before']!=proof.get('method_guard_after')):raise ValueError('actual SAT/residual/method guard proof missing')
    embeddings=proof.get('embeddings',[])
    if len(embeddings)!=proof['calls']['semantic'] or any(e.get('shape')!=[1,128] or e.get('finite') is not True
            or len(e.get('sha256',''))!=64 or e.get('semantic_text')!=record.get('semantic_text') for e in embeddings):raise ValueError('actual finite SAT128 metadata proof missing')
    return True


def validate_control(native,records):
    if len(records)!=2 or {r['variant'] for r in records}!=set(CONTROL_VARIANTS):raise ValueError('control requires baseline+semantic_only')
    baseline=next(r for r in records if r['variant']=='baseline')
    if native['actual_frame_batch']!=4 or any(r['actual_frame_batch']!=4 for r in records):raise ValueError('controls require actual batch4')
    metrics=native_engine()['METRICS'];error=max(abs(float(native['row'][m])-float(baseline['row'][m])) for m in metrics)
    geometry=lambda r:{k:r['roi'][k] for k in ('initial','reserved','final','history')}
    if error>1e-6 or geometry(native)!=geometry(baseline) or native['stages']!=baseline['stages']:raise ValueError('native BGscribble replay differs')
    for k in ('mask_threshold_used','dynamic_threshold_source'):
        if native['row'][k]!=baseline['row'][k]:raise ValueError('native baseline postprocess differs')
    reuse=reuse_engine()
    for r in [native,*records]:reuse['validate_reuse_proof'](r['reuse_execution'],r['variant'],True)
    if native['reuse_execution']['io_chain_sha256']!=baseline['reuse_execution']['io_chain_sha256']:raise ValueError('native baseline encoder inputs differ')
    return dict(passed=True,max_metric_error=error,roi_equal=True,stages_equal=True,actual_frame_batch=4,
        variants=list(CONTROL_VARIANTS),native_control_excluded=True,exact_encoder_reuse_passed=True)


def engine():return reuse_engine()
