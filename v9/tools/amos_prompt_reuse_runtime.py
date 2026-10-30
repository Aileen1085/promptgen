"""Additive, exact, CPU-RAM reuse over the immutable AMOS E490 runtime.

Only ``sam.forward_image`` and read-only file loads are memoized. PromptGen,
memory, ROI expansion, class extraction, and the ten native metrics still run.
No tensor is serialized. A worker owns one case and clears all RAM on finish.
"""
from __future__ import annotations

from collections import OrderedDict
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
from types import MethodType

import numpy as np
from tools import amos_prompt_ablation_restore_runtime as _restore

RESTORE_RUNTIME_SHA256 = "ad1c19d82449c84186160b876adfba14ea40f0faf41fc491a5a2f637dce929e3"
NATIVE_RUNTIME_SHA256 = "5387c458a6331491fd258718b8a2f370a594ec73a38db546702a116900f0b18e"
FEATURE_BUDGET_BYTES = 2 * 1024**3
EVENT_LIMIT = 64
METRICS = _restore.METRICS
Session = _restore.Session


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _is_tensor(value):
    torch = sys.modules.get("torch")
    return torch is not None and isinstance(value, torch.Tensor)


def _bytes(value):
    if _is_tensor(value):
        import torch
        value = value.detach().resolve_conj().resolve_neg().to(device="cpu").contiguous()
        return memoryview(value.reshape(-1).view(torch.uint8).numpy())
    return memoryview(np.ascontiguousarray(value)).cast("B")


def _cpu_copy(value):
    if _is_tensor(value):
        import torch
        return value.detach().resolve_conj().resolve_neg().to(device="cpu", copy=True, memory_format=torch.contiguous_format)
    if value.dtype.hasobject:
        raise TypeError("object array forbidden")
    return np.array(value, copy=True, order="C", subok=False)


def _spec(value):
    if _is_tensor(value):
        import torch
        if value.layout != torch.strided or value.is_quantized or value.device.type == "meta":
            raise TypeError("unsupported encoder tensor layout")
        return dict(kind="torch", shape=list(value.shape), dtype=str(value.dtype),
                    device=str(value.device), strides=list(value.stride()))
    return dict(kind="numpy", shape=list(value.shape), dtype=value.dtype.str,
                device="cpu", strides=list(value.strides))


def _equal(a, b):
    if _is_tensor(a) and _is_tensor(b):
        import torch
        return torch.equal(a, b)
    return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and np.array_equal(a, b)


class PackedTree:
    """Lossless tensor tree, preserving repeated tensor objects and layouts.

    ``vision_pos_enc`` batch samples are compressed only after exact equality
    of every sample. Equal CPU payloads are interned by BoundedMemo, including
    position templates across actual batches. Native tensor aliases survive
    each unpack; separate unpacks never share mutable payloads.
    """
    def __init__(self, graph, leaves):
        self.graph, self.leaves = graph, leaves
        self.digest = _sha(_json(dict(graph=graph, leaves=[{k:v for k,v in leaf.items() if k != "data"} for leaf in leaves])))
        self.payload_bytes = sum(leaf["nbytes"] for leaf in leaves)

    @classmethod
    def pack(cls, value):
        aliases, leaves = {}, []
        def visit(item, position=False):
            if isinstance(item, np.ndarray) or _is_tensor(item):
                identity=id(item)
                if identity in aliases:
                    return ["tensor", aliases[identity]]
                index=len(leaves); aliases[identity]=index
                spec=_spec(item)
                repeated = bool(position and len(item.shape) > 0 and int(item.shape[0]) > 1
                                and all(_equal(item[:1], item[i:i+1]) for i in range(1, int(item.shape[0]))))
                data=_cpu_copy(item[:1] if repeated else item)
                payload_id=_sha(_json(dict(kind=spec["kind"], dtype=spec["dtype"], shape=list(data.shape)))+_bytes(data))
                nbytes=int(data.numel()*data.element_size()) if _is_tensor(data) else int(data.nbytes)
                leaves.append(dict(spec=spec, repeated=repeated, payload_id=payload_id,nbytes=nbytes,data=data))
                return ["tensor",index]
            if type(item) is dict:
                if not all(type(key) is str for key in item):
                    raise TypeError("encoder tree dictionary keys must be strings")
                return ["dict", [[key,visit(child,position or key == "vision_pos_enc")] for key,child in item.items()]]
            if type(item) in (list,tuple):
                return [type(item).__name__,[visit(child,position) for child in item]]
            if item is None or type(item) in (bool,int,float,str):
                return ["literal",item]
            raise TypeError("unsupported native encoder output: "+type(item).__name__)
        graph=visit(value)
        return cls(graph,leaves)

    def unpack(self):
        values=[]
        for leaf in self.leaves:
            spec,data=leaf['spec'],leaf['data']
            shape,strides=tuple(spec['shape']),tuple(spec['strides'])
            if spec['kind']=='torch':
                import torch
                restored=torch.empty_strided(shape,strides,dtype=data.dtype,device=spec['device'])
                restored.copy_(data.to(device=spec['device']).expand(shape) if leaf['repeated'] else data.to(device=spec['device']))
            else:
                if any(stride<0 for stride in strides):
                    raise ValueError("negative numpy strides unsupported")
                size=data.dtype.itemsize+sum(max(0,dim-1)*stride for dim,stride in zip(shape,strides))
                restored=np.ndarray(shape,dtype=data.dtype,buffer=bytearray(size),strides=strides)
                np.copyto(restored,np.broadcast_to(data,shape) if leaf['repeated'] else data)
            values.append(restored)
        def restore(node):
            kind,value=node
            if kind=='tensor':return values[value]
            if kind=='literal':return value
            if kind=='dict':return {key:restore(child) for key,child in value}
            items=[restore(child) for child in value]
            return tuple(items) if kind=='tuple' else items
        return restore(self.graph)


def trees_equal(a,b):
    """Exact native-vs-reconstructed values, device, dtype, layout and aliases."""
    left_alias,right_alias={},{}
    def compare(left,right):
        if isinstance(left,np.ndarray) or _is_tensor(left):
            if not (isinstance(right,np.ndarray) or _is_tensor(right)) or _spec(left)!=_spec(right) or not _equal(left,right):return False
            left_alias.setdefault(id(left),len(left_alias));right_alias.setdefault(id(right),len(right_alias))
            return left_alias[id(left)]==right_alias[id(right)]
        if type(left) is not type(right):return False
        if type(left) is dict:return list(left)==list(right) and all(compare(left[k],right[k]) for k in left)
        if type(left) in (list,tuple):return len(left)==len(right) and all(compare(x,y) for x,y in zip(left,right))
        return left==right
    return compare(a,b)


class BoundedMemo:
    """LRU with a refcounted, exact-content CPU payload pool (positions included)."""
    def __init__(self,max_bytes):
        if int(max_bytes)<0:raise ValueError("negative RAM budget")
        self.max_bytes=int(max_bytes);self.entries=OrderedDict();self.pool={}
        self.retained_bytes=self.evictions=self.oversized_bypasses=0

    def get(self,key):
        value=self.entries.get(key)
        if value is not None:self.entries.move_to_end(key)
        return value

    def _drop(self,key):
        packed=self.entries.pop(key)
        for leaf in packed.leaves:
            identity=leaf['payload_id'];self.pool[identity]['refs']-=1
            if not self.pool[identity]['refs']:
                self.retained_bytes-=self.pool[identity]['nbytes'];del self.pool[identity]

    def put(self,key,packed):
        unique={leaf['payload_id']:leaf for leaf in packed.leaves}
        if sum(leaf['nbytes'] for leaf in unique.values())>self.max_bytes:
            self.oversized_bypasses+=1;return False
        if key in self.entries:self._drop(key)
        def added():return sum(leaf['nbytes'] for identity,leaf in unique.items() if identity not in self.pool)
        while self.entries and self.retained_bytes+added()>self.max_bytes:
            self._drop(next(iter(self.entries)));self.evictions+=1
        for leaf in packed.leaves:
            identity=leaf['payload_id']
            if identity not in self.pool:
                self.pool[identity]=dict(data=leaf['data'],nbytes=leaf['nbytes'],refs=0)
                self.retained_bytes+=leaf['nbytes']
            self.pool[identity]['refs']+=1;leaf['data']=self.pool[identity]['data']
        self.entries[key]=packed
        if self.retained_bytes>self.max_bytes:raise AssertionError("RAM budget exceeded")
        return True

    def clear(self):
        self.entries.clear();self.pool.clear();self.retained_bytes=0

    def snapshot(self):
        return dict(budget_bytes=self.max_bytes,retained_bytes=self.retained_bytes,entries=len(self.entries),
                    evictions=self.evictions,oversized_bypasses=self.oversized_bypasses)


class EncoderReuse:
    """Sequential inference-only exact memo over the saved native bound method."""
    def __init__(self,native,guard,context,*,fingerprint,max_bytes=FEATURE_BUDGET_BYTES,actual_frame_batch=4):
        self.native,self.guard,self.context=native,guard,context
        if actual_frame_batch not in (1,2,4):raise ValueError('invalid configured encoder frame batch')
        self.actual_frame_batch=actual_frame_batch
        self.fingerprint=fingerprint;self.initial_guard=guard();self.memo=BoundedMemo(max_bytes)
        self.native_reference={};self.reference_task=None;self.begin('baseline',False,None)

    def begin(self,variant,control,task_key):
        self.check_guard()
        self.variant,self.control,self.task_key=variant,bool(control),task_key
        self.events=[];self.event_count=0;self.event_chain='0'*64;self.io_chain='0'*64
        self.event_prefix_chain=self.io_prefix_chain='0'*64
        self.twin_done=False;self.hit_twin_events=[];self.hit_twin_count=0
        self.counters={name:0 for name in ('hits','misses','bypasses','native_calls','native_twins','native_control_matches')}
        self.seconds={name:0.0 for name in ('native_encoder','input_key','pack','restore','lookup')}
        if variant=='native_control':
            self.native_reference={};self.reference_task=copy.deepcopy(task_key)
        if control and variant!='native_control' and self.reference_task!=task_key:
            raise ValueError('missing same-task native control')

    def check_guard(self):
        if self.guard()!=self.initial_guard:raise ValueError('encoder parameter/buffer version/pointer/layout changed')

    def _native(self,images):
        # CUDA CPU transfers inside packing synchronize the real result. Explicit
        # synchronization is necessary here for honest standalone encoder timing.
        if _is_tensor(images) and images.device.type=='cuda':
            import torch
            torch.cuda.synchronize(images.device)
        start=time.perf_counter();value=self.native(images)
        if _is_tensor(images) and images.device.type=='cuda':
            import torch
            torch.cuda.synchronize(images.device)
        self.seconds['native_encoder']+=time.perf_counter()-start;self.counters['native_calls']+=1
        return value

    def _pack(self,value):
        start=time.perf_counter();packed=PackedTree.pack(value);self.seconds['pack']+=time.perf_counter()-start
        return packed

    def __call__(self,images):
        self.check_guard();context=copy.deepcopy(self.context())
        if context.get('grad') or context.get('training') or bool(getattr(images,'requires_grad',False)):
            raise ValueError('reuse is inference-only: eval and no_grad required')
        start=time.perf_counter()
        input_sha=_sha(_json(dict(spec=_spec(images),context=context,fingerprint=self.fingerprint,guard=self.initial_guard))+_bytes(images))
        self.seconds['input_key']+=time.perf_counter()-start
        start=time.perf_counter();cached=self.memo.get(input_sha);self.seconds['lookup']+=time.perf_counter()-start
        bypass=self.variant=='native_control'
        twin=False;native_match=False
        if bypass or cached is None:
            self.counters['bypasses' if bypass else 'misses']+=1
            native=self._native(images);packed=self._pack(native)
            if self.control:
                restored=packed.unpack()
                if not trees_equal(native,restored):raise ValueError('native encoder pack/unpack twin mismatch')
                if not bypass:
                    self.counters['native_twins']+=1;twin=True
            self.memo.put(input_sha,packed)
            value=native
        else:
            self.counters['hits']+=1;packed=cached
            start=time.perf_counter();value=packed.unpack();self.seconds['restore']+=time.perf_counter()-start
            if self.control and not self.twin_done:
                native=self._native(images);native_packed=self._pack(native)
                if packed.digest!=native_packed.digest or not trees_equal(native,value):raise ValueError('native encoder reuse twin mismatch')
                twin=True;self.twin_done=True;self.counters['native_twins']+=1
        if bypass:
            previous=self.native_reference.setdefault(input_sha,packed.digest)
            if previous!=packed.digest:raise ValueError('native encoder non-deterministic same-input output')
        elif self.control and self.variant=='baseline':
            if self.native_reference.get(input_sha)!=packed.digest:raise ValueError('baseline lacks identical actual native control input/output')
            native_match=True;self.counters['native_control_matches']+=1
        event=dict(input_sha256=input_sha,output_sha256=packed.digest,output_digest_kind='lossless_packed_native_tree',
                   actual_batch=int(images.shape[0]),mode='native_bypass' if bypass else ('hit' if cached is not None else 'miss'),
                   actual_native_twin=twin,native_control_match=native_match,
                   unique_payload_bytes=packed.payload_bytes,context=context)
        if twin:event.update(native_twin_input_sha256=input_sha,native_twin_output_sha256=packed.digest,exact_tree_equal=True)
        self.event_count+=1;self.event_chain=_sha(bytes.fromhex(self.event_chain)+_json(event))
        self.io_chain=_sha(bytes.fromhex(self.io_chain)+_json([input_sha,packed.digest]))
        if len(self.events)<EVENT_LIMIT:
            self.events.append(event);self.event_prefix_chain=self.event_chain;self.io_prefix_chain=self.io_chain
        if event['mode']=='hit' and twin:
            self.hit_twin_count+=1
            if len(self.hit_twin_events)<1:self.hit_twin_events.append(copy.deepcopy(event))
        self.check_guard()
        return value

    def proof(self):
        self.check_guard()
        return dict(schema='amos_exact_encoder_reuse_v1',scope='one_case_cpu_ram_only',fingerprint=self.fingerprint,
                    parameter_guard=self.initial_guard,parameter_guard_verified=True,variant=self.variant,task_key=self.task_key,
                    actual_frame_batch=self.actual_frame_batch,
                    control=self.control,
                    counters=dict(self.counters),seconds=dict(self.seconds),memo=self.memo.snapshot(),
                    events=copy.deepcopy(self.events),event_count=self.event_count,event_chain_sha256=self.event_chain,
                    io_chain_sha256=self.io_chain,hit_twin_count=self.hit_twin_count,hit_twin_events=copy.deepcopy(self.hit_twin_events),
                    event_prefix_chain_sha256=self.event_prefix_chain,io_prefix_chain_sha256=self.io_prefix_chain,
                    remaining_event_count=self.event_count-len(self.events),
                    event_limit=EVENT_LIMIT,position_codec='all_batch_samples_exact_equal_then_sample_repeat',
                    alias_codec='native_tensor_object_identity',disk_tensor_bytes=0)


def encoder_parameter_guard(sam):
    """forward_image reads the encoder AND the two decoder FPN projections."""
    values=[]
    modules=[('image_encoder',sam.image_encoder)]
    if getattr(sam,'use_high_res_features_in_sam',True):
        modules.extend((name,getattr(sam.sam_mask_decoder,name)) for name in ('conv_s0','conv_s1'))
    for prefix,module in modules:
        values.append((prefix,'training',bool(module.training)))
        for kind,items in (('parameter',module.named_parameters()),('buffer',module.named_buffers())):
            for name,value in items:
                values.append((prefix,kind,name,int(value._version),int(value.data_ptr()),tuple(value.shape),
                               tuple(value.stride()),str(value.dtype),str(value.device),str(value.layout)))
    if len(values)<=len(modules):raise ValueError('empty encoder parameter guard')
    return _sha(repr(values).encode())


def _context(sam):
    import torch
    def enabled(device):
        try:return bool(torch.is_autocast_enabled(device))
        except TypeError:return bool(torch.is_autocast_enabled() if device=='cuda' else torch.is_autocast_cpu_enabled())
    def dtype(device):
        try:return str(torch.get_autocast_dtype(device))
        except AttributeError:return str(torch.get_autocast_gpu_dtype() if device=='cuda' else torch.get_autocast_cpu_dtype())
    return dict(grad=torch.is_grad_enabled(),training=bool(sam.training or sam.image_encoder.training),
                cuda_autocast=enabled('cuda'),cpu_autocast=enabled('cpu'),cuda_dtype=dtype('cuda'),cpu_dtype=dtype('cpu'),
                inference_mode=bool(torch.is_inference_mode_enabled()),
                tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32),tf32_cudnn=bool(torch.backends.cudnn.allow_tf32),
                deterministic=bool(torch.are_deterministic_algorithms_enabled()),
                cudnn_benchmark=bool(torch.backends.cudnn.benchmark),cudnn_deterministic=bool(torch.backends.cudnn.deterministic))


def _file_signature(path):
    path=Path(path).resolve();stat=path.stat()
    return (str(path),stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns,stat.st_dev,stat.st_ino)


class PreparationReuse:
    """One signed case label plus one existing shared CT, never class tensors."""
    def __init__(self,inner,labels=None,ct_paths=None):
        self.inner=inner;self.arrays={};self.signatures={};self.counters=dict(label_hits=0,label_misses=0,ct_hits=0,ct_misses=0)
        self.labels={str(Path(p).resolve()) for p in (inner.label_paths if labels is None else labels)}
        if len(self.labels)!=1:raise ValueError('preparation reuse requires one signed case label source')
        self.ct_paths=None if ct_paths is None else {str(Path(p).resolve()) for p in ct_paths if p is not None}

    def load(self,kind,path,native):
        if path is None or not Path(path).is_file():return native(path)
        signature=_file_signature(path);key=(kind,signature[0])
        if kind=='label' and signature[0] not in self.labels:return native(path)
        if kind=='ct' and self.ct_paths is not None and signature[0] not in self.ct_paths:return native(path)
        if key in self.signatures and self.signatures[key]!=signature:raise ValueError('memo source changed: '+str(path))
        if key in self.arrays:
            self.counters[kind+'_hits']+=1;return self.arrays[key]
        value=native(path);self.counters[kind+'_misses']+=1
        if value is not None:
            if any(old_kind==kind and old_path!=signature[0] for old_kind,old_path in self.arrays):
                raise ValueError('preparation cache crossed one-case source boundary')
            # Keep only the canonical source array. Class extraction still runs.
            value=np.array(value,copy=True,subok=False);value.flags.writeable=False
            self.arrays[key]=value;self.signatures[key]=signature
        return value

    def snapshot(self):
        return dict(self.counters,retained_bytes=sum(v.nbytes for v in self.arrays.values()),
                    source_signatures=[list(value) for value in self.signatures.values()],
                    preprocessing_memo=False,class_extraction_native=True,ct_outer_callback_native=True)

    def clear(self):self.arrays.clear();self.signatures.clear()


def install_preparation(session):
    inner=session.namespace['inner'];tasks=session.namespace.get('tasks')
    labels=None;ct_paths=None
    if tasks:
        labels=[task.get('label_path',task.get('label')) for task in tasks]
        if any(label is None for label in labels):raise ValueError('missing signed one-case label source')
        if hasattr(inner,'_case_image_cache_path'):
            images=[task.get('image_path',task.get('image')) for task in tasks]
            if any(image is None for image in images):raise ValueError('missing signed one-case image source')
            ct_paths=[inner._case_image_cache_path(image) for image in images]
    memo=PreparationReuse(inner,labels,ct_paths)
    native_array=inner._load_array;native_ct=inner._load_case_image_cache
    inner._load_array=MethodType(lambda self,path:memo.load('label',path,native_array),inner)
    inner._load_case_image_cache=MethodType(lambda self,path:memo.load('ct',path,native_ct),inner)
    return memo


def _source_binding(paths):
    root=Path(paths['root']).resolve()
    for module,relative,expected in ((_restore,'tools/amos_prompt_ablation_restore_runtime.py',RESTORE_RUNTIME_SHA256),
                                     (_restore._native,'tools/amos_prompt_ablation_runtime.py',NATIVE_RUNTIME_SHA256)):
        path=root/relative
        if Path(module.__file__).resolve()!=path or _sha(path.read_bytes())!=expected:raise ValueError('signed runtime SHA/path mismatch: '+relative)
    return dict(restore_runtime_sha256=RESTORE_RUNTIME_SHA256,native_runtime_sha256=NATIVE_RUNTIME_SHA256,
                reuse_runtime_sha256=_sha(Path(__file__).read_bytes()))


def initialize(job,paths):
    binding=_source_binding(paths)
    # The old runtime must perform its real source checks, dataset/type hook
    # installation, model load and E490 tensor restoration before new wrappers.
    session=_restore.initialize(job,paths)
    ns=session.namespace;sam=ns['adapter'].sam
    protocol=Path(paths['protocol']).resolve()
    binding['protocol_sha256']=_sha(protocol.read_bytes())
    binding['native_program_sha256']=session.program.source_sha256
    # The signed protocol already inventories weights, native sources and data.
    # Include its exact bytes and loaded checkpoint path/stat in each key domain.
    weights=Path(ns['custom'].weights).resolve();binding['weights_signature']=list(_file_signature(weights))
    frozen_paths=[Path(_restore.__file__),Path(_restore._native.__file__),Path(__file__),protocol,weights]
    signatures={str(p.resolve()):_file_signature(p) for p in frozen_paths}
    def file_guard():
        for path,signature in signatures.items():
            if _file_signature(path)!=signature:raise ValueError('frozen runtime/weights/protocol source changed: '+path)
    fingerprint=_sha(_json(binding))
    engine=EncoderReuse(sam.forward_image,lambda:encoder_parameter_guard(sam),lambda:_context(sam),fingerprint=fingerprint,
                        actual_frame_batch=session.batch)
    sam.forward_image=engine
    session.reuse_engine=engine;session.reuse_preparation=install_preparation(session)
    session.reuse_binding=binding;session.reuse_file_guard=file_guard
    native_finish=session.finish
    def finish(self):return finish_session(self,native_finish)
    session.finish=MethodType(finish,session)
    # Keep all old audit fields unchanged, adding only a source-bound proof.
    session.input_audit['exact_reuse']=dict(source_binding=binding,fingerprint=fingerprint,
        feature_budget_bytes=FEATURE_BUDGET_BYTES,scope='one_case_cpu_ram_only',preprocessing_memo=False)
    return session


def finish_session(session,native_finish):
    engine=session.reuse_engine
    try:
        session.reuse_file_guard();engine.check_guard()
        proof=native_finish();engine.check_guard();session.reuse_file_guard()
        proof['reuse_execution']=dict(source_binding=session.reuse_binding,final_parameter_guard=engine.initial_guard,
            memo_before_clear=engine.memo.snapshot(),preparation_before_clear=session.reuse_preparation.snapshot(),
            successful_native_finish=True)
        return proof
    finally:
        engine.memo.clear();engine.native_reference.clear();session.reuse_preparation.clear()


def evaluate(session,task,variant):
    session.reuse_file_guard()
    key=list(_restore._native._key(task));engine=session.reuse_engine
    engine.begin(variant,bool(session.job.get('control') or variant=='native_control'),key)
    before=session.reuse_preparation.snapshot()
    record=_restore.evaluate(session,task,variant)
    proof=engine.proof();proof['source_binding']=copy.deepcopy(session.reuse_binding)
    proof['preparation']=dict(before=before,after=session.reuse_preparation.snapshot())
    record['reuse_execution']=proof
    session.reuse_file_guard()
    validate_reuse_proof(proof,variant,control=bool(session.job.get('control') or variant=='native_control'))
    return record


def validate_reuse_proof(proof,variant,control=False):
    if (proof.get('schema')!='amos_exact_encoder_reuse_v1' or proof.get('variant')!=variant
            or proof.get('scope')!='one_case_cpu_ram_only' or proof.get('disk_tensor_bytes')!=0
            or proof.get('parameter_guard_verified') is not True):raise ValueError('missing exact reuse proof')
    for field in ('fingerprint','parameter_guard','event_chain_sha256','io_chain_sha256'):
        if not isinstance(proof.get(field),str) or len(proof[field])!=64:
            raise ValueError('invalid reuse digest '+field)
    counts=proof.get('counters',{});memo=proof.get('memo',{})
    if any(type(counts.get(key)) is not int or counts[key]<0 for key in
           ('hits','misses','bypasses','native_calls','native_twins','native_control_matches')):
        raise ValueError('invalid reuse counters')
    calls=sum(int(counts.get(k,-1)) for k in ('hits','misses','bypasses'))
    if (calls<=0 or calls!=proof.get('event_count') or memo.get('retained_bytes',-1)<0
            or memo.get('retained_bytes',-1)>memo.get('budget_bytes',-1)
            or not 0<=memo.get('budget_bytes',-1)<=FEATURE_BUDGET_BYTES):raise ValueError('invalid reuse calls/budget')
    events=proof.get('events',[])
    if len(events)!=min(calls,EVENT_LIMIT):raise ValueError('missing bounded encoder events')
    if type(proof.get('control')) is not bool or (control and not proof['control']):raise ValueError('missing real control mode')
    control=bool(control or proof['control'])
    hit_twins=proof.get('hit_twin_count')
    if type(hit_twins) is not int or hit_twins not in (0,1) or hit_twins>counts['hits']:raise ValueError('invalid real hit-twin count')
    if counts['native_calls']!=counts['misses']+counts['bypasses']+hit_twins:raise ValueError('native encoder call arithmetic mismatch')
    expected_twins=counts['misses']+hit_twins if control and variant!='native_control' else 0
    if counts['native_twins']!=expected_twins:raise ValueError('native encoder twin arithmetic mismatch')
    expected_matches=calls if control and variant=='baseline' else 0
    if counts['native_control_matches']!=expected_matches:raise ValueError('native control match arithmetic mismatch')
    if proof.get('remaining_event_count')!=calls-len(events):raise ValueError('remaining event count mismatch')
    batch_cap=proof.get('actual_frame_batch')
    if batch_cap not in (1,2,4):raise ValueError('invalid configured encoder batch')
    def validate_event(event):
        if any(len(event.get(name,''))!=64 for name in ('input_sha256','output_sha256')):raise ValueError('missing actual encoder digests')
        if type(event.get('actual_batch')) is not int or not 1<=event['actual_batch']<=batch_cap:raise ValueError('unapproved native encoder batch')
        if event.get('context',{}).get('grad') or event.get('context',{}).get('training'):raise ValueError('invalid inference reuse context')
        if event.get('mode') not in ('hit','miss','native_bypass'):raise ValueError('invalid encoder reuse event mode')
        if any(type(event.get(name)) is not bool for name in ('actual_native_twin','native_control_match')):
            raise ValueError('missing actual encoder comparison flags')
        expected_match=bool(control and variant=='baseline')
        if event['native_control_match']!=expected_match:raise ValueError('encoder event native control match mismatch')
        if (variant=='native_control')!=(event['mode']=='native_bypass'):raise ValueError('encoder bypass mode differs from variant')
        if not control and event['actual_native_twin']:raise ValueError('production event claims native twin')
        if control and variant!='native_control' and event['mode']=='miss' and not event['actual_native_twin']:
            raise ValueError('control miss lacks pack/unpack native twin')
        if event.get('actual_native_twin') and (event.get('native_twin_input_sha256')!=event['input_sha256']
                or event.get('native_twin_output_sha256')!=event['output_sha256'] or event.get('exact_tree_equal') is not True):
            raise ValueError('invalid native encoder twin digests')
    prefix_counts=dict(hits=0,misses=0,bypasses=0)
    chain=io_chain='0'*64
    for event in events:
        validate_event(event)
        prefix_counts[{'hit':'hits','miss':'misses','native_bypass':'bypasses'}[event['mode']]]+=1
        chain=_sha(bytes.fromhex(chain)+_json(event));io_chain=_sha(bytes.fromhex(io_chain)+_json([event['input_sha256'],event['output_sha256']]))
    if any(prefix_counts[key]>counts[key] or (calls<=EVENT_LIMIT and prefix_counts[key]!=counts[key]) for key in prefix_counts):
        raise ValueError('encoder event modes/counters mismatch')
    for field,count_key in (('actual_native_twin','native_twins'),('native_control_match','native_control_matches')):
        total=sum(event[field] for event in events)
        if total>counts[count_key] or (calls<=EVENT_LIMIT and total!=counts[count_key]):raise ValueError('encoder comparison flag/counter mismatch')
    if chain!=proof.get('event_prefix_chain_sha256') or io_chain!=proof.get('io_prefix_chain_sha256'):raise ValueError('encoder prefix event chain mismatch')
    if calls<=EVENT_LIMIT and (chain!=proof['event_chain_sha256'] or io_chain!=proof['io_chain_sha256']):raise ValueError('encoder event chain mismatch')
    for event in proof.get('hit_twin_events',[]):validate_event(event)
    prefix_hit_twins=[event for event in events if event['mode']=='hit' and event['actual_native_twin']]
    if prefix_hit_twins and proof.get('hit_twin_events')!=prefix_hit_twins:raise ValueError('retained native hit twin differs from actual event prefix')
    if variant=='native_control' and (counts.get('hits')!=0 or counts.get('bypasses')!=calls or counts.get('native_calls')!=calls):
        raise ValueError('native control consumed reuse')
    if control and variant!='native_control':
        if counts.get('native_twins',0)<=0 or not any(e.get('actual_native_twin') for e in events):raise ValueError('missing actual native encoder twin')
        if counts['hits'] and (proof.get('hit_twin_count')!=1 or len(proof.get('hit_twin_events',[]))!=1
                or proof['hit_twin_events'][0].get('mode')!='hit' or proof['hit_twin_events'][0].get('actual_native_twin') is not True):
            raise ValueError('real reuse hit lacks actual native twin')
        if variant=='baseline' and (counts.get('hits',0)<=0 or counts.get('native_control_matches')!=calls):raise ValueError('baseline requires real hits and same-batch actual native control matches')
    return True


def validate_stage_coverage(proof,stages,actual_frame_batch):
    batches=[min(actual_frame_batch,stage['frames']-start) for stage in stages
             for start in range(0,stage['frames'],actual_frame_batch)]
    if (proof.get('actual_frame_batch')!=actual_frame_batch or proof.get('event_count')!=len(batches)
            or [event['actual_batch'] for event in proof['events']]!=batches[:EVENT_LIMIT]):
        raise ValueError('encoder reuse calls differ from actual native stage frame chunks')
    return True


def validate_record(record,task,variant,actual_frame_batch,control=False):
    _restore.validate_record(record,task,variant,actual_frame_batch,control=control)
    proof=record.get('reuse_execution',{});validate_reuse_proof(proof,variant,control)
    validate_stage_coverage(proof,record['stages'],actual_frame_batch)
    expected=_source_binding(dict(root=str(Path(__file__).resolve().parents[1])))
    if any(proof.get('source_binding',{}).get(key)!=value for key,value in expected.items()):raise ValueError('reuse source binding changed')
    if proof.get('task_key')!=list(_restore._native._key(task)) or len(proof.get('fingerprint',''))!=64:raise ValueError('reuse task/fingerprint mismatch')
    if proof['source_binding'].get('native_program_sha256')!=record.get('native_source_sha256'):
        raise ValueError('reuse native program source differs from native record')
    if (proof['fingerprint']!=_sha(_json(proof['source_binding']))
            or proof.get('actual_frame_batch')!=actual_frame_batch
            or record.get('runtime_input_audit',{}).get('exact_reuse',{}).get('source_binding')!=proof['source_binding']
            or record['runtime_input_audit']['exact_reuse'].get('fingerprint')!=proof['fingerprint']):
        raise ValueError('reuse frozen model/source fingerprint mismatch')
    return True


def validate_control(native_record,records):
    proof=_restore.validate_control(native_record,records)
    validate_reuse_proof(native_record.get('reuse_execution',{}),'native_control',True)
    for record in records:validate_reuse_proof(record.get('reuse_execution',{}),record['variant'],True)
    baseline=next(record for record in records if record['variant']=='baseline')
    if (native_record['reuse_execution']['io_chain_sha256']!=baseline['reuse_execution']['io_chain_sha256']
            or native_record['reuse_execution']['event_count']!=baseline['reuse_execution']['event_count']):
        raise ValueError('native-vs-baseline full encoder call chain differs')
    native_events=native_record['reuse_execution']['events'];baseline_events=baseline['reuse_execution']['events']
    if [(e['input_sha256'],e['output_sha256']) for e in native_events]!=[(e['input_sha256'],e['output_sha256']) for e in baseline_events]:
        raise ValueError('native-vs-baseline actual encoder input/output trees differ')
    return dict(proof,exact_encoder_reuse_passed=True,
                actual_native_encoder_twins=sum(r['reuse_execution']['counters']['native_twins'] for r in records),
                actual_baseline_encoder_hits=baseline['reuse_execution']['counters']['hits'],
                encoder_input_output_trees_equal=True,production_speedup_claimed=False)
