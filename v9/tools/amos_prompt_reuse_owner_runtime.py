"""Exact reuse with a guard over the saved native forward_image bound owner.

The frozen runtime's cache, native forwards, metrics and proofs are unchanged.
Only the observer resolves the object whose forward_image actually executes.
"""
from __future__ import annotations
import hashlib
import inspect
from pathlib import Path
import types

ROOT=Path(__file__).resolve().parents[1]
ANCESTOR='tools/amos_prompt_reuse_runtime.py'
ANCESTOR_SHA256='bd3f4c1274088edc8fea1cfc0c908d9a4e1b436471a85d2e2400be9f56d9725a'
_ENGINE=None


def ancestor_guard(root=None):
    path=Path(root or ROOT)/ANCESTOR
    if hashlib.sha256(path.read_bytes()).hexdigest()!=ANCESTOR_SHA256:raise ValueError('signed ancestor runtime SHA changed')


def native_owner(sam):
    """A bound method's owner, rather than a wrapper's similarly named attrs."""
    call=getattr(sam,'forward_image',None)
    reuse_class=engine()['EncoderReuse']
    if isinstance(call,reuse_class):call=call.native
    if not inspect.ismethod(call) or call.__self__ is None or call.__func__.__name__!='forward_image':
        raise ValueError('unsupported native forward_image: original bound method required')
    owner=call.__self__;function=call.__func__
    if inspect.getattr_static(type(owner),'forward_image',None) is not function:
        raise ValueError('native forward_image class function identity changed')
    return owner,function


def encoder_parameter_guard(sam):
    ancestor_guard();owner,function=native_owner(sam)
    if bool(owner.training):raise ValueError('native forward_image owner is in training mode; eval required')
    high_res=getattr(owner,'use_high_res_features_in_sam',None)
    if type(high_res) is not bool:raise ValueError('native forward_image high resolution flag missing')
    encoder=getattr(owner,'image_encoder',None)
    if encoder is None:raise ValueError('native forward_image image encoder missing')
    values=[('native_bound_owner',id(owner),id(function),function.__module__,function.__qualname__,bool(owner.training),high_res)]
    modules=[('image_encoder',encoder)]
    if high_res:
        decoder=getattr(owner,'sam_mask_decoder',None)
        for name in ('conv_s0','conv_s1'):
            module=getattr(decoder,name,None)
            if module is None:raise ValueError('native forward_image projection missing: '+name)
            modules.append((name,module))
    for prefix,module in modules:
        if not all(hasattr(module,name) for name in ('training','named_parameters','named_buffers')):
            raise ValueError('unsupported native encoder/projection module: '+prefix)
        if bool(module.training):raise ValueError('native forward_image dependency is in training mode; eval required: '+prefix)
        values.append((prefix,'module',id(module),'training',bool(module.training)))
        for kind,items in (('parameter',module.named_parameters()),('buffer',module.named_buffers())):
            for name,value in items:
                values.append((prefix,kind,name,int(value._version),int(value.data_ptr()),tuple(value.shape),
                               tuple(value.stride()),str(value.dtype),str(value.device),str(value.layout)))
    if len(values)<=1+len(modules):raise ValueError('empty native forward_image parameter guard')
    return hashlib.sha256(repr(values).encode()).hexdigest()


def engine():
    global _ENGINE
    ancestor_guard()
    if _ENGINE is None:
        private=types.ModuleType('_amos_prompt_reuse_owner_runtime_private');ns=private.__dict__
        ns['__file__']=str(Path(__file__).resolve())
        exec(compile((ROOT/ANCESTOR).read_bytes(),str(ROOT/ANCESTOR),'exec'),ns)
        ns['encoder_parameter_guard']=encoder_parameter_guard
        original_finish=ns['finish_session']
        def finish(session,native_finish):ancestor_guard();return original_finish(session,native_finish)
        ns['finish_session']=finish;_ENGINE=ns
    return _ENGINE


def initialize(job,paths):ancestor_guard(paths.get('root'));return engine()['initialize'](job,paths)
def evaluate(session,task,variant):ancestor_guard();return engine()['evaluate'](session,task,variant)
def validate_record(*args,**kwargs):ancestor_guard();return engine()['validate_record'](*args,**kwargs)
def validate_control(*args,**kwargs):ancestor_guard();return engine()['validate_control'](*args,**kwargs)
def finish_session(*args,**kwargs):ancestor_guard();return engine()['finish_session'](*args,**kwargs)


def __getattr__(name):
    if name.startswith('__'):raise AttributeError(name)
    ns=engine()
    if name not in ns:raise AttributeError(name)
    return ns[name]
