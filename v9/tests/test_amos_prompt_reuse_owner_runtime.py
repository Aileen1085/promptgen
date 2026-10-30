"""Observe the actual bound SAM forward_image owner through PEFT shadows."""
import importlib
from pathlib import Path
import types
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ENTRY=ROOT/'tools/amos_prompt_reuse_owner_runtime.py'


class Parameter:
    def __init__(self):self._version=0;self.shape=(1,);self.dtype='float32';self.device='cpu';self.layout='strided'
    def data_ptr(self):return id(self)
    def stride(self):return (1,)


class Layer:
    training=False
    def __init__(self):self.weight=Parameter()
    def named_parameters(self):return [('weight',self.weight)]
    def named_buffers(self):return []


class Native:
    training=False;use_high_res_features_in_sam=True
    def __init__(self):self.image_encoder=Layer();self.sam_mask_decoder=types.SimpleNamespace(conv_s0=Layer(),conv_s1=Layer())
    def forward_image(self,images):return {'vision_features':images+1,'backbone_fpn':[images+2,images+3]}


class Shadow:
    training=False
    def __init__(self,owner):
        self.forward_image=owner.forward_image;self.image_encoder=Layer();self.use_high_res_features_in_sam=True
        self.sam_mask_decoder=types.SimpleNamespace(core_decoder=types.SimpleNamespace(conv_s0=Layer(),conv_s1=Layer()))


class OwnerTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(ENTRY.exists(),'add source-bound actual forward_image owner runtime')
        self.m=importlib.import_module('tools.amos_prompt_reuse_owner_runtime')

    def test_shadow_decoder_without_conv_uses_actual_bound_native_owner(self):
        owner=Native();shadow=Shadow(owner);old=importlib.import_module('tools.amos_prompt_reuse_runtime')
        with self.assertRaises(AttributeError):old.encoder_parameter_guard(shadow)
        self.assertEqual(self.m.encoder_parameter_guard(owner),self.m.encoder_parameter_guard(shadow))
        original=self.m.encoder_parameter_guard(shadow)
        shadow.sam_mask_decoder.core_decoder.conv_s0.weight._version+=1;shadow.image_encoder.weight._version+=1
        self.assertEqual(self.m.encoder_parameter_guard(shadow),original)

    def test_actual_image_and_projection_mutations_change_guard(self):
        for part in ('image_encoder','conv_s0','conv_s1'):
            owner=Native();shadow=Shadow(owner);before=self.m.encoder_parameter_guard(shadow)
            layer=owner.image_encoder if part=='image_encoder' else getattr(owner.sam_mask_decoder,part)
            layer.weight._version+=1
            self.assertNotEqual(self.m.encoder_parameter_guard(shadow),before)

    def test_wrapped_engine_retains_same_owner_guard_and_refuses_changed_parameters(self):
        owner=Native();shadow=Shadow(owner);before=self.m.encoder_parameter_guard(shadow)
        engine=self.m.EncoderReuse(shadow.forward_image,lambda:self.m.encoder_parameter_guard(shadow),lambda:dict(training=False,grad=False),fingerprint='x')
        shadow.forward_image=engine
        self.assertEqual(self.m.encoder_parameter_guard(shadow),before)
        engine(np.ones((2,2),np.float32));engine(np.ones((2,2),np.float32));self.assertEqual(engine.proof()['counters']['hits'],1)
        owner.sam_mask_decoder.conv_s1.weight._version+=1
        with self.assertRaisesRegex(ValueError,'parameter'):engine(np.ones((2,2),np.float32))

    def test_callable_owner_and_class_function_identity_are_strict(self):
        owner=Native();shadow=Shadow(owner);before=self.m.encoder_parameter_guard(shadow)
        shadow.forward_image=Native().forward_image
        self.assertNotEqual(self.m.encoder_parameter_guard(shadow),before)
        for call in (lambda image:image,Native.forward_image,types.SimpleNamespace(native=owner.forward_image)):
            shadow.forward_image=call
            with self.assertRaisesRegex(ValueError,'bound|native|forward_image'):self.m.encoder_parameter_guard(shadow)
        owner=Native();owner.forward_image=types.MethodType(lambda self,image:image,owner)
        with self.assertRaisesRegex(ValueError,'function|forward_image'):self.m.encoder_parameter_guard(owner)

    def test_high_resolution_flag_tracks_only_dependencies_native_reads(self):
        owner=Native();owner.use_high_res_features_in_sam=False;owner.sam_mask_decoder=object()
        self.m.encoder_parameter_guard(owner)
        owner.use_high_res_features_in_sam=True
        with self.assertRaisesRegex(ValueError,'projection|conv'):self.m.encoder_parameter_guard(owner)

    def test_shadow_eval_never_hides_actual_native_training_mode(self):
        for name in ('owner','image_encoder','conv_s0','conv_s1'):
            owner=Native();shadow=Shadow(owner)
            module=owner if name=='owner' else owner.image_encoder if name=='image_encoder' else getattr(owner.sam_mask_decoder,name)
            module.training=True
            with self.assertRaisesRegex(ValueError,'training|eval'):self.m.encoder_parameter_guard(shadow)
            self.assertTrue(module.training);self.assertFalse(shadow.training)

    def test_ancestor_sha_checked_before_every_exported_operation(self):
        for name,args in [('initialize',({},{})),('evaluate',(None,None,None)),('validate_record',(None,None,None,None)),('validate_control',(None,None)),('finish_session',(None,None))]:
            with patch.object(self.m,'ANCESTOR_SHA256','0'*64):
                with self.assertRaisesRegex(ValueError,'SHA'):getattr(self.m,name)(*args)

    def test_compiled_frozen_runtime_uses_new_file_and_source_binding(self):
        ns=self.m.engine();old=importlib.import_module('tools.amos_prompt_reuse_runtime')
        self.assertEqual(Path(ns['__file__']),ENTRY)
        self.assertIs(ns['EncoderReuse'].__init__.__globals__,ns)
        self.assertIsNot(ns,old.__dict__)
        self.assertEqual(ns['METRICS'],old.METRICS);self.assertIs(ns['Session'],old.Session)


class TorchOwnerTests(unittest.TestCase):
    def test_real_torch_native_control_and_exact_hit_with_actual_dependencies(self):
        try:import torch
        except ImportError:self.skipTest('Torch not available locally; mandatory server test')
        m=importlib.import_module('tools.amos_prompt_reuse_owner_runtime')
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__();self.image_encoder=torch.nn.Conv2d(1,2,1);self.use_high_res_features_in_sam=True
                self.sam_mask_decoder=torch.nn.Module();self.sam_mask_decoder.conv_s0=torch.nn.Conv2d(2,2,1);self.sam_mask_decoder.conv_s1=torch.nn.Conv2d(2,2,1)
            def forward_image(self,images):
                feature=self.image_encoder(images)
                return {'vision_features':feature,'backbone_fpn':[self.sam_mask_decoder.conv_s0(feature),self.sam_mask_decoder.conv_s1(feature)]}
        owner=Tiny().eval();shadow=Shadow(owner);images=torch.arange(8,dtype=torch.float32).reshape(2,1,2,2)
        with torch.inference_mode():
            engine=m.EncoderReuse(shadow.forward_image,lambda:m.encoder_parameter_guard(shadow),lambda:dict(training=False,grad=False),fingerprint='tiny-owner')
            shadow.forward_image=engine;engine.begin('native_control',True,['amos','a',201]);native=engine(images)
            engine.begin('baseline',True,['amos','a',201]);baseline=engine(images);hit=engine(images)
            self.assertTrue(m.trees_equal(native,baseline));self.assertTrue(m.trees_equal(baseline,hit));self.assertGreater(engine.proof()['counters']['hits'],0)
            owner.sam_mask_decoder.conv_s0.weight.add_(1)
            with self.assertRaisesRegex(ValueError,'parameter'):engine(images)


if __name__=='__main__':unittest.main()
