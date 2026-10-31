"""Module proofs stay distinct from the constant native BG-scribble prompt."""
import copy
import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class ModuleRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((ROOT/'tools/amos_bg_scribble_module_runtime.py').exists(), 'add module runtime')
        self.m = importlib.import_module('tools.amos_bg_scribble_module_runtime')

    def test_frozen_native_prompt_changes_are_private_and_narrow(self):
        old = importlib.import_module('tools.amos_prompt_ablation_runtime')
        before = old.Session.modes
        native = self.m.native_engine()
        self.assertIs(old.Session.modes, before)
        self.assertIsNot(native['Session'], old.Session)
        for variant in ('native_control', *self.m.MODULE_VARIANTS):
            session = native['Session']({}, {})
            session.variant = variant
            self.assertEqual(session.modes()['background_mode'], 'scribble')
        self.assertEqual(native['PROMPT_VARIANT'], 'fg_scribble_bg_scribble')
        self.assertEqual(self.m.TRANSFORM_COUNTS, {'prompt_selectors': 3, 'background_default': 1})

    def test_changed_original_intervention_or_runtime_source_refused(self):
        with patch.object(self.m, 'INTERVENTION_SHA256', '0'*64):
            with self.assertRaisesRegex(ValueError, 'SHA'): self.m.source_guard()
        with self.assertRaisesRegex(ValueError, 'SHA'):
            self.m.native_engine(source=b'pass')

    def test_changed_prompt_anchor_count_is_refused_even_if_hash_is_rebound(self):
        payload=(ROOT/self.m.NATIVE).read_bytes().replace(b'background_mode="point"',b'background_mode="scribble"')
        import hashlib
        with patch.object(self.m,'NATIVE_SHA256',hashlib.sha256(payload).hexdigest()):
            with self.assertRaisesRegex(ValueError,'anchors'):self.m.transform_native(payload)

    def fixture(self):
        class Module:
            def forward(self, *a, **k): return a[0] if a else None
            def __call__(self, *a, **k): return self.forward(*a, **k)
        encoder = Module(); encoder.base_encoder = Module(); encoder.router = Module(); encoder.experts = [Module()]
        prompt = types.SimpleNamespace(encoder3d=encoder, semantic_dim=128,
            _semantic_embedding=lambda *a: np.ones((1,128)), semantic_encoder=Module())
        decoder = types.SimpleNamespace(prompt_3d_adapter=Module(), prompt_memory_adapter=Module())
        guard_calls = []
        session = types.SimpleNamespace(namespace={'prompt': prompt, 'adapter': types.SimpleNamespace(sam=types.SimpleNamespace(sam_mask_decoder=decoder))},
            restoration=dict(model='v10_2', epoch=490, weight_mode='live', prompt_parameters_exact=418,
                             decoder_parameters_exact=212, guards_passed=True, model_load_count=1),
            check_parameter_guard=lambda: guard_calls.append('parameters') or {'guard':'original'},
            reuse_engine=types.SimpleNamespace(check_guard=lambda: guard_calls.append('owner')))
        session.module_methods = self.m.method_snapshot(prompt, decoder)
        return session, prompt, decoder, guard_calls

    def test_each_condition_installs_original_intervention_and_restores_in_finally(self):
        session, prompt, decoder, guards = self.fixture(); task = {'case_id':'a'}; seen=[]
        def apply(p, d, variant):
            seen.append(variant)
            p.encoder3d.forward = lambda x: x
            return dict(variant=variant,calls={variant:1},memory_disabled=False)
        def evaluate(s, t, variant):
            self.assertEqual(t,task)
            self.assertNotEqual(self.m.method_snapshot(prompt,decoder), session.module_methods)
            raise RuntimeError('CUDA out of memory')
        with patch.object(self.m,'apply_intervention',side_effect=apply), patch.object(self.m,'reuse_engine',return_value={'evaluate':evaluate}):
            for variant in self.m.MODULE_VARIANTS:
                with self.assertRaisesRegex(RuntimeError,'out of memory'): self.m.evaluate(session,task,variant)
                self.assertEqual(self.m.method_snapshot(prompt,decoder),session.module_methods)
        self.assertEqual(seen,list(self.m.MODULE_VARIANTS));self.assertIn('owner',guards)

    def test_record_normalization_changes_only_label_fields_and_preserves_original(self):
        value=dict(variant='no_semantic',module_variant='no_semantic',prompt_variant=self.m.PROMPT_VARIANT,
                   metric_execution={'events':[{'variant':'no_semantic','input_sha256':'abc'}]},
                   prompt={'background_mode':'scribble'},roi={'initial':[1]},stages=[{'frames':4}])
        old=copy.deepcopy(value); normalized=self.m.normalized_record(value)
        self.assertEqual(value,old)
        self.assertEqual(normalized['variant'],self.m.PROMPT_VARIANT)
        self.assertEqual(normalized['metric_execution']['events'][0]['variant'],self.m.PROMPT_VARIANT)
        for key in ('prompt','roi','stages'):self.assertEqual(normalized[key],value[key])
        value['prompt_variant']='baseline'
        with self.assertRaisesRegex(ValueError,'prompt'):self.m.normalized_record(value)

    def test_owner_encoder_is_the_signed_guard_in_private_reuse_namespace(self):
        ns=self.m.reuse_engine();owner=importlib.import_module('tools.amos_prompt_reuse_owner_runtime')
        self.assertEqual(ns['FEATURE_BUDGET_BYTES'],2*1024**3)
        self.assertEqual(ns['encoder_parameter_guard'].__code__.co_code,owner.encoder_parameter_guard.__code__.co_code)
        self.assertIs(ns['_restore']._native.__dict__,self.m.native_engine())
        self.assertEqual(ns['_restore'].evaluate.__name__,'evaluate_prompt')

    def test_private_owner_guard_tracks_native_projection_even_with_shadow_decoder(self):
        from tests.test_amos_prompt_reuse_owner_runtime import Native, Shadow
        ns=self.m.reuse_engine();owner=Native();sam=Shadow(owner)
        guard=lambda:ns['encoder_parameter_guard'](sam)
        engine=ns['EncoderReuse'](sam.forward_image,guard,lambda:dict(training=False,grad=False),fingerprint='owner')
        sam.forward_image=engine
        images=np.ones((2,2),np.float32)
        engine(images);engine(images)
        self.assertEqual(engine.proof()['counters']['hits'],1)
        owner.sam_mask_decoder.conv_s0.weight._version+=1
        with self.assertRaisesRegex(ValueError,'parameter'):engine(images)

    def test_private_prompt_dispatch_always_uses_fixed_prompt_and_labels_metrics(self):
        ns=self.m.native_engine();reuse=self.m.reuse_engine();seen=[]
        def native(s,t,v):
            seen.append(v)
            return dict(variant=v,metric_execution={'events':[dict(variant=v)]})
        with patch.dict(ns,evaluate=native):
            for variant in self.m.MODULE_VARIANTS:
                record=reuse['_restore'].evaluate(object(),{},variant)
                self.assertEqual(record['module_variant'],variant)
                self.assertEqual(record['prompt_variant'],self.m.PROMPT_VARIANT)
                self.assertEqual(record['metric_execution']['events'][0]['variant'],variant)
            reuse['_restore'].evaluate(object(),{},'native_control')
        self.assertEqual(seen,[self.m.PROMPT_VARIANT]*6+['native_control'])

    def test_no_semantic_forbids_residual_mlp_as_well_as_text_encoder(self):
        session,prompt,decoder,_=self.fixture()
        prompt.semantic_residual_mlp=type(prompt.semantic_encoder)()
        before=self.m.method_snapshot(prompt,decoder)
        observed,forbidden=self.m.observe_methods(prompt,decoder,'no_semantic')
        try:
            with self.assertRaisesRegex(RuntimeError,'removed module'):prompt.semantic_residual_mlp(np.ones((1,128)))
            self.assertEqual(forbidden['semantic_residual'],1)
        finally:self.m.restore_methods(before)

    def test_module_proof_requires_real_calls_forbidden_zero_restored_methods_and_guards(self):
        for variant in self.m.MODULE_VARIANTS:
            calls={name:1 for name in self.m.required_names(variant)}
            forbidden={'router':0,'expert_0':0} if variant=='shared' else ({'text_encoder':0} if variant=='no_semantic' else {})
            original={'shared':['shared'],'no_3d_adapter':['no_3d_adapter'],'no_memory_adapter':['no_memory_adapter'],
                      'no_both_adapters':['no_3d_adapter','no_memory_adapter'],'no_semantic':['no_semantic']}.get(variant,[])
            proof=dict(schema='amos_bg_scribble_module_v1',variant=variant,prompt_variant=self.m.PROMPT_VARIANT,
                       intervention_sha256=self.m.INTERVENTION_SHA256,runtime_sha256=self.m.sha(self.m.__file__),
                       observed_calls=calls,forbidden_calls=forbidden,methods_restored=True,owner_encoder_guard_passed=True,
                       parameter_guard_before={'original':1},parameter_guard_after={'original':1},
                       method_guard_before='a'*64,method_guard_after='a'*64,
                       original=dict(variant=variant,memory_disabled=False,calls={name:1 for name in original}))
            self.m.validate_module_proof(proof,variant)
            for mutation in ('required','forbidden','restore','parameters'):
                bad=copy.deepcopy(proof)
                if mutation=='required':bad['observed_calls']={}
                if mutation=='forbidden':bad['forbidden_calls']['router']=1
                if mutation=='restore':bad['methods_restored']=False
                if mutation=='parameters':bad['parameter_guard_after']={}
                with self.assertRaises(ValueError):self.m.validate_module_proof(bad,variant)

    def test_successful_six_conditions_run_original_methods_and_never_share_decoder_outputs(self):
        class Tensor(np.ndarray):
            def expand(self,n,*shape):return np.broadcast_to(self,(n,*self.shape[1:])).view(Tensor)
            def contiguous(self):return np.ascontiguousarray(self).view(Tensor)
            def numel(self):return self.size
            def new_zeros(self,shape):return np.zeros(shape,dtype=self.dtype).view(Tensor)
        tensor=lambda shape:np.ones(shape,np.float32).view(Tensor)
        fake_torch=types.SimpleNamespace(Tensor=Tensor,cat=lambda values,dim:np.concatenate(values,axis=dim).view(Tensor))
        session,prompt,decoder,_=self.fixture();session.batch=4;session.job={'control':False}
        prompt._semantic_embedding=lambda *a:tensor((1,128));session.module_methods=self.m.method_snapshot(prompt,decoder)
        outputs=[]
        def forward(s,t,v):
            # Run each actual signed intervention. No previous decoder result
            # enters a subsequent condition's native task invocation.
            prompt.encoder3d(tensor((1,2,8)))
            prompt._semantic_embedding('liver',None,tensor((1,)))
            decoder.prompt_3d_adapter(tensor((1,2,8)),tensor((2,)),tensor((2,1,8)))
            decoder.prompt_memory_adapter(tensor((1,2,2,2)))
            value=dict(variant=v,module_variant=v,prompt_variant=self.m.PROMPT_VARIANT,
                       metric_execution={'events':[dict(variant=v)]},
                       stages=[dict(frames=2,memory_calls={'memory_encoder':2,'memory_attention':1})])
            outputs.append(value);return value
        with patch.dict(sys.modules,torch=fake_torch),patch.object(self.m,'reuse_engine',return_value={'evaluate':forward,'validate_record':lambda *a:True}):
            for variant in self.m.MODULE_VARIANTS:
                result=self.m.evaluate(session,{},variant)
                self.assertEqual(result['module_execution']['variant'],variant)
                self.assertEqual(self.m.method_snapshot(prompt,decoder),session.module_methods)
        self.assertEqual(len({id(value) for value in outputs}),6)

    def test_original_native_validator_accepts_real_bg_scribble_proof_without_normalizing_prompt_rasters(self):
        from tests.test_amos_prompt_ablation_runtime import FrozenProtocolTests
        from tools.amos_prompt_ablation_prompts import build_variant,variant_modes
        fixture=FrozenProtocolTests();fixture.setUp();record,task=fixture.record()
        # Produce a synthetic BG-scribble raster and its real transform audit;
        # this is a CPU validator check, never a claim of real GPU controls.
        ns=self.m.native_engine();modes=variant_modes(self.m.PROMPT_VARIANT)
        package=dict(coronal=np.zeros((12,5,7),np.uint8),sagittal=np.zeros((12,5,7),np.uint8),
                     coronal_slice=2,sagittal_slice=2,background_points_dhw=np.array([[0,0,2],[11,4,2],[0,2,0],[11,2,6]]),
                     background_point_planes=['coronal','coronal','sagittal','sagittal'])
        package['coronal'][3:5,1:4,2]=1;package['sagittal'][3:5,2,2:5]=1
        groups={'coronal':np.zeros((3,12,5),np.uint8),'sagittal':np.zeros((3,12,7),np.uint8)}
        for group in groups.values():group[1,:,-1]=1
        converted,prompt=build_variant(package,groups,self.m.PROMPT_VARIANT,8)
        prompt.update(group_builds=[{}],actual_consumption_calls=1);record['prompt']=prompt
        consume=ns['validate_consumption'](converted['groups'],converted['plane_ids'],0,modes,converted['groups'],
            slice_coords=converted['slice_coords'],expected_slice_coords=converted['slice_coords'])
        for stage in record['stages']:
            stage['bundle']=consume
            stage['prompt_calls']=[dict(stage['prompt_calls'][0],**consume)]
            for event in [*stage['prompt_calls'],*stage['global_pool_calls'],*stage['decoder_calls'],*stage['adapter_calls']]:
                event['type_ids']=[0,3]
        record['variant']=self.m.PROMPT_VARIANT
        record['metric_execution']['events'][0]['variant']=self.m.PROMPT_VARIANT
        ns['validate_record'](record,task,self.m.PROMPT_VARIANT,4)
        bad=copy.deepcopy(record);bad['prompt']['background_mode']='point'
        with self.assertRaisesRegex(ValueError,'mode'):ns['validate_record'](bad,task,self.m.PROMPT_VARIANT,4)


try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'real Torch intervention behavior runs on server')
class OriginalInterventionBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.m=importlib.import_module('tools.amos_bg_scribble_module_runtime')
        class Add(torch.nn.Module):
            def forward(self,x,*a,**k):return x+9
        self.prompt=types.SimpleNamespace(encoder3d=Add(),semantic_dim=128,_semantic_embedding=lambda *a:torch.ones(1,128),semantic_encoder=Add())
        self.prompt.encoder3d.base_encoder=Add();self.prompt.encoder3d.router=Add();self.prompt.encoder3d.experts=torch.nn.ModuleList([Add()])
        self.decoder=types.SimpleNamespace(prompt_3d_adapter=Add(),prompt_memory_adapter=Add())

    def test_all_original_interventions_preserve_exact_behavior(self):
        original=self.m.method_snapshot(self.prompt,self.decoder)
        for variant in self.m.MODULE_VARIANTS:
            with self.subTest(variant=variant):
                audit=self.m.apply_intervention(self.prompt,self.decoder,variant)
                if variant=='shared':
                    self.assertEqual(self.prompt.encoder3d(torch.tensor(1.)).item(),10)
                    with self.assertRaises(RuntimeError):self.prompt.encoder3d.router(torch.tensor(1.))
                    with self.assertRaises(RuntimeError):self.prompt.encoder3d.experts[0](torch.tensor(1.))
                if variant in ('no_3d_adapter','no_both_adapters'):
                    tokens=torch.arange(24.).reshape(1,3,8);sparse=torch.ones(4,2,8)
                    actual=self.decoder.prompt_3d_adapter(tokens,torch.arange(4),sparse)
                    self.assertTrue(torch.equal(actual,torch.cat((sparse,tokens.expand(4,-1,-1)),dim=1)))
                if variant in ('no_memory_adapter','no_both_adapters'):
                    features=torch.ones(1,2,2,2);self.assertIs(self.decoder.prompt_memory_adapter(features),features)
                if variant=='no_semantic':
                    actual=self.prompt._semantic_embedding('liver',None,torch.ones(1,dtype=torch.float64))
                    self.assertEqual(tuple(actual.shape),(1,128));self.assertEqual(actual.dtype,torch.float64)
                    self.assertEqual(actual.abs().sum().item(),0)
                    with self.assertRaises(RuntimeError):self.prompt.semantic_encoder(torch.ones(1))
                self.assertEqual(audit['memory_disabled'],False)
                self.m.restore_methods(original)


if __name__=='__main__':unittest.main()
