"""CPU boundary checks; real Torch/CUDA controls remain mandatory."""
import ast
import copy
import hashlib
import importlib
from pathlib import Path
import types
import unittest
from unittest.mock import patch

import numpy as np
try:
    import torch
except ImportError:
    torch=None

ROOT = Path(__file__).resolve().parents[1]


class SemanticRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((ROOT/'tools/amos_semantic_only_runtime.py').is_file(), 'semantic runtime is required')
        self.m = importlib.import_module('tools.amos_semantic_only_runtime')

    def test_private_namespace_binds_empty_modes_without_mutating_native(self):
        old = importlib.import_module('tools.amos_prompt_ablation_runtime')
        before = old.Session.modes
        ns = self.m.native_engine()
        self.assertIs(old.Session.modes, before)
        self.assertIsNot(ns['Session'], old.Session)
        s = ns['Session']({}, {})
        s.variant = 'semantic_only'
        self.assertEqual(s.modes()['plane_ids'], ())
        self.assertFalse(s.modes()['background_active'])
        s.variant = 'baseline'
        self.assertEqual(s.modes()['background_mode'], 'scribble')
        self.assertTrue(ns['TRANSFORM_PROOF']['reverse_ast_equal'])
        self.assertIs(ns['evaluate'].__globals__, ns)

    def test_source_and_anchor_changes_fail_closed(self):
        with self.assertRaisesRegex(ValueError, 'SHA'):
            self.m.native_engine(source=b'pass')
        payload = (ROOT/self.m.NATIVE).read_bytes().replace(b'background_mode="point"', b'background_mode="scribble"')
        with patch.object(self.m, 'NATIVE_SHA256', hashlib.sha256(payload).hexdigest()):
            with self.assertRaisesRegex(ValueError, 'anchor'):
                self.m.transform_native(payload)

    def test_full_ct_geometry_never_reads_prompt_or_target(self):
        ns = self.m.native_engine(); s = ns['Session']({}, {})
        s.variant = 'semantic_only'; s.namespace = {'audit': {'loaded_task_keys': []}}
        s.current = ('amos', 'a', 201)
        calls = []
        inner = types.SimpleNamespace(image_paths=['ct'], prompt_plane_size=8,
            _get_or_create_case_image=lambda p: calls.append(p) or np.ones((7, 4, 6), np.float32),
            _plane_prompt_group=lambda *a: self.fail('plane reader forbidden'))
        s.package = lambda *a: self.fail('prompt cache forbidden')
        for arbitrary_gt in (np.zeros((1, 2, 3)), np.ones((200, 1, 1))):
            indices = self.m.full_ct_roi(s, inner, arbitrary_gt)
            np.testing.assert_array_equal(indices, np.arange(7))
            bundle = s.converted(inner, 0, 201)
            self.assertEqual(bundle['groups'].shape, (0, 3, 8, 8))
            self.assertEqual(bundle['slice_coords'].shape, (0,))
            self.assertEqual(s.prompt_audit['canonical_shape_dhw'], [7, 4, 6])
        self.assertTrue(calls)

    def test_zero_only_adapter_delegates_nonempty_baseline(self):
        calls=[]
        p=types.SimpleNamespace(_mixed_prompt_groups=lambda x: calls.append(x) or 'baseline')
        s=types.SimpleNamespace(variant='semantic_only')
        self.m.install_zero_groups(s,p)
        empty=np.empty((0,3,8,8),np.float32)
        result=p._mixed_prompt_groups(empty)
        self.assertIs(result[0],empty)
        self.assertEqual(result[1:],('scribble','scribble',0,False))
        s.variant='baseline'; self.assertEqual(p._mixed_prompt_groups(np.ones((2,3,8,8))),'baseline')
        self.assertEqual(len(calls),1)
        s.variant='semantic_only'
        with self.assertRaises(ValueError):p._mixed_prompt_groups(np.zeros((1,3,8,8)))

    def test_strict_consumption_and_output_reject_phantom_spatial_input(self):
        m=self.m; modes=m.semantic_modes(); groups=np.empty((0,3,8,8),np.float32)
        event=m.validate_consumption(groups,np.empty(0,np.int64),0,modes,groups,
            slice_coords=np.empty(0,np.float32),expected_slice_coords=np.empty(0,np.float32))
        self.assertEqual(event['plane_ids'],[]);self.assertEqual(event['channel_nonzero'],[])
        with self.assertRaises(ValueError):m.validate_consumption(np.zeros((1,3,8,8)),np.array([0]),0,modes,groups)
        output=dict(prompt_3d_role_ids=np.array([0,0]),prompt_3d_type_ids=np.array([0,0]),
            foreground_prompt_mode='scribble',background_prompt_mode='scribble',foreground_channel=0,background_active=False)
        self.assertEqual(m.validate_output_types(output,modes)['role_ids'],[0])
        output['background_active']=True
        with self.assertRaises(ValueError):m.validate_output_types(output,modes)

    def test_postprocess_only_skips_components_for_verified_empty_semantic(self):
        calls=[]
        native=lambda *a,**k:calls.append('baseline') or ('native',{})
        threshold=lambda *a,**k:calls.append(('threshold',k)) or {'threshold':.6,'source':'fallback_missing_prompt_logits'}
        seed=np.zeros((2,3,4),bool)
        probs=np.full((2,1,3,4),.7,np.float32)
        kwargs=dict(fallback=.6,minimum=.7,maximum=.75,min_logit_separation=.5)
        pred,audit=self.m.postprocess_dispatch('semantic_only',True,probs,seed,seed,native=native,
            threshold=threshold,probability=lambda x:x,**kwargs)
        self.assertTrue(pred.all());self.assertEqual(audit['threshold'],.6)
        self.assertEqual(calls[0][0],'threshold')
        self.assertEqual(self.m.postprocess_dispatch('baseline',False,probs,seed,seed,native=native,threshold=threshold,
            probability=lambda x:x,**kwargs)[0],'native')
        for verified,fg in ((False,seed),(True,np.ones_like(seed))):
            with self.assertRaises(ValueError):self.m.postprocess_dispatch('semantic_only',verified,probs,fg,seed,
                native=native,threshold=threshold,probability=lambda x:x,**kwargs)

    def test_dynamic_local_import_is_removed_privately_and_body_reversible(self):
        source=(ROOT/'v10/infer/infer_eight_source_adaptive_v10_2.py').read_bytes()
        tree,proof=self.m.transform_decode(source,hashlib.sha256(source).hexdigest())
        fn=tree.body[0]
        imports=[a.name for n in ast.walk(fn) if isinstance(n,ast.ImportFrom) for a in n.names]
        self.assertNotIn('postprocess_v10_logits_from_prompts',imports)
        self.assertNotIn('generate_case_3d_tokens',imports)
        self.assertTrue(proof['reverse_ast_equal'])
        self.assertEqual(proof['postprocess_imports'],1)
        self.assertEqual(proof['memory_imports'],2)

    def test_scoring_alias_is_rebound_after_native_initialization_import(self):
        native=self.m.native_engine();old=lambda *a: 'unobserved'
        module=types.ModuleType('scorer_fixture')
        exec('def restore_and_score_full_volume(*a, **k):\n    return "observed"',module.__dict__)
        session=types.SimpleNamespace(namespace={'restore_and_score_full_volume':old})
        with patch.dict(native,_private_module=lambda name:module):
            self.m.bind_native_scoring(session)
        call=session.namespace['restore_and_score_full_volume']
        self.assertEqual(call(),'observed');self.assertIs(call.__globals__,module.__dict__)

    def test_baseline_controls_compare_scientific_geometry_not_audit_source_labels(self):
        ns=self.m.native_engine()
        row={m:.5 for m in ns['METRICS']};row.update(mask_threshold_used=.7,dynamic_threshold_source='prompt_logit_midpoint')
        native=dict(variant='native_control',row=row,roi=dict(initial=[1,2],reserved=[0,1,2],final=[1,2],history=[],builds=[{'source':'native_original'}]),
            stages=[{'frames':2}],actual_frame_batch=4,reuse_execution={'io_chain_sha256':'a'})
        replay=copy.deepcopy(native);replay['variant']='baseline';replay['roi']['builds'][0]['source']='actual_raw_active_foreground'
        semantic=copy.deepcopy(replay);semantic['variant']='semantic_only'
        with patch.object(self.m,'reuse_engine',return_value={'validate_reuse_proof':lambda *a:True}):
            self.assertTrue(self.m.validate_control(native,[replay,semantic])['roi_equal'])
            replay['roi']['final']=[0]
            with self.assertRaisesRegex(ValueError,'replay'):self.m.validate_control(native,[replay,semantic])

    def semantic_record(self):
        from tests.test_amos_prompt_ablation_runtime import FrozenProtocolTests
        record,task=FrozenProtocolTests().record();ns=self.m.native_engine();shape=[12,5,7];depth=12
        modes=self.m.semantic_modes();prompt={k:list(v) if isinstance(v,tuple) else v for k,v in modes.items()}
        prompt.update(canonical_shape_dhw=shape,groups_shape=[0,3,8,8],groups_sha256=hashlib.sha256(b'').hexdigest(),
            planes={},raw_foreground_z_support=[],raw_background_points_per_plane={},geometry_source='actual_readonly_case_CT_shape',
            full_xy=True,prompt_cache_read=False,selected_plane_read=False,gt_geometry_read=False,
            group_builds=[{'groups_sha256':'a'*64,'plane_ids':[]}],actual_consumption_calls=1)
        event=dict(plane_ids=[],slice_coords=[],foreground_mode_id=0,channel_nonzero=[],raster_sha256=prompt['groups_sha256'],groups_sha256='a'*64)
        token=dict(type_ids=[0],role_ids=[0])
        audit=dict(threshold=.6,source='fallback_missing_prompt_logits',semantic_component_bypass=True,
            foreground_seed_count=0,background_seed_count=0,raw_threshold_mask_retained=True)
        stage=dict(frames=depth,frame_coords=(2*np.arange(depth,dtype=np.float32)/(depth-1)-1).tolist(),
            settings={**ns['FROZEN_SETTINGS'],'sam2_frame_batch_size':4},memory_calls={'memory_encoder':depth,'memory_attention':1},
            bundle=event,prompt_calls=[dict(event,**token,semantic_text=record['semantic_text'],frames=depth,
                frame_coords=(2*np.arange(depth,dtype=np.float32)/(depth-1)-1).tolist(),aligned_prompt_exact_zero=True,
                dense_shape=[depth,4,2,2],sparse_shape=[depth,2,4],dense_sha256='d'*64,sparse_sha256='e'*64)],mixed_calls=[{}],
            global_pool_calls=[dict(token,pooled_features_exact_zero=True)],decoder_calls=[dict(token,anchor=0)],adapter_calls=[token]*depth,threshold_audit=audit)
        record.update(variant='semantic_only',prompt=prompt,stages=[stage],
            roi=dict(initial=list(range(depth)),reserved=list(range(depth)),final=list(range(depth)),history=[],
                builds=[dict(source='actual_readonly_case_CT_shape',depth=depth,indices=list(range(depth)),canonical_shape_dhw=shape)]))
        record['row'].update(mask_threshold_used=.6,dynamic_threshold_source='fallback_missing_prompt_logits',label_id=task['local_class_id'])
        record['metric_execution']['events'][0]['variant']='semantic_only'
        return record,task

    def test_strict_semantic_validator_rejects_label_frames_roles_and_fake_bypass(self):
        record,task=self.semantic_record();self.m.validate_semantic_record(record,task,4,True)
        for change in ('label','frames','roles','fallback','bypass','parameters','memory','seed','chunk_coords','pool','dense','groups_shape','build_plane'):
            bad=copy.deepcopy(record)
            if change=='label':bad['row']['label_id']=99
            if change=='frames':bad['roi']['final']=bad['roi']['final'][:-1]
            if change=='roles':bad['stages'][0]['adapter_calls'][0]={'type_ids':[0,3],'role_ids':[0,1]}
            if change=='fallback':bad['stages'][0]['threshold_audit']['threshold']=.7
            if change=='bypass':bad['stages'][0]['threshold_audit'].pop('semantic_component_bypass')
            if change=='parameters':bad['parameter_guard_after']={'prompt':'changed'}
            if change=='memory':bad['stages'][0]['memory_calls']['memory_encoder']=11
            if change=='seed':bad['stages'][0]['threshold_audit']['foreground_seed_count']=1
            if change=='chunk_coords':bad['stages'][0]['prompt_calls'][0]['frame_coords']=[0.]*12
            if change=='pool':bad['stages'][0]['global_pool_calls'][0]['pooled_features_exact_zero']=False
            if change=='dense':bad['stages'][0]['prompt_calls'][0]['dense_shape'][0]=11
            if change=='groups_shape':bad['prompt']['groups_shape'][0]=1
            if change=='build_plane':bad['prompt']['group_builds'][0]['plane_ids']=[0]
            with self.subTest(change=change),self.assertRaises(ValueError):self.m.validate_semantic_record(bad,task,4,True)

    def test_observers_verify_zero_native_pool_and_actual_chunk_outputs(self):
        stage={'prompt_calls':[],'global_pool_calls':[]}
        def forward(groups,video,**kwargs):
            stage['prompt_calls'].append({})
            return dict(aligned_prompt=np.zeros((2,3,8,8),np.float32),dense_prompt_embeddings=np.ones((2,4,2,2)),sparse_prompt_embeddings=np.ones((2,2,4)))
        def pool(features,**kwargs):stage['global_pool_calls'].append({});return {'unchanged':True}
        prompt=types.SimpleNamespace(forward_case_3d_tokens=forward,build_case_3d_tokens_from_global_pool=pool)
        session=types.SimpleNamespace(variant='semantic_only',active_stage=stage,namespace={'prompt':prompt})
        self.m.install_output_observers(session)
        result=prompt.forward_case_3d_tokens(np.empty((0,3,8,8)),np.ones((2,3,8,8)),frame_coords=np.array([-1.,1.]))
        self.assertTrue(result['dense_prompt_embeddings'].all())
        self.assertEqual(stage['prompt_calls'][0]['frame_coords'],[-1.,1.])
        self.assertTrue(stage['prompt_calls'][0]['aligned_prompt_exact_zero'])
        prompt.build_case_3d_tokens_from_global_pool(np.zeros((3,4)))
        self.assertTrue(stage['global_pool_calls'][0]['pooled_features_exact_zero'])
        with self.assertRaisesRegex(ValueError,'pool'):prompt.build_case_3d_tokens_from_global_pool(np.ones((3,4)))

    def test_semantic_embedding_metadata_requires_actual_finite128_and_nonempty_text(self):
        expected=np.arange(128,dtype=np.float32)[None]
        proof=self.m.embedding_metadata(expected,'spleen')
        self.assertEqual(proof['shape'],[1,128]);self.assertTrue(proof['finite']);self.assertEqual(len(proof['sha256']),64)
        self.assertNotIn('embedding',proof)
        for embedding,text in ((expected,''),(np.zeros((1,127)),'spleen'),(np.full((1,128),np.nan),'spleen')):
            with self.assertRaises(ValueError):self.m.embedding_metadata(embedding,text)

    def test_private_module_clones_function_globals_and_never_patches_source(self):
        original=types.ModuleType('original_fixture')
        exec('value=1\ndef score():\n    return value',original.__dict__)
        private=self.m._clone_module(original,'private_fixture')
        private.value=2
        self.assertEqual(private.score(),2);self.assertEqual(original.score(),1)
        self.assertIs(private.score.__globals__,private.__dict__)


@unittest.skipIf(torch is None,'local runtime has no Torch; mandatory on server')
class RealTorchBoundaryTests(unittest.TestCase):
    def test_actual_core_accepts_N0_and_retains_dense_semantic_outputs(self):
        from tools import amos_semantic_only_runtime as m
        from v10.model.cross_view_prompt_token_generator_v10 import CrossViewPromptTokenGeneratorV10
        torch.manual_seed(41);torch.set_num_threads(2)
        prompt=CrossViewPromptTokenGeneratorV10(prompt_dim=16,image_embedding_size=4,hidden_dim=4,
            num_tokens=2,work_size=16,view_feature_dim=4,semantic_dim=128,image_feature_mode='none',
            semantic_clip_ckpt='',background_branch=True,dropout=0,refinement_iters=1).eval()
        session=types.SimpleNamespace(variant='semantic_only');m.install_zero_groups(session,prompt)
        groups=torch.empty((0,3,16,16));coords=torch.linspace(-1,1,4)
        with torch.no_grad():
            out=prompt.forward_case_3d_tokens(groups,torch.rand((4,3,32,32)),plane_ids=torch.empty(0,dtype=torch.long),
                slice_coords=torch.empty(0),frame_coords=coords,foreground_mode_id=0,semantic_text='spleen',semantic_embedding=torch.rand(1,128))
        self.assertEqual(out['slice_prompt_embeddings'].shape[1],0)
        for name in ('dense_prompt_embeddings','sparse_prompt_embeddings','prompt_3d_tokens'):
            self.assertTrue(torch.isfinite(out[name]).all(),name)
        self.assertFalse(out['background_active']);self.assertEqual(set(out['prompt_3d_role_ids'].tolist()),{0})
        self.assertEqual(set(out['prompt_3d_type_ids'].tolist()),{0});self.assertFalse(out['aligned_prompt'].any())

    def test_actual_postprocess_retains_raw_seedless_threshold_mask(self):
        from tools import amos_semantic_only_runtime as m
        from v10.infer.infer_amos_sam2_promptgen_v10 import postprocess_v10_logits_from_prompts, dynamic_mask_threshold_from_prompt_logits
        logits=torch.tensor([[[[-2.,0.,.5,2.]]]])
        seed=np.zeros((1,1,4),bool);kw=dict(fallback=.6,minimum=.7,maximum=.75,min_logit_separation=.5)
        native,_=postprocess_v10_logits_from_prompts(logits,seed,seed,**kw)
        self.assertFalse(native.any())
        actual,audit=m.postprocess_dispatch('semantic_only',True,logits,seed,seed,native=postprocess_v10_logits_from_prompts,
            threshold=dynamic_mask_threshold_from_prompt_logits,**kw)
        np.testing.assert_array_equal(actual,(torch.sigmoid(logits)>=.6).squeeze(1).numpy())
        self.assertEqual(audit['source'],'fallback_missing_prompt_logits')


if __name__=='__main__':unittest.main()
