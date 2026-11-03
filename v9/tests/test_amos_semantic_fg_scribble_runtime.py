"""Real NumPy boundary tests; no CUDA predictions are simulated here."""
import ast
import copy
import hashlib
import importlib
from pathlib import Path
import types
import unittest
from unittest.mock import patch
import numpy as np
from tests.test_amos_prompt_ablation_prompts import fixture

ROOT=Path(__file__).resolve().parents[1]
try:
    import torch
except ImportError:
    torch=None


class ForegroundRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((ROOT/'tools/amos_semantic_fg_scribble_runtime.py').is_file(),'FG-only runtime implementation required')
        self.m=importlib.import_module('tools.amos_semantic_fg_scribble_runtime')

    def test_private_native_keeps_original_roi_and_postprocessing(self):
        old=importlib.import_module('tools.amos_prompt_ablation_runtime');before=old.Session.modes
        ns=self.m.native_engine();s=ns['Session']({},{});s.variant='semantic_fg_scribble'
        self.assertIs(old.Session.modes,before);self.assertIsNot(ns['Session'],old.Session)
        self.assertEqual(s.modes()['background_mode'],'none')
        s.variant='baseline';self.assertEqual(s.modes()['background_mode'],'scribble')
        source=ast.unparse(self.m.transform_native((ROOT/self.m.NATIVE).read_bytes())[0])
        self.assertNotIn('full_ct_roi',source);self.assertNotIn('empty_converted',source)
        self.assertNotIn('semantic_component_bypass',source)
        self.assertIn('value = native_post(*args, **kwargs)',source)
        self.assertIn('roi_indices(int(target.shape[0])',source)
        self.assertTrue(ns['TRANSFORM_PROOF']['reverse_ast_equal'])
        self.assertIs(ns['evaluate'].__globals__,ns)

    def test_signed_source_and_exact_anchors_fail_closed(self):
        m=self.m
        with self.assertRaisesRegex(ValueError,'SHA'):m.native_engine(source=b'pass')
        payload=(ROOT/m.NATIVE).read_bytes().replace(b'background_mode="point"',b'background_mode="scribble"')
        with patch.object(m,'NATIVE_SHA256',hashlib.sha256(payload).hexdigest()):
            with self.assertRaisesRegex(ValueError,'anchor'):m.transform_native(payload)

    def test_authenticated_s21_and_local_profile_cannot_infer(self):
        m=self.m
        self.assertEqual(m.classify_dependencies(dict(m.LOCAL_DEPENDENCIES)),'local_metadata_only')
        self.assertEqual(m.classify_dependencies(dict(m.S21_DEPENDENCIES)),'production_s21')
        mixed=dict(m.LOCAL_DEPENDENCIES);mixed[m.MEMORY_SOURCE]=m.S21_DEPENDENCIES[m.MEMORY_SOURCE]
        with self.assertRaises(ValueError):m.classify_dependencies(mixed)
        p=dict(sources=dict(m.S21_DEPENDENCIES),native_source_sha256=m.S21_DEPENDENCIES[m.NATIVE_PROGRAM])
        self.assertTrue(m.validate_s21_protocol(p,{'sha256':m.S21_PROTOCOL_SHA256},m.S21_PROTOCOL_SHA256))
        with self.assertRaises(ValueError):m.validate_s21_protocol(p,{'sha256':'0'*64},m.S21_PROTOCOL_SHA256)
        with patch.object(m,'dependency_profile',return_value=('local_metadata_only',dict(m.LOCAL_DEPENDENCIES))),patch.object(m,'reuse_engine') as reuse:
            with self.assertRaisesRegex(ValueError,'production'):m.initialize({}, {})
            reuse.assert_not_called()

    def test_target_consumption_and_tokens_reject_any_bg_or_box(self):
        from tools.amos_semantic_fg_scribble_prompts import build_variant,variant_modes
        p,g=fixture();bundle,_=build_variant(p,g,'semantic_fg_scribble',13);modes=variant_modes('semantic_fg_scribble')
        def consume(x):return self.m.validate_consumption(x,bundle['plane_ids'],0,modes,bundle['groups'],slice_coords=bundle['slice_coords'],expected_slice_coords=bundle['slice_coords'])
        event=consume(bundle['groups']);self.assertTrue(all(c[0]>0 and c[1:]==[0,0] for c in event['channel_nonzero']))
        for channel in (1,2):
            bad=bundle['groups'].copy();bad[0,channel,0,0]=1
            with self.assertRaises(ValueError):consume(bad)
        out=dict(prompt_3d_role_ids=np.array([0,0]),prompt_3d_type_ids=np.array([0,0]),foreground_prompt_mode='scribble',background_prompt_mode='none',foreground_channel=0,background_active=False)
        self.assertEqual(self.m.validate_output_types(out,modes)['role_ids'],[0])
        for field,value in (('background_active',True),('background_prompt_mode','scribble'),('prompt_3d_type_ids',np.array([0,3])),('prompt_3d_role_ids',np.array([0,1]))):
            bad=dict(out);bad[field]=value
            with self.assertRaises(ValueError):self.m.validate_output_types(bad,modes)

    def test_real_none_embedding_is_retained_in_native_source(self):
        source=(ROOT/'v10/model/cross_view_prompt_token_generator_v10.py').read_text(encoding='utf-8')
        self.assertIn('"none": 2',source)
        self.assertIn('self.BACKGROUND_MODE_IDS[bg_mode]',source)
        self.assertNotIn('install_zero_groups',Path(self.m.__file__).read_text(encoding='utf-8'))

    def test_embedding_metadata_requires_actual_finite_sat128(self):
        self.assertEqual(self.m.embedding_metadata(np.ones((1,128)),'spleen')['shape'],[1,128])
        for bad in (np.zeros((1,64)),np.full((1,128),np.nan)):
            with self.assertRaises(ValueError):self.m.embedding_metadata(bad,'spleen')

    def target_record(self):
        from tests.test_amos_prompt_ablation_runtime import FrozenProtocolTests
        from tools.amos_semantic_fg_scribble_prompts import build_variant,roi_indices,variant_modes
        r,t=FrozenProtocolTests().record();p,g=fixture();bundle,audit=build_variant(p,g,'semantic_fg_scribble',13)
        consume=self.m.validate_consumption(bundle['groups'],bundle['plane_ids'],0,variant_modes('semantic_fg_scribble'),bundle['groups'],slice_coords=bundle['slice_coords'],expected_slice_coords=bundle['slice_coords'])
        types_=dict(type_ids=[0],role_ids=[0],type_counts={'0':2})
        call=dict(consume,**types_,semantic_text='organ',aligned_channel_nonzero=[6,0,0],background_none_mode_id=2,background_none_embedding_sha256='a'*64)
        stage=r['stages'][0];stage.update(bundle=consume,prompt_calls=[call],mixed_calls=[dict(foreground='scribble',background='none',foreground_channel=0)],global_pool_calls=[dict(types_,background_active=False,background_mode='none')],decoder_calls=[types_],adapter_calls=[types_]*4)
        stage['threshold_audit']['foreground_only_native_postprocess']=dict(foreground_seed_count=6,background_seed_count=0,native_component_filter=True,native_scorer_sha256=self.m.S21_DEPENDENCIES[self.m.SCORER_SOURCE])
        audit.update(group_builds=[{}],actual_consumption_calls=1);r.update(prompt=audit,variant='semantic_fg_scribble')
        initial=roi_indices(12,audit['raw_foreground_z_support'],1,0,1).tolist();r['roi']['initial']=initial;r['roi']['builds'][0].update(indices=initial,raw_support=audit['raw_foreground_z_support'])
        r['metric_execution']['events'][0].update(variant='semantic_fg_scribble',full_mask_shape=[12,7,9])
        return r,t

    def test_target_record_requires_real_native_foreground_filter_and_no_bg(self):
        r,t=self.target_record();self.assertTrue(self.m.validate_native_record(r,t,'semantic_fg_scribble',4))
        for corrupt in ('raw_bg','raster_box','role','pool','none_id','filter','metric','memory','attention'):
            bad=copy.deepcopy(r);s=bad['stages'][0]
            if corrupt=='raw_bg':bad['prompt']['planes']['coronal']['raw_channel_nonzero'][1]=1
            if corrupt=='raster_box':bad['prompt']['planes']['coronal']['raster_channel_nonzero'][2]=1
            if corrupt=='role':s['adapter_calls'][0]['role_ids']=[0,1]
            if corrupt=='pool':s['global_pool_calls'][0]['background_active']=True
            if corrupt=='none_id':s['prompt_calls'][0]['background_none_mode_id']=0
            if corrupt=='filter':s['threshold_audit']['foreground_only_native_postprocess']['native_component_filter']=False
            if corrupt=='metric':bad['row']['dice']=float('nan')
            if corrupt=='memory':s['memory_calls']['memory_encoder']=0
            if corrupt=='attention':s['memory_calls']['memory_attention']=0
            with self.subTest(corrupt=corrupt),self.assertRaises(ValueError):self.m.validate_native_record(bad,t,'semantic_fg_scribble',4)

    def test_target_context_chunk_may_have_zero_aligned_fg(self):
        r,t=self.target_record();s=r['stages'][0];empty=copy.deepcopy(s['prompt_calls'][0]);empty['aligned_channel_nonzero']=[0,0,0]
        s['prompt_calls'].append(empty);s['mixed_calls'].append(dict(s['mixed_calls'][0]));r['prompt']['actual_consumption_calls']=2
        self.assertTrue(self.m.validate_native_record(r,t,'semantic_fg_scribble',4))
        s['prompt_calls'][0]['aligned_channel_nonzero']=[0,0,0]
        with self.assertRaises(ValueError):self.m.validate_native_record(r,t,'semantic_fg_scribble',4)

    def test_private_control_trace_namespaces_keep_original_and_production_limit64(self):
        import inspect
        self.assertIn('control',inspect.signature(self.m.reuse_engine).parameters,'private control reuse namespace is required')
        from tools import amos_prompt_reuse_owner_runtime as old
        original=old.engine();before=original['EVENT_LIMIT']
        control=self.m.reuse_engine(control=True);production=self.m.reuse_engine(control=False)
        self.assertIsNot(control,production);self.assertIsNot(control,original)
        self.assertEqual((control['EVENT_LIMIT'],production['EVENT_LIMIT'],original['EVENT_LIMIT']),(4096,64,before))
        self.assertEqual(control['FEATURE_BUDGET_BYTES'],2*1024**3)
        self.assertIs(control['validate_reuse_proof'].__globals__,control)

    def trace_engine(self,control):
        import inspect
        self.assertIn('control',inspect.signature(self.m.reuse_engine).parameters,'private control reuse namespace is required')
        ns=self.m.reuse_engine(control=control);calls=[]
        def native(images):
            calls.append(images.copy());top=images*2
            return dict(vision_features=top,backbone_fpn=[top],vision_pos_enc=[np.ones_like(top)])
        engine=ns['EncoderReuse'](native,lambda:'a'*64,lambda:dict(grad=False,training=False),fingerprint='b'*64,max_bytes=4096)
        return engine,ns,calls

    def test_real_tiny_control_encoder_over64_calls_retains_and_recomputes_complete_chains(self):
        engine,ns,calls=self.trace_engine(True);key=['amos','unit_trace',118];image=np.ones((1,2),np.float32)
        engine.begin('native_control',True,key);engine(image)
        engine.begin('semantic_fg_scribble',True,key)
        for _ in range(70):engine(image.copy())
        proof=engine.proof();self.assertEqual(proof['event_count'],70);self.assertEqual(len(proof['events']),70)
        self.assertEqual((proof['remaining_event_count'],proof['event_limit']),(0,4096))
        chain=io='0'*64
        for event in proof['events']:
            chain=hashlib.sha256(bytes.fromhex(chain)+ns['_json'](event)).hexdigest()
            io=hashlib.sha256(bytes.fromhex(io)+ns['_json']([event['input_sha256'],event['output_sha256']])).hexdigest()
        self.assertEqual(chain,proof['event_chain_sha256']);self.assertEqual(io,proof['io_chain_sha256'])
        self.assertTrue(self.m.validate_reuse_proof(proof,'semantic_fg_scribble',True))
        truncated=copy.deepcopy(proof);truncated['events']=truncated['events'][:64];truncated['remaining_event_count']=6
        with self.assertRaises(ValueError):self.m.validate_reuse_proof(truncated,'semantic_fg_scribble',True)
        changed=copy.deepcopy(proof);changed['event_limit']=64
        with self.assertRaises(ValueError):self.m.validate_reuse_proof(changed,'semantic_fg_scribble',True)
        with self.assertRaises(ValueError):self.m.validate_reuse_proof(proof,'semantic_fg_scribble',False)

    def test_control_trace_bound_exhaustion_rejected_before_encoder_execution(self):
        engine,ns,calls=self.trace_engine(True);engine.begin('native_control',True,['amos','unit_trace',118])
        image=np.ones((1,2),np.float32)
        for _ in range(4096):engine(image)
        self.assertEqual(len(engine.proof()['events']),4096);before=len(calls)
        with self.assertRaisesRegex(ValueError,'exhaust'):engine(image)
        self.assertEqual(len(calls),before);self.assertEqual(engine.event_count,4096)
        bad=engine.proof();bad['event_count']=4097
        with self.assertRaises(ValueError):self.m.validate_reuse_proof(bad,'native_control',True)

    def test_production_trace_retains_original64_prefix_and_control_flag_is_bound(self):
        engine,ns,calls=self.trace_engine(False);engine.begin('semantic_fg_scribble',False,['amos','unit_trace',118]);image=np.ones((1,2),np.float32)
        for _ in range(70):engine(image)
        p=engine.proof();self.assertEqual((p['event_limit'],len(p['events']),p['remaining_event_count']),(64,64,6))
        self.assertTrue(self.m.validate_reuse_proof(p,'semantic_fg_scribble',False))
        changed=copy.deepcopy(p);changed['event_limit']=4096
        with self.assertRaises(ValueError):self.m.validate_reuse_proof(changed,'semantic_fg_scribble',False)


@unittest.skipIf(torch is None,'local runtime has no Torch; mandatory on server')
class RealTorchForegroundTests(unittest.TestCase):
    def test_native_core_none_mode_retains_learned_none_embedding(self):
        from v10.model.cross_view_prompt_token_generator_v10 import CrossViewPromptTokenGeneratorV10
        torch.manual_seed(41);torch.set_num_threads(2)
        p=CrossViewPromptTokenGeneratorV10(prompt_dim=16,image_embedding_size=4,hidden_dim=4,num_tokens=2,work_size=16,view_feature_dim=4,semantic_dim=128,image_feature_mode='none',semantic_clip_ckpt='',background_branch=True,dropout=0,refinement_iters=1).eval()
        p._v10_forced_background_mode='none'
        groups=torch.zeros((2,3,16,16));groups[:,0,3:12,6]=1
        video=torch.rand((4,3,32,32));coords=torch.linspace(-1,1,4);semantic=torch.rand(1,128)
        kwargs=dict(plane_ids=torch.tensor([1,0]),slice_coords=torch.tensor([0.,0.]),frame_coords=coords,foreground_mode_id=0,semantic_text='spleen',semantic_embedding=semantic)
        projection_inputs=[]
        hook=p.mode_dense_projection.register_forward_pre_hook(lambda _module,args:projection_inputs.append(args[0].detach().clone()))
        try:
            with torch.no_grad():
                self.assertEqual(torch.count_nonzero(p.mode_dense_projection.weight).item(),0)
                self.assertEqual(torch.count_nonzero(p.mode_dense_projection.bias).item(),0)
                zero_out=p.forward_case_3d_tokens(groups,video,**kwargs)
                p.background_mode_embedding.weight[2].add_(1)
                zero_changed=p.forward_case_3d_tokens(groups,video,**kwargs)
                # Native initialization suppresses the dense mode bias. Prove
                # the learned none path with an explicit synthetic projection.
                p.background_mode_embedding.weight[2].sub_(1)
                p.mode_dense_projection.weight.copy_(torch.eye(p.prompt_dim))
                out=p.forward_case_3d_tokens(groups,video,**kwargs)
                p.background_mode_embedding.weight[2].add_(1)
                changed=p.forward_case_3d_tokens(groups,video,**kwargs)
        finally:hook.remove()
        self.assertEqual(len(projection_inputs),4)
        for before,after in ((projection_inputs[0],projection_inputs[1]),(projection_inputs[2],projection_inputs[3])):
            torch.testing.assert_close(after-before,torch.ones_like(before))
        torch.testing.assert_close(zero_changed['dense_prompt_embeddings'],zero_out['dense_prompt_embeddings'])
        torch.testing.assert_close(zero_changed['sparse_prompt_embeddings']-zero_out['sparse_prompt_embeddings'],torch.ones_like(zero_out['sparse_prompt_embeddings']))
        self.assertEqual((out['background_prompt_mode'],out['background_active']),('none',False))
        self.assertEqual(set(out['prompt_3d_role_ids'].tolist()),{0});self.assertEqual(set(out['prompt_3d_type_ids'].tolist()),{0})
        self.assertFalse(out['aligned_prompt'][:,1:].any());self.assertTrue(out['aligned_prompt'][:,0].any())
        torch.testing.assert_close(changed['sparse_prompt_embeddings']-out['sparse_prompt_embeddings'],torch.ones_like(out['sparse_prompt_embeddings']))
        torch.testing.assert_close(changed['dense_prompt_embeddings']-out['dense_prompt_embeddings'],torch.ones_like(out['dense_prompt_embeddings']))
        self.assertFalse(torch.equal(out['dense_prompt_embeddings'],changed['dense_prompt_embeddings']))
