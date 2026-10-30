"""Exact RAM reuse contracts; GPU/native controls remain a separate release gate."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

try:
    from tools import amos_prompt_reuse_runtime as reuse
except ImportError:
    reuse = None

ROOT = Path(__file__).resolve().parents[1]


class ReuseTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(reuse, "add the additive exact reuse runtime")

    def tree(self, batch=2):
        top = np.arange(batch*8, dtype=np.float32).reshape(batch, 2, 2, 2)
        position = np.broadcast_to(np.arange(8, dtype=np.float32).reshape(1,2,2,2), top.shape).copy()
        return dict(vision_features=top, backbone_fpn=[top], vision_pos_enc=[position])

    def test_packing_exact_positions_and_aliases_with_independent_hits(self):
        tree = self.tree()
        packed = reuse.PackedTree.pack(tree)
        self.assertLess(packed.payload_bytes, tree['vision_features'].nbytes + tree['vision_pos_enc'][0].nbytes)
        first, second = packed.unpack(), packed.unpack()
        self.assertIs(first['vision_features'], first['backbone_fpn'][0])
        self.assertEqual(first['vision_features'].dtype, tree['vision_features'].dtype)
        self.assertEqual(first['vision_features'].strides, tree['vision_features'].strides)
        self.assertTrue(reuse.trees_equal(tree, first))
        first['vision_features'].fill(99)
        self.assertTrue(reuse.trees_equal(tree, second))
        self.assertEqual(packed.digest, reuse.PackedTree.pack(second).digest)

    def test_nonidentical_position_samples_are_not_compressed(self):
        tree = self.tree(); tree['vision_pos_enc'][0][1] += 1
        packed = reuse.PackedTree.pack(tree)
        self.assertEqual(packed.payload_bytes, 128)
        self.assertTrue(reuse.trees_equal(tree, packed.unpack()))

    def test_budget_eviction_and_template_pool_are_exact_and_bounded(self):
        memo = reuse.BoundedMemo(150)
        first = self.tree(); second = self.tree(); second['vision_features'] += 3
        self.assertTrue(memo.put('a', reuse.PackedTree.pack(first)))
        self.assertTrue(memo.put('b', reuse.PackedTree.pack(second)))
        self.assertIsNone(memo.get('a'))
        self.assertTrue(reuse.trees_equal(second, memo.get('b').unpack()))
        self.assertLessEqual(memo.snapshot()['retained_bytes'], 150)
        self.assertGreater(memo.snapshot()['evictions'], 0)
        self.assertFalse(memo.put('huge', reuse.PackedTree.pack(np.zeros(100, np.float32))))
        memo.clear(); self.assertEqual(memo.snapshot()['retained_bytes'], 0)
        self.assertEqual(len(memo.pool), 0)

    def make_engine(self):
        calls=[]; state={'guard':hashlib.sha256(b'original').hexdigest(), 'context':dict(amp=False, dtype='float32', device='cpu', grad=False, training=False)}
        def native(images):
            calls.append(images.copy())
            top=images*2
            return dict(vision_features=top,backbone_fpn=[top],vision_pos_enc=[np.ones_like(top)])
        engine=reuse.EncoderReuse(native, lambda:state['guard'], lambda:state['context'], fingerprint=hashlib.sha256(b'frozen').hexdigest(), max_bytes=4096)
        return engine, state, calls

    def test_exact_input_content_batch_and_amp_are_in_key(self):
        engine,state,calls=self.make_engine(); engine.begin('baseline',False,['amos','a',1])
        images=np.arange(16,dtype=np.float32).reshape(2,2,2,2)
        engine(images); engine(images.copy())
        changed=images.copy(); changed[0,0,0,0]+=1; engine(changed)
        engine(images[:1]); state['context']['amp']=True; engine(images)
        self.assertEqual(len(calls),4)
        self.assertEqual(engine.proof()['counters']['hits'],1)
        self.assertEqual(engine.proof()['counters']['misses'],4)
        self.assertEqual(len({e['input_sha256'] for e in engine.proof()['events']}),4)

    def test_parameter_mutation_and_grad_or_training_refused(self):
        engine,state,_=self.make_engine(); engine.begin('baseline',False,['amos','a',1])
        state['guard']='mutated'
        with self.assertRaisesRegex(ValueError,'parameter'):engine(np.ones((1,2)))
        state['guard']=hashlib.sha256(b'original').hexdigest(); state['context']['grad']=True
        with self.assertRaisesRegex(ValueError,'inference'):engine(np.ones((1,2)))
        state['context']['grad']=False; state['context']['training']=True
        with self.assertRaisesRegex(ValueError,'inference'):engine(np.ones((1,2)))

    def test_native_control_bypasses_real_baseline_hit_gets_actual_native_twin(self):
        engine,state,calls=self.make_engine(); images=np.ones((2,2,2,2),np.float32)
        engine.begin('native_control',True,['amos','a',1]); native=engine(images); engine(images)
        native_proof=engine.proof()
        self.assertEqual(native_proof['counters']['hits'],0)
        self.assertEqual(native_proof['counters']['bypasses'],2)
        engine.begin('baseline',True,['amos','a',1]); result=engine(images)
        proof=engine.proof()
        self.assertTrue(reuse.trees_equal(native,result))
        self.assertEqual(len(calls),3)  # Two native bypasses and one real native twin.
        self.assertEqual(proof['counters']['hits'],1)
        self.assertEqual(proof['counters']['native_twins'],1)
        self.assertEqual(proof['counters']['native_control_matches'],1)
        reuse.validate_reuse_proof(proof,'baseline',True)
        forged=copy.deepcopy(proof); forged['counters']['native_twins']=0
        with self.assertRaises(ValueError):reuse.validate_reuse_proof(forged,'baseline',True)

    def test_actual_native_twin_mismatch_refused(self):
        engine,state,calls=self.make_engine(); images=np.ones((2,2,2,2),np.float32)
        engine.begin('native_control',True,['amos','a',1]); engine(images)
        old=engine.native
        def changed(x):
            value=old(x);value['vision_features']+=1;return value
        engine.native=changed;engine.begin('baseline',True,['amos','a',1])
        with self.assertRaisesRegex(ValueError,'twin'):engine(images)

    def test_all_call_io_chain_matches_native_even_after_bounded_events(self):
        engine,state,calls=self.make_engine(); images=np.ones((2,2,2,2),np.float32)
        engine.begin('native_control',True,['amos','a',1])
        for i in range(reuse.EVENT_LIMIT+2):engine(images)
        native=engine.proof()
        engine.begin('baseline',True,['amos','a',1])
        for i in range(reuse.EVENT_LIMIT+2):engine(images)
        baseline=engine.proof()
        self.assertEqual(len(baseline['events']),reuse.EVENT_LIMIT)
        self.assertEqual(native['io_chain_sha256'],baseline['io_chain_sha256'])
        self.assertEqual(native['event_count'],baseline['event_count'])
        engine(images)
        self.assertNotEqual(native['io_chain_sha256'],engine.proof()['io_chain_sha256'])

    def test_invalid_audit_counters_and_eval_context_are_rejected(self):
        engine,state,calls=self.make_engine();images=np.ones((2,2,2,2),np.float32)
        engine.begin('baseline',False,['amos','a',1]);engine(images)
        proof=engine.proof();proof['counters']['hits']=-1;proof['counters']['misses']=2
        with self.assertRaises(ValueError):reuse.validate_reuse_proof(proof,'baseline')
        proof=engine.proof();proof['events'][0]['context']['grad']=True
        with self.assertRaises(ValueError):reuse.validate_reuse_proof(proof,'baseline')

    def test_first_control_hit_twin_evidence_is_retained_after_event_limit(self):
        engine,state,calls=self.make_engine();images=np.ones((2,2,2,2),np.float32)
        engine.memo.max_bytes=1024*1024
        engine.begin('native_control',True,['amos','a',1]);engine(images)
        engine.begin('fg_box_bg_point',True,['amos','a',1])
        for index in range(reuse.EVENT_LIMIT+1):engine(images+index+1)
        engine(images)
        proof=engine.proof()
        self.assertGreater(proof['counters']['hits'],0)
        self.assertGreater(proof['hit_twin_count'],0)
        self.assertTrue(proof['hit_twin_events'][0]['actual_native_twin'])
        reuse.validate_reuse_proof(proof,'fg_box_bg_point',True)

    def test_native_remainder_batch_three_accepted_but_configured_cap_enforced(self):
        engine,state,calls=self.make_engine();images=np.ones((3,2,2,2),np.float32)
        engine.begin('baseline',False,['amos','a',1]);engine(images)
        proof=engine.proof();self.assertEqual(proof['events'][0]['actual_batch'],3)
        reuse.validate_reuse_proof(proof,'baseline')
        proof['actual_frame_batch']=2
        with self.assertRaises(ValueError):reuse.validate_reuse_proof(proof,'baseline')

    def test_complete_bounded_event_chains_are_recomputed(self):
        engine,state,calls=self.make_engine();engine.begin('baseline',False,['amos','a',1]);engine(np.ones((2,2,2,2),np.float32))
        proof=engine.proof();proof['event_chain_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'chain'):reuse.validate_reuse_proof(proof,'baseline')

    def test_native_twin_proof_contains_actual_comparison_digests(self):
        engine,state,calls=self.make_engine();images=np.ones((2,2,2,2),np.float32)
        engine.begin('native_control',True,['amos','a',1]);engine(images)
        engine.begin('baseline',True,['amos','a',1]);engine(images)
        twin=engine.proof()['hit_twin_events'][0]
        self.assertEqual(twin['native_twin_input_sha256'],twin['input_sha256'])
        self.assertEqual(twin['native_twin_output_sha256'],twin['output_sha256'])
        self.assertTrue(twin['exact_tree_equal'])

    def test_counter_arithmetic_and_hit_twin_digests_are_validated(self):
        engine,state,calls=self.make_engine();images=np.ones((2,2,2,2),np.float32)
        engine.begin('native_control',True,['amos','a',1]);engine(images)
        engine.begin('baseline',True,['amos','a',1]);engine(images)
        proof=engine.proof();proof['counters']['native_calls']+=1
        with self.assertRaises(ValueError):reuse.validate_reuse_proof(proof,'baseline',True)
        proof=engine.proof();proof['hit_twin_events'][0]['native_twin_output_sha256']='0'*64
        with self.assertRaises(ValueError):reuse.validate_reuse_proof(proof,'baseline',True)

    def test_overflow_prefix_and_remaining_count_are_bound(self):
        engine,state,calls=self.make_engine();images=np.ones((2,2,2,2),np.float32)
        engine.begin('baseline',False,['amos','a',1])
        for i in range(reuse.EVENT_LIMIT+2):engine(images)
        proof=engine.proof();self.assertEqual(proof['remaining_event_count'],2)
        reuse.validate_reuse_proof(proof,'baseline')
        proof['event_prefix_chain_sha256']='0'*64
        with self.assertRaises(ValueError):reuse.validate_reuse_proof(proof,'baseline')

    def test_real_stage_frame_chunks_bound_the_reuse_events(self):
        engine,state,calls=self.make_engine();engine.begin('baseline',False,['amos','a',1])
        for size in (4,4,3,4,1):engine(np.ones((size,2,2,2),np.float32))
        proof=engine.proof();reuse.validate_stage_coverage(proof,[{'frames':11},{'frames':5}],4)
        with self.assertRaises(ValueError):reuse.validate_stage_coverage(proof,[{'frames':12},{'frames':5}],4)
        with self.assertRaises(ValueError):reuse.validate_stage_coverage(proof,[{'frames':11}],4)

    def test_finish_runs_native_and_clears_ram_even_if_initial_guard_refuses(self):
        for failure in (False,True):
            engine,state,calls=self.make_engine();engine.begin('baseline',False,['amos','a',1]);engine(np.ones((2,2,2,2),np.float32))
            class Prep:
                cleared=False
                def snapshot(self):return {'retained_bytes':1}
                def clear(self):self.cleared=True
            prep=Prep();native_calls=[]
            def file_guard():
                if failure:raise ValueError('source changed')
            session=SimpleNamespace(reuse_engine=engine,reuse_preparation=prep,reuse_binding={},reuse_file_guard=file_guard)
            def native_finish():native_calls.append(True);return {'successful_native_completion':True}
            if failure:
                with self.assertRaisesRegex(ValueError,'source changed'):reuse.finish_session(session,native_finish)
                self.assertFalse(native_calls)
            else:
                result=reuse.finish_session(session,native_finish);self.assertTrue(result['successful_native_completion']);self.assertEqual(len(native_calls),1)
            self.assertEqual(engine.memo.retained_bytes,0);self.assertTrue(prep.cleared)

    def test_source_bindings_checked_before_old_initialize(self):
        self.assertEqual(reuse.RESTORE_RUNTIME_SHA256,hashlib.sha256((ROOT/'tools/amos_prompt_ablation_restore_runtime.py').read_bytes()).hexdigest())
        self.assertEqual(reuse.NATIVE_RUNTIME_SHA256,hashlib.sha256((ROOT/'tools/amos_prompt_ablation_runtime.py').read_bytes()).hexdigest())
        with patch.object(reuse,'RESTORE_RUNTIME_SHA256','0'*64),patch.object(reuse._restore,'initialize') as init:
            with self.assertRaisesRegex(ValueError,'SHA'):reuse.initialize({},dict(root=str(ROOT)))
        init.assert_not_called()

    def test_preparation_memo_preserves_real_class_extraction_and_ct_callback(self):
        with tempfile.TemporaryDirectory() as directory:
            label=Path(directory)/'label.npy';ct=Path(directory)/'ct.npy'
            np.save(label,np.arange(8,dtype=np.float32).reshape(2,2,2));np.save(ct,np.ones((2,2,2),np.float32))
            class Inner:
                label_paths=[str(label)];image_paths=[str(ct)]
                def _load_array(self,p): self.label_reads+=1;return np.load(p)
                def _load_case_image_cache(self,p):self.ct_reads+=1;return np.load(p)
                label_reads=ct_reads=0
                def target(self,c):return (self._load_array(str(label))==c).astype(np.uint8)
            inner=Inner();session=SimpleNamespace(namespace={'inner':inner},job={'case_id':'a'})
            prep=reuse.install_preparation(session)
            self.assertEqual(inner.target(1).sum(),1);self.assertEqual(inner.target(2).sum(),1)
            # The outer native read-only callback still executes on each access.
            audit=[]
            for unused in range(2):
                value=inner._load_case_image_cache(str(ct));audit.append({'mode':'read_only_hit','shape':list(value.shape)})
            self.assertEqual((inner.label_reads,inner.ct_reads,len(audit)),(1,1,2))
            self.assertEqual(prep.snapshot()['label_hits'],1)
            self.assertEqual(prep.snapshot()['ct_hits'],1)
            np.save(label,np.zeros((2,2,2),np.float32))
            with self.assertRaisesRegex(ValueError,'source changed'):inner.target(1)
            prep.clear();self.assertEqual(prep.snapshot()['retained_bytes'],0)

    def test_preparation_only_memoizes_signed_current_case_label(self):
        with tempfile.TemporaryDirectory() as directory:
            one=Path(directory)/'one.npy';other=Path(directory)/'other.npy'
            np.save(one,np.ones((2,2,2),np.float32));np.save(other,np.zeros((2,2,2),np.float32))
            class Inner:
                label_paths=[str(one),str(other)]
                def _load_array(self,p):return np.load(p)
                def _load_case_image_cache(self,p):return np.load(p)
            inner=Inner();session=SimpleNamespace(namespace={'inner':inner,'tasks':[{'label_path':str(one)}]},job={'case_id':'a'})
            prep=reuse.install_preparation(session);inner._load_array(str(other));inner._load_array(str(one))
            self.assertEqual(len(prep.arrays),1)
            self.assertEqual(prep.labels,{str(one.resolve())})

    def test_record_encoder_program_identity_cannot_disagree_with_native_record(self):
        engine,state,calls=self.make_engine();engine.begin('baseline',False,['amos','a',1]);engine(np.ones((2,2,2,2),np.float32))
        proof=engine.proof();binding=reuse._source_binding({'root':str(ROOT)});binding['native_program_sha256']='a'*64
        proof['source_binding']=binding;proof['fingerprint']=reuse._sha(reuse._json(binding))
        record={'reuse_execution':proof,'stages':[{'frames':2}],'native_source_sha256':'c'*64,
            'runtime_input_audit':{'exact_reuse':{'source_binding':binding,'fingerprint':proof['fingerprint']}}}
        with patch.object(reuse._restore,'validate_record',return_value=True):
            with self.assertRaisesRegex(ValueError,'program'):reuse.validate_record(record,{'source_name':'amos','case_id':'a','class_id':1},'baseline',4)

    def test_hit_twin_event_must_equal_the_actual_retained_hit(self):
        engine,state,calls=self.make_engine();images=np.ones((2,2,2,2),np.float32)
        engine.begin('native_control',True,['amos','a',1]);engine(images)
        engine.begin('baseline',True,['amos','a',1]);engine(images)
        proof=engine.proof();proof['hit_twin_events'][0]['native_control_match']=False
        with self.assertRaises(ValueError):reuse.validate_reuse_proof(proof,'baseline',True)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'real Torch tests run on server')
class TorchTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(reuse)

    def test_bfloat16_noncontiguous_exact_layout_and_aliases(self):
        import torch
        # GPU controls run only in the controller's explicitly leased workers.
        for device in ['cpu']:
            top=torch.arange(32,device=device,dtype=torch.bfloat16).reshape(2,2,2,4).transpose(2,3)
            pos=torch.ones((2,2,4,2),device=device,dtype=torch.bfloat16)
            value=dict(vision_features=top,backbone_fpn=[top],vision_pos_enc=[pos])
            packed=reuse.PackedTree.pack(value); restored=packed.unpack()
            self.assertTrue(reuse.trees_equal(value,restored))
            self.assertEqual(restored['vision_features'].stride(),top.stride())
            self.assertEqual(restored['vision_features'].device,top.device)
            self.assertEqual(restored['vision_features'].dtype,top.dtype)
            self.assertIs(restored['vision_features'],restored['backbone_fpn'][0])
            restored['vision_features'].zero_();self.assertTrue(torch.equal(packed.unpack()['vision_features'],top))

    def test_encoder_parameter_and_buffer_guard_includes_fpn_projection(self):
        import torch
        sam=SimpleNamespace(image_encoder=torch.nn.BatchNorm2d(2),sam_mask_decoder=SimpleNamespace(conv_s0=torch.nn.Conv2d(2,2,1),conv_s1=torch.nn.Conv2d(2,2,1)))
        original=reuse.encoder_parameter_guard(sam)
        with torch.no_grad():sam.sam_mask_decoder.conv_s0.weight.add_(1)
        self.assertNotEqual(reuse.encoder_parameter_guard(sam),original)
        original=reuse.encoder_parameter_guard(sam)
        sam.image_encoder.running_mean.add_(1)
        self.assertNotEqual(reuse.encoder_parameter_guard(sam),original)

    def test_actual_torch_engine_amp_control_twins_clone_and_pool(self):
        import torch
        class Sam(torch.nn.Module):
            def __init__(self):
                super().__init__();self.image_encoder=torch.nn.Conv2d(3,2,1)
                self.sam_mask_decoder=torch.nn.Module();self.sam_mask_decoder.conv_s0=torch.nn.Conv2d(2,2,1)
                self.sam_mask_decoder.conv_s1=torch.nn.Conv2d(2,2,1)
            def forward_image(self,x):
                top=self.sam_mask_decoder.conv_s0(self.image_encoder(x))
                pos=torch.ones_like(top)
                return dict(vision_features=top,backbone_fpn=[top],vision_pos_enc=[pos])
        for device in ['cpu']:
            sam=Sam().to(device).eval();images=torch.arange(96,device=device,dtype=torch.float32).reshape(2,3,4,4)
            engine=reuse.EncoderReuse(sam.forward_image,lambda:reuse.encoder_parameter_guard(sam),lambda:reuse._context(sam),
                fingerprint=hashlib.sha256(b'actual_torch').hexdigest(),max_bytes=4096)
            with torch.no_grad(),torch.autocast(device_type=device,dtype=torch.bfloat16):
                engine.begin('native_control',True,['amos','a',1]);native=engine(images);engine(images)
                self.assertEqual(engine.proof()['counters']['hits'],0)
                engine.begin('baseline',True,['amos','a',1]);hit=engine(images)
                self.assertTrue(reuse.trees_equal(native,hit));self.assertEqual(hit['vision_features'].dtype,torch.bfloat16)
                self.assertEqual(hit['vision_features'].device,images.device)
                self.assertIs(hit['vision_features'],hit['backbone_fpn'][0]);hit['vision_features'].zero_()
                self.assertTrue(torch.equal(native['vision_features'],engine(images)['vision_features']))
                reuse.validate_reuse_proof(engine.proof(),'baseline',True)
                engine.begin('fg_box_bg_point',False,['amos','a',1]);engine(images+1)
                self.assertEqual(len(engine.memo.pool),3) # Two distinct FPN payloads, one shared position sample.
                engine(images[:1]);self.assertEqual(engine.proof()['counters']['misses'],2)
                engine(images.to(dtype=torch.bfloat16));self.assertEqual(engine.proof()['counters']['misses'],3)
            engine.memo.max_bytes=512
            with torch.no_grad():
                engine(images);self.assertEqual(engine.proof()['counters']['misses'],4) # AMP is part of exact domain.
                self.assertGreater(engine.memo.evictions,0)
            with torch.no_grad():sam.sam_mask_decoder.conv_s1.weight.add_(1)
            with torch.no_grad(),self.assertRaisesRegex(ValueError,'parameter'):engine(images)


if __name__=='__main__':unittest.main()
