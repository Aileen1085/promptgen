"""Actual semantic consumption tests; model imports stay on the GPU worker."""
import copy
import hashlib
import importlib
import importlib.util
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


def catalogue():
    return [dict(dataset='totalseg', global_class_id=1, semantic_text='ignored'),
            dict(dataset='bone', global_class_id=10003, local_class_id=3,
                 label_value=3, class_name='rib_right_4',
                 semantic_base_text='rib_right_4', morphology_group='compact',
                 parent_semantic_key=None,
                 semantic_text='rib_right_4; compact target morphology'),
            dict(dataset='bone', global_class_id=10004, local_class_id=4,
                 label_value=4, class_name='vertebrae_L1（仅椎体）',
                 semantic_base_text='vertebrae_L1 (vertebral body only)',
                 morphology_group='compact', parent_semantic_key=None,
                 semantic_text='vertebrae_L1 (vertebral body only); compact target morphology')]


def task(gid=10003):
    row=next(x for x in catalogue() if x['global_class_id']==gid)
    return dict(source_name='bone', case_id='case-a', global_class_id=gid,
                local_class_id=row['local_class_id'], class_name=row['class_name'])


def prior_init(self, model, pending, fingerprint, on_class, **kwargs):
    # The frozen parent needs Torch for CT memoization, unrelated to this audit.
    self.model=model; self.on_class=on_class; self.current=None; self.floor=None
    self.pending={('bone',x['task']['case_id'],x['task']['global_class_id']):
                  (x['task'],tuple(x['floors'])) for x in pending}
    self.stage_rows={}; self.restoration=None
    self.training_audit={'training_dataset_constructions_skipped':0}
    self.memory_calls={'memory_encoder':0,'memory_attention':0}


def prior_begin(self, item):
    self.current=('bone',item['case_id'],item['global_class_id'])
    self.floor=None; self.stage_rows={}
    return self.pending[self.current][1]


class FakeEncoder:
    def __init__(self):
        self.calls=[]; self.output=object(); self.default_modality='ct'

    def forward(self, texts, modality=None):
        self.calls.append(texts)
        return self.output

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class FakePrompt:
    def __init__(self):
        self.semantic_enabled=True; self.semantic_encoder=FakeEncoder()
        self.output=object(); self.restored=False; self.expected_count=438

    def _semantic_embedding(self, semantic_text, semantic_embedding, reference):
        if not self.semantic_enabled:return None
        if semantic_embedding is None:
            semantic_embedding=self.semantic_encoder(semantic_text)
        return semantic_embedding

    def forward(self, semantic_text=None, semantic_embedding=None):
        return self._semantic_embedding(semantic_text, semantic_embedding, None)

    def _anatomy_bias(self, semantic_text, reference):
        return self.output


class SemanticRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('tools.bone_training_semantics_runtime'),
                             'actual-text semantic runtime has not been implemented')
        self.m=importlib.import_module('tools.bone_training_semantics_runtime')
        self.m.configure_semantics(catalogue())
        self.init_patch=patch.object(self.m.PriorSession,'__init__',prior_init)
        self.begin_patch=patch.object(self.m.PriorSession,'begin',prior_begin)
        self.init_patch.start();self.begin_patch.start()
        self.addCleanup(self.init_patch.stop);self.addCleanup(self.begin_patch.stop)
        self.saved=[];self.session=self.make_session()

    def make_session(self, model='v9_2'):
        return self.m.GridSession(model,[dict(task=task(gid),floors=[.35 if model=='v9_2' else .75])
                                        for gid in (10003,10004)],'frozen',self.saved.append)

    def begin(self, gid=10003):
        self.session.begin(task(gid));self.session.floor=.35 if self.session.model=='v9_2' else .75
        self.session.stage_rows={self.session.floor:[]}

    def record(self, gid=10003):
        value=dict(model=self.session.model,task=task(gid),
                   results=[dict(floor=self.session.floor,row=task(gid))])
        self.session.on_class(value)
        return value

    def restore(self, prompt, model='v9_2'):
        """Exercise the real delayed/synchronous guards with tiny module doubles."""
        state=dict(epoch=360 if model=='v9_2' else 490,
                   prompt_generator={'restored':True}, sam2_mask_decoder={'restored':True},
                   mask_decoder={'restored':True})
        decoder=SimpleNamespace(restored=False,expected_count=192 if model=='v9_2' else 212)
        prompt.expected_count=438 if model=='v9_2' else 418
        proofs=[]
        def compare(module, expected):
            if not module.restored:raise ValueError('decoder not yet restored')
            return module.expected_count
        if model=='v9_2':
            entry=ModuleType('finetune_multisource_sam2_v9_2_precision')
            result=(state,object())
            def native_load(path, module, *args, **kwargs):
                module.restored=True;return result
            entry.base=SimpleNamespace(load_prompt_generator_checkpoint=native_load)
            joint=ModuleType('finetune_totalseg_sam2_promptgen_v9_joint')
            class Adapter:
                def __init__(self):self.sam=SimpleNamespace(sam_mask_decoder=decoder)
            joint.SAM2MedicalAdapterV9=Adapter
            evaluator=ModuleType('infer.infer_eight_source_adaptive_v9_2')
            evaluator._install_exact_prompt_factory=lambda *a,**k:None
            modules={m.__name__:m for m in (entry,joint,evaluator)}
        else:
            entry=ModuleType('v10.finetune_totalseg_amos_magic_v10_2_joint')
            result=state
            def native_load(path, module, dec):
                module.restored=True;dec.restored=True;return result
            entry.make_prompt_v10_2=lambda *a,**k:prompt
            entry.load_init_v10_2=native_load
            modules={entry.__name__:entry}
        with patch.dict(sys.modules,modules), patch('tools.amos_module_sensitivity.compare_parameters',compare):
            self.m.install_restoration(self.session,proofs.append)
            if model=='v9_2':
                joint.SAM2MedicalAdapterV9()
                actual=entry.base.load_prompt_generator_checkpoint('checkpoint',prompt)
                self.assertIs(actual,result)
                self.assertFalse(proofs,'v9 decoder is loaded after prompt checkpoint loader')
                with self.assertRaisesRegex(ValueError,'not yet restored'):self.session.begin(task())
                self.assertFalse(proofs)
                decoder.restored=True
            else:
                actual=entry.load_init_v10_2('checkpoint',prompt,decoder)
                self.assertIs(actual,result)
            self.begin()
        self.assertEqual(proofs[0]['prompt_parameters_exact'],438 if model=='v9_2' else 418)
        self.assertEqual(proofs[0]['decoder_parameters_exact'],192 if model=='v9_2' else 212)
        return prompt

    def test_correct_consumption_records_literal_sha_and_preserves_native_return(self):
        prompt=self.restore(FakePrompt())
        text=catalogue()[1]['semantic_text']
        actual=prompt.forward(semantic_text=text)
        self.assertIs(actual,prompt.semantic_encoder.output)
        proof=self.record()['semantic_execution']
        self.assertEqual(proof['call_count'],1)
        self.assertEqual(proof['native_encoder_call_count'],1)
        event=proof['events'][0]
        self.assertEqual(event['key'],['bone','case-a',10003])
        self.assertEqual(event['floor'],.35)
        self.assertEqual(event['semantic_text'],text)
        self.assertEqual(event['text_sha256'],hashlib.sha256(text.encode('utf-8')).hexdigest())
        self.assertTrue(event['semantic_enabled'])
        self.m.validate_semantic_record(self.saved[0],catalogue())

    def test_bare_changed_and_wrong_class_text_are_rejected_before_encoder(self):
        prompt=self.restore(FakePrompt())
        for value in ('rib_right_4','right rib 4; compact target morphology',catalogue()[2]['semantic_text']):
            with self.subTest(text=value),self.assertRaises(ValueError):prompt.forward(value)
        self.assertFalse(prompt.semantic_encoder.calls)
        with self.assertRaises(ValueError):self.record()

    def test_disabled_missing_and_precomputed_semantics_fail(self):
        prompt=self.restore(FakePrompt())
        prompt.semantic_enabled=False
        with self.assertRaises(ValueError):prompt.forward(catalogue()[1]['semantic_text'])
        prompt.semantic_enabled=True
        with self.assertRaises(ValueError):prompt.forward()
        with self.assertRaises(ValueError):prompt.forward(catalogue()[1]['semantic_text'],object())
        self.assertFalse(prompt.semantic_encoder.calls)
        with self.assertRaises(ValueError):self.record()

    def test_encoder_bypass_or_missing_consumption_cannot_commit(self):
        prompt=self.restore(FakePrompt())
        with self.assertRaises(ValueError):self.record()
        with self.assertRaises(ValueError):prompt.semantic_encoder(catalogue()[1]['semantic_text'])
        prompt._anatomy_bias(catalogue()[1]['semantic_text'],None)
        with self.assertRaises(ValueError):self.record()

    def test_class_reset_prevents_previous_calls_from_satisfying_next_class(self):
        prompt=self.restore(FakePrompt())
        prompt.forward(catalogue()[1]['semantic_text']);self.record()
        self.begin(10004)
        with self.assertRaises(ValueError):self.record(10004)
        with self.assertRaises(ValueError):prompt.forward(catalogue()[1]['semantic_text'])
        prompt.forward(catalogue()[2]['semantic_text'])
        proof=self.record(10004)['semantic_execution']
        self.assertEqual(proof['call_count'],1)
        self.assertEqual(proof['events'][0]['key'][-1],10004)

    def test_body_only_qualifier_is_consumed_verbatim(self):
        prompt=self.restore(FakePrompt());self.begin(10004)
        text=catalogue()[2]['semantic_text']
        with self.assertRaises(ValueError):prompt.forward(text.replace(' (vertebral body only)',''))
        prompt.forward(text)
        self.assertIn('(vertebral body only)',self.record(10004)['semantic_execution']['events'][0]['semantic_text'])

    def test_native_encoder_and_prompt_errors_leave_no_success_and_restore_context(self):
        prompt=FakePrompt()
        def fail(texts,modality=None):raise RuntimeError('native encoder failure')
        prompt.semantic_encoder.forward=fail;self.restore(prompt)
        with self.assertRaisesRegex(RuntimeError,'native encoder failure'):prompt.forward(catalogue()[1]['semantic_text'])
        with self.assertRaises(ValueError):self.record()
        with self.assertRaises(ValueError):prompt.semantic_encoder(catalogue()[1]['semantic_text'])

    def test_prompt_cannot_skip_encoder_or_rewrite_its_actual_input(self):
        for bypass in (True,False):
            self.session=self.make_session()
            prompt=FakePrompt()
            def changed(semantic_text,semantic_embedding,reference):
                if bypass:return prompt.output
                return prompt.semantic_encoder('rib_right_4')
            prompt._semantic_embedding=changed
            self.restore(prompt)
            with self.subTest(bypass=bypass),self.assertRaises(ValueError):
                prompt.forward(catalogue()[1]['semantic_text'])
            with self.assertRaises(ValueError):self.record()

    def test_missing_encoder_and_wrong_sat_modality_cannot_supply_proof(self):
        prompt=FakePrompt();prompt.semantic_encoder=None
        with self.assertRaises(ValueError):self.restore(prompt)
        self.session=self.make_session();prompt=self.restore(FakePrompt())
        prompt.semantic_encoder.default_modality='mri'
        with self.assertRaises(ValueError):prompt.forward(catalogue()[1]['semantic_text'])
        with self.assertRaises(ValueError):self.record()

    def test_stage_binding_must_refer_to_an_actual_committed_native_stage(self):
        prompt=self.restore(FakePrompt());prompt.forward(catalogue()[1]['semantic_text'])
        record=self.record()
        record['results'][0]['stages']=[dict(stage='original native stage')]
        self.m.validate_semantic_record(record,catalogue())
        record['semantic_execution']['events'][0]['stage_index']=1
        with self.assertRaises(ValueError):self.m.validate_semantic_record(record,catalogue())

    def test_v10_global_pool_semantic_path_is_checked_and_native_result_untouched(self):
        self.session=self.make_session('v10_2')
        prompt=self.restore(FakePrompt(),'v10_2')
        text=catalogue()[1]['semantic_text']
        prompt.forward(text)
        self.assertIs(prompt._anatomy_bias(text,None),prompt.output)
        with self.assertRaises(ValueError):prompt._anatomy_bias('rib_right_4',None)
        proof=self.record()['semantic_execution']
        self.assertEqual(proof['call_count'],2)
        self.assertEqual(proof['native_encoder_call_count'],1)
        self.assertEqual(proof['events'][1]['native_path'],'_anatomy_bias')

    def test_frozen_catalogue_is_immutable_and_filtered_to_bone(self):
        rows=catalogue();self.m.configure_semantics(rows);self.session=self.make_session()
        rows[1]['semantic_text']='changed after freeze'
        rows[1]['morphology_group']='tubular'
        prompt=self.restore(FakePrompt());prompt.forward(catalogue()[1]['semantic_text'])
        record=self.record()
        self.m.validate_semantic_record(record,catalogue())
        for mutate in ('semantic_text','morphology_group'):
            bad=catalogue();bad[1][mutate]='tampered'
            with self.assertRaises(ValueError):self.m.validate_semantic_record(record,bad)
        irrelevant=catalogue();irrelevant[0]['semantic_text']='non-Bone change'
        self.m.validate_semantic_record(record,irrelevant)

    def test_duplicate_empty_or_unconfigured_catalogues_fail_closed(self):
        for rows in ([],[catalogue()[0]],catalogue()+[catalogue()[1]],
                     [dict(catalogue()[1],semantic_text='')]):
            with self.subTest(rows=rows),self.assertRaises(ValueError):self.m.configure_semantics(rows)
        with patch.object(self.m,'_CONFIG',None),self.assertRaises(ValueError):self.make_session()

    def test_commit_validation_rejects_tampered_event_task_count_and_control(self):
        prompt=self.restore(FakePrompt());prompt.forward(catalogue()[1]['semantic_text'])
        record=self.record()
        changes=[lambda x:x['semantic_execution']['events'][0].update(semantic_text='bare'),
                 lambda x:x['semantic_execution']['events'][0].update(key=['bone','other',10003]),
                 lambda x:x['semantic_execution']['events'][0].update(floor=.75),
                 lambda x:x['semantic_execution'].update(native_encoder_call_count=0),
                 lambda x:x['semantic_execution'].update(call_count=0),
                 lambda x:x['semantic_execution']['events'][0].update(semantic_enabled=False),
                 lambda x:x['task'].update(local_class_id=4)]
        for change in changes:
            bad=copy.deepcopy(record);change(bad)
            with self.subTest(change=change),self.assertRaises(ValueError):self.m.validate_semantic_record(bad,catalogue())
        with self.assertRaises(ValueError):self.m.validate_semantic_record(record,catalogue(),control=True)

    def test_control_and_frozen_exports_are_preserved(self):
        self.m.configure_semantics(catalogue(),control=True);self.session=self.make_session()
        prompt=self.restore(FakePrompt());prompt.forward(catalogue()[1]['semantic_text'])
        record=self.record();self.m.validate_semantic_record(record,catalogue(),control=True)
        self.assertTrue(self.session.shared_control)
        native=importlib.import_module('tools.bone_metric_speed_runtime')
        self.assertTrue(issubclass(self.m.GridSession,native.GridSession))
        for name in ('install_v92','install_v10','tag_carrier'):
            self.assertIs(getattr(self.m,name),getattr(native,name))


if __name__=='__main__':unittest.main()
