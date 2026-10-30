"""Offline fresh controls after a source-bound encoder-observer failure."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
ENTRY=ROOT/'tools/amos_prompt_reuse_owner_eval.py'


class OwnerControllerTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(ENTRY.exists(),'add offline owner-fix controller')
        self.m=importlib.import_module('tools.amos_prompt_reuse_owner_eval')
    def d(self):return Path(tempfile.mkdtemp(prefix='amos_owner_'))
    def write(self,p,value):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return p

    def test_bound_hashes_are64hex_and_failure_hash_has_exact_two_halves(self):
        for name in ('ANCESTOR_SHA256','RUNTIME_ANCESTOR_SHA256','OWNER_RUNTIME_SHA256','FAILURE_SHA256','PROBE_SHA256','PROBE_INVENTORY_SHA256','FAILED_PROTOCOL_SHA256'):
            self.assertRegex(getattr(self.m,name),r'\A[0-9a-f]{64}\Z',name)
        self.assertEqual(self.m.FAILURE_SHA256,'71d7a30a2cfba98aac4a3d0291ea4fc1'+'72c9cf36660574924cc795de65a44b6c')

    def test_private_engine_dispatches_new_entry_and_runtime_without_parent_mutation(self):
        old=importlib.import_module('tools.amos_prompt_reuse_resume_eval');oldctx=old.compile_context();before=dict(oldctx.__dict__)
        c=self.m.compile_context();ns=c.engine();self.assertEqual(before,oldctx.__dict__)
        for name in ('worker','make_attempt','merge','schedule_queue','class_records','main'):self.assertIs(ns[name].__globals__,ns)
        self.assertEqual(ns['OUT'],self.m.OUT);self.assertEqual(c.DRAINED,old.DRAINED)
        self.assertEqual(ns['runtime']().__name__,'tools.amos_prompt_reuse_owner_runtime')
        self.assertIn('tools/amos_prompt_reuse_owner_eval.py',ns['SOURCE_SEEDS']);self.assertIn('tools/amos_prompt_reuse_owner_runtime.py',ns['SOURCE_SEEDS'])
        d=self.d();job=dict(log=str(d/'a.log'),spec=str(d/'a.json'),gpu=2)
        with patch.object(ns['subprocess'],'Popen',return_value=object()) as spawn:
            _,handle=ns['Backend'](d,False).spawn(job);handle.close()
        self.assertEqual(Path(spawn.call_args.args[0][1]),ENTRY)
        with self.assertRaisesRegex(ValueError,'offline|signal'):c.signal_exact({},15)
        with self.assertRaisesRegex(ValueError,'offline|drain'):c.finish_parent()

    def test_compile_refuses_changed_signed_source(self):
        with self.assertRaisesRegex(ValueError,'SHA'):self.m.compile_context(source=b'pass')

    def test_offline_prepare_uses_certificate_and_twice_quiescence_without_process_capture(self):
        c=self.m.compile_context();events=[]
        with patch.object(self.m,'verify_failure',side_effect=lambda ctx:events.append('failure')),\
             patch.dict(c.__dict__,source_guard=lambda:events.append('sources'),verify_drained=lambda:events.append('certificate'),
                        require_quiescence=lambda:events.extend(['quiet1','quiet2']),verify_parent=lambda **kw:events.append('parent')):
            c.prepare_guard()
        self.assertEqual(events,['failure','sources','certificate','quiet1','quiet2','parent'])

    def failure_fixture(self):
        m=self.m;d=self.d();out=d/m.FAILED_OUT
        log=out/'logs/amos_native_control_a0.log';log.parent.mkdir(parents=True);log.write_text('AttributeError: decoder conv_s0')
        status=dict(phase='controls',pending=0,running={},failed=[dict(name='amos_native_control',returncode=1,log=str(log))])
        records=[dict(old=i) for i in range(680)]
        for name in ('protocol.json','protocol_signature.json','jobs.json','native_bootstrap_audit.json'):self.write(out/name,{})
        self.write(out/'status.json',status);self.write(out/'reuse_audit.json',dict(records=records))
        files={str(p.relative_to(d)).replace('\\','/'):m.sha(p) for p in out.rglob('*') if p.is_file()}
        value=dict(kind='amos_reuse_encoder_owner_failure_v1',output=m.FAILED_OUT,status=status,source_sha256=m.ANCESTOR_SHA256,
                   runtime_sha256=m.RUNTIME_ANCESTOR_SHA256,files=files,failed_log=str(log),failed_log_text=log.read_text(),related_checks=[[],[]],
                   audited_reused_classes=680,audited_reused_rows=4080,new_classes=0,new_rows=0,controls_passed=False,independent_gate_exists=False,merged_exists=False)
        path=self.write(d/m.FAILURE,value);native=unittest.mock.Mock(return_value={})
        def verify(files):
            for name,expected in files.items():
                if m.sha(d/name)!=expected:raise ValueError('frozen file changed')
        c=types.SimpleNamespace(os=os,verify_inventory=verify,engine=lambda:dict(native_verify=lambda out:None,native_class_records=native),
             _owner_ancestor_verify_reuse=lambda out:None,verify_parent=lambda:{i:{} for i in range(680)},_record_index=lambda found:records)
        return d,path,value,c,native

    def test_failed_control_lineage_refuses_changed_files_or_any_new_class(self):
        m=self.m;d,path,value,c,native=self.failure_fixture()
        with patch.object(m,'ROOT',d),patch.object(m,'FAILURE_SHA256',m.sha(path)),patch.object(m,'verify_probe',return_value={}):
            self.assertEqual(m.verify_failure(c),value)
            self.assertEqual(native.call_args_list,[unittest.mock.call(d/m.FAILED_OUT,False),unittest.mock.call(d/m.FAILED_OUT,True)])
            native.return_value={('unexpected',):{}}
            with self.assertRaisesRegex(ValueError,'committed'):m.verify_failure(c)
            native.return_value={};Path(value['failed_log']).write_text('changed')
            with self.assertRaisesRegex(ValueError,'changed'):m.verify_failure(c)

    def test_failed_control_cannot_substitute_new_gate_or_success(self):
        m=self.m;d,path,value,c,native=self.failure_fixture()
        with patch.object(m,'ROOT',d),patch.object(m,'FAILURE_SHA256',m.sha(path)),patch.object(m,'verify_probe',return_value={}):
            self.write(d/m.FAILED_OUT/'control_verification.json',dict(passed=True))
            with self.assertRaisesRegex(ValueError,'gate'):m.verify_failure(c)

    def test_diagnostic_binds_actual_owner_zero_predictions_finish_and_source(self):
        m=self.m;d=self.d();source=d/'sam2/sam2/modeling/sam2_base.py';source.parent.mkdir(parents=True);source.write_text('signed forward image source')
        protocol=self.write(d/m.FAILED_OUT/'protocol.json',dict(sources={'sam2/sam2/modeling/sam2_base.py':m.sha(source)}))
        restoration=dict(model='v10_2',epoch=490,weight_mode='live',prompt_parameters_exact=418,decoder_parameters_exact=212,guards_passed=True,model_load_count=1)
        probe=dict(kind='amos_actual_encoder_owner_probe_v1',predictions=0,outer_class='peft.peft_model.PeftModelForFeatureExtraction',
                   owner_class='sam2.modeling.sam2_base.SAM2Base',same_object=False,forward_qualname='SAM2Base.forward_image',forward_source=str(source),
                   outer_decoder='SAM2MaskDecoderWith3DPromptAdapterV10',owner_decoder='MaskDecoder',outer_highres=True,owner_highres=True,
                   owner_convs=dict(conv_s0=True,conv_s1=True),outer_convs=dict(conv_s0=False,conv_s1=False),protocol_sha256=m.sha(protocol),restoration=restoration,
                   finish=dict(restoration,successful_native_completion=True,full_tensor_comparison_phase='normal_worker_completion'),
                   resource=dict(gpu=2,gpu_uuid='GPU-own',visible_cuda_ordinal=0,minimum_free_mib=30720,free_mib_before_load=69327))
        path=self.write(d/m.PROBE,probe);folder=path.parent
        for name in ('spec.json','prepared/manifest.json','prepared/refs.json'):self.write(folder/name,{})
        files={str(p.relative_to(d)).replace('\\','/'):m.sha(p) for p in folder.rglob('*') if p.is_file()}
        inv=self.write(d/m.PROBE_INVENTORY,dict(kind='amos_encoder_owner_probe_inventory_v1',diagnostic_only=True,predictions=0,files=files))
        def verify(current):
            for name,expected in current.items():
                if m.sha(d/name)!=expected:raise ValueError('probe file changed')
        c=types.SimpleNamespace(verify_inventory=verify)
        with patch.object(m,'ROOT',d),patch.object(m,'PROBE_SHA256',m.sha(path)),patch.object(m,'PROBE_INVENTORY_SHA256',m.sha(inv)),patch.object(m,'FAILED_PROTOCOL_SHA256',m.sha(protocol)):
            self.assertEqual(m.verify_probe(c)['files'],files)
            source.write_text('changed same file')
            with self.assertRaisesRegex(ValueError,'source SHA'):m.verify_probe(c)


_old_tests=importlib.import_module('tests.test_amos_prompt_reuse_eval')


class OwnerWorkerExecutionTests(_old_tests.ChildWorkerExecutionTests):
    def setUp(self):
        c=importlib.import_module('tools.amos_prompt_reuse_owner_eval').compile_context()
        self.m=_old_tests._ChildGlobalsProxy(c.engine())
        self.task=dict(source_name='amos',case_id='0001',global_class_id=201,local_class_id=1,
                       class_name='spleen',image='data/a.nii.gz',label='data/b.nii.gz')


class OwnerQueueExecutionTests(_old_tests.ChildSchedulerExecutionTests):
    setUp=OwnerWorkerExecutionTests.setUp


if __name__=='__main__':unittest.main()
