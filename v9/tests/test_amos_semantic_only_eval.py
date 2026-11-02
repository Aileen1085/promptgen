"""Fresh one-row production, paired controls, unchanged native queue safety."""
import importlib
import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from tests.test_amos_bg_scribble_module_eval import Proxy
from tests import test_amos_prompt_ablation_eval as original

ROOT=Path(__file__).resolve().parents[1]


def controller():return Proxy(importlib.import_module('tools.amos_semantic_only_eval').compile_context().engine())


class SemanticControllerTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((ROOT/'tools/amos_semantic_only_eval.py').is_file(),'semantic controller required')
        self.entry=importlib.import_module('tools.amos_semantic_only_eval');self.m=controller()
        self.task=dict(source_name='amos',case_id='a',global_class_id=201,local_class_id=1,class_name='spleen')

    def test_private_current_bindings_and_fresh_semantic_scope(self):
        ns=self.m.ns
        for n in ('worker','schedule_queue','class_records','make_attempt','main','verify_worker_complete'):
            self.assertIs(ns[n].__globals__,ns,n)
        self.assertEqual(ns['VARIANTS'],('semantic_only',))
        self.assertEqual(ns['CONTROL_VARIANTS'],('baseline','semantic_only'))
        self.assertEqual(ns['runtime']().__name__,'tools.amos_semantic_only_runtime')
        self.assertEqual(ns['OUT'],'output/amos_v102_e490_floor070_semantic_only_20261102')
        self.assertEqual(ns['OLD_SHA'],'5e2bdbf2fa340c69b9d35bd079ed3d04332d95f9b1eea5871111c3ee55cf8777')
        self.assertTrue(ns['TRANSFORM_PROOF']['controller_reverse_ast_equal'])

    def test_one_production_two_controls_must_commit_atomically_same_batch(self):
        def row(v,b=4):return dict(variant=v,actual_frame_batch=b)
        with patch.object(self.m,'validate_record'):
            self.assertEqual(self.m.validate_class([row('semantic_only')],self.task),4)
            self.assertEqual(self.m.validate_class([row('baseline'),row('semantic_only')],self.task,True),4)
            for records,control in (([row('baseline')],False),([row('semantic_only')],True),
                                    ([row('baseline'),row('semantic_only',2)],True)):
                with self.assertRaises(ValueError):self.m.validate_class(records,self.task,control)

    def test_control_jobs_pin_selected_completed_s21_key_and_two_variants(self):
        with patch.object(self.m,'selected_control',return_value={'task':self.task}):
            jobs=self.m.control_jobs([self.task])
        self.assertEqual(jobs[0]['variants'],['baseline','semantic_only'])
        self.assertTrue(jobs[0]['control']);self.assertEqual(jobs[0]['frame_batch'],4)

    def test_signed_source_anchor_change_is_refused(self):
        with self.assertRaisesRegex(ValueError,'SHA'):self.entry.compile_context(source=b'pass')

    def test_local_profile_refuses_prepare_and_pipeline_before_any_output(self):
        from tools import amos_semantic_only_runtime as runtime
        out=ROOT/'output'/'semantic_local_profile_must_not_be_created'
        self.assertFalse(out.exists())

        with patch.object(runtime,'dependency_profile',return_value=('local_metadata_only',dict(runtime.LOCAL_DEPENDENCIES))),\
             patch.object(self.m,'_native_prepare') as prepare,patch.object(self.m,'_native_run_pipeline') as run:
            with self.assertRaisesRegex(ValueError,'production'):self.m.prepare(out)
            with self.assertRaisesRegex(ValueError,'production'):self.m.run_pipeline(out)
            prepare.assert_not_called();run.assert_not_called()
        self.assertFalse(out.exists())

    def test_status_and_merge_verification_reauthenticates_production_sources(self):
        with patch.object(self.m,'_source_guard',side_effect=RuntimeError('guard sentinel')) as guard:
            with self.assertRaisesRegex(RuntimeError,'guard sentinel'):self.m.verify(ROOT/'output'/'unused')
        self.assertEqual(guard.call_args.kwargs,{'production':True})

    def test_production_count_policy_is_1485_and_no_reference_rows_reused(self):
        self.assertEqual(self.m.PRODUCTION_ROWS,1485)
        self.assertEqual(self.m.policy_extension()['fresh_rows'],1485)
        self.assertEqual(self.m.policy_extension()['reused_rows'],0)
        self.assertEqual(self.m.policy_extension()['spatial_prompt_policy'],'none')
        self.assertEqual(self.m.policy_extension()['semantic_prompt_count'],1)
        self.assertEqual(self.m.policy_extension()['spatial_prompt_count'],0)
        self.assertEqual(self.m.policy_extension()['feedback_interactions'],0)


class SemanticAtomicTests(SemanticControllerTests):
    tasks=original.ControllerTests.tasks
    job=original.ControllerTests.job
    row=original.ControllerTests.row
    record=original.ControllerTests.record
    test_scope_exact_cases_classes_keys_and_training_exclusion=original.ControllerTests.test_scope_exact_cases_classes_keys_and_training_exclusion
    test_oom_ladder_and_resource75_do_not_change_protocol=original.ControllerTests.test_oom_ladder_and_resource75_do_not_change_protocol
    test_atomic_publication_refuses_replacement=original.ControllerTests.test_atomic_publication_refuses_replacement
    test_partial_results_are_never_completed_classes=original.ControllerTests.test_partial_results_are_never_completed_classes
    test_preflight_refuses_output_tmux_process_and_pipeline=original.ControllerTests.test_preflight_refuses_output_tmux_process_and_pipeline
    test_independent_real_gpu_control_gate_required=original.ControllerTests.test_independent_real_gpu_control_gate_required
    def test_pipeline_is_once_only_and_orders_independent_gate_before_full(self):
        with patch.object(self.m,'_source_guard'):
            original.ControllerTests.test_pipeline_is_once_only_and_orders_independent_gate_before_full(self)
    test_class_macro_and_fsum_do_not_follow_task_weighted_or_naive_sum=original.ControllerTests.test_class_macro_and_fsum_do_not_follow_task_weighted_or_naive_sum

    def setUp(self):
        super().setUp();native=tempfile.mkdtemp
        self.temp_patch=patch.object(tempfile,'mkdtemp',side_effect=lambda *a,**k:native(dir=str(ROOT/'output')))
        self.temp_patch.start();self.addCleanup(self.temp_patch.stop)

    def test_exact1485_fresh_rows_and100case_proofs_merge_once_verify_only(self):
        m=self.m;d=Path(tempfile.mkdtemp());tasks=self.tasks();jobs=m.case_jobs(tasks)
        m.atomic_new(d/'status.json',dict(phase='full',pending=0,running={},failed=[]))
        m.atomic_new(d/'manifest.json',dict(tasks=tasks));m.atomic_new(d/'jobs.json',dict(cases=jobs))
        m.atomic_new(d/'protocol.json',{});m.atomic_new(d/'control_verification.json',{})
        rows=[self.row(t) for t in tasks]
        found={m.key(t):dict(record={'records':[dict(variant='semantic_only',row=r)]},spec={'frame_batch':4}) for t,r in zip(tasks,rows)}
        with patch.object(m,'verify'),patch.object(m,'check_control_gate'),patch.object(m,'related',return_value=[]),\
             patch.object(m,'class_records',return_value=found),patch.object(m,'native_reference',return_value=rows),\
             patch.object(m,'verify_case_success') as caseproof:
            result=m.merge(d);self.assertEqual(result['fresh_rows'],1485);self.assertEqual(result['reused_rows'],0)
            self.assertEqual(set(result['variants']),{'semantic_only'});self.assertEqual(caseproof.call_count,100)
            before=m.sha(d/'merged_metrics.json');self.assertEqual(m.merge(d),result);self.assertEqual(m.sha(d/'merged_metrics.json'),before)
            with patch.object(m,'related',return_value=[{'args':['worker']}]),self.assertRaises(ValueError):m.merge(d)
            found.pop(next(iter(found)))
            with self.assertRaises(ValueError):m.merge(d)


class SemanticWorkerTests(unittest.TestCase):
    setup_worker=original.WorkerTests.setup_worker
    fake_runtime=original.WorkerTests.fake_runtime
    patches=original.WorkerTests.patches
    row=original.WorkerTests.row

    def setUp(self):
        self.m=controller();self.task=dict(source_name='amos',case_id='0001',global_class_id=201,local_class_id=1,class_name='spleen',image='data/a.nii.gz',label='data/b.nii.gz')
        native=tempfile.mkdtemp;self.temp_patch=patch.object(tempfile,'mkdtemp',side_effect=lambda *a,**k:native(dir=str(ROOT/'output')))
        self.temp_patch.start();self.addCleanup(self.temp_patch.stop)

    def test_production_worker_runs_exactly_one_prediction_per_key_and_normal_finish(self):
        from contextlib import ExitStack
        m=self.m;d,out,job,tasks,torch=self.setup_worker();fake,calls=self.fake_runtime(job)
        m.atomic_new(out/'control_verification.json',{'independent':'passed'});job['control_gate_sha256']=m.sha(out/'control_verification.json')
        Path(job['spec']).write_text(json.dumps(job))
        with ExitStack() as stack:
            for p in self.patches(d,torch,fake):stack.enter_context(p)
            m.worker(job);records=m.class_records(out);self.assertEqual(len(records),2)
            m.complete_case(out,job);m.verify_worker_complete(job)
        self.assertEqual(calls,[('initialize','0001'),(201,'semantic_only'),(202,'semantic_only')])
        self.assertEqual(m.read(Path(job['run'])/'native_worker_complete.json')['variants'],1)

    def test_control_worker_runs_native_replay_semantic_exactly_and_reports_two_variants(self):
        from contextlib import ExitStack
        m=self.m;d,out,job,tasks,torch=self.setup_worker();job.update(control=True,variants=list(m.CONTROL_VARIANTS),pending=tasks[:1],task_count=1)
        canonical={k:v for k,v in job.items() if k not in ('gpu','gpu_uuid','actual_frame_batch','run','spec','log','control_gate_sha256')}
        (out/'jobs.json').write_text(json.dumps(dict(cases=[],controls=[canonical])))
        Path(job['spec']).write_text(json.dumps(job));fake,calls=self.fake_runtime(job);fake.validate_control=lambda *a:{'passed':True}
        with ExitStack() as stack:
            for p in self.patches(d,torch,fake):stack.enter_context(p)
            m.worker(job);self.assertEqual(len(m.class_records(out,True)),1);m.verify_worker_complete(job)
        self.assertEqual(calls,[('initialize','0001'),(201,'native_control'),(201,'baseline'),(201,'semantic_only')])
        self.assertEqual(m.read(Path(job['run'])/'native_worker_complete.json')['variants'],2)

    def test_oom_in_second_class_retains_first_atomic_class_without_normal_completion(self):
        from contextlib import ExitStack
        m=self.m;d,out,job,tasks,torch=self.setup_worker();fake,calls=self.fake_runtime(job,fail_at=3)
        m.atomic_new(out/'control_verification.json',{'independent':'passed'});job['control_gate_sha256']=m.sha(out/'control_verification.json');Path(job['spec']).write_text(json.dumps(job))
        with ExitStack() as stack:
            for p in self.patches(d,torch,fake):stack.enter_context(p)
            with self.assertRaisesRegex(RuntimeError,'out of memory'):m.worker(job)
            self.assertEqual(len(m.class_records(out)),1)
        self.assertFalse((Path(job['run'])/'native_worker_complete.json').exists())


class SemanticSchedulerTests(original.SchedulerTests):
    def setUp(self):
        self.m=controller();self.task=dict(source_name='amos',case_id='0001',global_class_id=201,local_class_id=1,class_name='spleen')


if __name__=='__main__':unittest.main()
