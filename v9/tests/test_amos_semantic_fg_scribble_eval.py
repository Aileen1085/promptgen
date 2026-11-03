"""Single fresh condition and native queue/disk/atomic publication policy."""
import importlib
from pathlib import Path
import unittest
import tempfile
import copy
import json
import sys
import types
from unittest.mock import patch
from tests.test_amos_bg_scribble_module_eval import Proxy
from tests import test_amos_prompt_ablation_eval as original

ROOT=Path(__file__).resolve().parents[1]


class ForegroundControllerTests(unittest.TestCase):
    tasks=original.ControllerTests.tasks
    job=original.ControllerTests.job
    row=original.ControllerTests.row
    record=original.ControllerTests.record
    test_scope_exact_cases_classes_keys_and_training_exclusion=original.ControllerTests.test_scope_exact_cases_classes_keys_and_training_exclusion
    test_oom_ladder_and_resource75_do_not_change_protocol=original.ControllerTests.test_oom_ladder_and_resource75_do_not_change_protocol
    test_atomic_publication_refuses_replacement=original.ControllerTests.test_atomic_publication_refuses_replacement
    test_independent_real_gpu_control_gate_required=original.ControllerTests.test_independent_real_gpu_control_gate_required
    test_class_macro_and_fsum_do_not_follow_task_weighted_or_naive_sum=original.ControllerTests.test_class_macro_and_fsum_do_not_follow_task_weighted_or_naive_sum

    def setUp(self):
        self.assertTrue((ROOT/'tools/amos_semantic_fg_scribble_eval.py').is_file(),'FG-only controller implementation required')
        self.entry=importlib.import_module('tools.amos_semantic_fg_scribble_eval');self.m=Proxy(self.entry.compile_context().engine())
        self.task=dict(source_name='amos',case_id='amos_0218',global_class_id=118,local_class_id=1,class_name='spleen')
        native=tempfile.mkdtemp;self.temp_patch=patch.object(tempfile,'mkdtemp',side_effect=lambda *a,**k:native(dir=str(ROOT/'output')))
        self.temp_patch.start();self.addCleanup(self.temp_patch.stop)

    def test_isolated_sources_fresh_1485_target_no_failed_parent(self):
        m=self.m;ns=m.ns
        self.assertEqual(m.VARIANTS,('semantic_fg_scribble',));self.assertEqual(m.CONTROL_VARIANTS,('baseline','semantic_fg_scribble'))
        self.assertEqual(m.OUT,'output/amos_v102_e490_floor070_semantic_fg_scribble_20261103')
        self.assertEqual(m.PRODUCTION_ROWS,1485);self.assertEqual(m.policy_extension()['reused_rows'],0)
        self.assertEqual(m.policy_extension()['spatial_prompt_policy'],'native_cached_foreground_scribble_only')
        self.assertFalse(m.policy_extension()['full_original_CT'])
        self.assertNotIn('frozen_records',ns);self.assertNotIn('parent_history',ns)
        for name in ('worker','schedule_queue','class_records','make_attempt','main'):
            self.assertIs(ns[name].__globals__,ns)
        jobs=m.case_jobs(self.tasks());self.assertEqual(len(jobs),100)
        self.assertEqual(sum(len(j['pending']) for j in jobs),1485)

    def test_only_target_production_and_two_atomic_control_variants(self):
        def r(v):return dict(variant=v,actual_frame_batch=4)
        with patch.object(self.m,'validate_record'):
            self.assertEqual(self.m.validate_class([r('semantic_fg_scribble')],self.task),4)
            self.assertEqual(self.m.validate_class([r('baseline'),r('semantic_fg_scribble')],self.task,True),4)
            for rows,control in (([r('baseline')],False),([r('semantic_fg_scribble')],True)):
                with self.assertRaises(ValueError):self.m.validate_class(rows,self.task,control)

    def test_source_and_production_profile_fail_closed(self):
        with self.assertRaisesRegex(ValueError,'SHA'):self.entry.compile_context(source=b'pass')
        with patch.object(self.m,'_source_guard',side_effect=ValueError('production sentinel')),patch.object(self.m,'_native_prepare') as prepare:
            with self.assertRaisesRegex(ValueError,'production sentinel'):self.m.prepare(ROOT/'output/unused_fg')
            prepare.assert_not_called()

    def test_disk_reserve_preflight_and_every_dispatch_not_verify_or_oom(self):
        m=self.m
        with patch.object(m,'disk_free_bytes',return_value=3*1024**3-1),patch.object(m,'_native_preflight') as native:
            with self.assertRaisesRegex(ValueError,'3 GiB'):m.preflight(ROOT/m.OUT)
            native.assert_not_called()
            self.assertEqual(m.Backend(ROOT/m.OUT,False).available(),{})
        with patch.object(m,'disk_free_bytes',return_value=3*1024**3),patch.object(m,'gpu_snapshot',return_value={2:{'free':40000}}):
            self.assertIn(2,m.Backend(ROOT/m.OUT,False).available())
        with patch.object(m,'disk_free_bytes',side_effect=AssertionError('ordinary verify must not inspect reserve')),patch.object(m,'_source_guard'),patch.object(m,'_native_verify',return_value={'policy':{}}):
            with self.assertRaisesRegex(ValueError,'policy'):m.verify(ROOT/m.OUT)

    def test_control_jobs_require_same_selected_key_and_exact_batch4(self):
        with patch.object(self.m,'selected_control',return_value={'task':self.task}):jobs=self.m.control_jobs([self.task])
        self.assertEqual(jobs[0]['variants'],['baseline','semantic_fg_scribble']);self.assertEqual(jobs[0]['frame_batch'],4)

    def test_pipeline_waits_for_initial_disk_reserve_before_native_preflight(self):
        with patch.object(self.m,'_source_guard'),patch.object(self.m,'disk_free_bytes',side_effect=[0,1,3*1024**3]),patch.object(self.m.time,'sleep') as sleep,patch.object(self.m,'_native_run_pipeline') as run:
            self.m.run_pipeline(ROOT/self.m.OUT)
            self.assertEqual(sleep.call_count,2);run.assert_called_once()

    def test_protocol_pins_complete_control_trace_and_original_production_trace(self):
        p=self.m.policy_extension()
        self.assertIn('control_encoder_event_limit',p,'new signed protocol must pin complete control trace')
        self.assertEqual(p['control_encoder_event_limit'],4096)
        self.assertEqual(p['production_encoder_event_limit'],64)
        self.assertTrue(p['controls_require_complete_encoder_trace'])


def controller():return Proxy(importlib.import_module('tools.amos_semantic_fg_scribble_eval').compile_context().engine())

class ForegroundAtomicTests(ForegroundControllerTests):
    tasks=original.ControllerTests.tasks
    job=original.ControllerTests.job
    row=original.ControllerTests.row
    record=original.ControllerTests.record
    test_scope_exact_cases_classes_keys_and_training_exclusion=original.ControllerTests.test_scope_exact_cases_classes_keys_and_training_exclusion
    test_oom_ladder_and_resource75_do_not_change_protocol=original.ControllerTests.test_oom_ladder_and_resource75_do_not_change_protocol
    test_atomic_publication_refuses_replacement=original.ControllerTests.test_atomic_publication_refuses_replacement
    test_partial_results_are_never_completed_classes=original.ControllerTests.test_partial_results_are_never_completed_classes
    def test_preflight_refuses_output_tmux_process_and_pipeline(self):
        with patch.object(self.m,'preflight',self.m._native_preflight):
            original.ControllerTests.test_preflight_refuses_output_tmux_process_and_pipeline(self)
    test_independent_real_gpu_control_gate_required=original.ControllerTests.test_independent_real_gpu_control_gate_required
    def test_pipeline_is_once_only_and_orders_independent_gate_before_full(self):
        with patch.object(self.m,'_source_guard'):
            original.ControllerTests.test_pipeline_is_once_only_and_orders_independent_gate_before_full(self)
    test_class_macro_and_fsum_do_not_follow_task_weighted_or_naive_sum=original.ControllerTests.test_class_macro_and_fsum_do_not_follow_task_weighted_or_naive_sum

    def setUp(self):
        super().setUp()

    def test_exact1485_fresh_rows_and100case_proofs_merge_once_verify_only(self):
        m=self.m;d=Path(tempfile.mkdtemp());tasks=self.tasks();jobs=m.case_jobs(tasks)
        m.atomic_new(d/'status.json',dict(phase='full',pending=0,running={},failed=[]))
        m.atomic_new(d/'manifest.json',dict(tasks=tasks));m.atomic_new(d/'jobs.json',dict(cases=jobs))
        m.atomic_new(d/'protocol.json',{});m.atomic_new(d/'control_verification.json',{})
        rows=[self.row(t) for t in tasks]
        found={m.key(t):dict(record={'records':[dict(variant='semantic_fg_scribble',row=r)]},spec={'frame_batch':4}) for t,r in zip(tasks,rows)}
        with patch.object(m,'verify'),patch.object(m,'check_control_gate'),patch.object(m,'related',return_value=[]),\
             patch.object(m,'class_records',return_value=found),patch.object(m,'native_reference',return_value=rows),\
             patch.object(m,'verify_case_success') as caseproof:
            result=m.merge(d);self.assertEqual(result['fresh_rows'],1485);self.assertEqual(result['reused_rows'],0)
            self.assertEqual(set(result['variants']),{'semantic_fg_scribble'});self.assertEqual(caseproof.call_count,100)
            before=m.sha(d/'merged_metrics.json');self.assertEqual(m.merge(d),result);self.assertEqual(m.sha(d/'merged_metrics.json'),before)
            with patch.object(m,'related',return_value=[{'args':['worker']}]),self.assertRaises(ValueError):m.merge(d)
            found.pop(next(iter(found)))
            with self.assertRaises(ValueError):m.merge(d)


class ForegroundWorkerTests(unittest.TestCase):
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
        self.assertEqual(calls,[('initialize','0001'),(201,'semantic_fg_scribble'),(202,'semantic_fg_scribble')])
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
        self.assertEqual(calls,[('initialize','0001'),(201,'native_control'),(201,'baseline'),(201,'semantic_fg_scribble')])
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

    def test_worker_rechecks_real_resource_snapshot_before_model_load(self):
        from contextlib import ExitStack
        for available in ({},{2:dict(uuid='wrong',free=65000)},{2:dict(uuid='GPU-2',free=30719)}):
            m=self.m;d,out,job,tasks,torch=self.setup_worker();fake,calls=self.fake_runtime(job)
            m.atomic_new(out/'control_verification.json',{'independent':'passed'});job['control_gate_sha256']=m.sha(out/'control_verification.json');Path(job['spec']).write_text(json.dumps(job))
            with ExitStack() as stack:
                for p in self.patches(d,torch,fake):stack.enter_context(p)
                stack.enter_context(patch.object(m,'gpu_snapshot',return_value=available))
                with self.assertRaises(SystemExit) as error:m.worker(job)
                self.assertEqual(error.exception.code,75);self.assertFalse(calls)

    def test_worker_actual_cuda_uuid_and_small_tensor_probe_fail_closed(self):
        from contextlib import ExitStack
        for corrupt in ('uuid','probe'):
            m=self.m;d,out,job,tasks,torch=self.setup_worker();fake,calls=self.fake_runtime(job)
            m.atomic_new(out/'control_verification.json',{'independent':'passed'});job['control_gate_sha256']=m.sha(out/'control_verification.json');Path(job['spec']).write_text(json.dumps(job))
            if corrupt=='uuid':torch.cuda.get_device_properties=lambda _:types.SimpleNamespace(uuid='GPU-wrong')
            else:torch.ones=lambda *a,**k:types.SimpleNamespace(item=lambda:0)
            with ExitStack() as stack:
                for p in self.patches(d,torch,fake):stack.enter_context(p)
                with self.assertRaisesRegex(ValueError,'UUID/probe'):m.worker(job)
                self.assertFalse(calls)


class ForegroundSchedulerTests(original.SchedulerTests):
    def setUp(self):
        self.m=controller();self.task=dict(source_name='amos',case_id='0001',global_class_id=201,local_class_id=1,class_name='spleen')

    def test_disk_drop_after_gpu_snapshot_retains_pending_without_attempt_or_oom(self):
        m=self.m;j=self.job();b=original.FakeQueue(m,[j],[(0,0,'')])
        b.disk_ready=lambda:b.clock>=10
        m.schedule_queue([j],False,b)
        self.assertEqual(len(b.started),1);self.assertEqual(b.started[0]['attempt'],0)
        self.assertEqual(b.started[0]['frame_batch'],4);self.assertEqual(b.states[0]['pending'],1)
        self.assertFalse(b.states[0]['failed']);self.assertEqual(b.states[0]['running'],{})


