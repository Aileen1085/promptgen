import importlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class V2Tests(unittest.TestCase):
    def module(self):
        try: return importlib.import_module('tools.five_source_prompt_type_eval_v2')
        except ModuleNotFoundError: self.fail('additive v2 controller missing')

    def fake_torch(self):
        return SimpleNamespace(cuda=SimpleNamespace(get_device_properties=lambda i:SimpleNamespace(uuid='abc')),
            ones=lambda *a,**k:SimpleNamespace(item=lambda:1),set_num_threads=lambda n:None)

    def test_low_free_on_either_worker_check_is_resource_wait(self):
        m=self.module();job=dict(gpu=0,gpu_uuid='GPU-abc')
        for values in (['0, GPU-abc, 4836'],['0, GPU-abc, 50000','0, GPU-abc, 4836']):
            with patch.object(m.shutil,'disk_usage',return_value=SimpleNamespace(free=10*1024**3)),patch.object(m.subprocess,'check_output',side_effect=values),patch.dict('sys.modules',torch=self.fake_torch()):
                with self.assertRaises(SystemExit) as error: m.bind_resources(job,Path('.'))
                self.assertEqual(error.exception.code,75)

    def test_identity_mismatch_is_hard_error_and_disk_low_is_wait(self):
        m=self.module();job=dict(gpu=0,gpu_uuid='GPU-abc')
        with patch.object(m.shutil,'disk_usage',return_value=SimpleNamespace(free=1)):
            with self.assertRaises(SystemExit) as error:m.bind_resources(job,Path('.'))
            self.assertEqual(error.exception.code,75)
        with patch.object(m.shutil,'disk_usage',return_value=SimpleNamespace(free=10*1024**3)),patch.object(m.subprocess,'check_output',return_value='0, GPU-wrong, 4836'):
            with self.assertRaises(ValueError):m.bind_resources(job,Path('.'))

    def test_successful_checks_keep_both_actual_snapshots(self):
        m=self.module()
        with patch.object(m.shutil,'disk_usage',return_value=SimpleNamespace(free=10*1024**3)),patch.object(m.subprocess,'check_output',side_effect=['0, GPU-abc, 50000','0, GPU-abc, 49000']),patch.dict('sys.modules',torch=self.fake_torch()):
            proof=m.bind_resources(dict(gpu=0,gpu_uuid='GPU-abc'),Path('.'))
        self.assertEqual(proof['first_snapshot']['free_mib'],50000)
        self.assertEqual(proof['second_snapshot']['free_mib'],49000)
        self.assertTrue(proof['cuda_probe'])

    def test_one_production_job_per_case_all_three_conditions(self):
        m=self.module()
        tasks=[dict(source_name='magic',case_id='pancreas/13',global_class_id=i) for i in (1,2)]
        self.assertTrue(hasattr(m,'case_jobs'),'case-coupled jobs missing')
        jobs,controls=m.case_jobs(tasks)
        self.assertEqual(len(jobs),1)
        self.assertEqual(jobs[0]['variant'],'all')
        self.assertEqual(jobs[0]['tasks'],tasks)
        self.assertEqual(len(controls),1)

    def test_atomic_publication_rejects_incomplete_or_mixed_batch_cases(self):
        m=self.module();tasks=[dict(source_name='magic',case_id='pancreas/13',global_class_id=1)]
        self.assertTrue(hasattr(m,'publish_case'),'atomic case publisher missing')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=root/'attempt';run.mkdir()
            values=[(tasks[0],v,dict(actual_frame_batch=4,variant=v)) for v in m.VARIANTS]
            for malformed in (values[:-1],values[:2]+[(tasks[0],m.VARIANTS[2],dict(actual_frame_batch=2))]):
                with self.assertRaises(ValueError):m.publish_case(root,run,tasks,malformed,4)
                self.assertFalse(m.record_path(root,tasks[0],m.VARIANTS[0]).exists())
            m.publish_case(root,run,tasks,values,4)
            self.assertTrue(all(m.record_path(root,tasks[0],v).exists() for v in m.VARIANTS))
            with self.assertRaises(FileExistsError):m.publish_case(root,run,tasks,values,4)

class WholeWorkerTests(unittest.TestCase):
    def test_late_real_oom_publishes_zero_then_retries_all_conditions_at_batch_two(self):
        from tools import five_source_prompt_type_eval_v2 as m
        tasks=[dict(source_name='magic',case_id='pancreas/13',global_class_id=i) for i in (1,2)]
        protocol=dict(inventory=dict(tasks=tasks,refs=[dict(t,prompt_cache_name='p') for t in tasks],paths={}))
        calls=[]
        def initialize(job,paths):
            batch=job['frame_batch']
            def evaluate(task,variant):
                calls.append((batch,task['global_class_id'],variant))
                if batch==4 and task['global_class_id']==2 and variant=='fg_box_bg_point':raise RuntimeError('CUDA out of memory')
                return dict(actual_frame_batch=batch,variant=variant,row=task)
            return SimpleNamespace(evaluate=evaluate,finish=lambda:dict(successful_native_completion=True))
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            for source in m.SOURCES:
                m.atomic_new(out/'controls'/source/'control_pass.json',dict(passed=True,source=source,protocol_sha256='digest'))
            for batch in (2,4):m.atomic_new(out/'controls/magic'/('batch_'+str(batch))/'control_pass.json',dict(passed=True,source='magic',actual_frame_batch=batch,protocol_sha256='digest'))
            job=dict(source='magic',case_id='pancreas/13',tasks=tasks,variant='all',frame_batch=4,attempt=0)
            with patch.object(m,'protocol_at',return_value=(protocol,'digest')),patch.object(m,'bind_resources',return_value={}),patch.object(m,'verify_inputs',return_value={}),patch.object(m.runtime,'initialize',side_effect=initialize):
                with self.assertRaisesRegex(RuntimeError,'CUDA out of memory'):m.worker(out,job)
                self.assertFalse((out/'records').exists())
                job.update(frame_batch=2,attempt=1)
                m.worker(out,job)
            fresh=[call for call in calls if call[0]==2]
            self.assertEqual(len(fresh),6)
            self.assertEqual({(i,v) for _,i,v in fresh},{(i,v) for i in (1,2) for v in m.VARIANTS})
            self.assertTrue(all(m.read(m.record_path(out,t,v))['actual_frame_batch']==2 for t in tasks for v in m.VARIANTS))

class StatusCouplingTests(unittest.TestCase):
    def test_status_rejects_partial_or_mixed_batch_case_even_with_valid_individual_rows(self):
        from tools import five_source_prompt_type_eval_v2 as m
        task=dict(source_name='magic',case_id='pancreas/13',global_class_id=1)
        inputs={'v10/infer/infer_eight_source_adaptive_v10_2.py':'native'}
        protocol=dict(inventory=dict(tasks=[task],input_sha256=inputs))
        for batches in ([4,4],[4,2,4]):
            with tempfile.TemporaryDirectory() as directory:
                out=Path(directory);m.atomic_new(out/'protocol.json',protocol);digest=m.sha(out/'protocol.json')
                for variant,batch in zip(m.VARIANTS,batches):
                    record=dict(protocol_sha256=digest,control=False,input_sha256=inputs,native_source_sha256='native',actual_frame_batch=batch,normal_finish=dict(successful_native_completion=True))
                    m.atomic_new(m.record_path(out,task,variant),record)
                with patch.object(m,'validate_inventory',return_value=[task]),patch.object(m.runtime,'validate_record',return_value=True),patch.object(m,'require_batch_gate',return_value={}):
                    with self.assertRaises(ValueError):m.status(out)


if __name__ == '__main__':unittest.main()
