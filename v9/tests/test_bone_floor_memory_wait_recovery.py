import importlib
from pathlib import Path
import unittest
from unittest.mock import patch


class MemoryWaitRecoveryTests(unittest.TestCase):
    def setUp(self):self.m=importlib.import_module('tools.bone_floor_memory_wait_recovery')

    def job(self,batch=4,oom=0,ceiling=4,inherited=0,attempt=0,gpu=4):
        return dict(name='job',frame_batch=batch,oom_retries=oom,frame_batch_ceiling=ceiling,
            inherited_oom_retries=inherited,attempt=attempt,gpu=gpu,minimum_free_mib=30720,
            memory_wait_retries=0,not_before=0.,avoid_gpu_until=0.,last_failed_gpu=None)

    def test_batch_one_oom_waits_for_more_memory_without_stopping_other_jobs(self):
        j=self.job(batch=1,oom=4,ceiling=1,inherited=4,attempt=7)
        v=self.m.retry_job(j,1,'torch.OutOfMemoryError: CUDA out of memory',1000)
        self.assertEqual((v['attempt'],v['frame_batch'],v['oom_retries']),(8,1,5))
        self.assertEqual((v['minimum_free_mib'],v['memory_wait_retries'],v['last_failed_gpu']),(45000,1,4))
        self.assertEqual((v['not_before'],v['avoid_gpu_until']),(1060,1600))

    def test_batch_four_oom_halves_batch_and_preserves_ceiling(self):
        v=self.m.retry_job(self.job(),1,'CUDA out of memory',1000)
        self.assertEqual((v['frame_batch'],v['oom_retries'],v['frame_batch_ceiling']),(2,1,4))
        self.assertEqual(v['minimum_free_mib'],30720)

    def test_resource_race_is_requeued_without_fake_oom(self):
        v=self.m.retry_job(self.job(),75,'GPU_RESOURCE_WAIT',1000)
        self.assertEqual((v['frame_batch'],v['oom_retries'],v['attempt']),(4,0,1))

    def test_existing_wait_metadata_is_not_reset_by_defaults(self):
        self.assertEqual(self.m.memory_defaults({'minimum_free_mib':45000})['minimum_free_mib'],45000)

    def test_complete_oom_sequence_keeps_real_counters_and_saturates_at_one(self):
        j=self.job()
        for i,batch in enumerate((2,1,1),1):
            j=self.m.retry_job(j,1,'torch.OutOfMemoryError',1000+i)
            self.assertEqual((j['attempt'],j['oom_retries'],j['frame_batch']),(i,i,batch))
            self.m.n.validate_batch(j,None);self.m.validate_memory_policy(j,None)
        self.assertEqual(j['minimum_free_mib'],45000)

    def test_non_oom_error_remains_terminal(self):
        self.assertIsNone(self.m.retry_job(self.job(),1,'ValueError: changed signed source',1000))

    def test_pending_selector_skips_memory_blocked_head(self):
        pending=[{**self.job(),'minimum_free_mib':45000},{**self.job(),'name':'other'}]
        selected=self.m.select_pending(pending,5,{4:{'free':0},5:{'free':35000}},1000)
        self.assertEqual(selected['name'],'other');self.assertEqual(len(pending),1)

    def test_pending_selector_prefers_another_gpu_for_large_retry(self):
        j={**self.job(),'minimum_free_mib':45000,'last_failed_gpu':4,'avoid_gpu_until':1600}
        pending=[j];available={4:{'free':47000},6:{'free':47000}}
        self.assertIsNone(self.m.select_pending(pending,4,available,1100))
        self.assertEqual(self.m.select_pending(pending,6,available,1100)['name'],'job')

    def test_only_available_gpu_may_be_used_after_alternate_cooldown(self):
        j={**self.job(),'minimum_free_mib':45000,'last_failed_gpu':4,'avoid_gpu_until':1600}
        self.assertIsNone(self.m.select_pending([j],4,{4:{'free':47000}},1500))
        self.assertIsNotNone(self.m.select_pending([j],4,{4:{'free':47000}},1601))

    def test_retry_delay_does_not_block_other_tasks(self):
        pending=[{**self.job(),'not_before':1200},{**self.job(),'name':'ready'}]
        self.assertEqual(self.m.select_pending(pending,5,{5:{'free':47000}},1000)['name'],'ready')

    def test_artifacts_require_resource_audit_for_parent_new_jobs_only(self):
        old=self.m.ROOT/self.m.n.s.OUT/'jobs/task_a0'
        parent=self.m.ROOT/self.m.n.OUT/'jobs/task_a0'
        self.assertNotIn('resource_policy_audit.json',self.m.artifact_names({'run':str(old)}))
        self.assertIn('resource_policy_audit.json',self.m.artifact_names({'run':str(parent)}))

    def test_fresh_config_preserves_batch_cap_and_actual_guard(self):
        v=self.m.resume_settings(None,now=1000)
        self.assertEqual((v['frame_batch'],v['oom_retries'],v['minimum_free_mib']),(4,0,30720))
        self.assertNotIn(45000,self.m.engine(None).canary.__code__.co_consts)

    def test_old_native_spec_does_not_accept_proactive_cap_without_history(self):
        out=self.m.ROOT/'output/old_native';j=dict(name='x',model='v9_2',variant='baseline',source='bone',floor=.35,prepared='p',task_count=1,smoke=True,gpu=2,gpu_uuid='GPU-x',attempt=0,oom_retries=0,frame_batch=16)
        j.update({k:v for k,v in self.m.g.attempt_paths(out,'x',0).items() if k in ('run','log')})
        signed={k:j[k] for k in ('name','model','variant','source','floor','prepared','task_count','smoke')}
        self.m.validate_job_spec(j,signed,out)
        with self.assertRaises(ValueError):self.m.validate_job_spec({**j,'frame_batch':4},signed,out)

    def test_wait_policy_cannot_lower_frozen_minimum_or_forge_fields(self):
        limits={**self.job(),'minimum_free_mib':45000,'memory_wait_retries':1,'oom_retries':1,'last_failed_gpu':4}
        self.m.validate_memory_policy(limits,limits)
        for change in (dict(minimum_free_mib=1000),dict(minimum_free_mib=30720),dict(memory_wait_retries=-1),dict(memory_wait_retries=999),dict(last_failed_gpu=7),dict(last_failed_gpu=None),dict(not_before=float('nan'))):
            with self.assertRaises(ValueError):self.m.validate_memory_policy({**limits,**change},limits)

    def test_process_guard_includes_new_worker_and_excludes_status_self(self):
        for name in ('bone_floor_grid.py','bone_floor_recovery.py','bone_floor_resource_recovery.py','bone_floor_small_batch_recovery.py','bone_floor_memory_wait_recovery.py'):
            self.assertTrue(self.m.process_matches(['python',name,'worker'],2,1))
            self.assertFalse(self.m.process_matches(['python',name,'status'],2,1))
            self.assertFalse(self.m.process_matches(['python',name,'worker'],1,1))

    def test_native_startup_guard_uses_true_per_job_requirement(self):
        targets=[(self.m.r,'verify'),(self.m.g,'verify'),(self.m.g.amos,'verify'),
                 (self.m.r,'verify_job'),(self.m.g,'validate_job_spec'),(self.m.g,'run')]
        before=[getattr(obj,key) for obj,key in targets]
        try:
            with patch.object(self.m.r,'configure'):
                self.m.configure(self.m.ROOT/self.m.OUT,45000)
                self.assertEqual(self.m.g.run.__globals__['START_FREE_MIB'],45000)
        finally:
            for (obj,key),value in zip(targets,before):setattr(obj,key,value)

    def test_new_worker_keeps_signed_science_and_larger_memory_binding(self):
        out=(self.m.ROOT/self.m.OUT).resolve()
        j={**self.job(batch=1,oom=5,ceiling=1,inherited=4,attempt=8,gpu=6),
           'model':'v9_2','variant':'baseline','source':'bone','floor':.45,'prepared':'p',
           'task_count':83,'smoke':False,'gpu_uuid':'GPU-x','minimum_free_mib':45000,
           'memory_wait_retries':1,'last_failed_gpu':4}
        j.update({k:v for k,v in self.m.g.attempt_paths(out,j['name'],8).items() if k in ('run','log')})
        limits=dict(j);signed={k:j[k] for k in ('name','model','variant','source','floor','prepared','task_count','smoke')}
        with patch.object(self.m.b,'read',return_value={'resume':{j['name']:limits}}):
            self.m.validate_job_spec(j,signed,out)
            for change in (dict(floor=.5),dict(gpu=1),dict(frame_batch=4),dict(minimum_free_mib=30720),dict(oom_retries=0)):
                with self.assertRaises(ValueError):self.m.validate_job_spec({**j,**change},signed,out)


if __name__=='__main__':unittest.main()
