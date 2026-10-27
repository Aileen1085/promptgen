import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class SmallBatchRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.m = importlib.import_module('tools.bone_floor_small_batch_recovery')

    def test_proactive_batch_cap_is_not_reported_as_oom(self):
        self.assertEqual(self.m.resume_settings(), dict(attempt=0, oom_retries=0,
            frame_batch=4, frame_batch_ceiling=4, inherited_oom_retries=0))
        old=dict(attempt=2, oom_retries=1, frame_batch=8)
        self.assertEqual(self.m.resume_settings(old), dict(attempt=3, oom_retries=1,
            frame_batch=4, frame_batch_ceiling=4, inherited_oom_retries=1))

    def test_inherits_batch_one_and_next_attempt(self):
        old=dict(attempt=6, oom_retries=4, frame_batch=1)
        self.assertEqual(self.m.resume_settings(old, oom=True), dict(attempt=7,
            oom_retries=4, frame_batch=1, frame_batch_ceiling=1, inherited_oom_retries=4))

    def test_last_unwritten_oom_is_counted_before_proactive_cap(self):
        old=dict(attempt=0, oom_retries=0, frame_batch=16)
        self.assertEqual(self.m.resume_settings(old, oom=True), dict(attempt=1,
            oom_retries=1, frame_batch=4, frame_batch_ceiling=4, inherited_oom_retries=1))

    def test_canary_retry_keeps_original_ceiling_and_actual_oom_count(self):
        old=dict(attempt=0,oom_retries=1,frame_batch=2,frame_batch_ceiling=4,inherited_oom_retries=0)
        self.assertEqual(self.m.resume_settings(old), {**old,'attempt':1})

    def test_actual_batch_history_does_not_increase_or_reset(self):
        limits=dict(attempt=3,oom_retries=1,frame_batch=4,frame_batch_ceiling=4,inherited_oom_retries=1)
        self.m.validate_batch(limits, limits)
        self.m.validate_batch({**limits,'attempt':4,'oom_retries':2,'frame_batch':2}, limits)
        for patch in (dict(frame_batch=8),dict(oom_retries=0),dict(attempt=0),
                      dict(frame_batch_ceiling=8),dict(inherited_oom_retries=0),dict(frame_batch=2)):
            with self.assertRaises(ValueError):self.m.validate_batch({**limits,**patch},limits)

    def test_process_guard_covers_all_generations_and_excludes_self_status(self):
        for script in ('bone_floor_grid.py','bone_floor_recovery.py','bone_floor_resource_recovery.py','bone_floor_small_batch_recovery.py'):
            self.assertTrue(self.m.process_matches(['python',script,'worker'],2,1))
            self.assertFalse(self.m.process_matches(['python',script,'status'],2,1))
            self.assertFalse(self.m.process_matches(['python',script,'worker'],1,1))

    def test_resource_transform_changes_only_exact_memory_guard(self):
        source="def run(job):\n    if gpu['free']<45000:\n        raise SystemExit(75)\n    scientific_call(job)\n"
        expected=source.replace("gpu['free']<45000", "gpu['free']<START_FREE_MIB")
        self.assertEqual(self.m.resource_source(source),expected)
        for bad in ('def run(): pass',source+source):
            with self.assertRaises(ValueError):self.m.resource_source(bad)

    def test_rejects_non_guard_constant_substitution(self):
        with self.assertRaises(ValueError):self.m.resource_source('def scientific():\n    return 45000\n')

    def test_engine_keeps_worker_entry_and_new_guard_policy(self):
        engine=self.m.engine(None)
        self.assertEqual(engine.launch.__globals__['__file__'],self.m.__file__)
        self.assertEqual(engine.launch.__globals__['START_FREE_MIB'],32768)
        self.assertNotIn(45000,engine.launch.__code__.co_consts)
        self.assertNotIn(45000,engine.canary.__code__.co_consts)

    def test_new_path_validates_scientific_fields_and_honest_cap(self):
        out=(self.m.ROOT/self.m.OUT).resolve();limits=self.m.resume_settings()
        job=dict(name='x',model='v9_2',variant='baseline',source='bone',floor=.35,
            prepared='p',task_count=10,smoke=False,gpu=2,gpu_uuid='GPU-x',**limits)
        job.update({k:v for k,v in self.m.g.attempt_paths(out,'x',0).items() if k in ('run','log')})
        signed={k:job[k] for k in ('name','model','variant','source','floor','prepared','task_count','smoke')}
        with patch.object(self.m.b,'read',return_value={'resume':{'x':limits}}):
            self.m.validate_job_spec(job,signed,out)
            for change in (dict(floor=.4),dict(gpu=0),dict(frame_batch=16),dict(oom_retries=1),dict(source='totalseg')):
                with self.assertRaises(ValueError):self.m.validate_job_spec({**job,**change},signed,out)

    def test_old_path_uses_unchanged_native_validator(self):
        out=self.m.ROOT/'output/old_signed';job=dict(name='x',model='v9_2',variant='baseline',source='bone',floor=.35,
            prepared='p',task_count=1,smoke=True,gpu=2,gpu_uuid='GPU-x',attempt=0,oom_retries=0,frame_batch=16)
        job.update({k:v for k,v in self.m.g.attempt_paths(out,'x',0).items() if k in ('run','log')})
        signed={k:job[k] for k in ('name','model','variant','source','floor','prepared','task_count','smoke')}
        self.m.validate_job_spec(job,signed,out)
        with self.assertRaises(ValueError):self.m.validate_job_spec({**job,'frame_batch':4},signed,out)

    def test_forbidden_gpus_still_rejected(self):
        self.assertEqual(self.m.s.gpu_ids('2,4,5,6'),[2,4,5,6])
        for text in ('0','3','1','7','2,2'):
            with self.assertRaises(ValueError):self.m.s.gpu_ids(text)

    def test_reuse_bytes_remain_readonly_verified(self):
        folder=self.m.ROOT/'output';folder.mkdir(exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=folder,delete=False) as f:p=Path(f.name)
        try:
            p.write_text('frozen');files={str(p):self.m.s.sha(p)}
            self.m.s.verify_files(files);p.write_text('changed')
            with self.assertRaises(ValueError):self.m.s.verify_files(files)
        finally:p.unlink()


if __name__=='__main__':unittest.main()
