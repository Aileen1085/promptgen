import importlib
from pathlib import Path
from tempfile import NamedTemporaryFile
import unittest


class ResourceRecoveryTests(unittest.TestCase):
    def setUp(self): self.m = importlib.import_module('tools.bone_floor_resource_recovery')

    def jobs(self):
        smoke = [dict(name='smoke'+str(i), smoke=True, task_count=1) for i in range(18)]
        full = [dict(name='full'+str(i), smoke=False, task_count=114 if i==0 else 96) for i in range(144)]
        return smoke+full

    def test_resource_recovery_excludes_contended_and_forbidden_gpus(self):
        self.assertEqual(self.m.gpu_ids('2,4,5,6'), [2,4,5,6])
        for value in ('0', '3', '1', '7', '2,4,0', '', '2,2'):
            with self.assertRaises(ValueError): self.m.gpu_ids(value)

    def test_never_prepare_or_launch_while_old_workers_are_draining(self):
        state = dict(phase='full', running={}, failed=[dict(name='full1')], pending=142, done_in_phase=1)
        self.m.require_drained(state, [])
        for patch, processes in [({}, ['live_worker']), ({'running': {'2': {}}}, []),
                                  ({'failed': []}, []), ({'phase': 'smoke'}, [])]:
            with self.assertRaises(ValueError): self.m.require_drained({**state, **patch}, processes)

    def test_partition_reuses_only_complete_audited_names(self):
        complete = {'smoke'+str(i) for i in range(18)} | {'full0'}
        part = self.m.partition(self.jobs(), complete)
        self.assertEqual(part, dict(reused_jobs=19, pending_jobs=143, reused_tasks=132,
                                    new_tasks=13728, total_tasks=13860))
        remaining = self.m.pending_jobs(self.jobs(), complete, smoke=False)
        self.assertEqual(len(remaining), 143)
        self.assertNotIn('full0', {j['name'] for j in remaining})

    def test_partition_rejects_incomplete_smokes_and_wrong_total(self):
        complete = {'smoke'+str(i) for i in range(18)}
        for jobs, names in [(self.jobs(), complete-{'smoke0'}),
                             (self.jobs(), complete|{'unknown'}),
                             (self.jobs()[:-1], complete),
                             (self.jobs()+[self.jobs()[0]], complete)]:
            with self.assertRaises(ValueError): self.m.partition(jobs, names)

    def test_terminal_state_must_account_for_all_failed_pending_and_completed_jobs(self):
        jobs = self.jobs(); complete = {'smoke'+str(i) for i in range(18)}|{'full0'}
        state = dict(phase='full', running={}, failed=[dict(name='full1')], pending=142, done_in_phase=1)
        self.m.validate_state_partition(state, jobs, complete)
        for patch in ({'pending': 141}, {'done_in_phase': 2}, {'failed': [dict(name='full0')]}):
            with self.assertRaises(ValueError): self.m.validate_state_partition({**state, **patch}, jobs, complete)

    def test_process_guard_includes_all_three_entries_without_self_or_status(self):
        for script in ('bone_floor_grid.py','bone_floor_recovery.py','bone_floor_resource_recovery.py'):
            self.assertTrue(self.m.process_matches(['python',script,'launch'],2,1))
            self.assertTrue(self.m.process_matches(['python',script,'worker'],2,1))
            self.assertFalse(self.m.process_matches(['python',script,'status'],2,1))
            self.assertFalse(self.m.process_matches(['python',script,'launch'],1,1))
        self.assertTrue(self.m.process_matches(['python','bone_floor_resource_recovery.py','wait-launch'],2,1))

    def test_resource_recovery_preserves_old_reduced_batches_and_disjoint_attempts(self):
        latest=dict(attempt=8,oom_retries=4,frame_batch=1)
        limits=self.m.resume_settings(latest)
        self.assertEqual(limits,dict(attempt=9,oom_retries=4,frame_batch=1))
        self.m.validate_resume(limits,limits)
        for patch in (dict(frame_batch=16),dict(oom_retries=0),dict(attempt=0)):
            with self.assertRaises(ValueError):self.m.validate_resume({**limits,**patch},limits)

    def test_canary_must_be_fresh_one_task_baseline_at_same_floor_in_new_output(self):
        out=self.m.ROOT/'output/test_resource'; spec=dict(name='smoke',run=str(out/'jobs/smoke_a0'),
            model='v9_2',source='bone',variant='baseline',floor=.35,smoke=True,task_count=1,gpu=2)
        self.m.validate_canary_spec(spec,out,'v9_2')
        for patch in (dict(run=str(self.m.ROOT/'output/old/jobs/smoke_a0')), dict(task_count=2),
                      dict(floor=.40),dict(gpu=0),dict(model='v10_2'),dict(smoke=False)):
            with self.assertRaises(ValueError): self.m.validate_canary_spec({**spec,**patch},out,'v9_2')

    def test_readonly_reuse_bytes_cannot_change(self):
        folder=self.m.ROOT/'output'; folder.mkdir(exist_ok=True)
        with NamedTemporaryFile(dir=folder,delete=False) as f: q=Path(f.name)
        try:
            q.write_text('frozen'); files={str(q):self.m.sha(q)}; self.m.verify_files(files)
            q.write_text('changed')
            with self.assertRaises(ValueError): self.m.verify_files(files)
        finally: q.unlink()


if __name__=='__main__': unittest.main()
