import unittest
from tools.five_source_prompt_type_queue import retry_job, available_gpu, require_scope, add_batch_controls, lower_pending_batch


class QueueTests(unittest.TestCase):
    def test_real_oom_decreases_batch_and_retains_history(self):
        job=dict(source='magic', case_id='x', frame_batch=4, attempt=2, gpu=3)
        r=retry_job(job, 1, 'torch.OutOfMemoryError: CUDA out of memory', 100.)
        self.assertEqual((r['frame_batch'],r['attempt']), (2,3))
        self.assertEqual(job['frame_batch'],4)
        job.update(frame_batch=1)
        r=retry_job(job, 1, 'CUDA out of memory', 100.)
        self.assertEqual(r['minimum_free_mib'],45000)
        self.assertEqual(r['avoid_gpus']['3'],700.)
        self.assertIsNone(retry_job(job,1,'NFS failure',100.))

    def test_resource_wait_is_same_batch_and_selection_scans_other_cards(self):
        j=dict(frame_batch=4, attempt=0, gpu=0)
        r=retry_job(j,75,'CUDA out of memory is old text',100.)
        self.assertEqual(r['frame_batch'],4)
        rows={0:dict(free=81000,uuid='a'),3:dict(free=72000,uuid='b')}
        r.update(avoid_gpus={'0':1000.},not_before=0)
        self.assertEqual(available_gpu(r,rows,[],100.),3)
        self.assertIsNone(available_gpu(r,rows,[3],100.))

    def test_scope_never_silently_includes_unrequested_sources(self):
        self.assertEqual(require_scope([0,3]),[0,3])
        for values in ([0,0],[1],[7],[]):
            with self.assertRaises(ValueError):require_scope(values)

    def test_oom_fallback_schedules_one_matching_control_before_work(self):
        pending=[dict(source='magic',frame_batch=2,variant='baseline',case_id='pancreas/13')]
        templates=[dict(source='magic',frame_batch=4,variant='control',control=True,case_id='first',tasks=[])]
        expanded=add_batch_controls(pending,templates,set(),[])
        self.assertTrue(expanded[0]['control'])
        self.assertEqual(expanded[0]['frame_batch'],2)
        self.assertEqual(len(add_batch_controls(expanded,templates,set(),[])),2)
        self.assertEqual(len(add_batch_controls(pending,templates,{('magic',2)},[])),1)
        lowered=lower_pending_batch(expanded,'magic',2,1)
        self.assertEqual(lowered[-1]['frame_batch'],1)


if __name__=='__main__':unittest.main()
