import importlib
from pathlib import Path
import unittest


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.r=importlib.import_module('tools.v102_totalseg_oom_recovery')

    def test_requires_drained_original(self):
        good=dict(pending=0,running={},failed=[{'name':'bad'}])
        self.r.require_drained(good,[])
        for state,processes in [(dict(good,pending=1),[]),(dict(good,running={'0':{}}),[]),(good,['123'])]:
            with self.assertRaises(ValueError):self.r.require_drained(state,processes)

    def test_exact_reuse_and_pending_partition(self):
        jobs=[dict(name=str(i),task_count=1 if i==0 else (108 if i<7 else 1)) for i in range(58)]
        expected=sum(j['task_count'] for j in jobs)
        v=self.r.partition(jobs,{str(i) for i in range(52)},expected)
        self.assertEqual(v['reused_shards'],52)
        self.assertEqual(v['pending_shards'],6)
        self.assertEqual(v['reused_tasks']+v['new_tasks'],expected)
        with self.assertRaises(ValueError):self.r.partition(jobs,{'not-a-job'},expected)
        with self.assertRaises(ValueError):self.r.partition(jobs,{'0'},expected+1)
        with self.assertRaises(ValueError):self.r.partition(jobs[:-1]+[jobs[0]],{'0'},expected)

    def test_readonly_reuse_sha_guard(self):
        p=Path(__file__);before=p.read_bytes()
        self.r.verify_files({str(p):self.r.sha(p)})
        with self.assertRaises(ValueError):self.r.verify_files({str(p):'0'*64})
        self.assertEqual(p.read_bytes(),before)

    def test_rejects_unauthorized_gpu(self):
        self.assertEqual(self.r.gpu_ids('2,4,6'),[2,4,6])
        for text in ('1','7','2,2',''):
            with self.assertRaises(ValueError):self.r.gpu_ids(text)

    def test_configuration_does_not_modify_original_signature(self):
        native=self.r.f.signature
        self.r.configure()
        self.assertIs(self.r.f.signature,native)
        self.assertIs(self.r.g.verify,self.r.verify)
        self.assertIs(self.r.g.amos.merge,self.r.merge)

    def test_every_decode_stage_and_floor_guard(self):
        good=dict(dynamic_floor=.5,memory_group=1,actual_decode_stages=[dict(memory_group=1,threshold_min=.5)])
        self.r.validate_stage(good)
        for bad in (dict(good,dynamic_floor=.6),dict(good,memory_group=4),dict(good,actual_decode_stages=[]),
                    dict(good,actual_decode_stages=[dict(memory_group=1,threshold_min=.35)])):
            with self.assertRaises(ValueError):self.r.validate_stage(bad)


if __name__=='__main__':unittest.main()
