import unittest
from tools.five_source_prompt_type_anygpu_queue import select_gpu, restore_pending, next_unused_attempt
import tempfile
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
from tools import five_source_prompt_type_anygpu_queue as queue

class RoutingTests(unittest.TestCase):
    def test_any_free_gpu_including_five_and_eight_card_scan(self):
        rows={0:dict(free=4000),3:dict(free=5000),5:dict(free=35000),7:dict(free=48000)}
        self.assertEqual(select_gpu({},rows,[],100),7)
        self.assertEqual(select_gpu({},rows,[7],100),5)
        self.assertIsNone(select_gpu(dict(minimum_free_mib=45000),rows,[7],100))
        self.assertEqual(select_gpu(dict(avoid_gpus={'7':200}),rows,[],100),5)

    def test_preserve_completed_and_resume_same_batch_after_resource_wait(self):
        a=dict(source='magic',case_id='a',variant='control',frame_batch=4,attempt=0)
        b=dict(source='kits23',case_id='b',variant='control',frame_batch=1,attempt=25,gpu=3,minimum_free_mib=45000,avoid_gpus={'0':800})
        pending=restore_pending([a,b],{('magic','a','control')},[dict(b,returncode=75)],lambda j:'resource wait',100)
        self.assertEqual(len(pending),1)
        self.assertEqual((pending[0]['frame_batch'],pending[0]['attempt']),(1,26))
        self.assertEqual(pending[0]['minimum_free_mib'],45000)
        self.assertEqual(pending[0]['avoid_gpus'],{'0':800})

    def test_unknown_failure_is_not_blindly_retried(self):
        job=dict(source='magic',case_id='a',variant='all',frame_batch=4,attempt=0)
        with self.assertRaises(ValueError):
            restore_pending([job],set(),[dict(job,returncode=1)],lambda j:'bad input',100)

    def test_new_control_does_not_reuse_prior_attempt_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);path=out/'jobs'/'magic__case_a__control__b2__a0'
            path.mkdir(parents=True)
            job=dict(source='magic',case_id='a',variant='control',control=True,frame_batch=2,attempt=0)
            actual=next_unused_attempt(out,job)
            self.assertEqual(actual['attempt'],1)
            self.assertTrue(path.is_dir())

    def test_dispatch_scans_all_cards_but_caps_own_concurrency_at_four(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);calls=[];concurrent=[0,0]
            jobs=[dict(source='magic',case_id=str(n),variant='control',control=True,frame_batch=4,attempt=0) for n in range(5)]
            def spawn(argv,**kwargs):
                import json
                actual=json.loads(Path(argv[-1]).read_text());calls.append(actual)
                concurrent[0]+=1;concurrent[1]=max(concurrent)
                def poll():concurrent[0]-=1;return 0
                return SimpleNamespace(pid=100+len(calls),poll=poll)
            with patch.object(queue.overlay,'install',return_value=dict(protocol_sha256='a',policy_sha256='b')), \
                 patch.object(queue.overlay,'read_policy'),patch.object(queue,'recovery',return_value=(jobs,[],[],jobs)), \
                 patch.object(queue,'completed_job',return_value=True),patch.object(queue.old,'passed_batches',return_value=set()), \
                 patch.object(queue.old,'resources',side_effect=lambda gpus:{g:dict(free=81000-g,uuid='GPU-'+str(g)) for g in gpus}), \
                 patch.object(queue.shutil,'disk_usage',return_value=SimpleNamespace(free=10*1024**3)), \
                 patch.object(queue.subprocess,'Popen',side_effect=spawn),patch.object(queue.time,'sleep'):
                queue.run_phase(out,'controls')
            self.assertEqual(len(calls),5)
            self.assertEqual(concurrent[1],4)
            self.assertEqual([j['gpu'] for j in calls[:4]],[0,1,2,3])

if __name__=='__main__':unittest.main()
