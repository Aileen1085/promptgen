import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import tools
from tools import five_source_fg_bg_scribble_queue as queue
from tools.five_source_fg_bg_scribble_queue import select_gpu, retry_job, add_batch_controls, align_initial_batches, apply_retry_cap, allocate_attempt


class QueueTests(unittest.TestCase):
    def test_all_eight_cards_largest_adequate_free_and_busy_exclusion(self):
        rows={0:dict(free=8000),3:dict(free=6000),5:dict(free=35000),7:dict(free=80000)}
        self.assertEqual(select_gpu({},rows,[],100),7)
        self.assertEqual(select_gpu({},rows,[7],100),5)
        self.assertIsNone(select_gpu(dict(minimum_free_mib=45000),rows,[7],100))
        self.assertEqual(select_gpu(dict(avoid_gpus={'7':200}),rows,[],100),5)

    def test_real_oom_and_resource_wait_keep_required_semantics(self):
        job=dict(source='magic',case_id='a',frame_batch=4,attempt=2,gpu=5)
        retry=retry_job(job,1,'torch.OutOfMemoryError: CUDA out of memory',100)
        self.assertEqual((retry['frame_batch'],retry['attempt']),(2,3))
        wait=retry_job(job,75,'',100)
        self.assertEqual((wait['frame_batch'],wait['attempt']),(4,3))
        last=retry_job(dict(job,frame_batch=1),1,'CUDA out of memory',100)
        self.assertEqual((last['minimum_free_mib'],last['not_before'],last['avoid_gpus']['5']),(45000,160,700))
        self.assertIsNone(retry_job(job,1,'wrong source',100))

    def test_add_one_actual_matching_batch_control_and_preserve_pending_jobs(self):
        jobs=[dict(source='magic',case_id='a',frame_batch=2),dict(source='magic',case_id='b',frame_batch=2)]
        templates=[dict(source='magic',case_id='control',control=True,frame_batch=4)]
        result=add_batch_controls(jobs,templates,set(),[])
        self.assertEqual(len(result),3)
        self.assertTrue(result[0]['control']);self.assertEqual(result[0]['frame_batch'],2)
        self.assertEqual(len(add_batch_controls(jobs,templates,{('magic',2)},[])),2)
        self.assertEqual(len(add_batch_controls(jobs,templates,set(),result[:1])),2)

    def test_initial_case_batch_follows_parent_and_successful_source_control(self):
        jobs=[dict(source='magic',frame_batch=2),dict(source='magic',frame_batch=4),dict(source='kits23',frame_batch=1)]
        self.assertEqual([j['frame_batch'] for j in align_initial_batches(jobs,{'magic':4,'kits23':1})],[2,4,1])
        self.assertEqual([j['frame_batch'] for j in align_initial_batches(jobs,{'magic':2,'kits23':1})],[2,2,1])
        self.assertEqual(jobs[1]['frame_batch'],4)

    def test_late_high_batch_worker_oom_cannot_recreate_failed_higher_control(self):
        caps={}
        control=dict(source='magic',case_id='ctrl',variant='control',control=True,frame_batch=2,attempt=0,gpu=5)
        reduced=apply_retry_cap(control,retry_job(control,1,'CUDA out of memory',100),caps)
        self.assertEqual(reduced['frame_batch'],1)
        late=dict(source='magic',case_id='late',variant='all',frame_batch=4,attempt=0,gpu=7)
        reduced=apply_retry_cap(late,retry_job(late,1,'CUDA out of memory',110),caps)
        self.assertEqual(reduced['frame_batch'],1)
        wait=apply_retry_cap(late,retry_job(late,75,'',110),caps)
        self.assertEqual(wait['frame_batch'],4)

    def test_dynamic_control_attempt_numbers_never_reuse_older_attempts(self):
        control=dict(source='magic',case_id='ctrl',variant='control',control=True,frame_batch=2,attempt=0)
        history=[dict(control,attempt=0),dict(control,frame_batch=1,attempt=1)]
        next_job=allocate_attempt(control,history,[])
        self.assertEqual(next_job['attempt'],2)
        self.assertEqual(control['attempt'],0)

    def simulate_controls(self,fail_first=False):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);calls=[];concurrent=[0,0]
            sources=['magic','msd_task07','msd_task10','parse2022','kits23']
            jobs=[dict(source=s,case_id='a',variant='control',control=True,frame_batch=4,attempt=0) for s in sources]
            inputs={n:'a' for n in ['tools/five_source_fg_bg_scribble_queue.py','tools/five_source_fg_bg_scribble_eval.py','tools/five_source_fg_bg_scribble_runtime.py']}
            (out/'protocol.json').write_text(json.dumps(dict(inventory=dict(input_sha256=inputs),parent_actual_batches={})))
            (out/'control_jobs.json').write_text(json.dumps(jobs))
            (out/'jobs.json').write_text(json.dumps(jobs))
            def read(path):
                if path.name=='complete.json':return dict(passed=True,protocol_sha256='a',actual_frame_batch=4,normal_finish=dict(successful_native_completion=True))
                return json.loads(path.read_text())
            fake=SimpleNamespace(read=read,sha=lambda p:'a',SOURCES=sources,case_name=lambda s,c:s+'__'+c,
                                 validate_inventory=lambda inv:[],case_jobs=lambda tasks,batches:(jobs,jobs),validate_protocol=lambda p:None)
            def spawn(argv,**kwargs):
                actual=json.loads(Path(argv[-1]).read_text());calls.append(actual)
                concurrent[0]+=1;concurrent[1]=max(concurrent)
                rc=1 if fail_first and len(calls)==1 else 0
                def poll():concurrent[0]-=1;return rc
                return SimpleNamespace(pid=100+len(calls),poll=poll)
            with patch.object(tools,'five_source_fg_bg_scribble_eval',fake,create=True), \
                 patch.object(queue,'resources',side_effect=lambda gpus:{g:dict(free=81000-g,uuid='GPU-'+str(g)) for g in gpus}), \
                 patch.object(queue.shutil,'disk_usage',return_value=SimpleNamespace(free=10*1024**3)), \
                 patch.object(queue.subprocess,'Popen',side_effect=spawn),patch.object(queue.time,'sleep'):
                if fail_first:
                    with self.assertRaises(RuntimeError):queue.run_phase(out,'controls')
                else:queue.run_phase(out,'controls')
            state=json.loads((out/'queue_controls.json').read_text())
            return calls,concurrent,state

    def test_all_gpu_dispatch_keeps_concurrency_four(self):
        calls,concurrent,state=self.simulate_controls()
        self.assertEqual((len(calls),concurrent[1],concurrent[0]),(5,4,0))
        self.assertEqual(len({j['gpu'] for j in calls[:4]}),4)
        self.assertEqual(state['phase'],'complete')

    def test_non_oom_stops_new_dispatch_and_naturally_drains_own_workers(self):
        calls,concurrent,state=self.simulate_controls(fail_first=True)
        self.assertEqual((len(calls),concurrent[0]),(4,0))
        self.assertEqual(state['phase'],'failed')
        self.assertEqual((state['pending'],len(state['failed']),state['running']),(1,1,[]))


if __name__=='__main__':unittest.main()
