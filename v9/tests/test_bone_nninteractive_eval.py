import importlib.util
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]


def tasks():
    rows=[]
    for case in range(20):
        labels=list(range(1,39))
        if case==0: labels+=list(range(39,48))
        if case==1: labels+=[39]
        rows.extend(dict(source_name='bone',case_id=f'{case+1:04d}_phase_3',
                         global_class_id=10000+c,local_class_id=c,class_name=str(c),
                         image='case.nii.gz',label='label.nii.gz',split_role='validation') for c in labels)
    return rows


class BoneNNRunnerTests(unittest.TestCase):
    def setUp(self):
        path=ROOT/'tools/bone_nninteractive_eval.py'
        self.assertTrue(path.is_file(),'new Bone single-round runner is missing')
        spec=importlib.util.spec_from_file_location('bone_nn_runner_test',path)
        self.m=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.m)

    def completed(self,task):
        return dict(task=task,interaction=dict(prediction_trigger_count=1,prompt_addition_count=6,
                    foreground_scribbles=2,background_points=4),
                    metric_execution=dict(comparisons=1,max_error=0.,native_compared=True))

    @contextmanager
    def simulated_scheduler(self,jobs,advance):
        """Run the actual loop against an in-memory filesystem and process clock."""
        sim=SimpleNamespace(tick=0,records={},writes={},states=[],processes=[],logs={},active={})
        out=ROOT/'output'/'unused_scheduler_fixture';owner=self
        def read(path):
            owner.assertEqual(Path(path),out/'jobs.json')
            return deepcopy(jobs)
        def new(path,value):
            path=Path(path)
            if path in sim.writes:raise FileExistsError(path)
            sim.writes[path]=deepcopy(value)
        class Process:
            def __init__(self,argv,**kwargs):
                self.job=deepcopy(sim.writes[Path(argv[-1])]);self.pid=100+len(sim.processes)
                self.returncode=None;self.started=sim.tick;self.poll_ticks=[];self.handle=kwargs['stdout']
                owner.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],str(self.job['gpu']))
                owner.assertFalse(any(p.returncode is None and p.job['gpu']==self.job['gpu'] for p in sim.processes))
                sim.processes.append(self)
            def poll(self):self.poll_ticks.append(sim.tick);return self.returncode
            def wait(self):raise AssertionError('the scheduler must naturally poll active peers to exit')
            def terminate(self):raise AssertionError('worker signals are forbidden')
            kill=terminate
            send_signal=terminate
        def sleep(_seconds):
            sim.tick+=1
            owner.assertLess(sim.tick,10,'scheduler failed to finish the simulated timeline')
            advance(sim)
        with ExitStack() as stack:
            for target,kwargs in (
                ('read',dict(side_effect=read)),('new',dict(side_effect=new)),
                ('state_write',dict(side_effect=lambda _out,value:sim.states.append(deepcopy(value)))),
                ('class_records',dict(side_effect=lambda *_args:dict(sim.records))),
                ('gpu_snapshot',dict(return_value={g:dict(uuid=f'GPU-{g}',free=50000) for g in self.m.ALLOWED})),
            ):stack.enter_context(patch.object(self.m,target,**kwargs))
            stack.enter_context(patch.object(self.m.subprocess,'Popen',side_effect=Process))
            stack.enter_context(patch.object(self.m.time,'time',side_effect=lambda:1000.+60.*sim.tick))
            stack.enter_context(patch.object(self.m.time,'sleep',side_effect=sleep))
            stack.enter_context(patch.object(Path,'mkdir'))
            stack.enter_context(patch.object(Path,'open',side_effect=lambda *_args,**_kwargs:StringIO()))
            stack.enter_context(patch.object(Path,'read_text',autospec=True,side_effect=lambda path,**_kwargs:sim.logs.get(str(path),'')))
            yield sim,out

    def finish(self,sim,process):
        for task in process.job['tasks']:sim.records[self.m.key(task)]=self.completed(task)
        process.returncode=0

    def test_frozen_scope_and_original_local_ids(self):
        rows=tasks();self.assertEqual(len(self.m.validate_tasks(rows)),770)
        rows[0]['local_class_id']=74
        with self.assertRaisesRegex(ValueError,'label|local|mapping'):self.m.validate_tasks(rows)

    def test_scope_rejects_duplicate_or_omission(self):
        with self.assertRaises(ValueError):self.m.validate_tasks(tasks()[:-1])
        with self.assertRaises(ValueError):self.m.validate_tasks(tasks()+tasks()[:1])

    def test_jobs_keep_cases_whole_and_cover_exactly(self):
        jobs=self.m.plan_jobs(tasks())
        self.assertEqual(len(jobs),20)
        keys=[self.m.key(t) for j in jobs for t in j['tasks']]
        self.assertEqual(len(keys),len(set(keys)))
        self.assertEqual(len(keys),770)
        self.assertTrue(all(len({t['case_id'] for t in j['tasks']})==1 for j in jobs))

    def test_actual_single_round_evidence_required(self):
        good=dict(prediction_trigger_count=1,prompt_addition_count=6,
                  foreground_scribbles=2,background_points=4)
        self.m.validate_interaction(good)
        for field,bad in [('prediction_trigger_count',2),('prompt_addition_count',5),
                          ('foreground_scribbles',1),('background_points',2)]:
            with self.subTest(field=field),self.assertRaises(ValueError):
                self.m.validate_interaction(dict(good,**{field:bad}))

    def test_merge_same_complete_task_set_and_ten_metrics(self):
        rows=[dict(t,label_id=t['local_class_id'],**{n:.5 for n in self.m.METRICS}) for t in tasks()]
        result=self.m.merge_rows(rows,tasks())
        self.assertEqual(result['task_count'],770)
        self.assertEqual(result['mean']['dice'],.5)
        self.assertEqual(result['class_macro_dice'],.5)
        with self.assertRaises(ValueError):self.m.merge_rows(rows[:-1],tasks())
        with self.assertRaises(ValueError):self.m.merge_rows(rows+rows[:1],tasks())
        rows[0]['mean_slice_nsd_3mm']=float('nan')
        with self.assertRaisesRegex(ValueError,'metric|finite'):self.m.merge_rows(rows,tasks())

    def test_resource_retry_preserves_case_and_increments_attempt(self):
        job=dict(name='one',tasks=tasks()[:2],attempt=2,minimum_free_mib=30720,gpu=4)
        next_job=self.m.retry_job(job,oom=True,now=100.)
        self.assertEqual(next_job['attempt'],3)
        self.assertEqual(next_job['tasks'],job['tasks'])
        self.assertEqual(next_job['minimum_free_mib'],45000)
        self.assertEqual(next_job['not_before'],160.)
        self.assertEqual(next_job['avoid_gpu_until'],700.)
        same=self.m.retry_job(job,oom=False,now=100.)
        self.assertEqual(same['minimum_free_mib'],30720)
        self.assertEqual(job['attempt'],2)

    def test_paired_source_manifest_must_match_original_bone_hash(self):
        self.assertTrue(hasattr(self.m,'DATA_HASHES'),'original paired manifest hash guard missing')
        with patch.object(self.m,'sha',return_value='tampered'):
            with self.assertRaisesRegex(ValueError,'paired.*changed'):
                self.m.dataset_files()

    def test_first_gate_needs_observed_trigger_and_native_metric_comparison(self):
        self.assertTrue(hasattr(self.m,'gate_ready'),'real single-round gate helper missing')
        record=dict(interaction=dict(prediction_trigger_count=1,prompt_addition_count=6,
                                     foreground_scribbles=2,background_points=4),
                    metric_execution=dict(comparisons=1,max_error=0.,native_compared=True))
        self.assertTrue(self.m.gate_ready({('bone','case',10001):record}))
        self.assertFalse(self.m.gate_ready({}))
        record['metric_execution']['comparisons']=0
        with self.assertRaises(ValueError):self.m.gate_ready({'one':record})

    def test_gate_selects_the_actual_native_compared_prediction(self):
        interaction=dict(prediction_trigger_count=1,prompt_addition_count=6,foreground_scribbles=2,background_points=4)
        later=dict(interaction=interaction,metric_execution=dict(comparisons=1,max_error=0.,native_compared=False))
        first=dict(interaction=interaction,metric_execution=dict(comparisons=1,max_error=0.,native_compared=True))
        self.assertIs(self.m.gate_ready(dict(later=later,first=first)),first)

    def test_prompt_bytes_match_old_signed_budget_not_a_fresh_hash_only(self):
        self.assertTrue(hasattr(self.m,'frozen_prompt_hashes'),'signed original prompt binding missing')
        rows=tasks();refs=[dict(t,prompt_cache_name=t['case_id']+'_'+str(t['global_class_id'])+'.npz') for t in rows]
        budget=dict(tasks=770,per_case_class=[dict(key=list(self.m.key(t)),prompt_sha256='a'*64) for t in rows])
        self.assertEqual(len(self.m.frozen_prompt_hashes(rows,refs,budget,lambda n:'a'*64)),770)
        with self.assertRaisesRegex(ValueError,'prompt|signature'):
            self.m.frozen_prompt_hashes(rows,refs,budget,lambda n:'b'*64)

    def test_controller_error_waits_for_own_started_workers_without_signals(self):
        self.assertTrue(hasattr(self.m,'drain_workers'),'natural controller drain missing')
        events=[]
        class Process:
            def wait(self):events.append('wait')
        class Handle:
            def close(self):events.append('close')
        active={2:(Process(),{},Handle()),4:(Process(),{},Handle())}
        self.m.drain_workers(active)
        self.assertEqual(active,{})
        self.assertEqual(events,['wait','close','wait','close'])

    def test_log_close_error_does_not_skip_waiting_for_other_workers(self):
        events=[]
        class Process:
            def wait(self):events.append('wait')
        class Broken:
            def close(self):events.append('broken');raise OSError('test close')
        class Good:
            def close(self):events.append('close')
        active={2:(Process(),{},Broken()),4:(Process(),{},Good())}
        errors=self.m.drain_workers(active)
        self.assertEqual(active,{})
        self.assertEqual(events,['wait','broken','wait','close'])
        self.assertTrue(errors)

    def test_complete_marker_is_exclusive_fully_written_publication(self):
        self.assertTrue(hasattr(self.m,'atomic_new'),'atomic complete marker publication missing')
        folder=Path(tempfile.mkdtemp(prefix='bone-nn-marker-',dir=ROOT/'output'))
        path=folder/'one.done.json';payload=dict(values=list(range(1000)))
        self.m.atomic_new(path,payload)
        self.assertEqual(self.m.read(path),payload)
        with self.assertRaises(FileExistsError):self.m.atomic_new(path,dict(replacement=True))
        self.assertEqual(self.m.read(path),payload)

    def test_successful_exit_uses_fresh_complete_records(self):
        self.assertTrue(hasattr(self.m,'remaining_after_exit'),'post-exit complete proof refresh missing')
        task=tasks()[0]
        with patch.object(self.m,'class_records',return_value={self.m.key(task):dict(task=task)}) as observe:
            remaining,records=self.m.remaining_after_exit(None,{},dict(tasks=[task]))
        self.assertEqual(remaining,[])
        observe.assert_called_once()

    def test_scheduler_dispatches_one_case_until_real_compared_record_then_peers(self):
        jobs=self.m.plan_jobs(tasks())[:4]
        def advance(sim):
            if sim.tick in (1,2):
                self.assertEqual(len(sim.processes),1)
                self.assertFalse(sim.states[-1]['single_round_gate'])
            if sim.tick==2:
                task=sim.processes[0].job['tasks'][0]
                sim.records[self.m.key(task)]=self.completed(task)
            if sim.tick==3:
                self.assertEqual(len(sim.processes),4)
                for process in sim.processes:self.finish(sim,process)
        with self.simulated_scheduler(jobs,advance) as (sim,out):
            self.m._schedule_loop(out,{},sim.active)
            self.assertEqual([p.started for p in sim.processes],[0,2,2,2])
            self.assertEqual([p.job['gpu'] for p in sim.processes],[2,4,5,6])
            self.assertTrue(sim.writes[out/'single_round_gate.json']['metric_execution']['native_compared'])
            self.assertEqual(sim.states[-1]['pending'],0)
            self.assertEqual(sim.active,{})

    def test_scheduler_resource_exit_retries_only_unfinished_classes_in_new_attempt(self):
        for code,message,minimum in ((75,'resource wait',30720),(1,'torch.OutOfMemoryError: CUDA out of memory',45000)):
            with self.subTest(returncode=code):
                job=self.m.plan_jobs(tasks())[0];job['tasks']=job['tasks'][:2]
                committed=self.completed(job['tasks'][0])
                def advance(sim):
                    if sim.tick==1:
                        first=sim.processes[0];first.returncode=code
                        sim.records[self.m.key(job['tasks'][0])]=committed
                        sim.logs[first.job['log']]=message
                    if sim.tick==3:
                        self.assertEqual(len(sim.processes),2)
                        self.finish(sim,sim.processes[1])
                with self.simulated_scheduler([job],advance) as (sim,out):
                    self.m._schedule_loop(out,{},sim.active)
                    first,retry=sim.processes
                    self.assertEqual([first.job['attempt'],retry.job['attempt']],[0,1])
                    self.assertEqual(retry.started,2,'resource backoff was skipped')
                    self.assertEqual(sim.states[1]['pending'],1)
                    self.assertEqual(retry.job['tasks'],job['tasks'][1:])
                    self.assertEqual(retry.job['minimum_free_mib'],minimum)
                    self.assertNotEqual(first.job['gpu'],retry.job['gpu'])
                    for field in ('run','spec','log'):self.assertNotEqual(first.job[field],retry.job[field])
                    self.assertEqual(sim.writes[Path(first.job['spec'])],first.job)
                    self.assertIs(sim.records[self.m.key(job['tasks'][0])],committed)
                    self.assertEqual(sim.states[-1]['complete_tasks'],2)
                    self.assertEqual(sim.states[-1]['failed'],[])

    def test_scheduler_nonresource_failure_stops_dispatch_and_naturally_drains_peers(self):
        jobs=self.m.plan_jobs(tasks())[:6]
        def advance(sim):
            if sim.tick==1:
                task=sim.processes[0].job['tasks'][0]
                sim.records[self.m.key(task)]=self.completed(task)
            if sim.tick==2:
                self.assertEqual(len(sim.processes),4)
                failed=sim.processes[0];failed.returncode=1
                sim.logs[failed.job['log']]='ValueError: invalid source signature'
            if sim.tick in (3,4):
                self.assertEqual(len(sim.processes),4,'pending cases were dispatched after failure')
                self.assertTrue(sim.states[-1]['failed'])
            if sim.tick==4:
                for process in sim.processes[1:]:self.finish(sim,process)
        with self.simulated_scheduler(jobs,advance) as (sim,out):
            with self.assertRaisesRegex(RuntimeError,'non-resource failure.*drained'):
                self.m._schedule_loop(out,{},sim.active)
            self.assertEqual(sim.active,{})
            self.assertEqual(sim.states[-1]['pending'],2)
            self.assertEqual(sim.states[-1]['running'],{})
            self.assertEqual(len(sim.states[-1]['failed']),1)
            for process in sim.processes[1:]:
                self.assertEqual(process.poll_ticks[-3:],[2,3,4])
                self.assertTrue(process.handle.closed)

    def test_duplicate_controller_or_launch_marker_prevents_startup(self):
        for related,exception in ((['123'],RuntimeError),([],FileExistsError)):
            with self.subTest(related=related),patch.object(self.m,'verify',return_value={}), \
                    patch.object(self.m,'related',return_value=related), \
                    patch.object(self.m,'new',side_effect=FileExistsError('launched.json exists')) as publish, \
                    patch.object(self.m,'_schedule_loop') as loop:
                with self.assertRaises(exception):self.m.schedule(ROOT/'output'/'unused_scheduler_fixture')
                loop.assert_not_called()
                if related:publish.assert_not_called()
                else:self.assertEqual(Path(publish.call_args.args[0]).name,'launched.json')

    def test_worker_uuid_and_memory_guards_run_before_runtime_import(self):
        out=ROOT/'output'/'unused_scheduler_fixture';canonical=self.m.plan_jobs(tasks())[0]
        tag=canonical['name']+'_a0'
        job=dict(canonical,gpu=2,gpu_uuid='GPU-2',run=str(out/'runs'/tag),
                 spec=str(out/'job_specs'/(tag+'.json')),log=str(out/'logs'/(tag+'.log')))
        original_import=__import__
        def guarded_import(name,*args,**kwargs):
            if name=='tools.bone_nninteractive_runtime':raise AssertionError('runtime imported before GPU resource rejection')
            return original_import(name,*args,**kwargs)
        for gpu,exception in ((dict(uuid='GPU-wrong',free=50000),ValueError),
                              (dict(uuid='GPU-2',free=30719),SystemExit)):
            with self.subTest(gpu=gpu),patch.object(self.m,'verify',return_value={}), \
                    patch.object(self.m,'read',return_value=[canonical]), \
                    patch.object(self.m,'gpu_snapshot',return_value={2:gpu}), \
                    patch.object(Path,'mkdir',side_effect=AssertionError('attempt directory created before resource guard')), \
                    patch('builtins.__import__',side_effect=guarded_import):
                with self.assertRaises(exception) as caught:self.m.worker(out,job)
                if exception is SystemExit:self.assertEqual(caught.exception.code,75)

    def test_attempt_rejects_disallowed_gpu_or_memory_floor(self):
        out=ROOT/'output'/'unused_scheduler_fixture';canonical=self.m.plan_jobs(tasks())[0]
        tag=canonical['name']+'_a0'
        job=dict(canonical,gpu=2,gpu_uuid='GPU-2',run=str(out/'runs'/tag),
                 spec=str(out/'job_specs'/(tag+'.json')),log=str(out/'logs'/(tag+'.log')))
        with patch.object(self.m,'read',return_value=[canonical]):
            for changed in (dict(gpu=0),dict(gpu=3),dict(minimum_free_mib=30719)):
                with self.subTest(changed=changed),self.assertRaises(ValueError):
                    self.m.validate_attempt(dict(job,**changed),out)


if __name__=='__main__':unittest.main()
