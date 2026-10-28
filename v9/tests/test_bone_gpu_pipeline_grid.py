import ast
import copy
from contextlib import contextmanager
import hashlib
import importlib
import json
import signal
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


@contextmanager
def scratch_directory():
    # Keep tiny diagnostic fixtures: project policy forbids bulk cleanup.
    yield tempfile.mkdtemp(prefix='bone_pipeline_test_')


class PipelineGridTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((Path(__file__).resolve().parents[1]/'tools/bone_gpu_pipeline_grid.py').exists(),
                        'new independent pipeline runner is required')
        self.m=importlib.import_module('tools.bone_gpu_pipeline_grid')
        self.task=dict(source_name='bone',case_id='a',global_class_id=10003,class_name='bone')

    def canonical(self,name='case',control=False):
        return dict(name=name,model='v9_2',control=control,parent_history=[],variant='baseline',
                    source='bone',case_id='a',prepared='/prepared',pending=[dict(task=self.task,floors=[.35])],
                    task_count=1,attempt=0,oom_retries=0,frame_batch=4 if control else 2,
                    minimum_free_mib=30720,not_before=0.,avoid_gpus={},avoid_gpu_until=0.,last_failed_gpu=None)

    def test_engine_independent_output_sources_and_exact_runtime_run(self):
        e=self.m.engine()
        self.assertTrue(e.OUT.endswith('20261028_cpugpu'))
        self.assertEqual(Path(e.__file__).name,'bone_gpu_pipeline_grid.py')
        self.assertIn('tools/bone_gpu_pipeline_runtime.py',e.SOURCES)
        self.assertIn('tools/bone_gpu_pipeline_grid.py',e.SOURCES)
        self.assertIn('tools.bone_gpu_pipeline_runtime',e.run.__wrapped__.__code__.co_names)
        self.assertNotIn('tools.bone_metric_speed_runtime',e.run.__wrapped__.__code__.co_names)
        self.assertEqual(e.ALLOWED,(2,4,5,6))

    def test_run_configures_original_signed_job_before_native_run(self):
        seen=[];fake=types.SimpleNamespace(configure_job=lambda j:seen.append(('configure',j)))
        e=self.m.engine();original=self.canonical()
        with patch.dict('sys.modules',{'tools.bone_gpu_pipeline_runtime':fake}):
            wrapped=self.m.configured_run(lambda j:seen.append(('native',j)))
            wrapped(original)
        self.assertEqual(seen,[('configure',original),('native',original)])

    def test_parent_helpers_use_exact_metricfast_parent_and_unique_handoff(self):
        h=self.m.handoff_helpers()
        self.assertEqual(h.CURRENT_SCRIPT,'bone_metric_speed_grid.py')
        self.assertEqual(h.ENTRY,'bone_gpu_pipeline_grid.py')
        self.assertEqual(h.HANDOFF,'output/bone_cpugpu_handoff_20261028.json')
        self.assertEqual(h.STOPPED,'output/bone_cpugpu_handoff_stopped_20261028.json')
        self.assertEqual(h.DRAINED,'output/bone_cpugpu_handoff_drained_20261028.json')
        self.assertTrue(h.prior_engine().OUT.endswith('metricfast'))
        self.assertIn('output/bone_metricfast_pipeline_20261028.log',h.collect_parent.__code__.co_consts)
        self.assertIn('bone_cpugpu_checkpoint_stop_',h.finish_parent.__code__.co_consts)
        names=next(v for v in h.matching_processes.__code__.co_consts if isinstance(v,frozenset))
        self.assertTrue({'bone_shared_floor_probe_grid.py','bone_metric_speed_grid.py','bone_gpu_pipeline_grid.py'}<=names)

    def test_helper_source_and_run_ast_anchors_are_fail_closed(self):
        text=(self.m.ROOT/'tools/bone_metric_speed_grid.py').read_bytes().decode()
        self.assertEqual(hashlib.sha256(text.encode()).hexdigest(),self.m.PARENT_SHA)
        with self.assertRaisesRegex(ValueError,'frozen metric'):
            self.m.handoff_helpers(text+'\n# changed')
        e=self.m.prior_engine()
        with self.assertRaisesRegex(ValueError,'run.*anchor'):
            self.m.compile_pipeline_run(e,'def run(job):\n    return job\n')

    def test_scheduler_identity_targets_current_parent_not_ancestor(self):
        h=self.m.handoff_helpers();root=Path('/project')
        info=dict(pid=12,uid=1000,cwd=str(root),args=['python','tools/bone_metric_speed_grid.py','wait-launch'],starttime='900',state='T')
        h.validate_scheduler(info,root,1000)
        for changed in (dict(uid=1001),dict(cwd='/other'),dict(state='S'),
                        dict(args=['python','tools/bone_shared_floor_probe_grid.py','wait-launch']),
                        dict(args=['python','/other/bone_metric_speed_grid.py','wait-launch'])):
            with self.assertRaises(ValueError):h.validate_scheduler(dict(info,**changed),root,1000)

    def test_handoff_history_preserves_intentional_exit_and_real_oom_deadlines(self):
        h=self.m.handoff_helpers()
        spec=dict(frame_batch=2,oom_retries=1,attempt=3,gpu=4,minimum_free_mib=30720)
        stop=h.inherited_history(spec,'',100,signal.SIGTERM,True)
        self.assertEqual((stop['frame_batch'],stop['attempt'],stop['oom_retries']),(2,3,1))
        self.assertTrue(stop['intentional_checkpoint_stop'])
        retry=h.inherited_history(spec,'CUDA out of memory',100,256)
        self.assertEqual((retry['frame_batch'],retry['attempt'],retry['oom_retries']),(1,4,2))
        later=h.inherited_history(dict(spec,frame_batch=1),'CUDA out of memory',100,256)
        self.assertEqual((later['minimum_free_mib'],later['not_before'],later['avoid_gpu_until']),(45000,160,700))
        with self.assertRaises(ValueError):h.inherited_history(spec,'bad geometry',100,256)

    def test_attempt_signs_slot_before_atomic_write_and_cannot_overwrite(self):
        e=self.m.engine();job=self.canonical()
        with scratch_directory() as d:
            out=Path(d);available={2:dict(uuid='GPU-2',free=50000)}
            with patch.object(e,'class_records',return_value={}):
                actual=e.make_attempt(out,job,2,available,{},slot=1)
                self.assertEqual(actual['slot'],1)
                self.assertEqual(json.loads(Path(actual['spec']).read_text())['slot'],1)
                e.validate_attempt(actual,job,out)
                with self.assertRaisesRegex(ValueError,'already exist'):e.make_attempt(out,job,2,available,{},slot=2)
                for bad in (-1,3,True,'1'):
                    with self.assertRaises(ValueError):e.validate_attempt(dict(actual,slot=bad),job,out)
                for bad in (0,1,3,7):
                    with self.assertRaises(ValueError):e.validate_attempt(dict(actual,gpu=bad),job,out)
                with self.assertRaises(ValueError):e.validate_attempt(dict(actual,log='/other.log'),job,out)

    def test_attempt_paths_with_partial_run_or_log_are_never_reused(self):
        for kind in ('run','log'):
            e=self.m.engine();job=self.canonical()
            with scratch_directory() as d:
                out=Path(d);target=out/('runs' if kind=='run' else 'logs')/('case_a0'+('' if kind=='run' else '.log'))
                target.parent.mkdir();target.mkdir() if kind=='run' else target.write_text('partial')
                with patch.object(e,'class_records',return_value={}):
                    with self.assertRaises(ValueError):e.make_attempt(out,job,2,{2:dict(uuid='GPU-2')},{},slot=0)
                self.assertFalse((out/'job_specs/case_a0.json').exists())

    def test_host_slots_require_allowed_gpu_ram_and_preserve_resource_policy(self):
        e=self.m.engine();job=self.canonical();available={g:dict(free=50000,uuid=f'GPU-{g}') for g in e.ALLOWED}
        active={'2:0':object(),'2:1':object()};ram=128*1024**3
        self.assertTrue(self.m.resource_eligible(e,job,2,2,available,100,active,ram))
        self.assertFalse(self.m.resource_eligible(e,job,2,0,available,100,active,ram))
        self.assertFalse(self.m.resource_eligible(e,job,2,3,available,100,active,ram))
        self.assertFalse(self.m.resource_eligible(e,job,0,0,available,100,active,ram))
        self.assertFalse(self.m.resource_eligible(e,job,2,2,available,100,active,ram-1))
        for changed in (dict(not_before=101),dict(avoid_gpus={'2':101}),dict(minimum_free_mib=45000)):
            snapshot=copy.deepcopy(available);snapshot[2]['free']=44000
            self.assertFalse(self.m.resource_eligible(e,dict(job,**changed),2,2,snapshot,100,active,ram))

    def test_class_records_validate_actual_marker_attempt_including_controls(self):
        e=self.m.engine();calls=[];out=Path('/independent')
        actual=dict(slot=2,gpu=4);record={'pipeline_execution':{}}
        native=lambda q,control=False:{('v9_2',('bone','a',10003)):dict(marker=str(q/'runs/case_a7/classes/a.done.json'),record=record)}
        fake=types.SimpleNamespace(validate_pipeline_record=lambda r,j:calls.append((r,j)))
        wrapped=self.m.checked_class_records(e,native)
        with patch.dict('sys.modules',{'tools.bone_gpu_pipeline_runtime':fake}),patch.object(e.b,'read',return_value=actual) as read:
            wrapped(out,True)
        read.assert_called_once_with(out/'job_specs/case_a7.json')
        self.assertEqual(calls,[(record,actual)])

    def test_control_lease_exclusion_deduplicates_and_rejects_overlap_across_workers(self):
        def event(i,pid,start,end,**kw):return dict(id=i,pid=pid,gpu=2,start_monotonic=start,end_monotonic=end,nested=False,**kw)
        one=event('a',1,1.,2.);two=event('b',2,2.,3.)
        def item(start,events):return dict(record=dict(pipeline_execution=dict(startup_lease=start,native_probe_leases=[],raw_events=events)))
        records={'a':item(one,[one,dict(one,id='nested',nested=True,start_monotonic=1.2,end_monotonic=1.9)]),'b':item(two,[])}
        self.m.require_exclusive_leases(records)
        records['b']=item(event('b',2,1.9,3.),[])
        with self.assertRaisesRegex(ValueError,'overlap'):self.m.require_exclusive_leases(records)
        records['b']=item(event('b',2,1.9,3.,gpu_uuid='different'),[])
        with self.assertRaisesRegex(ValueError,'overlap'):self.m.require_exclusive_leases(records)

    def test_scheduler_starts_two_controls_on_same_gpu_independent_host_slots(self):
        e=self.m.engine();jobs=[self.canonical('control_a',True),self.canonical('control_b',True)];launched=[];states=[]
        class Proc:
            def __init__(self,job):self.job=job;self.pid=100+len(launched);self.returncode=None;self.polls=0
            def poll(self):
                self.polls+=1
                if self.polls>1:self.returncode=0
                return self.returncode
        with scratch_directory() as d:
            out=Path(d)
            def read(path):return dict(controls=jobs) if path.name=='jobs.json' else json.loads(path.read_text())
            def spawn(args,**kwargs):
                spec=json.loads(Path(args[-1]).read_text());proc=Proc(spec);launched.append(proc);return proc
            def records(q,control=False):return {(x.job['model'],e.p.key(t['task'])):dict(record={}) for x in launched if x.returncode==0 for t in x.job['pending']}
            with patch.object(e.b,'read',side_effect=read),patch.object(e.b,'write',side_effect=lambda p,v:states.append(v)),patch.object(e,'class_records',side_effect=records),patch.object(e.g,'gpu_snapshot',return_value={g:dict(free=60000,uuid=f'GPU-{g}') for g in e.ALLOWED}),patch.object(self.m,'host_available_bytes',return_value=2*1024**4),patch.object(e.subprocess,'Popen',side_effect=spawn),patch.object(e.time,'sleep'):
                e.schedule(out,True)
        self.assertEqual([(p.job['gpu'],p.job['slot']) for p in launched],[(2,0),(2,1)])
        running=next(s['running'] for s in states if len(s['running'])==2)
        self.assertEqual(set(running),{'2:0','2:1'})
        self.assertEqual({(v['gpu'],v['slot']) for v in running.values()},{(2,0),(2,1)})
        self.assertTrue(all(Path(p.job['spec']).name.startswith('control_') for p in launched))

    def scheduler_scenario(self,count,fatal=False,spawn_failure=False,proof_failure=False,output_failure=False):
        e=self.m.engine();jobs=[];launched=[];states=[];proof_errors=[]
        for i in range(count):
            job=self.canonical(f'case_{i}');job['pending'][0]['task']=dict(self.task,global_class_id=10003+i);jobs.append(job)
        class Proc:
            def __init__(self,job,index):self.job=job;self.pid=100+index;self.index=index;self.returncode=None;self.polls=0
            def poll(self):
                self.polls+=1
                if fatal and self.index==0:self.returncode=1
                elif self.polls>=2:self.returncode=0
                return self.returncode
        with scratch_directory() as d:
            out=Path(d)
            def read(path):return dict(cases=jobs) if path.name=='jobs.json' else json.loads(path.read_text())
            def spawn(args,**kwargs):
                if spawn_failure and len(launched)==2:raise OSError('cannot create host process')
                spec=json.loads(Path(args[-1]).read_text());proc=Proc(spec,len(launched));launched.append(proc);return proc
            def records(q,control=False):
                if proof_failure and len(launched)>=2:
                    proof_errors.append(len(launched))
                    raise ValueError('invalid class proof')
                return {(x.job['model'],e.p.key(t['task'])):dict(record={}) for x in launched if x.returncode==0 for t in x.job['pending']}
            def output(*args,**kwargs):
                if output_failure and len(launched)>=2:raise OSError('pipeline stdout failed')
            with patch.object(e.b,'read',side_effect=read),patch.object(e.b,'write',side_effect=lambda p,v:states.append(v)),patch.object(e,'class_records',side_effect=records),patch.object(e.g,'gpu_snapshot',return_value={g:dict(free=60000,uuid=f'GPU-{g}') for g in e.ALLOWED}),patch.object(self.m,'host_available_bytes',return_value=2*1024**4),patch.object(e.subprocess,'Popen',side_effect=spawn),patch.object(e.time,'sleep'),patch('builtins.print',side_effect=output):
                if fatal or spawn_failure or proof_failure or output_failure:
                    failure=None
                    try:e.schedule(out,False)
                    except Exception as exc:failure=exc
                    self.assertIsInstance(failure,RuntimeError,'failures must drain workers before final RuntimeError')
                else:e.schedule(out,False)
        if proof_failure:self.assertEqual(proof_errors,[2],'failed proof must not be read again while draining')
        return launched,states

    def test_schedule_caps_twelve_host_workers_three_per_allowed_physical_card(self):
        launched,states=self.scheduler_scenario(20)
        self.assertEqual(len(launched),20)
        self.assertEqual([(p.job['gpu'],p.job['slot']) for p in launched[:4]],[(2,0),(4,0),(5,0),(6,0)])
        self.assertEqual(max(len(s['running']) for s in states),12)
        for state in states:
            counts={}
            for key,worker in state['running'].items():
                self.assertEqual(key,f"{worker['gpu']}:{worker['slot']}")
                self.assertIn(worker['gpu'],(2,4,5,6));counts[worker['gpu']]=counts.get(worker['gpu'],0)+1
            self.assertTrue(all(v<=3 for v in counts.values()))

    def test_non_resource_failure_stops_dispatch_and_drains_other_eleven_workers(self):
        launched,states=self.scheduler_scenario(20,fatal=True)
        self.assertEqual(len(launched),12)
        self.assertEqual(states[-1]['running'],{})
        self.assertEqual(states[-1]['pending'],8)
        self.assertEqual(states[-1]['failed'][0]['returncode'],1)
        self.assertTrue(all(p.returncode is not None for p in launched))

    def test_spawn_failure_stops_dispatch_and_drains_started_workers(self):
        launched,states=self.scheduler_scenario(20,spawn_failure=True)
        self.assertEqual(len(launched),2)
        self.assertEqual(states[-1]['running'],{})
        self.assertTrue(all(p.returncode==0 for p in launched))

    def test_invalid_class_proof_latches_failure_and_drains_without_more_proof_reads(self):
        launched,states=self.scheduler_scenario(20,proof_failure=True)
        self.assertEqual(len(launched),2)
        self.assertTrue(all(p.returncode==0 for p in launched))
        self.assertEqual(states[-1]['running'],{})
        self.assertIn('invalid class proof',states[-1]['failed'][0]['scheduler_error'])

    def test_persistent_stdout_failure_still_drains_started_workers(self):
        launched,states=self.scheduler_scenario(20,output_failure=True)
        self.assertEqual(len(launched),2)
        self.assertTrue(all(p.returncode==0 for p in launched))
        self.assertEqual(states[-1]['running'],{})
        self.assertIn('pipeline stdout failed',states[-1]['failed'][0]['scheduler_error'])


if __name__=='__main__':unittest.main()
