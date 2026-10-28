"""CPU-only tests for cooperative native-scheduler checkpoint retirement."""
import copy
import hashlib
import importlib
import json
import signal
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        path=Path(__file__).resolve().parents[1]/'tools/bone_gpu_checkpoint_grid.py'
        self.assertTrue(path.exists(),'additive checkpoint helper is required')
        self.m=importlib.import_module('tools.bone_gpu_checkpoint_grid')

    def fixture(self):
        # Retain tiny fixtures; project policy prohibits recursive cleanup.
        root=Path(tempfile.mkdtemp(prefix='bone_checkpoint_'))
        h=self.m.handoff_helpers();h.ROOT=root
        s=self.m.base.prior_engine();s.ROOT=root;q=root/s.OUT
        q.mkdir(parents=True);(root/'output/bone_metricfast_pipeline_20261028.log').write_text('')
        def write(path,value):
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(value),encoding='utf-8')
        s.b=types.SimpleNamespace(read=lambda p:json.loads(Path(p).read_text()),write=write)
        s.verify=lambda p:None;s.check_controls=lambda p:None
        write(q/'protocol.json',dict(sources={}))
        write(q/'jobs.json',dict(cases=[],controls=[]))
        state=dict(phase='full',pending=1,running={},failed=[])
        records={};processes={};signals=[]
        scheduler=dict(pid=10,uid=1000,starttime='100',cwd=str(root),state='S',
                       args=['python','tools/bone_metric_speed_grid.py','wait-launch'])
        processes[10]=scheduler
        h.process_record=lambda pid:copy.deepcopy(processes.get(pid))
        h.matching_processes=lambda:[copy.deepcopy(p) for p in processes.values() if p['state']!='Z']
        s.class_records=lambda p,control=False:copy.deepcopy(records)
        value=types.SimpleNamespace(require_handoff=s.require_handoff,related=lambda:[str(p) for p in processes])
        write(q/'status.json',state)
        c=self.m.CheckpointHandoff(value,h,s,uid=1000,kill=lambda p,sig:signals.append((p,sig)))
        f=types.SimpleNamespace(root=root,q=q,h=h,s=s,c=c,write=write,state=state,records=records,
                                processes=processes,signals=signals,scheduler=scheduler,value=value)
        def worker(pid=20,name='case',attempt=0):
            task=dict(source_name='bone',case_id=name,global_class_id=10003,class_name='bone')
            canonical=dict(name=name,model='v9_2',control=False,parent_history=[],variant='baseline',
                source='bone',case_id=name,prepared=str(q/'prepared'/name),pending=[dict(task=task,floors=[.35])],
                task_count=1,attempt=0,oom_retries=0,frame_batch=2,minimum_free_mib=30720,
                not_before=0.,avoid_gpus={},avoid_gpu_until=0.,last_failed_gpu=None)
            catalog=s.b.read(q/'jobs.json')
            if not any(j['name']==name for j in catalog['cases']):catalog['cases'].append(canonical)
            write(q/'jobs.json',catalog)
            tag=f'{name}_a{attempt}';spec=dict(canonical,attempt=attempt,gpu=2,gpu_uuid='GPU-2',
                spec=str(q/'job_specs'/(tag+'.json')),run=str(q/'runs'/tag),log=str(q/'logs'/(tag+'.log')))
            write(Path(spec['spec']),spec);Path(spec['log']).parent.mkdir(exist_ok=True)
            Path(spec['log']).write_text('worker started\n')
            p=dict(pid=pid,uid=1000,starttime=str(pid*10),cwd=str(root),state='S',
                   args=['python','tools/bone_metric_speed_grid.py','worker','--job',spec['spec']])
            processes[pid]=p;state['running'][str(pid)]=dict(pid=pid,spec=spec['spec'],log=spec['log'],name=name)
            f.save();return spec,p
        def marker(spec,label='new'):
            run=Path(spec['run']);record=run/'classes'/(label+'.json')
            paths=[record,run/'restoration_start.json',run/'resource_policy_audit.json',Path(spec['spec']),
                   run/'prepared/manifest.json',run/'prepared/refs.json',run/'prepared_input_audit.json']
            for p in paths:
                if not p.exists():write(p,{})
            done=record.with_name(label+'.done.json')
            write(done,dict(job_name=spec['name'],record=str(record),files={str(p):s.sha(p) for p in paths},protocol_sha=s.sha(q/'protocol.json')))
            records[(spec['model'],s.p.key(spec['pending'][0]['task']))]=dict(marker=str(done),marker_sha=s.sha(done),record={})
            return done
        f.save=lambda:write(q/'status.json',state)
        f.worker=worker;f.marker=marker
        return f

    def test_engine_inherits_pipeline_and_adds_only_unique_source_paths(self):
        e=self.m.engine();old=self.m.base.engine()
        self.assertEqual(e.OUT,'output/bone_v92e360_v102e490_floor_grid_20261028_cpugpu_checkpoint')
        self.assertEqual(Path(e.__file__).name,'bone_gpu_checkpoint_grid.py')
        self.assertEqual(e.SOURCES,old.SOURCES+('tools/bone_gpu_checkpoint_grid.py',))
        self.assertEqual(e.run.__wrapped__.__code__,old.run.__wrapped__.__code__)
        self.assertEqual(e.schedule.__code__,old.schedule.__code__)
        self.assertEqual(e.ALLOWED,(2,4,5,6))

    def test_private_helpers_preserve_parent_collector_and_use_new_names(self):
        h=self.m.handoff_helpers();old=self.m.base.handoff_helpers()
        self.assertEqual(h.CURRENT_SCRIPT,'bone_metric_speed_grid.py')
        self.assertEqual(h.ENTRY,'bone_gpu_checkpoint_grid.py')
        for attr in ('HANDOFF','STOPPED','DRAINED'):
            self.assertIn('bone_cpugpu_checkpoint_handoff',getattr(h,attr))
            self.assertNotEqual(getattr(h,attr),getattr(old,attr))
        self.assertEqual(h.collect_parent.__code__,old.collect_parent.__code__)
        self.assertIn('bone_gpu_checkpoint_grid.py',self.m.PROCESS_NAMES)

    def test_source_guard_blocks_changed_reviewed_pipeline(self):
        with patch.object(self.m,'BASE_SHA','0'*64):
            with self.assertRaisesRegex(ValueError,'reviewed pipeline'):self.m.engine()

    def test_chunks_are_disjoint_exact_and_inherit_conservative_history(self):
        e=self.m.base.engine();tasks=[dict(source_name='bone',case_id='a',global_class_id=10000+i,class_name='b') for i in range(19)]
        histories={(model,floor,e.p.key(task)):dict(name='inherited',frame_batch=1,minimum_free_mib=45000,attempt=3,
                    oom_retries=2,not_before=900.,avoid_gpus={'2':1000.},avoid_gpu_until=1100.,last_failed_gpu=2)
                   for model in e.p.MODELS for floor in e.p.FLOORS for task in tasks}
        original=e.case_jobs(tasks,set(),histories)
        actual=self.m.chunk_case_jobs(e.case_jobs)(list(reversed(tasks)),set(),histories)
        def identities(jobs):
            return [(j['model'],floor,e.p.key(item['task'])) for j in jobs for item in j['pending'] for floor in item['floors']]
        self.assertEqual(set(identities(actual)),set(identities(original)))
        self.assertEqual(len(identities(actual)),len(set(identities(actual))))
        for job in actual:
            self.assertLessEqual(job['task_count'],8)
            parent=next(j for j in original if job['name'].startswith(j['name']+'_chunk'))
            self.assertEqual({k:v for k,v in job.items() if k not in ('name','pending','task_count')},
                             {k:v for k,v in parent.items() if k not in ('name','pending','task_count')})
        self.assertEqual(actual,self.m.chunk_case_jobs(e.case_jobs)(tasks,set(),histories))
        self.assertEqual([j['task_count'] for j in actual],[8,8,3,8,8,3])

    def test_capture_is_additive_and_preserves_original_failed_freeze_sha(self):
        f=self.fixture();f.worker();old=f.root/'output/bone_cpugpu_handoff_20261028.json';f.write(old,dict(old=True))
        f.c.capture();data=f.s.b.read(f.root/f.h.HANDOFF)
        self.assertEqual(data['prior_failed_freeze'][str(old)],f.s.sha(old))
        self.assertFalse(f.signals);self.assertFalse((f.root/f.h.STOPPED).exists())
        with self.assertRaises(FileExistsError):f.c.capture()

    def test_capture_rejects_wrong_scheduler_and_unsigned_worker(self):
        for field,bad in [('uid',999),('cwd','/other'),('starttime',''),('args',['python','other.py','wait-launch'])]:
            f=self.fixture();f.scheduler[field]=bad
            with self.assertRaises(ValueError):f.c.capture()
        f=self.fixture();spec,p=f.worker();f.write(Path(spec['spec']),dict(spec,frame_batch=8))
        with self.assertRaises(ValueError):f.c.capture()

    def test_worker_argv_spec_alias_and_mutated_canonical_catalog_are_rejected(self):
        f=self.fixture();spec,p=f.worker();alias=f.q/'job_specs/copied.json';f.write(alias,spec);p['args'][-1]=str(alias)
        with self.assertRaises(ValueError):f.c.capture()
        f=self.fixture();spec,p=f.worker();f.c.capture();f.marker(spec)
        catalog=f.s.b.read(f.q/'jobs.json');catalog['cases'][0]['parent_history'].append(dict(changed=True));f.write(f.q/'jobs.json',catalog)
        with self.assertRaises(ValueError):f.c.tick()
        self.assertFalse(f.signals)

    def test_no_fresh_marker_means_no_signal(self):
        f=self.fixture();spec,p=f.worker();f.marker(spec,'old');f.c.capture()
        f.c.tick();f.c.tick();self.assertEqual(f.signals,[])

    def test_checked_new_marker_writes_intent_before_only_one_exact_sigterm(self):
        f=self.fixture();spec,p=f.worker();f.c.capture();done=f.marker(spec)
        def kill(pid,sig):
            self.assertEqual((pid,sig),(20,signal.SIGTERM))
            ref=f.c.intents[spec['spec']];intent=f.s.b.read(ref['path'])
            self.assertEqual(intent['marker'],str(done));self.assertEqual(len(intent['files']),7)
            self.assertEqual(intent['spec_sha'],f.s.sha(spec['spec']))
            f.signals.append((pid,sig))
        f.c.kill=kill;f.c.tick();f.c.tick()
        self.assertEqual(f.signals,[(20,signal.SIGTERM)])

    def test_unvalidated_new_marker_fails_closed(self):
        f=self.fixture();spec,p=f.worker();f.c.capture();f.marker(spec);f.records.clear()
        with self.assertRaisesRegex(ValueError,'validation'):f.c.tick()
        self.assertFalse(f.signals)

    def test_marker_committed_after_first_record_snapshot_is_revalidated_before_signal(self):
        f=self.fixture();spec,p=f.worker();f.c.capture();f.marker(spec)
        snapshots=iter([{},copy.deepcopy(f.records)])
        f.s.class_records=lambda q,control=False:next(snapshots)
        f.c.tick();self.assertEqual(f.signals,[(20,signal.SIGTERM)])

    def test_changed_baseline_marker_or_spec_blocks_retirement(self):
        for what in ('marker','spec'):
            f=self.fixture();spec,p=f.worker();old=f.marker(spec,'old');f.c.capture()
            f.write(old if what=='marker' else Path(spec['spec']),dict(changed=True))
            with self.assertRaises(ValueError):f.c.tick()
            self.assertFalse(f.signals)

    def test_worker_pid_identity_change_blocks_signal(self):
        for changed in (dict(starttime='reused'),dict(uid=1001),dict(cwd='/wrong')):
            f=self.fixture();spec,p=f.worker();f.c.capture();f.marker(spec);p.update(changed)
            with self.assertRaises(ValueError):f.c.tick()
            self.assertFalse(f.signals)

    def acknowledge(self,f,spec):
        f.state['running']={};f.state['failed']=[dict(name=spec['name'],log=spec['log'],returncode=-15)];f.save()

    def test_only_native_exact_intent_acknowledgement_populates_controlled_exits(self):
        f=self.fixture();spec,p=f.worker();f.c.capture();f.marker(spec);f.c.tick()
        self.assertFalse(f.c.exits)
        self.acknowledge(f,spec);f.processes.pop(20);f.c.tick()
        self.assertEqual(f.c.exits[spec['spec']]['wait_status'],15)
        self.assertEqual(f.c.exits[spec['spec']]['intent'],f.c.intents[spec['spec']])

    def test_forged_unintended_or_changed_nonresource_failure_blocks(self):
        for field,bad in [('returncode',1),('log','/other'),('name','other')]:
            f=self.fixture();spec,p=f.worker();f.c.capture();f.marker(spec);f.c.tick();self.acknowledge(f,spec)
            f.state['failed'][0][field]=bad;f.save()
            with self.assertRaises(ValueError):f.c.tick()
        f=self.fixture();spec,p=f.worker();f.c.capture();self.acknowledge(f,spec)
        with self.assertRaises(ValueError):f.c.tick()

    def test_mutated_intent_or_checkpoint_proof_is_rejected_at_ack(self):
        for what in ('intent','proof','log'):
            f=self.fixture();spec,p=f.worker();f.c.capture();done=f.marker(spec);f.c.tick()
            intent=f.c.intents[spec['spec']]
            target=Path(intent['path']) if what=='intent' else (Path(spec['log']) if what=='log' else Path(f.s.b.read(done)['record']))
            target.write_text('changed');self.acknowledge(f,spec)
            with self.assertRaises(ValueError):f.c.tick()

    def test_worker_log_append_is_allowed_but_final_hash_is_recorded_after_reap(self):
        f=self.fixture();spec,p=f.worker();f.c.capture();f.marker(spec);f.c.tick()
        with Path(spec['log']).open('a') as handle:handle.write('next class preparation\n')
        self.acknowledge(f,spec);f.processes.pop(20);f.c.tick()
        self.assertEqual(f.c.exits[spec['spec']]['final_log_sha'],f.s.sha(spec['log']))

    def test_new_worker_launched_before_ack_is_captured_and_retired(self):
        f=self.fixture();one,p=f.worker();f.worker(21,'race');f.processes.pop(21);f.state['running'].pop('21');f.save()
        f.c.capture();f.marker(one);f.c.tick()
        two,p2=f.worker(21,'race');f.c.tick();self.assertIn(two['spec'],f.c.workers)
        f.marker(two);f.c.tick()
        self.assertEqual(f.signals,[(20,signal.SIGTERM),(21,signal.SIGTERM)])

    def test_reaped_success_race_requires_complete_class_positive_evidence(self):
        f=self.fixture();spec,p=f.worker();f.c.capture();f.marker(spec)
        f.processes.pop(20);f.state['running']={};f.save();f.c.tick()
        self.assertIn(spec['spec'],f.c.resolved);self.assertFalse(f.c.exits);self.assertFalse(f.signals)

    def test_reaped_worker_without_positive_status_or_log_evidence_blocks(self):
        f=self.fixture();spec,p=f.worker();f.c.capture();f.processes.pop(20)
        f.state['running']={};f.save()
        with self.assertRaisesRegex(ValueError,'positive'):f.c.tick()

    def test_real_resource_requeue_then_new_attempt_is_not_faked_as_sigterm(self):
        f=self.fixture();one,p=f.worker();f.c.capture();f.processes.pop(20)
        pipeline=f.root/'output/bone_metricfast_pipeline_20261028.log'
        pipeline.write_text('SHARED_START 2 20 case 1 batch 2\nSHARED_RESOURCE_REQUEUE case 2 30720\nSHARED_START 2 21 case 1 batch 2\n')
        two,p=f.worker(21,'case',1);f.state['running'].pop('20');f.save();f.c.tick()
        self.assertIn(one['spec'],f.c.resolved);self.assertNotIn(one['spec'],f.c.exits)
        self.assertIn(two['spec'],f.c.workers)

    def test_ack_race_waits_for_native_running_entry_to_disappear(self):
        f=self.fixture();spec,p=f.worker();f.c.capture();f.marker(spec);f.c.tick();f.processes.pop(20)
        f.c.tick();self.assertNotIn(spec['spec'],f.c.resolved)
        self.acknowledge(f,spec);f.c.tick();self.assertIn(spec['spec'],f.c.resolved)

    def test_observed_sigterm_cannot_be_relabelled_success_without_native_ack(self):
        f=self.fixture();spec,p=f.worker();f.c.capture();f.marker(spec);f.c.tick()
        p.update(state='Z',exit_status=15);f.state['running']={};f.save()
        with self.assertRaisesRegex(ValueError,'positive'):f.c.tick()

    def test_failed_entry_still_listed_running_cannot_resolve_worker_early(self):
        f=self.fixture();spec,p=f.worker();f.c.capture();f.marker(spec);f.c.tick()
        f.state['failed']=[dict(name=spec['name'],log=spec['log'],returncode=-15)];f.processes.pop(20);f.save()
        f.c.tick();self.assertNotIn(spec['spec'],f.c.resolved)

    def test_disappeared_worker_with_complete_marker_still_rejects_unknown_native_failure(self):
        f=self.fixture();spec,p=f.worker();f.c.capture();f.marker(spec);f.processes.pop(20)
        f.state.update(running={},failed=[dict(name=spec['name'],log=spec['log'],returncode=-9)]);f.save()
        with self.assertRaises(ValueError):f.c.tick()

    def test_zombie_actual_nonresource_status_cannot_hide_behind_complete_marker(self):
        f=self.fixture();spec,p=f.worker();f.c.capture();f.marker(spec)
        p.update(state='Z',exit_status=1<<8);f.state['running']={};f.save()
        with self.assertRaisesRegex(ValueError,'non-resource'):f.c.tick()

    def test_parent_identity_change_or_disappearance_with_running_queue_blocks(self):
        f=self.fixture();f.worker();f.c.capture();f.scheduler['starttime']='reused'
        with self.assertRaises(ValueError):f.c.tick()
        f=self.fixture();f.worker();f.c.capture();f.processes.pop(10)
        with self.assertRaises(ValueError):f.c.tick()

    def test_finish_requires_two_quiescence_checks_and_never_signals_scheduler(self):
        f=self.fixture();spec,p=f.worker();f.c.capture();f.marker(spec);f.c.tick()
        self.acknowledge(f,spec);f.processes.clear();f.c.tick()
        checks=[];f.value.related=lambda:checks.append(1) or []
        with patch.object(self.m.time,'sleep'):
            result=f.c.finish()
        self.assertEqual(len(checks),2);self.assertTrue(result['verified'])
        self.assertEqual(f.signals,[(20,signal.SIGTERM)])
        drained=f.s.b.read(f.root/f.h.DRAINED)
        self.assertEqual(drained['exits'][spec['spec']]['wait_status'],15)
        self.assertTrue((f.root/f.h.STOPPED).exists())

    def test_second_quiescence_detects_related_process_and_blocks_evidence(self):
        f=self.fixture();spec,p=f.worker();f.c.capture();f.marker(spec);f.c.tick()
        self.acknowledge(f,spec);f.processes.clear();f.c.tick();snapshots=iter([[],['99']])
        f.value.related=lambda:next(snapshots)
        with patch.object(self.m.time,'sleep'):
            with self.assertRaises(ValueError):f.c.finish()
        self.assertFalse((f.root/f.h.DRAINED).exists())


if __name__=='__main__':unittest.main()
