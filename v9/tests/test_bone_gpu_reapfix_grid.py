"""CPU reproductions of torn /proc reads during checkpoint retirement."""
import copy
import importlib
import json
import signal
import unittest
from pathlib import Path
from unittest.mock import patch
import test_bone_gpu_checkpoint_grid as previous_tests


class ReapFixTests(unittest.TestCase):
    def setUp(self):
        path=Path(__file__).resolve().parents[1]/'tools/bone_gpu_reapfix_grid.py'
        self.assertTrue(path.exists(),'additive reap-safe recovery entry is required')
        self.m=importlib.import_module('tools.bone_gpu_reapfix_grid')

    def fixture(self):
        prior=previous_tests.CheckpointTests('runTest');prior.setUp();f=prior.fixture()
        original=f.c
        f.c=self.m.CheckpointHandoff(f.value,f.h,f.s,uid=1000,kill=lambda p,s:f.signals.append((p,s)))
        f.original=original;f.clock=[0.];f.c.clock=lambda:f.clock[0]
        return f

    def captured(self):
        f=self.fixture();spec,p=f.worker();f.original.capture()
        # Use the original immutable capture, never generate a new one for recovery.
        log=f.root/self.m.checkpoint.PIPELINE
        log.write_text('Traceback (most recent call last):\nValueError: worker identity changed\nCHECKPOINT_PIPELINE_RC=1\n')
        f.spec=spec;f.process=p;return f

    def stopped_fixture(self):
        f=self.captured();f.marker(f.spec);f.original.tick()
        f.ref=copy.deepcopy(f.original.intents[f.spec['spec']])
        f.state['running']={};f.state['failed']=[dict(name=f.spec['name'],log=f.spec['log'],returncode=-15)]
        f.save();f.processes.pop(20)
        return f

    def restore(self,f):
        refs=list((f.root/'output').glob('bone_cpugpu_checkpoint_stop_*_20261028.json'))
        primary=refs[0] if refs else f.root/'output/missing.json'
        with patch.multiple(self.m,FAILED_LOG_SHA=f.s.sha(f.root/self.m.checkpoint.PIPELINE),
                HANDOFF_SHA=f.s.sha(f.root/f.h.HANDOFF),PRIMARY_INTENT_NAME=primary.name,
                PRIMARY_INTENT_SHA=f.s.sha(primary) if primary.exists() else '0'*64):
            return f.c.restore()

    def test_engine_inherits_frozen_runtime_controls_scheduler_and_chunking(self):
        e=self.m.engine();old=self.m.checkpoint.engine()
        self.assertEqual(e.OUT,old.OUT+'_reapfix')
        self.assertEqual(e.SOURCES,old.SOURCES+('tools/bone_gpu_reapfix_grid.py',))
        self.assertEqual(e.run.__wrapped__.__code__,old.run.__wrapped__.__code__)
        self.assertEqual(e.schedule.__code__,old.schedule.__code__)
        self.assertEqual(e.case_jobs.__code__,old.case_jobs.__code__)
        self.assertEqual(e.check_controls.__code__,old.check_controls.__code__)
        self.assertEqual(e.ALLOWED,(2,4,5,6))
        self.assertEqual(self.m.CHUNK_CLASSES,8)

    def test_helper_reuses_existing_handoff_paths_and_includes_new_entry(self):
        h=self.m.handoff_helpers();old=self.m.checkpoint.handoff_helpers()
        for name in ('HANDOFF','STOPPED','DRAINED'):self.assertEqual(getattr(h,name),getattr(old,name))
        self.assertEqual(h.collect_parent.__code__,old.collect_parent.__code__)
        self.assertIn('bone_gpu_reapfix_grid.py',self.m.PROCESS_NAMES)

    def test_frozen_checkpoint_and_test_hash_guards(self):
        for name in ('CHECKPOINT_SHA','CHECKPOINT_TEST_SHA'):
            with patch.object(self.m,name,'0'*64):
                with self.assertRaisesRegex(ValueError,'frozen checkpoint'):self.m.engine()

    def test_resume_preserves_original_capture_intent_and_positive_native_ack(self):
        f=self.stopped_fixture();before=f.s.sha(f.root/f.h.HANDOFF)
        with patch.object(f.c,'capture',side_effect=AssertionError('must never recapture')):
            result=self.restore(f)
        self.assertEqual(f.s.sha(f.root/f.h.HANDOFF),before)
        self.assertEqual(f.c.intents[f.spec['spec']],f.ref)
        self.assertEqual(f.c.exits[f.spec['spec']]['native_failure']['returncode'],-15)
        self.assertEqual(len(f.s.b.read(f.ref['path'])['files']),7)
        evidence=f.s.b.read(f.root/self.m.RESUME)
        self.assertEqual(evidence['handoff']['sha'],before)
        self.assertEqual(evidence['failed_waiter_log']['sha'],f.s.sha(f.root/self.m.checkpoint.PIPELINE))
        self.assertEqual(evidence['intents'][f.spec['spec']],f.ref)
        self.assertTrue(result['resumed']);self.assertEqual(f.signals,[(20,signal.SIGTERM)])

    def test_resume_rejects_mutated_intent_proof_or_log_prefix(self):
        for kind in ('intent','proof','log'):
            f=self.stopped_fixture();intent=f.s.b.read(f.ref['path'])
            target=Path(f.ref['path']) if kind=='intent' else (Path(intent['marker']) if kind=='proof' else Path(f.spec['log']))
            target.write_text('mutated')
            with self.assertRaises(ValueError):self.restore(f)
            self.assertFalse((f.root/self.m.RESUME).exists())

    def test_resume_accepts_log_appends_and_preserves_prefix_guard(self):
        f=self.stopped_fixture()
        with Path(f.spec['log']).open('a') as handle:handle.write('append after signal\n')
        self.restore(f);self.assertIn(f.spec['spec'],f.c.exits)

    def test_resume_rejects_wrong_failed_log_error_rc_hash_and_existing_outputs(self):
        for content in ('ValueError: other\nCHECKPOINT_PIPELINE_RC=1\n',
                        'ValueError: worker identity changed\nCHECKPOINT_PIPELINE_RC=0\n'):
            f=self.stopped_fixture();(f.root/self.m.checkpoint.PIPELINE).write_text(content)
            with self.assertRaises(ValueError):self.restore(f)
        f=self.stopped_fixture()
        with patch.object(self.m,'FAILED_LOG_SHA','0'*64):
            with self.assertRaises(ValueError):f.c.restore()
        for name in ('STOPPED','DRAINED'):
            f=self.stopped_fixture();f.write(f.root/getattr(f.h,name),{})
            with self.assertRaises(FileExistsError):self.restore(f)
        f=self.stopped_fixture();(f.root/self.m.OUT).mkdir(parents=True)
        with self.assertRaises(FileExistsError):self.restore(f)

    def test_resume_rejects_unintended_native_failure_before_pid_access(self):
        f=self.stopped_fixture();f.state['failed'][0]['returncode']=-9;f.save()
        with patch.object(f.h,'process_record',side_effect=AssertionError('no PID reads before failure audit')):
            with self.assertRaises(ValueError):self.restore(f)

    def test_exact_sigterm_torn_live_record_then_native_ack_does_not_fail(self):
        f=self.captured();f.marker(f.spec)
        # Initialize recovery from the same immutable capture without a second capture.
        f.original.tick();f.state['failed']=[dict(name=f.spec['name'],log=f.spec['log'],returncode=-15)]
        f.state['running']={};f.save();f.processes.pop(20);self.restore(f)
        # A second worker receives SIGTERM and becomes a torn R snapshot before ACK.
        # This helper directly transfers the already signed old capture state.
        f=self.fixture();spec,p=f.worker();f.c.capture();f.marker(spec)
        def kill(pid,sig):
            f.signals.append((pid,sig));p.update(state='R',args=[''],cwd='/proc/20/cwd')
        f.c.kill=kill
        self.assertFalse(f.c.tick());self.assertEqual(f.signals,[(20,signal.SIGTERM)])
        f.state['failed']=[dict(name=spec['name'],log=spec['log'],returncode=-15)];f.state['running']={};f.save()
        self.assertFalse(f.c.tick());self.assertIn(spec['spec'],f.c.resolved)
        self.assertEqual(f.signals,[(20,signal.SIGTERM)])

    def test_acknowledged_reaped_pid_is_never_read_or_signalled_again(self):
        f=self.stopped_fixture();self.restore(f)
        original=f.h.process_record
        def record(pid):
            if pid==20:raise AssertionError('native reaped PID must never be inspected again')
            return original(pid)
        f.h.process_record=record
        self.assertFalse(f.c.tick());self.assertIn(f.spec['spec'],f.c.resolved)
        self.assertEqual(f.signals,[(20,signal.SIGTERM)])

    def test_reused_pid_before_ack_defers_and_after_ack_never_signals_replacement(self):
        f=self.fixture();spec,p=f.worker();f.c.capture();f.marker(spec);f.c.tick()
        p.update(starttime='replacement',args=['python','other.py'],cwd='/elsewhere')
        self.assertFalse(f.c.tick());self.assertEqual(f.signals,[(20,signal.SIGTERM)])
        f.state['failed']=[dict(name=spec['name'],log=spec['log'],returncode=-15)];f.state['running']={};f.save()
        self.assertFalse(f.c.tick());self.assertEqual(f.signals,[(20,signal.SIGTERM)])

    def test_fixed_identity_deadline_never_slides_and_expired_wait_aborts(self):
        f=self.fixture();spec,p=f.worker();f.c.capture();p.update(args=[''],cwd='/proc/20/cwd')
        self.assertFalse(f.c.tick());deadline=f.c.identity_deadlines[spec['spec']]
        for now in (10.,30.,59.):
            f.clock[0]=now;self.assertFalse(f.c.tick());self.assertEqual(f.c.identity_deadlines[spec['spec']],deadline)
        f.clock[0]=60.
        with self.assertRaisesRegex(ValueError,'identity.*60'):f.c.tick()
        self.assertFalse(f.signals)

    def test_forged_identity_before_intent_never_gets_signal_permission(self):
        for changes in (dict(uid=999),dict(starttime='new'),dict(cwd='/other'),dict(args=['python','forged.py'])):
            f=self.fixture();spec,p=f.worker();f.c.capture();f.marker(spec);p.update(changes)
            self.assertFalse(f.c.tick());self.assertFalse(f.signals)
            f.clock[0]=60.
            with self.assertRaises(ValueError):f.c.tick()

    def test_disappeared_unacknowledged_worker_waits_for_positive_completion_or_requeue(self):
        f=self.fixture();spec,p=f.worker();f.c.capture();f.processes.pop(20)
        self.assertFalse(f.c.tick());self.assertNotIn(spec['spec'],f.c.resolved)
        f.marker(spec);f.state['running']={};f.save();self.assertFalse(f.c.tick())
        self.assertEqual(f.c.resolved[spec['spec']],'complete_class_records');self.assertFalse(f.signals)

    def test_source_spec_or_catalog_mutation_is_not_swallowed_as_identity_wait(self):
        for what in ('spec','catalog','protocol'):
            f=self.fixture();spec,p=f.worker();f.c.capture();p.update(args=[''])
            target=Path(spec['spec']) if what=='spec' else f.q/('jobs.json' if what=='catalog' else 'protocol.json')
            f.write(target,dict(changed=True))
            with self.assertRaises((ValueError,KeyError)):f.c.tick()
            self.assertFalse(f.signals)

    def test_resume_then_finish_keeps_twice_quiescence_and_adds_resume_evidence(self):
        f=self.stopped_fixture();self.restore(f);f.processes.clear();checks=[]
        f.value.related=lambda:checks.append(1) or []
        with patch.object(self.m.checkpoint.time,'sleep'):
            result=f.c.finish()
        self.assertEqual(len(checks),2);self.assertTrue(result['verified'])
        self.assertIn(str(f.root/self.m.RESUME),result['evidence'])
        self.assertEqual(f.s.b.read(f.root/f.h.DRAINED)['exits'][f.spec['spec']]['wait_status'],15)
        self.assertEqual(f.signals,[(20,signal.SIGTERM)])


if __name__=='__main__':unittest.main()
