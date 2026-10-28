"""Offline finalizer tests: no model, process signals, or identity waits."""
import copy
import importlib
import unittest
from pathlib import Path
from unittest.mock import patch
import test_bone_gpu_checkpoint_grid as previous_tests


class PostDrainTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((Path(__file__).resolve().parents[1]/'tools/bone_gpu_postdrain_grid.py').exists(),
                        'additive offline postdrain finalizer is required')
        self.m=importlib.import_module('tools.bone_gpu_postdrain_grid')

    def fixture(self):
        prior=previous_tests.CheckpointTests('runTest');prior.setUp();f=prior.fixture()
        specs=[f.worker(20+i,'case'+str(i))[0] for i in range(4)]
        f.c.capture()
        for spec in specs:f.marker(spec)
        f.c.tick()
        f.state.update(running={},pending=4,failed=[dict(name=s['name'],log=s['log'],returncode=-15) for s in specs]);f.save()
        f.processes.clear();f.specs=specs
        (f.root/self.m.reapfix.checkpoint.PIPELINE).write_text('ValueError: worker identity changed\nCHECKPOINT_PIPELINE_RC=1\n')
        (f.root/self.m.reapfix.PIPELINE).write_text('ValueError: unresolved worker identity after fixed 60 seconds: 425925\nREAPFIX_PIPELINE_RC=1\n')
        f.write(f.root/self.m.reapfix.RESUME,dict(original_resume=True))
        f.refs={str(f.root/self.m.reapfix.RESUME):f.s.sha(f.root/self.m.reapfix.RESUME),
                str(f.root/self.m.reapfix.PIPELINE):f.s.sha(f.root/self.m.reapfix.PIPELINE)}
        f.constants=dict(OLD_RESUME_SHA=f.s.sha(f.root/self.m.reapfix.RESUME),
                         FAILED_REAPFIX_LOG_SHA=f.s.sha(f.root/self.m.reapfix.PIPELINE),
                         FINAL_STATUS_SHA=f.s.sha(f.q/'status.json'))
        primary=next((f.root/'output').glob('bone_cpugpu_checkpoint_stop_20_*_20261028.json'))
        f.original_constants=dict(FAILED_LOG_SHA=f.s.sha(f.root/self.m.reapfix.checkpoint.PIPELINE),
            HANDOFF_SHA=f.s.sha(f.root/f.h.HANDOFF),PRIMARY_INTENT_NAME=primary.name,PRIMARY_INTENT_SHA=f.s.sha(primary))
        f.c=self.m.CheckpointHandoff(f.value,f.h,f.s,uid=1000)
        f.checks=[];f.value.related=lambda:f.checks.append(1) or []
        f.h.process_record=lambda p:(_ for _ in ()).throw(AssertionError('offline audit must not inspect old PIDs'))
        return f

    def restore(self,f):
        with patch.multiple(self.m,**f.constants),patch.multiple(self.m.reapfix,**f.original_constants),patch.object(self.m.reapfix.checkpoint.time,'sleep'):
            return f.c.restore()

    def test_engine_preserves_frozen_runtime_scheduler_controls_and_chunk8(self):
        e=self.m.engine();old=self.m.reapfix.engine()
        self.assertEqual(e.OUT,'output/bone_v92e360_v102e490_floor_grid_20261028_cpugpu_postdrain')
        self.assertEqual(e.SOURCES,old.SOURCES+('tools/bone_gpu_postdrain_grid.py',))
        for name in ('schedule','case_jobs','check_controls'):self.assertEqual(getattr(e,name).__code__,getattr(old,name).__code__)
        self.assertEqual(e.run.__wrapped__.__code__,old.run.__wrapped__.__code__)
        self.assertEqual(self.m.reapfix.CHUNK_CLASSES,8)
        with patch.object(self.m,'REAPFIX_SHA','0'*64):
            with self.assertRaisesRegex(ValueError,'frozen reapfix'):self.m.engine()
        with patch.object(self.m,'REAPFIX_TEST_SHA','0'*64):
            with self.assertRaisesRegex(ValueError,'frozen reapfix'):self.m.engine()

    def test_restores_exact_four_without_touching_old_resume_or_recapturing(self):
        f=self.fixture();before={p:f.s.sha(p) for p in f.refs}
        with patch.object(f.c,'capture',side_effect=AssertionError('must not recapture')):self.restore(f)
        self.assertEqual(len(f.c.exits),4);self.assertEqual(len(f.c.workers),4)
        self.assertEqual(len(f.checks),2)
        self.assertTrue((f.root/self.m.RESUME).exists())
        self.assertEqual({p:f.s.sha(p) for p in f.refs},before)
        for name in ('HANDOFF','STOPPED','DRAINED'):
            self.assertEqual(getattr(self.m.handoff_helpers(),name),getattr(f.h,name))

    def test_missing_extra_wrong_ack_or_running_worker_blocks_before_new_resume(self):
        for change in ('missing','extra','code','running'):
            f=self.fixture()
            if change=='missing':f.state['failed'].pop()
            elif change=='extra':f.state['failed'].append(copy.deepcopy(f.state['failed'][0]))
            elif change=='code':f.state['failed'][0]['returncode']=-9
            else:f.state['running']={'2':dict(pid=20,spec=f.specs[0]['spec'])}
            f.save();f.constants['FINAL_STATUS_SHA']=f.s.sha(f.q/'status.json')
            with self.assertRaises(ValueError):self.restore(f)
            self.assertFalse((f.root/self.m.RESUME).exists())

    def test_altered_handoff_sha_blocks_before_any_spec_path_read(self):
        f=self.fixture();f.write(f.root/f.h.HANDOFF,dict(workers=[dict(spec='/untrusted/arbitrary.json')]))
        original=f.s.b.read
        def read(path):
            if str(path)=='/untrusted/arbitrary.json':raise AssertionError('unverified handoff must not direct a spec read')
            return original(path)
        f.s.b.read=read
        with self.assertRaises(ValueError):self.restore(f)
        self.assertFalse((f.root/self.m.RESUME).exists())

    def test_related_process_on_second_check_blocks_all_new_audit_writes(self):
        f=self.fixture();snapshots=iter([[],['123']]);f.value.related=lambda:next(snapshots)
        with self.assertRaises(ValueError):self.restore(f)
        self.assertFalse((f.root/self.m.RESUME).exists())

    def test_old_resume_failed_logs_or_native_status_hash_mutation_blocks(self):
        for pathkind in ('resume','failed_log','status'):
            f=self.fixture()
            path=f.root/self.m.reapfix.RESUME if pathkind=='resume' else (f.q/'status.json' if pathkind=='status' else f.root/self.m.reapfix.PIPELINE)
            path.write_text('changed')
            with self.assertRaises(ValueError):self.restore(f)
            self.assertFalse((f.root/self.m.RESUME).exists())

    def test_original_seven_proof_and_log_prefix_guards_are_retained(self):
        for targetkind in ('proof','log'):
            f=self.fixture();spec=f.specs[0]
            target=Path(spec['log']) if targetkind=='log' else Path(spec['run'])/'classes/new.json'
            target.write_text('tampered')
            with self.assertRaises(ValueError):self.restore(f)
            self.assertFalse((f.root/self.m.RESUME).exists())

    def test_offline_finish_has_no_pid_reads_signals_or_identity_wait(self):
        f=self.fixture();self.restore(f)
        for old in f.c.workers.values():
            self.assertIsNone(f.c.immutable_worker(old));f.c.retire(old,None,{})
        with self.assertRaises(ValueError):f.c.retire(next(iter(f.c.workers.values())),dict(pid=123),{})
        with self.assertRaises(ValueError):f.c.kill(20,15)
        with patch.object(self.m.reapfix.checkpoint.time,'sleep'):
            result=f.c.finish()
        self.assertTrue(result['verified']);self.assertEqual(len(f.checks),4)
        self.assertIn(str(f.root/self.m.RESUME),result['evidence'])
        drained=f.s.b.read(f.root/f.h.DRAINED)
        self.assertEqual(len(drained['exits']),4)
        self.assertTrue(all(v['wait_status']==15 for v in drained['exits'].values()))

    def test_source_catalog_protocol_and_resume_tampering_after_restore_fail_closed(self):
        for kind in ('catalog','protocol','resume','failed_log'):
            f=self.fixture();self.restore(f)
            path=f.q/('jobs.json' if kind=='catalog' else 'protocol.json') if kind in ('catalog','protocol') else f.root/(self.m.RESUME if kind=='resume' else self.m.reapfix.PIPELINE)
            path.write_text('changed')
            with self.assertRaises(ValueError):f.c.tick()


if __name__=='__main__':unittest.main()
