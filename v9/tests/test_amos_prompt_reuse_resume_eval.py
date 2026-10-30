"""Fail-closed recovery of captured Linux workers during the R to Z transition."""
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
ENTRY=ROOT/'tools/amos_prompt_reuse_resume_eval.py'


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(ENTRY.exists(),'add isolated source-bound proc transition recovery')
        self.m=importlib.import_module('tools.amos_prompt_reuse_resume_eval')
        self.identity=dict(pid=124,uid=9,ppid=123,starttime=556,cwd='/project',
                           argv=['python','/project/tools/amos_prompt_ablation_native_manifest_eval.py','worker','--job','/project/spec'],state='R')

    def d(self):return Path(tempfile.mkdtemp(prefix='amos_procfix_'))
    def write(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8');return path
    def zombie(self):return dict(self.identity,state='Z',cwd=None,argv=[],raw_exit_status=0)

    def test_bound_sha256_constants_are64hex_and_bind_actual_failure_evidence(self):
        for name in ('CONTROLLER_SHA256','RUNTIME_SHA256','FAILURE_EVIDENCE_SHA256'):
            self.assertRegex(getattr(self.m,name),r'\A[0-9a-f]{64}\Z',name)
        actual=('e849aaeb6e1cd72ffa2a5661f88266ac'+'b3f5386c8bddc80fd7df14f0733d4d59')
        self.assertEqual(len(actual),64)
        self.assertEqual(self.m.FAILURE_EVIDENCE_SHA256,actual)

    def test_old_none_during_transition_resamples_actual_matching_zombie(self):
        old=importlib.import_module('tools.amos_prompt_reuse_eval');d=self.d();p=d/'124';p.mkdir()
        fields=['R','123']+['0']*48;fields[19]='556'
        (p/'stat').write_text('124 (python) '+' '.join(fields));(p/'status').write_text('Uid:\t9\t9\t9\t9\n')
        (p/'cmdline').write_bytes(b'python\0worker\0');(p/'wchan').write_text('0')
        with patch.object(old.os,'readlink',side_effect=FileNotFoundError('cwd vanished at exit')):
            self.assertIsNone(old.read_process(124,d))
        reader=unittest.mock.Mock(side_effect=[None,self.zombie()])
        with patch.object(self.m.time,'sleep') as sleep:
            got=self.m.read_captured_process(124,{124:self.identity},reader)
        self.assertEqual(got,self.zombie());self.assertEqual(reader.call_count,2);sleep.assert_called_once()

    def test_explicit_torn_snapshot_and_incomplete_live_snapshot_are_bounded(self):
        for initial in (ValueError('torn proc identity'),dict(self.identity,argv=[],cwd=None)):
            reader=unittest.mock.Mock(side_effect=[initial,self.zombie()])
            with patch.object(self.m.time,'sleep'):
                self.assertEqual(self.m.read_captured_process(124,{124:self.identity},reader),self.zombie())

    def test_permanent_missing_and_torn_never_fabricate_exit(self):
        for value in (None,ValueError('torn proc identity')):
            reader=unittest.mock.Mock(side_effect=lambda *a,**kw:(_ for _ in ()).throw(value) if isinstance(value,Exception) else value)
            with patch.object(self.m.time,'sleep'):
                with self.assertRaisesRegex(ValueError,'stable captured'):self.m.read_captured_process(124,{124:self.identity},reader)
            self.assertEqual(reader.call_count,self.m.SNAPSHOT_ATTEMPTS)

    def test_stable_uid_pid_ppid_start_argv_cwd_mismatch_refuses_immediately(self):
        for changes in (dict(uid=10),dict(pid=125),dict(ppid=2),dict(starttime=557),dict(argv=['other']),dict(cwd='/other')):
            reader=unittest.mock.Mock(return_value=dict(self.identity,**changes))
            with patch.object(self.m.time,'sleep') as sleep:
                with self.assertRaisesRegex(ValueError,'identity'):self.m.read_captured_process(124,{124:self.identity},reader)
            sleep.assert_not_called();reader.assert_called_once()

    def test_unknown_pid_or_permission_error_never_bypasses_identity_guard(self):
        reader=unittest.mock.Mock(return_value=self.zombie())
        with self.assertRaisesRegex(ValueError,'captured'):self.m.read_captured_process(125,{124:self.identity},reader)
        reader.assert_not_called()
        with self.assertRaises(PermissionError):self.m.read_captured_process(124,{124:self.identity},unittest.mock.Mock(side_effect=PermissionError('no')))

    def test_fresh_compiled_module_never_mutates_signed_parent_and_dispatches_new_entry(self):
        old=importlib.import_module('tools.amos_prompt_reuse_eval');parent=old.engine();before=dict(parent)
        context=self.m.compile_context();ns=context.engine()
        self.assertIsNot(ns,parent);self.assertEqual(before,parent)
        for name in ('worker','make_attempt','merge','schedule_queue','class_records','main'):self.assertIs(ns[name].__globals__,ns)
        self.assertEqual(ns['VARIANTS'],parent['VARIANTS']);self.assertEqual(ns['EVALUATION_ORDER'],parent['EVALUATION_ORDER'])
        self.assertEqual(ns['OUT'],self.m.OUT);self.assertEqual(ns['TMUX'],self.m.TMUX)
        d=self.d();job=dict(log=str(d/'worker.log'),spec=str(d/'job.json'),gpu=2)
        with patch.object(ns['subprocess'],'Popen',return_value=object()) as spawn:
            _,handle=ns['Backend'](d,False).spawn(job);handle.close()
        self.assertEqual(Path(spawn.call_args.args[0][1]),ENTRY)
        self.assertTrue(set(parent['SOURCE_SEEDS'])<=set(ns['SOURCE_SEEDS']))
        self.assertIn('tools/amos_prompt_reuse_resume_eval.py',ns['SOURCE_SEEDS'])

    def test_compiler_refuses_any_changed_signed_controller_bytes(self):
        old=ROOT/'tools/amos_prompt_reuse_eval.py'
        with self.assertRaisesRegex(ValueError,'SHA'):self.m.compile_context(source=old.read_bytes()+b'\n')

    def fixture(self):
        m=self.m;d=self.d();old=importlib.import_module('tools.amos_prompt_reuse_eval');h=d/old.HANDOFF;s=d/old.STOPPED
        self.write(h,{});self.write(s,{})
        log=d/'output/failed_helper.log';log.write_text('captured worker disappeared/reused; no positive exit status')
        scheduler=dict(self.identity,pid=123,ppid=1,state='T',argv=['python','tools/amos_prompt_ablation_native_manifest_eval.py','run'],cwd=str(d.resolve()))
        evidence=dict(kind='amos_reuse_proc_transition_failure_v1',helper_log=str(log),helper_log_sha256=m.sha(log),helper_log_text=log.read_text(),
                      handoff_sha256=m.sha(h),stopped_sha256=m.sha(s),source_sha256=m.CONTROLLER_SHA256,scheduler=scheduler,
                      workers=[dict(captured=self.identity,spec_sha256='signed')],related=[],new_out_exists=False)
        path=self.write(d/m.FAILURE_EVIDENCE,evidence)
        return d,path,evidence,dict(scheduler=scheduler,workers=[dict(identity=self.identity,spec_sha256='signed')])

    def test_failure_log_evidence_and_original_handoff_bind_exact_bytes(self):
        m=self.m;d,path,evidence,handoff=self.fixture()
        with patch.object(m,'ROOT',d),patch.object(m,'FAILURE_EVIDENCE_SHA256',m.sha(path)):
            result=m.verify_failure(handoff);self.assertEqual(result,evidence)
            Path(evidence['helper_log']).write_text('changed')
            with self.assertRaisesRegex(ValueError,'helper'):m.verify_failure(handoff)
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError,'SHA'):m.verify_failure(handoff)

    def test_resume_startup_checks_actual_frozen_parent_and_no_old_helper(self):
        m=self.m;context=self.m.compile_context();d,path,evidence,handoff=self.fixture()
        with patch.object(m,'ROOT',d),patch.object(m,'verify_failure',return_value=evidence),\
             patch.dict(context.__dict__,ROOT=d,verify_handoff=lambda:handoff,read_process=lambda pid:handoff['scheduler'],related=lambda:[]):
            with patch.dict(context.__dict__,source_guard=lambda:None):
                self.assertEqual(m.startup_guard(context),handoff)
                self.write(d/context.OUT,dict(partial=True))
                with self.assertRaises(FileExistsError):m.startup_guard(context)

    def test_new_resume_helpers_are_visible_to_related_and_block_duplicate_start(self):
        context=self.m.compile_context();d=self.d();p=d/'456';p.mkdir();(p/'cmdline').write_bytes(b'python\0tools/amos_prompt_reuse_resume_eval.py\0resume-run\0')
        self.assertEqual(context.related(d),[dict(pid=456,args=['python','tools/amos_prompt_reuse_resume_eval.py','resume-run'])])

    def test_missing_scheduler_allowed_only_after_original_term_and_cont(self):
        m=self.m;context=m.compile_context();scheduler=dict(self.identity,pid=123,ppid=1,state='T')
        handoff=dict(scheduler=scheduler,workers=[dict(identity=self.identity)])
        raw=unittest.mock.Mock(return_value=None);signals=[]
        with patch.dict(context.__dict__,verify_handoff=lambda:handoff,_procfix_original_reader=raw,
                        _procfix_original_signal=lambda identity,sig:signals.append((identity['pid'],sig))),patch.object(m.time,'sleep'):
            with self.assertRaisesRegex(ValueError,'stable captured'):context.read_process(123)
            with self.assertRaisesRegex(ValueError,'worker|scheduler'):context.signal_exact(self.identity,15)
            raw.return_value=scheduler
            context.signal_exact(scheduler,15);context.signal_exact(scheduler,18)
            raw.return_value=None
            self.assertIsNone(context.read_process(123))
            with self.assertRaisesRegex(ValueError,'stable captured'):context.read_process(124)
            raw.side_effect=ValueError('torn proc identity')
            with self.assertRaisesRegex(ValueError,'stable captured'):context.read_process(123)
        self.assertEqual(signals,[(123,15),(123,18)])

    def test_recovery_audit_and_source_inventory_bind_failed_helper_and_evidence(self):
        m=self.m;context=m.compile_context();d,path,evidence,handoff=self.fixture()
        with patch.object(m,'ROOT',d),patch.object(m,'verify_failure',return_value=evidence),\
             patch.dict(context.__dict__,ROOT=d,verify_handoff=lambda:handoff,_procfix_original_audit=lambda:dict(files={})):
            audit=context.build_reuse_audit()
        expected={m.FAILURE_EVIDENCE,'output/failed_helper.log','output/amos_prompt_reuse_handoff_20261030.json','output/amos_prompt_reuse_stopped_20261030.json'}
        self.assertEqual(set(audit['files']),expected)
        self.assertEqual(audit['proc_transition_recovery']['evidence_sha256'],m.FAILURE_EVIDENCE_SHA256)


_reuse_tests=importlib.import_module('tests.test_amos_prompt_reuse_eval')


class RecoveryWorkerExecutionTests(_reuse_tests.ChildWorkerExecutionTests):
    def setUp(self):
        context=importlib.import_module('tools.amos_prompt_reuse_resume_eval').compile_context()
        self.m=_reuse_tests._ChildGlobalsProxy(context.engine())
        self.task=dict(source_name='amos',case_id='0001',global_class_id=201,local_class_id=1,
                       class_name='spleen',image='data/a.nii.gz',label='data/b.nii.gz')


class RecoveryQueueExecutionTests(_reuse_tests.ChildSchedulerExecutionTests):
    setUp=RecoveryWorkerExecutionTests.setUp


if __name__=='__main__':unittest.main()
