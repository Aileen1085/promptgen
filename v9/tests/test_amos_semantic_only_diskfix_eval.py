"""CPU recovery contracts. Retained fixtures never establish a real GPU gate."""
import copy
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from tests.test_amos_bg_scribble_module_eval import Proxy
from tests import test_amos_prompt_ablation_eval as native
from tests import test_amos_semantic_only_eval as semantic

ROOT = Path(__file__).resolve().parents[1]
ENTRY = 'tools/amos_semantic_only_diskfix_eval.py'
TEST = 'tests/test_amos_semantic_only_diskfix_eval.py'


def controller():
    return Proxy(importlib.import_module('tools.amos_semantic_only_diskfix_eval').compile_context().engine())


class DiskfixTests(unittest.TestCase):
    tasks = native.ControllerTests.tasks
    job = native.ControllerTests.job
    row = native.ControllerTests.row
    record = native.ControllerTests.record

    def setUp(self):
        self.assertTrue((ROOT/ENTRY).exists(), 'new isolated ENOSPC controller required')
        self.entry = importlib.import_module('tools.amos_semantic_only_diskfix_eval')
        self.m = controller()
        self.task = dict(source_name='amos', case_id='0001', global_class_id=201,
                         local_class_id=1, class_name='spleen')
        original = tempfile.mkdtemp
        p = patch.object(tempfile, 'mkdtemp', side_effect=lambda *a, **k: original(dir=str(ROOT/'output')))
        p.start(); self.addCleanup(p.stop)

    def tempdir(self):
        return Path(tempfile.mkdtemp())

    def test_private_bindings_and_child_entry_leave_parent_untouched(self):
        parent = importlib.import_module('tools.amos_semantic_only_restored_eval')
        before = (parent._CONTEXT, parent.OUT, parent.TMUX, parent.PIPELINE)
        other = parent.compile_context().engine()
        for name in ('worker', 'schedule_queue', 'Backend', 'make_attempt', 'main',
                     'class_records', 'case_jobs', 'verify', 'verify_parent', 'related',
                     '_source_guard', 'engine', 'runtime', 'build_merged'):
            obj = self.m.ns[name]
            if name == 'Backend': obj = obj.available
            self.assertIs(obj.__globals__, self.m.ns, name)
        self.assertIs(self.m.engine(), self.m.ns)
        self.assertIsNot(self.m.parent_engine(), self.m.ns)
        self.assertIs(self.m._bootstrap_function.__globals__['engine'](), self.m.ns)
        self.assertEqual((parent._CONTEXT, parent.OUT, parent.TMUX, parent.PIPELINE), before)
        self.assertEqual(other['OUT'], parent.OUT)
        self.assertEqual(Path(self.m.ns['__file__']), ROOT/ENTRY)
        d = self.tempdir(); job = dict(log=str(d/'worker.log'), spec=str(d/'job.json'), gpu=2)
        with patch.object(self.m.subprocess, 'Popen', return_value=object()) as spawn:
            _, handle = self.m.Backend(d, False).spawn(job); handle.close()
        self.assertEqual(spawn.call_args.args[0], [sys.executable, str(ROOT/ENTRY), 'worker', '--job', job['spec']])
        self.assertEqual(spawn.call_args.kwargs['env']['V9_SHARED_CACHE_WRITE_POLICY'], 'read_only')

    def test_profile_source_and_failure_evidence_fail_before_writes(self):
        m = self.m; d = self.tempdir()
        with patch.object(m, '_source_guard', side_effect=ValueError('profile or SHA drift')), \
             patch.object(m, '_diskfix_native_prepare') as prepare, patch.object(m, '_diskfix_native_pipeline') as run:
            for f in (m.prepare, m.run_pipeline):
                with self.assertRaisesRegex(ValueError, 'drift'): f(d/m.OUT)
            prepare.assert_not_called(); run.assert_not_called()
        self.assertFalse((d/m.OUT).exists()); self.assertFalse((d/m.PIPELINE).exists())
        with self.assertRaisesRegex(ValueError, 'SHA'): self.entry.compile_context(source=b'pass')
        parent = importlib.import_module('tools.amos_semantic_only_restored_eval')
        with patch.object(parent, 'source_guard', side_effect=ValueError('S21 production')) as guard:
            with self.assertRaisesRegex(ValueError, 'S21'): self.entry.source_guard(production=True)
        guard.assert_called_once_with(production=True)

    def test_local_help_and_new_inventory(self):
        self.assertTrue(self.entry.source_guard(production=False))
        with patch.object(sys, 'argv', [str(ROOT/ENTRY), '--help']):
            with self.assertRaises(SystemExit) as caught: self.m.main()
        self.assertEqual(caught.exception.code, 0)
        self.assertIn(ENTRY, self.m.SOURCE_SEEDS); self.assertIn(TEST, self.m.SOURCE_SEEDS)
        self.assertIn(self.m.PARENT_ENTRY, self.m.SOURCE_SEEDS)
        d = self.tempdir()
        for name in self.m.FAILURE_PINS:
            p = d/name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b'evidence')
        with patch.object(self.m, 'ROOT', d), patch.object(self.m, '_diskfix_native_sources', return_value=[]):
            self.assertEqual(set(self.m.file_inventory(self.m.dependency_sources())), set(self.m.FAILURE_PINS))

    def test_each_pinned_failure_drift_rejected_before_prepare_or_log(self):
        m=self.m; d=self.tempdir(); parent=importlib.import_module('tools.amos_semantic_only_restored_eval')
        for name in (m.PARENT_ENTRY,m.PARENT_TEST):
            target=d/name; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes((ROOT/name).read_bytes())
        for name in [*m.FAILURE_PINS,m.PARENT_OUT+'/protocol.json',m.PARENT_OUT+'/control_verification.json',m.PARENT_PIPELINE]:
            path=d/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_text('pinned evidence')
        pins={name:m.sha(d/name) for name in m.FAILURE_PINS}
        with patch.object(m,'ROOT',d),patch.object(m,'FAILURE_PINS',pins), \
             patch.object(parent,'source_guard',return_value={'profile':'production_s21'}), \
             patch.object(m,'PARENT_PROTOCOL_SHA256',m.sha(d/m.PARENT_OUT/'protocol.json')), \
             patch.object(m,'PARENT_GATE_SHA256',m.sha(d/m.PARENT_OUT/'control_verification.json')), \
             patch.object(m,'PARENT_PIPELINE_SHA256',m.sha(d/m.PARENT_PIPELINE)), \
             patch.object(m,'_diskfix_native_prepare') as prepare,patch.object(m,'_diskfix_native_pipeline') as pipeline:
            self.assertTrue(m.source_guard(production=True))
            for name in pins:
                path=d/name; original=path.read_bytes(); path.write_bytes(original+b'drift')
                for operation in (m.prepare,m.run_pipeline):
                    with self.assertRaisesRegex(ValueError,'SHA'): operation(d/m.OUT)
                path.write_bytes(original)
            prepare.assert_not_called(); pipeline.assert_not_called()
        self.assertFalse((d/m.OUT).exists()); self.assertFalse((d/m.PIPELINE).exists())

    def test_exact_related_names_and_self_exclusion(self):
        d = self.tempdir()
        names = ['amos_semantic_only_eval.py', 'amos_semantic_only_restored_eval.py',
                 'amos_semantic_only_diskfix_eval.py', 'amos_semantic_only_diskfix_eval.py.bak']
        for i, name in enumerate(names):
            p = d/str(710001+i); p.mkdir(); (p/'cmdline').write_bytes(('python\0/work/'+name+'\0worker\0').encode())
        p = d/str(os.getpid()); p.mkdir(); (p/'cmdline').write_bytes(b'python\0amos_semantic_only_diskfix_eval.py\0run\0')
        self.assertEqual({r['pid'] for r in self.m.related(d)}, {710001, 710002, 710003})

    def test_reserve_preflight_and_backend_pause_recover(self):
        m = self.m; d = self.tempdir(); snapshot = {2: dict(free=40000, uuid='GPU-real')}
        absent = types.SimpleNamespace(returncode=1, stderr='')
        with patch.object(m, 'ROOT', d), patch.object(m.subprocess, 'run', return_value=absent), \
             patch.object(m, 'disk_free_bytes', return_value=m.DISK_RESERVE_BYTES-1):
            with self.assertRaisesRegex(ValueError, 'reserve'): m.preflight(d/m.OUT, processes=[], tmux_exists=False)
            with patch.object(m, 'gpu_snapshot') as gpu:
                self.assertEqual(m.Backend(d, False).available(), {}); gpu.assert_not_called()
        self.assertFalse((d/m.OUT).exists()); self.assertFalse((d/m.PIPELINE).exists())
        with patch.object(m, 'disk_free_bytes', return_value=m.DISK_RESERVE_BYTES), \
             patch.object(m, 'gpu_snapshot', return_value=snapshot):
            self.assertIs(m.Backend(d, False).available(), snapshot)
        with patch.object(m, 'ROOT', d), patch.object(m.subprocess, 'run', return_value=absent), \
             patch.object(m, 'disk_free_bytes', return_value=m.DISK_RESERVE_BYTES):
            (d/m.PARENT_OUT).mkdir(parents=True)
            m.preflight(d/m.OUT, processes=[], tmux_exists=False)
            for processes in ([dict(pid=1, args=['worker'])],):
                with self.assertRaises(ValueError): m.preflight(d/m.OUT, processes=processes, tmux_exists=False)
            (d/m.PIPELINE).write_text('existing')
            with self.assertRaises(FileExistsError): m.preflight(d/m.OUT, processes=[], tmux_exists=False, pipeline=d/m.PIPELINE)

    def test_history_next_attempt_preserves_actual_policy_without_returncode_guess(self):
        m = self.m; canonical = self.job()
        previous = dict(canonical, attempt=3, frame_batch=1, minimum_free_mib=45000,
                        oom_retries=2, not_before=500., avoid_gpus={'2': 900.}, gpu=2)
        value = m.advance_history(canonical, previous)
        self.assertEqual(value['attempt'], 4)
        for name in ('frame_batch', 'minimum_free_mib', 'oom_retries', 'not_before', 'avoid_gpus'):
            self.assertEqual(value[name], previous[name])
        self.assertEqual(value['pending'], canonical['pending'])
        self.assertNotIn('returncode', value); self.assertNotIn('gpu', value)
        self.assertEqual(m.advance_history(canonical, None), canonical)
        with patch.object(m, 'parent_history', side_effect=AssertionError('controls must be fresh')):
            control = m.case_jobs([self.task], strict=False)[0]
        self.assertEqual((control['attempt'], control['frame_batch'], control['oom_retries']), (0, 4, 0))

    def test_case_jobs_keep100_canonical_pending_and_unstarted_attempt0(self):
        m=self.m; tasks=self.tasks(); canonical=m._diskfix_native_case_jobs(tasks)
        history={j['name']:[] for j in canonical}
        history[canonical[0]['name']]=[dict(canonical[0],attempt=0,frame_batch=2,oom_retries=0,not_before=700.,avoid_gpus={'4':800.})]
        with patch.object(m,'parent_history',return_value=history): jobs=m.case_jobs(tasks)
        self.assertEqual(len(jobs),100); self.assertEqual(sum(j['task_count'] for j in jobs),1485)
        self.assertEqual((jobs[0]['attempt'],jobs[0]['frame_batch'],jobs[0]['oom_retries']),(1,2,0))
        self.assertEqual(jobs[0]['not_before'],700.); self.assertEqual(jobs[0]['avoid_gpus'],{'4':800.})
        self.assertTrue(all(j['attempt']==0 for j in jobs[1:]))
        self.assertEqual([j['pending'] for j in jobs],[j['pending'] for j in canonical])

    def test_partial_and_temporary_records_never_enter_union(self):
        with patch.object(self.m,'frozen_records',return_value={}):
            native.ControllerTests.test_partial_results_are_never_completed_classes(self)

    def certified_fixture(self):
        m = self.m; d = self.tempdir(); out = d/m.PARENT_OUT
        def fixture(path, value):
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(value),encoding='utf-8')
        # 100 complete canonical case lists: 9 absent classes in certified cases,
        # 6 in unstarted cases. No selection can consult a metric value.
        tasks = [dict(self.task, case_id=f'{c:04}', global_class_id=201+i, local_class_id=i+1)
                 for c in range(100) for i in range(14 if c<9 or 94<=c else 15)]
        jobs = m._diskfix_native_case_jobs(tasks)
        fixture(out/'jobs.json', dict(cases=jobs, controls=[]))
        found = {}
        for j in jobs[:36]:
            for t in j['pending']:
                base = out/'runs'/j['name']/'classes'/str(t['global_class_id'])
                spec = out/'job_specs'/(j['name']+'_a0.json'); log = out/'logs'/(j['name']+'_a0.log')
                if not spec.exists(): fixture(spec, dict(j, spec=str(spec), log=str(log))); log.parent.mkdir(exist_ok=True); log.write_text('proof')
                rec = base.with_suffix('.json'); marker = base.with_suffix('.done.json')
                fixture(rec, dict(task=t, records=[dict(variant='semantic_only', row=self.row(t))]))
                fixture(marker, dict(files={str(rec): m.sha(rec)}))
                found[m.key(t)] = dict(record=m.read(rec), marker=str(marker), marker_sha256=m.sha(marker), spec=m.read(spec))
        for j in jobs[:33]: fixture(out/'case_success'/(j['name']+'.json'), dict(job_name=j['name']))
        return d, out, jobs, found

    def test_selection_requires_published_case_certificates_and_preserves_all531(self):
        m = self.m; d, out, jobs, found = self.certified_fixture()
        self.assertEqual(len(found), 531)
        p = {'verify_case_success': lambda out, job, found: m.read(out/'case_success'/(job['name']+'.json'))}
        with patch.object(m, 'ROOT', d), patch.object(m, 'parent_engine', return_value=p):
            selected, cases = m.select_certified(found)
            self.assertEqual(len(selected), 486); self.assertEqual(len(cases), 33)
            self.assertTrue(all(v['spec']['name'] not in {j['name'] for j in jobs[33:]} for v in selected.values()))
            for v in found.values(): v['record']['records'][0]['row']['dice'] = 0.999
            self.assertEqual(m.select_certified(found)[1], cases)
            certificate = out/'case_success'/(jobs[0]['name']+'.json')
            certificate.rename(certificate.with_suffix('.retained'))
            with self.assertRaisesRegex(ValueError, '486|33|certified'): m.select_certified(found)

    def test_union_collisions_and_current_controls_are_separate(self):
        m = self.m
        with patch.object(m, 'frozen_records', return_value={('amos','a',1): 'old'}), \
             patch.object(m, '_diskfix_native_records', return_value={('amos','a',1): 'new'}):
            with self.assertRaisesRegex(ValueError, 'duplicate'): m.class_records('out')
            self.assertEqual(m.class_records('out', True), {('amos','a',1): 'new'})
        with self.assertRaisesRegex(ValueError, 'duplicate'): m.union_records({1: 1}, {1: 2})

    def test_parent_failure_verification_rejects_unrecorded_attempts(self):
        m = self.m; d, out, jobs, found = self.certified_fixture()
        read_text=Path.read_text
        private_paths={ROOT/'output/amos_semantic_enospc_failure_full_20261102.json',ROOT/m.FAILURE,
                       ROOT/m.INDEPENDENT,ROOT/m.ANCESTRY}
        def public_fixture_reads(path,*args,**kwargs):
            if path in private_paths:
                raise AssertionError('public fixture must not read private server evidence')
            return read_text(path,*args,**kwargs)
        guard=patch.object(Path,'read_text',autospec=True,side_effect=public_fixture_reads)
        guard.start(); self.addCleanup(guard.stop)
        failure = dict(kind='amos_semantic_shared_volume_enospc_v1',output=m.PARENT_OUT,
                       protocol_sha256=m.PARENT_PROTOCOL_SHA256,gate_sha256=m.PARENT_GATE_SHA256,
                       pipeline_sha256=m.PARENT_PIPELINE_SHA256,related_checks=[[],[]],tmux_exists=False,
                       status=dict(phase='full',running={},failed=[dict(stage='native_worker_complete',
                                   error="OSError(28, 'No space left on device')")]),runs=[])
        for path in (out/'job_specs').glob('*.json'):
            spec = m.read(path); spec.update(gpu=2, actual_frame_batch=4)
            path.write_text(json.dumps(spec))
            failure['runs'].append(dict(name=spec['name'], case=spec['case_id'], control=False, gpu=2,
                                       batch=4, attempt=0, oom_retries=0, files={}, log=spec['log'],
                                       log_sha256=m.sha(spec['log'])))
        m.atomic_new(out/'status.json', failure['status'])
        independent = dict(status=failure['status'], complete_class_records=531, certified_rows=486,
                           full_rehash=True, quiescence=[], failure_evidence_sha256=m.FAILURE_PINS[m.FAILURE],
                           certified_cases=[dict(name=j['name'], classes=len(j['pending'])) for j in jobs[:33]],
                           **{k:failure[k] for k in ('protocol_sha256','gate_sha256','pipeline_sha256')})
        for name, value in ((m.FAILURE,failure),(m.INDEPENDENT,independent),
                            (m.ANCESTRY,dict(passed=True,full_rehash=True,classes=1485))): m.atomic_new(d/name,value)
        p = dict(verify=lambda *a,**k: None, check_control_gate=lambda *a: None,
                 class_records=lambda *a: found,
                 verify_case_success=lambda out,j,f: m.read(out/'case_success'/(j['name']+'.json')))
        with patch.object(m,'ROOT',d), patch.object(m,'_source_guard'), patch.object(m,'parent_quiescence'), \
             patch.object(m,'parent_engine',return_value=p):
            self.assertEqual(len(m.verify_parent()),531)
            first = next((out/'job_specs').glob('*.json')); extra = m.read(first)
            extra.update(attempt=1, spec=str(first.with_name('extra_a1.json')))
            m.atomic_new(extra['spec'],extra)
            with self.assertRaisesRegex(ValueError,'attempt|captured'): m.verify_parent()

    def test_reuse_audit_freezes531_excluded_proofs_and_all_histories(self):
        m = self.m; d, out, jobs, found = self.certified_fixture()
        m.atomic_new(out/'protocol.json',dict(sources={},prepared={}))
        for name in [m.PARENT_PIPELINE,*m.FAILURE_PINS]:
            path=d/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_text('fixture')
        p = dict(verify_case_success=lambda out,j,f: m.read(out/'case_success'/(j['name']+'.json')),
                 class_records=lambda out,control=False: {} if control else found, validate_attempt=lambda *a: None)
        partial=out/'runs'/'excluded_partial.json.tmp'; partial.write_text('incomplete never imported')
        with patch.object(m,'ROOT',d), patch.object(m,'verify_parent',return_value=found), \
             patch.object(m,'_source_guard'),patch.object(m,'parent_quiescence'), \
             patch.object(m,'parent_engine',return_value=p):
            audit=m.build_reuse_audit()
            self.assertEqual((len(audit['records']),len(audit['all_parent_records']),len(audit['cases'])),(486,531,33))
            self.assertEqual(len(audit['histories']),100)
            self.assertEqual(sum(bool(v) for v in audit['histories'].values()),36)
            self.assertIn(str(partial.relative_to(d)).replace('\\','/'),audit['files'])
            self.assertTrue(all(str(Path(v['marker']).relative_to(d)).replace('\\','/') in audit['files'] for v in found.values()))
            fresh=d/'fresh'; m.atomic_new(fresh/'reuse_audit.json',audit)
            m.atomic_new(fresh/'native_bootstrap_audit.json',dict(reuse_audit_sha256=m.sha(fresh/'reuse_audit.json')))
            with patch.object(m,'verify_parent',side_effect=AssertionError('scheduler poll must not rehash model assets')), \
                 patch.object(m,'_source_guard'), patch.object(m,'parent_quiescence'):
                selected=m.frozen_records(fresh)
            self.assertEqual(len(selected),486)
            # The supplied union was already authenticated; per-case validation
            # must not rehash every model asset through the parent reader again.
            with patch.object(m,'frozen_records',side_effect=AssertionError('redundant heavyweight parent validation')):
                m.verify_case_success(fresh,jobs[0],selected)
            excluded=next(v for v in found.values() if v['spec']['name']==jobs[33]['name'])
            Path(excluded['marker']).write_text('drift')
            with self.assertRaisesRegex(ValueError,'changed'): m.frozen_records(fresh)

    def test_schedule_filters_only33_certified_cases_and_preserves_full_pending(self):
        m=self.m; d=self.tempdir()
        tasks=[dict(self.task,case_id=f'{c:04}',global_class_id=201+i,local_class_id=i+1)
               for c in range(100) for i in range(14 if c<9 or 94<=c else 15)]
        jobs=m._diskfix_native_case_jobs(tasks); m.atomic_new(d/'jobs.json',dict(cases=jobs,controls=[]))
        found={m.key(t):{} for j in jobs[:33] for t in j['pending']}
        with patch.object(m,'verify'),patch.object(m,'check_control_gate') as gate, \
             patch.object(m,'class_records',return_value=found),patch.object(m,'schedule_queue') as queue:
            m.schedule(d,False)
            pending=queue.call_args.args[0]
            self.assertEqual((len(pending),sum(j['task_count'] for j in pending)),(67,999))
            self.assertEqual(pending,jobs[33:]); gate.assert_called_once_with(d)

    def test_verification_allows_current_liveworkers_and_low_disk(self):
        m=self.m; live=[dict(pid=44,args=['python','amos_semantic_only_diskfix_eval.py','worker'])]
        absent=types.SimpleNamespace(returncode=1,stderr='')
        with patch.object(m,'related',return_value=live),patch.object(m.subprocess,'run',return_value=absent):
            m.parent_quiescence()
        with patch.object(m,'_source_guard'),patch.object(m,'verify_reuse'),patch.object(m,'verify_parent'), \
             patch.object(m,'_diskfix_native_verify',return_value={'verified':True}), \
             patch.object(m,'disk_free_bytes',side_effect=AssertionError('audit must not require reserve')):
            self.assertTrue(m.verify('out')['verified'])

    def test_bootstrap_freezes_reuse_sha_before_native_publication(self):
        m = self.m; d = self.tempdir(); audit = {'kind': 'fixture', 'files': {}}
        with patch.object(m, 'build_reuse_audit', return_value=audit), \
             patch.object(m, '_bootstrap_function', return_value={'native': True}) as boot:
            path = m._native_bootstrap_preflight(d, [], [])
        self.assertEqual(path, d/'native_bootstrap_audit.json')
        self.assertEqual(m.read(path)['reuse_audit_sha256'], m.sha(d/'reuse_audit.json'))
        boot.assert_called_once_with(root=ROOT, out=d, jobs=[], controls=[])

    def test_policy_and_merge1485_union_verify_only(self):
        m = self.m; d = self.tempdir(); tasks = self.tasks(); jobs = m._diskfix_native_case_jobs(tasks)
        for name, value in [('status.json', dict(phase='full', pending=0, running={}, failed=[])),
                            ('manifest.json',dict(tasks=tasks)), ('jobs.json',dict(cases=jobs)),
                            ('protocol.json',{}),('control_verification.json',{}),('reuse_audit.json',{})]:
            m.atomic_new(d/name, value)
        rows = [self.row(t) for t in tasks]
        found = {m.key(t): dict(record={'records':[dict(variant='semantic_only',row=r)]}, spec={'frame_batch':4}) for t,r in zip(tasks,rows)}
        with patch.object(m, 'verify'), patch.object(m, 'check_control_gate'), patch.object(m, 'related', return_value=[]), \
             patch.object(m, 'class_records', return_value=found), patch.object(m, 'native_reference', return_value=rows), \
             patch.object(m, 'verify_case_success') as cases:
            result = m.merge(d)
            self.assertEqual((result['rows'],result['cases'],result['fresh_rows'],result['reused_rows'],result['unused_parent_rows']), (1485,100,999,486,45))
            self.assertFalse(result['controls_in_final']); self.assertEqual(cases.call_count, 100)
            before = m.sha(d/'merged_metrics.json'); self.assertEqual(m.merge(d), result)
            self.assertEqual(m.sha(d/'merged_metrics.json'), before)
            found.pop(next(iter(found)))
            with self.assertRaises(ValueError): m.merge(d)
        self.assertEqual(m.policy_extension()['fresh_rows'],999)
        self.assertEqual(m.policy_extension()['semantic_prompt_count'],1)

    test_scope_exact_cases_classes_keys_and_training_exclusion = native.ControllerTests.test_scope_exact_cases_classes_keys_and_training_exclusion
    test_oom_ladder_and_resource75_do_not_change_protocol = native.ControllerTests.test_oom_ladder_and_resource75_do_not_change_protocol
    test_atomic_publication_refuses_replacement = native.ControllerTests.test_atomic_publication_refuses_replacement
    test_independent_real_gpu_control_gate_required = native.ControllerTests.test_independent_real_gpu_control_gate_required
    test_class_macro_and_fsum_do_not_follow_task_weighted_or_naive_sum = native.ControllerTests.test_class_macro_and_fsum_do_not_follow_task_weighted_or_naive_sum


class DiskfixWorkers(semantic.SemanticWorkerTests):
    def setUp(self):
        super().setUp(); self.m = controller()
        # These fixtures have no server parent. Only that read boundary is stubbed.
        p = patch.object(self.m, 'frozen_records', return_value={}); p.start(); self.addCleanup(p.stop)


class DiskfixScheduler(semantic.SemanticSchedulerTests):
    def setUp(self):
        super().setUp(); self.m = controller()


if __name__ == '__main__': unittest.main()
