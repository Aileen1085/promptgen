"""Recovery orchestration tests; CPU fixtures never establish a real GPU gate."""
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

from tests.test_amos_bg_scribble_module_eval import Proxy
from tests import test_amos_semantic_only_eval as semantic
from tests import test_amos_prompt_ablation_eval as native

ROOT = Path(__file__).resolve().parents[1]
ENTRY = 'tools/amos_semantic_only_restored_eval.py'
TEST = 'tests/test_amos_semantic_only_restored_eval.py'


def controller():
    return Proxy(importlib.import_module('tools.amos_semantic_only_restored_eval').compile_context().engine())


class RestoredControllerTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((ROOT/ENTRY).is_file(), 'independent restored-data controller is required')
        original_mkdtemp = tempfile.mkdtemp
        temp_patch = patch.object(tempfile, 'mkdtemp', side_effect=lambda *a, **k: original_mkdtemp(dir=str(ROOT/'output')))
        temp_patch.start(); self.addCleanup(temp_patch.stop)
        self.entry = importlib.import_module('tools.amos_semantic_only_restored_eval')
        self.old = importlib.import_module('tools.amos_semantic_only_eval')
        self.m = controller()
        self.task = dict(source_name='amos', case_id='0001', global_class_id=201,
                         local_class_id=1, class_name='spleen')

    def tempdir(self):
        # Retain every fixture under output: no recursive cleanup.
        return Path(tempfile.mkdtemp(prefix='semantic_restored_test_', dir=str(ROOT/'output')))

    def test_private_bindings_and_original_context_are_untouched(self):
        before = (self.old._CONTEXT, self.old.OUT, self.old.TMUX, self.old.PIPELINE)
        old_private = self.old.compile_context().engine()
        m = controller()
        for name in ('worker', 'schedule_queue', 'class_records', 'make_attempt', 'main',
                     'verify_worker_complete', 'related', 'preflight', 'dependency_sources',
                     '_source_guard', 'engine', 'runtime'):
            self.assertIs(m.ns[name].__globals__, m.ns, name)
        self.assertIs(m.engine(), m.ns)
        self.assertIs(m._bootstrap_function.__globals__['engine'](), m.ns)
        self.assertIsNot(m.ns, old_private)
        self.assertEqual((self.old._CONTEXT, self.old.OUT, self.old.TMUX, self.old.PIPELINE), before)
        self.assertEqual(old_private['OUT'], self.old.OUT)
        self.assertEqual(Path(m.ns['__file__']), ROOT/ENTRY)

    def test_new_paths_are_exclusive_and_do_not_replay_old_paths(self):
        self.assertEqual(self.m.OUT, 'output/amos_v102_e490_floor070_semantic_only_restored_20261102')
        self.assertEqual(self.m.TMUX, 'amos_v102_floor070_semantic_only_restored_20261102')
        self.assertEqual(self.m.PIPELINE, 'output/amos_semantic_only_restored_pipeline_20261102.log')
        with patch.object(self.m, '_source_guard'), patch.object(self.m.subprocess, 'run'):
            with self.assertRaisesRegex(ValueError, 'dedicated|recovery'):
                self.m.run_pipeline(ROOT/self.old.OUT)
        self.assertFalse((ROOT/self.m.OUT).exists())
        self.assertFalse((ROOT/self.m.PIPELINE).exists())

    def test_context_cache_is_independent_and_main_dispatches_current_namespace(self):
        sentinel = object()
        with patch.object(self.entry, '_CONTEXT', None), \
             patch.object(self.entry, 'compile_context', return_value=types.SimpleNamespace(engine=lambda: {'main': lambda: sentinel})) as compile_new:
            self.assertIs(self.entry.main(), sentinel)
            self.assertIs(self.entry.context(), self.entry.context())
            self.assertEqual(compile_new.call_count, 1)

    def test_source_drift_rejected_before_compiling_or_original_guard(self):
        d = self.tempdir()
        path = d/self.entry.ORIGINAL_ENTRY
        path.parent.mkdir(parents=True); path.write_bytes(b'changed original controller')
        with patch.object(self.entry, 'ROOT', d), patch.object(self.old, 'source_guard') as guard:
            with self.assertRaisesRegex(ValueError, 'SHA'):
                self.entry.compile_context()
            guard.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'SHA'):
            self.entry.compile_context(source=b'pass')

    def test_original_production_profile_guard_is_mandatory(self):
        with patch.object(self.old, 'source_guard', side_effect=ValueError('production S21 sentinel')) as guard:
            with self.assertRaisesRegex(ValueError, 'S21 sentinel'):
                self.entry.source_guard(production=True)
        guard.assert_called_once_with(production=True)

    def fixture_production_guard(self):
        d = self.tempdir()
        for name in (self.entry.ORIGINAL_ENTRY, self.entry.ORIGINAL_RUNTIME):
            target = d/name; target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT/name).read_bytes())
        for name in self.entry.EVIDENCE_SHA256:
            path = d/name; path.parent.mkdir(parents=True, exist_ok=True)
            if name == self.entry.DATA_READY:
                path.write_text(json.dumps(dict(self.entry.READINESS_REQUIRED, time=1793621159.0977242)), encoding='utf-8')
            else:
                path.write_text('retained provenance fixture\n', encoding='utf-8')
        pins = {name: hashlib.sha256((d/name).read_bytes()).hexdigest() for name in self.entry.EVIDENCE_SHA256}
        return d, pins

    def test_production_pins_all_evidence_and_validates_readiness_schema(self):
        d, pins = self.fixture_production_guard()
        with patch.object(self.entry, 'ROOT', d), patch.object(self.entry, 'EVIDENCE_SHA256', pins), \
             patch.object(self.old, 'source_guard', return_value={'profile': 'production_s21'}):
            proof = self.entry.source_guard(production=True)
            self.assertEqual(proof['recovery_evidence'], pins)
            payload = json.loads((d/self.entry.DATA_READY).read_text())
            payload['model_launched'] = True
            (d/self.entry.DATA_READY).write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, 'SHA'):
                self.entry.source_guard(production=True)
            pins[self.entry.DATA_READY] = self.entry.sha(d/self.entry.DATA_READY)
            with self.assertRaisesRegex(ValueError, 'readiness'):
                self.entry.source_guard(production=True)

    def test_missing_or_drifted_evidence_refused_before_pipeline_or_prepare_writes(self):
        d, pins = self.fixture_production_guard()
        with patch.object(self.m, 'ROOT', d), patch.object(self.m, 'EVIDENCE_SHA256', pins), \
             patch.object(self.old, 'source_guard', return_value={'profile': 'production_s21'}), \
             patch.object(self.m, '_native_prepare') as prepare, patch.object(self.m, '_native_run_pipeline') as run:
            for name in pins:
                path = d/name; before = path.read_bytes(); path.write_bytes(before+b'drift')
                for operation in (self.m.prepare, self.m.run_pipeline):
                    with self.assertRaisesRegex(ValueError, 'SHA'):
                        operation(d/self.m.OUT)
                path.write_bytes(before)
            ready = d/self.entry.DATA_READY
            renamed = ready.with_suffix('.retained')
            ready.rename(renamed)
            with self.assertRaises(FileNotFoundError):
                self.m.run_pipeline(d/self.m.OUT)
            prepare.assert_not_called(); run.assert_not_called()
        self.assertFalse((d/self.m.OUT).exists()); self.assertFalse((d/self.m.PIPELINE).exists())

    def test_nonproduction_help_does_not_require_server_evidence(self):
        self.assertTrue(self.entry.source_guard(production=False))
        with patch.object(sys, 'argv', [str(ROOT/ENTRY), '--help']):
            with self.assertRaises(SystemExit) as error:
                self.m.main()
        self.assertEqual(error.exception.code, 0)

    def test_related_uses_exact_old_and_new_argv_basenames_and_excludes_self(self):
        proc = self.tempdir()
        examples = [
            (810001, ['python', '/work/tools/amos_semantic_only_eval.py', 'run']),
            (810002, ['python', '/work/tools/amos_semantic_only_restored_eval.py', 'prepare']),
            (810003, ['python', '/work/tools/amos_semantic_only_restored_eval.py', 'worker']),
            (810004, ['python', '/work/tools/amos_semantic_only_restored_eval.py.bak', 'run']),
            (810005, ['python', 'other.py', '--description=amos_semantic_only_eval.py', 'run']),
            (810006, ['python', '/work/tools/amos_semantic_only_restored_eval.py', 'status']),
            (os.getpid(), ['python', '/work/tools/amos_semantic_only_restored_eval.py', 'run']),
        ]
        for pid, args in examples:
            path = proc/str(pid); path.mkdir(); (path/'cmdline').write_bytes(('\0'.join(args)+'\0').encode())
        self.assertEqual({r['pid'] for r in self.m.related(proc)}, {810001, 810002, 810003})

    def test_preflight_refuses_old_output_old_tmux_and_both_process_generations(self):
        d = self.tempdir(); out = d/self.m.OUT
        absent = types.SimpleNamespace(returncode=1, stderr='')
        present = types.SimpleNamespace(returncode=0, stderr='')
        with patch.object(self.m, 'ROOT', d), patch.object(self.m.subprocess, 'run', return_value=absent):
            self.m.preflight(out, processes=[], tmux_exists=False)
            for name in ('amos_semantic_only_eval.py', 'amos_semantic_only_restored_eval.py'):
                with self.assertRaisesRegex(ValueError, 'process|duplicate'):
                    self.m.preflight(out, processes=[dict(pid=42, args=['python', name, 'run'])], tmux_exists=False)
            with patch.object(self.m.subprocess, 'run', return_value=present):
                with self.assertRaisesRegex(ValueError, 'tmux|session'):
                    self.m.preflight(out, processes=[], tmux_exists=False)
            (d/self.old.OUT).mkdir(parents=True)
            with self.assertRaisesRegex(FileExistsError, 'original|old'):
                self.m.preflight(out, processes=[], tmux_exists=False)

    def test_old_pipeline_is_retained_evidence_and_new_pipeline_exclusive(self):
        d = self.tempdir(); out = d/self.m.OUT; old_log = d/self.old.PIPELINE
        old_log.parent.mkdir(parents=True); old_log.write_text('immutable original failure')
        absent = types.SimpleNamespace(returncode=1, stderr='')
        with patch.object(self.m, 'ROOT', d), patch.object(self.m.subprocess, 'run', return_value=absent):
            self.m.preflight(out, processes=[], tmux_exists=False, pipeline=d/self.m.PIPELINE)
            (d/self.m.PIPELINE).write_text('already started')
            with self.assertRaises(FileExistsError):
                self.m.preflight(out, processes=[], tmux_exists=False, pipeline=d/self.m.PIPELINE)
        self.assertEqual(old_log.read_text(), 'immutable original failure')

    def test_backend_child_executes_new_entry_and_retains_native_cache_readonly_env(self):
        d = self.tempdir(); job = dict(log=str(d/'child.log'), spec=str(d/'job.json'), gpu=2)
        with patch.object(self.m.subprocess, 'Popen', return_value=object()) as spawn:
            _, handle = self.m.Backend(d, False).spawn(job)
            handle.close()
        argv = spawn.call_args.args[0]
        self.assertEqual(argv, [sys.executable, str(ROOT/ENTRY), 'worker', '--job', job['spec']])
        self.assertEqual(spawn.call_args.kwargs['env']['V9_SHARED_CACHE_WRITE_POLICY'], 'read_only')

    def test_source_inventory_freezes_new_sources_and_provenance_without_prediction_claim(self):
        self.assertIn(ENTRY, self.m.SOURCE_SEEDS); self.assertIn(TEST, self.m.SOURCE_SEEDS)
        self.assertTrue(set(self.old.compile_context().engine()['SOURCE_SEEDS']) <= set(self.m.SOURCE_SEEDS))
        d, pins = self.fixture_production_guard()
        source = d/'retained.py'; source.write_text('pass\n')
        with patch.object(self.m, 'ROOT', d), patch.object(self.m, '_native_dependency_sources', return_value=[source]):
            inventory = self.m.file_inventory(self.m.dependency_sources())
            self.assertEqual(set(inventory), {'retained.py', *self.entry.EVIDENCE_SHA256})
            self.m.verify_inventory(inventory)
            (d/self.entry.DATA_READY).write_text('drift')
            with self.assertRaises(ValueError):
                self.m.verify_inventory(inventory)

    def test_scientific_policy_and_independent_gpu_gate_are_inherited(self):
        old = self.old.compile_context().engine()
        for name in ('VARIANTS', 'CONTROL_VARIANTS', 'PROMPT_VARIANT', 'PRODUCTION_ROWS',
                     'WEIGHTS', 'WEIGHT_SHA', 'ALLOWED', 'CACHE'):
            self.assertEqual(self.m.ns[name], old[name], name)
        self.assertEqual(self.m.policy_extension(), old['policy_extension']())
        self.assertEqual(self.m.runtime().__name__, 'tools.amos_semantic_only_runtime')
        native.ControllerTests.test_independent_real_gpu_control_gate_required(self)

    tasks = native.ControllerTests.tasks
    job = native.ControllerTests.job
    row = native.ControllerTests.row
    record = native.ControllerTests.record
    test_scope_exact_cases_classes_keys_and_training_exclusion = native.ControllerTests.test_scope_exact_cases_classes_keys_and_training_exclusion
    test_oom_ladder_and_resource75_do_not_change_protocol = native.ControllerTests.test_oom_ladder_and_resource75_do_not_change_protocol
    test_atomic_publication_refuses_replacement = native.ControllerTests.test_atomic_publication_refuses_replacement
    test_partial_results_are_never_completed_classes = native.ControllerTests.test_partial_results_are_never_completed_classes


class RestoredAtomicTests(semantic.SemanticAtomicTests):
    def setUp(self):
        # Reuse native atomic/merge checks with only the new controller substituted.
        self.entry = importlib.import_module('tools.amos_semantic_only_restored_eval')
        self.m = controller()
        self.task = dict(source_name='amos', case_id='a', global_class_id=201, local_class_id=1, class_name='spleen')
        original_mkdtemp = tempfile.mkdtemp
        temp_patch = patch.object(tempfile, 'mkdtemp', side_effect=lambda *a, **k: original_mkdtemp(dir=str(ROOT/'output')))
        temp_patch.start(); self.addCleanup(temp_patch.stop)

    # Original-specific path assertions are covered above for recovery paths.
    test_private_current_bindings_and_fresh_semantic_scope = None

    def test_preflight_refuses_output_tmux_process_and_pipeline(self):
        d = Path(tempfile.mkdtemp()); m = self.m
        with patch.object(m, 'ROOT', d), patch.object(m.subprocess, 'run', return_value=types.SimpleNamespace(returncode=1, stderr='')):
            out = d/m.OUT; log = d/'log'
            m.preflight(out, processes=[], tmux_exists=False, pipeline=log)
            for kwargs in (dict(processes=[{'pid': 42}], tmux_exists=False), dict(processes=[], tmux_exists=True)):
                with self.assertRaises(ValueError): m.preflight(out, **kwargs)
            log.write_text('existing')
            with self.assertRaises(FileExistsError): m.preflight(out, processes=[], tmux_exists=False, pipeline=log)
            out.mkdir(parents=True)
            with self.assertRaises(FileExistsError): m.preflight(out, processes=[], tmux_exists=False)


class RestoredWorkerTests(semantic.SemanticWorkerTests):
    def setUp(self):
        super().setUp(); self.m = controller()


class RestoredSchedulerTests(semantic.SemanticSchedulerTests):
    def setUp(self):
        super().setUp(); self.m = controller()


if __name__ == '__main__':
    unittest.main()
