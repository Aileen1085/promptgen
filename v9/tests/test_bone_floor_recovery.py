import ast
import importlib
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import SimpleNamespace
import unittest


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.r = importlib.import_module('tools.bone_floor_recovery')

    def test_validation_only_does_not_construct_training_quota(self):
        def forbidden(*a, **kw):
            raise ValueError('no candidates for source bone')
        audit = {}
        factory = self.r.validation_only_factory(forbidden, audit)
        value = factory([], SimpleNamespace(validation_only=True), return_target_class=True,
                        class_weights={10003: 1.0})
        self.assertEqual(audit['training_dataset_constructions_skipped'], 1)
        with self.assertRaises(RuntimeError):
            len(value)
        with self.assertRaises(RuntimeError):
            value[0]

    def test_non_validation_training_still_raises_original_error(self):
        def training(*a, **kw):
            raise ValueError('no candidates for source bone')
        f = self.r.validation_only_factory(training, {})
        with self.assertRaisesRegex(ValueError, 'no candidates'):
            f([], SimpleNamespace(validation_only=False), class_weights={3: 1})

    def test_non_training_factory_calls_are_unchanged(self):
        calls = []
        token = object()
        def original(*a, **kw):
            calls.append((a, kw)); return token
        f = self.r.validation_only_factory(original, {})
        args = SimpleNamespace(validation_only=True)
        self.assertIs(f(['val'], args, target_class=10003), token)
        self.assertEqual(calls, [((['val'], args), {'target_class': 10003})])

    def test_native_core_validation_branch_precedes_training_dataset_use(self):
        tree = ast.parse((self.r.ROOT/'v9_training_core.py').read_text(encoding='utf-8'))
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        branch = next(n for n in main.body if isinstance(n, ast.If)
                      and 'validation_only' in ast.unparse(n.test))
        self.assertTrue(any(isinstance(n, ast.Return) for n in ast.walk(branch)))
        # Loaded E360 must materialize lazy parameters; fail closed if warmup tries the sentinel.
        self.assertTrue(any(isinstance(n, ast.Name) and n.id == 'train_dataset'
                            and n.lineno > branch.lineno for n in ast.walk(main)))

    def test_recovery_requires_failed_exited_zero_prediction_state(self):
        self.r.validate_old_state({'phase': 'smoke', 'running': {},
            'failed': [{'name': str(i)} for i in range(4)]}, [], 0)
        for state, procs, count in [({'phase': 'full', 'running': {}, 'failed': [1]}, [], 0),
                                    ({'phase': 'smoke', 'running': {'2': 1}, 'failed': [1]}, [], 0),
                                    ({'phase': 'smoke', 'running': {}, 'failed': []}, [], 0),
                                    ({'phase': 'smoke', 'running': {}, 'failed': [1]}, ['123'], 0),
                                    ({'phase': 'smoke', 'running': {}, 'failed': [1]}, [], 1)]:
            with self.assertRaises(ValueError): self.r.validate_old_state(state, procs, count)

    def test_readonly_reuse_detects_changed_bytes(self):
        folder = self.r.ROOT/'output'; folder.mkdir(exist_ok=True)
        with NamedTemporaryFile(dir=folder, suffix='.test.json', delete=False) as f:
            q = Path(f.name)
        try:
            q.write_text('frozen')
            refs = {str(q): self.r.sha(q)}
            self.r.verify_files(refs)
            q.write_text('changed')
            with self.assertRaises(ValueError): self.r.verify_files(refs)
        finally:
            q.unlink()

    def test_process_guard_excludes_self_and_targets_actual_commands(self):
        self.assertTrue(self.r.process_matches(['python', 'tools/bone_floor_grid.py', 'launch'], 2, 1))
        self.assertTrue(self.r.process_matches(['python', 'tools/bone_floor_recovery.py', 'worker'], 2, 1))
        self.assertFalse(self.r.process_matches(['python', 'tools/bone_floor_recovery.py', 'status'], 2, 1))
        self.assertFalse(self.r.process_matches(['python', 'tools/bone_floor_recovery.py', 'launch'], 1, 1))


if __name__ == '__main__': unittest.main()
