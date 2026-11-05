"""CPU contract tests; synthetic arrays never represent patient predictions."""
import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

SCRIPT = Path(__file__).absolute().parents[1] / 'tools/capture_qualitative_nninteractive_case.py'


class CaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = None
        if SCRIPT.exists():
            spec = importlib.util.spec_from_file_location('native_capture', SCRIPT)
            cls.mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.mod)

    def setUp(self):
        self.assertIsNotNone(self.mod, 'isolated capture entry must exist')

    def test_observer_copies_only_full_binary_prediction_and_metric_parity(self):
        observer = self.mod.MaskObserver((2, 3, 4))
        pred = np.zeros((2, 3, 4), dtype=np.uint8); pred[0, 0, 0] = 1
        observer.record(pred, pred, {'dice': 1.})
        pred[:] = 0
        self.assertEqual(observer.mask.sum(), 1)
        observer.record(np.zeros((1, 3, 4)), np.zeros((1, 3, 4)), {'dice': 1.})
        self.assertEqual(observer.full_calls, 1)

    def test_observer_rejects_surrogates_conflicting_full_calls_and_bad_dice(self):
        for pred in (np.full((2, 3, 4), .3), np.full((2, 3, 4), np.nan)):
            with self.assertRaises(ValueError):
                self.mod.MaskObserver((2, 3, 4)).record(pred, pred, {'dice': 1.})
        observer = self.mod.MaskObserver((2, 3, 4))
        gt = np.ones((2, 3, 4), np.uint8)
        with self.assertRaises(ValueError):
            observer.record(gt, gt, {'dice': .1})
        observer.record(gt, gt, {'dice': 1.})
        with self.assertRaises(ValueError):
            observer.record(np.zeros_like(gt), gt, {'dice': 0.})

    def test_factory_hooks_restore_even_on_failure_and_keep_original_result(self):
        metric = types.SimpleNamespace(binary_prompt_metrics_from_masks=lambda p, t: {'dice': 1.})
        base = types.SimpleNamespace(metric_module=lambda: metric)
        native = types.SimpleNamespace(base_module=lambda: base)
        factory, metric_factory, original = native.base_module, base.metric_module, metric.binary_prompt_metrics_from_masks
        observer = self.mod.MaskObserver((2, 3, 4))
        with self.assertRaisesRegex(RuntimeError, 'delegate'):
            with self.mod.observe_native(native, observer):
                value = native.base_module().metric_module().binary_prompt_metrics_from_masks(np.ones((2, 3, 4)), np.ones((2, 3, 4)))
                self.assertEqual(value, {'dice': 1.})
                raise RuntimeError('delegate')
        self.assertIs(native.base_module, factory)
        self.assertIs(base.metric_module, metric_factory)
        self.assertIs(metric.binary_prompt_metrics_from_masks, original)
        self.assertEqual(observer.full_calls, 1)

    def test_reference_requires_exact_key_and_all_ten_finite_metrics(self):
        task = dict(source_name='amos', case_id='case-a', global_class_id=123)
        row = dict(task, **{m: 1. for m in self.mod.METRICS})
        self.mod.compare_reference(row, row)
        for bad in (dict(row, case_id='other'), dict(row, dice=.5), dict(row, iou=float('nan'))):
            with self.assertRaises(ValueError):
                self.mod.compare_reference(bad, row)

    def test_five_source_authorization_rejects_extra_task_and_promptgen(self):
        auth = [dict(source_name=s, case_id='x', global_class_id=i) for i, s in enumerate(self.mod.SOURCES)]
        spec = dict(schema_version=1, method='SAM2', task=auth[0], authorized_tasks=auth,
                    shape_dhw=[2, 3, 4], grid_verification=dict(verified=True, grid_id='physical', axes_dhw='ZXY', evidence='export check'))
        self.mod.validate_scope(spec)
        for bad in (dict(spec, task=dict(auth[0], case_id='other')), dict(spec, method='PromptGen'), dict(spec, authorized_tasks=auth[:-1])):
            with self.assertRaises(ValueError):
                self.mod.validate_scope(bad)

    def test_missing_observation_refuses_export_and_success_saves_no_ct(self):
        scratch = SCRIPT.parents[1] / 'output/_synthetic_native_capture_tests'
        scratch.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(dir=scratch))
        observer = self.mod.MaskObserver((2, 3, 4))
        with self.assertRaises(ValueError):
            self.mod.save_capture(root / 'missing', observer, {})
        self.assertFalse((root / 'missing').exists())
        observer.record(np.ones((2, 3, 4)), np.ones((2, 3, 4)), {'dice': 1.})
        self.mod.save_capture(root / 'valid', observer, dict(normal_completion=True))
        self.assertEqual(sorted(p.name for p in (root / 'valid').iterdir()), ['mask.npy', 'provenance.json'])
        np.testing.assert_array_equal(np.load(root / 'valid/mask.npy'), observer.mask)
        with self.assertRaises(FileExistsError):
            self.mod.save_capture(root / 'valid', observer, {})

    def test_capture_delegates_normally_binds_inputs_metrics_and_writes_only_mask(self):
        scratch = SCRIPT.parents[1] / 'output/_synthetic_native_capture_tests'
        scratch.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(dir=scratch))
        auth = [dict(source_name=s, case_id='synthetic', global_class_id=i) for i, s in enumerate(self.mod.SOURCES)]
        task = dict(auth[0], image='synthetic-ct.nii', label='synthetic-gt.nii', local_class_id=1,
                    class_name='synthetic', prompt_cache_name='synthetic.npz')
        reference = dict(task, **{m: 1. for m in self.mod.METRICS})
        job = dict(tasks=[task], output=str(root / 'original'), model='sam2')
        (root / 'job.json').write_text(json.dumps(job), encoding='utf-8')
        spec = dict(schema_version=1, method='SAM2', task=task, authorized_tasks=auth,
                    reference=reference, shape_dhw=[2, 3, 4], job_path='job.json', output_dir='capture',
                    grid_verification=dict(verified=True, grid_id='synthetic', axes_dhw='ZXY', evidence='synthetic'),
                    gpu_uuid='GPU-synthetic', source_sha256={str(SCRIPT): self.mod.sha(SCRIPT)},
                    input_sha256={'job.json': self.mod.sha(root / 'job.json')})
        manifest = root / 'manifest.json'
        manifest.write_text(json.dumps(spec), encoding='utf-8')
        def delegate(method, job_path, actual_job, uuid, observer):
            self.assertEqual(method, 'SAM2')
            observer.record(np.ones((2, 3, 4)), np.ones((2, 3, 4)), reference)
            (root / 'original').mkdir()
            (root / 'original/complete_metrics.json').write_text(json.dumps(dict(per_case_class=[reference])), encoding='utf-8')
        with patch.object(self.mod, 'resource_check', return_value=dict(gpu_uuid='GPU-synthetic')):
            with patch.object(self.mod, 'gpu_free_memory', return_value=40000):
                with patch.object(self.mod, 'required_bindings', return_value=([SCRIPT], [root / 'job.json'])):
                    with patch.object(self.mod, 'run_delegate', side_effect=delegate):
                        result = self.mod.capture(manifest)
        self.assertEqual(result, root / 'capture')
        info = json.loads((result / 'provenance.json').read_text(encoding='utf-8'))
        self.assertTrue(info['normal_completion'])
        self.assertEqual(info['actual_row'], reference)
        self.assertEqual(sorted(p.name for p in result.iterdir()), ['mask.npy', 'provenance.json'])

    def test_nninteractive_accepts_original_integer_gpu_only_when_uuid_matches(self):
        self.assertTrue(callable(getattr(self.mod, 'validate_nn_gpu_index', None)), 'physical integer index mapping guard required')
        inventory = '0, GPU-other\n3, GPU-selected\n4, GPU-four\n'
        with patch.object(self.mod.subprocess, 'check_output', return_value=inventory):
            self.assertEqual(self.mod.validate_nn_gpu_index(3, 'GPU-selected'),
                             dict(physical_index=3, gpu_uuid='GPU-selected'))
            for index, uuid in ((0, 'GPU-selected'), (3, 'GPU-four'), (5, 'GPU-selected'),
                                ('GPU-selected', 'GPU-selected'), (True, 'GPU-selected')):
                with self.subTest(index=index, uuid=uuid):
                    with self.assertRaisesRegex(ValueError, 'GPU'):
                        self.mod.validate_nn_gpu_index(index, uuid)

    def test_resource_requires_30gib_and_records_two_free_checks_and_cuda_probe(self):
        self.assertTrue(callable(getattr(self.mod, 'gpu_free_memory', None)), 'resource inventory helper must exist')
        with patch.object(self.mod, 'gpu_free_memory', side_effect=[40000, 39000]) as free:
            with patch.object(self.mod, 'cuda_probe', return_value=dict(actual_cuda_uuid='synthetic', cuda_tensor_probe=True)) as probe:
                audit = self.mod.resource_check('GPU-synthetic', 30720)
        self.assertEqual(free.call_count, 2)
        probe.assert_called_once_with('GPU-synthetic')
        self.assertEqual(audit['free_before_cuda_mib'], 40000)
        self.assertEqual(audit['free_after_cuda_mib'], 39000)
        self.assertTrue(audit['cuda_tensor_probe'])
        with self.assertRaises(ValueError):
            self.mod.resource_check('GPU-synthetic', 24000)

    def test_resource_refuses_insufficient_memory_before_or_after_probe(self):
        self.assertTrue(callable(getattr(self.mod, 'gpu_free_memory', None)), 'resource inventory helper must exist')
        with patch.object(self.mod, 'gpu_free_memory', return_value=30000):
            with patch.object(self.mod, 'cuda_probe') as probe:
                with self.assertRaises(ValueError):
                    self.mod.resource_check('GPU-synthetic', 30720)
                probe.assert_not_called()
        with patch.object(self.mod, 'gpu_free_memory', side_effect=[40000, 30000]):
            with patch.object(self.mod, 'cuda_probe', return_value=dict(actual_cuda_uuid='synthetic', cuda_tensor_probe=True)):
                with self.assertRaises(ValueError):
                    self.mod.resource_check('GPU-synthetic', 30720)

    def test_actual_task_details_are_immutable_not_just_tuple_key(self):
        self.assertTrue(callable(getattr(self.mod, 'validate_task_details', None)), 'full task detail guard required')
        task = dict(source_name='amos', case_id='synthetic', global_class_id=123,
                    image='ct.nii.gz', label='label.nii.gz', local_class_id=6,
                    class_name='liver', prompt_cache_name='prompt.npz')
        self.mod.validate_task_details(task, task)
        for name, value in [('image', 'other.nii.gz'), ('label', 'other.nii.gz'),
                            ('local_class_id', 5), ('class_name', 'spleen'), ('prompt_cache_name', 'other.npz')]:
            with self.subTest(field=name):
                with self.assertRaises(ValueError):
                    self.mod.validate_task_details(task, dict(task, **{name:value}))

    def test_required_inventory_cannot_omit_actual_files(self):
        self.assertTrue(callable(getattr(self.mod, 'verify_inventory', None)), 'derived mandatory inventory guard required')
        scratch = SCRIPT.parents[1] / 'output/_synthetic_native_capture_tests'
        scratch.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(dir=scratch))
        actual = root / 'actual_input.txt'
        actual.write_text('synthetic', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'mandatory'):
            self.mod.verify_inventory({}, [actual], root)
        result = self.mod.verify_inventory({'actual_input.txt': self.mod.sha(actual)}, [actual], root)
        self.assertEqual(result[str(actual)], self.mod.sha(actual))
        with self.assertRaisesRegex(ValueError, 'SHA'):
            self.mod.verify_inventory({'actual_input.txt': 'incorrect'}, [actual], root)

    def test_mandatory_dependencies_are_derived_for_every_native_method_and_magic(self):
        scratch = SCRIPT.parents[1] / 'output/_synthetic_native_capture_tests'
        scratch.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(dir=scratch)) / 'project'
        cache = root / 'v10/.cache/totalseg_sam2_promptgen_v10_prompt_roi_full_v1'
        for source in ('amos', 'magic'):
            package = cache / 'metadata/shared_adaptive_prompt_v1' / source / 'prompt.npz'
            package.parent.mkdir(parents=True)
            np.savez(package, case_image_name=np.asarray('case.npy'))
        case = cache / '_case_images_v1/case.npy'
        case.parent.mkdir()
        np.save(case, np.zeros((2, 3, 4)))
        split = root / 'configs/ct13_v10_2_split_20260928.json'
        split.parent.mkdir()
        split.write_text(json.dumps(dict(datasets=dict(magic=dict(val=[dict(case_id='synthetic', image='canonical-ct.nii', label='canonical-gt.nii')])))), encoding='utf-8')
        task = dict(source_name='amos', case_id='synthetic', global_class_id=1, image='raw-ct.nii',
                    label='raw-gt.nii', local_class_id=1, class_name='organ', prompt_cache_name='prompt.npz')
        with patch.object(self.mod, 'ROOT', root), patch.object(self.mod, 'background_binding', return_value=root / 'existing-bg.npz'):
            for method in self.mod.METHODS[1:]:
                sources, inputs = self.mod.required_bindings(dict(method=method, task=task), {}, root / 'job.json')
                self.assertIn(case, inputs)
                self.assertIn(split, inputs)
                self.assertIn(root / task['image'], inputs)
                self.assertIn(root / 'utils/metrics.py', sources)
                self.assertTrue(any('native_sam2_physical_eval.py' in str(p) for p in sources))
                self.assertTrue(any(p.suffix in ('.pt', '.pth', '.bin') for p in inputs))
                if method == 'VISTA3D':
                    self.assertIn(root / 'vista3d/official_loader.py', sources)
                if method != 'PAM':
                    self.assertIn(root / 'existing-bg.npz', inputs)
            _, inputs = self.mod.required_bindings(dict(method='SAM2', task=dict(task, source_name='magic')), {}, root / 'job.json')
        self.assertIn(root / 'canonical-ct.nii', inputs)
        self.assertIn(root / 'canonical-gt.nii', inputs)

    def test_background_binding_matches_original_signature_and_fails_closed_on_miss(self):
        self.assertTrue(callable(getattr(self.mod, 'background_binding', None)), 'preexisting background cache guard required')
        scratch = SCRIPT.parents[1] / 'output/_synthetic_native_capture_tests'
        scratch.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(dir=scratch))
        package = root / 'prompt.npz'
        np.savez(package, case_image_name=np.asarray('synthetic.npy'))
        canonical = root / 'canonical-label.nii'
        canonical.write_text('synthetic label stat only', encoding='utf-8')
        split = root / 'configs/ct13_v10_2_split_20260928.json'
        split.parent.mkdir()
        split.write_text(json.dumps(dict(datasets=dict(magic=dict(val=[dict(case_id='synthetic', label=str(canonical))])))), encoding='utf-8')
        spec = dict(method='SAM2', task=dict(source_name='magic', case_id='synthetic', label='raw-nrrd-must-not-be-used'), shape_dhw=[2, 3, 4])
        directory = root / 'preexisting-bg'
        stamp = canonical.stat()
        payload = dict(version='v9_training_vertical_bg_v1', margin_ratio=.08,
                       foreground_package_sha256=self.mod.sha(package), shape=[2,3,4],
                       label_signature=[str(canonical), stamp.st_size, stamp.st_mtime_ns])
        signature = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        expected = directory / (signature + '.npz')
        with patch.object(self.mod, 'ROOT', root):
            with self.assertRaisesRegex(ValueError, 'background'):
                self.mod.background_binding(spec, dict(background_cache_dir=str(directory)), package)
            self.assertFalse(directory.exists())
            directory.mkdir()
            np.savez(expected, signature=np.asarray(signature), coords_dhw=np.asarray([[0,0,0]], np.int32))
            self.assertEqual(self.mod.background_binding(spec, dict(background_cache_dir=str(directory)), package), expected)
            self.assertEqual(self.mod.background_binding(dict(spec, method='MedSAM2'), dict(background_cache_dir=str(directory)), package), expected)
            fixed_dir = root / 'output/native_sam2_amos_sparse_bg_pilot_20261007/background_scribbles_light'
            fixed_dir.mkdir(parents=True)
            fixed = fixed_dir / (signature + '.npz')
            np.savez(fixed, signature=np.asarray(signature), coords_dhw=np.asarray([[0,0,0]], np.int32))
            for method in ('SAM-Med3D Turbo', 'SegVol', 'VISTA3D'):
                self.assertEqual(self.mod.background_binding(dict(spec, method=method), {}, package), fixed)
            np.savez(expected, signature=np.asarray('wrong'), coords_dhw=np.asarray([[0,0,0]], np.int32))
            with self.assertRaisesRegex(ValueError, 'signature'):
                self.mod.background_binding(spec, dict(background_cache_dir=str(directory)), package)


if __name__ == '__main__':
    unittest.main()
