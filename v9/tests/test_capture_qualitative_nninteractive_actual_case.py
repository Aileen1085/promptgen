"""CPU tests for explicitly reporting actual reruns, never selecting a winner."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

SCRIPT = Path(__file__).absolute().parents[1] / 'tools/capture_qualitative_nninteractive_actual_case.py'


class ActualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = None
        if SCRIPT.exists():
            spec = importlib.util.spec_from_file_location('actual_capture', SCRIPT)
            cls.mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.mod)

    def setUp(self):
        self.assertIsNotNone(self.mod, 'separate actual-rerun observer must exist')

    def fixture(self):
        auth = [dict(source_name=s, case_id='synthetic', global_class_id=i) for i, s in enumerate(self.mod.strict.SOURCES)]
        return dict(schema_version=1, method='nnInteractive', task=auth[0], authorized_tasks=auth,
                    shape_dhw=[2,3,4], reference_policy='report_actual_rerun', capture_attempt=1,
                    grid_verification=dict(verified=True, grid_id='synthetic', axes_dhw='ZXY', evidence='synthetic'))

    def test_policy_is_explicit_nninteractive_only_first_frozen_rerun(self):
        spec = self.fixture()
        self.mod.validate_policy(spec)
        for updates in (dict(reference_policy='strict'), dict(method='SAM2'), dict(capture_attempt=2)):
            with self.assertRaises(ValueError):
                self.mod.validate_policy(dict(spec, **updates))
        del spec['reference_policy']
        with self.assertRaises(ValueError):
            self.mod.validate_policy(spec)

    def test_report_retains_large_nsd_and_tiny_dice_differences_without_failing_or_ranking(self):
        reference = dict(self.fixture()['task'], **{m:.8 for m in self.mod.strict.METRICS})
        actual = dict(reference, dice=.800011216, mean_slice_nsd_1mm=.72233,
                      mean_slice_nsd_2mm=.67711, mean_slice_nsd_3mm=.63502)
        report = self.mod.metric_report(actual, reference)
        self.assertFalse(report['all_ten_match_atol_1e_6'])
        self.assertAlmostEqual(report['actual_minus_historical']['dice'], .000011216)
        self.assertAlmostEqual(report['actual_minus_historical']['mean_slice_nsd_3mm'], -.16498)
        self.assertEqual(set(report['actual_minus_historical']), set(self.mod.strict.METRICS))
        self.assertNotIn('winner', report)
        self.assertTrue(self.mod.metric_report(reference, reference)['all_ten_match_atol_1e_6'])

    def test_original_strict_historical_parity_guard_is_unchanged(self):
        reference = dict(self.fixture()['task'], **{m:.8 for m in self.mod.strict.METRICS})
        actual = dict(reference, dice=.800011216)
        original = self.mod.strict.compare_reference
        self.assertFalse(self.mod.metric_report(actual, reference)['all_ten_match_atol_1e_6'])
        self.assertIs(self.mod.strict.compare_reference, original)
        with self.assertRaises(ValueError):
            original(actual, reference)

    def test_exact_keys_finite_all_ten_metrics_and_ranges_remain_mandatory(self):
        reference = dict(self.fixture()['task'], **{m:.8 for m in self.mod.strict.METRICS})
        for updates in (dict(case_id='wrong'), dict(dice=float('nan')), dict(iou=1.1),
                        dict(mean_slice_nsd_2mm=-.1), dict(pred_gt_ratio=-1), dict(precision=True)):
            with self.assertRaises(ValueError):
                self.mod.metric_report(dict(reference, **updates), reference)
        bad = dict(reference); del bad['recall']
        with self.assertRaises(ValueError):
            self.mod.metric_report(bad, reference)
        self.mod.metric_report(dict(reference, pred_gt_ratio=3.5), reference)

    def test_normal_completion_exports_observed_mask_and_actual_metrics_not_old_dice(self):
        root = SCRIPT.parents[1] / 'output/_synthetic_nninteractive_actual_tests'
        root.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(dir=root))
        spec = self.fixture()
        task = dict(spec['task'], image='synthetic.nii', label='synthetic-gt.nii',
                    local_class_id=1, class_name='organ', prompt_cache_name='prompt.npz')
        spec['task'] = task
        prepared = root / 'prepared'; prepared.mkdir()
        (prepared / 'manifest.json').write_text(json.dumps(dict(tasks=[task])), encoding='utf-8')
        (prepared / 'refs.json').write_text(json.dumps(dict(per_case_class=[task])), encoding='utf-8')
        job = dict(model='nninteractive', gpu=3, prepared=str(prepared), run=str(root / 'worker'))
        job_path = root / 'job.json'; job_path.write_text(json.dumps(job), encoding='utf-8')
        strict = self.mod.strict
        actual = dict(task, **{m:1. for m in strict.METRICS})
        spec.update(reference=dict(actual, dice=.9, mean_slice_nsd_1mm=.4), gpu_uuid='GPU-selected',
                    job_path='job.json', output_dir='capture', source_sha256={str(SCRIPT):strict.sha(SCRIPT),
                    str(Path(strict.__file__).absolute()):strict.sha(strict.__file__)},
                    input_sha256={'job.json':strict.sha(job_path)})
        manifest = root / 'manifest.json'; manifest.write_text(json.dumps(spec), encoding='utf-8')
        def delegate(method, path, supplied_job, uuid, observer):
            observer.record(np.ones((2,3,4)), np.ones((2,3,4)), actual)
            Path(job['run']).mkdir()
            (Path(job['run']) / 'complete_metrics.json').write_text(json.dumps(dict(per_case_class=[actual])), encoding='utf-8')
        with patch.object(strict, 'required_bindings', return_value=([Path(strict.__file__).absolute()], [job_path])), \
             patch.object(strict, 'resource_check', return_value=dict(cuda_tensor_probe=True)), \
             patch.object(strict, 'gpu_free_memory', return_value=40000), \
             patch.object(strict, 'validate_nn_gpu_index', return_value=dict(physical_index=3, gpu_uuid='GPU-selected')), \
             patch.object(strict, 'run_delegate', side_effect=delegate):
            output = self.mod.capture(manifest)
        report = json.loads((output / 'provenance.json').read_text(encoding='utf-8'))
        self.assertEqual(report['dice_3d'], 1.)
        self.assertEqual(report['actual_row']['dice'], 1.)
        self.assertEqual(report['reference']['dice'], .9)
        self.assertFalse(report['historical_comparison']['all_ten_match_atol_1e_6'])
        self.assertEqual(sorted(p.name for p in output.iterdir()), ['mask.npy', 'provenance.json'])
        self.assertTrue(report['normal_completion'])
        np.testing.assert_array_equal(np.load(output / 'mask.npy'), np.ones((2,3,4)))
        with self.assertRaises(FileExistsError):
            self.mod.capture(manifest)
        # A distinct synthetic normal-failure path cannot export even a captured mask.
        job['run'] = str(root / 'failed-worker')
        job_path.write_text(json.dumps(job), encoding='utf-8')
        spec['input_sha256']['job.json'] = strict.sha(job_path)
        spec['output_dir'] = 'failed-capture'
        manifest.write_text(json.dumps(spec), encoding='utf-8')
        def failed_delegate(method, path, supplied_job, uuid, observer):
            observer.record(np.ones((2,3,4)), np.ones((2,3,4)), actual)
            raise RuntimeError('synthetic native failure before normal completion')
        with patch.object(strict, 'required_bindings', return_value=([Path(strict.__file__).absolute()], [job_path])), \
             patch.object(strict, 'resource_check', return_value=dict(cuda_tensor_probe=True)), \
             patch.object(strict, 'gpu_free_memory', return_value=40000), \
             patch.object(strict, 'validate_nn_gpu_index', return_value=dict(physical_index=3, gpu_uuid='GPU-selected')), \
             patch.object(strict, 'run_delegate', side_effect=failed_delegate):
            with self.assertRaisesRegex(RuntimeError, 'native failure'):
                self.mod.capture(manifest)
        self.assertFalse((root / 'failed-capture').exists())


if __name__ == '__main__':
    unittest.main()
