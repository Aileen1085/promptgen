import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import v10_2_ct13_ema_diagnostic as diagnostic


def metrics():
    return {
        'validation_protocol': {
            'selected_tasks': 267, 'covered_classes': 163, 'requested_classes': 163,
            'all_nonempty_classes_covered': True, 'na_classes': [40, 41],
            'metric_space': 'raw_nifti_full_volume', 'anchor_coarse_crop': False,
        },
        'primary_threshold': 0.6,
        'selection': {'score': 0.7878666850761594},
        'mean': {'dice': 0.7878666850761594},
        'per_dataset': {},
    }


class DiagnosticTest(unittest.TestCase):
    def test_checkpoint_guard_rejects_ema_saved_epoch_files(self):
        path = Path('v10/output/v10_2_ct13_e490_lowpg_lr_control_10_20261004/20261004_121030/last.pth')
        state = {'epoch': 10, 'ema_state': {'num_updates': 2000},
                 'validation_metrics': metrics()}
        diagnostic.assert_checkpoint(path, state)
        for name in ('epoch010.pth', 'best.pth'):
            with self.assertRaises(ValueError):
                diagnostic.assert_checkpoint(path.with_name(name), state)
        state['ema_state']['num_updates'] = 1000
        with self.assertRaises(ValueError):
            diagnostic.assert_checkpoint(path, state)

    def test_rejects_incomplete_protocol(self):
        value = metrics()
        diagnostic.assert_protocol(value)
        value['validation_protocol']['covered_classes'] = 162
        with self.assertRaises(ValueError):
            diagnostic.assert_protocol(value)

    def test_rejects_wrong_na_and_threshold(self):
        for key, value in [('na_classes', [40, 41, 69]), ('anchor_coarse_crop', True)]:
            result = metrics()
            result['validation_protocol'][key] = value
            with self.assertRaises(ValueError):
                diagnostic.assert_protocol(result)
        result = metrics()
        result['primary_threshold'] = 0.65
        with self.assertRaises(ValueError):
            diagnostic.assert_protocol(result)

    def test_args_are_copied_without_changing_training_protocol(self):
        source = dict(validation_max_tasks=0, val_foreground_mode='scribble',
                      val_background_mode='scribble', mask_threshold=0.6,
                      anchor_coarse_crop=False, validation_thresholds=[.55, .6, .65],
                      validation_seed=2027, prompt_lr=2.5e-7,
                      train_cases_per_epoch=200)
        original = copy.deepcopy(source)
        result = diagnostic.evaluation_args(source, gpu=1)
        self.assertEqual(source, original)
        self.assertEqual(result.prompt_lr, source['prompt_lr'])
        self.assertEqual(result.validation_seed, 2027)
        self.assertEqual(result.gpu, '1')
        self.assertEqual(result.dataset_foreground_mode, 'scribble')

    def test_args_reject_truncated_validation(self):
        with self.assertRaises(ValueError):
            diagnostic.evaluation_args({'validation_max_tasks': 10}, gpu=1)

    def test_comparison_rejects_nonfinite_or_ema_reproduction_error(self):
        live, ema = metrics(), metrics()
        live['selection']['score'] += .003
        result = diagnostic.compare(live, ema, expected_ema=ema['selection']['score'])
        self.assertAlmostEqual(result['live_minus_ema_weighted_dice'], .003)
        with self.assertRaises(ValueError):
            diagnostic.compare(live, ema, expected_ema=.7)
        live['selection']['score'] = float('nan')
        with self.assertRaises(ValueError):
            diagnostic.compare(live, ema, expected_ema=.7878666850761594)

    def test_results_never_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'result.json'
            diagnostic.save_json_once(path, {'original': True})
            with self.assertRaises(FileExistsError):
                diagnostic.save_json_once(path, {'original': False})
            self.assertTrue(json.loads(path.read_text())['original'])


if __name__ == '__main__':
    unittest.main()
