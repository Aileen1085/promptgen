import unittest
from argparse import Namespace
from v10.v10_2_ct13_hard_source import COUNTS, INITIALIZATION, focused_args, source_ratios


class HardSourceTest(unittest.TestCase):
    def test_global_budget_and_hard_sources(self):
        self.assertEqual(sum(COUNTS.values()), 200)
        self.assertEqual(len(COUNTS), 13)
        self.assertEqual(COUNTS['lndb'], 18)
        self.assertEqual(COUNTS['covid19_20'], 20)
        self.assertAlmostEqual(sum(source_ratios().values()), 1)

    def test_only_schedule_and_sampling_change(self):
        source = dict(prompt_lr=2.5e-7, adapter_lr=5e-7, memory_adapter_lr=5e-7,
                      decoder_lr=2e-7, mask_threshold=.60, anchor_coarse_crop=False,
                      seg_pos_weight_max=2.5, model_input_size=1024,
                      val_foreground_mode='scribble', val_background_mode='scribble',
                      lr_warmup_epochs=2, ema_decay=.9995)
        result = focused_args(source, initialization=INITIALIZATION, output='new', gpu=1)
        for key, value in source.items():
            self.assertEqual(getattr(result, key), value, key)
        self.assertEqual(result.epochs, 20)
        self.assertEqual(result.sam_encoder_unfreeze_epoch, 21)
        self.assertEqual(result.train_cases_per_epoch, 200)
        self.assertEqual(result.resume_checkpoint, '')
        self.assertTrue(result.validate_before_train)

    def test_smoke_preserves_short_budget(self):
        result = focused_args({}, initialization=INITIALIZATION, output='new', gpu=1, smoke=True)
        self.assertEqual(result.epochs, 1)
        self.assertEqual(result.train_cases_per_epoch, 26)
        self.assertFalse(result.validate_before_train)


if __name__ == '__main__':
    unittest.main()
