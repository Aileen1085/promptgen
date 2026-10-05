import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import v10_2_ct13_decoder_control as control


class Parameter:
    def __init__(self, enabled=True):
        self.requires_grad = enabled

    def requires_grad_(self, enabled):
        self.requires_grad = enabled
        return self


class Module:
    def __init__(self, parameters):
        self.parameters_by_name = parameters

    def named_parameters(self):
        return iter(self.parameters_by_name.items())


class ControlTest(unittest.TestCase):
    def test_snapshot_callback_runs_after_warm_start_not_before(self):
        events = []
        def load(*args, **kwargs):
            events.append('loaded')
            return False
        def capture(*args):
            events.append('captured')
        wrapped = control.after_initialization(load, capture)
        self.assertFalse(wrapped('sam', 'tuning', {}, required=False))
        self.assertEqual(events, ['loaded', 'captured'])

    def test_encoder_frozen_and_core_arm_differs_without_freezing_adapters(self):
        for arm in ('frozen', 'full'):
            decoder = Module({'core_decoder.head.weight': Parameter(),
                              'prompt_3d_adapter.weight': Parameter(),
                              'prompt_memory_adapter.weight': Parameter()})
            sam = Module({'image_encoder.qkv.lora_A': Parameter()})
            sam.sam_mask_decoder = decoder
            control.freeze_control_parameters(sam, arm)
            self.assertFalse(dict(sam.named_parameters())['image_encoder.qkv.lora_A'].requires_grad)
            self.assertEqual(decoder.parameters_by_name['core_decoder.head.weight'].requires_grad, arm == 'full')
            self.assertTrue(decoder.parameters_by_name['prompt_3d_adapter.weight'].requires_grad)
            self.assertTrue(decoder.parameters_by_name['prompt_memory_adapter.weight'].requires_grad)

    def test_configuration_fresh_and_matched(self):
        source = dict(epochs=10, lr_schedule_epochs=40, validate_every=5,
                      train_cases_per_epoch=200, prompt_lr=2.5e-7, adapter_lr=5e-7,
                      memory_adapter_lr=5e-7, decoder_lr=2e-7, ema_decay=.9995,
                      sam_encoder_unfreeze_epoch=6, resume_checkpoint='old/last.pth',
                      resume_new_run=True, validate_before_train=True, seed=2027)
        original = copy.deepcopy(source)
        args = control.control_args(source, initialization='epoch490.pth', output='new', gpu=1)
        self.assertEqual(source, original)
        self.assertEqual(args.resume_checkpoint, '')
        self.assertEqual(args.prompt_generator_checkpoint, 'epoch490.pth')
        self.assertEqual(args.sam_encoder_unfreeze_epoch, 11)
        self.assertEqual(args.seed, 2027)
        self.assertEqual(args.ema_decay, .9995)
        self.assertEqual(args.epochs, 10)
        self.assertEqual(args.train_cases_per_epoch, 200)

    def test_smoke_is_separate_and_no_formal_validation(self):
        source = dict(epochs=10, validate_every=5, train_cases_per_epoch=200)
        args = control.control_args(source, initialization='epoch490.pth', output='smoke', gpu=1, smoke=True)
        self.assertEqual(args.epochs, 1)
        self.assertEqual(args.train_cases_per_epoch, 26)
        self.assertFalse(args.validate_before_train)
        self.assertEqual(args.validate_every, 5)
        self.assertEqual(args.sam_encoder_unfreeze_epoch, 11)

    def test_completed_diagnostic_gate_rejects_wrong_ema_or_incomplete(self):
        good = {'ema_weighted_dice': .7878666850761594, 'ema_reproduction_delta': 0,
                'live_weighted_dice': .7875015128070333}
        control.assert_diagnostic_gate(good)
        for bad in ({}, dict(good, ema_reproduction_delta=.01),
                    dict(good, live_weighted_dice=float('nan'))):
            with self.assertRaises(ValueError):
                control.assert_diagnostic_gate(bad)


if __name__ == '__main__':
    unittest.main()
