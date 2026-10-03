import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import v10_2_sam_finetuning as tuning


ROOT = Path(__file__).resolve().parents[1]


def worker_schedule_horizon(args):
    tree = ast.parse((ROOT / 'finetune_totalseg_sam2_promptgen_v10_joint.py').read_text(encoding='utf-8'))
    call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name) and node.func.id == 'apply_staged_learning_rates')
    value = next(item.value for item in call.keywords if item.arg == 'total_epochs')
    return eval(compile(ast.Expression(value), '<schedule-horizon>', 'eval'), {'args': args, 'getattr': getattr, 'int': int})


class ShortControlScheduleTests(unittest.TestCase):
    def test_entry_rejects_schedule_shorter_than_training(self):
        tree = ast.parse((ROOT / 'finetune_multisource_sam2_v10_2_encoder_decoder.py').read_text(encoding='utf-8'))
        validate = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_validate')
        namespace = {}
        exec(compile(ast.Module(body=[validate], type_ignores=[]), '<validate>', 'exec'), namespace)
        args = SimpleNamespace(sam_encoder_frame_batch_size=4, sam_encoder_lora_rank=8,
            sam_encoder_lora_alpha=16, encoder_lr=5e-7, decoder_lr=2e-7,
            sam_encoder_unfreeze_epoch=6, train_gradient_accumulation_steps=1,
            lr_warmup_epochs=2, epochs=10, lr_cosine_min_ratio=0.1,
            v10_3_metric_workers=2, v10_3_metric_inflight=2, lr_schedule_epochs=5)
        with self.assertRaisesRegex(ValueError, 'lr-schedule-epochs'):
            namespace['_validate'](args)

    def test_short_control_uses_original_schedule_horizon(self):
        args = SimpleNamespace(epochs=10, lr_schedule_epochs=40)
        self.assertEqual(worker_schedule_horizon(args), 40)

    def test_legacy_default_keeps_training_horizon(self):
        self.assertEqual(worker_schedule_horizon(SimpleNamespace(epochs=10)), 10)
        self.assertEqual(worker_schedule_horizon(SimpleNamespace(epochs=10, lr_schedule_epochs=0)), 10)

    def test_short_control_rates_match_first_ten_original_epochs(self):
        class Optimizer:
            def __init__(self):
                self.param_groups = [{'name': 'prompt_generator', 'lr': 1e-6}, {'name': 'sam_image_encoder', 'lr': 5e-7}]
        control, reference = Optimizer(), Optimizer()
        horizon = worker_schedule_horizon(SimpleNamespace(epochs=10, lr_schedule_epochs=40))
        for epoch in range(1, 11):
            actual = tuning.apply_staged_learning_rates(control, epoch=epoch, total_epochs=horizon, warmup_epochs=2, cosine_min_ratio=0.1, encoder_unfreeze_epoch=6)
            expected = tuning.apply_staged_learning_rates(reference, epoch=epoch, total_epochs=40, warmup_epochs=2, cosine_min_ratio=0.1, encoder_unfreeze_epoch=6)
            self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
