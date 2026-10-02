import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapter import VistaFeatureBridge, VistaMultiScaleFeatureBridge, VistaPromptAdapter


class VistaAdapterTests(unittest.TestCase):
    def test_vista_feature_axes_and_fpn_channels(self):
        bridge = VistaFeatureBridge()
        source = torch.zeros(1, 48, 4, 5, 2, requires_grad=True)  # R,A,Z
        source.data[0, 0, 1, 3, 0] = 1
        maps = bridge(source, frame_count=8)
        self.assertEqual(tuple(maps["fpn0"].shape), (8, 32, 4, 5))
        self.assertEqual(tuple(maps["fpn1"].shape), (8, 64, 4, 5))
        self.assertEqual(tuple(maps["top"].shape), (8, 256, 4, 5))
        maps["top"].sum().backward()
        self.assertIsNotNone(source.grad)

    def test_multiscale_fpn_uses_three_distinct_encoder_levels(self):
        bridge = VistaMultiScaleFeatureBridge()
        levels = {
            "fpn0": torch.ones(1, 192, 8, 8, 8),
            "fpn1": torch.ones(1, 384, 4, 4, 4) * 2,
            "top": torch.ones(1, 768, 2, 2, 2) * 3,
        }
        output = bridge(levels, frame_count=16)
        self.assertEqual(tuple(output["fpn0"].shape), (16, 32, 8, 8))
        self.assertEqual(tuple(output["fpn1"].shape), (16, 64, 4, 4))
        self.assertEqual(tuple(output["top"].shape), (16, 256, 2, 2))
        for name in levels:
            changed = dict(levels)
            changed[name] = torch.zeros_like(levels[name])
            other = bridge(changed, frame_count=16)
            self.assertFalse(torch.equal(output[name], other[name]))
            for unaffected in levels.keys() - {name}:
                self.assertTrue(torch.equal(output[unaffected], other[unaffected]))

    def test_multiscale_fpn_rejects_missing_or_wrong_encoder_level(self):
        bridge = VistaMultiScaleFeatureBridge()
        levels = {
            "fpn0": torch.ones(1, 192, 8, 8, 8),
            "fpn1": torch.ones(1, 384, 4, 4, 4),
            "top": torch.ones(1, 768, 2, 2, 2),
        }
        with self.assertRaisesRegex(ValueError, "top"):
            bridge({key: value for key, value in levels.items() if key != "top"}, 16)
        with self.assertRaisesRegex(ValueError, "fpn1"):
            bridge({**levels, "fpn1": torch.ones(1, 192, 4, 4, 4)}, 16)

    def test_case_tokens_and_dense_volume_have_vista_shapes(self):
        adapter = VistaPromptAdapter()
        tokens = torch.randn(1, 3, 256, requires_grad=True)
        dense = torch.randn(8, 256, 16, 20, requires_grad=True)
        roles = torch.tensor([0, 1, 0])
        adapted, residual = adapter(tokens, roles, dense, output_shape=(4, 5, 2))
        self.assertEqual(tuple(adapted.shape), (1, 3, 48))
        self.assertEqual(tuple(residual.shape), (1, 48, 4, 5, 2))
        (adapted.sum() + residual.sum()).backward()
        self.assertGreater(float(tokens.grad.abs().sum()), 0)
        self.assertGreater(float(dense.grad.abs().sum()), 0)

    def test_invalid_token_role_and_frame_shapes_fail(self):
        adapter = VistaPromptAdapter()
        with self.assertRaisesRegex(ValueError, "role"):
            adapter(torch.zeros(1, 2, 256), torch.tensor([0]), torch.zeros(8, 256, 4, 4), (4, 4, 2))
        with self.assertRaisesRegex(ValueError, "dense"):
            adapter(torch.zeros(1, 2, 256), torch.tensor([0, 1]), torch.zeros(8, 128, 4, 4), (4, 4, 2))


if __name__ == "__main__":
    unittest.main()
