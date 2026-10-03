import importlib.util
import unittest
from pathlib import Path


class MemoryInputGradientTests(unittest.TestCase):
    def test_helper_exists(self):
        self.assertTrue((Path(__file__).resolve().parents[1] / 'v10_memory_input_grad.py').is_file())

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'real tensor tests require torch')
    def test_trainable_current_feature_gets_exact_gradient_with_dropout(self):
        import torch
        from v10_memory_input_grad import prepare_memory_features

        class Conditioner(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(8, 8)
                self.dropout = torch.nn.Dropout(0.3)
                self.requires_grad_(False)

            def _prepare_memory_conditioned_features(self, current_vision_feats):
                return self.dropout(self.linear(current_vision_feats[0])).tanh()

        sam = Conditioner().train()
        x = torch.randn(4, 8, requires_grad=True)
        torch.manual_seed(37)
        reference = sam._prepare_memory_conditioned_features([x])
        reference.sum().backward()
        expected = x.grad.clone()
        x.grad = None
        torch.manual_seed(37)
        actual = prepare_memory_features(sam, training=True, current_vision_feats=[x])
        actual.sum().backward()
        torch.testing.assert_close(actual, reference)
        torch.testing.assert_close(x.grad, expected)
        self.assertGreater(float(x.grad.abs().sum()), 0.0)
        self.assertTrue(all(p.grad is None for p in sam.parameters()))

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'real tensor tests require torch')
    def test_frozen_and_evaluation_paths_remain_detached(self):
        import torch
        from v10_memory_input_grad import prepare_memory_features

        class Conditioner:
            def _prepare_memory_conditioned_features(self, current_vision_feats):
                return current_vision_feats[0].sin()

        for training, input_grad in ((False, True), (True, False)):
            x = torch.randn(4, requires_grad=input_grad)
            actual = prepare_memory_features(Conditioner(), training=training, current_vision_feats=[x])
            self.assertFalse(actual.requires_grad)
            torch.testing.assert_close(actual, x.sin())


if __name__ == '__main__':
    unittest.main()
