import unittest
from types import SimpleNamespace
from v9_2_frozen_hard_sources import assert_frozen_sam2, focus_ratios


class Parameter:
    def __init__(self, trainable):
        self.requires_grad = trainable


class Adapter:
    def __init__(self, encoder=False, core=False):
        self.parameters = [('sam.image_encoder.weight', Parameter(encoder)),
                           ('sam.sam_mask_decoder.core_decoder.weight', Parameter(core)),
                           ('sam.sam_mask_decoder.prompt_3d_adapter.weight', Parameter(True))]

    def named_parameters(self):
        return iter(self.parameters)


class FrozenTests(unittest.TestCase):
    def test_encoder_unfreeze_is_rejected(self):
        with self.assertRaises(RuntimeError):
            assert_frozen_sam2(Adapter(encoder=True))

    def test_decoder_unfreeze_is_rejected(self):
        with self.assertRaises(RuntimeError):
            assert_frozen_sam2(Adapter(core=True))

    def test_only_added_adapter_is_allowed(self):
        adapter = Adapter()
        assert_frozen_sam2(adapter)
        with self.assertRaises(RuntimeError):
            assert_frozen_sam2(adapter, [{'params': [adapter.parameters[0][1]]}])

    def test_quotas_preserve_200_and_replay_all_sources(self):
        ratios = focus_ratios()
        self.assertAlmostEqual(sum(ratios.values()), 1)
        self.assertEqual(len(ratios), 13)
        self.assertTrue(all(v > 0 for v in ratios.values()))
        self.assertEqual(round(ratios['parse2022'] * 200), 20)
        self.assertEqual(round(ratios['covid19_20'] * 200), 24)


if __name__ == '__main__':
    unittest.main()
