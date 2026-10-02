import sys
from hashlib import sha256
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train
from protocol import APPROVED_SPLIT_SHA256, APPROVED_VALIDATION_SHA256, require_approved_amos_protocol


class FakeDataset:
    shared_case_cache_dir = Path("/cache")

    def _case_image_cache_path(self, image):
        return "/cache/_case_images_v1/case.npy"


class EntryPointTests(unittest.TestCase):
    def test_paired_modes_initialize_prompt_adapter_identically(self):
        from adapter import VistaPromptAdapter
        from train import make_paired_feature_modules

        torch.manual_seed(20260928)
        single_bridge, single_adapter = make_paired_feature_modules("single")
        torch.manual_seed(20260928)
        multiscale_bridge, multiscale_adapter = make_paired_feature_modules("multiscale")
        self.assertNotEqual(type(single_bridge), type(multiscale_bridge))
        self.assertIsInstance(single_adapter, VistaPromptAdapter)
        for name, value in single_adapter.state_dict().items():
            self.assertTrue(torch.equal(value, multiscale_adapter.state_dict()[name]), name)

    def test_help_and_amos_only_defaults(self):
        args = train.parser().parse_args(["audit-data"])
        self.assertEqual(args.vista_hw, 96)
        self.assertEqual(args.depth, 96)
        self.assertEqual(args.stride, 48)
        self.assertIn("ct13_v10_2_split_20260928", str(args.split_json))
        self.assertEqual(args.feature_bridge_mode, "single")
        self.assertEqual(
            train.parser().parse_args(["train", "--feature-bridge-mode", "multiscale"]).feature_bridge_mode,
            "multiscale",
        )

    def test_checkpoint_architecture_isolated_by_feature_bridge_mode(self):
        self.assertEqual(train.architecture_for_feature_mode("single"),
                         "vista3d_promptgen_v102_amos_adapter_v1")
        self.assertEqual(train.architecture_for_feature_mode("multiscale"),
                         "vista3d_promptgen_v102_amos_multiscale_fpn_v1")
        with self.assertRaisesRegex(ValueError, "feature bridge"):
            train.require_matching_feature_mode(
                {"architecture": "vista3d_promptgen_v102_amos_adapter_v1"},
                "multiscale",
            )

    def test_different_same_size_split_is_not_approved(self):
        with self.assertRaisesRegex(ValueError, "split JSON SHA-256"):
            require_approved_amos_protocol(SimpleNamespace(
                split_sha256="different", validation_sha256=APPROVED_VALIDATION_SHA256))
        with self.assertRaisesRegex(ValueError, "validation JSON SHA-256"):
            require_approved_amos_protocol(SimpleNamespace(
                split_sha256=APPROVED_SPLIT_SHA256, validation_sha256="different"))

    def test_validate_requires_trained_vista_adapter(self):
        with self.assertRaisesRegex(ValueError, "--resume"):
            train.main(["validate"])

    def test_frozen_vista_weight_identity_must_match_resume(self):
        with self.assertRaisesRegex(ValueError, "VISTA3D weight SHA-256"):
            train.require_matching_vista_checkpoint(
                {"vista_checkpoint_sha256": "old"}, "new")
        train.require_matching_vista_checkpoint(
            {"vista_checkpoint_sha256": "same"}, "same")

    def test_audit_uses_approved_case_cache_and_fixed_task(self):
        case = SimpleNamespace(case_id="amos_1", image=Path("image.nii.gz"), classes=(118,))
        protocol = SimpleNamespace(train=(case,), val=(case,), tasks=(object(),),
                                   split_sha256="a", validation_sha256="b")
        bundle = {"groups": torch.ones(1)}
        view = SimpleNamespace(vista_image=torch.ones(1, 4, 4),
                               target=torch.ones(1, 1, 4, 4),
                               promptgen_video=torch.ones(1, 3, 4, 4), bundle=bundle)
        with patch.object(train, "read_case_view", return_value=view) as mocked:
            result = train.audit_data(protocol, FakeDataset(), FakeDataset(),
                                      Namespace(vista_hw=96, prompt_work_size=192))
        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(result["fixed_amos_validation_tasks"], 1)
        self.assertEqual(result["cache_reuse"], "existing case CT + v10.2 adaptive scribble metadata")


if __name__ == "__main__":
    unittest.main()
