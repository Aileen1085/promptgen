import sys
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend import VistaPromptGenModel
from adapter import VistaMultiScaleFeatureBridge


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 48, 1)

    def forward(self, image, *, with_point, with_label):
        assert with_point and not with_label
        return self.conv(F.avg_pool3d(image, 2)), None


class FakeTransformer(nn.Module):
    def forward(self, source, position, tokens):
        mixed = source + position
        summary = mixed.flatten(2).mean(-1)[:, None, :]
        return tokens + summary, mixed.flatten(2).transpose(1, 2)


class FakePointHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat_downsample = nn.Conv3d(48, 48, 1)
        self.mask_tokens = nn.Embedding(1, 48)
        self.supported_embed = nn.Embedding(1, 48)
        self.transformer = FakeTransformer()
        self.output_upscaling = nn.Conv3d(48, 48, 1)
        self.output_hypernetworks_mlps = nn.Linear(48, 48)

    def pe_layer(self, shape):
        return torch.zeros(48, *shape)


class FakeVista(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = FakeEncoder()
        self.point_head = FakePointHead()


class FakePyramidEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        channels = (1, 48, 96, 192, 384, 768)
        self.encoder.layers = nn.ModuleList([
            nn.ModuleDict({"blocks": nn.Conv3d(channels[i], channels[i + 1], 1)})
            for i in range(5)
        ])
        self.point_projection = nn.Conv3d(768, 48, 1)
        self.calls = 0

    def forward(self, image, *, with_point, with_label):
        assert with_point and not with_label
        self.calls += 1
        x = image
        for index, level in enumerate(self.encoder.layers):
            x = level["blocks"](x)
            if index < 4:
                x = F.avg_pool3d(x, 2)
        point_feature = F.interpolate(self.point_projection(x), size=image.shape[-3:])
        return point_feature, None


class FakeVistaPyramid(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = FakePyramidEncoder()
        self.point_head = FakePointHead()


class FakePromptGen(nn.Module):
    work_size = 8

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward_case_3d_tokens(self, groups, video, **kwargs):
        assert video.shape == (8, 3, 8, 8)
        assert set(kwargs["image_features"]) == {"fpn0", "fpn1", "top"}
        assert kwargs["semantic_text"] == "kidney"
        dense = kwargs["image_features"]["top"] * self.scale
        tokens = dense.mean((0, 2, 3))[None, None]
        return {"prompt_3d_tokens": tokens, "prompt_3d_role_ids": torch.tensor([0]),
                "dense_prompt_embeddings": dense}


class FakePromptGenPyramid(FakePromptGen):
    def forward_case_3d_tokens(self, groups, video, **kwargs):
        assert video.shape == (16, 3, 8, 8)
        features = kwargs["image_features"]
        assert tuple(features[name].shape[-2:] for name in ("fpn0", "fpn1", "top")) == (
            (8, 8), (4, 4), (2, 2)
        )
        dense = features["top"] * self.scale
        tokens = dense.mean((0, 2, 3))[None, None]
        return {"prompt_3d_tokens": tokens, "prompt_3d_role_ids": torch.tensor([0]),
                "dense_prompt_embeddings": dense}


class VistaBackendTests(unittest.TestCase):
    def test_multiscale_uses_one_frozen_encoder_pass_and_real_levels(self):
        model = VistaPromptGenModel(
            FakeVistaPyramid(), FakePromptGenPyramid(),
            feature_bridge=VistaMultiScaleFeatureBridge(),
            feature_bridge_mode="multiscale",
        )
        image = torch.randn(1, 1, 32, 32, 16)
        video = torch.randn(16, 3, 32, 32)
        bundle = {"groups": [], "plane_ids": torch.tensor([]),
                  "slice_coords": torch.tensor([]), "frame_coords": torch.arange(16)}
        logits = model.forward_patch(image, video, bundle, "kidney")
        self.assertEqual(tuple(logits.shape), (1, 1, 16, 32, 32))
        self.assertEqual(model.vista.image_encoder.calls, 1)
        self.assertTrue(torch.isfinite(logits).all())
        logits.square().mean().backward()
        self.assertTrue(all(parameter.grad is None for parameter in model.vista.parameters()))
        self.assertGreater(float(model.promptgen.scale.grad.abs()), 0)
        self.assertTrue(any(parameter.grad is not None for parameter in model.feature_bridge.parameters()))
        for index in (2, 3, 4):
            self.assertEqual(len(model.vista.image_encoder.encoder.layers[index]["blocks"]._forward_hooks), 0)

    def test_frozen_vista_but_promptgen_and_adapters_receive_gradient(self):
        model = VistaPromptGenModel(FakeVista(), FakePromptGen())
        model.train()
        self.assertFalse(model.vista.training)
        image = torch.randn(1, 1, 16, 16, 8)
        video = torch.randn(8, 3, 16, 16)
        bundle = {"groups": [], "plane_ids": torch.tensor([]),
                  "slice_coords": torch.tensor([]), "frame_coords": torch.arange(8)}
        logits = model.forward_patch(image, video, bundle, "kidney")
        self.assertEqual(tuple(logits.shape), (1, 1, 8, 16, 16))
        self.assertTrue(torch.isfinite(logits).all())
        logits.square().mean().backward()
        self.assertTrue(all(p.grad is None for p in model.vista.parameters()))
        self.assertIsNotNone(model.promptgen.scale.grad)
        self.assertGreater(float(model.promptgen.scale.grad.abs()), 0)
        self.assertTrue(any(p.grad is not None for p in model.prompt_adapter.parameters()))

    def test_reject_mismatched_depth(self):
        model = VistaPromptGenModel(FakeVista(), FakePromptGen())
        with self.assertRaisesRegex(ValueError, "depth"):
            model.forward_patch(torch.zeros(1, 1, 16, 16, 8), torch.zeros(7, 3, 16, 16), {}, "kidney")


if __name__ == "__main__":
    unittest.main()
