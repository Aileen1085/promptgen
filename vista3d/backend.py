"""Frozen VISTA3D point-head path with trainable 3D PromptGen adapters.

The public VISTA3D forward accepts clicks/class IDs, not dense embeddings.  This
module joins its frozen encoder and point-head transformer at their tensor
boundary; no GT-derived clicks or SAM2 decoder are used.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from adapter import VistaFeatureBridge, VistaMultiScaleFeatureBridge, VistaPromptAdapter


class VistaPromptGenModel(nn.Module):
    def __init__(self, vista: nn.Module, promptgen: nn.Module,
                 *, feature_bridge: VistaFeatureBridge | VistaMultiScaleFeatureBridge | None = None,
                 prompt_adapter: VistaPromptAdapter | None = None,
                 feature_bridge_mode: str = "single"):
        super().__init__()
        self.vista = vista
        self.vista.requires_grad_(False)
        self.vista.eval()
        self.promptgen = promptgen
        if feature_bridge_mode not in ("single", "multiscale"):
            raise ValueError("feature_bridge_mode must be single or multiscale")
        self.feature_bridge_mode = feature_bridge_mode
        self.feature_bridge = feature_bridge or (
            VistaFeatureBridge() if feature_bridge_mode == "single" else VistaMultiScaleFeatureBridge()
        )
        self.prompt_adapter = prompt_adapter or VistaPromptAdapter()

    def train(self, mode: bool = True):
        super().train(mode)
        self.vista.eval()
        return self

    def _encode_with_multiscale_levels(self, vista_image: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        encoder = self.vista.image_encoder
        layers = getattr(getattr(encoder, "encoder", None), "layers", None)
        if layers is None or len(layers) != 5:
            raise ValueError("VISTA multi-scale bridge requires the official five-stage SegResNet encoder")
        captured: dict[str, torch.Tensor] = {}
        handles = []
        try:
            for name, index in (("fpn0", 2), ("fpn1", 3), ("top", 4)):
                block = layers[index]["blocks"]
                handles.append(block.register_forward_hook(
                    lambda _module, _inputs, output, name=name: captured.__setitem__(name, output)
                ))
            with torch.no_grad():
                output = encoder(vista_image, with_point=True, with_label=False)
        finally:
            for handle in handles:
                handle.remove()
        feature = output[0] if isinstance(output, (tuple, list)) else output
        if set(captured) != {"fpn0", "fpn1", "top"}:
            raise RuntimeError("VISTA encoder did not emit all three multi-scale levels")
        spatial = [captured[name].shape[-3:] for name in ("fpn0", "fpn1", "top")]
        if not all(all(left > right for left, right in zip(spatial[i], spatial[i + 1])) for i in (0, 1)):
            raise ValueError("VISTA encoder levels must have decreasing native 3D resolution")
        return feature, captured

    def forward_patch(self, vista_image: torch.Tensor, promptgen_video: torch.Tensor,
                      bundle: dict, semantic_text: str) -> torch.Tensor:
        if vista_image.ndim != 5 or tuple(vista_image.shape[:2]) != (1, 1):
            raise ValueError("VISTA image must be (1,1,R,A,Z)")
        if promptgen_video.ndim != 4 or promptgen_video.shape[0] != vista_image.shape[-1]:
            raise ValueError("PromptGen video depth must match VISTA Z depth")
        if promptgen_video.shape[1] != 3:
            raise ValueError("PromptGen video must keep its three CT channels")
        if not semantic_text:
            raise ValueError("semantic text must identify the AMOS class")
        if self.feature_bridge_mode == "multiscale":
            feature, levels = self._encode_with_multiscale_levels(vista_image)
        else:
            with torch.no_grad():
                output = self.vista.image_encoder(vista_image, with_point=True, with_label=False)
                feature = output[0] if isinstance(output, (tuple, list)) else output
        if feature.ndim != 5 or feature.shape[0] != 1 or feature.shape[1] != 48:
            raise ValueError("official VISTA point feature must be (1,48,R,A,Z)")
        fpn = self.feature_bridge(
            levels if self.feature_bridge_mode == "multiscale" else feature,
            frame_count=promptgen_video.shape[0],
        )
        video = F.interpolate(promptgen_video.float(),
                              size=(self.promptgen.work_size, self.promptgen.work_size),
                              mode="bilinear", align_corners=False)
        generated = self.promptgen.forward_case_3d_tokens(
            bundle["groups"], video,
            plane_ids=bundle["plane_ids"],
            slice_coords=bundle["slice_coords"],
            frame_coords=bundle["frame_coords"],
            foreground_mode_id=bundle.get("v10_foreground_mode_id"),
            image_features=fpn,
            semantic_text=semantic_text,
        )
        tokens, dense = self.prompt_adapter(
            generated["prompt_3d_tokens"], generated["prompt_3d_role_ids"],
            generated["dense_prompt_embeddings"], tuple(feature.shape[-3:]))
        head = self.vista.point_head
        low = head.feat_downsample(feature + dense.to(dtype=feature.dtype))
        position = head.pe_layer(low.shape[-3:]).unsqueeze(0).to(device=low.device, dtype=low.dtype)
        native_tokens = torch.cat((head.mask_tokens.weight[None],
                                   tokens.to(dtype=low.dtype),
                                   head.supported_embed.weight[None]), dim=1)
        token_output, source = head.transformer(low, position, native_tokens)
        mask_token = head.output_hypernetworks_mlps(token_output[:, :1, :])
        b, c, r, a, z = low.shape
        decoded = head.output_upscaling(source.transpose(1, 2).reshape(b, c, r, a, z))
        b, c, r, a, z = decoded.shape
        logits = (mask_token @ decoded.reshape(b, c, r * a * z)).reshape(b, 1, r, a, z)
        logits = F.interpolate(logits, size=vista_image.shape[-3:], mode="trilinear", align_corners=False)
        return logits.permute(0, 1, 4, 2, 3).contiguous()
