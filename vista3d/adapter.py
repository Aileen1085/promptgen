"""Trainable 3D tensor boundary between unchanged PromptGen and frozen VISTA3D."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class VistaFeatureBridge(nn.Module):
    """Map VISTA `(R,A,Z)` encoder features into PromptGen axial FPN views."""

    def __init__(self, vista_dim: int = 48):
        super().__init__()
        self.projections = nn.ModuleDict({
            name: nn.Conv3d(vista_dim, channels, kernel_size=1)
            for name, channels in (("fpn0", 32), ("fpn1", 64), ("top", 256))
        })

    def forward(self, encoder_feature: torch.Tensor, frame_count: int) -> dict[str, torch.Tensor]:
        if encoder_feature.ndim != 5 or encoder_feature.shape[0] != 1:
            raise ValueError("VISTA encoder feature must be (1,C,R,A,Z)")
        if frame_count < 1:
            raise ValueError("frame_count must be positive")
        axial = encoder_feature.permute(0, 1, 4, 2, 3).contiguous()
        result = {}
        for name, projector in self.projections.items():
            projected = projector(axial)
            projected = F.interpolate(projected, size=(frame_count, *projected.shape[-2:]),
                                      mode="trilinear", align_corners=False)
            result[name] = projected[0].permute(1, 0, 2, 3).contiguous()
        return result


class VistaMultiScaleFeatureBridge(nn.Module):
    """Project three native 1/4, 1/8 and 1/16 VISTA encoder levels separately."""

    LEVELS = (("fpn0", 192, 32), ("fpn1", 384, 64), ("top", 768, 256))

    def __init__(self):
        super().__init__()
        self.projections = nn.ModuleDict({
            name: nn.Conv3d(source_channels, target_channels, kernel_size=1)
            for name, source_channels, target_channels in self.LEVELS
        })

    def forward(self, encoder_levels: dict[str, torch.Tensor], frame_count: int) -> dict[str, torch.Tensor]:
        if frame_count < 1:
            raise ValueError("frame_count must be positive")
        result = {}
        for name, source_channels, _ in self.LEVELS:
            if name not in encoder_levels:
                raise ValueError(f"missing VISTA encoder level {name}")
            feature = encoder_levels[name]
            if feature.ndim != 5 or tuple(feature.shape[:2]) != (1, source_channels):
                raise ValueError(f"VISTA encoder level {name} must be (1,{source_channels},R,A,Z)")
            axial = feature.permute(0, 1, 4, 2, 3).contiguous()
            projected = self.projections[name](axial)
            projected = F.interpolate(
                projected,
                size=(frame_count, *projected.shape[-2:]),
                mode="trilinear",
                align_corners=False,
            )
            result[name] = projected[0].permute(1, 0, 2, 3).contiguous()
        return result


class VistaPromptAdapter(nn.Module):
    """Preserve PromptGen 3D token/dense outputs and adapt them to VISTA's point head."""

    def __init__(self, prompt_dim: int = 256, vista_dim: int = 48):
        super().__init__()
        self.prompt_dim = int(prompt_dim)
        self.vista_dim = int(vista_dim)
        self.role_embedding = nn.Embedding(2, self.prompt_dim)
        nn.init.zeros_(self.role_embedding.weight)
        self.token_projection = nn.Sequential(nn.LayerNorm(self.prompt_dim),
                                              nn.Linear(self.prompt_dim, self.vista_dim))
        self.dense_projection = nn.Conv3d(self.prompt_dim, self.vista_dim, kernel_size=1)
        nn.init.normal_(self.token_projection[-1].weight, std=1e-3)
        nn.init.zeros_(self.token_projection[-1].bias)
        nn.init.normal_(self.dense_projection.weight, std=1e-3)
        nn.init.zeros_(self.dense_projection.bias)

    def forward(self, prompt_3d_tokens: torch.Tensor, role_ids: torch.Tensor,
                dense_prompt_embeddings: torch.Tensor,
                output_shape: tuple[int, int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        if prompt_3d_tokens.ndim != 3 or prompt_3d_tokens.shape[0] != 1 or prompt_3d_tokens.shape[-1] != self.prompt_dim:
            raise ValueError("PromptGen tokens must be (1,K,256)")
        if role_ids.ndim != 1 or role_ids.numel() != prompt_3d_tokens.shape[1] or not torch.all((role_ids == 0) | (role_ids == 1)):
            raise ValueError("one foreground/background role is required per token")
        if dense_prompt_embeddings.ndim != 4 or dense_prompt_embeddings.shape[1] != self.prompt_dim:
            raise ValueError("PromptGen dense prompt must be (Z,256,R,A)")
        if len(output_shape) != 3 or min(output_shape) < 1:
            raise ValueError("VISTA output feature shape must be positive (R,A,Z)")
        roles = role_ids.to(device=prompt_3d_tokens.device, dtype=torch.long)
        tokens = self.token_projection(prompt_3d_tokens + self.role_embedding(roles)[None])
        volume = dense_prompt_embeddings.permute(1, 2, 3, 0)[None].float()
        volume = F.interpolate(volume, size=output_shape, mode="trilinear", align_corners=False)
        residual = self.dense_projection(volume.to(dtype=self.dense_projection.weight.dtype))
        return tokens, residual
