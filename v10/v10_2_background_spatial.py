"""Physical prompt-distance residual for the isolated v10.2 phase-1 probe."""

from __future__ import annotations

import math
import importlib
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from torch import nn


def rescaled_spacing_after_square_crop(
    base_spacing: Sequence[float],
    full_side: int,
    bounds: Sequence[int],
    output_side: int,
) -> tuple[float, float, float]:
    """Return DHW millimetres per voxel after one square crop and resize."""

    if len(base_spacing) != 3 or any(
        not math.isfinite(float(value)) or float(value) <= 0.0
        for value in base_spacing
    ):
        raise ValueError("base spacing must contain three positive finite values")
    full_side = int(full_side)
    output_side = int(output_side)
    if full_side <= 0 or output_side <= 0 or len(bounds) != 4:
        raise ValueError("full and output sides must be positive")
    left, right, top, bottom = (int(value) for value in bounds)
    if not (
        0 <= left < right <= full_side
        and 0 <= top < bottom <= full_side
    ):
        raise ValueError("crop bounds must stay inside the full square")
    crop_width = right - left
    crop_height = bottom - top
    if crop_width != crop_height:
        raise ValueError("physical prompt crop must be square")
    scale = float(crop_width) / float(output_side)
    return (
        float(base_spacing[0]),
        float(base_spacing[1]) * scale,
        float(base_spacing[2]) * scale,
    )


def prompt_work_spacing_dhw(
    model_spacing_dhw: Sequence[float],
    model_hw: Sequence[int],
    work_size: int,
) -> tuple[float, float, float]:
    if len(model_spacing_dhw) != 3 or len(model_hw) != 2:
        raise ValueError("model spacing and shape must be DHW and HW")
    work_size = int(work_size)
    height, width = (int(value) for value in model_hw)
    if min(work_size, height, width) <= 0:
        raise ValueError("model and work sizes must be positive")
    return (
        float(model_spacing_dhw[0]),
        float(model_spacing_dhw[1]) * float(height) / float(work_size),
        float(model_spacing_dhw[2]) * float(width) / float(work_size),
    )


def _inner_dataset(dataset):
    current = dataset
    seen = set()
    while hasattr(current, "dataset") and id(current) not in seen:
        seen.add(id(current))
        nested = getattr(current, "dataset")
        if nested is None or nested is current:
            break
        current = nested
    return current


class PhysicalSpacingDataset:
    """Attach post-transform prompt spacing without changing shared datasets."""

    def __init__(self, dataset) -> None:
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def _attach(self, index: int, item):
        video, bundle, points, target, class_id = item
        inner = _inner_dataset(self.dataset)
        base_spacing = inner.effective_model_spacing_dhw(int(index))
        model_full_side = int(inner.model_input_size)
        reference_side = int(
            getattr(inner, "last_padded_square", 0) or model_full_side
        )
        bounds = getattr(inner, "last_patch_bounds", None)
        if bounds is None or not np.asarray(bounds).size:
            bounds = (0, reference_side, 0, reference_side)
        left, right, top, bottom = (int(value) for value in bounds)
        crop_width = right - left
        crop_height = bottom - top
        output_side = int(video.shape[-1])
        if (
            crop_width <= 0
            or crop_width != crop_height
            or tuple(video.shape[-2:]) != (output_side, output_side)
        ):
            raise ValueError("physical spacing requires positive square crop/output")
        scale = (
            float(model_full_side)
            * float(crop_width)
            / (float(reference_side) * float(output_side))
        )
        updated = dict(bundle)
        updated["physical_spacing_dhw"] = torch.tensor(
            (
                float(base_spacing[0]),
                float(base_spacing[1]) * scale,
                float(base_spacing[2]) * scale,
            ),
            dtype=torch.float32,
        )
        return video, updated, points, target, class_id

    def __getitem__(self, index):
        return self._attach(index, self.dataset[index])

    def get_item_for_class(self, index, class_id):
        return self._attach(
            index, self.dataset.get_item_for_class(index, class_id)
        )


def bind_physical_spacing_source_factory(owner_module):
    """Wrap every v10.2 source dataset, including non-TotalSeg sources."""

    owner_factory = owner_module.make_source_dataset
    if bool(getattr(owner_factory, "_v10_2_physical_spacing_bound", False)):
        return owner_factory
    data_module = importlib.import_module(owner_factory.__module__)
    original_factory = data_module.make_source_dataset

    def spatial_source_factory(*factory_args, **factory_kwargs):
        dataset = original_factory(*factory_args, **factory_kwargs)
        if isinstance(dataset, PhysicalSpacingDataset):
            return dataset
        return PhysicalSpacingDataset(dataset)

    spatial_source_factory._v10_2_physical_spacing_bound = True
    spatial_source_factory._v10_2_physical_spacing_original = original_factory
    data_module.make_source_dataset = spatial_source_factory
    owner_module.make_source_dataset = spatial_source_factory
    return spatial_source_factory


def generate_case_3d_tokens_with_physical_spacing(
    original_generate,
    prompt_gen,
    adapter,
    video_cpu,
    bundle_cpu,
    class_id,
    args,
    device,
):
    spacing = bundle_cpu.get("physical_spacing_dhw")
    if spacing is None:
        raise RuntimeError("physical-distance PromptGen requires physical spacing")
    if torch.is_tensor(spacing):
        spacing = tuple(float(value) for value in spacing.flatten().tolist())
    work_spacing = prompt_work_spacing_dhw(
        spacing, video_cpu.shape[-2:], prompt_gen.work_size
    )
    sentinel = object()
    previous = getattr(prompt_gen, "_physical_spacing_dhw", sentinel)
    prompt_gen._physical_spacing_dhw = work_spacing
    try:
        return original_generate(
            prompt_gen,
            adapter,
            video_cpu,
            bundle_cpu,
            class_id,
            args,
            device,
        )
    finally:
        if previous is sentinel:
            delattr(prompt_gen, "_physical_spacing_dhw")
        else:
            prompt_gen._physical_spacing_dhw = previous


def _distance_mm(mask: np.ndarray, spacing_dhw: Sequence[float]) -> np.ndarray:
    return distance_transform_edt(~mask.astype(bool), sampling=spacing_dhw).astype(
        np.float32, copy=False
    )


def physical_distance_features(
    aligned_prompt: torch.Tensor,
    *,
    spacing_dhw: Sequence[float],
    output_hw: tuple[int, int],
    distance_clip_mm: float,
) -> torch.Tensor:
    """Build bounded, prompt-only physical-distance features.

    The input is the aligned ``(D,C,H,W)`` prompt volume. Channel 0 is the
    active foreground prompt and channel 1 is the active background prompt.
    The result is detached by construction and contains no target information.
    """

    if aligned_prompt.ndim != 4 or int(aligned_prompt.shape[1]) < 2:
        raise ValueError("aligned_prompt must have shape (D,C,H,W) with C >= 2")
    if len(spacing_dhw) != 3 or any(
        not math.isfinite(float(value)) or float(value) <= 0.0
        for value in spacing_dhw
    ):
        raise ValueError("spacing_dhw must contain three positive finite values")
    clip = float(distance_clip_mm)
    if not math.isfinite(clip) or clip <= 0.0:
        raise ValueError("distance_clip_mm must be positive and finite")
    output_hw = tuple(int(value) for value in output_hw)
    if len(output_hw) != 2 or min(output_hw) <= 0:
        raise ValueError("output_hw must contain two positive values")

    source_hw = tuple(int(value) for value in aligned_prompt.shape[-2:])
    prompt = aligned_prompt.detach().float()
    prompt = F.adaptive_max_pool2d(prompt, output_hw)
    foreground = (prompt[:, 0] > 0).cpu().numpy()
    background = (prompt[:, 1] > 0).cpu().numpy()
    if not bool(foreground.any()):
        raise ValueError("foreground prompt is empty")

    feature_spacing = (
        float(spacing_dhw[0]),
        float(spacing_dhw[1]) * float(source_hw[0]) / float(output_hw[0]),
        float(spacing_dhw[2]) * float(source_hw[1]) / float(output_hw[1]),
    )
    foreground_distance = _distance_mm(foreground, feature_spacing)
    foreground_proximity = np.exp(
        -np.minimum(foreground_distance, clip) / clip
    ).astype(np.float32, copy=False)
    if bool(background.any()):
        background_distance = _distance_mm(background, feature_spacing)
        background_proximity = np.exp(
            -np.minimum(background_distance, clip) / clip
        ).astype(np.float32, copy=False)
        signed_contrast = np.tanh(
            (background_distance - foreground_distance) / clip
        ).astype(np.float32, copy=False)
        background_presence = np.ones_like(foreground_proximity, dtype=np.float32)
    else:
        background_proximity = np.zeros_like(foreground_proximity, dtype=np.float32)
        signed_contrast = np.zeros_like(foreground_proximity, dtype=np.float32)
        background_presence = np.zeros_like(foreground_proximity, dtype=np.float32)

    features = np.stack(
        (
            foreground_proximity,
            background_proximity,
            signed_contrast,
            background_presence,
        ),
        axis=0,
    )
    return torch.from_numpy(features).unsqueeze(0).to(
        device=aligned_prompt.device, dtype=aligned_prompt.dtype
    )


class PhysicalDistanceDenseResidual(nn.Module):
    """Zero-initialized residual from physical prompt distances to dense SAM prompts."""

    def __init__(
        self,
        *,
        prompt_dim: int,
        hidden_dim: int = 16,
        distance_clip_mm: float = 80.0,
    ) -> None:
        super().__init__()
        prompt_dim = int(prompt_dim)
        hidden_dim = int(hidden_dim)
        if prompt_dim <= 0 or hidden_dim <= 0:
            raise ValueError("prompt and hidden dimensions must be positive")
        groups = min(4, hidden_dim)
        while hidden_dim % groups:
            groups -= 1
        self.distance_clip_mm = float(distance_clip_mm)
        self.features = nn.Sequential(
            nn.Conv3d(4, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
        )
        self.output = nn.Conv3d(hidden_dim, prompt_dim, kernel_size=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, distance_features: torch.Tensor) -> torch.Tensor:
        if distance_features.ndim != 5 or int(distance_features.shape[1]) != 4:
            raise ValueError("distance features must have shape (B,4,D,H,W)")
        return self.output(self.features(distance_features))


def apply_physical_distance_residual(
    dense_prompt_embeddings: torch.Tensor,
    aligned_prompt: torch.Tensor,
    branch: PhysicalDistanceDenseResidual,
    *,
    spacing_dhw: Sequence[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add the soft spatial residual without masking the legacy dense prompt."""

    if dense_prompt_embeddings.ndim != 4:
        raise ValueError("dense prompt embeddings must have shape (D,C,H,W)")
    features = physical_distance_features(
        aligned_prompt,
        spacing_dhw=spacing_dhw,
        output_hw=tuple(int(value) for value in dense_prompt_embeddings.shape[-2:]),
        distance_clip_mm=branch.distance_clip_mm,
    )
    residual = branch(features).squeeze(0).permute(1, 0, 2, 3).contiguous()
    residual = residual.to(
        device=dense_prompt_embeddings.device,
        dtype=dense_prompt_embeddings.dtype,
    )
    return dense_prompt_embeddings + residual, features


def background_hard_negative_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    physical_features: torch.Tensor,
    *,
    hard_negative_ratio: float = 0.05,
    minimum: int = 1,
    maximum: int = 65536,
) -> torch.Tensor:
    """Penalize high-probability GT negatives, softly weighted by BG distance.

    ``target`` is intentionally confined to this training-only loss. Physical
    features are prompt-derived and are the only spatial signal used at
    inference time.
    """

    if logits.shape != target.shape or logits.ndim != 4:
        raise ValueError("logits and target must share shape (D,1,H,W)")
    if physical_features.ndim != 4 or int(physical_features.shape[1]) != 4:
        raise ValueError("physical features must have shape (D,4,H,W)")
    if int(physical_features.shape[0]) != int(logits.shape[0]):
        raise ValueError("physical features must preserve logit depth")
    ratio = float(hard_negative_ratio)
    if not 0.0 < ratio <= 1.0:
        raise ValueError("hard_negative_ratio must be within (0, 1]")
    minimum = int(minimum)
    maximum = int(maximum)
    if minimum < 1 or maximum < minimum:
        raise ValueError("hard-negative count bounds are invalid")
    if float(physical_features[:, 3].detach().amax()) <= 0.0:
        return logits.sum() * 0.0

    proximity = F.interpolate(
        physical_features[:, 1:2].float(),
        size=tuple(int(value) for value in logits.shape[-2:]),
        mode="bilinear",
        align_corners=False,
    ).to(device=logits.device, dtype=logits.dtype).clamp(0.0, 1.0)
    negative = target <= 0.5
    count = int(negative.sum().item())
    if count == 0:
        return logits.sum() * 0.0
    selected_count = min(maximum, max(minimum, int(math.ceil(count * ratio))))
    selected_count = min(selected_count, count)
    probability = torch.sigmoid(logits.detach())
    scores = probability.masked_fill(~negative, -1.0).flatten()
    selected = torch.topk(scores, selected_count, sorted=False).indices
    losses = F.binary_cross_entropy_with_logits(
        logits, torch.zeros_like(logits), reduction="none"
    ).flatten()[selected]
    weights = (0.25 + 0.75 * proximity).flatten()[selected]
    return (losses * weights).mean()


from v10.model.cross_view_prompt_token_generator_v10_1 import (  # noqa: E402
    CrossViewPromptTokenGeneratorV101,
)


class CrossViewPromptTokenGeneratorV102BackgroundSpatial(
    CrossViewPromptTokenGeneratorV101
):
    """v10.2 prompt generator with one additive physical-distance branch."""

    DEFAULT_PHYSICAL_DISTANCE_HIDDEN_DIM = 16
    DEFAULT_PHYSICAL_DISTANCE_CLIP_MM = 80.0

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.physical_distance_residual_enabled = True
        self.physical_distance_residual = PhysicalDistanceDenseResidual(
            prompt_dim=self.prompt_dim,
            hidden_dim=int(self.DEFAULT_PHYSICAL_DISTANCE_HIDDEN_DIM),
            distance_clip_mm=float(self.DEFAULT_PHYSICAL_DISTANCE_CLIP_MM),
        )

    def forward_case_3d_tokens(
        self,
        *args,
        physical_spacing_dhw: torch.Tensor | Sequence[float] | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        if physical_spacing_dhw is None:
            physical_spacing_dhw = getattr(
                self, "_physical_spacing_dhw", None
            )
        if physical_spacing_dhw is None:
            raise ValueError("physical_spacing_dhw is required")
        if torch.is_tensor(physical_spacing_dhw):
            spacing = tuple(
                float(value)
                for value in physical_spacing_dhw.detach().cpu().flatten().tolist()
            )
        else:
            spacing = tuple(float(value) for value in physical_spacing_dhw)
        outputs = super().forward_case_3d_tokens(*args, **kwargs)
        dense, features = apply_physical_distance_residual(
            outputs["dense_prompt_embeddings"],
            outputs["aligned_prompt"],
            self.physical_distance_residual,
            spacing_dhw=spacing,
        )
        outputs["dense_prompt_embeddings"] = dense
        outputs["physical_distance_features"] = features.squeeze(0).permute(
            1, 0, 2, 3
        ).contiguous()
        return outputs


def validate_phase1_args(args) -> None:
    if str(args.resume_checkpoint or ""):
        raise ValueError(
            "phase-1 must start a new isolated run via --prompt-generator-checkpoint"
        )
    if str(args.sam_tuning_mode or ""):
        raise ValueError("phase-1 keeps every SAM2 component frozen")
    if not bool(args.freeze_prompt_3d_adapter):
        raise ValueError("phase-1 requires --freeze-prompt-3d-adapter")
    if not bool(args.freeze_prompt_memory_adapter):
        raise ValueError("phase-1 requires --freeze-prompt-memory-adapter")
    if int(args.v10_2_physical_distance_hidden_dim) < 1:
        raise ValueError("physical-distance hidden dimension must be positive")
    if float(args.v10_2_physical_distance_clip_mm) <= 0.0:
        raise ValueError("physical-distance clip must be positive")
    if float(args.v10_2_background_hard_negative_weight) < 0.0:
        raise ValueError("hard-negative weight must be non-negative")
    if not 0.0 < float(args.v10_2_background_hard_negative_ratio) <= 1.0:
        raise ValueError("hard-negative ratio must be within (0, 1]")
    if int(args.v10_2_background_hard_negative_min) < 1:
        raise ValueError("hard-negative minimum must be positive")
    if int(args.v10_2_background_hard_negative_max) < int(
        args.v10_2_background_hard_negative_min
    ):
        raise ValueError("hard-negative maximum must be >= minimum")


def configure_phase1_prompt_trainability(prompt) -> None:
    for parameter in prompt.parameters():
        parameter.requires_grad = False
    branch = getattr(prompt, "physical_distance_residual", None)
    if branch is None:
        raise RuntimeError("phase-1 prompt lacks physical_distance_residual")
    for parameter in branch.parameters():
        parameter.requires_grad = True
