"""Isolated v10.2 phase-1 training for physical background constraints."""

from __future__ import annotations

import argparse
from datetime import datetime

import torch
import torch.multiprocessing as mp

from v10 import finetune_totalseg_amos_magic_v10_2_joint as v102
from v10.v10_2_background_spatial import (
    CrossViewPromptTokenGeneratorV102BackgroundSpatial,
    PhysicalSpacingDataset,
    background_hard_negative_loss,
    configure_phase1_prompt_trainability,
    generate_case_3d_tokens_with_physical_spacing,
    physical_distance_features,
    validate_phase1_args,
)
from utils.distributed_runtime import find_free_port, parse_gpu_ids


def parser() -> argparse.ArgumentParser:
    result = v102.parser()
    result.description = (
        "v10.2 Epoch390 isolated physical foreground/background distance probe"
    )
    result.add_argument(
        "--v10-2-physical-distance-hidden-dim", type=int, default=16
    )
    result.add_argument(
        "--v10-2-physical-distance-clip-mm", type=float, default=80.0
    )
    result.add_argument(
        "--v10-2-background-hard-negative-weight", type=float, default=0.10
    )
    result.add_argument(
        "--v10-2-background-hard-negative-ratio", type=float, default=0.05
    )
    result.add_argument(
        "--v10-2-background-hard-negative-min", type=int, default=256
    )
    result.add_argument(
        "--v10-2-background-hard-negative-max", type=int, default=65536
    )
    result.add_argument(
        "--freeze-prompt-3d-adapter",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    result.add_argument(
        "--freeze-prompt-memory-adapter",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    result.set_defaults(
        sam_tuning_mode="",
        freeze_prompt_3d_adapter=True,
        freeze_prompt_memory_adapter=True,
        dataset_foreground_mode="scribble",
        val_foreground_mode="scribble",
        val_background_mode="scribble",
        anchor_coarse_crop=False,
        patch_scaling=False,
    )
    return result


def make_prompt_background_spatial(args, adapter, device):
    cls = CrossViewPromptTokenGeneratorV102BackgroundSpatial
    cls.DEFAULT_PHYSICAL_DISTANCE_HIDDEN_DIM = int(
        args.v10_2_physical_distance_hidden_dim
    )
    cls.DEFAULT_PHYSICAL_DISTANCE_CLIP_MM = float(
        args.v10_2_physical_distance_clip_mm
    )
    cls.DEFAULT_TRAIN_BACKGROUND_MODES = ("scribble",)
    original = v102.v101.CrossViewPromptTokenGeneratorV101
    v102.v101.CrossViewPromptTokenGeneratorV101 = cls
    try:
        prompt = v102.make_prompt_v10_2(args, adapter, device)
    finally:
        v102.v101.CrossViewPromptTokenGeneratorV101 = original
    prompt._v10_2_args = args
    configure_phase1_prompt_trainability(prompt)
    return prompt


class BackgroundSpatialTrainingStep(v102.v101.v10.DDPTrainingStepV10):
    def __init__(self, prompt, adapter):
        super().__init__(prompt, adapter)
        for parameter in adapter.sam.sam_mask_decoder.prompt_3d_adapter.parameters():
            parameter.requires_grad = False
        for parameter in adapter.sam.sam_mask_decoder.prompt_memory_adapter.parameters():
            parameter.requires_grad = False

    def forward(self, batch, args, device):
        prepared, logits = self.joint(batch, args, device)
        loss = v102.v101.v10.segmentation_loss(
            logits,
            prepared[4],
            args.seg_pos_weight_max,
            args.boundary_loss_weight,
            args.boundary_dice_loss_weight,
            args.boundary_band_width,
            args.boundary_loss_size,
            args.volume_tversky_loss_weight,
            args.volume_tversky_fp_weight,
            args.volume_tversky_fn_weight,
        )
        offset = 0
        hard_negative_rows = []
        for frames, context in zip(prepared[5], prepared[7]):
            end = offset + int(frames)
            if float(args.prompt_consistency_loss_weight) > 0.0:
                loss = loss + float(args.prompt_consistency_loss_weight) * (
                    v102.v101.v10.prompt_consistency_loss(
                        logits[offset:end],
                        context["aligned_prompt"],
                        context["foreground_mode"],
                        context["background_mode"],
                        args.prompt_consistency_threshold,
                        args.prompt_consistency_foreground_weight,
                        args.prompt_consistency_background_weight,
                        args.prompt_consistency_apply_to_box,
                    )
                    / max(1, len(prepared[5]))
                )
            aligned = context["aligned_prompt"]
            features = physical_distance_features(
                aligned,
                spacing_dhw=(1.0, 1.0, 1.0),
                output_hw=tuple(int(value) for value in logits.shape[-2:]),
                distance_clip_mm=max(1.0, float(aligned.shape[-1]) / 4.0),
            ).squeeze(0).permute(1, 0, 2, 3).contiguous()
            hard_negative_rows.append(
                background_hard_negative_loss(
                    logits[offset:end],
                    prepared[4][offset:end],
                    features,
                    hard_negative_ratio=args.v10_2_background_hard_negative_ratio,
                    minimum=args.v10_2_background_hard_negative_min,
                    maximum=args.v10_2_background_hard_negative_max,
                )
            )
            offset = end
        if hard_negative_rows:
            loss = loss + float(
                args.v10_2_background_hard_negative_weight
            ) * torch.stack(hard_negative_rows).mean()
        return loss, sum(prepared[5])


def _bind_background_spatial(args) -> None:
    v102._bind_v10_2(args)
    base_factory = v102.v101.make_dataset_v10_1

    def spatial_factory(*factory_args, **factory_kwargs):
        dataset = base_factory(*factory_args, **factory_kwargs)
        if isinstance(dataset, PhysicalSpacingDataset):
            return dataset
        return PhysicalSpacingDataset(dataset)

    v102.v101.make_dataset_v10_1 = spatial_factory
    base_generate = v102.v10_memory.generate_case_3d_tokens

    def spatial_generate(prompt_gen, adapter, video_cpu, bundle_cpu, class_id, bound_args, device):
        return generate_case_3d_tokens_with_physical_spacing(
            base_generate,
            prompt_gen,
            adapter,
            video_cpu,
            bundle_cpu,
            class_id,
            bound_args,
            device,
        )

    v102.v10_memory.generate_case_3d_tokens = spatial_generate
    v102.v10.make_prompt = make_prompt_background_spatial
    v102.v10.load_init = v102.load_init_v10_2
    v102.v10.DDPTrainingStepV10 = BackgroundSpatialTrainingStep


def _worker(rank: int, world: int, port: int, stamp: str, args) -> None:
    v102._prepare_protocol(args)
    _bind_background_spatial(args)
    v102.v10.worker(rank, world, port, stamp, args)


def main() -> None:
    args = parser().parse_args()
    v102.v101._validate_extension_args(args)
    validate_phase1_args(args)
    v102._prepare_protocol(args)
    args.amp = True
    args.object_score_gate = False
    ids = parse_gpu_ids(args.gpu)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if len(ids) > 1:
        mp.spawn(
            _worker,
            nprocs=len(ids),
            args=(len(ids), find_free_port(), stamp, args),
            join=True,
        )
    else:
        _worker(0, 1, find_free_port(), stamp, args)


if __name__ == "__main__":
    main()
