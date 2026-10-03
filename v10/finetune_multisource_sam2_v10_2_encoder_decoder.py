"""Eight-source v10.2 with controlled SAM2 encoder and decoder fine-tuning."""

from __future__ import annotations

import argparse
from datetime import datetime

import torch.multiprocessing as mp

import finetune_totalseg_amos_magic_v10_2_joint as v102
from utils.distributed_runtime import find_free_port, parse_gpu_ids


def parser() -> argparse.ArgumentParser:
    result = v102.parser()
    result.description = (
        "v10.2 eight-source fine-tuning with Hiera encoder LoRA/full thaw probes, "
        "full mask-decoder tuning, grouped training decode and pipelined metrics"
    )
    result.add_argument("--v10-3-metric-workers", type=int, default=2)
    result.add_argument("--v10-3-metric-inflight", type=int, default=2)
    result.set_defaults(
        sam_tuning_mode="encoder_lora_full_decoder",
        sam_encoder_lora_rank=8,
        sam_encoder_lora_alpha=16,
        encoder_lr=5e-6,
        decoder_lr=2e-6,
        sam_encoder_frame_batch_size=4,
        v10_3_grouped_decode=True,
        v10_3_metric_pipeline=True,
        v10_3_local_patch=False,
        train_memory_group_size=4,
        eval_memory_group_size=1,
    )
    return result


def _validate(args) -> None:
    schedule_epochs = int(getattr(args, "lr_schedule_epochs", 0))
    if schedule_epochs < 0 or (schedule_epochs and schedule_epochs < int(args.epochs)):
        raise ValueError("--lr-schedule-epochs must be 0 or at least epochs")
    if int(args.sam_encoder_frame_batch_size) < 1:
        raise ValueError("--sam-encoder-frame-batch-size must be positive")
    if int(args.sam_encoder_lora_rank) < 1 or int(args.sam_encoder_lora_alpha) < 1:
        raise ValueError("encoder LoRA rank and alpha must be positive")
    if float(args.encoder_lr) <= 0.0 or float(args.decoder_lr) <= 0.0:
        raise ValueError("encoder and decoder learning rates must be positive")
    if int(args.sam_encoder_unfreeze_epoch) < 1:
        raise ValueError("--sam-encoder-unfreeze-epoch must be positive")
    if int(args.train_gradient_accumulation_steps) < 1:
        raise ValueError("--train-gradient-accumulation-steps must be positive")
    for name in ("decoder_transformer_lr", "decoder_head_lr"):
        value = getattr(args, name, None)
        if value is not None and float(value) <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if int(args.lr_warmup_epochs) < 0 or int(args.lr_warmup_epochs) > int(args.epochs):
        raise ValueError("--lr-warmup-epochs must be within [0, epochs]")
    if not 0.0 < float(args.lr_cosine_min_ratio) <= 1.0:
        raise ValueError("--lr-cosine-min-ratio must be within (0, 1]")
    if int(args.v10_3_metric_workers) < 1 or int(args.v10_3_metric_inflight) < 1:
        raise ValueError("metric workers and inflight counts must be positive")


def _worker(rank: int, world: int, port: int, stamp: str, args) -> None:
    v102._prepare_protocol(args)
    v102._bind_v10_2(args)
    v102.v10.worker(rank, world, port, stamp, args)


def main() -> None:
    args = parser().parse_args()
    v102.v101._validate_extension_args(args)
    _validate(args)
    v102._prepare_protocol(args)
    v102.v101._validate_resume_stage(args)
    v102._validate_resume_is_v10_2(args)
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
