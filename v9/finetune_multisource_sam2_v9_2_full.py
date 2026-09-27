"""Opt-in SAM2 v9.2 eight-source encoder+decoder fine-tuning entry.

Use the same dataset, prompt, sampling and validation callbacks as the v9.2
precision/DDP run. This module only changes trainability, gradient transport,
optimizer groups and checkpoint contents. It never resumes the old run in place.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys

import torch
from torch.utils.checkpoint import checkpoint

import finetune_multisource_sam2_v9_2_ddp as ddp
from v9_2_full_finetune_policy import select_encoder_parameter_names


base = ddp.base
v9_joint = ddp.legacy.multi92.v92.v91.v9
v92_joint = ddp.legacy.multi92.v92
_BASE_PARSE = base.parse_args
_BASE_FREEZE = base.freeze_module
_BASE_GROUPS = base.build_optimizer_groups
_BASE_PROMPT_LOAD = base.load_prompt_generator_checkpoint
_BASE_RESUME_LOAD = v92_joint._BASE_LOAD_RESUME
_BASE_EVALUATE = base.evaluate
_ADAPTER = None
_OPTIONS = None


def _parse_full_args():
    global _OPTIONS
    extension = argparse.ArgumentParser(add_help=False)
    extension.add_argument("--full-ft-encoder-scope", choices=("late", "all"), default="late")
    extension.add_argument("--full-ft-encoder-lr", type=float, default=2e-7)
    extension.add_argument("--full-ft-encoder-early-lr", type=float, default=5e-8)
    extension.add_argument("--full-ft-activation-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    options, remaining = extension.parse_known_args(sys.argv[1:])
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *remaining]
        args = _BASE_PARSE()
    finally:
        sys.argv = old_argv
    if bool(args.freeze_decoder):
        raise ValueError("Full fine-tuning requires --no-freeze-decoder")
    if not bool(args.load_decoder_from_prompt_checkpoint) and not args.resume_checkpoint:
        raise ValueError("Warm start must include --load-decoder-from-prompt-checkpoint")
    if not args.resume_checkpoint and not args.prompt_generator_checkpoint:
        raise ValueError("Specify a v9.2 --prompt-generator-checkpoint or a full-fine-tune --resume-checkpoint")
    if options.full_ft_encoder_lr <= 0 or options.full_ft_encoder_early_lr <= 0:
        raise ValueError("Encoder learning rates must be positive")
    if options.full_ft_encoder_early_lr > options.full_ft_encoder_lr:
        raise ValueError("Early Hiera layers must not have a higher LR than late layers")
    if args.resume_checkpoint and not args.resume_run_dir:
        raise ValueError("Resume requires --resume-run-dir to keep the checkpoint series together")
    args.full_finetune_image_encoder = True
    args.full_ft_encoder_scope = options.full_ft_encoder_scope
    args.full_ft_encoder_lr = options.full_ft_encoder_lr
    args.full_ft_encoder_early_lr = options.full_ft_encoder_early_lr
    args.full_ft_activation_checkpoint = options.full_ft_activation_checkpoint
    _OPTIONS = options
    return args


def _encoder_selection(encoder):
    names = [name for name, _parameter in encoder.named_parameters()]
    stage_ends = getattr(getattr(encoder, "trunk", None), "stage_ends", ())
    return select_encoder_parameter_names(names, stage_ends, _OPTIONS.full_ft_encoder_scope)


def _freeze_with_encoder_scope(module):
    global _ADAPTER
    _BASE_FREEZE(module)
    if not isinstance(module, base.SAM2MedicalAdapter):
        return
    _ADAPTER = module
    encoder = module.sam.image_encoder
    selected = _encoder_selection(encoder)
    for name, parameter in encoder.named_parameters():
        parameter.requires_grad_(name in selected)
    encoder.train()
    if _OPTIONS.full_ft_activation_checkpoint:
        original_forward = encoder.forward

        def checkpointed_forward(sample):
            if torch.is_grad_enabled():
                return checkpoint(original_forward, sample, use_reentrant=False)
            return original_forward(sample)

        encoder.forward = checkpointed_forward
    logging.info(
        "SAM2 full FT encoder scope=%s trainable=%d/%d activation_checkpoint=%s",
        _OPTIONS.full_ft_encoder_scope,
        sum(p.numel() for p in encoder.parameters() if p.requires_grad),
        sum(p.numel() for p in encoder.parameters()),
        _OPTIONS.full_ft_activation_checkpoint,
    )


def _encode_image_features_with_grad(self, video):
    # The legacy adapter method is decorated with no_grad and detaches FPN.
    # Evaluation already calls this method under a no_grad context.
    return self._compute_image_features(video)


def _groups_with_encoder(prompt_gen, decoder, args):
    groups = _BASE_GROUPS(prompt_gen, decoder, args)
    if _ADAPTER is None:
        raise RuntimeError("SAM2 adapter must be initialized before optimizer construction")
    encoder = _ADAPTER.sam.image_encoder
    late_names = select_encoder_parameter_names(
        [name for name, _parameter in encoder.named_parameters()],
        encoder.trunk.stage_ends,
        "late",
    )
    late = []
    early = []
    for name, parameter in encoder.named_parameters():
        if not parameter.requires_grad:
            continue
        (late if name in late_names else early).append(parameter)
    if not late:
        raise RuntimeError("No trainable Hiera late-stage/neck parameters")
    groups.append({"params": late, "lr": float(args.full_ft_encoder_lr), "name": "sam2_encoder_late"})
    if early:
        groups.append({"params": early, "lr": float(args.full_ft_encoder_early_lr), "name": "sam2_encoder_early"})
    identifiers = [id(parameter) for group in groups for parameter in group["params"]]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("A parameter appears in multiple full-fine-tune optimizer groups")
    return groups


class FullJointForward(ddp.DistributedJointForward):
    def __init__(self, prompt_gen, adapter):
        super().__init__(prompt_gen, adapter)
        # The base DDP wrapper keeps adapter only as a weakref. Explicitly
        # register the trainable encoder so DDP synchronizes its gradients.
        self.sam2_image_encoder = adapter.sam.image_encoder


def _save_full_checkpoint(path, prompt_gen, adapter, optimizer, scheduler, epoch,
                          val_dice, class_weights, args, best_val=None, bad_validations=0):
    payload = {
        "epoch": int(epoch),
        "val_dice": None if val_dice is None else float(val_dice),
        "best_val": None if best_val is None else float(best_val),
        "bad_validations": int(bad_validations),
        "prompt_cache_version": base.SAM2CachedAMOSDataset.PROMPT_CACHE_VERSION,
        "prompt_generator_arch": base.PROMPT_GENERATOR_ARCH,
        "joint_finetune_arch": base.JOINT_FINETUNE_ARCH,
        "prompt_channel_names": base.SAM2CachedAMOSDataset.PROMPT_CHANNEL_NAMES,
        "prompt_generator": prompt_gen.state_dict(),
        "sam2_mask_decoder": adapter.sam.sam_mask_decoder.state_dict(),
        "sam2_image_encoder": adapter.sam.image_encoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "class_weights": class_weights,
        "args": vars(args),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _resume_full_checkpoint(path, prompt_gen, adapter, optimizer, scheduler, freeze_decoder=False):
    state = torch.load(path, map_location="cpu")
    if "sam2_image_encoder" not in state:
        raise RuntimeError("Refusing full-fine-tune resume: checkpoint has no SAM2 image encoder")
    result = _BASE_RESUME_LOAD(
        path, prompt_gen, adapter, optimizer, scheduler, freeze_decoder=freeze_decoder
    )
    adapter.sam.image_encoder.load_state_dict(state["sam2_image_encoder"], strict=True)
    return result


def _load_full_warm_start(path, prompt_gen, **kwargs):
    result = _BASE_PROMPT_LOAD(path, prompt_gen, **kwargs)
    state = result[0]
    if "sam2_mask_decoder" not in state:
        raise RuntimeError(
            "Full fine-tuning requires a joint v9.2 warm-start checkpoint "
            "containing sam2_mask_decoder"
        )
    if "sam2_image_encoder" in state:
        if _ADAPTER is None:
            raise RuntimeError("SAM2 adapter was not initialized before warm start")
        _ADAPTER.sam.image_encoder.load_state_dict(state["sam2_image_encoder"], strict=True)
        logging.info("Warm-started SAM2 image encoder from %s", path)
    else:
        logging.info("Warm start has no encoder state; retaining original SAM2 Hiera-L encoder")
    return result


def _evaluate_with_encoder_mode(prompt_gen, adapter, *args, **kwargs):
    encoder = adapter.sam.image_encoder
    was_training = encoder.training
    encoder.eval()
    try:
        return _BASE_EVALUATE(prompt_gen, adapter, *args, **kwargs)
    finally:
        encoder.train(was_training)


def _install():
    base.parse_args = _parse_full_args
    base.freeze_module = _freeze_with_encoder_scope
    base.build_optimizer_groups = _groups_with_encoder
    base.JointTrainingForward = FullJointForward
    base.load_prompt_generator_checkpoint = _load_full_warm_start
    base.evaluate = _evaluate_with_encoder_mode
    v9_joint._original_save_ckpt = _save_full_checkpoint
    v92_joint._BASE_LOAD_RESUME = _resume_full_checkpoint
    base.SAM2MedicalAdapter.encode_image_features = _encode_image_features_with_grad


_install()


def main():
    ddp.main()


if __name__ == "__main__":
    main()
