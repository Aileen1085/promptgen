"""AMOS CT-only VISTA3D / existing 3D PromptGen experiment.

No code path here mutates or resumes the SAM2 v10.2 training run.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import random

import numpy as np
import torch

from backend import VistaPromptGenModel
from adapter import VistaFeatureBridge, VistaMultiScaleFeatureBridge, VistaPromptAdapter
from official_loader import load_official_vista3d
from factory import load_v102_promptgen
from metrics import binary_metrics
from protocol import load_amos_protocol, require_approved_amos_protocol
from training import (
    make_amos_dataset,
    pack_window,
    predict_complete_roi,
    read_case_view,
    segmentation_loss_compatible,
)
from data_bridge import sliding_windows
from training_control import ValidationController


ROOT = Path(__file__).resolve().parent.parent


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_matching_vista_checkpoint(state: dict, actual_sha256: str) -> None:
    if state.get("vista_checkpoint_sha256") != actual_sha256:
        raise ValueError("frozen VISTA3D weight SHA-256 differs from resume checkpoint")


def architecture_for_feature_mode(mode: str) -> str:
    architectures = {
        "single": "vista3d_promptgen_v102_amos_adapter_v1",
        "multiscale": "vista3d_promptgen_v102_amos_multiscale_fpn_v1",
    }
    if mode not in architectures:
        raise ValueError(f"unknown feature bridge mode: {mode}")
    return architectures[mode]


def require_matching_feature_mode(state: dict, mode: str) -> None:
    if state.get("architecture") != architecture_for_feature_mode(mode):
        raise ValueError("resume checkpoint feature bridge mode differs from requested mode")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("audit-data", "smoke", "validate", "train"))
    p.add_argument("--split-json", type=Path, default=ROOT / "configs/ct13_v10_2_split_20260928.json")
    p.add_argument("--validation-json", type=Path, default=ROOT / "configs/ct13_v10_2_validation_four_new_sources4_classweight_v3_20260928.json")
    p.add_argument("--shared-cache-dir", type=Path, default=ROOT / "v10/.cache/totalseg_sam2_promptgen_v10_prompt_roi_full_v1")
    p.add_argument("--prompt-plane-cache-dir", type=Path, default=ROOT / ".cache/prompt_planes")
    p.add_argument("--vista-source", type=Path)
    p.add_argument("--vista-checkpoint", type=Path)
    p.add_argument("--promptgen-checkpoint", type=Path)
    p.add_argument("--feature-bridge-mode", choices=("single", "multiscale"), default="single")
    p.add_argument("--resume", type=Path, help="Only a vista3d checkpoint, never a v10.2 optimizer")
    p.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "output/amos")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--train-tasks-per-epoch", type=int, default=200)
    p.add_argument("--validate-every", type=int, default=5)
    p.add_argument("--prompt-lr", type=float, default=2e-5)
    p.add_argument("--adapter-lr", type=float, default=5e-5)
    p.add_argument("--adapter-only-epochs", type=int, default=5,
                   help="Initially freeze the transferred PromptGen; VISTA3D remains frozen throughout")
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--depth", type=int, default=96)
    p.add_argument("--stride", type=int, default=48)
    p.add_argument("--vista-hw", type=int, default=96)
    p.add_argument("--prompt-work-size", type=int, default=192,
                   help="Must equal the transferred PromptGen work_size; downsample its video once")
    p.add_argument("--threshold", type=float, default=0.60)
    p.add_argument("--seed", type=int, default=20260928)
    p.add_argument("--lr-plateau-patience", type=int, default=2)
    p.add_argument("--early-stop-patience", type=int, default=4)
    p.add_argument("--min-dice-improvement", type=float, default=0.001)
    p.add_argument("--lr-factor", type=float, default=0.5)
    p.add_argument("--min-prompt-lr", type=float, default=2e-7)
    p.add_argument("--min-adapter-lr", type=float, default=5e-7)
    return p


def _prepare(args):
    for path in (args.split_json, args.validation_json):
        if not path.is_file():
            raise FileNotFoundError(
                f"audited AMOS protocol file is missing: {path}; copy the exact "
                "v10.2 JSON from the existing training environment, do not re-split"
            )
    protocol = load_amos_protocol(args.split_json, args.validation_json)
    require_approved_amos_protocol(protocol)
    if not args.shared_cache_dir.is_dir() or not args.prompt_plane_cache_dir.is_dir():
        raise FileNotFoundError("approved shared case CT / prompt-plane cache directories are required")
    train_data = make_amos_dataset(protocol, args.shared_cache_dir, args.prompt_plane_cache_dir, training=True)
    val_data = make_amos_dataset(protocol, args.shared_cache_dir, args.prompt_plane_cache_dir, training=False)
    return protocol, train_data, val_data


def audit_data(protocol, train_data, val_data, args):
    """Exercise real AMOS CT, GT and existing scribble prompt without a new split."""

    for dataset, case in ((train_data, protocol.train[0]), (val_data, protocol.val[0])):
        index = next(index for index, row in enumerate(protocol.train if dataset is train_data else protocol.val) if row.case_id == case.case_id)
        cache_path = dataset._case_image_cache_path(str(case.image))
        if not cache_path or Path(cache_path).parent != Path(dataset.shared_case_cache_dir) / "_case_images_v1":
            raise RuntimeError("AMOS dataset is not using the approved shared case CT cache")
        view = read_case_view(dataset, index, case.classes[0], case.image,
                              vista_hw=args.vista_hw, prompt_hw=args.prompt_work_size)
        if view.vista_image.shape[0] != view.target.shape[0]:
            raise RuntimeError("AMOS CT/GT/prompt ROI depth differs")
        if not torch.isfinite(view.vista_image).all() or not torch.isfinite(view.promptgen_video).all():
            raise RuntimeError("AMOS CT input contains non-finite values")
        if view.bundle["groups"].sum() <= 0:
            raise RuntimeError("AMOS cached scribble prompt is empty")
    return {
        "amos_train_cases": len(protocol.train),
        "amos_val_cases": len(protocol.val),
        "fixed_amos_validation_tasks": len(protocol.tasks),
        "split_sha256": protocol.split_sha256,
        "validation_sha256": protocol.validation_sha256,
        "cache_reuse": "existing case CT + v10.2 adaptive scribble metadata",
    }


def make_paired_feature_modules(mode: str):
    """Keep the shared prompt adapter's random initialization identical in both arms."""
    adapter = VistaPromptAdapter()
    if mode == "single":
        return VistaFeatureBridge(), adapter
    if mode == "multiscale":
        return VistaMultiScaleFeatureBridge(), adapter
    raise ValueError(f"unknown feature bridge mode: {mode}")


def _build_model(args):
    if args.depth < 16 or args.depth % 16 or args.vista_hw < 32 or args.vista_hw % 16:
        raise ValueError("VISTA patch depth and XY size must be positive multiples of 16")
    for name in ("vista_source", "vista_checkpoint", "promptgen_checkpoint"):
        if getattr(args, name) is None:
            raise ValueError(f"--{name.replace('_', '-')} is required")
    core = load_official_vista3d(args.vista_source, args.vista_checkpoint, args.device)
    prompt = load_v102_promptgen(args.promptgen_checkpoint, args.device)
    if int(prompt.work_size) != args.prompt_work_size:
        raise ValueError("--prompt-work-size differs from v10.2 checkpoint construction args")
    bridge, prompt_adapter = make_paired_feature_modules(args.feature_bridge_mode)
    model = VistaPromptGenModel(
        core, prompt, feature_bridge=bridge, prompt_adapter=prompt_adapter,
        feature_bridge_mode=args.feature_bridge_mode,
    ).to(args.device)
    return model


def _checkpoint(path: Path, model, optimizer, epoch: int, protocol, args, metrics=None,
                training_control=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save({
        "architecture": architecture_for_feature_mode(args.feature_bridge_mode),
        "epoch": epoch,
        "prompt_generator": model.promptgen.state_dict(),
        "vista_feature_bridge": model.feature_bridge.state_dict(),
        "vista_prompt_adapter": model.prompt_adapter.state_dict(),
        "optimizer": optimizer.state_dict(),
        "split_sha256": protocol.split_sha256,
        "validation_sha256": protocol.validation_sha256,
        "vista_checkpoint": str(args.vista_checkpoint),
        "vista_checkpoint_sha256": args.vista_checkpoint_sha256,
        "metrics": metrics,
        "training_control": training_control,
    }, temporary)
    os.replace(temporary, path)


def validate(model, protocol, val_data, args):
    model.eval()
    by_id = {case.case_id: index for index, case in enumerate(protocol.val)}
    results = []
    import nibabel as nib

    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16,
        enabled=str(args.device).startswith("cuda"),
    ):
        for number, task in enumerate(protocol.tasks, 1):
            index = by_id[task.case_id]
            view = read_case_view(val_data, index, task.global_class_id, task.image,
                                  vista_hw=args.vista_hw, prompt_hw=args.prompt_work_size)
            probability = predict_complete_roi(
                model, view, task.class_name, depth=args.depth,
                stride=args.stride, device=args.device,
            )
            truth = val_data._load_canonical_target(index, task.global_class_id)
            if tuple(truth.shape) != view.raw_shape:
                raise RuntimeError(f"canonical AMOS CT/GT grid mismatch: {task.case_id}")
            canonical = nib.as_closest_canonical(nib.load(str(task.image)))
            xyz_spacing = canonical.header.get_zooms()[:3]
            spacing = (xyz_spacing[2], xyz_spacing[0], xyz_spacing[1])
            metric = binary_metrics(
                probability.cpu().numpy() >= args.threshold,
                truth > 0,
                spacing_zyx=spacing,
            )
            results.append({"case_id": task.case_id, "class_id": task.global_class_id, "class_name": task.class_name, **metric})
            print(f"validation {number}/{len(protocol.tasks)} case={task.case_id} class={task.global_class_id} dice={metric['dice']:.5f}", flush=True)
    fields = ("dice", "iou", "precision", "recall", "specificity", "pred_gt_ratio", "nsd_1mm_3d")
    summary = {field: float(np.mean([row[field] for row in results])) for field in fields}
    finite_hd95 = [row["hd95_mm_3d"] for row in results
                   if row["hd95_mm_3d"] is not None and np.isfinite(row["hd95_mm_3d"])]
    summary["hd95_mm_3d"] = float(np.mean(finite_hd95)) if finite_hd95 else None
    summary.update({"tasks": len(results), "classes": len({row["class_id"] for row in results}), "split_sha256": protocol.split_sha256, "validation_sha256": protocol.validation_sha256})
    return {"summary": summary, "tasks": results}


def smoke_real_case(model, protocol, train_data, val_data, args):
    """One real cached AMOS gradient step plus one fixed full-volume task."""

    model.train()
    case = next(case for case in protocol.train if case.classes)
    index = protocol.train.index(case)
    view = read_case_view(train_data, index, case.classes[0], case.image,
                          vista_hw=args.vista_hw, prompt_hw=args.prompt_work_size)
    start, end = sliding_windows(len(view.frame_indices), depth=args.depth, stride=args.stride)[0]
    image, video, bundle, target, valid = pack_window(view, start, end, depth=args.depth, device=args.device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                        enabled=str(args.device).startswith("cuda")):
        logits = model.forward_patch(image, video, bundle, protocol.class_names[case.classes[0]])
        loss = segmentation_loss_compatible(logits, target, valid)
    if not torch.isfinite(loss):
        raise FloatingPointError("real AMOS VISTA3D/PromptGen loss is nonfinite")
    loss.backward()
    gradients = {
        "feature_bridge": any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in model.feature_bridge.parameters()),
        "prompt_adapter": any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in model.prompt_adapter.parameters()),
        "promptgen": any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in model.promptgen.parameters() if p.requires_grad),
    }
    if not all(gradients.values()):
        raise RuntimeError(f"real AMOS adapter gradient contract failed: {gradients}")
    if any(p.grad is not None for p in model.vista.parameters()):
        raise RuntimeError("frozen VISTA3D backbone accumulated gradients")
    model.zero_grad(set_to_none=True)
    task = protocol.tasks[0]
    val_index = next(i for i, row in enumerate(protocol.val) if row.case_id == task.case_id)
    val_view = read_case_view(val_data, val_index, task.global_class_id, task.image,
                              vista_hw=args.vista_hw, prompt_hw=args.prompt_work_size)
    model.eval()
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16,
        enabled=str(args.device).startswith("cuda"),
    ):
        probability = predict_complete_roi(model, val_view, task.class_name, depth=args.depth, stride=args.stride, device=args.device)
    truth = val_data._load_canonical_target(val_index, task.global_class_id)
    if probability.shape != truth.shape or not torch.isfinite(probability).all():
        raise RuntimeError("real AMOS full-volume prediction/GT contract failed")
    return {"train_case": case.case_id, "train_class": case.classes[0], "finite_loss": float(loss.detach()), "gradients": gradients, "validation_case": task.case_id, "validation_class": task.global_class_id, "full_volume_shape": list(truth.shape)}


def main(argv=None):
    args = parser().parse_args(argv)
    if args.mode == "validate" and args.resume is None:
        raise ValueError("validate requires --resume with a trained VISTA3D adapter checkpoint")
    if args.epochs < 1 or args.train_tasks_per_epoch < 1 or args.validate_every < 1:
        raise ValueError("epoch/task/validation settings must be positive")
    if not (0.0 < args.threshold < 1.0):
        raise ValueError("threshold must be in (0,1)")
    control_settings = {
        "min_delta": args.min_dice_improvement,
        "lr_patience": args.lr_plateau_patience,
        "stop_patience": args.early_stop_patience,
        "factor": args.lr_factor,
        "min_lrs": (args.min_prompt_lr, args.min_adapter_lr),
    }
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    protocol, train_data, val_data = _prepare(args)
    audit = audit_data(protocol, train_data, val_data, args)
    print(json.dumps(audit, indent=2), flush=True)
    if args.mode == "audit-data":
        return
    if args.vista_checkpoint is None or not args.vista_checkpoint.is_file():
        raise FileNotFoundError("--vista-checkpoint must point to the official research weight")
    args.vista_checkpoint_sha256 = file_sha256(args.vista_checkpoint)
    model = _build_model(args)
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        require_matching_feature_mode(state, args.feature_bridge_mode)
        require_matching_vista_checkpoint(state, args.vista_checkpoint_sha256)
        if state["split_sha256"] != protocol.split_sha256 or state["validation_sha256"] != protocol.validation_sha256:
            raise ValueError("resume protocol hash differs")
        model.promptgen.load_state_dict(state["prompt_generator"], strict=True)
        model.feature_bridge.load_state_dict(state["vista_feature_bridge"], strict=True)
        model.prompt_adapter.load_state_dict(state["vista_prompt_adapter"], strict=True)
    if args.mode == "validate":
        result = validate(model, protocol, val_data, args)
        print(json.dumps(result["summary"], indent=2), flush=True)
        return
    if args.mode == "smoke":
        print(json.dumps(smoke_real_case(model, protocol, train_data, val_data, args), indent=2), flush=True)
        return

    groups = [
        {"params": [parameter for parameter in model.promptgen.parameters() if parameter.requires_grad], "lr": args.prompt_lr},
        {"params": list(model.feature_bridge.parameters()) + list(model.prompt_adapter.parameters()), "lr": args.adapter_lr},
    ]
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    prompt_trainable = {id(parameter): parameter.requires_grad for parameter in model.promptgen.parameters()}
    start_epoch = 1
    if args.resume:
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"]) + 1
        control = ValidationController.from_checkpoint(state, **control_settings)
    else:
        control = ValidationController(
            best_dice=-1.0, significant_best_dice=-1.0, **control_settings
        )
    tasks = [(index, class_id) for index, case in enumerate(protocol.train) for class_id in case.classes]
    if not tasks:
        raise ValueError("AMOS CT train split has no nonempty class task")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        model.promptgen.set_morphology_routing_epoch(model.promptgen.vista_source_epoch + epoch)
        for parameter in model.promptgen.parameters():
            parameter.requires_grad_(prompt_trainable[id(parameter)] and epoch > args.adapter_only_epochs)
        epoch_loss = 0.0
        sample = random.Random(args.seed + epoch).choices(tasks, k=args.train_tasks_per_epoch)
        for step, (index, class_id) in enumerate(sample, 1):
            case = protocol.train[index]
            view = read_case_view(train_data, index, class_id, case.image,
                                  vista_hw=args.vista_hw, prompt_hw=args.prompt_work_size)
            windows = sliding_windows(len(view.frame_indices), depth=args.depth, stride=args.stride)
            start, end = windows[random.Random(args.seed + epoch * 100000 + step).randrange(len(windows))]
            image, video, bundle, target, valid = pack_window(view, start, end, depth=args.depth, device=args.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=str(args.device).startswith("cuda")):
                logits = model.forward_patch(image, video, bundle, protocol.class_names[class_id])
                loss = segmentation_loss_compatible(logits, target, valid)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite AMOS loss: epoch={epoch} step={step} case={case.case_id} class={class_id}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for group in groups for p in group["params"]], 1.0)
            optimizer.step()
            epoch_loss += float(loss.detach())
            print(f"epoch={epoch} step={step}/{len(sample)} case={case.case_id} class={class_id} loss={float(loss.detach()):.5f}", flush=True)
        metrics = None
        should_stop = False
        if epoch % args.validate_every == 0 or epoch == args.epochs:
            metrics = validate(model, protocol, val_data, args)
            metrics_path = args.out_dir / f"validation_epoch{epoch:03d}.json"
            metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
            dice = metrics["summary"]["dice"]
            decision = control.observe(dice, [group["lr"] for group in optimizer.param_groups])
            for group, lr in zip(optimizer.param_groups, decision.lrs):
                group["lr"] = lr
            should_stop = decision.should_stop
            print(
                f"epoch={epoch} val_dice={dice:.6f} best_dice={control.best_dice:.6f} "
                f"bad_validations={control.bad_validations}/{control.stop_patience} "
                f"lr_reduced={decision.reduced_lrs} early_stop={should_stop}",
                flush=True,
            )
            if decision.new_best:
                _checkpoint(args.out_dir / "best.pth", model, optimizer, epoch, protocol,
                            args, metrics["summary"], control.state_dict())
            _checkpoint(args.out_dir / f"epoch{epoch:03d}.pth", model, optimizer,
                        epoch, protocol, args, metrics["summary"], control.state_dict())
        _checkpoint(args.out_dir / "last.pth", model, optimizer, epoch, protocol, args,
                    None if metrics is None else metrics["summary"], control.state_dict())
        print(
            f"epoch={epoch} mean_loss={epoch_loss / len(sample):.5f} "
            f"prompt_lr={optimizer.param_groups[0]['lr']:.9g} "
            f"adapter_lr={optimizer.param_groups[1]['lr']:.9g} "
            f"time={datetime.now().isoformat()}", flush=True,
        )
        if should_stop:
            print(f"early_stop epoch={epoch} after {control.bad_validations} validations without meaningful Dice improvement", flush=True)
            break


if __name__ == "__main__":
    main()
