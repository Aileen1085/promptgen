"""v10.2: v10 memory plus latest v9.2 multi-dataset/input/expert policy."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import logging
from pathlib import Path
import sys


V10_ROOT = Path(__file__).resolve().parent
V9_ROOT = V10_ROOT.parent
for root in (V10_ROOT, V9_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import numpy as np
import torch
import torch.multiprocessing as mp

import finetune_totalseg_sam2_promptgen_v10_1_joint as v101
from model.v9_2_checkpoint_migration import migrate_v9_2_prompt_state
from utils.distributed_runtime import find_free_port, parse_gpu_ids
from v10_2_data import (
    FixedTaskV102Dataset,
    JointSourceQuotaV102Dataset,
    make_source_dataset,
)
from v10_2_multidataset_runtime import (
    SELECTION_WEIGHTS,
    SOURCE_ORDER,
    dataset_weighted_dice,
    validate_source_ratios,
)
from v9_1_multidataset_runtime import catalog_maps, load_and_validate_protocol
from v9_2_extended_protocol import (
    SOURCE_ORDER as EIGHT_SOURCE_ORDER,
    conditioning_text_from_metadata,
    validate_class_catalog_metadata,
)


v10 = v101.v10
v10_memory = v101.v10_memory
_BASE_BALANCED_EPOCH_INDICES = v10._balanced_epoch_indices
_BASE_EVALUATE = v10_memory.evaluate_memory_v10
_PROTOCOL: tuple[dict, dict] | None = None
_SEMANTIC_BY_ID: dict[int, str] = {}


def parser() -> argparse.ArgumentParser:
    result = v101.parser()
    result.description = (
        "v10.2 joint TotalSegmentator + AMOS + MAGIC training with grouped memory"
    )
    result.add_argument(
        "--multidataset-split-json",
        default=str(V9_ROOT / "configs/v9_1_multidataset_split_seed20260902.json"),
    )
    result.add_argument(
        "--multidataset-validation-json",
        default=str(V9_ROOT / "configs/v9_2_validation_t115_a45_m35_195.json"),
    )
    result.add_argument("--totalseg-sampling-ratio", type=float, default=0.30)
    result.add_argument("--amos-sampling-ratio", type=float, default=0.40)
    result.add_argument("--magic-sampling-ratio", type=float, default=0.30)
    result.add_argument("--msd-task07-sampling-ratio", type=float, default=0.10)
    result.add_argument("--msd-task10-sampling-ratio", type=float, default=0.075)
    result.add_argument("--msd-task08-sampling-ratio", type=float, default=0.10)
    result.add_argument("--parse2022-sampling-ratio", type=float, default=0.05)
    result.add_argument("--topcow2024-cta-sampling-ratio", type=float, default=0.125)
    result.add_argument("--v10-2-difficulty-sampling-cap", type=float, default=1.25)
    result.add_argument(
        "--v10-2-adaptive-scribble",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use per-component foreground scribble width for every data source.",
    )
    result.add_argument("--v10-2-scribble-min-radius", type=int, default=1)
    result.add_argument("--v10-2-scribble-max-radius", type=int, default=20)
    result.add_argument("--v10-2-scribble-width-fraction", type=float, default=0.15)
    result.add_argument(
        "--v10-2-scribble-min-plane-ratio",
        type=float,
        default=0.0,
        help=(
            "Minimum adaptive foreground-scribble radius as a fraction of the "
            "prompt plane's shorter side; 0 disables the semantic floor."
        ),
    )
    result.add_argument(
        "--v10-2-routing-policy",
        choices=("soft", "scheduled"),
        default="scheduled",
        help="Latest v9.2 defaults to soft-to-Top-2 scheduled expert routing.",
    )
    result.set_defaults(
        v10_1_stage="sat_morphology_role_memory_v92",
        train_memory_group_size=4,
        eval_memory_group_size=1,
        train_cases_per_epoch=200,
        prompt_semantic_dim=128,
    )
    return result


def extended_source_ratios(args) -> dict[str, float]:
    return {
        "totalseg": args.totalseg_sampling_ratio,
        "amos": args.amos_sampling_ratio,
        "magic": args.magic_sampling_ratio,
        "msd_task07": args.msd_task07_sampling_ratio,
        "msd_task10": args.msd_task10_sampling_ratio,
        "msd_task08": args.msd_task08_sampling_ratio,
        "parse2022": args.parse2022_sampling_ratio,
        "topcow2024_cta": args.topcow2024_cta_sampling_ratio,
    }


def _prepare_protocol(args) -> None:
    global _PROTOCOL, _SEMANTIC_BY_ID
    split_path = Path(args.multidataset_split_json)
    validation_path = Path(args.multidataset_validation_json)
    if not split_path.is_file():
        raise FileNotFoundError(
            f"Fixed multi-dataset split is missing: {split_path}. "
            "Copy the audited server protocol; do not regenerate a new split implicitly."
        )
    if not validation_path.is_file():
        raise FileNotFoundError(f"Fixed validation protocol is missing: {validation_path}")
    split, validation = load_and_validate_protocol(split_path, validation_path)
    source_order = tuple(split.get("source_order", SOURCE_ORDER))
    if source_order not in (SOURCE_ORDER, EIGHT_SOURCE_ORDER):
        raise ValueError(
            "v10.2 supports the audited three-source or eight-source protocol; "
            f"got source_order={source_order}"
        )
    catalogue = [dict(row) for row in split["class_catalog"]]
    if source_order == EIGHT_SOURCE_ORDER:
        validate_class_catalog_metadata(catalogue)
        _SEMANTIC_BY_ID = {
            int(row["global_class_id"]): conditioning_text_from_metadata(row)
            for row in catalogue
        }
        ratios = extended_source_ratios(args)
    else:
        _catalog, _SEMANTIC_BY_ID = catalog_maps(split)
        ratios = {
            "totalseg": args.totalseg_sampling_ratio,
            "amos": args.amos_sampling_ratio,
            "magic": args.magic_sampling_ratio,
        }
    _PROTOCOL = (split, validation)
    args.v10_2_source_order = source_order
    args.v10_2_class_catalog = catalogue
    args.multidataset_sampling_ratios = validate_source_ratios(
        ratios,
        source_order=source_order,
    )
    args.label_names = {
        int(row["global_class_id"]): str(row["class_name"])
        for row in split["class_catalog"]
    }
    args.validation_na_class_ids = tuple(
        sorted(
            int(row["global_class_id"])
            for row in catalogue
            if bool(row.get("not_applicable", False))
        )
    )
    args.num_classes = len(args.label_names)
    args.val_num_classes = args.num_classes


def semantic_text_for_global_class(class_id: int, args=None) -> str | None:
    if args is not None and not bool(getattr(args, "prompt_semantic_enabled", True)):
        return None
    try:
        return _SEMANTIC_BY_ID[int(class_id)]
    except KeyError as error:
        raise KeyError(f"missing semantic text for global class {class_id}") from error


def _apply_routing_policy(prompt, args) -> None:
    setter = getattr(prompt.encoder3d, "set_routing_policy", None)
    if not callable(setter):
        raise RuntimeError("v10.2 requires the v9.2 morphology routed encoder")
    if args.v10_2_routing_policy == "soft":
        prompt.encoder3d.set_routing_policy("soft")
    else:
        setter("scheduled")


def make_prompt_v10_2(args, adapter, device):
    prompt = v101.make_prompt_v10_1(args, adapter, device)
    _apply_routing_policy(prompt, args)
    return prompt


def load_init_v10_2(path, prompt, decoder):
    state = torch.load(path, map_location="cpu", weights_only=True)
    source_architecture = str(state.get("architecture", ""))
    target_architecture = v101.V10_1_TRAIN_ARCHITECTURES[prompt.v10_1_stage]
    if source_architecture != target_architecture:
        raise RuntimeError(
            "v10.2 must initialize from the matching v10.1 Stage-3 architecture; "
            f"expected {target_architecture!r}, got {source_architecture!r}"
        )
    source_prompt = state.get("prompt_generator", state)
    migrated, report = migrate_v9_2_prompt_state(
        source_prompt, prompt.state_dict()
    )
    prompt.load_state_dict(migrated, strict=True)
    v101._load_decoder_state(state, decoder, str(path))
    args = getattr(prompt, "_v10_2_args", None)
    if args is not None:
        _apply_routing_policy(prompt, args)
    logging.info(
        "v10.2 additive expert migration | source_arch=%s report=%s",
        source_architecture,
        report,
    )
    return state


def _validate_resume_is_v10_2(args) -> None:
    if not args.resume_checkpoint:
        return
    state = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=True)
    saved_args = state.get("args", {})
    if not isinstance(saved_args, dict) or "v10_2_routing_policy" not in saved_args:
        raise RuntimeError(
            "--resume-checkpoint must be a native v10.2 checkpoint. "
            "Use --prompt-generator-checkpoint for a model-only v10.1 Stage-3 transition."
        )


def make_dataset_v10_2(
    _pairs,
    args,
    target_class=None,
    return_target_class=False,
    class_weights=None,
    **_kwargs,
):
    if target_class is not None or not return_target_class:
        raise ValueError(
            "v10.2 joint training requires return_target_class and no fixed target"
        )
    if _PROTOCOL is None:
        raise RuntimeError("v10.2 multi-dataset protocol is not loaded")
    weights = class_weights or getattr(args, "validation_class_difficulty", {})
    return JointSourceQuotaV102Dataset(
        _PROTOCOL[0], args, weights, v101.make_dataset_v10_1
    )


def _discover_protocol_pairs(_root, split_name, limit=None):
    if _PROTOCOL is None:
        raise RuntimeError("v10.2 multi-dataset protocol is not loaded")
    entries = _PROTOCOL[0]["datasets"]["totalseg"][str(split_name)]
    pairs = [(str(row["image"]), str(row["label"])) for row in entries]
    return pairs if limit is None else pairs[: int(limit)]


def _load_global_label_map(_path):
    if _PROTOCOL is None:
        raise RuntimeError("v10.2 multi-dataset protocol is not loaded")
    return {
        int(row["global_class_id"]): str(row["class_name"])
        for row in _PROTOCOL[0]["class_catalog"]
    }


def _balanced_joint_indices(dataset, args, ctx, epoch):
    setter = getattr(dataset, "set_epoch", None)
    if callable(setter):
        setter(int(epoch))
    return _BASE_BALANCED_EPOCH_INDICES(dataset, args, ctx, epoch)


def _finite_mean(values):
    rows = [float(value) for value in values if np.isfinite(float(value))]
    return float(np.mean(rows)) if rows else None


@torch.no_grad()
def evaluate_multidataset_v10_2(prompt_gen, adapter, _pairs, _class_ids, args, device):
    if _PROTOCOL is None:
        raise RuntimeError("v10.2 multi-dataset protocol is not loaded")
    split, validation = _PROTOCOL
    source_order = tuple(split["source_order"])
    selection_weights = (
        dict(SELECTION_WEIGHTS)
        if source_order == SOURCE_ORDER
        else dict(args.multidataset_sampling_ratios)
    )
    ct13_validation_v2 = validation.get("selection_weight_policy") == "nonempty_class_count"
    if ct13_validation_v2:
        from ct13_validation_policy import class_count_weights
        selection_weights = class_count_weights(split)
    all_per_class: dict[str, dict[int, float]] = {}
    per_dataset: dict[str, dict[str, float]] = {}
    source_metrics: dict[str, dict] = {}
    original_factory = v10_memory.make_dataset
    original_semantic = v10_memory.semantic_text_for_class
    try:
        v10_memory.semantic_text_for_class = semantic_text_for_global_class
        for source in source_order:
            tasks = [
                dict(row)
                for row in validation["tasks"]
                if str(row["dataset"]) == source
            ]
            if not tasks:
                raise RuntimeError(f"fixed validation has no {source} tasks")
            entries_by_case = {
                str(row["case_id"]): dict(row)
                for row in split["datasets"][source]["val"]
            }
            case_ids = list(dict.fromkeys(str(row["case_id"]) for row in tasks))
            entries = [entries_by_case[case_id] for case_id in case_ids]
            dataset = FixedTaskV102Dataset(
                make_source_dataset(
                    source,
                    entries,
                    args,
                    v101.make_dataset_v10_1,
                    foreground_mode=str(args.val_foreground_mode),
                ),
                tasks,
            )
            v10_memory.make_dataset = lambda *_a, _dataset=dataset, **_k: _dataset
            source_args = copy.copy(args)
            source_args.validation_max_tasks = 0
            source_args.validation_class_dice = {}
            source_args.validation_class_difficulty = {}
            class_ids = sorted({int(row["global_class_id"]) for row in tasks})
            pairs = list(zip(dataset.image_paths, dataset.label_paths))
            _dice, _class_dice, metrics = _BASE_EVALUATE(
                prompt_gen, adapter, pairs, class_ids, source_args, device
            )
            source_metrics[source] = metrics
            per_dataset[source] = dict(metrics["mean"])
            for metric_name, values in metrics["per_class"].items():
                all_per_class.setdefault(metric_name, {}).update(
                    {int(key): float(value) for key, value in values.items()}
                )
            logging.info(
                "v10.2 validation source=%s tasks=%d metrics=%s",
                source,
                len(tasks),
                per_dataset[source],
            )
    finally:
        v10_memory.make_dataset = original_factory
        v10_memory.semantic_text_for_class = original_semantic

    expected = {
        int(row["global_class_id"])
        for row in split["class_catalog"]
        if not bool(row.get("not_applicable", False))
    }
    observed = set(all_per_class.get("dice", {}))
    if observed != expected:
        raise RuntimeError(
            "combined validation coverage mismatch "
            f"missing={sorted(expected - observed)} extra={sorted(observed - expected)}"
        )
    mean = {
        name: value
        for name, rows in all_per_class.items()
        if (value := _finite_mean(rows.values())) is not None
    }
    weighted_score = dataset_weighted_dice(
        per_dataset,
        weights=selection_weights,
        source_order=source_order,
    )
    threshold_keys = set.intersection(*[
        set(metrics.get("per_threshold", {})) for metrics in source_metrics.values()
    ])
    per_threshold = {}
    for threshold in sorted(threshold_keys):
        names = set.intersection(*[
            set(source_metrics[source]["per_threshold"][threshold])
            for source in source_order
        ])
        per_threshold[threshold] = {
            name: sum(
                selection_weights[source]
                * float(source_metrics[source]["per_threshold"][threshold][name])
                for source in source_order
            )
            for name in names
        }
    selected_tasks = sum(len([
        row for row in validation["tasks"] if row["dataset"] == source
    ]) for source in source_order)
    wall = sum(
        float(metrics.get("performance", {}).get("wall_seconds", 0.0))
        for metrics in source_metrics.values()
    )
    mode = f"fg={args.val_foreground_mode}|bg={args.val_background_mode}"
    result = {
        "mean": mean,
        "per_class": all_per_class,
        "per_dataset": per_dataset,
        "selection": {
            "name": (
                "dataset_weighted_dice_20_30_50"
                if source_order == SOURCE_ORDER
                else "eight_source_dataset_weighted_dice"
            ),
            "score": weighted_score,
            "weights": selection_weights,
            "class_macro_dice": mean.get("dice"),
        },
        "per_prompt_mode": {mode: {"count": selected_tasks, **mean}},
        "per_threshold": per_threshold,
        "primary_threshold": float(args.mask_threshold),
        "foreground_class_macro_dice": {
            str(args.val_foreground_mode): float(mean["dice"])
        },
        "validation_protocol": {
            "selected_tasks": selected_tasks,
            "covered_classes": len(observed),
            "requested_classes": len(expected),
            "na_classes": list(args.validation_na_class_ids),
            "all_classes_covered": True,
            "all_nonempty_classes_covered": True,
            "metric_space": "raw_nifti_full_volume",
            "fixed_task_json": str(args.multidataset_validation_json),
            "source_ratios": dict(args.multidataset_sampling_ratios),
            "anchor_coarse_crop": bool(args.anchor_coarse_crop),
            "eval_memory_group_size": int(args.eval_memory_group_size),
        },
        "performance": {
            "wall_seconds": wall,
            "tasks": selected_tasks,
            "seconds_per_task": wall / max(1, selected_tasks),
        },
    }
    if ct13_validation_v2:
        from ct13_validation_policy import case_class_macro
        result["case_class_macro"] = case_class_macro(source_metrics)
        result["selection"]["name"] = "ct13_nonempty_class_count_weighted_dice"
        result["validation_protocol"]["selection_weight_policy"] = "nonempty_class_count"
        logging.info("CT13 case_class_macro=%s selection=%s", result["case_class_macro"], result["selection"])
    return weighted_score, dict(all_per_class["dice"]), result


def _bind_v10_2(args) -> None:
    v101._bind_v10_1(args.v10_1_stage)
    v10.make_prompt = make_prompt_v10_2
    v10.make_dataset = make_dataset_v10_2
    v10_memory.make_dataset = make_dataset_v10_2
    v10_memory.semantic_text_for_class = semantic_text_for_global_class
    v10.load_init = load_init_v10_2
    v10.evaluate_memory_v10 = evaluate_multidataset_v10_2
    v10.discover_totalseg_split_pairs = _discover_protocol_pairs
    v10.load_label_map = _load_global_label_map
    v10._balanced_epoch_indices = _balanced_joint_indices


def _worker_v10_2(rank: int, world: int, port: int, stamp: str, args) -> None:
    # ``mp.spawn`` starts fresh interpreters, so the protocol loaded by the
    # parent process is not present in each worker's module globals.
    _prepare_protocol(args)
    _bind_v10_2(args)
    original_make_prompt = v10.make_prompt

    def prompt_with_args(bound_args, adapter, device):
        prompt = original_make_prompt(bound_args, adapter, device)
        prompt._v10_2_args = bound_args
        return prompt

    v10.make_prompt = prompt_with_args
    v10.worker(rank, world, port, stamp, args)


def main() -> None:
    args = parser().parse_args()
    v101._validate_extension_args(args)
    if int(args.train_memory_group_size) < 1 or int(args.eval_memory_group_size) < 1:
        raise ValueError("memory group sizes must be positive")
    if float(args.v10_2_difficulty_sampling_cap) <= 0.0:
        raise ValueError("--v10-2-difficulty-sampling-cap must be positive")
    if int(args.v10_2_scribble_min_radius) < 0:
        raise ValueError("--v10-2-scribble-min-radius must be non-negative")
    if int(args.v10_2_scribble_max_radius) < int(args.v10_2_scribble_min_radius):
        raise ValueError("adaptive scribble max radius must be >= min radius")
    if not 0.0 < float(args.v10_2_scribble_width_fraction) <= 0.5:
        raise ValueError("--v10-2-scribble-width-fraction must be in (0, 0.5]")
    if not 0.0 <= float(args.v10_2_scribble_min_plane_ratio) <= 0.10:
        raise ValueError("--v10-2-scribble-min-plane-ratio must be in [0, 0.10]")
    _prepare_protocol(args)
    v101._validate_resume_stage(args)
    _validate_resume_is_v10_2(args)
    args.amp = True
    args.object_score_gate = False
    ids = parse_gpu_ids(args.gpu)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if len(ids) > 1:
        mp.spawn(
            _worker_v10_2,
            nprocs=len(ids),
            args=(len(ids), find_free_port(), stamp, args),
            join=True,
        )
    else:
        _worker_v10_2(0, 1, find_free_port(), stamp, args)


if __name__ == "__main__":
    main()
