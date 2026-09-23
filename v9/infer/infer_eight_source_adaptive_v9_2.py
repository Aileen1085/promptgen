from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
from types import MethodType
from typing import Iterable, Mapping

import numpy as np

from infer.v9_2_eight_source_eval_protocol import checkpoint_plan, prompt_package_to_groups
from infer.magic_prompt_alignment_v9_2 import align_magic_prompt_package


PROMPT_PROTOCOL = "v9_2_adaptive_only_r1_20_f0p15_final_trigger"
TRAINING_BACKGROUND_SCRIBBLE_PROTOCOL = (
    "v9_2_adaptive_fg_r1_20_f0p15_bg_training_scribble_final_trigger"
)
MANIFEST_KIND = "independent_eight_source_validation_v1"


def evaluation_prompt_protocol(background_mode: str) -> str:
    mode = str(background_mode).strip().lower()
    if mode == "point":
        return PROMPT_PROTOCOL
    if mode == "scribble":
        return TRAINING_BACKGROUND_SCRIBBLE_PROTOCOL
    raise ValueError(f"unsupported background mode: {background_mode!r}")


def training_background_scribbles_for_package(
    dataset,
    *,
    index: int,
    class_id: int,
    package: Mapping[str, object],
) -> dict[str, np.ndarray]:
    """Generate channel-1 prompts through the dataset's training prompt builder."""
    target = dataset._load_canonical_target(int(index), int(class_id))
    output: dict[str, np.ndarray] = {}
    for plane in ("coronal", "sagittal"):
        slice_index = int(package[f"{plane}_slice"])
        training_group = np.asarray(
            dataset._plane_prompt_group(target, plane, slice_index), dtype=np.uint8
        )
        if training_group.ndim != 3 or training_group.shape[0] < 2:
            raise RuntimeError(
                f"training prompt builder returned invalid {plane} group {training_group.shape}"
            )
        output[plane] = training_group[1].copy()
    return output


def _task_key(row: Mapping[str, object]) -> tuple[str, str, int]:
    return (
        str(row.get("source_name", row.get("dataset"))),
        str(row["case_id"]),
        int(row.get("global_class_id", row.get("class_id"))),
    )


def load_independent_manifest(
    path: str | Path, *, sources: set[str] | None = None
) -> list[dict[str, object]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("manifest_kind") != MANIFEST_KIND:
        raise ValueError("v9.2 evaluation requires the independent eight-source validation manifest")
    all_rows = [dict(row) for row in payload.get("tasks", [])]
    if int(payload.get("expected_task_count", -1)) != len(all_rows):
        raise ValueError("independent manifest expected_task_count does not match its tasks")
    rows = [row for row in all_rows if sources is None or str(row["source_name"]) in sources]
    keys = [_task_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate independent validation task")
    if any(str(row.get("split_role")) != "validation" for row in rows):
        raise ValueError("independent manifest contains a non-validation task")
    if sources is not None and set(str(row["source_name"]) for row in rows) != set(sources):
        raise ValueError("one or more requested independent validation sources are absent")
    return rows


def shard_tasks_by_case(
    tasks: Iterable[Mapping[str, object]], *, shard_count: int, shard_index: int
) -> list[dict[str, object]]:
    """Balance complete cases across shards without splitting their classes."""
    if int(shard_count) < 1:
        raise ValueError("shard_count must be positive")
    if not 0 <= int(shard_index) < int(shard_count):
        raise ValueError("shard_index must be in [0, shard_count)")
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for task in tasks:
        row = dict(task)
        key = (str(row["source_name"]), str(row["case_id"]))
        grouped.setdefault(key, []).append(row)
    assignments: list[list[dict[str, object]]] = [[] for _ in range(int(shard_count))]
    loads = [0 for _ in assignments]
    for _case_key, rows in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        owner = min(range(len(assignments)), key=lambda value: (loads[value], value))
        assignments[owner].extend(rows)
        loads[owner] += len(rows)
    return sorted(assignments[int(shard_index)], key=_task_key)


def _task_path(row: Mapping[str, object], key: str) -> str:
    alternatives = (f"{key}_path", key)
    for candidate in alternatives:
        value = row.get(candidate)
        if value:
            return str(value)
    raise KeyError(f"task has no {key} path: {_task_key(row)}")


def resolve_model_task_paths(
    base_split: Mapping[str, object], tasks: Iterable[Mapping[str, object]]
) -> list[dict[str, object]]:
    """Resolve model-readable paths without changing task identity or prompt keys.

    MAGIC's independent manifest intentionally points at the raw NRRD source used by
    nnInteractive.  The v9.2 loader is NIfTI-only, so use the geometry-equivalent
    preprocessed NIfTI that is already recorded in the original validation split.
    Looking up validation entries only also prevents accidental train-case leakage.
    """
    rows = [dict(row) for row in tasks]
    datasets = dict(base_split.get("datasets", {}))
    magic_val = {
        str(entry["case_id"]): entry
        for entry in dict(datasets.get("magic", {})).get("val", [])
    }
    for row in rows:
        if str(row.get("source_name", row.get("dataset"))) != "magic":
            continue
        case_id = str(row["case_id"])
        entry = magic_val.get(case_id)
        if entry is None:
            raise ValueError(
                f"MAGIC case {case_id!r} is absent from the base validation split"
            )
        for key in ("image", "label"):
            resolved = _task_path(entry, key)
            row[key] = resolved
            if f"{key}_path" in row:
                row[f"{key}_path"] = resolved
    return rows


def build_selected_protocol(
    base_split: Mapping[str, object], tasks: Iterable[Mapping[str, object]]
) -> tuple[dict[str, object], dict[str, object]]:
    rows = [dict(row) for row in tasks]
    split = copy.deepcopy(dict(base_split))
    datasets = split.setdefault("datasets", {})
    selected_sources = sorted({str(row["source_name"]) for row in rows})
    validation_rows: list[dict[str, object]] = []
    for source in selected_sources:
        source_rows = [row for row in rows if str(row["source_name"]) == source]
        grouped: dict[str, dict[str, object]] = {}
        for row in source_rows:
            case_id = str(row["case_id"])
            entry = grouped.setdefault(
                case_id,
                {
                    "case_id": case_id,
                    "image": _task_path(row, "image"),
                    "label": _task_path(row, "label"),
                    "classes": [],
                    "modality": "ct",
                },
            )
            if entry["image"] != _task_path(row, "image") or entry["label"] != _task_path(row, "label"):
                raise ValueError(f"case paths disagree within independent manifest: {(source, case_id)}")
            entry["classes"].append(int(row["global_class_id"]))
            validation_rows.append(
                {
                    "case_id": case_id,
                    "image": entry["image"],
                    "label": entry["label"],
                    "dataset": source,
                    "global_class_id": int(row["global_class_id"]),
                    "local_class_id": int(row["local_class_id"]),
                    "class_name": str(row["class_name"]),
                }
            )
        for entry in grouped.values():
            entry["classes"] = sorted(set(int(value) for value in entry["classes"]))
        source_split = datasets.setdefault(source, {"train": [], "val": []})
        source_split["val"] = [grouped[key] for key in sorted(grouped)]
        validation_case_ids = set(grouped)
        source_split["train"] = [
            entry for entry in source_split.get("train", [])
            if str(entry.get("case_id")) not in validation_case_ids
        ]
    validation = {
        "protocol": "independent_eight_source_validation_v1",
        "tasks": sorted(validation_rows, key=lambda row: (row["dataset"], row["case_id"], row["global_class_id"])),
        "not_applicable": [],
        "source_order": list(split.get("source_order", selected_sources)),
    }
    return split, validation


def build_prompt_result_index(
    tasks: Iterable[Mapping[str, object]], rows: Iterable[Mapping[str, object]]
) -> dict[tuple[str, str, int], str]:
    expected = {_task_key(row) for row in tasks}
    index: dict[tuple[str, str, int], str] = {}
    for row in rows:
        if str(row.get("prompt_protocol")) != PROMPT_PROTOCOL:
            raise ValueError("nnInteractive prompt result uses an incompatible prompt protocol")
        key = _task_key(row)
        if key in index:
            raise ValueError(f"duplicate nnInteractive prompt result: {key}")
        index[key] = str(row["prompt_cache_name"])
    if set(index) != expected:
        raise ValueError(
            f"nnInteractive prompt result coverage mismatch: missing={sorted(expected-set(index))[:5]} "
            f"extra={sorted(set(index)-expected)[:5]}"
        )
    return index


def _extract_custom_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Evaluate v9.2 on the independent eight-source manifest with exact cached prompts."
    )
    parser.add_argument("--independent-task-manifest", required=True)
    parser.add_argument("--nninteractive-metrics", required=True)
    parser.add_argument("--shared-adaptive-prompt-root", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--v9-2-weights", required=True)
    parser.add_argument("--allow-uniform-weight", action="store_true")
    parser.add_argument("--source", required=True)
    parser.add_argument("--eval-out-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--sam2-frame-batch-size", type=int, default=4)
    parser.add_argument("--task-shard-count", type=int, default=1)
    parser.add_argument("--task-shard-index", type=int, default=0)
    parser.add_argument("--dynamic-threshold-fallback", type=float, default=0.60)
    parser.add_argument("--dynamic-threshold-min", type=float, default=0.35)
    parser.add_argument("--dynamic-threshold-max", type=float, default=0.75)
    parser.add_argument("--dynamic-threshold-min-logit-separation", type=float, default=0.50)
    parser.add_argument("--prompt-logit-neighborhood-radius", type=int, default=2)
    parser.add_argument("--exact-background-mode", choices=("point", "scribble"), default="point")
    parser.add_argument("--base-split-json", default="configs/v9_2_extended_split_4case_v2.json")
    return parser.parse_known_args(sys.argv[1:])


def _append_flag(arguments: list[str], flag: str, *values: object) -> None:
    arguments.extend([flag, *(str(value) for value in values)])


def _install_exact_prompt_factory(
    entry,
    tasks: list[dict[str, object]],
    prompt_index: dict[tuple[str, str, int], str],
    prompt_root: Path,
    audit: dict[str, object],
    background_mode: str,
) -> None:
    import torch
    nninteractive_infer = Path(__file__).resolve().parents[1] / "nninteractive_test" / "infer"
    if str(nninteractive_infer) not in sys.path:
        sys.path.insert(0, str(nninteractive_infer))
    from nninteractive_test.infer.adaptive_prompt_cache import _read
    from v9_family_shared_cache_entry import _enable

    original_factory = entry._make_source_dataset_adaptive
    task_by_image_class = {
        (os.path.normpath(_task_path(row, "image")), int(row["global_class_id"])): row
        for row in tasks
    }

    def factory(source: str, entries: list[dict], args, *, complete_prompt_roi: bool = False):
        dataset = original_factory(source, entries, args, complete_prompt_roi=complete_prompt_roi)
        dataset = _enable(dataset, source_name=source, item_cache_enabled=False)
        inner = dataset
        while hasattr(inner, "dataset"):
            inner = inner.dataset

        def exact_prompt_groups(self, index, class_id):
            image_path = os.path.normpath(str(self.image_paths[int(index)]))
            task = task_by_image_class[(image_path, int(class_id))]
            key = _task_key(task)
            cache_path = prompt_root / "metadata" / "shared_adaptive_prompt_v1" / source / prompt_index[key]
            package = _read(cache_path)
            if package is None:
                raise RuntimeError(f"missing or invalid reusable prompt cache: {cache_path}")
            if source == "magic":
                target = self._load_canonical_target(int(index), int(class_id))
                package = align_magic_prompt_package(
                    package, expected_shape_dhw=tuple(map(int, target.shape))
                )
            background_scribbles = None
            if background_mode == "scribble":
                background_scribbles = training_background_scribbles_for_package(
                    self,
                    index=int(index),
                    class_id=int(class_id),
                    package=package,
                )
            converted = prompt_package_to_groups(
                package,
                prompt_plane_size=int(self.prompt_plane_size),
                background_scribbles=background_scribbles,
            )
            audit.setdefault("loaded_task_keys", []).append(list(key))
            return {
                "groups": torch.from_numpy(np.asarray(converted["groups"], dtype=np.float32)),
                "plane_ids": torch.from_numpy(np.asarray(converted["plane_ids"], dtype=np.int64)),
                "slice_coords": torch.from_numpy(np.asarray(converted["slice_coords"], dtype=np.float32)),
                "canonical_depth": int(converted["canonical_depth"]),
            }

        inner._build_prompt_slice_groups = MethodType(exact_prompt_groups, inner)
        return dataset

    entry.multi._make_source_dataset = factory


def _preserve_all_background_points(source, fallback=None):
    return (source > 0).to(dtype=source.dtype)


def validate_checkpoint_selection(
    plan: Mapping[str, Mapping[str, object]],
    sources: set[str],
    actual_weight: str,
    allow_uniform_weight: bool,
) -> None:
    selected_weight_paths = {
        str(Path(plan[source]["path"]).resolve()) for source in sources
    }
    resolved_actual = str(Path(actual_weight).resolve())
    if selected_weight_paths != {resolved_actual} and not bool(allow_uniform_weight):
        raise ValueError(
            f"selected sources require checkpoint paths {sorted(selected_weight_paths)}, "
            f"got {resolved_actual}"
        )


def main() -> None:
    custom, remaining = _extract_custom_args()
    sources = {value.strip() for value in str(custom.source).split(",") if value.strip()}
    manifest_tasks = load_independent_manifest(custom.independent_task_manifest, sources=sources)
    manifest_tasks = shard_tasks_by_case(
        manifest_tasks,
        shard_count=custom.task_shard_count,
        shard_index=custom.task_shard_index,
    )
    if not manifest_tasks:
        raise ValueError("selected case shard is empty")
    selected_task_keys = {_task_key(row) for row in manifest_tasks}
    prompt_payload = json.loads(Path(custom.nninteractive_metrics).read_text(encoding="utf-8"))
    prompt_rows = [
        row
        for row in prompt_payload.get("per_case_class", [])
        if _task_key(row) in selected_task_keys
    ]
    prompt_index = build_prompt_result_index(manifest_tasks, prompt_rows)
    base_split = json.loads(Path(custom.base_split_json).read_text(encoding="utf-8"))
    tasks = resolve_model_task_paths(base_split, manifest_tasks)
    split, validation = build_selected_protocol(base_split, tasks)

    plan = checkpoint_plan(custom.checkpoint_dir)
    actual_weight = str(Path(custom.v9_2_weights).resolve())
    validate_checkpoint_selection(
        plan,
        sources,
        actual_weight,
        bool(custom.allow_uniform_weight),
    )

    # The legacy joint entry reads --gpu during import and rewrites
    # CUDA_VISIBLE_DEVICES.  Keep the requested physical GPU visible for that
    # import, then pass local ordinal 0 to the actual single-GPU evaluator.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(custom.gpu)
    import finetune_multisource_sam2_v9_2_precision as entry
    import v9_totalseg_runtime as validation_runtime
    from infer.v9_2_dynamic_validation_patch import (
        dynamic_threshold_settings,
        install_dynamic_validation_patch,
    )
    from model.v9_prompt_generator_modes import CrossViewPromptTokenGeneratorV7
    import torch

    visible_gpu_uuid = str(torch.cuda.get_device_properties(0).uuid)

    entry.multi.load_and_validate_protocol = lambda *_args, **_kwargs: (split, validation)
    audit: dict[str, object] = {
        "prompt_protocol": evaluation_prompt_protocol(custom.exact_background_mode),
        "source_prompt_protocol": PROMPT_PROTOCOL,
        "background_mode": str(custom.exact_background_mode),
        "background_generation": (
            "dataset_training_plane_prompt_group_channel_1"
            if custom.exact_background_mode == "scribble"
            else "cached_two_points_per_plane"
        ),
        "expected_task_count": len(tasks),
        "sources": sorted(sources),
        "checkpoint": actual_weight,
        "checkpoint_selection": {source: plan[source] for source in sorted(sources)},
        "uniform_checkpoint_override": bool(custom.allow_uniform_weight),
        "magic_model_input": "preprocessed_nifti_from_base_validation_split",
        "magic_prompt_alignment": "raw_nrrd_to_model_nifti_swap_flip_xy_v1",
        "requested_physical_gpu": int(custom.gpu),
        "visible_cuda_ordinal": 0,
        "visible_gpu_uuid": visible_gpu_uuid,
        "loaded_task_keys": [],
        "task_shard_count": int(custom.task_shard_count),
        "task_shard_index": int(custom.task_shard_index),
        "shard_strategy": "whole_case_greedy_balance",
        "dynamic_threshold": dynamic_threshold_settings(
            fallback=custom.dynamic_threshold_fallback,
            minimum=custom.dynamic_threshold_min,
            maximum=custom.dynamic_threshold_max,
            min_logit_separation=custom.dynamic_threshold_min_logit_separation,
            prompt_logit_neighborhood_radius=custom.prompt_logit_neighborhood_radius,
        ),
        "dynamic_threshold_rows": [],
    }
    _install_exact_prompt_factory(
        entry,
        tasks,
        prompt_index,
        Path(custom.shared_adaptive_prompt_root),
        audit,
        str(custom.exact_background_mode),
    )
    if custom.exact_background_mode == "point":
        CrossViewPromptTokenGeneratorV7._one_point_per_group = staticmethod(
            _preserve_all_background_points
        )
    install_dynamic_validation_patch(
        validation_runtime,
        audit["dynamic_threshold_rows"],
        **{
            key: value
            for key, value in audit["dynamic_threshold"].items()
            if key != "method"
        },
    )

    class_ids = sorted({int(row["global_class_id"]) for row in tasks})
    forwarded = list(remaining)
    mandatory = (
        ("--prompt-generator-checkpoint", custom.v9_2_weights),
        ("--out-dir", custom.eval_out_dir),
        ("--gpu", 0),
        ("--sam2-frame-batch-size", custom.sam2_frame_batch_size),
        ("--multidataset-validation-class-ids", ",".join(str(value) for value in class_ids)),
        ("--max-frames", 256),
        ("--validation-window-frames", 256),
        ("--model-input-size", 1024),
        ("--cache-image-size", 512),
        ("--prompt-work-size", 192),
        ("--prompt-semantic-dim", 128),
        ("--amp-dtype", "bfloat16"),
        ("--v9-eval-foreground-mode", "scribble"),
        ("--v9-eval-background-mode", custom.exact_background_mode),
        ("--v9-2-scribble-min-radius", 1),
        ("--v9-2-scribble-max-radius", 20),
        ("--v9-2-scribble-width-fraction", 0.15),
    )
    for values in mandatory:
        _append_flag(forwarded, *values)
    forwarded.extend(
        [
            "--validation-only",
            "--load-decoder-from-prompt-checkpoint",
            "--validation-complete-prompt-roi",
            "--validation-roi-auto-expand",
            "--v9-2-adaptive-scribble",
            "--no-v9-eval-random-prompt-modes",
            "--no-v9-decoder-feedback",
            "--amp",
        ]
    )
    os.environ["SAMPROMPT_RUN_STAMP"] = str(custom.run_name)
    sys.argv = [sys.argv[0], *forwarded]
    run_dir = Path(custom.eval_out_dir) / str(custom.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "evaluation_protocol.json").write_text(
        json.dumps({key: value for key, value in audit.items() if key != "loaded_task_keys"}, indent=2),
        encoding="utf-8",
    )
    entry.main()
    loaded = [tuple(value) for value in audit["loaded_task_keys"]]
    expected = [_task_key(row) for row in tasks]
    if set(loaded) != set(expected):
        raise RuntimeError(
            f"exact prompt injection coverage mismatch: loaded={len(loaded)} expected={len(expected)}"
        )
    audit["loaded_task_count"] = len(set(loaded))
    audit["prompt_load_call_count"] = len(loaded)
    audit["loaded_task_keys"] = [list(value) for value in sorted(set(loaded))]
    thresholds = [float(row["threshold"]) for row in audit["dynamic_threshold_rows"]]
    if len(thresholds) != len(expected):
        raise RuntimeError(
            f"dynamic threshold coverage mismatch: calibrated={len(thresholds)} expected={len(expected)}"
        )
    audit["dynamic_threshold_summary"] = {
        "count": len(thresholds),
        "minimum": float(min(thresholds)),
        "maximum": float(max(thresholds)),
        "mean": float(sum(thresholds) / len(thresholds)),
        "fallback_count": sum(
            str(row.get("source", "")).startswith("fallback")
            for row in audit["dynamic_threshold_rows"]
        ),
    }
    (run_dir / "exact_prompt_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
