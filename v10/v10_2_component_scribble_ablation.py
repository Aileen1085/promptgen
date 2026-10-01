"""Paired, evaluation-only test of per-3D-component two-plane scribbles.

The fixed CT13 held-out tasks, model, background prompts, and full-volume
metrics remain unchanged. This script never modifies a running trainer.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import shlex
import sys
import time

from v10_2_component_scribble import select_component_plane_slices


PATHOLOGY_CLASSES = {
    "magic": {133, 134, 135, 136, 137, 139},
    "msd_task07": {141},
    "msd_task10": {142},
    "msd_task08": {144},
    "lndb": {159},
    "msd_task06": {160},
    "covid19_20": {161},
    "kits23": {163, 164},
    "lnq2023_lite": {165},
}
CACHE_SALT = "per3d_component_26c_v1"


def pathology_tasks(split: dict, validation: dict) -> list[dict]:
    """Select only already-fixed held-out lesion/abnormal-node case-class tasks."""
    selected = []
    seen = set()
    for task in validation["tasks"]:
        source = str(task["dataset"])
        class_id = int(task["global_class_id"])
        if class_id not in PATHOLOGY_CLASSES.get(source, ()):
            continue
        case_id = str(task["case_id"])
        key = (source, case_id, class_id)
        if key in seen:
            raise ValueError(f"duplicate fixed validation task: {key}")
        seen.add(key)
        source_split = split["datasets"][source]
        val_ids = {str(row["case_id"]) for row in source_split["val"]}
        train_ids = {str(row["case_id"]) for row in source_split["train"]}
        if case_id not in val_ids or case_id in train_ids:
            raise ValueError(f"non-held-out pathology task: {key}")
        selected.append(dict(task))
    if not selected:
        raise ValueError("fixed validation contains no pathology tasks")
    return selected


def parse_launcher_args(command: str) -> list[str]:
    """Read training configuration as data; never execute the launcher."""
    tokens = shlex.split(command)
    for index, token in enumerate(tokens):
        if token.endswith("ct13_training_entry.py"):
            return tokens[index + 1:]
    raise ValueError("launcher does not contain the CT13 entry point")


def memory_safe_eval_args(options: list[str], *, gpu: int) -> list[str]:
    """Keep model/protocol options, changing only evaluation memory scheduling."""
    return list(options) + [
        "--gpu", str(gpu),
        "--sam2-frame-batch-size", "8",
        "--sam2-feature-cache-device", "cpu",
    ]


def _inner_dataset(dataset):
    current = dataset
    seen = set()
    while hasattr(current, "dataset") and id(current) not in seen:
        seen.add(id(current))
        current = current.dataset
    return current


def component_cache_namespace(dataset) -> None:
    """Salt prompt/ROI metadata only; preserve the shared case CT cache root."""
    inner = _inner_dataset(dataset)
    inner.PROMPT_PLANE_CACHE_VERSION = (
        f"{inner.PROMPT_PLANE_CACHE_VERSION}_{CACHE_SALT}"
    )
    inner.PROMPT_CACHE_VERSION = int(inner.PROMPT_CACHE_VERSION) + 1_000_000_000
    if not bool(getattr(inner, "v10_lightweight_roi_metadata", False)):
        raise ValueError("component test requires lightweight ROI metadata")


@contextmanager
def component_selector(enabled: bool):
    import v10_data
    import v9_2_adaptive_scribble

    if not enabled:
        yield
        return
    previous_data = v10_data.select_scribble_plane_slices
    previous_adaptive = v9_2_adaptive_scribble.select_scribble_plane_slices
    try:
        v10_data.select_scribble_plane_slices = select_component_plane_slices
        v9_2_adaptive_scribble.select_scribble_plane_slices = select_component_plane_slices
        yield
    finally:
        v10_data.select_scribble_plane_slices = previous_data
        v9_2_adaptive_scribble.select_scribble_plane_slices = previous_adaptive


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_metrics(metrics: dict) -> dict:
    import math

    return {
        name: float(value)
        for name, value in metrics.items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    }


def _summarize(rows: list[dict]) -> dict:
    import numpy as np

    names = ("dice", "iou", "precision", "recall", "pred_gt_ratio", "mean_slice_nsd_1mm")
    cohorts = {"all": rows, "multi_component": [row for row in rows if row["components"] >= 2]}
    for source in sorted({row["task"]["dataset"] for row in rows}):
        cohorts[source] = [row for row in rows if row["task"]["dataset"] == source]
    result = {}
    for cohort, members in cohorts.items():
        result[cohort] = {"tasks": len(members)}
        for name in names:
            base = [row["baseline"][name] for row in members if name in row["baseline"] and name in row["component"]]
            changed = [row["component"][name] for row in members if name in row["baseline"] and name in row["component"]]
            if base:
                result[cohort][name] = {
                    "baseline": float(np.mean(base)),
                    "component": float(np.mean(changed)),
                    "delta": float(np.mean(np.asarray(changed) - np.asarray(base))),
                    "improved_tasks": int(sum(after > before for before, after in zip(base, changed))),
                }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-launcher", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--max-tasks", type=int, default=0,
                        help="Optional pilot size; 0 evaluates all fixed pathology tasks")
    settings = parser.parse_args()
    if not settings.source_launcher.is_file() or not settings.checkpoint.is_file():
        raise FileNotFoundError("launcher or checkpoint is missing")
    if not settings.validation_json.is_file():
        raise FileNotFoundError(settings.validation_json)
    if settings.max_tasks < 0:
        raise ValueError("max-tasks must be non-negative")
    if "CUDA_VISIBLE_DEVICES" in os.environ and os.environ["CUDA_VISIBLE_DEVICES"] != str(settings.gpu):
        raise ValueError("inherited CUDA_VISIBLE_DEVICES disagrees with --gpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(settings.gpu)

    root = Path(__file__).resolve().parent.parent
    for path in (root / "v10", root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import numpy as np
    from scipy import ndimage
    import torch
    import ct13_training_entry as ct13
    import v9_2_extended_protocol as protocol
    import v10_2_data as data
    import finetune_totalseg_amos_magic_v10_2_joint as joint
    from v10_2_expert_ablation import save_json_once

    launch_text = settings.source_launcher.read_text(encoding="utf-8")
    launch_line = next(line for line in launch_text.splitlines() if "ct13_training_entry.py" in line)
    launch_args = parse_launcher_args(launch_line)
    args = joint.parser().parse_args(memory_safe_eval_args(launch_args, gpu=settings.gpu) + [
        "--multidataset-validation-json", str(settings.validation_json),
    ])
    args.resume_checkpoint = None
    args.dataset_foreground_mode = "scribble"
    if args.val_foreground_mode != "scribble" or args.val_background_mode != "scribble":
        raise ValueError("paired test requires fixed fg/bg scribble modes")
    if args.anchor_coarse_crop or args.patch_scaling or not args.v10_2_adaptive_scribble:
        raise ValueError("paired test requires active no-crop adaptive-scribble protocol")
    if int(args.scribble_num_slices) != 1:
        raise ValueError("paired test expects one slice per axis in the baseline")
    if tuple(json.loads(Path(args.multidataset_split_json).read_text())["source_order"]) != ct13.SOURCE_ORDER:
        raise ValueError("launcher split is not CT13")
    split = json.loads(Path(args.multidataset_split_json).read_text())
    protocol.SOURCE_ORDER = ct13.SOURCE_ORDER
    ct13.install_v10_protocol(joint, data, split)
    joint.v101._validate_extension_args(args)
    joint._prepare_protocol(args)
    joint._bind_v10_2(args)
    split, validation = joint._PROTOCOL
    tasks = pathology_tasks(split, validation)
    if settings.max_tasks:
        tasks = tasks[:settings.max_tasks]
    checkpoint_sha = _sha256(settings.checkpoint)
    validation_sha = _sha256(settings.validation_json)
    split_sha = _sha256(Path(args.multidataset_split_json))
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    settings.output_dir = settings.output_dir.resolve()
    if (settings.output_dir / "summary.json").exists():
        raise FileExistsError(settings.output_dir / "summary.json")

    args.amp = True
    args.object_score_gate = False
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.empty((1,), device=device)
    adapter = joint.v10.SAM2MedicalAdapterV10(args, device).to(device)
    prompt = joint.make_prompt_v10_2(args, adapter, device)
    prompt._v10_2_args = args
    state = joint.load_init_v10_2(settings.checkpoint, prompt, adapter.sam.sam_mask_decoder)
    adapter.eval()
    prompt.eval()
    entries = {
        source: {str(row["case_id"]): row for row in split["datasets"][source]["val"]}
        for source in PATHOLOGY_CLASSES
    }
    previous_factory = joint.v10_memory.make_dataset
    previous_semantic = joint.v10_memory.semantic_text_for_class
    joint.v10_memory.semantic_text_for_class = joint.semantic_text_for_global_class
    rows = []
    started = time.perf_counter()
    try:
        with torch.no_grad():
            for index, task in enumerate(tasks):
                pair_path = settings.output_dir / f"task_{index:03d}.json"
                if pair_path.exists():
                    saved = json.loads(pair_path.read_text(encoding="utf-8"))
                    if (saved["task"] != task or saved["checkpoint_sha256"] != checkpoint_sha
                            or saved["validation_sha256"] != validation_sha):
                        raise ValueError(f"existing paired task has different inputs: {pair_path}")
                    rows.append(saved)
                    continue
                source = str(task["dataset"])
                class_id = int(task["global_class_id"])
                entry = entries[source][str(task["case_id"])]
                seed = int(args.validation_seed) + index
                mode_results = {}
                component_count = None
                for mode in ("baseline", "component"):
                    enabled = mode == "component"
                    with component_selector(enabled):
                        random.seed(seed)
                        np.random.seed(seed)
                        torch.manual_seed(seed)
                        torch.cuda.manual_seed_all(seed)
                        dataset = joint.make_source_dataset(
                            source, [entry], args, joint.v101.make_dataset_v10_1,
                            foreground_mode="scribble",
                        )
                        if enabled:
                            component_cache_namespace(dataset)
                        inner = _inner_dataset(dataset)
                        if component_count is None:
                            target = inner._load_canonical_target(0, class_id)
                            _labels, component_count = ndimage.label(
                                target.astype(bool), structure=np.ones((3, 3, 3), dtype=bool)
                            )
                            del target, _labels
                        fixed = joint.FixedTaskV102Dataset(dataset, [task])
                        joint.v10_memory.make_dataset = lambda *_a, _fixed=fixed, **_k: _fixed
                        source_args = copy.copy(args)
                        source_args.validation_max_tasks = 0
                        source_args.validation_class_dice = {}
                        source_args.validation_class_difficulty = {}
                        _score, _class_dice, metrics = joint._BASE_EVALUATE(
                            prompt, adapter,
                            list(zip(fixed.image_paths, fixed.label_paths)),
                            [class_id], source_args, device,
                        )
                        validation_protocol = metrics["validation_protocol"]
                        if (validation_protocol["selected_tasks"] != 1
                                or validation_protocol["metric_space"] != "raw_nifti_full_volume"):
                            raise ValueError(f"incomplete full-volume task: {task} {mode}")
                        mode_results[mode] = _finite_metrics(metrics["mean"])
                        del fixed, dataset
                        torch.cuda.empty_cache()
                record = {
                    "task": task,
                    "seed": seed,
                    "components": int(component_count),
                    "baseline": mode_results["baseline"],
                    "component": mode_results["component"],
                    "checkpoint_sha256": checkpoint_sha,
                    "validation_sha256": validation_sha,
                }
                save_json_once(pair_path, record)
                rows.append(record)
                print(f"paired={index+1}/{len(tasks)} source={source} class={class_id} "
                      f"components={component_count} "
                      f"dice={record['baseline']['dice']:.5f}->{record['component']['dice']:.5f}",
                      flush=True)
    finally:
        joint.v10_memory.make_dataset = previous_factory
        joint.v10_memory.semantic_text_for_class = previous_semantic
    summary = {
        "checkpoint": str(settings.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_epoch": state.get("epoch"),
        "validation_json": str(settings.validation_json.resolve()),
        "validation_sha256": validation_sha,
        "split_sha256": split_sha,
        "gpu_physical": settings.gpu,
        "eval_frame_batch_size": int(args.sam2_frame_batch_size),
        "eval_feature_cache_device": str(args.sam2_feature_cache_device),
        "metric_space": "raw_nifti_full_volume",
        "protocol": "fixed pathology tasks; identical checkpoint/seed; baseline versus one coronal and one sagittal plane per 26-connected 3D target",
        "tasks": len(rows),
        "elapsed_seconds": time.perf_counter() - started,
        "metrics": _summarize(rows),
    }
    save_json_once(settings.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
