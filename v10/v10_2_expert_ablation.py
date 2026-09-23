"""Read-only fixed-protocol ablation for v10.2 morphology routes."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from uuid import uuid4

ROUTES = ("router", "shared", "expert:0", "expert:1", "expert:2", "expert:3")


def route_modes(values):
    modes = ROUTES if values is None else tuple(str(value) for value in values)
    if not modes or len(set(modes)) != len(modes) or any(mode not in ROUTES for mode in modes):
        raise ValueError(f"ablation modes must be unique members of {ROUTES}")
    return modes


def assert_fixed_validation(metrics):
    protocol = metrics.get("validation_protocol") or {}
    expected = {
        "selected_tasks": 271,
        "covered_classes": 156,
        "requested_classes": 156,
        "metric_space": "raw_nifti_full_volume",
        "all_nonempty_classes_covered": True,
    }
    mismatches = {
        key: (protocol.get(key), value)
        for key, value in expected.items()
        if protocol.get(key) != value
    }
    if mismatches:
        raise ValueError(f"v10.2 fixed validation protocol mismatch: {mismatches}")


def assert_eval_args(args):
    expected = {
        "validation_max_tasks": 0,
        "val_foreground_mode": "scribble",
        "val_background_mode": "scribble",
        "mask_threshold": 0.60,
        "anchor_coarse_crop": False,
    }
    mismatches = {
        key: (getattr(args, key, None), value)
        for key, value in expected.items()
        if getattr(args, key, None) != value
    }
    if Path(str(getattr(args, "multidataset_validation_json", ""))).name != (
        "v9_2_extended_validation_4case_v2.json"
    ):
        mismatches["multidataset_validation_json"] = getattr(
            args, "multidataset_validation_json", None
        )
    if mismatches:
        raise ValueError(f"expert ablation requires unchanged validation settings: {mismatches}")


def run_routes(modes, encoder, evaluate, *, reset_rng, write):
    """Evaluate each route under identical RNG, restoring the model on failure."""
    selected = route_modes(modes)
    original = encoder.route_override
    completed = []
    try:
        for route in selected:
            encoder.route_override = route
            reset_rng()
            metrics = evaluate()
            assert_fixed_validation(metrics)
            write(route, metrics)
            completed.append((route, metrics))
    finally:
        encoder.route_override = original
    return completed


def _json_default(value):
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def save_json_once(path, payload):
    """Publish a complete result atomically, never replacing an earlier run."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    temporary = destination.with_name(destination.name + "." + uuid4().hex + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
            encoding="utf-8",
        )
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    v10_root = Path(__file__).resolve().parent
    v9_root = v10_root.parent
    for root in (v10_root, v9_root):
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

    import numpy as np
    import torch
    import finetune_totalseg_amos_magic_v10_2_joint as joint
    from utils.distributed_runtime import parse_gpu_ids

    parser = joint.parser()
    parser.description = "Evaluation-only v10.2 morphology expert ablation"
    parser.add_argument("--ablation-output", required=True)
    parser.add_argument("--ablation-modes", nargs="+", default=None)
    args = parser.parse_args()
    modes = route_modes(args.ablation_modes)
    assert_eval_args(args)
    gpu_ids = parse_gpu_ids(args.gpu)
    if len(gpu_ids) != 1:
        raise ValueError("expert ablation requires exactly one GPU")
    checkpoint = Path(args.resume_checkpoint or args.prompt_generator_checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output = Path(args.ablation_output).resolve()
    for mode in modes:
        result_path = output / ("route_" + mode.replace(":", "_") + ".json")
        if result_path.exists():
            raise FileExistsError(result_path)

    joint.v101._validate_extension_args(args)
    joint._prepare_protocol(args)
    joint._bind_v10_2(args)
    _split, validation = joint._PROTOCOL
    if len(validation["tasks"]) != 271 or len(args.v10_2_source_order) != 8:
        raise ValueError("expert ablation requires the fixed 271-task eight-source protocol")
    args.amp = True
    args.object_score_gate = False
    args.dataset_foreground_mode = args.val_foreground_mode

    # The shared v10 entry point maps --gpu into CUDA_VISIBLE_DEVICES before
    # CUDA initialization. A single selected physical GPU is cuda:0 here.
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    adapter = joint.v10.SAM2MedicalAdapterV10(args, device).to(device)
    prompt = joint.make_prompt_v10_2(args, adapter, device)
    prompt._v10_2_args = args
    state = joint.load_init_v10_2(checkpoint, prompt, adapter.sam.sam_mask_decoder)
    adapter.eval()
    prompt.eval()
    checkpoint_hash = sha256_file(checkpoint)
    protocol_hash = sha256_file(args.multidataset_validation_json)

    def reset_rng():
        seed = int(args.validation_seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def evaluate():
        return joint.evaluate_multidataset_v10_2(
            prompt, adapter, [], [], args, device
        )[2]

    started = time.perf_counter()

    def write(route, metrics):
        result = {
            "route": route,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_epoch": state.get("epoch"),
            "validation_json": str(Path(args.multidataset_validation_json).resolve()),
            "validation_json_sha256": protocol_hash,
            "validation_seed": int(args.validation_seed),
            "gpu": int(gpu_ids[0]),
            "elapsed_from_start_seconds": time.perf_counter() - started,
            "metrics": metrics,
        }
        path = output / ("route_" + route.replace(":", "_") + ".json")
        save_json_once(path, result)
        print(
            f"route={route} selection_dice={metrics['selection']['score']:.6f} "
            f"result={path}", flush=True,
        )

    with torch.no_grad():
        run_routes(modes, prompt.encoder3d, evaluate, reset_rng=reset_rng, write=write)


if __name__ == "__main__":
    main()
