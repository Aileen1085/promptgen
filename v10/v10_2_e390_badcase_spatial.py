"""Paired, read-only E390/E20 spatial diagnosis on fixed validation tasks.

The output contains metrics and small spatial summaries only; no CT or masks
are written. This diagnostic is not a replacement for the 271-task validation.
"""

from __future__ import annotations

from pathlib import Path
import os
import random
import sys
import time


V10_ROOT = Path(__file__).resolve().parent
V9_ROOT = V10_ROOT.parent
for root in (V10_ROOT, V9_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def select_fixed_badcases(tasks, class_ids=(86, 116, 143, 144)):
    """Keep only requested tasks from the supplied fixed validation protocol."""
    requested = {int(value) for value in class_ids}
    if not requested:
        raise ValueError("at least one bad-case class is required")
    chosen = []
    seen = set()
    for row in tasks:
        class_id = int(row["global_class_id"])
        if class_id not in requested:
            continue
        key = (str(row["dataset"]), str(row["case_id"]), class_id)
        if key in seen:
            raise ValueError("duplicate fixed-validation task: {}".format(key))
        seen.add(key)
        chosen.append(dict(row))
    missing = requested - {int(row["global_class_id"]) for row in chosen}
    if missing:
        raise ValueError("missing fixed-validation classes: {}".format(sorted(missing)))
    return chosen


def build_badcase_record(task, full_metrics, roi_spatial, *, seed):
    """Separate official-space metrics from prompt-ROI diagnostic evidence."""
    if full_metrics.get("metric_space") != "raw_nifti_full_volume":
        raise ValueError("bad-case metrics must score the raw full volume")
    return {
        "task": {
            "dataset": str(task["dataset"]),
            "case_id": str(task["case_id"]),
            "global_class_id": int(task["global_class_id"]),
            "class_name": str(task.get("class_name", "")),
        },
        "seed": int(seed),
        "full_volume_metrics": dict(full_metrics),
        "prompt_roi_spatial": dict(roi_spatial),
    }


def main():
    import nibabel as nib
    import numpy as np
    import torch

    import finetune_totalseg_amos_magic_v10_2_joint as joint
    from v10_2_background_spatial import PhysicalSpacingDataset
    from v10_2_sam_finetuning import (
        configure_sam_finetuning,
        load_sam_tuning_from_checkpoint,
    )
    from v10_2_spatial_diagnostic import spatial_summary_for_roi
    from v10.v10_2_expert_ablation import save_json_once, sha256_file
    from utils.distributed_runtime import parse_gpu_ids

    parser = joint.parser()
    parser.description = "Paired E390/E20 fixed-validation bad-case spatial diagnosis"
    parser.add_argument("--diagnostic-output", required=True)
    args = parser.parse_args()
    if args.resume_checkpoint or not args.prompt_generator_checkpoint:
        raise ValueError("pass one model checkpoint without --resume-checkpoint")
    if bool(args.patch_scaling) or bool(args.anchor_coarse_crop):
        raise ValueError("fixed diagnosis requires no patch scaling/coarse crop")
    if str(args.val_foreground_mode) != "scribble" or str(args.val_background_mode) != "scribble":
        raise ValueError("fixed diagnosis requires fg/bg scribble")
    if float(args.mask_threshold) != 0.60:
        raise ValueError("fixed diagnosis requires mask threshold 0.60")
    gpu_ids = parse_gpu_ids(args.gpu)
    if len(gpu_ids) != 1:
        raise ValueError("select one physical GPU")
    output = Path(args.diagnostic_output)
    if output.exists():
        raise FileExistsError(output)
    checkpoint = Path(args.prompt_generator_checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
    joint.v101._validate_extension_args(args)
    joint._prepare_protocol(args)
    joint._bind_v10_2(args)
    split, validation = joint._PROTOCOL
    tasks = select_fixed_badcases(validation["tasks"])
    args.amp = True
    args.object_score_gate = False
    args.dataset_foreground_mode = "scribble"
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    adapter = joint.v10.SAM2MedicalAdapterV10(args, device).to(device)
    tuning = configure_sam_finetuning(adapter.sam, "encoder_lora_full_decoder")
    prompt = joint.make_prompt_v10_2(args, adapter, device)
    prompt._v10_2_args = args
    state = joint.load_init_v10_2(checkpoint, prompt, adapter.sam.sam_mask_decoder)
    tuning_loaded = load_sam_tuning_from_checkpoint(adapter.sam, tuning, state)
    protocol = (state.get("validation_metrics") or {}).get("validation_protocol") or {}
    if (
        int(protocol.get("selected_tasks", 0)) != 271
        or protocol.get("metric_space") != "raw_nifti_full_volume"
        or Path(str(protocol.get("fixed_task_json", ""))).name
        != Path(args.multidataset_validation_json).name
    ):
        raise ValueError("checkpoint and fixed 271-task validation protocol differ")
    prompt.eval()
    adapter.eval()

    source_entries = {
        source: {
            str(row["case_id"]): row
            for row in contents["val"]
        }
        for source, contents in split["datasets"].items()
    }
    original_semantic = joint.v10_memory.semantic_text_for_class
    joint.v10_memory.semantic_text_for_class = joint.semantic_text_for_global_class
    rows = []
    started = time.perf_counter()
    try:
        with torch.no_grad():
            for index, task in enumerate(tasks):
                seed = 2027 + index
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                source = str(task["dataset"])
                entry = source_entries[source][str(task["case_id"])]
                dataset = joint.make_source_dataset(
                    source, [entry], args, joint.v101.make_dataset_v10_1,
                    foreground_mode="scribble",
                )
                dataset.v10_foreground_mode = "scribble"
                if hasattr(dataset, "dataset"):
                    dataset.dataset.v10_foreground_mode = "scribble"
                fixed = joint.FixedTaskV102Dataset(dataset, [{**task, **entry}])
                fixed = PhysicalSpacingDataset(fixed)
                class_id = int(task["global_class_id"])
                video, bundle, _points, target, _ = fixed.get_item_for_class(0, class_id)
                if float(target.sum()) <= 0:
                    raise RuntimeError("fixed validation task has empty target")
                frame_indices = np.asarray(fixed.last_frame_indices, dtype=np.int64).copy()
                raw_target = fixed._load_total_target(fixed.label_paths[0], class_id)
                canonical = nib.as_closest_canonical(nib.load(str(fixed.image_paths[0])))
                zoom_x, zoom_y, zoom_z = canonical.header.get_zooms()[:3]
                raw_spacing = (float(zoom_z), float(zoom_x), float(zoom_y))
                bundle["v10_forced_background_mode"] = "scribble"
                tokens, anchor = joint.v10_memory.generate_case_3d_tokens(
                    prompt, adapter, video, bundle, class_id, args, device,
                )
                logits, _ = joint.v10_memory.memory_sliding_window_decode(
                    adapter, video, bundle, tokens, anchor, args, device,
                    training=False,
                )
                full_metrics = joint.v10_memory.full_volume_binary_metrics_from_logits(
                    logits, raw_target, frame_indices=frame_indices,
                    threshold=float(args.mask_threshold), spacing_dhw=raw_spacing,
                    full_side=int(video.shape[-1]),
                )
                roi_spatial = spatial_summary_for_roi(
                    logits, target, tokens["_aligned_prompt"],
                    spacing_dhw=bundle["physical_spacing_dhw"],
                    threshold=float(args.mask_threshold),
                )
                rows.append(build_badcase_record(task, full_metrics, roi_spatial, seed=seed))
                print("diagnosed={}/{} source={} class={} dice={:.5f}".format(
                    index + 1, len(tasks), source, class_id, full_metrics["dice"]
                ), flush=True)
                del dataset, fixed, video, bundle, target, tokens, logits
                torch.cuda.empty_cache()
    finally:
        joint.v10_memory.semantic_text_for_class = original_semantic
    save_json_once(output, {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_epoch": int(state.get("epoch", -1)),
        "sam_tuning_state_loaded": bool(tuning_loaded),
        "validation_json": str(args.multidataset_validation_json),
        "validation_sha256": sha256_file(args.multidataset_validation_json),
        "metric_scope": "fixed validation tasks; raw full-volume score and prompt-ROI spatial diagnosis",
        "elapsed_seconds": time.perf_counter() - started,
        "rows": rows,
    })
    print("diagnostic_result={} tasks={}".format(output, len(rows)), flush=True)


if __name__ == "__main__":
    main()
