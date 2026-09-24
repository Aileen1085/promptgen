"""Fixed-protocol full-volume evaluation for the v10.2 spatial branch."""

from __future__ import annotations

import json
from pathlib import Path
import random
import sys
import time

V10_ROOT = Path(__file__).resolve().parent
V9_ROOT = V10_ROOT.parent
for root in (V10_ROOT, V9_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from v10.v10_2_expert_ablation import (
    assert_eval_args,
    assert_fixed_validation,
    save_json_once,
    sha256_file,
)


def main() -> None:
    v10_root = Path(__file__).resolve().parent
    v9_root = v10_root.parent
    for root in (v10_root, v9_root):
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

    import numpy as np
    import torch
    import finetune_totalseg_amos_magic_v10_2_joint as joint
    from v10_2_background_spatial import (
        CrossViewPromptTokenGeneratorV102BackgroundSpatial,
        PhysicalSpacingDataset,
        bind_physical_spacing_source_factory,
        generate_case_3d_tokens_with_physical_spacing,
    )
    from utils.distributed_runtime import parse_gpu_ids

    parser = joint.parser()
    parser.description = "Fixed 271-task v10.2 physical-distance evaluation"
    parser.add_argument("--evaluation-output", required=True)
    parser.add_argument(
        "--v10-2-physical-distance-hidden-dim", type=int, default=16
    )
    parser.add_argument(
        "--v10-2-physical-distance-clip-mm", type=float, default=80.0
    )
    parser.add_argument(
        "--route-mode",
        choices=("router", "shared", "expert:0", "expert:1", "expert:2", "expert:3"),
        default="router",
    )
    args = parser.parse_args()
    assert_eval_args(args)
    gpu_ids = parse_gpu_ids(args.gpu)
    if len(gpu_ids) != 1:
        raise ValueError("background-spatial evaluation requires exactly one GPU")
    checkpoint = Path(
        args.resume_checkpoint or args.prompt_generator_checkpoint
    ).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output = Path(args.evaluation_output).resolve()
    if output.exists():
        raise FileExistsError(output)

    joint.v101._validate_extension_args(args)
    joint._prepare_protocol(args)
    joint._bind_v10_2(args)
    bind_physical_spacing_source_factory(joint)
    base_generate = joint.v10_memory.generate_case_3d_tokens

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

    joint.v10_memory.generate_case_3d_tokens = spatial_generate
    _split, validation = joint._PROTOCOL
    if len(validation["tasks"]) != 271 or len(args.v10_2_source_order) != 8:
        raise ValueError("evaluation requires the fixed 271-task eight-source protocol")
    args.amp = True
    args.object_score_gate = False
    args.dataset_foreground_mode = "scribble"

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    adapter = joint.v10.SAM2MedicalAdapterV10(args, device).to(device)
    cls = CrossViewPromptTokenGeneratorV102BackgroundSpatial
    cls.DEFAULT_PHYSICAL_DISTANCE_HIDDEN_DIM = int(
        args.v10_2_physical_distance_hidden_dim
    )
    cls.DEFAULT_PHYSICAL_DISTANCE_CLIP_MM = float(
        args.v10_2_physical_distance_clip_mm
    )
    original = joint.v101.CrossViewPromptTokenGeneratorV101
    joint.v101.CrossViewPromptTokenGeneratorV101 = cls
    try:
        prompt = joint.make_prompt_v10_2(args, adapter, device)
    finally:
        joint.v101.CrossViewPromptTokenGeneratorV101 = original
    prompt._v10_2_args = args
    state = joint.load_init_v10_2(
        checkpoint, prompt, adapter.sam.sam_mask_decoder
    )
    prompt.encoder3d.route_override = str(args.route_mode)
    prompt.eval()
    adapter.eval()

    seed = int(args.validation_seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    started = time.perf_counter()
    with torch.no_grad():
        metrics = joint.evaluate_multidataset_v10_2(
            prompt, adapter, [], [], args, device
        )[2]
    assert_fixed_validation(metrics)
    result = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_epoch": state.get("epoch"),
        "validation_json": str(Path(args.multidataset_validation_json).resolve()),
        "validation_json_sha256": sha256_file(args.multidataset_validation_json),
        "validation_seed": seed,
        "threshold": float(args.mask_threshold),
        "route": str(args.route_mode),
        "distance_hidden_dim": int(args.v10_2_physical_distance_hidden_dim),
        "distance_clip_mm": float(args.v10_2_physical_distance_clip_mm),
        "elapsed_seconds": time.perf_counter() - started,
        "metrics": metrics,
    }
    save_json_once(output, result)
    print(json.dumps({
        "output": str(output),
        "selection_dice": metrics["selection"]["score"],
        "precision": metrics["mean"].get("precision"),
        "recall": metrics["mean"].get("recall"),
        "pred_gt_ratio": metrics["mean"].get("pred_gt_ratio"),
        "mean_slice_nsd_1mm": metrics["mean"].get("mean_slice_nsd_1mm"),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
