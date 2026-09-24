"""Training-only task selection and counterfactual route ranking for v10.2.

This module deliberately does not import the validation evaluator: its inputs
must come from the training split, and all five counterfactuals use one task.
"""

from __future__ import annotations

import hashlib
import json
import os


COUNTERFACTUAL_ROUTES = (
    "shared", "expert:0", "expert:1", "expert:2", "expert:3"
)


def _path(value):
    return os.path.normcase(os.path.normpath(str(value))) if value else ""


def _task_key(row):
    return str(row["dataset"]), str(row["case_id"]), int(row["global_class_id"])


def _rank(seed, row):
    identity = "{}|{}|{}|{}".format(seed, row["dataset"], row["case_id"],
                                   row["global_class_id"])
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def select_training_tasks(split, validation, *, quotas, hard_classes, seed):
    """Choose a fixed train-only subset and fail closed on split overlap."""
    if not quotas or any(int(count) <= 0 for count in quotas.values()):
        raise ValueError("positive source quotas are required")
    datasets = split["datasets"]
    validation_cases = set()
    validation_paths = set()
    for source, contents in datasets.items():
        for row in contents.get("val", ()):  # split-level held-out cases
            validation_cases.add((str(source), str(row["case_id"])))
            validation_paths.update(filter(None, (_path(row.get("image")),
                                                   _path(row.get("label")))))
    for row in validation.get("tasks", ()):
        validation_cases.add((str(row["dataset"]), str(row["case_id"])))
        validation_paths.update(filter(None, (_path(row.get("image")),
                                               _path(row.get("label")))))

    selected = []
    for source, quota in quotas.items():
        if source not in datasets:
            raise ValueError("missing training source: {}".format(source))
        candidates = []
        for entry in datasets[source]["train"]:
            identity = (str(source), str(entry["case_id"]))
            paths = {_path(entry.get("image")), _path(entry.get("label"))} - {""}
            if identity in validation_cases or paths & validation_paths:
                raise ValueError("training candidate overlaps validation: {}".format(identity))
            for class_id in entry["classes"]:
                candidates.append({
                    "dataset": str(source),
                    "case_id": str(entry["case_id"]),
                    "image": str(entry["image"]),
                    "label": str(entry["label"]),
                    "global_class_id": int(class_id),
                })
        candidates.sort(key=lambda row: (_rank(seed, row), _task_key(row)))
        chosen = []
        if source == "totalseg":
            for class_id in hard_classes:
                match = next((row for row in candidates
                              if row["global_class_id"] == int(class_id)), None)
                if match is None:
                    raise ValueError("hard class {} absent from training".format(class_id))
                if _task_key(match) not in {_task_key(row) for row in chosen}:
                    chosen.append(match)
        if len(chosen) > int(quota):
            raise ValueError("quota cannot cover required hard classes")
        for row in candidates:
            if len(chosen) >= int(quota):
                break
            if _task_key(row) not in {_task_key(item) for item in chosen}:
                chosen.append(row)
        if len(chosen) < int(quota):
            raise ValueError("insufficient training tasks for {}".format(source))
        selected.extend(chosen)
    return selected


def subset_sha256(tasks):
    payload = json.dumps(tasks, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def evaluate_counterfactual_routes(encoder, evaluate):
    """Enumerate all five routes, restoring override even after failure."""
    original = encoder.route_override
    scores = {}
    try:
        for route in COUNTERFACTUAL_ROUTES:
            encoder.route_override = route
            scores[route] = evaluate()
    finally:
        encoder.route_override = original
    return scores


def final_mask_loss(logits, target):
    """Score the frozen SAM2 output using BCE plus soft per-frame Dice."""
    import torch
    import torch.nn.functional as F

    if tuple(logits.shape) != tuple(target.shape) or logits.ndim != 4:
        raise ValueError("final mask logits and target must share [D,1,H,W]")
    positive = target.float().sum().clamp_min(1.0)
    pos_weight = ((target.numel() - positive) / positive).clamp(1.0, 6.0)
    bce_sum = logits.new_zeros((), dtype=torch.float32)
    dice_sum = logits.new_zeros((), dtype=torch.float32)
    for start in range(0, int(logits.shape[0]), 8):
        current = logits[start:start + 8].float()
        truth = target[start:start + 8].float()
        bce_sum = bce_sum + F.binary_cross_entropy_with_logits(
            current, truth, pos_weight=pos_weight, reduction="sum"
        )
        probability = current.sigmoid()
        intersection = (probability * truth).sum(dim=(1, 2, 3))
        denominator = probability.sum(dim=(1, 2, 3)) + truth.sum(dim=(1, 2, 3))
        dice_sum = dice_sum + ((2.0 * intersection + 1e-6)
                               / (denominator + 1e-6)).sum()
    return float((bce_sum / target.numel()
                  + 1.0 - dice_sum / logits.shape[0]).item())


def auxiliary_route_losses(coarse_logits, target):
    """Rank the five low-resolution head channels against one task target."""
    import torch
    import torch.nn.functional as F

    if coarse_logits.ndim != 5 or coarse_logits.shape[0] != 1 or coarse_logits.shape[1] != 5:
        raise ValueError("auxiliary logits must have shape [1,5,D,H,W]")
    if target.ndim != 4 or target.shape[1] != 1:
        raise ValueError("target must have shape [D,1,H,W]")
    truth = target.permute(1, 0, 2, 3).unsqueeze(0).float()
    truth = F.interpolate(truth, size=coarse_logits.shape[-3:], mode="nearest")
    result = {}
    for index, route in enumerate(COUNTERFACTUAL_ROUTES):
        logit = coarse_logits[:, index:index + 1].float()
        bce = F.binary_cross_entropy_with_logits(logit, truth)
        probability = torch.sigmoid(logit)
        dice = 1.0 - (2.0 * (probability * truth).sum() + 1e-6) / (
            probability.sum() + truth.sum() + 1e-6
        )
        result[route] = float((bce + dice).item())
    return result


def router_route_probabilities(probabilities):
    """Map the unforced router's five probabilities to route names."""
    import torch

    if probabilities.ndim != 2 or probabilities.shape[1] != 5:
        raise ValueError("router probabilities must have shape [N,5]")
    if not bool(torch.isfinite(probabilities).all()) or bool((probabilities < 0).any()):
        raise ValueError("router probabilities must be finite and nonnegative")
    means = probabilities.float().mean(dim=0).tolist()
    if abs(sum(means) - 1.0) > 1e-3:
        raise ValueError("router probabilities must sum to one")
    return dict(zip(COUNTERFACTUAL_ROUTES, map(float, means)))


def prompt_regions(depth, *, chunk_size, halo, full_limit):
    """Mirror the inference prompt generator's core and halo schedule."""
    if min(int(depth), int(chunk_size), int(full_limit)) <= 0 or int(halo) < 0:
        raise ValueError("invalid prompt chunk geometry")
    cores = [(0, int(depth))] if depth <= full_limit else [
        (start, min(int(depth), start + int(chunk_size)))
        for start in range(0, int(depth), int(chunk_size))
    ]
    return [
        (start, end, max(0, start - int(halo)), min(int(depth), end + int(halo)))
        for start, end in cores
    ]


def aggregate_prompt_diagnostics(records, target):
    """Weight each halo observation by its unique core's frame count."""
    import torch

    if not records:
        raise ValueError("at least one prompt chunk is required")
    total = sum(int(core_end) - int(core_start)
                for core_start, core_end, _, _, _, _ in records)
    if total != int(target.shape[0]):
        raise ValueError("prompt cores must cover the complete target")
    weighted_probabilities = None
    weighted_auxiliary = {route: 0.0 for route in COUNTERFACTUAL_ROUTES}
    for core_start, core_end, region_start, region_end, probabilities, logits in records:
        weight = (int(core_end) - int(core_start)) / float(total)
        if not (0 <= region_start <= core_start < core_end <= region_end <= total):
            raise ValueError("prompt halo region is invalid")
        current = probabilities.detach().float().mean(dim=0, keepdim=True)
        weighted_probabilities = (current * weight if weighted_probabilities is None
                                  else weighted_probabilities + current * weight)
        losses = auxiliary_route_losses(logits, target[region_start:region_end])
        for route in COUNTERFACTUAL_ROUTES:
            weighted_auxiliary[route] += weight * losses[route]
    return weighted_probabilities, weighted_auxiliary


def run_frozen_model_route(prompt, adapter, video, bundle, class_id, args,
                           device, target, *, generate, decode):
    """Capture every prompt chunk, then decode all ROI frames with frozen SAM2."""
    regions = prompt_regions(
        int(video.shape[0]),
        chunk_size=max(1, int(args.prompt_depth_chunk_size)),
        halo=max(0, int(args.prompt_depth_halo)),
        full_limit=max(1, int(args.prompt_full_depth_limit)),
    )
    captured = []
    original = prompt.forward_case_3d_tokens
    had_instance_override = "forward_case_3d_tokens" in prompt.__dict__
    previous_override = prompt.__dict__.get("forward_case_3d_tokens")

    def capture(*call_args, **call_kwargs):
        outputs = original(*call_args, **call_kwargs)
        if len(captured) >= len(regions):
            raise RuntimeError("prompt generator emitted too many chunks")
        probabilities = outputs["morphology_route_probabilities"].detach().cpu()
        auxiliary = outputs["morphology_counterfactual_logits"].detach().cpu()
        captured.append((*regions[len(captured)], probabilities, auxiliary))
        return outputs

    try:
        prompt.forward_case_3d_tokens = capture
        tokens, anchor = generate(
            prompt, adapter, video, bundle, class_id, args, device
        )
    finally:
        if had_instance_override:
            prompt.forward_case_3d_tokens = previous_override
        else:
            delattr(prompt, "forward_case_3d_tokens")
    if len(captured) != len(regions):
        raise RuntimeError("prompt chunk count differs from full-ROI schedule")
    probabilities, auxiliary_losses = aggregate_prompt_diagnostics(captured, target)
    logits, _ = decode(
        adapter, video, bundle, tokens, anchor, args, device, training=False
    )
    if tuple(logits.shape) != tuple(target.shape):
        raise ValueError("decoded final mask shape differs from target")
    return logits.detach(), probabilities, auxiliary_losses


def evaluate_task_routes(encoder, run_route, target):
    """Score one unchanged prompt/target under router and all forced routes."""
    original = encoder.route_override
    try:
        encoder.route_override = "router"
        router_logits, probabilities, auxiliary_logits = run_route()
        router_loss = final_mask_loss(router_logits, target)
        router_probabilities = router_route_probabilities(probabilities)
        auxiliary_losses = (
            dict(auxiliary_logits) if isinstance(auxiliary_logits, dict)
            else auxiliary_route_losses(auxiliary_logits, target)
        )

        def evaluate_forced():
            logits, _probabilities, _auxiliary_logits = run_route()
            return final_mask_loss(logits, target)

        losses = evaluate_counterfactual_routes(encoder, evaluate_forced)
        return {
            "router_final_mask_loss": router_loss,
            "final_mask_losses": losses,
            "router_probabilities": router_probabilities,
            "auxiliary_losses": auxiliary_losses,
            "ranking": ranking_diagnostics(
                final_mask_losses=losses,
                router_probabilities=router_probabilities,
                auxiliary_losses=auxiliary_losses,
            ),
        }
    finally:
        encoder.route_override = original


def _ranks(values, *, descending=False):
    ordered = sorted(values, key=lambda key: (values[key], key), reverse=descending)
    result = {}
    for index, key in enumerate(ordered):
        result[key] = index + 1
    return result


def _spearman(first, second):
    mean = (len(first) + 1) / 2.0
    numerator = sum((first[key] - mean) * (second[key] - mean) for key in first)
    left = sum((value - mean) ** 2 for value in first.values())
    right = sum((value - mean) ** 2 for value in second.values())
    return numerator / (left * right) ** 0.5 if left and right else 0.0


def ranking_diagnostics(*, final_mask_losses, router_probabilities,
                        auxiliary_losses):
    routes = set(COUNTERFACTUAL_ROUTES)
    if any(set(rows) != routes for rows in
           (final_mask_losses, router_probabilities, auxiliary_losses)):
        raise ValueError("all five counterfactual routes are required")
    final_ranks = _ranks(final_mask_losses)
    router_ranks = _ranks(router_probabilities, descending=True)
    auxiliary_ranks = _ranks(auxiliary_losses)
    final_best = min(final_ranks, key=final_ranks.get)
    router_best = min(router_ranks, key=router_ranks.get)
    auxiliary_best = min(auxiliary_ranks, key=auxiliary_ranks.get)
    return {
        "final_mask_best": final_best,
        "router_best": router_best,
        "auxiliary_best": auxiliary_best,
        "router_top1_agreement": float(router_best == final_best),
        "auxiliary_top1_agreement": float(auxiliary_best == final_best),
        "router_rank_correlation": _spearman(final_ranks, router_ranks),
        "auxiliary_rank_correlation": _spearman(final_ranks, auxiliary_ranks),
        "oracle_gain_vs_router_top1": float(final_mask_losses[router_best]
                                             - final_mask_losses[final_best]),
    }


def summarize_results(results, *, hard_classes):
    """Aggregate task-level route evidence without selecting by validation."""
    if not results:
        raise ValueError("no completed diagnostic tasks")

    def summary(rows):
        metrics = ("router_top1_agreement", "auxiliary_top1_agreement",
                   "router_rank_correlation", "auxiliary_rank_correlation")
        result = {"tasks": len(rows), "oracle_gain_mean": sum(
            float(row["ranking"]["oracle_gain_vs_router_top1"])
            for row in rows) / len(rows)}
        for name in metrics:
            result[name + "_mean"] = sum(
                float(row["ranking"][name]) for row in rows
            ) / len(rows)
        return result

    sources = sorted({str(row["dataset"]) for row in results})
    per_source = {
        source: summary([row for row in results if row["dataset"] == source])
        for source in sources
    }
    per_hard_class = {}
    for class_id in hard_classes:
        rows = [row for row in results if int(row["global_class_id"]) == int(class_id)]
        if rows:
            per_hard_class[str(class_id)] = summary(rows)
    distributions = {}
    for label, key in (("router_selected_distribution", "router_best"),
                       ("final_mask_oracle_distribution", "final_mask_best"),
                       ("auxiliary_selected_distribution", "auxiliary_best")):
        distributions[label] = {
            route: sum(row["ranking"][key] == route for row in results)
            for route in COUNTERFACTUAL_ROUTES
        }
    return {"overall": summary(results), "per_source": per_source,
            "per_hard_class": per_hard_class, **distributions}


def main():
    """Run a read-only final-mask route diagnosis on hashed training tasks."""
    from pathlib import Path
    import random
    import sys
    import time

    import numpy as np
    import torch

    v10_root = Path(__file__).resolve().parent
    v9_root = v10_root.parent
    for root in (v10_root, v9_root):
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
    import finetune_totalseg_amos_magic_v10_2_joint as joint
    from v10.v10_2_expert_ablation import assert_eval_args, save_json_once, sha256_file
    from utils.distributed_runtime import parse_gpu_ids

    parser = joint.parser()
    parser.description = "Training-only frozen final-mask v10.2 route diagnosis"
    parser.add_argument("--diagnostic-output", required=True)
    args = parser.parse_args()
    assert_eval_args(args)
    if Path(args.multidataset_split_json).name != "v9_2_extended_split_4case_v2.json":
        raise ValueError("diagnosis requires the fixed eight-source split")
    if args.resume_checkpoint or not args.prompt_generator_checkpoint:
        raise ValueError("diagnosis requires the original model-only Epoch390 checkpoint")
    checkpoint = Path(args.prompt_generator_checkpoint)
    if sha256_file(checkpoint) != (
        "0c4efc6ecaf50b43b76d6ca5638511f44719806b7d81a6ab11e0ec1ab3807a46"
    ):
        raise ValueError("Epoch390 checkpoint hash mismatch")
    if sha256_file(args.multidataset_validation_json) != (
        "06cc6cde333b602343cc4d77052283784e395ae59189c813c8cea7528a7dc717"
    ):
        raise ValueError("fixed validation exclusion list hash mismatch")
    output = Path(args.diagnostic_output)
    if output.exists():
        raise FileExistsError(output)
    gpu_ids = parse_gpu_ids(args.gpu)
    if len(gpu_ids) != 1:
        raise ValueError("diagnosis requires one explicitly selected GPU")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])

    joint.v101._validate_extension_args(args)
    joint._prepare_protocol(args)
    joint._bind_v10_2(args)
    split, validation = joint._PROTOCOL
    quotas = {"totalseg": 6, "amos": 4, "msd_task10": 4}
    hard_classes = (69, 38, 116)
    tasks = select_training_tasks(
        split, validation, quotas=quotas, hard_classes=hard_classes, seed=2027
    )
    subset_hash = subset_sha256(tasks)
    args.amp = True
    args.object_score_gate = False
    args.dataset_foreground_mode = "scribble"
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    adapter = joint.v10.SAM2MedicalAdapterV10(args, device).to(device)
    prompt = joint.make_prompt_v10_2(args, adapter, device)
    prompt._v10_2_args = args
    state = joint.load_init_v10_2(
        checkpoint, prompt, adapter.sam.sam_mask_decoder
    )
    if int(state.get("epoch", -1)) != 390:
        raise ValueError("diagnosis checkpoint is not Epoch390")
    prompt.eval()
    adapter.eval()
    started = time.perf_counter()
    completed = []
    with torch.no_grad():
        for task_index, task in enumerate(tasks):
            seed = 2027 + task_index
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            source = str(task["dataset"])
            entry = next(row for row in split["datasets"][source]["train"]
                         if str(row["case_id"]) == str(task["case_id"]))
            dataset = joint.make_source_dataset(
                source, [entry], args, joint.v101.make_dataset_v10_1,
                foreground_mode="scribble",
            )
            dataset.v10_foreground_mode = "scribble"
            class_id = int(task["global_class_id"])
            video, bundle, _points, target, _ = dataset.get_item_for_class(0, class_id)
            if float(target.sum()) <= 0:
                raise RuntimeError("selected training task has no target: {}".format(task))
            bundle["v10_forced_background_mode"] = "scribble"

            def run_route():
                return run_frozen_model_route(
                    prompt, adapter, video, bundle, class_id, args, device,
                    target,
                    generate=joint.v10_memory.generate_case_3d_tokens,
                    decode=joint.v10_memory.memory_sliding_window_decode,
                )

            result = evaluate_task_routes(prompt.encoder3d, run_route, target)
            completed.append({
                "dataset": source,
                "case_id": task["case_id"],
                "global_class_id": class_id,
                "seed": seed,
                **result,
            })
            print("task={}/{} source={} class={} router_agree={}".format(
                task_index + 1, len(tasks), source, class_id,
                result["ranking"]["router_top1_agreement"]), flush=True)
            del dataset, video, bundle, target, result
            torch.cuda.empty_cache()
    payload = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_epoch": 390,
        "split_sha256": sha256_file(args.multidataset_split_json),
        "validation_exclusion_sha256": sha256_file(args.multidataset_validation_json),
        "subset_sha256": subset_hash,
        "subset_seed": 2027,
        "subset_quotas": quotas,
        "hard_classes": list(hard_classes),
        "gpu": int(gpu_ids[0]),
        "threshold": float(args.mask_threshold),
        "prompt_modes": ["scribble", "scribble"],
        "elapsed_seconds": time.perf_counter() - started,
        "tasks": completed,
        "summary": summarize_results(completed, hard_classes=hard_classes),
    }
    save_json_once(output, payload)
    print("diagnostic={} subset_sha256={}".format(output, subset_hash), flush=True)


if __name__ == "__main__":
    main()
