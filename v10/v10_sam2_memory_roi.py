from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
import gc
import math
import random
import time
import weakref

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from utils.distributed_runtime import all_gather_flat, context_from_env
from v10_data import make_dataset, semantic_text_for_class
from v10_metrics import full_volume_binary_metrics_from_logits
from v10_2_multidataset_runtime import memory_commit_positions


def _amp(args, device):
    amp_name = str(getattr(args, "amp_dtype", "float16")).lower()
    amp_dtype = torch.bfloat16 if amp_name in ("bfloat16", "bf16") else torch.float16
    return (
        torch.autocast("cuda", dtype=amp_dtype)
        if bool(args.amp) and device.type == "cuda" else nullcontext()
    )


def encode_case_image_views(adapter, video_cpu, args, device, training: bool):
    """Run SAM2's image encoder once and cache two logical views.

    The BCHW feature maps are the PromptGen view. The memory view is rebuilt
    as flattened HW,B,C tensors from the same storage, so feature tensors are
    not duplicated. Frozen encoders preserve the legacy detached CPU/GPU cache
    path. A trainable encoder keeps its graph on GPU and checkpoints each small
    frame chunk so gradients reach LoRA or thawed Hiera weights.
    """
    sam = adapter.sam
    train_image_encoder = bool(training) and any(
        parameter.requires_grad
        for name, parameter in sam.named_parameters()
        if name.startswith("image_encoder.") or ".image_encoder." in name
    )
    frame_batch = max(1, int(
        getattr(args, "sam_encoder_frame_batch_size", 4)
        if train_image_encoder else args.sam2_frame_batch_size
    ))
    requested_cache_device = str(
        getattr(args, "sam2_feature_cache_device", "auto")
    ).lower()
    if train_image_encoder:
        cache_device = device
    elif requested_cache_device == "auto":
        limit_name = (
            "sam2_feature_cache_gpu_max_frames"
            if training else "sam2_feature_cache_gpu_max_eval_frames"
        )
        default_limit = 128 if training else 384
        gpu_limit = max(0, int(getattr(args, limit_name, default_limit)))
        cache_device = (
            device if device.type == "cuda" and int(video_cpu.shape[0]) <= gpu_limit
            else torch.device("cpu")
        )
    elif requested_cache_device == "cuda":
        cache_device = device
    else:
        cache_device = torch.device("cpu")
    level_rows = None
    position_templates = None
    feature_sizes = None
    for start in range(0, int(video_cpu.shape[0]), frame_batch):
        end = min(int(video_cpu.shape[0]), start + frame_batch)
        images = video_cpu[start:end].to(device=device, dtype=torch.float32)
        gradient_context = nullcontext() if train_image_encoder else torch.no_grad()
        with gradient_context, _amp(args, device):
            if train_image_encoder:
                backbone = torch.utils.checkpoint.checkpoint(
                    sam.forward_image,
                    images,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                backbone = sam.forward_image(images)
            _, vision, positions, sizes = sam._prepare_backbone_features(backbone)
        maps = [
            value.permute(1, 2, 0).reshape(end - start, value.shape[2], *size)
            for value, size in zip(vision, sizes)
        ]
        if level_rows is None:
            level_rows = [[] for _ in maps]
            feature_sizes = [tuple(int(x) for x in size) for size in sizes]
            position_templates = [
                value[:, :1].detach().to(
                    device=cache_device, dtype=torch.bfloat16
                ).contiguous()
                for value in positions
            ]
        elif len(maps) != len(level_rows) or [tuple(int(x) for x in size) for size in sizes] != feature_sizes:
            raise RuntimeError("SAM2 image feature layout changed within one ROI")
        for rows, value in zip(level_rows, maps):
            cached = value if train_image_encoder else value.detach()
            rows.append(cached.to(
                device=cache_device, dtype=torch.bfloat16
            ).contiguous())
        del images, backbone, vision, positions, maps
    levels = [torch.cat(rows, dim=0) for rows in level_rows]
    prompt = {"top": levels[-1]}
    if len(levels) >= 2:
        prompt["fpn0"] = levels[0]
        prompt["fpn1"] = levels[1]
    return {
        "prompt": prompt,
        "levels": levels,
        "position_templates": position_templates,
        "feature_sizes": feature_sizes,
        "cache_device": str(cache_device),
    }


def generate_case_3d_tokens(
    prompt_gen, adapter, video_cpu, bundle_cpu, class_id, args, device
):
    """Generate learned dense/sparse prompts for every real ROI frame.

    Large ROIs are split only along depth. Every frame is evaluated by the
    PromptGen, and overlapping halo frames provide 3D context; no frame/token
    sampling or interpolation is used.
    """
    depth = int(video_cpu.shape[0])
    frame_batch = max(1, int(args.sam2_frame_batch_size))
    image_views = encode_case_image_views(
        adapter, video_cpu, args, device, training=bool(prompt_gen.training)
    )
    semantic_text = semantic_text_for_class(int(class_id), args)
    chunk_size = max(1, int(getattr(args, "prompt_depth_chunk_size", 48)))
    halo = max(0, int(getattr(args, "prompt_depth_halo", 24)))
    full_limit = max(1, int(getattr(args, "prompt_full_depth_limit", 96)))
    configured_checkpoint_threshold = int(
        getattr(prompt_gen, "FULL_DEPTH_CHECKPOINT_THRESHOLD", 96)
    )
    large_roi_limit = max(
        0, int(getattr(args, "sam2_feature_cache_gpu_max_frames", 128))
    )
    # Multiple large chunks retain separate backward graphs. For ROIs that
    # already use CPU feature offload, restore 3D-encoder recomputation to
    # keep memory bounded; common small/medium ROIs retain the fast path.
    if prompt_gen.training and depth > large_roi_limit:
        prompt_gen.FULL_DEPTH_CHECKPOINT_THRESHOLD = min(
            configured_checkpoint_threshold, 48
        )
    cores = [(0, depth)] if depth <= full_limit else [
        (start, min(depth, start + chunk_size)) for start in range(0, depth, chunk_size)
    ]

    requested_background = bundle_cpu.get("v10_forced_background_mode")
    if requested_background is not None:
        background_mode = str(requested_background)
    elif prompt_gen.training or bool(prompt_gen.DEFAULT_EVAL_RANDOM_PROMPT_MODES):
        background_mode = prompt_gen._choose(tuple(prompt_gen.DEFAULT_TRAIN_BACKGROUND_MODES))
    else:
        background_mode = str(prompt_gen.DEFAULT_EVAL_BACKGROUND_MODE)
    previous_background = getattr(prompt_gen, "_v10_forced_background_mode", None)
    previous_background_points = getattr(prompt_gen, "_v10_forced_background_points", None)
    prompt_gen._v10_forced_background_mode = background_mode
    if background_mode == "point":
        with torch.no_grad():
            groups_for_points = bundle_cpu["groups"].to(device)
            prompt_gen._v10_forced_background_points = prompt_gen._one_point_per_group(
                groups_for_points[:, 1:2]
            ).detach()

    dense_rows, legacy_sparse_rows, aligned_rows, strength_rows = [], [], [], []
    morphology_auxiliary = []
    pool_numerator_rows, pool_denominator_rows = [], []
    foreground_mode = foreground_channel = background_active = None
    try:
        for core_start, core_end in cores:
            region_start = max(0, core_start - halo)
            region_end = min(depth, core_end + halo)
            prompt_video_rows = []
            for start in range(region_start, region_end, frame_batch):
                end = min(region_end, start + frame_batch)
                images = video_cpu[start:end].to(device, dtype=torch.float32)
                prompt_video_rows.append(F.interpolate(
                    images, size=(prompt_gen.work_size, prompt_gen.work_size),
                    mode="bilinear", align_corners=False,
                ))
                del images
            prompt_video = torch.cat(prompt_video_rows, dim=0)
            image_features = None
            if prompt_gen.image_feature_mode != "none":
                image_features = {
                    key: value[region_start:region_end].to(device)
                    for key, value in image_views["prompt"].items()
                }
            context_kwargs = {}
            if bool(getattr(prompt_gen, "v10_3_global_context_enabled", False)):
                context_kwargs = {
                    "v10_3_global_thumbnail": bundle_cpu[
                        "v10_3_global_thumbnail"
                    ][region_start:region_end].to(device),
                    "v10_3_crop_geometry": bundle_cpu[
                        "v10_3_crop_geometry"
                    ].to(device),
                }
            with _amp(args, device):
                outputs = prompt_gen.forward_case_3d_tokens(
                    bundle_cpu["groups"].to(device), prompt_video,
                    plane_ids=bundle_cpu["plane_ids"].to(device),
                    slice_coords=bundle_cpu["slice_coords"].to(device),
                    frame_coords=bundle_cpu["frame_coords"][region_start:region_end].to(device),
                    foreground_mode_id=bundle_cpu["v10_foreground_mode_id"].to(device),
                    image_features=image_features,
                    semantic_text=semantic_text,
                    **context_kwargs,
                )
            if (
                prompt_gen.training
                and "morphology_route_logits" in outputs
                and "morphology_counterfactual_logits" in outputs
            ):
                morphology_auxiliary.append({
                    "route_logits": outputs["morphology_route_logits"],
                    "counterfactual_logits": outputs[
                        "morphology_counterfactual_logits"
                    ],
                    "region_start": int(region_start),
                    "region_end": int(region_end),
                })
            local_start, local_end = core_start - region_start, core_end - region_start
            dense = outputs["dense_prompt_embeddings"][local_start:local_end].contiguous()
            slice_sparse = outputs["slice_prompt_embeddings"][local_start:local_end]
            volume_count = int(outputs["volume_prompt_embeddings"].shape[1])
            slice_count = int(outputs["slice_prompt_embeddings"].shape[1])
            learned_fg_bg = outputs["sparse_prompt_embeddings"][
                local_start:local_end, slice_count + volume_count:
            ]
            legacy_sparse = torch.cat((slice_sparse, learned_fg_bg), dim=1).contiguous()
            aligned_core = outputs["aligned_prompt"][local_start:local_end].detach()
            strength = aligned_core[:, 0].float().sum(dim=(-2, -1))
            fusion_core = outputs["_fusion_features_3d"][:, :, local_start:local_end].float()
            prompt_core = outputs["aligned_prompt"][local_start:local_end].permute(
                1, 0, 2, 3
            ).unsqueeze(0).detach().float().clamp_min(0.0)
            pool_numerator_rows.append(torch.einsum(
                "bfzyx,bpzyx->pf", fusion_core, prompt_core
            ))
            pool_denominator_rows.append(prompt_core.sum(dim=(0, 2, 3, 4)))
            if prompt_gen.training:
                dense_rows.append(dense)
                legacy_sparse_rows.append(legacy_sparse)
                aligned_rows.append(aligned_core)
                strength_rows.append(strength)
            else:
                dense_rows.append(dense.detach().to(device="cpu", dtype=torch.float16))
                legacy_sparse_rows.append(legacy_sparse.detach().to(device="cpu", dtype=torch.float16))
                aligned_rows.append(aligned_core.to(device="cpu", dtype=torch.float16))
                strength_rows.append(strength.detach().cpu())
            if foreground_mode is None:
                foreground_mode = outputs["foreground_prompt_mode"]
                foreground_channel = int(outputs["foreground_channel"])
                background_active = bool(outputs["background_active"])
            elif (
                foreground_mode != outputs["foreground_prompt_mode"]
                or foreground_channel != int(outputs["foreground_channel"])
                or background_active != bool(outputs["background_active"])
            ):
                raise RuntimeError("Prompt mode changed between depth chunks")
            del outputs, prompt_video, prompt_video_rows, image_features
            if not prompt_gen.training and device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        prompt_gen.FULL_DEPTH_CHECKPOINT_THRESHOLD = configured_checkpoint_threshold
        if previous_background is None:
            delattr(prompt_gen, "_v10_forced_background_mode")
        else:
            prompt_gen._v10_forced_background_mode = previous_background
        if previous_background_points is None:
            if hasattr(prompt_gen, "_v10_forced_background_points"):
                delattr(prompt_gen, "_v10_forced_background_points")
        else:
            prompt_gen._v10_forced_background_points = previous_background_points

    prompt_strength = torch.cat(strength_rows)
    anchor = int(torch.argmax(prompt_strength).item()) if prompt_strength.numel() else depth // 2
    global_numerator = torch.stack(pool_numerator_rows).sum(dim=0)
    global_denominator = torch.stack(pool_denominator_rows).sum(dim=0).clamp_min(1e-6)
    pooled_features = global_numerator / global_denominator[:, None]
    with _amp(args, device):
        global_tokens = prompt_gen.build_case_3d_tokens_from_global_pool(
            pooled_features,
            foreground_mode=foreground_mode,
            background_mode=background_mode,
            foreground_channel=foreground_channel,
            background_active=background_active,
            semantic_text=semantic_text,
        )
        fuse_case_context = getattr(prompt_gen, "fuse_v10_3_case_context", None)
        if callable(fuse_case_context) and bool(
            getattr(prompt_gen, "v10_3_global_context_enabled", False)
        ):
            global_tokens = fuse_case_context(
                global_tokens,
                bundle_cpu["v10_3_global_thumbnail"].to(device),
                bundle_cpu["v10_3_crop_geometry"].to(device),
            )
    compact = {
        **global_tokens,
        "foreground_prompt_mode": foreground_mode,
        "background_prompt_mode": background_mode,
        "dense_prompt_embeddings": torch.cat(dense_rows, dim=0),
        "per_frame_sparse_embeddings": torch.cat(legacy_sparse_rows, dim=0),
        "_aligned_prompt": torch.cat(aligned_rows, dim=0),
        "_image_views": image_views,
        "_morphology_auxiliary": morphology_auxiliary,
    }
    if int(compact["dense_prompt_embeddings"].shape[0]) != depth:
        raise RuntimeError("PromptGen depth stitching did not preserve every ROI frame")
    return compact, anchor


def _compact_memory(sam, vision_feats, feat_sizes, high_res, obj_ptr, obj_score):
    with torch.no_grad():
        memory, position = sam._encode_new_memory(
            current_vision_feats=[value.detach() for value in vision_feats],
            feat_sizes=feat_sizes,
            pred_masks_high_res=high_res.detach(),
            object_score_logits=obj_score.detach(),
            is_mask_from_pts=False,
        )
    return {
        "maskmem_features": memory.to(dtype=torch.bfloat16, device="cpu"),
        "maskmem_pos_enc": [value.to(device="cpu") for value in position],
        "obj_ptr": obj_ptr.detach(),
    }


def _decode_external_prompt(
    sam, pix_feat, high_res_features, sparse, dense, apply_object_score_gate,
    detach_high_res: bool = False,
    materialize_high_res: bool = True,
):
    dense = dense.to(device=pix_feat.device, dtype=pix_feat.dtype)
    if tuple(dense.shape[-2:]) != tuple(pix_feat.shape[-2:]):
        dense = F.interpolate(dense, size=pix_feat.shape[-2:], mode="bilinear", align_corners=False)
    low, _ious, output_tokens, object_score = sam.sam_mask_decoder(
        image_embeddings=pix_feat,
        image_pe=sam.sam_prompt_encoder.get_dense_pe().to(
            device=pix_feat.device, dtype=pix_feat.dtype
        ),
        sparse_prompt_embeddings=sparse.to(dtype=pix_feat.dtype),
        dense_prompt_embeddings=dense.to(dtype=pix_feat.dtype),
        multimask_output=False, repeat_image=False,
        high_res_features=high_res_features,
    )
    if apply_object_score_gate and getattr(sam, "pred_obj_scores", False):
        low = torch.where(
            (object_score > 0)[:, :, None, None], low,
            low.new_full((), -1024.0),
        )
    if materialize_high_res:
        high_source = low.detach() if detach_high_res else low
        high = F.interpolate(
            high_source.float(), size=(sam.image_size, sam.image_size),
            mode="bilinear", align_corners=False,
        )
        obj_ptr = sam.obj_ptr_proj(output_tokens[:, 0])
        if getattr(sam, "pred_obj_scores", False):
            appearing = (
                object_score.sigmoid()
                if sam.soft_no_obj_ptr else (object_score > 0).float()
            )
            if sam.fixed_no_obj_ptr:
                obj_ptr = appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - appearing) * sam.no_obj_ptr
    else:
        # Checkpointed decoder calls must return tensors. Empty tensors avoid
        # the otherwise redundant 1024-square interpolation and object-pointer
        # projection for train frames that are not committed to memory.
        high = low.new_empty((0,))
        obj_ptr = low.new_empty((0,))
    return low, high, obj_ptr, object_score


def memory_sliding_window_decode(
    adapter, video_cpu, bundle_cpu, token_outputs, anchor,
    args, device, training: bool,
):
    """Decode every ROI frame while preserving SAM2 memory across windows."""
    sam = adapter.sam
    depth = int(video_cpu.shape[0])
    coords = bundle_cpu["frame_coords"]
    prompt_3d = token_outputs["prompt_3d_tokens"]
    role_ids = token_outputs["prompt_3d_role_ids"]
    type_ids = token_outputs["prompt_3d_type_ids"]
    adapter_3d = sam.sam_mask_decoder.prompt_3d_adapter
    memory_adapter = sam.sam_mask_decoder.prompt_memory_adapter
    dense_prompt_volume = token_outputs["dense_prompt_embeddings"]
    per_frame_sparse = token_outputs["per_frame_sparse_embeddings"]
    image_views = token_outputs.get("_image_views")
    if image_views is None:
        raise RuntimeError("v10 memory decode requires the shared SAM2 image-feature cache")
    logits = [None] * depth
    sparse_rows = [None] * depth
    decoder_checkpoint_depth = max(
        0, int(getattr(args, "decoder_checkpoint_min_frames", 96))
    )
    use_decoder_checkpoint = bool(
        getattr(args, "decoder_gradient_checkpoint", False)
    ) or depth > decoder_checkpoint_depth

    def decode_frame(
        frame_idx, vision, positions, sizes, local, output_dict, reverse, init,
        commit_memory=True,
    ):
        frame_vision = [value[:, local:local + 1] for value in vision]
        frame_position = [value[:, local:local + 1] for value in positions]
        high_res_features = [
            value.permute(1, 2, 0).view(1, value.size(2), *size)
            for value, size in zip(frame_vision[:-1], sizes[:-1])
        ] if len(frame_vision) > 1 else None
        with torch.no_grad(), _amp(args, device):
            pix_feat = sam._prepare_memory_conditioned_features(
                frame_idx=int(frame_idx), is_init_cond_frame=bool(init),
                current_vision_feats=frame_vision[-1:],
                current_vision_pos_embeds=frame_position[-1:],
                feat_sizes=sizes[-1:], output_dict=output_dict,
                num_frames=depth, track_in_reverse=bool(reverse),
            )
        with _amp(args, device):
            current_pix = frame_vision[-1].permute(1, 2, 0).reshape(
                1, frame_vision[-1].shape[2], *sizes[-1]
            )
            pix_feat = memory_adapter(
                pix_feat,
                current_pix,
                prompt_3d.to(device),
                is_init_frame=bool(init),
                prompt_3d_role_ids=role_ids,
            )
            adapter_sparse = adapter_3d(
                prompt_3d.to(device), coords[frame_idx:frame_idx + 1].to(device), None,
                role_ids=role_ids, type_ids=type_ids,
            )
            learned_sparse = per_frame_sparse[frame_idx:frame_idx + 1].to(
                device=device, dtype=adapter_sparse.dtype
            )
            sparse = torch.cat((learned_sparse, adapter_sparse), dim=1)
            dense = dense_prompt_volume[frame_idx:frame_idx + 1].to(device)

        def decoder_call(sparse_input):
            return _decode_external_prompt(
                sam, pix_feat, high_res_features, sparse_input, dense,
                bool(args.object_score_gate), detach_high_res=bool(training),
                materialize_high_res=bool(commit_memory or not training),
            )

        with _amp(args, device):
            if (
                training
                and not init
                and use_decoder_checkpoint
            ):
                low, high, obj_ptr, obj_score = checkpoint(
                    decoder_call, sparse, use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                low, high, obj_ptr, obj_score = decoder_call(sparse)
        current = None
        if commit_memory:
            with _amp(args, device):
                current = _compact_memory(
                    sam, frame_vision[-1:], sizes[-1:], high, obj_ptr, obj_score
                )
        # The SAM2 decoder predicts native low-resolution logits; 1024x1024
        # masks are only bilinear views used by memory/evaluation.  Retaining
        # gradients through those expanded masks wastes ~16x logits memory.
        logits[frame_idx] = low.float() if training else high.float().cpu()
        sparse_rows[frame_idx] = sparse
        return current

    def image_features(frame_ids):
        levels = [
            value.index_select(
                0, torch.tensor(frame_ids, dtype=torch.long, device=value.device)
            ).to(device)
            for value in image_views["levels"]
        ]
        vision = [
            value.flatten(2).permute(2, 0, 1).contiguous()
            for value in levels
        ]
        positions = [
            value.to(device=device, dtype=feature.dtype).expand(-1, len(frame_ids), -1)
            for value, feature in zip(image_views["position_templates"], vision)
        ]
        return levels, vision, positions, image_views["feature_sizes"]

    # The strongest prompt frame is the conditioning anchor.
    levels, vision, positions, sizes = image_features([anchor])
    empty_state = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
    anchor_output = decode_frame(
        anchor, vision, positions, sizes, 0, empty_state, False, True, True
    )
    del levels, vision, positions

    def run_direction(frame_ids, reverse):
        state = {
            "cond_frame_outputs": {int(anchor): anchor_output},
            "non_cond_frame_outputs": {},
        }
        window = max(1, int(args.sam2_frame_batch_size))
        group_size = max(1, int(getattr(
            args,
            "train_memory_group_size" if training else "eval_memory_group_size",
            1,
        )))
        commit_positions = set(memory_commit_positions(len(frame_ids), group_size))
        direction_position = 0
        for start in range(0, len(frame_ids), window):
            current_ids = frame_ids[start:start + window]
            levels, vision, positions, sizes = image_features(current_ids)
            for local, frame_idx in enumerate(current_ids):
                commit_memory = direction_position in commit_positions
                current = decode_frame(
                    frame_idx, vision, positions, sizes, local,
                    state, reverse, False, commit_memory,
                )
                if current is not None:
                    state["non_cond_frame_outputs"][int(frame_idx)] = current
                direction_position += 1
            del levels, vision, positions

    run_direction(list(range(anchor + 1, depth)), False)
    run_direction(list(range(anchor - 1, -1, -1)), True)
    return torch.cat(logits, dim=0), torch.cat(sparse_rows, dim=0)


@torch.no_grad()
def decode_anchor_coarse_mask(
    adapter, video_cpu, bundle_cpu, token_outputs, anchor, args, device
):
    """Decode only the anchor frame for inference-time adaptive XY cropping."""
    sam = adapter.sam
    depth = int(video_cpu.shape[0])
    image_views = token_outputs.get("_image_views")
    if image_views is None:
        raise RuntimeError("Anchor coarse decode requires SAM2 image features")
    anchor = int(anchor)
    levels = [
        value.index_select(
            0, torch.tensor([anchor], dtype=torch.long, device=value.device)
        ).to(device)
        for value in image_views["levels"]
    ]
    vision = [
        value.flatten(2).permute(2, 0, 1).contiguous()
        for value in levels
    ]
    positions = [
        value.to(device=device, dtype=feature.dtype).expand(-1, 1, -1)
        for value, feature in zip(image_views["position_templates"], vision)
    ]
    sizes = image_views["feature_sizes"]
    empty_state = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
    with _amp(args, device):
        pix_feat = sam._prepare_memory_conditioned_features(
            frame_idx=anchor, is_init_cond_frame=True,
            current_vision_feats=vision[-1:],
            current_vision_pos_embeds=positions[-1:],
            feat_sizes=sizes[-1:], output_dict=empty_state,
            num_frames=depth, track_in_reverse=False,
        )
        prompt_3d = token_outputs["prompt_3d_tokens"].to(device)
        pix_feat = sam.sam_mask_decoder.prompt_memory_adapter(
            pix_feat, vision[-1].permute(1, 2, 0).reshape(
                1, vision[-1].shape[2], *sizes[-1]
            ), prompt_3d, is_init_frame=True,
        )
        adapter_sparse = sam.sam_mask_decoder.prompt_3d_adapter(
            prompt_3d, bundle_cpu["frame_coords"][anchor:anchor + 1].to(device), None,
            role_ids=token_outputs["prompt_3d_role_ids"].to(device),
            type_ids=token_outputs["prompt_3d_type_ids"].to(device),
        )
        per_frame_sparse = token_outputs["per_frame_sparse_embeddings"]
        sparse = torch.cat((
            per_frame_sparse[anchor:anchor + 1].to(device=device, dtype=adapter_sparse.dtype),
            adapter_sparse,
        ), dim=1)
        dense = token_outputs["dense_prompt_embeddings"][anchor:anchor + 1].to(device)
        _low, high, _obj_ptr, _obj_score = _decode_external_prompt(
            sam, vision[-1].permute(1, 2, 0).reshape(
                1, vision[-1].shape[2], *sizes[-1]
            ),
            [
                value.permute(1, 2, 0).view(1, value.size(2), *size)
                for value, size in zip(vision[:-1], sizes[:-1])
            ] if len(vision) > 1 else None,
            sparse, dense, bool(args.object_score_gate), detach_high_res=True,
        )
    return high.float().cpu()


def anchor_coarse_square_crop(
    coarse_logits, full_side, threshold=0.5, margin_ratio=0.50, min_size=256
):
    """Get a conservative XY square crop; return None on empty/border masks."""
    mask = (torch.sigmoid(coarse_logits[0, 0]) >= float(threshold)).numpy()
    coords = np.argwhere(mask)
    if not coords.size:
        return None
    y0, x0 = coords.min(axis=0); y1, x1 = coords.max(axis=0) + 1
    h, w = int(y1 - y0), int(x1 - x0)
    if y0 <= 0 or x0 <= 0 or y1 >= mask.shape[0] or x1 >= mask.shape[1]:
        return None
    side = max(float(h), float(w)) * (1.0 + 2.0 * max(0.0, float(margin_ratio)))
    side = min(float(full_side), max(float(min_size), side))
    side_i = max(1, int(round(side)))
    cy, cx = (float(y0 + y1) * 0.5, float(x0 + x1) * 0.5)
    left = min(max(0, int(round(cx - side_i * 0.5))), full_side - side_i)
    top = min(max(0, int(round(cy - side_i * 0.5))), full_side - side_i)
    return (left, left + side_i, top, top + side_i)


class JointTrainingForwardMemoryV10(nn.Module):
    def __init__(self, prompt_gen, adapter):
        super().__init__()
        self.prompt_gen = prompt_gen
        self.mask_decoder = adapter.sam.sam_mask_decoder
        object.__setattr__(self, "_adapter_ref", weakref.ref(adapter))

    @property
    def adapter(self):
        result = self._adapter_ref()
        if result is None:
            raise RuntimeError("SAM2 adapter was released")
        return result

    @staticmethod
    def _prompt_derived_training_window(video, bundle, target, max_frames):
        depth = int(video.shape[0])
        max_frames = max(0, int(max_frames))
        if max_frames <= 0 or depth <= max_frames:
            return video, bundle, target
        mode_id = int(bundle["v10_foreground_mode_id"].item())
        foreground_channel = 0 if mode_id == 0 else 2
        prompt = bundle["groups"][:, foreground_channel].float().clamp_min(0.0)
        row_strength = prompt.amax(dim=(0, 2))
        active_rows = torch.nonzero(row_strength > 0, as_tuple=False).flatten()
        if active_rows.numel():
            # Every epoch may select another valid scribble/box z location,
            # so truncated BPTT covers different parts of a long organ without
            # using GT to position the window.
            selected = active_rows[
                torch.randint(int(active_rows.numel()), (1,)).item()
            ]
            z_coord = (
                2.0 * selected.float() / max(1, int(row_strength.numel()) - 1)
                - 1.0
            )
            center = int(torch.argmin(
                (bundle["frame_coords"].float() - z_coord).abs()
            ).item())
        else:
            center = depth // 2
        start = min(max(0, center - max_frames // 2), depth - max_frames)
        end = start + max_frames
        cropped_bundle = dict(bundle)
        cropped_bundle["frame_coords"] = bundle["frame_coords"][start:end].contiguous()
        return (
            video[start:end].contiguous(),
            cropped_bundle,
            target[start:end].contiguous(),
        )

    def forward(self, batch, args, device):
        targets, logits_rows, sparse_rows = [], [], []
        frame_counts, class_ids, consistency_contexts = [], [], []
        for video, bundle, _points, target, class_id in batch:
            video, bundle, target = self._prompt_derived_training_window(
                video, bundle, target,
                getattr(args, "train_max_roi_frames", 192),
            )
            if bool(getattr(args, "v10_3_local_patch", False)):
                from v10_3_patch_runtime import (
                    apply_local_patch,
                    local_patch_config_from_args,
                )
                video, bundle, target, _geometry = apply_local_patch(
                    None, video, bundle, target, local_patch_config_from_args(args)
                )
            elif hasattr(args, "v10_3_native_image_size"):
                from v10_3_patch_runtime import resize_full_frame_to_native
                video, target, _geometry = resize_full_frame_to_native(
                    video, target, int(args.v10_3_native_image_size)
                )
            class_id = int(class_id.item()) if torch.is_tensor(class_id) else int(class_id)
            tokens, anchor = generate_case_3d_tokens(
                self.prompt_gen, self.adapter, video, bundle,
                class_id, args, device,
            )
            decode = memory_sliding_window_decode
            if bool(getattr(args, "v10_3_grouped_decode", False)):
                from v10_3_memory_runtime import memory_grouped_decode_v103
                decode = memory_grouped_decode_v103
            case_logits, case_sparse = decode(
                self.adapter, video, bundle, tokens, anchor,
                args, device, training=True,
            )
            target_device = target.to(device)
            if tuple(target_device.shape[-2:]) != tuple(case_logits.shape[-2:]):
                # Max pooling preserves tiny positive structures and recall
                # when reducing a 1024 target to SAM2's native mask grid.
                target_device = F.adaptive_max_pool2d(
                    target_device.float(), case_logits.shape[-2:]
                )
            targets.append(target_device)
            logits_rows.append(case_logits); sparse_rows.append(case_sparse)
            frame_counts.append(int(video.shape[0])); class_ids.append(class_id)
            consistency_contexts.append({
                "aligned_prompt": tokens["_aligned_prompt"],
                "foreground_mode": tokens["foreground_prompt_mode"],
                "background_mode": tokens["background_prompt_mode"],
                "morphology_auxiliary": tokens.get("_morphology_auxiliary", []),
            })
        # The v10 loader intentionally uses one case per GPU. torch.cat still
        # allocates and copies a new tensor for a one-element list; for a
        # 300+ frame 1024x1024 ROI that needlessly duplicates more than 1 GiB
        # each of logits and target immediately before backward.
        logits = logits_rows[0] if len(logits_rows) == 1 else torch.cat(logits_rows)
        sparse = sparse_rows[0] if len(sparse_rows) == 1 else torch.cat(sparse_rows)
        target_volume = targets[0] if len(targets) == 1 else torch.cat(targets)
        dense_placeholder = logits.new_zeros((logits.shape[0], 1, 1, 1))
        prepared = (
            logits.new_empty((0,)), None, sparse, dense_placeholder,
            target_volume, frame_counts, class_ids, consistency_contexts,
        )
        return prepared, logits


def partition_validation_classes(allowed, tasks_by_class):
    allowed = {int(value) for value in allowed}
    covered = sorted(
        class_id for class_id in allowed if tasks_by_class.get(class_id)
    )
    missing = sorted(allowed - set(covered))
    return covered, missing


def numeric_validation_metrics(metrics, names):
    """Select finite numeric metrics while ignoring descriptive metadata."""
    output = {}
    for name in names:
        if name not in metrics:
            continue
        value = float(metrics[name])
        if np.isfinite(value):
            output[name] = value
    return output


@torch.no_grad()
def evaluate_memory_v10(prompt_gen, adapter, pairs, class_ids, args, device):
    started = time.perf_counter()
    prompt_gen.eval(); adapter.eval()
    dataset = make_dataset(pairs, args, return_target_class=True)
    allowed = {int(value) for value in class_ids}
    all_tasks = []
    for case_idx, label_path in enumerate(dataset.label_paths):
        present = dataset._load_label_classes_cached(label_path)
        all_tasks.extend((case_idx, int(c)) for c in present if int(c) in allowed)
    tasks_by_class = defaultdict(list)
    for task in all_tasks:
        tasks_by_class[task[1]].append(task)
    covered_classes, missing_classes = partition_validation_classes(
        allowed, tasks_by_class
    )
    if not covered_classes:
        raise RuntimeError("Official validation cases contain no requested non-empty classes")
    # First select one deterministic case for every class. Additional slots
    # are filled from the remaining case/class tasks. Thus max_tasks is never
    # allowed to silently remove a class from validation.
    validation_seed = int(getattr(args, "validation_seed", 2027))
    rng = random.Random(validation_seed)
    mandatory_tasks = [
        rng.choice(tasks_by_class[class_id]) for class_id in covered_classes
    ]
    rng.shuffle(mandatory_tasks)
    mandatory_set = set(mandatory_tasks)
    remaining_tasks = [task for task in all_tasks if task not in mandatory_set]
    previous_dice = {
        int(key): float(value)
        for key, value in getattr(args, "validation_class_dice", {}).items()
        if np.isfinite(float(value))
    }
    difficulty_weights = {
        int(key): max(0.05, float(value))
        for key, value in getattr(args, "validation_class_difficulty", {}).items()
        if np.isfinite(float(value))
    }
    if previous_dice or difficulty_weights:
        # Weighted sampling without replacement. Low-Dice classes receive a
        # smaller exponential key and therefore more of the extra 83 slots,
        # while the mandatory set still guarantees complete class coverage.
        def difficulty_key(task):
            class_id = int(task[1])
            hardness = difficulty_weights.get(
                class_id, max(0.05, 1.0 - previous_dice.get(class_id, 0.5))
            )
            return -math.log(max(rng.random(), 1e-12)) / hardness
        remaining_tasks.sort(key=difficulty_key)
    else:
        rng.shuffle(remaining_tasks)
    requested_max_tasks = max(0, int(getattr(args, "validation_max_tasks", 0)))
    effective_max_tasks = (
        len(all_tasks) if requested_max_tasks == 0
        else max(requested_max_tasks, len(mandatory_tasks))
    )
    tasks = mandatory_tasks + remaining_tasks[
        :max(0, effective_max_tasks - len(mandatory_tasks))
    ]
    foreground_modes = (
        ("scribble", "box")
        if str(getattr(args, "val_foreground_mode", "random")) == "random"
        else (str(args.val_foreground_mode),)
    )
    background_modes = (
        ("scribble", "point", "none")
        if str(getattr(args, "val_background_mode", "random")) == "random"
        else (str(args.val_background_mode),)
    )
    prompt_combinations = [
        (foreground, background)
        for foreground in foreground_modes
        for background in background_modes
    ]
    task_specs = [
        (case_idx, class_id, *prompt_combinations[index % len(prompt_combinations)])
        for index, (case_idx, class_id) in enumerate(tasks)
    ]
    distributed = context_from_env()
    task_specs = task_specs[distributed.rank::distributed.world_size]
    names = (
        "dice", "mean_slice_dice", "iou", "precision", "recall",
        "specificity", "pred_gt_ratio", "mean_slice_nsd_1mm",
        "mean_slice_nsd_2mm", "mean_slice_nsd_3mm",
    )
    thresholds = tuple(dict.fromkeys(
        float(value) for value in (
            *getattr(args, "validation_thresholds", (.55, .60, .65)),
            float(args.mask_threshold),
        )
    ))
    if any(not 0.0 < value < 1.0 for value in thresholds):
        raise ValueError(f"Validation thresholds must be between 0 and 1: {thresholds}")
    threshold_keys = {value: f"{value:.2f}" for value in thresholds}
    primary_threshold_key = threshold_keys[float(args.mask_threshold)]
    by_class = {name: defaultdict(list) for name in names}
    by_mode = defaultdict(lambda: defaultdict(list))
    by_threshold_class = {
        key: {name: defaultdict(list) for name in names}
        for key in threshold_keys.values()
    }
    foreground_dice_by_class = defaultdict(lambda: defaultdict(list))
    metric_pipeline = None
    local_patch_audit = {
        "tasks": 0,
        "initial_input_side_sum": 0.0,
        "final_input_side_sum": 0.0,
        "magnification_sum": 0.0,
        "expanded_tasks": 0,
        "expansion_steps": 0,
    }
    if bool(getattr(args, "v10_3_metric_pipeline", False)):
        from v10_3_metrics import OrderedMetricPipeline
        metric_pipeline = OrderedMetricPipeline(
            max_workers=int(getattr(args, "v10_3_metric_workers", 2)),
            max_inflight=int(getattr(args, "v10_3_metric_inflight", 2)),
        )

    def consume_threshold_metrics(meta, metrics_by_threshold):
        class_id, foreground_mode, combo = meta
        threshold_metrics = {
            threshold_keys[float(threshold)]: values
            for threshold, values in metrics_by_threshold.items()
        }
        metrics = threshold_metrics[primary_threshold_key]
        for threshold_key, threshold_values in threshold_metrics.items():
            for name in names:
                value = float(threshold_values[name])
                if np.isfinite(value):
                    by_threshold_class[threshold_key][name][class_id].append(value)
        for name, value in numeric_validation_metrics(metrics, by_class).items():
            by_class[name][class_id].append(value)
            by_mode[combo][name].append(value)
            if name == "dice":
                foreground_dice_by_class[foreground_mode][class_id].append(value)

    for case_idx, class_id, foreground_mode, background_mode in tqdm(
        task_specs, desc="Val v10 SAM2 memory ROI", leave=False,
        disable=not distributed.is_main,
        dynamic_ncols=True,
    ):
        dataset.v10_foreground_mode = foreground_mode
        video, bundle, _points, target, _ = dataset.get_item_for_class(case_idx, class_id)
        if float(target.sum()) <= 0: continue
        frame_indices = np.asarray(dataset.last_frame_indices, dtype=np.int64).copy()
        raw_target = dataset._load_total_target(dataset.label_paths[case_idx], class_id)
        canonical_image = nib.as_closest_canonical(
            nib.load(str(dataset.image_paths[case_idx]))
        )
        zoom_x, zoom_y, zoom_z = canonical_image.header.get_zooms()[:3]
        raw_spacing_dhw = (float(zoom_z), float(zoom_x), float(zoom_y))
        metric_full_side = int(video.shape[-1])
        bundle["v10_forced_background_mode"] = background_mode
        crop_bounds = None
        full_video = full_bundle = full_target = None
        initial_geometry = None
        if bool(getattr(args, "v10_3_local_patch", False)):
            from v10_3_patch_runtime import (
                apply_local_patch,
                local_patch_config_from_args,
            )
            full_video, full_bundle, full_target = video, dict(bundle), target
            video, bundle, target, geometry = apply_local_patch(
                dataset, video, bundle, target, local_patch_config_from_args(args)
            )
            initial_geometry = geometry
            crop_bounds = geometry.bounds
        elif hasattr(args, "v10_3_native_image_size"):
            from v10_3_patch_runtime import resize_full_frame_to_native
            video, target, geometry = resize_full_frame_to_native(
                video, target, int(args.v10_3_native_image_size)
            )
            crop_bounds = geometry.bounds
        tokens, anchor = generate_case_3d_tokens(
            prompt_gen, adapter, video, bundle, class_id, args, device
        )
        if (
            bool(getattr(args, "anchor_coarse_crop", False))
            and not bool(getattr(args, "v10_3_local_patch", False))
        ):
            coarse = decode_anchor_coarse_mask(
                adapter, video, bundle, tokens, anchor, args, device
            )
            crop_bounds = anchor_coarse_square_crop(
                coarse, int(video.shape[-1]),
                threshold=float(getattr(args, "anchor_coarse_threshold", 0.5)),
                margin_ratio=float(getattr(args, "anchor_coarse_margin_ratio", 0.50)),
                min_size=int(getattr(args, "anchor_coarse_min_size", 256)),
            )
            if crop_bounds is not None:
                left, right, top, bottom = crop_bounds
                full_side = int(video.shape[-1])
                prompt_data = {
                    "groups": bundle["groups"],
                    "plane_ids": bundle["plane_ids"],
                }
                cropped_bundle = dict(bundle)
                cropped_bundle["groups"] = dataset._scale_prompt_groups_to_patch(
                    prompt_data, crop_bounds, full_side
                )
                video = video[:, :, top:bottom, left:right].contiguous()
                video = F.interpolate(
                    video.float(), size=(int(dataset.model_input_size), int(dataset.model_input_size)),
                    mode="bilinear", align_corners=False,
                )
                bundle = cropped_bundle
                del tokens, coarse
                tokens, anchor = generate_case_3d_tokens(
                    prompt_gen, adapter, video, bundle, class_id, args, device
                )
        decode = memory_sliding_window_decode
        if bool(getattr(args, "v10_3_grouped_decode", False)):
            from v10_3_memory_runtime import memory_grouped_decode_v103
            decode = memory_grouped_decode_v103
        logits, _ = decode(
            adapter, video, bundle, tokens, anchor,
            args, device, training=False,
        )
        expansion_steps = 0
        if (
            initial_geometry is not None
            and bool(getattr(args, "v10_3_auto_expand_xy", True))
            and int(getattr(args, "v10_3_auto_expand_xy_steps", 2)) > 0
        ):
            from v10_3_patch_runtime import (
                apply_local_patch_bounds,
                expand_square_bounds_from_contacts,
                foreground_prompt_seed_mask,
                prompt_connected_xy_boundary_contacts,
            )
            max_expansion_steps = int(getattr(args, "v10_3_auto_expand_xy_steps", 2))
            for _expansion_index in range(max_expansion_steps):
                prediction = (
                    torch.sigmoid(logits.detach().float()).cpu().numpy()[:, 0]
                    > float(args.mask_threshold)
                )
                foreground_seed = foreground_prompt_seed_mask(
                    bundle, prediction.shape
                )
                contacts = prompt_connected_xy_boundary_contacts(
                    prediction,
                    foreground_seed,
                    band_pixels=int(getattr(args, "v10_3_auto_expand_xy_band", 2)),
                )
                expanded_bounds = expand_square_bounds_from_contacts(
                    crop_bounds,
                    full_side=metric_full_side,
                    contacts=contacts,
                    step_ratio=float(getattr(args, "v10_3_auto_expand_xy_ratio", 0.25)),
                )
                if expanded_bounds == crop_bounds:
                    break
                del video, bundle, target, tokens, logits, _
                video, bundle, target, geometry = apply_local_patch_bounds(
                    dataset,
                    full_video,
                    full_bundle,
                    full_target,
                    expanded_bounds,
                    int(getattr(args, "v10_3_native_image_size", 512)),
                )
                crop_bounds = geometry.bounds
                tokens, anchor = generate_case_3d_tokens(
                    prompt_gen, adapter, video, bundle, class_id, args, device
                )
                logits, _ = decode(
                    adapter, video, bundle, tokens, anchor,
                    args, device, training=False,
                )
                expansion_steps += 1
        if initial_geometry is not None:
            final_side = int(crop_bounds[1] - crop_bounds[0])
            local_patch_audit["tasks"] += 1
            local_patch_audit["initial_input_side_sum"] += float(initial_geometry.input_side)
            local_patch_audit["final_input_side_sum"] += float(final_side)
            local_patch_audit["magnification_sum"] += float(initial_geometry.magnification)
            local_patch_audit["expanded_tasks"] += int(expansion_steps > 0)
            local_patch_audit["expansion_steps"] += int(expansion_steps)
        combo = f"fg={tokens['foreground_prompt_mode']}|bg={tokens['background_prompt_mode']}"
        if bool(getattr(args, "v10_3_metric_pipeline", False)):
            from v10_3_metrics import full_volume_metrics_multi_threshold
            emitted = metric_pipeline.submit(
                (class_id, foreground_mode, combo),
                full_volume_metrics_multi_threshold,
                logits,
                raw_target,
                frame_indices=frame_indices,
                thresholds=thresholds,
                spacing_dhw=raw_spacing_dhw,
                crop_bounds=crop_bounds,
                full_side=metric_full_side,
            )
            for meta, threshold_metrics in emitted:
                consume_threshold_metrics(meta, threshold_metrics)
        else:
            threshold_metrics = {
                threshold: full_volume_binary_metrics_from_logits(
                    logits,
                    raw_target,
                    frame_indices=frame_indices,
                    threshold=threshold,
                    spacing_dhw=raw_spacing_dhw,
                    crop_bounds=crop_bounds,
                    full_side=metric_full_side,
                )
                for threshold in thresholds
            }
            consume_threshold_metrics(
                (class_id, foreground_mode, combo), threshold_metrics
            )
        del video, bundle, target, tokens, logits, _
        if full_video is not None:
            del full_video, full_bundle, full_target
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if metric_pipeline is not None:
        for meta, threshold_metrics in metric_pipeline.finish():
            consume_threshold_metrics(meta, threshold_metrics)
    payload = {
        "class": {n: {int(k): list(v) for k,v in r.items()} for n,r in by_class.items()},
        "mode": {m: {n: list(v) for n,v in r.items()} for m,r in by_mode.items()},
        "foreground_dice": {
            mode: {int(k): list(v) for k, v in rows.items()}
            for mode, rows in foreground_dice_by_class.items()
        },
        "threshold": {
            threshold: {
                name: {int(k): list(v) for k, v in rows.items()}
                for name, rows in metrics.items()
            }
            for threshold, metrics in by_threshold_class.items()
        },
        "local_patch_audit": dict(local_patch_audit),
    }
    gathered = all_gather_flat([payload], distributed)
    merged = {name: defaultdict(list) for name in names}
    modes = defaultdict(lambda: defaultdict(list))
    foreground_dice = defaultdict(lambda: defaultdict(list))
    threshold_class = {
        key: {name: defaultdict(list) for name in names}
        for key in threshold_keys.values()
    }
    merged_patch_audit = {
        key: sum(float(item.get("local_patch_audit", {}).get(key, 0.0)) for item in gathered)
        for key in local_patch_audit
    }
    for item in gathered:
        for name, rows in item["class"].items():
            for key, values in rows.items(): merged[name][int(key)].extend(values)
        for mode, rows in item["mode"].items():
            for name, values in rows.items(): modes[mode][name].extend(values)
        for mode, rows in item["foreground_dice"].items():
            for key, values in rows.items():
                foreground_dice[mode][int(key)].extend(values)
        for threshold, metric_rows in item["threshold"].items():
            for name, class_rows in metric_rows.items():
                for key, values in class_rows.items():
                    threshold_class[threshold][name][int(key)].extend(values)
    per = {n: {k: float(np.mean(v)) for k,v in r.items() if v} for n,r in merged.items()}
    mean = {n: float(np.mean(list(r.values()))) for n,r in per.items() if r}
    per_mode = {}
    for foreground_mode, background_mode in prompt_combinations:
        mode = f"fg={foreground_mode}|bg={background_mode}"
        rows = modes.get(mode, {})
        per_mode[mode] = {"count": len(rows.get("dice", [])),
                          "metric_counts": {name: len(values) for name, values in rows.items()}, **{
            name: float(np.mean(values)) for name, values in rows.items() if values
        }}
    foreground_class_macro_dice = {
        mode: float(np.mean([
            np.mean(values) for values in rows.values() if values
        ]))
        for mode, rows in foreground_dice.items()
        if any(rows.values())
    }
    per_threshold = {}
    for threshold, metric_rows in threshold_class.items():
        per_threshold[threshold] = {
            name: float(np.mean([
                np.mean(values) for values in class_rows.values() if values
            ]))
            for name, class_rows in metric_rows.items()
            if any(class_rows.values())
        }
    wall = time.perf_counter() - started
    task_count = sum(int(rows.get("count", 0)) for rows in per_mode.values())
    patch_tasks = max(1.0, merged_patch_audit["tasks"])
    patch_protocol = {
        "enabled": bool(getattr(args, "v10_3_local_patch", False)),
        "tasks": int(merged_patch_audit["tasks"]),
        "mean_initial_input_side": merged_patch_audit["initial_input_side_sum"] / patch_tasks,
        "mean_final_input_side": merged_patch_audit["final_input_side_sum"] / patch_tasks,
        "mean_initial_magnification": merged_patch_audit["magnification_sum"] / patch_tasks,
        "expanded_tasks": int(merged_patch_audit["expanded_tasks"]),
        "expansion_steps": int(merged_patch_audit["expansion_steps"]),
        "max_magnification": float(getattr(args, "v10_3_local_max_magnification", 1.0)),
        "preserve_background_support": bool(getattr(args, "v10_3_preserve_background_support", True)),
        "auto_expand_xy": bool(getattr(args, "v10_3_auto_expand_xy", False)),
    }
    result = {"mean": mean, "per_class": per, "per_prompt_mode": per_mode,
              "per_threshold": per_threshold,
              "primary_threshold": float(args.mask_threshold),
              "foreground_class_macro_dice": foreground_class_macro_dice,
              "validation_protocol": {
                  "seed": validation_seed,
                  "requested_max_tasks": requested_max_tasks,
                  "effective_max_tasks": effective_max_tasks,
                  "selected_tasks": len(tasks),
                  "covered_classes": len(covered_classes),
                  "requested_classes": len(allowed),
                  "na_classes": missing_classes,
                  "all_classes_covered": not missing_classes,
                  "all_nonempty_classes_covered": True,
                  "metric_space": "raw_nifti_full_volume",
                  "difficulty_weighted_extras": bool(previous_dice or difficulty_weights),
                  "difficulty_reference_classes": len(difficulty_weights or previous_dice),
                  "prompt_combinations": [
                      f"fg={foreground}|bg={background}"
                      for foreground, background in prompt_combinations
                  ],
                  "fixed_across_epochs": True,
                  "anchor_coarse_crop": bool(getattr(args, "anchor_coarse_crop", False)),
                  "anchor_coarse_threshold": float(getattr(args, "anchor_coarse_threshold", 0.5)),
                  "anchor_coarse_margin_ratio": float(getattr(args, "anchor_coarse_margin_ratio", 0.50)),
                  "local_patch": patch_protocol,
              },
              "performance": {"wall_seconds": wall, "tasks": task_count,
                              "seconds_per_task": wall / max(1, task_count)}}
    return mean.get("dice"), per.get("dice", {}), result
