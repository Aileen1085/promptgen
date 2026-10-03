"""Grouped SAM2 memory decoding for v10.3.

Frames inside one group observe the same group-start memory state and are sent
through the mask decoder as one batch. The final frame is committed to memory.
"""

from __future__ import annotations

from typing import Iterable, Sequence


def memory_groups(frame_ids: Sequence[int], group_size: int) -> tuple[tuple[int, ...], ...]:
    group_size = max(1, int(group_size))
    values = tuple(int(value) for value in frame_ids)
    return tuple(values[start:start + group_size] for start in range(0, len(values), group_size))


def bidirectional_memory_groups(
    *, depth: int, anchor: int, group_size: int
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    depth = int(depth)
    anchor = int(anchor)
    if depth <= 0 or not 0 <= anchor < depth:
        raise ValueError("anchor must be inside a positive depth")
    return (
        memory_groups(range(anchor + 1, depth), group_size),
        memory_groups(range(anchor - 1, -1, -1), group_size),
    )


def memory_grouped_decode_v103(
    adapter, video_cpu, bundle_cpu, token_outputs, anchor,
    args, device, training: bool,
):
    """Decode an ROI with one batched mask-decoder call per memory group."""

    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    from v10_sam2_memory_roi import (
        _amp,
        _compact_memory,
        _decode_external_prompt,
    )
    from v10_memory_input_grad import prepare_memory_features

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
        raise RuntimeError("v10.3 grouped decode requires the shared image-feature cache")
    logits = [None] * depth
    sparse_rows = [None] * depth
    checkpoint_depth = max(0, int(getattr(args, "decoder_checkpoint_min_frames", 96)))
    use_checkpoint = bool(getattr(args, "decoder_gradient_checkpoint", False)) or depth > checkpoint_depth

    def image_features(frame_ids: Iterable[int]):
        ids = tuple(int(value) for value in frame_ids)
        index_by_device = {}
        levels = []
        for value in image_views["levels"]:
            key = str(value.device)
            index = index_by_device.get(key)
            if index is None:
                index = torch.tensor(ids, dtype=torch.long, device=value.device)
                index_by_device[key] = index
            levels.append(value.index_select(0, index).to(device))
        vision = [value.flatten(2).permute(2, 0, 1).contiguous() for value in levels]
        positions = [
            value.to(device=device, dtype=feature.dtype).expand(-1, len(ids), -1)
            for value, feature in zip(image_views["position_templates"], vision)
        ]
        return levels, vision, positions, image_views["feature_sizes"]

    def prepare_frame(frame_idx, vision, positions, sizes, local, group_start_state, reverse, init):
        frame_vision = [value[:, local:local + 1] for value in vision]
        frame_position = [value[:, local:local + 1] for value in positions]
        high_res = [
            value.permute(1, 2, 0).reshape(1, value.size(2), *size)
            for value, size in zip(frame_vision[:-1], sizes[:-1])
        ] if len(frame_vision) > 1 else None
        with _amp(args, device):
            pix_feat = prepare_memory_features(
                sam, training=bool(training),
                frame_idx=int(frame_idx),
                is_init_cond_frame=bool(init),
                current_vision_feats=frame_vision[-1:],
                current_vision_pos_embeds=frame_position[-1:],
                feat_sizes=sizes[-1:],
                output_dict=group_start_state,
                num_frames=depth,
                track_in_reverse=bool(reverse),
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
                prompt_3d.to(device),
                coords[frame_idx:frame_idx + 1].to(device),
                None,
                role_ids=role_ids,
                type_ids=type_ids,
            )
            learned_sparse = per_frame_sparse[frame_idx:frame_idx + 1].to(
                device=device, dtype=adapter_sparse.dtype
            )
            sparse = torch.cat((learned_sparse, adapter_sparse), dim=1)
            dense = dense_prompt_volume[frame_idx:frame_idx + 1].to(device)
        return {
            "frame_idx": int(frame_idx),
            "frame_vision": frame_vision,
            "pix_feat": pix_feat,
            "high_res": high_res,
            "sparse": sparse,
            "dense": dense,
        }

    def decode_group(group, state, reverse, init=False):
        group = tuple(int(value) for value in group)
        group_start_state = state
        levels, vision, positions, sizes = image_features(group)
        prepared = [
            prepare_frame(frame_idx, vision, positions, sizes, local, group_start_state, reverse, init)
            for local, frame_idx in enumerate(group)
        ]
        batched_pix_feat = torch.cat([row["pix_feat"] for row in prepared], dim=0)
        batched_sparse = torch.cat([row["sparse"] for row in prepared], dim=0)
        batched_dense = torch.cat([row["dense"] for row in prepared], dim=0)
        batched_high_res = None
        if prepared[0]["high_res"] is not None:
            batched_high_res = [
                torch.cat([row["high_res"][level] for row in prepared], dim=0)
                for level in range(len(prepared[0]["high_res"]))
            ]

        def decoder_call(sparse_input):
            return _decode_external_prompt(
                sam,
                batched_pix_feat,
                batched_high_res,
                sparse_input,
                batched_dense,
                bool(args.object_score_gate),
                detach_high_res=bool(training),
                materialize_high_res=True,
            )

        with _amp(args, device):
            if training and not init and use_checkpoint:
                low, high, obj_ptr, obj_score = checkpoint(
                    decoder_call,
                    batched_sparse,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                low, high, obj_ptr, obj_score = decoder_call(batched_sparse)
        for local, row in enumerate(prepared):
            frame_idx = row["frame_idx"]
            logits[frame_idx] = low[local:local + 1].float() if training else high[local:local + 1].float().cpu()
            sparse_rows[frame_idx] = row["sparse"]

        commit_frame = group[-1]
        commit_local = len(group) - 1
        commit_row = prepared[commit_local]
        with _amp(args, device):
            compact = _compact_memory(
                sam,
                commit_row["frame_vision"][-1:],
                sizes[-1:],
                high[commit_local:commit_local + 1],
                obj_ptr[commit_local:commit_local + 1],
                obj_score[commit_local:commit_local + 1],
            )
        del levels, vision, positions, prepared
        return commit_frame, compact

    empty_state = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
    _, anchor_output = decode_group((int(anchor),), empty_state, False, init=True)
    group_size = max(1, int(getattr(
        args,
        "train_memory_group_size" if training else "eval_memory_group_size",
        1,
    )))
    forward_groups, reverse_groups = bidirectional_memory_groups(
        depth=depth, anchor=int(anchor), group_size=group_size
    )

    def run_direction(groups, reverse):
        state = {
            "cond_frame_outputs": {int(anchor): anchor_output},
            "non_cond_frame_outputs": {},
        }
        for group in groups:
            commit_frame, current = decode_group(group, state, reverse, init=False)
            state["non_cond_frame_outputs"][int(commit_frame)] = current

    run_direction(forward_groups, False)
    run_direction(reverse_groups, True)
    if any(value is None for value in logits) or any(value is None for value in sparse_rows):
        raise RuntimeError("v10.3 grouped decoder did not produce every ROI frame")
    return torch.cat(logits, dim=0), torch.cat(sparse_rows, dim=0)
