"""Run independent v10.2 evaluation with the original three-source prompt protocol.

This thin entry reuses the sharded full-volume v10.2 evaluator.  It changes only
the evaluation prompts and SAT text back to the three-source training settings.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from types import MethodType
from typing import Callable, Mapping


def plain_semantics(rows) -> dict[int, str]:
    result = {}
    for row in rows:
        class_id = int(row["global_class_id"])
        value = str(row.get("semantic_text") or "").strip()
        if not value or class_id in result:
            raise ValueError(f"missing or duplicate three-source semantic text: {class_id}")
        result[class_id] = value
    return result


def install_training_prompts_and_reserved_roi(
    dataset,
    *,
    source: str,
    tasks,
    audit: dict,
    background_mode: str,
    reserve_bounds: Callable | None = None,
    install_cache_policy: Callable | None = None,
):
    import numpy as np

    if background_mode != "scribble":
        raise ValueError("three-source validation requires foreground/background scribble")
    from v10.infer import infer_eight_source_adaptive_v10_2 as base
    if reserve_bounds is None:
        from v10.infer.infer_amos_sam2_promptgen_v10 import reserved_prompt_roi_bounds
        reserve_bounds = lambda bounds, depth: reserved_prompt_roi_bounds(
            bounds, canonical_depth=depth
        )
    if install_cache_policy is None:
        install_cache_policy = base.install_read_only_dynamic_roi_cache_policy
    inner = base._find_inner_dataset(dataset)
    install_cache_policy(inner)
    by_image_class = {
        (os.path.normpath(str(row.get("image_path", row.get("image")))),
         int(row["global_class_id"])): [source, str(row["case_id"]), int(row["global_class_id"])]
        for row in tasks
    }
    original_groups = inner._build_prompt_slice_groups
    original_roi = inner._prompt_derived_roi_indices

    def training_groups(self, index, class_id):
        key = by_image_class[(os.path.normpath(str(self.image_paths[int(index)])), int(class_id))]
        groups = original_groups(index, class_id)
        audit.setdefault("loaded_task_keys", []).append(key)
        return groups

    def reserve_roi(self, target_mask):
        initial = np.asarray(original_roi(target_mask), dtype=np.int32)
        if not initial.size or np.any(np.diff(initial) != 1):
            raise RuntimeError("initial training-prompt ROI must be contiguous")
        self._v10_2_initial_prompt_roi_indices = initial.copy()
        depth = int(np.asarray(target_mask).shape[0])
        start, end = reserve_bounds((int(initial[0]), int(initial[-1]) + 1), depth)
        audit.setdefault("roi_build_calls", []).append({
            "initial_bounds": [int(initial[0]), int(initial[-1]) + 1],
            "reserved_bounds": [int(start), int(end)],
        })
        return np.arange(start, end, dtype=np.int32)

    inner._build_prompt_slice_groups = MethodType(training_groups, inner)
    inner._prompt_derived_roi_indices = MethodType(reserve_roi, inner)
    return dataset, inner


def main() -> None:
    options = argparse.ArgumentParser(add_help=False)
    options.add_argument("--three-source-split-json", required=True)
    known, remaining = options.parse_known_args()
    if "--gpu" not in remaining:
        raise ValueError("--gpu is required before CUDA initialization")
    os.environ["CUDA_VISIBLE_DEVICES"] = remaining[remaining.index("--gpu") + 1]
    split = json.loads(Path(known.three_source_split_json).read_text(encoding="utf-8"))
    semantic_map = plain_semantics(split["class_catalog"])

    from v10.infer import infer_eight_source_adaptive_v10_2 as base
    import v10.finetune_totalseg_amos_magic_v10_2_joint as entry

    original_build = base.build_selected_protocol

    def build_three_source_protocol(base_split: Mapping, tasks):
        selected_split, validation = original_build(base_split, tasks)
        selected_split["source_order"] = ["totalseg", "amos", "magic"]
        return selected_split, validation

    def install(dataset, *, source, tasks, prompt_index, prompt_root, audit, background_mode):
        del prompt_index, prompt_root
        return install_training_prompts_and_reserved_roi(
            dataset, source=source, tasks=tasks, audit=audit,
            background_mode=background_mode,
        )

    base.build_selected_protocol = build_three_source_protocol
    base._install_exact_prompts_and_reserved_roi = install
    def protocol_name(mode):
        if mode != "scribble":
            raise ValueError("three-source requires scribble")
        return "v10_2_three_source_fixed_fg_bg_scribble_final_trigger"

    base.evaluation_prompt_protocol = protocol_name
    entry.conditioning_text_from_metadata = lambda row: semantic_map[int(row["global_class_id"])]
    sys.argv = [sys.argv[0], *remaining]
    base.main()


if __name__ == "__main__":
    main()
