"""Run only CT13's five new fixed-validation sources with the existing nnInteractive protocol."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
from unittest.mock import patch


EXPECTED_SOURCE_COUNTS = {
    "totalseg": 115,
    "amos": 45,
    "magic": 35,
    "msd_task07": 8,
    "msd_task10": 4,
    "msd_task08": 8,
    "parse2022": 4,
    "topcow2024_cta": 26,
    "lndb": 4,
    "msd_task06": 4,
    "covid19_20": 4,
    "kits23": 6,
    "lnq2023_lite": 4,
}
NEW_SOURCES = tuple(list(EXPECTED_SOURCE_COUNTS)[8:])
SOURCE_ORDER = tuple(EXPECTED_SOURCE_COUNTS)


def partition_validation_tasks(
    manifest: dict,
    *,
    selected_sources: tuple[str, ...] = NEW_SOURCES,
) -> set[tuple[str, str, int]]:
    """Check the immutable CT13 manifest and return only new-source task keys."""
    if tuple(manifest.get("source_order", ())) != SOURCE_ORDER:
        raise ValueError("CT13 source order differs from the fixed protocol")
    selected = set(selected_sources)
    if not selected or not selected <= set(NEW_SOURCES):
        raise ValueError("only the five new sources may be evaluated here")
    rows = manifest.get("tasks", [])
    if len(rows) != sum(EXPECTED_SOURCE_COUNTS.values()):
        raise ValueError(f"CT13 task count is {len(rows)}, expected 267")
    keys = [(str(row["dataset"]), str(row["case_id"]), int(row["global_class_id"])) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate case-class task in CT13 validation manifest")
    counts = Counter(key[0] for key in keys)
    if dict(counts) != EXPECTED_SOURCE_COUNTS:
        raise ValueError(f"CT13 per-source task counts differ: {dict(counts)}")
    na = {int(row["global_class_id"]) for row in manifest.get("not_applicable", [])}
    if na != {40, 41}:
        raise ValueError(f"CT13 N/A class set differs: {sorted(na)}")
    return {key for key in keys if key[0] in selected}


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    infer_root = Path(__file__).resolve().parent
    for candidate in (str(project_root), str(infer_root)):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)

    import eight_source_adaptive_protocol as protocol
    import infer_eight_source_adaptive_nninteractive as evaluator

    args = evaluator.parse_args()
    if not args.split_json or not args.validation_json:
        raise ValueError("--split-json and --validation-json must name the fixed CT13 files")
    if args.task_manifest:
        raise ValueError("--task-manifest is not supported for this CT13 fixed split")
    selected = tuple(value.strip() for value in args.source.split(",") if value.strip()) or NEW_SOURCES
    args.source = ",".join(selected)
    manifest = json.loads(Path(args.validation_json).read_text(encoding="utf-8"))
    expected_keys = partition_validation_tasks(manifest, selected_sources=selected)
    split = json.loads(Path(args.split_json).read_text(encoding="utf-8"))
    if tuple(split.get("source_order", ())) != SOURCE_ORDER:
        raise ValueError("CT13 split source order differs from validation manifest")

    # The established loader otherwise hard-codes eight sources. The extension is
    # restricted to this process; the evaluator and all metric/prompt code are reused.
    with patch.object(protocol, "EIGHT_SOURCES", SOURCE_ORDER):
        loaded = protocol.load_eight_source_tasks(
            args.split_json, args.validation_json, sources=selected, require_files=True
        )
        loaded_keys = {(task.source_name, task.case_id, task.global_class_id) for task in loaded}
        if loaded_keys != expected_keys:
            raise ValueError("loaded task keys differ from the fixed CT13 manifest")
        if args.max_tasks:
            raise ValueError("--max-tasks would truncate the fixed CT13 evaluation")
        evaluator.run_inference(args)


if __name__ == "__main__":
    main()
