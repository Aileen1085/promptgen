"""Additive import-order repair over the immutable AMOS prompt runtime.

CT13 captures the original eight-source order when it is imported. Load that
immutable expansion before the delegated runtime changes the legacy protocol's
order to thirteen. No protocol constants, model code, or validation are replaced.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools import amos_prompt_ablation_runtime as _native


NATIVE_RUNTIME_SHA256 = "5387c458a6331491fd258718b8a2f370a594ec73a38db546702a116900f0b18e"
EXPECTED_SOURCE_ORDER = (
    "totalseg", "amos", "magic", "msd_task07", "msd_task10", "msd_task08",
    "parse2022", "topcow2024_cta", "lndb", "msd_task06", "covid19_20",
    "kits23", "lnq2023_lite",
)

# Direct aliases preserve the original functions' single private native globals.
Session = _native.Session
evaluate = _native.evaluate
validate_record = _native.validate_record
validate_control = _native.validate_control
METRICS = _native.METRICS


def _preload_ct13(split):
    # This import must precede _native.initialize, exactly as in the established
    # ct13_full_validation_eval entry. It intentionally does not reset modules.
    import ct13_training_entry as entry
    import ct_thirteen_source_protocol as expansion

    if (tuple(expansion.OLD_SOURCE_ORDER) != EXPECTED_SOURCE_ORDER[:8]
            or tuple(expansion.SOURCE_ORDER) != EXPECTED_SOURCE_ORDER
            or tuple(entry.SOURCE_ORDER) != EXPECTED_SOURCE_ORDER):
        raise ValueError("CT13 import order already contaminated; use a fresh worker process")
    if tuple(split.get("source_order", ())) != EXPECTED_SOURCE_ORDER:
        raise ValueError("CT13 split source order differs from the frozen thirteen-source order")
    entry._ratios(split)  # Keep the real native ratio and source validation.
    return dict(captured_original_source_order=list(expansion.OLD_SOURCE_ORDER),
        source_order=list(entry.SOURCE_ORDER), native_ratios_passed=True,
        preloaded_before_native_initialize=True, native_runtime_sha256=NATIVE_RUNTIME_SHA256)


def initialize(job, paths):
    """Preload the immutable expansion, then invoke the signed runtime directly."""
    source_path = Path(paths["root"]).resolve()/"tools/amos_prompt_ablation_runtime.py"
    if (Path(_native.__file__).resolve() != source_path
            or hashlib.sha256(source_path.read_bytes()).hexdigest() != NATIVE_RUNTIME_SHA256):
        raise ValueError("signed native runtime SHA/path mismatch")
    split_path = _native._resolve(paths, job, "base_split_json", "configs/ct13_v10_2_split_20260928.json")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    proof = _preload_ct13(split)
    session = _native.initialize(job, paths)
    session.input_audit["ct13_import_order_restore"] = proof
    return session
