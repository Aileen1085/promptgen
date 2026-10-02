"""Run the existing v10.2 encoder-LoRA/full-decoder method on the audited CT13 data."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import sys

from ct13_training_entry import install_v10_protocol


_NATIVE_WORKER = None


def install_protocol(split_path, importer=importlib.import_module):
    split = json.loads(Path(split_path).read_text(encoding="utf-8"))
    if split.get("protocol_revision") != "ct13_v10_2_20260928":
        raise ValueError("CT13 v10.2 manifest family/revision mismatch")
    from ct_thirteen_source_protocol import SOURCE_ORDER

    protocol = importer("v9_2_extended_protocol")
    joint = importer("finetune_totalseg_amos_magic_v10_2_joint")
    data = importer("v10_2_data")
    protocol.SOURCE_ORDER = SOURCE_ORDER
    install_v10_protocol(joint, data, split)
    return split


def run_worker(rank, world, port, stamp, args, importer=importlib.import_module):
    install_protocol(args.multidataset_split_json, importer=importer)
    native = _NATIVE_WORKER
    if native is None:
        native = importer("finetune_multisource_sam2_v10_2_encoder_decoder")._worker
    native(rank, world, port, stamp, args)


def main():
    if os.environ.get("CT13_FAMILY") != "v10_2":
        raise ValueError("set CT13_FAMILY=v10_2 explicitly")
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / "v10"))
    full = importlib.import_module("finetune_multisource_sam2_v10_2_encoder_decoder")
    if "--help" in sys.argv or "-h" in sys.argv:
        full.main()
        return
    from ct13_training_entry import _split_path

    install_protocol(_split_path())
    global _NATIVE_WORKER
    _NATIVE_WORKER = full._worker
    full._worker = run_worker
    try:
        full.main()
    finally:
        full._worker = _NATIVE_WORKER
        _NATIVE_WORKER = None


if __name__ == "__main__":
    main()
