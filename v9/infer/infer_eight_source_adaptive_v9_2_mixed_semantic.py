"""Run exact-prompt eight-source v9.2 inference with mixed semantic text."""

from __future__ import annotations

import importlib
import sys


ENTRYPOINT_MODULE = (
    "finetune_multisource_sam2_v9_2_precision_mixed_semantic_eval"
)


def main() -> None:
    entrypoint = importlib.import_module(ENTRYPOINT_MODULE)
    sys.modules["finetune_multisource_sam2_v9_2_precision"] = entrypoint
    from infer.infer_eight_source_adaptive_v9_2 import main as original_main

    original_main()


if __name__ == "__main__":
    main()
