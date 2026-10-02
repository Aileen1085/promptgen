"""Compose the existing full-fine-tune launcher with the E360 CT13 protocol."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"launcher token must occur exactly once: {old!r}")
    return text.replace(old, new, 1)


def transform_launcher(text: str, root: Path) -> str:
    del root  # Paths in the generated launcher are resolved from its working directory.
    text = replace_once(
        text,
        'PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'PROJECT_ROOT="$(pwd -P)"',
    )
    text = replace_once(
        text,
        '"${PROJECT_ROOT}/v9_family_shared_cache_entry.py" ' + chr(92) +
        '\n  finetune_multisource_sam2_v9_2_full ' + chr(92),
        '"${PROJECT_ROOT}/sam2_v9_2_ct13_full_entry.py" ' + chr(92),
    )
    text = replace_once(
        text,
        'configs/v9_2_extended_split_4case_v2.json',
        'configs/ct13_v10_2_split_20260928.json',
    )
    text = replace_once(
        text,
        'configs/v9_2_extended_validation_4case_v2.json',
        'configs/ct13_v10_2_validation_four_new_sources4_classweight_v3_20260928.json',
    )
    text = replace_once(
        text,
        '--out-dir "${OUT_DIR}" --epochs 40 --validate-every 10',
        '--out-dir "${OUT_DIR}" --epochs "${EPOCHS:-80}" --validate-every 10',
    )
    text = replace_once(
        text,
        '  "$@"\n',
        '  --full-ft-lr-schedule plateau --full-ft-plateau-factor 0.5 ' + chr(92) +
        '\n  --full-ft-plateau-threshold 0.0001 --early-stop-patience 4 ' + chr(92) +
        '\n  --early-stop-min-delta 0.0001 ' + chr(92) +
        '\n  "$@"\n',
    )
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args()
    if extra[:1] == ["--"]:
        extra = extra[1:]
    root = Path(__file__).resolve().parents[1]
    template = transform_launcher((root / "run_train_sam2_v9_2_full.sh").read_text(), root)
    if args.dry_run:
        print(template)
        return
    for relative in (
        "configs/ct13_v10_2_split_20260928.json",
        "configs/ct13_v10_2_validation_four_new_sources4_classweight_v3_20260928.json",
    ):
        if not (root / relative).is_file():
            raise FileNotFoundError(root / relative)
    result = subprocess.run(["bash", "-s", "--", *extra], input=template, text=True, cwd=root, env=os.environ.copy())
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
