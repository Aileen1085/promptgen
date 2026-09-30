"""Generate, but do not execute, an exact-argv CT13 plateau continuation."""

import argparse
from pathlib import Path
import shlex


def last_option(argv, name):
    positions = [i for i, value in enumerate(argv[:-1]) if value == name]
    if not positions:
        raise ValueError(f"missing {name} in previous launcher")
    return argv[positions[-1] + 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-launcher", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--new-launcher", type=Path, required=True)
    args = parser.parse_args()

    lines = args.previous_launcher.read_text().splitlines()
    commands = [line[len("exec env "):] for line in lines if line.startswith("exec env ")]
    if len(commands) != 1:
        raise ValueError("expected exactly one exec env command")
    argv = shlex.split(commands[0])
    if "CT13_FAMILY=v10_2" not in argv[:3] or not any(
        Path(value).name == "ct13_training_entry.py" for value in argv
    ):
        raise ValueError("previous launcher is not the v10.2 CT13 entry")
    if Path(last_option(argv, "--multidataset-validation-json")) != args.validation:
        raise ValueError("validation protocol differs from completed run")
    if last_option(argv, "--gpu") != "5":
        raise ValueError("previous launcher did not use the verified physical GPU5")
    if last_option(argv, "--epochs") != "430":
        raise ValueError("previous launcher did not finish at E430")
    if not args.checkpoint.is_file() or not args.validation.is_file():
        raise FileNotFoundError("checkpoint or validation manifest is missing")
    if args.output_root.exists() or args.new_launcher.exists():
        raise FileExistsError("new output root or launcher already exists")

    argv += [
        "--resume-checkpoint", str(args.checkpoint),
        "--resume-lr-scale", "1.0", "--resume-new-run",
        "--out-dir", str(args.output_root),
        "--epochs", "510", "--validate-every", "10",
        "--no-validate-before-train",
        "--plateau-early-stop-patience", "4",
        "--plateau-lr-patience", "2",
        "--plateau-lr-factor", "0.5",
        "--plateau-min-lr-ratio", "0.1",
        "--plateau-min-delta", "0.001",
    ]
    text = "#!/bin/bash\nset -euo pipefail\ncd /home/bedicloud/sharestore2/al/v9\n"
    text += "exec env " + shlex.join(argv) + "\n"
    with args.new_launcher.open("x") as file:
        file.write(text)
    print(args.new_launcher)


if __name__ == "__main__":
    main()
