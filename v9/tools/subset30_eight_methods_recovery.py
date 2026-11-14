"""Resume the frozen subset30 evaluation after prompt handoff errors."""
import argparse
import os
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import subset30_eight_methods as run


def configure():
    # The cache reader has absolute imports to sibling modules. Keep the path
    # process-local and behind the installed nnInteractive bundle if present.
    sys.path.append(str(ROOT / "nninteractive_test/infer"))
    if "worker" not in sys.argv[1:2]:
        return
    name = sys.argv[2]
    if not name.startswith("v10_2_"):
        return
    import v10.infer.infer_eight_source_adaptive_v10_2 as evaluator
    from infer.infer_eight_source_adaptive_v9_2 import PROMPT_PROTOCOL

    original = evaluator.build_prompt_result_index

    def index(tasks, rows):
        # The frozen manifest was produced by the original FUMPE-style
        # generator. Attach its protocol identifier to the in-memory refs.
        verified = {}
        _, frozen = run.job_rows(name)
        for row in frozen:
            verified[run.common.task_key(row)] = row
        bound = []
        for row in rows:
            ref = verified[run.common.task_key(row)]
            if row != ref:
                raise ValueError("v10.2 reference differs from its frozen prompt record")
            run.read_prompt(ref)
            bound.append(dict(row, prompt_protocol=PROMPT_PROTOCOL))
        return original(tasks, bound)

    evaluator.build_prompt_result_index = index


def main():
    configure()
    if sys.argv[1:2] == ["worker"]:
        run.main()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("schedule", "verify", "drain"))
    parser.add_argument("--scheduler-pid", type=int)
    args = parser.parse_args()
    if args.command == "verify":
        manifest = run.common.read(run.OUT / "manifest.json")
        for source in run.SOURCES:
            row = next(row for row in manifest["tasks"] if row["source_name"] == source)
            if len(run.read_prompt(row)["background_points_dhw"]) != 4:
                raise ValueError("invalid shared background prompt")
        print("RECOVERY_PROMPT_IMPORT_OK", flush=True)
        return
    import fcntl
    with (run.OUT / "recovery_schedule.lock").open("a") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command == "drain":
            drain(args.scheduler_pid)
        print("RECOVERY_WAIT_ORIGINAL_QUEUE", flush=True)
        with (run.OUT / "schedule.lock").open("a") as prior:
            fcntl.flock(prior, fcntl.LOCK_EX)
        original_status = run.common.read(run.OUT / "status.json")
        run.common.write(run.OUT / f"status_before_recovery_{time.time_ns()}.json", original_status)
        if (run.OUT / "merged_metrics.json").is_file():
            print("EXISTING_FINAL_RESULTS", flush=True)
            return
        # Redirect only the child entry path. Source verification still uses
        # the original frozen evaluator file.
        original_popen = run.subprocess.Popen
        def popen(args, **kwargs):
            argv = list(args)
            if "worker" in argv and str(run.__file__) in argv:
                argv[argv.index(str(run.__file__))] = __file__
            return original_popen(argv, **kwargs)
        run.subprocess.Popen = popen
        try:
            run.schedule()
        finally:
            run.subprocess.Popen = original_popen


def drain(pid):
    if not pid or pid <= 1:
        raise ValueError("explicit original scheduler PID required")
    parent = Path("/proc") / str(pid)
    if not parent.exists():
        return
    argv = (parent / "cmdline").read_bytes().split(b"\0")
    if argv[-3:-1] != [b"tools/subset30_eight_methods.py", b"schedule"]:
        raise ValueError("PID is not the original subset30 scheduler")
    if (parent / "cwd").resolve() != ROOT:
        raise ValueError("original scheduler working directory mismatch")
    os.kill(pid, signal.SIGSTOP)
    try:
        print("ORIGINAL_DISPATCH_PAUSED", pid, flush=True)
        while True:
            children = (parent / "task" / str(pid) / "children").read_text().split()
            running = []
            for child in children:
                try:
                    status = (Path("/proc") / child / "stat").read_text().rsplit(")", 1)[1].split()[0]
                except FileNotFoundError:
                    continue
                if status != "Z":
                    running.append(child)
            if not running:
                break
            print("DRAINING_ORIGINAL_WORKERS", running, flush=True)
            time.sleep(30)
        print("ORIGINAL_WORKERS_DRAINED", flush=True)
        os.kill(pid, signal.SIGTERM)
    finally:
        try:
            os.kill(pid, signal.SIGCONT)
        except ProcessLookupError:
            pass


if __name__ == "__main__":
    main()
