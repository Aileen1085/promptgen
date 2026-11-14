"""Run LNQ first and restrict enhanced SegRap to v10.2."""
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import subset30_eight_methods as run
from tools import subset30_eight_methods_recovery as recovery


def permitted(job):
    return job["source"] != "segrap_cect" or job["method"] == "v10_2"


def summarize(jobs, tasks):
    groups = defaultdict(list)
    for job in jobs:
        record = run.common.read(run.OUT / "runs" / job["name"] / "complete.json")
        rows = record["per_case_class"]
        expected = {run.common.task_key(row): row["prompt_sha256"] for row in job["tasks"]}
        actual = {run.common.task_key(row): row["prompt_sha256"] for row in rows}
        if record["method"] != job["method"] or len(rows) != len(expected) or actual != expected:
            raise ValueError(f"result mismatch for {job['name']}")
        for row in rows:
            groups[job["source"], row["group"], job["method"]].append(row)

    def means(rows):
        return {metric: math.fsum(row[metric] for row in rows) / len(rows) for metric in run.METRICS}

    summary = {}
    for (source, group, method), rows in groups.items():
        expected = [row for row in tasks if row["source_name"] == source and row["group"] == group]
        if Counter(map(run.common.task_key, expected)) != Counter(map(run.common.task_key, rows)):
            raise ValueError(f"incomplete group {source}/{group}/{method}")
        result = dict(cases=len({row["patient_id"] for row in rows}), tasks=len(rows))
        if method == "v10_2":
            if any(not row.get("logits_sha256") or [f["floor"] for f in row["floors"]] != list(run.FLOORS)
                   for row in rows):
                raise ValueError("incomplete shared-logits grid")
            grid = {floor: means([row["floors"][i] for row in rows]) for i, floor in enumerate(run.FLOORS)}
            best = max(run.FLOORS, key=lambda f: grid[f]["dice"])
            result.update(floors=grid, best_floor=best, best_metrics=grid[best])
        else:
            result["metrics"] = means(rows)
        summary.setdefault(source, {}).setdefault(group, {})[method] = result
    run.common.write(run.OUT / "merged_metrics_scoped.json", dict(summary=summary))


def main():
    recovery.configure()
    if sys.argv[1:] != ["schedule"]:
        run.main()
        return
    original_jobs = run.jobs_for
    original_popen = run.subprocess.Popen
    original_write = run.common.write
    rank = {"lnq_subset30": 0, "segrap_ncct": 1, "segrap_cect": 2}
    tasks = run.common.read(run.OUT / "manifest.json")["tasks"]
    required = Counter()
    for job in original_jobs(tasks):
        if permitted(job):
            required[job["method"]] += len(job["tasks"])

    def jobs_for(rows):
        return sorted((job for job in original_jobs(rows) if permitted(job)),
                      key=lambda j: (rank[j["source"]], j["name"] != j["smoke"]))

    def popen(args, **kwargs):
        if "worker" in args and str(run.__file__) in args:
            args = [__file__ if arg == str(run.__file__) else arg for arg in args]
        return original_popen(args, **kwargs)

    def write(path, payload):
        if Path(path) == run.OUT / "status.json":
            payload = dict(payload)
            payload.pop("total_tasks_per_method", None)
            payload["required_tasks_by_method"] = dict(required)
            payload["priority_order"] = list(rank)
        return original_write(path, payload)

    run.jobs_for = jobs_for
    run.summarize = summarize
    run.subprocess.Popen = popen
    run.common.write = write
    print("SCOPED_QUEUE_LNQ_FIRST", dict(required), flush=True)
    run.schedule()


if __name__ == "__main__":
    main()
