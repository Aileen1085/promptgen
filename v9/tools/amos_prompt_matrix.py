"""Fixed AMOS subset prompt evaluation using the existing native inference paths."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from infer.native_sammed3d_prompts import sample_points
from tools import native_ct13_eval as native

OUT = ROOT / "output/amos_prompt_matrix_20261120"
SPLIT = ROOT / "configs/ct13_v10_2_split_20260928.json"
SOURCE = ROOT / "output/native_sam2_medsam2_ct13_full6094_20261007_gridfix/manifest.json"
REFERENCE = ROOT / "output/amos_v102_e490_floor070_prompt_ablations_20261031_reusefast_ownerfix"
SAM_ARMS = ("fg1_bg2", "fg2_bg2", "fg4_bg2", "fg2_bg1", "fg2_bg4")
ARMS = {
    "promptgen": ("fg_scribble_bg_scribble", "fg_scribble_bg_point", "fg_bbox_bg_point", "fg_bbox_bg_scribble"),
    "sam2": SAM_ARMS,
    "medsam2": SAM_ARMS,
    "nninteractive": ("bg_point2", "bg_point1", "bg_point4", "bg_scribble"),
    "segvol": ("official_fps2", "official_fps4", "official_axis4", "physical_fps4", "official_fps4_pe_aligned"),
    "vista3d": ("fit_fps4", "fixed_fps2", "fixed_fps4", "fixed_axis4"),
    "sammed3d": ("fps2", "fps4", "fps8", "axis4", "fps4_stretch"),
    "pam": ("coronal", "sagittal", "dual_or", "coronal_prob", "sagittal_prob", "coronal_margin",
            "sagittal_margin", "dual_prob", "dual_margin"),
}
PG_GPUS = {2, 4, 5, 6}
MINIMUM_MIB = 30720
PARALLEL_ARMS = {"sam2", "medsam2", "segvol", "vista3d", "sammed3d"}


def arm_groups(method):
    arms = ARMS[method]
    if method not in PARALLEL_ARMS:
        return (arms,)
    split = (len(arms) + 1) // 2
    return arms[:split], arms[split:]


def select_tasks(rows):
    amos = [row for row in rows if row["source_name"] == "amos"]
    cases = sorted({row["case_id"] for row in amos})
    if len({native.key(row) for row in amos}) != len(amos):
        raise ValueError("duplicate fixed holdout task")
    if len(cases) != 100 or len(amos) != 1485 or any(row["split_role"] != "validation" for row in amos):
        raise ValueError("expected the fixed AMOS holdout of 100 CTs and 1485 tasks")
    return sorted((row for row in amos if row["case_id"] in cases[:10]), key=native.key)


def vertical_line(target_plane):
    """The established AMOS vertical BG rule, retaining the original plane identity."""
    hits = np.argwhere(target_plane)
    if not len(hits):
        raise ValueError("empty target on an original scribble plane")
    height, width = target_plane.shape
    row_lo, col_lo = hits.min(axis=0)
    row_hi, col_hi = hits.max(axis=0)
    margin = max(2, int(round(min(target_plane.shape) * .08)))
    rows = slice(max(0, row_lo - margin), min(height, row_hi + margin + 1))
    columns = [c for c in (col_lo - margin, col_hi + margin) if 0 <= c < width]
    columns.extend((0, width - 1, max(0, col_lo - 1), min(width - 1, col_hi + 1)))
    mask = np.zeros_like(target_plane, dtype=bool)
    for c in dict.fromkeys(columns):
        available = ~target_plane[rows, c]
        if available.any():
            mask[rows, c] = available
            return mask
    outside = np.argwhere(~target_plane)
    if not len(outside):
        raise ValueError("no background on an original scribble plane")
    column = outside[len(outside) // 2, 1]
    mask[:, column] = ~target_plane[:, column]
    return mask


def background_supports(target, planes):
    coronal, sagittal = planes
    coronal_line = np.zeros_like(target, dtype=bool)
    sagittal_line = np.zeros_like(target, dtype=bool)
    coronal_line[:, :, coronal] = vertical_line(target[:, :, coronal])
    sagittal_line[:, sagittal, :] = vertical_line(target[:, sagittal, :])
    return coronal_line, sagittal_line


def face_coordinates(support, axis, index):
    points = np.argwhere(np.take(support, index, axis=axis))
    return np.insert(points, axis, index, axis=1)


def sample_faces(supports, spacing, cap, sampler, minimum_mm, *, planes=None):
    if planes is None:
        coordinates = tuple(np.argwhere(support) for support in supports)
    else:
        coordinates = tuple(face_coordinates(support, axis, plane)
                            for support, axis, plane in zip(supports, (2, 1), planes))
    selected = []
    for candidates in coordinates:
        if selected and len(selected[0]):
            previous = selected[0]
            distance = np.sum(((candidates[:, None] - previous) * spacing) ** 2, axis=2)
            candidates = candidates[np.all((distance >= minimum_mm ** 2) & (distance > 0), axis=1)]
        selected.append(sample_points(candidates, spacing, cap, minimum_mm, sampler).astype(np.int32))
    return tuple(selected)


def points_digest(supports, planes):
    # Preserve the original plane assignment even at the intersection.
    coordinates = [face_coordinates(mask, axis, plane).tolist()
                   for mask, axis, plane in zip(supports, (2, 1), planes)]
    return sha256(json.dumps(coordinates, separators=(",", ":")).encode()).hexdigest()


@dataclass
class Prompts:
    ct: np.ndarray
    target: np.ndarray
    foreground: tuple
    background: tuple
    spacing: tuple
    planes: tuple
    cache_name: str
    cache_sha256: str
    ct_name: str
    background_sha256: str

    def points(self, role, cap, sampler="fps", minimum_mm=3.0):
        supports = {"fg": self.foreground, "bg": self.background}[role]
        return sample_faces(supports, self.spacing, cap, sampler, minimum_mm, planes=self.planes)


def stat_identity(path):
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns]


class CaseLoader:
    def __init__(self):
        self.identity = None
        self.files = None
        self.base = native.base_module()

    def __call__(self, task):
        cache_name = task["prompt_cache_name"]
        if Path(cache_name).name != cache_name:
            raise ValueError("prompt package name must be local to the cache")
        path = native.CACHE / "metadata/shared_adaptive_prompt_v1/amos" / cache_name
        digest = native.sha(path)
        with np.load(path, allow_pickle=False) as package:
            ct_name = str(package["case_image_name"].item())
            size = tuple(int(v) for v in package["shape_dhw"])
            planes = tuple(int(v) for v in package["slices"].reshape(2))
            foreground = tuple(np.unpackbits(package[p + "_bits"].astype(np.uint8),
                              count=math.prod(size), bitorder="little").reshape(size).astype(bool)
                               for p in ("coronal", "sagittal"))
        if Path(ct_name).name != ct_name:
            raise ValueError("shared CT name must be local to the case cache")
        identity = (task["case_id"], task["image"], task["label"], ct_name)
        if self.identity != identity:
            image, affine = self.base.canonical(task["image"])
            mask, mask_affine = self.base.canonical(task["label"])
            shape = tuple(image.shape[i] for i in (2, 0, 1))
            if mask.shape != image.shape:
                raise ValueError("CT/GT array dimensions differ")
            native.assert_compatible_grid(affine, mask_affine, shape, image_qform=image.get_qform(),
                                          label_qform=mask.get_qform())
            ct_path = native.CACHE / "_case_images_v1" / ct_name
            ct = np.load(ct_path, mmap_mode="r", allow_pickle=False)
            if ct.shape != shape or ct.dtype != np.float32:
                raise ValueError("shared CT is not canonical raw float32 DHW")
            self.spacing = tuple(np.linalg.norm(affine[:3, :3], axis=0))
            self.labels = np.asarray(mask.dataobj).transpose(2, 0, 1)
            self.identity, self.ct = identity, ct
            self.files = dict(ct=stat_identity(ct_path), image=stat_identity(Path(task["image"])),
                              label=stat_identity(Path(task["label"])))
        target = self.labels == task["local_class_id"]
        for fg, axis, plane in zip(foreground, (2, 1), planes):
            if fg.shape != target.shape or not 0 <= plane < target.shape[axis]:
                raise ValueError("cached foreground geometry does not match the target")
            points = face_coordinates(fg, axis, plane)
            if not len(points) or not target[tuple(points.T)].all():
                raise ValueError("cached foreground is empty or does not align with the target")
            if np.count_nonzero(fg) != len(points):
                raise ValueError("foreground is outside its original selected plane")
        background = background_supports(target, planes)
        return Prompts(self.ct, target, foreground, background, self.spacing, planes, cache_name,
                       digest, ct_name, points_digest(background, planes))


def prompt_identity(prompts, file_stats):
    return dict(foreground_sha256=prompts.cache_sha256, background_sha256=prompts.background_sha256,
                ct_name=prompts.ct_name, planes=list(prompts.planes), spacing=list(prompts.spacing),
                files=file_stats)


def validate_arms(rows, method, group=None):
    expected = ARMS[method] if group is None else arm_groups(method)[group]
    if Counter(row["arm"] for row in rows) != Counter(expected):
        raise ValueError(f"incomplete, extra or duplicate conditions for {method}")
    for row in rows:
        values = row["metrics"]
        if set(values) != set(native.METRICS) or not all(math.isfinite(v) for v in values.values()):
            raise ValueError("incomplete or nonfinite ten-metric result")


def code_binding():
    sources = [str(Path(__file__).relative_to(ROOT))]
    sources.extend(f"tools/amos_prompt_matrix_{suffix}.py" for suffix in ("native", "interactive", "promptgen"))
    sources.extend(("tools/native_ct13_eval.py", "tools/native_sam2_physical_eval.py", "tools/fumpe_six_baselines.py",
                    "tools/native_segvol_pilot.py", "tools/native_vista3d_pilot.py", "tools/native_sammed3d_pilot.py",
                    "tools/pam_soft_fusion.py", "tools/pam_amos_three_view.py", "tools/four_source_eval.py",
                    "infer/native_sam2_physical_prompts.py", "infer/native_sammed3d_prompts.py", "utils/metrics.py",
                    "nninteractive_test/infer/adaptive_prompt_cache.py"))
    return {path: native.sha(ROOT / path) for path in sources}


def prepare():
    manifest = OUT / "protocol.json"
    if manifest.exists():
        raise FileExistsError("the AMOS prompt matrix is already prepared")
    rows = select_tasks(native.read(SOURCE)["tasks"])
    parent_keys = {native.key(row) for row in native.read(REFERENCE / "manifest.json")["tasks"]}
    if len(rows) != 149 or not all(native.key(row) in parent_keys for row in rows):
        raise ValueError("ten-case selection differs from the fixed AMOS reference")
    split = native.read(SPLIT)["datasets"]["amos"]
    selected = {row["case_id"] for row in rows}
    if (not selected <= {row["case_id"] for row in split["val"]}
            or selected & {row["case_id"] for row in split["train"]}):
        raise ValueError("fixed AMOS holdout identity mismatch")
    loader = CaseLoader()
    for row in rows:
        prompts = loader(row)
        if any(len(points) != 2 for points in prompts.points("bg", 2)):
            raise ValueError(f"two distinct BG points per face unavailable for {native.key(row)}")
        row["prompt_identity"] = prompt_identity(prompts, loader.files)
    payload = dict(tasks=rows, cases=sorted(selected), methods={m: list(arms) for m, arms in ARMS.items()},
                   groups={m: [list(arms) for arms in arm_groups(m)] for m in ARMS},
                   sources=code_binding(), split_sha=native.sha(SPLIT), manifest_sha=native.sha(SOURCE),
                   s21_protocol_sha=native.sha(REFERENCE / "protocol.json"))
    native.write(manifest, payload)
    print("PREPARED", len(selected), len(rows), len(rows) * sum(map(len, ARMS.values())), flush=True)


def protocol():
    return native.read(OUT / "protocol.json")


def verify_inputs(payload):
    if (payload["sources"] != code_binding() or payload["split_sha"] != native.sha(SPLIT)
            or payload["manifest_sha"] != native.sha(SOURCE)
            or payload["s21_protocol_sha"] != native.sha(REFERENCE / "protocol.json")):
        raise ValueError("signed input or runtime changed")


def gpu_inventory():
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.free",
                                      "--format=csv,noheader,nounits"], text=True)
    return {int(i): (uuid, int(free)) for i, uuid, free in
            ([p.strip() for p in line.split(",")] for line in output.splitlines())}


def task_file(method, task, group):
    return OUT / "records" / method / task["case_id"] / str(group) / (str(task["global_class_id"]) + ".json")


def certificate_file(method, case, group):
    return OUT / "cases" / method / (f"{case}_{group}.json")


def read_record(method, task, group, signed_sha):
    record = native.read(task_file(method, task, group))
    if record["task"] != list(native.key(task)) or record["protocol_sha256"] != signed_sha:
        raise ValueError("case record does not match the frozen input")
    if record["method"] != method or record["group"] != group or record["prompt_identity"] != task["prompt_identity"]:
        raise ValueError("case record belongs to a different method or prompt")
    validate_arms(record["rows"], method, group)
    return record


def worker(method, case_id, group, gpu, uuid, attempt, batch):
    import torch

    payload = protocol()
    verify_inputs(payload)
    signed_sha = native.sha(OUT / "protocol.json")
    initial = gpu_inventory().get(gpu)
    if initial is None or initial[0] != uuid or initial[1] < MINIMUM_MIB:
        raise SystemExit(75)
    current = str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-")
    if current != uuid.removeprefix("GPU-") or torch.ones(1, device="cuda").item() != 1:
        raise ValueError("CUDA device did not match the physical GPU")
    if method == "promptgen" and gpu not in PG_GPUS:
        raise ValueError("PromptGen worker GPU outside signed runtime policy")
    torch.set_num_threads(2)
    os.environ["V9_SHARED_CACHE_WRITE_POLICY"] = "read_only"
    workdir = OUT / "attempts" / method / case_id / str(group) / f"{attempt:03d}"
    workdir.mkdir(parents=True, exist_ok=False)
    loader = CaseLoader()
    metric = loader.base.metric_module().binary_prompt_metrics_from_masks
    tasks = [row for row in payload["tasks"] if row["case_id"] == case_id]
    if not tasks:
        raise ValueError("case missing from the signed AMOS subset")
    if method == "promptgen":
        from tools.amos_prompt_matrix_promptgen import PromptGen
        model = PromptGen(case_id, gpu, uuid, workdir, batch=batch)
    elif method in ("pam", "nninteractive"):
        from tools.amos_prompt_matrix_interactive import InteractivePredictor
        model = InteractivePredictor(method, gpu)
    else:
        from tools.amos_prompt_matrix_native import NativePredictor
        model = NativePredictor(method)
    for task in tasks:
        path = task_file(method, task, group)
        if path.exists():
            read_record(method, task, group, signed_sha)
            continue
        started = time.monotonic()
        prompts = loader(task)
        identity = prompt_identity(prompts, loader.files)
        if identity != task["prompt_identity"]:
            raise ValueError(f"changed frozen input {native.key(task)}")
        records = []
        kwargs = {"arms": arm_groups(method)[group]} if method in PARALLEL_ARMS else {}
        for arm, prediction, audit in model.evaluate(task, prompts, **kwargs):
            if method == "promptgen":
                scores = prediction
            else:
                if prediction.shape != prompts.target.shape or prediction.dtype != np.bool_:
                    raise ValueError("prediction must be a complete canonical binary volume")
                scores = metric(torch.from_numpy(np.ascontiguousarray(prediction)),
                                torch.from_numpy(prompts.target), tuple(prompts.spacing))
            records.append(dict(arm=arm, metrics={m: float(scores[m]) for m in native.METRICS}, audit=audit))
        validate_arms(records, method, group)
        native.write(path, dict(task=list(native.key(task)), method=method, group=group, rows=records,
                                prompt_identity=identity, protocol_sha256=signed_sha, gpu_uuid=uuid,
                                batch=batch if method == "promptgen" else None,
                                seconds=time.monotonic() - started))
        print("SCORED", method, case_id, group, task["global_class_id"], len(records), flush=True)
    final = model.finish() if method == "promptgen" else getattr(model, "assets", None)
    native.write(certificate_file(method, case_id, group),
                 dict(keys=[list(native.key(task)) for task in tasks], protocol_sha256=signed_sha,
                      method=method, group=group, gpu=gpu, uuid=uuid, attempt=attempt, proof=final))


def merge(payload):
    digest = native.sha(OUT / "protocol.json")
    report = {}
    for method, arms in ARMS.items():
        rows = {arm: [] for arm in arms}
        for case in payload["cases"]:
            case_tasks = [task for task in payload["tasks"] if task["case_id"] == case]
            for group in range(len(arm_groups(method))):
                certificate = native.read(certificate_file(method, case, group))
                if (certificate["protocol_sha256"] != digest or certificate["method"] != method
                        or certificate["group"] != group
                        or certificate["keys"] != [list(native.key(task)) for task in case_tasks]):
                    raise ValueError("case completion certificate differs from the protocol")
                for task in case_tasks:
                    for row in read_record(method, task, group, digest)["rows"]:
                        rows[row["arm"]].append(row["metrics"])
        report[method] = {arm: dict(tasks=len(items), averages={
            m: math.fsum(item[m] for item in items) / len(items) for m in native.METRICS})
            for arm, items in rows.items()}
    native.write(OUT / "merged_metrics.json", dict(protocol_sha256=digest, cases=payload["cases"], methods=report))
    print("AMOS_PROMPT_MATRIX_COMPLETE", 149 * sum(map(len, ARMS.values())), flush=True)


def run():
    import fcntl
    from tools.fumpe_six_baselines import worker_runtime

    with (OUT / "schedule.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        payload = protocol()
        verify_inputs(payload)
        pending = [(method, case, group) for case in payload["cases"] for method in ARMS
                   for group in range(len(arm_groups(method)))
                   if not certificate_file(method, case, group).exists()]
        # Warm each model path on the first case before dispatching secondary arm groups.
        pending.sort(key=lambda job: (payload["cases"].index(job[1]), job[2]))
        active, failed = {}, {}
        batches, ooms, cooldown = {}, Counter(), {}
        while pending or active:
            for index, (job, child, log, attempt) in list(active.items()):
                rc = child.poll()
                if rc is None:
                    continue
                log.close()
                del active[index]
                if rc == 75:
                    cooldown[job, index] = time.monotonic() + 60
                    pending.append(job)
                elif rc == 76 and ooms[job] < 4:
                    ooms[job] += 1
                    batches[job] = max(1, batches.get(job, 4) // 2)
                    cooldown[job, index] = time.monotonic() + 180
                    pending.append(job)
                elif rc != 0 or not certificate_file(*job).exists():
                    failed["/".join(map(str, job))] = dict(rc=rc, attempt=attempt)
            if shutil.disk_usage(OUT).free >= 3 * 1024 ** 3:
                available = gpu_inventory()
                for job in list(pending):
                    method, case, group = job
                    eligible = [(free, index, uuid) for index, (uuid, free) in available.items()
                                if index not in active and free >= MINIMUM_MIB
                                and (method != "promptgen" or index in PG_GPUS)
                                and cooldown.get((job, index), 0) <= time.monotonic()]
                    if not eligible:
                        continue
                    _, index, uuid = max(eligible)
                    executable, env = worker_runtime(method)
                    if method == "nninteractive":
                        executable = str(ROOT.parent / "conda-envs/nninteractive_eval/bin/python")
                    env.update(CUDA_VISIBLE_DEVICES=uuid, V9_SHARED_CACHE_WRITE_POLICY="read_only")
                    log_dir = OUT / "logs" / method / case / str(group)
                    log_dir.mkdir(parents=True, exist_ok=True)
                    attempt = len(list(log_dir.glob("*.log"))) + 1
                    log = (log_dir / f"{attempt:03d}.log").open("x")
                    cmd = [executable, "-B", str(Path(__file__)), "worker", method, case, str(group), str(index), uuid,
                           str(attempt), str(batches.get(job, 4))]
                    child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    active[index] = (job, child, log, attempt)
                    pending.remove(job)
                    print("DISPATCH", method, case, group, index, child.pid, attempt, flush=True)
            native.write(OUT / "status.json", dict(pending=pending, failed=failed,
                         active={index: dict(job=v[0], pid=v[1].pid, attempt=v[3]) for index, v in active.items()}))
            if pending or active:
                time.sleep(10)
        if failed:
            raise RuntimeError(f"case jobs failed: {failed}")
        merge(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for action in ("prepare", "run"):
        commands.add_parser(action)
    child = commands.add_parser("worker")
    child.add_argument("method", choices=tuple(ARMS))
    child.add_argument("case_id")
    child.add_argument("group", type=int)
    child.add_argument("gpu", type=int)
    child.add_argument("uuid")
    child.add_argument("attempt", type=int)
    child.add_argument("batch", type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "run":
        run()
    else:
        try:
            worker(args.method, args.case_id, args.group, args.gpu, args.uuid, args.attempt, args.batch)
        except RuntimeError as error:
            if "out of memory" not in str(error).lower() or "cuda" not in str(error).lower():
                raise
            import traceback
            traceback.print_exc()
            raise SystemExit(76) from error


if __name__ == "__main__":
    main()
