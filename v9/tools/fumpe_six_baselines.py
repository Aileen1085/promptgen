"""Compare native baselines using the frozen October 11 FUMPE prompts."""
import argparse
import importlib
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import four_source_data, four_source_eval as original
from tools import native_ct13_eval as native

OLD = ROOT / original.OUT
OUT = ROOT / "output/fumpe_six_baselines_20261114"
METHODS = ("sam2", "medsam2", "sammed3d", "segvol", "vista3d", "pam")
ARMS = {"sammed3d": "fps4", "segvol": "official_fps4", "vista3d": "fit_fps4"}
MIN_GPU_MIB = 30720
INPUTS = ("selected_cases.json", "refs.json", "manifest.json", "geometry_audit.json", "merged_metrics.json")


def prompt_path(ref):
    name = ref["prompt_cache_name"]
    if Path(name).name != name:
        raise ValueError("invalid cache filename")
    return ROOT / original.CACHE / "metadata/shared_adaptive_prompt_v1/fumpe" / name


def frozen_inputs():
    before = {name: native.sha(OLD / name) for name in INPUTS}
    cases = native.read(OLD / "selected_cases.json")["fumpe"]
    refs = [r for r in native.read(OLD / "refs.json")["per_case_class"] if r["source_name"] == "fumpe"]
    tasks = [r for r in native.read(OLD / "manifest.json")["tasks"] if r["source_name"] == "fumpe"]
    empty = [r for r in native.read(OLD / "geometry_audit.json")["not_applicable"] if r["source_name"] == "fumpe"]
    case_map = {c["case_id"]: c for c in cases}
    ids = {t["case_id"] for t in tasks}
    if (len(cases), len(case_map), len(tasks), len(ids), len(empty)) != (30, 30, 28, 28, 2):
        raise ValueError("original FUMPE case/target counts changed")
    if set(case_map) - ids != {r["case_id"] for r in empty if r["reason"] == "GT_empty"}:
        raise ValueError("unexpected excluded cases")
    if len(refs) != 28 or {native.key(r) for r in refs} != {native.key(t) for t in tasks}:
        raise ValueError("prompt reference mismatch")
    merged = native.read(OLD / "merged_metrics.json")
    baseline = {}
    for method in ("v10_2", "nninteractive"):
        rows = [r for r in merged[method]["per_case_class"] if r["source_name"] == "fumpe"]
        original.validate_results(rows, tasks)
        original.validate_prompt_signatures(rows, refs)
        baseline[method] = rows
    for ref in refs:
        if native.sha(prompt_path(ref)) != ref["prompt_sha256"]:
            raise ValueError("frozen prompt changed")
    if {name: native.sha(OLD / name) for name in INPUTS} != before:
        raise ValueError("original input changed during read")
    return case_map, sorted(refs, key=native.key), baseline, before


def load_inputs(case, ref):
    import numpy as np

    if native.sha(prompt_path(ref)) != ref["prompt_sha256"]:
        raise ValueError("prompt SHA mismatch")
    for field in ("image_path", "label_path"):
        if "FUMPE" not in Path(case[field]).parts:
            raise ValueError("case outside FUMPE")
    image, label, geometry = four_source_data.load_pair(case)
    target = (np.asarray(label.dataobj).transpose(2, 0, 1) == int(ref["local_class_id"])).astype(np.uint8)
    with np.load(prompt_path(ref), allow_pickle=False) as package:
        shape = tuple(int(n) for n in package["shape_dhw"])
        supports = [np.unpackbits(package[field], count=math.prod(shape), bitorder="little")
                    .reshape(shape).astype(bool) for field in ("coronal_bits", "sagittal_bits")]
        background = package["background_points_dhw"].astype(np.int32)
        cache_name = str(package["case_image_name"].item())
        slices = tuple(int(n) for n in package["slices"].reshape(2))
    audit = original.audit_prompt(dict(coronal=supports[0], sagittal=supports[1],
                                      background_points_dhw=background), target)
    if audit["background_count"] != 4 or not target.any():
        raise ValueError("wrong original prompt or empty target")
    if native.canonical_plane_slices(supports) != slices:
        raise ValueError("cached scribble planes differ from canonical coordinates")
    ct = np.asarray(image.dataobj, dtype=np.float32).transpose(2, 0, 1)
    cache = ROOT / original.CACHE / "_case_images_v1" / cache_name
    hit = bool(cache_name) and Path(cache_name).name == cache_name and cache.is_file()
    if hit:
        cached = np.load(cache, mmap_mode="r", allow_pickle=False)
        if cached.shape != ct.shape or not np.array_equal(cached, ct):
            raise ValueError("shared case CT differs from original canonical CT")
        ct = cached
    spacing = tuple(geometry["spacing_xyz_mm"][i] for i in (2, 0, 1))
    return ct, target, supports, background, spacing, slices, dict(
        audit, cached_ct_used=hit, background_points_dhw=background.tolist(), slices=slices)


def points_3d(foreground, background, spacing, slices, cap, method="fps"):
    """Apply the existing FG sampler and pass each original BG point through."""
    import numpy as np
    from infer.native_sammed3d_prompts import sample_points

    coords = np.argwhere(foreground)
    selected = [sample_points(coords[coords[:, axis] == plane], spacing, cap, 3., method)
                for axis, plane in zip((2, 1), slices)]
    positive = np.unique(np.concatenate(selected), axis=0)
    if not len(positive):
        raise ValueError("missing foreground points")
    return (np.concatenate((positive, background)).astype(float),
            np.r_[np.ones(len(positive), dtype=np.int64), np.zeros(len(background), dtype=np.int64)])


def load_model(method):
    if method in ("sam2", "medsam2"):
        assets = native.asset_spec(method)
        code = ROOT / assets["code_root"]
        sys.path.insert(0, str(code))
        from sam2 import build_sam
        if not Path(build_sam.__file__).resolve().is_relative_to(code.resolve()):
            raise ValueError("unexpected SAM implementation")
        weight = ROOT / assets["checkpoint"]
        model = build_sam.build_sam2_video_predictor(
            assets["config"], str(weight), device="cuda",
            hydra_overrides_extra=["++model.use_mask_input_as_output_without_sam=false",
                                   "++model.fill_hole_area=0"])
        if model.image_size != assets["image_size"] or model.num_maskmem != 7:
            raise ValueError("wrong native SAM configuration")
        return model, native.base_module(), {str(weight.relative_to(ROOT)): native.sha(weight)}
    if method == "pam":
        from tools import pam_nine_source_probability as pam
        model = pam.soft.SoftModels()
        return model, pam, model.audit
    pilot = importlib.import_module("tools.native_" + method + "_pilot")
    pilot.scribble_points = points_3d
    return pilot.load_model(), pilot, {str(pilot.WEIGHT.relative_to(ROOT)): pilot.WEIGHT_SHA}


def predict(method, model, adapter, ct, supports, bg, spacing, slices):
    import numpy as np
    import torch

    fg = supports[0] | supports[1]
    if method == "pam":
        payload = tuple(dict(plane=plane, slice=index, box=adapter.p.slice_box(support, plane, index))
                        for plane, index, support in zip(("coronal", "sagittal"), slices, supports))
        pred, stages = adapter.predict(ct, spacing, payload, model)
        return pred, dict(boxes=payload, stages=stages, background="unsupported")
    if method in ARMS:
        image = ct if method == "sammed3d" else adapter.normalize(ct)
        pred, stages = adapter.predict(model, image, fg, bg, spacing, slices, ARMS[method])
        return pred, dict(stages=stages, encoded_background=bg.tolist())
    args = SimpleNamespace(max_points=2, max_background_points=0, point_spacing_mm=3.,
                           dense_prior="none", sigma_mm=2., amplitude=4.)
    initial = adapter.prompt_bounds_3d(fg, bg, spacing)
    bounds = initial
    stages = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(8):
            _, (h0, h1), (w0, w1) = bounds
            roi, stage = adapter.predict_roi(model, ct[:, h0:h1, w0:w1], fg[:, h0:h1, w0:w1],
                                             bg - np.array([0, h0, w0]), spacing, bounds[0], args)
            stages.append(dict(stage, bounds_dhw=bounds))
            expanded = adapter.expand_bounds_3d(roi, fg, bounds, initial)
            if expanded == bounds:
                pred = np.zeros(ct.shape, dtype=bool)
                pred[tuple(slice(a, b) for a, b in bounds)] = roi
                return pred, dict(stages=stages, encoded_background=bg.tolist())
            bounds = expanded
    raise RuntimeError("native SAM AutoExpand did not converge")


def worker(method, uuid):
    import numpy as np
    import torch

    cases, refs, _, hashes = frozen_inputs()
    actual = str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-")
    if actual != uuid.removeprefix("GPU-") or torch.ones(1, device="cuda").item() != 1:
        raise ValueError("CUDA/physical GPU mapping mismatch")
    if not any(device == uuid for _, device in available()):
        raise RuntimeError("GPU below free-memory threshold after probe")
    torch.set_num_threads(2)
    model, adapter, weights = load_model(method)
    scorer = original.metric_module()
    destination = OUT / method
    destination.mkdir(exist_ok=True)
    native.write(destination / "protocol.json", dict(
        inputs=hashes, weights=weights, metric_sha=native.sha(original.metric_path()),
        entry_sha=native.sha(__file__), gpu_uuid=uuid, method=method))
    rows = []
    for ref in refs:
        ct, gt, fg, bg, spacing, slices, audit = load_inputs(cases[ref["case_id"]], ref)
        prediction, conversion = predict(method, model, adapter, ct, fg, bg, spacing, slices)
        if prediction.shape != gt.shape or prediction.dtype != np.bool_:
            raise ValueError("prediction must be binary on the full canonical CT grid")
        scored = scorer.binary_prompt_metrics_from_masks(
            torch.from_numpy(np.ascontiguousarray(prediction)), torch.from_numpy(gt), spacing_dhw=spacing)
        row = dict(ref, **{k: float(scored[k]) for k in original.METRICS},
                   prompt_input_audit=audit, conversion=conversion, method=method)
        original.validate_results([row], [ref])
        rows.append(row)
        native.write(destination / "progress.json", dict(per_case_class=rows))
        print(method, len(rows), "/ 28", ref["case_id"], f"Dice={row['dice']:.5f}", flush=True)
        del ct, gt, fg, bg, prediction
    original.validate_results(rows, refs)
    native.write(destination / "complete_metrics.json", dict(per_case_class=rows))


def available():
    inventory = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.free",
                                        "--format=csv,noheader,nounits"], text=True)
    entries = [line.split(",") for line in inventory.splitlines()]
    return sorted(((int(free), device.strip()) for _, device, free in entries
                   if int(free) >= MIN_GPU_MIB), reverse=True)


def worker_runtime(method):
    interpreters = {"segvol": ROOT / "segvol-3d/.venv/bin/python",
                    "vista3d": ROOT / "vista3d/.venv-vista/bin/python"}
    executable = interpreters.get(method, Path(sys.executable))
    env = dict(os.environ)
    if method == "pam":
        dependency_path = ROOT / "third_party/PAM_official_20261010/python_deps"
        paths = [str(dependency_path)]
        if env.get("PYTHONPATH"):
            paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(paths)
    if not executable.is_file():
        raise FileNotFoundError(executable)
    return str(executable), env


def schedule():
    import fcntl

    OUT.mkdir(exist_ok=True)
    with (OUT / "schedule.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _, refs, baseline, hashes = frozen_inputs()
        active, failed = {}, {}
        pending = [m for m in METHODS if not (OUT / m / "complete_metrics.json").is_file()]
        if any((OUT / (m + ".log")).exists() for m in pending):
            raise RuntimeError("unfinished prior attempt requires review")
        while pending or active:
            for method, (process, device) in list(active.items()):
                rc = process.poll()
                if rc is not None:
                    del active[method]
                    if rc:
                        failed[method] = rc
            used = {device for _, device in active.values()}
            if pending and shutil.disk_usage(OUT).free >= 3 * 1024**3:
                for _, device in available():
                    if not pending or device in used:
                        continue
                    method = pending.pop(0)
                    executable, env = worker_runtime(method)
                    env["CUDA_VISIBLE_DEVICES"] = device
                    with (OUT / (method + ".log")).open("x") as log:
                        process = subprocess.Popen([executable, "-B", __file__, "worker", method, device],
                                                   cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    active[method] = process, device
            native.write(OUT / "status.json", dict(pending=pending, failed=failed,
                         active={m: dict(pid=p.pid, gpu=d) for m, (p, d) in active.items()},
                         waiting_for_gpu=bool(pending and not active), input_sha256=hashes))
            if pending or active:
                time.sleep(45)
        if failed:
            raise RuntimeError(f"workers failed: {failed}")
        for method in METHODS:
            baseline[method] = native.read(OUT / method / "complete_metrics.json")["per_case_class"]
        result = {}
        for method, rows in baseline.items():
            original.validate_results(rows, refs)
            original.validate_prompt_signatures(rows, refs)
            result[method] = dict(per_case_class=rows, summary={
                k: math.fsum(r[k] for r in rows) / len(rows) for k in original.METRICS})
        native.write(OUT / "merged_metrics.json", dict(input_sha256=hashes, methods=result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--geometry", action="store_true")
    commands.add_parser("schedule")
    run = commands.add_parser("worker")
    run.add_argument("method", choices=METHODS)
    run.add_argument("uuid")
    args = parser.parse_args()
    if args.command == "verify":
        cases, refs, _, hashes = frozen_inputs()
        print("FUMPE_INPUTS_VERIFIED", len(refs), hashes)
        if args.geometry:
            for ref in refs:
                case_id = ref["case_id"]
                *_, audit = load_inputs(cases[case_id], ref)
                print("FUMPE_GRID_OK", case_id, audit["shape_dhw"], audit["slices"], flush=True)
    elif args.command == "worker":
        worker(args.method, args.uuid)
    else:
        schedule()


if __name__ == "__main__":
    main()
