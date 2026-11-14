"""Eight native/PromptGen methods on aligned SegRap and LNQ subset30 cases."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import four_source_eval as common
from tools import fumpe_six_baselines as native

OUT = ROOT / "output/segrap_lnq_subset30_eight_methods_20261114"
METHODS = ("v10_2", "nninteractive", *native.METHODS)
SOURCES = ("segrap_ncct", "segrap_cect", "lnq_subset30")
FLOORS = (.35, .45, .50, .55, .60, .65, .70)
PROMPTS = OUT / "metadata/shared_adaptive_prompt_v1"
METRICS = common.METRICS

# Packed mask ID and shared-region IDs from the SegRap23-Lite label inventory.
# https://huggingface.co/datasets/YongchengYAO/SegRap23-Lite
OARS = (
    ("brain", (1,)), ("brainstem", (2,)), ("optic chiasm", (3,)),
    ("left temporal lobe", (4, 6)), ("right temporal lobe", (5, 7)),
    ("left hippocampus", (8, 6)), ("right hippocampus", (9, 7)),
    ("left eye", (10,)), ("right eye", (11,)), ("left lens", (12,)),
    ("right lens", (13,)), ("left optic nerve", (14,)), ("right optic nerve", (15,)),
    ("left middle ear", (16, 20, 24, 30)), ("right middle ear", (17, 21, 25, 31)),
    ("left internal auditory canal", (18,)), ("right internal auditory canal", (19,)),
    ("left tympanic cavity", (22, 20)), ("right tympanic cavity", (23, 21)),
    ("left vestibular semicircular canal", (26, 24)), ("right vestibular semicircular canal", (27, 25)),
    ("left cochlea", (28,)), ("right cochlea", (29,)),
    ("left bony eustachian tube", (32, 30)), ("right bony eustachian tube", (33, 31)),
    ("pituitary gland", (34,)), ("oral cavity", (35,)), ("left mandible", (36,)),
    ("right mandible", (37,)), ("left submandibular gland", (38,)), ("right submandibular gland", (39,)),
    ("left parotid gland", (40,)), ("right parotid gland", (41,)), ("left mastoid", (42,)),
    ("right mastoid", (43,)), ("left temporomandibular joint", (44,)),
    ("right temporomandibular joint", (45,)), ("spinal cord", (46,)), ("esophagus", (47,)),
    ("larynx", (48, 51)), ("glottic larynx", (49,)), ("supraglottic larynx", (50,)),
    ("pharyngeal constrictor", (52, 51)), ("thyroid", (53,)), ("trachea", (54,)),
)
GTVS = (("nasopharyngeal primary tumor GTV", (1,)), ("lymph node tumor GTV", (2,)))
LNQ = (("mediastinal lymph nodes", (1,)),)


def signature(path):
    p = ROOT / path
    stat = p.stat()
    return [str(p.resolve()), stat.st_size, stat.st_mtime_ns]


def check_grid(first, second):
    import numpy as np
    if len(first.shape) != 3 or first.shape != second.shape:
        raise ValueError(f"CT/mask shape mismatch: {first.shape} and {second.shape}")
    if not np.allclose(first.affine, second.affine, rtol=0, atol=1e-4):
        raise ValueError("CT/mask physical affine mismatch")


def binary_label(labels, values):
    mask = labels == values[0]
    for code in values[1:]:
        mask |= labels == code
    return mask


def prepare():
    import nibabel as nib
    import numpy as np
    protocol, nn = common.nn_modules()
    from adaptive_prompt_cache import _write

    adaptive = protocol.load_adaptive_module(ROOT / "v9_2_adaptive_scribble.py")
    args = nn.parse_args()
    args.adaptive_min_radius, args.adaptive_max_radius, args.adaptive_width_fraction = 1, 20, .15
    args.background_points_per_plane = 2
    args.negative_point_mode, args.point_seed = "bbox_outside", 2026
    rows, catalogue, excluded, selected = [], {}, [], {}
    for dataset, choices, groups in (
        ("SegRap2023", (("segrap_ncct", "Images-CT"), ("segrap_cect", "Images-contrastCT")),
         (("OAR", "Masks-Task1", OARS, 3000, 54), ("GTV", "Masks-Task2", GTVS, 3100, 2))),
        ("LNQ2023-Lite", (("lnq_subset30", "Images"),), (("LNQ", "Masks", LNQ, 3200, 1),)),
    ):
        base = Path("../data") / dataset / "subset30"
        ids = (ROOT / base / "cases.txt").read_text().split()
        if len(ids) != 30 or len(set(ids)) != 30 or any(Path(i).name != i for i in ids):
            raise ValueError("expected 30 unique subset cases")
        selected[dataset] = ids
        for case in sorted(ids):
            filename = case + ".nii.gz"
            images = [(source, str(base / folder / filename)) for source, folder in choices]
            headers = [nib.as_closest_canonical(nib.load(str(ROOT / path))) for _, path in images]
            for group, folder, classes, offset, max_label in groups:
                label_path = str(base / folder / filename)
                label_img = nib.as_closest_canonical(nib.load(str(ROOT / label_path)))
                for header in headers:
                    check_grid(header, label_img)
                labels = np.asarray(label_img.dataobj).transpose(2, 0, 1)
                present = set(np.unique(labels).tolist())
                if not present.issubset(set(range(max_label + 1))):
                    raise ValueError(f"unexpected labels {case}/{group}: {present}")
                label_signature = signature(label_path)
                for index, (name, codes) in enumerate(classes, 1):
                    cid = offset + index
                    for source, _ in images:
                        catalogue[source, cid] = dict(dataset=source, global_class_id=cid,
                            class_name=name, local_class_id=codes[0], label_value=codes[0],
                            label_values=codes, semantic_text=name + " on CT",
                            semantic_key="target." + name, parent_semantic_key=None,
                            equivalence_group="target." + name, morphology_group="compact")
                    if present.isdisjoint(codes):
                        excluded.append(dict(dataset=dataset, case_id=case, group=group, target=name))
                        continue
                    gt = binary_label(labels, codes)
                    ref = protocol.AdaptiveValidationTask(
                        dataset.lower(), case, cid, codes[0], name, ROOT / images[0][1], ROOT / label_path)
                    package = nn._build_prompt_package(args, ref, gt, "memory_only", adaptive)
                    audit = common.audit_prompt(package, gt)
                    if audit["background_count"] != 4:
                        raise ValueError("expected four original-protocol background points")
                    prompt_name = f"{case}_{cid}.npz"
                    first_path = None
                    for source, image_path in images:
                        prompt_file = PROMPTS / source / prompt_name
                        if first_path is None:
                            _write(prompt_file, package)
                            first_path = prompt_file
                            prompt_sha = common.sha(prompt_file)
                        else:
                            prompt_file.parent.mkdir(parents=True, exist_ok=True)
                            os.link(first_path, prompt_file)
                        # A case ID within one model source must bind exactly one label volume.
                        # The key keeps Task 1 and Task 2 distinct without duplicating CT.
                        row_case = case if group != "GTV" else case + "_gtv"
                        rows.append(dict(source_name=source, dataset=dataset, group=group,
                            patient_id=case, case_id=row_case, global_class_id=cid, local_class_id=codes[0],
                            class_name=name, label_values=codes, image=image_path, label=label_path,
                            split_role="validation", image_signature=signature(image_path),
                            label_signature=label_signature, prompt_cache_name=prompt_name,
                            prompt_sha256=prompt_sha, prompt_audit=audit))
                    del package, gt
                del labels
            print("PREPARED", dataset, case, len(rows), flush=True)
    keys = {common.task_key(r) for r in rows}
    if not rows or len(keys) != len(rows):
        raise ValueError("empty or duplicate task inventory")
    for row in rows:
        if signature(row["image"]) != row["image_signature"] or signature(row["label"]) != row["label_signature"]:
            raise ValueError("data changed during preparation")
    split = dict(source_order=list(SOURCES), class_catalog=list(catalogue.values()),
                 datasets={s: dict(train=[], val=[]) for s in SOURCES})
    common.write(OUT / "split.json", split)
    common.write(OUT / "selected_cases.json", selected)
    common.write(OUT / "empty_targets.json", excluded)
    common.write(OUT / "manifest.json", common.manifest(rows))
    common.write(OUT / "protocol.json", dict(
        manifest_sha=common.sha(OUT/"manifest.json"), split_sha=common.sha(OUT/"split.json"),
        code_sha=common.sha(__file__), metric_sha=common.sha(common.metric_path()),
        floors=FLOORS, reference_floor=.35, selected_counts={s: len(v) for s,v in selected.items()},
        task_counts=dict(Counter((r["source_name"] + "/" + r["group"]) for r in rows))))
    print("PREPARATION_COMPLETE", len(rows), flush=True)


class CaseCache:
    """Read each current scan and label as canonical arrays in memory."""
    def __init__(self):
        self.cache = {}
        self.checked = {}

    def read(self, name, path, expected=None):
        import nibabel as nib
        import numpy as np
        if expected is not None and self.checked.get(name) != (str(path), expected):
            if signature(path) != expected:
                raise ValueError(f"changed data file {path}")
            self.checked[name] = (str(path), expected)
        previous = self.cache.get(name)
        if previous is None or previous[0] != str(path):
            image = nib.as_closest_canonical(nib.load(str(ROOT / path)))
            data = np.asarray(image.dataobj, dtype=np.float32 if name == "ct" else None).transpose(2, 0, 1)
            self.cache[name] = (str(path), image.affine, data)
        return self.cache[name][2]

    def task(self, row):
        ct = self.read("ct", row["image"], row["image_signature"])
        labels = self.read("label", row["label"], row["label_signature"])
        target = binary_label(labels, row["label_values"])
        import numpy as np
        spacing = np.linalg.norm(self.cache["ct"][1][:3, (2, 0, 1)], axis=0)
        return ct, target.astype(np.uint8), spacing


def read_prompt(row):
    from nninteractive_test.infer.adaptive_prompt_cache import _read
    path = PROMPTS / row["source_name"] / row["prompt_cache_name"]
    if common.sha(path) != row["prompt_sha256"]:
        raise ValueError("prompt contents changed")
    result = _read(path)
    if result is None:
        raise ValueError("invalid prompt package")
    return result


def install_vista_edge_guard(pilot):
    import numpy as np
    original = pilot.patch_predict

    def checked(model, image, points, labels):
        if np.any(points < -.500001) or np.any(points > np.asarray(image.shape) - .499999):
            raise ValueError("prompt outside the physical VISTA input grid")
        return original(model, image, np.maximum(points, 0), labels)

    pilot.patch_predict = checked


def run_baseline(method, rows, emit, gpu):
    import numpy as np
    import torch
    scorer = common.metric_module()
    cache = CaseCache()
    if method == "nninteractive":
        _, nn = common.nn_modules(str(gpu))
        nn.patch_nninteractive_trainer_for_current_nnunet()
        session = nn.nnInteractiveInferenceSession(device=torch.device("cuda:0"), use_torch_compile=False,
            verbose=False, torch_n_threads=2, do_autozoom=True, use_pinned_memory=True)
        session.initialize_from_trained_model_folder(str(ROOT.parent / "nninteractive_test_amos/checkpoints/nnInteractive_v1.0"))
    else:
        model, adapter, weights = native.load_model(method)
        if method == "vista3d":
            install_vista_edge_guard(adapter)
    for row in rows:
        ct, gt, spacing = cache.task(row)
        package = read_prompt(row)
        audit = common.audit_prompt(package, gt)
        if audit["background_count"] != 4:
            raise ValueError("background points changed")
        if method == "nninteractive":
            pred = nn.predict_with_nninteractive_cached(session, ct, package, torch.device("cuda:0"))
        else:
            supports = [package[name].astype(bool) for name in ("coronal", "sagittal")]
            bg = package["background_points_dhw"]
            slices = (package["coronal_slice"], package["sagittal_slice"])
            volume, _ = native.predict(method, model, adapter, ct, supports, bg, spacing, slices)
            pred = torch.from_numpy(np.ascontiguousarray(volume))
        if tuple(pred.shape) != tuple(gt.shape):
            raise ValueError("prediction not restored to the full CT")
        scores = scorer.binary_prompt_metrics_from_masks(pred, torch.from_numpy(gt), spacing_dhw=tuple(spacing))
        emit(dict(row, **{m: float(scores[m]) for m in METRICS}))
        del package, pred, gt
        if method != "nninteractive":
            del volume, supports


def run_v10(rows, emit, work, gpu, frame_batch):
    import numpy as np
    import v10.infer.infer_eight_source_adaptive_v10_2 as evaluator
    import v10.infer.infer_amos_sam2_promptgen_v10 as scorer
    import v10_2_data as data
    from tools.bone_shared_logits import tensor_digest

    source = rows[0]["source_name"]
    selected = {common.task_key(r): r for r in rows}
    by_label_id = {(r["label"], r["global_class_id"]): r for r in rows}
    cache = CaseCache()

    def load_ct(self, path):
        found = self._load_case_image_cache(self._case_image_cache_path(path))
        return found if found is not None else cache.read("ct", path)

    def load_array(self, path):
        return cache.read("label", path)

    def target(self, path, cid):
        row = by_label_id[str(Path(path)), int(cid)]
        labels = cache.read("label", path, row["label_signature"])
        return binary_label(labels, row["label_values"]).astype(np.uint8)

    data.SingleFileMulticlassV102Dataset._get_or_create_case_image = load_ct
    data.SingleFileMulticlassV102Dataset._load_array = load_array
    data.SingleFileMulticlassV102Dataset._load_total_target = target
    scorer.binary_prompt_metrics_from_masks = common.metric_module().binary_prompt_metrics_from_masks
    native_decode = evaluator._dynamic_adaptive_decode
    native_post = scorer.postprocess_v10_logits_from_prompts
    native_score = scorer.restore_and_score_full_volume
    decoded = {}

    def post(logits, fg, bg, **settings):
        import weakref
        decoded["capture"] = (weakref.ref(logits), fg, bg, settings)
        return native_post(logits, fg, bg, **settings)

    def decode(*args, **kwargs):
        result = native_decode(*args, **kwargs)
        logits, prediction, bounds, history, audit, geometry = result
        if decoded["capture"][0]() is not logits or geometry is not None:
            raise ValueError("shared-logits scoring needs the exact final v10.2 ROI tensor")
        decoded.update(task_class=int(args[4]), prediction=prediction, history=history)
        return result

    ordered = iter(sorted(rows, key=common.task_key))

    def score(logits, **kwargs):
        row = next(ordered)
        captured, fg, bg, options = decoded["capture"]
        if logits is not captured() or decoded["task_class"] != row["global_class_id"]:
            raise ValueError("final logits/task binding mismatch")
        base = native_score(logits, **kwargs)
        digest = tensor_digest(logits)
        floor_rows = []
        for floor in FLOORS:
            pred, audit = native_post(logits, fg, bg, **dict(options, minimum=floor))
            scores = base[0] if floor == .35 else native_score(
                logits, **dict(kwargs, prediction=pred, threshold=audit["threshold"]))[0]
            if floor == .35 and not np.array_equal(pred, decoded["prediction"]):
                raise ValueError("reference floor does not reproduce original prediction")
            floor_rows.append(dict(floor=floor, threshold=float(audit["threshold"]),
                calibration_source=str(audit["source"]), **{m: float(scores[m]) for m in METRICS}))
        emit(dict(row, logits_sha256=digest, roi_frames=logits.shape[0], floors=floor_rows))
        decoded.clear()
        return base

    scorer.postprocess_v10_logits_from_prompts = post
    scorer.restore_and_score_full_volume = score
    evaluator._dynamic_adaptive_decode = decode
    for row in selected.values():
        if signature(row["image"]) != row["image_signature"] or signature(row["label"]) != row["label_signature"]:
            raise ValueError("CT/GT changed")
        if common.sha(PROMPTS / source / row["prompt_cache_name"]) != row["prompt_sha256"]:
            raise ValueError("prompt changed")
    common.write(work / "manifest.json", common.manifest(rows))
    common.write(work / "refs.json", dict(per_case_class=rows))
    sys.argv = [sys.argv[0], "--independent-task-manifest", str(work/"manifest.json"),
        "--nninteractive-metrics", str(work/"refs.json"), "--shared-adaptive-prompt-root", str(OUT),
        "--weights", str(ROOT/common.WEIGHT), "--source", source, "--background-mode", "point",
        "--eval-out-dir", str(work), "--run-name", "result", "--gpu", str(gpu),
        "--sam2-frame-batch-size", str(frame_batch), "--dynamic-threshold-min", ".35",
        "--base-split-json", str(OUT/"split.json")]
    evaluator.main()


def job_rows(name):
    protocol = common.read(OUT/"protocol.json")
    for field, file in (("manifest_sha", OUT/"manifest.json"), ("split_sha", OUT/"split.json"),
                        ("code_sha", Path(__file__)), ("metric_sha", common.metric_path())):
        if common.sha(file) != protocol[field]:
            raise ValueError(f"changed signed input: {file}")
    spec = common.read(OUT/"jobs"/(name+".json"))
    return spec["method"], spec["tasks"]


def worker(name, gpu, uuid, frame_batch):
    import torch
    actual = str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-")
    if actual != uuid.removeprefix("GPU-") or torch.ones(1, device="cuda").item() != 1:
        raise ValueError("mismatched CUDA UUID")
    if not any(device == uuid for _, device in native.available()):
        raise SystemExit(75)
    torch.set_num_threads(2)
    method, rows = job_rows(name)
    out = OUT/"runs"/name
    out.mkdir(parents=True, exist_ok=True)
    work = out/("attempt_" + str(time.time_ns()))
    work.mkdir()
    from collections import OrderedDict
    completed = OrderedDict()
    expected = {common.task_key(row): row for row in rows}
    for old in sorted(out.glob("attempt_*/progress.json")):
        for row in common.read(old)["per_case_class"]:
            key = common.task_key(row)
            if key in completed or key not in expected or row["prompt_sha256"] != expected[key]["prompt_sha256"]:
                raise ValueError("prior task does not match the frozen job")
            completed[key] = row
    pending = [row for row in rows if common.task_key(row) not in completed]
    fresh = []

    def emit(row):
        parts = row["floors"] if method == "v10_2" else (row,)
        if any(not math.isfinite(part[m]) for part in parts for m in METRICS):
            raise ValueError("nonfinite result")
        key = common.task_key(row)
        if key in completed or key not in expected:
            raise ValueError("new result does not match a pending task")
        fresh.append(row)
        common.write(work/"progress.json", dict(per_case_class=fresh))
        completed[key] = row
        print("SCORED", name, len(completed), "/", len(rows), key, flush=True)

    try:
        if pending and method == "v10_2":
            run_v10(pending, emit, work, gpu, frame_batch)
        elif pending:
            run_baseline(method, pending, emit, gpu)
    except torch.cuda.OutOfMemoryError:
        if method == "v10_2" and frame_batch > 1:
            raise SystemExit(76)
        raise
    if set(completed) != set(expected):
        raise ValueError("incomplete/duplicate worker results")
    common.write(out/"complete.json", dict(method=method, per_case_class=list(completed.values())))


def jobs_for(tasks):
    jobs = []
    for source in SOURCES:
        rows = sorted((r for r in tasks if r["source_name"] == source), key=common.task_key)
        if not rows:
            raise ValueError("missing source")
        # Keep all classes for the same physical CT in one production shard.
        by_patient = defaultdict(list)
        for row in rows:
            by_patient[row["patient_id"]].append(row)
        loads = [[] for _ in range(10)]
        for group in sorted(by_patient.values(), key=lambda rows: -len(rows)):
            min(loads, key=len).extend(group)
        smoke = rows[:1]
        shards = [[row for row in part if row is not smoke[0]] for part in loads]
        shards = [part for part in shards if part]
        for method in METHODS:
            for index, shard in enumerate([smoke, *shards]):
                jobs.append(dict(name=f"{method}_{source}_{index:02}", method=method, tasks=shard,
                                 smoke=f"{method}_{source}_00", source=source))
    return jobs


def summarize(jobs, tasks):
    buckets = defaultdict(list)
    rows_by_method = defaultdict(list)
    for job in jobs:
        record = common.read(OUT/"runs"/job["name"]/"complete.json")
        if record["method"] != job["method"]:
            raise ValueError("method mismatch")
        rows_by_method[job["method"]].extend(record["per_case_class"])
    prompts = {common.task_key(r): r["prompt_sha256"] for r in tasks}
    for method in METHODS:
        rows = rows_by_method[method]
        if Counter(map(common.task_key, rows)) != Counter(map(common.task_key, tasks)):
            raise ValueError("method coverage mismatch")
        for row in rows:
            if row["prompt_sha256"] != prompts[common.task_key(row)]:
                raise ValueError("prompt mismatch")
            buckets[method,row["source_name"],row["group"]].append(row)
    summary = {}
    for (method, source, group), rows in buckets.items():
        def means(part):
            return {m: math.fsum(r[m] for r in part) / len(part) for m in METRICS}
        item = dict(cases=len({r["patient_id"] for r in rows}), tasks=len(rows))
        if method == "v10_2":
            if any([f["floor"] for f in r["floors"]] != list(FLOORS) for r in rows):
                raise ValueError("incomplete logit grid")
            grid = {floor: means([r["floors"][i] for r in rows]) for i, floor in enumerate(FLOORS)}
            best = max(FLOORS, key=lambda f: grid[f]["dice"])
            item.update(floors=grid, best_floor=best, best_metrics=grid[best])
        else:
            item["metrics"] = means(rows)
        summary.setdefault(source, {}).setdefault(group, {})[method] = item
    common.write(OUT/"merged_metrics.json", dict(summary=summary, per_method=rows_by_method))


def schedule():
    import fcntl
    with (OUT/"schedule.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        tasks = common.read(OUT/"manifest.json")["tasks"]
        jobs = jobs_for(tasks)
        for job in jobs:
            common.bind_protocol(OUT/"jobs"/(job["name"]+".json"),job)
        pending = [j for j in jobs if not (OUT/"runs"/j["name"]/"complete.json").is_file()]
        active, failed, retry, batches = {}, {}, {}, {}
        while pending or active:
            for name, (proc, device, job) in list(active.items()):
                rc = proc.poll()
                if rc is None:
                    continue
                del active[name]
                if rc in (75,76):
                    retry[name] = time.monotonic() + 45
                    if rc == 76:
                        batches[name] = batches.get(name,4) // 2
                    pending.append(job)
                elif rc or not (OUT/"runs"/name/"complete.json").is_file():
                    failed[name] = rc
            used = {d for _,d,_ in active.values()}
            if pending and shutil.disk_usage(OUT).free >= 3 * 1024**3:
                inventory = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.free",
                    "--format=csv,noheader,nounits"], text=True)
                gpu_rows = sorted((line.split(",") for line in inventory.splitlines()),
                                  key=lambda parts: -int(parts[2]))
                for index, raw_uuid, free in gpu_rows:
                    uuid = raw_uuid.strip()
                    if int(free) < native.MIN_GPU_MIB or uuid in used:
                        continue
                    choice = next((j for j in pending if j["smoke"] not in failed
                        and retry.get(j["name"],0) <= time.monotonic()
                        and (j["name"] == j["smoke"] or
                             (OUT/"runs"/j["smoke"]/"complete.json").is_file())), None)
                    if choice is None:
                        continue
                    m, name = choice["method"], choice["name"]
                    exe, env = native.worker_runtime(m)
                    if m == "nninteractive":
                        exe = str(ROOT.parent/"conda-envs/nninteractive_eval/bin/python")
                    env["CUDA_VISIBLE_DEVICES"] = uuid
                    env["V9_SHARED_CACHE_WRITE_POLICY"] = "read_only"
                    logdir = OUT/"logs"
                    logdir.mkdir(exist_ok=True)
                    with (logdir/(name+"_"+str(time.time_ns())+".log")).open("x") as log:
                        proc = subprocess.Popen([exe,"-B","-u",__file__,"worker",name,index.strip(),uuid,
                                                 str(batches.get(name,4))],
                            cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                    active[name] = proc,uuid,choice
                    pending.remove(choice)
                    used.add(uuid)
            blocked = [j for j in pending if j["smoke"] in failed]
            for job in blocked:
                failed[job["name"]] = "smoke_failed"
                pending.remove(job)
            completed = [j for j in jobs if (OUT/"runs"/j["name"]/"complete.json").is_file()]
            common.write(OUT/"status.json",dict(
                jobs=len(jobs), complete_jobs=len(completed), pending=len(pending), failed=failed,
                active={name: dict(pid=p.pid,gpu=d) for name,(p,d,_) in active.items()},
                scored=dict(Counter({m:sum(len(j["tasks"]) for j in completed if j["method"]==m) for m in METHODS})),
                total_tasks_per_method=len(tasks)))
            if active or pending:
                time.sleep(20)
        if failed:
            raise RuntimeError("one or more baseline shards failed; see status and preserved logs")
        summarize(jobs,tasks)
        print("SUBSET30_FINAL_PASS",len(tasks),len(METHODS),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    subs.add_parser("prepare")
    subs.add_parser("schedule")
    runner = subs.add_parser("worker")
    runner.add_argument("name")
    runner.add_argument("gpu")
    runner.add_argument("uuid")
    runner.add_argument("frame_batch",type=int,choices=(1,2,4))
    args = parser.parse_args()
    if args.command == "prepare":
        OUT.mkdir(exist_ok=False)
        prepare()
    elif args.command == "schedule":
        schedule()
    else:
        worker(args.name,args.gpu,args.uuid,args.frame_batch)


if __name__ == "__main__":
    main()
