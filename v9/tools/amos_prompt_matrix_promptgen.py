"""Evaluate the four paired AMOS prompt arms with the signed S21 runtime."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from types import MethodType

import numpy as np

from infer.v9_2_eight_source_eval_protocol import _resize_preserve
from tools.amos_prompt_ablation_prompts import variant_modes


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = "v10/output/v10_2_ct13_e470_zero_delta_e510_20261001/20261001_215853/epoch490.pth"
BASE_SPLIT = "configs/ct13_v10_2_split_20260928.json"
NATIVE_VARIANTS = {
    "fg_scribble_bg_scribble": "fg_scribble_bg_scribble",
    "fg_scribble_bg_point": "baseline",
    "fg_bbox_bg_point": "fg_box_bg_point",
    "fg_bbox_bg_scribble": "fg_box_bg_scribble",
}
ARMS = tuple(NATIVE_VARIANTS)
PLANES = ("coronal", "sagittal")


@dataclass(frozen=True)
class PlanePrompts:
    index: int
    foreground: np.ndarray
    background: np.ndarray
    background_points: np.ndarray


def prepare_prompts(prompts):
    """Project the input onto its recorded planes without consulting the target."""
    shape = prompts.ct.shape
    if len(shape) != 3 or min(shape) < 1:
        raise ValueError("expected one nonempty canonical DHW CT")
    background_points = prompts.points("bg", 2)
    planes = []
    for i, (name, axis) in enumerate(zip(PLANES, (2, 1))):
        index = int(prompts.planes[i])
        if not 0 <= index < shape[axis]:
            raise ValueError(f"{name} selected plane is outside the CT")
        masks = []
        for role, volumes in (("foreground", prompts.foreground), ("background", prompts.background)):
            volume = np.asarray(volumes[i])
            if volume.shape != shape or volume.dtype != np.bool_:
                raise ValueError(f"{name} {role} support must be a canonical DHW binary mask")
            mask = np.take(volume, index, axis=axis)
            count = np.count_nonzero(mask)
            if count == 0 or count != np.count_nonzero(volume):
                raise ValueError(f"{name} {role} support escapes the selected plane or is empty")
            masks.append(mask)
        foreground, background = masks
        points = np.asarray(background_points[i])
        if (points.shape != (2, 3) or points.dtype.kind not in "iu"
                or np.any(points < 0) or np.any(points >= np.asarray(shape))
                or len(np.unique(points, axis=0)) != 2):
            raise ValueError(f"{name} requires two distinct raw background points")
        coords = points[:, [0, 3 - axis]]
        if np.any(points[:, axis] != index) or not np.all(background[tuple(coords.T)]):
            raise ValueError(f"{name} background point is outside its cached stroke")
        planes.append(PlanePrompts(index, foreground, background, points.copy()))
    return tuple(map(int, shape)), tuple(planes)


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 ** 2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_protocol_file(root, relative, inventory):
    expected = inventory.get(relative)
    if expected is None or _sha(root / relative) != expected:
        raise ValueError(f"S21 input changed: {relative}")
    return expected


def _raster_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _box_outline(foreground):
    coordinates = np.argwhere(foreground)
    z0, c0 = map(int, coordinates.min(axis=0))
    z1, c1 = map(int, coordinates.max(axis=0))
    result = np.zeros_like(foreground)
    result[[z0, z1], c0:c1 + 1] = 1
    result[z0:z1 + 1, [c0, c1]] = 1
    return result, [z0, z1, c0, c1]


def rasterize(prepared, arm, size):
    """Build the fixed two-plane groups using native padding and raster resize."""
    modes = variant_modes(NATIVE_VARIANTS[arm])
    shape, planes = prepared
    depth, height, width = shape
    square = max(height, width)
    pad_top, pad_left = (square - height) // 2, (square - width) // 2
    foreground_channel = 2 if modes["foreground_mode"] == "box" else 0
    use_points = modes["background_mode"] == "point"
    supports, rasters, coords, evidence = [], [], [], {}
    for i, (name, plane, column_pad, normal_pad) in enumerate(zip(
            PLANES, planes, (pad_top, pad_left), (pad_left, pad_top))):
        raw = np.zeros((3, *plane.foreground.shape), dtype=np.uint8)
        if foreground_channel == 2:
            raw[2], box_bounds = _box_outline(plane.foreground)
        else:
            raw[0] = plane.foreground
            box_bounds = None
        if use_points:
            projected = plane.background_points[:, [0, i + 1]]
            raw[1, projected[:, 0], projected[:, 1]] = 1
        else:
            raw[1] = plane.background
        support = np.flatnonzero(np.any(raw[foreground_channel], axis=1))
        raster = _resize_preserve(raw, size, (column_pad, square - raw.shape[-1] - column_pad))
        raster = raster.astype(np.float32)
        coord = np.float32(2 * (plane.index + normal_pad) / max(1, square - 1) - 1.0)
        supports.append(support)
        rasters.append(raster)
        coords.append(coord)
        evidence[name] = dict(plane_id=modes["plane_ids"][i], slice_index=plane.index,
            slice_coord=float(coord), box_bounds_inclusive_z_column=box_bounds,
            raw_foreground_z_support=support.tolist(),
            background_points_dhw=plane.background_points.tolist() if use_points else [],
            raw_channel_nonzero=np.count_nonzero(raw, axis=(1, 2)).tolist(),
            raw_channel_sha256=[_raster_sha(channel) for channel in raw],
            raster_channel_nonzero=np.count_nonzero(raster, axis=(1, 2)).tolist(),
            raster_channel_sha256=[_raster_sha(channel) for channel in raster])
    converted = dict(groups=np.stack(rasters), plane_ids=np.asarray(modes["plane_ids"], dtype=np.int64),
                     slice_coords=np.asarray(coords, dtype=np.float32), canonical_depth=depth)
    audit = {key: list(value) if isinstance(value, tuple) else value for key, value in modes.items()}
    audit.update(variant=NATIVE_VARIANTS[arm], planes=evidence, canonical_shape_dhw=list(shape),
        foreground_channel=foreground_channel, prompt_plane_size=size,
        raw_foreground_z_support=np.unique(np.concatenate(supports)).tolist(),
        raw_background_points_per_plane=dict.fromkeys(PLANES, 2 if use_points else 0),
        groups_sha256=_raster_sha(converted["groups"]))
    return converted, audit


def _runtime_paths(case_id, workdir):
    from tools.amos_semantic_fg_scribble_runtime import S21_PROTOCOL, source_guard

    proof = source_guard(production=True)
    protocol_path = ROOT / S21_PROTOCOL
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    proof["weights_sha256"] = _verify_protocol_file(ROOT, WEIGHTS, protocol["assets"])
    proof["base_split_sha256"] = _verify_protocol_file(ROOT, BASE_SPLIT, protocol["training"])
    prepared = protocol_path.parent / "prepared" / f"amos_{case_id}"
    manifest, refs = prepared / "manifest.json", prepared / "refs.json"
    for path in (manifest, refs):
        proof[path.stem + "_sha256"] = _verify_protocol_file(
            ROOT, path.relative_to(ROOT).as_posix(), protocol["prepared"])
    run_dir = Path(workdir).resolve() / "promptgen_native" / case_id
    output_root = (ROOT / "output").resolve()
    if output_root not in run_dir.parents or protocol_path.parent in run_dir.parents:
        raise ValueError("PromptGen audit directory must be under the new project output")
    return dict(root=str(ROOT), manifest=str(manifest), refs=str(refs),
                weights=str(ROOT / WEIGHTS), base_split_json=str(ROOT / BASE_SPLIT),
                run_dir=str(run_dir), protocol=str(protocol_path),
                native_source_sha256=protocol["native_source_sha256"]), proof


class PromptGen:
    """One native model load per fresh process and case, with four arms per class."""

    def __init__(self, case_id, gpu_index, gpu_uuid, workdir, batch=4):
        from tools import amos_prompt_reuse_owner_runtime as runtime
        from tools.amos_prompt_ablation_runtime import _key

        if re.fullmatch(r"amos_\d{4}", case_id) is None:
            raise ValueError("expected one AMOS case ID")
        if gpu_index not in (2, 4, 5, 6) or batch not in (1, 2, 4):
            raise ValueError("invalid native physical GPU or frame batch")
        self.case_id = case_id
        self.runtime = runtime
        self.pending = None
        paths, self.source_proof = _runtime_paths(case_id, workdir)
        Path(paths["run_dir"]).mkdir(parents=True, exist_ok=False)
        self.session = runtime.initialize(dict(case_id=case_id, gpu=gpu_index, gpu_uuid=gpu_uuid,
                                               actual_frame_batch=batch), paths)
        self.inner = self.session.namespace["inner"]
        self.tasks = {_key(task): task for task in self.session.namespace["tasks"]}
        self.session.converted = MethodType(self._converted, self.session)
        self.inner._load_case_image_cache = MethodType(self._shared_ct, self.inner)

    def _converted(self, session, inner, index, class_id):
        key, native_variant, converted, audit, _ = self.pending
        if (inner is not self.inner or int(index) != 0 or int(class_id) != key[2]
                or session.current != key or session.variant != native_variant):
            raise ValueError("requested raster differs from the active native AMOS task/arm")
        session.prompt_audit = audit
        session.expected_groups = converted["groups"]
        session.expected_slice_coords = converted["slice_coords"]
        return converted

    def _shared_ct(self, inner, cache_path):
        prompts = self.pending[-1]
        if (inner is not self.inner or not cache_path or not Path(cache_path).is_file()
                or Path(cache_path).name != prompts.ct_name):
            raise ValueError("PromptGen and paired methods must use the same shared CT cache")
        return prompts.ct

    def evaluate(self, task, prompts):
        from tools.amos_prompt_ablation_runtime import METRICS, _key

        key = _key(task)
        native_task = self.tasks.get(key)
        if self.pending is not None or native_task is None or any(
                native_task[field] != task[field] for field in ("class_name", "local_class_id")):
            raise ValueError("task is outside the signed AMOS case/label manifest")
        ns = self.session.namespace
        cache_name = ns["prompt_index"][key]
        path = (Path(ns["custom"].shared_adaptive_prompt_root) / "metadata"
                / "shared_adaptive_prompt_v1" / "amos" / cache_name)
        if cache_name != prompts.cache_name or _sha(path) != prompts.cache_sha256:
            raise ValueError("paired foreground prompts differ from the native AMOS cache")
        if prompts.ct.dtype != np.float32:
            raise ValueError("PromptGen requires shared canonical raw float32 CT")
        if not np.array_equal(self.inner._load_canonical_target(0, key[2]), prompts.target):
            raise ValueError("paired AMOS target differs from the native label")
        prepared = prepare_prompts(prompts)
        results = []
        for arm, native_variant in NATIVE_VARIANTS.items():
            converted, audit = rasterize(prepared, arm, int(self.inner.prompt_plane_size))
            audit.update(prompt_cache_name=cache_name, prompt_cache_sha256=prompts.cache_sha256,
                         cached_background_sha256=prompts.background_sha256,
                         foreground_box_source="inclusive_bounds_of_cached_foreground_scribble",
                         background_point_source="two_points_on_each_cached_negative_stroke")
            self.pending = key, native_variant, converted, audit, prompts
            try:
                record = self.runtime.evaluate(self.session, native_task, native_variant)
            finally:
                self.pending = None
            if not record["ct_cache_access"] or any(
                    Path(event["path"]).name != prompts.ct_name or event["mode"] != "read_only_hit"
                    for event in record["ct_cache_access"]):
                raise ValueError("native PromptGen did not consume the paired shared case CT")
            if not np.allclose(record["metric_execution"]["events"][0]["spacing_dhw"], prompts.spacing,
                               rtol=1e-6, atol=1e-6):
                raise ValueError("paired methods disagree on native AMOS spacing")
            record["requested_prompt_arm"] = arm
            results.append((arm, {metric: float(record["row"][metric]) for metric in METRICS}, record))
        return results

    def finish(self):
        proof = self.session.finish()
        return dict(proof, source_profile=self.source_proof)
