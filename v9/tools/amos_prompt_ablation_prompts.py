"""Pure NumPy prompt interventions for the frozen AMOS E490 evaluation.

The caller supplies native selected-plane groups before resizing.  Cached FG
scribbles/BG point coordinates stay exact; box outlines and BG scribbles come
from the native plane builder.  No CT, GT or fallback ROI is accepted here.
"""
from __future__ import annotations

import hashlib
import numbers
from typing import Mapping

import numpy as np

from infer.v9_2_eight_source_eval_protocol import (
    _resize_preserve,
    prompt_package_to_groups,
)


VARIANTS = (
    "baseline", "sagittal_only", "coronal_only", "fg_box_bg_point",
    "fg_scribble_bg_scribble", "fg_box_bg_scribble",
)
_PLANES = ("coronal", "sagittal")
_PLANE_IDS = {"coronal": 1, "sagittal": 0}


def variant_modes(variant: str) -> dict:
    """Return native bundle mode IDs separately from learned adapter type IDs."""
    if variant not in VARIANTS:
        raise ValueError(f"unsupported prompt variant: {variant!r}")
    planes = (("sagittal",) if variant == "sagittal_only" else
              ("coronal",) if variant == "coronal_only" else _PLANES)
    foreground = "box" if variant in ("fg_box_bg_point", "fg_box_bg_scribble") else "scribble"
    background = "scribble" if variant in ("fg_scribble_bg_scribble", "fg_box_bg_scribble") else "point"
    return {
        "active_planes": planes,
        "plane_ids": tuple(_PLANE_IDS[p] for p in planes),
        "foreground_mode": foreground,
        "background_mode": background,
        "foreground_mode_id": int(foreground == "box"),
        "background_mode_id": int(background == "point"),
        "foreground_type_id": 2 if foreground == "box" else 0,
        "background_type_id": 4 if background == "point" else 3,
    }


def _integer(value, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _mask(value, name: str, shape=None) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "buif" or not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError(f"{name} must contain finite nonnegative mask values")
    if shape is not None and tuple(array.shape) != tuple(shape):
        raise ValueError(f"{name} shape mismatch: {array.shape} != {shape}")
    return (array > 0).astype(np.uint8)


def _validated_inputs(package, native_groups):
    if not isinstance(package, Mapping) or not isinstance(native_groups, Mapping):
        raise TypeError("package and native_groups must be mappings")
    if set(native_groups) != set(_PLANES):
        raise ValueError("native groups must contain exactly coronal and sagittal planes")
    try:
        volumes = {p: _mask(package[p], p) for p in _PLANES}
        shape = volumes["coronal"].shape
        if len(shape) != 3 or min(shape) <= 0 or volumes["sagittal"].shape != shape:
            raise ValueError("cached foreground volumes must share one nonempty DHW shape")
        depth, height, width = shape
        slices = {p: _integer(package[p + "_slice"], p + "_slice", 0) for p in _PLANES}
        if slices["coronal"] >= width or slices["sagittal"] >= height:
            raise ValueError("cached prompt slice lies outside canonical volume")
        fg = {
            "coronal": volumes["coronal"][:, :, slices["coronal"]].copy(),
            "sagittal": volumes["sagittal"][:, slices["sagittal"], :].copy(),
        }
        for plane in _PLANES:
            if np.count_nonzero(volumes[plane]) != np.count_nonzero(fg[plane]):
                raise ValueError(f"{plane} foreground support lies outside its selected plane")
        native = {
            "coronal": _mask(native_groups["coronal"], "coronal native group", (3, depth, height)),
            "sagittal": _mask(native_groups["sagittal"], "sagittal native group", (3, depth, width)),
        }
        points = np.asarray(package["background_points_dhw"])
        if (points.ndim != 2 or points.shape[1] != 3 or points.dtype.kind not in "uif"
                or not np.all(np.isfinite(points)) or not np.all(points == np.floor(points))):
            raise ValueError("background points must be an Nx3 array of finite integer coordinates")
        if np.any(points < 0) or np.any(points >= np.asarray(shape)):
            raise ValueError("background point lies outside canonical DHW volume")
        point_planes = [str(p) for p in package["background_point_planes"]]
        if len(point_planes) != len(points):
            raise ValueError("background point and plane counts differ")
        counts = {p: 0 for p in _PLANES}
        for point, plane in zip(points, point_planes):
            if plane not in _PLANE_IDS:
                raise ValueError(f"unsupported background point plane: {plane}")
            normal = 2 if plane == "coronal" else 1
            if int(point[normal]) != slices[plane]:
                raise ValueError(f"{plane} background point is outside its selected plane")
            counts[plane] += 1
        if counts != {"coronal": 2, "sagittal": 2}:
            raise ValueError(f"expected two raw background points per plane: {counts}")
    except KeyError as error:
        raise ValueError(f"missing prompt input field: {error.args[0]}") from error
    return tuple(int(n) for n in shape), slices, fg, native, points.astype(np.int64), point_planes


def _sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def build_variant(package, native_groups, variant: str, prompt_plane_size: int):
    """Return ``(native-schema converted groups, JSON-serializable audit)``.

    Native group channels are [FG scribble, BG scribble, FG box outline].
    Interventions replace raw channels before the identical native pixel-preserving
    resize.  Absent planes are removed, so no blank group can contribute tokens.
    Audit cardinalities describe raw point records and raster pixels separately.
    """
    modes = variant_modes(variant)
    size = _integer(prompt_plane_size, "prompt_plane_size", 1)
    shape, slices, fg, native, points, point_planes = _validated_inputs(package, native_groups)
    depth, height, width = shape
    # Reuse the native canonical coordinate convention, padding and result schema.
    exact = prompt_package_to_groups(package, prompt_plane_size=size)
    square = max(height, width)
    pad_top, pad_left = (square - height) // 2, (square - width) // 2
    active = modes["active_planes"]
    foreground_channel = 2 if modes["foreground_mode"] == "box" else 0
    raw_points, raster_pixels, raw_counts, planes, supports, rasters = {}, {}, {}, {}, [], []
    for plane in active:
        raw = np.zeros_like(native[plane])
        if foreground_channel == 2:
            raw[2] = native[plane][2]
        else:
            raw[0] = fg[plane]
        z = np.flatnonzero(np.any(raw[foreground_channel] > 0, axis=1)).astype(np.int32)
        if not z.size:
            raise ValueError(f"{plane} actual foreground {modes['foreground_mode']} support is empty")
        if foreground_channel == 2:
            # Native inclusive 2D GT-plane bbox is an outline, including valid
            # one-row/one-column degeneracies.  No GT membership test belongs here.
            coordinates = np.argwhere(raw[2] > 0)
            r0, c0 = coordinates.min(axis=0)
            r1, c1 = coordinates.max(axis=0)
            outline = np.zeros_like(raw[2])
            outline[[r0, r1], c0:c1 + 1] = 1
            outline[r0:r1 + 1, c0] = 1
            outline[r0:r1 + 1, c1] = 1
            if not np.array_equal(raw[2], outline):
                raise ValueError(f"{plane} foreground box must be a native inclusive rectangle outline")
        supports.append(z)
        if modes["background_mode"] == "scribble":
            raw[1] = native[plane][1]
            if not np.any(raw[1]):
                raise ValueError(f"{plane} native background scribble is empty")
            raw_points[plane] = 0
        else:
            raw_points[plane] = 0
            for (z_value, y, x), source_plane in zip(points, point_planes):
                if source_plane == plane:
                    raw[1, int(z_value), int(y if plane == "coronal" else x)] = 1
                    raw_points[plane] += 1
        raw_counts[plane] = (raw_points[plane] if modes["background_mode"] == "point"
                             else int(np.count_nonzero(raw[1])))
        before = pad_top if plane == "coronal" else pad_left
        after = square - raw.shape[-1] - before
        raster = _resize_preserve(raw, size, (before, after)).astype(np.float32)
        rasters.append(raster)
        raster_pixels[plane] = int(np.count_nonzero(raster[1]))
        index = _PLANES.index(plane)
        planes[plane] = {
            "plane_id": _PLANE_IDS[plane],
            "slice_index": slices[plane],
            "slice_coord": float(exact["slice_coords"][index]),
            "foreground_channel": foreground_channel,
            "foreground_type_id": modes["foreground_type_id"],
            "background_type_id": modes["background_type_id"],
            "raw_foreground_z_support": z.tolist(),
            "raw_background_point_count": raw_points[plane],
            "raw_shape": list(raw.shape),
            "raw_dtype": str(raw.dtype),
            "raw_sha256": _sha(raw),
            "raw_channel_sha256": [_sha(c) for c in raw],
            "raw_channel_nonzero": [int(np.count_nonzero(c)) for c in raw],
            "raster_shape": list(raster.shape),
            "raster_dtype": str(raster.dtype),
            "raster_sha256": _sha(raster),
            "raster_channel_sha256": [_sha(c) for c in raster],
            "raster_channel_nonzero": [int(np.count_nonzero(c)) for c in raster],
        }
    indices = [_PLANES.index(p) for p in active]
    converted = {
        "groups": np.stack(rasters, axis=0),
        "plane_ids": np.asarray(exact["plane_ids"])[indices].copy(),
        "slice_coords": np.asarray(exact["slice_coords"])[indices].copy(),
        "canonical_depth": depth,
        "background_points_per_plane": raw_counts,
        "background_mode": "point" if modes["background_mode"] == "point" else "training_scribble",
    }
    audit = {
        "variant": variant,
        **{k: list(v) if isinstance(v, tuple) else v for k, v in modes.items()},
        "canonical_shape_dhw": list(shape),
        "prompt_plane_size": size,
        "foreground_channel": foreground_channel,
        "raw_foreground_z_support": np.unique(np.concatenate(supports)).tolist(),
        "raw_background_points_per_plane": raw_points,
        "raster_background_pixels_per_plane": raster_pixels,
        "groups_sha256": _sha(converted["groups"]),
        "planes": planes,
    }
    return converted, audit


def roi_indices(depth, z_support, context_slices, context_ratio, minframes) -> np.ndarray:
    """Apply native adaptive-scribble ROI arithmetic to actual raw FG support.

    No GT fallback is permitted.  Reserved/expanded bounds remain the native
    runtime's responsibility after this initial contiguous index interval.
    """
    depth = _integer(depth, "depth", 1)
    context_slices = _integer(context_slices, "context_slices", 0)
    minframes = _integer(minframes, "minframes", 1)
    if (isinstance(context_ratio, (bool, np.bool_)) or not isinstance(context_ratio, numbers.Real)
            or not np.isfinite(context_ratio) or context_ratio < 0):
        raise ValueError("context_ratio must be finite and nonnegative")
    z_all = np.asarray(z_support)
    if (z_all.ndim != 1 or not z_all.size or z_all.dtype.kind not in "uif"
            or not np.all(np.isfinite(z_all)) or not np.all(z_all == np.floor(z_all))):
        raise ValueError("foreground z support must be a nonempty array of integer indices")
    if np.any(z_all < 0) or np.any(z_all >= depth):
        raise ValueError("foreground z support lies outside canonical depth")
    # Keep the arithmetic in v9_2_adaptive_scribble.py:284-296 unchanged.
    z0, z1 = int(z_all.min()), int(z_all.max())
    span = max(1, z1 - z0 + 1)
    context = max(int(context_slices), int(round(span * context_ratio)))
    start = max(0, z0 - context)
    end = min(depth, z1 + context + 1)
    if end - start < int(minframes):
        extra = int(minframes) - (end - start)
        before = min(start, extra // 2)
        after = min(depth - end, extra - before)
        before = min(start, extra - after)
        start -= before
        end += after
    return np.arange(start, end, dtype=np.int32)
