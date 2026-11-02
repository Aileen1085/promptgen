"""Pure NumPy inputs for full-CT inference without spatial prompts.

Geometry comes only from the canonical CT shape. Scribble mode IDs below are
native model compatibility tags; no scribble or selected plane is supplied.
"""
from __future__ import annotations

from collections.abc import Mapping
import numbers

import numpy as np


_BUNDLE_KEYS = frozenset(("groups", "plane_ids", "slice_coords", "canonical_depth"))


def _positive_integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _canonical_shape(canonical_shape_dhw) -> tuple[int, int, int]:
    if isinstance(canonical_shape_dhw, (str, bytes, Mapping)):
        raise ValueError("canonical_shape_dhw must contain three positive integers")
    try:
        shape = tuple(canonical_shape_dhw)
    except TypeError as error:
        raise ValueError("canonical_shape_dhw must contain three positive integers") from error
    if len(shape) != 3:
        raise ValueError("canonical_shape_dhw must contain three positive integers")
    return tuple(_positive_integer(n, "canonical_shape_dhw dimension") for n in shape)


def make_empty_bundle(canonical_shape_dhw, prompt_plane_size):
    """Build the native zero-group schema from CT shape and plane size only."""
    depth, _, _ = _canonical_shape(canonical_shape_dhw)
    size = _positive_integer(prompt_plane_size, "prompt_plane_size")
    return {
        "groups": np.empty((0, 3, size, size), dtype=np.float32),
        "plane_ids": np.empty((0,), dtype=np.int64),
        "slice_coords": np.empty((0,), dtype=np.float32),
        "canonical_depth": depth,
    }


def full_ct_indices(canonical_shape_dhw):
    """Return every original axial frame, without prompt or GT-derived bounds."""
    depth, _, _ = _canonical_shape(canonical_shape_dhw)
    return np.arange(depth, dtype=np.int32)


def validate_empty_bundle(bundle, canonical_shape_dhw, prompt_plane_size):
    """Reject phantom spatial input, extra metadata, or a shortened CT depth."""
    depth, _, _ = _canonical_shape(canonical_shape_dhw)
    size = _positive_integer(prompt_plane_size, "prompt_plane_size")
    if not isinstance(bundle, Mapping):
        raise TypeError("empty bundle must be a mapping")
    if set(bundle) != _BUNDLE_KEYS:
        raise ValueError("empty bundle must contain exactly the four native schema keys")
    for key, shape, dtype in (
        ("groups", (0, 3, size, size), np.float32),
        ("plane_ids", (0,), np.int64),
        ("slice_coords", (0,), np.float32),
    ):
        array = bundle[key]
        if not isinstance(array, np.ndarray):
            raise TypeError(f"{key} must be a NumPy array")
        if array.shape != shape or array.dtype != np.dtype(dtype):
            raise ValueError(f"{key} must have shape {shape} and dtype {np.dtype(dtype)}")
    if _positive_integer(bundle["canonical_depth"], "canonical_depth") != depth:
        raise ValueError("canonical_depth must equal the full original CT depth")


def semantic_modes():
    """Describe absent spatial input separately from native compatibility tags.

    Only foreground role/type 0 remains in the model output. Background scribble
    type 3 is a compatibility tag, and its role is inactive.
    """
    return {
        "spatial_prompt_policy": "none",
        "active_planes": (),
        "plane_ids": (),
        "foreground_mode": "scribble",
        "background_mode": "scribble",
        "foreground_mode_id": 0,
        "background_mode_id": 0,
        "foreground_type_id": 0,
        "background_type_id": 3,
        "background_active": False,
        "role_ids": (0,),
        "type_ids": (0,),
    }
