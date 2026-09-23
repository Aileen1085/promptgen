"""Map MAGIC raw NRRD prompt coordinates to the model's NIfTI space."""

from __future__ import annotations

from typing import Mapping

import numpy as np


def align_magic_prompt_package(
    package: Mapping[str, object],
    *,
    expected_shape_dhw: tuple[int, int, int] | None = None,
) -> dict[str, object]:
    raw_coronal = np.asarray(package["coronal"])
    raw_sagittal = np.asarray(package["sagittal"])
    if raw_coronal.ndim != 3 or raw_coronal.shape != raw_sagittal.shape:
        raise ValueError("MAGIC prompt volumes must share a DHW shape")
    depth, height, width = map(int, raw_coronal.shape)
    model_shape = (depth, width, height)
    if expected_shape_dhw is not None and model_shape != tuple(expected_shape_dhw):
        raise ValueError(
            f"transformed MAGIC prompt shape {model_shape} differs from target "
            f"{tuple(expected_shape_dhw)}"
        )

    def transform(mask: np.ndarray) -> np.ndarray:
        return np.transpose(mask, (0, 2, 1))[:, ::-1, ::-1].copy()

    coronal_slice = int(package["coronal_slice"])
    sagittal_slice = int(package["sagittal_slice"])
    if not (0 <= coronal_slice < width and 0 <= sagittal_slice < height):
        raise ValueError("MAGIC prompt slice is outside the raw volume")

    result = dict(package)
    result["coronal"] = transform(raw_sagittal)
    result["sagittal"] = transform(raw_coronal)
    result["coronal_slice"] = height - 1 - sagittal_slice
    result["sagittal_slice"] = width - 1 - coronal_slice

    points = np.asarray(package["background_points_dhw"], dtype=np.int32).reshape(-1, 3)
    if points.size and (
        np.any(points[:, 0] < 0) or np.any(points[:, 0] >= depth)
        or np.any(points[:, 1] < 0) or np.any(points[:, 1] >= height)
        or np.any(points[:, 2] < 0) or np.any(points[:, 2] >= width)
    ):
        raise ValueError("MAGIC background point is outside the raw volume")
    transformed_points = points.copy()
    transformed_points[:, 1] = width - 1 - points[:, 2]
    transformed_points[:, 2] = height - 1 - points[:, 1]
    result["background_points_dhw"] = transformed_points
    swap = {"coronal": "sagittal", "sagittal": "coronal"}
    result["background_point_planes"] = [
        swap[str(plane)] for plane in package["background_point_planes"]
    ]
    if len(result["background_point_planes"]) != len(points):
        raise ValueError("MAGIC background point and plane counts differ")

    if "bbox_dhw" in package:
        z0, y0, x0, z1, y1, x1 = map(int, np.asarray(package["bbox_dhw"]).reshape(6))
        result["bbox_dhw"] = np.asarray(
            [z0, width - 1 - x1, height - 1 - y1,
             z1, width - 1 - x0, height - 1 - y0],
            dtype=np.int32,
        )
    for suffix in ("ratio", "radius"):
        coronal_key, sagittal_key = f"coronal_{suffix}", f"sagittal_{suffix}"
        if coronal_key in package and sagittal_key in package:
            result[coronal_key], result[sagittal_key] = (
                package[sagittal_key], package[coronal_key]
            )
    return result
