"""Select prompt planes separately for each 3D connected target instance."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from utils.prompt_utils import select_scribble_plane_slices


def select_component_plane_slices(
    mask: np.ndarray,
    plane: str,
    num_slices: int = 1,
    slice_idx: int | None = None,
    selection_strategy: str = "central_area",
) -> list[int]:
    """Return one (or requested number of) intersecting planes per 26-connected GT object.

    This is only for simulated interactive prompts. The resulting scribbles,
    rather than the full GT bounding box, continue to determine the model ROI.
    """
    foreground = np.asarray(mask, dtype=bool)
    if foreground.ndim != 3:
        raise ValueError(f"expected 3D target, got shape={foreground.shape}")
    if plane not in ("coronal", "sagittal"):
        raise ValueError(f"unsupported prompt plane: {plane}")
    if not foreground.any():
        return []
    if slice_idx is not None:
        raise ValueError("per-component plane selection does not accept a fixed slice")
    labels, count = ndimage.label(foreground, structure=np.ones((3, 3, 3), dtype=bool))
    axis = 2 if plane == "coronal" else 1
    selected: set[int] = set()
    for component_id, bounds in enumerate(ndimage.find_objects(labels), start=1):
        if bounds is None:
            continue
        component = labels[bounds] == component_id
        for local_index in select_scribble_plane_slices(
            component,
            plane=plane,
            num_slices=num_slices,
            selection_strategy=selection_strategy,
        ):
            selected.add(int(bounds[axis].start) + int(local_index))
    return sorted(selected)
