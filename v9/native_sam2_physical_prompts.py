"""Non-learned, support-preserving physical geometry for native SAM2 prompts."""
from __future__ import annotations

import math
import numpy as np
from scipy.ndimage import label
from scipy.spatial import cKDTree


def map_voxels(points, source_affine, target_affine):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    homogeneous = np.concatenate((points, np.ones((len(points), 1))), axis=1)
    world = homogeneous @ np.asarray(source_affine, dtype=np.float64).T
    mapped = world @ np.linalg.inv(np.asarray(target_affine, dtype=np.float64)).T
    rounded = np.concatenate((np.rint(mapped[:, :3]), np.ones((len(points), 1))), axis=1)
    error = float(np.max(np.linalg.norm((rounded @ target_affine.T - world)[:, :3], axis=1))) if len(points) else 0.
    return mapped[:, :3], error


def reorient_grid(array, source_affine, target_affine, target_shape):
    """Exact signed-permutation grid transform; never interpolate annotations."""
    array = np.asarray(array)
    mapping = np.linalg.inv(source_affine) @ target_affine
    rotation = np.rint(mapping[:3, :3]).astype(int)
    if not np.allclose(mapping[:3, :3], rotation, atol=1e-5) or not np.allclose(mapping[:3, 3], np.rint(mapping[:3, 3]), atol=1e-4):
        raise ValueError('source and target are not on the same permutation grid')
    if not np.all(np.abs(rotation).sum(axis=0) == 1) or not np.all(np.abs(rotation).sum(axis=1) == 1):
        raise ValueError('invalid grid permutation')
    axes = np.argmax(np.abs(rotation), axis=0)
    out = array.transpose(tuple(int(i) for i in axes))
    for target_axis, source_axis in enumerate(axes):
        sign = rotation[source_axis, target_axis]
        expected_offset = array.shape[source_axis] - 1 if sign < 0 else 0
        if abs(mapping[source_axis, 3] - expected_offset) > 1e-4:
            raise ValueError('grid translation does not preserve the complete source extent')
        if sign < 0:
            out = np.flip(out, axis=target_axis)
    if out.shape != tuple(target_shape):
        raise ValueError('grid target shape mismatch')
    return out


def pad_square(image):
    image = np.asarray(image)
    h, w = image.shape[-2:]
    side = max(h, w); top = (side - h) // 2; left = (side - w) // 2
    padding = [(0, 0)] * image.ndim
    padding[-2:] = [(top, side - h - top), (left, side - w - left)]
    return np.pad(image, padding, mode='constant'), (left, top)


def pad_points(points_xy, offset):
    return np.asarray(points_xy, dtype=np.float32).reshape(-1, 2) + np.asarray(offset, dtype=np.float32)


def sample_scribble_points(mask, spacing_yx, max_points=16, min_spacing_mm=0.):
    """Deterministic physical farthest-point sampling; no GT or new locations."""
    if int(max_points) < 0 or float(min_spacing_mm) < 0:
        raise ValueError('point cap must be nonnegative; zero means all support')
    coords = np.argwhere(np.asarray(mask) > 0)
    if not len(coords):
        return np.zeros((0, 2), dtype=np.float32)
    if int(max_points) == 0 and float(min_spacing_mm) == 0:
        return coords[:, ::-1].astype(np.float32)
    physical = coords * np.asarray(spacing_yx)
    center = physical.mean(axis=0)
    selected = [int(np.argmin(((physical - center) ** 2).sum(axis=1)))]
    distance = np.full(len(coords), np.inf)
    cap = int(max_points) or len(coords)
    for _ in range(min(cap, len(coords)) - 1):
        distance = np.minimum(distance, ((physical - physical[selected[-1]]) ** 2).sum(axis=1))
        distance[selected] = -1.
        candidate = int(np.argmax(distance))
        if distance[candidate] < float(min_spacing_mm) ** 2:
            break
        selected.append(candidate)
    return coords[selected, ::-1].astype(np.float32)


def training_background_scribble_coords(target, coronal_slice, sagittal_slice, margin_ratio=.08):
    """Current v9 training vertical lines, simulated on the two prompt planes.

    GT is used here solely to simulate a user background scribble, not to
    select a prediction ROI. The resulting sparse coordinates are the prompt.
    """
    target = np.asarray(target, dtype=bool)
    coordinates = []
    for plane, index in (('coronal', int(coronal_slice)), ('sagittal', int(sagittal_slice))):
        mask = target[:, :, index] if plane == 'coronal' else target[:, index, :]
        hits = np.argwhere(mask)
        if not len(hits):
            raise ValueError('empty selected target prompt plane')
        r0, c0 = hits.min(axis=0); r1, c1 = hits.max(axis=0)
        margin = max(2, int(round(min(mask.shape) * float(margin_ratio))))
        row0 = max(0, int(r0) - margin)
        row1 = min(mask.shape[0] - 1, int(r1) + margin)
        candidates = []
        if c0 - margin >= 0: candidates.append(int(c0) - margin)
        if c1 + margin < mask.shape[1]: candidates.append(int(c1) + margin)
        candidates.extend([0, mask.shape[1] - 1, max(0, int(c0) - 1), min(mask.shape[1] - 1, int(c1) + 1)])
        scribble = np.zeros(mask.shape, bool)
        seen = set()
        for col in candidates:
            col = int(np.clip(col, 0, mask.shape[1] - 1))
            if col in seen: continue
            seen.add(col)
            line = np.zeros(mask.shape, bool)
            line[row0:row1 + 1, col] = True; line[mask] = False
            if line.any():
                scribble = line; break
        if not scribble.any() and (~mask).any():
            outside = np.argwhere(~mask)
            col = int(outside[len(outside) // 2, 1])
            line_rows = outside[outside[:, 1] == col, 0]
            if len(line_rows):
                scribble[int(line_rows.min()):int(line_rows.max()) + 1, col] = True
                scribble[mask] = False
        rows, cols = np.nonzero(scribble)
        fixed = np.full(len(rows), index)
        coordinates.append(np.column_stack((rows, cols, fixed) if plane == 'coronal' else (rows, fixed, cols)))
    return np.unique(np.concatenate(coordinates), axis=0).astype(np.int32)


def gaussian_prior(foreground, background_yx, spacing_yx, sigma_mm=2., amplitude=4., output_size=256):
    """Signed Gaussian evidence, neutral zero far away; not a full target mask."""
    foreground = np.asarray(foreground) > 0
    padded, (left, top) = pad_square(foreground)
    side = padded.shape[0]
    coords = (np.arange(output_size, dtype=np.float64) + .5) * side / output_size - .5
    yy, xx = np.meshgrid(coords, coords, indexing='ij')
    query = np.stack((yy.reshape(-1), xx.reshape(-1)), axis=1) * np.asarray(spacing_yx)

    def field(points):
        if not len(points):
            return np.zeros((output_size, output_size), np.float32)
        distances = cKDTree(points * np.asarray(spacing_yx)).query(query)[0]
        values = np.exp(-.5 * (distances / float(sigma_mm)) ** 2)
        values[distances > 4. * float(sigma_mm)] = 0.
        return values.reshape(output_size, output_size).astype(np.float32)

    fg = field(np.argwhere(padded))
    bg = np.asarray(background_yx, dtype=np.float64).reshape(-1, 2) + [top, left]
    return (float(amplitude) * (fg - field(bg))).astype(np.float32)


def prompt_z_bounds(foreground, background_dhw, spacing_z, context_mm=8., fraction=.2):
    coords = np.argwhere(foreground)
    if not len(coords):
        raise ValueError('empty foreground prompt')
    start, end = int(coords[:, 0].min()), int(coords[:, 0].max()) + 1
    context = max(int(math.ceil(context_mm / float(spacing_z))), int(math.ceil((end - start) * fraction)))
    start -= context; end += context
    bg = np.asarray(background_dhw).reshape(-1, 3)
    if len(bg):
        start = min(start, int(bg[:, 0].min())); end = max(end, int(bg[:, 0].max()) + 1)
    return max(0, start), min(foreground.shape[0], end)


def expand_z_bounds(prediction, foreground, bounds, initial_bounds):
    start, end = bounds; initial_start, initial_end = initial_bounds
    components, _ = label(np.asarray(prediction) > 0)
    touched = np.unique(components[np.asarray(foreground[start:end]) > 0])
    touched = touched[touched != 0]
    if not len(touched):
        return bounds
    connected = np.isin(components, touched)
    growth = int(math.ceil((end - start) * .5))
    limit = int(math.ceil((initial_end - initial_start) * .5))
    lower = max(0, initial_start - limit); upper = min(foreground.shape[0], initial_end + limit)
    if connected[0].any():
        start = max(lower, start - growth)
    if connected[-1].any():
        end = min(upper, end + growth)
    return start, end


def prompt_bounds_3d(foreground, background_dhw, spacing_dhw, context_mm=8., fraction=.2):
    """Prompt-only local crop, including every original negative point."""
    coords = np.argwhere(foreground)
    bg = np.asarray(background_dhw).reshape(-1, 3)
    z = prompt_z_bounds(foreground, bg, spacing_dhw[0], context_mm, fraction)
    positions = np.concatenate((coords, bg), axis=0)
    bounds = [z]
    for axis in (1, 2):
        start, end = int(positions[:, axis].min()), int(positions[:, axis].max()) + 1
        margin = max(int(math.ceil(context_mm / spacing_dhw[axis])), int(math.ceil((end - start) * fraction)))
        bounds.append((max(0, start - margin), min(foreground.shape[axis], end + margin)))
    return tuple(bounds)


def expand_bounds_3d(prediction, foreground, bounds, initial_bounds):
    slices = tuple(slice(start, end) for start, end in bounds)
    components, _ = label(np.asarray(prediction) > 0)
    touched = np.unique(components[np.asarray(foreground[slices]) > 0])
    touched = touched[touched != 0]
    if not len(touched):
        return bounds
    connected = np.isin(components, touched)
    expanded = []
    for axis, ((start, end), (initial_start, initial_end)) in enumerate(zip(bounds, initial_bounds)):
        growth = int(math.ceil((end - start) * .5))
        limit = int(math.ceil((initial_end - initial_start) * .5))
        if np.take(connected, 0, axis=axis).any():
            start = max(0, initial_start - limit, start - growth)
        if np.take(connected, -1, axis=axis).any():
            end = min(foreground.shape[axis], initial_end + limit, end + growth)
        expanded.append((start, end))
    return tuple(expanded)


def annotation_order(foreground, background_dhw, bounds_z):
    start, end = bounds_z
    fg = sorted(z for z in np.flatnonzero(foreground.any(axis=(1, 2))).tolist() if start <= z < end)
    bg_only = sorted(z for z in set(np.asarray(background_dhw)[:, 0].tolist()) - set(fg) if start <= z < end)
    return fg + bg_only, bg_only


def probability_gaussian_prior(foreground, background_yx, spacing_yx, sigma_mm=2., output_size=256):
    return np.clip(gaussian_prior(foreground, background_yx, spacing_yx,
        sigma_mm=sigma_mm, amplitude=1., output_size=output_size), 0., 1.)
