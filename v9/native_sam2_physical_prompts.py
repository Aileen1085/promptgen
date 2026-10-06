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


def sample_scribble_points(mask, spacing_yx, max_points=16):
    """Deterministic physical farthest-point sampling; no GT or new locations."""
    coords = np.argwhere(np.asarray(mask) > 0)
    if not len(coords):
        return np.zeros((0, 2), dtype=np.float32)
    physical = coords * np.asarray(spacing_yx)
    center = physical.mean(axis=0)
    selected = [int(np.argmin(((physical - center) ** 2).sum(axis=1)))]
    distance = np.full(len(coords), np.inf)
    for _ in range(min(int(max_points), len(coords)) - 1):
        distance = np.minimum(distance, ((physical - physical[selected[-1]]) ** 2).sum(axis=1))
        distance[selected] = -1.
        selected.append(int(np.argmax(distance)))
    return coords[selected, ::-1].astype(np.float32)


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
