import importlib.util
from pathlib import Path

import numpy as np
import pytest


PATH = Path(__file__).resolve().parents[1] / 'infer/native_sam2_physical_prompts.py'


def api():
    assert PATH.is_file(), 'physical scribble implementation is missing'
    spec = importlib.util.spec_from_file_location('native_physical_prompts', PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_physical_swap_flip_preserves_world_coordinates_and_values():
    m = api()
    raw = np.arange(3 * 4 * 5).reshape(3, 4, 5)
    source = np.diag([2., 3., 4., 1.])
    mapping = np.array([[1, 0, 0, 0], [0, 0, -1, 3], [0, -1, 0, 4], [0, 0, 0, 1.]])
    target = source @ mapping
    out = m.reorient_grid(raw, source, target, (3, 5, 4))
    assert np.array_equal(out, raw.transpose(0, 2, 1)[:, ::-1, ::-1])
    point, error = m.map_voxels(np.array([[1., 2., 3.]]), source, target)
    assert np.allclose(point, [[1, 1, 1]]) and error < 1e-9


def test_non_grid_affine_is_rejected():
    m = api()
    target = np.eye(4); target[1, 3] = 0.3
    with pytest.raises(ValueError, match='grid'):
        m.reorient_grid(np.zeros((3, 4, 5)), np.eye(4), target, (3, 4, 5))


def test_gaussian_is_physical_anisotropic_and_background_signed():
    m = api()
    fg = np.zeros((41, 41), bool); fg[20, 20] = True
    prior = m.gaussian_prior(fg, np.array([[5, 5]]), (2., 1.), sigma_mm=2., amplitude=4., output_size=41)
    assert prior[20, 20] == pytest.approx(4.)
    assert prior[5, 5] == pytest.approx(-4.)
    assert prior[21, 20] == pytest.approx(prior[20, 22], abs=.02)
    assert abs(prior[0, 40]) < 1e-4


def test_points_are_deterministic_and_only_on_existing_scribble():
    m = api()
    fg = np.zeros((25, 31), bool); fg[10, 3:25] = True; fg[3:20, 12] = True
    a = m.sample_scribble_points(fg, (1., 2.), 8)
    b = m.sample_scribble_points(fg, (1., 2.), 8)
    assert len(a) == 8 and np.array_equal(a, b)
    assert all(fg[int(y), int(x)] for x, y in a)
    assert len(np.unique(a, axis=0)) == 8


def test_pad_geometry_matches_image_and_point_order():
    m = api()
    image = np.zeros((5, 9)); image[2, 6] = 1
    padded, offset = m.pad_square(image)
    points = m.pad_points(np.array([[6., 2.]]), offset)
    assert padded.shape == (9, 9) and np.array_equal(points, [[6., 4.]])
    assert padded[int(points[0, 1]), int(points[0, 0])] == 1


def test_roi_preserves_background_z_and_uses_physical_context():
    m = api()
    fg = np.zeros((100, 10, 10), bool); fg[40:50, 4, 5] = True
    bounds = m.prompt_z_bounds(fg, np.array([[25, 0, 0], [70, 0, 0]]), spacing_z=2.)
    assert bounds == (25, 71)
    assert m.prompt_z_bounds(fg, np.zeros((0, 3)), spacing_z=2.) == (36, 54)


def test_expansion_only_on_prompt_connected_component_and_cap():
    m = api()
    fg = np.zeros((60, 5, 5), bool); fg[30, 2, 2] = True
    pred = np.zeros((20, 5, 5), bool); pred[:, 2, 2] = True
    assert m.expand_z_bounds(pred, fg, (20, 40), (20, 40)) == (10, 50)
    pred = np.zeros((40, 5, 5), bool); pred[:, 2, 2] = True
    assert m.expand_z_bounds(pred, fg, (10, 50), (20, 40)) == (10, 50)
    disconnected = np.zeros((20, 5, 5), bool); disconnected[0, 0, 0] = True
    assert m.expand_z_bounds(disconnected, fg, (20, 40), (20, 40)) == (20, 40)
