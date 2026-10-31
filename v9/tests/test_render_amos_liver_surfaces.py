"""Dependency-light tests for complete-mask surface preparation and RAS geometry."""
import importlib
from pathlib import Path
import tempfile
import unittest

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "tools/render_amos_liver_surfaces.py"


class SurfaceTests(unittest.TestCase):
    def renderer(self):
        self.assertTrue(SOURCE.exists(), "optional complete-mask surface renderer is missing")
        return importlib.import_module("tools.render_amos_liver_surfaces")

    def test_complete_volume_keeps_distant_components_and_boundary_voxels(self):
        r = self.renderer()
        mask = np.zeros((9, 13, 17), np.uint8)
        mask[0, 0, 0] = 1
        mask[4:7, 5:9, 8:12] = 1
        mask[-1, -1, -1] = 1
        volume = r.prepare_volume(mask)
        self.assertEqual(volume.shape, (11, 15, 19))
        np.testing.assert_array_equal(volume[1:-1, 1:-1, 1:-1], mask)
        self.assertEqual(int(volume.sum()), int(mask.sum()))
        self.assertEqual(volume[1, 1, 1], 1)
        self.assertEqual(volume[-2, -2, -2], 1)
        self.assertFalse(volume[0].any())
        self.assertFalse(volume[-1].any())
        self.assertFalse(volume[:, 0].any())
        self.assertFalse(volume[:, :, -1].any())

    def test_half_voxel_mesh_vertices_use_true_dhw_affine_and_remove_only_padding(self):
        r = self.renderer()
        affine = np.array([[0, .7, 0, -100], [0, 0, .9, -70], [3, 0, 0, -200], [0, 0, 0, 1.]])
        # A surface surrounding native DHW=(0,0,0) has +/-0.5 voxel offsets.
        padded_vertices = np.array([[.5, 1, 1], [1, .5, 1], [1, 1, .5], [9.5, 13, 17]])
        ras = r.vertices_to_ras(padded_vertices, affine)
        np.testing.assert_allclose(ras, [[-100, -70, -201.5], [-100.35, -70, -200],
                                         [-100, -70.45, -200], [-91.6, -55.6, -174.5]])

    def test_invalid_or_surrogate_values_are_not_treated_as_actual_binary_masks(self):
        r = self.renderer()
        with self.assertRaises(ValueError):
            r.prepare_volume(np.ones((4, 5)))
        with self.assertRaises(ValueError):
            r.prepare_volume(np.full((3, 4, 5), .7))
        with self.assertRaises(ValueError):
            r.prepare_volume(np.zeros((3, 4, 5), np.uint8))

    def test_surface_display_frame_is_shared_and_does_not_change_extracted_volume(self):
        r = self.renderer()
        gt = np.zeros((9, 13, 17), bool)
        gt[4:7, 5:9, 8:12] = True
        affine = np.array([[0, .7, 0, -100], [0, 0, .9, -70], [3, 0, 0, -200], [0, 0, 0, 1.]])
        lo, hi = r.display_limits(gt, affine, margin_mm=2.)
        # Same voxel-rounded/clipped physical frame as the primary CT cube.
        np.testing.assert_allclose(lo, [-98.6, -65.5, -191])
        np.testing.assert_allclose(hi, [-92.3, -57.4, -179])
        mask = gt.copy()
        mask[0, 0, 0] = True
        volume = r.prepare_volume(mask)
        self.assertEqual(int(volume.sum()), int(mask.sum()))
        self.assertEqual(volume[1, 1, 1], 1)

    def test_surface_output_cannot_overwrite_primary_or_previous_figures(self):
        r = self.renderer()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "figures"
            r.reserve_output(path)
            sentinel = path / "raw_ct_cube.png"
            sentinel.write_bytes(b"primary")
            with self.assertRaises(FileExistsError):
                r.reserve_output(path)
            self.assertEqual(sentinel.read_bytes(), b"primary")


if __name__ == "__main__":
    unittest.main()
