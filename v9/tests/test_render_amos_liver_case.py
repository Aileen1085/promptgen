"""Geometry/appearance regressions; no model or optional renderer imports needed."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "tools/render_amos_liver_case.py"


class RendererTests(unittest.TestCase):
    def renderer(self):
        self.assertTrue(SOURCE.exists(), "scientific renderer is not implemented")
        spec = importlib.util.spec_from_file_location("amos_renderer", SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_palette_can_change_for_rerender_without_changing_masks(self):
        r = self.renderer()
        r.configure_colors("#FFAA22", "#EE8899", "#88DDFF")
        image = np.zeros((2, 2), np.float32)
        mask = np.array([[True, False], [False, False]])
        original = mask.copy()
        np.testing.assert_array_equal(r.overlay_prediction(image, mask, alpha=1)[0, 0], [255, 170, 34])
        np.testing.assert_array_equal(mask, original)
        self.assertEqual(r.BG_COLOR, "#EE8899")
        self.assertEqual(r.GT_COLOR, "#88DDFF")
        with self.assertRaises(ValueError):
            r.configure_colors("orange", "#EE8899", "#88DDFF")
        self.assertEqual(r.FG_COLOR, "#FFAA22")

    def test_radiological_landmarks_and_points_share_pixels(self):
        r = self.renderer()
        a = np.arange(3 * 4 * 5).reshape(3, 4, 5)
        # RAS high X (patient right) lies at image left; high Y at top.
        self.assertEqual(r.oriented_slice(a, "axial", 1).shape, (5, 4))
        self.assertEqual(r.oriented_slice(a, "axial", 1)[0, 0], a[1, 3, 4])
        self.assertEqual(r.oriented_slice(a, "coronal", 2)[0, 0], a[2, 3, 2])
        self.assertEqual(r.oriented_slice(a, "sagittal", 2)[0, 0], a[2, 2, 4])
        points = np.array([[1, 3, 4], [0, 0, 0]])
        np.testing.assert_array_equal(r.project_points(points, a.shape, "axial", 1), [[0, 0]])
        np.testing.assert_array_equal(r.project_points(points, a.shape, "coronal", 4), [[0, 1]])
        np.testing.assert_array_equal(r.project_points(points, a.shape, "sagittal", 3), [[0, 1]])

    def test_full_axial_fov_and_physically_anisotropic_extent(self):
        r = self.renderer()
        a = np.zeros((5, 7, 11), np.float32)
        pred = np.zeros_like(a, dtype=bool)
        pred[2, 3, 5] = True
        rgb = r.overlay_prediction(r.oriented_slice(a, "axial", 2),
                                   r.oriented_slice(pred, "axial", 2), 50, 350, .32)
        self.assertEqual(rgb.shape, (11, 7, 3))
        self.assertEqual(r.slice_spacing((3., .7, .9), "axial"), (.9, .7))
        self.assertEqual(r.slice_spacing((3., .7, .9), "coronal"), (3., .7))
        self.assertEqual(r.slice_spacing((3., .7, .9), "sagittal"), (3., .9))
        # Pixel-centre grids include both FOV ends, including with decimation.
        grid = r.plane_grid(((0, 5), (0, 7), (0, 11)), "axial", 2, max_side=4)
        np.testing.assert_array_equal(grid[0, 0], [2, 6, 10])
        np.testing.assert_array_equal(grid[-1, -1], [2, 0, 0])
        affine = np.array([[0, .7, 0, -20], [0, 0, .9, -10], [3, 0, 0, 5], [0, 0, 0, 1.]])
        xyz = r.to_ras(grid, affine)
        np.testing.assert_allclose(xyz[0, 0], [-15.8, -1., 11.])
        np.testing.assert_allclose(xyz[-1, -1], [-20., -10., 11.])

    def test_translucent_prediction_preserves_ct_and_has_exact_palette(self):
        r = self.renderer()
        ct = np.array([[-1000., 50., 1000.], [50., 50., 50.]])
        mask = np.array([[0, 1, 0], [0, 0, 0]], bool)
        out = r.overlay_prediction(ct, mask, 50, 350, .32)
        plain = r.window_rgb(ct, 50, 350)
        np.testing.assert_array_equal(out[~mask], plain[~mask])
        expected = np.rint(.68 * plain[0, 1] + .32 * np.array([245, 184, 59])).astype(np.uint8)
        np.testing.assert_array_equal(out[0, 1], expected)
        self.assertEqual(r.FG_COLOR, "#F5B83B")
        self.assertEqual(r.BG_COLOR, "#F6A0A0")
        self.assertEqual(r.GT_COLOR, "#70E5FF")

    def test_slice_choice_uses_gt_and_native_prompt_frames_not_prediction(self):
        r = self.renderer()
        gt = np.zeros((5, 7, 8), bool)
        gt[1] = True  # larger, but does not have both actual prompt roles
        gt[3, :4, :4] = True
        gt[4, :2, :2] = True
        fg = np.array([[3, 1, 1], [4, 1, 1]])
        bg = np.array([[3, 6, 7], [4, 6, 7]])
        self.assertEqual(r.choose_axial_z(gt, fg, bg), (3, "max_gt_area_with_native_fg_bg_points"))
        self.assertEqual(r.choose_axial_z(gt, fg, np.empty((0, 3), int)), (1, "max_gt_area_fallback"))
        self.assertEqual(r.choose_axial_z(gt, fg, bg, 2), (2, "explicit_display_override"))
        with self.assertRaises(ValueError):
            r.choose_axial_z(gt, fg, bg, 10)

    def test_shared_views_leave_2d_uncropped_and_crop_3d_in_mm(self):
        r = self.renderer()
        gt = np.zeros((20, 30, 40), bool)
        gt[8:12, 10:15, 20:26] = True
        view = r.make_view_spec(gt, (3., 1., 2.), np.array([[10, 11, 22]]),
                                np.array([[10, 20, 22]]), 22, 12, elev=22, azim=115,
                                margin_mm=4.)
        self.assertEqual(view["axial_bounds_dhw"], [[0, 20], [0, 30], [0, 40]])
        self.assertEqual(view["display_crop_3d_dhw"], [[6, 14], [6, 19], [18, 28]])
        self.assertEqual(view["camera"], {"elev": 22., "azim": 115., "projection": "ortho"})
        self.assertEqual(view["axial_z"], 10)

    def test_prompt_panels_color_only_actual_support_and_no_gt(self):
        r = self.renderer()
        a = np.zeros((3, 5, 7), np.float32)
        rgb = r.prompt_rgb(a, "coronal", 3, [[1, 2, 3]], [[2, 1, 3]], 50, 350)
        base = r.window_rgb(r.oriented_slice(a, "coronal", 3), 50, 350)
        np.testing.assert_array_equal(rgb[1, 2], [245, 184, 59])
        np.testing.assert_array_equal(rgb[0, 3], [246, 160, 160])
        self.assertEqual(np.count_nonzero(np.any(rgb != base, axis=-1)), 2)

    def test_output_directory_refuses_overwrite_before_writing(self):
        r = self.renderer()
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "figure"
            r.reserve_output(output)
            sentinel = output / "existing.png"
            sentinel.write_bytes(b"preserve")
            with self.assertRaises(FileExistsError):
                r.reserve_output(output)
            self.assertEqual(sentinel.read_bytes(), b"preserve")

    def test_pair_validation_rejects_different_geometry_and_invalid_coordinates(self):
        r = self.renderer()
        meta = dict(schema="amos_liver_capture_v1", case_id="amos_0167", global_class_id=123,
                    label_id=6, shape_dhw=[3, 5, 7], canonical_axes="ZXY",
                    spacing_dhw_mm=[3., 1., 2.], ct_cache_path="shared/case.npy",
                    label_path="labels/case.nii.gz", selected_coronal_w=3, selected_sagittal_h=2,
                    affine_dhw_ras=[[0, 1, 0, 0], [0, 0, 2, 0], [3, 0, 0, 0], [0, 0, 0, 1]])
        def capture(method):
            arrays = dict(prediction_dhw=np.zeros((3, 5, 7), np.uint8),
                          prompt_fg_dhw=np.array([[1, 2, 3]], np.int32),
                          prompt_bg_dhw=np.array([[1, 4, 6]], np.int32))
            if method == "v10_2":
                for plane in ("coronal", "sagittal"):
                    for role in ("fg", "bg"):
                        arrays[f"{plane}_{role}_dhw"] = np.array([[1, 2, 3]], np.int32)
            return dict(meta=dict(meta, method=method), arrays=arrays)
        pair = {method: capture(method) for method in ("v10_2", "native_sam2")}
        r.validate_pair(pair)
        pair["native_sam2"]["meta"]["spacing_dhw_mm"] = [1., 1., 1.]
        with self.assertRaisesRegex(ValueError, "geometry"):
            r.validate_pair(pair)
        pair["native_sam2"]["meta"]["spacing_dhw_mm"] = [3., 1., 2.]
        pair["native_sam2"]["arrays"]["prompt_fg_dhw"][0, 1] = 99
        with self.assertRaisesRegex(ValueError, "coordinates"):
            r.validate_pair(pair)

    def test_invalid_camera_or_plane_cannot_silently_change_shared_geometry(self):
        r = self.renderer()
        gt = np.ones((3, 5, 7), bool)
        points = np.array([[1, 2, 3]])
        with self.assertRaisesRegex(ValueError, "plane"):
            r.make_view_spec(gt, (3., 1., 2.), points, points, 99, 2)
        with self.assertRaisesRegex(ValueError, "camera"):
            r.make_view_spec(gt, (3., 1., 2.), points, points, 3, 2, elev=float("nan"))
        with self.assertRaisesRegex(ValueError, "spacing"):
            r.make_view_spec(gt, (0., 1., 2.), points, points, 3, 2)


if __name__ == "__main__":
    unittest.main()
