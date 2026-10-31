"""Pure-NumPy contracts for the additive AMOS display revision."""
import importlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / "tools/render_amos_liver_revision.py"
AFFINE = np.array([[0, .7, 0, -100], [0, 0, .9, -70], [3, 0, 0, -200], [0, 0, 0, 1.]])


class RevisionTests(unittest.TestCase):
    def renderer(self):
        self.assertTrue(SOURCE.exists(), "revision renderer has not been implemented")
        return importlib.import_module("tools.render_amos_liver_revision")

    def test_main_line_is_one_longest_skeleton_path_inside_actual_support(self):
        r = self.renderer()
        pixels = np.array([(3, y) for y in range(1, 8)] + [(x, 3) for x in (1, 2, 4, 5, 6)])
        line, info = r.main_line(pixels, row_mm=5., col_mm=1.)
        self.assertEqual({tuple(line[0]), tuple(line[-1])}, {(3, 1), (3, 7)})
        self.assertTrue(all(tuple(p) in set(map(tuple, pixels)) for p in line))
        self.assertTrue(np.all(np.max(np.abs(np.diff(line, axis=0)), axis=1) == 1))
        self.assertEqual(len(set(map(tuple, line))), len(line))
        self.assertEqual(info["actual_support_pixels"], len(pixels))
        self.assertGreaterEqual(info["path_length_mm"], 30.)

    def test_thick_branched_support_is_thinned_without_creating_coordinates(self):
        r = self.renderer()
        pixels = np.array([(x, y) for y in range(2, 13) for x in (4, 5, 6)] +
                          [(x, y) for x in range(5, 10) for y in (6, 7, 8)])
        original = pixels.copy()
        line, info = r.main_line(pixels, 5., .8)
        self.assertGreater(len(line), 5)
        self.assertLess(len(line), len(set(map(tuple, pixels))))
        self.assertTrue(all(tuple(p) in set(map(tuple, pixels)) for p in line))
        np.testing.assert_array_equal(pixels, original)
        self.assertEqual(info["display_only"], True)

    def test_long_closed_support_component_is_not_lost_to_a_short_disconnected_line(self):
        r = self.renderer()
        pixels = np.array([(x, y) for x in range(9) for y in range(9) if x in (0, 8) or y in (0, 8)] + [(15, 0), (15, 1)])
        line, info = r.main_line(pixels, 1., 1.)
        self.assertGreater(info["path_length_mm"], 10.)
        self.assertTrue(np.all(line[:, 0] <= 8))

    def test_plot_air_margins_preserve_physical_mm_and_do_not_extend_ct_bounds(self):
        r = self.renderer()
        self.assertTrue(hasattr(r, "physical_view_limits"), "explicit physical air-margin helper is missing")
        bounds = [[0, 12], [1, 9], [2, 12]]
        original = [row[:] for row in bounds]
        low, high = r.physical_view_limits(AFFINE, bounds, headroom_mm=12.)
        np.testing.assert_allclose(low, [-103.3, -72.2, -212.])
        np.testing.assert_allclose(high, [-90.4, -56.1, -155.])
        self.assertEqual(bounds, original)

    def test_context_depth_includes_all_target_and_real_slices_to_ct_boundaries(self):
        r = self.renderer()
        gt = np.zeros((12, 10, 14), bool)
        gt[3:10, 2:8, 4:11] = True
        bounds, info = r.context_bounds(gt, (3., .7, .9), context=6)
        self.assertEqual(bounds[0], [0, 12])
        self.assertEqual(info["gt_depth_halfopen"], [3, 10])
        self.assertEqual(info["available_context_below_above"], [3, 2])
        spec = r.scene_spec(gt, gt, gt, AFFINE, bounds, coronal_w=7, sagittal_h=5)
        self.assertEqual(spec["bounds_dhw"][0], [0, 12])
        self.assertEqual(spec["intersection_dhw"][0], 11)
        grids = r.pres.face_grids(spec)
        self.assertEqual(int(grids["axial"][..., 0].min()), 11)
        self.assertEqual(set(np.unique(grids["coronal"][..., 0])), set(range(12)))
        self.assertIn(spec["front_face"], ("coronal", "sagittal"))

    def test_prompt_views_share_depth_physical_width_and_equal_scale(self):
        r = self.renderer()
        shape, spacing = (12, 10, 14), (3., .7, .9)
        arrays = {f"{p}_{role}_dhw": np.array([[3, 2, 4], [9, 7, 10]])
                  for p in ("coronal", "sagittal") for role in ("fg", "bg")}
        frame = r.prompt_frame(shape, spacing, [[0, 12], [1, 9], [2, 12]], arrays)
        self.assertEqual(frame["depth_halfopen"], [0, 12])
        self.assertEqual(frame["physical_height_mm"], 36.)
        self.assertEqual(frame["physical_width_mm"], 9.)
        self.assertEqual(frame["aspect"], "equal")
        self.assertEqual(frame["png_size"], [1152, 1152])

    def test_result_selection_uses_actual_paired_slice_scores_area_quality_and_spacing(self):
        r = self.renderer()
        gt = np.ones((10, 4, 5), bool)
        gt[9] = False
        gt[9, 0, 0] = True
        v10 = gt.copy()
        v10[0] = False  # Exclude a poor v10 slice despite low native recall.
        native = np.zeros_like(gt)
        for z, amount in enumerate([0, 17, 4, 5, 16, 6, 17, 8, 9, 0]):
            native[z].flat[:amount] = True
        selected, policy, rows = r.select_results(gt, v10, native, 5., count=3, min_spacing_mm=15.)
        self.assertEqual(selected, [2, 5, 8])
        self.assertAlmostEqual(rows[2]["native_recall"], .2)
        self.assertAlmostEqual(rows[2]["v10_dice"], 1.)
        self.assertEqual(policy["fallback"], None)
        self.assertNotIn(9, selected)

    def test_native_prompt_frames_have_both_actual_roles_and_result_plus_max_gt_anchors(self):
        r = self.renderer()
        gt = np.ones((10, 4, 5), bool)
        gt[6] = True
        gt[:6, :1] = False
        gt[7:, :1] = False
        fg = np.array([[z, 1, 1] for z in (1, 2, 4, 6, 8)])
        bg = np.array([[z, 3, 4] for z in (2, 4, 6, 8)])
        frames, policy = r.native_frames(gt, fg, bg, [1, 2, 4], count=3)
        self.assertEqual(frames, [2, 4, 6])
        self.assertEqual(policy["max_gt_anchor"], 6)
        self.assertNotIn(1, frames)

    def test_known_proofs_bind_current_capture_json_and_npz_before_loading_data(self):
        r = self.renderer()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proof = dict(case_id="amos_0167", sources={})
            for method in r.core.METHODS:
                (root / method).mkdir()
                proof["sources"][method] = {}
                for name, key in (("capture.json", "capture_json_sha256"), ("capture.npz", "capture_npz_sha256")):
                    path = root / method / name
                    path.write_bytes(f"{method}-{name}".encode())
                    proof["sources"][method][key] = r.core.file_sha(path)
            manifest = root / "proof.json"
            manifest.write_text(json.dumps(proof), encoding="utf-8")
            r.bind_sources(root, manifest)
            (root / "native_sam2/capture.npz").write_bytes(b"same-case-different-mask")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                r.bind_sources(root, manifest)


if __name__ == "__main__":
    unittest.main()
