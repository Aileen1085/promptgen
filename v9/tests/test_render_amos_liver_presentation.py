"""Scientific geometry and prompt-stroke tests without optional rendering packages."""
import importlib
import itertools
from pathlib import Path
import tempfile
import unittest

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "tools/render_amos_liver_presentation.py"
AFFINE = np.array([[0, .7, 0, -100], [0, 0, .9, -70], [3, 0, 0, -200], [0, 0, 0, 1.]])


class PresentationTests(unittest.TestCase):
    def renderer(self):
        self.assertTrue(SOURCE.exists(), "three-face presentation renderer is not implemented")
        return importlib.import_module("tools.render_amos_liver_presentation")

    def test_octant_retains_maximum_gt_and_camera_faces_all_exposed_normals(self):
        r = self.renderer()
        gt = np.zeros((8, 10, 12), bool)
        gt[:3, 5:, 6:] = True
        bounds = [[0, 8], [0, 10], [0, 12]]
        spec = r.choose_cutaway(gt, bounds, [3, 4, 5], AFFINE)
        self.assertEqual(spec["signs_dhw"], [-1, 1, 1])
        self.assertEqual(spec["bounds_dhw"], [[0, 4], [4, 10], [5, 12]])
        self.assertEqual(spec["retained_gt_voxels"], int(gt.sum()))
        self.assertEqual(len(spec["octant_gt_counts"]), 8)
        elev, azim = np.radians([spec["camera"]["elev"], spec["camera"]["azim"]])
        direction = np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])
        normals = -np.array(spec["signs_dhw"])[:, None] * (AFFINE[:3, :3] / [3., .7, .9]).T
        self.assertTrue(np.all(normals @ direction > .5))

    def test_three_faces_stay_at_real_slice_coordinates_and_meet_at_one_corner(self):
        r = self.renderer()
        gt = np.zeros((8, 10, 12), bool)
        gt[:3, 5:, 6:] = True
        spec = r.choose_cutaway(gt, [[0, 8], [0, 10], [0, 12]], [3, 4, 5], AFFINE)
        faces = r.face_grids(spec)
        self.assertEqual(set(faces), {"axial", "coronal", "sagittal"})
        for plane, axis in (("axial", 0), ("coronal", 2), ("sagittal", 1)):
            grid = faces[plane]
            self.assertTrue(np.all(grid[..., axis] == [3, 4, 5][axis]))
            self.assertTrue(np.any(np.all(grid == [3, 4, 5], axis=-1)))
            self.assertTrue(np.all(grid >= np.array(spec["bounds_dhw"])[:, 0]))
            self.assertTrue(np.all(grid < np.array(spec["bounds_dhw"])[:, 1]))
        # Three exposed CT faces, no internal crossed plane through another face.
        corners = list(itertools.product(*[(lo, hi - 1) for lo, hi in spec["bounds_dhw"]]))
        self.assertEqual(len(r.visible_edges(spec)), 9)
        self.assertEqual(len(corners), 8)

    def test_strokes_join_only_adjacent_actual_pixels_and_keep_isolated_support(self):
        r = self.renderer()
        # Coronal points project to (x,y)=(1,2),(2,2),(2,1),(5,0).
        points = np.array([[1, 5, 2], [1, 4, 2], [2, 4, 2], [3, 1, 2], [1, 5, 2]])
        segments, isolated = r.support_strokes(points, (4, 7, 6), "coronal", 2)
        support = {(1, 2), (2, 2), (2, 1), (5, 0)}
        self.assertEqual(len(segments), 3)
        np.testing.assert_array_equal(isolated, [[5, 0]])
        for segment in segments:
            self.assertTrue(all(tuple(p) in support for p in segment))
            self.assertEqual(np.max(np.abs(segment[1] - segment[0])), 1)
        empty, single = r.support_strokes([[1, 4, 3]], (4, 7, 6), "coronal", 2)
        self.assertEqual(empty.shape, (0, 2, 2))
        self.assertEqual(single.shape, (0, 2))

    def test_gt_contour_clipping_leaves_open_ends_without_fake_crop_boundary(self):
        r = self.renderer()
        path = np.array([[-1., 1, 1], [1, 1, 1], [3, 1, 1], [3, 3, 1], [1, 3, 1]])
        chunks = r.clip_polyline(path, [[0, 3], [0, 3], [0, 3]])
        self.assertEqual(len(chunks), 1)
        np.testing.assert_allclose(chunks[0], [[0, 1, 1], [1, 1, 1], [2, 1, 1]])
        self.assertFalse(np.array_equal(chunks[0][0], chunks[0][-1]))

    def test_contour_clipping_does_not_modify_shared_display_bounds(self):
        r = self.renderer()
        bounds = np.array([[0., 3.], [0., 3.], [0., 3.]])
        original = bounds.copy()
        r.clip_polyline(np.array([[0., 1, 1], [2., 1, 1]]), bounds)
        np.testing.assert_array_equal(bounds, original)

    def test_face_contours_inverse_orientation_matches_real_slice_pixels(self):
        r = self.renderer()
        np.testing.assert_allclose(r.pixels_to_dhw([[0, 0], [2.5, 1.5]], (4, 7, 6), "axial", 2),
                                   [[2, 6, 5], [2, 3.5, 3.5]])
        np.testing.assert_allclose(r.pixels_to_dhw([[0, 0]], (4, 7, 6), "coronal", 2), [[3, 6, 2]])
        np.testing.assert_allclose(r.pixels_to_dhw([[0, 0]], (4, 7, 6), "sagittal", 2), [[3, 2, 5]])

    def test_axial_sources_must_match_capture_case_and_shared_slice(self):
        r = self.renderer()
        manifest = dict(case_id="amos_0167", view=dict(axial_z=72),
                        window=dict(level_hu=50., width_hu=350.), prediction_alpha=.32)
        r.validate_axial_manifest(manifest, "amos_0167", 72)
        with self.assertRaises(ValueError):
            r.validate_axial_manifest(manifest, "amos_0167", 71)
        with self.assertRaises(ValueError):
            r.validate_axial_manifest(manifest, "amos_0001", 72)

    def test_same_case_axial_manifest_rejects_changed_capture_json_or_npz_for_either_method(self):
        r = self.renderer()
        self.assertTrue(hasattr(r, "validate_capture_binding"), "current capture hash binding is missing")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = dict(case_id="amos_0167", sources={})
            for method in r.core.METHODS:
                (root / method).mkdir()
                manifest["sources"][method] = {}
                for name, field in (("capture.json", "capture_json_sha256"), ("capture.npz", "capture_npz_sha256")):
                    path = root / method / name
                    path.write_bytes(f"original-{method}-{name}".encode())
                    manifest["sources"][method][field] = r.core.file_sha(path)
            r.validate_capture_binding(manifest, root)
            for method in r.core.METHODS:
                for name in ("capture.json", "capture.npz"):
                    with self.subTest(method=method, file=name):
                        path = root / method / name
                        original = path.read_bytes()
                        path.write_bytes(original + b"-new-same-case-capture")
                        with self.assertRaisesRegex(ValueError, "capture.*mismatch"):
                            r.validate_capture_binding(manifest, root)
                        path.write_bytes(original)

    def test_new_presentation_directory_refuses_existing_output(self):
        r = self.renderer()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "presentation"
            r.reserve_output(path)
            with self.assertRaises(FileExistsError):
                r.reserve_output(path)


if __name__ == "__main__":
    unittest.main()
