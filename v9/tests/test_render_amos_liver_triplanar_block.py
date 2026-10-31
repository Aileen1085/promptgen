"""Geometry contracts for fixed front/right/top CT and honest full-FOV scale."""
import importlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / 'tools/render_amos_liver_triplanar_block.py'
AFFINE = np.array([[0, .7, 0, -100], [0, 0, .9, -70], [3, 0, 0, -200], [0, 0, 0, 1.]])


class FixedBlockTests(unittest.TestCase):
    def renderer(self):
        self.assertTrue(SOURCE.exists(), 'fixed three-face renderer not implemented')
        return importlib.import_module('tools.render_amos_liver_triplanar_block')

    def test_requested_face_centers_project_front_left_right_side_and_top(self):
        r = self.renderer()
        spec = r.fixed_scene((12, 10, 14), AFFINE, coronal_w=7, sagittal_h=5)
        self.assertEqual(spec['front_face'], 'coronal')
        self.assertEqual(spec['right_face'], 'sagittal')
        self.assertEqual(spec['top_face'], 'axial')
        grids = r.pres.face_grids(spec)
        centers = {k: r.core.to_ras(v.reshape(-1, 3).mean(0), AFFINE) for k, v in grids.items()}
        projected = {k: r.project_camera(v, spec['camera']) for k, v in centers.items()}
        self.assertGreater(projected['sagittal'][0], projected['coronal'][0])
        self.assertGreater(projected['axial'][1], projected['coronal'][1])
        view = r.camera_direction(spec['camera'])
        for normal in spec['outward_normals_ras']:
            self.assertGreater(np.dot(view, normal), 0)

    def test_faces_share_exact_edges_and_preserve_all_real_scan_heights(self):
        r = self.renderer()
        spec = r.fixed_scene((12, 10, 14), AFFINE, 7, 5)
        self.assertEqual(spec['bounds_dhw'], [[0, 12], [0, 6], [7, 14]])
        self.assertEqual(spec['intersection_dhw'], [11, 5, 7])
        grids = r.pres.face_grids(spec)
        points = {k: set(map(tuple, v.reshape(-1, 3))) for k, v in grids.items()}
        self.assertEqual(points['coronal'] & points['sagittal'], {(z, 5, 7) for z in range(12)})
        self.assertEqual(points['coronal'] & points['axial'], {(11, x, 7) for x in range(6)})
        self.assertEqual(points['sagittal'] & points['axial'], {(11, 5, y) for y in range(7, 14)})

    def test_full_volume_limits_include_voxel_edges_and_do_not_depend_on_mask_bbox(self):
        r = self.renderer()
        low, high = r.full_volume_limits((12, 10, 14), AFFINE)
        np.testing.assert_allclose(low, [-100.35, -70.45, -201.5])
        np.testing.assert_allclose(high, [-93.35, -57.85, -165.5])
        np.testing.assert_allclose(high - low, [7., 12.6, 36.])

    def test_scale_fractions_use_complete_ct_not_target_crop(self):
        r = self.renderer()
        mask = np.zeros((8, 10, 12), np.uint8)
        mask[2:6, 2:7, 3:9] = 1
        original = mask.copy()
        info = r.scale_provenance(mask, [5., .7, .8])
        np.testing.assert_allclose(info['full_ct_fov_dhw_mm'], [40., 7., 9.6])
        self.assertEqual(info['mask_voxels'], 120)
        self.assertEqual(info['full_ct_voxels'], 960)
        self.assertAlmostEqual(info['mask_volume_fraction_of_full_ct'], .125)
        self.assertAlmostEqual(info['maximum_axial_area_fraction_of_full_ct'], .25)
        self.assertFalse(info['mask_scaled_separately'])
        np.testing.assert_array_equal(mask, original)

    def test_scale_rejects_invalid_masks_and_nonphysical_spacing(self):
        r = self.renderer()
        with self.assertRaises(ValueError):
            r.scale_provenance(np.full((2, 3, 4), 2), [1, 1, 1])
        with self.assertRaises(ValueError):
            r.scale_provenance(np.ones((2, 3, 4)), [5, 0, 1])

    def test_invalid_cut_planes_cannot_make_a_degenerate_or_outside_block(self):
        r = self.renderer()
        for coronal, sagittal in ((-1, 5), (13, 5), (7, 0), (14, 5), (7, 10)):
            with self.assertRaises(ValueError):
                r.fixed_scene((12, 10, 14), AFFINE, coronal, sagittal)

    def test_source_binding_refuses_changed_same_case_capture(self):
        r = self.renderer()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proof = dict(case_id='amos_0167', sources={})
            for method in r.core.METHODS:
                (root / method).mkdir()
                proof['sources'][method] = {}
                for name, key in (('capture.json', 'capture_json_sha256'), ('capture.npz', 'capture_npz_sha256')):
                    path = root / method / name
                    path.write_bytes((method + name).encode())
                    proof['sources'][method][key] = r.core.file_sha(path)
            path = root / 'proof.json'
            path.write_text(json.dumps(proof), encoding='utf8')
            r.bind_sources(root, path)
            (root / 'v10_2/capture.npz').write_bytes(b'changed-same-case-data')
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                r.bind_sources(root, path)


if __name__ == '__main__':
    unittest.main()
