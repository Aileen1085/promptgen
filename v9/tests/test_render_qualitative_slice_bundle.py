"""Synthetic slice-export contract tests; retain fixtures without bulk deletion."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / 'tools/render_qualitative_slice_bundle.py'


class SliceBundleTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(FILE.exists(), 'Slice-only renderer required')
        spec = importlib.util.spec_from_file_location('slice_bundle', FILE)
        self.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.m)
        scratch = ROOT / 'output/_synthetic_slice_bundle_tests'
        scratch.mkdir(parents=True, exist_ok=True)
        self.root = Path(tempfile.mkdtemp(dir=scratch))

    def fixture(self):
        gray = np.arange(256, dtype=np.uint8).reshape(16, 16)
        mask = np.zeros((16, 16), np.uint8)
        mask[4:12, 5:10] = 1
        assets = {}
        for name, arr in [('ct', gray), ('mask', mask)]:
            path = self.root / f'{name}.png'
            Image.fromarray(arr).save(path)
            assets[name] = dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        cases = []
        for source in self.m.SOURCES:
            cases.append(dict(source=source, case_id='synthetic', class_name='organ', slice_index=47,
                              shape_hw=[16, 16], spacing_hw=[2, 1], zoom_crop_yx=[3, 13, 3, 12],
                              display_transform={}, ct=assets['ct'], gt=assets['mask'],
                              methods=[dict(name=name, mask=assets['mask'], dice_3d=.123,
                                            metrics_evidence='synthetic full-volume row attestation')
                                       for name in self.m.METHODS]))
        manifest = dict(schema='five_source_display_slice_bundle_v1', export_scope='display_slices_only',
                        window_hu=[-160, 240], full_volume_audit=dict(verified=True, method_rows=40,
                        evidence='Synthetic fixture, not medical data'), cases=cases)
        path = self.root / 'manifest.json'
        path.write_text(json.dumps(manifest), encoding='utf-8')
        return manifest, path, gray

    def test_slice_view_rejects_another_z_and_ct_gray_roundtrip_exact(self):
        manifest, path, gray = self.fixture()
        cases, _ = self.m.load_bundle(path)
        case = cases[0]
        from tools.render_five_source_qualitative import blend_slice
        reproduced = blend_slice(case['ct'][47, :, :], None, (-160, 240))
        np.testing.assert_array_equal(reproduced[..., 0], gray)
        with self.assertRaises(IndexError):
            case['ct'][46, :, :]
        self.assertFalse(case['ct'][47, :, :].flags.writeable)

    def test_full_and_zoom_export_preserve_attested_3d_dice_not_slice_dice(self):
        _, path, _ = self.fixture()
        out = self.root / 'rendered'
        report = self.m.render_bundle(path, out, panel_pixels=100)
        for view in ('full', 'zoom'):
            self.assertTrue((out / view / 'five_source_comparison.png').is_file())
            self.assertTrue((out / view / 'five_source_comparison.pdf').is_file())
            self.assertTrue((out / view / '01_AMOS.pdf').is_file())
        self.assertEqual(report['cases'][0]['methods'][0]['dice_3d'], .123)  # slice masks are identical
        self.assertEqual(report['cases'][0]['slice_index'], 47)
        self.assertIn('not recomputed', (out / 'caption.txt').read_text())
        with self.assertRaises(FileExistsError):
            self.m.render_bundle(path, out)

    def test_hash_nonbinary_and_bad_dimensions_are_rejected_before_output(self):
        manifest, path, _ = self.fixture()
        manifest['cases'][0]['ct']['sha256'] = '0' * 64
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'SHA'):
            self.m.render_bundle(path, self.root / 'bad')
        self.assertFalse((self.root / 'bad').exists())
        for arr in (np.full((16, 16), 2, np.uint8), np.zeros((8, 16), np.uint8)):
            Image.fromarray(arr).save(self.root / 'bad.png')
            case = manifest['cases'][0]
            case['ct']['sha256'] = hashlib.sha256((self.root / 'ct.png').read_bytes()).hexdigest()
            case['gt'] = dict(path='bad.png', sha256=hashlib.sha256((self.root / 'bad.png').read_bytes()).hexdigest())
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                self.m.load_bundle(path)

    def test_zip_input_and_strict_audit_scope(self):
        manifest, path, _ = self.fixture()
        zipped = self.root / 'bundle.zip'
        with zipfile.ZipFile(zipped, 'w') as archive:
            for name in ('manifest.json', 'ct.png', 'mask.png'):
                archive.write(self.root / name, arcname=name)
        cases, _ = self.m.load_bundle(zipped)
        self.assertEqual(len(cases), 5)
        manifest['full_volume_audit']['method_rows'] = 8
        path.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            self.m.load_bundle(path)

    def test_binary_255_masks_and_common_crop_transform_preserve_actual_pixels(self):
        manifest, path, gray = self.fixture()
        mask = np.array(Image.open(self.root / 'mask.png')) * np.uint8(255)
        Image.fromarray(mask).save(self.root / 'mask255.png')
        manifest['cases'][0]['gt'] = dict(path='mask255.png', sha256=hashlib.sha256((self.root / 'mask255.png').read_bytes()).hexdigest())
        manifest['cases'][0]['display_transform'] = dict(transpose=True, flip_y=True)
        path.write_text(json.dumps(manifest))
        cases, _ = self.m.load_bundle(path)
        from tools.render_five_source_qualitative import _display, blend_slice
        case = dict(cases[0], crop_yx=cases[0]['zoom_crop_yx'])
        actual = blend_slice(_display(case['ct'], case), None, manifest['window_hu'])
        np.testing.assert_array_equal(actual[..., 0], gray[3:13, 3:12].T[::-1])
        np.testing.assert_array_equal(case['gt'][47, :, :], case['methods'][0]['mask'][47, :, :])
        self.assertEqual(case['spacing_dhw'], [None, 2, 1])  # no invented depth spacing

    def test_unverified_metric_evidence_and_escaping_paths_are_rejected(self):
        manifest, path, _ = self.fixture()
        manifest['cases'][0]['methods'][0]['metrics_evidence'] = ''
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'evidence'):
            self.m.load_bundle(path)
        manifest['cases'][0]['methods'][0]['metrics_evidence'] = 'synthetic full-volume row'
        manifest['cases'][0]['ct']['path'] = '../outside.png'
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'relative'):
            self.m.load_bundle(path)


if __name__ == '__main__':
    unittest.main()
