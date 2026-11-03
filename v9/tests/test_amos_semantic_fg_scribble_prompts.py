"""Exact cached foreground, native raster/ROI, and zero-background boundaries."""
import copy
import importlib
from pathlib import Path
import unittest
import numpy as np
from tests.test_amos_prompt_ablation_prompts import fixture
from tools import amos_prompt_ablation_prompts as native


class ForegroundPromptTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((Path(__file__).resolve().parents[1]/'tools/amos_semantic_fg_scribble_prompts.py').is_file(), 'FG-only prompt implementation required')
        self.m=importlib.import_module('tools.amos_semantic_fg_scribble_prompts')

    def test_exact_cached_fg_rasters_coordinates_and_raw_support(self):
        package,groups=fixture();before=copy.deepcopy(package)
        for size in (1,2,11,19):
            baseline,a=native.build_variant(package,groups,'fg_scribble_bg_scribble',size)
            actual,b=self.m.build_variant(package,groups,'semantic_fg_scribble',size)
            np.testing.assert_array_equal(actual['groups'][:,0],baseline['groups'][:,0])
            self.assertFalse(np.any(actual['groups'][:,1:]))
            for key in ('plane_ids','slice_coords'):np.testing.assert_array_equal(actual[key],baseline[key])
            self.assertEqual(b['raw_foreground_z_support'],a['raw_foreground_z_support'])
            for plane in ('coronal','sagittal'):
                for field in ('raw_channel_sha256','raster_channel_sha256'):
                    self.assertEqual(b['planes'][plane][field][0],a['planes'][plane][field][0])
            self.assertEqual(actual['background_mode'],'none')
            self.assertEqual(actual['background_points_per_plane'],{'coronal':0,'sagittal':0})
        np.testing.assert_array_equal(package['coronal'],before['coronal'])

    def test_native_none_id_and_absent_background_type(self):
        modes=self.m.variant_modes('semantic_fg_scribble')
        self.assertEqual((modes['foreground_mode'],modes['foreground_mode_id'],modes['foreground_type_id']),('scribble',0,0))
        self.assertEqual((modes['background_mode'],modes['background_mode_id'],modes['background_active'],modes['background_type_id']),('none',2,False,None))

    def test_baseline_delegates_unchanged(self):
        p,g=fixture()
        for variant in native.VARIANTS:
            x,a=self.m.build_variant(p,g,variant,13);y,b=native.build_variant(p,g,variant,13)
            self.assertEqual(a,b);self.assertEqual(self.m.variant_modes(variant),native.variant_modes(variant))
            for key in x:
                if isinstance(x[key],np.ndarray):np.testing.assert_array_equal(x[key],y[key])
                else:self.assertEqual(x[key],y[key])

    def test_roi_uses_cached_support_never_gt_bbox(self):
        p,g=fixture();_,a=self.m.build_variant(p,g,'semantic_fg_scribble',13)
        p['foreground_bbox_dhw']=[0,0,0,1,1,1]
        _,b=self.m.build_variant(p,g,'semantic_fg_scribble',13)
        self.assertEqual(a,b)
        np.testing.assert_array_equal(self.m.roi_indices(12,a['raw_foreground_z_support'],1,.2,4),native.roi_indices(12,a['raw_foreground_z_support'],1,.2,4))

    def test_invalid_inputs_and_empty_fg_fail_closed(self):
        for corrupt in ('empty','outside','nan'):
            p,g=fixture()
            if corrupt=='empty':p['coronal'][:]=0
            if corrupt=='outside':p['coronal'][0,0,0]=1
            if corrupt=='nan':g['coronal']=g['coronal'].astype(float);g['coronal'][0,0,0]=np.nan
            with self.subTest(corrupt=corrupt),self.assertRaises(ValueError):self.m.build_variant(p,g,'semantic_fg_scribble',13)
        with self.assertRaises(ValueError):self.m.variant_modes('unknown')
