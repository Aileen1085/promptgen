import importlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / 'tools/render_amos_liver_triplanar_context.py'
AFFINE = np.array([[0,.7,0,-100],[0,0,.9,-70],[3,0,0,-200],[0,0,0,1.]])


class ContextBlockTests(unittest.TestCase):
    def renderer(self):
        self.assertTrue(SOURCE.exists(), 'tissue-context renderer not implemented')
        return importlib.import_module('tools.render_amos_liver_triplanar_context')

    def test_posterior_quadrant_still_places_sagittal_on_right_and_axial_on_top(self):
        r=self.renderer();s=r.context_scene((12,10,14),AFFINE,7,3)
        self.assertEqual(s['bounds_dhw'],[[0,12],[3,10],[0,8]])
        self.assertEqual(s['intersection_dhw'],[11,3,7])
        grids=r.base.pres.face_grids(s)
        xy={p:r.base.project_camera(r.base.core.to_ras(g.reshape(-1,3).mean(0),AFFINE),s['camera']) for p,g in grids.items()}
        self.assertGreater(xy['sagittal'][0],xy['coronal'][0])
        self.assertGreater(xy['axial'][1],xy['coronal'][1])
        for normal in s['outward_normals_ras']:
            self.assertGreater(np.dot(normal,r.base.camera_direction(s['camera'])),0)

    def test_low_side_cut_is_selected_by_actual_gt_area_without_using_prediction(self):
        r=self.renderer();g=np.zeros((6,12,14),bool)
        g[2:5,2:9,4:10]=True;g[1:6,7:9,3:11]=True
        w,h,info=r.choose_planes(g,.35)
        areas=np.count_nonzero(g,axis=(0,2))
        self.assertEqual(h,next(i for i,v in enumerate(areas) if v>=.35*areas.max()))
        self.assertEqual(w,int(np.argmax(np.count_nonzero(g,axis=(0,1)))))
        self.assertEqual(info['selection_input'],'GT display geometry only; predictions not used')

    def test_real_faces_meet_along_complete_shared_edges(self):
        r=self.renderer();s=r.context_scene((12,10,14),AFFINE,7,3)
        p={k:set(map(tuple,g.reshape(-1,3))) for k,g in r.base.pres.face_grids(s).items()}
        self.assertEqual(p['coronal']&p['sagittal'],{(z,3,7) for z in range(12)})
        self.assertEqual(p['axial']&p['sagittal'],{(11,3,y) for y in range(8)})

    def test_scale_line_is_100mm_on_front_side_of_block(self):
        r=self.renderer();s=r.context_scene((89,512,512),np.array([[0,.782,0,0],[0,0,.782,0],[5,0,0,0],[0,0,0,1.]]),309,209)
        line=r.scale_line(s,np.array([[0,.782,0,0],[0,0,.782,0],[5,0,0,0],[0,0,0,1.]]))
        self.assertAlmostEqual(np.linalg.norm(line[1]-line[0]),100.)
        self.assertGreater(line[0,1],309*.782)

    def test_scene_rejects_outside_or_degenerate_context(self):
        r=self.renderer()
        for w,h in ((0,3),(14,3),(7,9),(7,-1)):
            with self.assertRaises(ValueError):r.context_scene((12,10,14),AFFINE,w,h)

    def test_reused_surface_rejects_changed_png_even_for_same_capture(self):
        r=self.renderer()
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);binding={'frozen':'capture-sha'}
            outputs={}
            for name in r.SURFACE_FILES:
                path=folder/name;path.write_bytes(name.encode());outputs[name]=r.base.core.file_sha(path)
            info=dict(complete_mask=True,smoothing=False,scale_provenance=dict(mask_scaled_separately=False,ct_and_mask_share_one_affine=True))
            manifest=dict(case_id='amos_0167',capture_binding=binding,input_sha256={},renderer_sha256=r.BASE_SHA,
                complete_surfaces={m:info for m in r.base.core.METHODS},outputs=outputs)
            (folder/'fixed_three_face_manifest.json').write_text(json.dumps(manifest))
            r.validate_surface_reuse(folder,binding,{})
            (folder/r.SURFACE_FILES[0]).write_bytes(b'wrong-result-image')
            with self.assertRaisesRegex(ValueError,'PNG changed'):r.validate_surface_reuse(folder,binding,{})


if __name__=='__main__':unittest.main()
