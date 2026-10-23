"""Geometry and scope contracts for isolated PAM evaluation."""
import importlib
import unittest
import numpy as np


class PAMContracts(unittest.TestCase):
    def setUp(self):
        self.p = importlib.import_module('tools.pam_amos_three_view')

    def test_orientation_and_spacing_roundtrip(self):
        raw = np.arange(5*7*11).reshape(5,7,11)
        for plane, order in [('coronal',(2,0,1)),('sagittal',(1,0,2))]:
            stack, spacing = self.p.orient(raw, (2.,3.,4.), plane)
            np.testing.assert_array_equal(stack, raw.transpose(order))
            self.assertEqual(tuple(spacing), tuple(np.array((2.,3.,4.))[list(order)]))
            np.testing.assert_array_equal(self.p.restore(stack, plane), raw)

    def test_box_uses_only_selected_slice(self):
        gt=np.zeros((5,7,11),bool);gt[1:4,2:6,8]=1
        box=self.p.slice_box(gt,'coronal',8)
        self.assertEqual(box,(2,1,6,4))
        gt[0,:,0]=True
        self.assertEqual(self.p.slice_box(gt,'coronal',8),box)
        with self.assertRaises(ValueError):self.p.slice_box(gt,'coronal',7)

    def test_box_crop_and_paste(self):
        image=np.arange(63).reshape(7,9)
        crop, bounds=self.p.box_crop(image,(2,1,7,5))
        np.testing.assert_array_equal(crop,image[1:5,2:7])
        mask=self.p.paste(np.ones((4,5),bool),image.shape,bounds)
        self.assertEqual(mask.sum(),20)
        self.assertTrue(mask[1:5,2:7].all())

    def test_union_no_third_prediction(self):
        ct=np.zeros((5,7,11),np.float32)
        payload=({'plane':'coronal','slice':8,'box':(2,1,6,4)},
                 {'plane':'sagittal','slice':3,'box':(2,1,8,4)})
        calls=[]
        def native(stack,spacing,prompt):
            calls.append(prompt['plane'])
            out=np.zeros_like(stack,dtype=bool);out[prompt['slice'],1:3,2:4]=1
            return out,{'box_decoder_calls':1,'prop_decoder_calls':3}
        rows,audits=self.p.predict_three(ct,(2,3,4),payload,native)
        self.assertEqual(calls,['coronal','sagittal'])
        np.testing.assert_array_equal(rows['dual_or'],rows['coronal']|rows['sagittal'])
        self.assertEqual(len(audits),2)

    def test_reject_partial_or_train_scope(self):
        rows=[dict(source_name='amos',case_id='a',global_class_id=1)]
        split={'datasets':{'amos':{'val':[{'case_id':'a'}],'train':[]}}}
        self.p.validate_scope(rows,split,expected_tasks=1,expected_cases=1)
        with self.assertRaises(ValueError):self.p.validate_scope(rows,split)
        split['datasets']['amos']['train']=[{'case_id':'a'}]
        with self.assertRaises(ValueError):self.p.validate_scope(rows,split,1,1)

    def test_checkpoint_prefix_rejects_collision(self):
        self.assertEqual(self.p.strip_state({'module.x':1}),{'x':1})
        with self.assertRaises(ValueError):self.p.strip_state({'x':1,'module.x':2})

    def test_vote_uses_positive_union_and_initial_component(self):
        a=np.zeros((3,5,5),int);a[:,2,2]=1;a[:,0,0]=1
        guide=np.zeros_like(a,bool);guide[1,2,2]=1
        out=self.p.finish_votes(a,np.ones(3),guide)
        self.assertEqual(out.sum(),3)

    def test_normalize_before_crop_preserves_slice_scale(self):
        image=np.arange(100,dtype=np.float32).reshape(10,10)
        full=self.p.normalize_u8(image)
        self.assertEqual(int(full[5,5]),int(55/99*255))
        crop,_=self.p.box_crop(full,(4,4,7,7))
        self.assertGreater(int(crop.min()),0)

    def test_guiding_mask_threshold_is_majority_not_any_vote(self):
        votes=np.array([[0,1,2]],dtype=np.int16)
        np.testing.assert_array_equal(self.p.guide_binary(votes,2),[[False,False,True]])


if __name__=='__main__':unittest.main()
