import importlib
import unittest
import numpy as np


class SoftFusionTests(unittest.TestCase):
    def setUp(self):
        self.p = importlib.import_module('tools.pam_soft_fusion')

    def test_per_pixel_coverage_and_overlap_mean(self):
        a = self.p.SoftVolume((1, 3, 4))
        a.add(0, (1, 1, 3, 3), np.full((2,2), .8), np.full((2,2), 2.))
        a.add(0, (2, 1, 4, 3), np.full((2,2), .4), np.full((2,2), -1.))
        prob, margin, valid = a.means()
        self.assertAlmostEqual(float(prob[0,1,2]), .6, places=6)
        self.assertAlmostEqual(float(margin[0,1,2]), .5)
        self.assertFalse(valid[0,0,0])
        self.assertEqual(float(prob[0,0,0]), 0.)

    def test_valid_view_average_not_background_fill(self):
        a=(np.array([.8,.8,0.,0.]),np.array([2.,2.,0.,0.]),np.array([1,1,0,0],bool))
        b=(np.array([.1,0.,.9,0.]),np.array([-3.,0.,3.,0.]),np.array([1,0,1,0],bool))
        prob, margin, valid=self.p.fuse(a,b)
        np.testing.assert_allclose(prob,[.45,.8,.9,0.])
        np.testing.assert_allclose(margin,[-.5,2.,3.,0.])
        np.testing.assert_array_equal(valid,[1,1,1,0])

    def test_probability_and_logit_mean_can_disagree(self):
        logits=np.array([2.,2.,-10.])
        a=(np.array([(1/(1+np.exp(-logits))).mean()]),np.array([logits.mean()]),np.array([True]))
        b=(np.array([1/(1+np.exp(-.1))]),np.array([.1]),np.array([True]))
        prob,margin,_=self.p.fuse(a,b)
        self.assertTrue(prob[0]>.5)
        self.assertFalse(margin[0]>0)

    def test_reject_nonfinite_or_invalid_probability(self):
        a=self.p.SoftVolume((1,2,2))
        for value in (float('nan'),float('inf'),1.1,-.1):
            with self.assertRaises(ValueError):a.add(0,(0,0,1,1),np.array([[value]]),np.array([[0.]]))

    def test_component_filter_uses_learned_seed_not_gt(self):
        pred=np.zeros((3,4,4),bool);pred[:,1,1]=1;pred[:,3,3]=1
        seed=np.zeros_like(pred);seed[1,1,1]=1
        out=self.p.seed_components(pred,seed)
        self.assertEqual(int(out.sum()),3)

    def test_scope_is_fixed_eight_cases_not_full_val(self):
        tasks=[{'case_id':f'c{i:02}','source_name':'amos','global_class_id':1} for i in range(10)]
        chosen=self.p.select_pilot(tasks)
        self.assertEqual([x['case_id'] for x in chosen],[f'c{i:02}' for i in range(8)])

    @unittest.skipUnless(importlib.util.find_spec('cv2'),'real OpenCV required; run on server isolated dependency')
    def test_soft_recording_preserves_native_argmax(self):
        import torch
        native=object.__new__(self.p.p.Models);native.torch=torch
        soft=object.__new__(self.p.SoftModels);soft.torch=torch
        outputs={'predictions':[torch.tensor([[[[0.,2.],[1.,-1.]],[[2.,0.],[1.,3.]]]]),
                                torch.tensor([[[[1.]],[[0.]]]])]}
        expected=native.output(outputs,(5,7))
        actual=soft.output(outputs,(5,7))
        np.testing.assert_array_equal(expected[0],actual[0][0])
        self.assertTrue(np.isfinite(actual[0][1]).all())
        self.assertTrue(((actual[0][1]>=0)&(actual[0][1]<=1)).all())

    def test_two_propagations_reused_for_all_six_soft_results(self):
        shape=(3,4,5);ct=np.zeros(shape);calls=[]
        payload=({'plane':'coronal','slice':2,'box':(0,0,4,3)},
                 {'plane':'sagittal','slice':1,'box':(0,0,5,3)})
        def models(stack,spacing,prompt):
            calls.append(prompt['plane']);seed=np.zeros(stack.shape,bool);seed[prompt['slice'],1,1]=1
            valid=np.ones(stack.shape,bool)
            return seed,(np.full(stack.shape,.6),np.ones(stack.shape),valid),seed,{}
        rows,_=self.p.predict(ct,(1,2,3),payload,models)
        self.assertEqual(calls,['coronal','sagittal'])
        self.assertEqual(set(rows),set(self.p.METHODS))
        for mask in rows.values():self.assertEqual(mask.shape,shape)

if __name__=='__main__':unittest.main()
