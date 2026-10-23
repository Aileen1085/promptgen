import importlib
import unittest
from unittest.mock import patch
import numpy as np


class NineSourceTests(unittest.TestCase):
    def setUp(self):self.m=importlib.import_module('tools.pam_nine_source_probability')

    def data(self):
        rows=[];split={'datasets':{}}
        for source,count in self.m.COUNTS.items():
            rows.extend(dict(source_name=source,case_id=source+'1',global_class_id=i+100,local_class_id=i+1) for i in range(count))
            split['datasets'][source]={'val':[{'case_id':source+'1'}],'train':[]}
        return rows,split

    def test_full_scope_counts_and_excluded_sources(self):
        rows,split=self.data();self.m.validate_scope(rows,split)
        self.assertEqual(len(rows),4548)
        self.assertFalse(set(self.m.COUNTS)&{'amos','lndb','msd_task06','covid19_20'})
        with self.assertRaises(ValueError):self.m.validate_scope(rows[:-1],split)

    def test_train_overlap_and_duplicate_rejected(self):
        rows,split=self.data();split['datasets']['magic']['train']=[{'case_id':'magic1'}]
        with self.assertRaises(ValueError):self.m.validate_scope(rows,split)
        rows,split=self.data();rows[-1]=dict(rows[-2])
        with self.assertRaises(ValueError):self.m.validate_scope(rows,split)

    def test_shards_keep_all_keys_and_smoke_all_sources(self):
        rows,_=self.data();jobs=self.m.make_jobs(rows)
        self.assertTrue(jobs[0]['smoke'])
        self.assertEqual({r['source_name'] for r in jobs[0]['tasks']},set(self.m.COUNTS))
        keys=[self.m.n.key(t) for j in jobs for t in j['tasks']]
        self.assertEqual(len(keys),len(set(keys)))
        self.assertEqual(set(keys),{self.m.n.key(t) for t in rows})
        for j in jobs[1:]:self.assertEqual(len({t['source_name'] for t in j['tasks']}),1)

    def test_probability_fusion_matches_pilot(self):
        q1=np.array([.8,.3,.8,0.]);q2=np.array([.1,.9,0.,0.])
        v1=np.array([1,1,1,0],bool);v2=np.array([1,1,0,0],bool)
        actual=self.m.probability_mean((q1,v1),(q2,v2))
        reference=self.m.soft.fuse((q1,q1,v1),(q2,q2,v2))
        np.testing.assert_allclose(actual[0],reference[0]);np.testing.assert_array_equal(actual[1],reference[2])

    def test_two_plane_calls_no_additional_forward(self):
        shape=(3,4,5);calls=[]
        def model(stack,spacing,prompt):
            calls.append(prompt['plane']);seed=np.zeros(stack.shape,bool);seed[prompt['slice'],1,1]=True
            return seed,(np.full(stack.shape,.6),np.zeros(stack.shape),np.ones(stack.shape,bool)),seed,{}
        payload=({'plane':'coronal','slice':2,'box':(0,0,3,3)}, {'plane':'sagittal','slice':1,'box':(0,0,3,3)})
        pred,audit=self.m.predict(np.zeros(shape),(2,3,4),payload,model)
        self.assertEqual(calls,['coronal','sagittal']);self.assertEqual(pred.shape,shape);self.assertEqual(len(audit),2)

    def test_loader_anisotropic_dhw_and_readonly_cache_hit_or_miss(self):
        from pathlib import Path
        image=np.arange(5*6*4,dtype=np.float32).reshape(5,6,4)
        label=np.zeros(image.shape,np.uint8);label[1:4,2:5,1:3]=2
        shape=(4,5,6);coro=np.zeros(shape,bool);sag=coro.copy()
        coro[1:3,1:4,3]=True;sag[1:3,2,2:5]=True
        package=dict(shape_dhw=np.array(shape),case_image_name=np.array('ct.npy'),
                     coronal_bits=np.packbits(coro.reshape(-1),bitorder='little'),
                     sagittal_bits=np.packbits(sag.reshape(-1),bitorder='little'))
        class Package:
            def __enter__(self):return package
            def __exit__(self,*a):return False
        class Image:
            def __init__(self,values):self.dataobj=values;self.shape=values.shape
            def get_qform(self):return np.diag([2.,3.,4.,1.])
        affine=np.eye(4);affine[:3,:3]=np.diag([2.,3.,4.])[:,[2,0,1]]
        task=dict(source_name='parse2022',case_id='one',prompt_cache_name='p.npz',local_class_id=2)
        split={'datasets':{'parse2022':{'val':[dict(case_id='one',image='ct.nii',label='gt.nii')]}}}
        for hit in (False,True):
            loader=self.m.Loader(split)
            loader.base.canonical=lambda path:(Image(image if str(path)=='ct.nii' else label),affine)
            def load(path,**kwargs):return image.transpose(2,0,1) if str(path).endswith('.npy') else Package()
            with self.subTest(hit=hit),patch.object(Path,'is_file',return_value=hit),patch.object(self.m.n,'sha',return_value='unchanged'),patch.object(np,'load',side_effect=load),patch.object(np,'save',side_effect=AssertionError('CT write')),patch.object(np,'savez_compressed',side_effect=AssertionError('cache write')):
                ct,gt,spacing,payload,audit=loader(task)
            np.testing.assert_array_equal(ct,image.transpose(2,0,1))
            np.testing.assert_array_equal(gt,label.transpose(2,0,1)==2)
            np.testing.assert_allclose(spacing,[4.,2.,3.])
            self.assertEqual(payload[0]['box'],(1,1,4,3));self.assertEqual(payload[1]['box'],(2,1,5,3))
            self.assertEqual(audit['magic_mapping_applications'],0);self.assertEqual(audit['case_cache_hit'],hit)

if __name__=='__main__':unittest.main()
