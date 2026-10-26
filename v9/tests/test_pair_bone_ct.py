import importlib
import json
from pathlib import Path
import unittest
import zipfile
import nibabel as nib
import numpy as np
from test_prepare_bone_test_data import fixture_directory


class PairBoneTests(unittest.TestCase):
    def setUp(self):self.m=importlib.import_module('tools.pair_bone_ct')

    def test_ct_zip_rejects_traversal_and_unexpected_members(self):
        with fixture_directory() as t:
            p=Path(t)
            for index,name in enumerate(['ct/../x','/ct/0001.nii.gz','ct/password.txt']):
                z=p/f'{index}.zip'
                with zipfile.ZipFile(z,'w') as f:f.writestr(name,b'bad')
                with self.assertRaises(ValueError):self.m.extract_ct(z,p/f'out{index}')

    def test_ct_extract_exclusive_hash_checked_resume(self):
        with fixture_directory() as t:
            p=Path(t);z=p/'ct.zip'
            with zipfile.ZipFile(z,'w') as f:f.writestr('ct/0001.nii.gz',b'example')
            first=self.m.extract_ct(z,p/'raw');self.m.extract_ct(z,p/'raw')
            self.assertEqual(first['files']['0001.nii.gz']['size'],7)
            (p/'raw/0001.nii.gz').write_bytes(b'changed')
            with self.assertRaises(ValueError):self.m.extract_ct(z,p/'raw')

    def test_native_lps_label_exact_canonical_roundtrip(self):
        xyz=np.arange(24,dtype=np.uint8).reshape(2,3,4)
        affine=np.diag([-0.7,-0.8,1.2,1]);affine[:3,3]=[20,30,-10]
        image=nib.Nifti1Image(np.zeros_like(xyz,dtype=np.int16),affine)
        canonical=nib.as_closest_canonical(image)
        labels=np.zeros(canonical.shape,np.uint8);labels[0,1,2]=3;labels[1,2,3]=5
        native=self.m.native_labels(labels.transpose(2,0,1),image)
        restored=nib.as_closest_canonical(nib.Nifti1Image(native,image.affine))
        np.testing.assert_array_equal(np.asanyarray(restored.dataobj),labels)
        self.assertEqual(native.shape,image.shape)

    def test_shape_and_degenerate_geometry_rejected(self):
        image=nib.Nifti1Image(np.zeros((2,3,4),np.int16),np.eye(4))
        with self.assertRaises(ValueError):self.m.native_labels(np.zeros((4,2,4),np.uint8),image)
        with self.assertRaises(ValueError):self.m.geometry(np.zeros((4,4)))

    def test_pair_missing_ct_explicit_and_saved_label_geometry(self):
        with fixture_directory() as t:
            p=Path(t);root=p/'bone';(root/'test_ready_v1/labels').mkdir(parents=True)
            ct=p/'ct';ct.mkdir();affine=np.diag([-1.,-2.,3.,1.])
            nib.save(nib.Nifti1Image(np.full((2,3,4),400,np.int16),affine),ct/'0001.nii.gz')
            rows=[]
            for case in ['0001','0002']:
                labels=np.ones((4,2,3),np.uint8)
                np.savez_compressed(root/f'test_ready_v1/labels/{case}.npz',labels_dhw=labels)
                rows.append(dict(case_id=case+'_phase_3',label_npz=f'test_ready_v1/labels/{case}.npz',present_foreground_ids=[1],label_counts={'1':24}))
            manifest=dict(cases=rows,classes=[dict(label_id=1,name='bone_a')],archive_sha256='labels',mapping_sha256='map')
            (root/'test_ready_v1/manifest.json').write_text(json.dumps(manifest))
            result=self.m.pair(root,ct,root/'paired_v1')
            self.assertEqual(result['ready_case_count'],1);self.assertFalse(result['test_ready'])
            self.assertEqual(result['missing_ct_cases'],['0002_phase_3'])
            a=result['cases'][0];saved=nib.load(root/a['label_nifti'])
            np.testing.assert_allclose(saved.affine,affine)
            self.assertEqual(saved.shape,(2,3,4));self.assertEqual(saved.get_data_dtype(),np.dtype('uint8'))
            self.assertTrue(a['test_ready']);self.assertEqual(result['ready_task_count'],1)
            with self.assertRaises(ValueError):self.m.pair(root,ct,root/'paired_v1')


if __name__=='__main__':unittest.main()
