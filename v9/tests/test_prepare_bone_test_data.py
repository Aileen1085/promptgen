import importlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from contextlib import contextmanager
import numpy as np
from PIL import Image


MAPPING = '0 | background | (0,0,0)\n1 | bone_a | (10,20,30)\n2 | bone_b | (40,50,60)\n'


@contextmanager
def fixture_directory():
    base=Path(__file__).resolve().parents[1]/'output/bone_prepare_test_fixtures'
    base.mkdir(parents=True,exist_ok=True)
    # Retain tiny fixtures: project forbids bulk/recursive deletion.
    yield tempfile.mkdtemp(prefix='fixture_',dir=base)


def png(array):
    b = io.BytesIO(); Image.fromarray(array).save(b, format='PNG'); return b.getvalue()


class BonePreparationTests(unittest.TestCase):
    def setUp(self): self.m = importlib.import_module('tools.prepare_bone_test_data')

    def archive(self, path, duplicate=True, conflict=False):
        canonical = np.zeros((3,4,2), np.uint8)
        canonical[0,1,0] = 1; canonical[2,3,1] = 1
        meta = dict(case='0001', phase='phase_3', canonical_shape=[3,4,2], slice_count=2,
                    canonical_axcodes=['R','A','S'], slice_axis=2,
                    png_orientation='canonical RAS axial; transpose; vertical flip; posterior/vertebra at bottom',
                    source_sha256='same', label_counts={'0':22,'1':2}, ids=[0,1],
                    label_id_dir='label_id_png', color_dir='color_png')
        palette = np.array([[0,0,0],[10,20,30],[40,50,60]], np.uint8)
        with zipfile.ZipFile(path,'w') as z:
            z.writestr('bone/mapping.txt', MAPPING)
            for folder in (['bone/0001','bone'] if duplicate else ['bone/0001']):
                z.writestr(folder+'/index.json',json.dumps(meta))
                for k in range(2):
                    ids=np.flipud(canonical[:,:,k].T)
                    if conflict and folder=='bone': ids=ids.copy();ids[0,0]=2
                    z.writestr(folder+f'/label_id_png/{k:04d}.png',png(ids))
                    z.writestr(folder+f'/color_png/{k:04d}.png',png(palette[ids]))
        return canonical

    def test_mapping_preserves_ids_names_and_rejects_duplicates(self):
        value=self.m.parse_mapping(MAPPING)
        self.assertEqual(value[1]['name'],'bone_a');self.assertEqual(value[2]['rgb'],[40,50,60])
        with self.assertRaises(ValueError): self.m.parse_mapping(MAPPING+'1 | other | (1,2,3)\n')
        with self.assertRaises(ValueError): self.m.parse_mapping(MAPPING+'3 | other | (10,20,30)\n')

    def test_paths_and_symlinks_rejected(self):
        for name in ['../escape','/abs','bone/../../escape','C:/escape','bone/../escape','bone\\..\\escape']:
            with self.assertRaises(ValueError): self.m.safe_relative(name)
        with fixture_directory() as temp:
            p=Path(temp);z=p/'bad.zip'
            with zipfile.ZipFile(z,'w') as out:
                info=zipfile.ZipInfo('bone/link');info.create_system=3;info.external_attr=(0o120777<<16)
                out.writestr(info,'/outside')
            with self.assertRaises(ValueError):self.m.extract_archive(z,p/'out')

    def test_inverse_transpose_flip_is_exact_not_double_flip(self):
        xy=np.arange(15,dtype=np.uint8).reshape(3,5)
        np.testing.assert_array_equal(self.m.restore_slice(np.flipud(xy.T)),xy)

    def test_full_prepare_actual_presence_and_missing_ct(self):
        with fixture_directory() as temp:
            p=Path(temp);expected=self.archive(p/'bone.zip')
            result=self.m.prepare(p/'bone.zip',p/'bone')
            self.assertFalse(result['test_ready']);self.assertEqual(len(result['cases']),1)
            self.assertEqual(result['present_foreground_ids'],[1]);self.assertEqual(result['absent_foreground_ids'],[2])
            self.assertEqual(len(result['duplicate_aliases']),1)
            c=result['cases'][0];self.assertIsNone(c['image']);self.assertIsNone(c['affine_ras_xyz'])
            with np.load(p/'bone'/c['label_npz'],allow_pickle=False) as f:
                np.testing.assert_array_equal(f['labels_dhw'],expected.transpose(2,0,1))
            self.assertEqual(result['classes'][1]['voxel_count'],2)
            self.assertEqual(result['classes'][1]['case_count'],1)
            before=(p/'bone/0001/index.json').read_bytes()
            self.m.extract_archive(p/'bone.zip',p/'bone')
            self.assertEqual(before,(p/'bone/0001/index.json').read_bytes())
            with self.assertRaises(ValueError):self.m.prepare(p/'bone.zip',p/'bone')

    def test_conflicting_same_case_not_silently_deduplicated(self):
        with fixture_directory() as temp:
            p=Path(temp);self.archive(p/'bone.zip',conflict=True)
            with self.assertRaises(ValueError):self.m.prepare(p/'bone.zip',p/'bone')

    def test_rgb_mismatch_rejected(self):
        with fixture_directory() as temp:
            p=Path(temp);self.archive(p/'bone.zip',duplicate=False)
            self.m.extract_archive(p/'bone.zip',p/'bone')
            image=p/'bone/0001/color_png/0000.png'
            v=np.array(Image.open(image));v[0,0]=[1,2,3];Image.fromarray(v).save(image)
            meta=json.loads((p/'bone/0001/index.json').read_text())
            with self.assertRaises(ValueError):self.m.scan_case(p/'bone/0001',meta,self.m.parse_mapping(MAPPING))

    def test_duplicate_zip_members_and_existing_changed_bytes_rejected(self):
        with fixture_directory() as temp:
            p=Path(temp);self.archive(p/'bone.zip',duplicate=False)
            self.m.extract_archive(p/'bone.zip',p/'bone')
            (p/'bone/mapping.txt').write_text('changed',encoding='utf-8')
            with self.assertRaises(ValueError):self.m.extract_archive(p/'bone.zip',p/'bone')
            with zipfile.ZipFile(p/'duplicate.zip','w') as z:
                z.writestr('bone/a','a');z.writestr('bone//a','b')
            with self.assertRaises(ValueError):self.m.extract_archive(p/'duplicate.zip',p/'out')

    def test_index_count_mismatch_fails_not_metadata_only_inventory(self):
        with fixture_directory() as temp:
            p=Path(temp);self.archive(p/'bone.zip',duplicate=False)
            self.m.extract_archive(p/'bone.zip',p/'bone')
            meta=json.loads((p/'bone/0001/index.json').read_text());meta['label_counts']['1']=99
            with self.assertRaises(ValueError):self.m.scan_case(p/'bone/0001',meta,self.m.parse_mapping(MAPPING))


if __name__=='__main__':unittest.main()
