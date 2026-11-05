"""CPU-only observer tests. Synthetic fixtures retained; no recursive cleanup."""
import importlib.util
from pathlib import Path
import tempfile
import types
import unittest

import numpy as np

FILE = Path(__file__).absolute().parents[1]/'tools/capture_qualitative_promptgen_case.py'


class CaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = None
        if FILE.exists():
            spec = importlib.util.spec_from_file_location('pgcapture', str(FILE))
            cls.m = importlib.util.module_from_spec(spec); spec.loader.exec_module(cls.m)

    def module(self):
        self.assertIsNotNone(self.m, 'capture module required')
        return self.m

    def test_observer_returns_unchanged_real_scorer_result_and_copies_only_mask(self):
        m = self.module(); mask = np.zeros((3,4,5), np.float32); mask[1,1,1] = 1
        metrics = {name: .5 for name in m.METRICS}
        result = (metrics, mask); original = lambda *a, **k: result
        ns = {'restore_and_score_full_volume': original}
        with m.observe_final_mask(ns, (3,4,5)) as captured:
            actual = ns['restore_and_score_full_volume'](None, raw_target=np.ones(mask.shape), frame_indices=[0,1,2])
            self.assertIs(actual, result)
        self.assertIs(ns['restore_and_score_full_volume'], original)
        mask[:] = 1
        self.assertEqual(int(captured['prediction_dhw'].sum()), 1)
        self.assertEqual(set(captured), {'prediction_dhw','metrics','frame_indices'})

    def test_hooks_restore_on_native_exception(self):
        m = self.module()
        def score(*a, **k): raise RuntimeError('original failure')
        obj = types.SimpleNamespace(restore_and_score_full_volume=score)
        with self.assertRaisesRegex(RuntimeError, 'original failure'):
            with m.observe_final_mask(obj, (2,3,4)):
                obj.restore_and_score_full_volume(None)
        self.assertIs(obj.restore_and_score_full_volume, score)

    def test_rejects_nonbinary_wrongshape_and_incomplete_frame_indices(self):
        m = self.module()
        for mask, frames in [(np.zeros((2,3,4)), [0,2]),
                             (np.full((2,3,4), .5), [0,1]),
                             (np.zeros((1,3,4)), [0])]:
            ns = {'restore_and_score_full_volume':lambda *a,**k:(dict(dice=.5),mask)}
            with self.subTest(shape=mask.shape, frames=frames):
                with self.assertRaises(ValueError):
                    with m.observe_final_mask(ns, (2,3,4)):
                        ns['restore_and_score_full_volume'](None, frame_indices=frames)

    def test_rejects_multiple_scorer_calls_and_absent_capture(self):
        m = self.module(); ns={'restore_and_score_full_volume':lambda *a,**k:({}, np.zeros((2,3,4)))}
        with self.assertRaises(ValueError):
            with m.observe_final_mask(ns, (2,3,4)):
                pass
        with self.assertRaises(ValueError):
            with m.observe_final_mask(ns, (2,3,4)):
                ns['restore_and_score_full_volume'](None, frame_indices=[0,1])
                ns['restore_and_score_full_volume'](None, frame_indices=[0,1])

    def test_all_ten_metrics_compared_no_nan_or_missing(self):
        m = self.module(); row={name:.7 for name in m.METRICS}
        self.assertEqual(set(m.compare_metrics(row,row)), set(m.METRICS))
        for name in m.METRICS:
            bad=dict(row); bad[name]=.8
            with self.assertRaises(ValueError):m.compare_metrics(bad,row)
        bad=dict(row);bad['dice']=float('nan')
        with self.assertRaises(ValueError):m.compare_metrics(bad,row)
        del bad['dice']
        with self.assertRaises(ValueError):m.compare_metrics(bad,row)

    def test_five_key_authorization_and_explicit_runner_binding(self):
        m=self.module(); entries=[]
        for source,runner in [('amos','amos_s21'),('magic','magic_s16'),('kits23','s1_e490'),
                              ('msd_task07','s1_e490'),('msd_task10','msd10_s7')]:
            entries.append(dict(task=dict(source_name=source,case_id='case',global_class_id=123),runner=runner))
        manifest=dict(schema='five_source_promptgen_capture_v1',export_authorized=True,entries=entries)
        self.assertEqual(m.authorized_entry(manifest,'magic')['runner'],'magic_s16')
        manifest['entries'][1]['runner']='s1_e490'
        with self.assertRaises(ValueError):m.authorized_entry(manifest,'magic')
        manifest['entries'][1]['runner']='magic_s16';manifest['entries'].append(entries[0])
        with self.assertRaises(ValueError):m.authorized_entry(manifest,'amos')

    def test_key_match_never_substitutes_local_class_or_case(self):
        m=self.module();task=dict(source_name='kits23',case_id='case',global_class_id=145,local_class_id=1)
        m.require_key(task,dict(task))
        for bad in [dict(task,global_class_id=1),dict(task,case_id='CASE'),dict(task,source_name='amos')]:
            with self.assertRaises(ValueError):m.require_key(bad,task)
        native=dict(source_name='kits23',case_id='case',class_id=145)
        m.require_key(native,task)
        with self.assertRaises(ValueError):m.require_key(dict(native,global_class_id=1),task)

    def test_resource_snapshot_enforces_uuid_free_and_amos_native_gpu_guard(self):
        m=self.module(); snapshot='3, GPU-abc, 55000\n2, GPU-def, 10000\n'
        self.assertEqual(m.resource_snapshot(snapshot,3,'GPU-abc','magic_s16')['free_mib'],55000)
        for gpu,uuid,runner in [(2,'GPU-def','s1_e490'),(3,'GPU-other','s1_e490'),(3,'GPU-abc','amos_s21')]:
            with self.assertRaises(ValueError):m.resource_snapshot(snapshot,gpu,uuid,runner)

    def test_checkpoint_path_epoch_and_inventory_binding(self):
        m=self.module();entry=dict(checkpoint=dict(path='weights/test.pth',epoch=490,sha256='a'*64),
                                  input_sha256={'weights/test.pth':'a'*64})
        m.check_checkpoint(entry,str(m.ROOT/'weights/test.pth'),490)
        for path,epoch in [('weights/other.pth',490),('weights/test.pth',15)]:
            with self.assertRaises(ValueError):m.check_checkpoint(entry,path,epoch)
        entry['input_sha256']['weights/test.pth']='b'*64
        with self.assertRaises(ValueError):m.check_checkpoint(entry,'weights/test.pth',490)

    def test_signed_parent_relative_inputs_are_readonly_and_hash_checked(self):
        m=self.module();output=FILE.parents[1]/'output';output.mkdir(exist_ok=True)
        fixture=Path(tempfile.mkdtemp(prefix='pg_parent_inventory_',dir=str(output)))
        project=fixture/'project';project.mkdir()
        sibling=fixture/'shared_data';sibling.mkdir()
        source=sibling/'case_signature.txt';source.write_bytes(b'CT reference fixture, not voxels')
        digest=m.sha(source);saved_root=m.ROOT
        try:
            m.ROOT=project
            relative='../shared_data/case_signature.txt'
            self.assertEqual(m.verify_inputs({relative:digest}),{relative:digest})
            self.assertEqual(source.read_bytes(),b'CT reference fixture, not voxels')
            with self.assertRaisesRegex(ValueError,'SHA mismatch'):
                m.verify_inputs({relative:'0'*64})
            with self.assertRaisesRegex(ValueError,'relative'):
                m.verify_inputs({str(source.absolute()):digest})
        finally:m.ROOT=saved_root

    def test_export_only_mask_no_ct_and_no_overwrite(self):
        m=self.module();root=FILE.parents[1]/'output';root.mkdir(exist_ok=True)
        out=Path(tempfile.mkdtemp(prefix='pg_observer_test_',dir=str(root)))
        pred=np.zeros((2,3,4),np.uint8)
        m.save_mask(out, pred, dict(method='PromptGen'))
        with np.load(str(out/'prediction.npz')) as payload:
            self.assertEqual(payload.files,['prediction_dhw'])
            self.assertTrue(np.array_equal(payload['prediction_dhw'],pred))
        with self.assertRaises(FileExistsError):m.save_mask(out,pred,{})


if __name__=='__main__':unittest.main()
