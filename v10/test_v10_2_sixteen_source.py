import copy
import importlib.util
import sys
import tempfile
import json
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tools'))


def external():
    datasets = {s: {'train': [f'train{i}' for i in range(n)],
        'test': [f'test{i}' for i in range(20)], 'val': [], 'training_enabled': True}
        for s, n in [('graz', 20), ('pengwin', 320), ('adrenal', 33)]}
    datasets['fumpe'] = {'train': [], 'test': [f'f{i}' for i in range(30)], 'val': [], 'training_enabled': False}
    return {'datasets': datasets}


class SixteenSourceTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('v10_2_sixteen_source_protocol'),
                             'sixteen-source protocol missing')
        import v10_2_sixteen_source_protocol as module
        return module

    def test_exact_partitions(self):
        self.assertEqual(self.module().validate_external_split(external()),
                         {'graz': 20, 'pengwin': 320, 'adrenal': 33})

    def test_reject_test_leakage(self):
        value = external(); value['datasets']['graz']['test'][0] = 'train0'
        with self.assertRaisesRegex(ValueError, 'overlap'):
            self.module().validate_external_split(value)

    def test_fumpe_never_trains(self):
        value = external(); value['datasets']['fumpe']['train'] = ['f1']
        with self.assertRaisesRegex(ValueError, 'FUMPE'):
            self.module().validate_external_split(value)

    def test_quotas_retain_original_budget(self):
        counts = self.module().source_counts()
        self.assertEqual(sum(counts.values()), 250)
        self.assertEqual(len(counts), 16)
        self.assertEqual([counts[s] for s in ('graz','pengwin','adrenal')], [10,30,10])
        self.assertEqual(sum(list(counts.values())[:13]), 200)

    def test_signature_independent_of_reference_path_class_and_prompt(self):
        record = {'source_name':'pengwin', 'case_id':'case1', 'row_index':3,
            'source_signatures':[{'path':'data/a.parquet', 'size_bytes':99, 'mtime_ns':1}]}
        other = copy.deepcopy(record); other.update(reference='elsewhere', global_class_id=2004, prompt='box')
        module = self.module()
        self.assertEqual(module.native_signature(record, image=True), module.native_signature(other, image=True))
        other['source_signatures'][0]['mtime_ns'] = 2
        self.assertNotEqual(module.native_signature(record, image=True), module.native_signature(other, image=True))

    def test_ct_signature_does_not_depend_on_seg(self):
        record = {'source_name':'adrenal', 'case_id':'a', 'seg_path':'seg.dcm',
            'source_signatures':[{'path':'ct.dcm','size_bytes':1,'mtime_ns':1},
                                 {'path':'seg.dcm','size_bytes':1,'mtime_ns':1}]}
        other = copy.deepcopy(record); other['source_signatures'][1]['mtime_ns'] = 9
        module = self.module()
        self.assertEqual(module.native_signature(record, image=True), module.native_signature(other, image=True))
        self.assertNotEqual(module.native_signature(record, image=False), module.native_signature(other, image=False))

    def test_native_reference_retains_one_case_only(self):
        self.assertIsNotNone(importlib.util.find_spec('v10_2_native_reference'), 'native loader missing')
        from v10_2_native_reference import NativeReferenceLoader
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for case in ('a','b'):
                p = Path(tmp)/case; p.mkdir()
                for name in ('image.nii.gz','label.nii.gz'):
                    f = p/name; f.write_text(json.dumps({'native_training_reference_v1': True,
                        'source_name':'graz','case_id':case}), encoding='utf-8'); paths.append(f)
            calls = []
            def read(record):
                calls.append(record['case_id']); return record['case_id']+' CT',record['case_id']+' GT',{}
            loader = NativeReferenceLoader(tmp, native_loader=read)
            self.assertEqual(loader.load(paths[0]), 'a CT')
            self.assertEqual(loader.load(paths[1]), 'a GT')
            self.assertEqual(loader.load(paths[2]), 'b CT')
            self.assertEqual(calls, ['a','b'])
            self.assertEqual(loader.current, ('graz','b'))

    def test_training_args_do_not_resume_old_optimizer(self):
        self.assertIsNotNone(importlib.util.find_spec('v10_2_sixteen_source_train'), 'training entry missing')
        from v10_2_sixteen_source_train import training_args
        args = training_args({'resume_checkpoint':'old', 'lr_warmup_epochs':2},
            initialization='best.pth', output='new', gpu=2, split='new.json', smoke=False)
        self.assertEqual(args.resume_checkpoint, '')
        self.assertEqual(args.prompt_generator_checkpoint, 'best.pth')
        self.assertEqual(args.train_cases_per_epoch, 250)
        self.assertEqual(args.epochs, 40)
        self.assertEqual(args.validate_every, 10)
        self.assertGreater(args.sam_encoder_unfreeze_epoch, args.epochs)

    def test_native_audit_returns_metadata_not_volume(self):
        import prepare_v10_2_sixteen_source as preparation
        self.assertTrue(hasattr(preparation, 'audit_record'), 'native audit interface missing')
        audit_record = preparation.audit_record
        audit = {'grid_verified':True,'label_values':[0,1],'source_signatures':[]}
        self.assertEqual(audit_record({}, native_reader=lambda r: (object(),object(),audit)), audit)
        with self.assertRaisesRegex(ValueError, 'geometry'):
            audit_record({}, native_reader=lambda r: (object(),object(),{'grid_verified':False}))

    def test_best_gate_rejects_nan_and_wrong_epoch(self):
        import v10_2_sixteen_source_train as entry
        self.assertTrue(hasattr(entry,'assert_best_state'), 'finite audited best gate missing')
        for epoch,score in [(10,float('nan')),(9,.7881604375711181)]:
            with self.assertRaises(ValueError):
                entry.assert_best_state({'epoch':epoch,'validation_metrics':{'selection':{'score':score}}})

    def test_native_paths_portable_and_do_not_change_embedded_names(self):
        import prepare_v10_2_sixteen_source as preparation
        self.assertTrue(hasattr(preparation,'portable_record'), 'portable native records missing')
        root = Path.cwd()
        record = {'source_name':'pengwin', 'parquet_path':str(root/'data'/'a.parquet'),
            'image_name':'embedded/a_image.nii.gz','source_signatures':[{'path':str(root/'data'/'a.parquet'),'size_bytes':1,'mtime_ns':1}]}
        result = preparation.portable_record(record,root)
        self.assertEqual(result['parquet_path'],'data/a.parquet')
        self.assertEqual(result['source_signatures'][0]['path'],'data/a.parquet')
        self.assertEqual(result['image_name'],record['image_name'])

    def test_prefetch_and_main_native_reads_are_serialized(self):
        from v10_2_native_reference import NativeReferenceLoader
        first = threading.Event(); second = threading.Event(); release = threading.Event()
        with tempfile.TemporaryDirectory() as tmp:
            paths=[]
            for case in ('a','b'):
                p=Path(tmp)/case; p.mkdir(); f=p/'image.nii.gz'
                f.write_text(json.dumps({'native_training_reference_v1':True,'source_name':'graz','case_id':case}))
                paths.append(f)
            def reader(record):
                if record['case_id']=='a': first.set(); release.wait(2)
                else: second.set()
                return record['case_id'],'label',{}
            loader=NativeReferenceLoader(tmp,native_loader=reader)
            a=threading.Thread(target=loader.load,args=(paths[0],)); b=threading.Thread(target=loader.load,args=(paths[1],))
            a.start(); self.assertTrue(first.wait(1)); b.start()
            try: self.assertFalse(second.wait(.1),'concurrent decoding breaks bounded cache ownership')
            finally: release.set(); a.join(); b.join()


if __name__ == '__main__': unittest.main()
