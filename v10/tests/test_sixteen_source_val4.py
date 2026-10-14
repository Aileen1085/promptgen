import copy
import importlib.util
import sys
import tempfile
import json
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v10_2_sixteen_source_protocol import SOURCE_ORDER


class Val4Tests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('v10_2_sixteen_source_validation'))
        import v10_2_sixteen_source_validation as module
        return module

    def fixture(self):
        split = dict(version=1, source_order=list(SOURCE_ORDER), datasets={}, class_catalog=[])
        heldout = {}
        for index, source in enumerate(SOURCE_ORDER):
            cid = index + 1
            split['class_catalog'].extend([
                dict(dataset=source, global_class_id=cid, class_name=source, not_applicable=False),
                dict(dataset=source, global_class_id=cid+100, class_name='absent', not_applicable=False)])
            rows = [dict(case_id=k, dataset=source, image=k+'.nii', label=k+'_gt.nii', classes=[cid])
                    for k in ['z','b','a','d','c','e']]
            split['datasets'][source] = dict(train=[dict(case_id='train')], val=rows if index<13 else [])
            heldout[source] = rows
        return split, heldout

    def test_order_all_tasks_na_and_immutable_training(self):
        split, heldout = self.fixture(); original=copy.deepcopy(split)
        view, manifest = self.module().build_validation_view(split, heldout)
        self.assertEqual(split, original)
        self.assertEqual(manifest['case_count'], 64)
        self.assertEqual(manifest['expected_tasks'], 64)
        for source in SOURCE_ORDER:
            self.assertEqual(manifest['case_ids'][source], ['z','b','a','d'])
            key='remaining_val' if source in SOURCE_ORDER[:13] else 'test'
            self.assertEqual([r['case_id'] for r in view['datasets'][source][key]], ['c','e'])
            if source in SOURCE_ORDER[:13]: self.assertNotIn('test',view['datasets'][source])
        self.assertEqual(len(manifest['covered_class_ids']),16)
        self.assertTrue(all(r['not_applicable'] for r in view['class_catalog'] if r['class_name']=='absent'))

    def test_reject_overlap_duplicates_short_and_unknown_classes(self):
        for mutation in ('overlap','duplicate','short','unknown'):
            split, heldout=self.fixture(); rows=heldout[SOURCE_ORDER[0]]
            if mutation=='overlap': rows[0]['case_id']='train'
            elif mutation=='duplicate': rows[1]['case_id']=rows[0]['case_id']
            elif mutation=='short': del rows[3:]
            else: rows[0]['classes']=[9999]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.module().build_validation_view(split,heldout)

    def test_metrics_guard_uses_new_counts_not_legacy267(self):
        view, val=self.module().build_validation_view(*self.fixture())
        metrics=dict(primary_threshold=.6, per_dataset={s:{} for s in SOURCE_ORDER},
            per_class={'dice':{cid:.8 for cid in val['covered_class_ids']}},
            case_class_macro={'count':64},validation_protocol=dict(selected_tasks=64,
            metric_space='raw_nifti_full_volume',anchor_coarse_crop=False,covered_classes=16))
        self.module().assert_validation_metrics(metrics,val)
        for field, value in [('selected_tasks',63),('metric_space','roi'),('anchor_coarse_crop',True)]:
            bad=copy.deepcopy(metrics); bad['validation_protocol'][field]=value
            with self.assertRaises(ValueError): self.module().assert_validation_metrics(bad,val)

    def test_entries_default_to_new_protocol_with_explicit_legacy_option(self):
        for name in ('v10_2_sixteen_source_train.py','v10_2_sixteen_source_full_decoder.py'):
            text=(Path(__file__).resolve().parents[1]/name).read_text()
            self.assertIn('--legacy-retention-validation',text)
            self.assertIn('assert_validation_metrics',text)

    def test_reject_reordered_remaining_test_even_with_updated_file_signature(self):
        module=self.module(); train,heldout=self.fixture()
        view,val=module.build_validation_view(train,heldout)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); external=root/'external.json'; t=root/'train.json'
            s=root/'view.json'; v=root/'tasks.json'
            external.write_text(json.dumps({'datasets':{src:{'test':[r['case_id'] for r in heldout[src]]}
                for src in SOURCE_ORDER[13:]}}))
            train['external_split']=str(external)
            t.write_text(json.dumps(train))
            s.write_text(json.dumps(view))
            val.update(training_split_sha256=module.sha256_file(t),split_file_sha256=module.sha256_file(s))
            v.write_text(json.dumps(val))
            module.load_validation_view(t,s,v)
            view['datasets'][SOURCE_ORDER[-1]]['test'].reverse()
            s.write_text(json.dumps(view)); val['split_file_sha256']=module.sha256_file(s)
            v.write_text(json.dumps(val))
            with self.assertRaisesRegex(ValueError,'test order'):
                module.load_validation_view(t,s,v)


if __name__=='__main__': unittest.main()
