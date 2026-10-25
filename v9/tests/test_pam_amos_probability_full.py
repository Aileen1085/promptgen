import importlib
from pathlib import Path
import unittest
from unittest.mock import patch


class AmosProbabilityTests(unittest.TestCase):
    def setUp(self):
        self.m = importlib.import_module('tools.pam_amos_probability_full')

    def data(self):
        rows = []
        for i in range(100):
            for c in range(15 if i < 85 else 14):
                rows.append(dict(source_name='amos', case_id=f'amos_{i:04d}',
                                 global_class_id=c+100, local_class_id=c+1))
        split = {'datasets': {'amos': {'val': [{'case_id': f'amos_{i:04d}'} for i in range(100)], 'train': []}}}
        return rows, split

    def row(self):
        return dict(source_name='amos', case_id='amos_0000', global_class_id=100,
                    local_class_id=1, method='dual_prob',
                    **{k: .5 for k in self.m.n.METRICS},
                    prompt_audit={'submission_sha256': 'boxes_same'},
                    propagation_audit=[{'guide_slices': [1], 'box_decoder_calls': 1,
                                        'prop_decoder_calls': 2, 'guiding_pixels': 3}]*2)

    def test_exact_scope_and_rejections(self):
        rows, split = self.data()
        self.m.validate_scope(rows, split)
        with self.assertRaises(ValueError): self.m.validate_scope(rows[:-1], split)
        duplicate = rows[:-1]+[dict(rows[0])]
        with self.assertRaises(ValueError): self.m.validate_scope(duplicate, split)
        split['datasets']['amos']['train'] = [split['datasets']['amos']['val'][0]]
        with self.assertRaises(ValueError): self.m.validate_scope(rows, split)
        rows, split = self.data(); rows[0]['source_name'] = 'magic'
        with self.assertRaises(ValueError): self.m.validate_scope(rows, split)

    def test_whole_case_smoke_and_exact_once(self):
        rows, _ = self.data(); jobs = self.m.make_jobs(rows)
        self.assertEqual(len(jobs), 25)
        self.assertTrue(jobs[0]['smoke'])
        self.assertEqual(len(jobs[0]['tasks']), 15)
        keys = [self.m.n.key(r) for j in jobs for r in j['tasks']]
        self.assertEqual(len(keys), 1485)
        self.assertEqual(len(set(keys)), 1485)
        ownership = {}
        for j in jobs:
            for r in j['tasks']:
                self.assertEqual(ownership.setdefault(r['case_id'], j['name']), j['name'])

    def test_pilot_controls_metrics_and_payload(self):
        row = self.row(); self.m.validate_pilot_control([row], [row])
        other = dict(row, dice=.7)
        with self.assertRaises(ValueError): self.m.validate_pilot_control([other], [row])
        other = dict(row, prompt_audit={'submission_sha256': 'changed'})
        with self.assertRaises(ValueError): self.m.validate_pilot_control([other], [row])
        other = dict(row, propagation_audit=[dict(row['propagation_audit'][0], guiding_pixels=4)]*2)
        with self.assertRaises(ValueError): self.m.validate_pilot_control([other], [row])
        unrelated = dict(row, case_id='unrelated')
        self.m.validate_pilot_control([unrelated], [row])

    def test_completion_guard_before_write(self):
        row = self.row(); written = []
        write = self.m.guarded_writer(lambda path, value: written.append(Path(path).name), [row])
        write(Path('all_metrics_live.json'), {'per_case_class': [dict(row, dice=.7)]})
        self.assertEqual(written, ['all_metrics_live.json'])
        with self.assertRaises(ValueError): write(Path('complete_metrics.json'), {'per_case_class': [dict(row, dice=.7)]})
        self.assertEqual(written, ['all_metrics_live.json'])
        write(Path('complete_metrics.json'), {'per_case_class': [row]})
        self.assertEqual(written[-1], 'complete_metrics.json')

    def test_configure_native_bindings_and_no_source_changes(self):
        from tools import pam_nine_source_probability as q
        files = [Path(q.__file__), self.m.ROOT/'tools/pam_soft_fusion.py', self.m.ROOT/'tools/pam_amos_three_view.py']
        before = [p.read_bytes() for p in files]
        names = ['COUNTS','PROTOCOL','validate_scope','assets_signature']
        old = {k:getattr(q,k) for k in names}
        oldp = {k:getattr(q.p,k) for k in ['assets_signature','progress','merge','__file__']}
        try:
            self.m.configure()
            self.assertEqual(q.COUNTS, {'amos':1485})
            self.assertIs(q.p.merge, self.m.merge)
            self.assertEqual(q.p.__file__, self.m.__file__)
            self.assertIs(q.validate_scope, self.m.validate_scope)
            self.assertEqual(q.PROTOCOL['methods'], ['dual_prob'])
            self.assertEqual(q.PROTOCOL['thresholds'], {'probability':.5})
            self.assertEqual(q.PROTOCOL['fusion'], 'per_voxel_query_mean_then_equal_valid_view_mean')
        finally:
            for k,v in old.items():setattr(q,k,v)
            for k,v in oldp.items():setattr(q.p,k,v)
        self.assertEqual(before, [p.read_bytes() for p in files])


if __name__ == '__main__': unittest.main()
