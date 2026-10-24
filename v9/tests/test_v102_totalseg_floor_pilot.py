import importlib
import unittest
from collections import Counter
from types import SimpleNamespace
from pathlib import Path
from uuid import uuid4


class TotalSegFloorPilotTests(unittest.TestCase):
    def setUp(self):
        self.t = importlib.import_module('tools.v102_totalseg_floor_pilot')

    def fixture(self):
        classes = [i for i in range(1, 118) if i not in (40, 41)]
        rows = [dict(source_name='totalseg', case_id='s%04d' % (i//115),
                     global_class_id=classes[i % 115]) for i in range(4034)]
        split = dict(datasets=dict(totalseg=dict(
            val=[dict(case_id='s%04d' % i) for i in range(36)],
            train=[dict(case_id='training')])) )
        return rows, split

    def test_stratified202_reproducible_not_metric_selected(self):
        rows, split = self.fixture()
        selected = self.t.select_subset(rows, split)
        self.assertEqual(len(selected), 202)
        self.assertEqual(len(set(map(self.t.b.key, selected))), 202)
        self.assertEqual(len({self.t.b.key(r)[2] for r in selected}), 115)
        reverse = [dict(r, dice=-999) for r in reversed(rows)]
        self.assertEqual(list(map(self.t.b.key, selected)),
                         list(map(self.t.b.key, self.t.select_subset(reverse, split))))
        counts = Counter(self.t.b.key(r)[2] for r in selected)
        self.assertTrue(all(n in (1, 2) for n in counts.values()))
        for wrong in (rows[:-1], rows[:-1]+rows[:1]):
            with self.assertRaises(ValueError):
                self.t.select_subset(wrong, split)
        split['datasets']['totalseg']['train'].append(dict(case_id='s0000'))
        with self.assertRaises(ValueError):
            self.t.select_subset(rows, split)

    def test_quota_rare_classes_and_NA_rejection(self):
        counts = {1:1, 2:33, 3:100}
        quotas = self.t.class_quotas(counts, 14)
        self.assertEqual(sum(quotas.values()), 14)
        self.assertEqual(quotas[1], 1)
        self.assertTrue(all(1 <= quotas[k] <= n for k,n in counts.items()))
        rows, split = self.fixture(); rows[0]['global_class_id'] = 40
        with self.assertRaises(ValueError):
            self.t.select_subset(rows, split)

    def test_seven_grids_exact_partition_and_whole_case(self):
        rows, split = self.fixture(); selected = self.t.select_subset(rows, split)
        self.assertEqual(self.t.FLOORS, (.45,.50,.55,.60,.65,.70,.75))
        jobs = self.t.make_jobs(selected)
        self.assertEqual(len(jobs), 91)
        self.assertTrue(all(j['smoke'] for j in jobs[:7]))
        self.assertEqual(sum(j['smoke'] for j in jobs), 7)
        self.assertEqual(sum(j['task_count'] for j in jobs), 1414)
        for floor in self.t.FLOORS:
            part = [r for j in jobs if j['floor']==floor for r in j['tasks']]
            self.assertEqual(sorted(map(self.t.b.key, part)), sorted(map(self.t.b.key, selected)))
            owner = {}
            for j in jobs:
                if j['floor']!=floor or j['smoke']: continue
                for r in j['tasks']:
                    self.assertEqual(owner.setdefault(r['case_id'],j['name']), j['name'])

    def test_floor_only_native_fallback_and_parser_boundary(self):
        for floor in self.t.FLOORS:
            a = SimpleNamespace(dynamic_threshold_min=.35, dynamic_threshold_max=.75,
                dynamic_threshold_fallback=.60, prompt_logit_neighborhood_radius=2)
            before = vars(a).copy(); self.t.freeze_settings(a, floor)
            self.assertEqual(vars(a), dict(before,dynamic_threshold_min=floor))
        a.eval_memory_group_size=4; a.dynamic_threshold_min=.35
        with self.assertRaises(ValueError): self.t.freeze_settings(a,.60)
        self.assertTrue(self.t.g.threshold_valid(dict(threshold=.60,source='fallback_insufficient_separation'),.75))
        self.assertFalse(self.t.g.threshold_valid(dict(threshold=.60,source='prompt_logit_midpoint'),.75))

    def test_exact_prompt_audit_and_legacy_source(self):
        r=dict(source_name='totalseg',case_id='s0000',global_class_id=1)
        a=dict(dynamic_threshold=dict(minimum=.75,maximum=.75,fallback=.60),
            loaded_task_keys=[['totalseg','s0000',1]],sources=['totalseg'],background_mode='point',
            dynamic_threshold_rows=[dict(source_name='totalseg',case_id='s0000',class_id=1,
                threshold=.60,source='fallback_insufficient_separation')])
        self.t.validate_prompt_audit(a,[r],'totalseg',.75)
        a['dynamic_threshold_rows'][0]['case_id']='other'
        with self.assertRaises(ValueError): self.t.validate_prompt_audit(a,[r],'totalseg',.75)

    def test_finite_paired_summary_and_status(self):
        r=dict(source_name='totalseg',case_id='s0000',global_class_id=1)
        rows=[dict(r,**{m:.7 for m in self.t.b.METRICS})]
        old=[dict(r,**{m:.5 for m in self.t.b.METRICS})]
        sixty=[dict(r,**{m:.6 for m in self.t.b.METRICS})]
        s=self.t.summarize(rows,old,sixty)
        for m in self.t.b.METRICS:
            self.assertAlmostEqual(s['delta_from_baseline'][m],.2)
            self.assertAlmostEqual(s['delta_from_floor60'][m],.1)
        rows[0]['dice']=float('inf')
        with self.assertRaises(ValueError): self.t.summarize(rows,old,sixty)
        out=Path(__file__).resolve().parents[1]/'.cache'/('totalseg_pilot_test_'+uuid4().hex)
        self.t.b.write(out/'jobs.json',[])
        self.assertEqual(self.t.status(out)['expected_new'],1414)


if __name__ == '__main__': unittest.main()
