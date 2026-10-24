import importlib
import unittest
from types import SimpleNamespace
from pathlib import Path
from uuid import uuid4


class TopCoWFloorTests(unittest.TestCase):
    def setUp(self):
        self.t = importlib.import_module('tools.v102_topcow_floor_grid')

    def fixture(self):
        tasks = [dict(source_name='topcow2024_cta', case_id='c%03d' % (i % 25),
                      global_class_id=200+i//25) for i in range(277)]
        split = dict(datasets=dict(topcow2024_cta=dict(
            val=[dict(case_id='c%03d' % i) for i in range(25)],
            train=[dict(case_id='train')])))
        return tasks, split

    def test_exact_heldout_scope(self):
        tasks, split = self.fixture()
        self.assertEqual(len(self.t.select_tasks(tasks, split)), 277)
        for wrong in (tasks[:-1], tasks[:-1]+tasks[:1]):
            with self.assertRaises(ValueError):
                self.t.select_tasks(wrong, split)
        split['datasets']['topcow2024_cta']['train'].append(dict(case_id='c000'))
        with self.assertRaises(ValueError):
            self.t.select_tasks(tasks, split)

    def test_four_floors_exact_partition_and_smoke_gate(self):
        tasks, _ = self.fixture()
        self.assertEqual(self.t.FLOORS, (.55, .60, .65, .70))
        jobs = self.t.make_jobs(tasks)
        self.assertEqual(len(jobs), 36)
        self.assertTrue(all(j['smoke'] for j in jobs[:4]))
        self.assertEqual(sum(j['smoke'] for j in jobs), 4)
        self.assertEqual(sum(j['task_count'] for j in jobs), 1108)
        for floor in self.t.FLOORS:
            chosen = [r for j in jobs if j['floor'] == floor for r in j['tasks']]
            self.assertEqual(sorted(map(self.t.b.key, chosen)), sorted(map(self.t.b.key, tasks)))
            owners = {}
            for j in jobs:
                if j['floor'] != floor or j['smoke']:
                    continue
                for r in j['tasks']:
                    prior = owners.setdefault(r['case_id'], j['name'])
                    self.assertEqual(prior, j['name'])

    def args(self, memory=True):
        value = SimpleNamespace(dynamic_threshold_min=.35, dynamic_threshold_max=.75,
            dynamic_threshold_fallback=.60, prompt_logit_neighborhood_radius=2)
        if memory:
            value.eval_memory_group_size = 1
        return value

    def test_only_min_changes_and_parser_boundary(self):
        for floor in self.t.FLOORS:
            for memory in (True, False):
                a = self.args(memory); before = vars(a).copy()
                self.t.freeze_settings(a, floor)
                self.assertEqual(vars(a), dict(before, dynamic_threshold_min=floor))
        for floor in (.35, .50, .75):
            with self.assertRaises(ValueError):
                self.t.freeze_settings(self.args(), floor)
        a = self.args(); a.eval_memory_group_size = 4
        with self.assertRaises(ValueError):
            self.t.freeze_settings(a, .60)

    def audit(self, floor=.65):
        return dict(dynamic_threshold=dict(minimum=floor, maximum=.75, fallback=.60),
            background_mode='point', sources=['topcow2024_cta'],
            loaded_task_keys=[['topcow2024_cta', 'c000', 200]],
            dynamic_threshold_rows=[dict(source_name='topcow2024_cta', case_id='c000',
                class_id=200, threshold=.60, source='fallback_insufficient_separation')])

    def test_fallback_and_exact_audit_keys(self):
        tasks, _ = self.fixture(); tasks = tasks[:1]
        self.t.validate_prompt_audit(self.audit(), tasks, 'topcow2024_cta', .65)
        a = self.audit(); a['dynamic_threshold_rows'][0]['source'] = 'prompt_logit_midpoint_clamped'
        with self.assertRaises(ValueError):
            self.t.validate_prompt_audit(a, tasks, 'topcow2024_cta', .65)
        a = self.audit(); a['dynamic_threshold_rows'][0]['case_id'] = 'other'
        with self.assertRaises(ValueError):
            self.t.validate_prompt_audit(a, tasks, 'topcow2024_cta', .65)
        a = self.audit(); a['dynamic_threshold_rows'] = []
        with self.assertRaises(ValueError):
            self.t.validate_prompt_audit(a, tasks, 'topcow2024_cta', .65)

    def test_paired_summary_all_metrics_and_nonfinite(self):
        tasks, _ = self.fixture(); task = tasks[0]
        a = [dict(task, **{m:.7 for m in self.t.b.METRICS})]
        baseline = [dict(task, **{m:.5 for m in self.t.b.METRICS})]
        sixty = [dict(task, **{m:.6 for m in self.t.b.METRICS})]
        summary = self.t.summarize(a, baseline, sixty)
        for m in self.t.b.METRICS:
            self.assertAlmostEqual(summary['delta_from_baseline'][m], .2)
            self.assertAlmostEqual(summary['delta_from_floor60'][m], .1)
        a[0]['dice'] = float('nan')
        with self.assertRaises(ValueError):
            self.t.summarize(a, baseline, sixty)

    def test_status_reports_topcow_not_previous_grid_total(self):
        # Retain only a tiny test metadata file; no recursive temporary cleanup.
        out = Path(__file__).resolve().parents[1]/'.cache'/('topcow_status_test_'+uuid4().hex)
        self.t.b.write(out/'jobs.json', [])
        result = self.t.status(out)
        self.assertEqual(result['expected_new'], 1108)
        self.assertEqual(set(result['per_floor']), {'0.55', '0.6', '0.65', '0.7'})


if __name__ == '__main__':
    unittest.main()
