import importlib
import unittest
from types import SimpleNamespace


class FloorGridTests(unittest.TestCase):
    def setUp(self):self.g=importlib.import_module('tools.v102_floor_grid')

    def args(self):return SimpleNamespace(dynamic_threshold_min=.35,dynamic_threshold_max=.75,
        dynamic_threshold_fallback=.60,prompt_logit_neighborhood_radius=2,eval_memory_group_size=1)

    def test_only_min_changes(self):
        a=self.args();before=vars(a).copy();self.g.freeze_settings(a,.55)
        self.assertEqual(vars(a),dict(before,dynamic_threshold_min=.55))

    def test_reject_nonfrozen_floor_and_native_settings(self):
        for floor in (.35,.60,.75,.1):
            with self.assertRaises(ValueError):self.g.freeze_settings(self.args(),floor)
        a=self.args();a.dynamic_threshold_max=.8
        with self.assertRaises(ValueError):self.g.freeze_settings(a,.55)

    def tasks(self):
        return [dict(source_name=s,case_id=str(i),global_class_id=1) for s in ('amos','magic') for i in range(4)]

    def test_grid_job_scope_and_smokes(self):
        tasks=self.tasks();jobs=self.g.make_jobs(tasks)
        self.assertEqual(sum(j['smoke'] for j in jobs),8)
        for floor in self.g.FLOORS:
            selected=[r for j in jobs if j['floor']==floor for r in j['tasks']]
            self.assertEqual(sorted(map(self.g.b.key,selected)),sorted(map(self.g.b.key,tasks)))

    def audit(self):
        return dict(dynamic_threshold=dict(minimum=.55,maximum=.75,fallback=.60),
            background_mode='point',sources=['magic'],loaded_task_keys=[['magic','0',1]],
            dynamic_threshold_rows=[dict(threshold=.6)],
            prompt_coordinate_transform='raw_magic_nrrd_to_preprocessed_nifti_swap_flip_xy_v1')

    def test_real_audit_rejects_wrong_floor_or_threshold_coverage(self):
        tasks=[r for r in self.tasks() if r['source_name']=='magic' and r['case_id']=='0']
        self.g.validate_prompt_audit(self.audit(),tasks,'magic',.55)
        with self.assertRaises(ValueError):self.g.validate_prompt_audit(self.audit(),tasks,'magic',.65)
        a=self.audit();a['dynamic_threshold_rows']=[]
        with self.assertRaises(ValueError):self.g.validate_prompt_audit(a,tasks,'magic',.55)

    def test_paired_summary_both_references(self):
        task=self.tasks()[0]
        rows=[dict(task,**{m:.7 for m in self.g.b.METRICS})]
        old=[dict(task,**{m:.5 for m in self.g.b.METRICS})]
        sixty=[dict(task,**{m:.6 for m in self.g.b.METRICS})]
        out=self.g.summarize(rows,old,sixty)
        self.assertAlmostEqual(out['delta_from_baseline']['dice'],.2)
        self.assertAlmostEqual(out['delta_from_floor60']['dice'],.1)
        rows[0]['dice']=float('nan')
        with self.assertRaises(ValueError):self.g.summarize(rows,old,sixty)

if __name__=='__main__':unittest.main()
