import importlib
import unittest
from types import SimpleNamespace


class FullWinnerTests(unittest.TestCase):
    def setUp(self):
        self.t=importlib.import_module('tools.v102_totalseg_best_floor_full')

    def fixture(self):
        cs=[c for c in range(1,118) if c not in (40,41)]
        rows=[dict(source_name='totalseg',case_id='s%04d'%(i%57),global_class_id=cs[i%115]) for i in range(4034)]
        split=dict(datasets=dict(totalseg=dict(val=[dict(case_id='s%04d'%i) for i in range(57)],train=[dict(case_id='train')])) )
        return rows,split

    def pilot(self):
        rows,split=self.fixture(); expected=self.t.p.select_subset(rows,split)
        def part(score):
            return dict(tasks=202,mean={m:score for m in self.t.b.METRICS},
                per_case_class=[dict(r,**{m:score for m in self.t.b.METRICS}) for r in expected])
        result=dict(floors={format(f,'.2f'):part(.8 if f in (.50,.55) else .7) for f in self.t.p.FLOORS},
            baseline035=part(.65),new_predictions=1414,readonly_baseline_tasks=202)
        return result,expected

    def test_waits_until_every_shard_and_task_audited(self):
        good=dict(jobs=91,audited=91,audited_tasks=1414,expected_new=1414,
            scheduler=dict(pending=0,running={},failed=[]))
        self.t.require_complete(good)
        for field,value in [('audited',90),('audited_tasks',1413)]:
            bad=dict(good,**{field:value})
            with self.assertRaises(ValueError):self.t.require_complete(bad)
        for field,value in [('pending',1),('running',{'0':{}}),('failed',[{}])]:
            bad=dict(good,scheduler=dict(good['scheduler'],**{field:value}))
            with self.assertRaises(ValueError):self.t.require_complete(bad)

    def test_global_task_dice_exact_tie_smaller_floor(self):
        result,expected=self.pilot(); selected=self.t.choose_winner(result,expected)
        self.assertEqual(selected['chosen_floor'],.50)
        self.assertAlmostEqual(selected['candidate_dice']['0.55'],.8)
        result['floors']['0.55']['mean']['mean_slice_dice']=.99
        with self.assertRaises(ValueError):self.t.choose_winner(result,expected)

    def test_missing_nonfinite_or_mean_mismatch_cannot_select(self):
        for mode in ('missing','nonfinite','mismatch','duplicate'):
            result,expected=self.pilot()
            if mode=='missing':del result['floors']['0.75']
            elif mode=='nonfinite':result['floors']['0.75']['per_case_class'][0]['dice']=float('nan')
            elif mode=='mismatch':result['floors']['0.75']['mean']['dice']=.99
            else:result['floors']['0.75']['per_case_class'][-1]=result['floors']['0.75']['per_case_class'][0]
            with self.assertRaises(ValueError):self.t.choose_winner(result,expected)

    def test_exact_full_scope_and_once_only_whole_case_jobs(self):
        rows,split=self.fixture(); full=self.t.full_tasks(rows,split)
        jobs=self.t.make_jobs(full,.50)
        self.assertEqual(sum(j['task_count'] for j in jobs),4034)
        self.assertEqual(sum(j['smoke'] for j in jobs),1)
        self.assertEqual({j['floor'] for j in jobs},{.50})
        self.assertEqual(sorted(self.t.b.key(r) for j in jobs for r in j['tasks']),sorted(map(self.t.b.key,rows)))
        owner={}
        for j in jobs:
            if j['smoke']:continue
            for r in j['tasks']:self.assertEqual(owner.setdefault(r['case_id'],j['name']),j['name'])
        for wrong in (rows[:-1],rows[:-1]+rows[:1]):
            with self.assertRaises(ValueError):self.t.full_tasks(wrong,split)
        split['datasets']['totalseg']['train'].append(dict(case_id='s0000'))
        with self.assertRaises(ValueError):self.t.full_tasks(rows,split)

    def test_only_minimum_changes_native_fallback_preserved(self):
        for floor in self.t.CANDIDATES:
            a=SimpleNamespace(dynamic_threshold_min=.35,dynamic_threshold_max=.75,
                dynamic_threshold_fallback=.60,prompt_logit_neighborhood_radius=2)
            original=vars(a).copy();self.t.freeze_settings(a,floor)
            self.assertEqual(vars(a),dict(original,dynamic_threshold_min=floor))
        self.assertTrue(self.t.g.threshold_valid(dict(threshold=.60,source='fallback_insufficient_separation'),.75))

    def test_audit_exact_keys_and_summary_without_floor60(self):
        row=dict(source_name='totalseg',case_id='s0000',global_class_id=1)
        a=dict(dynamic_threshold=dict(minimum=.50,maximum=.75,fallback=.60),sources=['totalseg'],background_mode='point',
            loaded_task_keys=[['totalseg','s0000',1]],dynamic_threshold_rows=[dict(source_name='totalseg',case_id='s0000',class_id=1,threshold=.55,source='prompt_logit_midpoint')])
        self.t.validate_prompt_audit(a,[row],'totalseg',.50)
        a['dynamic_threshold_rows'][0]['case_id']='other'
        with self.assertRaises(ValueError):self.t.validate_prompt_audit(a,[row],'totalseg',.50)
        v=[dict(row,**{m:.8 for m in self.t.b.METRICS})];old=[dict(row,**{m:.6 for m in self.t.b.METRICS})]
        s=self.t.summarize(v,old)
        self.assertAlmostEqual(s['delta_from_baseline']['dice'],.2)


if __name__=='__main__':unittest.main()
