import importlib
import unittest
from types import SimpleNamespace


class V92PilotTests(unittest.TestCase):
    def setUp(self):
        self.t=importlib.import_module('tools.v92_totalseg_floor_pilot')

    def rows(self):
        return [dict(source_name='totalseg',case_id='s%04d'%(i//5),global_class_id=i%115+1) for i in range(202)]

    def test_exact202_jobs_native_v92(self):
        jobs=self.t.make_jobs(self.rows())
        self.assertEqual(len(jobs),91)
        self.assertEqual(sum(j['task_count'] for j in jobs),1414)
        self.assertEqual(sum(j['smoke'] for j in jobs),7)
        for f in self.t.FLOORS:
            part=[r for j in jobs if j['floor']==f for r in j['tasks']]
            self.assertEqual(set(map(self.t.b.key,part)),set(map(self.t.b.key,self.rows())))
        self.assertTrue(all(j['model']=='v9_2' and j['variant']=='baseline' for j in jobs))

    def test_floor_only_preserves_native_probability_settings(self):
        for f in self.t.FLOORS:
            a=SimpleNamespace(dynamic_threshold_min=.35,dynamic_threshold_max=.75,
                dynamic_threshold_fallback=.60,prompt_logit_neighborhood_radius=2)
            before=vars(a).copy();self.t.freeze_settings(a,f)
            self.assertEqual(vars(a),dict(before,dynamic_threshold_min=f))
        a.dynamic_threshold_min=.5
        with self.assertRaises(ValueError):self.t.freeze_settings(a,.6)

    def test_v92_prompt_threshold_audit_and_fallback(self):
        row=dict(source_name='totalseg',case_id='s1',global_class_id=1)
        a=dict(background_mode='point',sources=['totalseg'],loaded_task_keys=[['totalseg','s1',1]],
            dynamic_threshold=dict(method='prompt_probability_midpoint',minimum=.75,maximum=.75,fallback=.60,prompt_logit_neighborhood_radius=2),
            dynamic_threshold_rows=[dict(class_id=1,threshold=.60,source='fallback_insufficient_separation')])
        self.t.validate_prompt_audit(a,[row],.75)
        a['dynamic_threshold']['method']='prompt_logit_midpoint'
        with self.assertRaises(ValueError):self.t.validate_prompt_audit(a,[row],.75)

    def test_actual_stage_preserves_context_scale_no_memory_requirement(self):
        a=SimpleNamespace(dynamic_threshold_min=.55,dynamic_threshold_max=.75,dynamic_threshold_fallback=.60,
            prompt_logit_neighborhood_radius=2,max_frames=192,validation_window_frames=192,model_input_size=1024)
        self.assertEqual(self.t.stage_record(a,.55,77)['frames'],77)
        a.validation_window_frames=256
        with self.assertRaises(ValueError):self.t.stage_record(a,.55,77)

    def test_winner_task_mean_tie_and_baseline_can_win(self):
        r=dict(source_name='totalseg',case_id='s1',global_class_id=1)
        parts={.35:[dict(r,**{m:.8 for m in self.t.b.METRICS})],.5:[dict(r,**{m:.8 for m in self.t.b.METRICS})]}
        self.assertEqual(self.t.winner(parts,[r])['chosen_floor'],.35)
        parts[.5][0]['dice']=.81
        self.assertEqual(self.t.winner(parts,[r])['chosen_floor'],.5)
        parts[.5][0]['dice']=float('nan')
        with self.assertRaises(ValueError):self.t.winner(parts,[r])

    def test_gpu_exclusions(self):
        self.assertEqual(self.t.gpu_ids('2,4,5,6'),[2,4,5,6])
        for bad in ('1','7','2,2',''):
            with self.assertRaises(ValueError):self.t.gpu_ids(bad)

    def test_v92_baseline_fallback_source_is_not_lost(self):
        row=dict(source_name='totalseg',case_id='s1',global_class_id=1,mask_threshold_used=.60)
        ts=[dict(class_id=1,threshold=.60,source='fallback_insufficient_separation')]
        self.assertEqual(self.t.attach_threshold_sources([row],ts)[0]['dynamic_threshold_source'],ts[0]['source'])
        ts[0]['class_id']=2
        with self.assertRaises(ValueError):self.t.attach_threshold_sources([row],ts)


if __name__=='__main__':unittest.main()
