import importlib
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np


class RunnerTests(unittest.TestCase):
    def setUp(self):self.m=importlib.import_module('tools.bone_floor_grid')

    def args(self):
        return SimpleNamespace(dynamic_threshold_min=.35,dynamic_threshold_max=.75,
            dynamic_threshold_fallback=.60,prompt_logit_neighborhood_radius=2,
            max_frames=192,validation_window_frames=192,model_input_size=1024,eval_memory_group_size=1)

    def test_only_requested_floor_changes_at_parser(self):
        for model in ('v9_2','v10_2'):
            a=self.args();before=vars(a).copy();self.m.freeze_settings(model,a,.65)
            self.assertEqual(vars(a),{**before,'dynamic_threshold_min':.65})
        a=self.args();a.dynamic_threshold_max=.8
        with self.assertRaises(ValueError):self.m.freeze_settings('v10_2',a,.55)

    def test_actual_stage_rejects_changed_context_or_memory(self):
        a=self.args();a.dynamic_threshold_min=.65
        record=self.m.stage_record('v9_2',a,['bone','0001_phase_3',10003],30)
        self.assertEqual(record['method'],'prompt_probability_midpoint')
        a.max_frames=96
        with self.assertRaises(ValueError):self.m.stage_record('v9_2',a,[],30)
        a=self.args();a.eval_memory_group_size=4
        with self.assertRaises(ValueError):self.m.stage_record('v10_2',a,[],30)

    def test_readonly_case_cache_ram_reuse_never_saves_ct(self):
        class Data:
            def __init__(self):self.calls=0
            def _case_image_cache_path(self,p):return 'unused.npy'
            def _load_case_image_cache(self,p):return None
            def _load_array(self,p):self.calls+=1;return np.full((3,4,5),self.calls,np.int16)
            def _save_case_image_cache(self,*a):raise AssertionError('original writer invoked')
        self.m.install_readonly_ct(Data)
        d=Data();a=d._get_or_create_case_image('a.nii.gz');b=d._get_or_create_case_image('a.nii.gz')
        self.assertIs(a,b);self.assertEqual(d.calls,1);self.assertEqual(a.dtype,np.float32)
        d._get_or_create_case_image('b.nii.gz');self.assertEqual(d.calls,2)
        with self.assertRaises(RuntimeError):d._save_case_image_cache('x',a)

    def test_readonly_case_cache_prefers_existing_shared_array(self):
        cached=np.zeros((2,3,4),np.float32)
        class Data:
            def _case_image_cache_path(self,p):return 'existing.npy'
            def _load_case_image_cache(self,p):return cached
            def _load_array(self,p):raise AssertionError('should use existing shared CT')
        self.m.install_readonly_ct(Data);self.assertIs(Data()._get_or_create_case_image('a'),cached)

    def test_gpu_restrictions(self):
        self.assertEqual(self.m.gpu_ids('2,4,5,6'),[2,4,5,6])
        for x in ('','1','7','0,0','8','2,7'):
            with self.assertRaises(ValueError):self.m.gpu_ids(x)

    def test_prompt_worker_uses_existing_nninteractive_runtime(self):
        cmd=self.m.prompt_worker_argv(Path('job.json'))
        self.assertEqual(Path(cmd[0]),self.m.ROOT.parent/'conda-envs/nninteractive_eval/bin/python')
        self.assertEqual(cmd[-3:],['prompt-worker','--job','job.json'])

    def test_attempts_have_disjoint_run_spec_and_log_paths(self):
        out=self.m.ROOT/'output/test_bone_queue'
        a=self.m.attempt_paths(out,'job',0);b=self.m.attempt_paths(out,'job',1)
        self.assertTrue(all(a[k]!=b[k] for k in ('run','spec','log')))
        self.assertEqual(Path(a['run']).parents[1],out)

    def test_worker_binding_rejects_unsigned_scope_or_gpu_changes(self):
        out=self.m.ROOT/'output/test_bone_queue';signed=dict(name='job',model='v9_2',variant='baseline',
            source='bone',floor=.55,prepared=str(out/'prepared/job'),task_count=10,smoke=False)
        paths=self.m.attempt_paths(out,'job',0)
        job=dict(signed,gpu=2,gpu_uuid='GPU-example',frame_batch=16,attempt=0,run=paths['run'],log=paths['log'])
        self.m.validate_job_spec(job,signed,out)
        for patch in (dict(gpu=1),dict(gpu=7),dict(prepared='/different'),dict(source='amos'),
                      dict(task_count=1),dict(smoke=True),dict(frame_batch=3),dict(attempt=-1),dict(run=str(out/'jobs/other'))):
            with self.assertRaises(ValueError):self.m.validate_job_spec({**job,**patch},signed,out)

    def test_confirmation_requires_every_task_and_native_fallback(self):
        tasks=[dict(source_name='bone',case_id='a',global_class_id=10001)]
        stage=dict(key=['bone','a',10001],floor=.70,threshold=.60,source='fallback_missing_prompt_logits',
            method='prompt_logit_midpoint',memory_group=1,frames=10,input_size=1024)
        self.m.validate_stages([stage],tasks,'v10_2',.70)
        with self.assertRaises(ValueError):self.m.validate_stages([],tasks,'v10_2',.70)
        with self.assertRaises(ValueError):self.m.validate_stages([{**stage,'source':'prompt_logit_midpoint'}],tasks,'v10_2',.70)
        with self.assertRaises(ValueError):self.m.validate_stages([{**stage,'memory_group':4}],tasks,'v10_2',.70)

    def test_prompt_metadata_has_exact_two_planes_and_four_negative_points(self):
        meta=dict(shape_dhw=np.array([10,20,30]),slices=np.array([8,6]),
            background_points_dhw=np.array([[0,1,8],[9,19,8],[0,6,0],[9,6,29]]),
            background_point_planes=np.array(['coronal','coronal','sagittal','sagittal']))
        record=self.m.validate_prompt_metadata(meta,[10,20,30])
        self.assertEqual(record['background_points_per_plane'],2)
        with self.assertRaises(ValueError):self.m.validate_prompt_metadata({**meta,'slices':np.array([9,6])},[10,20,30])
        with self.assertRaises(ValueError):self.m.validate_prompt_metadata({**meta,'background_point_planes':np.array(['coronal']*4)},[10,20,30])
        with self.assertRaises(ValueError):self.m.validate_prompt_metadata(meta,[11,20,30])


if __name__=='__main__':unittest.main()
