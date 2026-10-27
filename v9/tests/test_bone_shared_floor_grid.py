import importlib
import unittest
from pathlib import Path
from unittest.mock import patch
from tools.bone_floor_protocol import FLOORS


class SharedGridTests(unittest.TestCase):
    def setUp(self):
        self.m = importlib.import_module('tools.bone_shared_floor_grid')
        self.tasks = [dict(source_name='bone',case_id='a',global_class_id=10003,class_name='x'),dict(source_name='bone',case_id='a',global_class_id=10004,class_name='y')]

    def test_case_grouping_excludes_completed_floor_keys(self):
        completed = {('v9_2',.35,('bone','a',10003)),('v9_2',.4,('bone','a',10003))}
        jobs = self.m.case_jobs(self.tasks,completed,{})
        self.assertEqual(len(jobs),2)
        v9=next(j for j in jobs if j['model']=='v9_2')
        self.assertEqual(v9['pending'][0]['floors'],list(FLOORS[2:]))
        self.assertEqual(sum(len(x['floors']) for j in jobs for x in j['pending']),34)

    def test_existing_small_batch_and_high_memory_history_is_preserved(self):
        identity=('v9_2',.45,('bone','a',10003))
        history={identity:dict(frame_batch=1,minimum_free_mib=45000,attempt=9,oom_retries=6,name='old_shard')}
        job=next(j for j in self.m.case_jobs(self.tasks,set(),history) if j['model']=='v9_2')
        self.assertEqual(job['frame_batch'],1)
        self.assertEqual(job['minimum_free_mib'],45000)
        self.assertEqual(job['parent_history'][0]['attempt'],9)
        self.assertEqual(job['oom_retries'],0)

    def test_batch_one_oom_waits_without_stopping_others(self):
        j=dict(frame_batch=1,minimum_free_mib=30720,attempt=2,oom_retries=1,gpu=4)
        next_job=self.m.retry(j,1,'torch.OutOfMemoryError: CUDA out of memory',100)
        self.assertEqual(next_job['minimum_free_mib'],45000)
        self.assertEqual(next_job['frame_batch'],1)
        self.assertEqual(next_job['not_before'],160)
        self.assertEqual(next_job['avoid_gpu_until'],700)
        self.assertEqual(next_job['attempt'],3)
        self.assertIsNone(self.m.retry(j,1,'ValueError: bad geometry',100))

    def test_resource_race_does_not_fake_an_oom(self):
        j=dict(frame_batch=2,minimum_free_mib=30720,attempt=0,oom_retries=0,gpu=5)
        result=self.m.retry(j,75,'GPU_RESOURCE_WAIT',100)
        self.assertEqual(result['frame_batch'],2)
        self.assertEqual(result['oom_retries'],0)

    def test_non_oom_final_audit_failure_cannot_hide_behind_complete_classes(self):
        self.assertEqual(self.m.attempt_outcome(1,'ValueError: prompt scope mismatch',False),'fatal')
        self.assertEqual(self.m.attempt_outcome(1,'OutOfMemoryError: CUDA out of memory',True),'retry')
        self.assertEqual(self.m.attempt_outcome(0,'ok',False),'done')

    def test_cannot_handoff_while_any_old_worker_is_live(self):
        with self.assertRaises(ValueError):self.m.require_handoff([],['12'])
        self.m.require_handoff([dict(verified=True)],[])

    def test_bad_class_proof_rejected_before_reuse(self):
        with self.assertRaises(ValueError):
            self.m.validate_class(dict(model='v9_2',task=self.tasks[0],results=[]),{'task':self.tasks[0],'floors':[.35]})

    def test_control_requires_same_expansion_stages_not_just_same_metrics(self):
        row={k:.7 for k in self.m.p.METRICS}
        stage=dict(key=['bone','a',10003],floor=.35,frames=10,input_size=1024,method='prompt_probability_midpoint',context=192,memory='native_v92_no_cross_frame_memory',threshold=.6,source='calibrated')
        record=dict(results=[dict(floor=.35,row=row,stages=[stage])])
        original={.35:dict(row=row,stages=[stage])}
        self.m.compare_control(record,original)
        with self.assertRaises(ValueError):self.m.compare_control(record,{.35:dict(row=row,stages=[dict(stage,frames=11)])})

    def test_attempt_cannot_lower_inherited_memory_or_change_paths(self):
        canonical=dict(name='case_job',model='v9_2',control=False,source='bone',variant='baseline',case_id='a',prepared='/prepared/a',frame_batch=1,minimum_free_mib=45000,parent_history=[{'name':'old'}])
        out=Path('/project/output/shared')
        job=dict(canonical,gpu=2,gpu_uuid='GPU-test',attempt=3,oom_retries=1,run=str(out/'runs/case_job_a3'),spec=str(out/'job_specs/case_job_a3.json'),log=str(out/'logs/case_job_a3.log'))
        self.m.validate_attempt(job,canonical,out)
        with self.assertRaises(ValueError):self.m.validate_attempt(dict(job,minimum_free_mib=30720),canonical,out)
        with self.assertRaises(ValueError):self.m.validate_attempt(dict(job,run='/other/run'),canonical,out)
        with self.assertRaises(ValueError):self.m.validate_attempt(dict(job,parent_history=[]),canonical,out)
        with self.assertRaises(ValueError):self.m.validate_attempt(dict(job,variant='no_semantic'),canonical,out)
        with self.assertRaises(ValueError):self.m.validate_attempt(dict(job,source='amos'),canonical,out)

    def test_unstarted_parent_job_preserves_signed_resume_history(self):
        job=dict(name='old',model='v9_2',frame_batch=16)
        value=self.m.parent_limit(job,None,dict(frame_batch=1,minimum_free_mib=45000,attempt=9,oom_retries=6))
        self.assertEqual(value['frame_batch'],1)
        self.assertEqual(value['minimum_free_mib'],45000)
        self.assertEqual(value['attempt'],9)
        self.assertEqual(value['name'],'old')
        self.assertEqual(self.m.parent_limit(job,None,None)['frame_batch'],16)

    def test_grouped_case_preserves_cooldown_and_failed_gpu_exclusion(self):
        history={('v9_2',.45,('bone','a',10003)):dict(name='old',frame_batch=1,minimum_free_mib=45000,not_before=160,avoid_gpu_until=700,last_failed_gpu=4)}
        j=next(x for x in self.m.case_jobs(self.tasks,set(),history) if x['model']=='v9_2')
        self.assertFalse(self.m.eligible(j,4,{4:{'free':50000}},200))
        self.assertFalse(self.m.eligible(j,2,{2:{'free':50000}},150))
        self.assertTrue(self.m.eligible(j,2,{2:{'free':50000}},200))

    def test_retry_spec_refreshes_class_commits_after_worker_poll(self):
        job=dict(name='case_job',model='v9_2',control=False,attempt=2,pending=[dict(task=t,floors=[.35]) for t in self.tasks])
        newly_committed={('v9_2',('bone','a',10003)):dict()}
        with patch.object(self.m,'class_records',return_value=newly_committed),patch.object(self.m,'atomic_new'),patch.object(Path,'exists',return_value=False),patch.object(Path,'mkdir'):
            result=self.m.make_attempt(Path('/project/out'),job,2,{2:{'uuid':'GPU-test'}},{})
        self.assertEqual(result['task_count'],1)
        self.assertEqual(result['pending'][0]['task']['global_class_id'],10004)

    def test_older_whole_shard_does_not_require_later_resource_artifact(self):
        old=dict(run=str(self.m.ROOT/'output/bone_v92e360_v102e490_floor_grid_20261026_validationfix/jobs/a'))
        current=dict(run=str(self.m.ROOT/self.m.OLD/'jobs/a'))
        self.assertEqual(len(self.m.artifact_files(old)),6)
        self.assertEqual(len(self.m.artifact_files(current)),7)

    def test_legacy_v92_fallback_source_is_read_from_final_native_stage(self):
        task=dict(self.tasks[0],local_class_id=3)
        raw=dict(task,**{name:.7 for name in self.m.p.METRICS},mask_threshold_used=.6)
        payloads={'manifest.json':dict(tasks=[task]),'confirmation_audit.json':dict(stages=[dict(key=list(self.m.p.key(task)),source='fallback_missing_prompt_logits')]),'complete_metrics.json':dict(per_case_class=[raw])}
        spec=dict(run='/native/run',prepared='/native/prepared')
        with patch.object(self.m.g,'verify_job'),patch.object(self.m.b,'read',side_effect=lambda f:payloads[Path(f).name]):
            row=self.m.reused_rows(spec)[0]
        self.assertEqual(row['dynamic_threshold_source'],'fallback_missing_prompt_logits')
        self.assertEqual(row['label_id'],3)

    def test_final_results_reconstruct_original_shard_key_sets(self):
        task=self.tasks[0];row=dict(task,**{name:.7 for name in self.m.p.METRICS})
        shards=[dict(name='old35',model='v9_2',floor=.35,tasks=[task])]
        rows={('v9_2',.35,self.m.p.key(task)):row}
        self.assertEqual(self.m.validate_original_partition(shards,rows),{'old35':1})
        with self.assertRaises(ValueError):self.m.validate_original_partition(shards,{})

    def test_marker_must_bind_all_required_attempt_inputs(self):
        marker=Path('/project/out/runs/job_a0/classes/a__10003.done.json')
        with self.assertRaises(ValueError):self.m.require_marker_files(dict(files={},record='/elsewhere.json'),marker,Path('/project/out'),dict(name='job'))

    def test_class_marker_must_belong_to_its_actual_attempt_subset(self):
        spec=dict(pending=[dict(task=self.tasks[0],floors=[.35])])
        self.assertEqual(self.m.expected_class(spec,self.tasks[0])['floors'],[.35])
        with self.assertRaises(ValueError):self.m.expected_class(spec,self.tasks[1])
        with self.assertRaises(ValueError):self.m.expected_class(spec,dict(self.tasks[0],image='changed'))

    def test_prepared_inputs_cannot_change_task_paths_or_prompt_references(self):
        pending=[dict(task=self.tasks[0],floors=[.35])]
        canonical=dict(pending=pending);spec=dict(pending=pending,task_count=1)
        manifest=self.m.g.bone_payload([self.tasks[0]]);refs=dict(per_case_class=[dict(self.tasks[0],prompt_cache_name='exact.npz')])
        self.m.validate_worker_inputs(spec,canonical,manifest,refs,refs)
        bad=dict(manifest,tasks=[dict(self.tasks[0],image='other_ct')])
        with self.assertRaises(ValueError):self.m.validate_worker_inputs(spec,canonical,bad,refs,refs)

    def test_final_merge_rejects_fatal_audit_or_live_workers(self):
        state=dict(phase='full',pending=0,running={},failed=[])
        self.m.require_merge_state(state,[])
        with self.assertRaises(ValueError):self.m.require_merge_state(dict(state,failed=[dict(error='audit')]),[])
        with self.assertRaises(ValueError):self.m.require_merge_state(state,['12'])


if __name__=='__main__':unittest.main()
