import importlib
import copy
import hashlib
import unittest
from pathlib import Path
from unittest.mock import patch


class MetricSpeedRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.m=importlib.import_module('tools.bone_metric_speed_grid')
        self.task=dict(source_name='bone',case_id='a',global_class_id=10003,class_name='bone')

    def test_reuse_only_complete_class_floors_and_reject_duplicate_rows(self):
        native=[dict(model='v9_2',floor=.35,row=self.task)]
        records={('v9_2',('bone','a',10003)):dict(record=dict(model='v9_2',task=self.task,
            results=[dict(floor=.4,row=self.task),dict(floor=.45,row=self.task)]))}
        self.assertEqual(len(self.m.combine_completed(native,records)),3)
        records[next(iter(records))]['record']['results'][0]['floor']=.35
        with self.assertRaises(ValueError):self.m.combine_completed(native,records)

    def test_oom_retry_history_uses_original_failure_time_not_new_prepare(self):
        spec=dict(name='old',frame_batch=2,minimum_free_mib=30720,attempt=3,
            oom_retries=1,gpu=4,not_before=0.,avoid_gpu_until=0.,last_failed_gpu=None)
        value=self.m.inherited_history(spec,'torch.OutOfMemoryError: CUDA out of memory',100,256)
        self.assertEqual(value['frame_batch'],1)
        self.assertEqual(value['attempt'],4)
        self.assertEqual(value['oom_retries'],2)
        self.assertLessEqual(value['not_before'],160)
        value=self.m.inherited_history(dict(spec,frame_batch=1),'CUDA out of memory',100,256)
        self.assertEqual(value['minimum_free_mib'],45000)
        self.assertEqual(value['not_before'],160)
        self.assertEqual(value['avoid_gpu_until'],700)
        race=self.m.inherited_history(spec,'GPU_RESOURCE_WAIT',100,75<<8)
        self.assertEqual(race['oom_retries'],1)
        self.assertEqual(race['frame_batch'],2)
        with self.assertRaises(ValueError):self.m.inherited_history(spec,'ValueError: geometry mismatch',100,256)

    def test_scheduler_identity_requires_own_exact_script_mode_cwd_and_start(self):
        root=Path('/project');args=['python','tools/bone_shared_floor_probe_grid.py','wait-launch']
        info=dict(pid=12,uid=1000,cwd=str(root),args=args,starttime='900',state='T')
        self.m.validate_scheduler(info,root,1000)
        for changed in (dict(uid=1001),dict(cwd='/other'),dict(args=['python','other.py','wait-launch']),dict(args=['python','/other/bone_shared_floor_probe_grid.py','wait-launch']),dict(state='S')):
            with self.assertRaises(ValueError):self.m.validate_scheduler(dict(info,**changed),root,1000)
        self.m.validate_scheduler(dict(info,state='R'),root,1000,frozen=False)

    def test_intentional_checkpoint_exit_preserves_batch_without_fake_oom(self):
        spec=dict(frame_batch=1,oom_retries=7,attempt=9,gpu=4)
        result=self.m.inherited_history(spec,'',123,15,intentional=True)
        self.assertEqual(result['oom_retries'],7)
        self.assertEqual(result['frame_batch'],1)
        self.assertEqual(result['attempt'],9)
        self.assertTrue(result['intentional_checkpoint_stop'])
        self.assertEqual(self.m.inherited_history(spec,'',123,0,intentional=True),spec)
        with self.assertRaises(ValueError):self.m.inherited_history(spec,'',123,9,intentional=True)
        raced=self.m.inherited_history(spec,'CUDA out of memory',123,256,intentional=True)
        self.assertEqual(raced['oom_retries'],8)
        self.assertEqual(raced['minimum_free_mib'],45000)

    def test_historical_exit_requires_positive_scheduler_resource_requeue(self):
        name='shared_v9_2_a'
        log='SHARED_START 2 99 '+name+' 3 batch 2\nSHARED_RESOURCE_REQUEUE '+name+' 2 30720\n'
        self.assertEqual(self.m.recorded_resource_exit(name,'',log),75<<8)
        self.assertEqual(self.m.recorded_resource_exit(name,'CUDA out of memory',log),256)
        with self.assertRaises(ValueError):self.m.recorded_resource_exit(name,'CUDA out of memory','')
        for bad in ('','SHARED_RESOURCE_REQUEUE other 2 30720\n',log+'SHARED_START 4 100 '+name+' 3 batch 2\n'):
            with self.assertRaises(ValueError):self.m.recorded_resource_exit(name,'',bad)

    def test_engine_uses_new_output_worker_entry_and_runtime(self):
        e=self.m.engine()
        self.assertTrue(e.OUT.endswith('20261028_metricfast'))
        self.assertTrue(e.__file__.endswith('bone_metric_speed_grid.py'))
        self.assertIn('tools.bone_metric_speed_runtime',e.run.__code__.co_names)
        self.assertIn('tools/bone_fast_metrics.py',e.SOURCES)

    def test_lazy_intermediate_scores_preserve_private_boundary_and_force_once(self):
        runtime=importlib.import_module('tools.bone_metric_speed_runtime')
        calls=[]
        metric=runtime.deferred_metrics(lambda:calls.append(1) or {'dice':.75,'iou':.6},names=('dice','iou'))
        metric['_touches_roi_z_boundaries']=(True,False)
        copied={**metric}
        self.assertEqual(copied.pop('_touches_roi_z_boundaries'),(True,False))
        self.assertFalse(calls)
        runtime.materialize_metric(copied)
        self.assertEqual(copied['dice'],.75)
        self.assertEqual(copied['iou'],.6)
        self.assertEqual(calls,[1])
        unused=runtime.deferred_metrics(lambda:self.fail('unused adaptive stage was scored'))
        unused['_touches_roi_z_boundaries']=(True,True)
        self.assertTrue(any(unused.pop('_touches_roi_z_boundaries')))

    def test_runtime_final_only_metric_and_actual_native_control_on_cache_hit(self):
        import torch
        from tools import ct13_full_validation_eval as b
        runtime=importlib.import_module('tools.bone_metric_speed_runtime')
        old_threads=torch.get_num_threads();torch.set_num_threads(2)
        try:
            with patch.object(b,'metric_module',b.metric_module):
                session=runtime.GridSession('v9_2',[dict(task=self.task,floors=[.35,.4])],'test',lambda r:None)
                session.shared_control=True;session.begin(self.task)
                scorer=b.metric_module().binary_prompt_metrics_from_masks
                pred=torch.zeros(3,16,16);pred[:,3:9,3:9]=1
                target=pred.clone();target[:,8:11,3:9]=1
                session.floor=.35
                discarded=scorer(pred,target,(2.,.7,.8))
                discarded['_touches_roi_z_boundaries']=(True,False)
                self.assertTrue(any(discarded.pop('_touches_roi_z_boundaries')))
                del discarded
                self.assertFalse(session.metric_events)
                final=scorer(pred,target,(2.,.7,.8))
                with patch.object(runtime.PriorSession,'emit',lambda _s,_t,m,_a,**kw:m):
                    runtime.GridSession.emit(session,self.task,final,{})
                self.assertEqual(len(session.metric_events),1)
                self.assertTrue(session.metric_events[0]['native_compared'])
                session.floor=.4
                again=scorer(pred,target,(2.,.7,.8));runtime.materialize_metric(again)
                self.assertTrue(session.metric_events[-1]['cache_hit'])
                self.assertTrue(session.metric_events[-1]['native_compared'])
                self.assertEqual({k:final[k] for k in runtime.METRICS},{k:again[k] for k in runtime.METRICS})
                self.assertEqual(session.deferred_candidates,3)
                self.assertEqual(session.deferred_materialized,2)
                record=dict(task=self.task,results=[dict(floor=e['floor'],row=e['metrics']) for e in session.metric_events],
                    metric_execution=dict(events=session.metric_events,stats={'comparison_failures':0},
                        provider_sha256=hashlib.sha256((Path(runtime.__file__).parent/'bone_fast_metrics.py').read_bytes()).hexdigest()))
                runtime.validate_metric_record(record,True)
                bad=copy.deepcopy(record);bad['metric_execution']['events'][-1]['native_compared']=False
                with self.assertRaises(ValueError):runtime.validate_metric_record(bad,True)
        finally:torch.set_num_threads(old_threads)


if __name__=='__main__':unittest.main()
