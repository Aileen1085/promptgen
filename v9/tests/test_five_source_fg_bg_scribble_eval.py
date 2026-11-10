import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools import five_source_fg_bg_scribble_eval as e

class EvalTests(unittest.TestCase):
    def task(self,case='a',cid=1):return dict(source_name='magic',case_id=case,global_class_id=cid)
    def test_jobs_use_prior_case_actual_batch(self):
        tasks=[self.task(),self.task(cid=2),self.task('b')]
        jobs,controls=e.case_jobs(tasks,{('magic','a'):2,('magic','b'):4})
        self.assertEqual([j['frame_batch'] for j in jobs],[2,4]);self.assertEqual(len(jobs),2)
        self.assertEqual(len(controls),1);self.assertTrue(all(j['variant']=='all' for j in jobs))
    def test_whole_case_publication_no_partial_or_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);tasks=[self.task()];run=out/'run'
            records=[(tasks[0],v,dict(actual_frame_batch=2)) for v in e.VARIANTS]
            with self.assertRaises(ValueError):e.publish_case(out,run,tasks,records[:1],2)
            self.assertFalse((out/'records').exists())
            e.publish_case(out,run,tasks,records,2)
            self.assertTrue(all(e.record_path(out,tasks[0],v).exists() for v in e.VARIANTS))
            with self.assertRaises(FileExistsError):e.publish_case(out,out/'run2',tasks,records,2)
    def test_summary_pairs_by_key_and_records_actual_batches(self):
        tasks=[self.task(),self.task(cid=2)];records=[]
        for t in tasks:
            for v in e.VARIANTS:
                row=dict(t,**{m:(.3 if v=='baseline' else .5) for m in e.METRICS})
                records.append(dict(row=row,variant=v,actual_frame_batch=2))
        summary=e.summarize(list(reversed(records)),tasks)['magic']
        self.assertAlmostEqual(summary[e.VARIANTS[1]]['paired_delta_vs_baseline']['dice'],.2)
        self.assertEqual(summary[e.VARIANTS[1]]['actual_batches'],{'2':2})
        with self.assertRaises(ValueError):e.summarize(records+records[:1],tasks)
    def test_low_disk_is_retry_not_hard_failure(self):
        with patch.object(e.shutil,'disk_usage',return_value=type('D',(),{'free':0})()):
            with self.assertRaises(SystemExit) as c:e.bind_resources({},'.')
        self.assertEqual(c.exception.code,75)
    def test_queue_failure_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);e.atomic_new(p/'protocol.json',{});e.atomic_new(p/'queue_full.json',dict(phase='complete',failed=[1],pending=[],running=[],protocol_sha256=e.sha(p/'protocol.json')))
            with self.assertRaises(ValueError):e.require_queue_complete(p,'full')

class WholeWorkerTests(unittest.TestCase):
    def test_late_oom_and_missing_normal_finish_publish_zero_then_full_pair_retry(self):
        from types import SimpleNamespace
        tasks=[dict(source_name='magic',case_id='a',global_class_id=i) for i in (1,2)]
        protocol=dict(inventory=dict(tasks=tasks,refs=[dict(t,prompt_cache_name='p') for t in tasks],paths={}))
        calls=[]
        def initialize(job,paths):
            batch=job['frame_batch']
            def evaluate(task,variant):
                calls.append((batch,task['global_class_id'],variant))
                if batch==4 and task['global_class_id']==2 and variant==e.VARIANTS[1]:raise RuntimeError('CUDA out of memory')
                return dict(actual_frame_batch=batch,variant=variant,row=task)
            return SimpleNamespace(evaluate=evaluate,finish=lambda:dict(successful_native_completion=batch!=1))
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            for s in e.SOURCES:e.atomic_new(out/'controls'/s/'control_pass.json',dict(passed=True,source=s,protocol_sha256='digest'))
            for b in (1,2,4):e.atomic_new(out/'controls/magic'/f'batch_{b}'/'control_pass.json',dict(passed=True,source='magic',actual_frame_batch=b,protocol_sha256='digest'))
            job=dict(source='magic',case_id='a',tasks=tasks,variant='all',frame_batch=4,attempt=0)
            with patch.object(e,'protocol_at',return_value=(protocol,'digest')),patch.object(e,'bind_resources',return_value={}),patch.object(e,'verify_inputs',return_value={}),patch.object(e.runtime,'initialize',side_effect=initialize),patch.object(e,'validate_bound_record'):
                with self.assertRaisesRegex(RuntimeError,'CUDA out of memory'):e.worker(out,job)
                self.assertFalse((out/'records').exists())
                job.update(frame_batch=1,attempt=1)
                with self.assertRaisesRegex(ValueError,'normal finish'):e.worker(out,job)
                self.assertFalse((out/'records').exists())
                job.update(frame_batch=2,attempt=2);e.worker(out,job)
            self.assertEqual(len([c for c in calls if c[0]==2]),4)
            self.assertTrue(all(e.read(e.record_path(out,t,v))['actual_frame_batch']==2 for t in tasks for v in e.VARIANTS))
    def test_reject_invalid_completed_record_before_publication(self):
        from types import SimpleNamespace
        task=dict(source_name='magic',case_id='a',global_class_id=1)
        protocol=dict(inventory=dict(tasks=[task],refs=[dict(task,prompt_cache_name='p')],paths={}))
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            for s in e.SOURCES:e.atomic_new(out/'controls'/s/'control_pass.json',dict(passed=True,source=s,protocol_sha256='digest'))
            job=dict(source='magic',case_id='a',tasks=[task],variant='all',frame_batch=2)
            session=SimpleNamespace(evaluate=lambda t,v:dict(actual_frame_batch=2),finish=lambda:dict(successful_native_completion=True))
            with patch.object(e,'protocol_at',return_value=(protocol,'digest')),patch.object(e,'require_batch_gate'),patch.object(e,'bind_resources',return_value={}),patch.object(e,'verify_inputs',return_value={}),patch.object(e.runtime,'initialize',return_value=session),patch.object(e,'validate_bound_record',side_effect=ValueError('bad final record')):
                with self.assertRaisesRegex(ValueError,'bad final record'):e.worker(out,job)
            self.assertFalse((out/'records').exists())

class EvidenceTests(unittest.TestCase):
    def test_finite_metrics_and_whole_case_matched_batch_required(self):
        tasks=[dict(source_name='magic',case_id='a',global_class_id=i) for i in (1,2)]
        records=[dict(row=dict(t,**{m:.5 for m in e.METRICS}),variant=v,actual_frame_batch=2) for t in tasks for v in e.VARIANTS]
        records[0]['row']['dice']=float('nan')
        with self.assertRaises(ValueError):e.summarize(records,tasks)
        records[0]['row']['dice']=.5;records[0]['actual_frame_batch']=4
        with self.assertRaises(ValueError):e.summarize(records,tasks)
    def test_numeric_tree_rejects_nan_and_changed_counts(self):
        with self.assertRaises(ValueError):e.compare_numeric_tree({'dice':float('nan')},{'dice':.5})
        self.assertGreater(e.compare_numeric_tree({'actual_batches':{'2':1}},{'actual_batches':{'2':2}}),1e-12)
    def test_native_parity_checks_all_metrics_and_roi(self):
        import copy
        record=dict(actual_frame_batch=2,row={m:.5 for m in e.METRICS},roi=dict(initial=[1],reserved=[1],final=[1],history=[]),stages=[{}])
        for metric in e.METRICS:
            bad=copy.deepcopy(record);bad['row'][metric]+=.01
            with self.assertRaises(ValueError):e.compare_control(record,bad)
        bad=copy.deepcopy(record);bad['roi']['history']=[1]
        with self.assertRaises(ValueError):e.compare_control(record,bad)

class IndependentAuditTests(unittest.TestCase):
    def test_independent_math_class_macro_pair_delta_and_batches(self):
        tasks=[dict(source_name='magic',case_id=c,global_class_id=i) for c,i in [('a',1),('b',1),('c',2)]]
        records=[]
        for t,value in zip(tasks,[.2,.4,.9]):
            for variant in e.VARIANTS:
                records.append(dict(row=dict(t,**{m:value+(.05 if variant!= 'baseline' else 0) for m in e.METRICS}),variant=variant,actual_frame_batch=2))
        with patch.object(e,'summarize',side_effect=AssertionError('audit must not trust merge helper')):
            summary=e.independent_summary(records,tasks)
        self.assertAlmostEqual(summary['magic']['baseline']['mean']['dice'],.5)
        self.assertAlmostEqual(summary['magic']['baseline']['class_macro']['dice'],.6)
        self.assertAlmostEqual(summary['magic'][e.VARIANTS[1]]['paired_delta_vs_baseline']['dice'],.05)
        self.assertEqual(summary['magic']['baseline']['actual_batches'],{'2':3})
    def test_parent_signature_failure_creates_no_output(self):
        with tempfile.TemporaryDirectory() as d:
            parent=Path(d)/'parent';parent.mkdir();(parent/'protocol.json').write_text('{}')
            out=Path(d)/'out'
            with self.assertRaisesRegex(ValueError,'parent proof'):e.prepare(out,parent)
            self.assertFalse(out.exists())
    def test_status_rejects_partial_and_reports_fresh_target_reference_counts(self):
        task=dict(source_name='magic',case_id='a',global_class_id=1)
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);e.atomic_new(out/'protocol.json',dict(inventory=dict(tasks=[task])))
            e.atomic_new(e.record_path(out,task,'baseline'),dict(actual_frame_batch=2))
            with patch.object(e,'validate_protocol'),patch.object(e,'validate_inventory',return_value=[task]),patch.object(e,'validate_bound_record'),patch.object(e,'require_batch_gate'):
                with self.assertRaisesRegex(ValueError,'partial'):e.status(out)
                e.atomic_new(e.record_path(out,task,e.VARIANTS[1]),dict(actual_frame_batch=2))
                value=e.status(out)
            self.assertEqual((value['completed'],value['target_completed'],value['reference_completed']),(2,1,1))

class ProtocolHardeningTests(unittest.TestCase):
    def test_protocol_at_invokes_strict_protocol_validation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);e.atomic_new(p/'protocol.json',{})
            with patch.object(e,'validate_protocol',side_effect=ValueError('strict schema')),patch.object(e,'verify_inputs'):
                with self.assertRaisesRegex(ValueError,'strict schema'):e.protocol_at(p)
    def test_queue_digest_must_match_current_protocol(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);e.atomic_new(p/'protocol.json',{})
            e.atomic_new(p/'queue_full.json',dict(phase='complete',failed=[],pending=[],running=[],protocol_sha256='wrong'))
            with self.assertRaisesRegex(ValueError,'protocol'):e.require_queue_complete(p,'full')

class QueueEvidenceHardeningTests(unittest.TestCase):
    def test_missing_and_duplicate_production_completions_rejected(self):
        tasks=[dict(source_name='magic',case_id=c,global_class_id=1) for c in ('a','b')]
        protocol=dict(inventory=dict(tasks=tasks),parent_actual_batches={json.dumps(('magic',c)):2 for c in ('a','b')})
        job=dict(source='magic',case_id='a',tasks=[tasks[0]],variant='all',frame_batch=2)
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            for completed in ([job],[job,job]):
                with patch.object(e,'require_queue_complete',return_value=dict(completed=completed)):
                    with self.assertRaises(ValueError):e.validate_completed_jobs(out,'full',protocol,'digest')
    def test_strict_protocol_rejects_every_mutated_field(self):
        import copy
        good=dict(schema='new',variants=list(e.VARIANTS),production_task_count=346,inventory=dict(tasks=[1],refs=[2],paths={'x':'y'},case_inputs={'a':['b']},input_sha256={'old':'hash'}),parent_actual_batches={'a':2},source_config={'magic':{'epoch':490}})
        with patch.object(e,'expected_protocol',return_value=good),patch.object(e,'validate_inventory'):
            self.assertEqual(e.validate_protocol(good),good)
            for field in good:
                bad=copy.deepcopy(good);bad[field]=None
                with self.assertRaisesRegex(ValueError,'strict'):e.validate_protocol(bad)
