"""CPU tests of immutable prompt-ablation collection and real queue decisions."""
import copy
import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((Path(__file__).resolve().parents[1]/'tools/amos_prompt_ablation_eval.py').exists(),
                        'independent AMOS controller is required')
        self.m=importlib.import_module('tools.amos_prompt_ablation_eval')
        self.task=dict(source_name='amos',case_id='0001',global_class_id=201,
                       local_class_id=1,class_name='spleen',image='data/a.nii.gz',label='data/b.nii.gz')

    def tasks(self):
        return [dict(self.task,case_id=f'{c:04}',global_class_id=201+i,local_class_id=i+1)
                for c in range(100) for i in range(14 if c<15 else 15)]

    def job(self):return self.m.case_jobs([self.task],strict=False)[0]

    def row(self,task=None,**values):
        return dict(task or self.task,**{m:.5 for m in self.m.METRICS},label_id=1,
                    mask_threshold_used=.70,dynamic_threshold_source='prompt_logits',**values)

    def record(self,variant,batch=4):
        return dict(task=self.task,variant=variant,row=self.row(),actual_frame_batch=batch,
                    stages=[dict(floor=.70)],restoration={'proof':'runtime validated'})

    def test_scope_exact_cases_classes_keys_and_training_exclusion(self):
        m=self.m;tasks=self.tasks();m.validate_scope(tasks)
        split=dict(datasets={'amos':dict(val=[dict(case_id=f'{c:04}') for c in range(100)],train=[dict(case_id='train')])})
        m.validate_scope(tasks,split)
        for bad in (tasks[:-1],tasks+[tasks[0]],[],[dict(t,source_name='bone') for t in tasks]):
            with self.assertRaises(ValueError):m.validate_scope(bad)
        split['datasets']['amos']['train'].append(dict(case_id='0000'))
        with self.assertRaises(ValueError):m.validate_scope(tasks,split)

    def test_cases_keep_six_variants_fresh_and_controls_excluded(self):
        m=self.m;tasks=self.tasks();old=copy.deepcopy(tasks);jobs=m.case_jobs(tasks)
        self.assertEqual(len(jobs),100);self.assertEqual(tasks,old)
        self.assertEqual(sum(len(j['pending']) for j in jobs)*len(m.VARIANTS),8910)
        self.assertTrue(all(j['frame_batch']==4 and not j['control'] and j['model']=='v10_2' for j in jobs))
        control=m.control_jobs(tasks)[0]
        self.assertTrue(control['control']);self.assertEqual(control['task_count'],1)
        self.assertEqual(len(control['pending']),1)

    def test_oom_ladder_and_resource75_do_not_change_protocol(self):
        m=self.m;j=self.job()|dict(gpu=2)
        same=m.retry(j,75,'CUDA out of memory resource retry',100)
        self.assertEqual((same['frame_batch'],same['oom_retries']),(4,0))
        two=m.retry(j,1,'CUDA out of memory',100);one=m.retry(two,1,'torch.OutOfMemoryError',120)
        wait=m.retry(one,1,'CUDA out of memory',140)
        self.assertEqual((two['frame_batch'],one['frame_batch']), (2,1))
        self.assertEqual((wait['frame_batch'],wait['minimum_free_mib'],wait['not_before'],wait['avoid_gpus']['2']),(1,45000,200,740))
        self.assertFalse(m.eligible(wait,2,{2:dict(free=65000)},201))
        self.assertFalse(m.eligible(j,0,{0:dict(free=65000)},201))
        self.assertIsNone(m.retry(j,1,'bad protocol',100))

    def test_atomic_publication_refuses_replacement(self):
        m=self.m;d=Path(tempfile.mkdtemp());path=d/'record.json'
        m.atomic_new(path,{'x':1})
        with self.assertRaises(FileExistsError):m.atomic_new(path,{'x':2})
        self.assertEqual(m.read(path),{'x':1})

    def test_class_commit_requires_all_variants_same_actual_batch_and_eight_proofs(self):
        m=self.m;d=Path(tempfile.mkdtemp());run=d/'run';j=self.job()|dict(name='j',run=str(run))
        proofs={}
        for i in range(8):
            p=d/f'proof{i}.json';m.atomic_new(p,{'i':i});proofs[str(p)]=m.sha(p)
        records=[self.record(v) for v in m.VARIANTS]
        validator=lambda *a,**kw:None
        with patch.object(m,'validate_record',side_effect=validator):
            for bad in (records[:-1],records+[records[0]],[dict(r,actual_frame_batch=2) if i==2 else r for i,r in enumerate(records)]):
                with self.assertRaises(ValueError):m.commit_class(j,self.task,bad,proofs)
            self.assertFalse((run/'classes/0001__201.done.json').exists())
            marker=m.commit_class(j,self.task,records,proofs)
            self.assertEqual(len(m.read(marker)['files']),9)
            with self.assertRaises(FileExistsError):m.commit_class(j,self.task,records,proofs)

    def test_partial_results_are_never_completed_classes(self):
        m=self.m;d=Path(tempfile.mkdtemp());m.atomic_new(d/'runs/unfinished/classes/0001__201.json',{'partial':True})
        m.atomic_new(d/'jobs.json',{'cases':[self.job()],'controls':[]})
        self.assertEqual(m.class_records(d),{})

    def test_preflight_refuses_output_tmux_process_and_pipeline(self):
        m=self.m;d=Path(tempfile.mkdtemp());out=d/'out';log=d/'log'
        m.preflight(out,processes=[],tmux_exists=False,pipeline=log)
        for kwargs in (dict(processes=[{'pid':42}],tmux_exists=False),dict(processes=[],tmux_exists=True)):
            with self.assertRaises(ValueError):m.preflight(out,**kwargs)
        log.write_text('existing')
        with self.assertRaises(FileExistsError):m.preflight(out,processes=[],tmux_exists=False,pipeline=log)
        out.mkdir()
        with self.assertRaises(FileExistsError):m.preflight(out,processes=[],tmux_exists=False)

    def test_merge_guard_requires_drained_full_queue_and_native_worker_complete(self):
        m=self.m;state=dict(phase='full',pending=0,running={},failed=[])
        m.require_merge_state(state,[])
        for bad,live in ((state,[1]),(state|dict(pending=1),[]),(state|dict(running={'2':{}}),[]),(state|dict(failed=[1]),[]),(state|dict(phase='controls'),[])):
            with self.assertRaises(ValueError):m.require_merge_state(bad,live)
        d=Path(tempfile.mkdtemp());run=d/'runs/case';job=self.job()|dict(run=str(run),spec=str(d/'spec.json'))
        m.atomic_new(d/'spec.json',job);m.atomic_new(d/'protocol.json',{'frozen':True})
        with self.assertRaises((ValueError,FileNotFoundError)):m.verify_worker_complete(job)
        restoration=dict(model='v10_2',epoch=490,weight_mode='live',prompt_parameters_exact=418,decoder_parameters_exact=212,guards_passed=True,model_load_count=1)
        m.atomic_new(run/'restoration_start.json',restoration)
        m.atomic_new(run/'restoration_final.json',dict(restoration,initial_restoration=restoration,successful_native_completion=True,full_tensor_comparison_phase='normal_worker_completion'))
        m.atomic_new(run/'native_worker_complete.json',dict(native_worker_complete=True,classes=1,variants=6,
            restoration_final_sha256=m.sha(run/'restoration_final.json'),
            spec_sha256=m.sha(d/'spec.json'),protocol_sha256=m.sha(d/'protocol.json')))
        m.verify_worker_complete(job)

    def test_metric_keys_bounds_fsum_classmacro_and_exact_paired_keys(self):
        m=self.m;rows=[self.row(),self.row(dict(self.task,case_id='0002'))]
        result=m.summarize(rows);self.assertEqual(result['mean']['dice'],.5)
        self.assertEqual(result['class_macro_dice'],.5);self.assertEqual(result['fallback_count'],0)
        self.assertEqual(m.paired_delta(rows,rows)['dice'],0.)
        ratio=[dict(rows[0],pred_gt_ratio=3.)];m.validate_rows(ratio,ratio)
        for bad in ([dict(rows[0],dice=1.01)],[dict(rows[0],dice=float('nan'))],rows+[rows[0]]):
            with self.assertRaises(ValueError):m.validate_rows(bad,rows)
        with self.assertRaises(ValueError):m.paired_delta(rows,rows[:1])

    def test_frozen_file_inventory_rehashes_only_bound_dependencies(self):
        m=self.m;d=Path(tempfile.mkdtemp());a=d/'a.py';a.write_text('pass\n');frozen=m.file_inventory([a],d)
        (d/'unrelated.py').write_text('anything\n')
        m.verify_inventory(frozen,d)
        a.write_text('changed\n')
        with self.assertRaises(ValueError):m.verify_inventory(frozen,d)

    def test_independent_real_gpu_control_gate_required(self):
        m=self.m;d=Path(tempfile.mkdtemp());m.atomic_new(d/'protocol.json',{'frozen':True})
        with patch.object(m,'check_controls',return_value={'digest':'abc'}):
            with self.assertRaises(FileNotFoundError):m.check_control_gate(d)
            gate=dict(passed=True,real_gpu=True,reviewer='independent-review',protocol_sha256=m.sha(d/'protocol.json'),controls_digest='abc')
            m.atomic_new(d/'control_verification.json',gate);self.assertEqual(m.check_control_gate(d),gate)

    def test_pipeline_is_once_only_and_orders_independent_gate_before_full(self):
        m=self.m;d=Path(tempfile.mkdtemp());events=[]
        with patch.object(m,'ROOT',d),patch.object(m,'preflight',side_effect=lambda *a,**kw:events.append('preflight')),\
             patch.object(m,'prepare',side_effect=lambda *a,**kw:events.append('prepare')),\
             patch.object(m,'schedule',side_effect=lambda out,control:events.append('control' if control else 'full')),\
             patch.object(m,'wait_control_gate',side_effect=lambda out:events.append('gate')),\
             patch.object(m,'merge',side_effect=lambda out:events.append('merge')):
            m.run_pipeline(d/m.OUT)
            self.assertEqual(events,['preflight','prepare','control','gate','full','merge'])
            with self.assertRaises(FileExistsError):m.run_pipeline(d/m.OUT)

    def test_protocol_file_checksum_refuses_policy_or_inventory_mutation(self):
        m=self.m;d=Path(tempfile.mkdtemp());protocol=d/'protocol.json';m.atomic_new(protocol,dict(native_source_sha256='a'*64))
        m.atomic_new(d/'protocol_signature.json',dict(sha256=m.sha(protocol)))
        protocol.write_text(json.dumps(dict(native_source_sha256='b'*64)))
        with self.assertRaisesRegex(ValueError,'protocol'):m.verify(d)

    def test_prepare_freezes_original_100_case_scope_and_explicit_source_inventory(self):
        m=self.m;d=Path(tempfile.mkdtemp());out=d/m.OUT;tasks=self.tasks()
        catalog=[dict(dataset='amos',global_class_id=201+i,local_class_id=1+i,class_name=f'organ{i}') for i in range(15)]
        for t in tasks:
            t.update(class_name=catalog[t['local_class_id']-1]['class_name'],image=f'data/{t["case_id"]}/ct',label=f'data/{t["case_id"]}/gt')
        split=dict(class_catalog=catalog,source_order=['amos'],datasets={'amos':dict(train=[dict(case_id='train')],val=[] )})
        for c in range(100):
            selected=[t for t in tasks if t['case_id']==f'{c:04}']
            split['datasets']['amos']['val'].append(dict(case_id=f'{c:04}',image=selected[0]['image'],label=selected[0]['label'],classes=[t['global_class_id'] for t in selected]))
        refs=[dict(source_name='amos',case_id=t['case_id'],global_class_id=t['global_class_id'],prompt_cache_name=f'p{i}.npz') for i,t in enumerate(tasks)]
        m.atomic_new(d/m.TRAINING,split);m.atomic_new(d/m.HISTORY/'manifest.json',dict(tasks=tasks))
        historic=[dict(self.row(t),class_id=t['global_class_id']) for t in tasks]
        for row in historic:row.pop('local_class_id');row.pop('global_class_id')
        m.atomic_new(d/m.OLD,dict(floors={'0.70':dict(amos=dict(per_case_class=historic))}))
        files=[d/n for n in m.SOURCE_SEEDS]
        for f in files:f.parent.mkdir(parents=True,exist_ok=True);f.write_text('pass\n')
        for name in (m.WEIGHTS,'sam2/checkpoints/sam2.1_hiera_large.pt','models/sat-medical-text/text_encoder.pth','models/biolord-2023-c/config.json'):
            f=d/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_text('test bytes only')
        with patch.object(m,'ROOT',d),patch.object(m,'TRAINING_SHA',m.sha(d/m.TRAINING)),patch.object(m,'WEIGHT_SHA',m.sha(d/m.WEIGHTS)),\
             patch.object(m,'OLD_SHA',m.sha(d/m.OLD)),patch.object(m,'prompt_references',return_value=(refs,[])),\
             patch.object(m,'data_and_cache_inventory',return_value=dict(dataset={},prompts={},shared_ct={})),\
             patch.object(m,'dependency_sources',return_value=files):
            m.prepare(out,already_checked=True);m.verify(out)
            self.assertEqual(len(m.read(out/'jobs.json')['cases']),100)
            self.assertEqual(m.read(out/'protocol.json')['policy']['rows'],8910)
            self.assertEqual(set(m.read(out/'protocol.json')['sources']),set(m.SOURCE_SEEDS))
            (d/'later_unrelated.py').write_text('unrelated source')
            m.verify(out)
            (d/m.SOURCE_SEEDS[0]).write_text('modified controller')
            with self.assertRaises(ValueError):m.verify(out)

    def test_merge_excludes_controls_keeps_exact8910_rows_and_pairs_both_baselines(self):
        m=self.m;d=Path(tempfile.mkdtemp());tasks=self.tasks();found={};reference=[]
        for t in tasks:
            rows=[]
            for i,v in enumerate(m.VARIANTS):
                row=self.row(t)
                row.update({name:.4+.01*i for name in m.METRICS})
                rows.append(dict(variant=v,row=row))
            found[m.key(t)]=dict(record=dict(records=rows),spec=dict(frame_batch=4))
            reference.append(dict(rows[0]['row']))
        m.atomic_new(d/'manifest.json',dict(tasks=tasks));m.atomic_new(d/'jobs.json',dict(cases=[],controls=[]))
        m.atomic_new(d/'status.json',dict(phase='full',pending=0,running={},failed=[]))
        m.atomic_new(d/'protocol.json',dict(frozen=True));m.atomic_new(d/'control_verification.json',dict(passed=True))
        with patch.object(m,'verify') as verification,patch.object(m,'check_control_gate'),patch.object(m,'related',return_value=[]),\
             patch.object(m,'class_records',return_value=found),patch.object(m,'native_reference',return_value=reference):
            merged=m.merge(d)
            self.assertEqual(merged['rows'],8910);self.assertFalse(merged['controls_in_final']);self.assertFalse(merged['winner_selection'])
            self.assertEqual(sum(v['task_count'] for v in merged['variants'].values()),8910)
            for i,v in enumerate(m.VARIANTS):
                part=merged['variants'][v]
                self.assertEqual(len(part['paired_deltas']),1485)
                self.assertAlmostEqual(part['delta_from_fresh_baseline']['dice'],i*.01,places=14)
                self.assertAlmostEqual(part['delta_from_historical_baseline']['dice'],i*.01,places=14)
            self.assertEqual(verification.call_args_list[0].kwargs,dict(rehash=True))
            with self.assertRaises(FileExistsError):m.merge(d)

    def test_class_macro_and_fsum_do_not_follow_task_weighted_or_naive_sum(self):
        import math
        m=self.m;rows=[self.row(),self.row(dict(self.task,case_id='0002')),self.row(dict(self.task,case_id='0003',global_class_id=202))]
        rows[0].update(dice=0.,pred_gt_ratio=1e16);rows[1].update(dice=0.,pred_gt_ratio=1.);rows[2].update(dice=1.,pred_gt_ratio=1.)
        value=m.summarize(rows)
        self.assertEqual(value['mean']['dice'],1/3);self.assertEqual(value['class_macro_dice'],.5)
        self.assertEqual(value['mean']['pred_gt_ratio'],math.fsum([1e16,1.,1.])/3)

    def test_recursive_sources_include_executed_package_initializers(self):
        m=self.m;d=Path(tempfile.mkdtemp())
        sources={'tools/run.py':'from utils.child import value\n',
                 'tools/__init__.py':'import tools.startup\n','tools/startup.py':'pass\n',
                 'utils/child.py':'value=1\n','utils/__init__.py':'import utils.init_dependency\n',
                 'utils/init_dependency.py':'pass\n',m.NATIVE:'pass\n',
                 'v10/__init__.py':'import v10.startup\n','v10/startup.py':'pass\n','v10/infer/__init__.py':'pass\n'}
        for name,source in sources.items():
            p=d/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(source)
        with patch.object(m,'ROOT',d),patch.object(m,'SOURCE_SEEDS',('tools/run.py',m.NATIVE)):
            found={str(p.relative_to(d)).replace('\\','/') for p in m.dependency_sources()}
        self.assertTrue(set(sources)<=found,set(sources)-found)

    def test_prompt_cache_reference_must_equal_actual_native_ct_cache_path(self):
        m=self.m;d=Path(tempfile.mkdtemp());tasks=self.tasks();refs=[]
        for t in tasks:
            t['image']=f'data/{t["case_id"]}/ct';t['label']=f'data/{t["case_id"]}/gt'
            refs.append(dict(t,prompt_cache_name=t['case_id']+'.npz'))
        class Package:
            def __enter__(self):return {'case_image_name':types.SimpleNamespace(item=lambda:'stale.npy')}
            def __exit__(self,*args):pass
        import numpy as np
        with patch.object(m,'ROOT',d),patch.object(np,'load',return_value=Package()),\
             patch.object(m,'native_case_cache_path',return_value=d/m.CACHE/'_case_images_v1/actual.npy',create=True):
            with self.assertRaisesRegex(ValueError,'native.*cache|cache.*native'):m.data_and_cache_inventory(tasks,refs)

    def test_actual_ct_cache_access_binds_image_path_and_frozen_hit_or_ram_miss(self):
        m=self.m;d=Path(tempfile.mkdtemp());task=dict(self.task,image='data/case/ct')
        entry=dict(path='cache/case.npy',image='data/case/ct',present=True)
        good=dict(ct_cache_access=[dict(path=str(d/'cache/case.npy'),image=str(d/'data/case/ct'),mode='read_only_hit')])
        with patch.object(m,'ROOT',d):
            self.assertTrue(hasattr(m,'validate_ct_cache_access'),'actual CT access must be validated against frozen native path')
            m.validate_ct_cache_access(good,task,entry)
            for changes in (dict(path=str(d/'cache/other.npy')),dict(image=str(d/'data/other/ct')),dict(mode='current_case_ram_miss')):
                bad=copy.deepcopy(good);bad['ct_cache_access'][0].update(changes)
                with self.assertRaises(ValueError):m.validate_ct_cache_access(bad,task,entry)
            absent=dict(entry,present=False);miss=copy.deepcopy(good);miss['ct_cache_access'][0]['mode']='current_case_ram_miss'
            m.validate_ct_cache_access(miss,task,absent)
            with self.assertRaises(ValueError):m.validate_ct_cache_access(dict(ct_cache_access=[]),task,entry)

    def test_native_ct_path_uses_real_amos_source_signature_without_model_import(self):
        import hashlib
        import os
        m=self.m;d=Path(tempfile.mkdtemp());image=d/'ct_source';image.write_bytes(b'fixture source')
        path=m.native_case_cache_path(image,d/'cache')
        stat=image.stat();signature=(os.path.abspath(image),int(stat.st_mtime_ns),int(stat.st_size))
        expected={'version':1,'dataset':'amos_ct','format':'canonical_axial_first_raw_f32_npy','image':signature}
        name=hashlib.sha1(repr(sorted(expected.items())).encode('utf-8')).hexdigest()+'.npy'
        self.assertEqual(path,d/'cache/_case_images_v1'/name)
        self.assertEqual(m.native_case_cache_path(image,d/'cache'),path)
        image.write_bytes(b'changed fixture source')
        self.assertNotEqual(m.native_case_cache_path(image,d/'cache'),path)


class FakeQueue:
    def __init__(self,m,jobs,codes):self.m=m;self.jobs=jobs;self.codes=list(codes);self.done={};self.started=[];self.clock=0;self.states=[];self.polls=0
    def records(self):return self.done
    def available(self):return {g:dict(free=65000,uuid=f'GPU-{g}') for g in (2,4,5,6)}
    def now(self):return self.clock
    def sleep(self,s):self.clock+=s
    def status(self,state):self.states.append(copy.deepcopy(state))
    def emit(self,*a):pass
    def attempt(self,j,g,a,completed):return dict(j,gpu=g,run=j['name'],spec=j['name'],log=j['name'])
    def spawn(self,j):
        backend=self;code,delay,text=self.codes.pop(0);self.started.append(j);start=self.clock
        class Process:
            pid=len(backend.started);returncode=None
            def poll(self):
                backend.polls+=1
                if backend.clock-start<delay:return None
                self.returncode=code
                if code==0:
                    for t in j['pending']:backend.done[backend.m.key(t)]=dict(task=t)
                return code
        class Handle:
            def close(self):pass
        j['_text']=text
        return Process(),Handle()
    def log(self,j):return j['_text']
    def complete(self,j):pass


class SchedulerTests(unittest.TestCase):
    setUp=ControllerTests.setUp
    job=ControllerTests.job
    def test_fatal_stops_dispatch_and_naturally_drains_live_peers(self):
        m=self.m;jobs=[dict(self.job(),name=f'j{i}',pending=[dict(self.task,case_id=f'{i:04}')]) for i in range(5)]
        b=FakeQueue(m,jobs,[(1,0,'bad prompt'),(0,40,''),(0,40,''),(0,40,'')])
        with self.assertRaises(RuntimeError):m.schedule_queue(jobs,False,b)
        self.assertEqual(len(b.started),4);self.assertGreaterEqual(b.clock,40)
        self.assertEqual(b.states[-1]['running'],{});self.assertTrue(b.states[-1]['failed'])

    def test_oom_retries_independent_attempt_and_only_complete_classes_skip(self):
        m=self.m;j=self.job();b=FakeQueue(m,[j],[(1,0,'CUDA out of memory'),(75,0,'resource'),(0,0,'')])
        m.schedule_queue([j],False,b)
        self.assertEqual([x['frame_batch'] for x in b.started],[4,2,2])
        self.assertEqual([x['attempt'] for x in b.started],[0,1,2]);self.assertEqual(len(b.done),1)


class WorkerTests(unittest.TestCase):
    setUp=ControllerTests.setUp
    row=ControllerTests.row

    def setup_worker(self):
        m=self.m;d=Path(tempfile.mkdtemp());out=d/m.OUT
        tasks=[self.task,dict(self.task,global_class_id=202,local_class_id=2,class_name='kidney')]
        canonical=m.case_jobs(tasks,strict=False)[0];canonical['prepared']=str(out/'prepared'/canonical['name'])
        tag=canonical['name']+'_a0';run=out/'runs'/tag
        job=dict(canonical,gpu=2,gpu_uuid='GPU-2',actual_frame_batch=4,run=str(run),
                 spec=str(out/'job_specs'/(tag+'.json')),log=str(out/'logs'/(tag+'.log')),control_gate_sha256='gate')
        m.atomic_new(out/'jobs.json',dict(cases=[canonical],controls=[]))
        m.atomic_new(out/'protocol.json',dict(native_source_sha256='f'*64,data=dict(shared_ct={'0001':dict(path='cache/0001.npy',image=tasks[0]['image'],present=False)})))
        m.atomic_new(out/'protocol_signature.json',dict(sha256=m.sha(out/'protocol.json')))
        m.atomic_new(out/'refs.json',dict(per_case_class=[dict(t,prompt_cache_name=f'{i}.npz') for i,t in enumerate(tasks)]))
        m.atomic_new(job['spec'],job);Path(job['log']).parent.mkdir(parents=True);Path(job['log']).write_text('final worker log')
        torch=types.SimpleNamespace(cuda=types.SimpleNamespace(get_device_properties=lambda _:types.SimpleNamespace(uuid='GPU-2')),
                                    ones=lambda *a,**kw:types.SimpleNamespace(item=lambda:1),set_num_threads=lambda n:None)
        return d,out,job,tasks,torch

    def fake_runtime(self,job,fail_at=None):
        restoration=dict(model='v10_2',epoch=490,weight_mode='live',prompt_parameters_exact=418,decoder_parameters_exact=212,guards_passed=True,model_load_count=1)
        calls=[];session=types.SimpleNamespace(input_audit={'actual_semantics':'legacy','source_sha256':'f'*64},
            finish=lambda:dict(restoration,initial_restoration=restoration,successful_native_completion=True,full_tensor_comparison_phase='normal_worker_completion'))
        def initialize(j,paths):calls.append(('initialize',j['case_id']));return session
        def evaluate(s,task,variant):
            calls.append((task['global_class_id'],variant))
            if fail_at is not None and len(calls)==fail_at:raise RuntimeError('CUDA out of memory')
            return dict(variant=variant,row=self.row(task),actual_frame_batch=job['frame_batch'],
                        restoration=restoration,runtime_input_audit=session.input_audit,native_source_sha256='f'*64,
                        resource_binding=dict(gpu=2,gpu_uuid='GPU-2'),stages=[{}],
                        ct_cache_access=[dict(path=str(self.m.ROOT/'cache/0001.npy'),image=str(self.m.ROOT/task['image']),mode='current_case_ram_miss')])
        return types.SimpleNamespace(initialize=initialize,evaluate=evaluate),calls

    def patches(self,d,torch,fake):
        m=self.m
        return (patch.object(m,'ROOT',d),patch.object(m,'verify',return_value={'native_source_sha256':'f'*64}),
                patch.object(m,'check_control_gate'),
                patch.object(m,'gpu_snapshot',return_value={2:dict(uuid='GPU-2',free=65000)}),
                patch.dict(sys.modules,{'torch':torch}),patch.object(m,'runtime',return_value=fake),patch.object(m,'validate_record'))

    def test_case_worker_initializes_model_once_and_commits_six_variants_per_class(self):
        from contextlib import ExitStack
        m=self.m;d,out,job,tasks,torch=self.setup_worker();fake,calls=self.fake_runtime(job)
        # Use a real gate file and its real digest so no hash behavior is mocked.
        m.atomic_new(out/'control_verification.json',{'independent':'passed'});job['control_gate_sha256']=m.sha(out/'control_verification.json')
        Path(job['spec']).write_text(json.dumps(job))
        with ExitStack() as stack:
            for p in self.patches(d,torch,fake):stack.enter_context(p)
            m.worker(job)
            records=m.class_records(out);self.assertEqual(len(records),2)
            proof=m.complete_case(out,job);self.assertEqual(len(proof['markers']),2)
            m.verify_case_success(out,m.read(out/'jobs.json')['cases'][0],records)
            Path(job['log']).write_text('tampered')
            with self.assertRaises(ValueError):m.verify_case_success(out,m.read(out/'jobs.json')['cases'][0],records)
        self.assertEqual(sum(c[0]=='initialize' for c in calls),1)
        self.assertEqual(calls[1:7],[(201,v) for v in m.VARIANTS])
        self.assertEqual(len(list((Path(job['run'])/'classes').glob('*.done.json'))),2)

    def test_failure_in_later_class_retains_only_complete_first_class(self):
        from contextlib import ExitStack
        m=self.m;d,out,job,tasks,torch=self.setup_worker();fake,calls=self.fake_runtime(job,fail_at=10)
        m.atomic_new(out/'control_verification.json',{'independent':'passed'});job['control_gate_sha256']=m.sha(out/'control_verification.json')
        Path(job['spec']).write_text(json.dumps(job))
        with ExitStack() as stack:
            for p in self.patches(d,torch,fake):stack.enter_context(p)
            with self.assertRaisesRegex(RuntimeError,'CUDA out of memory'):m.worker(job)
            records=m.class_records(out);self.assertEqual(set(records),{m.key(tasks[0])})
        run=Path(job['run']);self.assertFalse((run/'native_worker_complete.json').exists())
        self.assertFalse((run/'classes/0001__202.done.json').exists())

    def test_worker_complete_guard_binds_exact_successful_spec_and_protocol(self):
        m=self.m;d,out,job,tasks,torch=self.setup_worker();run=Path(job['run'])
        m.atomic_new(run/'native_worker_complete.json',dict(native_worker_complete=True,classes=2,variants=6,
            spec_sha256='wrong',protocol_sha256=m.sha(out/'protocol.json')))
        with self.assertRaises(ValueError):m.verify_worker_complete(job)

    def test_successful_retry_binds_prior_complete_class_and_its_finished_attempt_log(self):
        from contextlib import ExitStack
        m=self.m;d,out,job,tasks,torch=self.setup_worker();fake,calls=self.fake_runtime(job,fail_at=10)
        m.atomic_new(out/'control_verification.json',{'independent':'passed'});job['control_gate_sha256']=m.sha(out/'control_verification.json')
        Path(job['spec']).write_text(json.dumps(job))
        with ExitStack() as stack:
            for p in self.patches(d,torch,fake):stack.enter_context(p)
            with self.assertRaises(RuntimeError):m.worker(job)
        tag=job['name']+'_a1';retry=dict(job,attempt=1,oom_retries=1,frame_batch=2,actual_frame_batch=2,
                    pending=[tasks[1]],task_count=1,run=str(out/'runs'/tag),spec=str(out/'job_specs'/(tag+'.json')),log=str(out/'logs'/(tag+'.log')))
        m.atomic_new(retry['spec'],retry);Path(retry['log']).write_text('successful later attempt')
        fake,calls=self.fake_runtime(retry)
        with ExitStack() as stack:
            for p in self.patches(d,torch,fake):stack.enter_context(p)
            m.worker(retry);records=m.class_records(out);self.assertEqual(len(records),2)
            proof=m.complete_case(out,retry)
            self.assertEqual(len(proof['contributing_attempts']),2)
            canonical=m.read(out/'jobs.json')['cases'][0];m.verify_case_success(out,canonical,records)
            Path(job['log']).write_text('changed failed attempt log')
            with self.assertRaises(ValueError):m.verify_case_success(out,canonical,records)


if __name__=='__main__':unittest.main()
