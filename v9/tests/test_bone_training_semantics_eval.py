"""Fixed-floor runner contracts and real dispatch/drain behavior (CPU only)."""
import copy
import importlib
import json
import math
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((Path(__file__).resolve().parents[1]/'tools/bone_training_semantics_eval.py').is_file(),
                        'new independent training semantic runner is required')
        self.m=importlib.import_module('tools.bone_training_semantics_eval')
        self.task=dict(source_name='bone',case_id='0001',global_class_id=10001,
                       local_class_id=1,class_name='rib_right_4')

    def tasks(self):
        return [dict(self.task,case_id=f'{c:04}',global_class_id=10001+i,local_class_id=1+i)
                for c in range(20) for i in range(47 if c<10 else 30)]

    def test_full_split_prepare_and_verify_use_real_alignment_rows(self):
        from v9_2_extended_protocol import class_metadata
        d=Path(tempfile.mkdtemp(prefix='bone_semantic_prepare_'));m=self.m;old=d/m.OLD;out=d/m.OUT
        tasks=self.tasks();bones=[];training=[]
        for i in range(47):
            name=f'rib_right_{i+1}'
            bones.append(dict(dataset='bone',global_class_id=10001+i,local_class_id=i+1,label_value=i+1,
                              class_name=name,semantic_text=f'right rib {i+1}',**class_metadata('bone',name)))
            training.append(dict(dataset='totalseg',global_class_id=i+1,local_class_id=i+1,
                                 class_name=name,semantic_text=name,**class_metadata('totalseg',name)))
        for t in tasks:t['class_name']=bones[t['local_class_id']-1]['class_name']
        split=dict(class_catalog=training+bones,source_order=['totalseg','bone'],datasets={'bone':{'val':[]}})
        values={'manifest.json':dict(tasks=tasks,expected_task_count=770),'refs.json':dict(per_case_class=tasks),
                'prompt_budget_audit.json':dict(tasks=770),'merged_metrics.json':dict(reference_only=True),
                'bone_split.json':split,'jobs.json':[dict(smoke=True,pending=[dict(task=tasks[0],floors=[.35])])]}
        for name,value in values.items():m.atomic_new(old/name,value)
        m.atomic_new(d/m.TRAINING,dict(class_catalog=training,other_training_metadata='preserve'))
        hashes={n:m.sha(old/n) for n in m.REFERENCE_HASHES}
        fake=types.SimpleNamespace(b=types.SimpleNamespace(),g=types.SimpleNamespace(amos=types.SimpleNamespace(),
                    dataset_signature=lambda **kw:{},bone_payload=lambda rows:dict(tasks=rows,expected_task_count=len(rows))))
        with patch.object(m,'ROOT',d),patch.object(m,'REFERENCE_HASHES',hashes),patch.object(m,'TRAINING_SHA',m.sha(d/m.TRAINING)),\
             patch.object(m,'engine',return_value=fake),patch.object(m,'training_checkpoint_audit',return_value={}),\
             patch.object(m,'verify_old_reference',return_value={'native_chain_verified':True}),\
             patch.object(m,'snapshot',return_value={'fake_frozen_assets':True}):
            m.prepare(out,already_checked=True);m.verify(out)
            aligned=[r for r in m.read(out/'bone_split.json')['class_catalog'] if r['dataset']=='bone']
            self.assertEqual(len(aligned),47)
            self.assertTrue(all(r['semantic_text']==r['class_name']+'; compact target morphology' for r in aligned))
            self.assertEqual(len(m.read(out/'jobs.json')['cases']),40)
            self.assertEqual(fake.b.SPLIT,str(Path(m.OUT)/'bone_split.json'))
            self.assertIs(fake.g.amos.verify,m.verify)

    def test_run_owns_fresh_pipeline_and_orders_controls_before_full_then_once_merge(self):
        self.assertTrue(hasattr(self.m,'run_pipeline'),'run must own exclusive pipeline creation')
        d=Path(tempfile.mkdtemp(prefix='bone_semantic_pipeline_'));m=self.m;events=[]
        with patch.object(m,'ROOT',d),patch.object(m,'preflight',side_effect=lambda out,**kw:events.append(('preflight',kw))),\
             patch.object(m,'prepare',side_effect=lambda *a,**kw:events.append(('prepare',kw))),\
             patch.object(m,'schedule',side_effect=lambda out,c:events.append(('schedule',c))),\
             patch.object(m,'check_controls',side_effect=lambda out:events.append(('controls_verified',True))),\
             patch.object(m,'merge',side_effect=lambda out:events.append(('merge',True))):
            m.run_pipeline(d/m.OUT)
            self.assertEqual([e[0] for e in events],['preflight','prepare','schedule','controls_verified','schedule','merge'])
            self.assertEqual(events[0][1]['pipeline'],d/m.PIPELINE)
            self.assertEqual([e[1] for e in events if e[0]=='schedule'],[True,False])
            with self.assertRaises(FileExistsError):m.run_pipeline(d/m.OUT)

    def test_control_marker_validates_real_semantic_consumption_with_control_phase(self):
        from tools import bone_training_semantics_runtime as runtime
        from tests.test_bone_training_semantics_runtime import FakePrompt,prior_init,prior_begin
        from v9_2_extended_protocol import class_metadata
        m=self.m;e=m.engine();d=Path(tempfile.mkdtemp(prefix='bone_semantic_control_'));out=d/m.OUT
        j=self.job('semantic_control_v9_2')|dict(control=True,prepared=str(out/'prepared/control'))
        run=out/'runs'/(j['name']+'_a0');marker=run/'classes/0001__10001.done.json'
        catalog=[dict(dataset='bone',global_class_id=10001,local_class_id=1,label_value=1,class_name='rib_right_4',
                      semantic_base_text='rib_right_4',semantic_text='rib_right_4; compact target morphology',**class_metadata('bone','rib_right_4'))]
        stage=dict(key=list(m.p.key(self.task)),floor=.35,frames=2,input_size=1024,method='prompt_probability_midpoint',
                   context=192,memory='native_v92_no_cross_frame_memory',threshold=.6,source='prompt_logits',raw_key='raw0')
        row=self.task|{n:.5 for n in m.p.METRICS}|dict(label_id=1,mask_threshold_used=.6,dynamic_threshold_source='prompt_logits')
        restoration=dict(model='v9_2',weight_mode='live',guards_passed=True,prompt_parameters_exact=438,decoder_parameters_exact=192,epoch=360)
        record=dict(model='v9_2',task=self.task,results=[dict(floor=.35,row=row,stages=[stage])],restoration=restoration,
                    training_audit={'training_dataset_constructions_skipped':1},raw_producers={'raw0':dict(raw_key='raw0',frames=2)},
                    metric_execution=dict(provider_sha256=m.sha(Path(m.__file__).parent/'bone_fast_metrics.py'),
                        stats={'comparison_failures':0},events=[dict(key=list(m.p.key(self.task)),floor=.35,metrics={n:.5 for n in m.p.METRICS},native_compared=True,max_abs_error=0.)]))
        runtime.configure_semantics(catalog,control=True)
        with patch.object(runtime.PriorSession,'__init__',prior_init),patch.object(runtime.PriorSession,'begin',prior_begin):
            session=runtime.GridSession('v9_2',j['pending'],'frozen',lambda r:None)
            session.begin(self.task);session.floor=.35;session.stage_rows={.35:[]}
            prompt=FakePrompt();runtime._wrap_prompt(session,prompt);prompt.forward(catalog[0]['semantic_text']);session.on_class(record)
        spec=j|dict(gpu=2,gpu_uuid='GPU-123',run=str(run),spec=str(out/'job_specs'/(run.name+'.json')),log=str(out/'logs'/(run.name+'.log')))
        prepared_hashes={}
        files={out/'jobs.json':dict(cases=[],controls=[j]),out/'bone_split.json':dict(class_catalog=catalog),out/'protocol.json':dict(frozen=True),
               out/'refs.json':dict(per_case_class=[self.task]),Path(spec['spec']):spec,
               run/'restoration_start.json':restoration,run/'resource_policy_audit.json':{k:spec[k] for k in ('gpu','gpu_uuid','frame_batch','minimum_free_mib','parent_history')}|dict(actual_start_free_mib=60000,new_oom_retries=0),
               run/'prepared/manifest.json':dict(tasks=[self.task],expected_task_count=1),run/'prepared/refs.json':dict(per_case_class=[self.task])}
        for path,value in files.items():m.atomic_new(path,value)
        prepared_hashes={str(run/'prepared'/n):m.sha(run/'prepared'/n) for n in ('manifest.json','refs.json')}
        m.atomic_new(run/'prepared_input_audit.json',dict(canonical_prepared=j['prepared'],files=prepared_hashes))
        record_path=marker.with_name('0001__10001.json');m.atomic_new(record_path,record)
        m.atomic_new(marker,dict(job_name=j['name'],record=str(record_path),protocol_sha=m.sha(out/'protocol.json'),
                                files={str(p):m.sha(p) for p in m.marker_paths(out,run,marker)}))
        with patch.object(m,'ROOT',d),patch.object(m,'engine',return_value=e):
            self.assertEqual(len(m.class_records(out,True)),1);self.assertEqual(m.class_records(out,False),{})
            m.validate_class(record,j['pending'][0],control=True)
            with self.assertRaises(ValueError):m.validate_class(record,j['pending'][0],control=False)
            bad=copy.deepcopy(record);bad['semantic_execution']['events'][0]['semantic_text']='right rib 4'
            with self.assertRaises(ValueError):m.validate_class(bad,j['pending'][0],control=True)

    def job(self,name='case',model='v9_2'):
        return self.m.case_jobs([self.task],{},strict=False)[0] | dict(name=name,model=model)

    def test_fresh_fixed_groups_keep_all_original_keys(self):
        tasks=self.tasks();before=copy.deepcopy(tasks);jobs=self.m.case_jobs(tasks,{})
        self.assertEqual(len(jobs),40);self.assertEqual(tasks,before)
        for model,floor in [('v9_2',.35),('v10_2',.75)]:
            selected=[j for j in jobs if j['model']==model]
            rows=[x for j in selected for x in j['pending']]
            self.assertEqual(len(rows),770)
            self.assertEqual({self.m.p.key(x['task']) for x in rows},{self.m.p.key(t) for t in tasks})
            self.assertTrue(all(x['floors']==[floor] for x in rows))
        self.assertTrue(all(j['frame_batch']==2 and not j['control'] for j in jobs))

    def test_duplicate_and_lost_scope_rejected(self):
        for tasks in [self.tasks()[:-1],self.tasks()+[self.task],[]]:
            with self.assertRaises(ValueError):self.m.case_jobs(tasks,{})

    def test_history_only_restricts_resources_no_prediction_reuse(self):
        h={('v9_2','0001'):dict(frame_batch=1,minimum_free_mib=45000,last_failed_gpu=2,avoid_gpu_until=900,not_before=60)}
        jobs=self.m.case_jobs([self.task],h,strict=False)
        j=next(j for j in jobs if j['model']=='v9_2')
        self.assertEqual((j['frame_batch'],j['minimum_free_mib'],j['avoid_gpus']), (1,45000,{'2':900}))
        self.assertEqual(len(j['pending']),1)
        self.assertNotIn('results',j);self.assertNotIn('completed',j)

    def test_two_controls_have_original_smoke_key_and_no_final_rows(self):
        controls=self.m.control_jobs(self.tasks(),self.m.p.key(self.tasks()[0]))
        self.assertEqual(len(controls),2)
        self.assertEqual({j['model'] for j in controls},{'v9_2','v10_2'})
        self.assertTrue(all(j['control'] and j['task_count']==1 and j['frame_batch']==2 for j in controls))
        self.assertEqual([j['pending'][0]['floors'] for j in controls],[[.35],[.75]])

    def test_resource_retries_are_75_same_batch_and_true_oom_2_1_wait(self):
        j=self.job()|dict(gpu=2)
        same=self.m.retry(j,75,'resource unavailable',100)
        self.assertEqual((same['frame_batch'],same['oom_retries']),(2,0))
        one=self.m.retry(j,1,'CUDA out of memory',100)
        self.assertEqual((one['frame_batch'],one['oom_retries']),(1,1))
        wait=self.m.retry(one,1,'torch.OutOfMemoryError',150)
        self.assertEqual((wait['frame_batch'],wait['minimum_free_mib'],wait['not_before'],wait['avoid_gpus']['2']), (1,45000,210,750))
        self.assertFalse(self.m.eligible(wait,2,{2:dict(free=70000)},300))
        self.assertIsNone(self.m.retry(j,1,'bad semantics',100))
        self.assertEqual(self.m.attempt_outcome(1,'bad semantics',False),'fatal')

    def test_merge_gate_requires_drained_success(self):
        state=dict(phase='full',pending=0,running={},failed=[])
        self.m.require_merge_state(state,[])
        for bad,live in [(state,[1]),(state|dict(failed=[{}]),[]),(state|dict(pending=1),[]),(state|dict(phase='controls'),[])]:
            with self.assertRaises(ValueError):self.m.require_merge_state(bad,live)

    def test_frozen_constants_and_modes(self):
        self.assertEqual(self.m.OUT,'output/bone_v92e360_v102e490_training_semantics_bestfloor_20261029')
        self.assertEqual(self.m.TMUX,'bone_v92_v102_training_semantics')
        self.assertEqual(self.m.ALLOWED,(2,4,5,6))
        self.assertEqual(self.m.TRAINING_SHA,'99b2eee8b4ae549537922614b0570665c6fa90387e21e04c69c04ed0652de022')
        self.assertEqual(set(self.m.REFERENCE_HASHES),{'manifest.json','refs.json','prompt_budget_audit.json','merged_metrics.json'})
        for mode in ('prepare','run','worker','status','merge'):
            self.assertEqual(self.m.parser().parse_args([mode]).mode,mode)

    def test_exclusive_atomic_publication_keeps_first_result(self):
        d=Path(tempfile.mkdtemp(prefix='bone_semantic_runner_'));f=d/'record.json'
        self.m.atomic_new(f,dict(value=1))
        with self.assertRaises(FileExistsError):self.m.atomic_new(f,dict(value=2))
        self.assertEqual(json.loads(f.read_text())['value'],1)

    def test_preflight_refuses_output_live_worker_tmux_and_pipeline(self):
        d=Path(tempfile.mkdtemp(prefix='bone_semantic_preflight_'));out=d/'new';pipeline=d/'pipeline'
        self.m.preflight(out,processes=[],tmux_exists=False,pipeline=pipeline)
        for live,tmux in [([dict(pid=7)],False),([],True)]:
            with self.assertRaises(ValueError):self.m.preflight(out,processes=live,tmux_exists=tmux,pipeline=pipeline)
        out.mkdir()
        with self.assertRaises(FileExistsError):self.m.preflight(out,processes=[],tmux_exists=False,pipeline=pipeline)

    def test_signature_requires_training_reference_source_weight_data_and_prompt_hashes(self):
        m=self.m
        with patch.object(m,'snapshot',return_value=dict(sources={'a':'1'},weights={'b':'2'},dataset={'c':'3'},prompts={'d':'4'},training={'sha256':m.TRAINING_SHA},reference=m.REFERENCE_HASHES)):
            d=Path(tempfile.mkdtemp(prefix='bone_semantic_signature_'))
            for name in m.PREPARED_ASSETS:self.m.atomic_new(d/name,{'test':name})
            s=m.signature(d)
            self.assertEqual(set(s['frozen']),{'sources','weights','dataset','prompts','training','reference'})
            m.atomic_new(d/'protocol.json',s)
            with patch.object(m,'validate_prepared'),patch.object(m,'bind_context'):
                m.verify(d)
                (d/'jobs.json').write_text('{}')
                with self.assertRaises(ValueError):m.verify(d)

    def test_source_inventory_covers_recursive_sam2_config_and_imported_native_chain(self):
        self.assertTrue(hasattr(self.m,'source_paths'),'complete native source/config inventory required')
        d=Path(tempfile.mkdtemp(prefix='bone_semantic_sources_'))
        names=['sam2/sam2/modeling/backbones/image.py','sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml',
               'tools/bone_floor_small_batch_recovery.py','tools/bone_gpu_postdrain_grid.py']
        for name in names:
            f=d/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_text('test')
        with patch.object(self.m,'ROOT',d):
            paths=self.m.source_paths()
            self.assertTrue(all(d/n in paths for n in names))
        with patch.object(self.m,'ROOT',Path(__file__).resolve().parents[1]):
            self.assertIn(Path(self.m.__file__),self.m.source_paths())

    def test_only_selected_case_cache_paths_are_frozen_absence_and_changes_detected(self):
        self.assertTrue(hasattr(self.m,'case_cache_signature'),'selected shared CT cache provenance is required')
        d=Path(tempfile.mkdtemp(prefix='bone_semantic_ctcache_'));tasks=self.tasks();refs=[t|dict(prompt_cache_name=str(i)+'.npz') for i,t in enumerate(tasks)]
        names={str(i)+'.npz':t['case_id']+'.npy' for i,t in enumerate(tasks)}
        reader=lambda path:names[path.name]
        resolver=lambda image,root:{model:root/'_case_images_v1'/(Path(image).name+'.npy') for model in self.m.FLOORS}
        for t in tasks:t['image']=t['case_id']
        result=self.m.case_cache_signature(tasks,refs,d,reader=reader,path_resolver=resolver)
        self.assertEqual(len(result),20);self.assertTrue(all(v['state']=='absent' for v in result.values()))
        selected=d/'_case_images_v1'/names['0.npz'];selected.parent.mkdir();selected.write_bytes(b'CT fixture')
        updated=self.m.case_cache_signature(tasks,refs,d,reader=reader,path_resolver=resolver)
        self.assertEqual(updated['0000']['sha256'],self.m.sha(selected));self.assertNotEqual(updated,result)
        (d/'_case_images_v1/unrelated.npy').write_bytes(b'irrelevant')
        self.assertEqual(updated,self.m.case_cache_signature(tasks,refs,d,reader=reader,path_resolver=resolver))
        names['1.npz']='another.npy'
        with self.assertRaises(ValueError):self.m.case_cache_signature(tasks,refs,d,reader=reader,path_resolver=resolver)

    def test_native_cache_resolver_executes_both_actual_bone_dataset_methods(self):
        self.assertTrue(hasattr(self.m,'native_case_cache_paths'),'actual native CT cache-path methods required')
        d=Path(tempfile.mkdtemp(prefix='bone_semantic_nativecache_'));image=d/'image.nii.gz';image.write_bytes(b'CT metadata fixture')
        paths=self.m.native_case_cache_paths(image,d)
        self.assertEqual(set(paths),{'v9_2','v10_2'});self.assertEqual(paths['v9_2'],paths['v10_2'])
        self.assertEqual(paths['v9_2'].parent,d/'_case_images_v1');self.assertEqual(paths['v9_2'].suffix,'.npy')
        image.write_bytes(b'changed source signature')
        self.assertNotEqual(paths,self.m.native_case_cache_paths(image,d))

    def test_native_engine_uses_new_runtime_configured_before_session(self):
        e=self.m.engine()
        self.assertEqual(e.OUT,self.m.OUT);self.assertIs(e.verify,self.m.verify)
        self.assertIs(e.class_records,self.m.class_records)
        names=e.native_run.__code__.co_names
        self.assertIn('tools.bone_training_semantics_runtime',names)
        self.assertIn('configure_semantics',names)
        self.assertNotIn('prepare',names);self.assertNotIn('finish_parent',names)
        self.assertIs(e.g.amos.verify,self.m.verify)

    def test_marker_requires_record_spec_restoration_resource_prepared_and_protocol(self):
        d=Path(tempfile.mkdtemp(prefix='bone_semantic_marker_'));j=self.job();run=d/'runs'/(j['name']+'_a0')
        marker=run/'classes/x.done.json';files=self.m.marker_paths(d,run,marker)
        self.assertEqual(len(files),8)
        info=dict(record=str(marker.with_name('x.json')),files={str(f):'sha' for f in files})
        self.m.require_marker_files(info,marker,d,j)
        for path in files:
            bad=copy.deepcopy(info);del bad['files'][str(path)]
            with self.assertRaises(ValueError):self.m.require_marker_files(bad,marker,d,j)

    def test_summary_fsum_macro_fallback_and_original_ids(self):
        rows=[]
        for case,gid,value in [('a',10001,.1),('b',10001,.9),('a',10002,.8)]:
            r=dict(self.task,case_id=case,global_class_id=gid,local_class_id=gid-10000,label_id=gid-10000,
                   mask_threshold_used=.6,dynamic_threshold_source='fallback_missing_prompt_logits')
            r.update({m:value for m in self.m.p.METRICS});rows.append(r)
        s=self.m.summarize(rows)
        self.assertEqual(s['task_count'],3);self.assertEqual(s['fallback_count'],3)
        self.assertEqual(s['mean']['dice'],math.fsum([.1,.9,.8])/3)
        self.assertEqual(s['class_macro_dice'],.65)
        self.assertEqual(s['per_case_class'],sorted(rows,key=self.m.p.key))

    def test_real_merge_1540_fresh_rows_excludes_controls_and_computes_paired_ten_deltas_once(self):
        m=self.m;d=Path(tempfile.mkdtemp(prefix='bone_semantic_merge_'));out=d/m.OUT;tasks=self.tasks();records={};old={};nn=[]
        for model,floor in m.FLOORS.items():
            old_rows=[]
            for i,t in enumerate(tasks):
                value=(i%13+1)/20
                row=t|{name:value for name in m.p.METRICS}|dict(label_id=t['local_class_id'],mask_threshold_used=.6,
                      dynamic_threshold_source='fallback_missing_prompt_logits' if i%2 else 'prompt_logits')
                records[(model,m.p.key(t))]=dict(record=dict(model=model,task=t,results=[dict(floor=floor,row=row)]),spec={'frame_batch':1 if i%3 else 2})
                old_rows.append(row|{name:value-.05 for name in m.p.METRICS})
                if model=='v9_2':nn.append(row|{name:value-.10 for name in m.p.METRICS})
            old[model]=dict(floors={format(floor,'.2f'):dict(per_case_class=old_rows)})
        m.atomic_new(out/'status.json',dict(phase='full',pending=0,running={},failed=[]))
        m.atomic_new(out/'manifest.json',dict(tasks=tasks));m.atomic_new(out/'protocol.json',{'signed_fixture':True})
        m.atomic_new(d/m.OLD/'merged_metrics.json',dict(models=old));m.atomic_new(d/m.NN_REFERENCE,dict(per_case_class=nn))
        with patch.object(m,'ROOT',d),patch.object(m,'related',return_value=[]),patch.object(m,'verify'),\
             patch.object(m,'snapshot'),patch.object(m,'check_controls',return_value={'two_controls':True}),\
             patch.object(m,'class_records',return_value=records):
            result=m.merge(out)
            self.assertEqual((result['rows'],result['predictions']),(1540,1540));self.assertFalse(result['controls_in_final'])
            self.assertEqual(m.read(out/'merged_metrics.json'),result)
            for model in m.FLOORS:
                summary=result['models'][model];rows=summary['per_case_class'];expected=m.summarize(rows)
                self.assertEqual(summary['mean'],expected['mean']);self.assertEqual(summary['fallback_count'],385)
                self.assertEqual(summary['class_macro_dice'],expected['class_macro_dice'])
                self.assertEqual(set(summary['delta_from_old_same_floor']),set(m.p.METRICS))
                for name in m.p.METRICS:
                    self.assertAlmostEqual(summary['delta_from_old_same_floor'][name],.05)
                    self.assertAlmostEqual(summary['delta_from_nninteractive'][name],.10)
            with self.assertRaises(FileExistsError):m.merge(out)

    def test_control_gate_checks_actual_native_metrics_and_batch_twins_no_old_equality(self):
        self.m.validate_native_twins({'native_batch_probe':{str(b):dict(exact_same_batch=True,raw_sha='a',native_raw_sha='a',encoder_shas=['b'],native_encoder_shas=['b']) for b in (1,2,4)}})
        for bad in [{}, {'native_batch_probe':{'1':{}}}]:
            with self.assertRaises(ValueError):self.m.validate_native_twins(bad)

    def test_checkpoint_audit_uses_actual_epoch_sat_modality_and_portable_catalog_args(self):
        self.assertTrue(hasattr(self.m,'checkpoint_metadata'),'actual checkpoint argument audit is required')
        args=dict(multidataset_split_json='/server/v9/'+self.m.TRAINING,prompt_semantic_dim=128,
                  prompt_semantic_template='{}',v9_1_sat_modality='ct',v9_1_sat_base_model='/server/v9/models/biolord-2023-c')
        value=self.m.checkpoint_metadata('v9_2',dict(epoch=360,args=args))
        self.assertEqual(value['catalog_args']['multidataset_split_json'],self.m.TRAINING)
        self.assertEqual(value['semantic_args']['v9_1_sat_base_model'],'models/biolord-2023-c')
        for bad in [args|dict(prompt_semantic_dim=64),args|dict(v9_1_sat_modality='mri'),
                    args|dict(prompt_semantic_enabled=False),args|dict(multidataset_split_json='wrong')]:
            with self.assertRaises(ValueError):self.m.checkpoint_metadata('v9_2',dict(epoch=360,args=bad))
        with self.assertRaises(ValueError):self.m.checkpoint_metadata('v9_2',dict(epoch=359,args=args))

    def test_attempt_paths_and_resource_cap_are_checked_before_launch(self):
        d=Path(tempfile.mkdtemp(prefix='bone_semantic_attempt_'));j=self.job()|dict(prepared='canonical')
        attempt=j|dict(gpu=2,gpu_uuid='GPU-123',run=str(d/'runs'/(j['name']+'_a0')),
                       spec=str(d/'job_specs'/(j['name']+'_a0.json')),log=str(d/'logs'/(j['name']+'_a0.log')))
        self.m.validate_attempt(attempt,j,d)
        for bad in [attempt|dict(gpu=0),attempt|dict(frame_batch=4),attempt|dict(run=str(d/'escape')),
                    attempt|dict(prepared='other'),attempt|dict(minimum_free_mib=1)]:
            with self.assertRaises(ValueError):self.m.validate_attempt(bad,j,d)


class FakeBackend:
    def __init__(self,jobs,*,codes=None,failure=None):
        self.jobs=jobs;self.codes=codes or {};self.failure=failure;self.started=[];self.polls={};self.closed=[];self.states=[];self.clock=0;self.done={};self.log_reads=0
    def records(self):
        if self.failure=='records' and self.started:raise ValueError('bad marker')
        return self.done
    def available(self):return {g:dict(free=70000,uuid='GPU-'+str(g)) for g in (2,4,5,6)}
    def now(self):return self.clock
    def sleep(self,_):self.clock+=60
    def attempt(self,j,g,a,completed):return j|dict(gpu=g,gpu_uuid=a[g]['uuid'],log=j['name'],spec=j['name'],run=j['name'])
    def spawn(self,j):
        b=self;self.started.append(j);name=j['name'];self.polls[name]=0
        class Proc:
            pid=len(b.started);returncode=None
            def poll(self):
                b.polls[name]+=1
                if b.polls[name]<2:return None
                self.returncode=b.codes.get(name,0)
                if self.returncode==0:
                    for item in j['pending']:b.done[(j['model'],b.key(item['task']))]={'record':{}}
                return self.returncode
        return Proc(),types.SimpleNamespace(close=lambda:self.closed.append(name))
    def log(self,j):
        self.log_reads+=1
        if self.failure=='log':raise OSError('cannot read worker log')
        return 'CUDA out of memory' if self.codes.get(j['name'])==1 else ''
    def status(self,s):
        self.states.append(copy.deepcopy(s))
        if self.failure=='status':raise OSError('disk full')
    def complete(self,j):pass
    def emit(self,*args):
        if self.failure=='logging' and self.started:raise OSError('pipeline broken')


class SchedulerTests(unittest.TestCase):
    setUp=RunnerTests.setUp
    job=RunnerTests.job
    def backend(self,n=7,**kw):
        jobs=[self.job('job'+str(i)) for i in range(n)]
        for i,j in enumerate(jobs):
            j['case_id']=str(i);j['pending'][0]['task']['case_id']=str(i)
        backend=FakeBackend(jobs,**kw);backend.key=self.m.p.key;return backend
    def test_dispatch_one_worker_each_physical_card_and_then_drain(self):
        b=self.backend();self.m.schedule_queue(b.jobs,False,b)
        self.assertEqual(len(b.started),7)
        self.assertEqual([j['gpu'] for j in b.started[:4]],[2,4,5,6])
        self.assertEqual(len(b.closed),7);self.assertEqual(b.states[-1]['running'],{})
    def test_fatal_worker_stops_new_dispatch_and_naturally_drains_started_peers(self):
        b=self.backend(codes={'job0':3})
        with self.assertRaises(RuntimeError):self.m.schedule_queue(b.jobs,False,b)
        self.assertEqual(len(b.started),4);self.assertEqual(len(b.closed),4)
        self.assertTrue(all(b.polls[j['name']]>=2 for j in b.started))
    def test_controller_proof_status_log_and_pipeline_errors_all_drain(self):
        for failure in ('records','status','log','logging'):
            b=self.backend(failure=failure)
            with self.subTest(failure=failure),self.assertRaises(RuntimeError):self.m.schedule_queue(b.jobs,False,b)
            self.assertLessEqual(len(b.started),4);self.assertEqual(len(b.closed),len(b.started))
            self.assertTrue(all(b.polls[j['name']]>=2 for j in b.started))

    def test_nonzero_after_last_commit_and_missing_worker_completion_are_fatal(self):
        for code in (1,75):
            b=self.backend(n=1,codes={'job0':code});spawn=b.spawn
            def commit_then_exit(j):
                proc,handle=spawn(j);poll=proc.poll
                def checked_poll():
                    value=poll()
                    if value is not None:
                        for item in j['pending']:b.done[(j['model'],b.key(item['task']))]={'record':{}}
                    return value
                proc.poll=checked_poll;return proc,handle
            b.spawn=commit_then_exit
            with self.subTest(code=code),self.assertRaises(RuntimeError):self.m.schedule_queue(b.jobs,False,b)
            self.assertEqual(len(b.closed),1)
        b=self.backend(n=1)
        def no_completion(j):raise ValueError('missing post-run worker completion')
        b.complete=no_completion
        with self.assertRaises(RuntimeError):self.m.schedule_queue(b.jobs,False,b)


if __name__=='__main__':unittest.main()
