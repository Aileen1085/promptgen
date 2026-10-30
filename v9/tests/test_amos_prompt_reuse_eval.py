"""Safety and scientific invariants of the additive AMOS reuse controller."""
import ast
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
ENTRY=ROOT/'tools/amos_prompt_reuse_eval.py'
PYTHON_SOURCE=ROOT/'tools/amos_prompt_ablation_eval.py'


class ReuseTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(ENTRY.exists(), 'add source-bound reuse controller')
        self.m=importlib.import_module('tools.amos_prompt_reuse_eval')

    def ns(self):return self.m.engine()
    def task(self,case='0001',gid=201):return dict(source_name='amos',case_id=case,global_class_id=gid,local_class_id=1)
    def folder(self):return Path(tempfile.mkdtemp(prefix='amos_reuse_'))
    def write(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8');return path

    def test_child_globals_are_fresh_and_dispatch_new_entry(self):
        m=self.m;parent=m.parent_engine();before=dict(parent);ns=m.engine()
        self.assertIsNot(ns,parent)
        for name in ('worker','make_attempt','merge','schedule_queue','main','class_records'):
            self.assertIs(ns[name].__globals__,ns,name)
        self.assertIs(ns['Backend'].spawn.__globals__,ns)
        self.assertEqual(before,parent)
        self.assertEqual(ns['VARIANTS'],parent['VARIANTS'])
        self.assertEqual(ns['EVALUATION_ORDER'],('baseline','fg_scribble_bg_scribble','fg_box_bg_point','fg_box_bg_scribble','sagittal_only','coronal_only'))
        d=self.folder();job=dict(log=str(d/'a.log'),spec=str(d/'a.json'),gpu=2)
        with patch.object(ns['subprocess'],'Popen',return_value=object()) as spawn:
            _,handle=ns['Backend'](d,False).spawn(job);handle.close()
        self.assertEqual(Path(spawn.call_args.args[0][1]),ENTRY)
        self.assertIn('tools/amos_prompt_reuse_runtime.py',ns['SOURCE_SEEDS'])
        self.assertTrue(set(parent['SOURCE_SEEDS'])<=set(ns['SOURCE_SEEDS']))

    def test_one_exact_worker_ast_edit_and_source_binding(self):
        m=self.m;source=PYTHON_SOURCE.read_bytes();parent=m.PARENT
        before,_=parent.transform_controller(source,str(PYTHON_SOURCE),parent.NATIVE_CONTROLLER_SHA256)
        after,proof=m.transform_controller(source,str(PYTHON_SOURCE),parent.NATIVE_CONTROLLER_SHA256)
        self.assertEqual(proof['worker_iteration_replacements'],1)
        restored=m.reverse_worker_transform(after)
        self.assertEqual(ast.dump(before,include_attributes=False),ast.dump(restored,include_attributes=False))
        worker=next(n for n in after.body if isinstance(n,ast.FunctionDef) and n.name=='worker')
        self.assertIn('for v in EVALUATION_ORDER',ast.unparse(worker))
        changed=source.replace(b'for v in VARIANTS]',b'for v in tuple(VARIANTS)]')
        with self.assertRaisesRegex(ValueError,'iteration|anchor'):
            m.transform_controller(changed,'changed',hashlib.sha256(changed).hexdigest())
        with self.assertRaisesRegex(ValueError,'SHA'):m.transform_controller(source,'changed','0'*64)

    def test_old_and_new_record_union_is_whole_and_disjoint(self):
        m=self.m;task=self.task();key=('amos','0001',201)
        old={key:{'spec':{'name':'amos_0001'},'record':{'task':task}}}
        new={('amos','0002',201):{'spec':{'name':'amos_0002'}}}
        self.assertEqual(m.union_records(old,new),{**old,**new})
        with self.assertRaisesRegex(ValueError,'duplicate'):m.union_records(old,old)
        ns=self.ns();d=self.folder();calls=[]
        with patch.dict(ns,_frozen_records=lambda out: calls.append(('parent',out)) or old,
                        native_class_records=lambda out,control=False: calls.append(('child',out,control)) or new):
            self.assertEqual(ns['class_records'](d),{**old,**new})
            self.assertEqual(ns['class_records'](d,True),new)
        self.assertEqual(calls,[('parent',d),('child',d,False),('child',d,True)])

    def test_old_keys_never_create_new_attempt(self):
        ns=self.ns();d=self.folder();task=self.task();job=ns['case_jobs']([task],False)[0]
        self.assertIsNone(ns['make_attempt'](d,job,2,{}, {ns['key'](task):{}}))
        self.assertFalse((d/'job_specs').exists())

    def test_history_requires_positive_exit_and_preserves_policy(self):
        m=self.m;ns=self.ns();task=self.task();job=ns['case_jobs']([task],False)[0]
        prior=dict(job,attempt=3,frame_batch=1,minimum_free_mib=45000,oom_retries=2,not_before=999.,avoid_gpus={'2':2222.},gpu=2)
        got=m.advance_history(job,prior,dict(returncode=0,log_text='',observed_at=10))
        self.assertEqual([got[k] for k in ('attempt','frame_batch','minimum_free_mib','oom_retries','not_before','avoid_gpus')],[4,1,45000,2,999.,{'2':2222.}])
        resource=m.advance_history(job,prior,dict(returncode=75,log_text='',observed_at=10))
        self.assertEqual((resource['attempt'],resource['frame_batch'],resource['oom_retries']),(4,1,2))
        oom=m.advance_history(job,dict(prior,frame_batch=4),dict(returncode=1,log_text='CUDA out of memory',observed_at=10))
        self.assertEqual((oom['attempt'],oom['frame_batch'],oom['oom_retries']),(4,2,3))
        for missing in (None,{},dict(returncode=1,log_text='warning',observed_at=10)):
            with self.assertRaises(ValueError):m.advance_history(job,prior,missing)
        self.assertEqual(m.advance_history(job,None,None),job)

    def test_per_file_inventory_mutation_refuses(self):
        m=self.m;d=self.folder();file=self.write(d/'proof.json',dict(complete=True))
        inventory=m.inventory([file],d);m.verify_inventory(inventory,d)
        file.write_text('{}')
        with self.assertRaisesRegex(ValueError,'changed'):m.verify_inventory(inventory,d)

    def test_only_owned_exact_stable_process_can_receive_signal(self):
        m=self.m;identity=dict(pid=123,uid=9,ppid=1,starttime=555,cwd='/project',argv=['python','/project/tools/amos_prompt_ablation_native_manifest_eval.py','run'],state='T')
        with patch.object(m,'read_process',return_value=dict(identity,starttime=556)),patch.object(m.os,'kill') as kill:
            with self.assertRaisesRegex(ValueError,'identity'):m.signal_exact(identity,signal.SIGTERM)
            kill.assert_not_called()
        with patch.object(m,'read_process',return_value=identity),patch.object(m.os,'kill') as kill:
            m.signal_exact(identity,getattr(signal,'SIGSTOP',19))
            kill.assert_called_once_with(123,getattr(signal,'SIGSTOP',19))

    def test_zombie_exit_requires_captured_identity_and_raw_normal_zero(self):
        m=self.m;before=dict(pid=123,uid=9,ppid=1,starttime=555,cwd='/project',argv=['python','script','worker'],state='R')
        zombie=dict(before,state='Z',cwd=None,argv=[],raw_exit_status=0)
        self.assertEqual(m.observed_exit(before,zombie),0)
        self.assertEqual(m.observed_exit(before,dict(zombie,raw_exit_status=75<<8)),75)
        for changed in (dict(zombie,starttime=556),dict(zombie,raw_exit_status=None),None):
            with self.assertRaises(ValueError):m.observed_exit(before,changed)

    def test_second_quiescence_failure_prevents_prepare(self):
        m=self.m
        with patch.object(m,'related',side_effect=[[],[{'pid':1}]]),patch.object(m.time,'sleep') as sleep:
            with self.assertRaisesRegex(ValueError,'quiescence'):m.require_quiescence()
            sleep.assert_called_once()

    def test_proc_reader_rejects_torn_starttime(self):
        m=self.m;d=self.folder();p=d/'123';p.mkdir()
        def stat(start):
            fields=['S','1']+['0']*47+[str(0)];fields[19]=str(start)
            return '123 (python tricky ) name) '+' '.join(fields)
        (p/'stat').write_text(stat(555));(p/'status').write_text('Uid:\t9\t9\t9\t9\n');(p/'cmdline').write_bytes(b'python\0script\0run\0');(p/'wchan').write_text('hrtimer_nanosleep')
        with patch.object(m.os,'readlink',return_value='/project'):
            got=m.read_process(123,d);self.assertEqual((got['starttime'],got['uid']),(555,9))
            original=Path.read_text;count=0
            def torn(path,*a,**kw):
                nonlocal count
                if path==p/'stat':count+=1;return stat(555 if count==1 else 556)
                return original(path,*a,**kw)
            with patch.object(Path,'read_text',torn):
                with self.assertRaisesRegex(ValueError,'torn'):m.read_process(123,d)

    def test_partial_class_and_mixed_batches_are_never_accepted(self):
        ns=self.ns();task=self.task()
        records=[dict(variant=v,actual_frame_batch=4) for v in ns['VARIANTS']]
        with patch.dict(ns,validate_record=lambda *a,**kw:None):
            self.assertEqual(ns['validate_class'](records,task),4)
            with self.assertRaisesRegex(ValueError,'six'):ns['validate_class'](records[:-1],task)
            records[-1]['actual_frame_batch']=2
            with self.assertRaisesRegex(ValueError,'same'):ns['validate_class'](records,task)

    def test_completed_worker_requires_actual_zero_native_finish_and_all_pending(self):
        m=self.m;p=m.parent_engine();d=self.folder();task=self.task();log=d/'worker.log';log.write_text('normal')
        spec=dict(spec=str(self.write(d/'spec.json',{})),log=str(log),pending=[task])
        identity=dict(pid=123,uid=9,ppid=1,starttime=555,cwd='/project',argv=['worker'],state='R')
        item=dict(identity=identity,spec=spec);z=dict(identity,state='Z',argv=[],cwd=None,raw_exit_status=0)
        with patch.dict(p,verify_worker_complete=lambda j:None,class_records=lambda out:{p['key'](task):{}}):
            row=m.validate_finished_worker(item,z);self.assertEqual(row['returncode'],0)
        with patch.dict(p,verify_worker_complete=lambda j:None,class_records=lambda out:{}):
            with self.assertRaisesRegex(ValueError,'coverage'):m.validate_finished_worker(item,z)
        with patch.dict(p,verify_worker_complete=lambda j:(_ for _ in ()).throw(ValueError('no native finish'))):
            with self.assertRaisesRegex(ValueError,'native finish'):m.validate_finished_worker(item,z)
        with self.assertRaisesRegex(ValueError,'non-resource'):m.validate_finished_worker(item,dict(z,raw_exit_status=1<<8))

    def test_fully_reused_case_accepts_only_normal_bound_completion(self):
        m=self.m;p=m.parent_engine();d=self.folder();task=self.task();canonical=dict(name='amos_0001',pending=[task])
        spec=dict(name='amos_0001',spec=str(d/'spec.json'),log=str(d/'worker.log'),run=str(d/'run'))
        for path in (d/'spec.json',d/'worker.log',d/'run/native_worker_complete.json',d/'run/restoration_final.json'):
            self.write(path,spec if path==d/'spec.json' else {})
        marker=self.write(d/'class.done.json',{});found={p['key'](task):dict(spec=spec,marker=str(marker),marker_sha256=m.sha(marker))}
        normal=dict(spec=spec['spec'],returncode=0)
        with patch.object(m,'parent_out',return_value=d),patch.dict(p,verify_worker_complete=lambda j:None):
            proof=m._old_case_proof(canonical,found,dict(workers={'123':normal}))
            self.assertEqual(proof['kind'],'captured_normal_worker_completion')
            for workers in ({},{'123':dict(normal,returncode=75)}):
                with self.assertRaisesRegex(ValueError,'normal completion'):m._old_case_proof(canonical,found,dict(workers=workers))
            with self.assertRaisesRegex(ValueError,'coverage'):m._old_case_proof(canonical,{},dict(workers={'123':normal}))

    def test_fresh_controls_ignore_reused_records_and_require_new_gate(self):
        m=self.m;ns=self.ns();d=self.folder()
        with patch.dict(ns,check_controls=lambda out:dict(digest='fresh digest')):
            with self.assertRaises(FileNotFoundError):ns['check_control_gate'](d)
            self.write(d/'protocol.json',{})
            self.write(d/'control_verification.json',dict(passed=True,real_gpu=True,reviewer='independent',
                protocol_sha256=m.PARENT_PROTOCOL_SHA256,controls_digest='fresh digest'))
            with self.assertRaisesRegex(ValueError,'gate'):ns['check_control_gate'](d)

    def test_attempt_policy_and_exact_pending_subset_survive_reuse(self):
        ns=self.ns();d=self.folder();tasks=[self.task(),self.task(gid=202)]
        canonical=ns['native_case_jobs'](tasks,False)[0];canonical.update(prepared=str(d/'prepared'),attempt=3,frame_batch=2,oom_retries=1)
        self.write(d/'jobs.json',dict(cases=[canonical],controls=[]));self.write(d/'control_verification.json',{})
        got=ns['make_attempt'](d,canonical,2,{2:dict(uuid='GPU-test')},{ns['key'](tasks[0]):{}})
        self.assertEqual(got['pending'],tasks[1:]);self.assertEqual((got['attempt'],got['frame_batch'],got['oom_retries']),(3,2,1))
        self.assertIn('_a3.json',got['spec'])

    def test_natural_drain_only_signals_scheduler_after_positive_exit(self):
        m=self.m;d=self.folder();self.write(d/m.HANDOFF,{});self.write(d/m.STOPPED,{})
        self.write(d/m.PARENT.OUT/'status.json',{});self.write(d/m.PARENT.PIPELINE,{})
        scheduler=dict(pid=123,uid=9,ppid=1,starttime=555,cwd='/project',argv=['run'],state='T')
        worker=dict(pid=124,uid=9,ppid=123,starttime=556,cwd='/project',argv=['worker'],state='R')
        zombie=dict(worker,state='Z',raw_exit_status=0);item=dict(identity=worker,spec={})
        handoff=dict(scheduler=scheduler,workers=[item]);seen=[]
        def proc(pid):return scheduler if pid==123 and len(seen)<2 else None if pid==123 else zombie
        def sig(identity,value):seen.append((identity['pid'],value))
        with patch.object(m,'ROOT',d),patch.object(m,'verify_handoff',return_value=handoff),patch.object(m,'read_process',side_effect=proc),\
             patch.object(m,'validate_finished_worker',return_value=dict(returncode=0)),patch.object(m,'signal_exact',side_effect=sig),\
             patch.object(m,'require_quiescence',return_value=[[],[]]),patch.object(signal,'SIGCONT',18,create=True):
            result=m.finish_parent()
        self.assertEqual(seen,[(123,signal.SIGTERM),(123,18)])
        self.assertEqual(result['related_checks'],[[],[]]);self.assertTrue((d/m.DRAINED).exists())

    def test_zero_exit_without_native_proof_prevents_scheduler_retirement(self):
        m=self.m;d=self.folder();scheduler=dict(pid=123,uid=9,ppid=1,starttime=555,cwd='/project',argv=['run'],state='T')
        worker=dict(pid=124,uid=9,ppid=123,starttime=556,cwd='/project',argv=['worker'],state='R')
        with patch.object(m,'ROOT',d),patch.object(m,'verify_handoff',return_value=dict(scheduler=scheduler,workers=[dict(identity=worker)])),\
             patch.object(m,'read_process',side_effect=lambda pid:scheduler if pid==123 else dict(worker,state='Z',raw_exit_status=0)),\
             patch.object(m,'validate_finished_worker',side_effect=ValueError('native guard missing')),patch.object(m,'signal_exact') as sig:
            with self.assertRaisesRegex(ValueError,'native guard'):m.finish_parent()
            sig.assert_not_called()

    def test_exact8910_merge_keeps_same_key_deltas_with_old_and_new(self):
        m=self.m;ns=self.ns();d=self.folder();tasks=[];found={}
        for case in range(100):
            for gid in range(14 if case<15 else 15):
                task=dict(self.task(f'{case:04}',201+gid),local_class_id=gid+1);tasks.append(task)
                row=dict(task,**{name:.5 for name in ns['METRICS']},dynamic_threshold_source='prompt_logit_midpoint',mask_threshold_used=.70)
                found[ns['key'](task)]=dict(spec=dict(name='amos_'+task['case_id'],frame_batch=4),record=dict(records=[dict(variant=v,row=dict(row,dice=.5+i*.01)) for i,v in enumerate(ns['VARIANTS'])]))
        old=dict(list(found.items())[:700]);new=dict(list(found.items())[700:]);historical=[dict(v['record']['records'][0]['row'],dice=.4) for v in found.values()]
        for name,value in [('status.json',dict(phase='full',pending=0,running={},failed=[])),('manifest.json',dict(tasks=tasks)),('jobs.json',dict(cases=ns['native_case_jobs'](tasks))),('protocol.json',{}),('control_verification.json',{})]:self.write(d/name,value)
        with patch.dict(ns,verify=lambda *a,**kw:None,check_control_gate=lambda out:None,related=lambda:[],
                        _frozen_records=lambda out:old,native_class_records=lambda out,control=False:new,
                        verify_case_success=lambda *a:None,native_reference=lambda:historical):
            result=ns['merge'](d)
            self.assertEqual(result['rows'],8910)
            for i,v in enumerate(ns['VARIANTS']):
                self.assertEqual(result['variants'][v]['task_count'],1485)
                self.assertAlmostEqual(result['variants'][v]['delta_from_fresh_baseline']['dice'],i*.01)
                self.assertAlmostEqual(result['variants'][v]['delta_from_historical_baseline']['dice'],.1+i*.01)
            with self.assertRaises(FileExistsError):ns['merge'](d)

    def test_children_filters_ppid_before_reading_other_users_cwd(self):
        m=self.m;read=Path.read_text
        def stat(path,*args,**kwargs):
            if str(path).replace('\\','/') in ('/proc/10/stat','/proc/11/stat'):
                fields=['S','123' if path.parent.name=='11' else '1']+['0']*48
                return path.parent.name+' (python) '+' '.join(fields)
            return read(path,*args,**kwargs)
        value=dict(pid=11,ppid=123)
        with patch.object(Path,'iterdir',return_value=iter([Path('/proc/10'),Path('/proc/11')])),\
             patch.object(Path,'read_text',stat),patch.object(m,'read_process',side_effect=lambda pid: value if pid==11 else (_ for _ in ()).throw(PermissionError('other user cwd'))):
            self.assertEqual(m.children(dict(pid=123)),[value])

    def test_reuse_audit_and_parent_files_must_remain_bound(self):
        m=self.m;d=self.folder();out=d/'new';f=self.write(d/'parent/proof.json',{})
        self.write(d/m.HANDOFF,{});self.write(d/m.DRAINED,{})
        audit=dict(kind='amos_complete_six_class_reuse_v1',parent_output=m.PARENT.OUT,
                   parent_protocol_sha256=m.PARENT_PROTOCOL_SHA256,parent_gate_sha256=m.PARENT_GATE_SHA256,
                   handoff_sha256=m.sha(d/m.HANDOFF),drained_sha256=m.sha(d/m.DRAINED),files=m.inventory([f],d))
        a=self.write(out/'reuse_audit.json',audit);self.write(out/'native_bootstrap_audit.json',dict(reuse_audit_sha256=m.sha(a)))
        with patch.object(m,'ROOT',d),patch.object(m,'verify_drained',return_value={}):
            self.assertEqual(m.verify_reuse(out),audit)
            f.write_text('changed')
            with self.assertRaisesRegex(ValueError,'changed'):m.verify_reuse(out)
            f.write_text('{}');a.write_text('{}')
            with self.assertRaisesRegex(ValueError,'chain'):m.verify_reuse(out)

    def test_child_help_works_without_historical_failure_files(self):
        import subprocess
        result=subprocess.run([sys.executable,str(ENTRY),'--help'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('freeze-parent',result.stdout);self.assertIn('wait-run',result.stdout)

    def test_freeze_capture_race_resumes_exact_scheduler_without_worker_signal(self):
        m=self.m;d=self.folder();scheduler=dict(pid=123,uid=9,ppid=1,starttime=555,cwd=str(d.resolve()),
             argv=['python',str(d/'tools/amos_prompt_ablation_native_manifest_eval.py'),'run'],state='S',wchan='hrtimer_nanosleep')
        worker=dict(pid=124,uid=9,ppid=123,starttime=556,cwd=str(d.resolve()),argv=['worker'],state='R')
        spec=dict(name='amos_0001',spec='/old/spec.json',log='/old/log')
        self.write(d/m.PARENT.OUT/'status.json',dict(phase='full',failed=[],running={'2':dict(pid=124,**spec)}))
        seen=[]
        with patch.object(m,'ROOT',d),patch.object(m.os,'getuid',return_value=9,create=True),patch.object(m,'verify_parent'),\
             patch.object(m,'read_process',side_effect=lambda pid:dict(scheduler,state='T') if seen else scheduler),\
             patch.object(m,'children',side_effect=[[worker],[]]),patch.object(m,'_bind_worker',side_effect=lambda value,s:dict(identity=value,spec=spec,spec_sha256='same')),\
             patch.object(m,'signal_exact',side_effect=lambda identity,sig:seen.append((identity['pid'],sig))),\
             patch.object(signal,'SIGSTOP',19,create=True),patch.object(signal,'SIGCONT',18,create=True):
            with self.assertRaisesRegex(ValueError,'worker set changed'):m.freeze_parent(123)
        self.assertEqual(seen,[(123,19),(123,18)])
        self.assertFalse((d/m.HANDOFF).exists())

    def test_scheduler_owner_script_start_and_mode_are_required(self):
        m=self.m;d=self.folder();scheduler=dict(pid=123,uid=9,ppid=1,starttime=555,cwd=str(d.resolve()),
              argv=['python',str(d/'tools/amos_prompt_ablation_native_manifest_eval.py'),'run'],state='S')
        with patch.object(m,'ROOT',d),patch.object(m.os,'getuid',return_value=9,create=True):
            self.assertEqual(m._scheduler(scheduler),scheduler)
            for flags in (['-B'],['-u'],['-B','-u'],['-u','-B']):
                row=dict(scheduler,argv=['python',*flags,'tools/amos_prompt_ablation_native_manifest_eval.py','run'])
                self.assertEqual(m._scheduler(row),row)
            for row in (dict(scheduler,uid=10),dict(scheduler,cwd='/other'),dict(scheduler,argv=['python','other','run']),dict(scheduler,argv=scheduler['argv'][:2]+['worker']),dict(scheduler,state='Z')):
                with self.assertRaisesRegex(ValueError,'owned'):m._scheduler(row)
            for flags in (['-c'],['-u','-c'],['-u','-u'],['--help']):
                with self.assertRaisesRegex(ValueError,'owned'):
                    m._scheduler(dict(scheduler,argv=['python',*flags,'tools/amos_prompt_ablation_native_manifest_eval.py','run']))

    def test_drained_record_cannot_substitute_an_unrelated_spec_or_log(self):
        m=self.m;d=self.folder();identity=dict(pid=124,uid=9,ppid=123,starttime=556,cwd='/project',argv=['worker'],state='R')
        spec=dict(spec=str(self.write(d/'spec.json',{})),log=str(self.write(d/'log',{})))
        handoff=dict(scheduler={},workers=[dict(identity=identity,spec=spec)])
        self.write(d/m.HANDOFF,{});self.write(d/m.STOPPED,{})
        row=dict(identity=dict(identity,state='Z',raw_exit_status=0),returncode=0,raw_exit_status=0,
                 spec='other.json',spec_sha256=m.sha(spec['spec']),log=spec['log'],log_sha256=m.sha(spec['log']),log_text='{}',
                 observed_at=1.,log_mtime_ns=Path(spec['log']).stat().st_mtime_ns)
        self.write(d/m.DRAINED,dict(kind='amos_natural_drain_complete_v1',handoff_sha256=m.sha(d/m.HANDOFF),stopped_sha256=m.sha(d/m.STOPPED),related_checks=[[],[]],retired_scheduler={},workers={'124':row},files={}))
        with patch.object(m,'ROOT',d),patch.object(m,'verify_handoff',return_value=handoff):
            with self.assertRaisesRegex(ValueError,'exit evidence'):m.verify_drained()

    def test_captured_status_binds_pid_name_spec_and_log(self):
        m=self.m;spec=dict(name='amos_0001',spec='/old/spec.json',log='/old/log')
        workers=[dict(identity=dict(pid=124),spec=spec)]
        queue=dict(phase='full',failed=[],running={'2':dict(pid=124,**{k:spec[k] for k in ('name','spec','log')})})
        m.validate_status_workers(queue,workers)
        for field in ('pid','name','spec','log'):
            wrong=copy.deepcopy(queue);wrong['running']['2'][field]='different'
            with self.assertRaisesRegex(ValueError,'status'):m.validate_status_workers(wrong,workers)

    def test_empty_capture_still_requires_scheduler_frozen_before_retirement(self):
        m=self.m;d=self.folder();scheduler=dict(pid=123,uid=9,ppid=1,starttime=555,cwd='/project',argv=['run'],state='T')
        with patch.object(m,'ROOT',d),patch.object(m,'verify_handoff',return_value=dict(scheduler=scheduler,workers=[])),\
             patch.object(m,'read_process',return_value=dict(scheduler,state='R')),patch.object(m,'signal_exact') as sig:
            with self.assertRaisesRegex(ValueError,'frozen'):m.finish_parent()
            sig.assert_not_called()

    def test_real101_native_bootstraps_bind_reuse_audit_before_protocol(self):
        from tests.test_amos_prompt_ablation_native_manifest_eval import NativeManifestTests
        m=self.m;fixture=NativeManifestTests();fixture.setUp();d,out,native,initial,task=fixture.fixture()
        ns=self.ns();jobs=[];tasks=[]
        for case in range(100):
            current=[dict(task,case_id=f'{case:04}',global_class_id=201+i,local_class_id=i+1,class_name=f'organ{i}',
                          image=f'data/{case:04}/ct',label=f'data/{case:04}/gt') for i in range(14 if case<15 else 15)]
            tasks.extend(current);name=f'amos_{case:04}';prepared=out/'prepared'/name
            self.write(prepared/'manifest.json',m.PARENT.native_manifest_payload(tasks=current,expected_task_count=len(current)))
            self.write(prepared/'refs.json',dict(per_case_class=[dict(t,prompt_protocol='v9_2_adaptive_only_r1_20_f0p15_final_trigger',prompt_cache_name=f'{case}_{i}.npz') for i,t in enumerate(current)]))
            jobs.append(dict(name=name,case_id=f'{case:04}',control=False,pending=current,task_count=len(current),frame_batch=4,prepared=str(prepared)))
        split_path=d/'configs/ct13_v10_2_split_20260928.json';split=json.loads(split_path.read_text());split['datasets']['amos']['val']=[dict(case_id=j['case_id']) for j in jobs];self.write(split_path,split)
        control=dict(jobs[0],name='amos_native_control',control=True,pending=tasks[:1],task_count=1,prepared=str(out/'prepared/amos_native_control'))
        self.write(Path(control['prepared'])/'manifest.json',m.PARENT.native_manifest_payload(tasks=tasks[:1],expected_task_count=1))
        refs=json.loads((Path(jobs[0]['prepared'])/'refs.json').read_text());refs['per_case_class']=refs['per_case_class'][:1]
        self.write(Path(control['prepared'])/'refs.json',refs)
        reuse=dict(kind='reuse fixture',files={'proof':'frozen SHA'})
        with patch.object(m,'build_reuse_audit',return_value=reuse),patch.dict(ns,ROOT=d):
            path=m.bootstrap(out,jobs,[control])
        audit=json.loads(path.read_text())
        self.assertEqual((audit['job_count'],audit['full_tasks'],audit['control_tasks']),(101,1485,1))
        self.assertEqual(audit['reuse_audit_sha256'],m.sha(out/'reuse_audit.json'))
        self.assertEqual(audit['reuse_files'],reuse['files'])
        self.assertEqual(audit['reuse_controller_transform']['worker_iteration_replacements'],1)
        self.assertFalse(audit['model_loaded']);self.assertFalse(audit['cuda_initialized'])
        self.assertFalse((out/'protocol.json').exists())

    def test_latest_attempt_history_requires_captured_or_parent_success_evidence(self):
        m=self.m;p=m.parent_engine();d=self.folder();task=self.task()
        canonical=p['case_jobs']([task],False)[0];canonical['prepared']=str(d/'prepared')
        self.write(d/'jobs.json',dict(cases=[canonical],controls=[]))
        spec=dict(canonical,gpu=2,gpu_uuid='GPU-x',actual_frame_batch=4,run=str(d/'runs/amos_0001_a0'),
                  spec=str(d/'job_specs/amos_0001_a0.json'),log=str(d/'logs/amos_0001_a0.log'))
        self.write(Path(spec['spec']),spec);self.write(Path(spec['log']),{})
        with patch.object(m,'parent_out',return_value=d),patch.object(m,'verify_parent',return_value={}),\
             patch.object(m,'verify_drained',return_value=dict(workers={})):
            with self.assertRaisesRegex(ValueError,'positive'):m.parent_history()
        evidence=dict(spec=spec['spec'],returncode=75,log_text='',observed_at=10.)
        with patch.object(m,'parent_out',return_value=d),patch.object(m,'verify_parent',return_value={}),\
             patch.object(m,'verify_drained',return_value=dict(workers={'124':evidence})):
            history=m.parent_history();self.assertEqual(history['amos_0001']['previous'],spec)
            self.assertEqual(m.advance_history(canonical,**history['amos_0001'])['attempt'],1)


class _ChildGlobalsProxy:
    """patch.object writes directly into the executed child function globals."""
    def __init__(self,ns):object.__setattr__(self,'_ns',ns)
    def __getattr__(self,name):
        try:return self._ns[name]
        except KeyError:raise AttributeError(name) from None
    def __setattr__(self,name,value):self._ns[name]=value
    def __delattr__(self,name):
        try:del self._ns[name]
        except KeyError:raise AttributeError(name) from None


_original=importlib.import_module('tests.test_amos_prompt_ablation_eval')


class ChildWorkerExecutionTests(_original.WorkerTests):
    def setUp(self):
        self.m=_ChildGlobalsProxy(importlib.import_module('tools.amos_prompt_reuse_eval').engine())
        self.task=dict(source_name='amos',case_id='0001',global_class_id=201,local_class_id=1,
                       class_name='spleen',image='data/a.nii.gz',label='data/b.nii.gz')

    def patches(self,d,torch,fake):
        return (*super().patches(d,torch,fake),patch.object(self.m,'_frozen_records',return_value={}))

    def test_case_worker_initializes_model_once_and_commits_six_variants_per_class(self):
        from contextlib import ExitStack
        m=self.m;d,out,job,tasks,torch=self.setup_worker();fake,calls=self.fake_runtime(job)
        m.atomic_new(out/'control_verification.json',{'independent':'passed'});job['control_gate_sha256']=m.sha(out/'control_verification.json')
        Path(job['spec']).write_text(json.dumps(job))
        with ExitStack() as stack:
            for p in self.patches(d,torch,fake):stack.enter_context(p)
            m.worker(job);records=m.class_records(out);self.assertEqual(len(records),2)
            proof=m.complete_case(out,job);self.assertEqual(len(proof['markers']),2)
            m.verify_case_success(out,m.read(out/'jobs.json')['cases'][0],records)
            Path(job['log']).write_text('tampered')
            with self.assertRaises(ValueError):m.verify_case_success(out,m.read(out/'jobs.json')['cases'][0],records)
        self.assertEqual(sum(c[0]=='initialize' for c in calls),1)
        self.assertEqual(calls[1:7],[(201,v) for v in m.EVALUATION_ORDER])
        self.assertEqual(calls[7:13],[(202,v) for v in m.EVALUATION_ORDER])
        self.assertEqual(m.VARIANTS,('baseline','sagittal_only','coronal_only','fg_box_bg_point','fg_scribble_bg_scribble','fg_box_bg_scribble'))

    def test_control_worker_executes_native_then_baseline_and_other_five(self):
        from contextlib import ExitStack
        m=self.m;d,out,job,tasks,torch=self.setup_worker();canonical=m.read(out/'jobs.json')['cases'][0]
        canonical.update(name='amos_native_control',control=True,pending=tasks[:1],task_count=1,prepared=str(out/'prepared/amos_native_control'))
        tag='amos_native_control_a0';job.update(canonical,run=str(out/'runs'/tag),spec=str(out/'job_specs'/(tag+'.json')),log=str(out/'logs'/(tag+'.log')))
        (out/'jobs.json').write_text(json.dumps(dict(cases=[],controls=[canonical])));m.atomic_new(job['spec'],job)
        fake,calls=self.fake_runtime(job);control_checks=[]
        fake.validate_control=lambda native,records:control_checks.append((native['variant'],[r['variant'] for r in records])) or dict(actual_gpu_control=True)
        with ExitStack() as stack:
            for p in self.patches(d,torch,fake):stack.enter_context(p)
            m.worker(job);found=m.class_records(out,True);self.assertEqual(len(found),1)
            m.verify_worker_complete(job)
            self.assertEqual(m.read(Path(job['run'])/'native_control.json')['proof'],dict(actual_gpu_control=True))
        self.assertEqual(calls[1:],[(201,v) for v in ('native_control',*m.EVALUATION_ORDER)])
        self.assertEqual(control_checks,[('native_control',list(m.EVALUATION_ORDER))])

    def test_parent_whole_class_plus_child_pending_continuation_has_composite_success(self):
        from contextlib import ExitStack
        module=importlib.import_module('tools.amos_prompt_reuse_eval');m=self.m
        old=_original.WorkerTests();old.m=_ChildGlobalsProxy(module.parent_engine());old.task=self.task.copy()
        od,oldout,oldjob,tasks,torch=old.setup_worker();oldfake,oldcalls=old.fake_runtime(oldjob,fail_at=10)
        old.m.atomic_new(oldout/'control_verification.json',{'independent':'old passed'});oldjob['control_gate_sha256']=old.m.sha(oldout/'control_verification.json')
        Path(oldjob['spec']).write_text(json.dumps(oldjob))
        with ExitStack() as oldstack:
            for p in old.patches(od,torch,oldfake):oldstack.enter_context(p)
            with self.assertRaisesRegex(RuntimeError,'CUDA out of memory'):old.m.worker(oldjob)
            retained=old.m.class_records(oldout);self.assertEqual(set(retained),{m.key(tasks[0])})
            d,out,job,_,torch=self.setup_worker();tag=job['name']+'_a1'
            job.update(attempt=1,oom_retries=1,frame_batch=2,actual_frame_batch=2,pending=tasks[1:],task_count=1,
                       run=str(out/'runs'/tag),spec=str(out/'job_specs'/(tag+'.json')),log=str(out/'logs'/(tag+'.log')))
            m.atomic_new(out/'control_verification.json',{'independent':'new passed'});job['control_gate_sha256']=m.sha(out/'control_verification.json')
            m.atomic_new(job['spec'],job);Path(job['log']).write_text('new normal final attempt')
            fake,calls=self.fake_runtime(job);reads=[]
            def frozen(asked):
                self.assertEqual(Path(asked),out);reads.append(oldout)
                return old.m.class_records(oldout)
            with ExitStack() as newstack:
                for p in self.patches(d,torch,fake):newstack.enter_context(p)
                newstack.enter_context(patch.object(m,'_frozen_records',side_effect=frozen))
                # Execute the actual child worker on only the uncommitted class.
                m.worker(job);found=m.class_records(out)
                self.assertEqual(set(found),{m.key(t) for t in tasks})
                proof=m.complete_case(out,job);canonical=m.read(out/'jobs.json')['cases'][0]
                m.verify_case_success(out,canonical,found)
                self.assertEqual(set(proof['contributing_attempts']),{oldjob['spec'],job['spec']})
                self.assertEqual(len(proof['markers']),2)
                Path(oldjob['log']).write_text('old contributing log changed')
                with self.assertRaises(ValueError):m.verify_case_success(out,canonical,found)
        self.assertTrue(reads);self.assertEqual(calls[1:],[(202,v) for v in m.EVALUATION_ORDER])
        self.assertFalse((Path(job['run'])/'classes/0001__201.done.json').exists())


class ChildSchedulerExecutionTests(_original.SchedulerTests):
    setUp=ChildWorkerExecutionTests.setUp

    def test_production_prefilters_all_reused_prefix_cases_before_real_loop(self):
        m=self.m;d=Path(tempfile.mkdtemp());jobs=[dict(self.job(),name=f'amos_{i:04}',case_id=f'{i:04}',pending=[dict(self.task,case_id=f'{i:04}')]) for i in range(41)]
        tasks=[j['pending'][0] for j in jobs];old={m.key(t):dict(task=t) for t in tasks[:-1]}
        m.atomic_new(d/'jobs.json',dict(cases=jobs,controls=[]));backend=_original.FakeQueue(m,jobs,[(0,0,'')]);backend.done=old.copy()
        seen=[];run_queue=m.schedule_queue
        def queue(selected,control,b):seen.extend(selected);return run_queue(selected,control,b)
        with patch.object(m,'verify'),patch.object(m,'check_control_gate'),patch.object(m,'_frozen_records',return_value=old),\
             patch.object(m,'native_class_records',return_value={}),patch.object(m,'Backend',return_value=backend),\
             patch.object(m,'schedule_queue',side_effect=queue):
            state=m.schedule(d,False)
        self.assertEqual([j['name'] for j in seen],['amos_0040'])
        self.assertEqual([j['name'] for j in backend.started],['amos_0040'])
        self.assertEqual(backend.clock,10)
        self.assertEqual((state['pending'],state['running'],state['failed'],state['committed_classes']),(0,{},[],41))
        self.assertEqual(len(m.read(d/'jobs.json')['cases']),41)

    def test_fully_reused1485_queue_finishes_without_gpu_query_or_sleep(self):
        m=self.m;d=Path(tempfile.mkdtemp())
        tasks=[dict(self.task,case_id=f'{c:04}',global_class_id=201+i,local_class_id=i+1) for c in range(100) for i in range(14 if c<15 else 15)]
        jobs=m.native_case_jobs(tasks);old={m.key(t):dict(task=t) for t in tasks}
        m.atomic_new(d/'jobs.json',dict(cases=jobs,controls=[]))
        with patch.object(m,'verify'),patch.object(m,'check_control_gate'),patch.object(m,'_frozen_records',return_value=old),\
             patch.object(m,'native_class_records',return_value={}),patch.object(m,'Backend') as backend,patch.object(m,'schedule_queue') as queue:
            state=m.schedule(d,False)
        backend.assert_not_called();queue.assert_not_called()
        self.assertEqual({k:state[k] for k in ('phase','pending','running','failed','committed_classes')},dict(phase='full',pending=0,running={},failed=[],committed_classes=1485))
        self.assertEqual(m.read(d/'status.json'),state)


if __name__=='__main__':unittest.main()
