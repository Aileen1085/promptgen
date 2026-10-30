"""Real native metadata startup contracts for the additive manifest repair."""
import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'tools/amos_prompt_ablation_eval.py'
NATIVE=ROOT/'v10/infer/infer_eight_source_adaptive_v10_2.py'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class NativeManifestTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((ROOT/'tools/amos_prompt_ablation_native_manifest_eval.py').exists(),'add native-contract checked manifest entry')
        self.m=importlib.import_module('tools.amos_prompt_ablation_native_manifest_eval');self.m._ENGINE=None

    def task(self):
        return dict(source_name='amos',case_id='0001',global_class_id=201,local_class_id=1,
                    class_name='spleen',image='data/ct.nii.gz',label='data/gt.nii.gz',split_role='validation')

    def test_actual_native_loader_rejects_original_minimal_manifest_then_accepts_payload(self):
        from infer.infer_eight_source_adaptive_v9_2 import load_independent_manifest
        d=Path(tempfile.mkdtemp());path=d/'manifest.json';task=self.task()
        path.write_text(json.dumps(dict(tasks=[task],expected_task_count=1)))
        with self.assertRaisesRegex(ValueError,'independent eight-source'):load_independent_manifest(path,sources={'amos'})
        payload=self.m.native_manifest_payload(tasks=[task],expected_task_count=1)
        path.write_text(json.dumps(payload));self.assertEqual(load_independent_manifest(path,sources={'amos'}),[task])
        self.assertEqual(payload['manifest_kind'],'independent_eight_source_validation_v1')
        with self.assertRaises(ValueError):self.m.native_manifest_payload(tasks=[task],expected_task_count=2)

    def test_exact_four_manifest_replacements_and_one_guard_insertion_only(self):
        m=self.m;tree,proof=m.transform_controller(OLD.read_bytes(),str(OLD),m.NATIVE_CONTROLLER_SHA256)
        self.assertEqual((proof['manifest_replacements'],proof['audit_insertions']),(4,1))
        restored=ast.fix_missing_locations(m.reverse_transform(tree))
        self.assertEqual(ast.dump(restored,include_attributes=False),ast.dump(ast.parse(OLD.read_bytes()),include_attributes=False))
        changed=OLD.read_bytes().replace(b'dict(tasks=tasks,expected_task_count=1485)',b'dict(tasks=tasks,other_count=1485)',1)
        with self.assertRaisesRegex(ValueError,'four|4'):m.transform_controller(changed,'changed.py',hashlib.sha256(changed).hexdigest())
        with self.assertRaisesRegex(ValueError,'SHA'):m.transform_controller(OLD.read_bytes(),str(OLD),'0'*64)

    def fixture(self,valid=True):
        m=self.m;d=Path(tempfile.mkdtemp());native=d/'v10/infer/infer_eight_source_adaptive_v10_2.py'
        native.parent.mkdir(parents=True);native.write_bytes(NATIVE.read_bytes())
        runtime=importlib.import_module('tools.amos_prompt_ablation_restore_runtime')
        split=dict(source_order=list(runtime.EXPECTED_SOURCE_ORDER),sampling_ratios={s:1/13 for s in runtime.EXPECTED_SOURCE_ORDER},
                   datasets={'amos':dict(train=[],val=[])},class_catalog=[])
        path=d/'configs/ct13_v10_2_split_20260928.json';path.parent.mkdir();path.write_text(json.dumps(split))
        task=self.task();out=d/m.OUT;prepared=out/'prepared/amos_0001';prepared.mkdir(parents=True)
        payload=m.native_manifest_payload(tasks=[task],expected_task_count=1) if valid else dict(tasks=[task],expected_task_count=1)
        (prepared/'manifest.json').write_text(json.dumps(payload))
        (prepared/'refs.json').write_text(json.dumps(dict(per_case_class=[dict(task,prompt_protocol='v9_2_adaptive_only_r1_20_f0p15_final_trigger',prompt_cache_name='cached.npz')])))
        job=dict(name='amos_0001',case_id='0001',control=False,pending=[task],task_count=1,frame_batch=4,prepared=str(prepared))
        return d,out,native,job,task

    def test_real_native_pre_cuda_prefix_passes_and_audits_no_weights_or_gpu(self):
        m=self.m;d,out,native,job,task=self.fixture();before=os.environ.get('CUDA_VISIBLE_DEVICES');argv=sys.argv[:]
        proof=m.run_native_bootstrap(root=d,out=out,jobs=[job],controls=[],strict=False)
        self.assertEqual(proof['job_count'],1);self.assertEqual(proof['jobs'][0]['actual_task_keys'],[['amos','0001',201]])
        self.assertEqual(proof['jobs'][0]['actual_prompt_keys'],[['amos','0001',201]])
        self.assertTrue(proof['ct13_preload']['native_ratios_passed']);self.assertFalse(proof['weights_opened'])
        self.assertFalse(proof['model_loaded']);self.assertFalse(proof['cuda_initialized'])
        self.assertEqual(os.environ.get('CUDA_VISIBLE_DEVICES'),before);self.assertEqual(sys.argv,argv)
        self.assertEqual(proof['native_source_sha256'],sha(native));self.assertFalse((d/'v10/output').exists())

    def test_actual_bootstrap_refuses_bad_manifest_prompt_protocol_and_split_ratios(self):
        m=self.m;d,out,native,job,task=self.fixture(False)
        with self.assertRaisesRegex(ValueError,'independent eight-source'):m.run_native_bootstrap(root=d,out=out,jobs=[job],controls=[],strict=False)
        prepared=Path(job['prepared']);(prepared/'manifest.json').write_text(json.dumps(m.native_manifest_payload(tasks=[task],expected_task_count=1)))
        refs=json.loads((prepared/'refs.json').read_text());refs['per_case_class'][0]['prompt_protocol']='wrong';(prepared/'refs.json').write_text(json.dumps(refs))
        with self.assertRaisesRegex(ValueError,'prompt protocol'):m.run_native_bootstrap(root=d,out=out,jobs=[job],controls=[],strict=False)
        split_path=d/'configs/ct13_v10_2_split_20260928.json';split=json.loads(split_path.read_text());split['sampling_ratios']={};split_path.write_text(json.dumps(split))
        with self.assertRaisesRegex(ValueError,'sampling ratios'):m.run_native_bootstrap(root=d,out=out,jobs=[job],controls=[],strict=False)

    def test_native_prefix_structure_is_source_checked_and_stops_before_cuda_assignment(self):
        m=self.m;source=NATIVE.read_bytes();program=m.compile_bootstrap(source,str(NATIVE),sha(NATIVE))
        self.assertEqual(program['last_statement'],'split, validation = build_selected_protocol(base_split, tasks)')
        self.assertNotIn('CUDA_VISIBLE_DEVICES',program['prefix_source'])
        self.assertNotIn('torch.load',program['prefix_source'])
        with self.assertRaisesRegex(ValueError,'SHA'):m.compile_bootstrap(source,str(NATIVE),'0'*64)

    def test_dispatch_single_globals_and_independent_controls_remain_required(self):
        m=self.m;ns=m.engine();old=importlib.import_module('tools.amos_prompt_ablation_eval');d=Path(tempfile.mkdtemp())
        for name in ('VARIANTS','ALLOWED','METRICS','WEIGHT_SHA','TRAINING_SHA','OLD_SHA'):self.assertEqual(ns[name],getattr(old,name))
        for name in ('worker','class_records','schedule_queue','merge','main'):self.assertIs(ns[name].__globals__,ns)
        job=dict(log=str(d/'worker.log'),spec=str(d/'spec.json'),gpu=2)
        with patch.object(ns['subprocess'],'Popen',return_value=types.SimpleNamespace(pid=1)) as spawn:
            process,handle=ns['Backend'](d,False).spawn(job);handle.close()
        self.assertEqual(Path(spawn.call_args.args[0][1]),ROOT/'tools/amos_prompt_ablation_native_manifest_eval.py')
        ns['atomic_new'](d/'protocol.json',{'frozen':True})
        with patch.dict(ns,check_controls=lambda out:dict(digest='real GPU control')):
            with self.assertRaises(FileNotFoundError):ns['check_control_gate'](d)
        self.assertEqual(sha(OLD),m.NATIVE_CONTROLLER_SHA256)

    def test_all101_real_native_bootstrap_jobs_and_immutable_audit_are_checked(self):
        m=self.m;d,out,native,initial,task=self.fixture();jobs=[];all_tasks=[]
        for case in range(100):
            tasks=[dict(task,case_id=f'{case:04}',global_class_id=201+i,local_class_id=i+1,class_name=f'organ{i}',
                        image=f'data/{case:04}/ct',label=f'data/{case:04}/gt') for i in range(14 if case<15 else 15)]
            all_tasks.extend(tasks);name=f'amos_{case:04}';prepared=out/'prepared'/name;prepared.mkdir(exist_ok=True)
            (prepared/'manifest.json').write_text(json.dumps(m.native_manifest_payload(tasks=tasks,expected_task_count=len(tasks))))
            (prepared/'refs.json').write_text(json.dumps(dict(per_case_class=[dict(t,prompt_protocol='v9_2_adaptive_only_r1_20_f0p15_final_trigger',prompt_cache_name=f'{case}_{i}.npz') for i,t in enumerate(tasks)])))
            jobs.append(dict(name=name,case_id=f'{case:04}',control=False,pending=tasks,task_count=len(tasks),frame_batch=4,prepared=str(prepared)))
        path=d/'configs/ct13_v10_2_split_20260928.json';split=json.loads(path.read_text())
        split['datasets']['amos']['val']=[dict(case_id=j['case_id']) for j in jobs];path.write_text(json.dumps(split))
        control=dict(jobs[0],name='amos_native_control',control=True,pending=all_tasks[:1],task_count=1,prepared=str(out/'prepared/amos_native_control'))
        prepared=Path(control['prepared']);prepared.mkdir()
        (prepared/'manifest.json').write_text(json.dumps(m.native_manifest_payload(tasks=control['pending'],expected_task_count=1)))
        refs=json.loads((Path(jobs[0]['prepared'])/'refs.json').read_text());refs['per_case_class']=refs['per_case_class'][:1]
        (prepared/'refs.json').write_text(json.dumps(refs))
        proof=m.run_native_bootstrap(root=d,out=out,jobs=jobs,controls=[control])
        self.assertEqual((proof['job_count'],proof['full_tasks'],proof['control_tasks']),(101,1485,1))
        ns=m.engine()
        with patch.dict(ns,ROOT=d):
            audit_path=ns['_native_bootstrap_preflight'](out,jobs,[control])
        self.assertTrue(audit_path.is_file());self.assertEqual(json.loads(audit_path.read_text())['job_count'],101)
        with patch.dict(ns,ROOT=d):self.assertEqual(ns['file_inventory']([audit_path])[str(audit_path.relative_to(d)).replace('\\','/')],sha(audit_path))
        with self.assertRaises(FileExistsError):
            with patch.dict(ns,ROOT=d):ns['_native_bootstrap_preflight'](out,jobs,[control])

    def failure_fixture(self):
        m=self.m;prior=importlib.import_module('tools.amos_prompt_ablation_restore_eval');d=Path(tempfile.mkdtemp())
        (d/'dependency_source.py').write_text('frozen source dependency\n')
        for name in ('tools/amos_prompt_ablation_eval.py','tools/amos_prompt_ablation_restore_eval.py','tools/amos_prompt_ablation_restore_runtime.py'):
            p=d/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes())
        def metadata(output,files):
            queue=dict(time=1,phase='controls',pending=0,running={},failed=[dict(name='amos_native_control',returncode=1,log=str(d/output/'logs/amos_native_control_a0.log'))],committed_classes=0)
            for name in files:
                p=d/name;p.parent.mkdir(parents=True,exist_ok=True)
                value=json.dumps(queue) if name.endswith('/status.json') else 'immutable failure fixture'
                if name.endswith('/protocol.json'):value=json.dumps(dict(sources={'dependency_source.py':sha(d/'dependency_source.py')}))
                p.write_text(value)
            return dict(model_loaded=False,controls_passed=False,full_started=False,committed_classes=0,queue=queue,files={name:sha(d/name) for name in files})
        first=metadata(prior.OLD_OUTPUT,prior.FAILURE_FILES);first['related_after_independent_second_check']=[]
        second=metadata(m.SECOND_OUTPUT,m.SECOND_FILES);second.update(ct13_import_order_passed=True,related_checks=[[],[]])
        first_path=d/prior.BOOTSTRAP_EVIDENCE;first_path.write_text(json.dumps(first));second_path=d/m.SECOND_EVIDENCE;second_path.write_text(json.dumps(second))
        return d,prior,first_path,second_path,first,second

    def test_both_failed_bootstrap_chains_are_verified_before_any_new_mutation(self):
        m=self.m;d,prior,p1,p2,first,second=self.failure_fixture()
        with patch.object(prior,'BOOTSTRAP_SHA256',sha(p1)),patch.object(m,'SECOND_SHA256',sha(p2)):
            self.assertEqual(m.verify_failures(d),(first,second))
            changed=d/m.SECOND_FILES[-1];changed.write_text('modified second pipeline')
            with self.assertRaisesRegex(ValueError,'second retained'):m.verify_failures(d)
        ns=m.engine()
        with patch.dict(ns,ROOT=d),patch.object(m,'verify_failures',side_effect=ValueError('changed failure evidence')):
            with self.assertRaisesRegex(ValueError,'failure'):ns['run_pipeline'](d/ns['OUT'])
        self.assertFalse((d/ns['PIPELINE']).exists());self.assertFalse((d/ns['OUT']).exists())

    def test_related_and_frozen_sources_cover_all_three_entries_and_both_evidence_chains(self):
        m=self.m;d=Path(tempfile.mkdtemp())
        for i,name in enumerate(('amos_prompt_ablation_eval.py','amos_prompt_ablation_restore_eval.py','amos_prompt_ablation_native_manifest_eval.py'),100):
            p=d/str(i);p.mkdir();(p/'cmdline').write_bytes(f'python\0{name}\0worker\0'.encode())
        self.assertEqual({p['pid'] for p in m.related(d)},{100,101,102})
        ns=m.engine();self.assertTrue(set(m.SECOND_FILES)<=set(ns['SOURCE_SEEDS']));self.assertIn(m.SECOND_EVIDENCE,ns['SOURCE_SEEDS'])
        with patch.object(m,'RESTORE_RUNTIME_SHA256','0'*64):
            m._ENGINE=None
            with self.assertRaisesRegex(ValueError,'SHA'):m.engine()

    def test_prior_protocol_nested_sources_refuse_changed_native_dependency(self):
        m=self.m;d,prior,p1,p2,first,second=self.failure_fixture();original=sha(d/m.SECOND_FILES[0])
        with patch.object(prior,'BOOTSTRAP_SHA256',sha(p1)),patch.object(m,'SECOND_SHA256',sha(p2)):
            m.verify_failures(d)
            (d/'dependency_source.py').write_text('changed dependency while signed protocols remain unchanged')
            self.assertEqual(sha(d/m.SECOND_FILES[0]),original)
            with self.assertRaisesRegex(ValueError,'source'):m.verify_failures(d)


if __name__=='__main__':unittest.main()
