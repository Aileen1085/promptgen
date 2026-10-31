"""Fresh native module controller: isolated globals, atomic classes and gate."""
import importlib
from pathlib import Path
import tempfile
import json
import unittest
from unittest.mock import patch

from tests import test_amos_prompt_ablation_eval as original


class Proxy:
    def __init__(self,ns):object.__setattr__(self,'ns',ns)
    def __getattr__(self,name):
        try:return self.ns[name]
        except KeyError:raise AttributeError(name) from None
    def __setattr__(self,name,value):self.ns[name]=value
    def __delattr__(self,name):
        try:del self.ns[name]
        except KeyError:raise AttributeError(name) from None


def controller():return Proxy(importlib.import_module('tools.amos_bg_scribble_module_eval').compile_context().engine())


class ModuleControllerTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((Path(__file__).resolve().parents[1]/'tools/amos_bg_scribble_module_eval.py').exists(),'add module controller')
        self.entry=importlib.import_module('tools.amos_bg_scribble_module_eval');self.m=controller()

    def test_main_worker_runtime_and_spawn_use_current_private_namespace(self):
        ns=self.m.ns
        for name in ('main','worker','make_attempt','merge','schedule_queue','class_records','check_controls','check_control_gate'):
            self.assertIs(ns[name].__globals__,ns,name)
        self.assertEqual(ns['VARIANTS'],self.entry.MODULE_VARIANTS)
        self.assertEqual(ns['runtime']().__name__,'tools.amos_bg_scribble_module_runtime')
        self.assertEqual(ns['PROMPT_VARIANT'],'fg_scribble_bg_scribble')
        self.assertNotIn('native_class_records',ns)
        self.assertFalse(any('reuse_audit' in name or 'frozen_records' in name for name in ns))
        d=Path(tempfile.mkdtemp());job=dict(log=str(d/'a.log'),spec=str(d/'a.json'),gpu=2)
        with patch.object(ns['subprocess'],'Popen',return_value=object()) as spawn:
            _,handle=ns['Backend'](d,False).spawn(job);handle.close()
        self.assertEqual(Path(spawn.call_args.args[0][1]),Path(self.entry.__file__))

    def test_changed_source_refused_and_native_bootstrap_uses_current_engine(self):
        with self.assertRaisesRegex(ValueError,'SHA'):self.entry.compile_context(source=b'pass')
        self.assertIs(self.m._bootstrap_function.__globals__['engine'](),self.m.ns)
        self.assertIn("'scribble'",self.m.BOOTSTRAP_SOURCE)
        self.assertIs(self.m.TRANSFORM_PROOF.get('bootstrap_reverse_ast_equal'),True)

    def test_prior_selected_reference_requires_batch4_same_key_metrics_roi_and_stages(self):
        native=dict(actual_frame_batch=4,row=dict({name:.7 for name in self.m.METRICS},source_name='amos',case_id='a',global_class_id=201),roi={'initial':[1],'reserved':[1,2],'final':[1],'history':[]},stages=[{'frames':4}],
                    prompt_variant=self.m.PROMPT_VARIANT)
        self.m.compare_selected_reference(native,native)
        import copy
        for field in ('batch','metric','roi','stages'):
            bad=copy.deepcopy(native)
            if field=='batch':bad['actual_frame_batch']=2
            if field=='metric':bad['row']['dice']+=1e-5
            if field=='roi':bad['roi']['final']=[2]
            if field=='stages':bad['stages'][0]['frames']=3
            with self.assertRaises(ValueError):self.m.compare_selected_reference(native,bad)

    def test_module_reference_is_completed_bg_scribble_not_old_floor035_or_bgpoint(self):
        self.assertEqual(self.m.OLD,self.entry.SELECTED_MERGED)
        self.assertEqual(self.m.OLD_SHA,'5e2bdbf2fa340c69b9d35bd079ed3d04332d95f9b1eea5871111c3ee55cf8777')
        self.assertEqual(self.m.OUT,'output/amos_v102_e490_floor070_bg_scribble_modules_20261031')


class ModuleSchedulerTests(original.SchedulerTests):
    def setUp(self):
        self.m=controller();self.task=dict(source_name='amos',case_id='0001',global_class_id=201,local_class_id=1,class_name='spleen')


class ModuleWorkerTests(original.WorkerTests):
    def setUp(self):
        self.m=controller();self.task=dict(source_name='amos',case_id='0001',global_class_id=201,local_class_id=1,class_name='spleen',image='data/a.nii.gz',label='data/b.nii.gz')


class ModuleAtomicPolicyTests(original.ControllerTests):
    def setUp(self):
        self.m=controller();self.task=dict(source_name='amos',case_id='0001',global_class_id=201,local_class_id=1,class_name='spleen')

    def test_protocol_file_checksum_refuses_policy_or_inventory_mutation(self):
        with patch.object(self.m,'_source_guard'):
            super().test_protocol_file_checksum_refuses_policy_or_inventory_mutation()

    def test_prepare_freezes_original_100_case_scope_and_explicit_source_inventory(self):
        m=self.m;d=Path(tempfile.mkdtemp());out=d/m.OUT;tasks=self.tasks()
        catalog=[dict(dataset='amos',global_class_id=201+i,local_class_id=1+i,class_name=f'organ{i}') for i in range(15)]
        for t in tasks:t.update(class_name=catalog[t['local_class_id']-1]['class_name'],image=f'data/{t["case_id"]}/ct',label=f'data/{t["case_id"]}/gt')
        split=dict(class_catalog=catalog,source_order=['amos'],datasets={'amos':dict(train=[dict(case_id='train')],val=[])})
        for c in range(100):
            selected=[t for t in tasks if t['case_id']==f'{c:04}']
            split['datasets']['amos']['val'].append(dict(case_id=f'{c:04}',image=selected[0]['image'],label=selected[0]['label'],classes=[t['global_class_id'] for t in selected]))
        refs=[dict(source_name='amos',case_id=t['case_id'],global_class_id=t['global_class_id'],prompt_cache_name=f'p{i}.npz') for i,t in enumerate(tasks)]
        m.atomic_new(d/m.TRAINING,split);m.atomic_new(d/m.HISTORY/'manifest.json',dict(tasks=tasks))
        previous=[self.row(t) for t in tasks]
        m.atomic_new(d/m.OLD,dict(variants={m.PROMPT_VARIANT:dict(per_case_class=previous)},rows=8910,floor=.70,weights='E490live'))
        m.atomic_new(d/m.SELECTED_AUDIT,dict(complete=True,data_rehash=True,signature_and_real_controls_pass=True,scheduler_workers_absent=True))
        selected_proof=d/'selected/class-proof.json';m.atomic_new(selected_proof,dict(independent_prior_fixture=True))
        files=[d/n for n in m.SOURCE_SEEDS]
        for f in files:f.parent.mkdir(parents=True,exist_ok=True);f.write_text('pass\n')
        for name in (m.WEIGHTS,'sam2/checkpoints/sam2.1_hiera_large.pt','models/sat-medical-text/text_encoder.pth','models/biolord-2023-c/config.json'):
            f=d/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_text('test bytes only')
        with patch.object(m,'ROOT',d),patch.object(m,'_source_guard'),patch.object(m,'TRAINING_SHA',m.sha(d/m.TRAINING)),\
             patch.object(m,'WEIGHT_SHA',m.sha(d/m.WEIGHTS)),patch.object(m,'OLD_SHA',m.sha(d/m.OLD)),\
             patch.object(m,'SELECTED_AUDIT_SHA256',m.sha(d/m.SELECTED_AUDIT)),\
             patch.object(m,'selected_control',return_value=dict(task=tasks[0],record={},files={str(selected_proof):m.sha(selected_proof)})),\
             patch.object(m,'prompt_references',return_value=(refs,[])),\
             patch.object(m,'data_and_cache_inventory',return_value=dict(dataset={},prompts={},shared_ct={})),\
             patch.object(m,'dependency_sources',return_value=files),patch.object(m,'_bootstrap_function',return_value={'cpu_fixture':True}):
            m.prepare(out,already_checked=True);m.verify(out)
            protocol=m.read(out/'protocol.json');jobs=m.read(out/'jobs.json')
            self.assertEqual((len(jobs['cases']),sum(j['task_count'] for j in jobs['cases'])),(100,1485))
            self.assertEqual(protocol['policy']['variants'],list(m.VARIANTS));self.assertEqual(protocol['policy']['prompt_variant'],m.PROMPT_VARIANT)
            self.assertEqual(protocol['policy']['fresh_module_rows'],8910);self.assertEqual(set(protocol['sources']),set(m.SOURCE_SEEDS))
            self.assertIn(str(selected_proof.relative_to(d)).replace('\\','/'),protocol['reference'])
            (d/'later_unrelated.py').write_text('unrelated');m.verify(out)
            # Re-signing a changed policy cannot bypass the semantic protocol check.
            changed=dict(protocol);changed['policy']=dict(protocol['policy'],prompt_variant='baseline')
            (out/'protocol.json').write_text(json.dumps(changed));(out/'protocol_signature.json').write_text(json.dumps({'sha256':m.sha(out/'protocol.json')}))
            with self.assertRaisesRegex(ValueError,'protocol'):m.verify(out)
            (out/'protocol.json').write_text(json.dumps(protocol));(out/'protocol_signature.json').write_text(json.dumps({'sha256':m.sha(out/'protocol.json')}))
            (d/m.SOURCE_SEEDS[0]).write_text('modified controller')
            with self.assertRaises(ValueError):m.verify(out)

    def test_merge_excludes_controls_keeps_exact8910_rows_and_pairs_both_baselines(self):
        m=self.m;d=Path(tempfile.mkdtemp());tasks=self.tasks();found={};reference=[]
        for t in tasks:
            rows=[]
            for i,v in enumerate(m.VARIANTS):
                row=self.row(t);row.update({name:.4+.01*i for name in m.METRICS})
                rows.append(dict(variant=v,row=row))
            found[m.key(t)]=dict(record=dict(records=rows),spec=dict(frame_batch=4))
            reference.append(dict(rows[0]['row']))
        m.atomic_new(d/'manifest.json',dict(tasks=tasks));m.atomic_new(d/'jobs.json',dict(cases=[],controls=[]))
        m.atomic_new(d/'status.json',dict(phase='full',pending=0,running={},failed=[]))
        m.atomic_new(d/'protocol.json',dict(frozen=True));m.atomic_new(d/'control_verification.json',dict(passed=True))
        with patch.object(m,'verify'),patch.object(m,'check_control_gate'),patch.object(m,'related',return_value=[]),\
             patch.object(m,'class_records',return_value=found),patch.object(m,'native_reference',return_value=reference):
            merged=m.merge(d);sha=m.sha(d/'merged_metrics.json')
            self.assertEqual((merged['rows'],merged['fresh_rows'],merged['reused_rows']),(8910,8910,0))
            self.assertFalse(merged['controls_in_final']);self.assertEqual(set(merged['variants']),set(m.VARIANTS))
            self.assertEqual(sum(v['task_count'] for v in merged['variants'].values()),8910)
            for i,v in enumerate(m.VARIANTS):
                part=merged['variants'][v];self.assertEqual(len(part['paired_deltas']),1485)
                self.assertAlmostEqual(part['delta_from_fresh_baseline']['dice'],i*.01,places=14)
                self.assertAlmostEqual(part['delta_from_selected_prompt_baseline']['dice'],i*.01,places=14)
            self.assertEqual(m.merge(d),merged);self.assertEqual(m.sha(d/'merged_metrics.json'),sha)
            first=next(iter(found));removed=found.pop(first)
            with self.assertRaisesRegex(ValueError,'fresh'):m.merge(d)
            found[first]=removed;removed['record']['records'][0]['row']['dice']=.99
            with self.assertRaisesRegex(ValueError,'existing'):m.merge(d)
            self.assertEqual(m.sha(d/'merged_metrics.json'),sha)

    def test_cases_keep_six_variants_fresh_and_controls_excluded(self):
        with patch.object(self.m,'selected_control',return_value={'task':self.tasks()[0]}):
            super().test_cases_keep_six_variants_fresh_and_controls_excluded()


if __name__=='__main__':unittest.main()
