import importlib
import json
from pathlib import Path
import tempfile
import unittest

class ControllerTests(unittest.TestCase):
    def module(self):
        try: return importlib.import_module('tools.five_source_prompt_type_eval')
        except ModuleNotFoundError: self.fail('five-source controller missing')

    def test_frozen_inventory_requires_exact_unique_task_counts(self):
        m = self.module()
        from tools.five_source_prompt_type_runtime import SOURCES
        tasks = [dict(source_name=s,case_id='case'+str(i),global_class_id=1,class_name='x') for s,c in SOURCES.items() for i in range(c['count'])]
        inventory = dict(schema='five_source_prompt_type_inventory_v1',tasks=tasks,refs=[dict(t,prompt_cache_name='p.npz') for t in tasks],input_sha256={'a':'0'*64},paths={})
        self.assertEqual(len(m.validate_inventory(inventory)),173)
        inventory['tasks'] = tasks[:-1]+[tasks[0]]
        with self.assertRaises(ValueError): m.validate_inventory(inventory)

    def test_atomic_new_record_does_not_replace_existing(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'record.json'
            m.atomic_new(path,{'a':1})
            with self.assertRaises(FileExistsError): m.atomic_new(path,{'a':2})
            self.assertEqual(json.loads(path.read_text()),{'a':1})

    def test_control_uses_fresh_matching_batch_and_native_stages(self):
        m = self.module()
        from tools.five_source_prompt_type_runtime import METRICS
        row = {n:.5 for n in METRICS}; row.update(mask_threshold_used=.6,dynamic_threshold_source='fallback')
        record = dict(row=row,actual_frame_batch=4,roi={n:[1,2] for n in ('initial','reserved','final','history')},stages=[{'frames':2}])
        self.assertTrue(m.compare_control(record,record)['passed'])
        other = dict(record,actual_frame_batch=2)
        with self.assertRaises(ValueError): m.compare_control(record,other)

class GateTests(unittest.TestCase):
    def test_matching_source_batch_gate_required(self):
        from tools import five_source_prompt_type_eval as m
        self.assertTrue(hasattr(m,'require_batch_gate'),'matching-batch gate missing')
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            m.atomic_new(out/'controls/magic/batch_4/control_pass.json',dict(passed=True,source='magic',actual_frame_batch=4,protocol_sha256='abc'))
            m.require_batch_gate(out,'magic',4,'abc')
            with self.assertRaises((ValueError,FileNotFoundError)): m.require_batch_gate(out,'magic',2,'abc')
            with self.assertRaises(ValueError): m.require_batch_gate(out,'magic',4,'changed')

class NativeManifestTests(unittest.TestCase):
    def test_magic_case_ids_are_encoded_without_changing_identity(self):
        from tools import five_source_prompt_type_eval as m
        names=[m.case_name('magic','pancreas/'+str(i)) for i in range(35)]
        self.assertEqual(len(set(names)),35)
        self.assertTrue(all('/' not in name and '\\' not in name for name in names))
        self.assertNotEqual(m.case_name('magic','pancreas/13'),m.case_name('magic','pancreas%2F13'))
        self.assertIn('%2F',names[13])

    def test_native_prepared_manifest_contract(self):
        from tools import five_source_prompt_type_eval as m
        self.assertTrue(hasattr(m,'native_manifest'),'native manifest builder missing')
        task=dict(source_name='magic',case_id='pancreas/13',global_class_id=128)
        value=m.native_manifest([task])
        self.assertEqual(value['manifest_kind'],'independent_eight_source_validation_v1')
        self.assertEqual(value['expected_task_count'],1)
        self.assertEqual(value['per_source_task_counts'],{'magic':1})


class ScopedInventoryTests(unittest.TestCase):
    def test_worker_hashes_common_owncase_and_only_own_selected_checkpoint(self):
        from tools import five_source_prompt_type_eval as m
        from tools.five_source_prompt_type_runtime import SOURCES
        self.assertTrue(hasattr(m,'worker_inputs'),'scoped worker inventory missing')
        paths={'code.py':'a','casea.npy':'b','caseb.npy':'c',**{c['checkpoint']:'d' for c in SOURCES.values()}}
        inventory=dict(input_sha256=paths,case_inputs={json.dumps(['magic','pancreas/13']):['casea.npy'],json.dumps(['parse2022','b']):['caseb.npy']})
        actual=m.worker_inputs(inventory,'magic','pancreas/13')
        self.assertEqual(set(actual),{'code.py','casea.npy',SOURCES['magic']['checkpoint']})
        with self.assertRaises(ValueError): m.worker_inputs(inventory,'magic','unknown')


if __name__ == '__main__': unittest.main()
