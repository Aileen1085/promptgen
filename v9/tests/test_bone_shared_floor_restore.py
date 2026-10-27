import importlib
import sys
import unittest
from types import SimpleNamespace, ModuleType
from unittest.mock import patch


class NativeRestoreOrderTests(unittest.TestCase):
    def test_native_batch_pairs_are_exact_and_batch_setting_is_restored(self):
        import numpy as np
        m=importlib.import_module('tools.bone_shared_floor_restore_runtime')
        args=SimpleNamespace(sam2_frame_batch_size=4)
        def forward():
            batch=args.sam2_frame_batch_size
            return np.full((1,),batch,dtype=np.float32),['encoder-'+str(batch)]
        records=m.probe_native_batches(forward,forward,args)
        self.assertEqual(set(records),{1,2,4})
        self.assertEqual(args.sam2_frame_batch_size,4)
        self.assertTrue(all(r['raw_sha']==r['native_raw_sha'] for r in records.values()))
        def wrong():return np.zeros(1,dtype=np.float32),['wrong']
        with self.assertRaises(ValueError):m.probe_native_batches(forward,wrong,args)
        self.assertEqual(args.sam2_frame_batch_size,4)

    def test_actual_attempt_guard_allows_control4_and_rejects_production4(self):
        from pathlib import Path
        module=importlib.import_module('tools.bone_shared_floor_restore_grid').engine()
        out=Path('out')
        canonical=dict(name='control',model='v10_2',control=True,parent_history=[],
            variant='baseline',source='bone',case_id='c',prepared='p',
            frame_batch=4,minimum_free_mib=30720)
        job=dict(canonical,gpu=2,gpu_uuid='GPU-actual',attempt=0,oom_retries=0,
            run=str(out/'runs/control_a0'),spec=str(out/'job_specs/control_a0.json'),
            log=str(out/'logs/control_a0.log'))
        module.validate_attempt(job,canonical,out)
        with self.assertRaises(ValueError):
            module.validate_attempt(dict(job,control=False),dict(canonical,control=False),out)

    def test_recovery_keeps_production_cap_but_uses_verified_control_batch(self):
        import ast
        from pathlib import Path
        module=importlib.import_module('tools.bone_shared_floor_restore_grid')
        source=Path('tools/bone_shared_floor_grid.py').read_text(encoding='utf-8')
        tree=ast.parse(module.recovery_source(source))
        prepare=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='prepare')
        self.assertIn('frame_batch=4',ast.unparse(prepare))
        cases=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='case_jobs')
        self.assertIn('frame_batch=min([2] +',ast.unparse(cases))
        self.assertIn('bone_shared_floor_restore_grid.py',ast.unparse(tree))

    def test_decoder_is_verified_after_native_restore_before_first_class(self):
        import torch
        runtime = importlib.import_module('tools.bone_shared_floor_restore_runtime')
        prompt = torch.nn.Linear(1, 1, bias=False)
        decoder = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            prompt.weight.fill_(0)
            decoder.weight.fill_(0)
        state = dict(epoch=360, prompt_generator={'weight': torch.ones(1, 1)},
                     sam2_mask_decoder={'weight': torch.full((1, 1), 2.)})
        base = SimpleNamespace()
        def native_load(path, module, **kwargs):
            module.load_state_dict(state['prompt_generator'])
            return state, None
        base.load_prompt_generator_checkpoint = native_load
        entry = ModuleType('finetune_multisource_sam2_v9_2_precision');entry.base = base
        joint = ModuleType('finetune_totalseg_sam2_promptgen_v9_joint')
        class Adapter:
            def __init__(self):self.sam = SimpleNamespace(sam_mask_decoder=decoder)
        joint.SAM2MedicalAdapterV9 = Adapter
        evaluator = ModuleType('infer.infer_eight_source_adaptive_v9_2')
        evaluator._install_exact_prompt_factory = lambda *a, **k: None
        modules = {m.__name__: m for m in (entry, joint, evaluator)}
        begun=[]
        session = SimpleNamespace(model='v9_2', restoration=None,
            training_audit={}, begin=lambda task: begun.append(task) or ('began', task))
        saved=[]
        with patch.dict(sys.modules, modules):
            runtime.install_restoration(session, saved.append)
            joint.SAM2MedicalAdapterV9()
            base.load_prompt_generator_checkpoint('unused', prompt)
            self.assertFalse(saved, 'decoder is not loaded by prompt checkpoint loader')
            with self.assertRaises(ValueError):session.begin('unrestored')
            self.assertFalse(saved)
            self.assertFalse(begun, 'no evaluation may begin with unrestored decoder')
            decoder.load_state_dict(state['sam2_mask_decoder'])
            self.assertEqual(session.begin('class-a'), ('began', 'class-a'))
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved[0]['prompt_parameters_exact'], 1)
            self.assertEqual(saved[0]['decoder_parameters_exact'], 1)
            self.assertEqual(session.begin('class-b'), ('began', 'class-b'))
            self.assertEqual(len(saved), 1, 'initial restoration proof is immutable')


if __name__ == '__main__':unittest.main()
