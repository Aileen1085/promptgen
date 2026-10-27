import ast
import importlib
import unittest
from pathlib import Path
from types import SimpleNamespace


class RuntimeTransformTests(unittest.TestCase):
    def setUp(self):
        self.m = importlib.import_module('tools.bone_shared_floor_runtime')

    def test_v92_extraction_preserves_postprocessing_and_one_raw_prefix(self):
        source = Path('infer/v9_2_dynamic_validation_patch.py').read_text(encoding='utf-8')
        changed = self.m.v92_source(source)
        tree = ast.parse(changed)
        functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        raw = next(n for n in functions if n.name == '_shared_raw_v92')
        full = next(n for n in functions if n.name == 'dynamic_full_volume_task')
        self.assertIn('encode_image_features_in_chunks', ast.unparse(raw))
        self.assertNotIn('calibrate_prompt_logits', ast.unparse(raw))
        self.assertNotIn('encode_image_features_in_chunks', ast.unparse(full))
        self.assertIn('calibrate_prompt_logits', ast.unparse(full))
        self.assertIn('total_frames = int(video.shape[0])', ast.unparse(full))
        self.assertIn('logits_cpu = []', ast.unparse(full))
        self.assertIn('_shared_raw_call', ast.unparse(full))

    def test_v92_transform_rejects_unexpected_native_layout(self):
        with self.assertRaises(ValueError):
            self.m.v92_source('def install_dynamic_validation_patch():\n    pass\n')

    def test_v10_loop_keeps_task_load_once_and_all_native_scoring(self):
        source = Path('v10/infer/infer_eight_source_adaptive_v10_2.py').read_text(encoding='utf-8')
        tree = ast.parse(self.m.v10_main_source(source))
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        task = next(n for n in ast.walk(main) if isinstance(n, ast.For) and ast.unparse(n.target) == 'task')
        loop = next(n for n in task.body if isinstance(n, ast.For) and ast.unparse(n.target) == '_shared_floor')
        self.assertNotIn('get_item_for_class', ast.unparse(loop))
        self.assertNotIn('_load_total_target', ast.unparse(loop))
        self.assertIn('_load_total_target', ast.unparse(task))
        self.assertIn('_dynamic_adaptive_decode', ast.unparse(loop))
        self.assertIn('_shared_score', ast.unparse(loop))
        self.assertIn('_shared_emit_v10', ast.unparse(loop))
        self.assertIn('if _shared_primary():', ast.unparse(loop))
        self.assertNotIn('del video', ast.unparse(loop))
        self.assertIn('del video, bundle, target', ast.unparse(task))

    def test_v10_transform_fails_closed(self):
        with self.assertRaises(ValueError):
            self.m.v10_main_source('def main():\n    pass\n')

    def test_v10_stage_transform_caches_before_postprocessing(self):
        source = Path('v10/infer/infer_eight_source_adaptive_v10_2.py').read_text(encoding='utf-8')
        tree = ast.parse(self.m.v10_decode_source(source))
        raw = self.m._function(tree, '_shared_compute_v10')
        self.assertIn('generate_case_3d_tokens', ast.unparse(raw))
        self.assertIn('decode(', ast.unparse(raw))
        self.assertNotIn('postprocess', ast.unparse(raw))
        self.assertIn('_shared_raw_call', ast.unparse(tree))

    def test_floor_dag_reuses_common_stage_but_diverges(self):
        calls = []
        def forward(state):
            calls.append(state)
            return {'root': .5, 'left': .3, 'right': .7}[state]
        def branch(value, floor):
            return ('left' if floor < value else 'right') if value == .5 else None
        results = self.m.walk_floors((.35, .65), 'root', forward, branch)
        self.assertEqual(calls, ['root', 'left', 'right'])
        self.assertEqual(results, {.35: .3, .65: .7})

    def test_native_carrier_output_is_explicitly_labeled_per_task(self):
        task=dict(source_name='bone',case_id='a',global_class_id=10003)
        payload={'per_case_class':[dict(task,dice=.7)]}
        value=self.m.tag_carrier(payload,[{'task':task,'floors':[.55,.6]}])
        self.assertTrue(value['carrier_only'])
        self.assertEqual(value['per_case_class'][0]['carrier_floor'],.55)
        self.assertTrue(value['per_case_class'][0]['carrier_only'])


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.m = importlib.import_module('tools.bone_shared_floor_runtime')
        import torch
        self.torch = torch
        self.task = dict(source_name='bone',case_id='case1',global_class_id=10003,class_name='test')
        self.saved = []
        self.s = self.m.GridSession('v9_2', [{'task':self.task,'floors':[.35,.65]}], 'weights-source-sha', self.saved.append, 4096, 4096)
        self.s.restoration = dict(model='v9_2',epoch=360,weight_mode='live',prompt_parameters_exact=438,decoder_parameters_exact=192,guards_passed=True)
        self.s.training_audit = dict(training_dataset_constructions_skipped=1)
        self.args = SimpleNamespace(dynamic_threshold_min=.35,dynamic_threshold_max=.75,dynamic_threshold_fallback=.6,prompt_logit_neighborhood_radius=2,model_input_size=1024,max_frames=192,validation_window_frames=192)

    def test_shared_raw_is_floor_independent_and_complete_class_is_atomic(self):
        calls=[];self.s.begin(self.task)
        video=self.torch.ones(2,1,2,2);bundle={'groups':self.torch.ones(1,3,2,2)}
        for floor in (.35,.65):
            self.s.set_floor(floor,self.args)
            value=self.s.raw(lambda: calls.append(1) or video.clone(),video,bundle,10003,self.args)
            self.s.capture(self.args,dict(threshold=max(.6,floor),source='calibrated'))
            row={k:.7 for k in self.m.METRICS}
            self.s.emit(self.task,row,dict(threshold=max(.6,floor),source='calibrated'))
            value.zero_()
        self.assertEqual(len(calls),1)
        self.assertFalse(self.saved)
        self.s.finish(self.task)
        self.assertEqual(len(self.saved),1)
        self.assertEqual(len(self.saved[0]['results']),2)
        self.assertEqual(self.saved[0]['cache_stats']['raw']['hits'],1)

    def test_incomplete_class_must_not_emit_completion(self):
        self.s.begin(self.task)
        with self.assertRaises(ValueError):self.s.finish(self.task)
        self.assertFalse(self.saved)

    def test_stage_memory_context_change_rejected(self):
        self.s.begin(self.task);self.s.set_floor(.35,self.args)
        self.args.model_input_size=512
        with self.assertRaises(ValueError):
            self.s.raw(lambda:self.torch.ones(1),self.torch.ones(2,1,2,2),{},10003,self.args)

    def test_native_v92_postprocess_settings_are_available_before_shared_forward(self):
        self.s.begin(self.task)
        args=SimpleNamespace(model_input_size=1024,max_frames=192,validation_window_frames=192)
        self.s.set_floor(.35,args)
        self.assertEqual(args.dynamic_threshold_max,.75)
        self.assertEqual(args.dynamic_threshold_fallback,.6)
        self.assertEqual(args.prompt_logit_neighborhood_radius,2)

    def test_no_cross_class_raw_cache(self):
        self.s.begin(self.task);self.s.set_floor(.35,self.args)
        self.s.raw(lambda:self.torch.ones(1),self.torch.ones(2,1,2,2),{},10003,self.args)
        self.s.begin(self.task)
        self.assertEqual(self.s.raw_memo.stats['bytes'],0)


if __name__ == '__main__':
    unittest.main()
