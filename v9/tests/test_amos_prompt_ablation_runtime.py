"""CPU contracts for the checked native AMOS runtime, never prediction substitutes."""
import hashlib
import copy
import unittest
from pathlib import Path

import numpy as np

try:
    from tools import amos_prompt_ablation_runtime as runtime
except ImportError:
    runtime = None


MINI = '''
events = []
def _task_key(task):
    return task
def main():
    events.append("load")
    count = 0
    tasks = [1, 2]
    def hook():
        return active
    for task in sorted(tasks, key=_task_key):
        count += 1
        events.append((task, hook()))
        row = {"count": count, "active": active}
    raise RuntimeError("native final aggregation must not execute")
if __name__ == "__main__":
    main()
'''


class NativeSplitTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(runtime, "add the checked native runtime")

    def compile(self, source=MINI):
        return runtime.compile_native(source, "native.py", hashlib.sha256(source.encode()).hexdigest())

    def test_one_initializer_many_unchanged_tasks_in_same_globals(self):
        program = self.compile()
        namespace = program.namespace()
        exec(program.initialization, namespace)
        namespace["active"] = "first"
        namespace["task"] = 1
        exec(program.task_body, namespace)
        namespace["active"] = "second"
        namespace["task"] = 2
        exec(program.task_body, namespace)
        self.assertEqual(namespace["events"], ["load", (1, "first"), (2, "second")])
        self.assertEqual(namespace["row"], {"count": 2, "active": "second"})
        self.assertIs(namespace["hook"].__globals__, namespace)

    def test_hash_mismatch_rejected_before_exec(self):
        with self.assertRaisesRegex(ValueError, "source SHA"):
            runtime.compile_native(MINI, "native.py", "0" * 64)

    def test_ambiguous_or_changed_loop_rejected(self):
        for changed in (
            MINI.replace("sorted(tasks, key=_task_key)", "tasks"),
            MINI.replace('    raise RuntimeError', '    for task in sorted(tasks, key=_task_key):\n        pass\n    raise RuntimeError'),
        ):
            with self.subTest(changed=changed):
                with self.assertRaisesRegex(ValueError, "task loop"):
                    self.compile(changed)

    def test_native_source_task_body_has_exact_ast(self):
        source = (Path(__file__).resolve().parents[1] / "v10/infer/infer_eight_source_adaptive_v10_2.py").read_text()
        program = self.compile(source)
        self.assertTrue(program.task_body_sha256)
        self.assertIn("torch.no_grad()", program.task_body_source)
        self.assertIn("restore_and_score_full_volume", program.task_body_source)
        self.assertNotIn("expected_keys =", program.task_body_source)


class ActualConsumptionTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(runtime, "add actual consumption validation")
        self.groups = np.zeros((1, 3, 8, 8), dtype=np.float32)
        self.groups[0, 2, 2:6, 2] = 1
        self.groups[0, 1, 1, 1] = 1
        self.groups[0, 1, 6, 6] = 1
        self.modes = dict(active_planes=("sagittal",), plane_ids=(0,), foreground_mode="box",
            background_mode="point", foreground_mode_id=1, foreground_type_id=2, background_type_id=4)

    def test_incoming_box_and_single_plane_observed_without_copy(self):
        event = runtime.validate_consumption(self.groups, np.array([0]), 1, self.modes, self.groups)
        self.assertEqual(event["plane_ids"], [0])
        self.assertEqual(event["channel_nonzero"], [[0, 2, 4]])

    def test_phantom_plane_wrong_channel_or_wrong_mode_rejected(self):
        for groups, planes, mode in (
            (np.repeat(self.groups, 2, axis=0), np.array([0, 1]), 1),
            (self.groups, np.array([0]), 0),
            (self.groups[:, [2, 1, 0]], np.array([0]), 1),
        ):
            with self.assertRaises(ValueError):
                runtime.validate_consumption(groups, planes, mode, self.modes, self.groups)

    def test_actual_output_types_and_modes_required(self):
        out = dict(prompt_3d_type_ids=np.array([2, 2, 4, 4]), prompt_3d_role_ids=np.array([0, 0, 1, 1]),
                   foreground_prompt_mode="box", background_prompt_mode="point", foreground_channel=2, background_active=True)
        event = runtime.validate_output_types(out, self.modes)
        self.assertEqual(event["type_ids"], [2, 4])
        out["prompt_3d_type_ids"] = np.array([0, 0, 4, 4])
        with self.assertRaisesRegex(ValueError, "type"):
            runtime.validate_output_types(out, self.modes)

    def test_actual_plane_coordinates_are_checked(self):
        import inspect
        self.assertIn("slice_coords", inspect.signature(runtime.validate_consumption).parameters)
        with self.assertRaisesRegex(ValueError, "slice coordinate"):
            runtime.validate_consumption(self.groups, np.array([0]), 1, self.modes, self.groups,
                slice_coords=np.array([.4]), expected_slice_coords=np.array([.2]))

    def test_decoder_adapter_receives_actual_role_and_type_ids(self):
        self.assertTrue(hasattr(runtime, "validate_adapter_types"), "observe actual decoder adapter types")
        event = runtime.validate_adapter_types(np.array([0, 0, 1]), np.array([2, 2, 4]), self.modes)
        self.assertEqual(event["type_ids"], [2, 4])
        with self.assertRaisesRegex(ValueError, "type"):
            runtime.validate_adapter_types(np.array([0, 0, 1]), np.array([0, 0, 4]), self.modes)


class FrozenProtocolTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(runtime, "add frozen settings validation")

    def test_frozen_settings_checked_on_actual_args(self):
        from types import SimpleNamespace
        args = SimpleNamespace(**runtime.FROZEN_SETTINGS, sam2_frame_batch_size=4)
        self.assertEqual(runtime.validate_settings(args, 4)["eval_memory_group_size"], 1)
        args.dynamic_threshold_min = .35
        with self.assertRaisesRegex(ValueError, "dynamic_threshold_min"):
            runtime.validate_settings(args, 4)

    def test_parameter_values_checked_not_only_key_counts(self):
        class Parameter:
            def __init__(self, value): self.value = value
            def detach(self): return self
            def cpu(self): return self
            def numpy(self): return np.array([self.value], dtype=np.float32)
        class Module:
            def named_parameters(self): return [("a", Parameter(1))]
        state = {"a": np.array([1], dtype=np.float32)}
        self.assertEqual(runtime.compare_parameters(Module(), state), 1)
        state["a"][0] = 2
        with self.assertRaisesRegex(ValueError, "not restored"):
            runtime.compare_parameters(Module(), state)

    def test_actual_parameter_version_mutation_detected_without_tensor_copy(self):
        self.assertTrue(hasattr(runtime, "parameter_guard"), "guard actual native parameter metadata")
        class Parameter:
            _version, shape, dtype, device = 2, (4, 5), "float32", "cuda:0"
            def data_ptr(self): return 17
            def cpu(self): raise AssertionError("per-evaluation guard must not transfer tensors")
        class Module:
            def named_parameters(self): return [("weight", parameter)]
        parameter = Parameter()
        before = runtime.parameter_guard(Module())
        parameter._version += 1
        self.assertNotEqual(runtime.parameter_guard(Module()), before)

    def test_control_requires_all_six_and_native_roi_stage_equality(self):
        from tools.amos_prompt_ablation_prompts import VARIANTS
        native = dict(row={name: .5 for name in runtime.METRICS}, roi={"initial": [1, 2], "final": [1, 2]},
            stages=[{"frames": 2}], actual_frame_batch=4)
        native["row"].update(mask_threshold_used=.7, dynamic_threshold_source="prompt_logit_midpoint")
        records = [{**native, "variant": v} for v in VARIANTS]
        self.assertEqual(runtime.validate_control(native, records)["max_metric_error"], 0)
        records[0] = {**records[0], "roi": {"initial": [0, 1], "final": [1, 2]}}
        with self.assertRaisesRegex(ValueError, "ROI"):
            runtime.validate_control(native, records)

    def record(self):
        from tools.amos_prompt_ablation_prompts import build_variant
        from types import SimpleNamespace
        task = dict(source_name="amos", case_id="case", global_class_id=10001, class_name="organ", local_class_id=1)
        package = dict(coronal=np.zeros((12, 5, 7), np.uint8), sagittal=np.zeros((12, 5, 7), np.uint8),
            coronal_slice=2, sagittal_slice=2,
            background_points_dhw=np.array([[0,0,2],[11,4,2],[0,2,0],[11,2,6]]),
            background_point_planes=["coronal","coronal","sagittal","sagittal"])
        package["coronal"][3:5, 1:4, 2] = 1
        package["sagittal"][3:5, 2, 2:5] = 1
        converted, audit = build_variant(package, {"coronal": np.zeros((3,12,5),np.uint8),
            "sagittal": np.zeros((3,12,7),np.uint8)}, "baseline", 8)
        modes = {k: audit[k] for k in ("foreground_mode", "foreground_mode_id", "foreground_type_id", "background_mode", "background_type_id", "plane_ids")}
        consume = runtime.validate_consumption(converted["groups"], converted["plane_ids"], 0, modes, converted["groups"],
            slice_coords=converted["slice_coords"],expected_slice_coords=converted["slice_coords"])
        types = dict(type_ids=[0,4],role_ids=[0,1],type_counts={"0":2,"4":2})
        args = SimpleNamespace(**runtime.FROZEN_SETTINGS,sam2_frame_batch_size=4)
        stage = dict(frames=4,frame_coords=(2*np.array([2,3,4,5],np.float32)/11-1).astype(float).tolist(),
            bundle=consume,settings=runtime.validate_settings(args,4),prompt_calls=[dict(consume,**types,semantic_text="organ")],
            mixed_calls=[{}],global_pool_calls=[types],decoder_calls=[dict(types,frames=4)],adapter_calls=[types]*4,
            memory_calls={"memory_encoder":4,"memory_attention":3},threshold_audit=dict(threshold=.7,source="prompt_logit_midpoint"))
        audit.update(group_builds=[{}],actual_consumption_calls=1)
        row=dict(source_name="amos",case_id="case",class_id=10001,class_name="organ",mask_threshold_used=.7,
            dynamic_threshold_source="prompt_logit_midpoint",metric_space="raw_nifti_full_volume",**{m:.5 for m in runtime.METRICS})
        restore=dict(model="v10_2",epoch=490,weight_mode="live",prompt_parameters_exact=418,decoder_parameters_exact=212,
            guards_passed=True,model_load_count=1,full_tensor_comparison_phase="post_load",
            per_evaluation_guard="parameter_version_pointer_shape_dtype_device")
        roi=dict(initial=[2,3,4,5],reserved=list(range(8)),final=[2,3,4,5],history=[],
            builds=[dict(depth=12,indices=[2,3,4,5],raw_support=[3,4],context_slices=1,context_ratio=0,minframes=1,
                         source="actual_raw_active_foreground")],
            native_reserved_calls=[dict(initial_bounds=[2,6],reserved_bounds=[0,8])])
        record=dict(variant="baseline",row=row,stages=[stage],prompt=audit,roi=roi,restoration=restore,
            parameter_guard_before={"prompt":"a","decoder":"b"},parameter_guard_after={"prompt":"a","decoder":"b"},parameter_guard_verified=True,
            actual_frame_batch=4,native_source_sha256="a"*64,task_body_ast_sha256="b"*64,
            resource_binding=dict(gpu=2,gpu_uuid="GPU-test",minimum_free_mib=30720,free_mib_before_load=50000,pid=1,visible_cuda_ordinal=0),
            semantic_text="organ",metric_protocol="native_complete_mask_axial_union_nonempty_ten",
            runtime_input_audit=dict(native_source_sha256="a"*64,task_body_ast_sha256="b"*64,semantics="_legacy_text_amos",model_input_size=1024),
            ct_cache_access=[dict(mode="read_only_hit")])
        record["metric_execution"] = dict(provider_sha256=runtime._sha((Path(runtime.__file__).parent/"bone_fast_metrics.py").read_bytes()),
            events=[dict(key=["amos","case",10001],variant="baseline",metrics={name:row[name] for name in runtime.METRICS},
                         input_sha256="c"*64,native_compared=True,max_abs_error=0,full_mask_shape=[12,5,7])],
            tracker=dict(comparison_failures=0,cache_tensor_bytes=0),cache_scope="class_local_metadata_only")
        return record, task

    def test_record_actual_evidence_validates(self):
        record, task = self.record()
        self.assertTrue(runtime.validate_record(record,task,"baseline",4))

    def test_record_rejects_transform_raster_or_plane_audit_changes(self):
        record, task = self.record()
        for field,value in (("groups_sha256","f"*64),("plane_ids",[1]),("raw_background_points_per_plane",{"coronal":4,"sagittal":0})):
            altered = copy.deepcopy(record)
            altered["prompt"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                runtime.validate_record(altered,task,"baseline",4)

    def test_record_rejects_wrong_fullmask_space_or_semantics(self):
        record, task = self.record()
        altered = copy.deepcopy(record)
        altered["row"]["metric_space"] = "partial_roi"
        with self.assertRaises(ValueError):
            runtime.validate_record(altered,task,"baseline",4)
        altered = copy.deepcopy(record)
        altered["stages"][0]["prompt_calls"][0]["semantic_text"] = "wrong"
        with self.assertRaises(ValueError):
            runtime.validate_record(altered,task,"baseline",4)

    def test_fast_metric_provider_observes_native_comparisons_and_actual_inputs(self):
        import types
        from unittest.mock import patch
        self.assertTrue(hasattr(runtime, "_install_metric_provider"), "install the existing exact shared-distance metric provider")
        installed = []
        class Tracker:
            def snapshot(self):
                return dict(last_input_sha256="c"*64,last_cache_hit=False,last_native_compared=True,
                            last_comparison_max_abs_error=0,comparison_failures=0)
        def install(native, compare_native, workers, cache_entries):
            installed.append((native, compare_native(), workers, cache_entries))
            return lambda *a, **k: {name:.5 for name in runtime.METRICS}, Tracker()
        provider = types.ModuleType("tools.bone_fast_metrics")
        provider.install_provider = install
        session = runtime.Session({"control":True},{})
        session.current,session.variant=("amos","case",10001),"baseline"
        session.metric_events=[]
        scorer,native=types.SimpleNamespace(),object()
        with patch.dict("sys.modules", {"tools.bone_fast_metrics":provider}):
            runtime._install_metric_provider(session,scorer,native)
            values=scorer.binary_prompt_metrics_from_masks(np.ones((2,3,4)),np.ones((2,3,4)),spacing_dhw=(1,1,1))
        self.assertEqual(installed,[(native,True,2,128)])
        self.assertEqual(values["dice"],.5)
        self.assertTrue(session.metric_events[-1]["native_compared"])
        self.assertEqual(session.metric_events[-1]["input_sha256"],"c"*64)

    def test_control_metric_provider_requires_actual_native_twin_and_row_binding(self):
        record,task=self.record()
        altered=copy.deepcopy(record)
        altered["metric_execution"]["events"][0]["native_compared"]=False
        with self.assertRaises(ValueError):
            runtime.validate_record(altered,task,"baseline",4,control=True)
        altered=copy.deepcopy(record)
        altered["metric_execution"]["events"][0]["metrics"]["dice"]=.51
        with self.assertRaises(ValueError):
            runtime.validate_record(altered,task,"baseline",4)


class DatasetHookTests(unittest.TestCase):
    def test_actual_raw_support_hook_installed_before_native_reserve_capture(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        inner = SimpleNamespace(_v10_active_idx=0, _v10_active_class=10001,
            roi_context_slices=1, roi_context_ratio=0, roi_min_frames=1,
            _prompt_derived_roi_indices=lambda target: np.arange(9, 12),
            _build_prompt_slice_groups=lambda *args: {})
        session = runtime.Session({}, {})
        session.variant = "sagittal_only"
        session.namespace = {"_find_inner_dataset": lambda dataset: dataset}
        def converted(*args):
            session.prompt_audit = {"raw_foreground_z_support": [3, 4]}
            return {}
        session.converted = converted
        captured = []
        def native_installer(dataset, **kwargs):
            original = dataset._prompt_derived_roi_indices
            captured.append(original)
            def reserved(self, target): return original(target)
            from types import MethodType
            dataset._prompt_derived_roi_indices = MethodType(reserved, dataset)
            return dataset, dataset
        with patch.dict("sys.modules", {"torch": SimpleNamespace(from_numpy=lambda x: x)}):
            _, installed = runtime._install_dataset(session, native_installer, inner)
        target = np.ones((20, 2, 2))
        np.testing.assert_array_equal(installed._prompt_derived_roi_indices(target), np.arange(2, 6))
        self.assertEqual(session.roi_calls[-1]["source"], "actual_raw_active_foreground")
        session.variant = "native_control"
        np.testing.assert_array_equal(installed._prompt_derived_roi_indices(target), np.arange(9, 12))

    def test_readonly_ct_cache_miss_loads_one_case_once_and_never_saves(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        calls = []
        inner = SimpleNamespace(_prompt_derived_roi_indices=lambda target: [],
            _build_prompt_slice_groups=lambda *args: {},
            _case_image_cache_path=lambda image: "existing.npy",
            _load_case_image_cache=lambda path: None,
            _load_array=lambda image: (calls.append(image) or np.zeros((4, 5, 6), np.float32)))
        session = runtime.Session({}, {})
        session.namespace = {"_find_inner_dataset": lambda dataset: dataset}
        with patch.dict("sys.modules", {"torch": SimpleNamespace(from_numpy=lambda x: x)}):
            runtime._install_dataset(session, lambda dataset, **kwargs: (dataset, dataset), inner)
        first = inner._get_or_create_case_image("case.nii.gz")
        self.assertIs(inner._get_or_create_case_image("case.nii.gz"), first)
        self.assertEqual(calls, ["case.nii.gz"])
        self.assertTrue(all(event["mode"] == "current_case_ram_miss" for event in session.cache_events))
        with self.assertRaisesRegex(ValueError, "writes forbidden"):
            inner._save_case_image_cache("existing.npy", first)

    def test_actual_native_prompt_cache_save_name_is_readonly(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        writes = []
        inner = SimpleNamespace(_prompt_derived_roi_indices=lambda target: [],
            _build_prompt_slice_groups=lambda *args: {},
            _save_prompt_cache=lambda *args: writes.append(args))
        session = runtime.Session({}, {})
        session.namespace = {"_find_inner_dataset": lambda dataset: dataset}
        with patch.dict("sys.modules", {"torch": SimpleNamespace(from_numpy=lambda x: x)}):
            runtime._install_dataset(session, lambda dataset, **kwargs: (dataset, dataset), inner)
        with self.assertRaisesRegex(ValueError, "writes forbidden"):
            inner._save_prompt_cache("prompt.npz", {"groups": np.zeros((1,3,8,8))})
        self.assertEqual(writes, [])

    def test_missing_native_hook_refused_before_any_forward(self):
        import types
        from unittest.mock import patch
        session = runtime.Session({}, {})
        session.namespace = {"prompt": types.SimpleNamespace()}
        v10 = types.ModuleType("v10")
        infer = types.ModuleType("v10.infer")
        scorer = types.ModuleType("v10.infer.infer_amos_sam2_promptgen_v10")
        v10.infer = infer
        infer.infer_amos_sam2_promptgen_v10 = scorer
        fake_modules = {"v10": v10, "v10.infer": infer, scorer.__name__: scorer,
                        "v10_sam2_memory_roi": types.ModuleType("v10_sam2_memory_roi")}
        with patch.dict("sys.modules", fake_modules):
            with self.assertRaisesRegex(ValueError, "missing actual PromptGen hook"):
                runtime._install_execution_hooks(session)


if __name__ == "__main__":
    unittest.main()
