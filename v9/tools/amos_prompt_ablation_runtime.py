"""Single-load native AMOS prompt interventions with actual execution proofs.

The checked native main is split at its sole task loop. Both code objects execute
in ONE private globals dictionary; neither native forwards nor the task body are
rewritten. This module writes no predictions, tensors, or new CT caches.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import MethodType, SimpleNamespace

METRICS = ("dice", "mean_slice_dice", "iou", "precision", "recall", "specificity",
           "pred_gt_ratio", "mean_slice_nsd_1mm", "mean_slice_nsd_2mm", "mean_slice_nsd_3mm")
FROZEN_SETTINGS = dict(eval_memory_group_size=1, dynamic_threshold_min=.70,
    dynamic_threshold_max=.75, dynamic_threshold_fallback=.60,
    dynamic_threshold_min_logit_separation=.50, prompt_logit_neighborhood_radius=2,
    patch_scaling=False, anchor_coarse_crop=False, object_score_gate=False, amp=True,
    sam2_feature_cache_device="cpu", sam2_feature_cache_gpu_max_eval_frames=0,
    v10_3_local_patch=False, v10_3_grouped_decode=False, v10_3_metric_pipeline=False)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


class NativeProgram:
    def __init__(self, source, filename, tree, main, loop):
        self.filename = str(filename)
        self.source_sha256 = _sha(source.encode())
        prefix = main.body[:main.body.index(loop)]
        outer = [node for node in tree.body if node is not main and not (
            isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
            and ast.unparse(node.test) == "__name__ == '__main__'")]
        self.definitions = compile(ast.fix_missing_locations(ast.Module(body=outer, type_ignores=[])), filename, "exec")
        self.initialization = compile(ast.fix_missing_locations(ast.Module(body=prefix, type_ignores=[])), filename, "exec")
        self.task_body = compile(ast.fix_missing_locations(ast.Module(body=loop.body, type_ignores=[])), filename, "exec")
        self.task_body_source = ast.unparse(ast.Module(body=loop.body, type_ignores=[]))
        self.task_body_sha256 = _sha(ast.dump(ast.Module(body=loop.body, type_ignores=[]), include_attributes=False).encode())

    def namespace(self):
        namespace = dict(__name__="_amos_prompt_ablation_native", __file__=self.filename)
        exec(self.definitions, namespace)
        return namespace


def compile_native(source, filename, expected_sha256):
    """Check source bytes, loop structure, and compile the untouched loop body."""
    payload = source if isinstance(source, bytes) else source.encode()
    if not expected_sha256 or _sha(payload) != expected_sha256:
        raise ValueError("native source SHA mismatch")
    source = payload.decode("utf-8")
    tree = ast.parse(source, filename=str(filename))
    mains = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"]
    if len(mains) != 1 or mains[0].args.args or mains[0].decorator_list:
        raise ValueError("native main structure mismatch")
    main = mains[0]
    loops = [node for node in main.body if isinstance(node, ast.For) and ast.unparse(node.target) == "task"]
    if len(loops) != 1 or ast.unparse(loops[0].iter) != "sorted(tasks, key=_task_key)" or loops[0].orelse:
        raise ValueError("native task loop mismatch")
    class LocalControl(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            # Nested native hooks retain their own normal return statements.
            pass
        visit_AsyncFunctionDef = visit_FunctionDef
        visit_Lambda = visit_FunctionDef
        def visit_Return(self, node):
            raise ValueError("native main contains function-local control flow")
        visit_Yield = visit_Return
        visit_YieldFrom = visit_Return
        visit_Nonlocal = visit_Return
    for node in main.body:
        LocalControl().visit(node)
    program = NativeProgram(source, str(filename), tree, main, loops[0])
    program.source_sha256 = _sha(payload)
    return program


def _array(value):
    import numpy as np
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        # Native prompt masks/types are float32/int64. Parameter comparisons use
        # torch.equal separately to preserve bfloat16 and exact dtype handling.
        value = value.numpy()
    return np.asarray(value)


def _digest(value):
    import numpy as np
    array = np.ascontiguousarray(_array(value))
    return _sha(str((array.shape, array.dtype.str)).encode() + array.tobytes())


def _scalar(value):
    return int(value.item()) if hasattr(value, "item") else int(value)


def validate_settings(args, batch):
    if batch not in (1, 2, 4) or getattr(args, "sam2_frame_batch_size", None) != batch:
        raise ValueError("actual native frame batch mismatch")
    for name, value in FROZEN_SETTINGS.items():
        if getattr(args, name, None) != value:
            raise ValueError("changed frozen native setting: " + name)
    return {**FROZEN_SETTINGS, "sam2_frame_batch_size": batch}


def compare_parameters(module, state):
    """Verify actual restored values, including lazily restored decoder modules."""
    import numpy as np
    count = 0
    for name, parameter in module.named_parameters():
        actual = parameter.detach().cpu()
        expected = state.get(name)
        if hasattr(actual, "dtype") and hasattr(actual, "device"):
            import torch
            equal = expected is not None and torch.equal(actual, expected.to(dtype=actual.dtype, device="cpu"))
        else:
            equal = expected is not None and np.array_equal(_array(actual), _array(expected))
        if not equal:
            raise ValueError("checkpoint parameter not restored exactly: " + name)
        count += 1
    if not count:
        raise ValueError("empty restoration audit")
    return count


def parameter_guard(module):
    """Cheap mutation/device/replacement guard, with no parameter CPU transfer."""
    values = [(name, int(parameter._version), int(parameter.data_ptr()),
               tuple(parameter.shape), str(parameter.dtype), str(parameter.device))
              for name, parameter in module.named_parameters()]
    if not values:
        raise ValueError("empty actual parameter guard")
    return _sha(repr(values).encode())


def validate_consumption(groups, plane_ids, foreground_mode_id, modes, expected, *, slice_coords=None, expected_slice_coords=None):
    """Inspect the arguments actually passed to PromptGen at every depth chunk."""
    import numpy as np
    groups, planes = _array(groups), _array(plane_ids).reshape(-1)
    expected = _array(expected)
    if tuple(int(x) for x in planes) != tuple(modes["plane_ids"]):
        raise ValueError("actual PromptGen contains a phantom or missing plane")
    if groups.ndim != 4 or groups.shape[0] != len(planes) or groups.shape[1] != 3:
        raise ValueError("actual PromptGen group layout mismatch")
    if _scalar(foreground_mode_id) != modes["foreground_mode_id"]:
        raise ValueError("actual PromptGen foreground mode id mismatch")
    if groups.shape != expected.shape or not np.array_equal(groups, expected):
        raise ValueError("actual PromptGen group raster differs from selected variant")
    if slice_coords is not None and (expected_slice_coords is None or not np.array_equal(_array(slice_coords), _array(expected_slice_coords))):
        raise ValueError("actual PromptGen slice coordinate mismatch")
    active = 0 if modes["foreground_mode"] == "scribble" else 2
    inactive = 2 if active == 0 else 0
    counts = np.count_nonzero(groups, axis=(2, 3))
    if np.any(counts[:, inactive]) or np.any(counts[:, active] <= 0) or np.any(counts[:, 1] <= 0):
        raise ValueError("actual PromptGen foreground/background channel mismatch")
    event = dict(plane_ids=planes.astype(int).tolist(), foreground_mode_id=_scalar(foreground_mode_id),
                groups_sha256=_digest(groups), raster_sha256=_sha(groups.tobytes()), channel_nonzero=counts.astype(int).tolist())
    if slice_coords is not None:
        event["slice_coords"] = _array(slice_coords).reshape(-1).astype(float).tolist()
    return event


def validate_adapter_types(role_ids, type_ids, modes):
    return validate_output_types(dict(prompt_3d_role_ids=role_ids, prompt_3d_type_ids=type_ids), modes, metadata=False)


def validate_output_types(output, modes, *, metadata=True):
    import numpy as np
    types = _array(output["prompt_3d_type_ids"]).reshape(-1)
    roles = _array(output["prompt_3d_role_ids"]).reshape(-1)
    if types.shape != roles.shape or not len(types) or set(roles.tolist()) != {0, 1}:
        raise ValueError("actual prompt role/type tokens missing")
    if not np.all(types[roles == 0] == modes["foreground_type_id"]) or not np.all(types[roles == 1] == modes["background_type_id"]):
        raise ValueError("actual learned prompt type mismatch")
    if metadata and (output.get("foreground_prompt_mode") != modes["foreground_mode"]
            or output.get("background_prompt_mode") != modes["background_mode"]
            or output.get("foreground_channel") != (0 if modes["foreground_mode"] == "scribble" else 2)
            or output.get("background_active") is not True):
        raise ValueError("actual PromptGen output mode mismatch")
    return dict(type_ids=sorted(set(int(x) for x in types)), role_ids=sorted(set(int(x) for x in roles)),
                type_counts={str(int(x)): int((types == x).sum()) for x in set(types.tolist())})


def _key(task):
    return (str(task.get("source_name", "amos")), str(task["case_id"]),
            int(task.get("global_class_id", task.get("class_id"))))


class Session:
    """One case, one native model load, and serial native task executions."""
    def __init__(self, job, paths):
        self.job, self.paths = copy.deepcopy(job), dict(paths)
        self.current = None
        self.variant = None
        self.active_stage = None
        self.stages, self.prompt_calls, self.group_builds, self.roi_calls = [], [], [], []
        self.memory_calls = dict(memory_encoder=0, memory_attention=0)
        self.raw_cache = None
        self.case_ram_ct = {}
        self.cache_events = []
        self.hooks = []
        self.model_loads = 0
        self.restoration = None

    def modes(self):
        from tools.amos_prompt_ablation_prompts import variant_modes
        return variant_modes("baseline" if self.variant == "native_control" else self.variant)

    def restore_proof(self):
        ns = self.namespace
        proof = dict(model="v10_2", epoch=int(ns["state"]["epoch"]), weight_mode="live",
            prompt_parameters_exact=compare_parameters(ns["prompt"], ns["state"]["prompt_generator"]),
            decoder_parameters_exact=compare_parameters(ns["adapter"].sam.sam_mask_decoder, ns["state"]["mask_decoder"]),
            guards_passed=True, model_load_count=self.model_loads,
            full_tensor_comparison_phase="post_load" if self.restoration is None else "normal_worker_completion",
            per_evaluation_guard="parameter_version_pointer_shape_dtype_device")
        if (proof["epoch"], proof["prompt_parameters_exact"], proof["decoder_parameters_exact"], self.model_loads) != (490, 418, 212, 1):
            raise ValueError("E490 live restoration/count mismatch")
        if self.restoration is None:
            self.restoration = proof
            self.initial_parameter_guard = self.check_parameter_guard(initialize=True)
        return dict(proof)

    def check_parameter_guard(self, *, initialize=False):
        ns = self.namespace
        value = dict(prompt=parameter_guard(ns["prompt"]),
                     decoder=parameter_guard(ns["adapter"].sam.sam_mask_decoder))
        if not initialize and value != self.initial_parameter_guard:
            raise ValueError("actual native parameter version/pointer/layout changed")
        return value

    def finish(self):
        """Normal worker completion requires one final full tensor comparison."""
        if getattr(self, "finished", False):
            raise ValueError("native session already finished")
        self.check_parameter_guard()
        proof = self.restore_proof()
        self.finished = True
        return dict(proof, initial_restoration=self.restoration,
                    parameter_guard=self.check_parameter_guard(), successful_native_completion=True)

    def package(self, inner, index, class_id):
        from nninteractive_test.infer.adaptive_prompt_cache import _read
        ns = self.namespace
        identity = (int(index), int(class_id))
        if self.raw_cache is not None and self.raw_cache[0] == identity:
            return self.raw_cache[1:]
        image = os.path.normpath(str(inner.image_paths[int(index)]))
        task = next(row for row in ns["tasks"] if os.path.normpath(str(row.get("image_path", row.get("image")))) == image
                    and int(row["global_class_id"]) == int(class_id))
        cache_path = Path(ns["custom"].shared_adaptive_prompt_root)/"metadata/shared_adaptive_prompt_v1/amos"/ns["prompt_index"][_key(task)]
        package = _read(cache_path)
        if package is None:
            raise ValueError("missing read-only adaptive prompt package: " + str(cache_path))
        # Only selected 2D planes generate the native box outline/background.
        target = inner._load_canonical_target(int(index), int(class_id))
        previous = inner.prompt_use_box, inner.prompt_use_background_scribble
        try:
            inner.prompt_use_box = inner.prompt_use_background_scribble = True
            native = {plane: inner._plane_prompt_group(target, plane, int(package[plane + "_slice"]))
                      for plane in ("coronal", "sagittal")}
        finally:
            inner.prompt_use_box, inner.prompt_use_background_scribble = previous
        self.raw_cache = (identity, package, native, str(cache_path), _sha(cache_path.read_bytes()))
        return self.raw_cache[1:]

    def converted(self, inner, index, class_id):
        from tools.amos_prompt_ablation_prompts import build_variant
        package, native, path, digest = self.package(inner, index, class_id)
        converted, audit = build_variant(package, native,
            "baseline" if self.variant == "native_control" else self.variant, int(inner.prompt_plane_size))
        audit = dict(audit, prompt_cache_path=path, prompt_cache_sha256=digest)
        self.prompt_audit = audit
        self.expected_groups = converted["groups"]
        self.expected_slice_coords = converted["slice_coords"]
        return converted


def _install_dataset(session, native_installer, dataset, **kwargs):
    """Install raw support before native reserve_roi captures original_roi."""
    import numpy as np
    import torch
    from tools.amos_prompt_ablation_prompts import roi_indices
    ns = session.namespace
    inner = ns["_find_inner_dataset"](dataset)
    native_roi = inner._prompt_derived_roi_indices

    def actual_roi(self, target):
        session.converted(self, self._v10_active_idx, self._v10_active_class)
        if session.variant == "native_control":
            result = native_roi(target)
        else:
            result = roi_indices(int(target.shape[0]), session.prompt_audit["raw_foreground_z_support"],
                self.roi_context_slices, self.roi_context_ratio, self.roi_min_frames)
        session.roi_calls.append(dict(source="native_original" if session.variant == "native_control" else "actual_raw_active_foreground",
            raw_support=list(session.prompt_audit["raw_foreground_z_support"]), indices=np.asarray(result).astype(int).tolist(),
            depth=int(target.shape[0]), context_slices=int(self.roi_context_slices),
            context_ratio=float(self.roi_context_ratio), minframes=int(self.roi_min_frames)))
        return result

    inner._prompt_derived_roi_indices = MethodType(actual_roi, inner)
    result, inner = native_installer(dataset, **kwargs)
    original_groups = inner._build_prompt_slice_groups

    def groups(self, index, class_id):
        converted = session.converted(self, index, class_id)
        if session.variant == "native_control":
            value = original_groups(index, class_id)
            for field in ("groups", "plane_ids", "slice_coords"):
                if not np.array_equal(_array(value[field]), converted[field]):
                    raise ValueError("native baseline converter mismatch: " + field)
        else:
            value = {field: torch.from_numpy(np.asarray(converted[field])) for field in ("groups", "plane_ids", "slice_coords")}
            value["canonical_depth"] = int(converted["canonical_depth"])
            ns["audit"]["loaded_task_keys"].append(list(session.current))
        session.group_builds.append(dict(groups_sha256=_digest(value["groups"]), plane_ids=_array(value["plane_ids"]).astype(int).tolist()))
        return value

    inner._build_prompt_slice_groups = MethodType(groups, inner)

    def readonly_ct(self, image):
        path = self._case_image_cache_path(image)
        volume = self._load_case_image_cache(path)
        hit = volume is not None
        if volume is None:
            if str(image) not in session.case_ram_ct:
                volume = np.asarray(self._load_array(image), dtype=np.float32)
                if volume.ndim == 4 and volume.shape[0] == 1:
                    volume = volume[0]
                if volume.ndim != 3:
                    raise ValueError("non-3D canonical CT")
                session.case_ram_ct[str(image)] = volume
            volume = session.case_ram_ct[str(image)]
        session.cache_events.append(dict(path=path, image=str(image), mode="read_only_hit" if hit else "current_case_ram_miss",
                                        shape=list(volume.shape), dtype=str(volume.dtype)))
        return volume

    inner._get_or_create_case_image = MethodType(readonly_ct, inner)
    def refuse_save(*_a, **_k):
        raise ValueError("CT cache writes forbidden during prompt ablation")
    inner._save_case_image_cache = refuse_save
    # Native dynamic ROI installer already disables all class/ROI cache loads
    # and saves. Disable prompt-plane writes as well; exact groups bypass them.
    for method in ("_save_prompt_cache", "_save_prompt_slice_cache"):
        if hasattr(inner, method):
            setattr(inner, method, refuse_save)
    session.dataset_bound = True
    return result, inner


def _install_execution_hooks(session):
    import v10_sam2_memory_roi as memory
    import v10.infer.infer_amos_sam2_promptgen_v10 as scorer
    ns = session.namespace
    prompt = ns["prompt"]
    for method in ("forward_case_3d_tokens", "_mixed_prompt_groups", "build_case_3d_tokens_from_global_pool"):
        if not callable(getattr(prompt, method, None)):
            raise ValueError("missing actual PromptGen hook: " + method)
    for name in session.memory_calls:
        module = getattr(ns["adapter"].sam, name, None)
        if module is None or not hasattr(module, "register_forward_pre_hook"):
            raise ValueError("missing actual SAM2 memory hook: " + name)
        def count(_module, _args, label=name):
            session.memory_calls[label] += 1
        session.hooks.append(module.register_forward_pre_hook(count))

    native_forward = prompt.forward_case_3d_tokens
    def forward(self, groups, video, *args, **kwargs):
        event = validate_consumption(groups, kwargs["plane_ids"], kwargs["foreground_mode_id"], session.modes(), session.expected_groups,
            slice_coords=kwargs["slice_coords"], expected_slice_coords=session.expected_slice_coords)
        if session.active_stage is None:
            raise ValueError("PromptGen executed outside actual decoder stage")
        if kwargs.get("semantic_text") != session.expected_semantics[session.current[2]]:
            raise ValueError("actual PromptGen legacy AMOS semantic text mismatch")
        output = native_forward(groups, video, *args, **kwargs)
        event.update(validate_output_types(output, session.modes()))
        event.update(semantic_text=str(kwargs.get("semantic_text")), frame_coords_sha256=_digest(kwargs["frame_coords"]),
                     stage=len(session.stages), frames=int(video.shape[0]))
        session.prompt_calls.append(event)
        session.active_stage["prompt_calls"].append(event)
        return output
    prompt.forward_case_3d_tokens = MethodType(forward, prompt)

    native_mixed = prompt._mixed_prompt_groups
    def mixed(self, incoming):
        import numpy as np
        result = native_mixed(incoming)
        value, foreground, background, channel, active = result
        modes = session.modes()
        if (foreground, background, int(channel), bool(active)) != (modes["foreground_mode"], modes["background_mode"],
                0 if modes["foreground_mode"] == "scribble" else 2, True):
            raise ValueError("actual mixed prompt mode mismatch")
        raw, actual = _array(incoming), _array(value)
        if not np.array_equal(actual[:, 0], raw[:, channel]) or not np.array_equal(actual[:, 1], raw[:, 1]):
            raise ValueError("native mixed path changed actual foreground/background raster")
        session.active_stage["mixed_calls"].append(dict(foreground=foreground, background=background,
            incoming_sha256=_digest(incoming), mixed_sha256=_digest(value), foreground_channel=int(channel)))
        return result
    prompt._mixed_prompt_groups = MethodType(mixed, prompt)

    native_pool = prompt.build_case_3d_tokens_from_global_pool
    def pool(self, *args, **kwargs):
        if (kwargs["foreground_mode"], kwargs["background_mode"], kwargs["background_active"]) != (
                session.modes()["foreground_mode"], session.modes()["background_mode"], True):
            raise ValueError("actual global-pool mode mismatch")
        output = native_pool(*args, **kwargs)
        session.active_stage["global_pool_calls"].append(validate_output_types(output, session.modes(), metadata=False))
        return output
    prompt.build_case_3d_tokens_from_global_pool = MethodType(pool, prompt)

    native_generate = memory.generate_case_3d_tokens
    def generate(prompt, adapter, video, bundle, class_id, args, device):
        validate_settings(args, session.batch)
        if _scalar(class_id) != session.current[2] or tuple(video.shape[-2:]) != (1024, 1024):
            raise ValueError("actual stage class/native1024 mismatch")
        if bundle.get("v10_forced_background_mode") != session.modes()["background_mode"]:
            raise ValueError("actual bundle forced background mode mismatch")
        consumption = validate_consumption(bundle["groups"], bundle["plane_ids"], bundle["v10_foreground_mode_id"],
            session.modes(), session.expected_groups, slice_coords=bundle["slice_coords"], expected_slice_coords=session.expected_slice_coords)
        stage = dict(frames=int(video.shape[0]), frame_coords_sha256=_digest(bundle["frame_coords"]),
            frame_coords=_array(bundle["frame_coords"]).reshape(-1).astype(float).tolist(),
            bundle=consumption, settings=validate_settings(args, session.batch),
            prompt_calls=[], mixed_calls=[], global_pool_calls=[], decoder_calls=[], adapter_calls=[],
            memory_before=dict(session.memory_calls))
        session.active_stage = stage
        return native_generate(prompt, adapter, video, bundle, class_id, args, device)
    memory.generate_case_3d_tokens = generate

    native_decode = memory.memory_sliding_window_decode
    def decode(adapter, video, bundle, tokens, anchor, args, device, training):
        if training or session.active_stage is None:
            raise ValueError("actual decoder execution outside inference stage")
        validate_settings(args, session.batch)
        event = validate_output_types(tokens, session.modes(), metadata=False)
        event.update(frames=int(video.shape[0]), anchor=int(anchor))
        session.active_stage["decoder_calls"].append(event)
        return native_decode(adapter, video, bundle, tokens, anchor, args, device, training)
    memory.memory_sliding_window_decode = decode

    adapter3d = getattr(ns["adapter"].sam.sam_mask_decoder, "prompt_3d_adapter", None)
    if adapter3d is None or not hasattr(adapter3d, "register_forward_pre_hook"):
        raise ValueError("missing actual decoder 3D prompt adapter hook")
    def adapter_types(_module, _args, kwargs):
        if session.active_stage is None:
            raise ValueError("3D prompt adapter outside actual stage")
        session.active_stage["adapter_calls"].append(validate_adapter_types(kwargs["role_ids"], kwargs["type_ids"], session.modes()))
    session.hooks.append(adapter3d.register_forward_pre_hook(adapter_types, with_kwargs=True))

    native_post = scorer.postprocess_v10_logits_from_prompts
    def post(*args, **kwargs):
        for name, key in (("minimum", "dynamic_threshold_min"), ("maximum", "dynamic_threshold_max"),
                          ("fallback", "dynamic_threshold_fallback"), ("min_logit_separation", "dynamic_threshold_min_logit_separation")):
            if kwargs.get(name) != FROZEN_SETTINGS[key]:
                raise ValueError("actual postprocess setting changed: " + name)
        value = native_post(*args, **kwargs)
        stage = session.active_stage
        if stage is None or not all(stage[name] for name in ("prompt_calls", "mixed_calls", "global_pool_calls", "decoder_calls", "adapter_calls")):
            raise ValueError("actual stage lacks consumed prompt/type hooks")
        stage["memory_calls"] = {
            name: session.memory_calls[name] - stage["memory_before"][name] for name in session.memory_calls}
        del stage["memory_before"]
        if stage["memory_calls"]["memory_encoder"] <= 0:
            raise ValueError("actual SAM2 memory encoder did not execute")
        stage["threshold_audit"] = _plain(value[1])
        session.stages.append(stage)
        session.active_stage = None
        return value
    scorer.postprocess_v10_logits_from_prompts = post
    session.execution_hooks_bound = True


def _plain(value):
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    raise ValueError("nonserializable proof value: " + type(value).__name__)


def _resolve(paths, job, key, default=None):
    value = paths.get(key, job.get(key, default))
    if value is None:
        raise ValueError("missing runtime path: " + key)
    path = Path(value)
    return path if path.is_absolute() else Path(paths["root"])/path


def _install_metric_provider(session, scorer, native_module):
    """Share native axial boundary distances, with native twins on all controls."""
    from tools.bone_fast_metrics import install_provider
    metric, tracker = install_provider(native_module,
        compare_native=lambda: bool(session.job.get("control") or session.variant == "native_control"),
        workers=2, cache_entries=128)
    session.metric_tracker = tracker
    session.metric_provider_sha256 = _sha((Path(__file__).parent/"bone_fast_metrics.py").read_bytes())
    session.metric_events = []
    session.metric_class = None
    def score(prediction, target, spacing_dhw=None):
        result = metric(prediction, target, spacing_dhw=spacing_dhw)
        state = tracker.snapshot()
        session.metric_events.append(dict(key=list(session.current), variant=session.variant,
            input_sha256=state["last_input_sha256"], cache_hit=state["last_cache_hit"],
            native_compared=state["last_native_compared"], max_abs_error=state["last_comparison_max_abs_error"],
            metrics={name: float(result[name]) for name in METRICS}, full_mask_shape=list(prediction.shape),
            spacing_dhw=list(spacing_dhw) if spacing_dhw is not None else None))
        return result
    scorer.binary_prompt_metrics_from_masks = score


def initialize(job, paths):
    """Load the native E490 model once. All paths belong to the frozen job."""
    session = Session(job, paths)
    root = Path(paths["root"]).resolve()
    session.batch = int(job.get("actual_frame_batch", job.get("frame_batch_size", job.get("frame_batch", 4))))
    if job.get("gpu") not in (2, 4, 5, 6) or session.batch not in (1, 2, 4):
        raise ValueError("invalid actual GPU/frame batch")
    native_path = _resolve(paths, job, "native_source", "v10/infer/infer_eight_source_adaptive_v10_2.py")
    expected = paths.get("native_source_sha256", job.get("native_source_sha256"))
    session.program = compile_native(native_path.read_bytes(), str(native_path), expected)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
    os.environ["V9_SHARED_CACHE_WRITE_POLICY"] = "read_only"
    for path in (root, root/"v10"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import torch
    torch.set_num_threads(2)
    visible = "GPU-" + str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-")
    if visible != job.get("gpu_uuid") or torch.ones(1, device="cuda:0").item() != 1:
        raise ValueError("actual GPU UUID/CUDA mismatch")
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True)
    gpu_rows = {int(row.split(",")[0]): [part.strip() for part in row.split(",")] for row in result.stdout.strip().splitlines()}
    actual = gpu_rows[job["gpu"]]
    minimum = int(job.get("minimum_free_mib", 30720))
    if actual[1] != visible:
        raise ValueError("physical GPU UUID mismatch")
    if minimum < 30720 or int(actual[2]) < minimum:
        raise SystemExit(75)
    session.resource_binding = dict(gpu=job["gpu"], gpu_uuid=visible, visible_cuda_ordinal=0,
        free_mib_before_load=int(actual[2]), minimum_free_mib=minimum, pid=os.getpid())

    split_path = _resolve(paths, job, "base_split_json", "configs/ct13_v10_2_split_20260928.json")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    import v9_2_extended_protocol as protocol
    protocol.SOURCE_ORDER = tuple(split["source_order"])
    import v10.finetune_totalseg_amos_magic_v10_2_joint as entry
    import v10_2_data as data
    from ct13_training_entry import install_v10_protocol
    install_v10_protocol(entry, data, split)
    from finetune_multisource_sam2_v9_2_precision_mixed_semantic_eval import _legacy_text
    native_text = entry.conditioning_text_from_metadata
    entry.conditioning_text_from_metadata = lambda row: _legacy_text(row) if row["dataset"] in ("amos", "totalseg") else native_text(row)
    # Keep the existing axial full-mask metric definition used by the prior AMOS
    # evaluation. This provider's native implementation remains the control.
    import v10.infer.infer_amos_sam2_promptgen_v10 as scorer
    from tools.ct13_full_validation_eval import metric_module
    _install_metric_provider(session, scorer, metric_module())

    ns = session.program.namespace()
    session.namespace = ns
    run_dir = _resolve(paths, job, "run_dir", job.get("run", "output/amos_prompt_ablation_runtime"))
    prepared = Path(job.get("prepared", run_dir))
    custom = SimpleNamespace(independent_task_manifest=str(_resolve(paths, job, "manifest", prepared/"manifest.json")),
        nninteractive_metrics=str(_resolve(paths, job, "refs", prepared/"refs.json")),
        shared_adaptive_prompt_root=str(_resolve(paths, job, "shared_adaptive_prompt_root", "v10/.cache/totalseg_sam2_promptgen_v10_prompt_roi_full_v1")),
        weights=str(_resolve(paths, job, "weights", "v10/output/v10_2_ct13_e470_zero_delta_e510_20261001/20261001_215853/epoch490.pth")),
        source="amos", background_mode="point", eval_out_dir=str(run_dir.parent), run_name=run_dir.name,
        gpu=str(job["gpu"]), sam2_frame_batch_size=session.batch, task_shard_count=1, task_shard_index=0,
        dynamic_threshold_fallback=.60, dynamic_threshold_min=.70, dynamic_threshold_max=.75,
        dynamic_threshold_min_logit_separation=.50, prompt_logit_neighborhood_radius=2,
        base_split_json=str(split_path), runtime_version="v10.2")
    ns["_extract_args"] = lambda: custom
    installer = ns["_install_exact_prompts_and_reserved_roi"]
    ns["_install_exact_prompts_and_reserved_roi"] = lambda dataset, **kwargs: _install_dataset(session, installer, dataset, **kwargs)
    session.variant = "baseline"
    exec(session.program.initialization, ns)
    session.model_loads += 1
    if not getattr(session, "dataset_bound", False):
        raise ValueError("native dataset hook did not execute")
    cases = {_key(task)[:2] for task in ns["tasks"]}
    if len(cases) != 1 or next(iter(cases)) != ("amos", str(job["case_id"])):
        raise ValueError("single-load runtime must own exactly its signed AMOS case")
    validate_settings(ns["args"], session.batch)
    if int(getattr(ns["args"], "model_input_size", 0)) != 1024 or int(getattr(ns["inner"], "model_input_size", 0)) != 1024:
        raise ValueError("native model preprocessing must be1024")
    session.expected_semantics = dict(entry._SEMANTIC_BY_ID)
    session.restore_proof()
    _install_execution_hooks(session)
    session.input_audit = dict(native_source_sha256=session.program.source_sha256,
        task_body_ast_sha256=session.program.task_body_sha256, semantics="_legacy_text_amos", cache_policy="readonly_CT_or_current_case_RAM",
        actual_frame_batch=session.batch, settings=validate_settings(ns["args"], session.batch),
        model_input_size=int(ns["args"].model_input_size), cache_image_size=int(ns["inner"].image_size))
    return session


def evaluate(session, task, variant):
    """Run one unchanged native task body; return only serializable evidence."""
    from tools.amos_prompt_ablation_prompts import VARIANTS
    if variant not in (*VARIANTS, "native_control") or not getattr(session, "execution_hooks_bound", False) or getattr(session, "finished", False):
        raise ValueError("invalid variant or missing actual execution hooks")
    ns = session.namespace
    if _key(task) not in {_key(value) for value in ns["tasks"]}:
        raise ValueError("task outside initialized case")
    session.current, session.variant = _key(task), variant
    if session.metric_class != session.current:
        session.metric_tracker.clear()
        session.metric_class = session.current
    session.metric_events = []
    modes = session.modes()
    for obj in (ns["dataset"], ns["inner"]):
        obj.v10_foreground_mode = modes["foreground_mode"]
    ns["args"].dataset_foreground_mode = ns["args"].val_foreground_mode = modes["foreground_mode"]
    ns["args"].val_background_mode = ns["custom"].background_mode = modes["background_mode"]
    session.stages, session.prompt_calls, session.group_builds, session.roi_calls = [], [], [], []
    session.active_stage = None
    cache_before = len(session.cache_events)
    before = session.check_parameter_guard()
    ns["task"] = dict(task)
    # Never let rows or native audit history retain voxel tensors across tasks.
    ns["rows"] = []
    ns["audit"]["loaded_task_keys"] = []
    ns["audit"]["roi_build_calls"] = []
    ns["audit"]["dynamic_threshold_rows"] = []
    exec(session.program.task_body, ns)
    if len(ns["rows"]) != 1 or not session.roi_calls or not session.group_builds or not session.stages:
        raise ValueError("missing actual native task/ROI/stage execution")
    after = session.check_parameter_guard()
    if before != after:
        raise ValueError("model state changed during inference")
    reserve = _array(ns["reserve_indices"]).astype(int).tolist()
    final = _array(ns["final_indices"]).astype(int).tolist()
    initial = _array(ns["initial_indices"]).astype(int).tolist()
    record = dict(variant=variant, row=dict(ns["row"]), stages=copy.deepcopy(session.stages),
        prompt=dict(session.prompt_audit, group_builds=session.group_builds,
                    actual_consumption_calls=len(session.prompt_calls)),
        roi=dict(initial=initial, reserved=reserve, final=final, builds=session.roi_calls,
                 native_reserved_calls=copy.deepcopy(ns["audit"]["roi_build_calls"]), history=_plain(ns["history"])),
        restoration=dict(session.restoration), parameter_guard_before=before, parameter_guard_after=after,
        parameter_guard_verified=True, resource_binding=session.resource_binding,
        actual_frame_batch=session.batch, native_source_sha256=session.program.source_sha256,
        task_body_ast_sha256=session.program.task_body_sha256, runtime_input_audit=session.input_audit,
        ct_cache_access=session.cache_events[cache_before:], metric_protocol="native_complete_mask_axial_union_nonempty_ten",
        semantic_text=session.expected_semantics[session.current[2]],
        metric_execution=dict(provider_sha256=session.metric_provider_sha256,
            events=session.metric_events, tracker=session.metric_tracker.snapshot(), cache_scope="class_local_metadata_only"),
        final_threshold_audit=_plain(ns["threshold_audit"]))
    if "local_class_id" in task:
        record["row"]["label_id"] = int(task["local_class_id"])
    record = _plain(record)
    validate_record(record, task, variant, session.batch, control=bool(session.job.get("control") or variant == "native_control"))
    # Native loop deletes its principal tensors. Its metric aliases otherwise
    # retain the last ROI logits; release aliases only after actual scoring.
    for name in ("metric_logits", "metric_prediction", "raw_target", "canonical", "_points", "restored", "patch"):
        ns.pop(name, None)
    return record


def validate_record(record, task, variant, actual_frame_batch, control=False):
    """Controller-callable structural and semantic validation of actual evidence."""
    from tools.amos_prompt_ablation_prompts import variant_modes, roi_indices
    modes = variant_modes("baseline" if variant == "native_control" else variant)
    row = record["row"]
    if record.get("variant") != variant or _key(row) != _key(task) or record.get("actual_frame_batch") != actual_frame_batch:
        raise ValueError("runtime row/task/variant/batch mismatch")
    if row.get("metric_space") != "raw_nifti_full_volume" or row.get("local_patch_bounds") is not None:
        raise ValueError("final metric is not native full-volume/full-XY restoration")
    if row.get("class_name") != task.get("class_name") or ("local_class_id" in task and row.get("label_id", task["local_class_id"]) != task["local_class_id"]):
        raise ValueError("original task labels changed")
    for name in METRICS:
        value = float(row[name])
        if not math.isfinite(value) or value < 0 or (name != "pred_gt_ratio" and value > 1):
            raise ValueError("invalid final fullmask metric: " + name)
    r = record.get("restoration", {})
    if (r.get("model"), r.get("epoch"), r.get("weight_mode"), r.get("prompt_parameters_exact"),
            r.get("decoder_parameters_exact"), r.get("guards_passed"), r.get("model_load_count")) != ("v10_2", 490, "live", 418, 212, True, 1):
        raise ValueError("missing exact live E490 restoration")
    if (r.get("full_tensor_comparison_phase") != "post_load"
            or r.get("per_evaluation_guard") != "parameter_version_pointer_shape_dtype_device"
            or record.get("parameter_guard_verified") is not True
            or not record.get("parameter_guard_before")
            or record.get("parameter_guard_before") != record.get("parameter_guard_after")):
        raise ValueError("actual parameter state not guarded before/after forward")
    prompt, roi, stages = record["prompt"], record["roi"], record["stages"]
    if not stages or not prompt.get("group_builds") or not prompt.get("raw_foreground_z_support") or not roi.get("builds"):
        raise ValueError("missing actual prompt/ROI/stage proof")
    for name, value in modes.items():
        if prompt.get(name) != (list(value) if isinstance(value, tuple) else value):
            raise ValueError("prompt transform mode/plane audit mismatch: " + name)
    if set(prompt["planes"]) != set(modes["active_planes"]):
        raise ValueError("prompt audit includes inactive plane")
    expected_points = 2 if modes["background_mode"] == "point" else 0
    if prompt.get("raw_background_points_per_plane") != {name: expected_points for name in modes["active_planes"]}:
        raise ValueError("raw background point budget mismatch")
    for build in roi["builds"]:
        if variant != "native_control":
            expected = roi_indices(build["depth"], prompt["raw_foreground_z_support"], build["context_slices"], build["context_ratio"], build["minframes"])
            if expected.tolist() != build["indices"] or build["source"] != "actual_raw_active_foreground":
                raise ValueError("ROI is not actual active raw foreground support")
        if build["indices"] != roi["initial"]:
            raise ValueError("initial ROI audit mismatch")
    calls = 0
    for stage in stages:
        if stage.get("settings") != {**FROZEN_SETTINGS, "sam2_frame_batch_size": actual_frame_batch}:
            raise ValueError("actual stage frozen settings mismatch")
        if stage.get("memory_calls", {}).get("memory_encoder", 0) != stage["frames"]:
            raise ValueError("actual stage lacks memory-group1 encoder coverage")
        for event in [stage["bundle"], *stage["prompt_calls"]]:
            if event["plane_ids"] != list(modes["plane_ids"]) or event["foreground_mode_id"] != modes["foreground_mode_id"]:
                raise ValueError("actual prompt plane/mode proof mismatch")
            if event["groups_sha256"] != stage["bundle"]["groups_sha256"]:
                raise ValueError("actual prompt raster changed inside stage")
            if event.get("raster_sha256") != prompt["groups_sha256"]:
                raise ValueError("actual raster is not the audited transform")
            expected_counts = [prompt["planes"][plane]["raster_channel_nonzero"] for plane in modes["active_planes"]]
            if event.get("channel_nonzero") != expected_counts:
                raise ValueError("actual prompt channel budget differs from transform")
        if not stage["prompt_calls"] or len(stage["mixed_calls"]) != len(stage["prompt_calls"]) or len(stage["global_pool_calls"]) != 1:
            raise ValueError("missing actual PromptGen/mixed/global-pool hook")
        if len(stage.get("decoder_calls", [])) != 1 or len(stage.get("adapter_calls", [])) != stage["frames"]:
            raise ValueError("missing actual decoder/3D-adapter hook")
        for event in [*stage["prompt_calls"], *stage["global_pool_calls"], *stage["decoder_calls"], *stage["adapter_calls"]]:
            if event["type_ids"] != sorted([modes["foreground_type_id"], modes["background_type_id"]]) or event["role_ids"] != [0, 1]:
                raise ValueError("actual prompt learned type proof mismatch")
        if any(event.get("semantic_text") != record.get("semantic_text") for event in stage["prompt_calls"]):
            raise ValueError("actual legacy AMOS semantic text mismatch")
        audit = stage["threshold_audit"]
        threshold = float(audit["threshold"])
        fallback = str(audit["source"]).startswith("fallback")
        if (fallback and threshold != .60) or (not fallback and not .70 <= threshold <= .75):
            raise ValueError("actual threshold/floor/fallback mismatch")
        calls += len(stage["prompt_calls"])
    if calls != prompt["actual_consumption_calls"] or row["mask_threshold_used"] != stages[-1]["threshold_audit"]["threshold"]:
        raise ValueError("final threshold/actual hook coverage mismatch")
    for name in ("native_source_sha256", "task_body_ast_sha256"):
        if not isinstance(record.get(name), str) or len(record[name]) != 64:
            raise ValueError("missing checked source/AST binding")
    binding = record.get("resource_binding", {})
    if binding.get("gpu") not in (2, 4, 5, 6) or binding.get("free_mib_before_load", 0) < binding.get("minimum_free_mib", 30720):
        raise ValueError("missing actual GPU resource proof")
    execution = record.get("metric_execution", {})
    expected_provider = _sha((Path(__file__).parent/"bone_fast_metrics.py").read_bytes())
    if (execution.get("provider_sha256") != expected_provider
            or execution.get("cache_scope") != "class_local_metadata_only"
            or execution.get("tracker", {}).get("cache_tensor_bytes") != 0
            or execution.get("tracker", {}).get("comparison_failures") != 0
            or len(execution.get("events", [])) != 1):
        raise ValueError("missing exact native metric provider proof")
    event = execution["events"][0]
    if (event.get("key") != list(_key(task)) or event.get("variant") != variant
            or not isinstance(event.get("input_sha256"), str) or len(event["input_sha256"]) != 64
            or event.get("full_mask_shape") != prompt["canonical_shape_dhw"]
            or any(abs(float(event["metrics"][name])-float(row[name])) > 1e-12 for name in METRICS)):
        raise ValueError("final row differs from actual full-mask metric inputs/output")
    if control and (event.get("native_compared") is not True or event.get("max_abs_error") is None
                    or float(event["max_abs_error"]) > 1e-6):
        raise ValueError("control lacks actual exact native metric twin")
    return True


def validate_control(native_record, records):
    """Compare actual unmodified baseline with fresh baseline at one real batch."""
    from tools.amos_prompt_ablation_prompts import VARIANTS
    if len(records) != len(VARIANTS) or {r["variant"] for r in records} != set(VARIANTS):
        raise ValueError("control requires all six fresh variants")
    baseline = next(record for record in records if record["variant"] == "baseline")
    if any(record["actual_frame_batch"] != native_record["actual_frame_batch"] for record in records):
        raise ValueError("native control and variants used different actual batches")
    error = max(abs(float(native_record["row"][name]) - float(baseline["row"][name])) for name in METRICS)
    if error > 1e-6:
        raise ValueError("fresh baseline differs from native control metrics")
    def geometry(record):
        return {key: record["roi"][key] for key in ("initial", "final", "reserved", "history") if key in record["roi"]}
    if geometry(native_record) != geometry(baseline):
        raise ValueError("fresh baseline ROI differs from native control")
    if native_record["stages"] != baseline["stages"]:
        raise ValueError("fresh baseline stages differ from native control")
    for name in ("mask_threshold_used", "dynamic_threshold_source"):
        if native_record["row"][name] != baseline["row"][name]:
            raise ValueError("fresh baseline postprocess differs from native control")
    return dict(passed=True, max_metric_error=error, roi_equal=True, stages_equal=True,
                variants=list(VARIANTS), actual_frame_batch=native_record["actual_frame_batch"], native_control_excluded=True)
