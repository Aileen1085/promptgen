"""Additive, RAM-only common-stage reuse for the native Bone floor grid."""
from __future__ import annotations

import ast
import copy
import inspect
import json
from pathlib import Path
import textwrap

from tools.bone_floor_protocol import METRICS, key, validate_rows, FLOORS


def _tree(source):
    return ast.parse(textwrap.dedent(source))


def _function(tree, name):
    found = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(found) != 1:
        raise ValueError('native function anchor mismatch: ' + name)
    return found[0]


def v92_source(source):
    """Extract only native raw forward, preserving its operations verbatim as AST."""
    tree = _tree(source)
    installer = _function(tree, 'install_dynamic_validation_patch')
    full = _function(installer, 'dynamic_full_volume_task')
    first = next((i for i, n in enumerate(full.body) if isinstance(n, ast.Assign)
                  and ast.unparse(n.targets[0]) == 'logits_cpu'), None)
    end = next((i for i, n in enumerate(full.body) if isinstance(n, ast.For)
                and ast.unparse(n.target) == '(name, value)'), None)
    if first != 0 or end is None or end <= first:
        raise ValueError('unexpected native v92 raw-stage boundaries')
    prefix = copy.deepcopy(full.body[first:end])
    if 'calibrate_prompt_logits' in ast.unparse(ast.Module(body=prefix, type_ignores=[])):
        raise ValueError('thresholding leaked into raw stage')
    raw = copy.deepcopy(full)
    raw.name = '_shared_raw_v92'
    raw.body = prefix + [ast.Return(ast.Name('logits', ast.Load()))]
    raw.decorator_list = []
    names = [a.arg for a in full.args.args + full.args.kwonlyargs]
    call = '_shared_raw_call(lambda: _shared_raw_v92(' + ', '.join(
        name if name not in [a.arg for a in full.args.kwonlyargs] else name + '=' + name
        for name in names) + '), video, prompt_bundle, class_id, args)'
    full.body = _tree('total_frames = int(video.shape[0])\nlogits_cpu = []\nlogits = ' + call).body + full.body[end:]
    installer.body.insert(installer.body.index(full), raw)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def v10_decode_source(source):
    tree = _tree(source)
    outer = _function(tree, '_dynamic_adaptive_decode')
    stage = _function(outer, 'decode_current_patch')
    first = stage.body[:2]
    if (len(first) != 2 or not all(isinstance(n, ast.Assign) for n in first)
            or ast.unparse(first[0].targets[0]) != '(current_tokens, current_anchor)'
            or ast.unparse(first[1].targets[0]) != '(current_logits, _unused_sparse)'
            or 'generate_case_3d_tokens' not in ast.unparse(first[0])
            or 'decode(' not in ast.unparse(first[1])):
        raise ValueError('unexpected v10 native raw-stage boundaries')
    helper = _tree('def _shared_compute_v10():\n    return None').body[0]
    helper.body = copy.deepcopy(first) + [ast.Return(ast.Name('current_logits', ast.Load()))]
    replacement = _tree('current_logits = _shared_raw_call(_shared_compute_v10, region_video, region_bundle, class_id, args)\ncurrent_tokens, current_anchor = None, None').body
    stage.body = [helper] + replacement + stage.body[2:]
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def v10_main_source(source):
    tree = _tree(source)
    main = _function(tree, 'main')
    candidates = [n for n in main.body if isinstance(n, ast.For) and ast.unparse(n.target) == 'task']
    if len(candidates) != 1:
        raise ValueError('native v10 task loop mismatch')
    task = candidates[0]
    starts = [i for i, n in enumerate(task.body) if isinstance(n, ast.With)
              and any('torch.no_grad()' == ast.unparse(x.context_expr) for x in n.items)]
    if len(starts) != 1 or not isinstance(task.body[-2], ast.Delete):
        raise ValueError('native v10 inference/cleanup anchors mismatch')
    start = starts[0]
    tail = copy.deepcopy(task.body[start:])
    invariant_names = ('raw_target', 'canonical', '(zoom_x, zoom_y, zoom_z)', 'spacing')
    invariants = [n for n in tail if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) in invariant_names]
    if len(invariants) != 4 or tuple(ast.unparse(n.targets[0]) for n in invariants) != invariant_names:
        raise ValueError('native GT/spacing load anchors changed')
    tail = [n for n in tail if n not in invariants]

    class Rewrite(ast.NodeTransformer):
        saves = 0
        scores = 0
        audits = 0
        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == 'restore_and_score_full_volume':
                node.func.id = '_shared_score'
                node.args.insert(0, ast.Name('restore_and_score_full_volume', ast.Load()))
                self.scores += 1
            return node
        def visit_Expr(self, node):
            self.generic_visit(node)
            value = ast.unparse(node.value)
            if value == 'rows.append(row)':
                self.saves += 1
                emit = _tree('_shared_emit_v10(task, row, threshold_audit, history, final_bounds, reserve_indices, patch_geometry, args)').body[0]
                guard = ast.If(ast.Call(ast.Name('_shared_primary', ast.Load()), [], []), [node], [])
                return [emit, guard]
            if value.startswith("audit['dynamic_threshold_rows'].append("):
                self.audits += 1
                return ast.If(ast.Call(ast.Name('_shared_primary', ast.Load()), [], []), [node], [])
            return node
        def visit_Delete(self, node):
            if ast.unparse(node) != 'del video, bundle, target, logits, prediction, _full_prediction':
                raise ValueError('unexpected v10 cleanup')
            return _tree('del logits, prediction, _full_prediction, metric_logits, metric_prediction').body[0]

    rewrite = Rewrite()
    wrapped = ast.Module(body=tail, type_ignores=[])
    wrapped = rewrite.visit(wrapped)
    if (rewrite.saves, rewrite.scores, rewrite.audits) != (1, 1, 1):
        raise ValueError('native v10 metric anchors changed')
    loop = _tree('for _shared_floor in _shared_begin_v10(task, args):\n    _shared_set_floor(_shared_floor, args)').body[0]
    loop.body += wrapped.body
    task.body = task.body[:start] + invariants + [loop] + _tree('_shared_finish_v10(task)\ndel video, bundle, target, raw_target, canonical').body
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def compile_function(function, transform, bindings):
    source = transform(inspect.getsource(function))
    namespace = dict(function.__globals__, **bindings)
    exec(compile(source, function.__code__.co_filename, 'exec'), namespace)
    return namespace[function.__name__]


def walk_floors(floors, initial, forward, branch):
    """Small executable DAG contract used to test branch-specific stage reuse."""
    memo = {}
    result = {}
    for floor in floors:
        state = initial
        seen = set()
        while True:
            if state in seen:
                raise ValueError('cyclic expansion state')
            seen.add(state)
            if state not in memo:
                memo[state] = forward(state)
            value = memo[state]
            state = branch(value, floor)
            if state is None:
                result[floor] = value
                break
    return result


def _plain(value):
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(x) for x in value]
    if isinstance(value, set):
        return sorted(_plain(x) for x in value)
    return {'python_type': type(value).__module__ + '.' + type(value).__name__, 'value': str(value)}


def tag_carrier(payload,pending):
    value=copy.deepcopy(payload);floors={key(x['task']):x['floors'][0] for x in pending}
    value['carrier_only']=True;value['shared_grid_results']='audited per-class floor records, not this native carrier output'
    value['carrier_floors']=[dict(key=list(k),floor=f) for k,f in floors.items()]
    for row in value.get('per_case_class',()):
        row['carrier_only']=True;row['carrier_floor']=floors[key(row)]
    return value


class GridSession:
    """One restored model/case; class-local raw memo and case-local encoder LRU."""
    def __init__(self, model, pending, fingerprint, on_class, raw_bytes=6*1024**3, feature_bytes=2*1024**3):
        from tools.bone_shared_logits import TensorMemo
        self.model, self.fingerprint, self.on_class = model, fingerprint, on_class
        self.pending = {key(x['task']): (x['task'], tuple(x['floors'])) for x in pending}
        if len(self.pending) != len(pending) or not self.pending:
            raise ValueError('duplicate/empty pending class inventory')
        for task, floors in self.pending.values():
            if not floors or len(floors) != len(set(floors)) or any(f not in FLOORS for f in floors):
                raise ValueError('invalid requested floors')
        self.raw_memo, self.feature_memo = TensorMemo(raw_bytes), TensorMemo(feature_bytes)
        self.memory_calls = {'memory_encoder': 0, 'memory_attention': 0}
        self.restoration = None
        self.training_audit = {'training_dataset_constructions_skipped': 0}
        self.current = None
        self.floor = None
        self.raw_proofs = {}
        self.last_stage = None
        self.stage_rows = {}
        self.results = {}
        self.case = None

    def begin(self, task):
        identity = key(task)
        if identity not in self.pending:
            raise ValueError('unexpected task in shared case worker')
        if self.case != identity[:2]:
            self.feature_memo.clear()
            self.case = identity[:2]
        self.raw_memo.clear()
        self.current = identity
        self.floor = None
        self.raw_proofs = {}
        self.last_stage = None
        self.stage_rows, self.results = {}, {}
        return self.pending[identity][1]

    def set_floor(self, floor, args):
        if self.current is None or floor not in self.pending[self.current][1]:
            raise ValueError('floor is not pending for current class')
        self.floor = float(floor)
        for name,value in dict(dynamic_threshold_max=.75,dynamic_threshold_fallback=.60,
                prompt_logit_neighborhood_radius=2,dynamic_threshold_min_logit_separation=.50).items():
            if hasattr(args,name) and getattr(args,name)!=value:
                raise ValueError('changed frozen native postprocess setting: '+name)
            setattr(args,name,value)
        args.dynamic_threshold_min = self.floor
        self.stage_rows.setdefault(self.floor, [])

    def _settings(self, args):
        return {k: _plain(v) for k, v in vars(args).items() if k != 'dynamic_threshold_min'}

    def raw(self, compute, video, bundle, class_id, args):
        from tools.bone_floor_grid import stage_record
        if self.current is None or int(class_id) != self.current[2]:
            raise ValueError('raw class binding mismatch')
        stage_record(self.model, args, self.current, video.shape[0])
        if args.dynamic_threshold_min != self.floor:
            raise ValueError('raw floor context mismatch')
        before = dict(self.memory_calls)
        inputs = dict(video=video, bundle=bundle, class_id=int(class_id), settings=self._settings(args))
        value = self.raw_memo.call(self.fingerprint, inputs, compute)
        event = dict(self.raw_memo.records[-1])
        digest = event['key']
        hit = event['event'] == 'hit'
        if not hit:
            calls = {k: self.memory_calls[k] - before[k] for k in before}
            if self.model == 'v10_2' and calls['memory_encoder'] <= 0:
                raise ValueError('native memory did not execute on shared-stage producer')
            self.raw_proofs[digest] = dict(raw_key=digest, actual_memory_calls=calls,
                frames=int(video.shape[0]), settings=self._settings(args), output_digest=event['output_digest'],
                inputs_digest=event['inputs_digest'],namespace_digest=event['namespace_digest'])
        if digest not in self.raw_proofs:
            raise ValueError('cache hit without current-class producer proof')
        self.last_stage = dict(raw_key=digest, cache_hit=hit, producer=self.raw_proofs[digest])
        return value

    def capture(self, args, audit):
        from tools.bone_floor_grid import stage_record
        if self.last_stage is None:
            raise ValueError('postprocessing without raw-stage proof')
        row = stage_record(self.model, args, self.current, self.last_stage['producer']['frames'])
        row.update(threshold=float(audit['threshold']), source=str(audit['source']),
                   raw_key=self.last_stage['raw_key'], cache_hit=self.last_stage['cache_hit'])
        self.stage_rows[self.floor].append(row)

    def emit(self, task, metric, audit, **geometry):
        if key(task) != self.current or self.floor in self.results:
            raise ValueError('duplicate/misaligned shared floor result')
        row = {**task, **{name: float(metric[name]) for name in METRICS},
               'label_id': int(task.get('local_class_id', key(task)[2]-10000)),
               'mask_threshold_used': float(audit['threshold']),
               'dynamic_threshold_source': str(audit['source'])}
        validate_rows([row], [task])
        self.results[self.floor] = dict(floor=self.floor, row=row,
            stages=copy.deepcopy(self.stage_rows[self.floor]), geometry=_plain(geometry))

    def finish(self, task):
        from tools.bone_floor_grid import validate_stages
        expected = self.pending[key(task)][1]
        if key(task) != self.current or set(expected) != set(self.results):
            raise ValueError('incomplete class floor grid')
        restoration = self.restoration or {}
        counts = (438, 192, 360) if self.model == 'v9_2' else (418, 212, 490)
        if (restoration.get('model') != self.model or restoration.get('weight_mode') != 'live'
                or not restoration.get('guards_passed') or restoration.get('epoch') != counts[2]
                or restoration.get('prompt_parameters_exact') != counts[0]
                or restoration.get('decoder_parameters_exact') != counts[1]):
            raise ValueError('class completion without exact initial restoration proof')
        if self.training_audit['training_dataset_constructions_skipped'] != (1 if self.model == 'v9_2' else 0):
            raise ValueError('training-factory skip count mismatch')
        for floor, result in self.results.items():
            validate_stages(result['stages'], [task], self.model, floor)
            if result['row']['mask_threshold_used'] != result['stages'][-1]['threshold']:
                raise ValueError('final floor/threshold binding mismatch')
        record = dict(model=self.model, task=task, results=[self.results[f] for f in expected],
            restoration=restoration, training_audit=dict(self.training_audit),
            raw_producers=copy.deepcopy(self.raw_proofs),
            cache_stats=dict(raw=dict(self.raw_memo.stats), image=dict(self.feature_memo.stats)),
            metric_protocol='native final complete mask ten metrics; independent per-floor AutoExpand',
            carrier_floor=expected[0], carrier_only=True)
        self.on_class(record)
        self.raw_memo.clear()

    def encoder(self, native, *args, **kwargs):
        """Exact tensor inputs and native device layout; never key on case IDs alone."""
        import torch
        def devices(value):
            if torch.is_tensor(value): return str(value.device)
            if isinstance(value, dict): return {k: devices(v) for k, v in value.items()}
            if isinstance(value, list): return [devices(v) for v in value]
            if isinstance(value, tuple): return tuple(devices(v) for v in value)
            return None
        def restore(value, layout):
            if torch.is_tensor(value): return value.to(layout)
            if isinstance(value, dict): return {k: restore(v, layout[k]) for k,v in value.items()}
            if isinstance(value, list): return [restore(v,l) for v,l in zip(value,layout)]
            if isinstance(value, tuple): return tuple(restore(v,l) for v,l in zip(value,layout))
            return value
        # Adapter is a restored fixed model, represented by fingerprint, never serialized.
        data = [x for x in args if torch.is_tensor(x)]
        plain = [(self._settings(x) if hasattr(x, '__dict__') else _plain(x))
                 for x in args if not torch.is_tensor(x) and not hasattr(x, 'parameters')]
        def produce():
            output = native(*args, **kwargs)
            return output, devices(output)
        value, layout = self.feature_memo.call(self.fingerprint + ':image:' + native.__name__,
            dict(data=data, plain=plain, kwargs=_plain(kwargs)), produce)
        return restore(value, layout)


def install_v92(session):
    import infer.v9_2_dynamic_validation_patch as dynamic
    import infer.totalseg_dynamic_runtime as thresholding
    native_calibrate = thresholding.calibrate_prompt_logits
    def calibrate(logits, bundle, args):
        result = native_calibrate(logits, bundle, args)
        session.capture(args, result[1])
        return result
    thresholding.calibrate_prompt_logits = calibrate
    native_install = dynamic.install_dynamic_validation_patch
    shared_install = compile_function(native_install, v92_source, {'_shared_raw_call': session.raw})

    def install(runtime, audit_rows, **settings):
        true_adaptive = runtime._evaluate_adaptive_prompt_roi_task
        functions = {}
        for floor in FLOORS:
            runtime._evaluate_adaptive_prompt_roi_task = true_adaptive
            private = []
            shared_install(runtime, private, **dict(settings, minimum=floor))
            functions[floor] = (runtime._evaluate_full_volume_task, runtime._evaluate_adaptive_prompt_roi_task, private)
        base_full = functions[.35][0]
        native_encoder = runtime.base.encode_image_features_in_chunks
        runtime.base.encode_image_features_in_chunks = lambda *a, **k: session.encoder(native_encoder, *a, **k)
        def grid(*args, **kwargs):
            class_id = int(args[5]);identity = next(k for k in session.pending if k[2] == class_id)
            task = session.pending[identity][0];floors = session.begin(task)
            carrier = None;carrier_audit = None
            old_floor = getattr(args[6], 'dynamic_threshold_min', .35)
            try:
                for floor in floors:
                    session.set_floor(floor, args[6])
                    full, adaptive, private = functions[floor];private.clear()
                    runtime._evaluate_full_volume_task = full
                    metric = adaptive(*args, **kwargs)
                    if len(private) != 1: raise ValueError('native v92 final audit count mismatch')
                    session.emit(task, metric, private[-1])
                    if floor == floors[0]: carrier, carrier_audit = metric, dict(private[-1])
                session.finish(task)
                audit_rows.append(carrier_audit)
                return carrier
            finally:
                runtime._evaluate_full_volume_task = base_full
                args[6].dynamic_threshold_min = old_floor
        runtime._evaluate_adaptive_prompt_roi_task = grid
        runtime._evaluate_full_volume_task = base_full
    dynamic.install_dynamic_validation_patch = install


def install_v10(session):
    import v10.infer.infer_eight_source_adaptive_v10_2 as evaluator
    import v10.infer.infer_amos_sam2_promptgen_v10 as scorer
    import v10_sam2_memory_roi as memory
    native_score = scorer.restore_and_score_full_volume
    native_post = scorer.postprocess_v10_logits_from_prompts
    native_encoder = memory.encode_case_image_views
    memory.encode_case_image_views = lambda *a, **k: session.encoder(native_encoder, *a, **k)
    def post(*args, **kwargs):
        if float(kwargs['minimum']) != session.floor: raise ValueError('v10 active floor mismatch')
        result = native_post(*args, **kwargs)
        session.capture(session.active_args, result[1])
        return result
    scorer.postprocess_v10_logits_from_prompts = post
    evaluator._dynamic_adaptive_decode = compile_function(evaluator._dynamic_adaptive_decode,
        v10_decode_source, {'_shared_raw_call': session.raw})
    def begin(task, args):
        session.active_args = args
        return session.begin(task)
    def primary(): return session.floor == session.pending[session.current][1][0]
    def score(carrier, *args, **kwargs):
        return (carrier if primary() else native_score)(*args, **kwargs)
    def emit(task, row, audit, history, bounds, indices, geometry, args):
        session.emit(task, row, audit, history=history, bounds=bounds,
            final_frame_indices=indices[bounds[0]:bounds[1]].tolist(),
            patch_bounds=list(geometry.bounds) if geometry is not None else None)
    evaluator.main = compile_function(evaluator.main, v10_main_source, dict(
        _shared_begin_v10=begin, _shared_set_floor=session.set_floor, _shared_primary=primary,
        _shared_score=score, _shared_emit_v10=emit, _shared_finish_v10=session.finish))


def install_restoration(session, save_proof):
    """Audit actual restored parameters before the first class can be committed."""
    from tools.amos_module_sensitivity import compare_parameters
    def proof(prompt, decoder, state):
        decoder_key = 'sam2_mask_decoder' if session.model == 'v9_2' else 'mask_decoder'
        value = dict(model=session.model, epoch=int(state['epoch']), weight_mode='live',
            prompt_parameters_exact=compare_parameters(prompt, state['prompt_generator']),
            decoder_parameters_exact=compare_parameters(decoder, state[decoder_key]), guards_passed=True)
        session.restoration = value
        save_proof(value)
    if session.model == 'v9_2':
        import finetune_multisource_sam2_v9_2_precision as entry
        import finetune_totalseg_sam2_promptgen_v9_joint as joint
        import infer.infer_eight_source_adaptive_v9_2 as evaluator
        from tools.bone_floor_recovery import validation_only_factory
        holder = {}
        native_init = joint.SAM2MedicalAdapterV9.__init__
        def init(self, *args, **kwargs):
            native_init(self, *args, **kwargs)
            holder['decoder'] = self.sam.sam_mask_decoder
        joint.SAM2MedicalAdapterV9.__init__ = init
        native_load = entry.base.load_prompt_generator_checkpoint
        def load(path, prompt, *args, **kwargs):
            result = native_load(path, prompt, *args, **kwargs)
            proof(prompt, holder['decoder'], result[0])
            return result
        entry.base.load_prompt_generator_checkpoint = load
        native_bind = evaluator._install_exact_prompt_factory
        def bind(module, *args, **kwargs):
            result = native_bind(module, *args, **kwargs)
            module.base.make_dataset = validation_only_factory(module.base.make_dataset, session.training_audit)
            return result
        evaluator._install_exact_prompt_factory = bind
    else:
        import v10.finetune_totalseg_amos_magic_v10_2_joint as entry
        native_make = entry.make_prompt_v10_2
        def make(args, adapter, device):
            for name in session.memory_calls:
                def count(_module, _args, label=name): session.memory_calls[label] += 1
                getattr(adapter.sam, name).register_forward_pre_hook(count)
            return native_make(args, adapter, device)
        entry.make_prompt_v10_2 = make
        native_load = entry.load_init_v10_2
        def load(path, prompt, decoder):
            result = native_load(path, prompt, decoder)
            proof(prompt, decoder, result)
            return result
        entry.load_init_v10_2 = load
