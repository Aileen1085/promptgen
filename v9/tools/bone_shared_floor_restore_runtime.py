"""Independent pre-class restoration audit for the frozen shared Bone runtime."""
import copy
import inspect
from tools.bone_shared_floor_runtime import (
    GridSession as OriginalGridSession, install_v92,
    install_v10 as original_install_v10, tag_carrier,
    install_restoration as original_install_restoration,
)


class GridSession(OriginalGridSession):
    def __init__(self, model, pending, fingerprint, on_class, **kwargs):
        self.native_batch_probe = {}
        def committed(record):
            record['native_batch_probe'] = copy.deepcopy(self.native_batch_probe)
            return on_class(record)
        super().__init__(model, pending, fingerprint, committed, **kwargs)


def probe_native_batches(shared, native, args):
    from tools.bone_shared_logits import tensor_digest
    original = args.sam2_frame_batch_size
    records = {}
    try:
        for batch in (1, 2, 4):
            args.sam2_frame_batch_size = batch
            value, features = shared()
            raw_sha = tensor_digest(value)
            del value
            value, native_features = native()
            native_sha = tensor_digest(value)
            del value
            if raw_sha != native_sha or features != native_features or not features:
                raise ValueError('native/shared same-batch raw or encoder mismatch: '+str(batch))
            records[batch] = dict(raw_sha=raw_sha, native_raw_sha=native_sha,
                encoder_shas=features, native_encoder_shas=native_features,
                exact_same_batch=True)
    finally:
        args.sam2_frame_batch_size = original
    return records


def install_v10(session):
    import v10.infer.infer_eight_source_adaptive_v10_2 as evaluator
    import v10_sam2_memory_roi as memory
    native_decode = evaluator._dynamic_adaptive_decode
    native_encoder = memory.encode_case_image_views
    original_install_v10(session)
    original_raw = session.raw
    # The compiled raw callback is bound during installation, so replace its
    # bound function's implementation through the session object before install.
    def raw(compute, video, bundle, class_id, args):
        if getattr(session, 'shared_control', False) and not session.native_batch_probe:
            if getattr(args, 'v10_3_local_patch', False) or getattr(args, 'v10_3_native_image_size', None) is not None:
                raise ValueError('native twin probe requires original v10.2 full-frame stage')
            cells = inspect.getclosurevars(compute).nonlocals
            feature_shas = []
            from tools.bone_shared_logits import tensor_digest
            def encode(*a, **k):
                result = native_encoder(*a, **k)
                feature_shas.append(tensor_digest(result))
                return result
            previous_encoder = memory.encode_case_image_views
            previous_capture = session.capture
            def shared():
                feature_shas.clear()
                return compute(), list(feature_shas)
            def native():
                feature_shas.clear()
                result = native_decode(cells['adapter'], cells['prompt'], video, bundle,
                    class_id, args, cells['device'], (0, int(video.shape[0])))
                if tuple(result[2]) != (0, int(video.shape[0])):
                    raise ValueError('native twin changed exact stage bounds')
                return result[0], list(feature_shas)
            try:
                memory.encode_case_image_views = encode
                session.capture = lambda *a, **k: None
                session.native_batch_probe = probe_native_batches(shared, native, args)
            finally:
                memory.encode_case_image_views = previous_encoder
                session.capture = previous_capture
        return original_raw(compute, video, bundle, class_id, args)
    session.raw = raw
    # Rebind the installed decode callback; no other native operations change.
    from tools.bone_shared_floor_runtime import compile_function, v10_decode_source
    evaluator._dynamic_adaptive_decode = compile_function(native_decode,
        v10_decode_source, {'_shared_raw_call': session.raw})


def install_restoration(session, save_proof):
    if session.model != 'v9_2':
        return original_install_restoration(session, save_proof)
    import finetune_multisource_sam2_v9_2_precision as entry
    import finetune_totalseg_sam2_promptgen_v9_joint as joint
    import infer.infer_eight_source_adaptive_v9_2 as evaluator
    from tools.amos_module_sensitivity import compare_parameters
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
        if 'state' in holder:raise ValueError('unexpected repeated checkpoint load')
        holder.update(prompt=prompt, state=result[0])
        return result
    entry.base.load_prompt_generator_checkpoint = load
    native_begin = session.begin
    def begin(task):
        # Native main loads sam2_mask_decoder after the prompt loader returns.
        # Check both restored modules at the first class, before any forward.
        if session.restoration is None:
            state = holder['state']
            proof = dict(model='v9_2', epoch=int(state['epoch']), weight_mode='live',
                prompt_parameters_exact=compare_parameters(holder['prompt'], state['prompt_generator']),
                decoder_parameters_exact=compare_parameters(holder['decoder'], state['sam2_mask_decoder']),
                guards_passed=True)
            save_proof(proof)
            session.restoration = proof
        return native_begin(task)
    session.begin = begin
    native_bind = evaluator._install_exact_prompt_factory
    def bind(module, *args, **kwargs):
        result = native_bind(module, *args, **kwargs)
        module.base.make_dataset = validation_only_factory(module.base.make_dataset, session.training_audit)
        return result
    evaluator._install_exact_prompt_factory = bind
