"""Isolated five-source native prompt interventions; no source writes or arrays exported."""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import MethodType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
E490 = 'v10/output/v10_2_ct13_e470_zero_delta_e510_20261001/20261001_215853/epoch490.pth'
HARD = 'v10/output/v10_2_ct13_hard_source_20_20261006/20261006_143104'
SOURCES = {
    'magic': dict(count=35, epoch=490, floor=.60, checkpoint=E490),
    'msd_task07': dict(count=40, epoch=490, floor=.35, checkpoint=E490),
    'msd_task10': dict(count=20, epoch=15, floor=.35, checkpoint=HARD+'/epoch015.pth'),
    'parse2022': dict(count=20, epoch=20, floor=.35, checkpoint=HARD+'/epoch020.pth'),
    'kits23': dict(count=58, epoch=490, floor=.35, checkpoint=E490),
}
VARIANTS = ('baseline', 'fg_box_bg_point', 'fg_box_bg_scribble')
METRICS = ('dice','mean_slice_dice','iou','precision','recall','specificity',
           'pred_gt_ratio','mean_slice_nsd_1mm','mean_slice_nsd_2mm','mean_slice_nsd_3mm')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''): digest.update(block)
    return digest.hexdigest()


def key(row):
    value = row.get('global_class_id', row.get('class_id'))
    if value is None or ('class_id' in row and int(row['class_id']) != int(value)):
        raise ValueError('missing/conflicting global class ID')
    return str(row['source_name']), str(row['case_id']), int(value)


def resource_snapshot(text, gpu, expected_uuid):
    rows = {int(p[0]): p for p in ([x.strip() for x in line.split(',')] for line in text.splitlines() if line.strip())}
    if gpu not in (0,3) or gpu not in rows or rows[gpu][1] != expected_uuid or int(rows[gpu][2]) < 30720:
        raise ValueError('physical GPU UUID/free memory/allowed GPU requirement failed')
    return dict(gpu=gpu,gpu_uuid=expected_uuid,free_mib=int(rows[gpu][2]),minimum_free_mib=30720)


def oom_next_batch(error, batch):
    return batch//2 if batch in (2,4) and ('cuda out of memory' in error.lower() or 'torch.OutOfMemoryError' in error) else None


def private_hooks(source):
    path = ROOT/'tools/amos_prompt_ablation_runtime.py'
    spec = importlib.util.spec_from_file_location('_five_source_private_hooks', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.FROZEN_SETTINGS = dict(module.FROZEN_SETTINGS, dynamic_threshold_min=SOURCES[source]['floor'])
    module.validate_record = validate_record
    return module


def validate_restoration(proof, source):
    config = SOURCES[source]
    if (proof.get('model'),proof.get('epoch'),proof.get('weight_mode'),proof.get('prompt_parameters_exact'),
            proof.get('decoder_parameters_exact'),proof.get('guards_passed'),proof.get('model_load_count'),proof.get('checkpoint')) != (
            'v10_2',config['epoch'],'live',418,212,True,1,config['checkpoint']):
        raise ValueError('source-specific exact checkpoint restoration required')
    if config['epoch'] != 490:
        encoder = proof.get('encoder_restore',{})
        if encoder.get('epoch') != config['epoch'] or encoder.get('all_tensors_exact') is not True or encoder.get('restored_tensor_count',0) <= 0:
            raise ValueError('source-specific exact encoder restoration required')
    return True


def validate_record(record, task, variant, actual_frame_batch, control=False):
    from tools.amos_prompt_ablation_prompts import variant_modes, roi_indices
    row, source = record['row'], key(task)[0]
    config = SOURCES[source]
    if variant not in (*VARIANTS,'native_control') or record['variant'] != variant or key(row) != key(task):
        raise ValueError('task/variant mismatch')
    if row.get('metric_space') != 'raw_nifti_full_volume' or row.get('local_patch_bounds') is not None:
        raise ValueError('native full-volume metric required')
    if row.get('class_name') != task.get('class_name') or ('local_class_id' in task and row.get('label_id') != task['local_class_id']):
        raise ValueError('task label identity changed')
    for name in METRICS:
        value = float(row[name])
        if not math.isfinite(value) or value < 0 or (name != 'pred_gt_ratio' and value > 1):
            raise ValueError('nonfinite/invalid metric '+name)
    restoration = record['restoration']
    validate_restoration(restoration,source)
    binding = record['resource_binding']
    resource_snapshot(f"{binding['gpu']}, {binding['gpu_uuid']}, {binding['free_mib_before_load']}",binding['gpu'],binding['gpu_uuid'])
    if not binding.get('cuda_probe') or not binding.get('first_snapshot') or not binding.get('second_snapshot'):
        raise ValueError('both free-memory checks and actual CUDA probe required')
    for name in ('native_source_sha256','task_body_ast_sha256'):
        if not isinstance(record.get(name),str) or len(record[name]) != 64: raise ValueError('checked source/AST binding missing')
    if 'normal_finish' in record:
        finish=record['normal_finish'];validate_restoration(finish,source)
        if finish.get('successful_native_completion') is not True or finish.get('full_tensor_comparison_phase') != 'normal_worker_completion':
            raise ValueError('normal worker completion tensor proof missing')
    if not record['parameter_guard_verified'] or record['parameter_guard_before'] != record['parameter_guard_after']:
        raise ValueError('actual parameter mutation')
    if actual_frame_batch not in (1,2,4) or record['actual_frame_batch'] != actual_frame_batch:
        raise ValueError('frame batch mismatch')
    prompt, roi, stages = record['prompt'], record['roi'], record['stages']
    modes = variant_modes('baseline' if variant == 'native_control' else variant)
    if not stages or not prompt['group_builds'] or not roi['builds']:
        raise ValueError('actual prompt/ROI/stages missing')
    for name, value in modes.items():
        if prompt.get(name) != (list(value) if isinstance(value,tuple) else value): raise ValueError('mode audit mismatch')
    for build in roi['builds']:
        if build['indices'] != roi['initial']: raise ValueError('ROI initial mismatch')
        if variant != 'native_control' and build['indices'] != roi_indices(build['depth'],prompt['raw_foreground_z_support'],build['context_slices'],build['context_ratio'],build['minframes']).tolist():
            raise ValueError('ROI not derived from actual foreground')
    calls = 0
    for stage in stages:
        expected = dict(private_hooks(source).FROZEN_SETTINGS,sam2_frame_batch_size=actual_frame_batch)
        if stage['settings'] != expected: raise ValueError('source-specific native settings changed')
        if stage['memory_calls']['memory_encoder'] != stage['frames']: raise ValueError('memory group1 coverage mismatch')
        if not stage['prompt_calls'] or len(stage['mixed_calls']) != len(stage['prompt_calls']) or len(stage['global_pool_calls']) != 1 or len(stage['decoder_calls']) != 1 or len(stage['adapter_calls']) != stage['frames']:
            raise ValueError('actual consumption hook coverage incomplete')
        for event in [stage['bundle'],*stage['prompt_calls']]:
            if event['plane_ids'] != list(modes['plane_ids']) or event['foreground_mode_id'] != modes['foreground_mode_id'] or event['raster_sha256'] != prompt['groups_sha256']:
                raise ValueError('actual consumed prompt/coordinate/type mismatch')
            counts = [prompt['planes'][plane]['raster_channel_nonzero'] for plane in modes['active_planes']]
            if event['channel_nonzero'] != counts: raise ValueError('actual consumed raster differs')
        for event in [*stage['prompt_calls'],*stage['global_pool_calls'],*stage['decoder_calls'],*stage['adapter_calls']]:
            if event['type_ids'] != sorted([modes['foreground_type_id'],modes['background_type_id']]) or event['role_ids'] != [0,1]:
                raise ValueError('actual learned type/role mismatch')
        if any(event['semantic_text'] != record['semantic_text'] for event in stage['prompt_calls']): raise ValueError('source semantic mismatch')
        audit = stage['threshold_audit']; threshold = float(audit['threshold'])
        if (str(audit['source']).startswith('fallback') and threshold != .60) or not config['floor'] <= threshold <= .75:
            raise ValueError('source threshold mismatch')
        calls += len(stage['prompt_calls'])
    if calls != prompt['actual_consumption_calls'] or row['mask_threshold_used'] != stages[-1]['threshold_audit']['threshold']:
        raise ValueError('actual consumption/final threshold mismatch')
    metric = record['metric_execution']
    if metric['provider_sha256'] != sha(ROOT/'tools/bone_fast_metrics.py') or metric['tracker']['cache_tensor_bytes'] != 0 or metric['tracker']['comparison_failures'] != 0 or len(metric['events']) != 1:
        raise ValueError('actual full-mask metric provider mismatch')
    event = metric['events'][0]
    if event['key'] != list(key(task)) or event['full_mask_shape'] != prompt['canonical_shape_dhw'] or any(float(event['metrics'][m]) != float(row[m]) for m in METRICS):
        raise ValueError('scorer/row mismatch')
    if control and (not event['native_compared'] or event['max_abs_error'] is None or event['max_abs_error'] > 1e-6): raise ValueError('native metric twin missing')
    if not record.get('ct_cache_access') or any(event.get('mode') != 'read_only_hit' for event in record['ct_cache_access']):
        raise ValueError('existing case-level shared CT cache reuse required')
    return True


def initialize(job, paths):
    """Checked native initialization + unchanged task body in one private namespace."""
    source = job['source']; config = SOURCES[source]; hook = private_hooks(source)
    class Session(hook.Session):
        def package(self, inner, index, class_id):
            from nninteractive_test.infer.adaptive_prompt_cache import _read
            ns = self.namespace; identity = int(index),int(class_id)
            if self.raw_cache is not None and self.raw_cache[0] == identity: return self.raw_cache[1:]
            task = next(t for t in ns['tasks'] if key(t)[2] == int(class_id))
            path = Path(ns['custom'].shared_adaptive_prompt_root)/'metadata/shared_adaptive_prompt_v1'/source/ns['prompt_index'][key(task)]
            package = _read(path)
            if package is None: raise ValueError('missing frozen prompt package '+str(path))
            target = inner._load_canonical_target(int(index),int(class_id))
            package = ns['align_cached_prompt_package'](source,package,expected_shape_dhw=tuple(target.shape))
            previous = inner.prompt_use_box,inner.prompt_use_background_scribble
            try:
                inner.prompt_use_box = inner.prompt_use_background_scribble = True
                native = {p:inner._plane_prompt_group(target,p,int(package[p+'_slice'])) for p in ('coronal','sagittal')}
            finally: inner.prompt_use_box,inner.prompt_use_background_scribble = previous
            self.raw_cache = identity,package,native,str(path),sha(path)
            return self.raw_cache[1:]

        def restore_proof(self):
            ns = self.namespace
            proof = dict(model='v10_2',epoch=int(ns['state']['epoch']),weight_mode='live',
                prompt_parameters_exact=hook.compare_parameters(ns['prompt'],ns['state']['prompt_generator']),
                decoder_parameters_exact=hook.compare_parameters(ns['adapter'].sam.sam_mask_decoder,ns['state']['mask_decoder']),
                guards_passed=True,model_load_count=self.model_loads,checkpoint=config['checkpoint'],
                full_tensor_comparison_phase='post_load' if self.restoration is None else 'normal_worker_completion',
                per_evaluation_guard='parameter_version_pointer_shape_dtype_device')
            if proof['epoch'] != config['epoch'] or proof['model_load_count'] != 1: raise ValueError('source checkpoint mismatch')
            if config['epoch'] != 490:
                import torch
                named = dict(ns['adapter'].sam.named_parameters())
                stored = ns['state']['sam_tuning_state']['image_encoder']
                for name in self.encoder_names:
                    if not torch.equal(named[name].detach().cpu(),stored[name].to(dtype=named[name].dtype)):
                        raise ValueError('encoder delta not exact '+name)
                proof['encoder_restore'] = dict(epoch=config['epoch'],restored_tensor_count=len(self.encoder_names),all_tensors_exact=True)
            if self.restoration is None:
                self.restoration = proof; self.initial_parameter_guard = self.check_parameter_guard(initialize=True)
            validate_restoration(proof,source)
            return dict(proof)

    session = Session(job,paths); session.batch = int(job['frame_batch'])
    if session.batch not in (1,2,4): raise ValueError('frame batch cap4')
    session.resource_binding = dict(job['resource_binding'])
    for path in (ROOT,ROOT/'v10'):
        if str(path) not in sys.path: sys.path.insert(0,str(path))
    import ct13_training_entry
    import v9_2_extended_protocol as protocol
    split_path = ROOT/paths['base_split_json']; split = json.loads(split_path.read_text(encoding='utf-8'))
    protocol.SOURCE_ORDER = tuple(split['source_order'])
    import v10.finetune_totalseg_amos_magic_v10_2_joint as entry
    import v10_2_data as data
    ct13_training_entry.install_v10_protocol(entry,data,split)
    from finetune_multisource_sam2_v9_2_precision_mixed_semantic_eval import _legacy_text
    native_text = entry.conditioning_text_from_metadata
    entry.conditioning_text_from_metadata = lambda row: _legacy_text(row) if row['dataset'] in ('amos','totalseg') else native_text(row)
    native_path = ROOT/'v10/infer/infer_eight_source_adaptive_v10_2.py'
    session.program = hook.compile_native(native_path.read_bytes(),str(native_path),paths['native_source_sha256'])
    ns = session.program.namespace(); session.namespace = ns
    from tools.ct13_full_validation_eval import metric_module
    import v10.infer.infer_amos_sam2_promptgen_v10 as scorer
    hook._install_metric_provider(session,scorer,metric_module())
    run_dir = Path(job['run']); prepared = Path(job['prepared'])
    custom = SimpleNamespace(independent_task_manifest=str(prepared/'manifest.json'),nninteractive_metrics=str(prepared/'refs.json'),
        shared_adaptive_prompt_root=str(ROOT/paths['shared_adaptive_prompt_root']),weights=str(ROOT/config['checkpoint']),
        source=source,background_mode='point',eval_out_dir=str(run_dir.parent),run_name=run_dir.name,gpu=str(job['gpu']),
        sam2_frame_batch_size=session.batch,task_shard_count=1,task_shard_index=0,dynamic_threshold_fallback=.60,
        dynamic_threshold_min=config['floor'],dynamic_threshold_max=.75,dynamic_threshold_min_logit_separation=.50,
        prompt_logit_neighborhood_radius=2,base_split_json=str(split_path),runtime_version='v10.2')
    ns['_extract_args'] = lambda:custom
    installer = ns['_install_exact_prompts_and_reserved_roi']
    def install(dataset,**kwargs):
        result,inner=hook._install_dataset(session,installer,dataset,**kwargs)
        def readonly_existing(self,image):
            path=self._case_image_cache_path(image)
            volume=self._load_case_image_cache(path)
            if volume is None: raise ValueError('required existing case CT cache missing '+str(path))
            session.cache_events.append(dict(path=str(path),image=str(image),mode='read_only_hit',shape=list(volume.shape),dtype=str(volume.dtype)))
            return volume
        inner._get_or_create_case_image=MethodType(readonly_existing,inner)
        return result,inner
    ns['_install_exact_prompts_and_reserved_roi'] = install
    session.variant = 'baseline'
    exec(session.program.initialization,ns)
    session.model_loads += 1
    if not getattr(session,'dataset_bound',False) or {key(t)[:2] for t in ns['tasks']} != {(source,str(job['case_id']))}:
        raise ValueError('native single-case ownership mismatch')
    if config['epoch'] != 490:
        from v10_2_sam_finetuning import configure_sam_finetuning,load_sam_tuning_from_checkpoint
        from tools.v102_source_best_full import restore_encoder_delta
        tuning = restore_encoder_delta(ns['adapter'].sam,ns['args'].sam_tuning_mode,ns['state'],configure_sam_finetuning,load_sam_tuning_from_checkpoint)
        session.encoder_names = list(tuning.encoder_names)
        if not session.encoder_names: raise ValueError('empty encoder delta')
        for parameter in ns['adapter'].sam.parameters(): parameter.requires_grad_(False)
    hook.validate_settings(ns['args'],session.batch)
    if int(ns['args'].model_input_size) != 1024 or int(ns['inner'].model_input_size) != 1024: raise ValueError('native1024 required')
    session.expected_semantics = dict(entry._SEMANTIC_BY_ID)
    session.restore_proof()
    hook._install_execution_hooks(session)
    session.input_audit = dict(native_source_sha256=session.program.source_sha256,task_body_ast_sha256=session.program.task_body_sha256,
        semantics='native_expanded_metadata',cache_policy='readonly_CT_or_current_case_RAM',actual_frame_batch=session.batch,
        settings=hook.validate_settings(ns['args'],session.batch),model_input_size=1024,cache_image_size=int(ns['inner'].image_size),
        coordinate_transform='raw_magic_nrrd_to_preprocessed_nifti_swap_flip_xy_v1' if source == 'magic' else 'identity')
    session.evaluate = lambda task,variant:hook.evaluate(session,task,variant)
    return session
