"""CT16 matched continuation: staged Encoder LoRA and full SAM2 Decoder core."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT/'v10', ROOT/'tools', ROOT/'third_party/four_source_eval_deps'):
    sys.path.insert(0, str(directory))
from v10_2_sixteen_source_protocol import SOURCE_ORDER, source_counts, validate_external_split
from v10_2_sixteen_source_validation import (
    load_validation_view, assert_validation_metrics, VALIDATION_SPLIT, VALIDATION_TASKS)

INITIALIZATION = 'v10/output/v10_2_ct13_e490_decoder_frozen_control_10_20261005/20261005_155255/epoch010.pth'
TRAIN_SPLIT = 'configs/v10_2_sixteen_source_train_20261012.json'
OLD_SPLIT = 'configs/ct13_v10_2_split_20260928.json'
OLD_VALIDATION = 'configs/ct13_v10_2_validation_four_new_sources4_classweight_v3_20260928.json'


def assert_best_state(state):
    score = float(state['validation_metrics']['selection']['score'])
    if (int(state.get('epoch', -1)) != 10 or not math.isfinite(score)
            or abs(score-.7881604375711181) > 1e-9):
        raise ValueError('requires finite audited best A E10')
    return score


def training_args(source, *, initialization, output, gpu, split, smoke=False):
    values = copy.deepcopy(source)
    values.update(epochs=1 if smoke else 40, lr_schedule_epochs=40,
        train_cases_per_epoch=16 if smoke else 250, validate_every=10,
        validate_before_train=not smoke, prompt_generator_checkpoint=str(initialization),
        resume_checkpoint='', resume_new_run=False, out_dir=str(output), gpu=str(gpu),
        multidataset_split_json=str(split), sam_encoder_unfreeze_epoch=1 if smoke else 6,
        sam_tuning_mode='encoder_lora_full_decoder', sam_encoder_lora_rank=8,
        sam_encoder_lora_alpha=16, encoder_lr=5e-7, decoder_lr=2e-7,
        decoder_transformer_lr=None, decoder_head_lr=None,
        amp=True, object_score_gate=False, v10_2_poscap_skip_validation=bool(smoke))
    if smoke: values['lr_warmup_epochs'] = 1
    return argparse.Namespace(**values)


def assert_full_decoder_lora_policy(sam):
    """Validate native SAM policy before the worker enables the adapters."""
    encoder = list(sam.image_encoder.named_parameters())
    lora = [(name, p) for name, p in encoder if 'qkv' in name and 'lora_' in name]
    if not lora or not all(p.requires_grad for _, p in lora):
        raise RuntimeError('Encoder LoRA parameters must be trainable')
    if any(p.requires_grad for name, p in encoder
           if not ('qkv' in name and 'lora_' in name)):
        raise RuntimeError('Encoder base parameters must remain frozen')
    core = [(name, p) for name, p in sam.sam_mask_decoder.named_parameters()
            if not name.startswith(('prompt_3d_adapter.', 'prompt_memory_adapter.'))]
    if not core or not all(p.requires_grad for _, p in core):
        raise RuntimeError('Decoder core parameters must all be trainable')
    for name, p in sam.named_parameters():
        parts=name.split('.')
        if p.requires_grad and not ('image_encoder' in parts or 'sam_mask_decoder' in parts):
            raise RuntimeError('Unexpected trainable SAM parameter: '+name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initialization', default=INITIALIZATION)
    parser.add_argument('--split', default=TRAIN_SPLIT)
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--expected-gpu-uuid', required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--legacy-retention-validation', action='store_true',
                        help='Explicitly retain historical CT13-only validation instead of CT16 val4')
    cli = parser.parse_args(); os.chdir(ROOT)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cli.gpu)
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    import torch
    import finetune_multisource_sam2_v10_2_encoder_decoder as full
    import v10_2_data as data
    import v9_2_extended_protocol as protocol
    from v9_1_multidataset_runtime import load_and_validate_protocol
    from v9_2_extended_protocol import validate_class_catalog_metadata, conditioning_text_from_metadata
    from v10_2_ct13_ema_diagnostic import assert_protocol, SPLIT_SHA256, VALIDATION_SHA256, sha256_file, save_json_once
    from v10_2_native_reference import NativeReferenceLoader
    from utils.distributed_runtime import find_free_port

    if sha256_file(OLD_SPLIT) != SPLIT_SHA256 or sha256_file(OLD_VALIDATION) != VALIDATION_SHA256:
        raise ValueError('OriginalCT13 validation signature changed')
    retention, validation = load_and_validate_protocol(OLD_SPLIT, OLD_VALIDATION)
    split = json.loads(Path(cli.split).read_text())
    if tuple(split['source_order']) != SOURCE_ORDER or split['base_split_sha256'] != SPLIT_SHA256:
        raise ValueError('CT16 manifest source order/base signature mismatch')
    external = json.loads(Path(split['external_split']).read_text())
    validate_external_split(external)
    if sha256_file(split['external_split']) != split['external_split_sha256']:
        raise ValueError('external partition changed')
    validate_class_catalog_metadata(split['class_catalog'])
    for source in SOURCE_ORDER[:13]:
        if split['datasets'][source] != retention['datasets'][source]:
            raise ValueError('original source partitions changed: '+source)
    for source in SOURCE_ORDER[13:]:
        rows = split['datasets'][source]
        if {r['case_id'] for r in rows['train']} != set(external['datasets'][source]['train']) or rows['val']:
            raise ValueError('external training partition mismatch')
    eval_split, eval_validation = ((retention, validation) if cli.legacy_retention_validation
                                  else load_validation_view(cli.split))
    validation_path = OLD_VALIDATION if cli.legacy_retention_validation else VALIDATION_TASKS
    validation_scope = ('legacy CT13 retention only' if cli.legacy_retention_validation
                        else eval_split['validation_scope'])
    initialization = Path(cli.initialization).resolve()
    if sha256_file(initialization) != sha256_file(ROOT/INITIALIZATION):
        raise ValueError('initialization differs from audited best A E10 weights')
    state = torch.load(initialization, map_location='cpu', mmap=True, weights_only=False)
    assert_protocol(state['validation_metrics'])
    score = assert_best_state(state)
    output = Path(cli.output).resolve()
    if ROOT not in output.parents: raise ValueError('output must stay under project')
    args = training_args(state['args'], initialization=initialization, output=output,
                         gpu=cli.gpu, split=cli.split, smoke=cli.smoke)
    args.multidataset_validation_json = validation_path
    for key, expected in [('prompt_lr',2.5e-7),('adapter_lr',5e-7),('memory_adapter_lr',5e-7),('ema_decay',.9995)]:
        if getattr(args,key) != expected: raise ValueError('unexpected inherited '+key)
    device = torch.device('cuda:0'); torch.cuda.set_device(device)
    actual = str(torch.cuda.get_device_properties(device).uuid).removeprefix('GPU-')
    if actual != cli.expected_gpu_uuid.removeprefix('GPU-') or torch.ones(1,device=device).item() != 1:
        raise RuntimeError('real CUDA tensor/UUID check failed')
    torch.cuda.synchronize()
    output.mkdir(parents=True, exist_ok=False)
    save_json_once(output/'training_manifest.json', dict(initialization=str(initialization.relative_to(ROOT)),
        initialization_sha256=sha256_file(initialization), initialization_retention_dice=score,
        split=cli.split, split_sha256=sha256_file(cli.split), external_split_sha256=split['external_split_sha256'],
        validation=validation_path, validation_sha256=sha256_file(validation_path),
        validation_scope=validation_scope, test_used_for_selection=False, fumpe_training=False,
        selected_test_cases_promoted_to_validation=not cli.legacy_retention_validation,
        validation_split=OLD_SPLIT if cli.legacy_retention_validation else VALIDATION_SPLIT,
        source_counts=source_counts(), epochs=args.epochs, cases_per_epoch=args.train_cases_per_epoch,
        physical_gpu=cli.gpu, gpu_uuid=actual, encoder_base_frozen=True,
        encoder_lora_trainable=True, encoder_unfreeze_epoch=args.sam_encoder_unfreeze_epoch,
        encoder_lora_rank=8, encoder_lora_alpha=16, decoder_core_frozen=False,
        optimizer='fresh AdamW', ema='fresh', smoke=cli.smoke))
    del state
    NativeReferenceLoader(ROOT/split['native_reference_root']).install()
    joint = full.v102
    protocol.SOURCE_ORDER = SOURCE_ORDER
    joint.EIGHT_SOURCE_ORDER = data.EIGHT_SOURCE_ORDER = SOURCE_ORDER
    native_prepare = joint._prepare_protocol
    native_evaluate = joint.evaluate_multidataset_v10_2
    native_configure = joint.v10.configure_sam_finetuning
    baseline_checked = False

    def prepare(bound_args):
        joint._PROTOCOL = (split, validation)
        joint._SEMANTIC_BY_ID = {int(r['global_class_id']):conditioning_text_from_metadata(r) for r in split['class_catalog']}
        bound_args.v10_2_source_order = SOURCE_ORDER
        bound_args.v10_2_class_catalog = copy.deepcopy(split['class_catalog'])
        ratios = ({s:1/16 for s in SOURCE_ORDER} if cli.smoke else
                  {s:n/250 for s,n in source_counts().items()})
        bound_args.multidataset_sampling_ratios = ratios
        bound_args.label_names = {int(r['global_class_id']):r['class_name'] for r in split['class_catalog']}
        bound_args.validation_na_class_ids = tuple(
            int(r['global_class_id']) for r in eval_split['class_catalog'] if r.get('not_applicable'))
        bound_args.num_classes = len(bound_args.label_names)
        bound_args.val_num_classes = len(eval_split['class_catalog'])

    def evaluate(*values, **keywords):
        nonlocal baseline_checked
        old = joint._PROTOCOL
        # Independent validation view; the training partitions/catalog remain unchanged.
        joint._PROTOCOL = (eval_split, eval_validation)
        try: result = native_evaluate(*values, **keywords)
        finally: joint._PROTOCOL = old
        if cli.legacy_retention_validation:
            assert_protocol(result[2])
        else:
            assert_validation_metrics(result[2], eval_validation)
        if not math.isfinite(float(result[0])): raise ValueError('nonfinite retention metric')
        if not baseline_checked:
            if cli.legacy_retention_validation and abs(float(result[0])-score) > .002:
                raise ValueError('best A E10 baseline reproduction failed')
            baseline_checked = True
        result[2]['training_source_count'] = 16
        result[2]['validation_scope'] = validation_scope
        result[2]['new_source_validation_evaluated'] = not cli.legacy_retention_validation
        result[2]['new_source_test_evaluated'] = False  # remaining heldout16/source never used
        return result

    def configure(sam, mode):
        tuning = native_configure(sam, mode)
        assert_full_decoder_lora_policy(sam)
        print('CT16 encoder_base_frozen=True encoder_lora_trainable=True '
              'decoder_core_frozen=False adapters_trainable=True',flush=True)
        return tuning

    joint._prepare_protocol = prepare
    joint.evaluate_multidataset_v10_2 = evaluate
    joint.v10.configure_sam_finetuning = configure
    full.v102.v101._validate_extension_args(args); full._validate(args)
    try:
        full._worker(0,1,find_free_port(),datetime.now().strftime('%Y%m%d_%H%M%S'),args)
        print('CT16 SMOKE COMPLETE' if cli.smoke else 'CT16 TRAINING COMPLETE',flush=True)
    finally:
        joint._prepare_protocol = native_prepare
        joint.evaluate_multidataset_v10_2 = native_evaluate
        joint.v10.configure_sam_finetuning = native_configure


if __name__ == '__main__': main()
