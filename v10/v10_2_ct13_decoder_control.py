"""Matched E490 CT13 controls: Encoder frozen, Decoder core frozen or trainable."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys

from v10_2_ct13_ema_diagnostic import (
    EXPECTED_EMA_SCORE, SPLIT_SHA256, VALIDATION_SHA256, assert_checkpoint,
    assert_protocol, evaluation_args, save_json_once, sha256_file,
)

INITIALIZATION = 'v10/output/v10_2_ct13_e470_zero_delta_e510_20261001/20261001_215853/epoch490.pth'


def assert_diagnostic_gate(comparison):
    values = [comparison.get(key, float('nan')) for key in
              ('live_weighted_dice', 'ema_weighted_dice', 'ema_reproduction_delta')]
    if (not all(math.isfinite(float(value)) for value in values)
            or abs(float(values[1]) - EXPECTED_EMA_SCORE) > .002
            or abs(float(values[2])) > .002):
        raise ValueError('Complete, finite, reproduced EMA diagnostic is required')


def control_args(source, *, initialization, output, gpu, smoke=False):
    values = copy.deepcopy(source)
    values.update(epochs=1 if smoke else 10, lr_schedule_epochs=40,
                  train_cases_per_epoch=26 if smoke else 200, validate_every=5,
                  validate_before_train=not smoke,
                  prompt_generator_checkpoint=str(initialization), resume_checkpoint='',
                  resume_new_run=False, out_dir=str(output), gpu=str(gpu),
                  sam_encoder_unfreeze_epoch=11, amp=True, object_score_gate=False)
    return argparse.Namespace(**values)


def freeze_control_parameters(sam, arm):
    if arm not in ('frozen', 'full'):
        raise ValueError('Decoder arm must be frozen or full')
    for name, parameter in sam.named_parameters():
        if name.startswith('image_encoder.') or '.image_encoder.' in name:
            parameter.requires_grad_(False)
    for name, parameter in sam.sam_mask_decoder.named_parameters():
        if not name.startswith(('prompt_3d_adapter.', 'prompt_memory_adapter.')):
            parameter.requires_grad_(arm == 'full')


def after_initialization(native_load, capture):
    def wrapped(sam, tuning, state, **kwargs):
        result = native_load(sam, tuning, state, **kwargs)
        capture(sam, tuning)
        return result
    return wrapped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configuration-checkpoint', required=True)
    parser.add_argument('--diagnostic-results', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--arm', choices=('frozen', 'full'), required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--expected-gpu-uuid', required=True)
    parser.add_argument('--smoke', action='store_true')
    cli = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)
    output = Path(cli.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    for directory in (root, root / 'v10'):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cli.gpu)
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    import torch
    from ct13_v10_2_full_entry import install_protocol
    import finetune_multisource_sam2_v10_2_encoder_decoder as full
    from utils.distributed_runtime import find_free_port

    diagnostic = Path(cli.diagnostic_results).resolve()
    assert_diagnostic_gate(json.loads((diagnostic / 'comparison.json').read_text()))
    for arm in ('live', 'ema'):
        assert_protocol(json.loads((diagnostic / f'{arm}.json').read_text()))
    manifest = json.loads((diagnostic / 'manifest.json').read_text())
    if manifest['split_sha256'] != SPLIT_SHA256 or manifest['validation_sha256'] != VALIDATION_SHA256:
        raise ValueError('Diagnostic protocol signatures changed')
    configuration = Path(cli.configuration_checkpoint).resolve()
    state = torch.load(configuration, map_location='cpu', weights_only=False, mmap=True)
    assert_checkpoint(configuration, state)
    evaluation_args(state['args'], gpu=cli.gpu)  # Reject truncated/altered formal validation.
    args = control_args(state['args'], initialization=root / INITIALIZATION,
                        output=output, gpu=cli.gpu, smoke=cli.smoke)
    if (args.prompt_lr != 2.5e-7 or args.adapter_lr != 5e-7 or args.memory_adapter_lr != 5e-7
            or args.decoder_lr != 2e-7 or args.ema_decay != .9995):
        raise ValueError('Requires the unchanged low-PG learning rates and EMA')
    for path, expected in ((args.multidataset_split_json, SPLIT_SHA256),
                           (args.multidataset_validation_json, VALIDATION_SHA256)):
        if sha256_file(path) != expected:
            raise ValueError(f'Protocol signature mismatch: {Path(path).name}')
    install_protocol(args.multidataset_split_json)
    full.v102.v101._validate_extension_args(args)
    full._validate(args)
    full.v102._prepare_protocol(args)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    uuid = str(torch.cuda.get_device_properties(device).uuid).removeprefix('GPU-')
    if uuid != cli.expected_gpu_uuid.removeprefix('GPU-'):
        raise ValueError('CUDA ordinal/UUID mismatch')
    if torch.ones(1, device=device).item() != 1:
        raise RuntimeError('Real CUDA tensor probe failed')
    torch.cuda.synchronize()
    save_json_once(output / 'control_manifest.json', {
        'arm': cli.arm, 'encoder_frozen': True, 'smoke': cli.smoke,
        'initialization': INITIALIZATION, 'configuration_checkpoint': str(configuration.relative_to(root)),
        'initialization_sha256': sha256_file(root / INITIALIZATION),
        'configuration_sha256': manifest['checkpoint_sha256'],
        'physical_gpu': cli.gpu, 'uuid': uuid, 'epochs': args.epochs,
        'global_cases_per_epoch': args.train_cases_per_epoch,
        'lr_schedule_epochs': args.lr_schedule_epochs, 'seed': args.seed,
        'split_sha256': SPLIT_SHA256, 'validation_sha256': VALIDATION_SHA256,
        'prompt_lr': args.prompt_lr, 'adapter_lr': args.adapter_lr,
        'memory_adapter_lr': args.memory_adapter_lr, 'decoder_lr': args.decoder_lr,
        'ema_decay': args.ema_decay, 'optimizer': 'fresh AdamW', 'ema': 'fresh',
    })
    del state
    runtime = full.v102.v10
    native_configure = runtime.configure_sam_finetuning
    native_load = runtime.load_sam_tuning_from_checkpoint
    original_evaluate = full.v102.evaluate_multidataset_v10_2
    snapshots = {}

    def configure(sam, mode):
        tuning = native_configure(sam, mode)
        freeze_control_parameters(sam, cli.arm)
        if any(p.requires_grad for n, p in sam.named_parameters() if n.startswith('image_encoder.')):
            raise RuntimeError('Encoder unexpectedly trainable')
        core = {n: p for n, p in sam.sam_mask_decoder.named_parameters()
                if not n.startswith(('prompt_3d_adapter.', 'prompt_memory_adapter.'))}
        if not core or any(p.requires_grad != (cli.arm == 'full') for p in core.values()):
            raise RuntimeError('Decoder core freeze policy failed')
        print(f'CONTROL arm={cli.arm} encoder_trainable=0 decoder_core_trainable={sum(p.requires_grad for p in core.values())}', flush=True)
        return tuning

    def capture(sam, tuning):
        adapters = {n: p for n, p in sam.sam_mask_decoder.named_parameters()
                    if n.startswith(('prompt_3d_adapter.', 'prompt_memory_adapter.'))}
        if not adapters or not all(p.requires_grad for p in adapters.values()):
            raise RuntimeError('Prompt adapters unexpectedly frozen')
        if cli.smoke:
            snapshots['sam'] = sam
            snapshots['encoder'] = {n: p.detach().cpu().clone() for n, p in sam.named_parameters()
                                    if n in tuning.encoder_names}
            snapshots['core'] = {n: p.detach().cpu().clone() for n, p in sam.sam_mask_decoder.named_parameters()
                                 if n not in adapters}
            snapshots['adapters'] = {n: p.detach().cpu().clone() for n, p in adapters.items()}

    def evaluate(*values, **keywords):
        result = original_evaluate(*values, **keywords)
        assert_protocol(result[2])
        if not math.isfinite(float(result[0])):
            raise ValueError('Nonfinite formal validation')
        if not snapshots.get('baseline_checked'):
            if abs(float(result[0]) - .7874224293948384) > .002:
                raise ValueError('Frozen E490 baseline reproduction failed before training')
            snapshots['baseline_checked'] = True
        return result

    runtime.configure_sam_finetuning = configure
    runtime.load_sam_tuning_from_checkpoint = after_initialization(native_load, capture)
    full.v102.evaluate_multidataset_v10_2 = evaluate
    try:
        full._worker(0, 1, find_free_port(), datetime.now().strftime('%Y%m%d_%H%M%S'), args)
        if cli.smoke:
            sam = snapshots['sam']
            named = dict(sam.named_parameters())
            encoder_same = all(torch.equal(named[n].detach().cpu(), value)
                               for n, value in snapshots['encoder'].items())
            decoder = dict(sam.sam_mask_decoder.named_parameters())
            changed = sum(not torch.equal(decoder[n].detach().cpu(), value)
                          for n, value in snapshots['core'].items())
            adapter_changed = sum(not torch.equal(decoder[n].detach().cpu(), value)
                                  for n, value in snapshots['adapters'].items())
            if not encoder_same or ((changed > 0) != (cli.arm == 'full')) or not adapter_changed:
                raise RuntimeError(f'Actual update policy failed: encoder_same={encoder_same} decoder_changed={changed}')
            save_json_once(output / 'smoke_update_check.json',
                           {'encoder_unchanged': encoder_same, 'decoder_changed_tensors': changed,
                            'adapter_changed_tensors': adapter_changed, 'arm': cli.arm, 'status': 'passed'})
            print('SMOKE UPDATE POLICY PASSED', encoder_same, changed, flush=True)
    finally:
        runtime.configure_sam_finetuning = native_configure
        runtime.load_sam_tuning_from_checkpoint = native_load
        full.v102.evaluate_multidataset_v10_2 = original_evaluate


if __name__ == '__main__':
    main()
