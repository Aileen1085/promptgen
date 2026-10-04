"""Read-only CT13 comparison of live parameters and EMA from the SAME last.pth."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
import sys
import time

from v10_2_expert_ablation import save_json_once, sha256_file

SPLIT_SHA256 = '99b2eee8b4ae549537922614b0570665c6fa90387e21e04c69c04ed0652de022'
VALIDATION_SHA256 = 'd02bde8c7de9b9a20672df2cdc0ca39775a591eef33b89457cdac291c144dc39'


def assert_protocol(metrics):
    protocol = metrics.get('validation_protocol') or {}
    expected = dict(selected_tasks=267, covered_classes=163, requested_classes=163,
                    all_nonempty_classes_covered=True, metric_space='raw_nifti_full_volume',
                    anchor_coarse_crop=False)
    errors = {key: (protocol.get(key), value) for key, value in expected.items()
              if protocol.get(key) != value}
    if sorted(protocol.get('na_classes') or []) != [40, 41]:
        errors['na_classes'] = protocol.get('na_classes')
    if metrics.get('primary_threshold') != .60:
        errors['primary_threshold'] = metrics.get('primary_threshold')
    if errors:
        raise ValueError(f'CT13 diagnostic protocol mismatch: {errors}')


def evaluation_args(source, *, gpu):
    source = copy.deepcopy(source)
    expected = dict(validation_max_tasks=0, val_foreground_mode='scribble',
                    val_background_mode='scribble', mask_threshold=.60,
                    anchor_coarse_crop=False, validation_thresholds=[.55, .60, .65])
    errors = {key: (source.get(key), value) for key, value in expected.items()
              if source.get(key) != value}
    if errors:
        raise ValueError(f'Checkpoint validation args mismatch: {errors}')
    source.update(gpu=str(gpu), dataset_foreground_mode='scribble', amp=True,
                  object_score_gate=False)
    return argparse.Namespace(**source)


def compare(live, ema, *, expected_ema):
    for metrics in (live, ema):
        assert_protocol(metrics)
        if not math.isfinite(float(metrics['selection']['score'])):
            raise ValueError('Nonfinite weighted Dice')
    ema_score = float(ema['selection']['score'])
    error = ema_score - float(expected_ema)
    if abs(error) > .002:
        raise ValueError(f'EMA E10 reproduction failed: delta={error:.9f}')
    return {
        'live_weighted_dice': float(live['selection']['score']),
        'ema_weighted_dice': ema_score,
        'live_minus_ema_weighted_dice': float(live['selection']['score']) - ema_score,
        'ema_reproduction_delta': error,
        'per_source': {
            source: {'live': live['per_dataset'][source], 'ema': ema['per_dataset'][source]}
            for source in ema.get('per_dataset', {})
        },
        'interpretation': 'Same checkpoint; EMA/live comparison is not a frozen-decoder causal control.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--expected-gpu-uuid', required=True)
    cli = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    output = Path(cli.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(cli.checkpoint).resolve()
    for path in (root, root / 'v10'):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cli.gpu)
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(output / 'diagnostic.log')])

    import numpy as np
    import torch
    from ct13_v10_2_full_entry import install_protocol
    import finetune_totalseg_amos_magic_v10_2_joint as joint
    from v10_2_sam_finetuning import configure_sam_finetuning, load_sam_tuning_from_checkpoint
    from v10_training_ema import TrainableParameterEMA

    state = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    if int(state['epoch']) != 10 or not state.get('ema_state'):
        raise ValueError('Requires completed low-PG E10 last.pth with live model AND EMA state')
    args = evaluation_args(state['args'], gpu=cli.gpu)
    for path, expected in ((args.multidataset_split_json, SPLIT_SHA256),
                           (args.multidataset_validation_json, VALIDATION_SHA256)):
        if sha256_file(path) != expected:
            raise ValueError(f'Protocol signature changed: {Path(path).name}')
    install_protocol(args.multidataset_split_json)
    joint.v101._validate_extension_args(args)
    joint._prepare_protocol(args)
    joint._bind_v10_2(args)
    if len(joint._PROTOCOL[1]['tasks']) != 267 or len(args.v10_2_source_order) != 13:
        raise ValueError('Requires exactly 13 sources and 267 fixed tasks')
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    uuid = str(torch.cuda.get_device_properties(device).uuid)
    expected_uuid = cli.expected_gpu_uuid.removeprefix('GPU-')
    if uuid.removeprefix('GPU-') != expected_uuid:
        raise ValueError(f'GPU mapping mismatch: requested={cli.expected_gpu_uuid}, actual={uuid}')
    probe = torch.ones(1, device=device)
    torch.cuda.synchronize()
    if probe.item() != 1:
        raise RuntimeError('CUDA tensor probe failed')
    logging.info('GPU physical=%d internal=cuda:0 uuid=%s actual_tensor=OK', cli.gpu, uuid)

    adapter = joint.v10.SAM2MedicalAdapterV10(args, device).to(device)
    tuning = configure_sam_finetuning(adapter.sam, args.sam_tuning_mode)
    for module in (adapter.sam.sam_mask_decoder.prompt_3d_adapter,
                   adapter.sam.sam_mask_decoder.prompt_memory_adapter):
        for parameter in module.parameters():
            parameter.requires_grad = True
    prompt = joint.make_prompt_v10_2(args, adapter, device)
    prompt._v10_2_args = args
    loaded = joint.load_init_v10_2(checkpoint, prompt, adapter.sam.sam_mask_decoder)
    load_sam_tuning_from_checkpoint(adapter.sam, tuning, loaded, required=True)
    ema = TrainableParameterEMA({'prompt': prompt, 'sam': adapter.sam}, args.ema_decay)
    ema.load_state_dict(state['ema_state'])
    expected_ema = float(state['validation_metrics']['selection']['score'])
    del loaded, state
    adapter.eval()
    prompt.eval()
    joint.v10.CrossViewPromptTokenGeneratorV10.DEFAULT_EVAL_RANDOM_PROMPT_MODES = False
    joint.v10.CrossViewPromptTokenGeneratorV10.DEFAULT_EVAL_BACKGROUND_MODE = args.val_background_mode
    save_json_once(output / 'manifest.json', {
        'checkpoint': str(checkpoint.relative_to(root)), 'checkpoint_sha256': sha256_file(checkpoint),
        'epoch': 10, 'ema_updates': ema.num_updates, 'ema_decay': ema.decay,
        'physical_gpu': cli.gpu, 'uuid': uuid, 'validation_seed': args.validation_seed,
        'split_sha256': SPLIT_SHA256, 'validation_sha256': VALIDATION_SHA256,
        'expected_ema_weighted_dice': expected_ema,
        'cache_policy': 'Existing case CT and lightweight metadata only; no model/CT copies.',
    })
    results = {}
    for name in ('live', 'ema'):
        seed = int(args.validation_seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        start = time.monotonic()
        logging.info('BEGIN arm=%s epoch=10 ema_updates=%d', name, ema.num_updates)
        with torch.no_grad(), (ema.average_parameters() if name == 'ema' else nullcontext()):
            metrics = joint.evaluate_multidataset_v10_2(prompt, adapter, [], [], args, device)[2]
        assert_protocol(metrics)
        save_json_once(output / f'{name}.json', metrics)
        results[name] = metrics
        logging.info('COMPLETE arm=%s weighted_dice=%.9f seconds=%.1f',
                     name, metrics['selection']['score'], time.monotonic() - start)
        torch.cuda.empty_cache()
    result = compare(results['live'], results['ema'], expected_ema=expected_ema)
    save_json_once(output / 'comparison.json', result)
    logging.info('DIAGNOSTIC COMPLETE live=%.9f ema=%.9f delta=%+.9f reproduction=%+.9f',
                 result['live_weighted_dice'], result['ema_weighted_dice'],
                 result['live_minus_ema_weighted_dice'], result['ema_reproduction_delta'])


if __name__ == '__main__':
    main()
