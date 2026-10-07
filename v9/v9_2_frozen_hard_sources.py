"""E360 continuation: source-quota experiment with enforced frozen SAM2."""
from __future__ import annotations
import hashlib
import json
import logging
from pathlib import Path
import sys

QUOTAS = dict(totalseg=24, amos=12, magic=24, msd_task07=6,
              msd_task10=6, msd_task08=30, parse2022=20,
              topcow2024_cta=10, lndb=20, msd_task06=6,
              covid19_20=24, kits23=8, lnq2023_lite=10)
HARD_SOURCES = ('msd_task08', 'parse2022', 'lndb', 'covid19_20', 'lnq2023_lite')


def focus_ratios():
    assert sum(QUOTAS.values()) == 200
    return {name: count / 200. for name, count in QUOTAS.items()}


def assert_frozen_sam2(adapter, groups=()):
    native = [(n, p) for n, p in adapter.named_parameters()
              if '.prompt_3d_adapter.' not in n]
    trainable = [n for n, p in native if p.requires_grad]
    if trainable:
        raise RuntimeError('Frozen-SAM2 violation: ' + ', '.join(trainable[:8]))
    ids = {id(p) for _, p in native}
    if any(id(p) in ids for group in groups for p in group['params']):
        raise RuntimeError('Frozen SAM2 native parameter was included in optimizer')


def core_fingerprint(decoder):
    import torch
    digest = hashlib.sha256()
    core = decoder.core_decoder
    for name, tensor in sorted(core.state_dict().items()):
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main():
    import sam2_v9_2_ct13_entry as parent
    if sys.argv[1:2] == ['--launch']:
        import subprocess
        original_run = subprocess.run
        def run(*args, **kwargs):
            if 'input' in kwargs:
                kwargs['input'] = kwargs['input'].replace(
                    '/sam2_v9_2_ct13_entry.py', '/v9_2_frozen_hard_sources.py')
            return original_run(*args, **kwargs)
        subprocess.run = run
        return parent.launch(sys.argv[2:])
    parent.configured_ratios = lambda _split: focus_ratios()
    import finetune_multisource_sam2_v9_2_ddp as ddp
    base = ddp.base
    build = base.build_optimizer_groups
    fingerprint = [None]
    optimizer_groups = []
    def build_checked(prompt, decoder, args):
        if not args.freeze_decoder:
            raise RuntimeError('This experiment requires --freeze-decoder')
        groups = build(prompt, decoder, args)
        if any(p.requires_grad for p in decoder.core_decoder.parameters()):
            raise RuntimeError('SAM2 decoder core is trainable')
        native_ids = {id(p) for p in decoder.core_decoder.parameters()}
        if any(id(p) in native_ids for group in groups for p in group['params']):
            raise RuntimeError('SAM2 decoder core is in optimizer')
        optimizer_groups[:] = groups
        args.frozen_hard_source_quotas = dict(QUOTAS)
        logging.info('FROZEN_SAM2 optimizer_groups=%s quotas=%s',
                     [g.get('name') for g in groups], QUOTAS)
        return groups
    base.build_optimizer_groups = build_checked
    forward = base.JointTrainingForward
    class CheckedForward(forward):
        def __init__(self, prompt, adapter):
            assert_frozen_sam2(adapter, optimizer_groups)
            # Init checkpoint loading happens after optimizer construction.
            fingerprint[0] = core_fingerprint(adapter.sam.sam_mask_decoder)
            logging.info('FROZEN_SAM2 post-warmstart core_sha256=%s', fingerprint[0])
            logging.info('FROZEN_SAM2 all native parameters requires_grad=False; only added prompt_3d_adapter allowed')
            super().__init__(prompt, adapter)
    base.JointTrainingForward = CheckedForward
    save = base.save_ckpt
    def save_checked(path, prompt, adapter, *args, **kwargs):
        assert_frozen_sam2(adapter, optimizer_groups)
        actual = core_fingerprint(adapter.sam.sam_mask_decoder)
        if actual != fingerprint[0]:
            raise RuntimeError('Frozen SAM2 decoder core changed during training')
        logging.info('FROZEN_SAM2 checkpoint verified unchanged core %s', actual)
        return save(path, prompt, adapter, *args, **kwargs)
    base.save_ckpt = save_checked
    evaluate = base.evaluate
    def evaluate_checked(*args, **kwargs):
        result = evaluate(*args, **kwargs)
        metrics = result[2]
        per_source = metrics['per_dataset']
        hard = sum(per_source[s]['dice'] for s in HARD_SOURCES) / len(HARD_SOURCES)
        logging.info('HARD_SOURCE_MONITOR equal_source_dice=%.6f sources=%s', hard,
                     {s: per_source[s] for s in HARD_SOURCES})
        metrics['hard_source_monitor'] = {'dice': hard, 'sources': list(HARD_SOURCES)}
        return result
    base.evaluate = evaluate_checked
    parent.main()


if __name__ == '__main__':
    main()
