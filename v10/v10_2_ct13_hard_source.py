"""Sampling-only hard-source continuation of the audited frozen-SAM2 CT13 control."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

INITIALIZATION = 'v10/output/v10_2_ct13_e490_decoder_frozen_control_10_20261005/20261005_155255/epoch010.pth'
COUNTS = dict(totalseg=30, amos=14, magic=28, msd_task07=8, msd_task10=6,
              msd_task08=24, parse2022=6, topcow2024_cta=20, lndb=18,
              msd_task06=6, covid19_20=20, kits23=12, lnq2023_lite=8)


def source_ratios():
    if sum(COUNTS.values()) != 200 or len(COUNTS) != 13 or min(COUNTS.values()) < 1:
        raise ValueError('Requires thirteen positive quotas and global 200 budget')
    return {name: count / 200 for name, count in COUNTS.items()}


def focused_args(source, *, initialization, output, gpu, smoke=False):
    values = copy.deepcopy(source)
    values.update(epochs=1 if smoke else 20, lr_schedule_epochs=40,
                  train_cases_per_epoch=26 if smoke else 200, validate_every=5,
                  validate_before_train=not smoke, prompt_generator_checkpoint=str(initialization),
                  resume_checkpoint='', resume_new_run=False, out_dir=str(output), gpu=str(gpu),
                  sam_encoder_unfreeze_epoch=21, amp=True, object_score_gate=False,
                  v10_2_poscap_skip_validation=bool(smoke))
    if smoke:
        values['lr_warmup_epochs'] = 1
    return argparse.Namespace(**values)


def main():
    root = Path(__file__).resolve().parent.parent
    for directory in (root, root / 'v10'):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    import v10_2_ct13_decoder_control as control
    if '--help' not in sys.argv and '-h' not in sys.argv:
        if '--arm' not in sys.argv or sys.argv[sys.argv.index('--arm') + 1] != 'frozen':
            raise ValueError('Hard-source control must freeze both original SAM2 encoder and decoder')
    import ct13_v10_2_full_entry as entry
    native_install = entry.install_protocol
    native_save = control.save_json_once

    def install(*args, **kwargs):
        result = native_install(*args, **kwargs)
        import finetune_totalseg_amos_magic_v10_2_joint as joint
        if tuple(result['source_order']) != tuple(COUNTS):
            raise ValueError('Audited source order changed')
        joint.extended_source_ratios = lambda _args: source_ratios()
        return result

    def save(path, payload):
        if Path(path).name == 'control_manifest.json':
            payload = dict(payload, global_source_counts=COUNTS,
                           sampling_ratios=source_ratios(),
                           experiment_variable='source quotas only; LR/loss/model/prompt unchanged')
        return native_save(path, payload)

    control.INITIALIZATION = INITIALIZATION
    control.control_args = focused_args
    control.save_json_once = save
    entry.install_protocol = install
    control.main()


if __name__ == '__main__':
    main()
