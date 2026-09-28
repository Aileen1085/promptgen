"""Save an exact-argv resume launcher; does not stop or start any process."""
import argparse
import json
from pathlib import Path
import shlex
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    argv = Path(f'/proc/{args.pid}/cmdline').read_bytes().decode().split('\0')[:-1]
    if not any(Path(v).name == 'ct13_training_entry.py' for v in argv):
        raise ValueError('not the requested CT13 process')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if 'optimizer' not in checkpoint or 'ema_state' not in checkpoint:
        raise ValueError('not a full optimizer/EMA checkpoint')
    validation = json.loads(args.validation.read_text())
    if validation.get('selection_weight_policy') != 'nonempty_class_count':
        raise ValueError('wrong validation protocol')
    argv += ['--resume-checkpoint', str(args.checkpoint.resolve()),
             '--resume-lr-scale', '1.0', '--resume-new-run',
             '--multidataset-validation-json', str(args.validation.resolve())]
    text = '#!/bin/bash\nset -euo pipefail\ncd /home/bedicloud/sharestore2/al/v9\n'
    text += 'exec env CT13_FAMILY=v10_2 ' + shlex.join(argv) + '\n'
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(text)
    print(json.dumps({'resume_epoch': checkpoint['epoch'], 'next_epoch': int(checkpoint['epoch'])+1,
                     'optimizer_lrs': [g['lr'] for g in checkpoint['optimizer']['param_groups']],
                     'launcher': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
