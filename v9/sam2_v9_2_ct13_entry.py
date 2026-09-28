"""Additive CT13 SAM2 v9.2 precision/DDP entry; no v10 model imports."""
from __future__ import annotations
import json
import logging
import os
from pathlib import Path
import subprocess
import sys

from ct13_training_entry import _ratios
from ct_thirteen_source_protocol import SOURCE_ORDER
from ct13_validation_policy import class_count_weights


def configured_ratios(split):
    return _ratios(split)


def evaluate_with_class_weights(evaluator, weights, prompt, adapter, pairs, classes, args, device):
    training_ratios = args.multidataset_sampling_ratios
    try:
        args.multidataset_sampling_ratios = dict(weights)
        return evaluator(prompt, adapter, pairs, classes, args, device)
    finally:
        args.multidataset_sampling_ratios = training_ratios


def launch(extra):
    root = Path(__file__).resolve().parent
    template = (root / 'run_train_sam2_v9_2_precision_eight_source.sh').read_text()
    old = 'exec "${PYTHON}" -u "${PROJECT_ROOT}/v9_family_shared_cache_entry.py" \\\n  finetune_multisource_sam2_v9_2_precision \\'
    new = 'exec "${PYTHON}" -u -m torch.distributed.run --standalone --nproc_per_node=2 "${PROJECT_ROOT}/sam2_v9_2_ct13_entry.py" \\'
    replacements = {
        'PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"': 'PROJECT_ROOT='+json.dumps(str(root)),
        old: new,
        'configs/v9_2_extended_split_4case_v2.json': 'configs/ct13_v10_2_split_20260928.json',
        'configs/v9_2_extended_validation_4case_v2.json': 'configs/ct13_v10_2_validation_topcow2_classweight_v2_20260928.json',
    }
    for before, after in replacements.items():
        if template.count(before) != 1:
            raise ValueError('unexpected launcher template: '+before)
        template = template.replace(before, after, 1)
    if '--dry-run' in extra:
        print(template)
        return
    result = subprocess.run(['bash','-s','--',*extra],input=template,text=True,cwd=root)
    raise SystemExit(result.returncode)


def main():
    if sys.argv[1:2] == ['--launch']:
        return launch(sys.argv[2:])
    root = Path(__file__).resolve().parent
    def path_arg(option, default):
        return Path(sys.argv[sys.argv.index(option)+1]) if option in sys.argv else root/default
    split = json.loads(path_arg('--multidataset-split-json','configs/ct13_v10_2_split_20260928.json').read_text())
    validation = json.loads(path_arg('--multidataset-validation-json','configs/ct13_v10_2_validation_topcow2_classweight_v2_20260928.json').read_text())
    ratios = configured_ratios(split)
    if validation.get('selection_weight_policy') != 'nonempty_class_count':
        raise ValueError('CT13 requires the fixed class-count validation v2')
    weights = class_count_weights(split)
    import finetune_multisource_sam2_v9_2_ddp as ddp
    import v9_2_extended_protocol as protocol
    legacy = ddp.legacy
    legacy.SOURCE_ORDER = protocol.SOURCE_ORDER = ddp.multi.SOURCE_ORDER = SOURCE_ORDER
    legacy.validate_source_ratios = lambda supplied: dict(supplied) if set(supplied) == set(SOURCE_ORDER) else dict(ratios)
    from v9_family_shared_cache_entry import install_multidataset_factories
    install_multidataset_factories(ddp.multi)
    original = ddp.base.evaluate
    def evaluate(prompt, adapter, pairs, classes, args, device):
        result = evaluate_with_class_weights(original,weights,prompt,adapter,pairs,classes,args,device)
        score, per_class, metrics = result
        metrics['selection']['name'] = 'ct13_nonempty_class_macro_dice'
        args.v9_2_selection_metric = metrics['selection']['name']
        logging.info('CT13 validation | tasks=%d covered_classes=%d selection=%s',len(validation['tasks']),len(per_class),metrics['selection'])
        # Legacy evaluator writes the new-run JSON. Correct its descriptive name,
        # without changing any metric, score or historical eight-source file.
        epoch = int(legacy.multi92.v92.CURRENT_SAVE_EPOCH)
        for handler in logging.getLogger().handlers:
            if isinstance(handler,logging.FileHandler):
                record = Path(handler.baseFilename).parent/f'validation_epoch{epoch:03d}.json'
                if record.is_file():
                    payload=json.loads(record.read_text())
                    payload['metrics']['selection']=metrics['selection']
                    record.write_text(json.dumps(payload,indent=2),encoding='utf-8')
                break
        return score,per_class,metrics
    ddp.base.evaluate = evaluate
    ddp.main()

if __name__ == '__main__':
    main()
