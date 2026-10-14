"""Fixed heldout native cases; reuse CT16 training evaluation unchanged."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import sys

SOURCES = ('graz', 'pengwin', 'adrenal')
ROOT = Path(__file__).resolve().parents[1]

def select_cases(split, count=4):
    if count < 1: raise ValueError('positive case count required')
    selected = {}
    for source in SOURCES:
        row = split['datasets'][source]
        cases = list(row['test'])
        if len(cases) < count or len(set(cases)) != len(cases):
            raise ValueError('requested number of unique heldout cases required')
        if set(cases) & set(row['train']):
            raise ValueError('training/test overlap')
        selected[source] = cases[:count]
    return selected

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def enable_adapter_ema(decoder):
    # Match the checkpoint EMA membership; evaluation remains torch.no_grad().
    for module in (decoder.prompt_3d_adapter, decoder.prompt_memory_adapter):
        for parameter in module.parameters(): parameter.requires_grad = True

def save(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--expected-gpu-uuid', required=True)
    parser.add_argument('--cases-per-source', type=int, default=4)
    cli = parser.parse_args()
    os.chdir(ROOT)
    for path in (ROOT, ROOT/'v10', ROOT/'tools', ROOT/'third_party/four_source_eval_deps'):
        sys.path.insert(0, str(path))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cli.gpu)
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
    import numpy as np
    import torch
    import finetune_totalseg_amos_magic_v10_2_joint as joint
    import v10_2_data as data
    from four_source_data import discover, load_pair
    from prepare_v10_2_sixteen_source import portable_record
    from v10_2_native_reference import NativeReferenceLoader
    from v9_2_extended_protocol import conditioning_text_from_metadata
    from v10_2_sam_finetuning import configure_sam_finetuning, load_sam_tuning_from_checkpoint
    from v10_2_ct13_decoder_control import freeze_control_parameters
    from v10_training_ema import TrainableParameterEMA
    external_path = Path('configs/graz_pengwin_adrenal_test20_train_rest_v1_20261012.json')
    external = json.loads(external_path.read_text())
    chosen = select_cases(external, cli.cases_per_source)
    out = Path(cli.output).resolve()
    if ROOT not in out.parents: raise ValueError('output must be inside project')
    out.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    base = json.loads(Path('configs/v10_2_sixteen_source_train_20261012.json').read_text())
    state = torch.load(cli.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    args = argparse.Namespace(**copy.deepcopy(state['args']))
    catalog = [r for r in base['class_catalog'] if r['dataset'] in SOURCES]
    refs = ROOT/'v10/.cache/totalseg_sam2_promptgen_v10_prompt_roi_full_v1/metadata/native_eval_first4_v1'
    found = discover((ROOT/'../data').resolve())
    split = {'source_order': list(SOURCES), 'class_catalog': catalog, 'datasets': {}}
    tasks = []; absent = []
    for source in SOURCES:
        index = {r['case_id']: r for r in found[source]}; entries = []
        for case in chosen[source]:
            record = portable_record(index[case]); image, label, audit = load_pair(record)
            if not audit['grid_verified']: raise ValueError('native grid not verified')
            present = set(int(x) for x in audit['label_values'])
            ids = [r['global_class_id'] for r in catalog if r['dataset']==source and r['label_value'] in present]
            for r in catalog:
                if r['dataset']==source and r['label_value'] not in present: absent.append([source,case,r['global_class_id']])
            directory = refs/source/case; directory.mkdir(parents=True, exist_ok=True)
            ref = dict(record, native_training_reference_v1=True)
            for role in ('image.nii.gz','label.nii.gz'):
                path = directory/role
                if path.exists():
                    if json.loads(path.read_text()) != ref: raise ValueError('native reference changed')
                else: save(path, ref)
            row = dict(dataset=source, case_id=case, image=str((directory/'image.nii.gz').relative_to(ROOT)),
                       label=str((directory/'label.nii.gz').relative_to(ROOT)), classes=ids, modality='ct')
            entries.append(row)
            tasks.extend(dict(row, global_class_id=k) for k in ids)
            del image, label
            print('AUDIT',source,case,ids,flush=True)
        split['datasets'][source] = {'train': [], 'val': entries}
    observed = {t['global_class_id'] for t in tasks}
    split['class_catalog'] = [r for r in catalog if r['global_class_id'] in observed]
    validation = {'tasks': tasks, 'selection_weight_policy':'nonempty_class_count'}
    args.multidataset_sampling_ratios = {s:1/3 for s in SOURCES}
    args.validation_na_class_ids = (); args.multidataset_validation_json = str(out/'tasks.json')
    args.label_names = {r['global_class_id']:r['class_name'] for r in catalog}
    args.v10_2_source_order = SOURCES; args.v10_2_class_catalog = catalog
    args.validation_max_tasks = 0; args.gpu = str(cli.gpu)
    joint._PROTOCOL = (split, validation)
    joint._SEMANTIC_BY_ID = {r['global_class_id']:conditioning_text_from_metadata(r) for r in catalog}
    joint.EIGHT_SOURCE_ORDER = data.EIGHT_SOURCE_ORDER = SOURCES
    NativeReferenceLoader(refs).install(); joint._bind_v10_2(args)
    device = torch.device('cuda:0'); torch.cuda.set_device(device)
    uuid = str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')
    if uuid != cli.expected_gpu_uuid.removeprefix('GPU-') or torch.ones(1,device=device).item()!=1: raise ValueError('CUDA/UUID mismatch')
    adapter = joint.v10.SAM2MedicalAdapterV10(args, device).to(device)
    tuning = configure_sam_finetuning(adapter.sam, args.sam_tuning_mode)
    freeze_control_parameters(adapter.sam, 'frozen')
    enable_adapter_ema(adapter.sam.sam_mask_decoder)
    prompt = joint.make_prompt_v10_2(args, adapter, device); prompt._v10_2_args = args
    loaded = joint.load_init_v10_2(cli.checkpoint,prompt,adapter.sam.sam_mask_decoder)
    load_sam_tuning_from_checkpoint(adapter.sam,tuning,loaded,required=True)
    ema = TrainableParameterEMA({'prompt':prompt,'sam':adapter.sam},args.ema_decay)
    ema.load_state_dict(state['ema_state'])
    manifest = dict(checkpoint=cli.checkpoint, checkpoint_sha256=sha(cli.checkpoint), epoch=state['epoch'],
        selected_cases=chosen, cases_per_source=cli.cases_per_source, task_count=len(tasks), absent_case_classes=absent,
        external_split_sha256=sha(external_path), weight_mode='EMA', thresholds=list(args.validation_thresholds),
        foreground=args.val_foreground_mode, background=args.val_background_mode, gpu_uuid=uuid,
        scope='one fixed E40 evaluation; no split mutation or checkpoint selection')
    save(out/'manifest.json',manifest); save(out/'tasks.json',validation)
    del state, loaded
    seed=int(args.validation_seed); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    prompt.eval(); adapter.eval()
    joint.v10.CrossViewPromptTokenGeneratorV10.DEFAULT_EVAL_RANDOM_PROMPT_MODES=False
    joint.v10.CrossViewPromptTokenGeneratorV10.DEFAULT_EVAL_BACKGROUND_MODE=args.val_background_mode
    with torch.no_grad(), ema.average_parameters():
        score, classes, metrics = joint.evaluate_multidataset_v10_2(prompt,adapter,[],[],args,device)
    if metrics['validation_protocol']['selected_tasks'] != len(tasks): raise ValueError('incomplete evaluation')
    save(out/'metrics.json',metrics)
    print('COMPLETE',score,metrics['per_dataset'],flush=True)

if __name__=='__main__': main()
