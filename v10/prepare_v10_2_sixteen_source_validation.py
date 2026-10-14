"""Publish CT16 val4 metadata, reusing existing CT and native reference paths."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT/'v10', ROOT/'tools', ROOT/'third_party/four_source_eval_deps'):
    sys.path.insert(0, str(p))
from v10_2_sixteen_source_protocol import SOURCE_ORDER, validate_external_split
from v10_2_sixteen_source_validation import build_validation_view, sha256_file, VALIDATION_SPLIT, VALIDATION_TASKS
from prepare_v10_2_sixteen_source import portable_record, write_once, audit_record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-split',default='configs/v10_2_sixteen_source_train_20261012.json')
    p.add_argument('--data-root',default='../data')
    p.add_argument('--output-split',default=VALIDATION_SPLIT)
    p.add_argument('--output-tasks',default=VALIDATION_TASKS)
    args=p.parse_args(); os.chdir(ROOT)
    from four_source_data import discover
    split=json.loads(Path(args.training_split).read_text())
    external=json.loads(Path(split['external_split']).read_text()); validate_external_split(external)
    if sha256_file(split['external_split']) != split['external_split_sha256']:
        raise ValueError('external split changed')
    heldout={s:copy.deepcopy(split['datasets'][s]['val']) for s in SOURCE_ORDER[:13]}
    found=discover(Path(args.data_root).resolve())
    references=ROOT/split['native_reference_root']
    for source in SOURCE_ORDER[13:]:
        inventory={r['case_id']:r for r in found[source]}
        heldout[source]=[]
        classes=[r for r in split['class_catalog'] if r['dataset']==source]
        for index, case in enumerate(external['datasets'][source]['test']):
            record=portable_record(inventory[case])
            directory=references/source/case
            # Only first4 require geometry/label audits; the remainder is never evaluated.
            if index<4:
                audit_path=directory/'geometry.json'
                audit=json.loads(audit_path.read_text()) if audit_path.exists() else audit_record(record)
                if not audit['grid_verified'] or audit['source_signatures'] != record['source_signatures']:
                    raise ValueError('native geometry/signature mismatch')
                present=set(audit['label_values'])
                if present-{0}-{r['label_value'] for r in classes}:
                    raise ValueError('unknown native label')
                ids=[r['global_class_id'] for r in classes if r['label_value'] in present]
                write_once(audit_path,audit)
                reference=dict(record,native_training_reference_v1=True)
                write_once(directory/'image.nii.gz',reference)
                write_once(directory/'label.nii.gz',reference)
            else:
                ids=[]
            heldout[source].append(dict(dataset=source,case_id=case,
                image=str((directory/'image.nii.gz').relative_to(ROOT)),
                label=str((directory/'label.nii.gz').relative_to(ROOT)),classes=ids,modality='ct'))
            if index<4: print('VAL_AUDIT',source,case,ids,flush=True)
    view, validation=build_validation_view(split,heldout)
    # Portable copies of existing source paths; no original split is modified.
    for dataset in view['datasets'].values():
        # Only IDs are needed for leakage checking in this evaluation-only view.
        dataset['train']=[dict(case_id=r['case_id']) for r in dataset['train']]
        for row in dataset['val']+dataset.get('test',[])+dataset.get('remaining_val',[]):
            for key in ('image','label'):
                row[key]=Path(os.path.relpath(Path(row[key]).resolve(),ROOT)).as_posix()
    for task in validation['tasks']:
        for key in ('image','label'):
            task[key]=Path(os.path.relpath(Path(task[key]).resolve(),ROOT)).as_posix()
    write_once(args.output_split,view)
    validation.update(split_file_sha256=sha256_file(args.output_split),
        training_split_sha256=sha256_file(args.training_split),
        external_split_sha256=split['external_split_sha256'])
    write_once(args.output_tasks,validation)
    print(json.dumps(dict(cases=validation['case_count'],tasks=validation['expected_tasks'],
        covered=len(validation['covered_class_ids']),na=validation['na_class_ids'],
        case_ids=validation['case_ids']),indent=2),flush=True)


if __name__=='__main__': main()
