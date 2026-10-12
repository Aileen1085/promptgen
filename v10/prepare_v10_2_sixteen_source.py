"""Audit373 native training cases and write lightweight CT16 references only."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT/'v10', ROOT/'tools', ROOT/'third_party/four_source_eval_deps'):
    sys.path.insert(0, str(directory))
from v10_2_sixteen_source_protocol import SOURCE_ORDER, TRAIN_SOURCES, source_counts, validate_external_split


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_once(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != value:
            raise ValueError('different frozen metadata exists: '+str(path))
        return
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='../data')
    parser.add_argument('--external-split', default='configs/graz_pengwin_adrenal_test20_train_rest_v1_20261012.json')
    parser.add_argument('--base-split', default='configs/ct13_v10_2_split_20260928.json')
    parser.add_argument('--out', default='configs/v10_2_sixteen_source_train_20261012.json')
    parser.add_argument('--reference-root', default='v10/.cache/totalseg_sam2_promptgen_v10_prompt_roi_full_v1/metadata/native_training_v1')
    args = parser.parse_args(); os.chdir(ROOT)
    import numpy as np
    from four_source_data import discover, load_pair
    from four_source_eval import CLASSES
    external = json.loads(Path(args.external_split).read_text()); validate_external_split(external)
    split = json.loads(Path(args.base_split).read_text())
    if tuple(split['source_order']) != SOURCE_ORDER[:13]:
        raise ValueError('originalCT13 source order changed')
    references = Path(args.reference_root).resolve()
    if ROOT not in references.parents:
        raise ValueError('references must stay under project')
    found = discover(Path(args.data_root).resolve())
    catalog = copy.deepcopy(split['class_catalog'])
    for source in TRAIN_SOURCES:
        classes = [dict(r, not_applicable=False, conditioning='semantic_text_not_id_embedding') for r in CLASSES[source]]
        catalog.extend(classes)
        index = {r['case_id']:r for r in found[source]}
        source_split = external['datasets'][source]
        if not set(source_split['train']+source_split['test']).issubset(index):
            raise ValueError('external inventory missing case')
        rows = []
        for number, case in enumerate(source_split['train'], 1):
            record = index[case]
            directory = references/source/case
            audit_path = directory/'geometry.json'
            if audit_path.is_file():
                audit = json.loads(audit_path.read_text())
                if audit['source_signatures'] != record['source_signatures']:
                    raise ValueError('native source signature changed')
                present = set(audit['label_values'])
            else:
                image, label, audit = load_pair(record)
                present = set(int(v) for v in np.unique(np.asanyarray(label.dataobj)))
                if not audit['grid_verified']:
                    raise ValueError('unverified native geometry')
                write_once(audit_path, audit)
                del image, label
            unknown = present - {0} - {r['label_value'] for r in classes}
            if unknown:
                raise ValueError('unexpected native labels: '+str(unknown))
            ids = [r['global_class_id'] for r in classes if r['label_value'] in present]
            if not ids:
                raise ValueError('empty training target: '+source+'/'+case)
            reference = dict(record, native_training_reference_v1=True)
            image_path, label_path = directory/'image.nii.gz', directory/'label.nii.gz'
            write_once(image_path, reference); write_once(label_path, reference)
            rows.append(dict(dataset=source, case_id=case, image=str(image_path.relative_to(ROOT)),
                label=str(label_path.relative_to(ROOT)), classes=ids, modality='ct'))
            print(f'AUDIT {source} {number}/{len(source_split["train"])} {case} classes={ids}', flush=True)
        split['datasets'][source] = dict(train=rows, val=[], test_case_ids=source_split['test'])
    split.update(source_order=list(SOURCE_ORDER), class_catalog=catalog,
        protocol_revision='v10_2_sixteen_source_train_v1_20261012',
        sampling_quotas_250=source_counts(), sampling_ratios={s:n/250 for s,n in source_counts().items()},
        external_split=args.external_split, external_split_sha256=sha(args.external_split),
        base_split=args.base_split, base_split_sha256=sha(args.base_split),
        native_reference_root=str(references.relative_to(ROOT)),
        validation_scope='unchangedCT13 retention only; new sources have no validation; tests never select best')
    split.pop('sha256', None)
    write_once(args.out, split)
    print('COMPLETE373 native train /60 heldout test; FUMPE excluded', flush=True)


if __name__ == '__main__': main()
