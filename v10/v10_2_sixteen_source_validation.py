"""Versioned CT16 first-four-heldout validation; no voxel/cache writes."""
from __future__ import annotations
import copy
import hashlib
import json
import math
from pathlib import Path
from v10_2_sixteen_source_protocol import SOURCE_ORDER

VALIDATION_SPLIT = 'configs/v10_2_sixteen_source_val4_split_20261014.json'
VALIDATION_TASKS = 'configs/v10_2_sixteen_source_val4_tasks_20261014.json'


def build_validation_view(split, heldout):
    if tuple(split['source_order']) != SOURCE_ORDER or set(heldout) != set(SOURCE_ORDER):
        raise ValueError('exactly sixteen ordered sources required')
    view = copy.deepcopy(split)
    catalog = {int(r['global_class_id']): r for r in view['class_catalog']}
    tasks, cases, origins = [], {}, {}
    for index, source in enumerate(SOURCE_ORDER):
        rows = copy.deepcopy(heldout[source])
        ids = [str(r['case_id']) for r in rows]
        training = {str(r['case_id']) for r in split['datasets'][source]['train']}
        if len(rows) < 4 or len(ids) != len(set(ids)) or training & set(ids):
            raise ValueError('short/duplicate/overlapping heldout source: '+source)
        selected = rows[:4]
        cases[source] = ids[:4]
        origins[source] = 'existing_val' if index < 13 else 'promoted_from_test'
        view['datasets'][source]['val'] = selected
        view['datasets'][source]['remaining_val' if index < 13 else 'test'] = rows[4:]
        if index >= 13:
            view['datasets'][source]['test_case_ids'] = ids[4:]
        for row in selected:
            classes = sorted(set(int(c) for c in row['classes']))
            if not classes:
                raise ValueError('selected case has no nonempty targets: '+source+'/'+row['case_id'])
            for cid in classes:
                if cid not in catalog or catalog[cid]['dataset'] != source:
                    raise ValueError('unknown/cross-source validation class')
                tasks.append(dict(dataset=source, case_id=str(row['case_id']),
                    global_class_id=cid, class_name=catalog[cid]['class_name'],
                    image=row['image'], label=row['label']))
    covered = {t['global_class_id'] for t in tasks}
    for cid, row in catalog.items():
        row['not_applicable'] = cid not in covered
        row['not_applicable_reason'] = None if cid in covered else 'no_nonempty_GT_in_fixed_four_cases'
    view['validation_scope'] = 'CT16 fixed first4 per source; selected new test cases promoted to validation'
    view.pop('sha256', None)
    validation = dict(version=1, protocol='v10_2_sixteen_source_first4_heldout_v1_20261014',
        source_order=list(SOURCE_ORDER), tasks=tasks, case_ids=cases, case_origins=origins,
        case_count=64, expected_tasks=len(tasks), covered_class_ids=sorted(covered),
        na_class_ids=sorted(set(catalog)-covered), selection_weight_policy='nonempty_class_count',
        primary_threshold=.60, thresholds=[.55,.60,.65], metric_space='raw_nifti_full_volume',
        foreground='adaptive_scribble', background='adaptive_scribble', anchor_coarse_crop=False,
        test_used_for_selection=False, selected_test_cases_promoted_to_validation=True,
        fumpe_included=False)
    return view, validation


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_validation_view(training_split, split_path=VALIDATION_SPLIT, tasks_path=VALIDATION_TASKS):
    from v9_1_multidataset_runtime import load_and_validate_protocol
    view, validation = load_and_validate_protocol(split_path, tasks_path)
    if validation.get('split_file_sha256') != sha256_file(split_path):
        raise ValueError('validation split signature mismatch')
    if validation.get('training_split_sha256') != sha256_file(training_split):
        raise ValueError('training split signature mismatch')
    train = json.loads(Path(training_split).read_text())
    for source in SOURCE_ORDER:
        if ([r['case_id'] for r in view['datasets'][source]['train']] !=
                [r['case_id'] for r in train['datasets'][source]['train']]):
            raise ValueError('validation view changed training cases')
    rebuilt_view, rebuilt = build_validation_view(train, {
        s:view['datasets'][s]['val']+view['datasets'][s][
            'remaining_val' if s in SOURCE_ORDER[:13] else 'test'] for s in SOURCE_ORDER})
    for key in ('tasks','case_ids','case_origins','expected_tasks','covered_class_ids','na_class_ids'):
        if validation[key] != rebuilt[key]:
            raise ValueError('invalid fixed validation '+key)
    for source in SOURCE_ORDER[:13]:
        rows=view['datasets'][source]['val']+view['datasets'][source]['remaining_val']
        original=train['datasets'][source]['val']
        if [r['case_id'] for r in rows] != [r['case_id'] for r in original]:
            raise ValueError('original val order changed')
        for row, fixed in zip(rows,original):
            if (row['classes'] != fixed['classes'] or any(
                    Path(row[key]).resolve() != Path(fixed[key]).resolve() for key in ('image','label'))):
                raise ValueError('original val source/label/classes changed')
    external = json.loads(Path(train['external_split']).read_text())
    for source in SOURCE_ORDER[13:]:
        ids=external['datasets'][source]['test']
        rows=view['datasets'][source]['val']+view['datasets'][source]['test']
        if ([r['case_id'] for r in rows] != ids or
                view['datasets'][source]['test_case_ids'] != ids[4:]):
            raise ValueError('external test order changed')
    return view, validation


def assert_validation_metrics(metrics, validation):
    p = metrics['validation_protocol']
    observed = {int(c) for c in metrics['per_class']['dice']}
    if (p['selected_tasks'] != validation['expected_tasks']
        or observed != set(validation['covered_class_ids'])
        or p['covered_classes'] != len(observed)
        or p['metric_space'] != 'raw_nifti_full_volume' or p['anchor_coarse_crop']
        or abs(float(metrics['primary_threshold'])-.60) > 1e-9
        or set(metrics['per_dataset']) != set(SOURCE_ORDER)
        or metrics['case_class_macro']['count'] != validation['expected_tasks']
        or not all(math.isfinite(float(v)) for v in metrics['per_class']['dice'].values())):
        raise ValueError('CT16 val4 task/coverage/protocol/finite mismatch')
    if 'selection' in metrics:
        metrics['selection']['name'] = 'ct16_val4_nonempty_class_count_weighted_dice'
