"""Pure, independent bookkeeping for the paired Bone dynamic-floor evaluation.

This module reads no image data, imports no model code, and writes no files.
Global IDs serve bookkeeping only; local_class_id/label_value retain NIfTI IDs.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import re


FLOORS = (.35, .40, .45, .50, .55, .60, .65, .70, .75)
MODELS = ('v9_2', 'v10_2')
GID_OFFSET = 10000
METRICS = ('dice', 'mean_slice_dice', 'iou', 'precision', 'recall', 'specificity',
           'pred_gt_ratio', 'mean_slice_nsd_1mm', 'mean_slice_nsd_2mm', 'mean_slice_nsd_3mm')
NATIVE_FALLBACK_SOURCES = frozenset((
    'fallback_missing_prompt_logits', 'fallback_insufficient_separation',
))


def _integer(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f'invalid {field}: expected integer')
    try:
        return int(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f'invalid {field}: expected integer') from exc


def _text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'missing/empty {field}')
    return value


def key(row):
    """Return the common CT13 task identity, accepting metric-row aliases."""
    if not isinstance(row, Mapping):
        raise ValueError('invalid task row')
    source = _text(row.get('source_name', row.get('dataset')), 'task source')
    case = _text(row.get('case_id'), 'case_id')
    gid = _integer(row.get('global_class_id', row.get('class_id')), 'global_class_id')
    return source, case, gid


def _unique_index(rows, description):
    result = {}
    for row in rows:
        identity = key(row)
        if identity in result:
            raise ValueError(f'duplicate {description}: {identity}')
        result[identity] = row
    return result


def _class_index(classes):
    if not isinstance(classes, (list, tuple)) or not classes:
        raise ValueError('missing/empty class definitions')
    result = {}
    for row in classes:
        if not isinstance(row, Mapping):
            raise ValueError('invalid class definition')
        local = _integer(row.get('label_id'), 'label_id')
        if local < 0:
            raise ValueError('negative label_id')
        if local in result:
            raise ValueError(f'duplicate label_id definition: {local}')
        _text(row.get('name'), 'class name')
        for field in ('voxel_count', 'case_count'):
            if _integer(row.get(field), field) < 0:
                raise ValueError(f'negative class {field}')
        result[local] = row
    return result


def _present_class(classes, local):
    if local <= 0 or local not in classes:
        raise ValueError(f'missing foreground class definition: {local}')
    meta = classes[local]
    if int(meta['voxel_count']) <= 0 or int(meta['case_count']) <= 0:
        raise ValueError(f'empty foreground class: {local}')
    return meta


def _dataset_path(value, root, *, image=False):
    """Require safe root-relative text; permit only the audited 0009 CT link."""
    if not isinstance(value, str) or not value.strip() or '\x00' in value:
        raise ValueError('missing/invalid dataset path')
    windows = PureWindowsPath(value)
    relative = PurePosixPath(value.replace('\\', '/'))
    if (windows.drive or windows.root or relative.is_absolute()
            or '..' in relative.parts or not relative.parts):
        raise ValueError(f'dataset path must be root relative without traversal: {value}')
    resolved = (root / Path(*relative.parts)).resolve()
    if resolved.is_relative_to(root):
        return str(resolved)
    # Dataset preparation deliberately retained this one supplemental CT link.
    if (image and relative.parts == ('ct_raw', '0009.nii.gz')
            and resolved == root.parent / '0009.nii.gz'):
        return str(resolved)
    raise ValueError(f'dataset path escapes root: {value}')


def make_tasks(paired, dataset_root):
    """Build CT13-shaped tasks from ready paired cases and actual local labels.

    The generic helper intentionally has no fixed case/class/task-count checks;
    the runner separately verifies the signed full inventory before inference.
    """
    if not isinstance(paired, Mapping) or paired.get('test_ready') is not True:
        raise ValueError('paired manifest is not test ready')
    classes = _class_index(paired.get('classes'))
    cases = paired.get('cases')
    if not isinstance(cases, (list, tuple)) or not cases:
        raise ValueError('missing/empty paired cases')
    root = Path(dataset_root).resolve()
    seen_cases = set()
    rows = []
    for case in cases:
        if not isinstance(case, Mapping) or case.get('test_ready') is not True:
            raise ValueError('case is not test ready')
        case_id = _text(case.get('case_id'), 'case_id')
        if case_id in seen_cases:
            raise ValueError(f'duplicate case_id: {case_id}')
        seen_cases.add(case_id)
        present = case.get('present_foreground_ids')
        if not isinstance(present, (list, tuple)) or not present:
            raise ValueError(f'missing/empty foreground classes: {case_id}')
        ids = [_integer(value, 'foreground label_id') for value in present]
        if len(ids) != len(set(ids)):
            raise ValueError(f'duplicate foreground label_id: {case_id}')
        image = _dataset_path(case.get('image'), root, image=True)
        label = _dataset_path(case.get('label_nifti'), root)
        for local in ids:
            meta = _present_class(classes, local)
            rows.append(dict(source_name='bone', case_id=case_id,
                             global_class_id=GID_OFFSET + local, local_class_id=local,
                             class_name=meta['name'], image=image, label=label,
                             split_role='validation'))
    return sorted(rows, key=key)


def _semantic_text(name):
    body_only = '仅椎体' in name
    vertebra = re.fullmatch(r'vertebrae?_([CTLS])(\d+)(?:[（(].*?[）)])?', name)
    if vertebra:
        region, number = vertebra.groups()
        adjective = {'C': 'cervical', 'T': 'thoracic', 'L': 'lumbar', 'S': 'sacral'}[region]
        text = f'{adjective} vertebra {region}{number}'
        return f'vertebral body of {text}' if body_only else text
    rib = re.fullmatch(r'rib_(left|right)_(\d+)', name)
    if rib:
        return f'{rib[1]} rib {rib[2]}'
    sided = re.fullmatch(r'(.+)_(left|right)', name)
    if sided:
        anatomy, side = sided.groups()
        anatomy = {'phalanges_foot': 'toe phalanges', 'tarsal': 'tarsal bones',
                   'metatarsal': 'metatarsal bones', 'hip': 'hip bone',
                   'clavicula': 'clavicle'}.get(anatomy, anatomy.replace('_', ' '))
        return f'{side} {anatomy}'
    return name.replace('_', ' ')


def _bone_catalogue_row(local, meta):
    name = meta['name']
    semantic = 'bone.' + re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')
    if '仅椎体' in name:
        semantic += '_body_only'
    # These are metadata hints in the model's existing allowed vocabulary.
    if name.startswith(('rib_', 'femur_', 'tibia_', 'fibula_', 'humerus_', 'clavicula_')):
        morphology = 'tubular'
    elif name.startswith(('hip_', 'scapula_')):
        morphology = 'sheet'
    else:
        morphology = 'compact'
    return dict(dataset='bone', global_class_id=GID_OFFSET + local,
                local_class_id=local, label_value=local, class_name=name,
                semantic_text=_semantic_text(name), semantic_key=semantic,
                parent_semantic_key=None, equivalence_group=semantic,
                morphology_group=morphology)


def _validate_bone_tasks(rows):
    indexed = _unique_index(rows, 'Bone task')
    if not indexed:
        raise ValueError('empty Bone tasks')
    for (source, case, gid), row in indexed.items():
        local = _integer(row.get('local_class_id'), 'local_class_id')
        if source != 'bone' or local <= 0 or gid != GID_OFFSET + local:
            raise ValueError(f'inconsistent Bone task label IDs: {source}/{case}/{gid}')
    return indexed


def augment_split(base, tasks, classes):
    """Return an independent split with original sources intact plus Bone val."""
    indexed = _validate_bone_tasks(tasks)
    definitions = _class_index(classes)
    split = deepcopy(base)
    if ('bone' in split['source_order'] or 'bone' in split['datasets']
            or 'bone' in split.get('sampling_ratios', {})):
        raise ValueError('Bone source already present')
    existing_gids = {int(row['global_class_id']) for row in split['class_catalog']}
    selected = sorted({identity[2] - GID_OFFSET for identity in indexed})
    catalog = []
    for local in selected:
        if GID_OFFSET + local in existing_gids:
            raise ValueError(f'global class ID collision: {GID_OFFSET + local}')
        catalog.append(_bone_catalogue_row(local, _present_class(definitions, local)))

    cases = {}
    for identity, row in sorted(indexed.items()):
        _, case_id, gid = identity
        local = gid - GID_OFFSET
        if row.get('class_name') != definitions[local]['name']:
            raise ValueError(f'inconsistent Bone task class name: {identity}')
        image = _text(row.get('image'), 'image path')
        label = _text(row.get('label'), 'label path')
        entry = cases.setdefault(case_id, dict(dataset='bone', case_id=case_id,
                                                image=image, label=label, classes=[], modality='ct'))
        if image != entry['image'] or label != entry['label']:
            raise ValueError(f'inconsistent paths within Bone case: {case_id}')
        entry['classes'].append(gid)
    split['source_order'] = list(split['source_order']) + ['bone']
    split['class_catalog'].extend(catalog)
    split['datasets']['bone'] = dict(train=[], val=list(cases.values()))
    split['sampling_ratios'] = {source: float(ratio) * .99
                                for source, ratio in split['sampling_ratios'].items()}
    split['sampling_ratios']['bone'] = .01
    return split


def make_jobs(rows, shard_count=8):
    """Schedule all real smokes first, then disjoint balanced whole-case shards."""
    if isinstance(shard_count, bool) or not isinstance(shard_count, int) or shard_count < 1:
        raise ValueError('invalid shard_count')
    indexed = _validate_bone_tasks(rows)
    ordered = sorted(indexed.items())
    first_case = ordered[0][0][:2]
    first_rows = [(identity, row) for identity, row in ordered if identity[:2] == first_case]
    smoke_key = next((identity for identity, row in first_rows
                      if int(row['local_class_id']) == 3), first_rows[0][0])
    grouped = defaultdict(list)
    for identity, row in ordered:
        if identity != smoke_key:
            grouped[identity[:2]].append(row)
    shards = [[] for _ in range(shard_count)]
    loads = [0] * shard_count
    for case, values in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        position = min(range(shard_count), key=lambda index: (loads[index], index))
        shards[position].extend(values)
        loads[position] += len(values)

    smokes, full = [], []
    for model in MODELS:
        for floor in FLOORS:
            prefix = f'{model}_bone_floor{floor:.2f}'
            common = dict(model=model, variant='baseline', source='bone', floor=floor,
                          frame_batch=16, attempt=0)
            smokes.append(dict(common, name=prefix + '_smoke', smoke=True,
                               tasks=[deepcopy(indexed[smoke_key])], task_count=1))
            for number, values in enumerate(shards):
                if values:
                    tasks = deepcopy(sorted(values, key=key))
                    full.append(dict(common, name=prefix + f'_full_{number:02d}', smoke=False,
                                     tasks=tasks, task_count=len(tasks)))
    return smokes + full


def validate_rows(rows, expected):
    """Reject duplicates, missing/extra keys and any absent/nonfinite metric."""
    actual = _unique_index(rows, 'metric task')
    reference = _unique_index(expected, 'expected task')
    if set(actual) != set(reference):
        raise ValueError('metric task coverage mismatch')
    for identity, row in actual.items():
        for metric in METRICS:
            try:
                value = float(row[metric])
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f'missing/invalid metric {metric} in {identity}') from exc
            if not math.isfinite(value):
                raise ValueError(f'nonfinite metric {metric} in {identity}')


def threshold_valid(row, floor):
    """Audit the selected threshold; only the two native fallbacks bypass floor."""
    try:
        source = row.get('dynamic_threshold_source', row.get('source'))
        value = float(row['threshold'])
        floor = float(floor)
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        return False
    if not isinstance(source, str) or not source or not math.isfinite(value) or not math.isfinite(floor):
        return False
    if not 0 <= floor <= .75:
        return False
    if source.startswith('fallback'):
        return source in NATIVE_FALLBACK_SOURCES and value == .60
    return floor <= value <= .75


def _means(indexed):
    count = len(indexed)
    if not count:
        raise ValueError('empty metric rows')
    return {name: math.fsum(float(indexed[k][name]) for k in sorted(indexed)) / count
            for name in METRICS}


def summarize(rows, baseline, floor60):
    """Summarize task means and same-task deltas, preserving per-task rows."""
    validate_rows(rows, rows)
    validate_rows(baseline, rows)
    validate_rows(floor60, rows)
    current = _unique_index(rows, 'metric task')
    original = _unique_index(baseline, 'baseline task')
    sixty = _unique_index(floor60, 'floor60 task')
    mean = _means(current)
    ordered = sorted(current)
    count = len(ordered)

    def delta(reference):
        return {name: math.fsum(float(current[k][name]) - float(reference[k][name])
                                for k in ordered) / count for name in METRICS}

    by_class = defaultdict(list)
    for identity in ordered:
        by_class[identity[0], identity[2]].append(float(current[identity]['dice']))
    macro = math.fsum(math.fsum(values) / len(values) for values in by_class.values()) / len(by_class)
    return dict(task_count=count, mean=mean, per_case_class=[deepcopy(current[k]) for k in ordered],
                delta_from_baseline=delta(original), delta_from_floor60=delta(sixty),
                class_macro_dice=macro,
                fallback_count=sum(str(row.get('dynamic_threshold_source', '')).startswith('fallback')
                                   for row in current.values()))


def winner(parts):
    """Choose one global floor by full-volume task Dice, exact ties lower floor."""
    if not isinstance(parts, Mapping) or not parts:
        raise ValueError('empty floor candidates')
    reference = next(iter(parts.values()))
    candidates = []
    for floor, rows in parts.items():
        if isinstance(floor, bool):
            raise ValueError('invalid floor candidate')
        try:
            floor = float(floor)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError('invalid floor candidate') from exc
        if not math.isfinite(floor) or floor not in FLOORS:
            raise ValueError('invalid floor candidate')
        validate_rows(rows, reference)
        indexed = _unique_index(rows, 'metric task')
        candidates.append((floor, _means(indexed)['dice']))
    chosen, dice = min(candidates, key=lambda item: (-item[1], item[0]))
    return dict(chosen_floor=chosen, task_mean_dice=dice,
                rule='highest math.fsum task-mean full-volume dice; exact ties choose smaller floor')
