"""Pure, audited Bone semantics using the original training text builder.

No images, models, files, or input rows are changed. ``semantic_text`` in an
aligned row is the final text consumed by the native Bone inference hooks;
``semantic_base_text`` is the input to the training conditioning builder.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import re

from v9_2_extended_protocol import (
    MORPHOLOGY_GROUPS,
    REQUIRED_CLASS_METADATA,
    class_metadata,
    conditioning_text_from_metadata,
)


IDENTITY_FIELDS = ('dataset', 'global_class_id', 'local_class_id',
                   'label_value', 'class_name')
BODY_ONLY_QUALIFIER = ' (vertebral body only)'
CONDITIONING_BUILDER = 'v9_2_extended_protocol.conditioning_text_from_metadata'
METADATA_BUILDER = 'v9_2_extended_protocol.class_metadata'
_BODY_SUFFIXES = ('（仅椎体）', '(仅椎体)')


def _text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'missing/invalid {field}')
    return value


def _positive_id(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f'invalid {field}: expected positive integer')
    if isinstance(value, str) and not re.fullmatch(r'[0-9]+', value):
        raise ValueError(f'invalid {field}: expected positive integer')
    number = int(value)
    if number <= 0:
        raise ValueError(f'invalid {field}: expected positive integer')
    return number


def _rows(rows, description, *, allow_empty=False):
    if (not isinstance(rows, Sequence) or isinstance(rows, (str, bytes))
            or (not rows and not allow_empty)):
        raise ValueError(f'missing/invalid {description} catalog')
    if any(not isinstance(row, Mapping) for row in rows):
        raise ValueError(f'invalid {description} catalog row')
    return deepcopy(list(rows))


def _check_identities(rows, description, *, bone=False):
    globals_seen, locals_seen = set(), set()
    for row in rows:
        dataset = _text(row.get('dataset'), f'{description} dataset')
        gid = _positive_id(row.get('global_class_id'), f'{description} global_class_id')
        local = _positive_id(row.get('local_class_id'), f'{description} local_class_id')
        name = _text(row.get('class_name'), f'{description} class_name')
        if name != name.strip():
            raise ValueError(f'invalid {description} class_name whitespace')
        # Other sources may use one binary mask per class (MAGIC local ID 1).
        unique_local = bone or dataset == 'totalseg'
        if gid in globals_seen or (unique_local and (dataset, local) in locals_seen):
            raise ValueError(f'duplicate {description} identity: {dataset}/{gid}/{local}')
        globals_seen.add(gid)
        locals_seen.add((dataset, local))
        if bone:
            label = _positive_id(row.get('label_value'), 'Bone label_value')
            if dataset != 'bone' or label != local:
                raise ValueError('inconsistent Bone dataset/local_class_id/label_value identity')
            _text(row.get('semantic_text'), 'original Bone semantic_text')


def _scope(class_name):
    suffix = next((value for value in _BODY_SUFFIXES if class_name.endswith(value)), None)
    base = class_name[:-len(suffix)] if suffix else class_name
    if (not base or base != base.strip()
            or any(char in base for char in '()（）[]【】')
            or '仅椎体' in base):
        raise ValueError(f'unknown/malformed annotation scope suffix: {class_name!r}')
    if suffix and re.fullmatch(r'vertebrae?_[CTLS][0-9]+', base) is None:
        raise ValueError(f'vertebral body scope suffix on non-vertebral class: {class_name!r}')
    return base, suffix


def _is_semantic_metadata(field):
    return (field in REQUIRED_CLASS_METADATA or field == 'conditioning'
            or field.startswith(('semantic_', 'parent_semantic_')))


def _metadata(row):
    # Eligibility and label identity are never inherited from the training row.
    return {key: deepcopy(value) for key, value in row.items()
            if _is_semantic_metadata(key)
            and key not in ('semantic_text', 'semantic_base_text')}


def _validate_metadata(row):
    for field in REQUIRED_CLASS_METADATA:
        if field not in row:
            raise ValueError(f'missing semantic metadata: {field}')
        if field == 'parent_semantic_key' and row[field] is None:
            continue
        _text(row[field], f'semantic metadata {field}')
    if row['morphology_group'] not in MORPHOLOGY_GROUPS:
        raise ValueError(f'unsupported morphology_group: {row["morphology_group"]!r}')


def align_catalog(bone_rows, training_catalog):
    """Return independent ``(aligned_rows, audit_rows)`` in Bone input order.

    Only literal class names from training ``dataset='totalseg'`` may match.
    The two explicit Chinese body-only suffixes are the only matching exception.
    Other anatomy uses its raw token and the native training metadata function.
    Audit rows retain the original row, full matched training row, exact metadata,
    old/base/final texts, matching mode and annotation scope for recomputation.
    """
    bones = _rows(bone_rows, 'Bone')
    training = _rows(training_catalog, 'training', allow_empty=True)
    _check_identities(bones, 'Bone', bone=True)
    _check_identities(training, 'training')
    by_name = {}
    for source in training:
        if source['dataset'] != 'totalseg':
            continue
        name = source['class_name']
        if name in by_name:
            raise ValueError(f'ambiguous TotalSeg training class_name: {name!r}')
        _text(source.get('semantic_text'), 'training semantic_text')
        _validate_metadata(source)
        by_name[name] = source

    aligned, audits = [], []
    for original in bones:
        name, suffix = _scope(original['class_name'])
        source = by_name.get(name)
        metadata = (_metadata(source) if source is not None
                    else class_metadata(original['dataset'], name))
        _validate_metadata(metadata)
        base_text = source['semantic_text'] if source is not None else name
        if suffix:
            base_text += BODY_ONLY_QUALIFIER
        row = {key: deepcopy(value) for key, value in original.items()
               if not _is_semantic_metadata(key)}
        row.update(deepcopy(metadata))
        row['semantic_base_text'] = base_text
        row['semantic_text'] = base_text
        # Call the real training builder exactly once on the unconditioned base.
        row['semantic_text'] = conditioning_text_from_metadata(row)
        if suffix and BODY_ONLY_QUALIFIER not in row['semantic_text']:
            raise ValueError('lost vertebral body-only annotation scope')
        audit = {field: deepcopy(original[field]) for field in IDENTITY_FIELDS}
        audit.update(
            schema_version=1,
            original_semantic_text=original['semantic_text'],
            semantic_base_text=base_text,
            semantic_text=row['semantic_text'],
            match_mode='exact_totalseg' if source is not None else 'new_anatomy',
            matching_class_name=name,
            annotation_scope='vertebral_body_only' if suffix else 'whole_anatomy',
            scope_suffix=suffix,
            training_row=deepcopy(source),
            original_row=deepcopy(original),
            semantic_metadata=deepcopy(metadata),
            conditioning_builder=CONDITIONING_BUILDER,
            metadata_builder=None if source is not None else METADATA_BUILDER,
        )
        aligned.append(row)
        audits.append(audit)
    return aligned, audits


def _same_tree(actual, expected):
    if isinstance(expected, Mapping):
        return (isinstance(actual, Mapping) and actual.keys() == expected.keys()
                and all(_same_tree(actual[key], value) for key, value in expected.items()))
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, (list, tuple)):
        return (len(actual) == len(expected)
                and all(_same_tree(a, b) for a, b in zip(actual, expected)))
    return actual == expected


def validate_alignment(bone_rows, training_catalog, aligned_rows, audit_rows):
    """Recompute and reject any changed aligned row, identity, text or audit.

    Returns ``None`` on success; raises ``ValueError`` on invalid alignment.
    Neither supplied originals nor candidate outputs are mutated.
    """
    expected_rows, expected_audits = align_catalog(bone_rows, training_catalog)
    if not _same_tree(aligned_rows, expected_rows):
        raise ValueError('semantic alignment rows differ from native recomputation')
    if not _same_tree(audit_rows, expected_audits):
        raise ValueError('semantic alignment provenance differs from native recomputation')
