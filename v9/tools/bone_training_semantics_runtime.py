"""Add literal-text consumption evidence to the frozen native Bone runtime.

Only Python call boundaries are observed. Native values are returned untouched;
no tensor, image, logit, mask, parameter, or new disk cache is produced here.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
from collections.abc import Mapping
from pathlib import Path

from tools.bone_floor_protocol import key
from tools.bone_metric_speed_runtime import (
    GridSession as PriorSession, install_v92, install_v10, tag_carrier,
    install_restoration as _native_install_restoration,
)
from v9_2_extended_protocol import conditioning_text_from_metadata


_CONFIG = None
_SCHEMA = 'bone-native-semantic-execution-v1'


def _sha(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _digest(value):
    return _sha(json.dumps(value, ensure_ascii=False, sort_keys=True,
                           separators=(',', ':'), allow_nan=False))


def _freeze(catalog, control):
    if isinstance(catalog, (str, bytes, Mapping)):
        raise ValueError('semantic catalog must be the class_catalog row sequence')
    rows = {}
    for original in catalog:
        if not isinstance(original, Mapping):
            raise ValueError('invalid semantic catalog row')
        if original.get('dataset') != 'bone':
            continue
        row = copy.deepcopy(dict(original))
        gid = row.get('global_class_id')
        if isinstance(gid, bool) or not isinstance(gid, int) or gid in rows:
            raise ValueError('duplicate or invalid Bone semantic global ID')
        text, base = row.get('semantic_text'), row.get('semantic_base_text')
        if not isinstance(text, str) or not text.strip() or not isinstance(base, str) or not base.strip():
            raise ValueError('missing final or base semantic text')
        try:
            conditioned = conditioning_text_from_metadata(dict(row, semantic_text=base))
        except (KeyError, TypeError) as exc:
            raise ValueError('missing training semantic metadata') from exc
        if conditioned != text:
            raise ValueError('final semantic text differs from native training builder')
        rows[gid] = row
    if not rows:
        raise ValueError('empty Bone semantic catalog')
    return dict(rows=rows, catalog_sha256=_digest([rows[g] for g in sorted(rows)]),
                control=bool(control))


def configure_semantics(catalog, control=False):
    """Freeze the final Bone strings and complete row provenance for one worker."""
    global _CONFIG
    value = _freeze(catalog, control)
    _CONFIG = value
    return dict(catalog_sha256=value['catalog_sha256'], classes=len(value['rows']),
                control=value['control'])


def _expected(config, task):
    identity = key(task)
    if identity[0] != 'bone' or identity[2] not in config['rows']:
        raise ValueError('task is absent from frozen Bone semantic catalog')
    row = config['rows'][identity[2]]
    for field in ('class_name', 'local_class_id', 'label_value'):
        if field in row and field in task and row[field] != task[field]:
            raise ValueError('semantic task identity changed: ' + field)
    return identity, row


def validate_semantic_record(record, catalog, control=False):
    """Verify the saved actual-consumption proof independently of worker state."""
    config = _freeze(catalog, control)
    identity, row = _expected(config, record['task'])
    proof = record.get('semantic_execution', {})
    expected = row['semantic_text']
    bindings = dict(schema=_SCHEMA, catalog_sha256=config['catalog_sha256'],
                    catalog_row_sha256=_digest(row), global_class_id=identity[2],
                    expected_semantic_text=expected, expected_text_sha256=_sha(expected),
                    model=record['model'], control=bool(control), semantic_enabled=True,
                    provider_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if any(proof.get(name) != value for name, value in bindings.items()):
        raise ValueError('missing or changed semantic execution/catalog binding')
    events = proof.get('events', [])
    if not events or proof.get('call_count') != len(events):
        raise ValueError('class completion without positive actual semantic calls')
    floors = {float(result['floor']) for result in record['results']}
    stages = {float(result['floor']): result['stages'] for result in record['results']
              if 'stages' in result}
    encoder_calls = 0
    for index, event in enumerate(events, 1):
        if (event.get('key') != list(identity) or event.get('model') != record['model']
                or event.get('floor') not in floors or event.get('call_index') != index
                or type(event.get('stage_index')) is not int or event['stage_index'] < 0
                or event.get('semantic_text') != expected or event.get('text_sha256') != _sha(expected)
                or event.get('semantic_enabled') is not True or event.get('native_return_present') is not True):
            raise ValueError('actual semantic text/class/stage binding mismatch')
        if event['floor'] in stages and event['stage_index'] >= len(stages[event['floor']]):
            raise ValueError('semantic call refers to a missing native stage')
        texts = event.get('encoder_texts', [])
        count = event.get('native_encoder_calls')
        if (type(count) is not int or count != len(texts) or count < 0
                or event.get('encoder_text_sha256') != [_sha(t) for t in texts]
                or any(t != expected for t in texts)):
            raise ValueError('native semantic encoder consumption mismatch')
        if event.get('native_path') == '_semantic_embedding':
            if count < 1 or event.get('precomputed_embedding') is not False:
                raise ValueError('native semantic embedding bypassed literal encoder input')
        elif event.get('native_path') != '_anatomy_bias' or count != 0:
            raise ValueError('unknown native semantic consumption path')
        encoder_calls += count
    if encoder_calls < 1 or proof.get('native_encoder_call_count') != encoder_calls:
        raise ValueError('no successful native semantic encoder calls for class')


class GridSession(PriorSession):
    def __init__(self, model, pending, fingerprint, on_class, **kwargs):
        if _CONFIG is None:
            raise ValueError('configure_semantics must precede GridSession construction')
        self._semantic_config = copy.deepcopy(_CONFIG)
        for item in pending:
            _expected(self._semantic_config, item['task'])
        self.semantic_events = []
        self._semantic_active = None
        self.shared_control = self._semantic_config['control']

        def committed(record):
            identity, row = _expected(self._semantic_config, record['task'])
            text = row['semantic_text']
            record['semantic_execution'] = dict(
                schema=_SCHEMA, model=self.model, control=self.shared_control,
                catalog_sha256=self._semantic_config['catalog_sha256'],
                catalog_row_sha256=_digest(row), global_class_id=identity[2],
                expected_semantic_text=text, expected_text_sha256=_sha(text),
                semantic_enabled=True, events=copy.deepcopy(self.semantic_events),
                call_count=len(self.semantic_events),
                native_encoder_call_count=sum(e['native_encoder_calls'] for e in self.semantic_events),
                provider_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            validate_semantic_record(record, self._semantic_config['rows'].values(),
                                     self._semantic_config['control'])
            return on_class(record)
        super().__init__(model, pending, fingerprint, committed, **kwargs)

    def begin(self, task):
        _expected(self._semantic_config, task)
        if self._semantic_active is not None:
            raise ValueError('class changed inside native semantic consumption')
        value = super().begin(task)
        self.semantic_events = []
        return value


def _literal(session, prompt, value):
    if getattr(prompt, 'semantic_enabled', None) is not True:
        raise ValueError('native semantic prompts are disabled or missing')
    if session.current is None or session.floor is None:
        raise ValueError('native semantic call lacks active class/floor')
    task, floors = session.pending[session.current]
    identity, row = _expected(session._semantic_config, task)
    if session.current != identity or session.floor not in floors:
        raise ValueError('native semantic class/floor context mismatch')
    if not isinstance(value, str) or value != row['semantic_text']:
        raise ValueError('actual semantic text differs from frozen training-aligned text')
    return dict(key=list(identity), model=session.model, floor=session.floor,
                stage_index=len(session.stage_rows.get(session.floor, [])),
                semantic_text=value, text_sha256=_sha(value), semantic_enabled=True,
                native_encoder_calls=0, encoder_texts=[], encoder_text_sha256=[])


def _wrap_prompt(session, prompt):
    """Link successful native embedding calls to their actual SAT forward calls."""
    native = getattr(prompt, '_semantic_embedding', None)
    encoder = getattr(prompt, 'semantic_encoder', None)
    native_encode = getattr(encoder, 'forward', None)
    if not callable(native) or not callable(native_encode):
        raise ValueError('restored prompt lacks native semantic encoder path')
    if getattr(prompt, '_bone_semantic_audit_installed', False):
        raise ValueError('unexpected repeated semantic audit installation')
    signature = inspect.signature(native)
    encoder_signature = inspect.signature(native_encode)

    def encode(*args, **kwargs):
        event = session._semantic_active
        if event is None:
            raise ValueError('native semantic encoder called outside prompt consumption')
        bound = encoder_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        text = bound.arguments.get('texts')
        _literal(session, prompt, text)
        if text != event['semantic_text']:
            raise ValueError('semantic encoder changed literal prompt text')
        modality = bound.arguments.get('modality') or getattr(encoder, 'default_modality', None)
        if modality != 'ct':
            raise ValueError('native semantic encoder changed frozen CT modality')
        value = native_encode(*args, **kwargs)
        if value is None:
            raise ValueError('native semantic encoder returned no embedding')
        event['native_encoder_calls'] += 1
        event['encoder_texts'].append(text)
        event['encoder_text_sha256'].append(_sha(text))
        return value

    def embedding(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        event = _literal(session, prompt, bound.arguments.get('semantic_text'))
        if bound.arguments.get('semantic_embedding') is not None:
            raise ValueError('precomputed semantic embedding bypasses actual literal text')
        if session._semantic_active is not None:
            raise ValueError('nested native semantic embedding context')
        event.update(native_path='_semantic_embedding', precomputed_embedding=False)
        session._semantic_active = event
        try:
            value = native(*args, **kwargs)
            if value is None or not event['native_encoder_calls']:
                raise ValueError('prompt did not consume native semantic encoder output')
            event.update(native_return_present=True, call_index=len(session.semantic_events) + 1)
            session.semantic_events.append(event)
            return value
        finally:
            session._semantic_active = None

    native_anatomy = getattr(prompt, '_anatomy_bias', None)
    if callable(native_anatomy):
        anatomy_signature = inspect.signature(native_anatomy)
        def anatomy(*args, **kwargs):
            bound = anatomy_signature.bind(*args, **kwargs)
            event = _literal(session, prompt, bound.arguments.get('semantic_text'))
            value = native_anatomy(*args, **kwargs)
            if value is None:
                raise ValueError('native anatomy path returned no semantic bias')
            event.update(native_path='_anatomy_bias', native_return_present=True,
                         call_index=len(session.semantic_events) + 1)
            session.semantic_events.append(event)
            return value

    # Assignment happens only after the complete native path has been checked.
    encoder.forward = encode
    prompt._semantic_embedding = embedding
    if callable(native_anatomy):
        prompt._anatomy_bias = anatomy
    prompt._bone_semantic_audit_installed = True


def install_restoration(session, save_proof):
    """Keep native exact guards, then instrument the prompt after its load returns.

    The inherited v9 helper postpones decoder verification to session.begin,
    after native main has loaded the decoder. The v10 helper stays synchronous.
    """
    _native_install_restoration(session, save_proof)
    if session.model == 'v9_2':
        import finetune_multisource_sam2_v9_2_precision as entry
        owner, name = entry.base, 'load_prompt_generator_checkpoint'
    elif session.model == 'v10_2':
        import v10.finetune_totalseg_amos_magic_v10_2_joint as entry
        owner, name = entry, 'load_init_v10_2'
    else:
        raise ValueError('unsupported native semantic model')
    restored_load = getattr(owner, name)
    def load(path, prompt, *args, **kwargs):
        result = restored_load(path, prompt, *args, **kwargs)
        _wrap_prompt(session, prompt)
        return result
    setattr(owner, name, load)
