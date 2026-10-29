"""Independently audit completed Bone semantic results using only the standard library.

Rows originate in immutable class markers. Merged rows and summaries are comparison
targets only. Source/model/data signatures and native control execution must also be
verified with the evaluation runner; this tool does not import that runtime.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = 'output/bone_v92e360_v102e490_training_semantics_bestfloor_20261029'
OLD = 'output/bone_v92e360_v102e490_floor_grid_20261028_cpugpu_postdrain/merged_metrics.json'
NN = 'output/bone_nninteractive_single_round_20261028/merged_metrics.json'
OLD_SHA = '7554dd78885c7caf13327abd1c51a4ff34537bd70430869da9bb936dafb0a60e'
NN_SHA = '885357d422c93f6ea7feb37abdf67a1334cc7be0c500ef107a63a4a0408727b1'
EXPECTED_SCOPE = (20, 47, 770)
FLOORS = {'v9_2': .35, 'v10_2': .75}
METRICS = ('dice', 'mean_slice_dice', 'iou', 'precision', 'recall', 'specificity',
           'pred_gt_ratio', 'mean_slice_nsd_1mm', 'mean_slice_nsd_2mm', 'mean_slice_nsd_3mm')
FALLBACKS = {'fallback_missing_prompt_logits', 'fallback_insufficient_separation'}
ATOL = 1e-12


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('duplicate JSON field: '+key)
        value[key] = item
    return value


def _reject_constant(value):
    raise ValueError('nonfinite JSON number: '+value)


def _read(path, tracked):
    path = Path(path)
    raw = path.read_bytes()
    value = hashlib.sha256(raw).hexdigest()
    if str(path) in tracked and tracked[str(path)] != value:
        raise ValueError('input changed during audit: '+str(path))
    tracked[str(path)] = value
    return json.loads(raw.decode('utf-8'), object_pairs_hook=_object, parse_constant=_reject_constant)


def _integer(value):
    if type(value) is int:
        return value
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        return int(value)
    raise ValueError('invalid integer identity')


def _key(row):
    source = row.get('source_name', row.get('dataset'))
    case = row.get('case_id')
    gid = _integer(row.get('global_class_id', row.get('class_id')))
    if source != 'bone' or not isinstance(case, str) or not case or gid <= 10000:
        raise ValueError('invalid Bone task identity')
    return source, case, gid


def _number(value, description):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('invalid/nonfinite '+description)
    return float(value)


def _metrics(row):
    for name in METRICS:
        value = _number(row[name], name)
        if value < 0 or (name != 'pred_gt_ratio' and value > 1):
            raise ValueError('out-of-bounds metric: '+name)


def _index(rows, *, metrics=False):
    result = {}
    for row in rows:
        identity = _key(row)
        if identity in result:
            raise ValueError('duplicate task key: '+repr(identity))
        if metrics:
            _metrics(row)
        result[identity] = row
    return result


def _labels(row, task, *, metric=False):
    identity = _key(task)
    label = identity[2]-10000
    if (_key(row) != identity or type(task.get('local_class_id')) is not int
            or task['local_class_id'] != label or type(row.get('local_class_id')) is not int
            or row['local_class_id'] != label
            or row.get('class_name') != task.get('class_name')):
        raise ValueError('original task/class/label identity changed')
    for item in (task, row):
        if 'label_value' in item and (type(item['label_value']) is not int or item['label_value'] != label):
            raise ValueError('original label_value changed')
    if metric and (type(row.get('label_id')) is not int or row['label_id'] != label):
        raise ValueError('original metric label_id changed')


def live_processes(out):
    """Read process command lines; never signal or modify a process."""
    proc = Path('/proc')
    if not proc.is_dir():
        return []
    found = []
    for entry in proc.iterdir():
        if not entry.name.isdecimal() or int(entry.name) == os.getpid():
            continue
        try:
            args = (entry/'cmdline').read_bytes().decode(errors='replace').split('\0')
        except OSError:
            continue
        if (any(Path(a).name == 'bone_training_semantics_eval.py' for a in args)
                and ('run' in args or ('worker' in args and any(str(out/'job_specs') in a for a in args)))):
            found.append(int(entry.name))
    return found


def _drained(out, tracked):
    state = _read(out/'status.json', tracked)
    if (state.get('phase') != 'full' or state.get('pending') != 0
            or state.get('running') != {} or state.get('failed') != [] or live_processes(out)):
        raise ValueError('audit requires a naturally drained successful full queue')


def _jobs(payload, tasks):
    jobs = {}
    coverage = {model: set() for model in FLOORS}
    control_keys = []
    cases, _, _ = EXPECTED_SCOPE
    if len(payload['cases']) != 2*cases or len(payload['controls']) != 2:
        raise ValueError('case/control job inventory changed')
    for phase, control in (('cases', False), ('controls', True)):
        phase_models = Counter()
        for job in payload[phase]:
            name, model = job['name'], job['model']
            if name in jobs or model not in FLOORS or job.get('control') is not control:
                raise ValueError('ambiguous or misclassified job')
            pending = job['pending']
            indexed = _index([item['task'] for item in pending])
            if not indexed or len(indexed) != job['task_count'] or (control and len(indexed) != 1):
                raise ValueError('invalid job task inventory')
            for item in pending:
                task = item['task']; identity = _key(task)
                if (tasks.get(identity) != task or item['floors'] != [FLOORS[model]]
                        or task['case_id'] != job['case_id']):
                    raise ValueError('job changed original task/fixed floor')
                if control:
                    control_keys.append(identity)
                else:
                    if identity in coverage[model]:
                        raise ValueError('overlapping full jobs')
                    coverage[model].add(identity)
            jobs[name] = job; phase_models[model] += 1
        if dict(phase_models) != {model: 1 if control else cases for model in FLOORS}:
            raise ValueError('case/model job partition changed')
    if any(keys != set(tasks) for keys in coverage.values()) or len(set(control_keys)) != 1:
        raise ValueError('full/control task coverage changed')
    return jobs


def _marker(marker, out, jobs, tasks, tracked):
    info = _read(marker, tracked); job = jobs[info['job_name']]
    run = marker.parents[1]
    prefix = job['name']+'_a'
    if not run.name.startswith(prefix) or not run.name[len(prefix):].isdecimal():
        raise ValueError('marker outside its declared job attempt')
    record_path = marker.with_name(marker.name.removesuffix('.done.json')+'.json')
    spec_path = out/'job_specs'/(run.name+'.json')
    paths = [record_path, run/'restoration_start.json', run/'resource_policy_audit.json', spec_path,
             run/'prepared/manifest.json', run/'prepared/refs.json', run/'prepared_input_audit.json', out/'protocol.json']
    if info['record'] != str(record_path) or set(info['files']) != {str(p) for p in paths}:
        raise ValueError('marker must bind exactly eight expected proof files')
    for path in paths:
        if sha(path) != info['files'][str(path)]:
            raise ValueError('immutable marker file SHA mismatch: '+str(path))
    if info['protocol_sha'] != tracked[str(out/'protocol.json')]:
        raise ValueError('class protocol binding changed')
    spec = _read(spec_path, tracked)
    for field in ('name', 'model', 'control', 'source', 'variant', 'case_id', 'prepared', 'parent_history'):
        if spec.get(field) != job.get(field):
            raise ValueError('task spec/canonical job mismatch: '+field)
    if (type(spec['attempt']) is not int or spec['attempt'] < 0 or run.name != prefix+str(spec['attempt'])
            or spec['run'] != str(run) or spec['spec'] != str(spec_path)
            or spec['log'] != str(out/'logs'/(run.name+'.log'))
            or spec['frame_batch'] not in (1, 2) or spec['frame_batch'] > job['frame_batch']):
        raise ValueError('attempt path/frame-batch binding changed')
    expected = {_key(item['task']): item for item in job['pending']}
    pending = spec['pending']; subset = _index([item['task'] for item in pending])
    if not pending or len(subset) != spec['task_count'] or any(expected.get(_key(x['task'])) != x for x in pending):
        raise ValueError('worker spec is not a unique canonical task subset')
    manifest = _read(run/'prepared/manifest.json', tracked)
    refs = _read(run/'prepared/refs.json', tracked)
    frozen_refs = _index(_read(out/'refs.json', tracked)['per_case_class'])
    if (manifest['tasks'] != [x['task'] for x in pending] or manifest['expected_task_count'] != len(pending)
            or refs['per_case_class'] != [frozen_refs[_key(x['task'])] for x in pending]):
        raise ValueError('worker prepared task/reference subset changed')
    prepared = _read(run/'prepared_input_audit.json', tracked)
    if prepared != dict(canonical_prepared=job['prepared'], files={str(run/'prepared'/n):tracked[str(run/'prepared'/n)] for n in ('manifest.json','refs.json')}):
        raise ValueError('worker prepared-input proof mismatch')
    record = _read(record_path, tracked); task = record['task']; identity = _key(task)
    if (task != tasks.get(identity) or sum(item['task'] == task for item in pending) != 1
            or record['model'] != job['model'] or record['semantic_execution'].get('control') is not job['control']):
        raise ValueError('class/spec/phase binding changed')
    _labels(task, tasks[identity])
    if marker.name != task['case_id']+'__'+str(identity[2])+'.done.json':
        raise ValueError('class marker filename identity changed')
    if record['restoration'] != _read(run/'restoration_start.json', tracked):
        raise ValueError('record restoration proof differs')
    resource = _read(run/'resource_policy_audit.json', tracked)
    if (any(resource.get(k) != spec[k] for k in ('gpu','gpu_uuid','frame_batch','minimum_free_mib','parent_history'))
            or resource['actual_start_free_mib'] < spec['minimum_free_mib'] or resource['new_oom_retries'] != spec['oom_retries']):
        raise ValueError('resource/spec proof mismatch')
    if len(record['results']) != 1:
        raise ValueError('exactly one fixed-floor class result required')
    result = record['results'][0]; row = result['row']; floor = FLOORS[job['model']]
    if result['floor'] != floor or not result['stages']:
        raise ValueError('class fixed floor/native stage missing')
    _labels(row, task, metric=True); _metrics(row)
    final = result['stages'][-1]
    threshold = _number(row['mask_threshold_used'], 'final mask threshold')
    source = row['dynamic_threshold_source']
    if (final['key'] != list(identity) or final['floor'] != floor
            or final['threshold'] != threshold or final['source'] != source
            or not isinstance(source, str) or not source):
        raise ValueError('final native-stage threshold/source binding changed')
    if source.startswith('fallback'):
        if source not in FALLBACKS or threshold != .60:
            raise ValueError('unknown/changed final fallback')
    elif not floor <= threshold <= .75:
        raise ValueError('final threshold outside fixed native bounds')
    proof = dict(path=str(marker), sha256=tracked[str(marker)], job=job['name'], control=job['control'],
                 model=job['model'], key=list(identity), files=info['files'])
    return job, row, spec['frame_batch'], proof


def _paired(rows, reference):
    other = _index(reference, metrics=True)
    if set(other) != set(rows):
        raise ValueError('paired reference task keys differ')
    return {name: math.fsum(float(rows[k][name])-float(other[k][name]) for k in sorted(rows))/len(rows) for name in METRICS}


def _close(actual, expected, description):
    if abs(_number(actual, description)-expected) > ATOL:
        raise ValueError('merged value differs from immutable rows: '+description)


def audit_results(out, old_reference, nn_reference):
    """Validate a completed run and return a report without writing any files."""
    try:
        return _audit(Path(out).resolve(), Path(old_reference).resolve(), Path(nn_reference).resolve())
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError('malformed audit artifact: '+str(error)) from error


def _audit(out, old_reference, nn_reference):
    tracked = {}; _drained(out, tracked)
    merged = _read(out/'merged_metrics.json', tracked)
    _read(out/'protocol.json', tracked)
    manifest = _read(out/'manifest.json', tracked); tasks = _index(manifest['tasks'])
    cases, classes, count = EXPECTED_SCOPE
    if (len(tasks) != count or len({k[1] for k in tasks}) != cases or len({k[2] for k in tasks}) != classes
            or manifest['expected_task_count'] != count):
        raise ValueError('fixed Bone case/class/task scope changed')
    names = {}
    for identity, task in tasks.items():
        _labels(task, task)
        if identity[2] in names and names[identity[2]] != task['class_name']:
            raise ValueError('class name differs between cases')
        names[identity[2]] = task['class_name']
    jobs = _jobs(_read(out/'jobs.json', tracked), tasks)
    refs = _index(_read(out/'refs.json', tracked)['per_case_class'])
    if set(refs) != set(tasks):
        raise ValueError('frozen prompt reference keys differ')
    rows = {m: {} for m in FLOORS}; batches = {m: Counter() for m in FLOORS}
    controls = set(); proofs = []
    for marker in sorted((out/'runs').glob('*/classes/*.done.json')):
        job, row, batch, proof = _marker(marker, out, jobs, tasks, tracked)
        identity = _key(row); model = job['model']; proofs.append(proof)
        if job['control']:
            control_key = model, identity
            if control_key in controls:
                raise ValueError('duplicate control record')
            controls.add(control_key)
        else:
            if identity in rows[model]:
                raise ValueError('duplicate fresh class record')
            rows[model][identity] = row; batches[model][str(batch)] += 1
    if len(controls) != 2 or any(set(part) != set(tasks) for part in rows.values()):
        raise ValueError('missing fresh class/control markers')
    if sha(old_reference) != OLD_SHA or sha(nn_reference) != NN_SHA:
        raise ValueError('frozen old/nnInteractive reference SHA changed')
    old = _read(old_reference, tracked); nn = _read(nn_reference, tracked)
    nn_rows = nn.get('per_case_class', nn.get('models', {}).get('nninteractive', {}).get('per_case_class', []))
    if (merged['rows'] != 2*count or merged['predictions'] != 2*count or merged['task_count_per_model'] != count
            or merged['controls_in_final'] is not False or set(merged['models']) != set(FLOORS)
            or merged['protocol_sha'] != tracked[str(out/'protocol.json')]
            or merged['nninteractive_reference_sha256'] != NN_SHA):
        raise ValueError('merged scope/provenance/control declaration changed')
    summaries = {}
    for model, floor in FLOORS.items():
        part = rows[model]; ordered = sorted(part); by_class = defaultdict(list)
        for identity in ordered:
            by_class[identity[2]].append(float(part[identity]['dice']))
        summary = dict(task_count=count, floor=floor,
            mean={n: math.fsum(float(part[k][n]) for k in ordered)/count for n in METRICS},
            class_macro_dice=math.fsum(math.fsum(v)/len(v) for _,v in sorted(by_class.items()))/classes,
            fallback_count=sum(part[k]['dynamic_threshold_source'] in FALLBACKS for k in ordered),
            actual_frame_batches={str(b):batches[model][str(b)] for b in (1,2)},
            delta_from_old_same_floor=_paired(part, old['models'][model]['floors'][format(floor,'.2f')]['per_case_class']),
            delta_from_nninteractive=_paired(part, nn_rows))
        claimed = merged['models'][model]
        claimed_rows = _index(claimed['per_case_class'], metrics=True)
        for identity, row in claimed_rows.items():
            if identity not in tasks:
                raise ValueError('merged task key not in frozen manifest')
            _labels(row, tasks[identity], metric=True)
        if claimed_rows != part:
            raise ValueError('merged rows differ from immutable fresh class rows')
        for name in ('task_count', 'floor', 'fallback_count', 'actual_frame_batches'):
            if claimed[name] != summary[name]:
                raise ValueError('merged summary binding differs: '+name)
        _close(claimed['class_macro_dice'], summary['class_macro_dice'], 'class_macro_dice')
        for group in ('mean', 'delta_from_old_same_floor', 'delta_from_nninteractive'):
            if set(claimed[group]) != set(METRICS):
                raise ValueError('merged ten-metric summary fields changed')
            for name in METRICS:
                _close(claimed[group][name], summary[group][name], group+'/'+name)
        summaries[model] = summary
    _drained(out, tracked)
    for path, value in tracked.items():
        if sha(path) != value:
            raise ValueError('input changed before audit completion: '+path)
    return dict(schema='independent-bone-training-semantics-results-v1', passed=True,
        fresh_rows=2*count, excluded_control_records=len(controls), cases=cases, classes=classes,
        task_count_per_model=count, fixed_floors=FLOORS.copy(), metric_names=list(METRICS),
        comparison_atol=ATOL, models=summaries, class_markers=proofs, input_sha256=tracked,
        validation_scope='immutable-result integrity and independent arithmetic; native source/model/data/control verification is separate')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/OUT)
    parser.add_argument('--old-reference', type=Path, default=ROOT/OLD)
    parser.add_argument('--nn-reference', type=Path, default=ROOT/NN)
    parser.add_argument('--audit-output', type=Path)
    args = parser.parse_args(argv)
    output = args.audit_output or args.out/'independent_results_audit.json'
    if output.exists():
        raise FileExistsError('independent audit output already exists: '+str(output))
    report = audit_results(args.out, args.old_reference, args.nn_reference)
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    with output.open('x', encoding='utf-8') as stream:
        stream.write(serialized); stream.flush(); os.fsync(stream.fileno())
    print('INDEPENDENT_BONE_RESULTS_AUDIT_PASS', report['fresh_rows'], str(output), flush=True)
    return report


if __name__ == '__main__':
    main()
