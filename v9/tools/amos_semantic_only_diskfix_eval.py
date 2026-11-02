"""Independent semantic-only recovery after the immutable shared-volume ENOSPC.

Only whole cases with an already published native parent certificate are reused.
The failed run remains failed; all other cases obtain fresh native completion.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PARENT_ENTRY = 'tools/amos_semantic_only_restored_eval.py'
PARENT_TEST = 'tests/test_amos_semantic_only_restored_eval.py'
PARENT_ENTRY_SHA256 = '981b1b4b12082eba3d4e0fc0235ad142f8e41278d9b86238b87270e1ea3cbcfe'
PARENT_TEST_SHA256 = 'e948868829b4c4a0379dd808f0f527b00a6a86aefb98cbc5999b07ee738d7433'
PARENT_OUT = 'output/amos_v102_e490_floor070_semantic_only_restored_20261102'
PARENT_TMUX = 'amos_v102_floor070_semantic_only_restored_20261102'
PARENT_PIPELINE = 'output/amos_semantic_only_restored_pipeline_20261102.log'
PARENT_PROTOCOL_SHA256 = '5946b7a010067c383f6d747d18869a34cc77af4c917216fe5ef05237c976b923'
PARENT_GATE_SHA256 = 'b4ec117dde344f511c7b43b088bfd1572ae1b7ba20d2cbede305d517ee673b84'
PARENT_PIPELINE_SHA256 = '4f3d28553a295cd542c11dbdfa08b395b3803cb35ea6ad340bdeee299521c8e4'
OUT = 'output/amos_v102_e490_floor070_semantic_only_diskfix_20261103'
TMUX = 'amos_v102_floor070_semantic_only_diskfix_20261103'
PIPELINE = 'output/amos_semantic_only_diskfix_pipeline_20261103.log'
FAILURE = 'output/amos_semantic_diskfull_failure_20261102.json'
INDEPENDENT = 'output/amos_semantic_diskfull_independent_audit_20261102.json'
ANCESTRY = 'output/amos_semantic_diskfull_ancestry_rehash_20261102.json'
FAILURE_PINS = {
    FAILURE: '218957daf43d64d3df02cb0773496b300db50c654d91c51ce703b99c4e147936',
    INDEPENDENT: 'd3049593e9f4ab463724607d0516ddf988113f6a5180e168fe2fa4aa760a974c',
    ANCESTRY: 'a26e07075b8c54db995c059f6076ddbb8b025f749d5fbde970ebb29724638201',
}
RECOVERY_SOURCES = ('tools/amos_semantic_only_diskfix_eval.py',
                    'tests/test_amos_semantic_only_diskfix_eval.py', PARENT_ENTRY, PARENT_TEST)
DISK_RESERVE_BYTES = 3 * 1024**3
REUSED_ROWS = 486
FRESH_ROWS = 999
CANDIDATE_ROWS = 531
REUSED_CASES = 33
UNUSED_PARENT_ROWS = 45
OLD_ENTRIES = frozenset(('amos_semantic_only_eval.py', 'amos_semantic_only_restored_eval.py'))
ENTRIES = OLD_ENTRIES | {'amos_semantic_only_diskfix_eval.py'}
AUDIT_KIND = 'amos_semantic_enospc_certified_case_reuse_v1'
HISTORY_FIELDS = ('frame_batch', 'minimum_free_mib', 'oom_retries', 'not_before', 'avoid_gpus')
_CONTEXT = None
_PARENT_READER = None


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def source_guard(*, production=False):
    for name, expected in ((PARENT_ENTRY, PARENT_ENTRY_SHA256), (PARENT_TEST, PARENT_TEST_SHA256)):
        if sha(ROOT/name) != expected:
            raise ValueError('signed ENOSPC parent source SHA changed: '+name)
    # The unchanged restored guard authenticates original runtime and S21 profile.
    parent = importlib.import_module('tools.amos_semantic_only_restored_eval')
    proof = parent.source_guard(production=production)
    if production:
        for name, expected in FAILURE_PINS.items():
            if sha(ROOT/name) != expected:
                raise ValueError('ENOSPC evidence SHA changed: '+name)
        for name, expected in ((PARENT_OUT+'/protocol.json', PARENT_PROTOCOL_SHA256),
                               (PARENT_OUT+'/control_verification.json', PARENT_GATE_SHA256),
                               (PARENT_PIPELINE, PARENT_PIPELINE_SHA256)):
            if sha(ROOT/name) != expected:
                raise ValueError('ENOSPC parent evidence SHA changed: '+name)
        return dict(proof, enospc_evidence=dict(FAILURE_PINS))
    return proof


def related(proc_root=None):
    proc = Path('/proc') if proc_root is None else Path(proc_root)
    if not proc.exists():
        return []
    rows = []
    for folder in proc.iterdir():
        if not folder.name.isdigit() or int(folder.name) == os.getpid():
            continue
        try:
            args = (folder/'cmdline').read_bytes().decode(errors='replace').split('\0')
        except OSError:
            continue
        if any(Path(arg).name in ENTRIES for arg in args) and any(arg in ('prepare', 'run', 'worker') for arg in args):
            rows.append(dict(pid=int(folder.name), args=args))
    return rows


def parent_quiescence():
    old = [row for row in related() if any(Path(arg).name in OLD_ENTRIES for arg in row['args'])]
    if old:
        raise ValueError('failed parent is no longer quiescent: '+repr(old))
    for name in (PARENT_TMUX, 'amos_v102_floor070_semantic_only_20261102'):
        result = subprocess.run(['tmux', 'has-session', '-t', name], capture_output=True, text=True)
        if result.returncode not in (0, 1):
            raise RuntimeError('old tmux inspection failed: '+result.stderr)
        if result.returncode == 0:
            raise ValueError('old semantic tmux session exists')


def disk_free_bytes():
    # Query the destination filesystem even before the fresh output exists.
    return shutil.disk_usage(ROOT).free


def preflight(out, *, processes=None, tmux_exists=None, pipeline=None):
    if Path(out) != ROOT/OUT:
        raise ValueError('fresh dedicated ENOSPC recovery output required')
    if disk_free_bytes() < DISK_RESERVE_BYTES:
        raise ValueError('shared output filesystem below 3 GiB disk reserve')
    # Skip only the restored wrapper's ban on existence of its failed ancestor.
    _native_preflight(out, processes=processes, tmux_exists=tmux_exists,
                      pipeline=ROOT/PIPELINE if pipeline is None else pipeline)
    parent_quiescence()


def parent_engine():
    global _PARENT_READER
    if _PARENT_READER is None:
        _PARENT_READER = importlib.import_module('tools.amos_semantic_only_restored_eval').compile_context().engine()
    return _PARENT_READER


def select_certified(found):
    """Select by immutable published certificates and canonical coverage only."""
    p = parent_engine(); out = ROOT/PARENT_OUT
    jobs = read(out/'jobs.json')['cases']; chosen = {}; cases = {}
    expected = {key(t) for j in jobs for t in j['pending']}
    if len(found) != CANDIDATE_ROWS or not set(found) <= expected:
        raise ValueError('parent531 complete canonical class set changed')
    for job in jobs:
        certificate = out/'case_success'/(job['name']+'.json')
        if not certificate.exists():
            continue
        subset = {k: v for k, v in found.items() if v['spec']['name'] == job['name']}
        if set(subset) != {key(t) for t in job['pending']}:
            raise ValueError('published certified case lacks complete canonical coverage')
        p['verify_case_success'](out, job, found)
        cases[job['name']] = dict(kind='parent_case_success', path=str(certificate), sha256=sha(certificate))
        chosen = union_records(chosen, subset)
    if len(cases) != REUSED_CASES or len(chosen) != REUSED_ROWS:
        raise ValueError('exact33 certified cases and486 reusable rows required')
    return chosen, cases


def verify_parent(*, rehash=False):
    _source_guard(production=True); parent_quiescence()
    p = parent_engine(); out = ROOT/PARENT_OUT
    p['verify'](out, rehash=rehash); p['check_control_gate'](out)
    failure = read(ROOT/FAILURE); independent = read(ROOT/INDEPENDENT); ancestry = read(ROOT/ANCESTRY)
    status = read(out/'status.json')
    if (failure.get('kind') != 'amos_semantic_shared_volume_enospc_v1' or failure.get('output') != PARENT_OUT
            or failure.get('protocol_sha256') != PARENT_PROTOCOL_SHA256 or failure.get('gate_sha256') != PARENT_GATE_SHA256
            or failure.get('pipeline_sha256') != PARENT_PIPELINE_SHA256
            or status != failure.get('status') or status != independent.get('status')
            or status.get('phase') != 'full' or status.get('running') != {}
            or status.get('failed') != [dict(stage='native_worker_complete', error="OSError(28, 'No space left on device')")]
            or failure.get('related_checks') != [[], []] or failure.get('tmux_exists') is not False
            or independent.get('complete_class_records') != CANDIDATE_ROWS or independent.get('certified_rows') != REUSED_ROWS
            or independent.get('full_rehash') is not True or independent.get('quiescence') != []
            or independent.get('failure_evidence_sha256') != FAILURE_PINS[FAILURE]
            or any(independent.get(k) != failure.get(k) for k in ('protocol_sha256', 'gate_sha256', 'pipeline_sha256'))
            or ancestry.get('passed') is not True or ancestry.get('full_rehash') is not True or ancestry.get('classes') != 1485):
        raise ValueError('captured ENOSPC failure/audit contract changed')
    found = p['class_records'](out)
    selected, cases = select_certified(found)
    observed = sorted((dict(name=name, classes=sum(v['spec']['name'] == name for v in selected.values())) for name in cases), key=lambda r: r['name'])
    if observed != sorted(independent['certified_cases'], key=lambda r: r['name']):
        raise ValueError('independently audited33 certificate set changed')
    specs = {str(path): read(path) for path in (out/'job_specs').glob('*.json')}
    captured = failure['runs']
    actual_attempts = [(spec['name'], spec['attempt']) for spec in specs.values()]
    captured_attempts = [(run['name'], run['attempt']) for run in captured]
    if (len(actual_attempts) != len(set(actual_attempts))
            or len(captured_attempts) != len(set(captured_attempts))
            or set(actual_attempts) != set(captured_attempts)):
        raise ValueError('captured parent attempt set changed')
    for run in captured:
        matches = [spec for spec in specs.values() if spec['name'] == run['name'] and spec['attempt'] == run['attempt']]
        if len(matches) != 1:
            raise ValueError('captured parent attempt missing or ambiguous')
        spec = matches[0]
        if (any(spec.get(k) != run[k] for k in ('control', 'gpu', 'attempt', 'oom_retries'))
                or spec.get('case_id') != run['case'] or spec.get('frame_batch') != run['batch']
                or spec.get('log') != run['log'] or sha(spec['log']) != run['log_sha256']):
            raise ValueError('captured ENOSPC attempt/log changed')
        for path, expected_sha in run['files'].items():
            if sha(path) != expected_sha:
                raise ValueError('captured ENOSPC native finish proof changed')
    return found


def advance_history(canonical, previous):
    value = copy.deepcopy(canonical)
    if previous is not None:
        value.update({name: copy.deepcopy(previous[name]) for name in HISTORY_FIELDS})
        value['attempt'] = previous['attempt'] + 1
    return value


def parent_history():
    # Independent new attempts inherit actual state; ENOSPC is never a CUDA OOM.
    verify_parent(); p = parent_engine(); out = ROOT/PARENT_OUT
    jobs = {j['name']: j for j in read(out/'jobs.json')['cases']}
    history = {name: [] for name in jobs}
    for path in sorted((out/'job_specs').glob('*.json')):
        spec = read(path)
        if spec['control']:
            continue
        if spec['name'] not in jobs:
            raise ValueError('foreign parent attempt history')
        p['validate_attempt'](spec, jobs[spec['name']], out)
        if str(path) != spec['spec']:
            raise ValueError('parent attempt filename differs from signed path')
        history[spec['name']].append(spec)
    for attempts in history.values():
        attempts.sort(key=lambda j: j['attempt'])
        if [j['attempt'] for j in attempts] != list(range(len(attempts))):
            raise ValueError('parent attempt history gaps or duplicate attempts')
    return history


def case_jobs(tasks, strict=True):
    jobs = _diskfix_native_case_jobs(tasks, strict)
    if not strict:
        return jobs
    history = parent_history()
    if set(history) != {j['name'] for j in jobs}:
        raise ValueError('canonical100 case attempt history changed')
    return [advance_history(j, history[j['name']][-1] if history[j['name']] else None) for j in jobs]


def union_records(old, new):
    if set(old) & set(new):
        raise ValueError('duplicate reused/new committed class')
    return {**old, **new}


def _record_index(found):
    return [dict(key=list(k), marker=v['marker'], marker_sha256=v['marker_sha256'],
                 spec=v['spec']['spec'], spec_sha256=sha(v['spec']['spec'])) for k, v in sorted(found.items())]


def build_reuse_audit():
    found = verify_parent(rehash=True); selected, cases = select_certified(found)
    out = ROOT/PARENT_OUT; protocol = read(out/'protocol.json')
    # Freeze the entire small proof tree, including excluded rows and partial
    # files. This inventories bytes; no CT, model or prediction is copied.
    tree = sorted(path for path in out.rglob('*') if path.is_file())
    paths = [*tree, ROOT/PARENT_PIPELINE, *(ROOT/name for name in FAILURE_PINS)]
    for field in ('sources', 'prepared'):
        paths.extend(ROOT/name for name in protocol[field])
    # Native validation authenticates the old controls but never reuses their gate.
    parent_engine()['class_records'](out, True)
    return dict(kind=AUDIT_KIND, parent_output=PARENT_OUT,
                parent_protocol_sha256=PARENT_PROTOCOL_SHA256, parent_gate_sha256=PARENT_GATE_SHA256,
                failure_evidence=dict(FAILURE_PINS), candidate_rows=CANDIDATE_ROWS,
                reused_rows=REUSED_ROWS, unused_parent_rows=UNUSED_PARENT_ROWS, fresh_rows=FRESH_ROWS,
                reused_cases=REUSED_CASES, records=_record_index(selected), all_parent_records=_record_index(found),
                histories=parent_history(), cases=cases,
                parent_tree=[str(path.relative_to(out)).replace('\\', '/') for path in tree], files=file_inventory(paths))


def verify_reuse(out):
    out = Path(out); audit = read(out/'reuse_audit.json'); bootstrap = read(out/'native_bootstrap_audit.json')
    expected = dict(kind=AUDIT_KIND, parent_output=PARENT_OUT,
                    parent_protocol_sha256=PARENT_PROTOCOL_SHA256, parent_gate_sha256=PARENT_GATE_SHA256,
                    failure_evidence=dict(FAILURE_PINS), candidate_rows=CANDIDATE_ROWS,
                    reused_rows=REUSED_ROWS, unused_parent_rows=UNUSED_PARENT_ROWS, fresh_rows=FRESH_ROWS,
                    reused_cases=REUSED_CASES)
    if (any(audit.get(k) != v for k, v in expected.items())
            or bootstrap.get('reuse_audit_sha256') != sha(out/'reuse_audit.json')):
        raise ValueError('immutable ENOSPC reuse audit chain changed')
    verify_inventory(audit['files'])
    parent = ROOT/PARENT_OUT
    if sorted(str(path.relative_to(parent)).replace('\\', '/') for path in parent.rglob('*') if path.is_file()) != audit['parent_tree']:
        raise ValueError('failed parent proof tree changed')
    return audit


def frozen_records(out):
    _source_guard(production=True); parent_quiescence()
    audit = verify_reuse(out)
    # Endpoints verify parent protocol/data/assets in full. Queue polls still
    # rehash every frozen proof and validate all531 native class records, without
    # repeatedly hashing the same model assets between dispatch decisions.
    found = parent_engine()['class_records'](ROOT/PARENT_OUT)
    selected, cases = select_certified(found)
    if (_record_index(found) != audit['all_parent_records'] or _record_index(selected) != audit['records']
            or cases != audit['cases']):
        raise ValueError('frozen531 parent /486 selected class proofs changed')
    return selected


def class_records(out, control=False):
    if control:
        return _diskfix_native_records(out, True)
    return union_records(frozen_records(out), _diskfix_native_records(out, False))


def verify_case_success(out, canonical, found):
    if canonical['control']:
        return _diskfix_native_case_success(out, canonical, found)
    audit = verify_reuse(out)
    name = canonical['name']
    if name not in audit['cases']:
        return _diskfix_native_case_success(out, canonical, found)
    expected = {key(t) for t in canonical['pending']}
    selected = {k: v for k, v in found.items() if v['spec']['name'] == name}
    frozen = [row for row in audit['records'] if tuple(row['key']) in expected]
    if set(selected) != expected or _record_index(selected) != frozen:
        raise ValueError('reused case must have only complete frozen parent classes')
    p = parent_engine(); parent = ROOT/PARENT_OUT
    previous = next(j for j in read(parent/'jobs.json')['cases'] if j['name'] == name)
    if previous['pending'] != canonical['pending']:
        raise ValueError('reused canonical case task list changed')
    p['verify_case_success'](parent, previous, selected)
    certificate = parent/'case_success'/(name+'.json')
    if audit['cases'][name] != dict(kind='parent_case_success', path=str(certificate), sha256=sha(certificate)):
        raise ValueError('immutable parent case certificate changed')
    return audit['cases'][name]


def _native_bootstrap_preflight(out, jobs, controls):
    out = Path(out); audit = build_reuse_audit()
    atomic_new(out/'reuse_audit.json', audit)
    native = _bootstrap_function(root=ROOT, out=out, jobs=jobs, controls=controls)
    native['reuse_audit_sha256'] = sha(out/'reuse_audit.json')
    path = out/'native_bootstrap_audit.json'; atomic_new(path, native)
    return path


def dependency_sources():
    return list(dict.fromkeys([*_diskfix_native_sources(), *(ROOT/name for name in FAILURE_PINS)]))


def policy_extension():
    return dict(_diskfix_native_policy(), fresh_rows=FRESH_ROWS, reused_rows=REUSED_ROWS,
                unused_parent_rows=UNUSED_PARENT_ROWS, reused_cases=REUSED_CASES,
                reuse_policy='published_native_full_case_certificates_only')


def prepare(out, **kwargs):
    _source_guard(production=True); verify_parent(rehash=True)
    return _diskfix_native_prepare(out, **kwargs)


def run_pipeline(out):
    _source_guard(production=True); verify_parent(rehash=True)
    return _diskfix_native_pipeline(out)


def verify(out, **kwargs):
    _source_guard(production=True); verify_reuse(out)
    verify_parent(rehash=kwargs.get('rehash', False))
    return _diskfix_native_verify(out, **kwargs)


def schedule(out, control):
    if control:
        return _diskfix_native_schedule(out, True)
    verify(out, rehash=False); check_control_gate(out)
    committed = class_records(out)
    jobs = read(Path(out)/'jobs.json')['cases']
    pending = [j for j in jobs if any(key(t) not in committed for t in j['pending'])]
    if pending:
        return schedule_queue(pending, False, Backend(out, False))
    state = dict(time=time.time(), phase='full', pending=0, running={}, failed=[], committed_classes=len(committed))
    write_status(Path(out)/'status.json', state)
    return state


def build_merged(out):
    # All scientific arithmetic and1485 by-key paired comparisons stay native.
    value = _diskfix_native_build_merged(out)
    value.update(fresh_rows=FRESH_ROWS, reused_rows=REUSED_ROWS, unused_parent_rows=UNUSED_PARENT_ROWS,
                 reused_cases=REUSED_CASES, fresh_cases=100-REUSED_CASES,
                 reuse_audit_sha256=sha(Path(out)/'reuse_audit.json'),
                 parent_protocol_sha256=PARENT_PROTOCOL_SHA256, parent_gate_sha256=PARENT_GATE_SHA256,
                 failure_evidence=dict(FAILURE_PINS))
    return value


def compile_context(*, source=None):
    source_guard()
    private = importlib.import_module('tools.amos_semantic_only_restored_eval').compile_context(source=source)
    ns = private.__dict__
    ns.update({name: value for name, value in globals().items() if name.isupper() and not name.startswith('_')})
    ns.update(__name__='_amos_semantic_diskfix_controller_private', __doc__=__doc__,
              __file__=str(Path(__file__).resolve()), shutil=shutil, _PARENT_READER=None)
    for name, original in (('prepare', 'prepare'), ('pipeline', 'run_pipeline'), ('verify', 'verify'),
                           ('case_jobs', 'case_jobs'), ('records', 'class_records'), ('sources', 'dependency_sources'),
                           ('policy', 'policy_extension'), ('case_success', 'verify_case_success'),
                           ('schedule', 'schedule'), ('build_merged', 'build_merged')):
        ns['_diskfix_native_'+name] = ns[original]
    ns['SOURCE_SEEDS'] = tuple(dict.fromkeys((*ns['SOURCE_SEEDS'], *RECOVERY_SOURCES)))
    functions = (sha, source_guard, related, parent_quiescence, disk_free_bytes, preflight, parent_engine,
                 select_certified, verify_parent, advance_history, parent_history, case_jobs, union_records,
                 _record_index, build_reuse_audit, verify_reuse, frozen_records, class_records, verify_case_success,
                 _native_bootstrap_preflight, dependency_sources, policy_extension, prepare, run_pipeline, verify,
                 schedule, build_merged)
    for fn in functions:
        bound = types.FunctionType(fn.__code__, ns, fn.__name__, fn.__defaults__, fn.__closure__)
        bound.__kwdefaults__ = fn.__kwdefaults__; ns[fn.__name__] = bound
    ns['_source_guard'] = ns['source_guard']
    # Inherited methods and every child argv use the same isolated globals.
    exec('''
def engine(): return globals()
def runtime(): return importlib.import_module('tools.amos_semantic_only_runtime')
_DiskfixNativeBackend = Backend
class Backend(_DiskfixNativeBackend):
    def available(self):
        if disk_free_bytes() < DISK_RESERVE_BYTES: return {}
        return super().available()
''', ns)
    return private


def context():
    global _CONTEXT
    if _CONTEXT is None:
        _CONTEXT = compile_context()
    return _CONTEXT


def main():
    return context().engine()['main']()


if __name__ == '__main__':
    main()
