"""Replicate the AMOS prompt matrix on a pre-seeded, disjoint ten-case draw."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import math
from pathlib import Path
import random
import sys

ENTRY = Path(__file__).resolve()
ROOT = ENTRY.parents[1]
sys.path.insert(0, str(ROOT))
from tools import amos_prompt_matrix as matrix
from tools import magic_vista3d_retry as edge
from tools import native_vista3d_pilot as vista

native = matrix.native
PRIOR = matrix.OUT
OUT = ROOT / 'output/amos_prompt_matrix_random10_20261121'
SEED = 20261121
NUM_CASES = 10
SOURCE_BINDING = matrix.code_binding


def select_random_tasks(rows, split, previous):
    old = matrix.select_tasks(rows)  # Also validates all 100 CTs and 1485 nonempty keys.
    amos = [r for r in rows if r['source_name'] == 'amos']
    val = [r['case_id'] for r in split['val']]
    train = {r['case_id'] for r in split['train']}
    if len(val) != 100 or set(val) != {r['case_id'] for r in amos} or set(val) & train:
        raise ValueError('expected the fixed AMOS holdout with no training overlap')
    exclude = sorted({r['case_id'] for r in old})
    if (sorted(previous['cases']) != exclude or
            Counter(map(native.key, previous['tasks'])) != Counter(map(native.key, old))):
        raise ValueError('archived ten-case matrix does not match the original cohort')
    chosen = set(random.Random(SEED).sample(sorted(set(val) - set(exclude)), NUM_CASES))
    return sorted((dict(r) for r in amos if r['case_id'] in chosen), key=native.key)


def groups():
    return {m: [list(a) for a in matrix.arm_groups(m)] for m in matrix.ARMS}


def methods():
    return {m: list(arms) for m, arms in matrix.ARMS.items()}


def inputs():
    code = SOURCE_BINDING()
    for p in ('tools/amos_prompt_matrix.py', 'tools/amos_prompt_matrix_random10.py',
              'tools/magic_vista3d_retry.py', 'infer/native_vista3d_prompts.py'):
        code[p] = native.sha(ROOT / p)
    return dict(sources=code, split_sha=native.sha(matrix.SPLIT), manifest_sha=native.sha(matrix.SOURCE),
                s21_protocol_sha=native.sha(matrix.REFERENCE / 'protocol.json'),
                s21_manifest_sha=native.sha(matrix.REFERENCE / 'manifest.json'),
                archived_protocol_sha=native.sha(PRIOR / 'protocol.json'),
                archived_summary_sha=native.sha(PRIOR / 'merged_metrics.json'))


def candidates():
    old = native.read(PRIOR / 'protocol.json')
    split = native.read(matrix.SPLIT)['datasets']['amos']
    rows = select_random_tasks(native.read(matrix.SOURCE)['tasks'], split, old)
    ref = Counter(map(native.key, native.read(matrix.REFERENCE / 'manifest.json')['tasks']))
    if any(ref[native.key(row)] != 1 for row in rows):
        raise ValueError('selected task missing from the fixed S21 reference')
    return rows, sorted(old['cases'])


def verify_inputs(payload):
    if (payload['bindings'] != inputs() or payload['seed'] != SEED or payload['sample_count'] != NUM_CASES
            or payload['methods'] != methods() or payload['groups'] != groups()
            or len(payload['cases']) != NUM_CASES
            or set(payload['cases']) & set(payload['excluded_cases'])
            or sorted({r['case_id'] for r in payload['tasks']}) != payload['cases']):
        raise ValueError('random-ten AMOS protocol or its signed sources changed')


def prepare():
    path = OUT / 'protocol.json'
    if path.exists():
        raise FileExistsError('random-ten protocol already exists')
    before = inputs()
    rows, excluded = candidates()
    cases = sorted({r['case_id'] for r in rows})
    if len(cases) != NUM_CASES:
        raise ValueError('number of selected cases changed')
    for case in cases:
        folder = matrix.REFERENCE / 'prepared' / f'amos_{case}'
        if any(not (folder / filename).is_file() for filename in ('manifest.json', 'refs.json')):
            raise ValueError(f'missing original PromptGen references for {case}')
    loader = matrix.CaseLoader()
    counts = Counter()
    for row in rows:
        prompts = loader(row)
        if any(len(nodes) != 2 for nodes in prompts.points('bg', 2)):
            raise ValueError(f'two distinct BG points per face unavailable for {native.key(row)}')
        row['prompt_identity'] = matrix.prompt_identity(prompts, loader.files)
        counts[row['case_id']] += 1
    payload = dict(seed=SEED, sample_count=NUM_CASES, excluded_cases=excluded, cases=cases,
                   tasks=rows, methods=methods(), groups=groups(), bindings=before)
    verify_inputs(payload)
    native.write(path, payload)
    print('AMOS_RANDOM10_PREPARED', len(cases), len(rows), len(rows) * 41, dict(counts), flush=True)


def summary(payload):
    digest = native.sha(OUT / 'protocol.json')
    expected, excluded = candidates()
    if (list(map(native.key, expected)) != list(map(native.key, payload['tasks']))
            or excluded != payload['excluded_cases']):
        raise ValueError('protocol tasks do not reproduce the frozen random draw')
    count = len(expected)
    rows = {}
    for method, arms in matrix.ARMS.items():
        by_arm = {a: [] for a in arms}
        for case in payload['cases']:
            tasks = [t for t in payload['tasks'] if t['case_id'] == case]
            for group in range(len(matrix.arm_groups(method))):
                cert = native.read(matrix.certificate_file(method, case, group))
                if (cert['method'] != method or cert['group'] != group or cert['protocol_sha256'] != digest
                        or cert['keys'] != [list(native.key(t)) for t in tasks]):
                    raise ValueError('case certificate does not match the sealed protocol')
                for t in tasks:
                    record = matrix.read_record(method, t, group, digest)
                    if record['gpu_uuid'] != cert['uuid']:
                        raise ValueError('worker record and case certificate have different GPU UUIDs')
                    for item in record['rows']:
                        by_arm[item['arm']].append(item['metrics'])
        if any(len(r) != count for r in by_arm.values()):
            raise ValueError(f'incomplete coverage for {method}')
        rows[method] = {a: dict(tasks=count, averages={metric: math.fsum(r[metric] for r in rs) / count
                        for metric in native.METRICS}) for a, rs in by_arm.items()}
    return dict(seed=SEED, cases=payload['cases'], tasks=count, rows=count * sum(map(len, rows.values())),
                protocol_sha256=digest, methods=rows)


def merge(payload):
    verify_inputs(payload)
    path = OUT / 'merged_metrics.json'
    record = summary(payload)
    if path.exists():
        if native.read(path) != record:
            raise ValueError('saved random-ten result differs from committed rows')
    else:
        native.write(path, record)
    print('AMOS_RANDOM10_COMPLETE', record['rows'], flush=True)


@contextmanager
def runner():
    values = dict(OUT=OUT, __file__=str(ENTRY), __doc__=__doc__,
                  prepare=prepare, verify_inputs=verify_inputs, merge=merge)
    previous = {k: getattr(matrix, k) for k in values}
    patch = vista.patch_predict
    vars(matrix).update(values)
    vista.patch_predict = edge.patched_predict
    try:
        yield
    finally:
        vars(matrix).update(previous)
        vista.patch_predict = patch


def main():
    with runner():
        if sys.argv[1:] == ['verify']:
            p = matrix.protocol()
            verify_inputs(p)
            if native.read(OUT / 'merged_metrics.json') != summary(p):
                raise ValueError('saved random-ten result differs from committed rows')
            print('AMOS_RANDOM10_VERIFY_PASS', len(p['tasks']) * sum(map(len, matrix.ARMS.values())))
        else:
            matrix.main()


if __name__ == '__main__':
    main()
