"""Recover only the three unpublished VISTA3D groups from the fixed MAGIC run."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

ENTRY = Path(__file__).resolve()
sys.path.insert(0, str(ENTRY.parents[1]))
from tools import magic_baseline_prompt_matrix as magic
from tools import native_vista3d_pilot as vista

matrix, native, out = magic.matrix, magic.native, magic.OUT
preflight = out / 'vista3d_edge_retry_preflight.json'
report = out / 'vista3d_edge_retry_result.json'
old_patch_predict = vista.patch_predict


def project_grid_edge(points, shape):
    """Only snap points that would be dropped despite lying inside the edge voxel."""
    p = np.asarray(points, dtype=float)
    shape = np.asarray(shape, dtype=int)
    if (shape.shape != (3,) or np.any(shape < 1) or p.ndim != 2 or p.shape[1] != 3
            or not np.isfinite(p).all() or np.any(p < -0.50000001)
            or np.any(p > shape - 0.49999999)):
        raise ValueError('VISTA point lies outside the physical grid')
    padded = np.maximum(shape, 128)
    q = p + (padded - shape) // 2
    return np.where(q < 0, 0., np.where(q >= padded, shape - 1., p))


def patched_predict(model, image, points, labels):
    effective = project_grid_edge(points, image.shape)
    ids = np.flatnonzero(np.any(effective != points, axis=1))
    logits, audits = old_patch_predict(model, image, effective, labels)
    if len(ids):
        audits[0]['magic_edge_projection'] = dict(
            indices=ids.tolist(), original=points[ids].tolist(), effective=effective[ids].tolist(),
            model_grid_xyz=list(image.shape))
        print('VISTA_EDGE_POINTS_PROJECTED', audits[0]['magic_edge_projection'], flush=True)
    return logits, audits


def validate_missing(jobs, status):
    if (len(jobs) != 3 or len(set(jobs)) != 3 or status['active'] or status['pending']
            or any(method != 'vista3d' or group != 0 for method, _, group in jobs)
            or set(status['failed']) != {'/'.join(map(str, job)) for job in jobs}
            or any(v['rc'] != 1 for v in status['failed'].values())):
        raise ValueError('unexpected retry scope; refusing to dispatch')
    return sorted(jobs)


def inventory(payload, digest):
    hashes, missing, scored = {}, [], 0
    for task in payload['tasks']:
        case = task['case_id']
        for method, groups in payload['groups'].items():
            for group, _ in enumerate(groups):
                record_path = matrix.task_file(method, task, group)
                cert_path = matrix.certificate_file(method, case, group)
                if not cert_path.is_file() and not record_path.is_file():
                    missing.append((method, case, group))
                    continue
                record = matrix.read_record(method, task, group, digest)
                cert = native.read(cert_path)
                if (cert['keys'] != [list(native.key(task))] or cert['protocol_sha256'] != digest
                        or cert['method'] != method or cert['group'] != group
                        or cert['uuid'] != record['gpu_uuid']):
                    raise ValueError(f'invalid existing group {method}/{case}/{group}')
                for path in (record_path, cert_path):
                    hashes[path.relative_to(out).as_posix()] = native.sha(path)
                scored += len(record['rows'])
    return hashes, missing, scored


def bind_retry():
    return dict(source_sha256=native.sha(ENTRY),
                geometry_sha256=native.sha(magic.ROOT / 'infer/native_vista3d_prompts.py'),
                protocol_sha256=native.sha(out / 'protocol.json'))


def worker(case, gpu, uuid, attempt):
    spec = native.read(preflight)
    job = ('vista3d', case, 0)
    if spec['bindings'] != bind_retry() or list(job) not in spec['jobs']:
        raise ValueError('worker does not match the recovery scope')
    if matrix.certificate_file(*job).exists():
        raise ValueError('refusing to rerun a completed group')
    task, = (t for t in matrix.protocol()['tasks'] if t['case_id'] == case)
    if matrix.task_file(job[0], task, job[2]).exists():
        raise ValueError('refusing to overwrite an existing record')
    vista.patch_predict = patched_predict
    try:
        matrix.worker(*job, gpu, uuid, attempt, 4)
    finally:
        vista.patch_predict = old_patch_predict


def dispatch(jobs):
    from tools.fumpe_six_baselines import worker_runtime

    executable, env = worker_runtime('vista3d')
    env.update(V9_SHARED_CACHE_WRITE_POLICY='read_only')
    pending, active, failed, excluded = list(jobs), {}, {}, {}
    last_launch = time.monotonic()
    while pending or active:
        for gpu, (job, child, log) in list(active.items()):
            rc = child.poll()
            if rc is None:
                continue
            log.close()
            del active[gpu]
            if rc == 75:
                excluded[job, gpu] = time.monotonic() + 60
                pending.append(job)
            elif rc or not matrix.certificate_file(*job).is_file():
                failed[job] = rc
            print('MAGIC_VISTA_RETRY_EXIT', job, gpu, rc, flush=True)
        if pending and shutil.disk_usage(out).free >= 3 * 1024**3:
            cards = sorted(matrix.gpu_inventory().items(), key=lambda item: item[1][1], reverse=True)
            for gpu, (uuid, free) in cards:
                if gpu in active or free < matrix.MINIMUM_MIB:
                    continue
                job = next((j for j in pending if excluded.get((j, gpu), 0) <= time.monotonic()), None)
                if job is None:
                    continue
                method, case, group = job
                logdir = out / 'logs' / method / case / str(group)
                attempt = 1 + max(int(p.stem) for p in logdir.glob('[0-9][0-9][0-9].log'))
                log = (logdir / f'{attempt:03d}.log').open('x')
                child = subprocess.Popen([executable, '-B', str(ENTRY), 'worker', case, str(gpu), uuid,
                                          str(attempt)], cwd=magic.ROOT, stdout=log,
                                         stderr=subprocess.STDOUT, env={**env, 'CUDA_VISIBLE_DEVICES': uuid})
                active[gpu] = (job, child, log)
                pending.remove(job)
                last_launch = time.monotonic()
                print('MAGIC_VISTA_RETRY_DISPATCH', job, gpu, child.pid, attempt, flush=True)
        if pending and not active and time.monotonic() - last_launch > 900:
            raise RuntimeError('no eligible GPU or disk capacity for the remaining scoped groups')
        if pending or active:
            time.sleep(5)
    if failed:
        raise RuntimeError(f'recovery jobs failed: {failed}')


def recover():
    import fcntl

    with (out / 'schedule.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any(p.exists() for p in (preflight, report, out / 'merged_metrics.json')):
            raise ValueError('recovery already started or merged; refusing to overwrite')
        payload = matrix.protocol()
        magic.verify_inputs(payload)
        bindings = bind_retry()
        digest = bindings['protocol_sha256']
        before, missing, count = inventory(payload, digest)
        jobs = validate_missing(missing, native.read(out / 'status.json'))
        if count != 1289 or len(before) != 834:
            raise ValueError('existing result count does not match the audited run')
        native.write(preflight, dict(bindings=bindings, jobs=jobs, preserved_sha256=before))
        print('MAGIC_VISTA_RETRY_PREFLIGHT', count, len(before), jobs, flush=True)
        dispatch(jobs)
        after, missing, count = inventory(payload, digest)
        if (missing or count != 1295 or len(after) != 840 or
                any(after.get(path) != sha for path, sha in before.items()) or bindings != bind_retry()):
            raise ValueError('scoped recovery failed the preservation audit')
        magic.verify_inputs(payload)
        magic.merge(payload)
        native.write(report, dict(bindings=bindings, recovered_groups=jobs, preserved_files=len(before),
                                  preserved_scores=1289, recovered_scores=6, total=count,
                                  new_files_sha256={k: v for k, v in after.items() if k not in before},
                                  merged_sha256=native.sha(out / 'merged_metrics.json')))
        print('MAGIC_VISTA_RETRY_PASS', count, 'preserved_files', len(before), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    subs.add_parser('run')
    w = subs.add_parser('worker')
    w.add_argument('case')
    w.add_argument('gpu', type=int)
    w.add_argument('uuid')
    w.add_argument('attempt', type=int)
    args = parser.parse_args()
    with magic.runner():
        if args.command == 'run':
            recover()
        else:
            worker(args.case, args.gpu, args.uuid, args.attempt)


if __name__ == '__main__':
    main()
