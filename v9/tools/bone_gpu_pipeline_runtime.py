"""CPU floor overlap with one exclusive, audited GPU producer per physical card.

The inherited model, feature/raw memo, metrics and postprocessing stay unchanged.
Only small flock files are written; native producers must already return CPU data.
"""
from contextlib import contextmanager
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time
import uuid

from tools.bone_metric_speed_runtime import (
    GridSession as PriorSession, install_v92, install_v10 as _install_v10,
    install_restoration, tag_carrier,
)

_JOB = None
_LOCKS = {}


def _binding(job):
    if (type(job.get('gpu')) is not int or job['gpu'] not in (2, 4, 5, 6)
            or type(job.get('slot')) is not int or job['slot'] not in (0, 1, 2)
            or type(job.get('minimum_free_mib')) is not int or job['minimum_free_mib'] not in (30720, 45000)
            or not isinstance(job.get('gpu_uuid'), str) or not job['gpu_uuid'].startswith('GPU-')):
        raise ValueError('invalid signed pipeline GPU/UUID/slot/memory policy')
    spec = Path(job['spec']).resolve()
    payload = spec.read_bytes()
    if json.loads(payload) != job:
        raise ValueError('pipeline job differs from its signed spec')
    run = Path(job['run']).resolve()
    if run.parent.name != 'runs' or spec != run.parents[1]/'job_specs'/(run.name+'.json'):
        raise ValueError('pipeline run/spec output binding mismatch')
    return dict(spec_path=str(spec), spec_sha256=hashlib.sha256(payload).hexdigest(),
                run=str(run), output=str(run.parents[1]),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                **{name: job[name] for name in ('gpu', 'gpu_uuid', 'slot', 'minimum_free_mib')})


def configure_job(job):
    """Bind the original immutable attempt before the compiled run changes prepared."""
    global _JOB
    _binding(job)
    _JOB = copy.deepcopy(job)


def _gpu_snapshot():
    from tools.bone_floor_grid import gpu_snapshot
    return gpu_snapshot()


def _flock(handle, exclusive):
    import fcntl  # Import/help remains usable on Windows; execution requires POSIX.
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_UN)


def _synchronize():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _empty_cache():
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class GpuLease:
    """Process-reentrant flock, including cleanup and fresh checks inside the lock."""
    def __init__(self, job, *, snapshot=None, synchronize=None, empty_cache=None):
        self.binding = _binding(job)
        self.path = Path(self.binding['output'])/'gpu_locks'/('gpu_'+str(job['gpu'])+'.lock')
        self.snapshot = snapshot or _gpu_snapshot
        self.synchronize = synchronize or _synchronize
        self.empty_cache = empty_cache or _empty_cache

    @contextmanager
    def __call__(self, phase):
        state = _LOCKS.setdefault((os.getpid(), str(self.path)),
                                 SimpleLockState())
        with state.mutex:
            outer = state.depth == 0
            handle, locked, event = None, False, None
            try:
                if outer:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    handle = self.path.open('a+')
                    _flock(handle, True)
                    locked = True
                    state.events = []
                actual = self.snapshot()[self.binding['gpu']]
                if actual['uuid'] != self.binding['gpu_uuid']:
                    raise ValueError('physical GPU UUID changed inside exclusive lease')
                if actual['free'] < self.binding['minimum_free_mib']:
                    raise SystemExit(75)
                event = dict(id=uuid.uuid4().hex, pid=os.getpid(), phase=phase,
                    **{k: self.binding[k] for k in ('gpu', 'gpu_uuid', 'slot', 'minimum_free_mib')},
                    free_mib=actual['free'], lock_path=str(self.path),
                    start_monotonic=time.monotonic(), end_monotonic=None,
                    synchronized=False, nested=not outer,
                    parent_id=state.events[0]['id'] if not outer else None)
                state.events.append(event)
                state.depth += 1
                try:
                    yield event
                finally:
                    state.depth -= 1
                    event['end_monotonic'] = time.monotonic()
                    if outer:
                        try:
                            self.synchronize()
                            for item in state.events:
                                item['synchronized'] = True
                        finally:
                            try:
                                self.empty_cache()
                            finally:
                                event['end_monotonic'] = time.monotonic()
            finally:
                if outer and handle is not None:
                    try:
                        if locked:
                            _flock(handle, False)
                    finally:
                        handle.close()


class SimpleLockState:
    def __init__(self):
        self.mutex = threading.RLock()
        self.depth = 0
        self.events = []


def assert_cpu_tree(value):
    """Check, never relocate or cast, the native raw output."""
    import torch
    if torch.is_tensor(value) and value.device.type != 'cpu':
        raise ValueError('native raw output must already be CPU')
    if isinstance(value, dict):
        for child in value.values():
            assert_cpu_tree(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            assert_cpu_tree(child)
    return value


def _restored(model, restoration):
    r = restoration or {}
    prompt, decoder, epoch = (438, 192, 360) if model == 'v9_2' else (418, 212, 490)
    return (r.get('model') == model and r.get('epoch') == epoch
        and r.get('weight_mode') == 'live' and r.get('guards_passed') is True
        and r.get('prompt_parameters_exact') == prompt and r.get('decoder_parameters_exact') == decoder)


class GridSession(PriorSession):
    def __init__(self, model, pending, fingerprint, on_class, **kwargs):
        if _JOB is None:
            raise ValueError('configure_job must bind the signed attempt first')
        self.pipeline_job = copy.deepcopy(_JOB)
        if self.pipeline_job['model'] != model:
            raise ValueError('pipeline model differs from signed job')
        self.pipeline_binding = _binding(self.pipeline_job)
        self.gpu_lease = GpuLease(self.pipeline_job)
        self.raw_events, self.native_probe_leases = [], []
        self._startup = self.gpu_lease('startup')
        self.startup_lease = self._startup.__enter__()
        def committed(record):
            record['pipeline_execution'] = dict(self.pipeline_binding,
                startup_lease=copy.deepcopy(self.startup_lease),
                native_probe_leases=copy.deepcopy(self.native_probe_leases),
                raw_events=copy.deepcopy(self.raw_events))
            validate_pipeline_record(record, self.pipeline_job)
            return on_class(record)
        try:
            super().__init__(model, pending, fingerprint, committed, **kwargs)
        except BaseException:
            self.close()
            raise

    def close(self):
        if self._startup is not None:
            startup, self._startup = self._startup, None
            startup.__exit__(None, None, None)

    def begin(self, task):
        if self._startup is not None:
            if not _restored(self.model, self.restoration):
                raise ValueError('startup lease requires exact native restoration before release')
            self.close()
        value = super().begin(task)
        self.raw_events = []
        return value

    def raw(self, compute, video, bundle, class_id, args):
        if self._startup is not None:
            raise ValueError('raw producer attempted before restoration/begin')
        produced = []
        def guarded():
            with self.gpu_lease('raw') as event:
                value = assert_cpu_tree(compute())
                event['cpu_output'] = True
                produced.append(event)
                return value
        value = super().raw(guarded, video, bundle, class_id, args)
        if produced:
            event = produced[0]
            event['raw_key'] = self.last_stage['raw_key']
            self.raw_proofs[event['raw_key']]['gpu_lease'] = copy.deepcopy(event)
            self.raw_events.append(copy.deepcopy(event))
        return value


def install_v10(session):
    # The native control twin executes before the inherited raw callback.
    from tools import bone_shared_floor_restore_runtime as restore
    native = restore.probe_native_batches
    native = getattr(native, '_pipeline_original', native)
    def probe(shared, control, args):
        def checked(compute):
            def run():
                return assert_cpu_tree(compute())
            return run
        with session.gpu_lease('native_probe') as event:
            value = native(checked(shared), checked(control), args)
            event.update(cpu_output=True, batches=[1, 2, 4])
        session.native_probe_leases.append(copy.deepcopy(event))
        return value
    probe._pipeline_original = native
    restore.probe_native_batches = probe
    return _install_v10(session)


def validate_pipeline_record(record, job):
    """Validate against the actual signed attempt, never a spec claimed by a proof."""
    binding = _binding(job)
    proof = record.get('pipeline_execution', {})
    if any(proof.get(k) != v for k, v in binding.items()) or record.get('model') != job['model']:
        raise ValueError('pipeline execution differs from signed job/source binding')
    if not _restored(record['model'], record.get('restoration')):
        raise ValueError('pipeline startup lacks exact native restoration proof')
    lock_path = str(Path(binding['output'])/'gpu_locks'/('gpu_'+str(job['gpu'])+'.lock'))
    startup = proof.get('startup_lease', {})
    def check(event, phase):
        if (not event.get('id') or event.get('phase') != phase
                or any(event.get(k) != job[k] for k in ('gpu', 'gpu_uuid', 'slot', 'minimum_free_mib'))
                or event.get('lock_path') != lock_path or event.get('synchronized') is not True
                or type(event.get('pid')) is not int or event['pid'] <= 0
                or event['pid'] != startup.get('pid')
                or type(event.get('free_mib')) is not int or event['free_mib'] < job['minimum_free_mib']):
            raise ValueError('invalid actual GPU lease identity/guard/synchronization')
        start, end = event.get('start_monotonic'), event.get('end_monotonic')
        if (type(start) not in (int, float) or type(end) not in (int, float)
                or not math.isfinite(start) or not math.isfinite(end) or not 0 < start <= end):
            raise ValueError('invalid GPU lease interval')
        if phase != 'startup' and (event.get('cpu_output') is not True or start < startup['end_monotonic']):
            raise ValueError('GPU producer lacks native CPU output/restoration ordering')
    check(startup, 'startup')
    events = proof.get('raw_events', [])
    producers = record.get('raw_producers', {})
    if not events or len(events) != len(producers) or len({e.get('raw_key') for e in events}) != len(events):
        raise ValueError('pipeline lease coverage differs from actual raw producers')
    for event in events:
        check(event, 'raw')
        if producers.get(event.get('raw_key'), {}).get('gpu_lease') != event:
            raise ValueError('raw producer lacks its actual GPU lease evidence')
    probes = proof.get('native_probe_leases', [])
    for event in probes:
        check(event, 'native_probe')
        if event.get('batches') != [1, 2, 4]:
            raise ValueError('native batch twins lack GPU lease coverage')
    if job.get('control') and job['model'] == 'v10_2' and not probes:
        raise ValueError('control lacks native twin GPU lease')
    all_events = [startup, *probes, *events]
    by_id = {event['id']: event for event in all_events}
    if len(by_id) != len(all_events):
        raise ValueError('duplicate GPU lease identity')
    roots = []
    for event in all_events:
        if event.get('nested') is False and event.get('parent_id') is None:
            roots.append(event)
        elif event.get('nested') is True:
            parent = by_id.get(event.get('parent_id'), {})
            if (parent.get('nested') is not False or parent['start_monotonic'] > event['start_monotonic']
                    or event['end_monotonic'] > parent['end_monotonic']):
                raise ValueError('nested GPU lease lacks its enclosing exclusive lease')
        else:
            raise ValueError('invalid GPU lease nesting')
    roots.sort(key=lambda event: event['start_monotonic'])
    if any(left['end_monotonic'] > right['start_monotonic'] for left, right in zip(roots, roots[1:])):
        raise ValueError('exclusive GPU lease intervals overlap')
