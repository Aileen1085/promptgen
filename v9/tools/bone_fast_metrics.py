"""Exact, source-bound acceleration for native binary PromptGen metrics.

``metric, tracker = install_provider(native_module, compare_native=False,
workers=2, cache_entries=128)`` returns a drop-in
``binary_prompt_metrics_from_masks`` callable.  Install that callable at the
runtime import boundary; this module never patches the native module itself.
Only ``mean_slice_nsd_scores`` is replaced in a private copy of the binary
function's globals.  All volume reductions, epsilon formulas, and slice Dice
remain the original code.

Each nonempty paired slice uses the native surface/float32 point construction,
one full ``torch.cdist`` matrix, and one minimum per direction.  The same minima
serve the native 1/2/3 mm comparisons.  At most two ordered slice jobs run at a
time.  There are no distance transforms or approximate distances.

The bounded LRU retains only SHA-256 input signatures and scalar metric dicts.
Signatures hash chunked packed bits of ``pred.float() > .5`` and verified binary
``target.float()``, including shape, dtype, strides, devices, reduction threads
and spacing.  Nonbinary targets bypass the cache.  No masks, tensors, CT data,
logits or files are retained.
``tracker.clear()`` clears the LRU at a class boundary; counters remain cumulative
so callers can subtract two ``snapshot()`` values for per-class evidence.

``compare_native`` may be a bool or a zero-argument predicate.  Enabled calls,
including LRU hits, execute the independently cloned, unpatched native metric
and assert absolute error <= 1e-6 for every returned scalar.  Snapshots include
source hashes, counters/times and bounded last-call metadata.  No per-call event
history is retained; callers can record task/floor context around the callable.
"""
from __future__ import annotations

from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
import functools
import hashlib
import math
from pathlib import Path
import struct
import threading
import time
from types import FunctionType, ModuleType
from typing import Callable, Iterator

import numpy as np
import torch


_TOLERANCES = (1.0, 2.0, 3.0)
_NSD_NAMES = tuple(f"mean_slice_nsd_{int(t)}mm" for t in _TOLERANCES)
_SIGNATURE_CHUNK_VOXELS = 1 << 20


class MetricTracker:
    """Cumulative scalar evidence and a class-local, bounded metadata-only LRU."""

    def __init__(self, native_module: ModuleType, workers: int, cache_entries: int):
        self._lock = threading.Lock()
        self._cache: OrderedDict[bytes, dict[str, float]] = OrderedDict()
        self._capacity = cache_entries
        native_path = Path(native_module.__file__).resolve()
        provider_path = Path(__file__).resolve()
        self._metadata = {
            "native_source_path": str(native_path),
            "native_source_sha256": hashlib.sha256(native_path.read_bytes()).hexdigest(),
            "provider_source_sha256": hashlib.sha256(provider_path.read_bytes()).hexdigest(),
            "workers": workers,
            "cache_capacity": cache_entries,
            "cache_tensor_bytes": 0,
            "comparison_atol": 1e-6,
            "metric_names": list(native_module.PROMPTGEN_BINARY_METRIC_NAMES),
            "last_input_sha256": None,
            "last_cache_hit": False,
            "last_native_compared": False,
            "last_comparison_max_abs_error": None,
        }
        self._stats = dict.fromkeys((
            "calls", "fast_calls", "native_calls", "comparisons", "comparison_failures",
            "cache_hits", "cache_misses", "cache_bypasses", "cache_evictions", "cache_clear_calls",
            "fast_cdist_calls", "native_cdist_calls", "fast_nsd_slices",
        ), 0)
        self._stats.update(dict.fromkeys((
            "fast_metric_seconds", "native_metric_seconds", "signature_seconds", "total_metric_seconds",
            "fast_nsd_seconds", "fast_cdist_seconds", "native_cdist_seconds", "comparison_max_abs_error",
        ), 0.0))

    def _add(self, **values):
        with self._lock:
            for name, value in values.items():
                self._stats[name] += value

    def _last(self, **values):
        with self._lock:
            self._metadata.update(values)

    def clear(self):
        """Discard class-local memo entries without resetting cumulative evidence."""
        with self._lock:
            self._cache.clear()
            self._stats["cache_clear_calls"] += 1

    def snapshot(self) -> dict:
        """Return JSON-friendly scalar metadata; no input data or tensors."""
        with self._lock:
            result = dict(self._metadata, **self._stats, cache_entries=len(self._cache))
            result["metric_names"] = list(result["metric_names"])
        # Aliases make the evidence straightforward to bind in runtime records.
        result.update(
            native_compares=result["comparisons"],
            native_mismatches=result["comparison_failures"],
            native_seconds=result["native_metric_seconds"],
            fast_seconds=result["fast_metric_seconds"],
            nsd_cdist_calls=result["fast_cdist_calls"],
            source_sha256=result["native_source_sha256"],
        )
        return result

    def _get(self, key):
        with self._lock:
            if key not in self._cache:
                self._stats["cache_misses"] += 1
                return None
            self._stats["cache_hits"] += 1
            self._cache.move_to_end(key)
            return dict(self._cache[key])

    def _put(self, key, result):
        with self._lock:
            self._cache[key] = dict(result)
            self._cache.move_to_end(key)
            if len(self._cache) > self._capacity:
                self._cache.popitem(last=False)
                self._stats["cache_evictions"] += 1

    def _compare(self, actual, expected):
        max_error = 0.0
        try:
            if set(actual) != set(expected):
                raise AssertionError(f"metric keys differ: fast={sorted(actual)}, native={sorted(expected)}")
            for name, native_value in expected.items():
                fast_value = actual[name]
                if math.isnan(native_value) and math.isnan(fast_value):
                    continue
                if fast_value == native_value:
                    continue
                error = abs(fast_value - native_value)
                max_error = max(max_error, error if math.isfinite(error) else math.inf)
                if not math.isclose(fast_value, native_value, rel_tol=0.0, abs_tol=1e-6):
                    raise AssertionError(f"native metric mismatch {name}: fast={fast_value!r}, native={native_value!r}, atol=1e-6")
        except AssertionError:
            self._add(comparison_failures=1)
            raise
        else:
            self._add(comparisons=1)
        finally:
            with self._lock:
                self._stats["comparison_max_abs_error"] = max(self._stats["comparison_max_abs_error"], max_error)
                self._metadata["last_comparison_max_abs_error"] = max_error


def _chunks(tensor: torch.Tensor) -> Iterator[torch.Tensor]:
    """C-order chunks; even noncontiguous inputs never require a volume copy."""
    if tensor.numel() <= _SIGNATURE_CHUNK_VOXELS:
        yield tensor
    else:
        for index in range(int(tensor.shape[0])):
            yield from _chunks(tensor[index])


def _input_signature(pred, target, spacing_dhw) -> bytes | None:
    digest = hashlib.sha256(b"bone-native-binary-metrics-v3\0")
    digest.update(struct.pack("!I", pred.ndim))
    for dimension in pred.shape:
        digest.update(struct.pack("!Q", dimension))
    # Native float32 volume reductions can vary across layouts, devices and
    # reduction thread settings for huge volumes.  Dtype also matters: .float()
    # preserves expanded float32 strides but materializes converted float64.
    digest.update(struct.pack("!Q", torch.get_num_threads()))
    for tensor in (pred, target):
        dtype = str(tensor.dtype).encode("ascii")
        digest.update(struct.pack("!I", len(dtype)))
        digest.update(dtype)
        device = str(tensor.device).encode("ascii")
        digest.update(struct.pack("!I", len(device)))
        digest.update(device)
        for stride in tensor.stride():
            digest.update(struct.pack("!q", stride))
    digest.update(b"none" if spacing_dhw is None else b"spacing" + struct.pack("!ddd", *(float(x) for x in spacing_dhw)))
    for chunk in _chunks(target.detach()):
        normalized = chunk.float()
        if not bool(torch.all((normalized == 0) | (normalized == 1))):
            return None
        digest.update(np.packbits(normalized.bool().cpu().numpy(), axis=None).tobytes())
    digest.update(b"prediction\0")
    for chunk in _chunks(pred.detach()):
        normalized = chunk.float() > .5
        digest.update(np.packbits(normalized.cpu().numpy(), axis=None).tobytes())
    return digest.digest()


def _clone(function, namespace):
    result = FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


class _NativeTorch:
    """Count actual native cdist calls without mutating the global torch module."""

    def __init__(self, tracker):
        self._tracker = tracker

    def __getattr__(self, name):
        return getattr(torch, name)

    def cdist(self, *args, **kwargs):
        self._tracker._add(native_cdist_calls=1)
        started = time.perf_counter()
        try:
            return torch.cdist(*args, **kwargs)
        finally:
            self._tracker._add(native_cdist_seconds=time.perf_counter() - started)


def _original_binary(native_module, tracker):
    namespace = dict(native_module.binary_prompt_metrics_from_masks.__globals__)
    namespace["torch"] = _NativeTorch(tracker)
    namespace["nsd_score"] = _clone(native_module.nsd_score, namespace)
    namespace["mean_slice_nsd_scores"] = _clone(native_module.mean_slice_nsd_scores, namespace)
    return _clone(native_module.binary_prompt_metrics_from_masks, namespace)


def _slice_scores(pred, target, spacing_hw, surface_voxels, tracker):
    # This is native nsd_score with its distance work shared across tolerances.
    with torch.no_grad():
        pred = pred.bool()
        target = target.bool()
        if pred.sum() == 0 and target.sum() == 0:
            return None  # native mean_slice_nsd_scores skips these slices
        tracker._add(fast_nsd_slices=1)
        if pred.sum() == 0 or target.sum() == 0:
            return (0.0, 0.0, 0.0)
        pred_surface = surface_voxels(pred)
        target_surface = surface_voxels(target)
        if pred_surface.sum() == 0 or target_surface.sum() == 0:
            return (0.0, 0.0, 0.0)
        pred_points = torch.nonzero(pred_surface, as_tuple=False).float()
        target_points = torch.nonzero(target_surface, as_tuple=False).float()
        if pred_points.numel() == 0 or target_points.numel() == 0:
            return (0.0, 0.0, 0.0)
        if spacing_hw is not None:
            spacing_t = torch.as_tensor(spacing_hw, dtype=pred_points.dtype, device=pred_points.device)
            if spacing_t.numel() != pred_points.shape[1]:
                raise ValueError(f"spacing length {int(spacing_t.numel())} must match mask ndim {pred_points.shape[1]}")
            pred_points = pred_points * spacing_t
            target_points = target_points * spacing_t
        tracker._add(fast_cdist_calls=1)
        started = time.perf_counter()
        try:
            distances = torch.cdist(pred_points, target_points)
        finally:
            tracker._add(fast_cdist_seconds=time.perf_counter() - started)
        pred_distances = distances.min(dim=1).values
        target_distances = distances.min(dim=0).values
        output = []
        for tolerance in _TOLERANCES:
            pred_close = (pred_distances <= tolerance).float().mean()
            target_close = (target_distances <= tolerance).float().mean()
            output.append(float(.5 * (pred_close + target_close)))
        return tuple(output)


def _ordered_scores(pairs, spacing_hw, surface_voxels, tracker, workers):
    if workers == 1:
        for pred, target in pairs:
            yield _slice_scores(pred, target, spacing_hw, surface_voxels, tracker)
        return
    # Executor.map submits the whole input eagerly on older Python.  Explicitly
    # bound both the jobs in flight and the completed, ordered results to two.
    iterator = iter(pairs)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="bone-nsd") as executor:
        pending = deque()
        for _ in range(workers):
            pair = next(iterator, None)
            if pair is not None:
                pending.append(executor.submit(_slice_scores, *pair, spacing_hw, surface_voxels, tracker))
        while pending:
            yield pending.popleft().result()
            pair = next(iterator, None)
            if pair is not None:
                pending.append(executor.submit(_slice_scores, *pair, spacing_hw, surface_voxels, tracker))


def install_provider(
    native_module: ModuleType,
    compare_native: bool | Callable[[], bool] = False,
    workers: int = 2,
    cache_entries: int = 128,
):
    """Return ``(metric_callable, tracker)``; leave all module exports untouched."""
    if workers not in (1, 2):
        raise ValueError("workers must be 1 or 2")
    if not isinstance(cache_entries, int) or cache_entries < 0:
        raise ValueError("cache_entries must be a nonnegative integer")
    tracker = MetricTracker(native_module, workers, cache_entries)
    original_binary = _original_binary(native_module, tracker)
    native_surface = native_module.surface_voxels
    native_pairs = native_module.axial_slice_pairs

    def mean_slice_nsd_scores(pred, target, spacing_dhw=None):
        started = time.perf_counter()
        try:
            pairs = native_pairs(pred, target)
            spacing_hw = None
            if spacing_dhw is not None:
                if len(spacing_dhw) != 3:
                    raise ValueError(f"spacing_dhw must have length 3, got {len(spacing_dhw)}")
                spacing_hw = (float(spacing_dhw[1]), float(spacing_dhw[2]))
            scores = [[], [], []]
            for values in _ordered_scores(pairs, spacing_hw, native_surface, tracker, workers):
                if values is not None:
                    for index, value in enumerate(values):
                        scores[index].append(value)
            return {name: float(np.mean(values)) if values else float("nan") for name, values in zip(_NSD_NAMES, scores)}
        finally:
            tracker._add(fast_nsd_seconds=time.perf_counter() - started)

    # Preserve native float32 volume reductions and every other native formula.
    fast_globals = dict(native_module.binary_prompt_metrics_from_masks.__globals__)
    fast_globals["mean_slice_nsd_scores"] = mean_slice_nsd_scores
    fast_binary = _clone(native_module.binary_prompt_metrics_from_masks, fast_globals)

    @functools.wraps(native_module.binary_prompt_metrics_from_masks)
    def metric(pred, target, spacing_dhw=None):
        total_started = time.perf_counter()
        tracker._add(calls=1)
        tracker._last(last_input_sha256=None, last_cache_hit=False, last_native_compared=False, last_comparison_max_abs_error=None)
        try:
            if pred.shape != target.shape:
                raise ValueError(f"pred shape {tuple(pred.shape)} must match target shape {tuple(target.shape)}")
            if spacing_dhw is not None and len(spacing_dhw) != 3:
                raise ValueError(f"spacing_dhw must have length 3, got {len(spacing_dhw)}")
            signature_started = time.perf_counter()
            try:
                key = _input_signature(pred, target, spacing_dhw) if cache_entries else None
            finally:
                tracker._add(signature_seconds=time.perf_counter() - signature_started)
            actual = None
            if key is None:
                tracker._add(cache_bypasses=1)
            else:
                actual = tracker._get(key)
                tracker._last(last_input_sha256=key.hex(), last_cache_hit=actual is not None)
            if actual is None:
                tracker._add(fast_calls=1)
                started = time.perf_counter()
                try:
                    actual = fast_binary(pred, target, spacing_dhw=spacing_dhw)
                finally:
                    tracker._add(fast_metric_seconds=time.perf_counter() - started)
                if key is not None:
                    tracker._put(key, actual)
            comparison_enabled = compare_native() if callable(compare_native) else compare_native
            if comparison_enabled:
                tracker._add(native_calls=1)
                tracker._last(last_native_compared=True)
                started = time.perf_counter()
                try:
                    expected = original_binary(pred, target, spacing_dhw=spacing_dhw)
                finally:
                    tracker._add(native_metric_seconds=time.perf_counter() - started)
                tracker._compare(actual, expected)
            return dict(actual)
        finally:
            tracker._add(total_metric_seconds=time.perf_counter() - total_started)

    return metric, tracker
