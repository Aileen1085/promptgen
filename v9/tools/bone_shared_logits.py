"""Content-bound, CPU-only memoization for shared raw-logit computations.

The caller must bind the model/checkpoint and every setting in ``namespace``
or ``inputs``. Only a threshold floor applied *after* the shared computation
may be excluded. No tensors, features, or CT data are written to disk.

The byte budget counts cloned tensor/array payloads, not Python container or
digest-record overhead. Calls are sequential; this is not a concurrent cache.
"""

from collections import OrderedDict
import hashlib
import struct

import numpy as np
import torch


_PRIMITIVES = (type(None), bool, int, float, complex, str, bytes)


def _check_leaf(value):
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided or value.is_quantized or value.device.type == "meta":
            raise TypeError("Only materialized, strided, non-quantized tensors are supported")
        return
    if isinstance(value, (np.ndarray, np.generic)):
        if type(value) is not np.ndarray and not isinstance(value, np.generic):
            raise TypeError("NumPy array subclasses are unsupported")
        if value.dtype.hasobject or value.dtype.metadata:
            raise TypeError("Object arrays and dtypes with metadata are unsupported")
        return
    if type(value) not in _PRIMITIVES:
        raise TypeError("Unsupported memo value type: " + type(value).__qualname__)


def _walk(value, leaf, active=None):
    """Map supported trees, detecting cycles and never retaining source leaves."""
    active = set() if active is None else active
    if type(value) not in (dict, list, tuple):
        _check_leaf(value)
        return leaf(value)
    identity = id(value)
    if identity in active:
        raise ValueError("Recursive containers are unsupported")
    active.add(identity)
    try:
        if type(value) is dict:
            return {_walk(key, leaf, active): _walk(item, leaf, active)
                    for key, item in value.items()}
        items = [_walk(item, leaf, active) for item in value]
        return tuple(items) if type(value) is tuple else items
    finally:
        active.remove(identity)


def _cpu_tensor(value):
    # Resolve logical conjugate/negative views before copying raw storage.
    return value.detach().resolve_conj().resolve_neg().to(
        device="cpu", copy=True, memory_format=torch.contiguous_format)


def cpu_clone(value):
    """Return a detached CPU clone of every tensor/array and mutable container."""
    def clone(leaf):
        if isinstance(leaf, torch.Tensor):
            return _cpu_tensor(leaf)
        if isinstance(leaf, np.ndarray):
            return np.array(leaf, copy=True, order="C", subok=False)
        if isinstance(leaf, np.generic):
            return leaf.copy()
        return leaf

    return _walk(value, clone)


def to_device(value, device):
    """Clone a tree, placing tensors on ``device`` and leaving arrays as arrays.

    This explicitly restores hit tensors for a GPU consumer. Even when the
    requested device is already the source device, no mutable storage is shared.
    """
    def clone(leaf):
        if isinstance(leaf, torch.Tensor):
            return leaf.detach().resolve_conj().resolve_neg().to(
                device=device, copy=True, memory_format=torch.contiguous_format)
        if isinstance(leaf, np.ndarray):
            return np.array(leaf, copy=True, order="C", subok=False)
        if isinstance(leaf, np.generic):
            return leaf.copy()
        return leaf

    return _walk(value, clone)


def tensor_bytes(value):
    """Count payload bytes of the independent CPU snapshot (including aliases).

    A repeated leaf is cloned separately at each occurrence, so each occurrence
    contributes bytes. Python primitive/container overhead is not included.
    """
    total = 0

    def count(leaf):
        nonlocal total
        if isinstance(leaf, torch.Tensor):
            total += leaf.numel() * leaf.element_size()
        elif isinstance(leaf, (np.ndarray, np.generic)):
            total += leaf.nbytes
        return None

    _walk(value, count)
    return total


def _framed(hasher, value):
    hasher.update(struct.pack("!Q", len(value)))
    hasher.update(value)


def tensor_digest(value):
    """SHA-256 of typed values, shapes, dtypes and contiguous logical raw bytes.

    Dictionary order, tensor device, strides and autograd state do not affect
    identity. Dict/list/tuple types and NumPy/Torch types remain distinct.
    Unsupported objects fail explicitly rather than relying on repr/pickle.
    bfloat16 is viewed as uint8 before NumPy access, with no numeric conversion.
    """
    active = set()

    def digest(item):
        hasher = hashlib.sha256()
        kind = type(item)
        if kind in (dict, list, tuple):
            identity = id(item)
            if identity in active:
                raise ValueError("Recursive containers are unsupported")
            active.add(identity)
            try:
                _framed(hasher, kind.__name__.encode("ascii"))
                _framed(hasher, str(len(item)).encode("ascii"))
                if kind is dict:
                    for key_digest, value_digest in sorted((digest(key), digest(val))
                                                           for key, val in item.items()):
                        _framed(hasher, key_digest)
                        _framed(hasher, value_digest)
                else:
                    for child in item:
                        _framed(hasher, digest(child))
            finally:
                active.remove(identity)
            return hasher.digest()

        _check_leaf(item)
        if isinstance(item, torch.Tensor):
            _framed(hasher, b"torch.Tensor")
            _framed(hasher, str(item.dtype).encode("ascii"))
            _framed(hasher, digest(tuple(item.shape)))
            raw = item.detach().resolve_conj().resolve_neg().to(
                device="cpu", memory_format=torch.contiguous_format)
            # Flatten first so scalar tensors can also be reinterpreted as bytes.
            _framed(hasher, memoryview(raw.reshape(-1).view(torch.uint8).numpy()))
        elif isinstance(item, (np.ndarray, np.generic)):
            _framed(hasher, b"numpy.ndarray" if isinstance(item, np.ndarray) else b"numpy.scalar")
            _framed(hasher, digest(item.dtype.descr if item.dtype.fields else item.dtype.str))
            _framed(hasher, digest(tuple(item.shape)))
            raw = np.ascontiguousarray(item).reshape(-1).view(np.uint8)
            _framed(hasher, memoryview(raw))
        else:
            _framed(hasher, kind.__name__.encode("ascii"))
            if kind is str:
                _framed(hasher, item.encode("utf-8"))
            elif kind is bytes:
                _framed(hasher, item)
            elif kind is float:
                _framed(hasher, struct.pack("!d", item))
            elif kind is complex:
                _framed(hasher, struct.pack("!dd", item.real, item.imag))
            elif kind is not type(None):
                _framed(hasher, str(item).encode("ascii"))
        return hasher.digest()

    return digest(value).hex()


class TensorMemo:
    """LRU cache whose retained feature trees contain detached CPU copies only.

    A miss returns the producer's original result. A hit returns a fresh CPU
    clone; use :func:`to_device` when the native consumer needs GPU tensors.
    ``records`` exposes digest-only producer/hit audit records. ``clear`` releases
    payloads but preserves counters and records for case-level audit summaries.
    """

    def __init__(self, max_bytes):
        if type(max_bytes) is not int:
            raise TypeError("max_bytes must be an integer")
        if max_bytes < 0:
            raise ValueError("max_bytes must be nonnegative")
        self.max_bytes = max_bytes
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self._bytes = 0
        self._entries = OrderedDict()
        self._records = []

    @property
    def bytes(self):
        return self._bytes

    @property
    def stats(self):
        return {"hits": self.hits, "misses": self.misses,
                "evictions": self.evictions, "bytes": self._bytes}

    @property
    def records(self):
        return [record.copy() for record in self._records]

    def clear(self):
        self._entries.clear()
        self._bytes = 0

    def call(self, namespace, inputs, compute):
        namespace_digest = tensor_digest(namespace)
        inputs_digest = tensor_digest(inputs)
        key = hashlib.sha256(b"TensorMemo-v1\0" + bytes.fromhex(namespace_digest)
                             + bytes.fromhex(inputs_digest)).hexdigest()
        record = {"key": key, "namespace_digest": namespace_digest,
                  "inputs_digest": inputs_digest}
        if key in self._entries:
            self.hits += 1
            snapshot, size, output_digest = self._entries[key]
            self._entries.move_to_end(key)
            result = cpu_clone(snapshot)
            self._records.append({**record, "event": "hit", "cached": True,
                                  "output_digest": output_digest})
            return result

        self.misses += 1
        result = compute()
        size = tensor_bytes(result)
        cached = self.max_bytes > 0 and size <= self.max_bytes
        if cached:
            snapshot = cpu_clone(result)
            output_digest = tensor_digest(snapshot)
            while self._bytes + size > self.max_bytes:
                _, (_, evicted_size, _) = self._entries.popitem(last=False)
                self._bytes -= evicted_size
                self.evictions += 1
            self._entries[key] = (snapshot, size, output_digest)
            self._bytes += size
        else:
            output_digest = tensor_digest(result)
        self._records.append({**record, "event": "producer", "cached": cached,
                              "output_digest": output_digest})
        return result
