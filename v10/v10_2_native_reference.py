"""Scoped JSON references to native CT pairs, with bounded in-memory loading."""
from __future__ import annotations
import json
import threading
from pathlib import Path
from v10_2_sixteen_source_protocol import native_signature


class NativeReferenceLoader:
    def __init__(self, reference_root, native_loader=None):
        self.root = Path(reference_root).resolve()
        self.current = self.pair = self.original = None
        self.lock = threading.RLock()
        if native_loader is None:
            from four_source_data import load_pair
            native_loader = load_pair
        self.native_loader = native_loader

    def record(self, path):
        p = Path(path).resolve()
        if self.root not in p.parents:
            return None
        value = json.loads(p.read_text(encoding='utf-8'))
        if not value.pop('native_training_reference_v1', False):
            raise ValueError('native reference marker missing')
        if p.name not in ('image.nii.gz','label.nii.gz'):
            raise ValueError('unknown native reference role')
        return value

    def load(self, path, *args, **kwargs):
        record = self.record(path)
        if record is None:
            return self.original(path, *args, **kwargs)
        identity = record['source_name'], record['case_id']
        with self.lock:
            if identity != self.current:
                self.pair = self.native_loader(record)
                self.current = identity
            result = self.pair[0 if Path(path).name == 'image.nii.gz' else 1]
        return result

    def install(self):
        import nibabel as nib
        import v10_data
        self.original = nib.load
        original_signature = v10_data.SAM2CachedAMOSDataset._source_signature
        def signature(path):
            record = self.record(path)
            return (original_signature(path) if record is None else
                    native_signature(record, image=Path(path).name == 'image.nii.gz'))
        nib.load = self.load
        v10_data.SAM2CachedAMOSDataset._source_signature = staticmethod(signature)
        return self
