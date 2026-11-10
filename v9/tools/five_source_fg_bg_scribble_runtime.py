"""Additive FG/BG scribble runtime with a private native scientific namespace."""
from pathlib import Path
import importlib.util

ROOT = Path(__file__).absolute().parents[1]
_spec = importlib.util.spec_from_file_location('_fg_bg_scribble_private_native', ROOT/'tools/five_source_prompt_type_runtime.py')
_native = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_native)
VARIANTS = ('baseline', 'fg_scribble_bg_scribble')
SOURCES, METRICS = _native.SOURCES, _native.METRICS
sha, key = _native.sha, _native.key

def resource_snapshot(text, gpu, expected_uuid):
    rows = {int(p[0]): p for p in ([v.strip() for v in line.split(',')] for line in text.splitlines() if line.strip())}
    if gpu not in range(8) or gpu not in rows or rows[gpu][1] != expected_uuid:
        raise ValueError('physical GPU identity/UUID mismatch')
    if int(rows[gpu][2]) < 30720:
        raise ValueError('record GPU free-memory requirement failed')
    return dict(gpu=gpu,gpu_uuid=expected_uuid,free_mib=int(rows[gpu][2]),minimum_free_mib=30720)

_native.VARIANTS = VARIANTS
_native.resource_snapshot = resource_snapshot
initialize = _native.initialize
_validate_native = _native.validate_record

def validate_record(record, task, variant, actual_frame_batch, control=False):
    _validate_native(record,task,variant,actual_frame_batch,control=control)
    binding=record['resource_binding']
    minimum=max(30720,int(binding.get('minimum_free_mib',30720)))
    if binding['gpu'] not in range(8) or binding.get('cuda_probe') is not True:
        raise ValueError('physical GPU/probe binding mismatch')
    for name in ('first_snapshot','second_snapshot'):
        snapshot=binding[name]
        if (snapshot.get('gpu'),snapshot.get('gpu_uuid'))!=(binding['gpu'],binding['gpu_uuid']) or snapshot.get('free_mib',0)<minimum:
            raise ValueError('repeated free-memory/UUID binding mismatch')
    if binding['free_mib_before_load']!=binding['second_snapshot']['free_mib']:
        raise ValueError('second free-memory snapshot differs from load binding')
    return True

_native.validate_record = validate_record
validate_restoration = _native.validate_restoration
oom_next_batch = _native.oom_next_batch
