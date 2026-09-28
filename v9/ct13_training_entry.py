"""Explicit CT13 entry; historical scripts and live processes stay unchanged."""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
from ct_thirteen_source_protocol import SOURCE_ORDER


def _ratios(split):
    if tuple(split.get('source_order', ())) != SOURCE_ORDER:
        raise ValueError('CT13 requires the audited thirteen-source manifest')
    values = {str(k): float(v) for k, v in split['sampling_ratios'].items()}
    if set(values) != set(SOURCE_ORDER) or abs(sum(values.values())-1.) > 1e-8 or min(values.values()) <= 0:
        raise ValueError('invalid CT13 manifest sampling ratios')
    return values


def install_v9_protocol(precision, shared_protocol, split):
    ratios = _ratios(split)
    precision.SOURCE_ORDER = SOURCE_ORDER
    shared_protocol.SOURCE_ORDER = SOURCE_ORDER
    precision.validate_source_ratios = lambda _supplied: dict(ratios)


def install_v10_protocol(joint, data, split):
    ratios = _ratios(split)
    joint.EIGHT_SOURCE_ORDER = SOURCE_ORDER
    data.EIGHT_SOURCE_ORDER = SOURCE_ORDER
    joint.extended_source_ratios = lambda _args: dict(ratios)


def _split_path():
    option = '--multidataset-split-json'
    if option not in sys.argv:
        raise ValueError('CT13 requires an explicit --multidataset-split-json')
    return Path(sys.argv[sys.argv.index(option)+1])


def main():
    family = os.environ.get('CT13_FAMILY', '')
    if family not in ('v9_2', 'v10_2'):
        raise ValueError('set CT13_FAMILY=v9_2 or v10_2 explicitly')
    root = Path(__file__).resolve().parent
    split_path = _split_path()
    split = json.loads(split_path.read_text())
    _ratios(split)
    if split.get('protocol_revision') != f'ct13_{family}_20260928':
        raise ValueError('manifest family/revision mismatch')
    import v9_2_extended_protocol as protocol
    protocol.SOURCE_ORDER = SOURCE_ORDER
    if family == 'v9_2':
        import finetune_multisource_sam2_v9_2_full as module
        install_v9_protocol(module.ddp.legacy, protocol, split)
        from v9_family_shared_cache_entry import install_multidataset_factories
        install_multidataset_factories(module.multi)
    else:
        sys.path.insert(0, str(root/'v10'))
        import finetune_totalseg_amos_magic_v10_2_joint as module
        import v10_2_data as data
        install_v10_protocol(module, data, split)
    module.main()


if __name__ == '__main__':
    main()
