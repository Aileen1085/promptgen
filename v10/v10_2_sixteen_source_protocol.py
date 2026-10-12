"""Explicit CT13 rehearsal plus three native training sources; tests stay held out."""
from __future__ import annotations
import hashlib
import json

TRAIN_SOURCES = ('graz', 'pengwin', 'adrenal')
EXPECTED_TRAIN = dict(graz=20, pengwin=320, adrenal=33)
OLD_COUNTS = dict(totalseg=40, amos=18, magic=32, msd_task07=10, msd_task10=8,
    msd_task08=22, parse2022=5, topcow2024_cta=25, lndb=8, msd_task06=6,
    covid19_20=8, kits23=12, lnq2023_lite=6)
SOURCE_ORDER = (*OLD_COUNTS, *TRAIN_SOURCES)


def source_counts():
    return dict(OLD_COUNTS, graz=10, pengwin=30, adrenal=10)


def validate_external_split(value):
    datasets = value['datasets']
    f = datasets['fumpe']
    if f['train'] or f.get('val') or f.get('training_enabled') or len(set(f['test'])) != 30:
        raise ValueError('FUMPE must remain test-only with its original30 cases')
    for source, count in EXPECTED_TRAIN.items():
        row = datasets[source]
        train, test = set(row['train']), set(row['test'])
        if train & test:
            raise ValueError('train/test overlap: '+source)
        if (len(train) != count or len(test) != 20 or len(row['train']) != count
                or len(row['test']) != 20 or row.get('val') or not row['training_enabled']):
            raise ValueError('incorrect frozen partition: '+source)
    return dict(EXPECTED_TRAIN)


def native_signature(record, *, image):
    signatures = record['source_signatures']
    if image and record['source_name'] == 'adrenal':
        signatures = [r for r in signatures if r['path'] != record['seg_path']]
    elif image and record['source_name'] == 'graz':
        signatures = signatures[:1]
    identity = dict(version='native_ras_reader_v1', source=record['source_name'],
        case=record['case_id'], parquet_row=record.get('row_index'),
        signatures=sorted((r['path'], int(r['mtime_ns']), int(r['size_bytes'])) for r in signatures))
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    # Match the inherited (path,mtime,size) shape without reference-file timestamps.
    return 'native:'+digest, 0, sum(int(r['size_bytes']) for r in signatures)
