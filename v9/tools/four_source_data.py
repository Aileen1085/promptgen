"""Read four native public CT sources into a verified canonical RAS XYZ grid.

No reader creates CT caches or resamples a label. Discovery records contain only
paths, IDs and parquet row references; raw HU values are preserved.
"""
from __future__ import annotations

import gzip
import io
import random
import re
from pathlib import Path

import numpy as np

SOURCES = ('fumpe', 'graz', 'pengwin', 'adrenal')
GEOMETRY_TOLERANCE_MM = 1e-4


def select_cases(records, count=30, seed=20261011):
    rows = sorted(records, key=lambda r: str(r['case_id']))
    if len({str(r['case_id']) for r in rows}) != len(rows):
        raise ValueError('duplicate case IDs')
    if count <= 0 or len(rows) < count:
        raise ValueError(f'insufficient cases: requested {count}, found {len(rows)}')
    return random.Random(seed).sample(rows, count)


def _integer_labels(values, maximum):
    a = np.asarray(values)
    if not np.all(np.isfinite(a)) or np.any(a != np.rint(a)) or np.any(a < 0) or np.any(a > maximum):
        raise ValueError(f'invalid label values; require integers in 0..{maximum}')
    return a.astype(np.uint8)


def merge_pengwin_labels(values):
    a = _integer_labels(values, 200)
    return np.where(a > 0, (a.astype(np.int16) - 1) // 50 + 1, 0).astype(np.uint8)


def _grid_corners(shape, affine):
    from itertools import product
    corners = np.array(list(product(*[(0, n - 1) for n in shape])), dtype=np.float64)
    return corners @ affine[:3, :3].T + affine[:3, 3]


def _original_geometry(image):
    import nibabel as nib
    affine = np.asarray(image.affine, dtype=np.float64)
    if (affine.shape != (4, 4) or not np.all(np.isfinite(affine)) or
            not np.allclose(affine[3], [0, 0, 0, 1], rtol=0, atol=1e-12) or
            abs(np.linalg.det(affine[:3, :3])) < 1e-12 or
            np.any(nib.affines.voxel_sizes(affine) <= 0)):
        raise ValueError('nonfinite or singular image affine')
    qform, qcode = image.get_qform(coded=True)
    sform, scode = image.get_sform(coded=True)
    disagreement = bool(qcode and scode and not np.allclose(
        _grid_corners(image.shape, qform), _grid_corners(image.shape, sform),
        rtol=0, atol=GEOMETRY_TOLERANCE_MM))
    transform = nib.orientations.ornt_transform(nib.orientations.io_orientation(affine),
                                                nib.orientations.axcodes2ornt(('R', 'A', 'S')))
    return dict(axis_codes=list(nib.aff2axcodes(affine)), affine=affine.tolist(),
                orientation_transform=transform.tolist(),
                header=dict(qform_code=int(qcode), sform_code=int(scode),
                            qform=None if qform is None else qform.tolist(),
                            sform=None if sform is None else sform.tolist(),
                            qform_sform_disagreement=disagreement))


def source_signatures(record):
    """Lightweight source identity for manifests and cache invalidation."""
    if record['source_name'] == 'adrenal':
        paths = list(record['ct_paths']) + [record['seg_path']]
    elif record['source_name'] == 'pengwin':
        paths = [record['parquet_path']]
    else:
        paths = [record['image_path'], record['label_path']]
    signatures = []
    for path in paths:
        stat = Path(path).stat()
        signatures.append(dict(path=str(path), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns))
    return signatures


def _canonical_pair(image, label, source_name, audit=None):
    import nibabel as nib
    if len(image.shape) != 3 or len(label.shape) != 3:
        raise ValueError('CT and label must both be three dimensional')
    original_image, original_label = _original_geometry(image), _original_geometry(label)
    image = nib.as_closest_canonical(image)
    label = nib.as_closest_canonical(label)
    if image.shape != label.shape or not np.allclose(
            _grid_corners(image.shape, image.affine), _grid_corners(label.shape, label.affine),
            rtol=0, atol=GEOMETRY_TOLERANCE_MM):
        raise ValueError('CT/label physical grid geometry mismatch; resampling is prohibited')
    ct = np.asarray(image.dataobj, dtype=np.float32)
    if not np.all(np.isfinite(ct)):
        raise ValueError('nonfinite CT values')
    raw = np.asarray(label.dataobj)
    labels = merge_pengwin_labels(raw) if source_name == 'pengwin' else _integer_labels(raw, 2 if source_name == 'graz' else 1)
    image = nib.Nifti1Image(ct, image.affine)
    label = nib.Nifti1Image(labels, image.affine)
    result = dict(audit or {})
    for name, original in [('image', original_image), ('label', original_label)]:
        result[f'original_{name}_axis_codes'] = original['axis_codes']
        result[f'original_{name}_affine'] = original['affine']
        result[f'original_{name}_header'] = original['header']
        result[f'{name}_to_ras_orientation_transform'] = original['orientation_transform']
    result.update(grid_verified=True, canonical_axis_codes=list(nib.aff2axcodes(image.affine)),
                  array_order='XYZ', ct_normalization='none', label_values=np.unique(labels).tolist(),
                  shape_xyz=list(image.shape), spacing_xyz_mm=list(nib.affines.voxel_sizes(image.affine)),
                  affine_ras=image.affine.tolist(), geometry_tolerance_mm=GEOMETRY_TOLERANCE_MM)
    return image, label, result


def _nrrd_image(path):
    import nibabel as nib
    import SimpleITK as sitk
    image = sitk.ReadImage(str(path))
    if image.GetDimension() != 3:
        raise ValueError('NRRD must be a scalar 3D image')
    affine = np.eye(4)
    affine[:3, :3] = np.asarray(image.GetDirection()).reshape(3, 3) @ np.diag(image.GetSpacing())
    affine[:3, 3] = image.GetOrigin()
    affine = np.diag([-1., -1., 1., 1.]) @ affine
    return nib.Nifti1Image(sitk.GetArrayFromImage(image).transpose(2, 1, 0), affine)


def _embedded_nifti(blob):
    import nibabel as nib
    data = bytes(blob)
    if data[:2] == b'\x1f\x8b':
        data = gzip.decompress(data)
    holder = nib.FileHolder(fileobj=io.BytesIO(data))
    # The bytes are a complete single-file NIfTI, not an Analyze pair.
    image = nib.Nifti1Image.from_file_map({'image': holder})
    # Materialize proxy-scaled values once and retain the original forms/codes
    # for orientation audits. A fresh header would silently erase qform data.
    values = np.asanyarray(image.dataobj)
    header = image.header.copy()
    header.set_data_dtype(values.dtype)
    header.set_slope_inter(None, None)
    return nib.Nifti1Image(values, image.affine, header=header)


def _parquet_paths(path):
    import duckdb
    connection = duckdb.connect()
    try:
        return connection.execute('SELECT image.path, label.path FROM read_parquet(?)', [str(path)]).fetchall()
    finally:
        connection.close()


def _parquet_pair(record):
    import duckdb
    connection = duckdb.connect()
    try:
        row = connection.execute(
            'SELECT image.bytes, label.bytes, image.path, label.path FROM read_parquet(?) LIMIT 1 OFFSET ?',
            [record['parquet_path'], int(record['row_index'])]).fetchone()
        if row is None or tuple(row[2:]) != (record['image_name'], record['label_name']):
            raise ValueError('parquet row identity changed')
        return _embedded_nifti(row[0]), _embedded_nifti(row[1])
    finally:
        connection.close()


def _sequence(item, name):
    return list(getattr(item, name, []))


def _referenced_series(seg):
    result = set()
    for entry in _sequence(seg, 'ReferencedSeriesSequence'):
        result.add(str(entry.SeriesInstanceUID))
    for study in _sequence(seg, 'StudiesContainingOtherReferencedInstancesSequence'):
        for entry in _sequence(study, 'ReferencedSeriesSequence'):
            result.add(str(entry.SeriesInstanceUID))
    return result


def _frame_sources(frame):
    return {str(source.ReferencedSOPInstanceUID)
            for derivation in _sequence(frame, 'DerivationImageSequence')
            for source in _sequence(derivation, 'SourceImageSequence')}


def _functional_value(frame, shared, sequence_name, attribute):
    entries = _sequence(frame, sequence_name) or _sequence(shared, sequence_name)
    if len(entries) != 1 or not hasattr(entries[0], attribute):
        raise ValueError(f'missing or ambiguous SEG {sequence_name}.{attribute}')
    return np.asarray(getattr(entries[0], attribute), dtype=np.float64)


def dicom_pair_from_datasets(ct_slices, seg):
    """Place binary SEG frames on their referenced native CT planes exactly."""
    import nibabel as nib
    slices = list(ct_slices)
    if len(slices) < 2:
        raise ValueError('CT needs at least two physical planes')
    series = {str(s.SeriesInstanceUID) for s in slices}
    if len(series) != 1 or (_referenced_series(seg) and _referenced_series(seg) != series):
        raise ValueError('SEG referenced CT series mismatch')
    if {str(s.FrameOfReferenceUID) for s in slices} != {str(seg.FrameOfReferenceUID)}:
        raise ValueError('SEG/CT FrameOfReferenceUID mismatch')
    if str(seg.SegmentationType) != 'BINARY':
        raise ValueError('require BINARY ACC tumor SEG')
    segments = _sequence(seg, 'SegmentSequence')
    if len(segments) != 1:
        raise ValueError('expected exactly one ACC tumor segment')
    descriptor = str(getattr(segments[0], 'SegmentLabel', '')).lower()
    if not any(word in descriptor for word in ('mass', 'tumor', 'tumour', 'acc', 'carcinoma')):
        raise ValueError(f'unrecognized ACC segment label: {descriptor}')
    segment_number = int(segments[0].SegmentNumber)
    orientation = np.asarray(slices[0].ImageOrientationPatient, dtype=np.float64)
    spacing = np.asarray(slices[0].PixelSpacing, dtype=np.float64)
    if orientation.shape != (6,) or spacing.shape != (2,) or np.any(spacing <= 0):
        raise ValueError('invalid CT orientation/pixel spacing')
    x, y = orientation[:3], orientation[3:]
    if not np.allclose([np.dot(x, x), np.dot(y, y), np.dot(x, y)], [1, 1, 0], rtol=0, atol=1e-6):
        raise ValueError('invalid CT direction cosines')
    normal = np.cross(x, y)
    for s in slices:
        if not np.allclose(s.ImageOrientationPatient, orientation, rtol=0, atol=1e-6) or not np.allclose(s.PixelSpacing, spacing, rtol=0, atol=GEOMETRY_TOLERANCE_MM):
            raise ValueError('CT slice orientation/spacing mismatch')
        if (int(s.Rows), int(s.Columns)) != (int(slices[0].Rows), int(slices[0].Columns)):
            raise ValueError('CT slice dimensions mismatch')
    slices.sort(key=lambda s: float(np.dot(s.ImagePositionPatient, normal)))
    positions = np.array([s.ImagePositionPatient for s in slices], dtype=np.float64)
    steps = np.diff(positions, axis=0)
    if np.dot(steps[0], normal) <= GEOMETRY_TOLERANCE_MM or not np.allclose(steps, steps[0], rtol=0, atol=GEOMETRY_TOLERANCE_MM):
        raise ValueError('duplicate or nonuniform CT slice positions')
    # Reject in-plane drifting planes; no affine approximation/resampling.
    if not np.allclose(steps[0], normal * np.dot(steps[0], normal), rtol=0, atol=GEOMETRY_TOLERANCE_MM):
        raise ValueError('CT plane origin drift')
    uids = [str(s.SOPInstanceUID) for s in slices]
    if len(set(uids)) != len(uids):
        raise ValueError('duplicate CT SOP UID')
    uid_to_z = {uid: z for z, uid in enumerate(uids)}
    ct = np.stack([np.asarray(s.pixel_array, dtype=np.float32).T * float(getattr(s, 'RescaleSlope', 1)) +
                   float(getattr(s, 'RescaleIntercept', 0)) for s in slices], axis=2)
    mask = np.zeros(ct.shape, dtype=np.uint8)
    shared_items = _sequence(seg, 'SharedFunctionalGroupsSequence')
    if len(shared_items) != 1:
        raise ValueError('missing shared SEG functional groups')
    shared = shared_items[0]
    frames = _sequence(seg, 'PerFrameFunctionalGroupsSequence')
    pixels = np.asarray(seg.pixel_array)
    if pixels.ndim == 2:
        pixels = pixels[None]
    if pixels.shape != (len(frames), int(slices[0].Rows), int(slices[0].Columns)):
        raise ValueError('SEG frame dimensions mismatch')
    _integer_labels(pixels, 1)
    placed = set()
    for frame, plane in zip(frames, pixels):
        if not np.allclose(_functional_value(frame, shared, 'PlaneOrientationSequence', 'ImageOrientationPatient'), orientation, rtol=0, atol=1e-6):
            raise ValueError('SEG/CT orientation mismatch')
        if not np.allclose(_functional_value(frame, shared, 'PixelMeasuresSequence', 'PixelSpacing'), spacing, rtol=0, atol=GEOMETRY_TOLERANCE_MM):
            raise ValueError('SEG/CT pixel spacing mismatch')
        identified = _sequence(frame, 'SegmentIdentificationSequence')
        if len(identified) != 1 or int(identified[0].ReferencedSegmentNumber) != segment_number:
            raise ValueError('SEG segment number mismatch')
        sources = _frame_sources(frame)
        if len(sources) != 1 or not sources.issubset(uid_to_z):
            raise ValueError('SEG referenced SOP missing or ambiguous')
        z = uid_to_z[next(iter(sources))]
        position = _functional_value(frame, shared, 'PlanePositionSequence', 'ImagePositionPatient')
        if not np.allclose(position, positions[z], rtol=0, atol=GEOMETRY_TOLERANCE_MM):
            raise ValueError('SEG/CT physical plane mismatch')
        if z in placed:
            raise ValueError('duplicate SEG frame for CT plane')
        placed.add(z); mask[:, :, z] = plane.T
    affine = np.eye(4)
    affine[:3, :3] = np.column_stack([x * spacing[1], y * spacing[0], steps[0]])
    affine[:3, 3] = positions[0]
    affine = np.diag([-1., -1., 1., 1.]) @ affine
    return _canonical_pair(nib.Nifti1Image(ct, affine), nib.Nifti1Image(mask, affine), 'adrenal',
                           dict(seg_frames_placed=len(placed), ct_planes=len(slices),
                                referenced_series_uid=next(iter(series)), frame_of_reference_uid=str(seg.FrameOfReferenceUID),
                                segment_label=str(segments[0].SegmentLabel)))


def _discover_adrenal(root):
    import pydicom
    rows = []
    for patient in sorted((root / 'segmentations').iterdir()):
        if not patient.is_dir():
            continue
        seg_paths = sorted(patient.rglob('*.dcm'))
        if len(seg_paths) != 1:
            raise ValueError(f'{patient.name}: expected one DICOM SEG, found {len(seg_paths)}')
        seg = pydicom.dcmread(seg_paths[0], stop_before_pixels=True)
        referenced = _referenced_series(seg)
        referenced_sops = set().union(*(_frame_sources(f) for f in _sequence(seg, 'PerFrameFunctionalGroupsSequence')))
        candidates = {}
        for path in sorted((root / 'images' / patient.name).rglob('*.dcm')):
            header = pydicom.dcmread(path, stop_before_pixels=True)
            if str(getattr(header, 'Modality', '')) != 'CT':
                continue
            uid = str(header.SeriesInstanceUID)
            candidates.setdefault(uid, []).append((str(path), str(header.SOPInstanceUID)))
        matched = [uid for uid, group in candidates.items()
                   if (not referenced or uid in referenced) and referenced_sops.issubset({sop for _, sop in group})]
        if len(matched) != 1:
            raise ValueError(f'{patient.name}: SEG referenced CT series missing or ambiguous')
        rows.append(dict(source_name='adrenal', case_id=patient.name,
                         ct_paths=[path for path, _ in candidates[matched[0]]],
                         seg_path=str(seg_paths[0]), ct_series_uid=matched[0]))
    return rows


def discover(data_root):
    root = Path(data_root)
    found = {source: [] for source in SOURCES}
    for image in sorted((root / 'FUMPE' / 'images').glob('*.nii.gz')):
        label = root / 'FUMPE' / 'masks' / image.name
        if not label.is_file():
            raise ValueError(f'missing FUMPE mask: {label}')
        found['fumpe'].append(dict(source_name='fumpe', case_id=image.name.removesuffix('.nii.gz'),
                                  image_path=str(image), label_path=str(label)))
    for case in sorted((root / 'Aortic-Dissection-Graz' / 'dataset').glob('cta*s')):
        match = re.fullmatch(r'cta(\d+)s', case.name)
        if not match:
            continue
        image = case / f'{case.name}.nrrd'; label = case / f'mask{match[1]}.nrrd'
        if not image.is_file() or not label.is_file():
            raise ValueError(f'missing Graz image/mask pair: {case}')
        found['graz'].append(dict(source_name='graz', case_id=case.name, image_path=str(image), label_path=str(label)))
    for shard in sorted((root / 'PENGWIN2026' / 'data').glob('train-*-of-00020.parquet')):
        for index, (image_name, label_name) in enumerate(_parquet_paths(shard)):
            case_id = Path(image_name).name.removesuffix('_image.nii.gz')
            if Path(label_name).name != f'{case_id}_label.nii.gz':
                raise ValueError(f'PENGWIN image/label identity mismatch: {image_name}, {label_name}')
            found['pengwin'].append(dict(source_name='pengwin', case_id=case_id, parquet_path=str(shard),
                                        row_index=index, image_name=image_name, label_name=label_name))
    adrenal = root / 'Adrenal-ACC-Ki67-Seg'
    if (adrenal / 'segmentations').is_dir():
        found['adrenal'] = _discover_adrenal(adrenal)
    for source, records in found.items():
        if len({record['case_id'] for record in records}) != len(records):
            raise ValueError(f'duplicate {source} case IDs')
        for record in records:
            record['source_signatures'] = source_signatures(record)
    return found


def load_pair(record):
    import nibabel as nib
    source = record['source_name']
    if source not in SOURCES:
        raise ValueError(f'unknown source: {source}')
    signatures = source_signatures(record)
    if record.get('source_signatures', signatures) != signatures:
        raise ValueError('source file signature changed since discovery')
    if source == 'adrenal':
        import pydicom
        image, label, audit = dicom_pair_from_datasets([pydicom.dcmread(path) for path in record['ct_paths']],
                                                      pydicom.dcmread(record['seg_path']))
    elif source == 'pengwin':
        image, label = _parquet_pair(record)
    elif source == 'graz':
        image, label = _nrrd_image(record['image_path']), _nrrd_image(record['label_path'])
    else:
        image, label = nib.load(record['image_path']), nib.load(record['label_path'])
    if source != 'adrenal':
        image, label, audit = _canonical_pair(image, label, source)
    if source_signatures(record) != signatures:
        raise ValueError('source file signature changed during load')
    audit.update(source_name=source, case_id=record['case_id'], source_signatures=signatures)
    if source == 'pengwin':
        audit.update(parquet_row_index=int(record['row_index']), embedded_image_name=record['image_name'],
                     embedded_label_name=record['label_name'])
    return image, label, audit
