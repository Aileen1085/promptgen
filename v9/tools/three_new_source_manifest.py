"""Freeze external 30-case cohorts and audit headers without exporting voxels."""
import argparse
import json
import random
import re
from pathlib import Path
import numpy as np

FILENAME = 'test30_v102_nninteractive_20261013.json'
RELATIVE = {'mswal': 'MSWAL', 'imagetbad': 'ImageTBAD/dataset/imageTBAD',
            'headcta': 'Head-CTA-Aneurysm'}


def select30(rows):
    rows = sorted(rows, key=lambda r: r['case_id'])
    if len(rows) < 30 or len({r['case_id'] for r in rows}) != len(rows):
        raise ValueError('require at least 30 unique CT cases')
    return sorted(random.Random(20261013).sample(rows, 30), key=lambda r: r['case_id'])


def world_to_index(points, affine):
    inv = np.linalg.inv(affine)
    return np.asarray(points) @ inv[:3, :3].T + inv[:3, 3]


def dicom_grid(rows):
    if len(rows) < 2: raise ValueError('require 3D CT series')
    first = rows[0]; orient = np.asarray(first['orientation'], float)
    if orient.shape != (6,) or not np.all(np.isfinite(orient)):
        raise ValueError('invalid orientation')
    x, y = orient[:3], orient[3:]; normal = np.cross(x, y)
    if not np.allclose([np.linalg.norm(x), np.linalg.norm(y), x @ y], [1, 1, 0], atol=1e-5):
        raise ValueError('invalid direction cosines')
    for r in rows:
        if (r['series'] != first['series'] or r['rows'] != first['rows'] or
                r['columns'] != first['columns'] or
                not np.allclose(r['orientation'], orient, atol=1e-6) or
                not np.allclose(r['pixel_spacing'], first['pixel_spacing'], atol=1e-6)):
            raise ValueError('mixed series/grid')
    positions = np.array([r['position'] for r in rows], float)
    if not np.all(np.isfinite(positions)) or len({r['sop'] for r in rows}) != len(rows):
        raise ValueError('invalid positions/duplicate SOP')
    order = np.argsort(positions @ normal); ordered = positions[order]
    step = (ordered[-1] - ordered[0]) / (len(rows) - 1)
    if step @ normal <= 1e-6 or not np.allclose(np.diff(ordered, axis=0), step, atol=1e-3, rtol=0):
        raise ValueError('duplicate/nonuniform CT slices')
    spacing = np.asarray(first['pixel_spacing'], float)
    if spacing.shape != (2,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError('invalid pixel spacing')
    lps = np.eye(4); lps[:3, :3] = np.column_stack((x * spacing[1], y * spacing[0], step))
    lps[:3, 3] = ordered[0]
    return order.tolist(), np.diag([-1., -1., 1., 1.]) @ lps


def discover(root, source):
    root = Path(root); rows = []
    if source == 'headcta':
        meshes = {}
        for p in sorted((root/'Annotations').glob('*.stl')):
            match = re.fullmatch(r'an_(\d+)(?:_\d+)?', p.stem)
            if not match: raise ValueError('unknown STL ID: ' + p.name)
            meshes.setdefault(match[1].zfill(3), []).append(str(p))
        for cid, paths in sorted(meshes.items()):
            ct = sorted((root/'Studies'/cid).glob('*.dcm'))
            if not ct: raise ValueError('STL without CT: ' + cid)
            rows.append(dict(source_name=source, case_id=cid, ct_paths=list(map(str, ct)), stl_paths=paths))
    else:
        pattern = 'imagesTr/*_0000.nii.gz' if source == 'mswal' else '*_image.nii.gz'
        for p in sorted(root.glob(pattern)):
            cid = p.name[:-12] if source == 'mswal' else p.name[:-13]
            label = root/'labelsTr'/(cid+'.nii.gz') if source == 'mswal' else root/(cid+'_label.nii.gz')
            if not label.is_file(): raise ValueError('CT without GT: ' + str(p))
            rows.append(dict(source_name=source, case_id=cid, image_path=str(p), label_path=str(label)))
    return rows


def header_audit(rec):
    if rec['source_name'] != 'headcta':
        import nibabel as nib
        from four_source_data import _grid_corners, _original_geometry
        a, b = nib.load(rec['image_path']), nib.load(rec['label_path'])
        original = [_original_geometry(v) for v in (a, b)]
        a, b = nib.as_closest_canonical(a), nib.as_closest_canonical(b)
        if a.shape != b.shape or not np.allclose(_grid_corners(a.shape, a.affine),
                _grid_corners(b.shape, b.affine), rtol=0, atol=1e-4):
            raise ValueError('CT/GT physical grid mismatch: ' + rec['case_id'])
        return dict(grid_verified=True, original=original, shape=list(a.shape), affine_ras=a.affine.tolist())
    import pydicom
    rows = []
    for p in rec['ct_paths']:
        d = pydicom.dcmread(p, stop_before_pixels=True)
        if str(d.Modality) != 'CT': raise ValueError('non CT slice')
        rows.append(dict(position=list(d.ImagePositionPatient), orientation=list(d.ImageOrientationPatient),
            pixel_spacing=list(d.PixelSpacing), rows=int(d.Rows), columns=int(d.Columns),
            series=str(d.SeriesInstanceUID), sop=str(d.SOPInstanceUID)))
    order, affine = dicom_grid(rows)
    return dict(ct_grid_verified=True, stl_alignment_verified=False, shape=[rows[0]['columns'], rows[0]['rows'], len(rows)],
                affine_ras=affine.tolist(), ordered_ct_paths=[rec['ct_paths'][i] for i in order],
                note='STL coordinate convention/voxelization must pass independent audit before inference')


def freeze(data_root, audit=False):
    for source, rel in RELATIVE.items():
        root = Path(data_root)/rel; selected = select30(discover(root, source))
        payload = dict(source=source, seed=20261013, selected_count=30,
                       selection='random sampled native CT/GT pairs; no metrics used', cases=selected)
        path = root/FILENAME
        if path.exists():
            if json.loads(path.read_text()) != payload: raise ValueError('frozen selection changed')
        else: path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        print('FROZEN', source, path, [r['case_id'] for r in selected], flush=True)
        if audit:
            results=[]
            for rec in selected:
                try: results.append(dict(case_id=rec['case_id'], audit=header_audit(rec)))
                except Exception as e: results.append(dict(case_id=rec['case_id'], error=repr(e)))
                print('HEADER', source, rec['case_id'], 'error' if 'error' in results[-1] else 'pass', flush=True)
            target = root/'test30_geometry_headers_20261013.json'
            target.write_text(json.dumps(results, indent=2), encoding='utf-8')
            if any('error' in r for r in results): raise ValueError('header audit failures; no cases replaced')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True); p.add_argument('--audit-headers', action='store_true')
    a = p.parse_args(); freeze(a.data_root, a.audit_headers)
