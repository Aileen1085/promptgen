"""Compose audited display-only PNG slices; no volume loading or model execution.

Accept manifest.json or ZIP containing manifest.json and relative PNG assets.
Schema five_source_display_slice_bundle_v1 requires export_scope=display_slices_only,
window_hu=[low,high], full_volume_audit={verified:true,method_rows:40,evidence:...},
and five cases (one AMOS/MAGIC/KiTS23/MSD10/MSD07). Each case supplies source,
case_id, class_name, slice_index (original zero-based axial index), shape_hw,
spacing_hw, zoom_crop_yx, display_transform, ct/gt={path,sha256}, and eight methods
{name,mask:{path,sha256},dice_3d,metrics_evidence}. CT is a full-FOV, already
windowed 8-bit L-mode PNG; masks are L-mode binary 0/1 or 0/255 PNGs. Images are
stored BEFORE the common display transform/crop. Physical-grid and full-volume
metric verification belong to the upstream exporter, and their evidence is
retained verbatim. This renderer never computes a volume metric from a slice.
"""
import argparse
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path, PurePosixPath
import sys
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.render_five_source_qualitative import METHODS, SOURCES, _case_row, sha256, source_name


class SliceView:
    """One real plane indexed by its original z; deliberately no 3D array facade."""
    def __init__(self, plane, original_z):
        self.plane = np.asarray(plane)
        self.plane.flags.writeable = False
        self.original_z = original_z

    def __getitem__(self, index):
        if (not isinstance(index, tuple) or len(index) != 3 or
                isinstance(index[0], bool) or not isinstance(index[0], (int, np.integer)) or
                index[0] != self.original_z or not all(isinstance(i, slice) for i in index[1:])):
            raise IndexError('SliceView only permits the exported original axial index and two slices')
        return self.plane[index[1], index[2]]


class _Reader:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.archive = zipfile.ZipFile(self.path) if self.path.suffix.lower() == '.zip' else None
        self.base = self.path.parent
        if self.archive and len(self.archive.namelist()) != len(set(self.archive.namelist())):
            self.archive.close()
            raise ValueError('Duplicate ZIP member names')

    def read(self, relative):
        name = PurePosixPath(str(relative))
        if (name.is_absolute() or '..' in name.parts or '\\' in str(relative) or ':' in str(relative)
                or str(name) in ('', '.')):
            raise ValueError('Asset paths must be relative bundle paths')
        if self.archive:
            return self.archive.read(str(name))
        target = (self.base / str(name)).resolve()
        if not target.is_relative_to(self.base):
            raise ValueError('Asset escapes bundle directory')
        return target.read_bytes()

    def close(self):
        if self.archive:
            self.archive.close()


def _evidence(value):
    return bool(value) and isinstance(value, (str, dict)) and (not isinstance(value, str) or bool(value.strip()))


def _png(reader, asset, shape, binary=False):
    payload = reader.read(asset['path'])
    if hashlib.sha256(payload).hexdigest() != asset['sha256']:
        raise ValueError('PNG SHA mismatch: ' + asset['path'])
    with Image.open(BytesIO(payload)) as image:
        if image.format != 'PNG' or image.mode != 'L':
            raise ValueError('Require 8-bit grayscale L-mode PNG: ' + asset['path'])
        plane = np.array(image)
    if plane.shape != tuple(shape) or plane.dtype != np.uint8:
        raise ValueError('PNG dimensions/dtype mismatch: ' + asset['path'])
    if binary:
        values = set(np.unique(plane).tolist())
        if not (values <= {0, 1} or values <= {0, 255}):
            raise ValueError('Mask PNG must be binary 0/1 or 0/255')
        plane = (plane > 0).astype(np.uint8)
    return plane


def load_bundle(bundle):
    """Validate every exported asset before reserving output; strict five/eight scope."""
    reader = _Reader(bundle)
    try:
        payload = reader.read('manifest.json') if reader.archive else reader.path.read_bytes()
        manifest = json.loads(payload.decode('utf-8'))
        if (manifest.get('schema') != 'five_source_display_slice_bundle_v1'
                or manifest.get('export_scope') != 'display_slices_only'):
            raise ValueError('Audited display_slices_only schema required')
        audit = manifest.get('full_volume_audit', {})
        if (audit.get('verified') is not True or audit.get('method_rows') != 40 or
                not _evidence(audit.get('evidence'))):
            raise ValueError('Verified upstream full-volume 40-method-row evidence required')
        window = manifest['window_hu']
        if len(window) != 2 or not np.isfinite(window).all() or window[0] >= window[1]:
            raise ValueError('Invalid CT window_hu')
        entries = manifest['cases']
        if len(entries) != 5 or {source_name(c['source']) for c in entries} != set(SOURCES):
            raise ValueError('Need exactly one case-class per five sources')
        cases = []
        for entry in entries:
            shape, spacing, z = entry['shape_hw'], entry['spacing_hw'], entry['slice_index']
            if (len(shape) != 2 or any(type(v) is not int or v <= 0 for v in shape)
                    or len(spacing) != 2 or not np.isfinite(spacing).all() or min(spacing) <= 0
                    or type(z) is not int or z < 0
                    or any(not isinstance(entry.get(k), str) or not entry[k].strip() for k in ('case_id', 'class_name'))):
                raise ValueError('Invalid case identity, dimensions, spacing, or original slice index')
            crop = entry['zoom_crop_yx']
            if (len(crop) != 4 or any(type(v) is not int for v in crop)
                    or not (0 <= crop[0] < crop[1] <= shape[0] and 0 <= crop[2] < crop[3] <= shape[1])):
                raise ValueError('Invalid common zoom crop')
            transform = entry['display_transform']
            if (not isinstance(transform, dict) or set(transform) - {'transpose', 'flip_y', 'flip_x'}
                    or any(type(v) is not bool for v in transform.values())):
                raise ValueError('Invalid common display transform')
            methods = entry['methods']
            if len(methods) != 8 or {m['name'] for m in methods} != set(METHODS):
                raise ValueError('Need exactly eight main-table methods')
            gray = _png(reader, entry['ct'], shape)
            gt = _png(reader, entry['gt'], shape, binary=True)
            result_methods = []
            for name in METHODS:
                method = next(m for m in methods if m['name'] == name)
                dice = float(method['dice_3d'])
                if not math.isfinite(dice) or not 0 <= dice <= 1 or not _evidence(method.get('metrics_evidence')):
                    raise ValueError('Finite attested full-volume Dice and metric evidence required')
                mask = _png(reader, method['mask'], shape, binary=True)
                result_methods.append(dict(name=name, dice_3d=dice, mask=SliceView(mask, z)))
            ct = gray.astype(np.float64) / 255 * (window[1] - window[0]) + window[0]
            cases.append(dict(source=source_name(entry['source']), case_id=entry['case_id'],
                              class_name=entry['class_name'], slice_index=z, spacing_dhw=[None, *spacing],
                              crop_yx=[0, shape[0], 0, shape[1]], zoom_crop_yx=crop,
                              display_transform=transform, ct=SliceView(ct, z), gt=SliceView(gt, z),
                              methods=result_methods))
        cases.sort(key=lambda c: SOURCES.index(c['source']))
        return cases, manifest
    finally:
        reader.close()


def render_bundle(bundle, output, panel_pixels=512, alpha=.4, gt_contour=False):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Output directory must be fresh')
    if type(panel_pixels) is not int or panel_pixels < 96 or not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('Invalid panel size or opacity')
    cases, manifest = load_bundle(bundle)
    output.mkdir(parents=True, exist_ok=False)
    for view in ('full', 'zoom'):
        folder = output / view
        folder.mkdir()
        rows = []
        for i, case in enumerate(cases):
            display = dict(case)
            if view == 'zoom':
                display['crop_yx'] = case['zoom_crop_yx']
            row = _case_row(display, manifest['window_hu'], alpha, gt_contour, panel_pixels)
            stem = f"{i + 1:02d}_{case['source']}"
            row.save(folder / f'{stem}.png', dpi=(300, 300))
            row.save(folder / f'{stem}.pdf', resolution=300, quality=95, subsampling=0)
            rows.append(row)
        combined = Image.new('RGB', (rows[0].width, sum(r.height for r in rows)), 'black')
        top = 0
        for row in rows:
            combined.paste(row, (0, top))
            top += row.height
        combined.save(folder / 'five_source_comparison.png', dpi=(300, 300))
        combined.save(folder / 'five_source_comparison.pdf', resolution=300, quality=95, subsampling=0)
    caption = ('Post-hoc illustrative examples, not representative averages. Only exported display slices '
               'were transferred. CT is the upstream windowed full-FOV grayscale PNG. All columns within '
               'a row use identical original axial index, window, crop, spacing, and display transform. '
               'Full and zoom views use the same slice. Orange denotes segmentation at opacity '
               f'{alpha:g}. ' + ('Cyan outlines denote GT in all prediction columns. ' if gt_contour else '') +
               'Labels report upstream verified full-volume 3D Dice, not recomputed from these slices. '
               'Native method-specific prompt budgets are retained. Original slice indices are zero-based.\n')
    report = dict(manifest, renderer_sha256=sha256(Path(__file__)),
                  compositor_sha256=sha256(ROOT / 'tools/render_five_source_qualitative.py'),
                  bundle_sha256=sha256(bundle), panel_pixels=panel_pixels, alpha=alpha,
                  gt_contour=gt_contour, dpi=300, caption=caption,
                  metric_scope='upstream_full_volume_attested_not_recomputed_from_slice')
    (output / 'caption.txt').write_text(caption, encoding='utf-8')
    (output / 'provenance.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path, help='manifest.json or bundle.zip')
    parser.add_argument('output', type=Path, help='Fresh output directory')
    parser.add_argument('--panel-pixels', type=int, default=512)
    parser.add_argument('--alpha', type=float, default=.4)
    parser.add_argument('--gt-contour', action='store_true')
    args = parser.parse_args()
    render_bundle(args.bundle, args.output, args.panel_pixels, args.alpha, args.gt_contour)
    print('Rendered audited five-source slice bundle:', args.output)


if __name__ == '__main__':
    main()
