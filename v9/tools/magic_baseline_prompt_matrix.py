"""Run the 37 baseline arms on fixed MAGIC val; audit original AMOS PAM fusion pairs."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import amos_prompt_matrix as matrix
from tools import amos_prompt_matrix_interactive as interactive
from tools import pam_soft_fusion as soft

native = matrix.native
OUT = ROOT / 'output/magic_baseline_prompt_matrix_20261121'
METHODS = {method: arms for method, arms in matrix.ARMS.items() if method != 'promptgen'}
AMOS_MATRIX = matrix.OUT / 'protocol.json'
PAM_MAGIC = ROOT / 'output/pam_probability_nine_source_full4548_20261023/merged_metrics.json'
PAM_OR = ROOT / 'output/pam_amos_three_view_full1485_20261023/merged_metrics.json'
PAM_PROB = ROOT / 'output/pam_probability_amos_full1485_20261025/merged_metrics.json'
NATIVE_BINDING = matrix.code_binding
NATIVE_INTERACTIVE = interactive.InteractivePredictor
PLANES = ('coronal', 'sagittal')


def select_magic(rows, split):
    chosen = sorted((row for row in rows if row['source_name'] == 'magic'), key=native.key)
    dataset = split['datasets']['magic']
    cases = {row['case_id'] for row in chosen}
    if (len(chosen) != 35 or len(cases) != 35
            or len({native.key(row)[2] for row in chosen}) != 7
            or cases != {row['case_id'] for row in dataset['val']}
            or cases & {row['case_id'] for row in dataset['train']}
            or any(row['split_role'] != 'validation' for row in chosen)):
        raise ValueError('expected all 35 unique MAGIC validation cases, seven classes, train0')
    for case in cases:
        path = PurePosixPath(case)
        if not path.parts or path.is_absolute() or '..' in path.parts or '\\' in case or ':' in case:
            raise ValueError(f'invalid MAGIC case key {case!r}')
    return chosen


def map_foreground(fg, raw_planes, source, canonical, target):
    base = native.base_module()
    order = np.rint((np.linalg.inv(source) @ canonical)[:3, :3]).astype(int)
    mapped = {}
    for support, raw_axis, index in zip(fg, (2, 1), raw_planes, strict=True):
        original = matrix.face_coordinates(support, raw_axis, index)
        if len(original) == 0 or len(original) != np.count_nonzero(support):
            raise ValueError('invalid tagged raw scribble plane')
        grid = base.reorient_grid(support, source, canonical, target.shape)
        axes = np.flatnonzero(order[raw_axis])
        if len(axes) != 1 or int(axes[0]) not in (1, 2) or int(axes[0]) in mapped:
            raise ValueError('native planes do not map to orthogonal canonical coronal/sagittal')
        axis = int(axes[0])
        coordinates, error = base.map_voxels(original, source, canonical)
        integer = np.rint(coordinates).astype(int)
        if (error >= 1e-3 or (integer < 0).any() or (integer >= target.shape).any()
                or not grid[tuple(integer.T)].all() or len(integer) != np.count_nonzero(grid)
                or not target[grid].all() or len(set(integer[:, axis])) != 1):
            raise ValueError('MAGIC foreground physical alignment failed')
        mapped[axis] = grid, int(integer[0, axis])
    return tuple(mapped[i][0] for i in (2, 1)), tuple(mapped[i][1] for i in (2, 1))


def gt_boxes(target, planes):
    return tuple(dict(plane=plane, slice=int(index), box=soft.p.slice_box(target, plane, index))
                 for plane, index in zip(PLANES, planes, strict=True))


def box_digest(boxes):
    return hashlib.sha256(json.dumps(boxes, sort_keys=True).encode()).hexdigest()


def load_case_ct(base, image, cache_path, raw_shape, source_affine, canonical, read_raw=None):
    shape = tuple(image.shape[i] for i in (2, 0, 1))

    def map_raw(raw):
        if raw.shape != raw_shape or raw.dtype != np.float32:
            raise ValueError('MAGIC case CT is not raw float32 on the prompt grid')
        return base.reorient_grid(raw, source_affine, canonical, shape)

    if cache_path.is_file():
        return np.ascontiguousarray(map_raw(np.load(cache_path, mmap_mode='r', allow_pickle=False))), True
    ct = np.ascontiguousarray(np.asarray(image.dataobj, dtype=np.float32).transpose(2, 0, 1))
    if read_raw is not None and not np.array_equal(ct, map_raw(read_raw())):
        raise ValueError('existing MAGIC NIfTI and physical raw CT differ for missing shared cache')
    return ct, False


class CaseLoader:
    """Map cached raw CT and tagged strokes to the existing MAGIC NIfTI scoring grid."""

    def __init__(self, verify_fallback_ct=False):
        self.base = native.base_module()
        self.entries = {row['case_id']: row for row in native.read(matrix.SPLIT)['datasets']['magic']['val']}
        self.case = None
        self.verify_fallback_ct = verify_fallback_ct

    def __call__(self, task):
        import SimpleITK as sitk

        name = task['prompt_cache_name']
        if not name or Path(name).name != name:
            raise ValueError('invalid prompt cache name')
        package_path = native.CACHE / 'metadata/shared_adaptive_prompt_v1/magic' / name
        with np.load(package_path, allow_pickle=False) as package:
            ct_name = str(package['case_image_name'].item())
            shape = tuple(int(v) for v in package['shape_dhw'])
            raw_planes = tuple(int(v) for v in package['slices'].reshape(2))
            supports = tuple(np.unpackbits(package[plane + '_bits'].astype(np.uint8), count=math.prod(shape),
                                           bitorder='little').reshape(shape).astype(bool) for plane in PLANES)
        if not ct_name or Path(ct_name).name != ct_name or task['source_name'] != 'magic':
            raise ValueError('invalid MAGIC cache name or source')
        key = (task['case_id'], ct_name)
        if self.case != key:
            entry = self.entries[task['case_id']]
            image_path = Path(entry.get('image', entry.get('image_path')))
            label_path = Path(entry.get('label', entry.get('label_path')))
            image, self.affine = self.base.canonical(image_path)
            label, label_affine = self.base.canonical(label_path)
            self.target = native.select_target(np.asarray(label.dataobj).transpose(2, 0, 1), task)
            if self.target.shape != (image.shape[2], image.shape[0], image.shape[1]) or not self.target.any():
                raise ValueError('invalid MAGIC complete-volume label')
            native.assert_compatible_grid(self.affine, label_affine, self.target.shape,
                                          image_qform=image.get_qform(), label_qform=label.get_qform())
            raw_path = Path(task['image'])
            reader = sitk.ImageFileReader()
            reader.SetFileName(str(raw_path))
            reader.ReadImageInformation()
            lps = np.eye(4)
            lps[:3, :3] = (np.asarray(reader.GetDirection()).reshape(3, 3) @ np.diag(reader.GetSpacing()))[:, [2, 1, 0]]
            lps[:3, 3] = reader.GetOrigin()
            self.source_affine = np.diag([-1., -1., 1., 1.]) @ lps
            ct_path = native.CACHE / '_case_images_v1' / ct_name
            read_raw = (lambda: sitk.GetArrayFromImage(sitk.ReadImage(str(raw_path), sitk.sitkFloat32)))
            self.ct, cache_hit = load_case_ct(self.base, image, ct_path, shape, self.source_affine, self.affine,
                                              read_raw if self.verify_fallback_ct else None)
            self.spacing = tuple(np.linalg.norm(self.affine[:3, :3], axis=0))
            self.files = {k: matrix.stat_identity(p) for k, p in dict(ct=ct_path if cache_hit else image_path, image=image_path,
                          label=label_path, raw_geometry=raw_path).items()}
            self.files['ct_source'] = 'shared_read_only' if cache_hit else 'existing_nifti_verified_against_raw'
            self.case = key
        foreground, planes = map_foreground(supports, raw_planes, self.source_affine, self.affine, self.target)
        background = matrix.background_supports(self.target, planes)
        self.files['pam_submission_sha256'] = box_digest(gt_boxes(self.target, planes))
        return matrix.Prompts(self.ct, self.target, foreground, background, self.spacing, planes, name,
                              native.sha(package_path), ct_name, matrix.points_digest(background, planes))


class InteractivePredictor(NATIVE_INTERACTIVE):
    def _evaluate_pam(self, prompts):
        payload = gt_boxes(prompts.target, prompts.planes)
        scores, audits = self.fusion.predict(prompts.ct, prompts.spacing, payload, self.models)
        if (tuple(a['plane'] for a in audits) != PLANES or
                any(a['box_decoder_calls'] != 1 or a['gt_mask_supplied'] for a in audits)):
            raise ValueError('PAM requires two native box evaluations and no GT masks')
        for arm in self.fusion.METHODS:
            yield arm, scores[arm], dict(boxes=payload, submission_sha256=box_digest(payload),
                  gt_used_only_for_selected_2d_box=True, gt_mask_supplied=False, three_D_GT_ROI=False,
                  background_prompt_count=0, native_views=audits, fusion_extra_calls=0)


def paired_pam(tasks, binary, prob):
    indexes = []
    keys = [native.key(t) for t in tasks]
    unique = set(keys)
    if not keys or len(unique) != len(keys):
        raise ValueError('invalid paired AMOS scope')
    for rows, method in ((binary, 'dual_or'), (prob, 'dual_prob')):
        subset = [r for r in rows if r['method'] == method and native.key(r) in unique]
        native.validate_rows(subset, tasks)
        indexes.append({native.key(r): r for r in subset})
    pairs = []
    for key in keys:
        a, b = (index[key] for index in indexes)
        x, y = (r['prompt_audit'] for r in (a, b))
        if (x['submission_sha256'] != y['submission_sha256'] or any(
                pa['submission_sha256'] != box_digest(pa['boxes']) or pa['three_D_GT_ROI']
                or not pa['gt_used_only_for_selected_2D_box'] for pa in (x, y))):
            raise ValueError(f'original two-dimensional PAM boxes do not match for {key}')
        pa, pb = (r['propagation_audit'] for r in (a, b))
        fields = ('plane', 'slice', 'box_decoder_calls', 'prop_decoder_calls', 'guide_slices', 'guiding_pixels')
        if (any(tuple(view['plane'] for view in trace) != PLANES for trace in (pa, pb))
                or any(u['box_decoder_calls'] != 1 or u['gt_mask_supplied'] or v['gt_mask_supplied']
                       or any(u.get(f) != v.get(f) for f in fields) for u, v in zip(pa, pb, strict=True))):
            raise ValueError(f'original PAM propagation does not match for {key}')
        pairs.append(dict(key=key, box_sha256=x['submission_sha256'],
                          dual_or={m: a[m] for m in native.METRICS}, dual_prob={m: b[m] for m in native.METRICS}))
    means = {method: {m: math.fsum(p[method][m] for p in pairs) / len(pairs) for m in native.METRICS}
             for method in ('dual_or', 'dual_prob')}
    delta = {m: math.fsum(p['dual_prob'][m] - p['dual_or'][m] for p in pairs) / len(pairs) for m in native.METRICS}
    return dict(tasks=len(keys), cases=len({k[1] for k in keys}), averages=means,
                delta_probability_minus_union=delta, per_key=pairs, new_forward_calls=0)


def audit_amos(union_sha256, probability_sha256):
    archive_hashes = {PAM_OR: union_sha256.lower(), PAM_PROB: probability_sha256.lower()}
    if any(native.sha(path) != digest for path, digest in archive_hashes.items()):
        raise ValueError('archived PAM artifacts differ from documented SHA-256')
    tasks = matrix.select_tasks(native.read(matrix.SOURCE)['tasks'])
    previous = native.read(AMOS_MATRIX)['tasks']
    if len(tasks) != 149 or {native.key(t) for t in tasks} != {native.key(t) for t in previous}:
        raise ValueError('original PAM audit must use the same 149 AMOS matrix keys')
    result = paired_pam(tasks, *(native.read(path)['per_case_class'] for path in archive_hashes))
    result['archive_sha256'] = {str(p.relative_to(ROOT)): h for p, h in archive_hashes.items()}
    result['amos_matrix_protocol_sha256'] = native.sha(AMOS_MATRIX)
    write_or_verify(OUT / 'amos_pam_original_boxes_review.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'per_key'}, indent=2))


def bindings():
    source = NATIVE_BINDING()
    source['tools/amos_prompt_matrix.py'] = native.sha(ROOT / 'tools/amos_prompt_matrix.py')
    source['tools/magic_baseline_prompt_matrix.py'] = native.sha(Path(__file__))
    return source


def freeze_inputs():
    return dict(code=bindings(), split=native.sha(matrix.SPLIT), manifest=native.sha(matrix.SOURCE),
                original_pam=native.sha(PAM_MAGIC))


def verify_inputs(payload):
    if (payload['inputs'] != freeze_inputs() or payload['methods'] != {m: list(a) for m, a in METHODS.items()}
            or payload['groups'] != {m: [list(a) for a in matrix.arm_groups(m)] for m in METHODS}):
        raise ValueError('MAGIC frozen inputs changed')


def prepare():
    path = OUT / 'protocol.json'
    if path.exists():
        raise FileExistsError('MAGIC matrix already prepared')
    inputs = freeze_inputs()
    tasks = select_magic(native.read(matrix.SOURCE)['tasks'], native.read(matrix.SPLIT))
    old = [r for r in native.read(PAM_MAGIC)['per_case_class'] if r['source_name'] == 'magic' and r['method'] == 'dual_prob']
    native.validate_rows(old, tasks)
    old_boxes = {native.key(r): r['prompt_audit']['submission_sha256'] for r in old}
    loader = CaseLoader(verify_fallback_ct=True)
    for row in tasks:
        prompts = loader(row)
        if old_boxes[native.key(row)] != loader.files['pam_submission_sha256']:
            raise ValueError(f'MAGIC PAM main-table box differs on {native.key(row)}')
        row['prompt_identity'] = matrix.prompt_identity(prompts, loader.files)
        print('MAGIC_PREFLIGHT', row['case_id'], flush=True)
    if inputs != freeze_inputs():
        raise ValueError('MAGIC inputs changed while preparing')
    native.write(path, dict(inputs=inputs, tasks=tasks, cases=sorted(t['case_id'] for t in tasks),
                 methods={m: list(a) for m, a in METHODS.items()},
                 groups={m: [list(a) for a in matrix.arm_groups(m)] for m in METHODS}))
    print('MAGIC_PREPARED 35 cases / 35 tasks / 37 arms / 1295 rows', flush=True)


def write_or_verify(path, data):
    if path.exists():
        if native.read(path) != json.loads(json.dumps(data)):
            raise ValueError(f'previous output differs: {path}')
    else:
        native.write(path, data)


def summary(payload):
    digest = native.sha(OUT / 'protocol.json')
    report = {}
    for method, arms in METHODS.items():
        metrics = {arm: [] for arm in arms}
        for task in payload['tasks']:
            key = list(native.key(task))
            for group, _ in enumerate(matrix.arm_groups(method)):
                certificate = native.read(matrix.certificate_file(method, task['case_id'], group))
                record = matrix.read_record(method, task, group, digest)
                if (certificate['keys'] != [key] or certificate['protocol_sha256'] != digest
                        or certificate['method'] != method or certificate['group'] != group
                        or certificate['uuid'] != record['gpu_uuid']):
                    raise ValueError('mismatched MAGIC case certificate')
                for row in record['rows']:
                    if method == 'pam' and (row['audit']['submission_sha256'] != task['prompt_identity']['files']['pam_submission_sha256']
                            or row['audit']['gt_mask_supplied'] or row['audit']['fusion_extra_calls'] != 0):
                        raise ValueError('MAGIC PAM scoring record does not use the original boxes')
                    metrics[row['arm']].append(row['metrics'])
        if any(len(rows) != 35 for rows in metrics.values()):
            raise ValueError('incomplete MAGIC arm')
        report[method] = {arm: dict(tasks=35, averages={m: math.fsum(r[m] for r in rows) / 35 for m in native.METRICS})
                          for arm, rows in metrics.items()}
    return dict(protocol_sha256=digest, cases=payload['cases'], methods=report, rows=1295)


def merge(payload):
    write_or_verify(OUT / 'merged_metrics.json', summary(payload))
    print('MAGIC_BASELINE_PROMPT_MATRIX_COMPLETE 1295', flush=True)


@contextmanager
def runner():
    # Reuse the proven scheduler/worker without modifying the historical runner on disk.
    values = dict(OUT=OUT, ARMS=METHODS, prepare=prepare, merge=merge, CaseLoader=CaseLoader, __doc__=__doc__,
                  verify_inputs=verify_inputs, __file__=str(Path(__file__).resolve()))
    original = {key: getattr(matrix, key) for key in values}
    vars(matrix).update(values)
    interactive.InteractivePredictor = InteractivePredictor
    try:
        yield
    finally:
        vars(matrix).update(original)
        interactive.InteractivePredictor = NATIVE_INTERACTIVE


def main():
    if sys.argv[1:2] == ['audit-amos']:
        parser = argparse.ArgumentParser(description='Audit archived AMOS PAM prediction pairs.')
        parser.add_argument('--union-sha256', required=True)
        parser.add_argument('--probability-sha256', required=True)
        audit_amos(**vars(parser.parse_args(sys.argv[2:])))
        return
    with runner():
        if sys.argv[1:] == ['verify']:
            payload = matrix.protocol()
            verify_inputs(payload)
            if native.read(OUT / 'merged_metrics.json') != summary(payload):
                raise ValueError('saved MAGIC metrics do not match the committed records')
            print('MAGIC_BASELINE_PROMPT_MATRIX_VERIFY_PASS 1295')
        else:
            matrix.main()


if __name__ == '__main__':
    main()
