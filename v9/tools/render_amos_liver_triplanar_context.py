"""Tissue-bearing front/right/top CT faces; reuse verified full-FOV surfaces."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import sys

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import render_amos_liver_triplanar_block as base

CAMERA = dict(elev=18., azim=112., projection='ortho')
BASE_SHA = 'b45b077f51b263820e57a6d6a24edf55a5c3cb6a78a504709ae66806af64fa0f'
SURFACE_FILES = ['v10_2_full_ct_surface.png', 'native_sam2_full_ct_surface.png', 'full_ct_surface_comparison.png']


def context_scene(shape, affine, coronal_w, sagittal_h):
    if len(shape) != 3 or min(shape) < 2 or not (0 < coronal_w < shape[2] and 0 <= sagittal_h < shape[1] - 1):
        raise ValueError('Nondegenerate in-volume tissue cut planes required')
    basis = np.asarray(affine, float)[:3, :3]
    basis = basis / np.linalg.norm(basis, axis=0)
    if not np.allclose(basis, [[0, 1, 0], [0, 0, 1], [1, 0, 0]], atol=1e-5):
        raise ValueError('Expected canonical RAS ZXY')
    return dict(bounds_dhw=[[0, int(shape[0])], [int(sagittal_h), int(shape[1])], [0, int(coronal_w) + 1]],
                intersection_dhw=[int(shape[0]) - 1, int(sagittal_h), int(coronal_w)],
                front_face='coronal', right_face='sagittal', top_face='axial', camera=dict(CAMERA),
                outward_normals_ras=np.column_stack((basis[:, 0], -basis[:, 1], basis[:, 2])).T.tolist(),
                scope='Full real scan height; tissue-bearing XY cutaway, not the complete mask surface')


def choose_planes(gt, fraction=.35):
    if gt.ndim != 3 or not gt.any() or not 0 < fraction <= 1:
        raise ValueError('Nonempty actual GT and side-area fraction in (0,1] required')
    coronal = int(np.argmax(np.count_nonzero(gt, axis=(0, 1))))
    areas = np.count_nonzero(gt, axis=(0, 2))
    sagittal = int(np.flatnonzero(areas >= fraction * areas.max())[0])
    return coronal, sagittal, dict(selection_input='GT display geometry only; predictions not used',
        coronal_selection='Maximum complete GT coronal area', sagittal_selection='First low-H plane reaching side-area fraction',
        side_area_fraction=float(fraction), actual_side_area_voxels=int(areas[sagittal]), maximum_side_area_voxels=int(areas.max()))


def scale_line(spec, affine):
    import itertools
    corners = np.array(list(itertools.product(*[(lo, hi - 1) for lo, hi in spec['bounds_dhw']])))
    xyz = base.core.to_ras(corners, affine)
    low, high = xyz.min(0), xyz.max(0)
    start = np.array([low[0] + 12., high[1] + 3., low[2]])
    return np.array([start, start + [100., 0, 0]])


@contextmanager
def front_scale_bar(spec, affine):
    """Only this isolated CPU process's plotting helper is scoped and restored."""
    original = base.scale_bar

    def draw(ax, low, high):
        from mpl_toolkits.mplot3d.proj3d import proj_transform
        line = scale_line(spec, affine)
        ax.plot(*line.T, color='white', linewidth=2.1, zorder=8)
        middle = line.mean(0)
        x, y, _ = proj_transform(*middle, ax.get_proj())
        xy = ax.figure.transFigure.inverted().transform(ax.transData.transform([x, y]))
        ax.figure.text(xy[0], xy[1] - .018, '100 mm', ha='center', va='top', color='white', fontsize=9)
        return dict(length_mm=100., endpoints_ras_mm=line.tolist(), label='2D offset below projected true scale bar')

    base.scale_bar = draw
    try:
        yield
    finally:
        base.scale_bar = original


def validate_surface_reuse(folder, binding, inputs):
    manifest = json.loads((folder / 'fixed_three_face_manifest.json').read_text())
    if (manifest.get('case_id') != 'amos_0167' or manifest.get('capture_binding') != binding
            or manifest.get('input_sha256') != inputs or manifest.get('renderer_sha256') != BASE_SHA):
        raise ValueError('Saved full-FOV surface source binding mismatch')
    for method in base.core.METHODS:
        info = manifest['complete_surfaces'][method]
        if (info.get('complete_mask') is not True or info.get('smoothing') is not False
                or info['scale_provenance'].get('mask_scaled_separately') is not False
                or info['scale_provenance'].get('ct_and_mask_share_one_affine') is not True):
            raise ValueError('Full CT scale/complete-mask proof missing')
    for name in SURFACE_FILES:
        if base.core.file_sha(folder / name) != manifest['outputs'].get(name):
            raise ValueError('Saved full-FOV surface PNG changed: ' + name)
    return manifest


def face_diagnostics(ct, gt, affine, pair, spec):
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    result = {}
    for plane, grid in base.pres.face_grids(spec).items():
        idx = tuple(np.moveaxis(grid, -1, 0))
        truth = gt[idx]
        area = np.prod([spacing[a] for a in range(3) if a != base.core.AXIS[plane]])
        info = dict(non_air_fraction=float(np.mean(ct[idx] > -500)), gt_area_mm2=float(truth.sum() * area))
        for method, item in pair.items():
            prediction = item['arrays']['prediction_dhw'][idx].astype(bool)
            info[method + '_recall'] = float(np.count_nonzero(truth & prediction) / truth.sum()) if truth.any() else None
        result[plane] = info
    return result


def render(args):
    root, output = Path(args.root).resolve(), Path(args.output).resolve()
    old = Path(args.surfaces_from).resolve() if args.surfaces_from else root / 'fixed_three_face_v4'
    if base.core.file_sha(base.__file__) != BASE_SHA:
        raise ValueError('Read-only base renderer changed')
    proof = root / 'figures_v1/render_manifest.json'
    binding = base.bind_sources(root, proof)
    ct, gt, affine, pair = base.core.load_inputs(root, Path(__file__).resolve().parents[1])
    inputs = base.input_inventory(pair)
    previous = validate_surface_reuse(old, binding, inputs)
    coronal, sagittal, selection = choose_planes(gt, args.side_area_fraction)
    spec = context_scene(ct.shape, affine, coronal, sagittal)
    diagnostics = face_diagnostics(ct, gt, affine, pair, spec)
    if diagnostics['sagittal']['non_air_fraction'] < .2 or diagnostics['axial']['non_air_fraction'] < .2:
        raise ValueError('Selected faces are mostly air; do not publish an unreadable CT block')
    base.core.reserve_output(output)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from skimage.measure import find_contours
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    base.core.configure_colors('#F5B83B', '#F6A0A0', '#70E5FF')
    with front_scale_bar(spec, affine):
        layout = base.draw_block(plt, find_contours, ct, gt, affine, None, spec, 'Original CT | Three orthogonal faces', output / 'raw_ct_block.png')
        for method in base.core.METHODS:
            base.draw_block(plt, find_contours, ct, gt, affine, pair[method]['arrays']['prediction_dhw'], spec,
                            base.core.NAMES[method] + ' | CT + prediction', output / (method + '_block.png'))
    base.assemble(output, ['raw_ct_block.png', 'v10_2_block.png', 'native_sam2_block.png'], 'three_face_comparison.png')
    for name in SURFACE_FILES:
        shutil.copyfile(old / name, output / name)
    if binding != base.bind_sources(root, proof) or inputs != base.input_inventory(pair):
        raise ValueError('Read-only captured predictions/inputs changed')
    validate_surface_reuse(old, binding, inputs)
    manifest = dict(schema='amos_liver_tissue_context_v1', case_id='amos_0167', capture_binding=binding, input_sha256=inputs,
        renderer_sha256=base.core.file_sha(__file__), base_renderer_sha256=BASE_SHA, scene=spec, selection=selection,
        face_diagnostics=diagnostics, projected_face_layout=layout, inference_rerun=False,
        reused_full_ct_surfaces=dict(source_manifest_sha256=base.core.file_sha(old / 'fixed_three_face_manifest.json'),
                                    png_sha256={n: previous['outputs'][n] for n in SURFACE_FILES}, complete_surfaces=previous['complete_surfaces']),
        whole_case_metrics=previous['whole_case_metrics'], colors=dict(prediction='#F5B83B', gt='#70E5FF'),
        outputs={p.name: base.core.file_sha(p) for p in sorted(output.glob('*.png'))})
    (output / 'tissue_context_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(output=str(output), selected=[coronal, sagittal], diagnostics=diagnostics, complete=True)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True, help='New directory only')
    parser.add_argument('--surfaces-from', help='Verified previous full-CT-scale surface directory')
    parser.add_argument('--side-area-fraction', type=float, default=.35, help='Display-only lower-H plane threshold')
    render(parser.parse_args())


if __name__ == '__main__':
    main()
