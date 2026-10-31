"""Fixed front-coronal/right-sagittal/top-axial CT, plus full-CT-scale masks.

Read-only rendering of the already captured amos_0167 predictions. The cutaway
is a displayed XY subset; the separate surface figures contain every predicted
voxel and use the complete original CT field of view. No inference is imported.
"""
import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import render_amos_liver_case as core
from tools import render_amos_liver_presentation as pres
from tools import render_amos_liver_surfaces as surfaces

SETTINGS = dict(level=50., window=350., alpha=.32)
CAMERA = dict(elev=18., azim=-72., projection='ortho')


def bind_sources(root, proof):
    manifest = json.loads(Path(proof).read_text(encoding='utf8'))
    if manifest.get('case_id') != 'amos_0167':
        raise ValueError('Only the existing amos_0167 captures are authorized')
    return pres.validate_capture_binding(manifest, root)


def camera_direction(camera):
    elev, azim = np.radians([camera['elev'], camera['azim']])
    return np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])


def project_camera(points, camera):
    """Screen right/up orthographic coordinates; never rescale anatomy."""
    elev, azim = np.radians([camera['elev'], camera['azim']])
    right = [-np.sin(azim), np.cos(azim), 0.]
    up = [-np.sin(elev) * np.cos(azim), -np.sin(elev) * np.sin(azim), np.cos(elev)]
    return np.asarray(points) @ np.array([right, up]).T


def fixed_scene(shape, affine, coronal_w, sagittal_h):
    if len(shape) != 3 or min(shape) < 2 or not (0 <= coronal_w < shape[2] - 1 and 0 < sagittal_h < shape[1]):
        raise ValueError('Nondegenerate real cut planes within the CT are required')
    basis = np.asarray(affine, float)[:3, :3]
    basis = basis / np.linalg.norm(basis, axis=0)
    if not np.allclose(basis, [[0, 1, 0], [0, 0, 1], [1, 0, 0]], atol=1e-5):
        raise ValueError('Expected canonical RAS ZXY geometry')
    # Posterior-facing coronal boundary, right sagittal boundary, superior top.
    # Unlike the previous GT-max quadrant, face order is fixed by geometry.
    return dict(bounds_dhw=[[0, int(shape[0])], [0, int(sagittal_h) + 1], [int(coronal_w), int(shape[2])]],
                intersection_dhw=[int(shape[0]) - 1, int(sagittal_h), int(coronal_w)],
                front_face='coronal', right_face='sagittal', top_face='axial', camera=dict(CAMERA),
                outward_normals_ras=np.column_stack((basis[:, 0], basis[:, 1], -basis[:, 2])).T.tolist(),
                scope='Real CT cutaway with full scan height; displayed XY subset, not a complete mask surface')


def full_volume_limits(shape, affine):
    if len(shape) != 3 or min(shape) < 2:
        raise ValueError('A three-dimensional CT shape is required')
    corners = np.array(list(itertools.product(*[(-.5, n - .5) for n in shape])))
    xyz = core.to_ras(corners, affine)
    return xyz.min(0), xyz.max(0)


def scale_provenance(mask, spacing):
    mask, spacing = np.asarray(mask), np.asarray(spacing, float)
    if (mask.ndim != 3 or min(mask.shape) < 2 or not np.isin(mask, [0, 1]).all() or not mask.any()
            or spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0)):
        raise ValueError('Actual nonempty binary mask and positive finite physical spacing required')
    count = int(np.count_nonzero(mask))
    occupied = np.argwhere(mask)
    return dict(full_ct_shape_dhw=list(mask.shape), full_ct_fov_dhw_mm=(np.array(mask.shape) * spacing).tolist(),
                mask_voxels=count, full_ct_voxels=int(mask.size), mask_volume_fraction_of_full_ct=count / mask.size,
                maximum_axial_area_fraction_of_full_ct=float(np.count_nonzero(mask, axis=(1, 2)).max() / np.prod(mask.shape[1:])),
                mask_extent_dhw_mm=((occupied.max(0) - occupied.min(0) + 1) * spacing).tolist(),
                mask_scaled_separately=False, ct_and_mask_share_one_affine=True)


def input_inventory(pair):
    expected = {}
    for item in pair.values():
        for name, value in item['meta']['provenance']['input_sha256'].items():
            if name in expected and expected[name] != value:
                raise ValueError('Paired input signatures disagree')
            expected[name] = value
    for name, value in expected.items():
        if core.file_sha(name) != value:
            raise ValueError('Frozen CT/GT/prompt/reference input changed: ' + name)
    return expected


def frame(ax, low, high, camera):
    low, high = np.asarray(low), np.asarray(high)
    margin = np.array([8., 8., 12.])
    ax.set_xlim(low[0] - margin[0], high[0] + margin[0])
    ax.set_ylim(low[1] - margin[1], high[1] + margin[1])
    ax.set_zlim(low[2] - margin[2], high[2] + margin[2])
    ax.set_box_aspect(high - low + 2 * margin)
    ax.view_init(elev=camera['elev'], azim=camera['azim'])
    ax.set_axis_off()


def wire_box(ax, low, high, alpha=.38):
    corners = np.array(list(itertools.product(*zip(low, high))))
    for a, b in itertools.combinations(corners, 2):
        if np.count_nonzero(a != b) == 1:
            ax.plot(*np.array([a, b]).T, color='#A8BFCB', linewidth=.65, alpha=alpha, zorder=1)


def scale_bar(ax, low, high):
    length = 100.
    start = np.array([low[0] + 12., low[1] - 3., low[2]])
    end = start + [length, 0, 0]
    ax.plot(*np.array([start, end]).T, color='white', linewidth=2.1, zorder=8)
    ax.text(*(start + [length / 2, 0, -8.]), '100 mm', color='white', fontsize=9, ha='center', zorder=8)
    return dict(length_mm=length, endpoints_ras_mm=[start.tolist(), end.tolist()])


def new_axes(plt):
    fig = plt.figure(figsize=(6.4, 7.2), facecolor=pres.DARK)
    ax = fig.add_axes([.01, .065, .98, .84], projection='3d', proj_type='ortho', computed_zorder=False, facecolor=pres.DARK)
    return fig, ax


def draw_block(plt, find_contours, ct, gt, affine, prediction, spec, title, path):
    fig, ax = new_axes(plt)
    # Explicit painter order: top, right side, front; contours only on that face.
    grids = {plane: core.plane_grid(spec['bounds_dhw'], plane, spec['intersection_dhw'][core.AXIS[plane]], max_side=320)
             for plane in ('axial', 'sagittal', 'coronal')}
    for layer, (plane, grid) in enumerate(grids.items()):
        idx = tuple(np.moveaxis(grid, -1, 0))
        rgb = (core.window_rgb(ct[idx], SETTINGS['level'], SETTINGS['window']) if prediction is None else
               core.overlay_prediction(ct[idx], prediction[idx], SETTINGS['level'], SETTINGS['window'], SETTINGS['alpha']))
        xyz = core.to_ras(grid, affine)
        ax.plot_surface(*np.moveaxis(xyz, -1, 0), facecolors=rgb / 255., rstride=1, cstride=1,
                        linewidth=0, shade=False, antialiased=False, zorder=2 + layer)
        if prediction is not None:
            index = spec['intersection_dhw'][core.AXIS[plane]]
            for yx in find_contours(core.oriented_slice(gt, plane, index).astype(np.uint8), .5):
                dhw = pres.pixels_to_dhw(yx[:, ::-1], gt.shape, plane, index)
                for line in pres.clip_polyline(dhw, spec['bounds_dhw']):
                    coords = core.to_ras(line, affine) + .04 * np.asarray(spec['outward_normals_ras'][core.AXIS[plane]])
                    ax.plot(*coords.T, color=core.GT_COLOR, linewidth=1.35, linestyle=(0, (4, 3)), zorder=7)
    for edge in core.to_ras(pres.visible_edges(spec), affine):
        ax.plot(*edge.T, color='#C4D8E1', linewidth=.75, zorder=8)
    xyz = core.to_ras(np.array(list(itertools.product(*[(lo, hi - 1) for lo, hi in spec['bounds_dhw']]))), affine)
    low, high = xyz.min(0), xyz.max(0)
    frame(ax, low, high, spec['camera'])
    scale = scale_bar(ax, low, high)
    fig.suptitle(title, color='white', fontsize=14, y=.97)
    fig.text(.5, .025, 'Front: coronal  |  Right: sagittal  |  Top: axial\nAll 89 real CT slices in height; displayed XY cutaway',
             color='#E0EAF0', ha='center', fontsize=9)
    fig.savefig(path, dpi=180, facecolor=pres.DARK)
    plt.close(fig)
    centers = {p: core.to_ras(g.reshape(-1, 3).mean(0), affine) for p, g in grids.items()}
    return dict(face_centers_screen_mm={p: project_camera(c, spec['camera']).tolist() for p, c in centers.items()}, scale_bar=scale)


def draw_full_surface(plt, find_contours, marching_cubes, ct, gt, affine, mask, title, path):
    vertices, faces, _, _ = marching_cubes(surfaces.prepare_volume(mask), .5, step_size=1, allow_degenerate=False)
    xyz = surfaces.vertices_to_ras(vertices, affine)
    fig, ax = new_axes(plt)
    bounds = [[0, int(n)] for n in ct.shape]
    low, high = full_volume_limits(ct.shape, affine)
    # Full original 512x512 FOV on every CT reference plane, no target bbox crop.
    context_z = [0, int(np.flatnonzero(np.any(gt, axis=(1, 2)))[0]) - 6, ct.shape[0] - 1]
    context_z = sorted(set(max(0, int(z)) for z in context_z))
    for z in context_z:
        grid = core.plane_grid(bounds, 'axial', z, max_side=210)
        idx = tuple(np.moveaxis(grid, -1, 0))
        coords = core.to_ras(grid, affine)
        rgba = np.concatenate((core.window_rgb(ct[idx], SETTINGS['level'], SETTINGS['window']) / 255.,
                               np.full(grid.shape[:2] + (1,), .20)), axis=-1)
        ax.plot_surface(*np.moveaxis(coords, -1, 0), facecolors=rgba, shade=False, rstride=1, cstride=1,
                        antialiased=False, linewidth=0, zorder=1)
    wire_box(ax, low, high)
    ax.plot_trisurf(*xyz.T, triangles=faces, color=core.FG_COLOR, alpha=.42, edgecolor='none', linewidth=0, shade=True, zorder=3)
    occupied = np.flatnonzero(np.any(gt, axis=(1, 2)))
    zs = sorted(set([int(occupied[0]), int(occupied[-1])] + [53, 56, 60, 64]))
    zs = [z for z in zs if 0 <= z < gt.shape[0]]
    for z in zs:
        for yx in find_contours(core.oriented_slice(gt, 'axial', z).astype(np.uint8), .5):
            coords = core.to_ras(pres.pixels_to_dhw(yx[:, ::-1], gt.shape, 'axial', z), affine)
            ax.plot(*coords.T, color=core.GT_COLOR, linewidth=.9, linestyle=(0, (4, 3)), zorder=5)
    frame(ax, low, high, CAMERA)
    scale = scale_bar(ax, low, high)
    fig.suptitle(title, color='white', fontsize=14, y=.97)
    fig.text(.5, .023, 'Complete mask inside the FULL original CT field of view\n400.4 x 400.4 x 445.0 mm | Same affine and scale; no organ-only zoom',
             color='#E0EAF0', ha='center', fontsize=9)
    fig.savefig(path, dpi=180, facecolor=pres.DARK)
    plt.close(fig)
    return dict(vertices=len(vertices), faces=len(faces), complete_mask=True, smoothing=False, marching_cubes_step=1,
                ct_context_axial_indices=context_z, ct_context_fov='full original XY', full_volume_limits_ras_mm=[low.tolist(), high.tolist()],
                camera=dict(CAMERA), scale_bar=scale, scale_provenance=scale_provenance(mask, np.linalg.norm(affine[:3, :3], axis=0)))


def assemble(output, files, name):
    from PIL import Image
    images = [Image.open(output / file).convert('RGB') for file in files]
    try:
        panel = Image.new('RGB', (sum(p.width for p in images), max(p.height for p in images)), pres.DARK)
        x = 0
        for image in images:
            panel.paste(image, (x, 0))
            x += image.width
        panel.save(output / name)
    finally:
        for image in images:
            image.close()


def render(args):
    root, output = Path(args.root).resolve(), Path(args.output).resolve()
    proof = root / 'figures_v1/render_manifest.json'
    binding = bind_sources(root, proof)
    ct, gt, affine, pair = core.load_inputs(root, Path(__file__).resolve().parents[1])
    if pair['v10_2']['meta']['case_id'] != 'amos_0167':
        raise ValueError('Only amos_0167 may be rendered')
    inputs_before = input_inventory(pair)
    sources = {str(Path(module.__file__).resolve()): core.file_sha(module.__file__) for module in (core, pres, surfaces)}
    coronal = int(np.argmax(np.count_nonzero(gt, axis=(0, 1))))
    sagittal = int(np.argmax(np.count_nonzero(gt, axis=(0, 2))))
    spec = fixed_scene(ct.shape, affine, coronal, sagittal)
    core.reserve_output(output)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from skimage.measure import find_contours, marching_cubes
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    core.configure_colors('#F5B83B', '#F6A0A0', '#70E5FF')
    layout = draw_block(plt, find_contours, ct, gt, affine, None, spec, 'Original CT | Three orthogonal faces', output / 'raw_ct_block.png')
    meshes = {}
    for method in core.METHODS:
        mask = pair[method]['arrays']['prediction_dhw']
        draw_block(plt, find_contours, ct, gt, affine, mask, spec, core.NAMES[method] + ' | CT + prediction', output / (method + '_block.png'))
        meshes[method] = draw_full_surface(plt, find_contours, marching_cubes, ct, gt, affine, mask,
            core.NAMES[method] + ' | Complete 3D mask at CT scale', output / (method + '_full_ct_surface.png'))
    assemble(output, ['raw_ct_block.png', 'v10_2_block.png', 'native_sam2_block.png'], 'three_face_comparison.png')
    assemble(output, ['v10_2_full_ct_surface.png', 'native_sam2_full_ct_surface.png'], 'full_ct_surface_comparison.png')
    if inputs_before != input_inventory(pair) or binding != bind_sources(root, proof):
        raise ValueError('Read-only inputs changed during rendering')
    if any(core.file_sha(p) != digest for p, digest in sources.items()):
        raise ValueError('Existing renderer dependency changed during rendering')
    manifest = dict(schema='amos_liver_fixed_three_face_v1', case_id='amos_0167', inference_rerun=False,
        capture_binding=binding, input_sha256=inputs_before, previous_renderer_sha256=sources, renderer_sha256=core.file_sha(__file__),
        scene=spec, projected_face_layout=layout, complete_surfaces=meshes, settings=SETTINGS,
        whole_case_metrics={m: pair[m]['meta']['metrics'] for m in core.METHODS},
        scope='Fixed three-face real CT cutaway plus complete actual masks at full original CT scale; no independent mask scaling',
        outputs={p.name: core.file_sha(p) for p in sorted(output.glob('*.png'))})
    (output / 'fixed_three_face_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf8')
    print(json.dumps(dict(output=str(output), bounds=spec['bounds_dhw'], camera=spec['camera'], complete=True)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, help='Existing read-only capture root')
    parser.add_argument('--output', required=True, help='New figure directory; existing paths refused')
    render(parser.parse_args())


if __name__ == '__main__':
    main()
