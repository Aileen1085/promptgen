"""Optional complete 3D prediction surfaces; the textured CT cubes remain primary."""
import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.render_amos_liver_case import (FG_COLOR, GT_COLOR, METHODS, NAMES, file_sha,
                                        load_inputs, make_view_spec, reserve_output, to_ras)


def prepare_volume(mask):
    """Keep every component and voxel; a zero halo closes CT-edge surfaces."""
    mask = np.asarray(mask)
    if mask.ndim != 3 or not np.isin(mask, [0, 1]).all() or not mask.any():
        raise ValueError("Expected a nonempty actual binary DHW prediction")
    return np.pad(mask.astype(np.uint8), 1)


def vertices_to_ras(vertices, affine):
    return to_ras(np.asarray(vertices, float) - 1., affine)


def display_limits(gt, affine, margin_mm=30.):
    spacing = np.linalg.norm(np.asarray(affine)[:3, :3], axis=0)
    empty = np.empty((0, 3), int)
    crop = make_view_spec(gt, spacing, empty, empty, 0, 0, margin_mm=margin_mm)["display_crop_3d_dhw"]
    corners = list(itertools.product(*[(lo, hi - 1) for lo, hi in crop]))
    physical = to_ras(corners, affine)
    return physical.min(0), physical.max(0)


def render(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.measure import find_contours, marching_cubes

    root, output = Path(args.root).resolve(), Path(args.output).resolve()
    project = Path(__file__).resolve().parents[1]
    ct, gt, affine, pair = load_inputs(root, project)
    del ct  # Only its existing readonly mapping was opened; no CT data is copied.
    native, meta = pair["native_sam2"]["arrays"], pair["v10_2"]["meta"]
    view = make_view_spec(gt, meta["spacing_dhw_mm"], native["prompt_fg_dhw"], native["prompt_bg_dhw"],
                          meta["selected_coronal_w"], meta["selected_sagittal_h"], args.elev, args.azim,
                          args.margin_mm)
    lo, hi = display_limits(gt, affine, args.margin_mm)
    if not 0 < args.alpha <= 1:
        raise ValueError("Surface alpha must be in (0,1]")
    reserve_output(output)
    # Sparse actual axial GT contours reveal shape without a dense wireframe.
    occupied = np.flatnonzero(np.any(gt, axis=(1, 2)))
    step = max(1, int(round(8. / meta["spacing_dhw_mm"][0])))
    zs = sorted(set(range(int(occupied[0]), int(occupied[-1]) + 1, step)) | {int(occupied[-1])})
    contours = []
    for z in zs:
        for hw in find_contours(np.pad(gt[z], 1).astype(np.uint8), .5):
            contours.append(to_ras(np.column_stack((np.full(len(hw), z), hw - 1.)), affine))
    details = {}
    for method in METHODS:
        mask = pair[method]["arrays"]["prediction_dhw"]
        vertices, faces, _, _ = marching_cubes(prepare_volume(mask), .5, step_size=1, allow_degenerate=False)
        xyz = vertices_to_ras(vertices, affine)
        fig = plt.figure(figsize=(5.4, 5.7))
        ax = fig.add_axes([0, .06, 1, .85], projection="3d", proj_type="ortho")
        ax.plot_trisurf(*xyz.T, triangles=faces, color=FG_COLOR, alpha=args.alpha,
                        edgecolor="none", linewidth=0, antialiased=True, shade=True)
        for coords in contours:
            ax.plot(*coords.T, color=GT_COLOR, linewidth=.85, linestyle=(0, (4, 3)), alpha=.9)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(np.maximum(hi - lo, 1.))
        ax.view_init(elev=args.elev, azim=args.azim)
        ax.set_axis_off()
        dice = pair[method]["meta"]["metrics"].get("dice")
        title = f"{NAMES[method]} | Complete 3D mask stack"
        fig.suptitle(title + (f"\nDice {dice:.4f}" if dice is not None else ""), fontsize=13, y=.97)
        fig.text(.5, .025, "Actual prediction surface | Dashed cyan: GT contours\nShared display frame; physical RAS scale",
                 ha="center", fontsize=9)
        path = output / f"{method}_complete_mask_surface.png"
        fig.savefig(path, dpi=180, facecolor="white")
        plt.close(fig)
        details[method] = dict(vertices=len(vertices), faces=len(faces), prediction_voxels=int(mask.sum()),
                               capture_json_sha256=file_sha(root / method / "capture.json"),
                               capture_npz_sha256=file_sha(root / method / "capture.npz"),
                               png_sha256=file_sha(path))
    manifest = dict(schema="amos_liver_surfaces_v1", case_id=meta["case_id"], view=view,
                    extraction="Full prediction_dhw; zero halo; marching cubes level .5; step_size=1; no smoothing",
                    mesh_source="Actual final binary masks, never logits/surrogate/GT", gt_contour_axial_indices=zs,
                    prediction_color=FG_COLOR, gt_color=GT_COLOR, surface_alpha=args.alpha,
                    display_limits_ras_mm=[lo.tolist(), hi.tolist()], methods=details,
                    renderer_sha256=file_sha(__file__))
    (output / "surface_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(output=str(output), methods=details)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True, help="New directory; existing paths are refused")
    parser.add_argument("--elev", type=float, default=24.)
    parser.add_argument("--azim", type=float, default=120.)
    parser.add_argument("--margin-mm", type=float, default=30.)
    parser.add_argument("--alpha", type=float, default=.42)
    render(parser.parse_args())


if __name__ == "__main__":
    main()
