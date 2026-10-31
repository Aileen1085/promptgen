"""Render the frozen AMOS liver capture, without importing any inference/model code.

CT is read once through its existing readonly case-cache reference. DHW means
(Z,X,Y), canonical RAS. Only figures and lightweight JSON are written. Rerenders
require a new output directory; no existing result can be silently overwritten.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


FG_COLOR = "#F5B83B"
BG_COLOR = "#F6A0A0"
GT_COLOR = "#70E5FF"
METHODS = ("v10_2", "native_sam2")
NAMES = {"v10_2": "v10.2", "native_sam2": "Native SAM2"}
AXIS = {"axial": 0, "coronal": 2, "sagittal": 1}


def configure_colors(foreground, background, gt):
    global FG_COLOR, BG_COLOR, GT_COLOR
    values = (foreground, background, gt)
    if any(not isinstance(c, str) or len(c) != 7 or c[0] != "#"
           or any(v not in "0123456789abcdefABCDEF" for v in c[1:]) for c in values):
        raise ValueError("Colors must use #RRGGBB format")
    FG_COLOR, BG_COLOR, GT_COLOR = values


def oriented_slice(volume, plane, index):
    if plane == "axial":
        return volume[index].T[::-1, ::-1]
    if plane == "coronal":
        return volume[:, :, index][::-1, ::-1]
    if plane == "sagittal":
        return volume[:, index, :][::-1, ::-1]
    raise ValueError(f"Unknown plane: {plane}")


def project_points(points, shape, plane, index):
    points = np.asarray(points).reshape(-1, 3)
    p = points[points[:, AXIS[plane]] == index]
    if plane == "axial":
        return np.column_stack((shape[1] - 1 - p[:, 1], shape[2] - 1 - p[:, 2]))
    column = 1 if plane == "coronal" else 2
    return np.column_stack((shape[column] - 1 - p[:, column], shape[0] - 1 - p[:, 0]))


def slice_spacing(spacing, plane):
    d, h, w = map(float, spacing)
    return {"axial": (w, h), "coronal": (d, h), "sagittal": (d, w)}[plane]


def color_rgb(color):
    return np.array([int(color[i:i + 2], 16) for i in (1, 3, 5)], dtype=np.uint8)


def window_rgb(image, level=50, window=350):
    if not np.isfinite(level) or not np.isfinite(window) or window <= 0:
        raise ValueError("Window must be positive and level/window finite")
    gray = np.rint(np.clip((np.asarray(image, np.float32) - level) / window + .5, 0, 1) * 255)
    return np.repeat(gray.astype(np.uint8)[..., None], 3, axis=-1)


def overlay_prediction(image, mask, level=50, window=350, alpha=.32):
    if not 0 <= alpha <= 1:
        raise ValueError("Alpha must be in [0,1]")
    rgb = window_rgb(image, level, window)
    mask = np.asarray(mask, bool)
    rgb[mask] = np.rint((1 - alpha) * rgb[mask] + alpha * color_rgb(FG_COLOR)).astype(np.uint8)
    return rgb


def prompt_rgb(ct, plane, index, fg, bg, level=50, window=350):
    rgb = window_rgb(oriented_slice(ct, plane, index), level, window)
    for points, color in ((fg, FG_COLOR), (bg, BG_COLOR)):
        xy = project_points(points, ct.shape, plane, index).astype(int)
        if len(xy):
            rgb[xy[:, 1], xy[:, 0]] = color_rgb(color)
    return rgb


def choose_axial_z(gt, native_fg, native_bg, override=None):
    if override is not None:
        if not 0 <= override < gt.shape[0]:
            raise ValueError("Axial display override is outside volume")
        return int(override), "explicit_display_override"
    fg, bg = (np.asarray(p).reshape(-1, 3) for p in (native_fg, native_bg))
    common = sorted(set(fg[:, 0]) & set(bg[:, 0]))
    area = np.count_nonzero(gt, axis=(1, 2))
    candidates = [int(z) for z in common if 0 <= z < len(area) and area[int(z)] > 0]
    if candidates:
        return max(candidates, key=lambda z: (area[z], -z)), "max_gt_area_with_native_fg_bg_points"
    return int(np.argmax(area)), "max_gt_area_fallback"


def make_view_spec(gt, spacing, fg, bg, coronal_w, sagittal_h,
                   elev=24., azim=120., margin_mm=30., axial_z=None):
    if not np.all(np.isfinite(spacing)) or np.any(np.asarray(spacing) <= 0):
        raise ValueError("Physical spacing must be finite and positive")
    if not np.all(np.isfinite([elev, azim])):
        raise ValueError("Display camera must be finite")
    if not (0 <= coronal_w < gt.shape[2] and 0 <= sagittal_h < gt.shape[1]):
        raise ValueError("Selected prompt plane is outside volume")
    if not np.any(gt) or margin_mm < 0 or not np.isfinite(margin_mm):
        raise ValueError("Nonempty liver GT and nonnegative finite display margin required")
    z, reason = choose_axial_z(gt, fg, bg, axial_z)
    crop = []
    for axis, (size, step) in enumerate(zip(gt.shape, spacing)):
        occupied = np.flatnonzero(np.any(gt, axis=tuple(i for i in range(3) if i != axis)))
        pad = int(np.ceil(margin_mm / step))
        crop.append([max(0, int(occupied[0]) - pad), min(size, int(occupied[-1]) + pad + 1)])
    return dict(axial_z=z, axial_selection=reason, coronal_w=int(coronal_w), sagittal_h=int(sagittal_h),
                axial_bounds_dhw=[[0, int(n)] for n in gt.shape], display_crop_3d_dhw=crop,
                display_crop_source="GT bbox plus physical margin; display only",
                display_margin_mm=float(margin_mm), spacing_dhw_mm=list(map(float, spacing)),
                camera=dict(elev=float(elev), azim=float(azim), projection="ortho"))


def plane_grid(bounds, plane, index, max_side=180):
    axes = [np.unique(np.rint(np.linspace(lo, hi - 1, min(max_side, hi - lo))).astype(int))
            for lo, hi in bounds]
    if plane == "axial":
        h, w = np.meshgrid(axes[1][::-1], axes[2][::-1])
        return np.stack((np.full_like(h, index), h, w), axis=-1)
    column = 1 if plane == "coronal" else 2
    c, d = np.meshgrid(axes[column][::-1], axes[0][::-1])
    fixed = np.full_like(d, index)
    return np.stack((d, c, fixed) if plane == "coronal" else (d, fixed, c), axis=-1)


def to_ras(points, affine):
    affine = np.asarray(affine, float)
    return np.asarray(points) @ affine[:3, :3].T + affine[:3, 3]


def reserve_output(path):
    Path(path).mkdir(parents=True, exist_ok=False)


def validate_pair(pair):
    ref = pair[METHODS[0]]["meta"]
    for method in METHODS:
        meta, arrays = pair[method]["meta"], pair[method]["arrays"]
        if meta.get("schema") != "amos_liver_capture_v1" or meta.get("method") != method:
            raise ValueError(f"Invalid capture schema/method: {method}")
        if meta.get("global_class_id") != 123 or meta.get("label_id") != 6 or meta.get("canonical_axes") != "ZXY":
            raise ValueError("Expected AMOS liver class123/local6 in canonical ZXY")
        for field in ("case_id", "shape_dhw", "spacing_dhw_mm", "affine_dhw_ras", "ct_cache_path", "label_path"):
            if meta.get(field) != ref.get(field):
                raise ValueError(f"Capture geometry/source mismatch: {field}")
        shape = tuple(meta["shape_dhw"])
        if tuple(arrays["prediction_dhw"].shape) != shape or not np.isin(arrays["prediction_dhw"], [0, 1]).all():
            raise ValueError("Prediction must be the actual binary full-volume mask")
        for name, points in arrays.items():
            if not name.endswith("_dhw") or name == "prediction_dhw":
                continue
            if points.ndim != 2 or points.shape[1] != 3 or not np.issubdtype(points.dtype, np.integer):
                raise ValueError(f"Invalid prompt coordinates: {name}")
            if np.any(points < 0) or np.any(points >= np.array(shape)):
                raise ValueError(f"Out-of-volume prompt coordinates: {name}")
            for plane, field in (("coronal", "selected_coronal_w"), ("sagittal", "selected_sagittal_h")):
                if name.startswith(plane) and np.any(points[:, AXIS[plane]] != meta[field]):
                    raise ValueError(f"Prompt coordinates disagree with selected {plane} plane")


def resolve_source(value, project):
    path = Path(value)
    return path if path.is_absolute() else project / path


def load_inputs(root, project):
    import nibabel as nib

    pair = {}
    for method in METHODS:
        folder = root / method
        meta = json.loads((folder / "capture.json").read_text(encoding="utf-8"))
        keys = ["prediction_dhw", "prompt_fg_dhw", "prompt_bg_dhw"]
        if method == "v10_2":
            keys += [f"{p}_{role}_dhw" for p in ("coronal", "sagittal") for role in ("fg", "bg")]
        # Deliberately do not decompress/copy the captured logits for rendering.
        with np.load(folder / "capture.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in keys}
        pair[method] = dict(meta=meta, arrays=arrays)
    validate_pair(pair)
    meta = pair["v10_2"]["meta"]
    ct = np.load(resolve_source(meta["ct_cache_path"], project), mmap_mode="r", allow_pickle=False)
    label = nib.as_closest_canonical(nib.load(str(resolve_source(meta["label_path"], project))))
    gt = np.asarray(label.dataobj).transpose(2, 0, 1) == 6
    affine = label.affine[:, [2, 0, 1, 3]]
    if ct.shape != tuple(meta["shape_dhw"]) or gt.shape != ct.shape:
        raise ValueError("Shared CT/GT/capture shape mismatch")
    if not np.allclose(affine, meta["affine_dhw_ras"], atol=1e-4, rtol=0):
        raise ValueError("Canonical label/capture affine mismatch")
    if not np.allclose(np.linalg.norm(affine[:3, :3], axis=0), meta["spacing_dhw_mm"], atol=1e-4, rtol=0):
        raise ValueError("Capture physical spacing/affine mismatch")
    return ct, gt, affine, pair


def save_figure(fig, path):
    fig.savefig(path, dpi=180, facecolor="white", bbox_inches=None)


def draw_2d(plt, ct, gt, arrays, spacing, plane, index, title, path, args, prompts=False):
    fig, ax = plt.subplots(figsize=(5.4, 5.7))
    fig.subplots_adjust(left=.02, right=.98, bottom=.05, top=.91)
    image = oriented_slice(ct, plane, index)
    rgb = (window_rgb(image, args.level, args.window) if prompts else
           overlay_prediction(image, oriented_slice(arrays["prediction_dhw"], plane, index),
                              args.level, args.window, args.alpha))
    # aspect corrects anisotropy while preserving every original axial pixel.
    row_mm, col_mm = slice_spacing(spacing, plane)
    ax.imshow(rgb, interpolation="nearest", origin="upper", aspect=row_mm / col_mm)
    if prompts:
        for role, color in (("fg", FG_COLOR), ("bg", BG_COLOR)):
            key = f"{plane}_{role}_dhw" if f"{plane}_{role}_dhw" in arrays else f"prompt_{role}_dhw"
            xy = project_points(arrays[key], ct.shape, plane, index)
            if len(xy):
                ax.scatter(xy[:, 0], xy[:, 1], s=30 if plane == "axial" else 4,
                           c=color, edgecolors="white" if plane == "axial" else "none",
                           linewidths=.6, marker="o", zorder=4)
    else:
        truth = oriented_slice(gt, plane, index)
        if truth.any() and not truth.all():
            ax.contour(truth, levels=[.5], colors=[GT_COLOR], linewidths=1.5, linestyles="dashed")
    ax.set_xlim(-.5, image.shape[1] - .5)
    ax.set_ylim(image.shape[0] - .5, -.5)
    ax.set_facecolor("black")
    ax.set_axis_off()
    ax.set_title(title, fontsize=13, pad=10)
    fig.text(.5, .02, f"{plane.capitalize()} index {index} | L/W {args.level:g}/{args.window:g} HU",
             ha="center", fontsize=9, color="#555555")
    save_figure(fig, path)
    plt.close(fig)


def draw_cube(plt, ct, gt, affine, arrays, view, title, path, args, mode="raw"):
    fig = plt.figure(figsize=(5.4, 5.7))
    ax = fig.add_axes([0, .04, 1, .86], projection="3d", proj_type="ortho")
    bounds = view["axial_bounds_dhw"] if mode != "result" else view["display_crop_3d_dhw"]
    planes = (("axial", view["axial_z"]), ("coronal", view["coronal_w"]), ("sagittal", view["sagittal_h"]))
    # Native prompts are actual axial points; v10 prompts are actual two-plane supports.
    for plane, index in planes:
        if not bounds[AXIS[plane]][0] <= index < bounds[AXIS[plane]][1]:
            continue
        grid = plane_grid(bounds, plane, index)
        dhw = tuple(np.moveaxis(grid, -1, 0))
        image = ct[dhw]
        rgb = (overlay_prediction(image, arrays["prediction_dhw"][dhw], args.level, args.window, args.alpha)
               if mode == "result" else window_rgb(image, args.level, args.window))
        xyz = to_ras(grid, affine)
        ax.plot_surface(xyz[..., 0], xyz[..., 1], xyz[..., 2], facecolors=rgb / 255.,
                        rstride=1, cstride=1, shade=False, linewidth=0, antialiased=False)
        if mode == "result":
            # Extract 2D contours at full resolution; map every vertex back to RAS.
            full_grid = plane_grid(bounds, plane, index, max_side=max(ct.shape))
            truth = gt[tuple(np.moveaxis(full_grid, -1, 0))]
            if truth.any() and not truth.all():
                temp, axis = plt.subplots()
                contour = axis.contour(truth, levels=[.5])
                for segment in contour.allsegs[0]:
                    if len(segment) < 2:
                        continue
                    origin = full_grid[0, 0].astype(float)
                    dc = full_grid[0, 1] - full_grid[0, 0]
                    dr = full_grid[1, 0] - full_grid[0, 0]
                    points = origin + segment[:, :1] * dc + segment[:, 1:] * dr
                    coords = to_ras(points, affine)
                    ax.plot(*coords.T, c=GT_COLOR, lw=1.4, linestyle=(0, (4, 3)), zorder=6)
                plt.close(temp)
    if mode == "prompts":
        for role, color in (("fg", FG_COLOR), ("bg", BG_COLOR)):
            points = arrays[f"prompt_{role}_dhw"]
            if len(points):
                xyz = to_ras(points, affine)
                ax.scatter(*xyz.T, c=color, s=15 if len(points) < 50 else 2.5,
                           depthshade=False, edgecolors="none", zorder=10)
    corners = np.array(list(itertools.product(*[(lo, hi - 1) for lo, hi in bounds])))
    physical = to_ras(corners, affine)
    lo, hi = physical.min(0), physical.max(0)
    for a, b in itertools.combinations(range(8), 2):
        if np.count_nonzero(corners[a] != corners[b]) == 1:
            ax.plot(*physical[[a, b]].T, c="#AEB9C4", lw=.65, alpha=.7)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(np.maximum(hi - lo, 1))
    ax.view_init(elev=view["camera"]["elev"], azim=view["camera"]["azim"])
    ax.set_axis_off()
    fig.suptitle(title, fontsize=13, y=.96)
    fig.text(.5, .025, "RAS physical scale | " + ("Shared display zoom" if mode == "result" else "Full CT field of view"),
             ha="center", fontsize=9, color="#555555")
    save_figure(fig, path)
    plt.close(fig)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw

    project = Path(__file__).resolve().parents[1]
    root, output = Path(args.root).resolve(), Path(args.output).resolve()
    ct, gt, affine, pair = load_inputs(root, project)
    selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    meta = pair["v10_2"]["meta"]
    native = pair["native_sam2"]["arrays"]
    view = make_view_spec(gt, meta["spacing_dhw_mm"], native["prompt_fg_dhw"], native["prompt_bg_dhw"],
                          meta["selected_coronal_w"], meta["selected_sagittal_h"], args.elev, args.azim,
                          args.margin_mm, args.axial_z)
    window_rgb(np.zeros((1, 1)), args.level, args.window)
    if not 0 <= args.alpha <= 1:
        raise ValueError("Alpha must be in [0,1]")
    reserve_output(output)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    draw_cube(plt, ct, gt, affine, None, view, f"{meta['case_id']} | CT orthogonal planes",
              output / "raw_ct_cube.png", args)
    for method in METHODS:
        capture = pair[method]
        arrays, method_meta = capture["arrays"], capture["meta"]
        name = NAMES[method]
        dice = method_meta["metrics"].get("dice")
        score = f" | Dice {dice:.4f}" if dice is not None else ""
        draw_cube(plt, ct, gt, affine, arrays, view, f"{name} | Actual input prompts",
                  output / f"{method}_prompts_cube.png", args, "prompts")
        draw_2d(plt, ct, gt, arrays, meta["spacing_dhw_mm"], "axial", view["axial_z"],
                name + score, output / f"{method}_axial_prediction.png", args)
        draw_cube(plt, ct, gt, affine, arrays, view, f"{name} | CT + prediction" + score,
                  output / f"{method}_result_cube.png", args, "result")
        for plane in (("coronal", "sagittal") if method == "v10_2" else ("axial",)):
            index = view[{"axial": "axial_z", "coronal": "coronal_w", "sagittal": "sagittal_h"}[plane]]
            draw_2d(plt, ct, gt, arrays, meta["spacing_dhw_mm"], plane, index,
                    f"{name} | Actual {plane} prompts", output / f"{method}_{plane}_prompts.png", args, True)
    # Lossless panel assembly keeps identical individual panel dimensions.
    panel_files = [output / f"{method}_{suffix}.png" for method in METHODS
                   for suffix in ("prompts_cube", "axial_prediction", "result_cube")]
    with Image.open(panel_files[0]) as first:
        width, height = first.size
    panel = Image.new("RGB", (3 * width, 2 * height + 76), "white")
    for i, path in enumerate(panel_files):
        with Image.open(path) as tile:
            panel.paste(tile.convert("RGB"), ((i % 3) * width, (i // 3) * height))
    text = (f"{meta['case_id']} | AMOS liver | v10.2-selected best case; same-case Native SAM2\n"
            f"Orange: prediction/FG; pale red: BG; cyan dashed: GT. Axial full FOV; shared 3D display zoom. "
            f"L/W={args.level:g}/{args.window:g} HU; alpha={args.alpha:g}.")
    ImageDraw.Draw(panel).multiline_text((16, 2 * height + 12), text, fill="#303B44", spacing=7)
    panel.save(output / "comparison_panel.png")
    manifest = dict(schema="amos_liver_render_v1", case_id=meta["case_id"], selection=selection,
                    selection_sha256=file_sha(root / "selection.json"), view=view,
                    window=dict(level_hu=args.level, width_hu=args.window), prediction_alpha=args.alpha,
                    colors=dict(foreground_prediction=FG_COLOR, background=BG_COLOR, gt=GT_COLOR),
                    orientation=dict(axial="array[z].T[::-1,::-1]", coronal="array[:,:,w][::-1,::-1]",
                                     sagittal="array[:,h,:][::-1,::-1]", coordinates="DHW=(Z,X,Y), RAS"),
                    sources={method: dict(capture_json_sha256=file_sha(root / method / "capture.json"),
                                          capture_npz_sha256=file_sha(root / method / "capture.npz"),
                                          metrics=pair[method]["meta"]["metrics"],
                                          threshold_probability=pair[method]["meta"].get("threshold_probability"))
                             for method in METHODS},
                    renderer_sha256=file_sha(__file__), ct_cache_reference=meta["ct_cache_path"],
                    label_reference=meta["label_path"], full_mask_source="prediction_dhw (actual final capture)",
                    cube_geometry="Three actual orthogonal CT sections and box edges; no invented anatomy",
                    outputs={p.name: file_sha(p) for p in sorted(output.glob("*.png"))})
    (output / "render_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(output=str(output), axial_z=view["axial_z"], images=len(manifest["outputs"]))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Capture root containing selection.json and both method directories")
    parser.add_argument("--output", required=True, help="New figure directory (existing paths are refused)")
    parser.add_argument("--level", type=float, default=50.)
    parser.add_argument("--window", type=float, default=350.)
    parser.add_argument("--alpha", type=float, default=.32)
    parser.add_argument("--elev", type=float, default=24.)
    parser.add_argument("--azim", type=float, default=120.)
    parser.add_argument("--axial-z", type=int, help="Explicit display-only axial override")
    parser.add_argument("--margin-mm", type=float, default=30., help="GT-framed display-only 3D margin; 2D always full FOV")
    parser.add_argument("--foreground-color", default=FG_COLOR, help="Foreground/prediction #RRGGBB")
    parser.add_argument("--background-color", default=BG_COLOR, help="Background prompt #RRGGBB")
    parser.add_argument("--gt-color", default=GT_COLOR, help="Dashed ground-truth contour #RRGGBB")
    args = parser.parse_args()
    configure_colors(args.foreground_color, args.background_color, args.gt_color)
    render(args)


if __name__ == "__main__":
    main()
