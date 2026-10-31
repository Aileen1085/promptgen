"""Additive presentation: real three-face CT cutaway and actual support strokes.

The cutaway is a display-only octant, not the complete prediction. Complete mask
surfaces and full-FOV axial results remain separate. No inference is performed.
"""
import argparse
import itertools
import json
from pathlib import Path
import shutil
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import render_amos_liver_case as core

reserve_output = core.reserve_output
DARK = "#111A20"


def choose_cutaway(gt, bounds, intersection, affine):
    bounds, cut = np.asarray(bounds, int), np.asarray(intersection, int)
    if np.any(cut < bounds[:, 0]) or np.any(cut >= bounds[:, 1]):
        raise ValueError("Actual section intersection is outside display bounds")
    candidates = []
    for signs in itertools.product((-1, 1), repeat=3):
        box = [[int(lo if sign < 0 else c), int(c + 1 if sign < 0 else hi)]
               for (lo, hi), c, sign in zip(bounds, cut, signs)]
        count = int(np.count_nonzero(gt[tuple(slice(lo, hi) for lo, hi in box)]))
        candidates.append(dict(signs_dhw=list(signs), bounds_dhw=box, retained_gt_voxels=count))
    # Deterministic lexicographic sign tie-breaking; no prediction enters selection.
    best = max(candidates, key=lambda candidate: candidate["retained_gt_voxels"]).copy()
    if np.any(np.diff(np.asarray(best["bounds_dhw"]), axis=1) < 2):
        raise ValueError("Selected octant is degenerate at the actual section intersection")
    basis = np.asarray(affine)[:3, :3]
    basis = basis / np.linalg.norm(basis, axis=0)
    outward = -basis * np.asarray(best["signs_dhw"])
    direction = outward.sum(1)
    direction /= np.linalg.norm(direction)
    best.update(intersection_dhw=cut.tolist(),
                camera=dict(elev=float(np.degrees(np.arcsin(direction[2]))),
                            azim=float(np.degrees(np.arctan2(direction[1], direction[0]))), projection="ortho"),
                outward_normals_ras=outward.T.tolist(), octant_gt_counts=candidates)
    return best


def face_grids(spec):
    return {plane: core.plane_grid(spec["bounds_dhw"], plane, spec["intersection_dhw"][axis],
                                   max_side=max(hi - lo for lo, hi in spec["bounds_dhw"]))
            for plane, axis in core.AXIS.items()}


def visible_edges(spec):
    corners = np.array(list(itertools.product(*[(lo, hi - 1) for lo, hi in spec["bounds_dhw"]])))
    cut = np.asarray(spec["intersection_dhw"])
    return np.array([[a, b] for a, b in itertools.combinations(corners, 2)
                     if np.count_nonzero(a != b) == 1 and np.any((a == b) & (a == cut))])


def support_strokes(points, shape, plane, index):
    pixels = {tuple(p) for p in core.project_points(points, shape, plane, index).astype(int)}
    edges, connected = [], set()
    for x, y in sorted(pixels):
        for dx, dy in ((1, 0), (0, 1), (1, 1), (-1, 1)):
            q = (x + dx, y + dy)
            if q in pixels:
                edges.append([(x, y), q])
                connected.update(((x, y), q))
    return (np.asarray(edges, int).reshape(-1, 2, 2),
            np.asarray(sorted(pixels - connected), int).reshape(-1, 2))


def pixels_to_dhw(xy, shape, plane, index):
    x, y = np.asarray(xy, float).T
    fixed = np.full(len(x), index)
    if plane == "axial":
        return np.column_stack((fixed, shape[1] - 1 - x, shape[2] - 1 - y))
    d = shape[0] - 1 - y
    return np.column_stack((d, shape[1] - 1 - x, fixed) if plane == "coronal"
                           else (d, fixed, shape[2] - 1 - x))


def clip_polyline(points, bounds):
    """Clip actual contour segments; never close them along a display-crop edge."""
    low, high = np.array(bounds, float, copy=True).T
    high -= 1
    result, current = [], []
    for p, q in zip(points[:-1], points[1:]):
        delta, enter, leave = q - p, 0., 1.
        for axis in range(3):
            if abs(delta[axis]) < 1e-12:
                if p[axis] < low[axis] or p[axis] > high[axis]:
                    enter, leave = 1., 0.
                    break
            else:
                a, b = sorted(((low[axis] - p[axis]) / delta[axis], (high[axis] - p[axis]) / delta[axis]))
                enter, leave = max(enter, a), min(leave, b)
        if enter >= leave:
            if current:
                result.append(np.array(current))
                current = []
            continue
        a, b = p + enter * delta, p + leave * delta
        if current and not np.allclose(current[-1], a):
            result.append(np.array(current))
            current = []
        if not current:
            current.append(a)
        current.append(b)
    if current:
        result.append(np.array(current))
    return result


def validate_axial_manifest(manifest, case_id, axial_z):
    if manifest["case_id"] != case_id or manifest["view"]["axial_z"] != axial_z:
        raise ValueError("Existing axial figures must match the capture case and shared slice")


def validate_capture_binding(manifest, root):
    actual = {}
    for method in core.METHODS:
        expected = manifest.get("sources", {}).get(method, {})
        actual[method] = {}
        for name, field in (("capture.json", "capture_json_sha256"), ("capture.npz", "capture_npz_sha256")):
            digest = core.file_sha(Path(root) / method / name)
            if expected.get(field) != digest:
                raise ValueError(f"Axial source capture hash mismatch: {method}/{name}")
            actual[method][field] = digest
    return actual


def draw_prompt(plt, ct, arrays, spacing, plane, index, title, path, settings, crop=None):
    from matplotlib.collections import LineCollection

    fig, ax = plt.subplots(figsize=(5.4, 5.7))
    fig.subplots_adjust(left=.02, right=.98, bottom=.07, top=.91)
    image = core.oriented_slice(ct, plane, index)
    row, col = core.slice_spacing(spacing, plane)
    ax.imshow(core.window_rgb(image, settings["level"], settings["window"]), origin="upper",
              interpolation="nearest", aspect=row / col)
    stats = {}
    for role, color in (("fg", core.FG_COLOR), ("bg", core.BG_COLOR)):
        key = f"{plane}_{role}_dhw" if plane != "axial" else f"prompt_{role}_dhw"
        points = arrays[key]
        xy = core.project_points(points, ct.shape, plane, index)
        if plane == "axial":
            ax.scatter(*xy.T, s=32, c=color, edgecolors="white", linewidths=.6, zorder=4)
            stats[role] = dict(points_on_plane=len(xy))
        else:
            segments, isolated = support_strokes(points, ct.shape, plane, index)
            ax.add_collection(LineCollection(segments, colors=[color], linewidths=.75,
                                              capstyle="round", joinstyle="round", zorder=4))
            if len(isolated):
                ax.scatter(*isolated.T, s=1.5, c=color, marker="o", linewidths=0, zorder=4)
            stats[role] = dict(actual_support_pixels=len(xy), adjacent_edges=len(segments), isolated_pixels=len(isolated))
    if crop is None:
        ax.set_xlim(-.5, image.shape[1] - .5)
        ax.set_ylim(image.shape[0] - .5, -.5)
    else:
        column = 1 if plane == "coronal" else 2
        ax.set_xlim(ct.shape[column] - crop[column][1] - .5, ct.shape[column] - crop[column][0] - .5)
        ax.set_ylim(ct.shape[0] - crop[0][0] - .5, ct.shape[0] - crop[0][1] - .5)
    ax.set_axis_off()
    ax.set_title(title, fontsize=13, pad=10)
    description = "Display closeup; actual support only" if crop is not None else "Full CT field of view; actual prompts only"
    fig.text(.5, .025, f"{plane.capitalize()} {index} | {description}", ha="center", fontsize=8)
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return stats


def draw_cutaway(plt, find_contours, ct, gt, affine, prediction, spec, title, path, settings):
    fig = plt.figure(figsize=(5.4, 5.7), facecolor=DARK)
    ax = fig.add_axes([0, .04, 1, .87], projection="3d", proj_type="ortho", computed_zorder=False, facecolor=DARK)
    for plane, grid in face_grids(spec).items():
        dhw = tuple(np.moveaxis(grid, -1, 0))
        image = ct[dhw]
        rgb = (core.window_rgb(image, settings["level"], settings["window"]) if prediction is None else
               core.overlay_prediction(image, prediction[dhw], settings["level"], settings["window"], settings["alpha"]))
        xyz = core.to_ras(grid, affine)
        ax.plot_surface(*np.moveaxis(xyz, -1, 0), facecolors=rgb / 255., rstride=1, cstride=1,
                        linewidth=0, shade=False, antialiased=False, zorder=2)
        if prediction is not None:
            axis = core.AXIS[plane]
            index = spec["intersection_dhw"][axis]
            truth = core.oriented_slice(gt, plane, index)
            for yx in find_contours(truth.astype(np.uint8), .5):
                actual = pixels_to_dhw(yx[:, ::-1], gt.shape, plane, index)
                for segment in clip_polyline(actual, spec["bounds_dhw"]):
                    coords = core.to_ras(segment, affine) + .03 * np.array(spec["outward_normals_ras"][axis])
                    ax.plot(*coords.T, color=core.GT_COLOR, linewidth=1.3, linestyle=(0, (4, 3)), zorder=5)
    edges = core.to_ras(visible_edges(spec), affine)
    for edge in edges:
        ax.plot(*edge.T, color="#CDD9DE", linewidth=.7, alpha=.8, zorder=6)
    corners = np.array(list(itertools.product(*[(lo, hi - 1) for lo, hi in spec["bounds_dhw"]])))
    physical = core.to_ras(corners, affine)
    lo, hi = physical.min(0), physical.max(0)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo)
    ax.view_init(elev=spec["camera"]["elev"], azim=spec["camera"]["azim"])
    ax.set_axis_off()
    fig.suptitle(title, color="white", fontsize=13, y=.96)
    d, h, w = spec["intersection_dhw"]
    fig.text(.5, .026, f"Display-only cutaway | D {d} / H {h} / W {w}\nActual CT sections; physical mm scale",
             ha="center", color="#D7E0E5", fontsize=8)
    fig.savefig(path, dpi=180, facecolor=DARK)
    plt.close(fig)


def assemble_panel(output, case_id):
    from PIL import Image, ImageDraw, ImageFont
    from matplotlib import font_manager

    width, height = 972, 1026
    top, footer = 90, 140
    panel = Image.new("RGB", (4 * width, 2 * height + top + footer), "white")
    rows = [["v10_2_coronal_prompts_closeup.png", "v10_2_sagittal_prompts_closeup.png",
             "v10_2_axial_prediction.png", "v10_2_result_cube.png"],
            [None, None, "native_sam2_axial_prediction.png", "native_sam2_result_cube.png"]]
    for row, files in enumerate(rows):
        for column, name in enumerate(files):
            if name is not None:
                with Image.open(output / name) as image:
                    panel.paste(image.convert("RGB"), (column * width, top + row * height))
    with Image.open(output / "native_sam2_axial_prompts.png") as image:
        panel.paste(image.convert("RGB"), (width // 2, top + height))
    font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 30)
    draw = ImageDraw.Draw(panel)
    for x, text in ((width, "Input prompts"), (int(2.5 * width), "Full-FOV axial result"), (int(3.5 * width), "3D CT cutaway")):
        draw.text((x, 28), text, fill="#172C38", font=font, anchor="mt")
    caption = (f"{case_id} | AMOS liver | v10.2-selected best case; same-case Native SAM2\n"
               "Orange: FG/prediction. Pale red: BG. Dashed cyan: GT. Prompt closeups and 3D octant are display-only.\n"
               "Axial results preserve the original full field of view. Complete 3D mask surfaces are supplied separately.")
    draw.multiline_text((28, top + 2 * height + 14), caption, fill="#344752", font=font, spacing=8)
    panel.save(output / "comparison_panel.png")


def render(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.measure import find_contours

    root, axial, output = (Path(p).resolve() for p in (args.root, args.axial_figures, args.output))
    previous = json.loads((axial / "render_manifest.json").read_text(encoding="utf-8"))
    capture_hashes = validate_capture_binding(previous, root)
    ct, gt, affine, pair = core.load_inputs(root, Path(__file__).resolve().parents[1])
    meta = pair["v10_2"]["meta"]
    z = previous["view"]["axial_z"]
    validate_axial_manifest(previous, meta["case_id"], z)
    core.configure_colors(*(previous["colors"][k] for k in ("foreground_prediction", "background", "gt")))
    settings = dict(level=previous["window"]["level_hu"], window=previous["window"]["width_hu"], alpha=previous["prediction_alpha"])
    native = pair["native_sam2"]["arrays"]
    view = core.make_view_spec(gt, meta["spacing_dhw_mm"], native["prompt_fg_dhw"], native["prompt_bg_dhw"],
                               meta["selected_coronal_w"], meta["selected_sagittal_h"], margin_mm=args.margin_mm, axial_z=z)
    cut = [z, view["sagittal_h"], view["coronal_w"]]
    spec = choose_cutaway(gt, view["display_crop_3d_dhw"], cut, affine)
    for method in core.METHODS:
        name = f"{method}_axial_prediction.png"
        if core.file_sha(axial / name) != previous["outputs"][name]:
            raise ValueError(f"Existing axial PNG checksum mismatch: {name}")
    reserve_output(output)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    draw_cutaway(plt, find_contours, ct, gt, affine, None, spec, "CT | Three-face cutaway", output / "raw_ct_cube.png", settings)
    prompt_info = {}
    for method in core.METHODS:
        arrays = pair[method]["arrays"]
        shutil.copyfile(axial / f"{method}_axial_prediction.png", output / f"{method}_axial_prediction.png")
        dice = pair[method]["meta"]["metrics"]["dice"]
        draw_cutaway(plt, find_contours, ct, gt, affine, arrays["prediction_dhw"], spec,
                      f"{core.NAMES[method]} | Dice {dice:.4f}", output / f"{method}_result_cube.png", settings)
        planes = ("coronal", "sagittal") if method == "v10_2" else ("axial",)
        for plane in planes:
            index = cut[core.AXIS[plane]]
            title = f"{core.NAMES[method]} | {plane.capitalize()} prompts"
            prompt_info[f"{method}_{plane}"] = draw_prompt(plt, ct, arrays, meta["spacing_dhw_mm"], plane, index,
                title, output / f"{method}_{plane}_prompts.png", settings)
            if method == "v10_2":
                draw_prompt(plt, ct, arrays, meta["spacing_dhw_mm"], plane, index, title,
                            output / f"{method}_{plane}_prompts_closeup.png", settings, view["display_crop_3d_dhw"])
    assemble_panel(output, meta["case_id"])
    manifest = dict(schema="amos_liver_presentation_v1", case_id=meta["case_id"], shared_view=view, cutaway=spec,
                    cutaway_scope="Display-only octant subset; complete mask surfaces supplied separately; metrics unchanged",
                    gt_contour_policy="Full real section contours clipped to exposed faces, never closed along crop borders",
                    gt_contour_outward_display_offset_mm=.03, settings=settings, colors=previous["colors"], prompts=prompt_info,
                    prompt_stroke_policy="Only 8-neighbor edges between actual support pixels; isolated supports retained",
                    prompt_closeup_bounds_dhw=view["display_crop_3d_dhw"], axial_source_manifest_sha256=core.file_sha(axial / "render_manifest.json"),
                    axial_source_png_sha256={f"{m}_axial_prediction.png": previous["outputs"][f"{m}_axial_prediction.png"] for m in core.METHODS},
                    current_capture_binding_sha256=capture_hashes,
                    capture_json_sha256={m: core.file_sha(root / m / "capture.json") for m in core.METHODS},
                    renderer_sha256=core.file_sha(__file__), outputs={p.name: core.file_sha(p) for p in sorted(output.glob("*.png"))})
    (output / "presentation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(output=str(output), cutaway=spec)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Existing capture root")
    parser.add_argument("--axial-figures", required=True, help="Verified core figure directory with render_manifest.json")
    parser.add_argument("--output", required=True, help="New presentation directory; existing paths are refused")
    parser.add_argument("--margin-mm", type=float, default=30., help="GT-derived display margin only")
    render(parser.parse_args())


if __name__ == "__main__":
    main()
