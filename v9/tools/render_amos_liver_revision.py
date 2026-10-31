"""Saved-mask-only AMOS 0167 display revision; no model imports or inference."""
import argparse
import heapq
import itertools
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import render_amos_liver_case as core
from tools import render_amos_liver_presentation as pres
from tools import render_amos_liver_surfaces as surfaces

SETTINGS = dict(level=50., window=350., alpha=.32)
NEIGHBORS = [(x, y) for x in (-1, 0, 1) for y in (-1, 0, 1) if x or y]


def bind_sources(root, proof):
    manifest = json.loads(Path(proof).read_text(encoding="utf-8"))
    if manifest.get("case_id") != "amos_0167":
        raise ValueError("This revision is authorized only for amos_0167")
    return pres.validate_capture_binding(manifest, root)


def thin_pixels(pixels):
    """Zhang-Suen thinning deletes support pixels only; no coordinate is added."""
    pixels = np.unique(np.asarray(pixels, int).reshape(-1, 2), axis=0)
    if not len(pixels):
        return pixels
    low = pixels.min(0) - 1
    relative = pixels - low
    mask = np.zeros((relative[:, 1].max() + 2, relative[:, 0].max() + 2), np.uint8)
    mask[relative[:, 1], relative[:, 0]] = 1
    changed = True
    while changed:
        changed = False
        for phase in (0, 1):
            p = [mask[:-2, 1:-1], mask[:-2, 2:], mask[1:-1, 2:], mask[2:, 2:],
                 mask[2:, 1:-1], mask[2:, :-2], mask[1:-1, :-2], mask[:-2, :-2]]
            count = sum(p)
            transitions = sum((a == 0) & (b == 1) for a, b in zip(p, p[1:] + p[:1]))
            first = p[0] * p[2] * (p[4] if phase == 0 else p[6])
            second = (p[2] if phase == 0 else p[0]) * p[4] * p[6]
            remove = (mask[1:-1, 1:-1] == 1) & (count >= 2) & (count <= 6) & (transitions == 1) & (first == 0) & (second == 0)
            if remove.any():
                mask[1:-1, 1:-1][remove] = 0
                changed = True
    return np.argwhere(mask)[:, ::-1] + low


def main_line(pixels, row_mm, col_mm):
    actual = set(map(tuple, np.asarray(pixels, int).reshape(-1, 2)))
    nodes = set(map(tuple, thin_pixels(list(actual))))
    graph = {p: [(q, float(np.hypot(dx * col_mm, dy * row_mm))) for dx, dy in NEIGHBORS
                 if (q := (p[0] + dx, p[1] + dy)) in nodes] for p in nodes}
    starts, remaining = [], set(nodes)
    while remaining:
        component, pending = set(), [min(remaining)]
        while pending:
            node = pending.pop()
            if node not in component:
                component.add(node)
                pending.extend(neighbor for neighbor, _ in graph[node] if neighbor not in component)
        remaining -= component
        endpoints = sorted(p for p in component if len(graph[p]) <= 1)
        starts.extend(endpoints or sorted(component))
    best, length = [], -1.
    # Exact longest shortest path among skeleton endpoints; cycles use every node.
    for start in starts:
        dist, previous, queue = {start: 0.}, {}, [(0., start)]
        while queue:
            cost, node = heapq.heappop(queue)
            if cost != dist[node]:
                continue
            for neighbor, weight in graph[node]:
                candidate = cost + weight
                if candidate < dist.get(neighbor, float("inf")):
                    dist[neighbor], previous[neighbor] = candidate, node
                    heapq.heappush(queue, (candidate, neighbor))
        end = max(sorted(dist), key=lambda node: dist[node])
        if dist[end] > length:
            path = [end]
            while path[-1] != start:
                path.append(previous[path[-1]])
            best, length = path[::-1], dist[end]
    line = np.asarray(best, int).reshape(-1, 2)
    return line, dict(actual_support_pixels=len(actual), skeleton_pixels=len(nodes), display_line_pixels=len(line),
                      path_length_mm=max(length, 0.), display_only=True, all_line_nodes_in_actual_support=all(p in actual for p in best))


def context_bounds(gt, spacing, context=6, margin_mm=30.):
    if context < 0 or margin_mm < 0 or not gt.any():
        raise ValueError("Nonempty GT and nonnegative display context required")
    occupied = [np.flatnonzero(np.any(gt, axis=tuple(a for a in range(3) if a != axis))) for axis in range(3)]
    target = [[int(p[0]), int(p[-1]) + 1] for p in occupied]
    padding = [context] + [int(np.ceil(margin_mm / s)) for s in spacing[1:]]
    bounds = [[max(0, lo - pad), min(size, hi + pad)] for (lo, hi), pad, size in zip(target, padding, gt.shape)]
    return bounds, dict(gt_bounds_dhw=target, gt_depth_halfopen=target[0], display_depth_halfopen=bounds[0],
                        requested_context_slices_each_side=context,
                        available_context_below_above=[target[0][0] - bounds[0][0], bounds[0][1] - target[0][1]])


def prompt_frame(shape, spacing, bounds, arrays):
    depth = list(bounds[0])
    intervals, centers = {}, {}
    for plane, axis in (("coronal", 1), ("sagittal", 2)):
        support = np.concatenate([arrays[f"{plane}_{role}_dhw"] for role in ("fg", "bg")])
        lo, hi = bounds[axis]
        if len(support):
            lo, hi = min(lo, int(support[:, axis].min())), max(hi, int(support[:, axis].max()) + 1)
            depth = [min(depth[0], int(support[:, 0].min())), max(depth[1], int(support[:, 0].max()) + 1)]
        intervals[plane] = [lo, hi]
        centers[plane] = (shape[axis] - (lo + hi) / 2. - .5) * spacing[axis]
    width = max((hi - lo) * spacing[1 if plane == "coronal" else 2] for plane, (lo, hi) in intervals.items())
    return dict(depth_halfopen=depth, physical_height_mm=(depth[1] - depth[0]) * spacing[0],
                physical_width_mm=float(width), centers_from_display_left_mm=centers, native_horizontal_bounds=intervals,
                aspect="equal", png_size=[1152, 1152], padding="Black canvas where common physical width exceeds CT coverage")


def paired_scores(gt, v10, native, axis=0):
    other = tuple(a for a in range(3) if a != axis)
    area = np.count_nonzero(gt, axis=other)
    rows = [dict(index=int(i), gt_area_voxels=int(n)) for i, n in enumerate(area)]
    for name, prediction in (("v10", v10), ("native", native)):
        tp = np.count_nonzero(gt & prediction.astype(bool), axis=other)
        amount = np.count_nonzero(prediction, axis=other)
        for i, row in enumerate(rows):
            row[f"{name}_dice"] = float(2 * tp[i] / (area[i] + amount[i])) if area[i] + amount[i] else 1.
            row[f"{name}_recall"] = float(tp[i] / area[i]) if area[i] else 0.
    for row in rows:
        row["dice_gap"] = row["v10_dice"] - row["native_dice"]
    return rows


def select_results(gt, v10, native, spacing_z, count=4, min_spacing_mm=15.):
    rows = paired_scores(gt, v10, native)
    cutoff = .1 * max(r["gt_area_voxels"] for r in rows)
    candidates = [r for r in rows if r["gt_area_voxels"] >= cutoff and r["v10_dice"] >= .95]
    ordered = sorted(candidates, key=lambda r: (r["native_recall"], -r["dice_gap"], -r["gt_area_voxels"], r["index"]))
    selected = []
    for row in ordered:
        if all(abs(row["index"] - z) * spacing_z >= min_spacing_mm for z in selected):
            selected.append(row["index"])
        if len(selected) >= count:
            break
    fallback = None
    if not selected:
        selected = [max(rows, key=lambda r: r["gt_area_voxels"])["index"]]
        fallback = "No eligible high-v10 slice; show max-GT-area slice with actual scores"
    elif len(selected) < count:
        fallback = "Fewer eligible spatially separated slices than requested; criteria retained"
    policy = dict(qualitative_selection=True, min_gt_area_fraction=.1, v10_min_slice_dice=.95,
                  rank="native recall ascending, Dice gap descending, GT area descending, z ascending",
                  min_z_spacing_mm=min_spacing_mm, requested_count=count, fallback=fallback)
    return sorted(selected), policy, rows


def native_frames(gt, fg, bg, results, count=5):
    available = sorted(set(np.asarray(fg)[:, 0]) & set(np.asarray(bg)[:, 0]))
    available = [int(z) for z in available if gt[int(z)].any()]
    if not available:
        raise ValueError("No axial frame has both actual native FG and BG points")
    anchor = max(available, key=lambda z: (np.count_nonzero(gt[z]), -z))
    selected = list(dict.fromkeys([z for z in results if z in available] + [anchor]))[:count]
    for position in np.linspace(0, len(available) - 1, count):
        remaining = [z for z in available if z not in selected]
        if len(selected) >= count or not remaining:
            break
        selected.append(min(remaining, key=lambda z: abs(available.index(z) - position)))
    return sorted(selected), dict(available_actual_fg_bg_frames=available, result_anchors=[z for z in results if z in available], max_gt_anchor=anchor)


def scene_spec(gt, v10, native, affine, bounds, coronal_w, sagittal_h):
    choices = []
    for signs in itertools.product((-1, 1), repeat=2):
        box = [list(bounds[0])] + [[lo if sign < 0 else cut, cut + 1 if sign < 0 else hi]
                                   for (lo, hi), cut, sign in zip(bounds[1:], (sagittal_h, coronal_w), signs)]
        choices.append((int(np.count_nonzero(gt[tuple(slice(lo, hi) for lo, hi in box)])), signs, box))
    retained, signs, box = max(choices, key=lambda item: item[0])
    spacing = np.linalg.norm(np.asarray(affine)[:3, :3], axis=0)
    spec = dict(bounds_dhw=box, intersection_dhw=[box[0][1] - 1, sagittal_h, coronal_w], retained_gt_voxels=retained)
    areas, metrics = {}, {}
    for plane, grid in pres.face_grids(spec).items():
        if plane == "axial":
            continue
        idx = tuple(np.moveaxis(grid, -1, 0))
        truth = gt[idx]
        area = float(truth.sum() * np.prod(spacing[[a for a in range(3) if a != core.AXIS[plane]]]))
        areas[plane] = area
        metrics[plane] = dict(gt_area_mm2=area)
        for name, prediction in (("v10", v10), ("native", native)):
            pred = prediction[idx].astype(bool)
            tp = np.count_nonzero(truth & pred)
            metrics[plane][f"{name}_dice"] = float(2 * tp / (truth.sum() + pred.sum())) if truth.sum() + pred.sum() else 1.
            metrics[plane][f"{name}_recall"] = float(tp / truth.sum()) if truth.sum() else 0.
    front = max(areas, key=areas.get)
    basis = np.asarray(affine)[:3, :3] / spacing
    normals = np.column_stack((basis[:, 0], -signs[0] * basis[:, 1], -signs[1] * basis[:, 2]))
    weight = np.ones(3)
    weight[0], weight[core.AXIS[front]] = 1.2, 2.4
    direction = normals @ weight
    direction /= np.linalg.norm(direction)
    spec.update(front_face=front, visible_face_pair_metrics=metrics, outward_normals_ras=normals.T.tolist(),
                camera=dict(elev=float(np.degrees(np.arcsin(direction[2]))), azim=float(np.degrees(np.arctan2(direction[1], direction[0]))), projection="ortho"),
                scope="Full target Z plus real CT context; XY quadrant is display-only", signs_hw=list(signs))
    return spec


def choose_scenes(gt, v10, native, affine, bounds, coronal_override=None, sagittal_override=None):
    tables, largest, contrast = {}, {}, {}
    for plane, fraction in (("coronal", .8), ("sagittal", .65)):
        axis = core.AXIS[plane]
        rows = paired_scores(gt, v10, native, axis)
        largest[plane] = max(rows, key=lambda row: row["gt_area_voxels"])["index"]
        cutoff = fraction * max(row["gt_area_voxels"] for row in rows)
        candidates = [row for row in rows if row["gt_area_voxels"] >= cutoff and row["v10_dice"] >= .95]
        contrast[plane] = min(candidates, key=lambda row: (row["native_recall"], -row["dice_gap"], -row["gt_area_voxels"], row["index"]))["index"] if candidates else largest[plane]
        tables[plane] = dict(rows=rows, contrast_min_gt_area_fraction=fraction, contrast_v10_min_dice=.95)
    for plane, override in (("coronal", coronal_override), ("sagittal", sagittal_override)):
        if override is not None:
            if not bounds[core.AXIS[plane]][0] <= override < bounds[core.AXIS[plane]][1]:
                raise ValueError("Display plane override lies outside the target/context frame")
            contrast[plane] = int(override)
    scenes = {}
    for name, chosen in (("largest_area", largest), ("native_underseg", contrast)):
        spec = scene_spec(gt, v10, native, affine, bounds, chosen["coronal"], chosen["sagittal"])
        spec["whole_plane_pair_metrics"] = {p: tables[p]["rows"][chosen[p]] for p in chosen}
        spec["selection"] = "Maximum GT area" if name == "largest_area" else "Low native recall among substantial-area, high-v10 planes; optional display overrides recorded"
        scenes[name] = spec
    return scenes, tables


def draw_matched_prompt(plt, ct, arrays, spacing, plane, index, frame, path):
    from matplotlib.collections import LineCollection

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    fig.subplots_adjust(left=.03, right=.97, bottom=.075, top=.91)
    image = core.oriented_slice(ct, plane, index)
    row_mm, col_mm = core.slice_spacing(spacing, plane)
    extent = [-.5 * col_mm, (image.shape[1] - .5) * col_mm, (image.shape[0] - .5) * row_mm, -.5 * row_mm]
    ax.imshow(core.window_rgb(image, SETTINGS["level"], SETTINGS["window"]), extent=extent,
              origin="upper", interpolation="nearest", aspect="equal")
    fg = core.project_points(arrays[f"{plane}_fg_dhw"], ct.shape, plane, index)
    line, info = main_line(fg, row_mm, col_mm)
    if len(line):
        ax.plot(line[:, 0] * col_mm, line[:, 1] * row_mm, color=core.FG_COLOR, linewidth=2.,
                solid_capstyle="round", solid_joinstyle="round", marker="o" if len(line) == 1 else None, markersize=2)
    edges, isolated = pres.support_strokes(arrays[f"{plane}_bg_dhw"], ct.shape, plane, index)
    ax.add_collection(LineCollection(edges * [col_mm, row_mm], colors=[core.BG_COLOR], linewidths=3., capstyle="round", joinstyle="round"))
    if len(isolated):
        ax.scatter(isolated[:, 0] * col_mm, isolated[:, 1] * row_mm, c=core.BG_COLOR, s=10, linewidths=0)
    center, width = frame["centers_from_display_left_mm"][plane], frame["physical_width_mm"]
    low, high = frame["depth_halfopen"]
    ax.set_xlim(center - width / 2., center + width / 2.)
    ax.set_ylim((ct.shape[0] - low - .5) * row_mm, (ct.shape[0] - high - .5) * row_mm)
    ax.set_facecolor("black")
    ax.set_axis_off()
    ax.set_title(f"v10.2 | {plane.capitalize()} prompts", fontsize=14, pad=12)
    fig.text(.5, .025, f"Plane {index} | Shared physical field: {width:.1f} x {frame['physical_height_mm']:.1f} mm",
             ha="center", fontsize=9)
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    info.update(display_xy_pixels=line.tolist(), display_dhw=pres.pixels_to_dhw(line, ct.shape, plane, index).astype(int).tolist(),
                foreground_linewidth_pt=2., background_linewidth_pt=3., background_actual_support_pixels=len(arrays[f"{plane}_bg_dhw"]))
    return info


def physical_view_limits(affine, bounds, headroom_mm=12.):
    corners = np.array(list(itertools.product(*[(lo, hi - 1) for lo, hi in bounds])))
    xyz = core.to_ras(corners, affine)
    lo, hi = xyz.min(0), xyz.max(0)
    lo -= [4., 4., headroom_mm]
    hi += [4., 4., headroom_mm]
    return lo, hi


def axes_frame(ax, affine, bounds, camera, headroom_mm=12.):
    lo, hi = physical_view_limits(affine, bounds, headroom_mm)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo)
    ax.view_init(elev=camera["elev"], azim=camera["azim"])
    ax.set_axis_off()


def draw_context_cube(plt, find_contours, ct, gt, affine, prediction, spec, title, path):
    fig = plt.figure(figsize=(5.4, 6.2), facecolor=pres.DARK)
    ax = fig.add_axes([0, .04, 1, .88], projection="3d", proj_type="ortho", computed_zorder=False, facecolor=pres.DARK)
    for plane, grid in pres.face_grids(spec).items():
        idx = tuple(np.moveaxis(grid, -1, 0))
        image = ct[idx]
        rgb = (core.window_rgb(image, SETTINGS["level"], SETTINGS["window"]) if prediction is None else
               core.overlay_prediction(image, prediction[idx], SETTINGS["level"], SETTINGS["window"], SETTINGS["alpha"]))
        xyz = core.to_ras(grid, affine)
        ax.plot_surface(*np.moveaxis(xyz, -1, 0), facecolors=rgb / 255., rstride=1, cstride=1,
                        shade=False, antialiased=False, linewidth=0, zorder=2)
        if prediction is not None:
            axis = core.AXIS[plane]
            index = spec["intersection_dhw"][axis]
            for yx in find_contours(core.oriented_slice(gt, plane, index).astype(np.uint8), .5):
                for segment in pres.clip_polyline(pres.pixels_to_dhw(yx[:, ::-1], gt.shape, plane, index), spec["bounds_dhw"]):
                    coords = core.to_ras(segment, affine) + .03 * np.array(spec["outward_normals_ras"][axis])
                    ax.plot(*coords.T, c=core.GT_COLOR, lw=1.4, linestyle=(0, (4, 3)), zorder=5)
    for edge in core.to_ras(pres.visible_edges(spec), affine):
        ax.plot(*edge.T, c="#CBD9E0", lw=.7, zorder=6)
    axes_frame(ax, affine, spec["bounds_dhw"], spec["camera"])
    fig.suptitle(title, y=.96, color="white", fontsize=13)
    lo, hi = spec["bounds_dhw"][0]
    fig.text(.5, .025, f"Complete target depth + real CT context: D {lo}–{hi - 1}\nShared physical scale; display-only XY cutaway",
             color="#DCE7ED", fontsize=8, ha="center")
    fig.savefig(path, dpi=180, facecolor=pres.DARK)
    plt.close(fig)


def draw_surface_context(plt, find_contours, marching_cubes, ct, gt, affine, mask, bounds, camera, title, path, result_zs):
    vertices, faces, _, _ = marching_cubes(surfaces.prepare_volume(mask), .5, step_size=1, allow_degenerate=False)
    xyz = surfaces.vertices_to_ras(vertices, affine)
    fig = plt.figure(figsize=(5.4, 6.2), facecolor=pres.DARK)
    ax = fig.add_axes([0, .04, 1, .88], projection="3d", proj_type="ortho", computed_zorder=False, facecolor=pres.DARK)
    context_planes = [bounds[0][0], bounds[0][1] - 1]
    for z in context_planes:
        grid = core.plane_grid(bounds, "axial", z, max_side=220)
        idx = tuple(np.moveaxis(grid, -1, 0))
        coords = core.to_ras(grid, affine)
        rgba = np.concatenate((core.window_rgb(ct[idx], SETTINGS["level"], SETTINGS["window"]) / 255.,
                               np.full(grid.shape[:2] + (1,), .22)), axis=-1)
        ax.plot_surface(*np.moveaxis(coords, -1, 0), facecolors=rgba, shade=False, rstride=1, cstride=1,
                        antialiased=False, linewidth=0, zorder=1)
    ax.plot_trisurf(*xyz.T, triangles=faces, color=core.FG_COLOR, alpha=.48, edgecolor="none", lw=0, shade=True, zorder=3)
    occupied = np.flatnonzero(np.any(gt, axis=(1, 2)))
    contour_zs = sorted(set(result_zs) | {int(occupied[0]), int(occupied[-1])})
    for z in contour_zs:
        for yx in find_contours(core.oriented_slice(gt, "axial", z).astype(np.uint8), .5):
            coords = core.to_ras(pres.pixels_to_dhw(yx[:, ::-1], gt.shape, "axial", z), affine)
            ax.plot(*coords.T, c=core.GT_COLOR, lw=.9, linestyle=(0, (4, 3)), zorder=5)
    axes_frame(ax, affine, bounds, camera)
    fig.suptitle(title, color="white", y=.96, fontsize=13)
    fig.text(.5, .025, f"Complete actual mask | Real CT context at D {context_planes[0]} / {context_planes[1]}\nDashed cyan: actual GT contours; no mask smoothing",
             color="#DCE7ED", fontsize=8, ha="center")
    fig.savefig(path, dpi=180, facecolor=pres.DARK)
    plt.close(fig)
    return dict(vertices=len(vertices), faces=len(faces), complete_mask_extracted=True, context_axial_planes=context_planes,
                context_opacity=.22, surface_alpha=.48, gt_contour_zs=contour_zs, smoothing=False, marching_cubes_step_size=1)


def panel_grid(output, rows, name, headers, caption):
    from PIL import Image, ImageDraw, ImageFont
    from matplotlib import font_manager

    first = next(file for row in rows for file in row if file)
    with Image.open(output / first) as image:
        width, height = image.size
    panel = Image.new("RGB", (width * len(headers), height * len(rows) + 175), "white")
    font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 27)
    draw = ImageDraw.Draw(panel)
    for column, header in enumerate(headers):
        draw.text((int((column + .5) * width), 22), header, fill="#203B49", font=font, anchor="mt")
    for row, files in enumerate(rows):
        for column, file in enumerate(files):
            if file:
                with Image.open(output / file) as image:
                    # Contact sheets preserve aspect; individual source PNGs stay untouched.
                    image = image.convert("RGB")
                    image.thumbnail((width, height), Image.Resampling.LANCZOS)
                    panel.paste(image, (column * width + (width - image.width) // 2, 70 + row * height + (height - image.height) // 2))
    draw.multiline_text((20, 75 + height * len(rows)), caption, fill="#344E5D", font=font, spacing=6)
    panel.save(output / name)


def render(args):
    root, output = Path(args.root).resolve(), Path(args.output).resolve()
    proof = Path(args.proof_manifest).resolve() if args.proof_manifest else root / "figures_v1/render_manifest.json"
    hashes = bind_sources(root, proof)  # All four source hashes, before data/output.
    if args.result_count < 1 or args.context < 0 or args.min_z_spacing_mm <= 0:
        raise ValueError("Positive result count/spacing and nonnegative context required")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.measure import find_contours, marching_cubes
    from types import SimpleNamespace

    ct, gt, affine, pair = core.load_inputs(root, Path(__file__).resolve().parents[1])
    if pair["v10_2"]["meta"]["case_id"] != "amos_0167":
        raise ValueError("Only the confirmed amos_0167 case may be rendered")
    core.configure_colors("#F5B83B", "#F6A0A0", "#70E5FF")
    spacing = pair["v10_2"]["meta"]["spacing_dhw_mm"]
    v10, native = (pair[m]["arrays"] for m in core.METHODS)
    bounds, depth_info = context_bounds(gt, spacing, args.context)
    frame = prompt_frame(ct.shape, spacing, bounds, v10)
    results, selection, score_rows = select_results(gt, v10["prediction_dhw"], native["prediction_dhw"], spacing[0],
                                                    args.result_count, args.min_z_spacing_mm)
    prompt_zs, prompt_policy = native_frames(gt, native["prompt_fg_dhw"], native["prompt_bg_dhw"], results)
    scenes, plane_scores = choose_scenes(gt, v10["prediction_dhw"], native["prediction_dhw"], affine, bounds,
                                         args.contrast_coronal, args.contrast_sagittal)
    core.reserve_output(output)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    lines = {}
    for plane, field in (("coronal", "selected_coronal_w"), ("sagittal", "selected_sagittal_h")):
        index = pair["v10_2"]["meta"][field]
        lines[plane] = draw_matched_prompt(plt, ct, v10, spacing, plane, index, frame, output / f"v10_2_{plane}_prompts.png")
    native_info = {}
    for z in prompt_zs:
        native_info[str(z)] = pres.draw_prompt(plt, ct, native, spacing, "axial", z, f"Native SAM2 | Actual points | z {z}",
                                               output / f"native_axial_prompts_z{z:03d}.png", SETTINGS)
    settings = SimpleNamespace(**SETTINGS)
    for z in results:
        row = score_rows[z]
        for method, prefix in (("v10_2", "v10"), ("native_sam2", "native")):
            title = f"{core.NAMES[method]} | z {z}\nSlice Dice {row[prefix + '_dice']:.4f} | Recall {row[prefix + '_recall']:.4f}"
            core.draw_2d(plt, ct, gt, pair[method]["arrays"], spacing, "axial", z, title,
                         output / f"{method}_result_z{z:03d}.png", settings)
    mesh_info = {}
    for scene, spec in scenes.items():
        draw_context_cube(plt, find_contours, ct, gt, affine, None, spec, f"CT context | {scene.replace('_', ' ')}",
                          output / f"{scene}_raw_ct_cube.png")
        for method in core.METHODS:
            draw_context_cube(plt, find_contours, ct, gt, affine, pair[method]["arrays"]["prediction_dhw"], spec,
                              f"{core.NAMES[method]} | Full target depth", output / f"{scene}_{method}_cube.png")
    for method in core.METHODS:
        mesh_info[method] = draw_surface_context(plt, find_contours, marching_cubes, ct, gt, affine,
            pair[method]["arrays"]["prediction_dhw"], bounds, scenes["largest_area"]["camera"],
            f"{core.NAMES[method]} | Complete 3D mask + CT", output / f"{method}_surface_context.png", results)
    panel_grid(output, [[f"native_axial_prompts_z{z:03d}.png" if z in prompt_zs else None,
                         f"v10_2_result_z{z:03d}.png", f"native_sam2_result_z{z:03d}.png"] for z in results],
               "qualitative_panel.png", ["Actual native prompts", "v10.2", "Native SAM2"],
               "amos_0167 | Selected high-v10 / low-native-recall slices; these are per-slice scores.\nAll axial views retain full CT FOV. Orange: prediction. Dashed cyan: GT.")
    panel_grid(output, [[f"{s}_raw_ct_cube.png", f"{s}_v10_2_cube.png", f"{s}_native_sam2_cube.png"] for s in scenes],
               "3d_comparison_panel.png", ["Real CT context", "v10.2", "Native SAM2"],
               "Same camera, physical scale and real depth for paired methods.\nFull GT depth + available CT context; XY cutaway is display-only.")
    panel_grid(output, [["v10_2_coronal_prompts.png", "v10_2_sagittal_prompts.png"]],
               "matched_v10_prompts.png", ["Coronal", "Sagittal"],
               "Identical physical field and PNG size; equal scale, no stretching.\nFG: one display mainline within actual support. BG: actual support, thicker stroke.")
    panel_grid(output, [[f"native_axial_prompts_z{z:03d}.png" for z in prompt_zs]],
               "native_prompt_views.png", [f"Actual points | z {z}" for z in prompt_zs],
               "Every displayed slice has actual foreground and background points; no generated prompts.")
    manifest = dict(schema="amos_liver_revision_v1", case_id="amos_0167", capture_binding_sha256=hashes,
        proof_manifest_sha256=core.file_sha(proof), renderer_sha256=core.file_sha(__file__), settings=SETTINGS,
        colors=dict(foreground=core.FG_COLOR, background=core.BG_COLOR, gt=core.GT_COLOR), depth=depth_info,
        axes_air_margin_mm=dict(z_each_side=12., xy_each_side=4., invented_ct_slices=0), display_bounds_dhw=bounds,
        prompt_common_frame=frame, foreground_display_lines=lines, native_prompt_frames=prompt_policy,
        displayed_native_prompt_z=prompt_zs, native_prompt_counts=native_info, result_selection=selection,
        selected_result_z=results, selected_result_metrics=[score_rows[z] for z in results], all_axial_metrics=score_rows,
        scenes=scenes, plane_metrics=plane_scores, full_mask_surface_context=mesh_info,
        whole_case_metrics={m: pair[m]["meta"]["metrics"] for m in core.METHODS},
        limitations="Qualitatively selected slices/planes, not independent representative or average performance; no re-inference",
        outputs={p.name: core.file_sha(p) for p in sorted(output.glob("*.png"))})
    (output / "revision_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(output=str(output), selected_result_z=results, native_prompt_z=prompt_zs, depth=depth_info)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True, help="New directory; existing paths refused")
    parser.add_argument("--proof-manifest", help="Known render manifest binding both capture JSON/NPZ files; default root/figures_v1/render_manifest.json")
    parser.add_argument("--result-count", type=int, default=4)
    parser.add_argument("--context", type=int, default=6, help="Requested real CT slices above/below complete target, clipped to CT bounds")
    parser.add_argument("--min-z-spacing-mm", type=float, default=15.)
    parser.add_argument("--contrast-coronal", type=int, help="Optional real display-plane override for contrast scene")
    parser.add_argument("--contrast-sagittal", type=int, help="Optional real display-plane override for contrast scene")
    render(parser.parse_args())


if __name__ == "__main__":
    main()
