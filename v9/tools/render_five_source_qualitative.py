"""Render real, already exported canonical-DHW CT/masks; never run inference.

JSON schema v1: {"schema_version":1,"cases":[{"source":"AMOS",
"case_id":"...","class_name":"...","ct_path":"existing_case.npy",
"gt_path":"gt.npy","spacing_dhw":[3,1,1],"gt_grid_id":"verified-grid",
"grid_verification":{"verified":true,"grid_id":"verified-grid",
"axes_dhw":"ZXY","evidence":"Describe physical grid check at export"},
"methods":[{"name":"PromptGen","grid_id":"verified-grid",
"mask_path":"prediction.npy","dice_3d":0.9}]}]}.

Paths resolve relative to this manifest. CT must be an existing shared case .npy;
GT and masks are binary .npy or .npz (key "mask") on that exact full CT grid.
Optional slice_index and crop_yx=[y0,y1,x0,x1] affect display only. The default
slice is the first GT-largest axial area slice; default crop is the full plane.
Optional display_transform={transpose:false,flip_y:false,flip_x:false} applies
uniformly after slicing/cropping. DHW means stored axial frame, row, column;
no R/L or patient orientation is inferred. The exporter must align every mask
to the CT grid before supplying this manifest. spacing_dhw follows array axes.
grid_verification records the exporter's physical-grid check and array-axis
order (ZXY or ZYX). GT and every prediction must reference that verified grid.
This is an upstream attestation; shape equality alone cannot establish affine
alignment. In particular, raw NRRD and canonical NIfTI may require reorientation.
The optional dice_3d must equal recomputed full-volume binary Dice (atol 1e-6).
CT and masks are opened read-only; only PNG/PDF and lightweight metadata are
written, into a new directory. PDFs embed the CT raster at 300 dpi.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont


SOURCES = ("AMOS", "MAGIC", "KiTS23", "MSD10", "MSD07")
METHODS = ("PromptGen", "nnInteractive", "SAM2", "MedSAM2", "SAM-Med3D Turbo",
           "SegVol", "VISTA3D", "PAM")
ORANGE = (255, 138, 0)
CYAN = (0, 230, 255)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_name(value):
    key = re.sub(r"[^a-z0-9]", "", str(value).lower())
    aliases = {"amos": "AMOS", "magic": "MAGIC", "kits23": "KiTS23",
               "msd10": "MSD10", "msd10colon": "MSD10",
               "msd07": "MSD07", "msd07pancreas": "MSD07"}
    if key not in aliases:
        raise ValueError(f"Unknown source {value!r}")
    return aliases[key]


def _load(path, is_ct=False):
    path = Path(path).resolve()
    if path.suffix.lower() == ".npy":
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
    elif path.suffix.lower() == ".npz" and not is_ct:
        with np.load(path, allow_pickle=False) as archive:
            arr = archive["mask"]
        arr.flags.writeable = False
    else:
        raise ValueError("CT requires existing shared .npy; masks require .npy or .npz key mask")
    if arr.ndim != 3 or any(n == 0 for n in arr.shape):
        raise ValueError(f"Expected nonempty canonical DHW array: {path}")
    if arr.dtype.kind not in "biuf":
        raise ValueError(f"Array must be real numeric: {path}")
    for frame in arr:
        if not np.isfinite(frame).all():
            raise ValueError(f"Array must be finite: {path}")
        if not is_ct and not np.logical_or(frame == 0, frame == 1).all():
            raise ValueError(f"Mask must be binary (0/1): {path}")
    return arr, {"path": str(path), "sha256": sha256(path), "shape_dhw": list(arr.shape),
                 "dtype": str(arr.dtype)}


def load_case(entry, base, expected_methods=METHODS):
    """Validate and map one full-volume case; choose only its display geometry."""
    base = Path(base)
    source = source_name(entry["source"])
    grid = entry.get("grid_verification", {})
    if (not isinstance(grid, dict) or grid.get("verified") is not True
            or not isinstance(grid.get("grid_id"), str) or not grid["grid_id"].strip()
            or grid.get("axes_dhw") not in ("ZXY", "ZYX")
            or not isinstance(grid.get("evidence"), str) or not grid["evidence"].strip()
            or entry.get("gt_grid_id") != grid["grid_id"]):
        raise ValueError("Need verified physical grid binding, axes_dhw and export check evidence")
    for key in ("case_id", "class_name"):
        if not isinstance(entry[key], str) or not entry[key].strip():
            raise ValueError(f"Nonempty {key} is required")
    spacing = np.asarray(entry["spacing_dhw"], dtype=float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError("spacing_dhw must have three finite positive values")
    ct, ct_info = _load(base / entry["ct_path"], is_ct=True)
    gt, gt_info = _load(base / entry["gt_path"])
    if gt.shape != ct.shape:
        raise ValueError("GT and CT shapes differ")
    areas = np.asarray([np.count_nonzero(frame) for frame in gt])
    if not areas.any():
        raise ValueError("Selected illustrative GT must be nonempty")
    z = entry.get("slice_index", int(areas.argmax()))
    if isinstance(z, bool) or not isinstance(z, int) or not 0 <= z < ct.shape[0]:
        raise ValueError("slice_index must be an in-range integer")
    crop = entry.get("crop_yx", [0, ct.shape[1], 0, ct.shape[2]])
    if (not isinstance(crop, list) or len(crop) != 4 or
            any(isinstance(i, bool) or not isinstance(i, int) for i in crop) or
            not (0 <= crop[0] < crop[1] <= ct.shape[1] and 0 <= crop[2] < crop[3] <= ct.shape[2])):
        raise ValueError("Invalid display crop_yx")
    transform = entry.get("display_transform", {})
    if (not isinstance(transform, dict) or set(transform) - {"transpose", "flip_y", "flip_x"}
            or any(type(v) is not bool for v in transform.values())):
        raise ValueError("Invalid display_transform")
    names = [method["name"] for method in entry["methods"]]
    if len(names) != len(set(names)) or set(names) != set(expected_methods):
        raise ValueError(f"Need exactly the requested methods: {list(expected_methods)}")
    methods = []
    for name in expected_methods:
        method = next(m for m in entry["methods"] if m["name"] == name)
        if method.get("grid_id") != grid["grid_id"]:
            raise ValueError(f"Prediction physical grid differs from verified CT grid: {name}")
        mask, info = _load(base / method["mask_path"])
        if mask.shape != ct.shape:
            raise ValueError(f"Mask shape differs from full CT: {name}")
        total = int(areas.sum()) + sum(np.count_nonzero(frame) for frame in mask)
        intersection = sum(np.count_nonzero(np.logical_and(a, b)) for a, b in zip(mask, gt))
        dice = float(2 * intersection / total)
        if "dice_3d" in method:
            reported = float(method["dice_3d"])
            if not math.isfinite(reported) or abs(reported - dice) > 1e-6:
                raise ValueError(f"Full-volume Dice mismatch for {name}: supplied={reported}, actual={dice}")
        methods.append({"name": name, "mask": mask, "dice_3d": dice,
                        "reported_dice_3d": method.get("dice_3d"), "input": info})
    return {"source": source, "case_id": entry["case_id"], "class_name": entry["class_name"],
            "spacing_dhw": spacing.tolist(), "ct": ct, "gt": gt, "methods": methods,
            "slice_index": z, "slice_selection": "explicit_post_hoc_display" if "slice_index" in entry
            else "first_gt_largest_axial_area_post_hoc_display",
            "crop_yx": crop, "display_transform": transform, "grid_verification": grid,
            "inputs": {"ct": ct_info, "gt": gt_info}}


def blend_slice(ct, mask, window=(-160, 240), alpha=.4, contour=None):
    """Composite orange binary mask onto windowed CT; optional cyan GT contour."""
    low, high = window
    gray = np.rint(np.clip((np.asarray(ct, dtype=float) - low) / (high - low), 0, 1) * 255)
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    if mask is not None:
        selected = np.asarray(mask, dtype=bool)
        rgb[selected] = (1 - alpha) * rgb[selected] + alpha * np.asarray(ORANGE)
    rgb = np.rint(rgb).astype(np.uint8)
    if contour is not None:
        mask = np.asarray(contour, dtype=bool)
        padded = np.pad(mask, 1)
        interior = (mask & padded[:-2, 1:-1] & padded[2:, 1:-1] &
                    padded[1:-1, :-2] & padded[1:-1, 2:])
        rgb[mask & ~interior] = CYAN
    return rgb


def physical_panel(rgb, spacing_hw, max_pixels=360):
    """Preserve physical width/height, with bounded raster dimensions."""
    height, width = rgb.shape[:2]
    physical = np.asarray([width * spacing_hw[1], height * spacing_hw[0]], dtype=float)
    size = np.maximum(1, np.rint(physical / physical.max() * max_pixels).astype(int))
    return Image.fromarray(rgb).resize(tuple(size), Image.Resampling.NEAREST)


def _display(array, case):
    y0, y1, x0, x1 = case["crop_yx"]
    image = array[case["slice_index"], y0:y1, x0:x1]
    transform = case["display_transform"]
    if transform.get("transpose", False):
        image = image.T
    if transform.get("flip_y", False):
        image = image[::-1]
    if transform.get("flip_x", False):
        image = image[:, ::-1]
    return image


def _font(size):
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def _wrap_label(text, font, width):
    words, lines, current = text.split(), [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and font.getlength(candidate) > width:
            lines.append(current)
            current = ""
        # Preserve labels even when a case ID or method token exceeds a cell.
        for character in word:
            candidate = current + character
            if current and font.getlength(candidate) > width:
                lines.append(current)
                current = ""
            current += character
        current += " "
    if current.strip():
        lines.append(current.strip())
    return [line.rstrip() for line in lines] or [""]


def _row_layout(case, panel_pixels):
    """Measure text before composing; 10 columns at 7 inches retain 5–6 pt labels."""
    gap = 10
    font_size = max(12, panel_pixels // 9)
    font = _font(font_size)
    line_height = math.ceil(font_size * 1.25)
    padding = max(8, font_size // 3)
    titles = ["CT", "Ground truth"] + [m["name"] for m in case["methods"]]
    lines = [_wrap_label(title, font, panel_pixels) for title in titles]
    metric_y = padding + max(map(len, lines)) * line_height + padding
    header = metric_y + line_height + padding
    width = len(titles) * (panel_pixels + gap) + gap
    return dict(font_size=font_size, line_height=line_height, gap=gap, width=width,
                title_lines=lines, title_y=padding, metric_y=metric_y,
                header=header, footer=line_height + 2 * padding)


def _case_row(case, window, alpha, contour, panel_pixels):
    layout = _row_layout(case, panel_pixels)
    gap, header, footer = layout["gap"], layout["header"], layout["footer"]
    columns = len(case["methods"]) + 2
    cell = panel_pixels + gap
    row = Image.new("RGB", (columns * cell + gap, header + panel_pixels + footer), "black")
    draw = ImageDraw.Draw(row)
    font = _font(layout["font_size"])
    ct, gt = _display(case["ct"], case), _display(case["gt"], case)
    spacing = case["spacing_dhw"][1:]
    if case["display_transform"].get("transpose", False):
        spacing = spacing[::-1]
    items = [(None, None, None), (None, gt, None)]
    items += [(f"3D Dice {m['dice_3d']:.3f}", _display(m["mask"], case), gt if contour else None)
              for m in case["methods"]]
    for i, (metric, mask, gt_contour) in enumerate(items):
        center = gap + i * cell + panel_pixels / 2
        for line_idx, line in enumerate(layout["title_lines"][i]):
            draw.text((center, layout["title_y"] + line_idx * layout["line_height"]),
                      line, font=font, fill="white", anchor="mt")
        if metric:
            draw.text((center, layout["metric_y"]), metric, font=font, fill="white", anchor="mt")
        panel = physical_panel(blend_slice(ct, mask, window, alpha, gt_contour), spacing, panel_pixels)
        row.paste(panel, (gap + i * cell + (panel_pixels - panel.width) // 2,
                         header + (panel_pixels - panel.height) // 2))
    label = (f"{case['source']} | {case['case_id']} | {case['class_name']} | axial index {case['slice_index']} (0-based)"
             f" | HU [{window[0]:g}, {window[1]:g}]")
    draw.text((gap, header + panel_pixels + layout["title_y"]), label, font=font, fill="white", anchor="lt")
    return row


def render_manifest(manifest_path, output_dir, expected_methods=METHODS, window=(-160, 240),
                    alpha=.4, gt_contour=False, panel_pixels=360, require_five=True):
    """Validate all inputs before reserving a fresh output directory."""
    manifest_path, output_dir = Path(manifest_path).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory must be new: {output_dir}")
    if (len(window) != 2 or not np.isfinite(window).all() or window[0] >= window[1]
            or not math.isfinite(alpha) or not 0 <= alpha <= 1
            or not isinstance(panel_pixels, int) or panel_pixels < 96):
        raise ValueError("Invalid window, opacity, or panel pixel size (minimum 96)")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not manifest.get("cases"):
        raise ValueError("Need schema_version 1 and nonempty cases")
    sources = [source_name(c["source"]) for c in manifest["cases"]]
    if require_five and (len(sources) != 5 or set(sources) != set(SOURCES)):
        raise ValueError("Need exactly one case-class per AMOS/MAGIC/KiTS23/MSD10/MSD07")
    if len(expected_methods) == 0 or len(set(expected_methods)) != len(expected_methods):
        raise ValueError("Requested method names must be nonempty and unique")
    cases = [load_case(c, manifest_path.parent, expected_methods) for c in manifest["cases"]]
    keys = [(c["source"], c["case_id"], c["class_name"]) for c in cases]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate case-class")
    cases.sort(key=lambda c: SOURCES.index(c["source"]))
    output_dir.mkdir(parents=True, exist_ok=False)
    rows, records = [], []
    for i, case in enumerate(cases):
        row = _case_row(case, window, alpha, gt_contour, panel_pixels)
        stem = f"{i + 1:02d}_{case['source']}"
        row.save(output_dir / f"{stem}.png", dpi=(300, 300))
        row.save(output_dir / f"{stem}.pdf", resolution=300, quality=95, subsampling=0)
        rows.append(row)
        record = {k: v for k, v in case.items() if k not in ("ct", "gt", "methods")}
        record["methods"] = [{k: v for k, v in m.items() if k != "mask"} for m in case["methods"]]
        record["outputs"] = [f"{stem}.png", f"{stem}.pdf"]
        records.append(record)
    combined = Image.new("RGB", (rows[0].width, sum(row.height for row in rows)), "black")
    top = 0
    for row in rows:
        combined.paste(row, (0, top))
        top += row.height
    combined.save(output_dir / "five_source_comparison.png", dpi=(300, 300))
    combined.save(output_dir / "five_source_comparison.pdf", resolution=300, quality=95, subsampling=0)
    caption = ("Post-hoc illustrative case-class examples (not representative averages). "
               "CT, ground truth and each method use the identical axial slice, CT window, "
               "display crop, transform and physical aspect within each row. Default slices "
               "are selected by largest GT axial area after inference; any manual display "
               "selection is recorded in provenance. Orange fill denotes binary segmentation "
               f"at opacity {alpha:g}. " + ("Cyan outlines show GT in all prediction columns. " if gt_contour else "") +
               "3D Dice is recomputed on full-volume masks, not on displayed slices. "
               "Native method-specific prompt budgets are retained; equal prompt budgets "
               "are not implied. Images follow stored DHW axes and the recorded display "
               "transform; anatomical side labels are not inferred.\n")
    (output_dir / "caption.txt").write_text(caption, encoding="utf-8")
    report = {"schema_version": 1, "manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
              "renderer_sha256": sha256(Path(__file__)), "window_hu": list(window), "alpha": alpha,
              "mask_color_rgb": list(ORANGE), "gt_contour": gt_contour, "gt_contour_rgb": list(CYAN),
              "dpi": 300, "panel_max_pixels": panel_pixels, "expected_methods": list(expected_methods),
              "mask_alignment_contract": "All arrays on identical full canonical-DHW CT grid; exporter attests physical alignment, renderer verifies binding/shape but cannot independently recover affine from NPY",
              "caption": caption, "cases": records}
    (output_dir / "provenance.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_dir", type=Path, help="Must not already exist")
    parser.add_argument("--window", nargs=2, type=float, default=(-160, 240), metavar=("LOW", "HIGH"))
    parser.add_argument("--alpha", type=float, default=.4)
    parser.add_argument("--gt-contour", action="store_true")
    parser.add_argument("--panel-pixels", type=int, default=360)
    parser.add_argument("--methods", nargs="+", default=METHODS, help="Default: all eight main-table methods")
    parser.add_argument("--allow-subset-sources", action="store_true", help="Explicit partial diagnostic rendering only")
    args = parser.parse_args()
    report = render_manifest(args.manifest, args.output_dir, args.methods, args.window,
                             args.alpha, args.gt_contour, args.panel_pixels, not args.allow_subset_sources)
    print(f"Rendered {len(report['cases'])} case-class rows into {args.output_dir}")


if __name__ == "__main__":
    main()
