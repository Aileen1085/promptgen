"""Synthetic geometry/metric tests; fixtures are not medical result figures."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "tools/render_five_source_qualitative.py"
spec = importlib.util.spec_from_file_location("five_render", SCRIPT) if SCRIPT.exists() else None
renderer = importlib.util.module_from_spec(spec) if spec else None
if spec:
    spec.loader.exec_module(renderer)


class RendererTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(renderer, "Scientific renderer is not implemented")
        # Leave synthetic temporary fixtures rather than recursively delete files.
        scratch = SCRIPT.parents[1] / "output" / "_synthetic_renderer_tests"
        scratch.mkdir(parents=True, exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix="test_", dir=scratch))

    def fixture(self, source="AMOS", case_id="case1"):
        ct = np.linspace(-300, 500, 3 * 8 * 10, dtype=np.float32).reshape(3, 8, 10)
        gt = np.zeros(ct.shape, dtype=np.uint8)
        gt[0, 2:4, 3:5] = 1
        gt[1, 1:6, 2:8] = 1
        pred = gt.copy()
        pred[2, 0, 0] = 1  # off selected slice: Dice must still count this FP
        for name, arr in (("ct", ct), ("gt", gt), ("pred", pred)):
            np.save(self.root / f"{case_id}_{name}.npy", arr)
        return {"source": source, "case_id": case_id, "class_name": "organ",
                "ct_path": f"{case_id}_ct.npy", "gt_path": f"{case_id}_gt.npy",
                "grid_verification": {"verified": True, "grid_id": case_id, "axes_dhw": "ZYX",
                                      "evidence": "Synthetic arrays created on identical grid"},
                "gt_grid_id": case_id,
                "spacing_dhw": [3, 2, 1],
                "methods": [{"name": "PromptGen", "mask_path": f"{case_id}_pred.npy", "grid_id": case_id}]}

    def test_largest_gt_display_slice_and_full_volume_dice(self):
        case = renderer.load_case(self.fixture(), self.root, ["PromptGen"])
        self.assertEqual(case["slice_index"], 1)
        self.assertAlmostEqual(case["methods"][0]["dice_3d"], 68 / 69)
        self.assertEqual(case["crop_yx"], [0, 8, 0, 10])
        self.assertFalse(case["ct"].flags.writeable)

    def test_rejects_wrong_shape_nonbinary_nonfinite_and_metric_mismatch(self):
        entry = self.fixture()
        for bad in (np.zeros((2, 8, 10)), np.full((3, 8, 10), .5)):
            np.save(self.root / "bad.npy", bad)
            entry["methods"][0]["mask_path"] = "bad.npy"
            with self.assertRaises(ValueError):
                renderer.load_case(entry, self.root, ["PromptGen"])
        entry = self.fixture()
        entry["methods"][0]["dice_3d"] = .1
        with self.assertRaisesRegex(ValueError, "Dice"):
            renderer.load_case(entry, self.root, ["PromptGen"])
        ct = np.load(self.root / "case1_ct.npy")
        ct[0, 0, 0] = np.nan
        np.save(self.root / "case1_ct.npy", ct)
        with self.assertRaisesRegex(ValueError, "finite"):
            renderer.load_case(self.fixture_for_existing(), self.root, ["PromptGen"])

    def fixture_for_existing(self):
        return {"source": "AMOS", "case_id": "case1", "class_name": "organ",
                "ct_path": "case1_ct.npy", "gt_path": "case1_gt.npy",
                "grid_verification": {"verified": True, "grid_id": "case1", "axes_dhw": "ZYX",
                                      "evidence": "Synthetic arrays created on identical grid"},
                "gt_grid_id": "case1",
                "spacing_dhw": [3, 2, 1],
                "methods": [{"name": "PromptGen", "mask_path": "case1_pred.npy", "grid_id": "case1"}]}

    def test_same_crop_spacing_and_alpha_without_affecting_unmasked_pixels(self):
        ct = np.zeros((4, 6), dtype=np.float32)
        mask = np.zeros_like(ct, dtype=bool)
        mask[1, 2] = 1
        base = renderer.blend_slice(ct, None, (-160, 240), .4)
        overlay = renderer.blend_slice(ct, mask, (-160, 240), .4)
        np.testing.assert_array_equal(base[~mask], overlay[~mask])
        np.testing.assert_array_equal(overlay[1, 2], [163, 116, 61])
        panel = renderer.physical_panel(overlay, [2, 1], max_pixels=120)
        self.assertEqual(panel.size, (90, 120))  # 6mm width / 8mm height

    def test_fresh_outputs_caption_hashes_and_inputs_unchanged(self):
        entries = [self.fixture(source, f"case{i}") for i, source in enumerate(renderer.SOURCES)]
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps({"schema_version": 1, "cases": entries}), encoding="utf-8")
        before = {p.name: renderer.sha256(p) for p in self.root.glob("*.npy")}
        out = self.root / "figures"
        report = renderer.render_manifest(manifest, out, expected_methods=["PromptGen"], panel_pixels=100)
        self.assertEqual(len(report["cases"]), 5)
        self.assertTrue((out / "five_source_comparison.pdf").is_file())
        with Image.open(out / "five_source_comparison.png") as im:
            self.assertEqual(im.info["dpi"][0], 299.9994)
        self.assertIn("post-hoc", (out / "caption.txt").read_text(encoding="utf-8").lower())
        saved = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["cases"][0]["inputs"]["ct"]["sha256"], before["case0_ct.npy"])
        self.assertEqual(before, {p.name: renderer.sha256(p) for p in self.root.glob("*.npy")})
        self.assertEqual(list(out.glob("*.npy")), [])
        with self.assertRaises(FileExistsError):
            renderer.render_manifest(manifest, out, expected_methods=["PromptGen"], panel_pixels=100)

    def test_all_eight_methods_and_five_sources_are_default_requirements(self):
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps({"schema_version": 1, "cases": [self.fixture()]}), encoding="utf-8")
        with self.assertRaises(ValueError):
            renderer.render_manifest(manifest, self.root / "bad")
        self.assertFalse((self.root / "bad").exists())

    def test_full_default_ten_columns_and_uniform_display_transform(self):
        entries = [self.fixture(source, f"case{i}") for i, source in enumerate(renderer.SOURCES)]
        for entry in entries:
            mask_path = entry["methods"][0]["mask_path"]
            entry["methods"] = [{"name": name, "mask_path": mask_path, "grid_id": entry["case_id"]}
                                for name in renderer.METHODS]
            entry["display_transform"] = {"transpose": True, "flip_y": True, "flip_x": True}
            entry["crop_yx"] = [1, 7, 2, 9]
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps({"schema_version": 1, "cases": entries}), encoding="utf-8")
        report = renderer.render_manifest(manifest, self.root / "full", panel_pixels=100, gt_contour=True)
        self.assertEqual(report["expected_methods"], list(renderer.METHODS))
        with Image.open(self.root / "full/five_source_comparison.png") as im:
            self.assertEqual(im.width, 1110)
            self.assertGreater(im.height, 5 * 100)
            self.assertEqual(im.height % 5, 0)
        case = renderer.load_case(entries[0], self.root)
        native = case["gt"][case["slice_index"], 1:7, 2:9]
        np.testing.assert_array_equal(renderer._display(case["gt"], case), native.T[::-1, ::-1])
        overlay = renderer.blend_slice(np.zeros((3, 3)), np.ones((3, 3)), contour=np.ones((3, 3)))
        np.testing.assert_array_equal(overlay[0, 0], renderer.CYAN)
        self.assertFalse(np.array_equal(overlay[1, 1], renderer.CYAN))

    def test_bad_spacing_crop_and_missing_methods_rejected(self):
        for updates in ({"spacing_dhw": [1, 0, 1]}, {"crop_yx": [-1, 8, 0, 10]},
                        {"slice_index": 3}, {"methods": []}):
            entry = self.fixture()
            entry.update(updates)
            with self.assertRaises(ValueError):
                renderer.load_case(entry, self.root, ["PromptGen"])
        self.assertEqual(renderer.source_name("MSD10 Colon"), "MSD10")
        self.assertEqual(renderer.source_name("msd07_pancreas"), "MSD07")

    def test_missing_or_conflicting_physical_grid_attestation_rejected(self):
        entry = self.fixture()
        del entry["grid_verification"]
        with self.assertRaisesRegex(ValueError, "grid"):
            renderer.load_case(entry, self.root, ["PromptGen"])
        entry = self.fixture()
        entry["methods"][0]["grid_id"] = "different-physical-grid"
        with self.assertRaisesRegex(ValueError, "grid"):
            renderer.load_case(entry, self.root, ["PromptGen"])

    def test_paper_labels_remain_five_point_at_seven_inches_without_overlap(self):
        from PIL import ImageDraw
        self.assertIsNotNone(getattr(renderer, "_row_layout", None), "Measured paper layout is required")
        entry = self.fixture()
        entry["methods"] = [{"name": name, "mask_path": "case1_pred.npy", "grid_id": "case1"}
                            for name in renderer.METHODS]
        case = renderer.load_case(entry, self.root)
        for pixels in (360, 512):
            with self.subTest(panel_pixels=pixels):
                layout = renderer._row_layout(case, pixels)
                paper_points = layout["font_size"] * 7 * 72 / layout["width"]
                self.assertGreaterEqual(paper_points, 5)
                self.assertLessEqual(paper_points, 6)
                row = renderer._case_row(case, (-160, 240), .4, False, pixels)
                draw = ImageDraw.Draw(row)
                font = renderer._font(layout["font_size"])
                long_case = dict(case, methods=[dict(m) for m in case["methods"]])
                long_case["methods"][4]["name"] += " (native full-volume)"
                wrapped = renderer._row_layout(long_case, pixels)
                self.assertGreater(len(wrapped["title_lines"][-4]), 1)
                self.assertGreater(wrapped["header"], layout["header"])
                for lines in layout["title_lines"]:
                    for i, line in enumerate(lines):
                        bbox = draw.textbbox((0, layout["title_y"] + i * layout["line_height"]),
                                             line, font=font, anchor="lt")
                        self.assertLessEqual(bbox[2] - bbox[0], pixels)
                        self.assertLess(bbox[3], layout["metric_y"])
                metric_box = draw.textbbox((0, layout["metric_y"]), "3D Dice 0.986", font=font, anchor="lt")
                self.assertLess(metric_box[3], layout["header"])
                # The CT image remains identical despite the larger text/header.
                physical = renderer.physical_panel(renderer.blend_slice(case["ct"][1], None), [2, 1], pixels)
                x = layout["gap"] + (pixels - physical.width) // 2
                y = layout["header"] + (pixels - physical.height) // 2
                np.testing.assert_array_equal(np.asarray(row.crop((x, y, x + physical.width, y + physical.height))),
                                              np.asarray(physical))


if __name__ == "__main__":
    unittest.main()
