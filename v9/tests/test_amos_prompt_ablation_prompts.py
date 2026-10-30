"""Behavioral tests for the frozen AMOS prompt-input interventions (NumPy only)."""
from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import unittest

import numpy as np

from infer.v9_2_eight_source_eval_protocol import (
    _resize_preserve,
    prompt_package_to_groups,
)


def fixture():
    shape = (12, 7, 9)
    coronal = np.zeros(shape, dtype=np.uint8)
    sagittal = np.zeros(shape, dtype=np.uint8)
    coronal[2:5, 2, 4] = 1
    sagittal[7:10, 3, 3] = 1
    package = {
        "coronal": coronal,
        "sagittal": sagittal,
        "coronal_slice": 4,
        "sagittal_slice": 3,
        "background_points_dhw": np.array(
            [[0, 0, 4], [1, 0, 4], [0, 3, 0], [1, 3, 0]], dtype=np.int32
        ),
        "background_point_planes": ["coronal", "coronal", "sagittal", "sagittal"],
        "class_id": 17,
        # Deliberately irrelevant GT-derived metadata: never use this as ROI support.
        "foreground_bbox_dhw": [0, 0, 0, 11, 6, 8],
    }
    groups = {}
    for plane, width, bounds in (
        ("coronal", 7, (1, 1, 5, 5)),
        ("sagittal", 9, (6, 1, 10, 7)),
    ):
        group = np.zeros((3, 12, width), dtype=np.uint8)
        group[0, 0, 0] = 1  # Native newly simulated FG must not replace cached FG.
        group[1, :, -1] = 1
        r0, c0, r1, c1 = bounds
        group[2, [r0, r1], c0:c1 + 1] = 1
        group[2, r0:r1 + 1, [c0, c1]] = 1
        groups[plane] = group
    return package, groups


def expected_replacement(package, groups, foreground, background, size):
    expected = prompt_package_to_groups(package, prompt_plane_size=size)
    for i, plane in enumerate(("coronal", "sagittal")):
        padding = (1, 1) if plane == "coronal" else (0, 0)
        native = _resize_preserve(groups[plane], size, padding)
        if foreground == "box":
            expected["groups"][i, 0] = 0
            expected["groups"][i, 2] = native[2]
        if background == "scribble":
            expected["groups"][i, 1] = native[1]
    return expected


class PromptAblationTests(unittest.TestCase):
    def api(self):
        name = "tools.amos_prompt_ablation_prompts"
        self.assertIsNotNone(importlib.util.find_spec(name), "prompt transformer is missing")
        return importlib.import_module(name)

    def test_exact_six_variant_mode_and_native_type_contract(self):
        api = self.api()
        self.assertEqual(api.VARIANTS, ("baseline", "sagittal_only", "coronal_only",
                                      "fg_box_bg_point", "fg_scribble_bg_scribble",
                                      "fg_box_bg_scribble"))
        for variant in api.VARIANTS:
            modes = api.variant_modes(variant)
            foreground = "box" if "fg_box" in variant else "scribble"
            background = "scribble" if "bg_scribble" in variant else "point"
            planes = (("sagittal",) if variant == "sagittal_only" else
                      ("coronal",) if variant == "coronal_only" else
                      ("coronal", "sagittal"))
            self.assertEqual(modes["active_planes"], planes)
            self.assertEqual(modes["plane_ids"], tuple(1 if p == "coronal" else 0 for p in planes))
            self.assertEqual(modes["foreground_mode"], foreground)
            self.assertEqual(modes["background_mode"], background)
            self.assertEqual(modes["foreground_mode_id"], int(foreground == "box"))
            self.assertEqual(modes["background_mode_id"], int(background == "point"))
            self.assertEqual(modes["foreground_type_id"], 2 if foreground == "box" else 0)
            self.assertEqual(modes["background_type_id"], 4 if background == "point" else 3)

    def test_unknown_variant_is_refused(self):
        with self.assertRaisesRegex(ValueError, "variant"):
            self.api().variant_modes("unknown")

    def test_baseline_equals_native_converter_every_key(self):
        package, groups = fixture()
        for size in (1, 2, 11, 19):
            expected = prompt_package_to_groups(package, prompt_plane_size=size)
            actual, audit = self.api().build_variant(package, groups, "baseline", size)
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                if isinstance(expected[key], np.ndarray):
                    np.testing.assert_array_equal(actual[key], expected[key])
                    self.assertEqual(actual[key].dtype, expected[key].dtype)
                else:
                    self.assertEqual(actual[key], expected[key])
            self.assertEqual(audit["raw_foreground_z_support"], [2, 3, 4, 7, 8, 9])

    def test_single_plane_removes_foreground_and_background_group(self):
        package, groups = fixture()
        native = prompt_package_to_groups(package, prompt_plane_size=15)
        for variant, plane, index, support in (
            ("coronal_only", "coronal", 0, [2, 3, 4]),
            ("sagittal_only", "sagittal", 1, [7, 8, 9]),
        ):
            converted, audit = self.api().build_variant(package, groups, variant, 15)
            np.testing.assert_array_equal(converted["groups"], native["groups"][[index]])
            np.testing.assert_array_equal(converted["plane_ids"], native["plane_ids"][[index]])
            np.testing.assert_array_equal(converted["slice_coords"], native["slice_coords"][[index]])
            self.assertEqual(converted["background_points_per_plane"], {plane: 2})
            self.assertEqual(set(audit["planes"]), {plane})
            self.assertEqual(audit["raw_foreground_z_support"], support)
            self.assertEqual(audit["raw_background_points_per_plane"], {plane: 2})

    def test_raw_two_points_survive_audit_even_when_raster_collides(self):
        package, groups = fixture()
        converted, audit = self.api().build_variant(package, groups, "baseline", 2)
        self.assertEqual(audit["raw_background_points_per_plane"], {"coronal": 2, "sagittal": 2})
        self.assertEqual(audit["raster_background_pixels_per_plane"], {"coronal": 1, "sagittal": 1})
        self.assertEqual(int(converted["groups"][:, 1].sum()), 2)

    def test_duplicate_raw_points_not_replaced_to_force_raster_cardinality(self):
        package, groups = fixture()
        package["background_points_dhw"][1] = package["background_points_dhw"][0]
        converted, audit = self.api().build_variant(package, groups, "coronal_only", 15)
        self.assertEqual(audit["raw_background_points_per_plane"], {"coronal": 2})
        self.assertEqual(int(converted["groups"][0, 1].sum()), 1)

    def test_box_replaces_only_foreground_uses_outline_channel_two(self):
        package, groups = fixture()
        expected = expected_replacement(package, groups, "box", "point", 13)
        converted, audit = self.api().build_variant(package, groups, "fg_box_bg_point", 13)
        np.testing.assert_array_equal(converted["groups"], expected["groups"])
        self.assertFalse(converted["groups"][:, 0].any())
        self.assertEqual(audit["raw_foreground_z_support"], list(range(1, 11)))
        # A native simulated outline need not lie inside the sparse cached scribble.
        self.assertEqual(audit["planes"]["coronal"]["raw_channel_nonzero"][2], 16)

    def test_background_scribble_replaces_only_background(self):
        package, groups = fixture()
        expected = expected_replacement(package, groups, "scribble", "scribble", 13)
        converted, audit = self.api().build_variant(package, groups, "fg_scribble_bg_scribble", 13)
        np.testing.assert_array_equal(converted["groups"], expected["groups"])
        self.assertEqual(converted["background_mode"], "training_scribble")
        self.assertEqual(converted["background_points_per_plane"], {"coronal": 12, "sagittal": 12})
        self.assertEqual(audit["raw_background_points_per_plane"], {"coronal": 0, "sagittal": 0})
        self.assertEqual(audit["raw_foreground_z_support"], [2, 3, 4, 7, 8, 9])

    def test_both_replacements_use_same_native_geometry(self):
        package, groups = fixture()
        expected = expected_replacement(package, groups, "box", "scribble", 13)
        converted, audit = self.api().build_variant(package, groups, "fg_box_bg_scribble", 13)
        np.testing.assert_array_equal(converted["groups"], expected["groups"])
        self.assertEqual(audit["foreground_type_id"], 2)
        self.assertEqual(audit["background_type_id"], 3)

    def test_input_immutability_and_no_output_array_aliases(self):
        api = self.api()
        for variant in api.VARIANTS:
            package, groups = fixture()
            saved_package, saved_groups = copy.deepcopy(package), copy.deepcopy(groups)
            converted, _ = api.build_variant(package, groups, variant, 13)
            converted["groups"][:] = 0
            converted["plane_ids"][:] = -1
            for key, saved in saved_package.items():
                if isinstance(saved, np.ndarray):
                    np.testing.assert_array_equal(package[key], saved)
                else:
                    self.assertEqual(package[key], saved)
            for key in groups:
                np.testing.assert_array_equal(groups[key], saved_groups[key])

    def test_audit_records_actual_raw_and_raster_shape_hashes_geometry(self):
        package, groups = fixture()
        converted, audit = self.api().build_variant(package, groups, "fg_box_bg_scribble", 13)
        self.assertEqual(audit["canonical_shape_dhw"], [12, 7, 9])
        for i, plane in enumerate(("coronal", "sagittal")):
            plane_audit = audit["planes"][plane]
            self.assertEqual(plane_audit["plane_id"], int(converted["plane_ids"][i]))
            self.assertEqual(plane_audit["slice_index"], package[plane + "_slice"])
            self.assertEqual(plane_audit["slice_coord"], float(converted["slice_coords"][i]))
            self.assertEqual(plane_audit["raw_shape"], list(groups[plane].shape))
            self.assertEqual(plane_audit["raster_shape"], [3, 13, 13])
            raster = converted["groups"][i]
            self.assertEqual(plane_audit["raster_sha256"], hashlib.sha256(raster.tobytes()).hexdigest())
            self.assertEqual(plane_audit["raster_channel_sha256"],
                             [hashlib.sha256(c.tobytes()).hexdigest() for c in raster])
            self.assertEqual(plane_audit["raster_channel_nonzero"],
                             [int(np.count_nonzero(c)) for c in raster])
            raw = groups[plane].copy()
            raw[0] = 0
            self.assertEqual(plane_audit["raw_sha256"], hashlib.sha256(raw.tobytes()).hexdigest())

    def test_invalid_shapes_plane_coords_points_and_raw_groups_are_refused(self):
        mutations = [
            lambda p, g: p.update(sagittal=np.zeros((11, 7, 9), dtype=np.uint8)),
            lambda p, g: p.update(coronal_slice=9),
            lambda p, g: p.update(sagittal_slice=-1),
            lambda p, g: p.update(coronal_slice=4.5),
            lambda p, g: p["background_points_dhw"].__setitem__((0, 0), 12),
            lambda p, g: p["background_points_dhw"].__setitem__((0, 2), 5),
            lambda p, g: p["background_point_planes"].__setitem__(0, "axial"),
            lambda p, g: p.update(background_points_dhw=p["background_points_dhw"][:3]),
            lambda p, g: p.update(background_points_dhw=p["background_points_dhw"].astype(float) + 0.5),
            lambda p, g: g.update(coronal=np.zeros((3, 11, 7), dtype=np.uint8)),
            lambda p, g: g.update(coronal=np.zeros((2, 12, 7), dtype=np.uint8)),
            lambda p, g: g.pop("sagittal"),
            lambda p, g: g.update(axial=np.zeros((3, 7, 9), dtype=np.uint8)),
            lambda p, g: p["coronal"].__setitem__((3, 2, 1), 1),
            lambda p, g: p.update(coronal=np.full((12, 7, 9), np.nan)),
            lambda p, g: g.update(coronal=np.full((3, 12, 7), np.nan)),
        ]
        api = self.api()
        for mutation in mutations:
            package, groups = fixture()
            mutation(package, groups)
            with self.subTest(mutation=mutation), self.assertRaises((ValueError, TypeError)):
                api.build_variant(package, groups, "baseline", 13)

    def test_empty_actual_foreground_no_hidden_bbox_or_native_scribble_fallback(self):
        api = self.api()
        for variant, empty in (("coronal_only", "coronal"), ("sagittal_only", "sagittal")):
            package, groups = fixture()
            package[empty][:] = 0
            with self.assertRaisesRegex(ValueError, "foreground"):
                api.build_variant(package, groups, variant, 13)
        package, groups = fixture()
        groups["coronal"][2] = 0
        with self.assertRaisesRegex(ValueError, "foreground|box"):
            api.build_variant(package, groups, "fg_box_bg_point", 13)

    def test_empty_selected_background_scribble_refused(self):
        package, groups = fixture()
        groups["sagittal"][1] = 0
        with self.assertRaisesRegex(ValueError, "background"):
            self.api().build_variant(package, groups, "fg_scribble_bg_scribble", 13)

    def test_box_refuses_filled_or_broken_outline_but_allows_degenerate_native_line(self):
        api = self.api()
        for modification in ("fill", "break"):
            package, groups = fixture()
            if modification == "fill":
                groups["coronal"][2, 1:6, 1:6] = 1
            else:
                groups["coronal"][2, 1, 2] = 0
            with self.subTest(modification=modification), self.assertRaisesRegex(ValueError, "outline"):
                api.build_variant(package, groups, "fg_box_bg_point", 13)
        package, groups = fixture()
        groups["coronal"][2] = 0
        groups["coronal"][2, 3, 1:6] = 1
        _, audit = api.build_variant(package, groups, "fg_box_bg_point", 13)
        self.assertEqual(audit["planes"]["coronal"]["raw_foreground_z_support"], [3])

    def test_prompt_plane_size_positive_integer(self):
        package, groups = fixture()
        for size in (0, -1, 3.5, True):
            with self.subTest(size=size), self.assertRaises((ValueError, TypeError)):
                self.api().build_variant(package, groups, "baseline", size)

    def test_roi_native_context_rounding_and_minframes_arithmetic(self):
        api = self.api()
        examples = (
            ((20, [8, 10], 1, .5, 1), range(6, 13)),
            ((20, [8, 12], 0, .5, 1), range(6, 15)),  # round(2.5)==2
            ((20, [1], 0, 0., 8), range(0, 8)),
            ((20, [18], 0, 0., 8), range(12, 20)),
            ((20, [8, 10], 0, 0., 8), range(6, 14)),
            ((5, [2], 0, 0., 20), range(0, 5)),
        )
        for args, expected in examples:
            actual = api.roi_indices(*args)
            np.testing.assert_array_equal(actual, np.asarray(list(expected), dtype=np.int32))
            self.assertEqual(actual.dtype, np.int32)

    def test_roi_uses_raw_foreground_support_before_resizing(self):
        package, groups = fixture()
        _, audit = self.api().build_variant(package, groups, "coronal_only", 1)
        indices = self.api().roi_indices(12, audit["raw_foreground_z_support"], 0, 0., 1)
        np.testing.assert_array_equal(indices, [2, 3, 4])

    def test_roi_refuses_empty_invalid_support_and_parameters(self):
        api = self.api()
        for args in ((12, [], 0, 0., 1), (12, [-1], 0, 0., 1),
                     (12, [12], 0, 0., 1), (12, [1.5], 0, 0., 1),
                     (0, [0], 0, 0., 1), (12, [1], -1, 0., 1),
                     (12, [1], 0, -0.1, 1), (12, [1], 0, np.nan, 1),
                     (12, [1], 0, 0., 0)):
            with self.subTest(args=args), self.assertRaises((ValueError, TypeError)):
                api.roi_indices(*args)


if __name__ == "__main__":
    unittest.main()
