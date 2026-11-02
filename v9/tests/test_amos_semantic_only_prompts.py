"""NumPy-only contracts for semantic-only inputs with no spatial information."""
from __future__ import annotations

import importlib
import importlib.util
import inspect
import unittest

import numpy as np


class SemanticOnlyInputTests(unittest.TestCase):
    def api(self):
        name = "tools.amos_semantic_only_prompts"
        self.assertIsNotNone(
            importlib.util.find_spec(name), "semantic-only input helpers are missing"
        )
        return importlib.import_module(name)

    def test_empty_bundle_has_only_native_empty_arrays_and_ct_depth(self):
        api = self.api()
        for shape, size in (((9, 11, 13), 192), ((1, 2, 3), 1), ((17, 5, 2), 224)):
            with self.subTest(shape=shape, size=size):
                bundle = api.make_empty_bundle(shape, size)
                self.assertEqual(
                    set(bundle), {"groups", "plane_ids", "slice_coords", "canonical_depth"}
                )
                for key, expected_shape, dtype in (
                    ("groups", (0, 3, size, size), np.float32),
                    ("plane_ids", (0,), np.int64),
                    ("slice_coords", (0,), np.float32),
                ):
                    self.assertIsInstance(bundle[key], np.ndarray)
                    self.assertEqual(bundle[key].shape, expected_shape)
                    self.assertEqual(bundle[key].dtype, np.dtype(dtype))
                    self.assertEqual(bundle[key].size, 0)
                self.assertIs(type(bundle["canonical_depth"]), int)
                self.assertEqual(bundle["canonical_depth"], shape[0])
                api.validate_empty_bundle(bundle, shape, size)

    def test_integer_numpy_shape_and_size_are_accepted(self):
        api = self.api()
        shape = np.array([9, 11, 13], dtype=np.int64)
        bundle = api.make_empty_bundle(shape, np.int32(7))
        self.assertEqual(bundle["groups"].shape, (0, 3, 7, 7))
        self.assertEqual(bundle["canonical_depth"], 9)
        np.testing.assert_array_equal(api.full_ct_indices(shape), np.arange(9, dtype=np.int32))

    def test_invalid_canonical_shapes_are_rejected_by_all_shape_entry_points(self):
        api = self.api()
        invalid_shapes = (
            None, True, 9, "913", (), (9,), (9, 11), (9, 11, 13, 1),
            (0, 11, 13), (-1, 11, 13), (9, 0, 13), (9, 11, -1),
            (True, 11, 13), (9, np.bool_(True), 13), (9, 11, False),
            (9.0, 11, 13), (9, 11.5, 13), (9, 11, float("nan")),
            (float("inf"), 11, 13), ((9,), 11, 13),
            np.array([[9, 11, 13]], dtype=np.int32),
        )
        valid = api.make_empty_bundle((9, 11, 13), 7)
        for shape in invalid_shapes:
            with self.subTest(shape=shape):
                with self.assertRaises((TypeError, ValueError)):
                    api.make_empty_bundle(shape, 7)
                with self.assertRaises((TypeError, ValueError)):
                    api.full_ct_indices(shape)
                with self.assertRaises((TypeError, ValueError)):
                    api.validate_empty_bundle(valid, shape, 7)

    def test_invalid_prompt_plane_sizes_are_rejected(self):
        api = self.api()
        valid = api.make_empty_bundle((9, 11, 13), 7)
        for size in (None, True, np.bool_(False), 0, -1, 7.0, 1.5, float("nan"),
                     float("inf"), "7", (7,)):
            with self.subTest(size=size):
                with self.assertRaises((TypeError, ValueError)):
                    api.make_empty_bundle((9, 11, 13), size)
                with self.assertRaises((TypeError, ValueError)):
                    api.validate_empty_bundle(valid, (9, 11, 13), size)

    def test_validator_rejects_phantom_groups_planes_and_slice_coordinates(self):
        api = self.api()
        for key, phantom in (
            ("groups", np.zeros((1, 3, 7, 7), dtype=np.float32)),
            ("plane_ids", np.array([0], dtype=np.int64)),
            ("plane_ids", np.array([1], dtype=np.int64)),
            ("slice_coords", np.array([0.0], dtype=np.float32)),
            ("slice_coords", np.array([0.5], dtype=np.float32)),
        ):
            with self.subTest(key=key, shape=phantom.shape):
                bundle = api.make_empty_bundle((9, 11, 13), 7)
                bundle[key] = phantom
                with self.assertRaises(ValueError):
                    api.validate_empty_bundle(bundle, (9, 11, 13), 7)

    def test_validator_rejects_wrong_layout_channels_and_plane_size(self):
        api = self.api()
        for key, wrong_shape, dtype in (
            ("groups", (0, 7, 7, 3), np.float32),
            ("groups", (0, 2, 7, 7), np.float32),
            ("groups", (0, 4, 7, 7), np.float32),
            ("groups", (0, 3, 6, 7), np.float32),
            ("groups", (0, 3, 7, 6), np.float32),
            ("groups", (0, 3, 7), np.float32),
            ("plane_ids", (0, 1), np.int64),
            ("slice_coords", (1, 0), np.float32),
        ):
            with self.subTest(key=key, shape=wrong_shape):
                bundle = api.make_empty_bundle((9, 11, 13), 7)
                bundle[key] = np.empty(wrong_shape, dtype=dtype)
                with self.assertRaises(ValueError):
                    api.validate_empty_bundle(bundle, (9, 11, 13), 7)

    def test_validator_rejects_wrong_dtypes_and_implicit_array_coercion(self):
        api = self.api()
        for key, dtype in (
            ("groups", np.float64), ("groups", np.uint8),
            ("plane_ids", np.int32), ("plane_ids", np.float32),
            ("slice_coords", np.float64), ("slice_coords", np.int64),
        ):
            with self.subTest(key=key, dtype=dtype):
                bundle = api.make_empty_bundle((9, 11, 13), 7)
                bundle[key] = bundle[key].astype(dtype)
                with self.assertRaises((TypeError, ValueError)):
                    api.validate_empty_bundle(bundle, (9, 11, 13), 7)
        for key in ("groups", "plane_ids", "slice_coords"):
            with self.subTest(nonarray_key=key):
                bundle = api.make_empty_bundle((9, 11, 13), 7)
                bundle[key] = []
                with self.assertRaises((TypeError, ValueError)):
                    api.validate_empty_bundle(bundle, (9, 11, 13), 7)

    def test_validator_requires_full_canonical_depth(self):
        api = self.api()
        for depth in (8, 10, 0, -1, True, 9.0, float("nan"), "9"):
            with self.subTest(depth=depth):
                bundle = api.make_empty_bundle((9, 11, 13), 7)
                bundle["canonical_depth"] = depth
                with self.assertRaises((TypeError, ValueError)):
                    api.validate_empty_bundle(bundle, (9, 11, 13), 7)

    def test_validator_rejects_missing_keys_and_every_hidden_metadata_key(self):
        api = self.api()
        valid = api.make_empty_bundle((9, 11, 13), 7)
        for key in valid:
            with self.subTest(missing=key):
                bundle = dict(valid)
                del bundle[key]
                with self.assertRaises(ValueError):
                    api.validate_empty_bundle(bundle, (9, 11, 13), 7)
        for key in ("gt", "target", "prompt_package", "prompt_cache", "selected_planes",
                    "coronal_slice", "sagittal_slice", "center_plane", "foreground_support",
                    "raw_foreground_z_support", "foreground_bbox_dhw", "roi_indices",
                    "background_points_dhw", "audit", "arbitrary_hidden_metadata"):
            with self.subTest(extra=key):
                bundle = dict(valid, **{key: None})
                with self.assertRaises(ValueError):
                    api.validate_empty_bundle(bundle, (9, 11, 13), 7)
        for bundle in (None, [], ()):
            with self.subTest(nonmapping=bundle):
                with self.assertRaises((TypeError, ValueError)):
                    api.validate_empty_bundle(bundle, (9, 11, 13), 7)

    def test_full_ct_indices_cover_all_original_frames_in_order(self):
        api = self.api()
        for depth in (1, 9, 33, 513):
            with self.subTest(depth=depth):
                actual = api.full_ct_indices((depth, 11, 13))
                self.assertIsInstance(actual, np.ndarray)
                self.assertEqual(actual.dtype, np.dtype(np.int32))
                np.testing.assert_array_equal(actual, np.arange(depth, dtype=np.int32))

    def test_input_apis_accept_no_gt_package_cache_plane_or_roi_arguments(self):
        api = self.api()
        signatures = (
            (api.make_empty_bundle, ("canonical_shape_dhw", "prompt_plane_size"), ((9, 11, 13), 7)),
            (api.full_ct_indices, ("canonical_shape_dhw",), ((9, 11, 13),)),
            (api.validate_empty_bundle, ("bundle", "canonical_shape_dhw", "prompt_plane_size"),
             (api.make_empty_bundle((9, 11, 13), 7), (9, 11, 13), 7)),
            (api.semantic_modes, (), ()),
        )
        for function, names, args in signatures:
            with self.subTest(function=function.__name__):
                self.assertEqual(tuple(inspect.signature(function).parameters), names)
                for key in ("gt", "target", "prompt_package", "cache", "selected_planes", "roi"):
                    with self.assertRaises(TypeError):
                        function(*args, **{key: object()})

    def test_modes_express_no_spatial_input_with_native_scribble_compatibility(self):
        api = self.api()
        self.assertEqual(api.semantic_modes(), {
            "spatial_prompt_policy": "none",
            "active_planes": (), "plane_ids": (),
            "foreground_mode": "scribble", "background_mode": "scribble",
            "foreground_mode_id": 0, "background_mode_id": 0,
            "foreground_type_id": 0, "background_type_id": 3,
            "background_active": False, "role_ids": (0,), "type_ids": (0,),
        })

    def test_builders_return_fresh_state(self):
        api = self.api()
        first = api.make_empty_bundle((9, 11, 13), 7)
        second = api.make_empty_bundle((9, 11, 13), 7)
        self.assertIsNot(first, second)
        for key in ("groups", "plane_ids", "slice_coords"):
            self.assertIsNot(first[key], second[key])
        indices = api.full_ct_indices((9, 11, 13))
        indices[0] = 7
        self.assertEqual(api.full_ct_indices((9, 11, 13))[0], 0)
        modes = api.semantic_modes()
        modes["background_active"] = True
        self.assertIs(api.semantic_modes()["background_active"], False)


if __name__ == "__main__":
    unittest.main()
