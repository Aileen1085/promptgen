"""Regression tests for per-3D-component two-plane validation prompts."""

import unittest

import numpy as np

from utils.prompt_utils import select_scribble_plane_slices
from v10_2_component_scribble import select_component_plane_slices
from v10_2_component_scribble_ablation import (
    component_cache_namespace,
    memory_safe_eval_args,
    parse_launcher_args,
    pathology_tasks,
)


class ComponentPlaneSelectionTests(unittest.TestCase):
    def setUp(self):
        mask = np.zeros((12, 14, 16), dtype=np.uint8)
        mask[2:5, 2:5, 2:5] = 1
        mask[8:11, 10:13, 12:15] = 1
        self.mask = mask

    def test_each_disconnected_target_gets_coronal_and_sagittal_plane(self):
        for plane, axis in (("coronal", 2), ("sagittal", 1)):
            with self.subTest(plane=plane):
                old = select_scribble_plane_slices(self.mask, plane, num_slices=1)
                new = select_component_plane_slices(self.mask, plane, num_slices=1)
                self.assertEqual(len(old), 1)
                self.assertEqual(len(new), 2)
                for bounds in ((slice(2, 5), slice(2, 5), slice(2, 5)),
                               (slice(8, 11), slice(10, 13), slice(12, 15))):
                    component = self.mask[bounds]
                    indices = range(bounds[axis].start, bounds[axis].stop)
                    self.assertTrue(any(index in new for index in indices))
                    self.assertTrue(component.any())

    def test_empty_target_has_no_component_plane(self):
        empty = np.zeros((4, 5, 6), dtype=np.uint8)
        self.assertEqual(select_component_plane_slices(empty, "coronal", num_slices=1), [])

    def test_selection_does_not_change_existing_single_component(self):
        mask = self.mask.copy()
        mask[8:11, 10:13, 12:15] = 0
        for plane in ("coronal", "sagittal"):
            self.assertEqual(
                select_component_plane_slices(mask, plane, num_slices=1),
                select_scribble_plane_slices(mask, plane, num_slices=1),
            )


class PathologyCohortTests(unittest.TestCase):
    def test_only_fixed_heldout_pathology_tasks_are_selected(self):
        validation = {"tasks": [
            {"dataset": "magic", "case_id": "val1", "global_class_id": 135},
            {"dataset": "lndb", "case_id": "val2", "global_class_id": 159},
            {"dataset": "totalseg", "case_id": "val3", "global_class_id": 69},
            {"dataset": "magic", "case_id": "val4", "global_class_id": 138},
        ]}
        split = {"datasets": {
            "magic": {"train": [{"case_id": "train1"}], "val": [{"case_id": "val1"}]},
            "lndb": {"train": [{"case_id": "train2"}], "val": [{"case_id": "val2"}]},
        }}
        selected = pathology_tasks(split, validation)
        self.assertEqual([(row["dataset"], row["global_class_id"]) for row in selected],
                         [("magic", 135), ("lndb", 159)])

    def test_launcher_config_is_parsed_without_executing_training(self):
        command = "exec env CT13_FAMILY=v10_2 /bin/python -u /proj/ct13_training_entry.py --gpu 5 --validation-seed 2027"
        self.assertEqual(parse_launcher_args(command),
                         ["--gpu", "5", "--validation-seed", "2027"])

    def test_memory_safe_eval_changes_only_batching_and_feature_storage(self):
        options = ["--gpu", "5", "--mask-threshold", "0.60"]
        result = memory_safe_eval_args(options, gpu=2)
        self.assertEqual(result[-6:], [
            "--gpu", "2", "--sam2-frame-batch-size", "8",
            "--sam2-feature-cache-device", "cpu",
        ])
        self.assertEqual(result[:len(options)], options)

    def test_component_cache_is_isolated_without_changing_case_ct_root(self):
        class Dataset:
            PROMPT_PLANE_CACHE_VERSION = "adaptive_v2"
            PROMPT_CACHE_VERSION = 2100000
            cache_dir = "/shared/case_ct"
            v10_lightweight_roi_metadata = True
        dataset = Dataset()
        component_cache_namespace(dataset)
        self.assertEqual(dataset.PROMPT_PLANE_CACHE_VERSION,
                         "adaptive_v2_per3d_component_26c_v1")
        self.assertEqual(dataset.PROMPT_CACHE_VERSION, 1002100000)
        self.assertEqual(dataset.cache_dir, "/shared/case_ct")


if __name__ == "__main__":
    unittest.main()
