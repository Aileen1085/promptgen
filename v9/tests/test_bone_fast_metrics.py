"""Exact metric acceleration against the project's native (not copied) metrics."""
import gc
import importlib
import importlib.util
import math
from pathlib import Path
import threading
import time
import unittest
from unittest.mock import patch
import weakref

import torch


ROOT = Path(__file__).resolve().parents[1]


def native_module():
    spec = importlib.util.spec_from_file_location("_bone_test_native_metrics", ROOT / "utils" / "metrics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExactMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.native = native_module()

    def provider(self, **kwargs):
        self.assertTrue((ROOT / "tools" / "bone_fast_metrics.py").is_file(), "the exact metric provider has not been implemented")
        module = importlib.import_module("tools.bone_fast_metrics")
        return module.install_provider(self.native, **kwargs)

    def assertMetricsEqual(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for key, value in expected.items():
            with self.subTest(metric=key):
                if math.isnan(value):
                    self.assertTrue(math.isnan(actual[key]))
                else:
                    self.assertEqual(actual[key], value)

    def masks(self, depth=5, side=19):
        pred = torch.zeros(depth, side, side)
        target = torch.zeros_like(pred)
        pred[:, 3:12, 4:13] = 1
        target[:, 4:14, 6:15] = 1
        pred[0] = 0
        target[1] = 0
        pred[-1] = 0
        target[-1] = 0
        return pred, target

    def test_native_equivalence_supported_shapes_empty_slices_and_full_volume_fn(self):
        fast, tracker = self.provider(workers=1)
        pred, target = self.masks()
        pairs = [(pred[2], target[2]), (pred, target), (pred[:, None], target[:, None]),
                 (pred[None, None], target[None, None]), (pred.transpose(1, 2), target.transpose(1, 2)),
                 (torch.stack((pred, target), dim=1)[None], torch.stack((target, pred), dim=1)[None]),
                 (torch.stack((pred, target), dim=1), torch.stack((target, pred), dim=1))]
        for p, t in pairs:
            with self.subTest(shape=tuple(p.shape), contiguous=p.is_contiguous()):
                self.assertMetricsEqual(fast(p, t, (2.7, .61, 1.29)), self.native.binary_prompt_metrics_from_masks(p, t, (2.7, .61, 1.29)))
        # A missed target beyond prediction support remains a full-volume false negative.
        before = fast(pred, target)
        target[-1, 1, 1] = 1
        after = fast(pred, target)
        self.assertLess(after["recall"], before["recall"])
        self.assertMetricsEqual(after, self.native.binary_prompt_metrics_from_masks(pred, target))
        self.assertEqual(tracker.snapshot()["fast_calls"], 9)

    def test_empty_masks_keep_native_nan_and_one_sided_semantics(self):
        fast, _ = self.provider(workers=2)
        empty = torch.zeros(4, 7, 9)
        full = torch.ones_like(empty)
        for p, t in ((empty, empty), (empty, full), (full, empty)):
            self.assertMetricsEqual(fast(p, t), self.native.binary_prompt_metrics_from_masks(p, t))

    def test_float32_threshold_negative_values_and_exact_tolerance_boundaries(self):
        fast, _ = self.provider(workers=1)
        pred = torch.zeros(3, 9, 11, dtype=torch.float64)
        target = torch.zeros_like(pred)
        for z, distance in enumerate((1, 2, 3)):
            pred[z, 3, 3] = 1
            target[z, 3, 3 + distance] = 1
        pred[:, 0, 0] = -1
        pred[:, 0, 1] = .5 + 1e-10  # rounds to .5 before the native comparison
        pred[:, 0, 2] = .5
        self.assertMetricsEqual(fast(pred, target, (1., 1., 1.)), self.native.binary_prompt_metrics_from_masks(pred, target, (1., 1., 1.)))
        for spacing in ((1., .7, 1. + 1e-5), (1., .7, 1. - 1e-5)):
            self.assertMetricsEqual(fast(pred, target, spacing), self.native.binary_prompt_metrics_from_masks(pred, target, spacing))

    def test_one_cdist_and_two_directed_minima_per_nonempty_paired_slice(self):
        fast, tracker = self.provider(workers=1)
        pred, target = self.masks(depth=7)
        real = torch.cdist
        real_min = torch.Tensor.min
        min_calls = []

        def counted_min(tensor, *args, **kwargs):
            min_calls.append(kwargs.get("dim"))
            return real_min(tensor, *args, **kwargs)

        with patch.object(torch.Tensor, "min", new=counted_min):
            with patch.object(torch, "cdist", wraps=real) as counted:
                expected = self.native.binary_prompt_metrics_from_masks(pred, target)
                native_count = counted.call_count
                self.assertEqual(len(min_calls), native_count * 2)
                counted.reset_mock()
                min_calls.clear()
                actual = fast(pred, target)
                fast_count = counted.call_count
                self.assertEqual(min_calls, [1, 0] * fast_count)
        self.assertEqual(native_count, 3 * fast_count)
        self.assertEqual(fast_count, 4)
        self.assertEqual(tracker.snapshot()["fast_cdist_calls"], fast_count)
        self.assertMetricsEqual(actual, expected)

    def test_two_workers_are_bounded_ordered_and_numerically_deterministic(self):
        pred, target = self.masks(depth=11)
        serial, _ = self.provider(workers=1)
        expected = serial(pred, target, (1., .77, 1.13))
        parallel, _ = self.provider(workers=2)
        lock = threading.Lock()
        active = 0
        peak = 0
        real = torch.cdist

        def observed(*args, **kwargs):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                time.sleep(.01)
                return real(*args, **kwargs)
            finally:
                with lock:
                    active -= 1

        with patch.object(torch, "cdist", side_effect=observed):
            actual = parallel(pred, target, (1., .77, 1.13))
        self.assertEqual(peak, 2)
        self.assertMetricsEqual(actual, expected)

    def test_cache_exact_normalized_inputs_changes_and_class_clear(self):
        fast, tracker = self.provider(workers=1, cache_entries=8)
        pred, target = self.masks()
        expected = fast(pred, target, (1., 1., 1.))
        equivalent_pred = pred * 6 - 2
        self.assertMetricsEqual(fast(equivalent_pred, target.clone(), (1., 1., 1.)), expected)
        self.assertEqual(tracker.snapshot()["cache_hits"], 1)
        # The returned dictionary must not alias the memoized values.
        expected["dice"] = -123
        self.assertNotEqual(fast(pred, target, (1., 1., 1.))["dice"], -123)
        changed_pred = pred.clone()
        changed_pred[2, 0, 0] = 1
        changed_target = target.clone()
        changed_target[-1, 0, 0] = 1
        for p, t, s in ((changed_pred, target, (1., 1., 1.)), (pred, changed_target, (1., 1., 1.)),
                        (pred, target, (1., 1., 1.1)), (pred[:, None], target[:, None], (1., 1., 1.))):
            self.assertMetricsEqual(fast(p, t, s), self.native.binary_prompt_metrics_from_masks(p, t, s))
        self.assertEqual(tracker.snapshot()["cache_misses"], 5)
        tracker.clear()
        self.assertEqual(tracker.snapshot()["cache_entries"], 0)
        fast(pred, target, (1., 1., 1.))
        self.assertEqual(tracker.snapshot()["cache_misses"], 6)

    def test_weighted_target_bypasses_binary_cache_and_preserves_native_volume_formulas(self):
        fast, tracker = self.provider(workers=1)
        pred, target = self.masks()
        binary = fast(pred, target)
        weighted = target.clone()
        weighted[2, 5, 8] = .25
        expected = self.native.binary_prompt_metrics_from_masks(pred, weighted)
        self.assertNotEqual(binary["dice"], expected["dice"])
        for _ in range(2):
            self.assertMetricsEqual(fast(pred, weighted), expected)
        self.assertEqual(tracker.snapshot()["cache_bypasses"], 2)
        self.assertEqual(tracker.snapshot()["cache_hits"], 0)

    def test_chunked_signatures_check_late_target_values_and_float32_pred_semantics(self):
        module = importlib.import_module("tools.bone_fast_metrics")
        fast, tracker = self.provider(workers=1)
        pred, target = self.masks()
        with patch.object(module, "_SIGNATURE_CHUNK_VOXELS", 32):
            baseline = fast(pred, target)
            self.assertMetricsEqual(fast(pred.clone(), target.clone()), baseline)
            weighted = target.clone()
            weighted[-1, -1, -1] = .5
            self.assertMetricsEqual(fast(pred, weighted), self.native.binary_prompt_metrics_from_masks(pred, weighted))
            changed = pred.double()
            changed[-1, -1, -1] = .5 + 1e-10
            self.assertMetricsEqual(fast(changed, target), baseline)
            changed[-1, -1, -1] = .5001
            self.assertMetricsEqual(fast(changed, target), self.native.binary_prompt_metrics_from_masks(changed, target))
        self.assertEqual(tracker.snapshot()["cache_hits"], 1)
        self.assertEqual(tracker.snapshot()["cache_misses"], 3)
        self.assertEqual(tracker.snapshot()["cache_bypasses"], 1)

    def test_cache_separates_input_strides_and_native_reduction_thread_settings(self):
        fast, tracker = self.provider(workers=1)
        pred, target = self.masks()
        fast(pred, target)
        # Same logical values with another physical traversal can change native
        # float32 reductions for huge volumes.  Conservatively separate layouts.
        noncontig_pred = pred.transpose(1, 2).contiguous().transpose(1, 2)
        noncontig_target = target.transpose(1, 2).contiguous().transpose(1, 2)
        self.assertFalse(noncontig_pred.is_contiguous())
        self.assertMetricsEqual(fast(noncontig_pred, noncontig_target), self.native.binary_prompt_metrics_from_masks(noncontig_pred, noncontig_target))
        self.assertEqual(tracker.snapshot()["cache_misses"], 2)
        try:
            torch.set_num_threads(1)
            self.assertMetricsEqual(fast(pred, target), self.native.binary_prompt_metrics_from_masks(pred, target))
            self.assertEqual(tracker.snapshot()["cache_misses"], 3)
        finally:
            torch.set_num_threads(2)

    def test_cache_separates_dtype_dependent_float_conversion_layouts(self):
        fast, tracker = self.provider(workers=1, compare_native=True)
        generator = torch.Generator().manual_seed(0)
        row = (torch.rand(1, 1, 2, 1, 19, generator=generator) < .9).float()
        target32 = row.expand(1, 1, 2, 19, 19)
        target64 = row.double().expand_as(target32)
        self.assertEqual(target32.stride(), target64.stride())
        self.assertNotEqual(target32.float().stride(), target64.float().stride())
        pred = torch.ones(target32.shape)
        for target in (target32, target64):
            self.assertMetricsEqual(fast(pred, target), self.native.binary_prompt_metrics_from_masks(pred, target))
        # .float() leaves expanded float32 strides intact, but converting an
        # expanded float64 tensor materializes a different reduction layout.
        # Huge binary volumes can then have different native float32 sums.
        snap = tracker.snapshot()
        self.assertEqual(snap["cache_misses"], 2)
        self.assertEqual(snap["cache_hits"], 0)
        self.assertEqual(snap["comparisons"], 2)
        self.assertEqual(snap["comparison_max_abs_error"], 0.)

    def test_cache_is_bounded_metadata_only_and_retains_no_input_tensors(self):
        fast, tracker = self.provider(workers=2, cache_entries=2)
        pred, target = self.masks()
        pred_ref, target_ref = weakref.ref(pred), weakref.ref(target)
        for x_spacing in (1., 1.1, 1.2):
            fast(pred, target, (1., 1., x_spacing))
        snap = tracker.snapshot()
        self.assertEqual(snap["cache_entries"], 2)
        self.assertEqual(snap["cache_evictions"], 1)
        self.assertEqual(snap["cache_tensor_bytes"], 0)
        del pred, target
        gc.collect()
        self.assertIsNone(pred_ref())
        self.assertIsNone(target_ref())
        fast(torch.zeros(3, 19, 19), torch.zeros(3, 19, 19), (1., 1., 1.))

    def test_dynamic_native_comparison_includes_cache_hits_and_original_native_globals(self):
        enabled = [False]
        fast, tracker = self.provider(workers=1, compare_native=lambda: enabled[0])
        pred, target = self.masks()
        expected = self.native.binary_prompt_metrics_from_masks(pred, target)
        fast(pred, target)
        enabled[0] = True
        # Installing the wrapper as the public export cannot make the baseline call itself.
        self.native.binary_prompt_metrics_from_masks = fast
        self.native.mean_slice_nsd_scores = lambda *a, **kw: {"mean_slice_nsd_1mm": -999}
        self.assertMetricsEqual(fast(pred, target), expected)
        snap = tracker.snapshot()
        self.assertEqual(snap["cache_hits"], 1)
        self.assertEqual(snap["native_calls"], 1)
        self.assertEqual(snap["comparisons"], 1)
        self.assertEqual(snap["native_cdist_calls"], 6)
        self.assertEqual(snap["comparison_max_abs_error"], 0.)
        self.assertEqual(snap["native_source_sha256"], __import__("hashlib").sha256((ROOT / "utils" / "metrics.py").read_bytes()).hexdigest())

    def test_native_control_detects_corrupt_fast_result_even_on_cache_hit(self):
        enabled = [False]
        fast, tracker = self.provider(workers=1, compare_native=lambda: enabled[0])
        pred, target = self.masks()
        fast(pred, target)
        self.native.surface_voxels = lambda mask: torch.zeros_like(mask, dtype=torch.bool)
        # Existing cache remains exact; corrupt only the tracker-held cached metadata.
        cached = next(iter(tracker._cache.values()))
        cached["dice"] += .01
        enabled[0] = True
        with self.assertRaisesRegex(AssertionError, "dice"):
            fast(pred, target)
        self.assertEqual(tracker.snapshot()["comparison_failures"], 1)

    def test_native_validation_and_worker_bounds(self):
        fast, _ = self.provider(workers=1)
        with self.assertRaisesRegex(ValueError, "shape"):
            fast(torch.zeros(2, 3), torch.zeros(3, 2))
        with self.assertRaisesRegex(ValueError, "length 3"):
            fast(torch.zeros(2, 3), torch.zeros(2, 3), (1., 1.))
        for workers in (0, 3, 8):
            with self.assertRaisesRegex(ValueError, "workers"):
                self.provider(workers=workers)


if __name__ == "__main__":
    unittest.main()
