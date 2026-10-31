"""CPU contract tests; real GPU/reference parity is checked by the capture CLI."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / 'tools/capture_amos_liver_case.py'
try:
    from tools import capture_amos_liver_case as cap
except ImportError:
    cap = None


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(cap, 'single-case capture implementation is missing')

    def test_help_does_not_import_torch_or_native_models(self):
        result = subprocess.run([sys.executable, str(SOURCE), '--help'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--method', result.stdout)
        self.assertNotIn('torch', sys.modules)

    def test_decode_observer_retains_continuous_logits_not_metric_surrogate(self):
        true_logits = np.array([[[[-1.7, 0.33], [2.4, -0.29]]]], np.float32)
        post = true_logits[:, 0] > .4
        decoded = (true_logits, post, (1, 2), [], {'threshold': .7}, None)
        full = np.zeros((3, 2, 2)); full[1] = post[0]
        ns = {'_dynamic_adaptive_decode': lambda: decoded,
              'restore_and_score_full_volume': lambda *a, **k: ({'dice': 1}, full)}
        original = ns.copy()
        with cap.observe_v10(ns) as observed:
            self.assertIs(ns['_dynamic_adaptive_decode'](), decoded)
            ns['restore_and_score_full_volume'](post[:, None].astype(np.float32),
                frame_indices=np.array([1]), spacing_dhw=(3, 1, 1))
        np.testing.assert_array_equal(observed['logits_roi'], true_logits[:, 0])
        np.testing.assert_array_equal(observed['prediction_dhw'], full)
        self.assertEqual(ns, original)

    def test_v10_hooks_restore_on_exception(self):
        ns = {'_dynamic_adaptive_decode': object(), 'restore_and_score_full_volume': object()}
        original = ns.copy()
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            with cap.observe_v10(ns):
                raise RuntimeError('injected')
        self.assertEqual(ns, original)

    def test_final_stage_and_reverse_visits_overwrite_previous_logits(self):
        stages = cap.NativeFrames()
        stages.begin()
        stages.record(0, np.full((4, 4), 9.))
        stages.begin()
        stages.record(0, np.full((4, 4), -.3))
        stages.record(1, np.full((4, 4), 1.4))
        stages.record(0, np.full((4, 4), .2))
        prediction = np.ones((2, 2, 4), bool)
        logits = stages.finish(prediction, (0, 1))
        self.assertEqual(logits.dtype, np.float32)
        np.testing.assert_array_equal(logits[0], np.full((2, 4), np.float32(.2)))
        self.assertEqual(len(stages.frames), 2)
        with self.assertRaisesRegex(ValueError, 'threshold'):
            stages.finish(~prediction, (0, 1))

    def test_missing_native_frame_fails(self):
        stages = cap.NativeFrames(); stages.begin(); stages.record(0, np.ones((3, 3)))
        with self.assertRaisesRegex(ValueError, 'frame'):
            stages.finish(np.ones((2, 3, 3), bool), (0, 0))

    def test_full_volume_placement_uses_dhw_without_transpose(self):
        roi = np.array([[[1, 0, 1], [0, 1, 0]], [[0, 1, 0], [1, 0, 1]]], bool)
        placed = cap.place_roi(roi, (5, 6, 8), [1, 4], ((1, 5), (2, 4), (3, 6)))
        np.testing.assert_array_equal(placed[1, 2:4, 3:6], roi[0])
        np.testing.assert_array_equal(placed[4, 2:4, 3:6], roi[1])
        self.assertEqual(int(placed.sum()), int(roi.sum()))

    def test_v10_native_postprocessed_mask_mapping_matches_nearest_then_unpad(self):
        self.assertTrue(hasattr(cap, 'restore_v10_prediction'), 'native mask mapping helper is missing')
        roi = np.array([[[1, 0], [0, 1]]], bool)
        result = cap.restore_v10_prediction(roi, (3, 2, 4), [1], None)
        np.testing.assert_array_equal(result[1], [[1, 1, 0, 0], [0, 0, 1, 1]])
        self.assertFalse(result[0].any())
        geom = types.SimpleNamespace(bounds=(1, 3, 0, 2), full_side=4)
        result = cap.restore_v10_prediction(roi, (3, 4, 4), [2], geom)
        expected = np.zeros((4, 4), np.uint8); expected[:2, 1:3] = roi[0]
        np.testing.assert_array_equal(result[2], expected)

    def test_actual_cache_bindings_must_match_selected_readonly_inputs(self):
        self.assertTrue(hasattr(cap, 'validate_actual_inputs'), 'actual input binding helper is missing')
        selection = dict(ct_cache_path='shared/case.npy', image_path='data/image.nii.gz',
                         prompt_package_path='shared/prompt.npz', prompt_package_sha256='a'*64)
        event = dict(path='shared/case.npy', image='data/image.nii.gz', mode='read_only_hit')
        cap.validate_actual_inputs(selection, 'shared/prompt.npz', 'a'*64, [event])
        for changed in (dict(event, mode='current_case_ram_miss'), dict(event, path='other.npy')):
            with self.assertRaisesRegex(ValueError, 'CT'):
                cap.validate_actual_inputs(selection, 'shared/prompt.npz', 'a'*64, [changed])
        with self.assertRaisesRegex(ValueError, 'prompt'):
            cap.validate_actual_inputs(selection, 'other.npz', 'a'*64, [event])

    def test_plane_coordinates_use_fixed_w_for_coronal_fixed_h_for_sagittal(self):
        mask = np.zeros((4, 5), bool); mask[2, 3] = True
        np.testing.assert_array_equal(cap.plane_coords(mask, 'coronal', 7), [[2, 3, 7]])
        np.testing.assert_array_equal(cap.plane_coords(mask, 'sagittal', 7), [[2, 7, 3]])
        self.assertEqual(cap.plane_coords(~np.ones((2, 3), bool), 'coronal', 1).shape, (0, 3))

    def test_native_points_convert_xy_and_local_roi_offsets(self):
        stage = {'bounds_dhw': [[1, 5], [10, 20], [30, 40]],
                 'point_provenance': [{'z': 3, 'fg_sampled_xy': [[4, 2]], 'bg_original_yx': [[1, 5]]}]}
        fg, bg = cap.native_points(stage)
        np.testing.assert_array_equal(fg, [[3, 12, 34]])
        np.testing.assert_array_equal(bg, [[3, 11, 35]])

    def test_capture_write_rejects_overwrite_and_ct_voxels(self):
        arrays = dict(logits_roi=np.array([[[-.4, .8]]], np.float32),
                      prediction_dhw=np.zeros((1, 1, 2), np.uint8), frame_indices=np.array([0]),
                      bounds_dhw=np.array([[0, 1], [0, 1], [0, 2]]),
                      prompt_fg_dhw=np.empty((0, 3), int), prompt_bg_dhw=np.empty((0, 3), int))
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            metadata = cap.save_capture(out, {'schema': 'amos_liver_capture_v1'}, arrays)
            self.assertEqual(len(metadata['capture_npz_sha256']), 64)
            with self.assertRaises(FileExistsError):
                cap.save_capture(out, {}, arrays)
            with np.load(out/'capture.npz', allow_pickle=False) as saved:
                self.assertEqual(set(saved.files), set(arrays))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'unexpected'):
                cap.save_capture(Path(directory), {}, dict(arrays, ct=np.zeros((1, 2, 3))))

    def test_nonfinite_or_binary_surrogate_logits_rejected(self):
        for value in (np.array([[[np.nan]]]), np.array([[[0., 1.]]]), np.array([[[True, False]]])):
            with self.assertRaises(ValueError):
                cap.continuous_logits(value)

    def test_all_ten_reference_metrics_checked(self):
        row = {m: .9 for m in cap.METRICS}
        cap.compare_metrics(row, row)
        for metric in cap.METRICS:
            changed = dict(row, **{metric: .8})
            with self.assertRaisesRegex(ValueError, metric):
                cap.compare_metrics(changed, row)

    def test_selected_v10_roi_and_threshold_metadata_must_match(self):
        row = dict({m: .9 for m in cap.METRICS}, roi_final_frames=49, roi_reserved_frames=69,
                   roi_initial_frames=49, roi_auto_expand_iterations=0, mask_threshold_used=.7)
        for field in ('roi_final_frames', 'roi_reserved_frames', 'roi_initial_frames',
                      'roi_auto_expand_iterations', 'mask_threshold_used'):
            changed = dict(row, **{field: 3})
            with self.assertRaisesRegex(ValueError, field):
                cap.compare_metrics(changed, row)

    def test_native_observer_keeps_original_outputs_and_only_final_stage(self):
        class Model:
            image_size = 1024
            value = .3
            def propagate_in_video(self, *a, **k):
                for frame in (0, 1, 0):
                    yield frame, [1], np.full((1, 1, 4, 4), self.value, np.float32)
        model = Model(); returned = []
        def predict(*a):
            outputs = list(model.propagate_in_video())
            returned.append(outputs)
            return np.full((2, 2, 4), model.value > 0), {'square_pad_left_top': (0, 1)}
        original = object()
        base = types.SimpleNamespace(predict_roi=predict, overlay=original)
        builder = types.SimpleNamespace(build_sam2_video_predictor=lambda *a, **k: model)
        with cap.observe_native(base, builder, {'config': 'config', 'image_size': 1024}) as observed:
            builder.build_sam2_video_predictor('old', 'weights')
            base.predict_roi(); model.value = -.4; last = base.predict_roi()
            base.overlay(None, None, None, None, np.zeros((3, 2, 4), bool))
        self.assertIs(base.overlay, original)
        self.assertEqual(len(returned), 2)
        self.assertEqual(returned[-1][0][2][0, 0, 0, 0], np.float32(-.4))
        np.testing.assert_array_equal(observed['logits_roi'], np.full((2, 2, 4), np.float32(-.4)))
        np.testing.assert_array_equal(observed['roi_prediction'], last[0])

    def test_native_hooks_restore_after_failure(self):
        class Model:
            def propagate_in_video(self, *a, **k):
                yield 0, [1], np.full((1, 1, 2, 2), .3)
        model = Model()
        def predict(*a):
            list(model.propagate_in_video())
            raise RuntimeError('injected native error')
        base = types.SimpleNamespace(predict_roi=predict, overlay=object())
        builder = types.SimpleNamespace(build_sam2_video_predictor=lambda *a, **k: model)
        old = (base.predict_roi, base.overlay, builder.build_sam2_video_predictor, model.propagate_in_video)
        with self.assertRaisesRegex(RuntimeError, 'injected native'):
            with cap.observe_native(base, builder, {'config': 'config', 'image_size': 1024}):
                model.image_size = 1024
                builder.build_sam2_video_predictor('ignored', 'weights')
                base.predict_roi(model, None, None, None, None, None, None)
        self.assertEqual((base.predict_roi, base.overlay, builder.build_sam2_video_predictor, model.propagate_in_video), old)


if __name__ == '__main__':
    unittest.main()
