"""Behavior checks for the additive single-round nnInteractive runtime."""
import copy
from contextlib import nullcontext
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import uuid
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / 'nninteractive_test/infer/prompt_protocol.py'
if not HELPER.is_file():
    HELPER = ROOT.parent / 'nninteractive_test_amos/infer/prompt_protocol.py'
spec = importlib.util.spec_from_file_location('_bone_single_native_prompt', HELPER)
native_prompt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native_prompt)


def test_folder():
    # Keep tiny synthetic fixtures; project policy forbids recursive cleanup.
    folder = ROOT / 'output'
    folder.mkdir(exist_ok=True)
    return nullcontext(tempfile.mkdtemp(prefix='bone-single-runtime-test-', dir=folder))


def package():
    coronal = np.zeros((5, 7, 9), dtype=np.uint8)
    sagittal = np.zeros_like(coronal)
    coronal[1:4, 2:5, 4] = 1
    sagittal[1:4, 3, 3:6] = 1
    return dict(coronal=coronal, sagittal=sagittal, coronal_slice=4,
                sagittal_slice=3,
                background_points_dhw=np.array([[0, 0, 4], [4, 6, 4],
                                                 [0, 3, 0], [4, 3, 8]]),
                background_point_planes=['coronal', 'coronal', 'sagittal', 'sagittal'],
                case_image_name='shared.npy')


class RecordingSession:
    def __init__(self):
        self.preprocess_calls = 0
        self.reset_calls = 0
        self.predictions = []
        self.additions = []
        self.buffers = []
        self.target_buffer = None
        self.do_autozoom = True

    def set_image(self, image):
        self.preprocess_calls += 1
        self.image = image.copy()
        self.additions = []

    def _finish_preprocessing_and_initialize_interactions(self):
        pass

    def reset_interactions(self):
        self.reset_calls += 1
        self.additions = []
        if self.target_buffer is not None:
            self.target_buffer.zero_()

    def set_target_buffer(self, buffer):
        assert not bool(buffer.any()), 'previous class mask leaked into fresh target'
        self.target_buffer = buffer
        self.buffers.append(buffer)

    def add_point_interaction(self, coordinates, include_interaction, run_prediction=True):
        self.additions.append(('point', tuple(coordinates), include_interaction))
        if run_prediction:
            self._predict()

    def add_scribble_interaction(self, scribble_image, include_interaction, run_prediction=True):
        self.additions.append(('scribble', scribble_image.copy(), include_interaction))
        if run_prediction:
            self._predict()

    def _predict(self):
        self.predictions.append(list(self.additions))
        self.target_buffer[1:4, 2:5, 3:6] = 1


class SingleRoundTests(unittest.TestCase):
    def runtime(self):
        self.assertTrue((ROOT / 'tools/bone_nninteractive_runtime.py').is_file(),
                        'single-round runtime is not implemented')
        return importlib.import_module('tools.bone_nninteractive_runtime')

    def predictor(self, session, helper=None):
        return self.runtime().SingleRoundPredictor(
            session, helper or native_prompt.apply_two_plane_prompt)

    def test_actual_predict_sees_all_six_native_additions_once(self):
        session = RecordingSession()
        prediction, audit = self.predictor(session).predict(np.ones((5, 7, 9), np.float32), package())
        self.assertEqual(len(session.predictions), 1)
        seen = session.predictions[0]
        self.assertEqual([v[0] for v in seen], ['point'] * 4 + ['scribble'] * 2)
        self.assertEqual([v[2] for v in seen], [False] * 4 + [True] * 2)
        self.assertEqual([v[1] for v in seen[:4]], [tuple(p) for p in package()['background_points_dhw']])
        self.assertTrue(torch.equal(prediction, torch.from_numpy(np.pad(np.ones((3, 3, 3)), ((1, 1), (2, 2), (3, 3)))).float()))
        self.assertEqual(prediction.device.type, 'cpu')
        self.assertEqual(audit['prediction_trigger_count'], 1)
        self.assertEqual(audit['prompt_addition_count'], 6)
        self.assertEqual(audit['background_points'], 4)
        self.assertEqual(audit['foreground_scribbles'], 2)
        self.assertEqual(audit['events'][-1]['kind'], 'predict')
        self.assertEqual(audit['events'][-1]['prompt_addition_count'], 6)
        self.assertTrue(session.do_autozoom)
        self.assertFalse(bool(session.target_buffer.any()))

    def test_same_case_reuses_preprocessing_and_new_case_does_not(self):
        session = RecordingSession()
        predictor = self.predictor(session)
        image = np.ones((5, 7, 9), np.float32)
        first, audit1 = predictor.predict(image, package())
        second, audit2 = predictor.predict(image, package())
        self.assertEqual(session.preprocess_calls, 1)
        self.assertFalse(audit1['case_image_reused'])
        self.assertTrue(audit2['case_image_reused'])
        self.assertEqual(audit2['preprocess_calls'], 1)
        self.assertIsNot(session.buffers[0], session.buffers[1])
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(all(len(v) == 6 for v in session.predictions))
        predictor.predict(image.copy(), package())
        self.assertEqual(session.preprocess_calls, 2)

    def test_early_actual_trigger_is_rejected_and_methods_restored(self):
        session = RecordingSession()
        original = {name: getattr(session, name) for name in
                    ('_predict', 'add_point_interaction', 'add_scribble_interaction')}
        def early(s, *_args):
            s._predict()
        with self.assertRaisesRegex(ValueError, 'complete|six|6|early'):
            self.predictor(session, early).predict(np.ones((5, 7, 9)), package())
        self.assertEqual(len(session.predictions), 0)
        for name, method in original.items():
            self.assertEqual(getattr(session, name), method)
            self.assertNotIn(name, session.__dict__)

    def test_second_actual_trigger_is_rejected_before_native_call(self):
        session = RecordingSession()
        def twice(s, *args):
            native_prompt.apply_two_plane_prompt(s, *args)
            s._predict()
        with self.assertRaisesRegex(ValueError, 'second|one|1'):
            self.predictor(session, twice).predict(np.ones((5, 7, 9)), package())
        self.assertEqual(len(session.predictions), 1)

    def test_missing_or_changed_addition_is_rejected(self):
        for helper in (
                lambda s, c, g, p: None,
                lambda s, c, g, p: native_prompt.apply_two_plane_prompt(s, c, g, p[::-1]),
                lambda s, c, g, p: native_prompt.apply_two_plane_prompt(s, g, c, p)):
            with self.subTest(helper=helper), self.assertRaises(ValueError):
                self.predictor(RecordingSession(), helper).predict(np.ones((5, 7, 9)), package())

    def test_exception_restores_instance_overrides_and_resets_state(self):
        session = RecordingSession()
        def custom_predict():
            raise RuntimeError('native prediction failed')
        session._predict = custom_predict
        with self.assertRaisesRegex(RuntimeError, 'native prediction failed'):
            self.predictor(session).predict(np.ones((5, 7, 9)), package())
        self.assertIs(session._predict, custom_predict)
        self.assertFalse(session.additions)
        self.assertFalse(bool(session.target_buffer.any()))

    def test_invalid_prompt_geometry_and_budget_fail_before_preprocessing(self):
        changes = []
        bad = package(); bad['sagittal'] = np.ones((2, 2, 2)); changes.append(bad)
        bad = package(); bad['coronal'][0, 0, 0] = 1; changes.append(bad)
        bad = package(); bad['coronal'][:] = 0; changes.append(bad)
        bad = package(); bad['background_points_dhw'] = bad['background_points_dhw'][:3]; changes.append(bad)
        bad = package(); bad['background_point_planes'] = ['sagittal'] * 4; changes.append(bad)
        bad = package(); bad['background_points_dhw'][0, 0] = 99; changes.append(bad)
        bad = package(); bad['background_points_dhw'][0, 2] = 1; changes.append(bad)
        for bad in changes:
            with self.subTest(change=bad.keys()):
                session = RecordingSession()
                with self.assertRaises(ValueError):
                    self.predictor(session).predict(np.ones((5, 7, 9)), bad)
                self.assertEqual(session.preprocess_calls, 0)


class CaseRuntimeTests(unittest.TestCase):
    def runtime(self):
        return importlib.import_module('tools.bone_nninteractive_runtime')

    def fixture(self, folder):
        root = Path(folder)
        model = root / 'model'
        (model / 'fold_0').mkdir(parents=True)
        for name in ('plans.json', 'dataset.json', 'inference_session_class.json'):
            (model / name).write_text(json.dumps({'dataset_name': 'native-test'}), encoding='utf-8')
        (model / 'fold_0/checkpoint_final.pth').write_bytes(b'fake-model-test-only')
        (root / 'native.py').write_text('# signed test dependency', encoding='utf-8')
        image_path, label_path = root / 'image.nii.gz', root / 'label.nii.gz'
        image_path.write_bytes(b'fake-CT-test-only'); label_path.write_bytes(b'fake-GT-test-only')
        image = np.ones((5, 7, 9), np.float32)
        labels = np.zeros(image.shape, np.int32)
        labels[1:4, 2:5, 3:6] = 1
        labels[0, 0, 0] = 2
        cache = root / 'cache'
        (cache / '_case_images_v1').mkdir(parents=True)
        np.save(cache / '_case_images_v1/shared.npy', image)
        prompt_root = cache / 'metadata/shared_adaptive_prompt_v1/bone'
        prompt_root.mkdir(parents=True)
        tasks, refs, hashes = [], [], {}
        for local_id in (1, 2):
            name = f'prompt{local_id}.npz'
            (prompt_root / name).write_bytes(b'frozen-prompt-test-only')
            hashes[name] = hashlib.sha256((prompt_root / name).read_bytes()).hexdigest()
            task = dict(source_name='bone', case_id='Bone001', global_class_id=10000 + local_id,
                        local_class_id=local_id, class_name=f'label{local_id}',
                        image='image.nii.gz', label='label.nii.gz', split_role='validation')
            tasks.append(task)
            refs.append(dict(source_name='bone', case_id='Bone001', global_class_id=10000 + local_id,
                             prompt_cache_name=name))
        model_files = {str(p.relative_to(root)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in model.rglob('*') if p.is_file()}
        config = dict(model_dir='model', cache_root='cache', prompt_hashes=hashes,
                      prompt_refs=refs, model_files=model_files,
                      sources={'native.py': hashlib.sha256((root / 'native.py').read_bytes()).hexdigest()})
        # Fake-session provenance is explicit; this remains a CPU behavior test.
        for source in (Path(__file__), HELPER):
            relative = os.path.relpath(source, root).replace(os.sep, '/')
            config['sources'][relative] = hashlib.sha256(source.read_bytes()).hexdigest()
        job = dict(name='Bone001', case_id='Bone001', tasks=tasks, gpu=2, gpu_uuid='GPU-test',
                   minimum_free_mib=30720, run=str(root / 'run'), spec=str(root / 'job.json'), log=str(root / 'log'))
        session = RecordingSession()
        session.trainer_name = 'native-test-trainer'
        session.initialize_from_trained_model_folder = lambda path: None
        constructor_calls = []
        def constructor(**kwargs):
            constructor_calls.append(kwargs)
            return session
        loads = []
        def load_raw(path, canonical):
            loads.append(Path(path).name)
            return labels if Path(path).name == 'label.nii.gz' else image
        protocol = SimpleNamespace(EIGHT_SOURCES=('totalseg',),
            AdaptiveValidationTask=lambda **kw: SimpleNamespace(**kw),
            load_task_target=lambda t, source_loader: (source_loader(t.label_path) == t.local_class_id).astype(np.uint8))
        ev = SimpleNamespace(nnInteractiveInferenceSession=constructor,
            patch_nninteractive_trainer_for_current_nnunet=lambda: None,
            load_raw_axial_volume=load_raw,
            load_axial_shape_spacing_dhw=lambda p, canonical: (image.shape, (2., 1., 1.)),
            apply_two_plane_prompt=native_prompt.apply_two_plane_prompt)
        cache_mod = SimpleNamespace(
            shared_case_cache_path=lambda *args: cache / '_case_images_v1/shared.npy',
            _valid_case_cache=lambda p: np.load(p, mmap_mode='r'))
        prompts = SimpleNamespace(_read=lambda p: package())
        resource = dict(gpu=2, gpu_uuid='GPU-test', actual_start_free_mib=40000,
                        minimum_free_mib=30720, cuda_uuid='GPU-test')
        return SimpleNamespace(**locals())

    def test_case_reads_frozen_prompts_and_shared_ct_once_preserving_local_ids(self):
        runtime = self.runtime()
        self.assertTrue(callable(getattr(runtime, 'run_case', None)), 'run_case has not been implemented')
        with test_folder() as folder:
            f = self.fixture(folder)
            records = []
            with patch('tools.ct13_full_validation_eval.nn_modules', return_value=(f.protocol, f.ev)), \
                    patch.object(runtime, '_gpu_resource', return_value=f.resource), \
                    patch.dict('sys.modules', multisource_split_protocol=f.cache_mod, adaptive_prompt_cache=f.prompts):
                runtime.run_case(f.root, f.job, f.config, records.append)
            self.assertEqual(len(records), 2)
            self.assertEqual(f.session.preprocess_calls, 1)
            self.assertEqual(f.loads, ['label.nii.gz'])
            self.assertEqual(len(f.constructor_calls), 1)
            self.assertTrue(f.constructor_calls[0]['do_autozoom'])
            self.assertEqual([r['row']['label_id'] for r in records], [1, 2])
            self.assertEqual([r['row']['gt_voxels'] for r in records], [27, 1])
            self.assertEqual(records[0]['metric_execution']['comparisons'], 1)
            self.assertLessEqual(records[0]['metric_execution']['max_error'], 1e-6)
            self.assertIn('session_source', records[0]['model'])
            self.assertEqual(records[0]['model']['session_source']['sha256'], hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            self.assertEqual(records[0]['model']['prompt_helper_source']['sha256'], hashlib.sha256(HELPER.read_bytes()).hexdigest())
            for record, task in zip(records, f.tasks):
                runtime.validate_record(record, task, f.config, f.job)
                self.assertEqual(record['task'], task)
                self.assertEqual(record['prompt_sha256'], f.hashes[record['prompt_cache_name']])
                self.assertEqual(record['geometry']['shape_dhw'], [5, 7, 9])
                json.dumps(record, allow_nan=False)
            damaged = copy.deepcopy(records[0]); damaged['row']['label_id'] = 10001
            with self.assertRaises(ValueError):
                runtime.validate_record(damaged, f.tasks[0], f.config, f.job)
            damaged = copy.deepcopy(records[0]); damaged['interaction']['events'][-1]['prompt_addition_count'] = 5
            with self.assertRaises(ValueError):
                runtime.validate_record(damaged, f.tasks[0], f.config, f.job)

    def test_changed_prompt_signature_fails_without_regeneration_or_commit(self):
        runtime = self.runtime()
        self.assertTrue(callable(getattr(runtime, 'run_case', None)), 'run_case has not been implemented')
        with test_folder() as folder:
            f = self.fixture(folder)
            f.config['prompt_hashes']['prompt1.npz'] = '0' * 64
            records = []
            with patch('tools.ct13_full_validation_eval.nn_modules', return_value=(f.protocol, f.ev)), \
                    patch.object(runtime, '_gpu_resource', return_value=f.resource), \
                    patch.dict('sys.modules', multisource_split_protocol=f.cache_mod, adaptive_prompt_cache=f.prompts), \
                    self.assertRaisesRegex(ValueError, 'prompt|signature'):
                runtime.run_case(f.root, f.job, f.config, records.append)
            self.assertFalse(records)
            self.assertFalse(f.session.predictions)

    def test_cache_miss_keeps_case_ct_in_ram_without_writing_a_new_cache(self):
        runtime = self.runtime()
        with test_folder() as folder:
            f = self.fixture(folder)
            f.cache_mod._valid_case_cache = lambda path: None
            before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in f.cache.rglob('*') if p.is_file()}
            records = []
            with patch('tools.ct13_full_validation_eval.nn_modules', return_value=(f.protocol, f.ev)), \
                    patch.object(runtime, '_gpu_resource', return_value=f.resource), \
                    patch.dict('sys.modules', multisource_split_protocol=f.cache_mod, adaptive_prompt_cache=f.prompts):
                runtime.run_case(f.root, f.job, f.config, records.append)
            self.assertEqual(f.loads.count('image.nii.gz'), 1)
            self.assertEqual(f.loads.count('label.nii.gz'), 1)
            self.assertTrue(all(r['geometry']['image_cache_source'] == 'source_memory_only' for r in records))
            self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in f.cache.rglob('*') if p.is_file()})

    def test_imported_session_cannot_shadow_frozen_source_bundle(self):
        runtime = self.runtime()
        with test_folder() as folder:
            f = self.fixture(folder)
            f.config['sources'].pop(os.path.relpath(__file__, f.root).replace(os.sep, '/'))
            with patch('tools.ct13_full_validation_eval.nn_modules', return_value=(f.protocol, f.ev)), \
                    patch.object(runtime, '_gpu_resource', return_value=f.resource), \
                    patch.dict('sys.modules', multisource_split_protocol=f.cache_mod, adaptive_prompt_cache=f.prompts), \
                    self.assertRaisesRegex(ValueError, 'source|bundle|session'):
                runtime.run_case(f.root, f.job, f.config, lambda r: self.fail('shadowed native session committed a result'))
            self.assertFalse(f.session.predictions)


class ResourceTests(unittest.TestCase):
    def test_below_30gib_exits_75_before_cuda_or_model_initialization(self):
        runtime = importlib.import_module('tools.bone_nninteractive_runtime')
        job = dict(gpu=2, gpu_uuid='GPU-test', minimum_free_mib=30720)
        with patch.object(runtime.subprocess, 'check_output', return_value='GPU-test, 30719\n'), \
                patch('torch.cuda.is_available', side_effect=AssertionError('CUDA was initialized')):
            with self.assertRaises(SystemExit) as raised:
                runtime._gpu_resource(job)
        self.assertEqual(raised.exception.code, 75)

    def test_cuda_driver_uuid_must_match_assigned_physical_gpu(self):
        runtime = importlib.import_module('tools.bone_nninteractive_runtime')
        physical = 'GPU-11111111-2222-3333-4444-555555555555'
        job = dict(gpu=2, gpu_uuid=physical, minimum_free_mib=30720)
        def driver(cuda_uuid):
            def device(pointer, index):
                pointer._obj.value = index
                return 0
            def uuid_call(pointer, device):
                for index, value in enumerate(uuid.UUID(cuda_uuid[4:]).bytes):
                    pointer._obj[index] = value
                return 0
            return SimpleNamespace(cuInit=lambda flags: 0, cuDeviceGet=device,
                                   cuDeviceGetUuid=uuid_call, cuDeviceGetUuid_v2=uuid_call)
        probe = torch.zeros(1)
        for actual, valid in ((physical, True), ('GPU-aaaaaaaa-2222-3333-4444-555555555555', False)):
            with patch.object(runtime.subprocess, 'check_output', return_value=f'{physical}, 40000\n'), \
                    patch('torch.cuda.is_available', return_value=True), \
                    patch('torch.cuda.device_count', return_value=1), \
                    patch('torch.cuda.set_device'), patch('torch.cuda.synchronize'), \
                    patch('torch.zeros', return_value=probe), patch('ctypes.CDLL', return_value=driver(actual)):
                if valid:
                    resource = runtime._gpu_resource(job)
                    self.assertEqual(resource['cuda_uuid'], physical)
                    self.assertEqual(resource['actual_start_free_mib'], 40000)
                else:
                    with self.assertRaisesRegex(ValueError, 'UUID'):
                        runtime._gpu_resource(job)


if __name__ == '__main__':
    unittest.main()
