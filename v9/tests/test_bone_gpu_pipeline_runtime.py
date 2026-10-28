import copy
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch


def _process_lease(job, start, queue, fail):
    from tools.bone_gpu_pipeline_runtime import GpuLease
    start.wait(10)
    try:
        with GpuLease(job, snapshot=lambda: {2: dict(uuid='GPU-two', free=50000)},
                      synchronize=lambda: None, empty_cache=lambda: None)('raw') as event:
            time.sleep(.06)
            if fail:
                raise RuntimeError('forward failure')
    except RuntimeError:
        pass
    queue.put(event)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(Path('tools/bone_gpu_pipeline_runtime.py').exists(), 'GPU lease runtime is missing')
        self.m = importlib.import_module('tools.bone_gpu_pipeline_runtime')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.job = dict(gpu=2, gpu_uuid='GPU-two', slot=0, minimum_free_mib=30720,
                        run=str(self.out/'runs'/'case_a0'), spec=str(self.out/'job_specs'/'case_a0.json'),
                        model='v9_2', control=False)
        Path(self.job['spec']).parent.mkdir()
        self.save_job()
        self.events = []
        self.snapshot = {2: dict(uuid='GPU-two', free=50000)}
        for name, callback in (
            ('_flock', lambda handle, exclusive: self.events.append('lock' if exclusive else 'unlock')),
            ('_gpu_snapshot', lambda: copy.deepcopy(self.snapshot)),
            ('_synchronize', lambda: self.events.append('sync')),
            ('_empty_cache', lambda: self.events.append('empty')),
        ):
            p = patch.object(self.m, name, callback)
            p.start(); self.addCleanup(p.stop)

    def save_job(self):
        Path(self.job['spec']).write_text(json.dumps(self.job), encoding='utf-8')

    def configured(self):
        self.m.configure_job(self.job)
        return self.m.GpuLease(self.job)

    def test_configuration_binds_signed_spec_and_rejects_resource_changes(self):
        self.configured()
        for field, value in (('gpu', 0), ('slot', 3), ('minimum_free_mib', 100), ('gpu_uuid', '')):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.m.configure_job(dict(self.job, **{field: value}))
        with self.assertRaises(ValueError):
            self.m.configure_job(dict(self.job, run=self.job['run']+'other'))

    def test_exception_releases_after_synchronizing_and_nested_calls_share_lock(self):
        outer, inner = self.configured(), self.m.GpuLease(self.job)
        with self.assertRaisesRegex(RuntimeError, 'OOM'):
            with outer('raw') as event:
                with inner('native_probe') as nested:
                    self.events.append('forward')
                    raise RuntimeError('OOM')
        self.assertEqual(self.events, ['lock', 'forward', 'sync', 'empty', 'unlock'])
        self.assertTrue(event['synchronized'])
        self.assertTrue(nested['nested'])
        self.assertEqual(nested['parent_id'], event['id'])
        with outer('raw'):
            pass
        self.assertEqual(self.events.count('unlock'), 2)

    def test_memory_wait_uses_fresh_snapshot_after_exclusive_lock(self):
        lease = self.configured()
        self.snapshot[2]['free'] = 30719
        with self.assertRaises(SystemExit) as caught:
            with lease('raw'):
                self.fail('low-memory forward ran')
        self.assertEqual(caught.exception.code, 75)
        self.assertEqual(self.events[0], 'lock')
        self.assertEqual(self.events[-1], 'unlock')
        self.snapshot[2]['free'] = 50000
        with lease('raw') as event:
            pass
        self.assertEqual(event['free_mib'], 50000)

    def test_uuid_guard_and_cleanup_failure_release_lock(self):
        lease = self.configured()
        self.snapshot[2]['uuid'] = 'GPU-wrong'
        with self.assertRaisesRegex(ValueError, 'UUID'):
            with lease('raw'):
                self.fail('wrong GPU forward ran')
        self.assertEqual(self.events[-1], 'unlock')
        self.snapshot[2]['uuid'] = 'GPU-two'
        with patch.object(self.m, '_synchronize', side_effect=RuntimeError('sync failed')):
            failing = self.m.GpuLease(self.job)
            with self.assertRaisesRegex(RuntimeError, 'sync failed'):
                with failing('raw'):
                    pass
        self.assertEqual(self.events[-1], 'unlock')
        with lease('raw'):
            pass

    def new_session(self, callback=None):
        from tools import ct13_full_validation_eval as b
        p = patch.object(b, 'metric_module', b.metric_module)
        p.start(); self.addCleanup(p.stop)
        self.m.configure_job(self.job)
        self.task = dict(source_name='bone', case_id='case1', global_class_id=10003, class_name='test')
        session = self.m.GridSession(self.job['model'], [dict(task=self.task, floors=[.35, .4])],
                                     'test-source', callback or (lambda record: None))
        self.addCleanup(session.close)
        return session

    def restoration(self, session):
        session.restoration = dict(model='v9_2', epoch=360, weight_mode='live',
            prompt_parameters_exact=438, decoder_parameters_exact=192, guards_passed=True)
        session.training_audit = dict(training_dataset_constructions_skipped=1)

    def test_startup_covers_initialization_and_waits_for_restoration(self):
        session = self.new_session()
        self.assertEqual(self.events, ['lock'])
        with self.assertRaisesRegex(ValueError, 'restoration'):
            session.begin(self.task)
        self.assertEqual(self.events, ['lock'])
        self.restoration(session)
        session.begin(self.task)
        self.assertEqual(self.events, ['lock', 'sync', 'empty', 'unlock'])
        session.begin(self.task)
        self.assertEqual(self.events.count('unlock'), 1)

    def test_failed_parent_initialization_releases_startup(self):
        self.m.configure_job(self.job)
        with patch.object(self.m.PriorSession, '__init__', side_effect=RuntimeError('model init')):
            with self.assertRaisesRegex(RuntimeError, 'model init'):
                self.m.GridSession('v9_2', [], 'test', lambda r: None)
        self.assertEqual(self.events, ['lock', 'sync', 'empty', 'unlock'])

    def test_actual_raw_producer_is_guarded_cache_hits_and_cpu_metrics_are_outside(self):
        import torch
        saved = []
        session = self.new_session(saved.append)
        self.restoration(session); session.begin(self.task)
        args = SimpleNamespace(model_input_size=1024, max_frames=192, validation_window_frames=192)
        video = torch.ones(2, 1, 2, 2)
        calls = []
        def compute():
            calls.append(self.events[-1])
            return video.clone()
        from tools import bone_floor_protocol as p
        for floor in (.35, .4):
            session.set_floor(floor, args)
            session.raw(compute, video, {}, 10003, args)
            self.assertEqual(self.events[-1], 'unlock')
            session.capture(args, dict(threshold=.6, source='calibrated'))
            session.emit(self.task, dict.fromkeys(p.METRICS, .7), dict(threshold=.6, source='calibrated'))
        # Use the real shared completion callback; metric proof remains the parent's responsibility.
        with patch('tools.bone_metric_speed_runtime.validate_metric_record'):
            session.finish(self.task)
        self.assertEqual(calls, ['lock'])
        self.assertEqual(self.events.count('unlock'), 2)
        record = saved[0]
        self.assertIn('metric_execution', record)
        self.m.validate_pipeline_record(record, self.job)
        self.assertEqual(len(record['pipeline_execution']['raw_events']), 1)
        producer = next(iter(record['raw_producers'].values()))
        self.assertTrue(producer['gpu_lease']['cpu_output'])
        for bad in ('slot', 'gpu_uuid', 'source_sha256', 'spec_sha256'):
            invalid = copy.deepcopy(record)
            invalid['pipeline_execution'][bad] = 'changed'
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.m.validate_pipeline_record(invalid, self.job)
        invalid = copy.deepcopy(record)
        next(iter(invalid['raw_producers'].values())).pop('gpu_lease')
        with self.assertRaises(ValueError):
            self.m.validate_pipeline_record(invalid, self.job)
        invalid = copy.deepcopy(record)
        invalid['pipeline_execution']['raw_events'][0]['synchronized'] = False
        with self.assertRaises(ValueError):
            self.m.validate_pipeline_record(invalid, self.job)
        for field, value in (('id', record['pipeline_execution']['startup_lease']['id']),
                             ('nested', True), ('parent_id', 'invented-parent')):
            invalid = copy.deepcopy(record)
            event = invalid['pipeline_execution']['raw_events'][0]
            event[field] = value
            invalid['raw_producers'][event['raw_key']]['gpu_lease'] = copy.deepcopy(event)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.m.validate_pipeline_record(invalid, self.job)
        invalid = copy.deepcopy(record)
        invalid['restoration']['guards_passed'] = False
        with self.assertRaises(ValueError):
            self.m.validate_pipeline_record(invalid, self.job)
        self.job.update(model='v10_2', control=True)
        self.save_job()
        invalid = copy.deepcopy(record)
        invalid['model'] = 'v10_2'
        invalid['restoration'].update(model='v10_2', epoch=490,
                                      prompt_parameters_exact=418, decoder_parameters_exact=212)
        invalid['pipeline_execution'].update(self.m._binding(self.job))
        with self.assertRaisesRegex(ValueError, 'native twin'):
            self.m.validate_pipeline_record(invalid, self.job)

    def test_raw_low_memory_and_forward_error_release_without_producer_proof(self):
        import torch
        session = self.new_session()
        self.restoration(session); session.begin(self.task)
        args = SimpleNamespace(model_input_size=1024, max_frames=192, validation_window_frames=192)
        session.set_floor(.35, args)
        video = torch.ones(2, 1, 2, 2)
        self.snapshot[2]['free'] = 1
        with self.assertRaises(SystemExit) as caught:
            session.raw(lambda: self.fail('insufficient-memory producer ran'), video, {}, 10003, args)
        self.assertEqual(caught.exception.code, 75)
        self.assertEqual(self.events[-1], 'unlock')
        self.snapshot[2]['free'] = 50000
        def oom():
            raise RuntimeError('CUDA out of memory')
        with self.assertRaisesRegex(RuntimeError, 'CUDA out of memory'):
            session.raw(oom, video, {}, 10003, args)
        self.assertEqual(self.events[-3:], ['sync', 'empty', 'unlock'])
        self.assertFalse(session.raw_proofs)
        self.assertFalse(session.raw_events)
        session.raw(lambda: video.clone(), video, {}, 10003, args)
        self.assertEqual(len(session.raw_events), 1)

    def test_raw_cpu_assertion_fails_without_device_transfer(self):
        import torch
        with self.assertRaisesRegex(ValueError, 'CPU'):
            self.m.assert_cpu_tree({'logits': [torch.empty(1, device='meta')]})
        value = {'logits': [torch.ones(1)]}
        self.assertIs(self.m.assert_cpu_tree(value), value)

    def test_native_probe_is_wrapped_before_inherited_installer(self):
        from tools import bone_shared_floor_restore_runtime as restore
        session = self.new_session()
        self.restoration(session); session.begin(self.task)
        native = restore.probe_native_batches
        self.addCleanup(setattr, restore, 'probe_native_batches', native)
        def inherited(s):
            self.assertIsNot(restore.probe_native_batches, native)
        with patch.object(self.m, '_install_v10', inherited):
            self.m.install_v10(session)
        import torch
        args = SimpleNamespace(sam2_frame_batch_size=2)
        def raw():
            self.assertEqual(self.events[-1], 'lock')
            return torch.ones(1), ['encoder-sha']
        result = restore.probe_native_batches(raw, raw, args)
        self.assertEqual(set(result), {1, 2, 4})
        self.assertEqual(args.sam2_frame_batch_size, 2)
        event = session.native_probe_leases[0]
        self.assertTrue(event['synchronized'])
        self.assertTrue(event['cpu_output'])
        self.assertEqual(self.events[-1], 'unlock')

    def test_native_probe_failure_restores_batch_and_releases_lease(self):
        from tools import bone_shared_floor_restore_runtime as restore
        session = self.new_session()
        self.restoration(session); session.begin(self.task)
        native = restore.probe_native_batches
        self.addCleanup(setattr, restore, 'probe_native_batches', native)
        with patch.object(self.m, '_install_v10', lambda s: None):
            self.m.install_v10(session)
        args = SimpleNamespace(sam2_frame_batch_size=2)
        def oom():
            raise RuntimeError('native CUDA out of memory')
        with self.assertRaisesRegex(RuntimeError, 'native CUDA out of memory'):
            restore.probe_native_batches(oom, oom, args)
        self.assertEqual(args.sam2_frame_batch_size, 2)
        self.assertEqual(self.events[-3:], ['sync', 'empty', 'unlock'])
        self.assertFalse(session.native_probe_leases)

    @unittest.skipUnless(os.name == 'posix', 'real flock exclusion requires POSIX/server')
    def test_real_processes_exclude_each_other_and_release_after_exception(self):
        self.configured()
        context = multiprocessing.get_context('spawn')
        start, queue = context.Event(), context.Queue()
        workers = [context.Process(target=_process_lease, args=(self.job, start, queue, n == 0)) for n in range(3)]
        for process in workers:
            process.start()
        start.set()
        records = [queue.get(timeout=20) for _ in workers]
        for process in workers:
            process.join(timeout=10)
            self.assertEqual(process.exitcode, 0)
        records.sort(key=lambda event: event['start_monotonic'])
        self.assertTrue(all(left['end_monotonic'] <= right['start_monotonic']
                            for left, right in zip(records, records[1:])))


if __name__ == '__main__':
    unittest.main()
