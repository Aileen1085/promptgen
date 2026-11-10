import importlib
import unittest


class ProtocolTests(unittest.TestCase):
    def runtime(self):
        try:
            return importlib.import_module('tools.five_source_prompt_type_runtime')
        except ModuleNotFoundError:
            self.fail('five-source runtime is missing')

    def test_exact_source_configurations(self):
        m = self.runtime()
        self.assertEqual({s: (c['epoch'], c['floor']) for s, c in m.SOURCES.items()}, {
            'magic': (490, .60), 'msd_task07': (490, .35), 'msd_task10': (15, .35),
            'parse2022': (20, .35), 'kits23': (490, .35)})
        self.assertEqual(sum(c['count'] for c in m.SOURCES.values()) * len(m.VARIANTS), 519)

    def test_resource_uuid_memory_and_fresh_allowed_gpus(self):
        m = self.runtime()
        self.assertEqual(m.resource_snapshot('0, GPU-a, 30720', 0, 'GPU-a')['gpu'], 0)
        for text, gpu, uuid in [('0, GPU-a, 30719',0,'GPU-a'), ('0, GPU-a, 40000',0,'GPU-b'), ('2, GPU-a, 50000',2,'GPU-a')]:
            with self.assertRaises(ValueError): m.resource_snapshot(text,gpu,uuid)

    def test_private_hook_module_preserves_signed_module_settings(self):
        m = self.runtime()
        from tools import amos_prompt_ablation_runtime as signed
        before = dict(signed.FROZEN_SETTINGS)
        private = m.private_hooks('magic')
        self.assertEqual(private.FROZEN_SETTINGS['dynamic_threshold_min'], .60)
        self.assertEqual(signed.FROZEN_SETTINGS, before)
        self.assertIsNot(private, signed)

    def test_oom_retry_only_actual_cuda_error_and_stops_at_one(self):
        m = self.runtime()
        self.assertEqual(m.oom_next_batch('torch.OutOfMemoryError: CUDA out of memory',4),2)
        self.assertEqual(m.oom_next_batch('CUDA out of memory',2),1)
        self.assertIsNone(m.oom_next_batch('CUDA out of memory',1))
        self.assertIsNone(m.oom_next_batch('some ordinary failure',4))


class RestorationTests(unittest.TestCase):
    def test_source_specific_restoration_rejects_missing_encoder_or_wrong_checkpoint(self):
        from tools import five_source_prompt_type_runtime as m
        valid = dict(model='v10_2',epoch=15,weight_mode='live',prompt_parameters_exact=418,decoder_parameters_exact=212,
            guards_passed=True,model_load_count=1,checkpoint=m.SOURCES['msd_task10']['checkpoint'],
            encoder_restore=dict(epoch=15,restored_tensor_count=4,all_tensors_exact=True))
        self.assertTrue(hasattr(m,'validate_restoration'), 'source restoration validator missing')
        self.assertTrue(m.validate_restoration(valid,'msd_task10'))
        for patch in (dict(encoder_restore={}),dict(checkpoint=m.E490),dict(epoch=490),dict(prompt_parameters_exact=1)):
            with self.assertRaises(ValueError): m.validate_restoration(dict(valid,**patch),'msd_task10')


if __name__ == '__main__': unittest.main()
