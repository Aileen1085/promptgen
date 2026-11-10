import unittest
from tools import five_source_fg_bg_scribble_runtime as r
from tools import five_source_prompt_type_runtime as original

class RuntimeTests(unittest.TestCase):
    def test_isolated_variant_extension(self):
        self.assertEqual(r.VARIANTS, ('baseline','fg_scribble_bg_scribble'))
        self.assertNotIn('fg_scribble_bg_scribble', original.VARIANTS)
        self.assertEqual(r._native.VARIANTS, r.VARIANTS)
    def test_all_physical_cards_and_floor(self):
        for gpu in range(8):
            self.assertEqual(r.resource_snapshot(f'{gpu}, GPU-test, 30720',gpu,'GPU-test')['gpu'],gpu)
        for text,gpu,uuid in [('8, GPU-test, 50000',8,'GPU-test'),('1, GPU-wrong, 50000',1,'GPU-test'),('1, GPU-test, 30719',1,'GPU-test')]:
            with self.assertRaises(ValueError):r.resource_snapshot(text,gpu,uuid)
    def test_private_hooks_keep_target_validator(self):
        self.assertIs(r._native.private_hooks('magic').validate_record,r._native.validate_record)
        self.assertIs(r._native.resource_snapshot,r.resource_snapshot)

class BindingTests(unittest.TestCase):
    def test_repeated_binding_and_large_retry_floor_are_checked(self):
        from unittest.mock import patch
        good=dict(gpu=7,gpu_uuid='GPU-test',minimum_free_mib=45000,free_mib_before_load=46000,cuda_probe=True,
                  first_snapshot=dict(gpu=7,gpu_uuid='GPU-test',free_mib=47000),
                  second_snapshot=dict(gpu=7,gpu_uuid='GPU-test',free_mib=46000))
        import copy
        with patch.object(r,'_validate_native',return_value=True):
            self.assertTrue(r.validate_record(dict(resource_binding=good),{},'baseline',2))
            for field,value in [('gpu_uuid','GPU-wrong'),('free_mib',44999),('gpu',3)]:
                bad=copy.deepcopy(good);bad['first_snapshot'][field]=value
                with self.assertRaises(ValueError):r.validate_record(dict(resource_binding=bad),{},'baseline',2)
            bad=copy.deepcopy(good);bad['free_mib_before_load']=47000
            with self.assertRaises(ValueError):r.validate_record(dict(resource_binding=bad),{},'baseline',2)
