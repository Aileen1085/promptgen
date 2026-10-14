import unittest
from v10.v10_2_external_first4_eval import select_cases, enable_adapter_ema

class SelectionTests(unittest.TestCase):
    def fixture(self):
        return {'datasets': {s: {'train': ['train'], 'test': ['d','b','a','c','e']} for s in ('graz','pengwin','adrenal')}}
    def test_preserves_saved_order(self):
        self.assertEqual(select_cases(self.fixture())['graz'], ['d','b','a','c'])
    def test_overlap_rejected(self):
        x=self.fixture(); x['datasets']['graz']['train']=['a']
        with self.assertRaises(ValueError): select_cases(x)
    def test_duplicates_rejected(self):
        x=self.fixture(); x['datasets']['graz']['test']=['a']*5
        with self.assertRaises(ValueError): select_cases(x)
    def test_short_list_rejected(self):
        x=self.fixture(); x['datasets']['graz']['test']=['a']
        with self.assertRaises(ValueError): select_cases(x)
    def test_adapter_parameters_included_in_ema(self):
        class Parameter:
            requires_grad=False
        class Module:
            def parameters(self): return [parameter]
        class Decoder:
            prompt_3d_adapter=Module()
            prompt_memory_adapter=Module()
        parameter=Parameter()
        enable_adapter_ema(Decoder())
        self.assertTrue(parameter.requires_grad)
