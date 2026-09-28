import unittest
from types import SimpleNamespace
from pathlib import Path

class CT13Tests(unittest.TestCase):
    def test_manifest_ratios_not_eight_source_defaults(self):
        from sam2_v9_2_ct13_entry import configured_ratios
        from ct_thirteen_source_protocol import SOURCE_ORDER
        quotas = dict(zip(SOURCE_ORDER, (40,18,32,10,8,22,5,25,8,6,8,12,6)))
        split = dict(source_order=list(SOURCE_ORDER), sampling_ratios={s:q/200 for s,q in quotas.items()})
        self.assertEqual(configured_ratios(split), split['sampling_ratios'])
        with self.assertRaises(ValueError):
            configured_ratios(dict(source_order=['totalseg'],sampling_ratios={'totalseg':1}))

    def test_selection_restores_training_ratios_even_on_failure(self):
        from sam2_v9_2_ct13_entry import evaluate_with_class_weights
        args = SimpleNamespace(multidataset_sampling_ratios={'training':1})
        def evaluator(*unused):
            self.assertEqual(args.multidataset_sampling_ratios, {'validation':1})
            raise RuntimeError('probe')
        with self.assertRaisesRegex(RuntimeError, 'probe'):
            evaluate_with_class_weights(evaluator, {'validation':1}, None,None,None,None,args,None)
        self.assertEqual(args.multidataset_sampling_ratios, {'training':1})

    def test_launcher_is_additive_precision_not_full_or_medsam(self):
        text = Path('run_train_sam2_v9_2_thirteen_source.sh').read_text()
        self.assertIn('sam2_v9_2_ct13_entry.py',text)
        self.assertIn('ct13_v10_2_validation_topcow2_classweight_v2_20260928.json',text)
        self.assertIn('Refusing',text)
        self.assertNotIn('finetune_multisource_sam2_v9_2_full',text)

if __name__ == '__main__':
    unittest.main()
