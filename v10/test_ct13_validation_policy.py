import unittest
import importlib.util


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('ct13_validation_policy'), 'validation policy helper missing')

    def test_weights_use_nonempty_catalog_counts_not_sampling(self):
        from ct13_validation_policy import class_count_weights
        split = {'source_order': ['a', 'b'], 'class_catalog': [
            {'dataset': 'a', 'global_class_id': 1},
            {'dataset': 'a', 'global_class_id': 2},
            {'dataset': 'b', 'global_class_id': 3},
            {'dataset': 'b', 'global_class_id': 4, 'not_applicable': True}]}
        self.assertEqual(class_count_weights(split), {'a': 2/3, 'b': 1/3})

    def test_two_fixed_topcow_and_new_class_tasks(self):
        from ct13_validation_policy import validation_two_per_class
        tasks = [{'dataset': 'topcow2024_cta', 'global_class_id': 1, 'case_id': str(i)} for i in range(4)]
        tasks += [{'dataset': 'lndb', 'global_class_id': 2, 'case_id': str(i)} for i in range(2)]
        split = {'class_catalog': [{'dataset': 'topcow2024_cta', 'global_class_id': 1}, {'dataset': 'lndb', 'global_class_id': 2}]}
        result = validation_two_per_class(split, {'tasks': tasks})
        self.assertEqual([t['case_id'] for t in result['tasks'][:2]], ['0', '1'])
        self.assertEqual(len(result['tasks']), 4)
        self.assertEqual(len(tasks), 6)
        with self.assertRaisesRegex(ValueError, 'two distinct'):
            validation_two_per_class(split, {'tasks': tasks[:-1]})

    def test_task_macro_reuses_values_and_counts(self):
        from ct13_validation_policy import case_class_macro
        sources = {'a': {'per_prompt_mode': {'scribble': {'count': 2, 'dice': .8, 'precision': .6}}},
                   'b': {'per_prompt_mode': {'scribble': {'count': 1, 'dice': .2, 'precision': .3}}}}
        result = case_class_macro(sources)
        self.assertEqual(result['count'], 3)
        self.assertAlmostEqual(result['mean']['dice'], .6)
        self.assertAlmostEqual(result['mean']['precision'], .5)


if __name__ == '__main__':
    unittest.main()
