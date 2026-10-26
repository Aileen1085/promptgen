"""Invariant and failure-protection tests for the independent Bone floor grid."""
import copy
import importlib.util
import math
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / 'tools' / 'bone_floor_protocol.py'
METRICS = ('dice', 'mean_slice_dice', 'iou', 'precision', 'recall', 'specificity',
           'pred_gt_ratio', 'mean_slice_nsd_1mm', 'mean_slice_nsd_2mm', 'mean_slice_nsd_3mm')


def paired_fixture():
    return {
        'test_ready': True,
        'classes': [
            {'label_id': 0, 'name': 'background', 'voxel_count': 900, 'case_count': 2},
            {'label_id': 1, 'name': 'vertebrae_T10（仅椎体）', 'voxel_count': 40, 'case_count': 1},
            {'label_id': 3, 'name': 'vertebrae_L1（仅椎体）', 'voxel_count': 70, 'case_count': 2},
            {'label_id': 21, 'name': 'tibia_left', 'voxel_count': 80, 'case_count': 2},
            {'label_id': 55, 'name': 'rib_left_5', 'voxel_count': 4, 'case_count': 1},
            {'label_id': 74, 'name': 'rib_right_12', 'voxel_count': 0, 'case_count': 0},
        ],
        'cases': [
            {'case_id': '0001_phase_3', 'image': 'ct_raw/0001.nii.gz',
             'label_nifti': 'paired_v1/labels/0001_phase_3.nii.gz',
             'test_ready': True, 'present_foreground_ids': [1, 3, 21]},
            {'case_id': '0009_phase_3', 'image': 'ct_raw/0009.nii.gz',
             'label_nifti': 'paired_v1/labels/0009_phase_3.nii.gz',
             'test_ready': True, 'present_foreground_ids': [3, 21, 55]},
        ],
    }


def base_fixture():
    return {
        'source_order': ['totalseg', 'amos'],
        'class_catalog': [
            {'dataset': 'totalseg', 'global_class_id': 3, 'local_class_id': 3,
             'class_name': 'different_CT13_class', 'semantic_text': 'legacy wording'},
            {'dataset': 'amos', 'global_class_id': 118, 'local_class_id': 1, 'class_name': 'liver'},
        ],
        'datasets': {
            'totalseg': {'train': [{'case_id': 'old_train'}], 'val': [{'case_id': 'old_val'}]},
            'amos': {'train': [], 'val': [], 'custom_metadata': {'keep': [1, 2]}},
        },
        'sampling_ratios': {'totalseg': .7, 'amos': .3},
        'custom': {'history': ['keep']},
    }


class BoneFloorProtocolTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(MODULE.is_file(), 'independent Bone protocol module is missing')
        spec = importlib.util.spec_from_file_location('bone_floor_protocol_tested', MODULE)
        self.p = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.p)
        self.paired = paired_fixture()
        # These paths intentionally need not exist: this module performs no CT IO.
        self.data_root = ROOT / '__nonexistent_bone_protocol_fixture__'

    def tasks(self):
        return self.p.make_tasks(self.paired, self.data_root)

    def metric_rows(self, scores=(.2, .4, .6, .8, .5, .7)):
        return [{**task, **{name: score for name in METRICS},
                 'dynamic_threshold_source': 'adaptive', 'threshold': .6}
                for task, score in zip(self.tasks(), scores)]

    def test_constants_match_frozen_protocol_and_import_needs_no_models(self):
        self.assertEqual(self.p.FLOORS, (.35, .40, .45, .50, .55, .60, .65, .70, .75))
        self.assertEqual(self.p.MODELS, ('v9_2', 'v10_2'))
        self.assertEqual(self.p.GID_OFFSET, 10000)
        self.assertEqual(self.p.METRICS, METRICS)
        script = """
import builtins, importlib.util, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'numpy', 'model', 'v10'):
        raise AssertionError('pure protocol imported model dependencies: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
spec = importlib.util.spec_from_file_location('pure_bone', sys.argv[1])
spec.loader.exec_module(importlib.util.module_from_spec(spec))
"""
        result = subprocess.run([sys.executable, '-c', script, str(MODULE)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_preserves_raw_labels_names_paths_and_input(self):
        before = copy.deepcopy(self.paired)
        rows = self.tasks()
        self.assertEqual(len(rows), 6)
        self.assertEqual(self.paired, before)
        self.assertEqual({r['global_class_id'] for r in rows}, {10001, 10003, 10021, 10055})
        for row in rows:
            self.assertEqual(row['source_name'], 'bone')
            self.assertEqual(row['global_class_id'], 10000 + row['local_class_id'])
            self.assertEqual(row['split_role'], 'validation')
            original = next(c for c in self.paired['cases'] if c['case_id'] == row['case_id'])
            self.assertEqual(row['image'], str((self.data_root / original['image']).resolve()))
            self.assertEqual(row['label'], str((self.data_root / original['label_nifti']).resolve()))
        self.assertEqual(rows[0]['class_name'], 'vertebrae_T10（仅椎体）')

    def test_generic_toy_can_include_47_actual_classes_without_770_assumption(self):
        self.paired['classes'] = [dict(label_id=i, name=f'bone_{i}', voxel_count=2, case_count=2)
                                  for i in range(1, 48)]
        for case in self.paired['cases']:
            case['present_foreground_ids'] = list(range(1, 48))
        self.assertEqual(len(self.tasks()), 94)

    def test_rejects_unready_missing_or_empty_inventory(self):
        mutations = [
            lambda x: x.update(test_ready=False),
            lambda x: x['cases'][0].update(test_ready=False),
            lambda x: x['cases'][0].update(present_foreground_ids=[]),
            lambda x: x['cases'][0].update(present_foreground_ids=[74]),
            lambda x: x['cases'][0].update(present_foreground_ids=[999]),
            lambda x: x['classes'][1].update(name=' '),
            lambda x: x['classes'][1].update(voxel_count=0),
            lambda x: x['classes'][1].update(case_count=0),
            lambda x: x.update(cases=[]),
            lambda x: x.pop('classes'),
            lambda x: x['cases'][0].pop('label_nifti'),
        ]
        for mutate in mutations:
            data = paired_fixture()
            mutate(data)
            with self.subTest(data=data), self.assertRaises(ValueError):
                self.p.make_tasks(data, self.data_root)

    def test_rejects_duplicate_cases_class_definitions_and_presence_ids(self):
        variants = []
        data = paired_fixture(); data['cases'].append(copy.deepcopy(data['cases'][0])); variants.append(data)
        data = paired_fixture(); data['classes'].append(copy.deepcopy(data['classes'][1])); variants.append(data)
        data = paired_fixture(); data['cases'][0]['present_foreground_ids'].append(3); variants.append(data)
        for data in variants:
            with self.subTest(data=data), self.assertRaisesRegex(ValueError, 'duplicate'):
                self.p.make_tasks(data, self.data_root)

    def test_rejects_cross_platform_absolute_traversal_and_missing_paths(self):
        for field in ('image', 'label_nifti'):
            for value in ('../escape.nii.gz', 'ct_raw/../escape.nii.gz', '/escape.nii.gz',
                          r'C:\escape.nii.gz', r'\\host\share\escape.nii.gz',
                          r'ct_raw\..\escape.nii.gz', 'C:escape.nii.gz', '', None):
                data = paired_fixture(); data['cases'][0][field] = value
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, 'path'):
                    self.p.make_tasks(data, self.data_root)

    def test_allows_only_documented_0009_image_symlink_escape(self):
        original_resolve = Path.resolve
        root = self.data_root.resolve()
        target = root.parent / '0009.nii.gz'

        def resolve(path, *args, **kwargs):
            if path == root / 'ct_raw' / '0009.nii.gz':
                return target
            return original_resolve(path, *args, **kwargs)

        with patch.object(Path, 'resolve', resolve):
            row = next(r for r in self.tasks() if r['case_id'] == '0009_phase_3')
        self.assertEqual(row['image'], str(target))

        def other_escape(path, *args, **kwargs):
            if path == root / 'ct_raw' / '0001.nii.gz':
                return root.parent / 'outside.nii.gz'
            return original_resolve(path, *args, **kwargs)

        with patch.object(Path, 'resolve', other_escape), self.assertRaisesRegex(ValueError, 'path'):
            self.tasks()

        def wrong_0009_target(path, *args, **kwargs):
            if path == root / 'ct_raw' / '0009.nii.gz':
                return root.parent / 'different.nii.gz'
            return original_resolve(path, *args, **kwargs)

        with patch.object(Path, 'resolve', wrong_0009_target), self.assertRaisesRegex(ValueError, 'path'):
            self.tasks()

    def test_split_is_deepcopy_preserves_legacy_and_avoids_local_id_collision(self):
        base = base_fixture(); rows = self.tasks()
        before_base = copy.deepcopy(base); before_rows = copy.deepcopy(rows)
        result = self.p.augment_split(base, rows, self.paired['classes'])
        self.assertEqual(base, before_base)
        self.assertEqual(rows, before_rows)
        self.assertEqual(result['source_order'], base['source_order'] + ['bone'])
        self.assertEqual(result['class_catalog'][:2], base['class_catalog'])
        self.assertEqual({r['global_class_id'] for r in result['class_catalog']}, {3, 118, 10001, 10003, 10021, 10055})
        self.assertAlmostEqual(result['sampling_ratios']['totalseg'], .7 * .99)
        self.assertAlmostEqual(result['sampling_ratios']['amos'], .3 * .99)
        self.assertEqual(result['sampling_ratios']['bone'], .01)
        self.assertEqual(result['datasets']['bone']['train'], [])
        val = result['datasets']['bone']['val']
        self.assertEqual(len(val), 2)
        for case in val:
            self.assertEqual(case['modality'], 'ct')
            self.assertEqual(set(case['classes']), {r['global_class_id'] for r in rows if r['case_id'] == case['case_id']})
        result['datasets']['amos']['custom_metadata']['keep'].append(9)
        result['custom']['history'].append('new')
        self.assertEqual(base, before_base)

    def test_bone_semantics_preserve_body_only_and_use_readable_anatomical_text(self):
        additions = [(6, 'vertebrae_L5'), (10, 'tarsal_left'), (22, 'metatarsal_left'),
                     (4, 'phalanges_foot_left')]
        for label, name in additions:
            self.paired['classes'].append(dict(label_id=label, name=name, voxel_count=1, case_count=1))
            self.paired['cases'][0]['present_foreground_ids'].append(label)
        split = self.p.augment_split(base_fixture(), self.tasks(), self.paired['classes'])
        bone = {r['local_class_id']: r for r in split['class_catalog'] if r['dataset'] == 'bone'}
        texts = {i: r['semantic_text'] for i, r in bone.items()}
        self.assertEqual(texts[1], 'vertebral body of thoracic vertebra T10')
        self.assertEqual(texts[3], 'vertebral body of lumbar vertebra L1')
        self.assertEqual(texts[6], 'lumbar vertebra L5')
        self.assertEqual(texts[21], 'left tibia')
        self.assertEqual(texts[10], 'left tarsal bones')
        self.assertEqual(texts[22], 'left metatarsal bones')
        self.assertEqual(texts[4], 'left toe phalanges')
        self.assertEqual(texts[55], 'left rib 5')
        for local, row in bone.items():
            self.assertEqual(row['global_class_id'], 10000 + local)
            self.assertEqual(row['label_value'], local)
            self.assertIsNone(row['parent_semantic_key'])
            self.assertTrue(row['semantic_key']); self.assertTrue(row['equivalence_group'])
            self.assertIn(row['morphology_group'], ('compact', 'tubular', 'sheet', 'sheet_hollow', 'large_context'))

    def test_split_rejects_global_collision_and_inconsistent_task_metadata(self):
        base = base_fixture(); base['class_catalog'][0]['global_class_id'] = 10003
        with self.assertRaisesRegex(ValueError, 'collision'):
            self.p.augment_split(base, self.tasks(), self.paired['classes'])
        rows = self.tasks(); rows[0]['local_class_id'] = 3
        with self.assertRaises(ValueError):
            self.p.augment_split(base_fixture(), rows, self.paired['classes'])
        rows = self.tasks(); rows[1]['image'] = 'different_case_image.nii.gz'
        with self.assertRaises(ValueError):
            self.p.augment_split(base_fixture(), rows, self.paired['classes'])

    def test_grid_has_one_shared_smoke_per_model_floor_and_exact_full_coverage(self):
        rows = self.tasks(); before = copy.deepcopy(rows)
        jobs = self.p.make_jobs(list(reversed(rows)), shard_count=8)
        self.assertEqual(rows, before)
        self.assertEqual(len({job['name'] for job in jobs}), len(jobs))
        self.assertTrue(all(job['smoke'] for job in jobs[:18]))
        self.assertTrue(all(not job['smoke'] for job in jobs[18:]))
        expected = {self.p.key(row) for row in rows}
        shared_smoke = ('bone', '0001_phase_3', 10003)
        for model in self.p.MODELS:
            for floor in self.p.FLOORS:
                selected = [j for j in jobs if j['model'] == model and j['floor'] == floor]
                smoke = [j for j in selected if j['smoke']]
                self.assertEqual(len(smoke), 1)
                self.assertEqual([self.p.key(r) for r in smoke[0]['tasks']], [shared_smoke])
                keys = [self.p.key(r) for j in selected for r in j['tasks']]
                self.assertEqual(set(keys), expected); self.assertEqual(len(keys), len(expected))
                full_cases = {}
                for job in selected:
                    self.assertEqual(job['task_count'], len(job['tasks']))
                    self.assertTrue(job['tasks'])
                    self.assertEqual((job['variant'], job['source'], job['frame_batch'], job['attempt']),
                                     ('baseline', 'bone', 16, 0))
                    if not job['smoke']:
                        for row in job['tasks']:
                            self.assertEqual(full_cases.setdefault(row['case_id'], job['name']), job['name'])
        jobs[0]['tasks'][0]['class_name'] = 'mutated only in output'
        self.assertEqual(rows, before)

    def test_smoke_without_label_three_falls_back_to_first_key(self):
        rows = [r for r in self.tasks() if r['local_class_id'] != 3]
        expected = min(self.p.key(r) for r in rows)
        for job in self.p.make_jobs(rows, shard_count=2):
            if job['smoke']:
                self.assertEqual(self.p.key(job['tasks'][0]), expected)

    def test_grid_rejects_empty_duplicate_nonbone_tasks_and_bad_shard_count(self):
        rows = self.tasks()
        for bad in ([], rows + rows[:1], [{**rows[0], 'source_name': 'totalseg'}]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.p.make_jobs(bad)
        for count in (0, -1, 1.5, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.p.make_jobs(rows, shard_count=count)

    def test_metric_validation_requires_exact_unique_keys_and_all_ten_finite_metrics(self):
        rows = self.metric_rows(); tasks = self.tasks()
        self.p.validate_rows(list(reversed(rows)), tasks)
        self.assertEqual(self.p.key({'dataset': 'bone', 'case_id': 'a', 'class_id': '10003'}), ('bone', 'a', 10003))
        bad_rows = [rows[:-1], rows + rows[:1], [{**rows[0], 'case_id': 'wrong'}] + rows[1:]]
        for name in METRICS:
            for value in (math.nan, math.inf, -math.inf, None, 'bad'):
                bad_rows.append([{**rows[0], name: value}] + rows[1:])
            bad = copy.deepcopy(rows); bad[0].pop(name); bad_rows.append(bad)
        for bad in bad_rows:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.p.validate_rows(bad, tasks)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.p.validate_rows(rows, tasks + tasks[:1])

    def test_native_fallback_point_six_bypasses_higher_floor_only_when_recorded(self):
        self.assertTrue(self.p.threshold_valid({'dynamic_threshold_source': 'fallback_missing_prompt_logits', 'threshold': .6}, .75))
        self.assertTrue(self.p.threshold_valid({'source': 'fallback_insufficient_separation', 'threshold': .6}, .7))
        for floor in self.p.FLOORS:
            self.assertTrue(self.p.threshold_valid({'source': 'adaptive', 'threshold': floor}, floor))
            self.assertTrue(self.p.threshold_valid({'source': 'adaptive', 'threshold': .75}, floor))
        for row in ({'source': 'fallback_missing_prompt_logits', 'threshold': .65},
                    {'source': 'fallback_unrecognized', 'threshold': .6},
                    {'source': 'fallback_unrecognized', 'threshold': .7},
                    {'source': 'adaptive', 'threshold': .6},
                    {'source': 'adaptive', 'threshold': .751}, {'source': 'adaptive', 'threshold': math.nan},
                    {'source': 'adaptive'}, {'threshold': .7}):
            with self.subTest(row=row):
                self.assertFalse(self.p.threshold_valid(row, .7))

    def test_summary_uses_task_means_paired_deltas_class_macro_and_fallbacks(self):
        rows = self.metric_rows()
        baseline = [{**r, **{m: r[m] - .1 for m in METRICS}} for r in reversed(rows)]
        sixty = [{**r, **{m: r[m] + .05 for m in METRICS}} for r in rows]
        rows[0]['dynamic_threshold_source'] = 'fallback_missing_prompt_logits'
        result = self.p.summarize(rows, baseline, sixty)
        self.assertEqual(result['task_count'], 6)
        for name in METRICS:
            self.assertAlmostEqual(result['mean'][name], 3.2 / 6)
            self.assertAlmostEqual(result['delta_from_baseline'][name], .1)
            self.assertAlmostEqual(result['delta_from_floor60'][name], -.05)
        self.assertEqual(len(result['per_case_class']), 6)
        self.assertAlmostEqual(result['class_macro_dice'], (.2 + (.4 + .8) / 2 + (.6 + .5) / 2 + .7) / 4)
        self.assertEqual(result['fallback_count'], 1)
        with self.assertRaises(ValueError):
            self.p.summarize(rows, baseline[:-1], sixty)
        bad = copy.deepcopy(sixty); bad[0]['mean_slice_nsd_3mm'] = math.nan
        with self.assertRaises(ValueError):
            self.p.summarize(rows, baseline, bad)

    def test_winner_is_one_global_volume_dice_choice_ignoring_axial_score(self):
        lower = self.metric_rows((0, 1, 0, 1, 0, 1))
        higher = self.metric_rows((.6,) * 6)
        for row in lower:
            row['mean_slice_dice'] = 1
        for row in higher:
            row['mean_slice_dice'] = 0
        result = self.p.winner({.35: lower, .65: higher})
        self.assertEqual(result['chosen_floor'], .65)
        self.assertAlmostEqual(result['task_mean_dice'], .6)
        self.assertIn('dice', result['rule'].lower())
        tied = self.p.winner({.7: higher, .4: copy.deepcopy(higher)})
        self.assertEqual(tied['chosen_floor'], .4)

    def test_winner_uses_stable_sum_and_never_rounds_near_ties(self):
        rows = self.metric_rows((1, 1e-16, 1e-16, 0, 0, 0))
        result = self.p.winner({.35: rows})
        self.assertEqual(result['task_mean_dice'], math.fsum(r['dice'] for r in rows) / 6)
        better = copy.deepcopy(rows); better[0]['dice'] = math.nextafter(1, math.inf)
        self.assertEqual(self.p.winner({.35: rows, .75: better})['chosen_floor'], .75)

    def test_winner_rejects_incomplete_duplicate_nonfinite_or_empty_candidates(self):
        rows = self.metric_rows()
        bad = copy.deepcopy(rows); bad[0]['recall'] = math.inf
        for candidates in ({}, {.35: []}, {.35: rows, .4: rows[:-1]},
                           {.35: rows, .4: rows + rows[:1]}, {.35: rows, .4: bad}):
            with self.subTest(candidates=candidates), self.assertRaises(ValueError):
                self.p.winner(candidates)


if __name__ == '__main__':
    unittest.main()
