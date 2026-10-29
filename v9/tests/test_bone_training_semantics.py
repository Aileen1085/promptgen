"""Pure Bone semantic alignment contract and fail-closed provenance tests."""
import copy
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest

from v9_2_extended_protocol import class_metadata, conditioning_text_from_metadata


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / 'tools' / 'bone_training_semantics.py'
IDENTITY = ('dataset', 'global_class_id', 'local_class_id', 'label_value', 'class_name')


def bone(local=55, name='rib_right_4'):
    return dict(dataset='bone', global_class_id=10000 + local,
                local_class_id=local, label_value=local, class_name=name,
                semantic_text='right rib 4', semantic_key='bone.old',
                parent_semantic_key=None, equivalence_group='bone.old',
                morphology_group='tubular', not_applicable=False,
                extra={'preserve': [1, 2]})


def training(gid=95, name='rib_right_4', **updates):
    result = dict(dataset='totalseg', global_class_id=gid, local_class_id=gid,
                  class_name=name, semantic_text=name,
                  conditioning='semantic_text_not_id_embedding',
                  not_applicable=True, **class_metadata('totalseg', name))
    result.update(updates)
    return result


class BoneTrainingSemanticsTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(MODULE.is_file(), 'pure semantic alignment module is missing')
        spec = importlib.util.spec_from_file_location('bone_semantics_tested', MODULE)
        self.p = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.p)

    def test_exact_rib_and_femur_reuse_raw_training_names_and_compact_metadata(self):
        bones = [bone(), bone(19, 'femur_left')]
        catalog = [training(), training(80, 'femur_left')]
        aligned, audit = self.p.align_catalog(bones, catalog)
        self.assertEqual(len(aligned), 2)
        for old, row, proof, source in zip(bones, aligned, audit, catalog):
            self.assertEqual({f: row[f] for f in IDENTITY}, {f: old[f] for f in IDENTITY})
            self.assertEqual(row['semantic_text'], source['class_name'] + '; compact target morphology')
            self.assertEqual(row['semantic_base_text'], source['semantic_text'])
            self.assertEqual(row['semantic_key'], source['semantic_key'])
            self.assertEqual(row['morphology_group'], 'compact')
            self.assertIs(row['not_applicable'], False, 'training eligibility must not hide Bone classes')
            self.assertEqual(proof['match_mode'], 'exact_totalseg')
            self.assertEqual(proof['training_row'], source)
            self.assertEqual(proof['original_row'], old)
            self.assertEqual(proof['original_semantic_text'], old['semantic_text'])
            self.assertEqual(proof['semantic_text'], row['semantic_text'])
            self.assertEqual(proof['annotation_scope'], 'whole_anatomy')
        self.assertIsNone(self.p.validate_alignment(bones, catalog, aligned, audit))

    def test_only_exact_totalseg_class_name_matches_are_eligible(self):
        source = training(dataset='amos', semantic_text='AMOS custom name')
        cases = [([source], 'rib_right_4'),
                 ([training(name='RIB_RIGHT_4')], 'rib_right_4'),
                 ([training()], 'rib_right_04')]
        for catalog, name in cases:
            with self.subTest(name=name, catalog=catalog):
                aligned, audit = self.p.align_catalog([bone(name=name)], catalog)
                self.assertEqual(audit[0]['match_mode'], 'new_anatomy')
                self.assertIsNone(audit[0]['training_row'])
                self.assertEqual(aligned[0]['semantic_base_text'], name)
                self.assertEqual(aligned[0]['semantic_key'], class_metadata('bone', name)['semantic_key'])

    def test_preserves_native_parent_formatting_and_copies_all_semantic_metadata(self):
        source = training(parent_semantic_key='organ.skeletal_system',
                          semantic_aliases={'raw': ['training-token']})
        old = bone()
        old['semantic_obsolete'] = 'custom old metadata'
        aligned, audit = self.p.align_catalog([old], [source])
        self.assertEqual(aligned[0]['semantic_text'], conditioning_text_from_metadata(source))
        self.assertIn(' || parent_tissue: normal skeletal system tissue || parent_scale: 0.25',
                      aligned[0]['semantic_text'])
        self.assertEqual(aligned[0]['semantic_aliases'], source['semantic_aliases'])
        self.assertNotIn('semantic_obsolete', aligned[0])
        self.assertEqual(audit[0]['semantic_metadata']['parent_semantic_key'], 'organ.skeletal_system')

    def test_both_explicit_chinese_body_suffixes_retain_matched_base_and_scope(self):
        for suffix in ('（仅椎体）', '(仅椎体)'):
            with self.subTest(suffix=suffix):
                old = bone(3, 'vertebrae_L1' + suffix)
                aligned, audit = self.p.align_catalog([old], [training(25, 'vertebrae_L1')])
                row = aligned[0]
                self.assertEqual(row['class_name'], old['class_name'])
                self.assertEqual(row['semantic_base_text'], 'vertebrae_L1 (vertebral body only)')
                self.assertEqual(row['semantic_text'],
                                 'vertebrae_L1 (vertebral body only); compact target morphology')
                self.assertEqual(audit[0]['annotation_scope'], 'vertebral_body_only')
                self.assertEqual(audit[0]['scope_suffix'], suffix)
                self.assertEqual(audit[0]['matching_class_name'], 'vertebrae_L1')
                self.p.validate_alignment([old], [training(25, 'vertebrae_L1')], aligned, audit)

    def test_unseen_tibia_and_foot_phalanges_use_raw_token_and_native_metadata(self):
        for name in ('tibia_left', 'fibula_left', 'patella_right', 'tarsal_left',
                     'metatarsal_right', 'phalanges_foot_left'):
            with self.subTest(name=name):
                old = bone(21, name)
                aligned, audit = self.p.align_catalog([old], [training()])
                expected = class_metadata('bone', name)
                self.assertEqual(aligned[0]['semantic_base_text'], name)
                self.assertEqual(aligned[0]['semantic_text'],
                                 conditioning_text_from_metadata(dict(class_name=name,
                                                                      semantic_text=name, **expected)))
                for key, value in expected.items():
                    self.assertEqual(aligned[0][key], value)
                self.assertEqual(audit[0]['match_mode'], 'new_anatomy')
                self.assertIsNone(audit[0]['training_row'])
                self.assertEqual(audit[0]['metadata_builder'], 'v9_2_extended_protocol.class_metadata')

    def test_unmatched_vertebra_preserves_scope_using_original_token(self):
        old = bone(1, 'vertebrae_C1（仅椎体）')
        aligned, audit = self.p.align_catalog([old], [training()])
        self.assertEqual(aligned[0]['semantic_base_text'], 'vertebrae_C1 (vertebral body only)')
        self.assertEqual(aligned[0]['semantic_key'], class_metadata('bone', 'vertebrae_C1')['semantic_key'])
        self.assertEqual(audit[0]['annotation_scope'], 'vertebral_body_only')

    def test_inputs_and_nested_outputs_are_independent_copies(self):
        bones, catalog = [bone()], [training(semantic_aliases={'raw': ['token']})]
        before = copy.deepcopy((bones, catalog))
        aligned, audit = self.p.align_catalog(bones, catalog)
        self.p.validate_alignment(bones, catalog, aligned, audit)
        self.assertEqual((bones, catalog), before)
        aligned[0]['extra']['preserve'].append(3)
        aligned[0]['semantic_aliases']['raw'].append('changed')
        audit[0]['original_row']['extra']['preserve'].append(4)
        audit[0]['training_row']['semantic_aliases']['raw'].append('audit changed')
        self.assertEqual((bones, catalog), before)
        self.assertEqual(audit[0]['semantic_metadata']['semantic_aliases']['raw'], ['token'])

    def test_rejects_ambiguous_training_class_names_even_when_metadata_agrees(self):
        with self.assertRaisesRegex(ValueError, 'ambiguous|duplicate'):
            self.p.align_catalog([bone()], [training(), training(96)])

    def test_rejects_duplicate_ids_in_either_catalog(self):
        duplicate_local = bone(56, 'femur_left')
        duplicate_local['local_class_id'] = duplicate_local['label_value'] = 55
        duplicate_global = bone(56, 'femur_left')
        duplicate_global['global_class_id'] = 10055
        for bones, catalog in (([bone(), bone()], [training()]),
                               ([bone(), duplicate_local], [training()]),
                               ([bone(), duplicate_global], [training()]),
                               ([bone()], [training(), training(name='femur_left')])):
            with self.subTest(bones=bones, catalog=catalog), self.assertRaisesRegex(ValueError, 'duplicate'):
                self.p.align_catalog(bones, catalog)

    def test_non_totalseg_binary_mask_classes_can_share_local_label_values(self):
        catalog = [training(),
                   training(133, 'hcc', dataset='magic', local_class_id=1),
                   training(134, 'liver_mts', dataset='magic', local_class_id=1)]
        try:
            aligned, audit = self.p.align_catalog([bone()], catalog)
        except ValueError as exc:
            self.fail(f'valid per-class binary-mask local label IDs were rejected: {exc}')
        self.assertEqual(aligned[0]['semantic_text'], 'rib_right_4; compact target morphology')
        self.assertEqual(audit[0]['training_row'], catalog[0])

    def test_rejects_unknown_or_malformed_scope_suffixes(self):
        for name in ('vertebrae_L1(whole)', 'vertebrae_L1（椎体）',
                     'vertebrae_L1 (vertebral body only)', 'vertebrae_L1（仅椎体)',
                     'vertebrae_L1（仅椎体）extra', 'vertebrae_L1[body only]',
                     'rib_right_4（仅椎体）'):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'scope|suffix'):
                self.p.align_catalog([bone(name=name)], [training()])

    def test_rejects_malformed_or_missing_training_semantic_metadata(self):
        mutations = [lambda x: x.pop('semantic_text'),
                     lambda x: x.update(semantic_text=' '),
                     lambda x: x.update(semantic_text=None),
                     lambda x: x.update(semantic_text={'bad': 'text'}),
                     lambda x: x.pop('semantic_key'),
                     lambda x: x.update(semantic_key=None),
                     lambda x: x.update(semantic_key=''),
                     lambda x: x.update(equivalence_group=[]),
                     lambda x: x.pop('parent_semantic_key'),
                     lambda x: x.update(parent_semantic_key=''),
                     lambda x: x.update(parent_semantic_key=3),
                     lambda x: x.pop('morphology_group'),
                     lambda x: x.update(morphology_group='long_bone')]
        for mutate in mutations:
            source = training()
            mutate(source)
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, 'semantic|morphology|metadata'):
                self.p.align_catalog([bone()], [source])

    def test_rejects_missing_malformed_or_inconsistent_bone_identity(self):
        mutations = [lambda x: x.pop('global_class_id'),
                     lambda x: x.pop('local_class_id'),
                     lambda x: x.pop('label_value'),
                     lambda x: x.update(global_class_id=True),
                     lambda x: x.update(local_class_id=0),
                     lambda x: x.update(label_value=56),
                     lambda x: x.update(dataset='totalseg'),
                     lambda x: x.update(class_name=''),
                     lambda x: x.update(class_name=' rib_right_4'),
                     lambda x: x.update(semantic_text=None)]
        for mutate in mutations:
            old = bone()
            mutate(old)
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.p.align_catalog([old], [training()])

    def test_validator_rejects_changed_text_scope_identity_metadata_and_provenance(self):
        bones, catalog = [bone(3, 'vertebrae_L1（仅椎体）')], [training(25, 'vertebrae_L1')]
        aligned, audit = self.p.align_catalog(bones, catalog)
        mutations = [lambda rows, proofs: rows[0].update(semantic_text='vertebrae_L1'),
                     lambda rows, proofs: rows[0].update(semantic_base_text='vertebrae_L1'),
                     lambda rows, proofs: rows[0].update(label_value=4),
                     lambda rows, proofs: rows[0].update(class_name='vertebrae_L1'),
                     lambda rows, proofs: rows[0].update(morphology_group='tubular'),
                     lambda rows, proofs: proofs[0].update(match_mode='new_anatomy'),
                     lambda rows, proofs: proofs[0].update(annotation_scope='whole_anatomy'),
                     lambda rows, proofs: proofs[0]['training_row'].update(semantic_text='paraphrase'),
                     lambda rows, proofs: proofs[0]['original_row'].update(label_value=4),
                     lambda rows, proofs: proofs[0].pop('semantic_metadata'),
                     lambda rows, proofs: proofs.clear(),
                     lambda rows, proofs: rows.append(copy.deepcopy(rows[0]))]
        for mutate in mutations:
            rows, proofs = copy.deepcopy((aligned, audit))
            mutate(rows, proofs)
            with self.subTest(rows=rows, proofs=proofs), self.assertRaisesRegex(ValueError, 'alignment|provenance'):
                self.p.validate_alignment(bones, catalog, rows, proofs)

    def test_validator_rejects_identity_type_changes_despite_numeric_equality(self):
        bones, catalog = [bone()], [training()]
        aligned, audit = self.p.align_catalog(bones, catalog)
        for target in ('aligned', 'audit'):
            rows, proofs = copy.deepcopy((aligned, audit))
            (rows if target == 'aligned' else proofs)[0]['label_value'] = 55.0
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, 'alignment|provenance'):
                self.p.validate_alignment(bones, catalog, rows, proofs)

    def test_empty_bone_and_non_row_catalogs_fail_with_value_error(self):
        for bones, catalog in (([], [training()]), ([None], [training()]),
                               ([bone()], [None]), ({'rows': [bone()]}, [training()])):
            with self.subTest(bones=bones, catalog=catalog), self.assertRaises(ValueError):
                self.p.align_catalog(bones, catalog)

    def test_import_and_alignment_need_no_torch_numpy_or_model_code(self):
        script = '''
import builtins, importlib.util, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'numpy', 'model', 'v10'):
        raise AssertionError('semantic alignment imported model dependencies: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
spec = importlib.util.spec_from_file_location('pure_bone_semantics', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
from v9_2_extended_protocol import class_metadata
row = dict(dataset='bone', global_class_id=10021, local_class_id=21,
           label_value=21, class_name='tibia_left', semantic_text='left tibia',
           **class_metadata('bone', 'tibia_left'))
aligned, audit = module.align_catalog([row], [])
module.validate_alignment([row], [], aligned, audit)
'''
        result = subprocess.run([sys.executable, '-X', 'utf8', '-c', script, str(MODULE)],
                                cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
