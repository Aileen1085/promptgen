"""Fresh-process regressions for the additive CT13 import-order repair."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("totalseg", "amos", "magic", "msd_task07", "msd_task10", "msd_task08", "parse2022", "topcow2024_cta",
           "lndb", "msd_task06", "covid19_20", "kits23", "lnq2023_lite")


def split_fixture():
    return dict(source_order=list(SOURCES), sampling_ratios={name:1/13 for name in SOURCES})

try:
    from tools import amos_prompt_ablation_restore_runtime as restored
except ImportError:
    restored = None


def fresh_process(source):
    result = subprocess.run([sys.executable, "-c", source, str(ROOT), json.dumps(split_fixture())], cwd=ROOT,
                            text=True, capture_output=True, timeout=30)
    if result.returncode:
        raise AssertionError(result.stdout + result.stderr)
    return json.loads(result.stdout.strip())


class ImportOrderTests(unittest.TestCase):
    def test_actual_native_order_reproduces_eighteen_source_contamination(self):
        proof = fresh_process('''
import json, sys
from pathlib import Path
root=Path(sys.argv[1]);sys.path.insert(0,str(root))
split=json.loads(sys.argv[2])
import v9_2_extended_protocol as protocol
before=list(protocol.SOURCE_ORDER)
protocol.SOURCE_ORDER=tuple(split['source_order'])
import ct13_training_entry as entry
try:
    entry._ratios(split)
except ValueError as error:
    print(json.dumps(dict(original=len(before), manifest=len(split['source_order']),
                          expanded=len(entry.SOURCE_ORDER), error=str(error), torch_imported='torch' in sys.modules)))
else:
    raise AssertionError('the old import ordering no longer reproduces the native failure')
''')
        self.assertEqual((proof["original"], proof["manifest"], proof["expanded"]), (8,13,18))
        self.assertIn("thirteen-source manifest", proof["error"])
        self.assertFalse(proof["torch_imported"])

    def test_wrapper_preloads_real_expansion_before_native_mutation(self):
        self.assertIsNotNone(restored, "add the isolated CT13 import-order wrapper")
        proof = fresh_process('''
import json, sys, tempfile
from pathlib import Path
from types import SimpleNamespace
root=Path(sys.argv[1]);sys.path.insert(0,str(root))
from tools import amos_prompt_ablation_runtime as native
from tools import amos_prompt_ablation_restore_runtime as restored
split=json.loads(sys.argv[2])
calls=[]
def initialize(job, paths):
    # These are the original runtime's dependency-changing lines, in its exact
    # order. The real CT13 installer and native _ratios are deliberately used.
    import v9_2_extended_protocol as protocol
    protocol.SOURCE_ORDER=tuple(split['source_order'])
    from ct13_training_entry import install_v10_protocol
    entry,data=SimpleNamespace(),SimpleNamespace()
    install_v10_protocol(entry,data,split)
    calls.append((entry.EIGHT_SOURCE_ORDER,data.EIGHT_SOURCE_ORDER))
    return SimpleNamespace(input_audit={})
native.initialize=initialize
with tempfile.NamedTemporaryFile(mode='w',suffix='.json',encoding='utf-8',delete=False) as handle:
    json.dump(split,handle)
    split_path=Path(handle.name)
try:
    session=restored.initialize(dict(case_id='amos_0001'),dict(root=str(root),base_split_json=str(split_path)))
finally:
    split_path.unlink()  # One explicit metadata file, never a recursive cleanup.
import ct13_training_entry as entry
import ct_thirteen_source_protocol as expansion
print(json.dumps(dict(calls=len(calls),source_count=len(entry.SOURCE_ORDER),
    original_count=len(expansion.OLD_SOURCE_ORDER),actual=list(calls[0][0]),
    audit=session.input_audit['ct13_import_order_restore'],torch_imported='torch' in sys.modules)))
''')
        self.assertEqual((proof["calls"], proof["source_count"], proof["original_count"]), (1,13,8))
        self.assertEqual(tuple(proof["actual"]), restored.EXPECTED_SOURCE_ORDER)
        self.assertTrue(proof["audit"]["native_ratios_passed"])
        self.assertTrue(proof["audit"]["preloaded_before_native_initialize"])
        self.assertFalse(proof["torch_imported"])

    def test_contaminated_process_refused_without_resetting_native_constants(self):
        self.assertIsNotNone(restored, "refuse an already contaminated process")
        proof = fresh_process('''
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);sys.path.insert(0,str(root))
split=json.loads(sys.argv[2])
import v9_2_extended_protocol as old
old.SOURCE_ORDER=tuple(split['source_order'])
import ct13_training_entry as entry
from tools import amos_prompt_ablation_restore_runtime as restored
before=entry.SOURCE_ORDER
try:
    restored._preload_ct13(split)
except ValueError as error:
    print(json.dumps(dict(error=str(error),count=len(entry.SOURCE_ORDER),unchanged=entry.SOURCE_ORDER is before)))
else:
    raise AssertionError('contaminated CT13 expansion must be refused')
''')
        self.assertIn("CT13", proof["error"])
        self.assertEqual(proof["count"],18)
        self.assertTrue(proof["unchanged"])


class SignedDelegationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(restored, "add the signed runtime delegation wrapper")

    def test_signed_native_source_unchanged_and_direct_exports_preserve_globals(self):
        from tools import amos_prompt_ablation_runtime as native
        self.assertEqual(hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(), restored.NATIVE_RUNTIME_SHA256)
        for name in ("Session", "evaluate", "validate_record", "validate_control"):
            self.assertIs(getattr(restored,name),getattr(native,name))
        self.assertIs(restored.evaluate.__globals__, native.__dict__)

    def test_source_mismatch_refused_before_delegation(self):
        from tools import amos_prompt_ablation_runtime as native
        with patch.object(restored,"NATIVE_RUNTIME_SHA256","0"*64), patch.object(native,"initialize") as initialize:
            with self.assertRaisesRegex(ValueError,"runtime SHA"):
                restored.initialize({},dict(root=str(ROOT)))
        initialize.assert_not_called()

    def test_native_ratios_validation_is_not_bypassed(self):
        split=split_fixture()
        restored._preload_ct13(split)
        bad=dict(split,sampling_ratios={})
        with self.assertRaisesRegex(ValueError,"sampling ratios"):
            restored._preload_ct13(bad)
        bad=dict(split,source_order=list(reversed(split['source_order'])))
        with self.assertRaisesRegex(ValueError,"source order"):
            restored._preload_ct13(bad)


if __name__ == "__main__":
    unittest.main()
