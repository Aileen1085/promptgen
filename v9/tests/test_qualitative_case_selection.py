"""Synthetic contract tests; workspace CLI fixtures retained for diagnostics.

No automatic directory cleanup: project rules prohibit recursive deletion.
"""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


MODULE_PATH = Path(__file__).absolute().parents[1] / "tools" / "qualitative_case_selection.py"


def row(case="case-a", cls="liver", dice=0.95, source="AMOS", **extra):
    result = dict(source=source, case_id=case, class_id=cls,
                  dice=dice, metric_scope="full_volume")
    result.update(extra)
    return result


class SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if MODULE_PATH.exists():
            spec = importlib.util.spec_from_file_location("selection", str(MODULE_PATH))
            cls.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.module)
        else:
            cls.module = None

    def select(self, methods, **kwargs):
        self.assertIsNotNone(self.module, "selection tool must exist")
        return self.module.select_cases(methods, sources=["AMOS"], **kwargs)

    def fixture(self):
        return {"PromptGen": [row(), row("case-b", dice=.91)],
                "SAM2": [row(dice=.70), row("case-b", dice=.40)],
                "PAM": [row(dice=.90), row("case-b", dice=.60)]}

    def test_inclusive_gap_boundary_respects_float_roundoff(self):
        good = self.select({"PromptGen": [row(dice=.95)], "SAM2": [row(dice=.90)]})
        self.assertIsNotNone(good['sources']['AMOS']['selected'])
        below = self.select({"PromptGen": [row(dice=.949999)], "SAM2": [row(dice=.90)]})
        self.assertIsNone(below['sources']['AMOS']['selected'])
        tied = self.select({"PromptGen": [row(dice=.90)], "SAM2": [row(dice=.90)]}, min_gap=0)
        self.assertIsNone(tied['sources']['AMOS']['selected'])

    def test_ranks_against_strongest_comparator_and_records_rule(self):
        result = self.select(self.fixture())
        chosen = result["sources"]["AMOS"]["selected"]
        self.assertEqual(chosen["case_id"], "case-b")
        self.assertEqual(chosen["strongest_comparator"], "PAM")
        self.assertAlmostEqual(chosen["gap"], .31)
        self.assertTrue(chosen["all_method_win"])
        self.assertEqual(result["rule"]["score"], "promptgen_dice + gap")
        self.assertEqual(chosen["metrics"]["PromptGen"], .91)

    def test_no_fabricated_win_or_selection_when_criteria_unmet(self):
        data = {"PromptGen": [row(dice=.90)], "SAM2": [row(dice=.96)]}
        result = self.select(data)["sources"]["AMOS"]
        self.assertIsNone(result["selected"])
        self.assertEqual(result["status"], "criteria_unmet")
        self.assertFalse(result["candidates"][0]["all_method_win"])
        fallback = self.select(data, allow_best_available=True)["sources"]["AMOS"]
        self.assertEqual(fallback["status"], "best_available_criteria_unmet")
        self.assertFalse(fallback["selected"]["meets_criteria"])

    def test_exact_case_and_class_join_reports_missing_methods(self):
        data = {"PromptGen": [row(), row(cls="spleen")],
                "SAM2": [row(dice=.2), row("CASE-A", dice=.1)]}
        result = self.select(data)["sources"]["AMOS"]
        self.assertEqual(result["common_count"], 1)
        self.assertEqual(result["selected"]["class_id"], "liver")
        excluded = result["excluded_missing_methods"]
        self.assertEqual(len(excluded), 2)
        self.assertEqual(excluded[1]["missing_methods"], ["SAM2"])

    def test_rejects_alias_colliding_duplicates(self):
        data = self.fixture()
        data["SAM2"].append(row(source="amos", dice=.7))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.select(data)

    def test_rejects_invalid_metrics_scopes_sources_and_ids(self):
        for field, value in [("dice", float("nan")), ("dice", float("inf")),
                             ("dice", -.1), ("dice", 90), ("dice", True),
                             ("dice", ".9"), ("metric_scope", "roi"),
                             ("metric_scope", "slice"), ("source", "MSD"),
                             ("source", "TotalSeg"), ("case_id", ""),
                             ("case_id", " case-a"), ("class_id", 1)]:
            with self.subTest(field=field, value=value):
                data = self.fixture()
                data["PromptGen"][0][field] = value
                with self.assertRaises(ValueError):
                    self.select(data)
        data = self.fixture()
        del data["SAM2"][0]["metric_scope"]
        with self.assertRaises(ValueError):
            self.select(data)

    def test_explicit_source_aliases_keep_msd_tasks_separate(self):
        self.assertIsNotNone(self.module, "selection tool must exist")
        aliases = [("amos", "AMOS"), ("magic", "MAGIC"), ("kits23", "KiTS23"),
                   ("msd_task10", "MSD10"), ("Task10_Colon", "MSD10"),
                   ("msd_task07", "MSD07"), ("Task07_Pancreas", "MSD07")]
        for alias, expected in aliases:
            with self.subTest(alias=alias):
                self.assertEqual(self.module.normalize_source(alias), expected)
        with self.assertRaises(ValueError):
            self.module.normalize_source("Task7")

    def test_tie_break_and_output_are_independent_of_input_order(self):
        data = {"PromptGen": [row("b"), row("a")],
                "SAM2": [row("a", dice=.2), row("b", dice=.2)]}
        result = self.select(data)
        reordered = {key: list(reversed(value)) for key, value in reversed(list(data.items()))}
        self.assertEqual(result, self.select(reordered))
        self.assertEqual(result["sources"]["AMOS"]["selected"]["case_id"], "a")

    def test_requires_target_and_at_least_one_comparator(self):
        for data in [{"PromptGen": [row()]}, {"SAM2": [row()]}]:
            with self.assertRaises(ValueError):
                self.select(data)

    def test_absolute_quality_threshold_excludes_low_quality_large_gap(self):
        data = {"PromptGen": [row("bad", dice=.6), row("good", dice=.9)],
                "SAM2": [row("bad", dice=.0), row("good", dice=.7)]}
        chosen = self.select(data)["sources"]["AMOS"]["selected"]
        self.assertEqual(chosen["case_id"], "good")

    def test_absent_source_and_disjoint_keys_have_no_selected_case(self):
        data = {"PromptGen": [row()], "SAM2": [row("different")]}
        result = self.select(data)["sources"]["AMOS"]
        self.assertEqual(result["status"], "no_common_tasks")
        self.assertIsNone(result["selected"])
        empty = self.select({"PromptGen": [], "SAM2": []})["sources"]["AMOS"]
        self.assertEqual(empty["status"], "no_common_tasks")

    def test_configuration_validation(self):
        for kwargs in [dict(top_k=0), dict(top_k=True), dict(min_dice=-1),
                       dict(min_gap=float("nan")), dict(min_gap=-.1)]:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    self.select(self.fixture(), **kwargs)

    def test_five_source_selections_are_independent_and_top_k_does_not_drop_selection(self):
        self.assertIsNotNone(self.module, "selection tool must exist")
        data = {"PromptGen": [], "SAM2": []}
        for source in self.module.DEFAULT_SOURCES:
            data["PromptGen"].append(row(source=source))
            data["SAM2"].append(row(source=source, dice=.4))
        result = self.module.select_cases(data, top_k=1)
        self.assertEqual(set(result["sources"]), set(self.module.DEFAULT_SOURCES))
        for source, info in result["sources"].items():
            self.assertEqual(info["selected"]["source"], source)
            self.assertEqual(info["common_count"], 1)

    def test_cli_rejects_duplicate_json_method_keys(self):
        self.assertTrue(MODULE_PATH.exists(), "selection tool must exist")
        fixture_root = MODULE_PATH.parents[1] / "output"
        fixture_root.mkdir(exist_ok=True)
        temp = tempfile.mkdtemp(prefix="selection_duplicate_test_", dir=str(fixture_root))
        input_path = Path(temp) / "duplicate.json"
        output_path = Path(temp) / "selection.json"
        input_path.write_text('{"PromptGen": [], "SAM2": [], "SAM2": []}', encoding="utf-8")
        completed = subprocess.run(
            [sys.executable, str(MODULE_PATH), "--input", str(input_path),
             "--output", str(output_path)], capture_output=True, universal_newlines=True)
        self.assertEqual(completed.returncode, 2, completed.stderr)
        self.assertIn("duplicate JSON key", completed.stderr)
        self.assertFalse(output_path.exists())

    def test_cli_writes_deterministic_json_and_input_signature(self):
        self.assertTrue(MODULE_PATH.exists(), "selection tool must exist")
        fixture_root = MODULE_PATH.parents[1] / "output"
        fixture_root.mkdir(exist_ok=True)
        temp = tempfile.mkdtemp(prefix="selection_roundtrip_test_", dir=str(fixture_root))
        input_path = Path(temp) / "normalized.json"
        output_path = Path(temp) / "selection.json"
        input_path.write_text(json.dumps(self.fixture()), encoding="utf-8")
        command = [sys.executable, str(MODULE_PATH), "--input", str(input_path),
                   "--output", str(output_path), "--sources", "AMOS", "--top-k", "1"]
        completed = subprocess.run(command, capture_output=True, universal_newlines=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(len(result["input_sha256"]), 64)
        self.assertEqual(len(result["sources"]["AMOS"]["candidates"]), 1)
        self.assertEqual(result["sources"]["AMOS"]["selected"]["case_id"], "case-b")
        self.assertIn("post-hoc", result["interpretation"])
        original = output_path.read_bytes()
        repeated = subprocess.run(command, capture_output=True, universal_newlines=True)
        self.assertEqual(repeated.returncode, 2)
        self.assertEqual(output_path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
