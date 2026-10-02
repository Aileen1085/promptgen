"""Safety and composition checks for the additive CT13 full-fine-tune path."""
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CT13FullTests(unittest.TestCase):
    def test_entry_installs_full_hooks_before_ct13_evaluation(self):
        source = (ROOT / "sam2_v9_2_ct13_full_entry.py").read_text()
        self.assertLess(
            source.index("import finetune_multisource_sam2_v9_2_full"),
            source.index("from sam2_v9_2_ct13_entry import main"),
        )
        self.assertIn("return ct13_main()", source)

    def test_composed_launcher_keeps_e360_protocol_and_metric_driven_stopping(self):
        from tools.launch_ct13_full_from_v92 import transform_launcher

        template = (ROOT / "run_train_sam2_v9_2_full.sh").read_text()
        result = transform_launcher(template, ROOT)
        self.assertIn("sam2_v9_2_ct13_full_entry.py", result)
        self.assertIn("configs/ct13_v10_2_split_20260928.json", result)
        self.assertIn(
            "configs/ct13_v10_2_validation_four_new_sources4_classweight_v3_20260928.json",
            result,
        )
        self.assertIn("--full-ft-lr-schedule plateau", result)
        self.assertIn("--early-stop-patience 4", result)
        self.assertIn("--early-stop-min-delta 0.0001", result)
        self.assertIn("V9_SHARED_CASE_CACHE_DIR", result)
        self.assertNotIn("v9_2_extended_split_4case_v2.json", result)
        self.assertNotIn("v9_2_extended_validation_4case_v2.json", result)

    def test_wrong_launcher_template_is_rejected(self):
        from tools.launch_ct13_full_from_v92 import transform_launcher

        with self.assertRaisesRegex(ValueError, "exactly once"):
            transform_launcher("echo unsafe", ROOT)

    def test_dry_run_can_render_without_local_dataset_manifests(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools/launch_ct13_full_from_v92.py"), "--dry-run"],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ct13_v10_2_validation_four_new_sources4_classweight_v3", result.stdout)


if __name__ == "__main__":
    unittest.main()
