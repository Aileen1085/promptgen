import json
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ct_thirteen_source_protocol import SOURCE_ORDER


class FullEntryTests(unittest.TestCase):
    def require_entry(self):
        self.assertIsNotNone(
            importlib.util.find_spec("ct13_v10_2_full_entry"),
            "CT13 full-tuning entrypoint is missing",
        )

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "split.json"
        self.ratios = {source: 1 / len(SOURCE_ORDER) for source in SOURCE_ORDER}
        self.path.write_text(json.dumps({
            "protocol_revision": "ct13_v10_2_20260928",
            "source_order": list(SOURCE_ORDER),
            "sampling_ratios": self.ratios,
        }), encoding="utf-8")
        self.joint = SimpleNamespace()
        self.data = SimpleNamespace()
        self.protocol = SimpleNamespace()
        self.native_calls = []
        self.full = SimpleNamespace(_worker=lambda *args: self.native_calls.append(args))
        self.modules = {
            "finetune_totalseg_amos_magic_v10_2_joint": self.joint,
            "v10_2_data": self.data,
            "v9_2_extended_protocol": self.protocol,
            "finetune_multisource_sam2_v10_2_encoder_decoder": self.full,
        }

    def importer(self, name):
        return self.modules[name]

    def test_installs_thirteen_source_ratios_in_joint_and_data(self):
        self.require_entry()
        from ct13_v10_2_full_entry import install_protocol

        install_protocol(self.path, importer=self.importer)

        self.assertEqual(self.joint.EIGHT_SOURCE_ORDER, SOURCE_ORDER)
        self.assertEqual(self.data.EIGHT_SOURCE_ORDER, SOURCE_ORDER)
        self.assertEqual(self.protocol.SOURCE_ORDER, SOURCE_ORDER)
        self.assertEqual(self.joint.extended_source_ratios(None), self.ratios)

    def test_rejects_wrong_manifest_revision(self):
        self.require_entry()
        from ct13_v10_2_full_entry import install_protocol

        payload = json.loads(self.path.read_text(encoding="utf-8"))
        payload["protocol_revision"] = "ct13_v9_2_20260928"
        self.path.write_text(json.dumps(payload), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "revision"):
            install_protocol(self.path, importer=self.importer)

    def test_worker_reinstalls_protocol_before_native_full_tuning_worker(self):
        self.require_entry()
        from ct13_v10_2_full_entry import run_worker

        def native(*args):
            self.assertEqual(self.joint.EIGHT_SOURCE_ORDER, SOURCE_ORDER)
            self.assertEqual(self.protocol.SOURCE_ORDER, SOURCE_ORDER)
            self.native_calls.append(args)

        self.full._worker = native
        args = SimpleNamespace(multidataset_split_json=str(self.path))
        run_worker(0, 2, 2345, "stamp", args, importer=self.importer)

        self.assertEqual(len(self.native_calls), 1)
        self.assertEqual(self.native_calls[0], (0, 2, 2345, "stamp", args))


if __name__ == "__main__":
    unittest.main()
