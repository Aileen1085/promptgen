import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train
from factory import load_v102_promptgen


class SegVolEntrypointTests(unittest.TestCase):
    def test_help_does_not_require_segvol_or_weights(self):
        with redirect_stdout(StringIO()):
            with self.assertRaises(SystemExit) as caught:
                train.parser().parse_args(["--help"])
        self.assertEqual(caught.exception.code, 0)

    def test_missing_segvol_inputs_fail_before_training(self):
        args = train.parser().parse_args(["train"])
        with self.assertRaisesRegex(ValueError, "segvol-source"):
            train._build_model(args)

    def test_missing_promptgen_checkpoint_is_explicit(self):
        with self.assertRaises(FileNotFoundError):
            load_v102_promptgen("not-a-real-v10.2-checkpoint.pth", "cpu")

    def test_plateau_and_early_stop_arguments_are_available(self):
        args = train.parser().parse_args(["train"])
        self.assertEqual(args.lr_plateau_patience, 2)
        self.assertEqual(args.early_stop_patience, 4)
        self.assertEqual(args.min_dice_improvement, 0.001)
        self.assertEqual(args.lr_factor, 0.5)

    def test_checkpoint_persists_training_control(self):
        class Stateful:
            def state_dict(self):
                return {}

        model = SimpleNamespace(
            promptgen=Stateful(), feature_bridge=Stateful(), prompt_adapter=Stateful()
        )
        protocol = SimpleNamespace(split_sha256="split", validation_sha256="validation")
        args = SimpleNamespace(segvol_checkpoint="segvol-weights")
        control = {"best_dice": 0.812719094235435, "bad_validations": 2}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pth"
            train._checkpoint(path, model, Stateful(), 45, protocol, args,
                              {"dice": 0.80}, training_control=control)
            saved = torch.load(path, map_location="cpu", weights_only=True)
        self.assertEqual(saved["training_control"], control)


if __name__ == "__main__":
    unittest.main()
