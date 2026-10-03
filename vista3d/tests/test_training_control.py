import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training_control import ValidationController


class ValidationControllerTests(unittest.TestCase):
    def make_controller(self):
        return ValidationController(
            best_dice=0.812719094235435,
            significant_best_dice=0.812719094235435,
            min_delta=0.001,
            lr_patience=2,
            stop_patience=4,
            factor=0.5,
            min_lrs=(2e-7, 5e-7),
        )

    def test_plateau_reduces_both_lrs_then_stops_after_four_checks(self):
        control = self.make_controller()
        lrs = [2e-5, 5e-5]
        first = control.observe(0.8120, lrs)
        self.assertEqual(first.lrs, (2e-5, 5e-5))
        self.assertFalse(first.should_stop)
        second = control.observe(0.8110, first.lrs)
        self.assertEqual(second.lrs, (1e-5, 2.5e-5))
        self.assertTrue(second.reduced_lrs)
        third = control.observe(0.8100, second.lrs)
        self.assertFalse(third.should_stop)
        fourth = control.observe(0.8090, third.lrs)
        self.assertTrue(fourth.should_stop)
        self.assertEqual(fourth.lrs, second.lrs)

    def test_small_absolute_best_is_saved_but_does_not_reset_patience(self):
        control = self.make_controller()
        result = control.observe(0.8130, [2e-5, 5e-5])
        self.assertTrue(result.new_best)
        self.assertEqual(control.best_dice, 0.8130)
        self.assertEqual(control.bad_validations, 1)

    def test_meaningful_improvement_resets_patience_and_restores_state(self):
        control = self.make_controller()
        control.observe(0.8100, [2e-5, 5e-5])
        result = control.observe(0.8140, [2e-5, 5e-5])
        self.assertTrue(result.new_best)
        self.assertEqual(control.bad_validations, 0)
        self.assertEqual(control.bad_since_decay, 0)
        restored = ValidationController.from_state_dict(control.state_dict())
        self.assertEqual(restored.state_dict(), control.state_dict())

    def test_lr_floor_is_respected(self):
        control = self.make_controller()
        control.observe(0.81, [3e-7, 6e-7])
        result = control.observe(0.80, [3e-7, 6e-7])
        self.assertEqual(result.lrs, (2e-7, 5e-7))

    def test_old_epoch40_checkpoint_seeds_best_without_resetting_it(self):
        control = ValidationController.from_checkpoint(
            {"epoch": 40, "metrics": {"dice": 0.812719094235435}},
            min_delta=0.001,
            lr_patience=2,
            stop_patience=4,
            factor=0.5,
            min_lrs=(2e-7, 5e-7),
        )
        self.assertEqual(control.best_dice, 0.812719094235435)
        self.assertFalse(control.observe(0.81, [2e-5, 5e-5]).new_best)

    def test_new_checkpoint_restores_patience(self):
        control = self.make_controller()
        control.observe(0.81, [2e-5, 5e-5])
        resumed = ValidationController.from_checkpoint(
            {"epoch": 45, "metrics": {"dice": 0.81}, "training_control": control.state_dict()},
            min_delta=0.001,
            lr_patience=2,
            stop_patience=4,
            factor=0.5,
            min_lrs=(2e-7, 5e-7),
        )
        self.assertEqual(resumed.bad_validations, 1)
        self.assertEqual(resumed.best_dice, 0.812719094235435)

    def test_zero_delta_equal_dice_counts_as_no_improvement(self):
        control = ValidationController(0.8, 0.8, min_delta=0)
        result = control.observe(0.8, [2e-5, 5e-5])
        self.assertFalse(result.new_best)
        self.assertEqual(control.bad_validations, 1)

    def test_zero_delta_tiny_new_best_resets_both_counters(self):
        control = ValidationController(0.8, 0.8, min_delta=0)
        control.observe(0.79, [2e-5, 5e-5])
        result = control.observe(0.80000000001, [2e-5, 5e-5])
        self.assertTrue(result.new_best)
        self.assertEqual(control.bad_validations, 0)
        self.assertEqual(control.bad_since_decay, 0)

    def test_zero_delta_four_equal_checks_reduce_then_stop(self):
        control = ValidationController(0.8, 0.8, min_delta=0)
        lrs = [2e-5, 5e-5]
        for number in range(1, 5):
            result = control.observe(0.8, lrs)
            lrs = result.lrs
            self.assertEqual(result.should_stop, number == 4)
        self.assertEqual(lrs, (1e-5, 2.5e-5))

    def test_explicit_policy_reset_retains_best_not_last_dice(self):
        control = self.make_controller()
        control.observe(0.81, [2e-5, 5e-5])
        resumed = ValidationController.from_checkpoint(
            {"metrics": {"dice": 0.81}, "training_control": control.state_dict()},
            reset_policy=True, min_delta=0,
        )
        self.assertEqual(resumed.best_dice, control.best_dice)
        self.assertEqual(resumed.significant_best_dice, control.best_dice)
        self.assertEqual(resumed.bad_validations, 0)
        self.assertEqual(resumed.bad_since_decay, 0)
        self.assertEqual(resumed.min_delta, 0)
        self.assertEqual(resumed.observe(control.best_dice, [2e-5, 5e-5]).new_best, False)
        self.assertEqual(resumed.bad_validations, 1)

    def test_policy_change_without_explicit_reset_is_rejected(self):
        control = self.make_controller()
        with self.assertRaisesRegex(ValueError, "setting differs"):
            ValidationController.from_checkpoint(
                {"training_control": control.state_dict()}, min_delta=0,
            )


if __name__ == "__main__":
    unittest.main()
