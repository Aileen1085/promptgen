"""Validation-driven LR reductions and early stopping for SegVol-3D."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ValidationDecision:
    lrs: tuple
    new_best: bool
    reduced_lrs: bool
    should_stop: bool


@dataclass
class ValidationController:
    best_dice: float
    significant_best_dice: float
    min_delta: float = 0.001
    lr_patience: int = 2
    stop_patience: int = 4
    factor: float = 0.5
    min_lrs: tuple = (2e-7, 5e-7)
    bad_validations: int = 0
    bad_since_decay: int = 0
    lr_reductions: int = 0

    def __post_init__(self):
        if not math.isfinite(self.best_dice) or not math.isfinite(self.significant_best_dice):
            raise ValueError("best Dice values must be finite")
        if self.min_delta < 0 or not math.isfinite(self.min_delta):
            raise ValueError("min_delta must be finite and nonnegative")
        if self.lr_patience < 1 or self.stop_patience < 1:
            raise ValueError("validation patience must be positive")
        if not 0 < self.factor < 1:
            raise ValueError("LR factor must be in (0,1)")
        if not self.min_lrs or any(not math.isfinite(lr) or lr < 0 for lr in self.min_lrs):
            raise ValueError("minimum LRs must be finite and nonnegative")

    def observe(self, dice, lrs):
        if not math.isfinite(dice) or not 0 <= dice <= 1:
            raise ValueError("validation Dice must be finite and in [0,1]")
        lrs = tuple(float(lr) for lr in lrs)
        if len(lrs) != len(self.min_lrs):
            raise ValueError("one minimum LR is required per optimizer group")
        if any(not math.isfinite(lr) or lr < 0 for lr in lrs):
            raise ValueError("optimizer LRs must be finite and nonnegative")

        new_best = dice > self.best_dice
        if new_best:
            self.best_dice = dice
        meaningful = dice >= self.significant_best_dice + self.min_delta
        if meaningful:
            self.significant_best_dice = dice
            self.bad_validations = 0
            self.bad_since_decay = 0
        else:
            self.bad_validations += 1
            self.bad_since_decay += 1

        should_stop = self.bad_validations >= self.stop_patience
        reduced_lrs = False
        if not should_stop and self.bad_since_decay >= self.lr_patience:
            lower = tuple(max(floor, lr * self.factor) for lr, floor in zip(lrs, self.min_lrs))
            reduced_lrs = lower != lrs
            lrs = lower
            self.bad_since_decay = 0
            if reduced_lrs:
                self.lr_reductions += 1
        return ValidationDecision(lrs, new_best, reduced_lrs, should_stop)

    def state_dict(self):
        return {
            "best_dice": self.best_dice,
            "significant_best_dice": self.significant_best_dice,
            "min_delta": self.min_delta,
            "lr_patience": self.lr_patience,
            "stop_patience": self.stop_patience,
            "factor": self.factor,
            "min_lrs": tuple(self.min_lrs),
            "bad_validations": self.bad_validations,
            "bad_since_decay": self.bad_since_decay,
            "lr_reductions": self.lr_reductions,
        }

    @classmethod
    def from_state_dict(cls, state):
        return cls(**state)

    @classmethod
    def from_checkpoint(cls, checkpoint, **settings):
        if checkpoint.get("training_control") is not None:
            controller = cls.from_state_dict(checkpoint["training_control"])
            for name, value in settings.items():
                if getattr(controller, name) != value:
                    raise ValueError("resumed training-control setting differs: " + name)
            return controller
        metrics = checkpoint.get("metrics")
        if not metrics or "dice" not in metrics:
            raise ValueError("old checkpoint has no validated Dice baseline")
        return cls(
            best_dice=float(metrics["dice"]),
            significant_best_dice=float(metrics["dice"]),
            **settings
        )
