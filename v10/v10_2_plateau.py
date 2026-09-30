"""Opt-in validation-driven LR decay and early stopping for CT13 continuation."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite


@dataclass
class ValidationPlateau:
    best: float
    min_delta: float
    lr_patience: int
    stop_patience: int
    factor: float
    min_lr_ratio: float
    bad_validations: int = 0
    lr_bad_validations: int = 0
    initial_rates: list[float] = field(default_factory=list)

    def __post_init__(self):
        if not 0 < self.factor < 1:
            raise ValueError("LR factor must be in (0, 1)")
        if not 0 < self.min_lr_ratio <= 1:
            raise ValueError("minimum LR ratio must be in (0, 1]")
        if self.lr_patience < 1 or self.stop_patience < 1:
            raise ValueError("patience values must be positive")
        if self.min_delta < 0:
            raise ValueError("minimum improvement must be nonnegative")

    def observe(self, score, groups):
        score = float(score)
        if not isfinite(score):
            raise ValueError("validation score must be finite")
        if not self.initial_rates:
            self.initial_rates = [float(g.get("schedule_base_lr", g["lr"])) for g in groups]
        if len(groups) != len(self.initial_rates):
            raise ValueError("optimizer parameter group count changed")
        if score > self.best + self.min_delta:
            self.best = score
            self.bad_validations = 0
            self.lr_bad_validations = 0
            return False, False
        self.bad_validations += 1
        self.lr_bad_validations += 1
        reduced = self.lr_bad_validations >= self.lr_patience
        if reduced:
            self.lr_bad_validations = 0
            for group, initial in zip(groups, self.initial_rates):
                old = float(group.get("schedule_base_lr", group["lr"]))
                new = max(initial * self.min_lr_ratio, old * self.factor)
                group["schedule_base_lr"] = new
                if old > 0:
                    group["lr"] = new
        return reduced, self.bad_validations >= self.stop_patience

    def state_dict(self):
        return {
            "best": self.best, "min_delta": self.min_delta,
            "lr_patience": self.lr_patience, "stop_patience": self.stop_patience,
            "factor": self.factor, "min_lr_ratio": self.min_lr_ratio,
            "bad_validations": self.bad_validations,
            "lr_bad_validations": self.lr_bad_validations,
            "initial_rates": list(self.initial_rates),
        }

    @classmethod
    def from_state_dict(cls, state):
        return cls(**state)
