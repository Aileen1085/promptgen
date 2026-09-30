"""Opt-in, validation-driven LR decay for the v9.2 full-fine-tune entry.

The shared training loop calls ``scheduler.step()`` after every epoch.  This
scheduler deliberately changes rates only when ``observe`` receives the fixed
formal validation score, so an unvalidated training loss cannot drive LR.
"""

from __future__ import annotations

from math import isfinite


class ValidationPlateauScheduler:
    def __init__(self, optimizer, *, eta_min, factor=0.5, threshold=1e-4):
        if not 0.0 < factor < 1.0:
            raise ValueError("plateau factor must be in (0, 1)")
        if eta_min <= 0 or threshold < 0:
            raise ValueError("plateau minimum LR must be positive and threshold nonnegative")
        self.optimizer = optimizer
        self.eta_min = float(eta_min)
        self.factor = float(factor)
        self.threshold = float(threshold)
        self.best = float("-inf")
        self.num_bad_epochs = 0
        self.transferred_from_cosine = False

    def step(self):
        """Keep LR fixed between formal validations."""
        return None

    def set_baseline(self, score):
        score = float(score)
        if not isfinite(score):
            raise ValueError("validation baseline must be finite")
        self.best = score
        self.num_bad_epochs = 0

    def observe(self, score):
        """Return whether this validation reduced at least one LR group."""
        score = float(score)
        if not isfinite(score):
            raise ValueError("validation score must be finite")
        if score > self.best + self.threshold:
            self.best = score
            self.num_bad_epochs = 0
            return False
        self.num_bad_epochs += 1
        old_rates = self.get_last_lr()
        for group in self.optimizer.param_groups:
            old = float(group["lr"])
            group["lr"] = max(self.eta_min, old * self.factor)
        self.num_bad_epochs = 0
        return any(after < before for before, after in zip(old_rates, self.get_last_lr()))

    def get_last_lr(self):
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def state_dict(self):
        return {
            "kind": "validation_plateau_v1",
            "eta_min": self.eta_min,
            "factor": self.factor,
            "threshold": self.threshold,
            "best": self.best,
            "num_bad_epochs": self.num_bad_epochs,
        }

    def load_state_dict(self, state):
        if state.get("kind") != "validation_plateau_v1":
            if "T_max" not in state:
                raise ValueError("unrecognized LR scheduler checkpoint")
            self.transferred_from_cosine = True
            return
        for name in ("eta_min", "factor", "threshold", "best"):
            setattr(self, name, float(state[name]))
        self.num_bad_epochs = int(state["num_bad_epochs"])
        self.transferred_from_cosine = False
