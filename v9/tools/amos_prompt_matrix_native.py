"""Native SAM2, MedSAM2 and volumetric point baselines for the AMOS prompt matrix."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
POINT_SPACING_MM = 3.
SAM_METHODS = ("sam2", "medsam2")
VOLUME_METHODS = ("segvol", "vista3d", "sammed3d")
SAM_ARMS = {
    "fg1_bg2": (1, 2),
    "fg2_bg2": (2, 2),
    "fg4_bg2": (4, 2),
    "fg2_bg1": (2, 1),
    "fg2_bg4": (2, 4),
}


def _pilot(method):
    if method not in VOLUME_METHODS:
        raise ValueError(f"Unknown native point baseline {method!r}")
    return importlib.import_module(f"tools.native_{method}_pilot")


def arms_for(method):
    if method in SAM_METHODS:
        return tuple(SAM_ARMS)
    return tuple(_pilot(method).ARMS)


class NativePredictor:
    """Load one native model and return full-grid boolean masks for in-memory scoring."""

    def __init__(self, method):
        self.method = method
        if method in SAM_METHODS:
            from tools.fumpe_six_baselines import load_model

            self.model, self.adapter, self.assets = load_model(method)
            self.arms = SAM_ARMS
        else:
            self.adapter = _pilot(method)
            self.model = self.adapter.load_model()
            self.assets = {self.adapter.WEIGHT.relative_to(ROOT).as_posix(): self.adapter.WEIGHT_SHA}
            self.arms = self.adapter.ARMS

    def evaluate(self, task, prompts, *, arms=None):
        foreground = prompts.foreground[0] | prompts.foreground[1]
        background = np.argwhere(prompts.background[0] | prompts.background[1])
        selected_arms = self.arms if arms is None else tuple(arms)
        image = prompts.ct
        if self.method in ("segvol", "vista3d"):
            image = self.adapter.normalize(image)
        for arm in selected_arms:
            if self.method in SAM_METHODS:
                prediction, audit = self._predict_sam(image, foreground, background, prompts.spacing, arm)
            else:
                prediction, audit = self._predict_volume(image, foreground, background, prompts, arm)
            if prediction.shape != prompts.ct.shape or prediction.dtype != np.dtype(bool):
                raise ValueError(f"{self.method}/{arm} returned an invalid full-volume mask")
            yield arm, prediction, dict(audit, threshold_probability=.5, semantic_input=False,
                                        gt_used_for_roi=False, assets=self.assets)
            del prediction

    def _predict_sam(self, ct, foreground, background, spacing, arm):
        import torch

        fg_cap, bg_cap = SAM_ARMS[arm]
        args = SimpleNamespace(max_points=fg_cap, max_background_points=bg_cap,
                               point_spacing_mm=POINT_SPACING_MM, dense_prior="none",
                               sigma_mm=2., amplitude=4.)
        adapter = self.adapter
        initial = adapter.prompt_bounds_3d(foreground, background, spacing)
        bounds = initial
        stages = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(8):
                depth, (h0, h1), (w0, w1) = bounds
                offset = np.array([0, h0, w0])
                roi, stage = adapter.predict_roi(
                    self.model, ct[:, h0:h1, w0:w1], foreground[:, h0:h1, w0:w1],
                    background - offset, spacing, depth, args,
                )
                stages.append(dict(stage, bounds_dhw=bounds))
                expanded = adapter.expand_bounds_3d(roi, foreground, bounds, initial)
                if expanded == bounds:
                    break
                bounds = expanded
            else:
                raise RuntimeError("Native SAM AutoExpand did not converge")
        prediction = np.zeros(ct.shape, dtype=bool)
        prediction[tuple(slice(start, end) for start, end in bounds)] = roi
        return prediction, dict(stages=stages, foreground_cap_per_axial=fg_cap,
                                background_cap_per_axial=bg_cap, minimum_spacing_mm=POINT_SPACING_MM,
                                dense_prior="none", point_coordinates="DHW ROI offsets in bounds_dhw")

    def _predict_volume(self, image, foreground, background, prompts, arm):
        cap, sampler, *_ = self.arms[arm]
        positive = prompts.points("fg", cap, sampler, POINT_SPACING_MM)
        negative = prompts.points("bg", cap, sampler, POINT_SPACING_MM)
        if not sum(map(len, positive)) or not sum(map(len, negative)):
            raise ValueError(f"{self.method}/{arm} requires both prompt roles")
        points = np.concatenate((*positive, *negative)).astype(float)
        roles = np.repeat(np.array([1, 1, 0, 0], dtype=np.int64), [len(p) for p in (*positive, *negative)])
        original = self.adapter.scribble_points
        self.adapter.scribble_points = lambda *_args, **_kwargs: (points, roles)
        try:
            prediction, stages = self.adapter.predict(
                self.model, image, foreground, background, prompts.spacing, prompts.planes, arm,
            )
        finally:
            self.adapter.scribble_points = original
        return prediction, dict(stages=stages, cap_per_plane_per_role=cap, sampler=sampler,
                                minimum_spacing_mm=POINT_SPACING_MM,
                                foreground_points_dhw=[p.tolist() for p in positive],
                                background_points_dhw=[p.tolist() for p in negative])
