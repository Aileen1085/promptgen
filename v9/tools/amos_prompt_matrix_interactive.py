"""Native nnInteractive and PAM adapters for frozen AMOS two-plane prompts."""

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PLANE_NAMES = ('coronal', 'sagittal')
NN_ARMS = (('bg_point2', 2), ('bg_point1', 1), ('bg_point4', 4), ('bg_scribble', None))


def _load_nninteractive(gpu_index):
    import torch
    from tools.four_source_eval import nn_modules

    argv = sys.argv
    try:
        _, evaluator = nn_modules(gpu_index)
    finally:
        sys.argv = argv
    evaluator.patch_nninteractive_trainer_for_current_nnunet()
    session = evaluator.nnInteractiveInferenceSession(
        device=torch.device('cuda:0'), use_torch_compile=False, verbose=False,
        torch_n_threads=2, do_autozoom=True, use_pinned_memory=True,
    )
    session.initialize_from_trained_model_folder(
        str(ROOT.parent / 'nninteractive_test_amos/checkpoints/nnInteractive_v1.0'),
    )
    # The native session applies its calibrated CUDA autocast inside _predict.
    return session


class InteractivePredictor:
    """Load one native method in its process with the assigned GPU visible as cuda:0."""

    def __init__(self, method, gpu_index):
        self.method = method
        self.image = None
        if method == 'nninteractive':
            self.session = _load_nninteractive(gpu_index)
        elif method == 'pam':
            from tools import pam_soft_fusion

            self.fusion = pam_soft_fusion
            self.models = pam_soft_fusion.SoftModels()
        else:
            raise ValueError(f'Unknown AMOS interactive method {method!r}')

    def evaluate(self, task, prompts):
        if self.method == 'pam':
            yield from self._evaluate_pam(prompts)
            return

        reused = prompts.ct is self.image
        if not reused:
            self.session.set_image(np.asarray(prompts.ct, dtype=np.float32)[None])
            self.image = prompts.ct
        for arm, point_cap in NN_ARMS:
            prediction, audit = self._predict_nninteractive(prompts, point_cap)
            yield arm, prediction, {**audit, 'case_image_reused': reused}
            reused = True

    def _predict_nninteractive(self, prompts, point_cap):
        import torch

        session = self.session
        buffer = torch.zeros(prompts.ct.shape, dtype=torch.uint8, device='cpu')
        session.set_target_buffer(buffer)
        events = []
        point_counts = [0, 0]
        triggers = 0
        native_predict = session._predict
        had_instance_predict = '_predict' in vars(session)

        def count_predict(*args, **kwargs):
            nonlocal triggers
            last = events[-1] if events else {}
            if triggers or not last.get('run_prediction'):
                raise RuntimeError('nnInteractive requires one native prediction after both foreground scribbles')
            triggers += 1
            return native_predict(*args, **kwargs)

        def add_scribble(plane, mask, include, trigger):
            events.append(dict(kind='scribble', plane=plane, role='foreground' if include else 'background',
                               voxels=int(np.count_nonzero(mask)), run_prediction=trigger))
            session.add_scribble_interaction(np.asarray(mask, dtype=np.uint8),
                                             include_interaction=include, run_prediction=trigger)

        try:
            session._predict = count_predict
            if point_cap is None:
                for plane, scribble in zip(PLANE_NAMES, prompts.background):
                    add_scribble(plane, scribble, False, False)
            else:
                groups = prompts.points('bg', point_cap, sampler='fps', minimum_mm=3.0)
                point_counts = [len(group) for group in groups]
                for plane, points in zip(PLANE_NAMES, groups):
                    for point in points:
                        dhw = tuple(int(index) for index in point)
                        events.append(dict(kind='point', plane=plane, role='background',
                                           coordinates_dhw=list(dhw), run_prediction=False))
                        session.add_point_interaction(dhw, include_interaction=False, run_prediction=False)

            coronal, sagittal = prompts.foreground
            add_scribble('coronal', coronal, True, False)
            add_scribble('sagittal', sagittal, True, True)
            if triggers != 1:
                raise RuntimeError('nnInteractive requires one native prediction after both foreground scribbles')
            prediction = buffer.numpy().astype(bool)
        finally:
            if had_instance_predict:
                session._predict = native_predict
            else:
                del session._predict
            session.reset_interactions()

        return prediction, dict(
            prediction_trigger_count=triggers, prompt_addition_count=len(events), foreground_scribbles=2,
            background_point_cap_per_plane=point_cap, background_point_counts=point_counts,
            background_scribbles=2 if point_cap is None else 0, events=events,
        )

    def _evaluate_pam(self, prompts):
        fusion = self.fusion
        payload = tuple(
            dict(plane=plane, slice=index, box=fusion.p.slice_box(scribble, plane, index))
            for plane, index, scribble in zip(PLANE_NAMES, prompts.planes, prompts.foreground)
        )
        predictions, view_audits = fusion.predict(prompts.ct, prompts.spacing, payload, self.models)
        if [audit['plane'] for audit in view_audits] != list(PLANE_NAMES) or any(
            audit['box_decoder_calls'] != 1 for audit in view_audits
        ):
            raise RuntimeError('PAM requires one native box evaluation per view')
        audit = dict(
            prompt_boxes_xyxy_exclusive=[list(prompt['box']) for prompt in payload],
            prompt_slices=[int(prompt['slice']) for prompt in payload], view_evaluations=len(view_audits),
            native_views=view_audits, background_prompt_count=0, gt_mask_supplied=False,
        )
        for arm in fusion.METHODS:
            yield arm, predictions[arm], dict(audit)
