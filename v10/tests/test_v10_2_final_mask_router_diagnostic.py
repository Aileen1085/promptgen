"""Contracts for the training-only final-mask route diagnosis."""

import unittest
import subprocess
import sys
import importlib.util


class TrainingSubsetTests(unittest.TestCase):
    def test_cli_exposes_training_only_diagnostic_output(self):
        if importlib.util.find_spec("peft") is None:
            self.skipTest("full v10 runtime dependencies are not installed locally")
        completed = subprocess.run(
            [sys.executable, "-m", "v10.v10_2_final_mask_router_diagnostic", "--help"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--diagnostic-output", completed.stdout)

    def test_subset_is_deterministic_and_covers_required_sources_and_classes(self):
        from v10.v10_2_final_mask_router_diagnostic import select_training_tasks

        def entry(source, case, classes):
            return {
                "case_id": case,
                "image": "/train/{}/{}.nii.gz".format(source, case),
                "label": "/labels/{}/{}.nii.gz".format(source, case),
                "classes": classes,
            }

        split = {"datasets": {
            "totalseg": {"train": [entry("totalseg", "t1", [38, 69, 116]),
                                   entry("totalseg", "t2", [38, 69, 116])], "val": []},
            "amos": {"train": [entry("amos", "a1", [118]),
                               entry("amos", "a2", [119])], "val": []},
            "msd_task10": {"train": [entry("msd_task10", "m1", [142]),
                                     entry("msd_task10", "m2", [142])], "val": []},
        }}
        validation = {"tasks": []}
        kwargs = {"quotas": {"totalseg": 3, "amos": 2, "msd_task10": 2},
                  "hard_classes": (38, 69, 116), "seed": 2027}
        first = select_training_tasks(split, validation, **kwargs)
        self.assertEqual(first, select_training_tasks(split, validation, **kwargs))
        self.assertEqual(len(first), 7)
        self.assertEqual({38, 69, 116}, {r["global_class_id"] for r in first
                                         if r["dataset"] == "totalseg"})
        self.assertEqual({"amos", "msd_task10", "totalseg"},
                         {r["dataset"] for r in first})

    def test_rejects_validation_case_even_when_class_differs(self):
        from v10.v10_2_final_mask_router_diagnostic import select_training_tasks

        row = {"case_id": "a1", "image": "/train/a1.nii.gz",
               "label": "/train/a1-label.nii.gz", "classes": [118]}
        split = {"datasets": {"amos": {"train": [row], "val": []}}}
        validation = {"tasks": [{"dataset": "amos", "case_id": "a1",
                                  "global_class_id": 119,
                                  "image": "/another/path.nii.gz"}]}
        with self.assertRaisesRegex(ValueError, "validation"):
            select_training_tasks(split, validation, quotas={"amos": 1},
                                  hard_classes=(), seed=2027)

    def test_rejects_validation_path_even_when_case_id_differs(self):
        from v10.v10_2_final_mask_router_diagnostic import select_training_tasks

        row = {"case_id": "a1", "image": "/shared/image.nii.gz",
               "label": "/train/a1-label.nii.gz", "classes": [118]}
        split = {"datasets": {"amos": {"train": [row], "val": []}}}
        validation = {"tasks": [{"dataset": "amos", "case_id": "a2",
                                  "global_class_id": 118,
                                  "image": "/shared/image.nii.gz"}]}
        with self.assertRaisesRegex(ValueError, "validation"):
            select_training_tasks(split, validation, quotas={"amos": 1},
                                  hard_classes=(), seed=2027)


class RouteRankingTests(unittest.TestCase):
    def test_enumerates_shared_and_four_experts_then_restores_route(self):
        from v10.v10_2_final_mask_router_diagnostic import evaluate_counterfactual_routes

        class Encoder:
            route_override = "router"

        encoder = Encoder()
        observed = []

        def evaluate():
            observed.append(encoder.route_override)
            return float(len(observed))

        scores = evaluate_counterfactual_routes(encoder, evaluate)
        self.assertEqual(observed, ["shared", "expert:0", "expert:1",
                                    "expert:2", "expert:3"])
        self.assertEqual(set(scores), set(observed))
        self.assertEqual(encoder.route_override, "router")

    def test_restores_route_on_error(self):
        from v10.v10_2_final_mask_router_diagnostic import evaluate_counterfactual_routes

        class Encoder:
            route_override = "router"

        encoder = Encoder()
        with self.assertRaisesRegex(RuntimeError, "fail"):
            evaluate_counterfactual_routes(
                encoder, lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        self.assertEqual(encoder.route_override, "router")

    def test_final_mask_loss_ranking_and_agreement(self):
        from v10.v10_2_final_mask_router_diagnostic import ranking_diagnostics

        result = ranking_diagnostics(
            final_mask_losses={"shared": 0.3, "expert:0": 0.1,
                               "expert:1": 0.2, "expert:2": 0.4,
                               "expert:3": 0.5},
            router_probabilities={"shared": 0.1, "expert:0": 0.6,
                                  "expert:1": 0.2, "expert:2": 0.05,
                                  "expert:3": 0.05},
            auxiliary_losses={"shared": 0.5, "expert:0": 0.4,
                              "expert:1": 0.1, "expert:2": 0.2,
                              "expert:3": 0.3},
        )
        self.assertEqual(result["final_mask_best"], "expert:0")
        self.assertEqual(result["router_best"], "expert:0")
        self.assertEqual(result["router_top1_agreement"], 1.0)
        self.assertEqual(result["auxiliary_top1_agreement"], 0.0)
        self.assertAlmostEqual(result["oracle_gain_vs_router_top1"], 0.0)

    def test_final_mask_loss_ranks_sam2_logits_against_same_target(self):
        import torch
        from v10.v10_2_final_mask_router_diagnostic import final_mask_loss

        target = torch.tensor([[[[1.0, 0.0]]]])
        correct = torch.tensor([[[[5.0, -5.0]]]])
        wrong = -correct
        self.assertLess(final_mask_loss(correct, target),
                        final_mask_loss(wrong, target))

    def test_auxiliary_losses_rank_five_channels_without_using_final_masks(self):
        import torch
        from v10.v10_2_final_mask_router_diagnostic import auxiliary_route_losses

        target = torch.ones(1, 1, 2, 2)
        coarse = torch.full((1, 5, 1, 1, 1), -4.0)
        coarse[:, 2] = 4.0
        losses = auxiliary_route_losses(coarse, target)
        self.assertEqual(set(losses), {"shared", "expert:0", "expert:1",
                                      "expert:2", "expert:3"})
        self.assertEqual(min(losses, key=losses.get), "expert:1")

    def test_router_probabilities_are_read_before_route_override(self):
        import torch
        from v10.v10_2_final_mask_router_diagnostic import router_route_probabilities

        probabilities = torch.tensor([[0.1, 0.2, 0.4, 0.2, 0.1]])
        mapped = router_route_probabilities(probabilities)
        self.assertAlmostEqual(mapped["expert:1"], 0.4)
        with self.assertRaises(ValueError):
            router_route_probabilities(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))

    def test_task_runner_uses_unforced_router_and_same_target_for_all_masks(self):
        import torch
        from v10.v10_2_final_mask_router_diagnostic import evaluate_task_routes

        class Encoder:
            route_override = "expert:3"

        encoder = Encoder()
        target = torch.tensor([[[[1.0, 0.0]]]])
        seen = []

        def run_route():
            seen.append(encoder.route_override)
            correct = torch.tensor([[[[5.0, -5.0]]]])
            logits = correct if encoder.route_override == "expert:1" else -correct
            probability = torch.tensor([[0.1, 0.1, 0.6, 0.1, 0.1]])
            coarse = torch.zeros(1, 5, 1, 1, 1)
            return logits, probability, coarse

        result = evaluate_task_routes(encoder, run_route, target)
        self.assertEqual(seen, ["router", "shared", "expert:0", "expert:1",
                                "expert:2", "expert:3"])
        self.assertEqual(encoder.route_override, "expert:3")
        self.assertEqual(result["ranking"]["final_mask_best"], "expert:1")
        self.assertEqual(result["ranking"]["router_best"], "expert:1")

    def test_prompt_regions_cover_each_core_once_with_halo(self):
        from v10.v10_2_final_mask_router_diagnostic import prompt_regions

        self.assertEqual(prompt_regions(10, chunk_size=4, halo=2, full_limit=6),
                         [(0, 4, 0, 6), (4, 8, 2, 10), (8, 10, 6, 10)])
        self.assertEqual(prompt_regions(6, chunk_size=4, halo=2, full_limit=6),
                         [(0, 6, 0, 6)])

    def test_aggregates_router_and_auxiliary_across_all_prompt_chunks(self):
        import torch
        from v10.v10_2_final_mask_router_diagnostic import aggregate_prompt_diagnostics

        target = torch.ones(6, 1, 2, 2)
        first_aux = torch.full((1, 5, 1, 1, 1), -4.0)
        second_aux = first_aux.clone()
        first_aux[:, 1] = 4.0
        second_aux[:, 1] = 4.0
        records = [
            (0, 2, 0, 3, torch.tensor([[0.1, 0.6, 0.1, 0.1, 0.1]]), first_aux),
            (2, 6, 1, 6, torch.tensor([[0.1, 0.1, 0.6, 0.1, 0.1]]), second_aux),
        ]
        probabilities, auxiliary = aggregate_prompt_diagnostics(records, target)
        self.assertAlmostEqual(probabilities[0, 1].item(), 0.6 * 2 / 6 + 0.1 * 4 / 6)
        self.assertAlmostEqual(probabilities[0, 2].item(), 0.1 * 2 / 6 + 0.6 * 4 / 6)
        self.assertEqual(min(auxiliary, key=auxiliary.get), "expert:0")

    def test_frozen_route_capture_restores_prompt_method(self):
        import torch
        from types import SimpleNamespace
        from v10.v10_2_final_mask_router_diagnostic import run_frozen_model_route

        class Prompt(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder3d = SimpleNamespace(route_override="router")

            def forward_case_3d_tokens(self, *_args, **_kwargs):
                return {
                    "morphology_route_probabilities": torch.tensor(
                        [[0.2, 0.4, 0.2, 0.1, 0.1]]),
                    "morphology_counterfactual_logits": torch.ones(1, 5, 1, 1, 1),
                }

        prompt = Prompt()
        original = prompt.forward_case_3d_tokens

        def generate(model, *_args):
            model.forward_case_3d_tokens()
            return {"tokens": True}, 0

        def decode(_adapter, _video, _bundle, _tokens, _anchor,
                   _args, _device, training):
            self.assertFalse(training)
            return torch.zeros(2, 1, 2, 2), None

        args = SimpleNamespace(prompt_depth_chunk_size=4, prompt_depth_halo=2,
                               prompt_full_depth_limit=96)
        target = torch.zeros(2, 1, 2, 2)
        logits, probability, auxiliary = run_frozen_model_route(
            prompt, object(), torch.zeros(2, 1, 2, 2), {}, 38, args,
            torch.device("cpu"), target, generate=generate, decode=decode)
        self.assertEqual(tuple(logits.shape), tuple(target.shape))
        self.assertAlmostEqual(probability[0, 1].item(), 0.4)
        self.assertEqual(len(auxiliary), 5)
        self.assertEqual(prompt.forward_case_3d_tokens, original)

    def test_summary_reports_source_hard_class_and_route_distribution(self):
        from v10.v10_2_final_mask_router_diagnostic import summarize_results

        def row(source, class_id, final_best, router_best, gain):
            return {"dataset": source, "global_class_id": class_id,
                    "ranking": {"final_mask_best": final_best,
                                "router_best": router_best,
                                "auxiliary_best": "shared",
                                "router_top1_agreement": float(final_best == router_best),
                                "auxiliary_top1_agreement": float(final_best == "shared"),
                                "router_rank_correlation": 0.4,
                                "auxiliary_rank_correlation": 0.2,
                                "oracle_gain_vs_router_top1": gain}}

        summary = summarize_results([
            row("amos", 118, "expert:3", "shared", 0.3),
            row("amos", 119, "shared", "shared", 0.0),
            row("totalseg", 69, "expert:0", "expert:1", 0.2),
        ], hard_classes=(69, 38, 116))
        self.assertEqual(summary["overall"]["tasks"], 3)
        self.assertAlmostEqual(summary["per_source"]["amos"]["oracle_gain_mean"], 0.15)
        self.assertEqual(summary["per_hard_class"]["69"]["tasks"], 1)
        self.assertEqual(summary["router_selected_distribution"]["shared"], 2)


if __name__ == "__main__":
    unittest.main()
