"""Regression tests for rollback, saved best shapes and interrupted retries."""
from pathlib import Path
import json
import os
import sys
import tempfile
import time
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if os.environ.get("MOTHEYE_DEPENDENCY_PATH"):
    sys.path.append(os.environ["MOTHEYE_DEPENDENCY_PATH"])

import torch
from optimizer.common import load_config
from optimizer.engine import train
from optimizer.physics import NonphysicalResult


def quadratic(logits):
    return .2 + .01 * float(logits.detach().square().sum())


def valid_result(logits):
    return dict(objective=quadratic(logits), gradient=.02 * logits.detach(), seconds=.01)


class BacktrackingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def request(self, folder="trial", steps=1, lr=.04):
        config = load_config(HERE / "config.json")
        config.update(adam_steps=steps, patience=100, scheduler_patience=100)
        return dict(config=config, params=dict(lr=lr, init_top=8., delta_ratio=.9),
                    lattice="hexagonal", trial_folder=str(self.directory / folder),
                    trial_number=0, signature="test-policy-identity", deadline=None)

    def checkpoint(self, request):
        return torch.load(Path(request["trial_folder"]) / "adam_checkpoint.pt",
                          map_location="cpu", weights_only=True)

    def assert_optimizer_matches(self, left, right):
        torch.testing.assert_close(left["logits"], right["logits"], atol=1e-14, rtol=0.)
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(left["optimizer"]["state"][0][key],
                                       right["optimizer"]["state"][0][key], atol=1e-14, rtol=0.)
        self.assertEqual(left["scheduler"]["last_epoch"], right["scheduler"]["last_epoch"])

    def test_retry_restores_both_moments_and_step_count(self):
        request = self.request()
        calls = []
        def evaluate(logits, *args, **kwargs):
            calls.append(logits.detach().clone())
            if len(calls) in (2, 3):
                raise NonphysicalResult("injected power failure")
            return valid_result(logits)
        result = train(request, torch.device("cpu"), evaluator=evaluate)
        control = self.request("control", lr=.01)
        train(control, torch.device("cpu"), evaluator=lambda x, *a, **k: valid_result(x))
        actual = self.checkpoint(request)
        self.assert_optimizer_matches(actual, self.checkpoint(control))
        self.assertEqual(result["completed_steps"], 1)
        self.assertEqual(result["recovery_attempts"], 2)
        self.assertEqual(actual["history"][-1]["accepted_update_retries"], 2)
        self.assertAlmostEqual(actual["optimizer"]["param_groups"][0]["lr"], .01)

    def test_four_halved_retries_then_early_completion(self):
        request = self.request()
        calls = 0
        def evaluate(logits, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls > 1:
                raise NonphysicalResult("injected power failure")
            return valid_result(logits)
        result = train(request, torch.device("cpu"), evaluator=evaluate)
        checkpoint = self.checkpoint(request)
        self.assertEqual(calls, 6)  # Initial valid point + original proposal + four retries.
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["numerical_early_stop"])
        self.assertEqual(result["termination_reason"], "nonphysical_retry_exhausted")
        self.assertEqual(result["completed_steps"], 0)
        self.assertEqual(result["recovery_attempts"], 5)
        self.assertEqual(checkpoint["optimizer"]["state"], {})
        rates = [event["learning_rates"][0] for event in checkpoint["recovery_events"]]
        self.assertEqual(rates, [.04, .02, .01, .005, .0025])
        self.assertAlmostEqual(result["best_loss"], quadratic(torch.tensor(result["best_logits"])))

    def test_early_stop_returns_best_normal_shape_not_last_normal_shape(self):
        request = self.request(steps=3, lr=.5)
        calls = 0
        def evaluate(logits, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls > 2:
                raise NonphysicalResult("injected failure after an accepted uphill update")
            return valid_result(logits)
        result = train(request, torch.device("cpu"), evaluator=evaluate)
        checkpoint = self.checkpoint(request)
        self.assertEqual(result["best_step"], 0)
        self.assertEqual(result["completed_steps"], 1)
        self.assertLess(result["best_loss"], checkpoint["current_evaluation"]["objective"])
        self.assertAlmostEqual(result["best_loss"], quadratic(torch.tensor(result["best_logits"], dtype=torch.float64)))

    def test_retry_budget_resets_after_a_successful_update(self):
        request = self.request(steps=2)
        calls = 0
        def evaluate(logits, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls in (2, 4):
                raise NonphysicalResult("one failure per update")
            return valid_result(logits)
        result = train(request, torch.device("cpu"), evaluator=evaluate)
        checkpoint = self.checkpoint(request)
        self.assertEqual(result["completed_steps"], 2)
        self.assertEqual([h["accepted_update_retries"] for h in checkpoint["history"]], [0, 1, 1])
        self.assertEqual(checkpoint["retry_index"], 0)

    def test_interrupt_preserves_the_next_halved_retry(self):
        request = self.request()
        calls = 0
        def evaluate(logits, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise NonphysicalResult("first proposal fails")
            if calls == 3:
                raise KeyboardInterrupt()
            return valid_result(logits)
        with self.assertRaises(KeyboardInterrupt):
            train(request, torch.device("cpu"), evaluator=evaluate)
        checkpoint = self.checkpoint(request)
        self.assertEqual(checkpoint["retry_index"], 1)
        self.assertEqual(checkpoint["next_step"], 0)
        self.assertAlmostEqual(checkpoint["optimizer"]["param_groups"][0]["lr"], .02)
        resumed = train(request, torch.device("cpu"), evaluator=lambda x, *a, **k: valid_result(x))
        control = self.request("control", lr=.02)
        train(control, torch.device("cpu"), evaluator=lambda x, *a, **k: valid_result(x))
        self.assert_optimizer_matches(self.checkpoint(request), self.checkpoint(control))
        self.assertEqual(resumed["recovery_attempts"], 1)

    def test_oom_preserves_valid_checkpoint_and_propagates(self):
        request = self.request(steps=2)
        calls = 0
        def evaluate(logits, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise torch.OutOfMemoryError("injected OOM")
            return valid_result(logits)
        with self.assertRaises(torch.OutOfMemoryError):
            train(request, torch.device("cpu"), evaluator=evaluate)
        self.assertEqual(self.checkpoint(request)["next_step"], 1)
        self.assertEqual(self.checkpoint(request)["recovery_events"], [])
        train(request, torch.device("cpu"), evaluator=lambda x, *a, **k: valid_result(x))
        control = self.request("control", steps=2)
        train(control, torch.device("cpu"), evaluator=lambda x, *a, **k: valid_result(x))
        self.assert_optimizer_matches(self.checkpoint(request), self.checkpoint(control))

    def test_initial_nonphysical_point_has_no_fabricated_best_value(self):
        request = self.request()
        def evaluate(*args, **kwargs):
            raise NonphysicalResult("invalid initial point")
        with self.assertRaises(NonphysicalResult):
            train(request, torch.device("cpu"), evaluator=evaluate)
        checkpoint = self.checkpoint(request)
        self.assertIsNone(checkpoint["best_logits"])
        self.assertEqual(checkpoint["history"], [])
        self.assertFalse((Path(request["trial_folder"]) / "best.json").exists())

    def test_nonfinite_gradient_is_rejected(self):
        request = self.request()
        calls = 0
        def evaluate(logits, *args, **kwargs):
            nonlocal calls
            calls += 1
            result = valid_result(logits)
            if calls == 2:
                result["gradient"][0] = float("nan")
            return result
        result = train(request, torch.device("cpu"), evaluator=evaluate)
        self.assertEqual(result["recovery_attempts"], 1)
        self.assertTrue(torch.isfinite(self.checkpoint(request)["current_evaluation"]["gradient"]).all())

    def test_finished_checkpoint_is_idempotent_and_recreates_best_file(self):
        request = self.request()
        result = train(request, torch.device("cpu"), evaluator=lambda x, *a, **k: valid_result(x))
        (Path(request["trial_folder"]) / "best.json").unlink()
        def unexpected(*args, **kwargs):
            raise AssertionError("A finished checkpoint must not evaluate again")
        resumed = train(request, torch.device("cpu"), evaluator=unexpected)
        self.assertEqual(result, resumed)
        self.assertTrue((Path(request["trial_folder"]) / "best.json").exists())

    def test_pruning_keeps_a_valid_best_shape(self):
        request = self.request(steps=3)
        request["config"]["pruner_interval_steps"] = 1
        class Trial:
            def __init__(self):
                self.steps = []
            def report(self, value, step):
                self.steps.append(step)
            def should_prune(self):
                return self.steps[-1] == 1
        trial = Trial()
        result = train(request, torch.device("cpu"), trial=trial,
                       evaluator=lambda x, *a, **k: valid_result(x))
        self.assertEqual(trial.steps, [0, 1])
        self.assertEqual(result["status"], "pruned")
        self.assertEqual(result["completed_steps"], 1)
        self.assertAlmostEqual(result["best_loss"], quadratic(torch.tensor(result["best_logits"], dtype=torch.float64)))

    def test_time_budget_pause_keeps_pending_retry(self):
        request = self.request()
        calls = 0
        def evaluate(logits, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                request["deadline"] = time.time() - 1.
                raise NonphysicalResult("failure just before the time limit")
            return valid_result(logits)
        paused = train(request, torch.device("cpu"), evaluator=evaluate)
        self.assertEqual(paused["status"], "paused")
        self.assertEqual(self.checkpoint(request)["retry_index"], 1)
        request["deadline"] = None
        resumed = train(request, torch.device("cpu"), evaluator=lambda x, *a, **k: valid_result(x))
        self.assertEqual(resumed["completed_steps"], 1)
        self.assertEqual(resumed["recovery_attempts"], 1)

    def test_ordinary_error_propagates_without_backtracking(self):
        request = self.request()
        calls = 0
        def evaluate(logits, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ValueError("injected ordinary configuration error")
            return valid_result(logits)
        with self.assertRaises(ValueError):
            train(request, torch.device("cpu"), evaluator=evaluate)
        self.assertEqual(self.checkpoint(request)["recovery_events"], [])

    def test_legacy_checkpoint_is_rejected(self):
        request = self.request()
        folder = Path(request["trial_folder"])
        folder.mkdir()
        torch.save(dict(signature=request["signature"], params=request["params"]), folder / "adam_checkpoint.pt")
        with self.assertRaisesRegex(ValueError, "identity differs"):
            train(request, torch.device("cpu"), evaluator=lambda x, *a, **k: valid_result(x))

    def test_zero_retry_limit_still_keeps_the_initial_best(self):
        request = self.request()
        request["config"]["nonphysical_max_retries"] = 0
        calls = 0
        def evaluate(logits, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls > 1:
                raise NonphysicalResult("failed proposal")
            return valid_result(logits)
        result = train(request, torch.device("cpu"), evaluator=evaluate)
        self.assertEqual(calls, 2)
        self.assertTrue(result["numerical_early_stop"])
        self.assertEqual(result["completed_steps"], 0)


if __name__ == "__main__":
    unittest.main()
