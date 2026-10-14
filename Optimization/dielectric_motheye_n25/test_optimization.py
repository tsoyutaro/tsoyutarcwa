"""Physics parity, differentiable shape, and loss/checkpoint integrity."""
from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from .common import HERE, load_config
from .engine import train
from .geometry import baseline_logits, radii_tensor, radii_values
from .physics import solve_train, solve_verify


def quadratic(logits, config, lattice, waves, order, grid, device, *, gradient):
    value = .01+.001*((logits-.2)**2).mean()
    g = torch.autograd.grad(value, logits)[0] if gradient else None
    return {"objective": float(value.detach()), "gradient": g, "seconds": .001, "rows": []}


class PhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config = load_config(HERE/"config.json", smoke=True)
        cls.device = torch.device("cpu")

    def test_softmax_baseline_bounds_and_gauge_invariance(self):
        c = self.config
        x = torch.tensor(baseline_logits(c), dtype=torch.float64, requires_grad=True)
        r = radii_tensor(x, c)
        expected = torch.tensor([5+90*(i+.5)/15 for i in range(15)], dtype=torch.float64)
        self.assertTrue(torch.allclose(r, expected, atol=1e-10, rtol=1e-10))
        self.assertTrue(torch.allclose(r, radii_tensor(x+100, c), atol=1e-10, rtol=1e-10))
        extreme = radii_tensor(torch.linspace(-100, 100, 16, dtype=torch.float64), c)
        self.assertTrue(bool((extreme[1:] > extreme[:-1]).all()))
        self.assertGreater(float(extreme[0]), c["radius_min_nm"])
        self.assertLess(float(extreme[-1]), c["radius_max_nm"])

    def test_streamed_verification_equals_retained_training_for_both_lattices(self):
        c = self.config
        radii = torch.tensor(radii_values(baseline_logits(c), c), dtype=torch.float64)
        for lattice in ("square", "hexagonal"):
            for wave in (400., 550.):
                _, a = solve_train(radii, c, lattice, wave, 2, 48, self.device, release=False)
                _, b = solve_train(radii, c, lattice, wave, 2, 48, self.device)
                d = solve_verify(radii, c, lattice, wave, 2, 48, self.device)
                for metric in ("reflectance", "transmittance"):
                    self.assertAlmostEqual(a[metric], b[metric], delta=1e-10)
                    self.assertAlmostEqual(a[metric], d[metric], delta=1e-10)
                self.assertLess(abs(d["energy_residual"]), 1e-6)

    def test_checkpointing_preserves_gradients_and_directional_finite_difference(self):
        c = self.config
        for lattice in ("square", "hexagonal"):
            gradients = []
            for checkpointed in (False, True):
                x = torch.tensor(baseline_logits(c), dtype=torch.float64, requires_grad=True)
                value, _ = solve_train(radii_tensor(x, c), c, lattice, 550., 1, 32, self.device,
                                       checkpoint_layers=checkpointed)
                gradients.append(torch.autograd.grad(value, x)[0])
            self.assertTrue(torch.allclose(*gradients, atol=1e-9, rtol=1e-6))
            d = torch.linspace(-1, 1, 16, dtype=torch.float64)
            d /= d.norm()
            x = torch.tensor(baseline_logits(c), dtype=torch.float64)
            h = 1e-4
            values = [solve_verify(radii_tensor(x+sign*h*d, c), c, lattice, 550., 1, 32, self.device)["reflectance"]
                      for sign in (1, -1)]
            fd = (values[0]-values[1])/(2*h)
            ad = float(torch.dot(gradients[0], d))
            self.assertAlmostEqual(ad, fd, delta=1e-7+.005*abs(fd))


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.c = load_config(HERE/"config.json", smoke=True)
        self.c["adam_steps"] = 3
        self.c["patience"] = 10
        self.base = {"config": self.c, "lattice": "square", "signature": "unit",
                     "trial_number": 0, "params": {"lr": .03, "init_top": 8., "delta_ratio": .8}}

    def call(self, folder, evaluator=quadratic):
        with contextlib.redirect_stdout(io.StringIO()):
            return train({**self.base, "trial_folder": str(folder)}, torch.device("cpu"), evaluator=evaluator)

    def test_saved_best_loss_matches_saved_best_shape(self):
        result = self.call(self.root/"complete")
        best = torch.tensor(result["best_logits"], dtype=torch.float64)
        value = quadratic(best, None, None, None, None, None, None, gradient=False)["objective"]
        self.assertAlmostEqual(value, result["best_loss"], delta=1e-15)

    def test_interruption_and_resume_equal_uninterrupted_adam(self):
        complete = self.call(self.root/"continuous")
        count = 0
        def interrupt(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 2:
                raise KeyboardInterrupt()
            return quadratic(*args, **kwargs)
        with self.assertRaises(KeyboardInterrupt):
            self.call(self.root/"resumed", interrupt)
        resumed = self.call(self.root/"resumed")
        self.assertEqual(complete["best_logits"], resumed["best_logits"])
        self.assertEqual(complete["best_loss"], resumed["best_loss"])

    def test_changed_identity_rejects_adam_checkpoint(self):
        self.call(self.root/"identity")
        self.base["signature"] = "changed"
        with self.assertRaisesRegex(ValueError, "identity differs"):
            self.call(self.root/"identity")


class OptunaResumeTests(unittest.TestCase):
    def test_tpe_sequence_survives_pause_and_resume(self):
        try:
            import optuna
        except ImportError:
            self.skipTest("Optuna integration requires requirements.txt")
        from . import run
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        c = load_config(HERE/"config.json", smoke=True)
        c["trials"] = 12  # Exercise TPE after its random startup phase.
        base = {"config": c, "lattice": "square", "signature": "sampler-test"}
        def completed(request):
            params = request["params"]
            score = ((params["init_top"]-20)/100)**2 + params["lr"]*.001 + params["delta_ratio"]*.01
            x = baseline_logits(c)
            return {"status": "completed", "best_loss": score, "best_logits": x,
                    "radii_nm": radii_values(x, c), "best_step": 0, "completed_steps": 2}
        with tempfile.TemporaryDirectory() as directory:
            continuous, resumed = Path(directory)/"continuous", Path(directory)/"resumed"
            continuous.mkdir()
            resumed.mkdir()
            with contextlib.redirect_stdout(io.StringIO()), patch.object(run, "launch", side_effect=lambda folder, req: completed(req)):
                a, status = run.optimize_lattice(continuous, base, None)
            self.assertEqual(status, "completed")
            def pause(folder, request):
                return {"status": "paused"} if request["trial_number"] == 3 else completed(request)
            with contextlib.redirect_stdout(io.StringIO()), patch.object(run, "launch", side_effect=pause):
                _, status = run.optimize_lattice(resumed, base, None)
            self.assertEqual(status, "paused")
            with contextlib.redirect_stdout(io.StringIO()), patch.object(run, "launch", side_effect=lambda folder, req: completed(req)):
                b, status = run.optimize_lattice(resumed, base, None)
            self.assertEqual(status, "completed")
            self.assertEqual([t.params for t in a.trials], [t.params for t in b.trials])
            self.assertEqual([t.value for t in a.trials], [t.value for t in b.trials])


if __name__ == "__main__":
    unittest.main()
