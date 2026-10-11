"""Lossless physics, shape gradients and conservative method selection."""
from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from .compare import (comparison_exit_code, main, recommendation, scan_axis,
                      validate_final_spectrum, converge_with_spectrum)
from .model import (Geometry, Numerical, Study, choose_candidate, physical,
                    solve, trapezoid_weights)

HERE = Path(__file__).resolve().parent


def row(r, w=550., t=None):
    t = 1 - r if t is None else t
    return {"wavelength_nm": w, "reflectance": r, "transmittance": t, "absorptance": 1 - r - t}


def fake_reports():
    reports = {}
    for method, forward_cost, gradient_cost in (("nv", 1., 3.), ("asr", 2., 2.)):
        spectrum = [row(.01, w) for w in (400., 550., 700.)]
        reports["hexagonal/" + method] = {"status": "converged", "tolerance": .001,
            "energy_tolerance": 1e-6, "anchor_spectrum": spectrum, "final_spectrum": spectrum,
            "final_spectrum_complete": True, "final_spectrum_mesh_validation": {"complete": True, "passed": True}, "audit": {
                "forward_physical": True, "forward": {"median_seconds": forward_cost},
                "gradient": {"median_seconds": gradient_cost}, "gradient_passed": True,
                "gradient_per_nm": {"height_nm": 1e-4, "base_radius_nm": 2e-4},
                "settings": {"integration": {"device": "cpu"}, "wavelengths": [400., 550., 700.],
                             "atol_per_nm": 1e-7, "rtol": .05}}}
    return reports


class Guards(unittest.TestCase):
    def decision(self, reports, require_spectrum=True):
        return recommendation(reports, "hexagonal", .001, 1e-7, .05, require_spectrum)

    def test_late_instability_and_artificial_loss_block_convergence(self):
        spectra = {i: [row(r)] for i, r in enumerate((.01, .01001, .01002, .03), 1)}
        self.assertFalse(choose_candidate([1, 2, 3, 4], spectra, .001, 1e-6)[1])
        lossy = {i: [row(.2, t=.7)] for i in (1, 2, 3)}
        self.assertFalse(choose_candidate([1, 2, 3], lossy, .001, 1e-6)[1])
        self.assertFalse(physical(row(.2, t=.7), 1e-6))

    def test_late_two_intervals_select_middle(self):
        spectra = {i: [row(r)] for i, r in enumerate((.03, .01, .01001, .01002), 1)}
        selected, passed, _ = choose_candidate([1, 2, 3, 4], spectra, .001, 1e-6)
        self.assertTrue(passed)
        self.assertEqual(selected, 3)

    def test_scan_selects_earliest_stable_fine_candidate_without_cumulative_drift(self):
        class FakeStudy:
            values = (.01, .0101, .0102, .0103, .0104)

            def spectrum(self, n, wavelengths):
                return [row(self.values[n.order - 1])]

        study = FakeStudy()
        scan = scan_axis(study, "order", [1, 2, 3, 4, 5], Numerical(4, 4, 64), [550.], .001, 1e-6, 1)
        self.assertEqual(scan["selected"], 2)
        # Each local step is small, but accumulated error excludes order 2.
        study.values = (.01, .0107, .0114, .0121, .0128)
        scan = scan_axis(study, "order", [1, 2, 3, 4, 5], Numerical(4, 4, 64), [550.], .001, 1e-6, 1)
        self.assertEqual(scan["selected"], 4)
        study.values = (.01, .0101, .0102, .0103, .02)
        scan = scan_axis(study, "order", [1, 2, 3, 4, 5], Numerical(4, 4, 64), [550.], .001, 1e-6, 1)
        self.assertFalse(scan["converged"])

    def test_nonuniform_wavelength_average(self):
        weights = trapezoid_weights([400, 500, 700])
        self.assertAlmostEqual(sum(weights), 1)
        self.assertAlmostEqual(sum(w * y for w, y in zip(weights, (.1, .2, .4))), .25)

    def test_full_mesh_validation_detects_peak_between_original_anchors_and_joint_drift(self):
        class FakeStudy:
            lattice, method = "hexagonal", "asr"
            peak = True

            def spectrum(self, n, wavelengths):
                refined_axes = sum((n.order > 2, n.slices > 4, n.grid > 64))
                return [row(.01 + (.003 if self.peak and w == 645 else .0004 * refined_axes), w)
                        for w in wavelengths]

        study = FakeStudy()
        spectrum = [row(.01, w) for w in (400., 550., 645., 700.)]
        with redirect_stdout(io.StringIO()):
            result = validate_final_spectrum(study, Numerical(2, 4, 64),
                {"order": [1, 2, 3], "slices": [2, 4, 6], "grid": [32, 64, 96]}, spectrum, .001, 1e-6)
            self.assertFalse(result["passed"])
            self.assertTrue(result["complete"])
            self.assertEqual(result["checks"][0]["worst_wavelength_nm"], 645.)
            study.peak = False
            result = validate_final_spectrum(study, Numerical(2, 4, 64),
                {"order": [1, 2, 3], "slices": [2, 4, 6], "grid": [32, 64, 96]}, spectrum, .001, 1e-6)
            self.assertTrue(all(c["passed"] for c in result["checks"][:3]))
            self.assertFalse(result["checks"][-1]["passed"])
            self.assertFalse(result["passed"])

    def test_three_anchors_promote_hidden_peak_then_validate_again(self):
        class FakeStudy:
            lattice, method = "hexagonal", "asr"
            cases = {}

            def evaluate(self, n, wavelength):
                # Only 645 nm needs M>=3; the original three points look flat.
                return row(.01 + (.01 if wavelength == 645 and n.order < 3 else 0), wavelength)

            def spectrum(self, n, wavelengths):
                return [self.evaluate(n, w) for w in wavelengths]

        for limit, expected_status in ((0, "final_spectrum_not_converged"), (2, "converged")):
            report = {"history": [], "recommendation": {"order": 4, "slices": 3, "grid": 64}}
            with redirect_stdout(io.StringIO()):
                numerical = converge_with_spectrum(FakeStudy(), Numerical(4, 3, 64),
                    {"order": [1, 2, 3, 4, 5], "slices": [2, 3, 4], "grid": [32, 64, 96]},
                    (400., 550., 700.), (400., 550., 645., 700.), tolerance=.001,
                    energy_tolerance=1e-6, max_cycles=3, run_final_spectrum=True,
                    max_anchor_refinements=limit, report=report, save=lambda: None)
            self.assertEqual(report["status"], expected_status)
            if limit:
                self.assertEqual(numerical.order, 4)
                self.assertIn(645., report["anchor_wavelengths_nm"])
                self.assertEqual(report["anchor_refinements"][0]["added_wavelengths_nm"], (645.,))
                self.assertTrue(report["final_spectrum_mesh_validation"]["passed"])
            else:
                self.assertFalse(report["final_spectrum_mesh_validation"]["passed"])

    def test_forward_and_gradient_cost_choose_separately(self):
        choice = self.decision(fake_reports())
        self.assertEqual(choice["status"], "gradient_validated")
        self.assertEqual(choice["derivative_free_method"]["method"], "nv")
        self.assertEqual(choice["gradient_method"]["method"], "asr")

    def test_disagreement_prevents_speed_recommendation(self):
        reports = fake_reports()
        reports["hexagonal/asr"]["final_spectrum"][0] = row(.1, 400)
        choice = self.decision(reports)
        self.assertEqual(choice["status"], "methods_disagree")
        self.assertIsNone(choice["derivative_free_method"])

    def test_looser_cached_tolerance_and_different_hardware_are_rejected(self):
        reports = fake_reports()
        reports["hexagonal/nv"]["tolerance"] = .01
        self.assertEqual(self.decision(reports)["status"], "insufficient_evidence")
        reports = fake_reports()
        reports["hexagonal/nv"]["audit"]["settings"]["integration"] = {"device": "cuda:0"}
        self.assertEqual(self.decision(reports)["status"], "insufficient_evidence")

    def test_incomplete_full_spectrum_and_different_benchmark_wavelengths(self):
        reports = fake_reports()
        reports["hexagonal/nv"]["final_spectrum_complete"] = False
        self.assertEqual(self.decision(reports)["status"], "insufficient_evidence")
        reports = fake_reports()
        del reports["hexagonal/nv"]["final_spectrum_mesh_validation"]
        self.assertEqual(self.decision(reports)["status"], "insufficient_evidence")
        reports["hexagonal/nv"]["final_spectrum_mesh_validation"] = None
        self.assertEqual(self.decision(reports)["status"], "insufficient_evidence")
        reports = fake_reports()
        reports["hexagonal/nv"]["audit"]["settings"]["wavelengths"] = [550.]
        self.assertEqual(self.decision(reports)["status"], "insufficient_evidence")

    def test_gradient_disagreement_and_near_equal_timings(self):
        reports = fake_reports()
        reports["hexagonal/asr"]["audit"]["gradient_per_nm"]["height_nm"] = 1e-3
        choice = self.decision(reports)
        self.assertEqual(choice["status"], "forward_validated")
        self.assertIsNone(choice["gradient_method"])
        reports = fake_reports()
        reports["hexagonal/asr"]["audit"]["forward"]["median_seconds"] = 1.05
        self.assertEqual(self.decision(reports)["derivative_free_method"]["method"], "comparable")

    def test_gradient_failure_cannot_report_success_for_requested_audit(self):
        reports = fake_reports()
        for report in reports.values():
            report["audit"]["gradient_passed"] = False
        choices = {"hexagonal": self.decision(reports)}
        self.assertEqual(comparison_exit_code(reports, reports, choices, ["hexagonal"], require_gradients=True), 2)
        self.assertEqual(comparison_exit_code(reports, reports, choices, ["hexagonal"], require_gradients=False), 0)
        reports["hexagonal/nv"]["status"] = "calculation_failed"
        self.assertEqual(comparison_exit_code(reports, reports, choices, ["hexagonal"], require_gradients=False), 1)

    def test_dry_run_has_no_torch_dependency(self):
        import builtins
        original = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise AssertionError("Torch was imported in preflight")
            return original(name, *args, **kwargs)

        with tempfile.TemporaryDirectory(dir=HERE) as folder, patch("builtins.__import__", guarded), redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--device", "cuda", "--dry-run", "--output-root", folder]), 0)
            plan = json.loads((Path(folder) / "plan.json").read_text())
            self.assertEqual(plan["assumptions"]["geometry"]["index"], 2.5)
            self.assertEqual(plan["substrate_first_diffraction_threshold_nm"]["square"], 500)
            self.assertEqual(plan["anchor_wavelengths_nm"], [400., 550., 700.])
            self.assertEqual(len(plan["gradient_audit"]["wavelengths_nm"]), 11)
            self.assertEqual(main(["--device", "cuda", "--dry-run", "--output-root", folder,
                "--anchor-wavelengths", "400,430,436,475,495,505,550,600,645,650,700"]), 0)
            dense_plan = json.loads((Path(folder) / "plan.json").read_text())
            self.assertEqual(plan["signature"], dense_plan["signature"])


class Physics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(2)
        cls.device = torch.device("cpu")

    def test_four_cases_lossless_including_open_orders_and_rayleigh_threshold(self):
        g, n = Geometry(), Numerical(2, 3, 64)
        for lattice in ("hexagonal", "square"):
            for method in ("nv", "asr"):
                for w in (400., 500.):
                    with self.subTest(lattice=lattice, method=method, wavelength=w):
                        result, reflected = solve(w, n, g, method=method, lattice=lattice, device=self.device)
                        self.assertTrue(physical(result, 1e-6))
                        self.assertEqual(result["method"], method)
                        del reflected

    def test_four_cases_shape_autograd_matches_finite_differences(self):
        g, n = Geometry(), Numerical(2, 4, 64)
        for lattice in ("hexagonal", "square"):
            for method in ("nv", "asr"):
                h = self.torch.tensor(500., dtype=self.torch.float64, requires_grad=True)
                b = self.torch.tensor(95., dtype=self.torch.float64, requires_grad=True)
                _, reflected = solve(550., n, g, method=method, lattice=lattice, device=self.device,
                                     parameters={"height_nm": h, "base_radius_nm": b})
                derivatives = self.torch.autograd.grad(reflected, (h, b))
                for parameter, step, derivative in zip(("height_nm", "base_radius_nm"), (.1, .02), derivatives):
                    with self.subTest(lattice=lattice, method=method, parameter=parameter):
                        plus, _ = solve(550., n, replace(g, **{parameter: getattr(g, parameter) + step}),
                                        method=method, lattice=lattice, device=self.device)
                        minus, _ = solve(550., n, replace(g, **{parameter: getattr(g, parameter) - step}),
                                         method=method, lattice=lattice, device=self.device)
                        fd = (plus["reflectance"] - minus["reflectance"]) / (2 * step)
                        ad = float(derivative)
                        self.assertAlmostEqual(ad, fd, delta=1e-7 + .01 * abs(fd))

    def test_real_checkpoint_reuse_and_signature_guard(self):
        with tempfile.TemporaryDirectory(dir=HERE) as folder, redirect_stdout(io.StringIO()):
            kwargs = dict(geometry=Geometry(), method="nv", lattice="square", device=self.device,
                          use_symmetry=True, signature="unit-check", checkpoint=Path(folder) / "checkpoint.json")
            original = Study(**kwargs)
            result = original.evaluate(Numerical(1, 2, 32), 550.)
            resumed = Study(**kwargs)
            with patch("studies.dielectric_motheye_n25.model.solve", side_effect=AssertionError("recalculated")):
                self.assertEqual(resumed.evaluate(Numerical(1, 2, 32), 550.), result)
            with self.assertRaisesRegex(ValueError, "signature differs"):
                Study(**{**kwargs, "signature": "changed-geometry"})

    def test_asr_layer_checkpoint_preserves_powers_and_both_shape_gradients(self):
        for lattice in ("hexagonal", "square"):
            outputs = []
            retained_bytes = []
            for checkpointed in (False, True):
                h = self.torch.tensor(500., dtype=self.torch.float64, requires_grad=True)
                b = self.torch.tensor(95., dtype=self.torch.float64, requires_grad=True)
                saved = []

                def pack(tensor):
                    saved.append(tensor.numel() * tensor.element_size())
                    return tensor

                with self.torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                    _, reflected = solve(550., Numerical(2, 4, 64), Geometry(),
                        method="asr", lattice=lattice, device=self.device,
                        parameters={"height_nm": h, "base_radius_nm": b},
                        checkpoint_layers=checkpointed)
                retained_bytes.append(sum(saved))
                outputs.append([float(reflected.detach()), *map(float, self.torch.autograd.grad(reflected, (h, b)))])
            with self.subTest(lattice=lattice):
                for a, b in zip(*outputs):
                    self.assertAlmostEqual(a, b, delta=1e-10 + abs(a) * 1e-8)
                # Counts tensors saved for backward, not CUDA peak memory.
                self.assertLess(retained_bytes[1], retained_bytes[0] * .5)

    def test_square_radial_large_radius_has_stable_trace_and_exact_mirror_masks(self):
        from rcwa_solver_auto import ASROptions, AutoRCWA, GroupTheoryOptions, Lattice, OutputSpec
        sim = AutoRCWA(freq=200/550, order=[12, 12], lattice=Lattice.square(1.),
            asr=ASROptions(circle_profile="radial", circle_G=.03, grid=(128, 128)),
            outputs=OutputSpec(smatrix_size="half", fields="none"),
            group_theory=GroupTheoryOptions(enabled=True, symmetry="c2v", polarization="x", strict=True),
            dtype=self.torch.complex128, device=self.device, verify_cascade=False)
        sim.add_input_layer(eps=1.)
        sim.add_output_layer(eps=6.25)
        sim.set_incident_angle(0., 0.)
        mapping = sim.build_square_circle_asr_mapping(128, 128, 92.75/200)
        self.assertGreater(float(mapping.det_j.min()), .001)
        transform, _ = sim._build_circle_conversion_matrices(mapping)
        eb, _ = sim._c2v_source_bases((.5, .5), "x")
        self.assertLess(float(self.torch.linalg.cond(eb.mH @ transform @ eb)), 1e5)
        # Include points exactly on the computational/material boundary.
        boundary = sim.build_square_circle_asr_mapping(128, 128, .25)
        mask = boundary.matched_outer_mask
        mirror = (-self.torch.arange(128)) % 128
        self.assertTrue(self.torch.equal(mask, mask[mirror, :]))
        self.assertTrue(self.torch.equal(mask, mask[:, mirror]))


if __name__ == "__main__":
    unittest.main()
