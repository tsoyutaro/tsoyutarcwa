"""Numerical safeguards and small CPU checks, not a full convergence study."""
from __future__ import annotations

import json
import io
from contextlib import redirect_stdout
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from .converge import main, spectrum_statistics
from .model import (Geometry, Numerical, PMMAIndex, Study, choose_candidate,
                    load_tabulated_nk, physical, signature_for)

HERE = Path(__file__).resolve().parent


def row(r, t=.2, w=550.):
    return {"wavelength_nm": w, "reflectance": r, "transmittance": t,
            "absorptance": 1 - r - t, "passivity_warning": False}


class Safeguards(unittest.TestCase):
    def test_early_plateau_cannot_pass_late_instability(self):
        spectra = {i: [row(r)] for i, r in enumerate((.2, .2001, .2002, .4), 1)}
        selected, passed, _ = choose_candidate([1, 2, 3, 4], spectra, .005)
        self.assertFalse(passed)
        self.assertEqual(selected, 4)

    def test_final_two_intervals_pass_and_choose_middle(self):
        spectra = {i: [row(r)] for i, r in enumerate((.4, .2, .2001, .2002), 1)}
        selected, passed, _ = choose_candidate([1, 2, 3, 4], spectra, .005)
        self.assertTrue(passed)
        self.assertEqual(selected, 3)

    def test_nonphysical_flat_results_do_not_pass(self):
        spectra = {i: [row(.9)] for i in (1, 2, 3)}
        self.assertFalse(choose_candidate([1, 2, 3], spectra, .005)[1])
        self.assertFalse(physical(row(.9)))
        with self.assertRaises(ValueError):
            physical(row(float("nan")))

    def test_material_edge_and_cr_nodes(self):
        pmma = PMMAIndex(HERE / "data/Szczurowski.csv", "linear-to-400")
        self.assertAlmostEqual(pmma(400), 1.5058386407194826, places=12)
        self.assertAlmostEqual(pmma(404.7), 1.5051867964397, places=12)
        with self.assertRaises(ValueError):
            pmma(399)
        with self.assertRaises(ValueError):
            PMMAIndex(HERE / "data/Szczurowski.csv", "error")(400)
        cr = load_tabulated_nk(HERE / "data/Johnson.csv")
        for w, n, k in zip(cr.wavelength_nm, cr.n, cr.k):
            self.assertEqual(cr(w), complex(n, k) ** 2)

    def test_geometry_layer_thickness_and_valley_position(self):
        geometry = Geometry()
        for nz in (4, 40, 60, 80, 100):
            layers = geometry.profile(nz)
            self.assertEqual(len(layers), nz)
            self.assertAlmostEqual(sum(l.bottom_depth_nm - l.top_depth_nm for l in layers), 500)
            valleys = [l for l in layers if l.kind == "valley"]
            self.assertEqual(valleys[0].top_depth_nm, 470)
            self.assertAlmostEqual(sum(l.bottom_depth_nm - l.top_depth_nm for l in valleys), 30)
            self.assertTrue(all(l.outer_radius_nm is None for l in valleys))
        with self.assertRaises(ValueError):
            replace(geometry, cr_side_thickness_nm=35).validate()

    def test_statistics_are_actual_r_t_a_and_wavelength_weighted(self):
        rows = [row(.1, .2, 400), row(.2, .3, 500), row(.4, .1, 700)]
        stats = spectrum_statistics(rows)["powers"]
        self.assertAlmostEqual(stats["reflectance"]["wavelength_mean_trapezoidal"], .25)
        self.assertAlmostEqual(sum(stats[name]["wavelength_mean_trapezoidal"]
                                   for name in stats), 1)

    def test_dry_run_does_not_load_torch(self):
        import builtins
        original = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise AssertionError("Dry run attempted to import Torch")
            return original(name, *args, **kwargs)

        with tempfile.TemporaryDirectory(dir=HERE) as folder, patch("builtins.__import__", guarded), redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--dry-run", "--device", "cuda", "--output-root", folder]), 0)
            plan = json.loads((Path(folder) / "plan.json").read_text())
            self.assertEqual(plan["assumptions"]["geometry"]["cr_side_thickness_nm"], 30.)
            self.assertTrue((Path(folder) / "geometry.svg").exists())


class SmallPhysics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2)
        cls.torch = torch

    def test_lossless_flux_and_default_cap_compatibility(self):
        from studies.pmma_gold_motheye.common import GeometryConfig, NumericalConfig, simulate_case
        geometry = GeometryConfig(base_radius_nm=65, gold_thickness_nm=30)
        numerical = NumericalConfig(2, 4, 32)
        kwargs = dict(cascade="redheffer", use_symmetry=True, device=self.torch.device("cpu"),
                      valley_gold_thickness_nm=30)
        # The patterned metal surrogate is lossless. R+T must independently be 1.
        default = simulate_case(550, numerical, geometry, lambda w: 3 + 0j, **kwargs)
        explicit = simulate_case(550, numerical, geometry, lambda w: 3 + 0j,
                                 top_cap_thickness_nm=30, **kwargs)
        self.assertAlmostEqual(default["reflectance"] + default["transmittance"], 1, delta=1e-8)
        for name in ("reflectance", "transmittance", "absorptance"):
            self.assertAlmostEqual(default[name], explicit[name], delta=1e-12)
        homogeneous = simulate_case(550, numerical, replace(geometry, pmma_index=1),
                                    lambda w: 1 + 0j, top_cap_thickness_nm=0, **kwargs)
        refined = simulate_case(550, replace(numerical, grid=64), replace(geometry, pmma_index=1),
                                lambda w: 1 + 0j, top_cap_thickness_nm=0, **kwargs)
        # A nontrivial coordinate map has a finite quadrature error even in air.
        # Refining its grid must recover near-zero reflection and unit transmission.
        self.assertLess(refined["reflectance"], homogeneous["reflectance"] / 20)
        self.assertAlmostEqual(refined["reflectance"], 0, delta=1e-6)
        self.assertAlmostEqual(refined["transmittance"], 1, delta=1e-6)
        self.assertEqual(homogeneous["total_pattern_layers"], 4)

    def test_cr_wrapper_resume_and_mismatched_signature(self):
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            kwargs = dict(geometry=Geometry(), cr_csv=HERE / "data/Johnson.csv",
                          pmma_csv=HERE / "data/Szczurowski.csv", pmma_policy="linear-to-400",
                          device=self.torch.device("cpu"), radial_mapping="double", cascade="redheffer",
                          use_symmetry=True, signature=signature_for({"test": "Cr"}),
                          checkpoint=Path(folder) / "checkpoint.json")
            first = Study(**kwargs)
            result = first.evaluate(Numerical(2, 4, 32), 550.)
            self.assertTrue(physical(result))
            self.assertIn("epsilon_cr_imag", result)
            self.assertNotIn("epsilon_gold_imag", result)
            resumed = Study(**kwargs)
            with patch("studies.pmma_gold_motheye.common.simulate_case", side_effect=AssertionError("recalculated")):
                self.assertEqual(resumed.evaluate(Numerical(2, 4, 32), 550.), result)
            kwargs["signature"] = signature_for({"test": "different geometry"})
            with self.assertRaisesRegex(ValueError, "signature differs"):
                Study(**kwargs)


if __name__ == "__main__":
    unittest.main()
