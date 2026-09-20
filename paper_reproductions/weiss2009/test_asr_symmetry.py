"""Compare complete C2v eigensystems with full solves on CPU."""
from __future__ import annotations
import json
import sys
import unittest
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paper_reproductions.weiss2009 import reproduce as r
from rcwa_ext import UnsupportedCombinationError


class SymmetryTests(unittest.TestCase):
    def test_dielectric_and_metal_both_profiles(self):
        torch.set_num_threads(4)
        for profile in r.PROFILES:
            for metal in (False, True):
                with self.subTest(profile=profile, metal=metal):
                    args = dict(order=2, profile=profile, grid=64,
                        frequency_thz=370.0 if metal else r.DIELECTRIC_FREQUENCY_THZ,
                        period_um=r.METAL_PERIOD_UM if metal else r.DIELECTRIC_PERIOD_UM,
                        radius_um=r.METAL_RADIUS_UM if metal else r.DIELECTRIC_RADIUS_UM,
                        height_um=r.METAL_HEIGHT_UM if metal else r.DIELECTRIC_HEIGHT_UM,
                        epsilon=r.gold_drude_epsilon(370) if metal else r.DIELECTRIC_EPSILON,
                        dtype=torch.complex128, device=torch.device("cpu"),
                        cascade="redheffer", both_polarizations=True)
                    full = r.simulate_scattering(**args)
                    reduced = r.simulate_scattering(**args, use_symmetry=True)
                    for pol in ("x", "y"):
                        for key in ("R", "T", "A", "R0", "T0"):
                            self.assertLess(abs(full[key + "_" + pol] - reduced[key + "_" + pol]), 1e-9)
                    diagnostic = json.loads(reduced["symmetry_diagnostics"])[-1]
                    self.assertTrue(diagnostic["applied"])
                    self.assertEqual(len(diagnostic["blocks"]), 4)
                    self.assertEqual(sum(diagnostic["blocks"]), 2 * 25)
                    self.assertLess(diagnostic["max_invariance_residual"], 1e-8)

    def test_broken_invariance_is_rejected(self):
        sim = r._new_simulation(frequency_thz=370, order=1, period_um=0.7,
            profile="weiss2009", grid=32, dtype=torch.complex128,
            device=torch.device("cpu"), cascade="redheffer", smatrix_size="half",
            use_symmetry=True)
        torch.manual_seed(6)
        matrix = torch.randn((18, 18), dtype=torch.complex128)
        with self.assertRaises(UnsupportedCombinationError):
            sim._group_theory_eigendecomposition(matrix, matrix, [(0.35, 0.35)], 0)


if __name__ == "__main__":
    unittest.main()
