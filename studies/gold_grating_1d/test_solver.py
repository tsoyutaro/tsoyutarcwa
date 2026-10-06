"""Regression for high-order false Ky rejection and actual nonzero Ky."""
from __future__ import annotations

import math
import unittest
from common import HERE, load_config
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch unavailable; run in the RCWA environment")
class WavevectorChecks(unittest.TestCase):
    def setUp(self):
        from solver import _simulation
        self.simulation = _simulation
        self.config, self.model = load_config(HERE/"config.json")

    def test_pre_fix_failure_is_reproduced_then_exact_orthogonal_cell_passes(self):
        from rcwa_ext.auto import AutoRCWA
        with torch.no_grad():
            sim = self.simulation(48, 700, 200, self.model(700), "cpu")
            sim.cos_zeta, sim.sin_zeta = math.cos(math.pi/2), math.sin(math.pi/2)
            AutoRCWA._kvectors(sim)
            self.assertGreater(float(torch.max(torch.abs(sim.Ky_norm))), 1e-14)
            with self.assertRaisesRegex(ValueError, "K_y=0"):
                sim.add_ridge(0.01, 0.2, self.model(700), 576, analytic=True)
            sim._kvectors()
            self.assertEqual(float(torch.max(torch.abs(sim.Ky_norm))), 0.0)
            sim.add_ridge(0.01, 0.2, self.model(700), 576, analytic=True)
            self.assertEqual(sim.layer_N, 1)

    def test_exact_ky_and_decoupled_ports_at_larger_order(self):
        with torch.no_grad():
            sim = self.simulation(56, 700, 200, self.model(700), "cpu")
            self.assertEqual(float(torch.max(torch.abs(sim.Ky_norm_dn))), 0.0)
            n = sim.order_N
            for port in (sim.Vf, sim.Vi, sim.Vo):
                self.assertEqual(float(torch.max(torch.abs(port[:n, :n]))), 0.0)
                self.assertEqual(float(torch.max(torch.abs(port[n:, n:]))), 0.0)

    def test_actual_nonzero_ky_still_rejected(self):
        with torch.no_grad():
            sim = self.simulation(3, 700, 200, self.model(700), "cpu")
            sim.set_incident_angle(0.1, 0.5)
            with self.assertRaisesRegex(ValueError, "K_y=0"):
                sim.add_ridge(0.01, 0.2, self.model(700), 576, analytic=True)


if __name__ == "__main__":
    unittest.main()
