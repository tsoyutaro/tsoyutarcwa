"""Physical/amplitude parity and matrix-size regressions for TE/TM streaming."""
from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

from common import HERE, load_config

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch unavailable; run in the RCWA environment")
class PolarizationSeparationChecks(unittest.TestCase):
    def setUp(self):
        from solver import Grating1D, _simulation, port_results, simulate
        self.grating = Grating1D
        self.simulation = _simulation
        self.ports = port_results
        self.simulate = simulate
        self.config, self.model = load_config(HERE/"config.json")

    def make_simulation(self, *, separated=True, size="half", fields="none", eps=None):
        from rcwa_ext.config import GroupTheoryOptions, Lattice, OutputSpec
        sim = self.grating(freq=200/550, order=[4, 0],
                           lattice=Lattice.square(1), cascade="redheffer",
                           outputs=OutputSpec(smatrix_size=size, fields=fields),
                           group_theory=GroupTheoryOptions(enabled=False),
                           polarization_separated=separated,
                           dtype=torch.complex128, device="cpu")
        sim.add_input_layer(eps=1, mu=1)
        sim.add_output_layer(eps=self.model(550) if eps is None else eps, mu=1)
        sim.set_incident_angle(0, 0)
        return sim

    def assert_ports_close(self, left, right, tolerance=1e-9):
        for pol in ("TE", "TM"):
            for name in left[pol]:
                self.assertLessEqual(abs(left[pol][name]-right[pol][name]), tolerance,
                                     (pol, name, left[pol][name], right[pol][name]))

    def test_trapezoid_observables_match_dense_at_all_configured_wavelengths(self):
        numbers = {"order": 8, "slices": 20, "grid": 192}
        with torch.no_grad():
            for analytic in (False, True):
                for wave in self.config["wavelengths_nm"]:
                    with self.subTest(analytic=analytic, wavelength=wave):
                        config = copy.deepcopy(self.config)
                        config["solver"]["fourier_coefficients"] = "analytic" if analytic else "sampled"
                        config["solver"]["polarization_separated"] = True
                        reduced = self.simulate(config, numbers, wave, self.model(wave), "cpu")
                        config["solver"]["polarization_separated"] = False
                        dense = self.simulate(config, numbers, wave, self.model(wave), "cpu")
                        self.assert_ports_close(reduced["polarizations"], dense["polarizations"])
                        self.assertTrue(reduced["polarization_separated"])
                        self.assertFalse(dense["polarization_separated"])
                        for audit in reduced["selected_layer_diagnostics"]:
                            self.assertLess(audit["relative_eigen_residual"], 1e-11)

    def test_all_requested_smatrix_blocks_match_dense(self):
        with torch.no_grad():
            for size in ("full", "half", "quarter"):
                with self.subTest(size=size):
                    reduced = self.make_simulation(size=size)
                    dense = self.make_simulation(size=size, separated=False)
                    for sim in (reduced, dense):
                        for fill, thickness in ((.05, 0), (.23, .17), (.65, .21), (.95, .09)):
                            sim.add_ridge(thickness, fill, self.model(550), 192, analytic=True)
                        sim.solve_global_smatrix()
                    self.assertIsNone(reduced._separated_public_s)
                    for actual, expected in zip(reduced.S, dense.S):
                        torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)
                    n = reduced.order_N
                    for block in reduced.S:
                        self.assertEqual(float(block[:n, n:].abs().max()), 0)
                        self.assertEqual(float(block[n:, :n].abs().max()), 0)

    def test_empty_homogeneous_and_mixed_stacks_match_dense(self):
        with torch.no_grad():
            for layers in ((), ("gold",), ("air", "ridge", "gold", "ridge")):
                with self.subTest(layers=layers):
                    reduced = self.make_simulation()
                    dense = self.make_simulation(separated=False)
                    for sim in (reduced, dense):
                        for layer in layers:
                            if layer == "ridge":
                                sim.add_ridge(.07, .41, self.model(550), 192, analytic=True)
                            else:
                                sim.add_layer(.15, eps=self.model(550) if layer == "gold" else 1)
                    self.assert_ports_close(self.ports(reduced), self.ports(dense))
                    self.assertIsNone(reduced._separated_public_s)

    def test_lossless_flux_and_finite_lossy_layer(self):
        with torch.no_grad():
            sim = self.make_simulation(eps=2.25)
            for fill in (.15, .4, .7, .9):
                sim.add_ridge(.25, fill, 2.25, 192, analytic=True)
            for pol in self.ports(sim).values():
                self.assertLess(abs(1-pol["reflectance"]-pol["power_into_substrate"]), 1e-10)
            dense = self.make_simulation(eps=2.25, separated=False)
            reduced = self.make_simulation(eps=2.25)
            for current in (dense, reduced):
                current.add_layer(.15, eps=self.model(550), mu=1.3)
            self.assert_ports_close(self.ports(reduced), self.ports(dense))

    def test_eigensolves_and_cascades_remain_n_by_n_and_release_layers(self):
        with torch.no_grad():
            sim = self.simulation(12, 550, 200, self.model(550), "cpu")
            with patch.object(sim, "_eig", wraps=sim._eig) as eig, \
                 patch.object(sim, "_solve", wraps=sim._solve) as solve:
                for fill in (.05, .5, .95):
                    sim.add_ridge(.1, fill, self.model(550), 192, analytic=True, audit=True)
                self.ports(sim)
                for call in eig.call_args_list+solve.call_args_list:
                    self.assertEqual(call.args[0].shape[-2:], (sim.order_N, sim.order_N))
            self.assertIsNone(sim._separated_public_s)
            for name in ("P", "Q", "E_eigvec", "H_eigvec", "kz_norm", "eps_conv", "mu_conv",
                         "layer_S11", "layer_S12", "layer_S21", "layer_S22"):
                self.assertEqual(len(getattr(sim, name)), 0, name)
            self.assertEqual(len(sim.thickness), 3)
            for block in sim._separated_stack_s:
                self.assertEqual(block.shape, (2, sim.order_N, sim.order_N))

    def test_last_operators_match_dense_without_retaining_cartesian_modes(self):
        with torch.no_grad():
            reduced = self.make_simulation()
            dense = self.make_simulation(separated=False)
            for sim in (reduced, dense):
                sim.add_ridge(.1, .37, self.model(550), 192, analytic=True)
            for actual, expected in zip(reduced.cartesian_operators(), dense.cartesian_operators()):
                torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-13)

    def test_fields_use_existing_cartesian_path_and_solver_cache_is_invalidated(self):
        with torch.no_grad():
            fields = self.make_simulation(size="full", fields="all")
            self.assertFalse(fields._separated_enabled)
            fields.add_ridge(.1, .3, self.model(550), 192, analytic=True)
            self.assertEqual(fields.E_eigvec[-1].shape, (18, 18))
            sim = self.make_simulation()
            sim.add_ridge(.1, .3, self.model(550), 192, analytic=True)
            sim.solve_global_smatrix()
            first = sim.S
            sim.add_ridge(.1, .7, self.model(550), 192, analytic=True)
            self.assertIsNone(sim._separated_global_s)
            with self.assertRaises(AttributeError):
                _ = sim.S
            sim.solve_global_smatrix()
            self.assertIsNot(sim.S, first)

    def test_default_free_space_ports_and_raster_fallback(self):
        from rcwa_ext.config import Lattice, OutputSpec
        with torch.no_grad():
            simulations = [self.grating(freq=200/550, order=[3, 0],
                                        lattice=Lattice.square(1),
                                        outputs=OutputSpec(smatrix_size="half", fields="none"),
                                        polarization_separated=separated,
                                        dtype=torch.complex128, device="cpu")
                           for separated in (True, False)]
            for sim in simulations:
                sim.set_incident_angle(0, 0)
                sim.add_ridge(.1, .37, self.model(550), 192, analytic=True)
                sim.solve_global_smatrix()
            for actual, expected in zip(simulations[0].S, simulations[1].S):
                torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)
            sim = self.make_simulation()
            sim.solve_global_smatrix()
            material = torch.full((64, 1), 2.25, dtype=torch.complex128)
            sim.add_layer(.1, eps=material)
            self.assertFalse(sim._separated_enabled)
            self.assertIsNone(sim._separated_global_s)
            sim.solve_global_smatrix()
            self.assertEqual(sim.S[0].shape, (18, 18))
            with self.assertRaisesRegex(RuntimeError, "Cartesian"):
                sim.solve_polarized_smatrix()


if __name__ == "__main__":
    unittest.main()
