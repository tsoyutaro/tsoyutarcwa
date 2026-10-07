"""Independent physical checks for the 1-D Vallius reproduction solver.

Run from the repository outputs directory with
``python -m unittest paper_reproductions.vallius2002.validation.test_solver``.
Fabry--Perot expectations use closed-form Fresnel amplitudes and do not call
the Fourier solver. ASR truncation is checked separately from exact FMM flux.
"""

from __future__ import annotations

import cmath
import math
import unittest
from functools import partial

import numpy as np

from ..solver import LayerSpec, PreparedStack

# This suite is the independent SciPy baseline even on GPU-equipped hosts.
PreparedStack = partial(PreparedStack, device="cpu", backend="scipy")


def slab_power(wavelength, thickness, epsilon, epsilon_in=1.0, epsilon_out=1.0):
    """Normal-incidence isotropic slab power for exp(-i omega t)."""
    n0, n1, n2 = map(cmath.sqrt, (epsilon_in, epsilon, epsilon_out))
    r01, r12 = (n0 - n1) / (n0 + n1), (n1 - n2) / (n1 + n2)
    phase = cmath.exp(2j * math.pi * n1 * thickness / wavelength)
    denominator = 1 + r01 * r12 * phase**2
    reflection = (r01 + r12 * phase**2) / denominator
    transmission = 4 * n0 * n1 * phase / ((n0 + n1) * (n1 + n2) * denominator)
    reflected = abs(reflection)**2
    transmitted = (n2.real / n0.real) * abs(transmission)**2
    return transmitted, reflected, 1 - transmitted - reflected


class PhysicalSolverTests(unittest.TestCase):
    def solve(self, layers, polarization, wavelength, **kwargs):
        return PreparedStack(layers, polarization=polarization, method="fmm",
                             harmonics=kwargs.pop("harmonics", 15), **kwargs).solve(wavelength)

    def assert_power_finite(self, result):
        for key in ("T0", "T", "R", "A"):
            self.assertTrue(np.isfinite(result[key]), key)
        for key in ("T_orders", "R_orders"):
            self.assertTrue(np.all(np.isfinite(result[key])), key)
            self.assertGreaterEqual(np.min(result[key]), -1e-12, key)
        self.assertAlmostEqual(float(np.sum(result["T_orders"])), result["T"], delta=1e-11)
        self.assertAlmostEqual(float(np.sum(result["R_orders"])), result["R"], delta=1e-11)

    def test_homogeneous_slab_matches_independent_fabry_perot(self):
        for polarization in ("TE", "TM"):
            for count in (1, 5, 8):
                for wavelength in (0.73, 1.17):
                    with self.subTest(polarization=polarization, harmonics=count, wavelength=wavelength):
                        thickness = 0.237
                        layer = LayerSpec(thickness, (0.0, 1.0), (4.0,))
                        result = self.solve([layer], polarization, wavelength, harmonics=count)
                        expected_t, expected_r, expected_a = slab_power(wavelength, thickness, 4.0)
                        self.assertAlmostEqual(result["T0"], expected_t, delta=2e-10)
                        self.assertAlmostEqual(result["R"], expected_r, delta=2e-10)
                        self.assertAlmostEqual(result["A"], expected_a, delta=2e-10)
                        self.assert_power_finite(result)

    def test_unequal_exterior_media_have_correct_transmitted_flux(self):
        # The n_out/n_in Fresnel factor catches accidental amplitude-squared
        # reporting when the incident and transmitted materials differ.
        layer = LayerSpec(0.18, (0.0, 1.0), (6.25,))
        for polarization in ("TE", "TM"):
            with self.subTest(polarization=polarization):
                result = self.solve([layer], polarization, 0.93,
                                    epsilon_in=1.0, epsilon_out=2.25)
                expected_t, expected_r, _ = slab_power(0.93, 0.18, 6.25, 1.0, 2.25)
                self.assertAlmostEqual(result["T"], expected_t, delta=2e-10)
                self.assertAlmostEqual(result["R"], expected_r, delta=2e-10)
                self.assertAlmostEqual(result["T"] + result["R"], 1.0, delta=2e-10)

    def test_passive_slab_absorption_matches_fresnel_solution(self):
        epsilon = 3.2 + 0.8j
        layer = LayerSpec(0.16, (0.0, 1.0), (epsilon,))
        expected_t, expected_r, expected_a = slab_power(0.91, 0.16, epsilon)
        for polarization in ("TE", "TM"):
            with self.subTest(polarization=polarization):
                result = self.solve([layer], polarization, 0.91)
                self.assertAlmostEqual(result["T"], expected_t, delta=2e-10)
                self.assertAlmostEqual(result["R"], expected_r, delta=2e-10)
                self.assertAlmostEqual(result["A"], expected_a, delta=2e-10)
                self.assertGreater(result["A"], 0)

    def test_lossless_fmm_conserves_all_propagating_orders(self):
        layers = [LayerSpec(0.21, (0.0, 0.37, 1.0), (4.0, 1.0)),
                  LayerSpec(0.13, (0.0, 0.19, 0.64, 1.0), (1.0, 2.25, 1.0))]
        for polarization in ("TE", "TM"):
            for wavelength in (0.72, 1.21):
                with self.subTest(polarization=polarization, wavelength=wavelength):
                    result = self.solve(layers, polarization, wavelength, harmonics=17)
                    self.assert_power_finite(result)
                    self.assertAlmostEqual(result["T"] + result["R"], 1.0, delta=2e-9)
                    self.assertGreaterEqual(result["T"] + 1e-12, result["T0"])

    def test_fmm_power_is_invariant_under_periodic_cell_translation(self):
        original = LayerSpec(0.29, (0.0, 0.3, 1.0), (4.0, 1.0))
        translated = LayerSpec(0.29, (0.0, 0.2, 0.5, 1.0), (1.0, 4.0, 1.0))
        for polarization in ("TE", "TM"):
            for wavelength in (0.83, 1.13):
                with self.subTest(polarization=polarization, wavelength=wavelength):
                    first = self.solve([original], polarization, wavelength)
                    second = self.solve([translated], polarization, wavelength)
                    for key in ("T0", "T", "R"):
                        self.assertAlmostEqual(first[key], second[key], delta=2e-9, msg=key)

    def test_normalized_spectrum_is_invariant_under_length_rescaling(self):
        scale = 3.2
        original = LayerSpec(0.23, (0.0, 0.4, 1.0), (4.0, 1.0))
        rescaled = LayerSpec(0.23 * scale, (0.0, 0.4 * scale, scale),
                            (4.0, 1.0), period=scale)
        for polarization in ("TE", "TM"):
            first = self.solve([original], polarization, 0.89)
            second = self.solve([rescaled], polarization, 0.89 * scale)
            for key in ("T0", "T", "R"):
                self.assertAlmostEqual(first[key], second[key], delta=2e-9, msg=key)

    def test_equal_regions_with_unit_slope_recover_identity_map_and_fmm(self):
        layer = LayerSpec(0.23, (0.0, 0.5, 1.0), (4.0, 1.0))
        for polarization in ("TE", "TM"):
            with self.subTest(polarization=polarization):
                adaptive = PreparedStack([layer], polarization, "asr", 15,
                                         oversampling=1, G=1.0, quadrature=128)
                mapping = adaptive.prepared_layers[0]
                np.testing.assert_allclose(mapping.x, mapping.u, rtol=0, atol=2e-15)
                np.testing.assert_allclose(mapping.resolution, 1.0, rtol=0, atol=2e-15)
                first = self.solve([layer], polarization, 0.87)
                second = adaptive.solve(0.87)
                for key in ("T0", "T", "R"):
                    self.assertAlmostEqual(first[key], second[key], delta=2e-10, msg=key)

    def test_adaptive_map_preserves_region_widths_and_joins_continuously(self):
        G, points = 0.001, 192
        layer = LayerSpec(0.23, (0.0, 0.3, 1.0), (4.0, 1.0))
        stack = PreparedStack([layer], "TE", "asr", 7, quadrature=points, G=G)
        mapping = stack.prepared_layers[0]
        self.assertTrue(np.all(np.diff(mapping.x) > 0))
        self.assertTrue(np.all(mapping.resolution > 0))
        self.assertAlmostEqual(float(mapping.weights.sum()), 1.0, delta=2e-14)
        for region, (left, right) in enumerate(((0.0, 0.3), (0.3, 1.0))):
            block = slice(region * points, (region + 1) * points)
            integral = float(np.sum(mapping.weights[block] * mapping.resolution[block]))
            self.assertAlmostEqual(integral, right - left, delta=2e-13)
            # Gauss nodes exclude the endpoints. Their nearest values are
            # extrapolated by the specified common interface slope G; the
            # next error is cubic in the tiny distance to the interface.
            near_left, near_right = region * points, (region + 1) * points - 1
            u_left, u_right = region / 2, (region + 1) / 2
            x_left = mapping.x[near_left] - G * (mapping.u[near_left] - u_left)
            x_right = mapping.x[near_right] + G * (u_right - mapping.u[near_right])
            self.assertAlmostEqual(float(x_left), left, delta=2e-12)
            self.assertAlmostEqual(float(x_right), right, delta=2e-12)
            self.assertAlmostEqual(float(mapping.resolution[near_left]), G, delta=1e-6)
            self.assertAlmostEqual(float(mapping.resolution[near_right]), G, delta=1e-6)

    def test_asr_cyclic_seam_merge_preserves_translated_spectrum(self):
        original = LayerSpec(0.29, (0.0, 0.3, 1.0), (4.0, 1.0))
        translated = LayerSpec(0.29, (0.0, 0.2, 0.5, 1.0), (1.0, 4.0, 1.0))
        for polarization in ("TE", "TM"):
            with self.subTest(polarization=polarization):
                first = PreparedStack([original], polarization, "asr", 15, quadrature=128).solve(0.87)
                second = PreparedStack([translated], polarization, "asr", 15, quadrature=128).solve(0.87)
                for key in ("T0", "T", "R"):
                    self.assertAlmostEqual(first[key], second[key], delta=2e-9, msg=key)

    def test_asr_projection_error_decreases_with_retained_modes(self):
        # Raw x-space boundary projections are finite truncations. Their
        # lossless flux error need not be zero at low N, but must converge.
        layer = LayerSpec(0.23, (0.0, 0.3, 1.0), (4.0, 1.0))
        for polarization in ("TE", "TM"):
            with self.subTest(polarization=polarization):
                reference = self.solve([layer], polarization, 0.87, harmonics=101, diagnostics=False)
                coarse = PreparedStack([layer], polarization, "asr", 5,
                                       quadrature=128, diagnostics=False).solve(0.87)
                fine = PreparedStack([layer], polarization, "asr", 25,
                                     quadrature=128, diagnostics=False).solve(0.87)
                self.assert_power_finite(coarse)
                self.assert_power_finite(fine)
                coarse_error = abs(coarse["T0"] - reference["T0"])
                fine_error = abs(fine["T0"] - reference["T0"])
                self.assertLess(fine_error, coarse_error / 5)
                coarse_flux = abs(coarse["T"] + coarse["R"] - 1)
                fine_flux = abs(fine["T"] + fine["R"] - 1)
                self.assertLess(fine_flux, coarse_flux / 5)


if __name__ == "__main__":
    unittest.main()
