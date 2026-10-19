"""Analytic and independent physical checks for one-dimensional TM scattering.

Run from the repository outputs directory with::

    python -m unittest paper_reproductions.mohamad2020.validation.test_solver -v

Planar expectations below use two-by-two analytic characteristic matrices,
without calling the Fourier solver or its matrix-building helpers.  ``N`` is
the maximum retained order, so its Fourier basis contains ``2*N + 1`` modes.
"""

from __future__ import annotations

import cmath
from dataclasses import dataclass
import math
import unittest

import numpy as np

from ..solver import solve


METHODS = ("rcwa", "rcwa-fff", "dm-fff")


@dataclass(frozen=True)
class Homogeneous:
    period_nm: float = 640.0
    depth_nm: float = 136.0
    n_in: complex = 1.0
    n_out: complex = 1.45
    n_layer: complex = 1.8

    def sample(self, x, z):
        return np.full_like(np.asarray(x, dtype=float), self.n_layer**2,
                            dtype=np.complex128)

    def tangent(self, x):
        x = np.asarray(x, dtype=float)
        return np.ones_like(x), np.zeros_like(x)


@dataclass(frozen=True)
class Layered:
    thicknesses_nm: tuple[float, ...] = (72.0, 44.0)
    indices: tuple[complex, ...] = (1.7, 2.2 + 0.18j)
    period_nm: float = 640.0
    n_in: complex = 1.0
    n_out: complex = 1.45

    @property
    def depth_nm(self):
        return sum(self.thicknesses_nm)

    def sample(self, x, z):
        index = min(int(np.searchsorted(np.cumsum(self.thicknesses_nm),
                                        z, side="right")), len(self.indices) - 1)
        return np.full_like(np.asarray(x, dtype=float), self.indices[index]**2,
                            dtype=np.complex128)

    def tangent(self, x):
        x = np.asarray(x, dtype=float)
        return np.ones_like(x), np.zeros_like(x)


@dataclass(frozen=True)
class Lamellar:
    period_nm: float = 640.0
    depth_nm: float = 112.0
    n_in: complex = 1.0
    n_out: complex = 1.2
    n_ridge: complex = 2.0
    n_groove: complex = 1.0
    fraction: float = 0.375
    shift_nm: float = 0.0

    def sample(self, x, z):
        position = np.mod(np.asarray(x, dtype=float) - self.shift_nm,
                          self.period_nm) / self.period_nm
        return np.where(position < self.fraction,
                        self.n_ridge**2, self.n_groove**2).astype(np.complex128)

    def tangent(self, x):
        # Vertical interfaces: tangent is along z, hence Eq. (11)'s limit.
        x = np.asarray(x, dtype=float)
        return np.zeros_like(x), np.ones_like(x)


def planar_stack(wavelength_nm, angle_deg, n_in, n_out,
                 thicknesses_nm=(), indices=()):
    """Exact h-amplitudes and power for a finite isotropic planar stack.

    The independent two-dimensional state here is [H_y, E_x/Z0].
    Forward-wave E/H is a=q/epsilon, where q=k_z/k_0.  Incidence is from
    a real, lossless medium and the time dependence is exp(-i*omega*t).
    """
    alpha = complex(n_in) * math.sin(math.radians(angle_deg))

    def admittance(n):
        epsilon = complex(n)**2
        q = cmath.sqrt(epsilon - alpha**2)
        if q.imag < -1e-14 or (abs(q.imag) <= 1e-14 and q.real < 0):
            q = -q
        return q, q / epsilon

    _, a_in = admittance(n_in)
    _, a_out = admittance(n_out)
    if len(indices) == 1:
        # Closed-form Fabry--Perot uses only decaying exponentials and remains
        # well conditioned for an optically opaque slab.
        q, a = admittance(indices[0])
        r01, r12 = (a_in - a) / (a_in + a), (a - a_out) / (a + a_out)
        phase = cmath.exp(2j * math.pi * q * thicknesses_nm[0] / wavelength_nm)
        denominator = 1 + r01 * r12 * phase**2
        r = (r01 + r12 * phase**2) / denominator
        t = (1 + r01) * (1 + r12) * phase / denominator
        return r, t, abs(r)**2, a_out.real / a_in.real * abs(t)**2
    transfer = np.eye(2, dtype=np.complex128)
    for thickness, n in zip(thicknesses_nm, indices):
        q, a = admittance(n)
        phase = 2 * math.pi * q * thickness / wavelength_nm
        cosine, sine = cmath.cos(phase), cmath.sin(phase)
        layer = np.array([[cosine, 1j * sine / a],
                          [1j * a * sine, cosine]], dtype=np.complex128)
        transfer = layer @ transfer
    forward = np.array([1.0, a_in], dtype=np.complex128)
    backward = np.array([1.0, -a_in], dtype=np.complex128)
    outgoing = np.array([1.0, a_out], dtype=np.complex128)
    matrix = np.column_stack((transfer @ backward, -outgoing))
    r, t = np.linalg.solve(matrix, -transfer @ forward)
    reflected = abs(r)**2
    transmitted = a_out.real / a_in.real * abs(t)**2
    return r, t, reflected, transmitted


class SolverTests(unittest.TestCase):
    def run_solver(self, geometry, method="rcwa", **kwargs):
        return solve(geometry,
                     wavelength_nm=kwargs.pop("wavelength_nm", 810.0),
                     angle_deg=kwargs.pop("angle_deg", 23.0),
                     max_order=kwargs.pop("max_order", 3),
                     method=method, section_nm=kwargs.pop("section_nm", 1.0),
                     nfft=kwargs.pop("nfft", 1024), **kwargs)

    def assert_power_bookkeeping(self, result):
        for values in (result.R_orders, result.T_orders):
            self.assertTrue(np.all(np.isfinite(values)))
            self.assertGreaterEqual(float(np.min(values)), -1e-11)
        for value in (result.R, result.T, result.A):
            self.assertTrue(np.isfinite(value))
        self.assertAlmostEqual(float(np.sum(result.R_orders)), result.R, delta=1e-11)
        self.assertAlmostEqual(float(np.sum(result.T_orders)), result.T, delta=1e-11)
        self.assertAlmostEqual(result.R + result.T + result.A, 1.0, delta=1e-11)
        zero = int(np.flatnonzero(result.orders == 0)[0])
        self.assertAlmostEqual(result.reflection(0), result.R_orders[zero], delta=1e-13)
        self.assertAlmostEqual(result.transmission(0), result.T_orders[zero], delta=1e-13)

    def assert_planar(self, result, expected, tolerance):
        r, t, reflected, transmitted = expected
        zero = int(np.flatnonzero(result.orders == 0)[0])
        self.assertAlmostEqual(result.r[zero], r, delta=tolerance)
        self.assertAlmostEqual(result.t[zero], t, delta=tolerance)
        self.assertAlmostEqual(result.R, reflected, delta=tolerance)
        self.assertAlmostEqual(result.T, transmitted, delta=tolerance)
        self.assertAlmostEqual(result.reflection(0), reflected, delta=tolerance)
        self.assertAlmostEqual(result.transmission(0), transmitted, delta=tolerance)
        nonzero = result.orders != 0
        np.testing.assert_allclose(result.R_orders[nonzero], 0.0, atol=tolerance)
        np.testing.assert_allclose(result.T_orders[nonzero], 0.0, atol=tolerance)
        self.assert_power_bookkeeping(result)

    def test_zero_depth_interface_matches_oblique_tm_fresnel(self):
        for angle in (0.0, 37.0, math.degrees(math.atan(1.5))):
            geometry = Homogeneous(depth_nm=0.0, n_in=1.0, n_out=1.5)
            expected = planar_stack(810.0, angle, 1.0, 1.5)
            for method in METHODS:
                with self.subTest(angle=angle, method=method):
                    result = self.run_solver(geometry, method, angle_deg=angle)
                    self.assert_planar(result, expected, 2e-10)

    def test_dielectric_slab_matches_fabry_perot_with_unequal_ports(self):
        geometry = Homogeneous(n_layer=1.83, n_in=1.0, n_out=1.45)
        expected = planar_stack(810.0, 23.0, geometry.n_in, geometry.n_out,
                                (geometry.depth_nm,), (geometry.n_layer,))
        for method in METHODS:
            with self.subTest(method=method):
                result = self.run_solver(geometry, method)
                self.assert_planar(result, expected, 3e-8)
                self.assertAlmostEqual(result.R + result.T, 1.0, delta=3e-8)

    def test_absorbing_slab_matches_complex_fresnel_solution(self):
        geometry = Homogeneous(n_layer=1.7 + 0.3j)
        expected = planar_stack(730.0, 31.0, geometry.n_in, geometry.n_out,
                                (geometry.depth_nm,), (geometry.n_layer,))
        for method in METHODS:
            with self.subTest(method=method):
                result = self.run_solver(geometry, method, wavelength_nm=730.0,
                                         angle_deg=31.0)
                self.assert_planar(result, expected, 3e-8)
                self.assertGreater(result.A, 0.0)

    def test_planar_bilayer_matches_independent_characteristic_matrix(self):
        geometry = Layered()
        expected = planar_stack(710.0, 19.0, geometry.n_in, geometry.n_out,
                                geometry.thicknesses_nm, geometry.indices)
        # The z steps coincide exactly with both sharp planar boundaries.
        # RK quadrature of smooth profiles is checked separately below.
        for method in ("rcwa", "rcwa-fff"):
            with self.subTest(method=method):
                result = self.run_solver(geometry, method, wavelength_nm=710.0,
                                         angle_deg=19.0)
                self.assert_planar(result, expected, 2e-9)

    def test_vertical_tangent_fff_recovers_lamellar_rcwa(self):
        geometry = Lamellar()
        reference = self.run_solver(geometry, "rcwa", section_nm=0.5)
        for method in ("rcwa-fff", "dm-fff"):
            with self.subTest(method=method):
                result = self.run_solver(geometry, method, section_nm=0.5)
                np.testing.assert_allclose(result.R_orders, reference.R_orders,
                                           rtol=2e-7, atol=2e-8)
                np.testing.assert_allclose(result.T_orders, reference.T_orders,
                                           rtol=2e-7, atol=2e-8)
                self.assert_power_bookkeeping(result)

    def test_lamellar_rcwa_matches_independent_vallius_solver(self):
        from ...vallius2002.solver import LayerSpec, PreparedStack

        geometry = Lamellar(n_ridge=1.3 + 7.6j, depth_nm=80.0)
        layer = LayerSpec(geometry.depth_nm,
                          (0.0, geometry.fraction * geometry.period_nm,
                           geometry.period_nm),
                          (geometry.n_ridge**2, geometry.n_groove**2),
                          period=geometry.period_nm)
        reference = PreparedStack([layer], polarization="TM", method="fmm",
                                  harmonics=7, epsilon_in=geometry.n_in**2,
                                  epsilon_out=geometry.n_out**2,
                                  angle=math.radians(23.0), device="cpu",
                                  backend="scipy", diagnostics=False).solve(810.0)
        result = self.run_solver(geometry, "rcwa", nfft=16384)
        # Vallius uses exact lamellar coefficients; this solver samples by FFT.
        self.assertAlmostEqual(result.R, reference["R"], delta=5e-5)
        self.assertAlmostEqual(result.T, reference["T"], delta=5e-5)
        np.testing.assert_allclose(result.R_orders, reference["R_orders"], atol=5e-5)
        np.testing.assert_allclose(result.T_orders, reference["T_orders"], atol=5e-5)

    def test_lossless_lamellar_conserves_propagating_power(self):
        geometry = Lamellar()
        for method in METHODS:
            with self.subTest(method=method):
                result = self.run_solver(geometry, method)
                self.assert_power_bookkeeping(result)
                self.assertAlmostEqual(result.R + result.T, 1.0, delta=2e-7)
                alpha = (geometry.n_in * math.sin(math.radians(23.0))
                         + result.orders * 810.0 / geometry.period_nm)
                for power, n in ((result.R_orders, geometry.n_in),
                                 (result.T_orders, geometry.n_out)):
                    evanescent = abs(alpha) > n
                    np.testing.assert_allclose(power[evanescent], 0.0, atol=1e-13)

    def test_passive_lamellar_metal_has_nonnegative_absorption(self):
        geometry = Lamellar(n_ridge=1.3 + 7.6j, depth_nm=96.0)
        for method in METHODS:
            with self.subTest(method=method):
                result = self.run_solver(geometry, method)
                self.assert_power_bookkeeping(result)
                self.assertGreaterEqual(result.A, -2e-7)
                self.assertLessEqual(result.R + result.T, 1.0 + 2e-7)

    def test_periodic_translation_preserves_diffraction_powers(self):
        original = Lamellar()
        shifted = Lamellar(shift_nm=original.period_nm / 8)
        for method in METHODS:
            with self.subTest(method=method):
                first = self.run_solver(original, method)
                second = self.run_solver(shifted, method)
                np.testing.assert_allclose(first.R_orders, second.R_orders,
                                           rtol=2e-8, atol=2e-9)
                np.testing.assert_allclose(first.T_orders, second.T_orders,
                                           rtol=2e-8, atol=2e-9)

    def test_common_length_rescaling_preserves_scattering(self):
        original = Lamellar()
        scale = 2.5
        scaled = Lamellar(period_nm=original.period_nm * scale,
                          depth_nm=original.depth_nm * scale)
        for method in METHODS:
            with self.subTest(method=method):
                first = self.run_solver(original, method)
                second = self.run_solver(scaled, method,
                                         wavelength_nm=810.0 * scale,
                                         section_nm=scale)
                np.testing.assert_allclose(first.R_orders, second.R_orders,
                                           rtol=2e-8, atol=2e-9)
                np.testing.assert_allclose(first.T_orders, second.T_orders,
                                           rtol=2e-8, atol=2e-9)

    def test_dm_refinement_reduces_error_against_analytic_slab(self):
        geometry = Homogeneous(depth_nm=240.0, n_layer=2.4 + 0.2j)
        expected = planar_stack(450.0, 23.0, geometry.n_in, geometry.n_out,
                                (geometry.depth_nm,), (geometry.n_layer,))
        errors = []
        for step in (20.0, 10.0):
            result = self.run_solver(geometry, "dm-fff", wavelength_nm=450.0,
                                     max_order=0, section_nm=step, dm_substeps=4)
            errors.append(max(abs(result.r[0] - expected[0]),
                              abs(result.t[0] - expected[1])))
        self.assertGreater(errors[0], 1e-12)
        self.assertLess(errors[1], errors[0] / 6)
        self.assertLess(errors[1], 0.02)

    def test_thick_absorbing_slab_does_not_overflow_evanescent_propagation(self):
        geometry = Homogeneous(depth_nm=2400.0, n_layer=1.3 + 7.6j,
                               n_out=1.0)
        expected = planar_stack(810.0, 23.0, 1.0, 1.0,
                                (geometry.depth_nm,), (geometry.n_layer,))
        for method in METHODS:
            with self.subTest(method=method):
                result = self.run_solver(geometry, method, max_order=1,
                                         section_nm=8.0)
                self.assert_planar(result, expected, 2e-6)

    def test_stored_fields_match_boundary_conditions_and_analytic_slab_fields(self):
        geometry = Homogeneous(depth_nm=48.0, n_layer=1.7 + 0.3j)
        wavelength, angle = 730.0, 31.0
        expected_r, _, _, _ = planar_stack(
            wavelength, angle, geometry.n_in, geometry.n_out,
            (geometry.depth_nm,), (geometry.n_layer,))
        alpha0 = geometry.n_in * math.sin(math.radians(angle))
        q_in = cmath.sqrt(geometry.n_in**2 - alpha0**2)
        q_layer = cmath.sqrt(geometry.n_layer**2 - alpha0**2)
        v_in = q_in / geometry.n_in**2
        v_layer = q_layer / geometry.n_layer**2
        h_at_zero = 1 + expected_r
        e_at_zero = v_in * (1 - expected_r)
        forward = 0.5 * (h_at_zero + e_at_zero / v_layer)
        backward = 0.5 * (h_at_zero - e_at_zero / v_layer)
        for method in METHODS:
            with self.subTest(method=method):
                result = self.run_solver(geometry, method, wavelength_nm=wavelength,
                                         angle_deg=angle, max_order=2, nfft=256,
                                         section_nm=3.0, store_fields=True)
                fields = result.fields
                self.assertIsNotNone(fields)
                self.assertEqual(fields["z_nm"][0], 0.0)
                self.assertEqual(fields["z_nm"][-1], geometry.depth_nm)
                self.assertEqual(fields["Ex_coeff"].shape,
                                 (len(fields["z_nm"]), len(result.orders)))
                self.assertEqual(fields["h_coeff"].shape, fields["Ex_coeff"].shape)
                alpha = alpha0 + result.orders * wavelength / geometry.period_nm
                vi = np.sqrt(geometry.n_in**2 - alpha**2 + 0j) / geometry.n_in**2
                vo = np.sqrt(geometry.n_out**2 - alpha**2 + 0j) / geometry.n_out**2
                incident = np.asarray(result.orders == 0, dtype=complex)
                np.testing.assert_allclose(fields["h_coeff"][0], incident + result.r,
                                           atol=2e-9)
                np.testing.assert_allclose(fields["Ex_coeff"][0], vi * (incident - result.r),
                                           atol=2e-9)
                np.testing.assert_allclose(fields["h_coeff"][-1], result.t, atol=2e-9)
                np.testing.assert_allclose(fields["Ex_coeff"][-1], vo * result.t, atol=2e-9)
                phase = np.exp(2j * math.pi * q_layer * fields["z_nm"] / wavelength)
                expected_h = forward * phase + backward / phase
                expected_e = v_layer * (forward * phase - backward / phase)
                zero = int(np.flatnonzero(result.orders == 0)[0])
                np.testing.assert_allclose(fields["h_coeff"][:, zero], expected_h,
                                           rtol=2e-8, atol=2e-9)
                np.testing.assert_allclose(fields["Ex_coeff"][:, zero], expected_e,
                                           rtol=2e-8, atol=2e-9)


if __name__ == "__main__":
    unittest.main()
