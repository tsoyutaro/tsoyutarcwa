"""Regression checks for stiff propagation, varying media, and run status."""
from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.integrate import solve_ivp

from .. import reproduce
from ..solver import solve
from .test_solver import Homogeneous, Layered, planar_stack


class SmoothLayer(Homogeneous):
    def sample(self, x, z):
        n = 1.6 + .5 * np.sin(2 * np.pi * z / self.depth_nm) + .03j
        return np.full(np.asarray(x).shape, n * n, dtype=complex)


class ThinLayer(Layered):
    def constant_slice_boundaries(self, x):
        return np.cumsum(self.thicknesses_nm)[:-1]


def independent_smooth_scattering(geometry, wavelength, angle):
    # N=0 Maxwell ODE directly, integrated independently by DOP853. This uses
    # neither the solver's Fourier generator nor its scattering conversion.
    alpha = geometry.n_in * np.sin(np.deg2rad(angle))
    k0 = 2 * np.pi / wavelength

    def rhs(z, state):
        eps = geometry.sample(np.array([0.]), z)[0]
        generator = 1j * k0 * np.array([[0, 1 - alpha * alpha / eps], [eps, 0]])
        return (generator @ state.reshape(2, 2)).ravel()

    p = solve_ivp(rhs, (0, geometry.depth_nm), np.eye(2, dtype=complex).ravel(),
                  method="DOP853", rtol=2e-13, atol=2e-14).y[:, -1].reshape(2, 2)
    vi = np.sqrt(geometry.n_in**2 - alpha**2 + 0j) / geometry.n_in**2
    vo = np.sqrt(geometry.n_out**2 - alpha**2 + 0j) / geometry.n_out**2
    forward, backward = np.array([vi, 1]), np.array([-vi, 1])
    outgoing = np.array([vo, 1])
    return np.linalg.solve(np.column_stack((p @ backward, -outgoing)), -p @ forward)


class IntegrationTests(unittest.TestCase):
    def test_material_events_resolve_layer_missed_by_trial_quadrature(self):
        geometry = ThinLayer(thicknesses_nm=(12., .05, 3.95), indices=(1.2, 3. + 2j, 1.2))
        exact = planar_stack(450., 23., geometry.n_in, geometry.n_out,
                             geometry.thicknesses_nm, geometry.indices)
        result = solve(geometry, 450., 23., 0, section_nm=16., dm_substeps=1, nfft=32)
        np.testing.assert_allclose([result.r[0], result.t[0]], exact[:2], atol=2e-11)
        self.assertEqual(result.metadata["integration_diagnostics"]["constant_slice_steps"], 3)

    def test_stiff_lossless_slab_matches_analytic_amplitudes_and_passivity(self):
        geometry = Homogeneous(n_layer=20., depth_nm=5., n_out=1.)
        exact = planar_stack(40., 23., 1., 1., (5.,), (20.,))
        result = solve(geometry, 40., 23., 0, section_nm=5., dm_substeps=1, nfft=32)
        np.testing.assert_allclose([result.r[0], result.t[0]], exact[:2], atol=2e-10)
        self.assertAlmostEqual(result.R + result.T, 1., delta=2e-10)
        legacy = solve(geometry, 40., 23., 0, section_nm=5., dm_substeps=1,
                       nfft=32, dm_integrator="rk4")
        self.assertGreater(abs(legacy.R + legacy.T - 1), .01)

    def test_adaptive_magnus_matches_independent_noncommuting_ode(self):
        geometry = SmoothLayer(depth_nm=136.)
        exact = independent_smooth_scattering(geometry, 450., 31.)
        results = []
        for tolerance in (1e-5, 1e-7):
            result = solve(geometry, 450., 31., 0, section_nm=34., dm_substeps=1,
                           nfft=32, dm_rtol=tolerance, dm_atol=tolerance * .001)
            results.append(result)
        errors = [np.max(np.abs(np.array([r.r[0], r.t[0]]) - exact)) for r in results]
        self.assertLess(errors[1], 2e-7)
        self.assertLess(errors[1], errors[0] / 4)
        stats = results[1].metadata["integration_diagnostics"]
        self.assertGreater(stats["rejected_steps"], 0)
        self.assertLessEqual(stats["max_accepted_error_ratio"], 1.)
        self.assertGreater(stats["accepted_steps"], 4)

    def test_cf4_has_fourth_order_for_varying_noncommuting_ode(self):
        geometry = SmoothLayer(depth_nm=136.)
        exact = independent_smooth_scattering(geometry, 450., 31.)
        errors = []
        for substeps in (1, 2):
            result = solve(geometry, 450., 31., 0, section_nm=17., dm_substeps=substeps,
                           nfft=32, dm_integrator="cf4")
            errors.append(np.max(np.abs(np.array([result.r[0], result.t[0]]) - exact)))
        self.assertLess(errors[1], errors[0] / 10)

    def test_unphysical_points_produce_nonzero_exit_status(self):
        class Runner:
            def __init__(self, args):
                self.out, self.failed, self.records = args.output_dir, 0, []

            def smoke(self):
                self.records.append({"status": "UNPHYSICAL"})

            def save_metadata(self):
                pass

        with patch.object(reproduce, "Runner", Runner), redirect_stdout(io.StringIO()):
            self.assertEqual(reproduce.main(["--study", "smoke"]), 3)

    def test_existing_results_are_preserved_without_explicit_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            sentinel = path / "settings.json"
            sentinel.write_text("original", encoding="utf-8")
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    reproduce.main(["--study", "smoke", "--output-dir", str(path)])
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "original")


if __name__ == "__main__":
    unittest.main()
