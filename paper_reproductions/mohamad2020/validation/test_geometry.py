"""GIL endpoint checks from paper Fig.7b,c, independent of propagation."""
import unittest
import numpy as np

from ..geometries import triangular_gold


class GeometryTests(unittest.TestCase):
    def test_gil_is_centered_and_uses_each_interfaces_dielectric(self):
        geometry = triangular_gold(gil_nm=5.)
        slope = 2 * geometry.height_nm / geometry.period_nm
        air_edge, substrate_edge = 50 / slope, 100 / slope
        x = np.array([air_edge - 3., air_edge - 1.25, air_edge,
                      air_edge + 1.25, air_edge + 3., substrate_edge - 1.25,
                      substrate_edge, substrate_edge + 1.25, substrate_edge + 3.])
        expected_n = np.array([geometry.n_out, .75 * geometry.n_out + .25 * geometry.n_metal,
                               .5 * (geometry.n_out + geometry.n_metal),
                               .25 * geometry.n_out + .75 * geometry.n_metal,
                               geometry.n_metal, .25 * geometry.n_in + .75 * geometry.n_metal,
                               .5 * (geometry.n_in + geometry.n_metal),
                               .75 * geometry.n_in + .25 * geometry.n_metal, geometry.n_in])
        np.testing.assert_allclose(geometry.sample(x, 100.), expected_n**2, atol=1e-12)
        np.testing.assert_allclose(geometry.sample(geometry.period_nm - x, 100.), expected_n**2, atol=1e-12)

    def test_absent_intersection_does_not_invent_gil_at_valley_or_apex(self):
        geometry = triangular_gold(gil_nm=5.)
        self.assertAlmostEqual(geometry.sample(np.array([0.]), 25.)[0], geometry.n_metal**2)
        self.assertAlmostEqual(geometry.sample(np.array([200.]), geometry.height_nm + 25.)[0], geometry.n_metal**2)

    def test_linear_index_gil_is_locally_passive_and_periodic(self):
        geometry = triangular_gold(gil_nm=5.)
        x = np.linspace(0, 400, 2048, endpoint=False)
        for z in (1., 25., 49.9, 50., 100., 250., 283.2, 300., 332.):
            eps = geometry.sample(x, z)
            self.assertGreaterEqual(float(eps.imag.min()), -1e-13)
            np.testing.assert_allclose(geometry.sample(x + 400, z), eps, atol=1e-11)

    def test_legacy_dielectric_side_is_explicit_and_distinct(self):
        centered = triangular_gold(gil_nm=5.)
        legacy = triangular_gold(gil_nm=5., gil_side="dielectric")
        edge = 100 / (2 * centered.height_nm / centered.period_nm)
        point = np.array([edge + 2.5])
        self.assertAlmostEqual(centered.sample(point, 100.)[0], centered.n_in**2)
        self.assertAlmostEqual(legacy.sample(point, 100.)[0], (.5 * (centered.n_in + centered.n_metal))**2)


if __name__ == "__main__":
    unittest.main()
