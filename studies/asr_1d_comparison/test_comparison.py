"""Physical adapter parity and safeguards against false convergence."""
import copy
import unittest

import torch
from threadpoolctl import threadpool_limits

from .adapters import (METRICS, STUDIES, existing_li_case, layer_specs, load_study,
                       nested_intervals, scalar_case)
from .compare import healthy, summarize


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.limits = threadpool_limits(limits=1)

    @classmethod
    def tearDownClass(cls):
        cls.limits.restore_original_limits()

    def test_actual_study_li_matches_scalar_fmm_for_both_ports_and_polarizations(self):
        for study in STUDIES:
            config, model, _ = load_study(study)
            for wave in (400., 550., 700.):
                with self.subTest(study=study, wave=wave):
                    materials = model(wave)
                    expected = existing_li_case(study, config, 6, 4, wave, materials, "cpu")
                    actual = scalar_case(study, config, 6, 4, wave, materials, method="fmm")
                    for pol in ("TE", "TM"):
                        for metric in METRICS[study]:
                            self.assertAlmostEqual(actual["polarizations"][pol][metric],
                                                   expected["polarizations"][pol][metric], delta=2e-9)

    def test_layer_order_thickness_materials_and_exact_valley_interface(self):
        for study in STUDIES:
            config, model, _ = load_study(study)
            materials = model(550.)
            layers, substrate = layer_specs(study, config, 100, materials)
            period = config["geometry"]["period_nm"]
            if study == "gold_grating_1d":
                self.assertEqual(len(layers), 100)
                self.assertEqual(substrate, materials)
                self.assertLess(layers[0].breaks[2]-layers[0].breaks[1], layers[-1].breaks[2]-layers[-1].breaks[1])
                self.assertAlmostEqual(sum(s.thickness for s in layers)*period, 500.)
            else:
                self.assertEqual(len(layers), 101)
                self.assertEqual(substrate, materials[0])
                self.assertEqual(layers[0].name, "cap_1")
                self.assertAlmostEqual(layers[0].thickness*period, 30.)
                self.assertAlmostEqual(sum(s.thickness for s in layers)*period, 530.)
                self.assertAlmostEqual(sum(s.thickness for s in layers if s.name.startswith("valley"))*period, 30.)
                self.assertIn(materials[0], layers[-1].epsilon)
                self.assertIn(materials[1], layers[-1].epsilon)
                self.assertNotIn(1., layers[-1].epsilon)

    def test_nested_three_materials_and_full_width_are_exact(self):
        edges, eps = nested_intervals(.2, .6, 2.25, -5+1j, 1.)
        self.assertEqual(eps, (1., -5+1j, 2.25, -5+1j, 1.))
        self.assertAlmostEqual(edges[3]-edges[2], .2)
        _, eps = nested_intervals(.2, 1., 2.25, -5+1j, -5+1j)
        self.assertEqual(eps, (-5+1j, 2.25, -5+1j))

    def test_healthy_asr_case_matches_tensor_backend(self):
        study = "pmma_gold_grating_1d"
        config, model, _ = load_study(study)
        outputs = [scalar_case(study, config, 12, 8, 550., model(550.),
                               oversampling=4, backend=backend) for backend in ("scipy", "torch")]
        for pol in ("TE", "TM"):
            self.assertTrue(healthy(outputs[0]["polarizations"][pol], METRICS[study], 1e-7))
            for metric in METRICS[study]:
                self.assertAlmostEqual(outputs[0]["polarizations"][pol][metric],
                                       outputs[1]["polarizations"][pol][metric], delta=2e-7)


class ReportingTests(unittest.TestCase):
    def test_nonpassive_transmission_cannot_pass_or_get_reference_error(self):
        plan = dict(wavelengths_nm=[550.], orders=[2, 4, 8], asr_ratios=[4], slices=10,
                    reference_order=16, reference_check_order=12,
                    tolerance=.005, passivity_tolerance=1e-7)
        values = dict(reflectance=.2, transmittance=.3, absorptance=.5)
        cases = {f"{method}|10|{order}|550": dict(polarizations={p: copy.deepcopy(values) for p in ("TE", "TM")})
                 for method, orders in (("li", [2, 4, 8, 12, 16]), ("asr_r4", [2, 4, 8])) for order in orders}
        cases["asr_r4|10|8|550"]["polarizations"]["TM"].update(transmittance=2., absorptance=-1.2)
        report, rows, _ = summarize("pmma_gold_grating_1d", dict(cases=cases), plan)
        self.assertFalse(report["methods"]["asr_r4"]["tail_stable_at_tested_wavelengths"])
        self.assertEqual(report["methods"]["asr_r4"]["invalid_polarization_cases"], 1)
        invalid = next(r for r in rows if r["series"] == "asr_r4" and r["order"] == 8 and r["polarization"] == "TM")
        self.assertIsNone(invalid["difference_from_finite_li_reference_transmittance"])
        self.assertFalse(report["methods"]["asr_r4"]["tail_and_finite_reference_agree_within_tolerance"])

    def test_passive_flat_tail_far_from_reference_is_not_consistent(self):
        plan = dict(wavelengths_nm=[550.], orders=[2, 4, 8], asr_ratios=[4], slices=10,
                    reference_order=16, reference_check_order=12,
                    tolerance=.005, passivity_tolerance=1e-7)
        reference = dict(reflectance=.2, transmittance=.3, absorptance=.5)
        plateau = dict(reflectance=.4, transmittance=.3, absorptance=.3)
        cases = {f"{method}|10|{order}|550": dict(polarizations={p: copy.deepcopy(values) for p in ("TE", "TM")})
                 for method, orders, values in (("li", [2, 4, 8, 12, 16], reference),
                                               ("asr_r4", [2, 4, 8], plateau)) for order in orders}
        report, _, _ = summarize("pmma_gold_grating_1d", dict(cases=cases), plan)
        status = report["methods"]["asr_r4"]
        self.assertTrue(status["tail_stable_at_tested_wavelengths"])
        self.assertAlmostEqual(status["last_order_max_absolute_difference_from_finite_reference"], .2)
        self.assertFalse(status["last_order_within_reference_tolerance"])
        self.assertFalse(status["tail_and_finite_reference_agree_within_tolerance"])


if __name__ == "__main__":
    unittest.main()
