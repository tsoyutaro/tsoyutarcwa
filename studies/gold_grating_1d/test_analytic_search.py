"""Check that the analytic follow-up cannot recommend incomplete/coupled sweeps."""
from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from common import HERE, Study, load_config
from run_analytic import run_search


class AnalyticSearchChecks(unittest.TestCase):
    def setUp(self):
        self.config, self.model = load_config(HERE/"config.json")
        self.config["solver"]["fourier_coefficients"] = "analytic"
        self.config["wavelengths_nm"] = [700]
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.study = Study(self.config, self.model, Path(self.temp.name), "cpu")
        self.values = {"order": [2, 4, 6, 8], "slices": [2, 4, 6, 8]}

    def search(self, calculate):
        with contextlib.redirect_stdout(io.StringIO()):
            return run_search(self.study, self.values, calculate=calculate, max_rounds=6)

    def test_missing_cases_do_not_recommend(self):
        report = self.search(False)
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["axes_with_missing_cases"], ["order", "slices"])
        self.assertNotIn("recommended_numerics", report)
        self.assertIsNone(report["grid_used"])

    def test_joint_candidate_is_rechecked_and_promoted(self):
        calls = []
        def mock(config, numbers, wave, eps, device):
            calls.append((numbers["order"], numbers["slices"]))
            self.assertEqual(config["solver"]["fourier_coefficients"], "analytic")
            interaction = 0.02 if numbers["order"] < 8 and numbers["slices"] < 8 else 0
            r = 0.7+interaction
            return dict(numbers, wavelength_nm=wave, runtime_seconds=0,
                        coefficient_method="analytic", grid_used=None,
                        polarizations={p: {"reflectance": r, "power_into_substrate": 0.05,
                                           "relief_absorptance": 0.95-r}
                                       for p in ("TE", "TM")})
        self.study.simulate = mock
        report = self.search(True)
        self.assertEqual(report["status"], "converged_within_tested_values")
        self.assertEqual(report["verification_rounds"][0]["numerics"], {"order": 6, "slices": 6})
        self.assertEqual(report["recommended_numerics"], {"order": 8, "slices": 8})
        self.assertGreater(len(report["verification_rounds"]), 1)
        self.assertNotIn("grid", report["recommended_numerics"])
        self.assertTrue(all(d["passes_tolerance"] for d in
                            report["verification_rounds"][-1]["reference_comparison"].values()))
        previous_calls = len(calls)
        repeated = self.search(False)
        self.assertEqual(repeated["recommended_numerics"], report["recommended_numerics"])
        self.assertEqual(len(calls), previous_calls)

    def test_sampled_configuration_is_rejected(self):
        self.config["solver"]["fourier_coefficients"] = "sampled"
        with self.assertRaisesRegex(ValueError, "analytic"):
            self.search(False)


if __name__ == "__main__":
    unittest.main()
