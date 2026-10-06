"""Convergence regression checks; run with Python only, without torch."""
from __future__ import annotations

import contextlib
import copy
import io
import tempfile
import unittest
from pathlib import Path
from common import HERE, Study, case_key, load_config
from run_all import stable_candidate, run_search


class ReportingChecks(unittest.TestCase):
    def setUp(self):
        self.config, self.model = load_config(HERE/"config.json")
        self.config["wavelengths_nm"] = [550, 700]
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.study = Study(self.config, self.model, Path(self.temp.name), "cpu")
        self.fixed = {"order": 24, "slices": 140, "grid": 576}

    @staticmethod
    def row(numbers, wave, reflectance):
        return dict(numbers, wavelength_nm=wave, polarizations={p: {
            "reflectance": reflectance, "power_into_substrate": 0.05,
            "relief_absorptance": 0.95-reflectance} for p in ("TE", "TM")})

    def seed(self, values, rs):
        for value, r in zip(values, rs):
            for wave in self.config["wavelengths_nm"]:
                numbers = dict(self.fixed, order=value)
                self.study.checkpoint["cases"][case_key(numbers, wave)] = self.row(numbers, wave, r)

    def test_small_early_step_does_not_hide_high_order_oscillation(self):
        self.seed([4, 8, 12, 16], [0.7, 0.701, 0.68, 0.71])
        report = self.study.sweep_report("order", [4, 8, 12, 16], self.fixed)
        self.assertEqual(report["status"], "not_converged")
        self.assertIsNone(stable_candidate(report))

    def test_missing_wavelength_cannot_pass(self):
        self.seed([4, 8, 12], [0.7, 0.701, 0.702])
        del self.study.checkpoint["cases"][case_key(dict(self.fixed, order=8), 700)]
        report = self.study.sweep_report("order", [4, 8, 12], self.fixed)
        self.assertEqual(report["status"], "incomplete")

    def test_tm_failure_is_not_hidden_by_te(self):
        self.seed([4, 8, 12], [0.7, 0.701, 0.702])
        row = self.study.checkpoint["cases"][case_key(dict(self.fixed, order=12), 700)]
        row["polarizations"]["TM"]["reflectance"] = 0.72
        row["polarizations"]["TM"]["relief_absorptance"] = 0.23
        self.assertEqual(self.study.sweep_report("order", [4, 8, 12], self.fixed)["status"], "not_converged")

    def test_stable_unphysical_values_cannot_pass(self):
        self.seed([4, 8, 12], [1.1, 1.1, 1.1])
        self.assertEqual(self.study.sweep_report("order", [4, 8, 12], self.fixed)["status"], "not_converged")

    def test_changed_geometry_cannot_reuse_checkpoint(self):
        self.study.save()
        changed = copy.deepcopy(self.config)
        changed["geometry"]["height_nm"] = 501
        with self.assertRaisesRegex(ValueError, "differ"):
            Study(changed, self.model, Path(self.temp.name), "cpu")

    def test_combinations_are_rechecked_after_initial_sweeps_pass(self):
        # High-other-parameter sweeps are stable. An interaction error only
        # appears when all three cheap candidates are used together.
        values = {"order": [2, 4, 6, 8], "slices": [2, 4, 6, 8],
                  "grid": [48, 64, 96, 128]}
        config = copy.deepcopy(self.config)
        config["wavelengths_nm"] = [550]
        study = Study(config, self.model, Path(self.temp.name)/"joint", "cpu")
        def mock(config, numbers, wave, eps, device):
            interaction = 0.02 if (numbers["order"] < 8 and numbers["slices"] < 8
                                   and numbers["grid"] < 128) else 0
            row = self.row(numbers, wave, 0.7+interaction)
            row["runtime_seconds"] = 0
            return row
        study.simulate = mock
        with contextlib.redirect_stdout(io.StringIO()):
            report = run_search(study, values, calculate=True, max_rounds=6)
        self.assertEqual(report["status"], "converged_within_tested_values")
        self.assertEqual(report["verification_rounds"][0]["numerics"],
                         {"order": 6, "slices": 6, "grid": 96})
        self.assertNotEqual(report["recommended_numerics"], report["verification_rounds"][0]["numerics"])
        self.assertGreater(len(report["verification_rounds"]), 1)
        self.assertTrue(all(d["passes_tolerance"] for d in
                            report["verification_rounds"][-1]["reference_comparison"].values()))


if __name__ == "__main__":
    unittest.main()
