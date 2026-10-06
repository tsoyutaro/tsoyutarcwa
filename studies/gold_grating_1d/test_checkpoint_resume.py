"""Strict provenance and parity checks before reusing pre-fix saved results."""
from __future__ import annotations

import contextlib
import copy
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from common import (HERE, case_key, digest, load_config, read_json,
                    signature_inputs, write_json)
from resume_checkpoint import (OLD_SOLVER_SHA256, SOLVER_PATH,
                               resume_orthogonal_checkpoint)


class ResumeChecks(unittest.TestCase):
    def setUp(self):
        self.config, self.model = load_config(HERE/"config.json")
        self.config["solver"]["fourier_coefficients"] = "analytic"
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.path = self.output/"checkpoint.json"
        old_inputs = signature_inputs(self.config)
        old_inputs["source_sha256"][SOLVER_PATH] = OLD_SOLVER_SHA256
        self.old = {"signature": digest(old_inputs), "inputs": old_inputs,
                    "cases": {}, "errors": {"48|300|576|700": {"error": "old Ky check"}}}
        for order, waves in ((32, [400, 700]), (40, [400, 700]), (48, [400, 650])):
            for wave in waves:
                row = self.row({"order": order, "slices": 300, "grid": 576}, wave)
                self.old["cases"][case_key(row, wave)] = row
        write_json(self.path, self.old)
        self.original_bytes = self.path.read_bytes()

    @staticmethod
    def row(numbers, wave, r=0.7):
        return dict(numbers, wavelength_nm=wave, coefficient_method="analytic", grid_used=None,
                    runtime_seconds=1, polarizations={pol: {
                        "reflectance": r, "power_into_substrate": 0.05,
                        "relief_absorptance": 0.95-r} for pol in ("TE", "TM")})

    def migrate(self, simulate, calculate=True):
        with patch.dict(sys.modules, {"solver": types.SimpleNamespace(simulate=simulate)}), \
             contextlib.redirect_stdout(io.StringIO()):
            return resume_orthogonal_checkpoint(self.config, self.model, self.output, "cpu",
                                               calculate=calculate)

    def test_known_fix_keeps_backup_values_and_checks_three_largest_orders(self):
        calls = []
        def simulate(config, numbers, wave, eps, device):
            calls.append((numbers["order"], wave))
            return self.row(numbers, wave, 0.7+1e-11)
        record = self.migrate(simulate)
        self.assertEqual(calls, [(32, 700), (40, 700), (48, 650)])
        self.assertEqual(record["reused_cases"], 6)
        self.assertEqual(Path(record["backup"]).read_bytes(), self.original_bytes)
        updated = read_json(self.path)
        self.assertEqual(updated["signature"], digest(signature_inputs(self.config)))
        self.assertEqual(updated["errors"], self.old["errors"])
        for key, original in self.old["cases"].items():
            self.assertEqual(updated["cases"][key]["polarizations"], original["polarizations"])
            self.assertEqual(updated["cases"][key]["orthogonal_fix_reused_from"]["signature"], self.old["signature"])
        self.migrate(simulate)
        self.assertEqual(len(calls), 3)

    def test_unrelated_source_change_cannot_be_migrated(self):
        changed = copy.deepcopy(self.old)
        changed["inputs"]["source_sha256"]["rcwa_ext/nvm.py"] = "unrecognized"
        changed["signature"] = digest(changed["inputs"])
        write_json(self.path, changed)
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "beyond"):
            self.migrate(lambda *args: self.fail("Should not solve"))
        self.assertEqual(self.path.read_bytes(), before)

    def test_parity_failure_preserves_original_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "parity failed"):
            self.migrate(lambda cfg, nums, wave, eps, dev: self.row(nums, wave, 0.71))
        self.assertEqual(self.path.read_bytes(), self.original_bytes)
        self.assertFalse(list(self.output.glob("checkpoint_before*.json")))

    def test_report_only_does_not_solve_or_change_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "First rerun"):
            self.migrate(lambda *args: self.fail("Should not solve"), calculate=False)
        self.assertEqual(self.path.read_bytes(), self.original_bytes)


if __name__ == "__main__":
    unittest.main()
