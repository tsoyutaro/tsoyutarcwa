"""Guard against false 'matched' verdicts; no torch dependency.

Synthetic rows here test the verifier only and are never reference/RCWA data.
"""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from paper_reproductions.weiss2009 import verify_fig4a as verify


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.reference, self.metadata = verify.load_reference()
        self.args = verify.parser().parse_args([])
        self.frequencies = list(map(float, self.reference["frequency_THz"]))
        self.grids = [256, 512]
        self.rows = []
        for grid in self.grids:
            for point in self.reference:
                row = {"grid": grid, "frequency_THz": float(point["frequency_THz"]),
                       "order": 12, "minimum_jacobian": 1e-4}
                row.update({f"{c}_{p}": float(point[c]) for c in verify.CHANNELS for p in ("x", "y")})
                row.update({f"{c}0_{p}": float(point[c]) for c in ("T", "R") for p in ("x", "y")})
                self.rows.append(row)

    def analyze(self, rows=None, frequencies=None, grids=None):
        return verify.analyze(self.rows if rows is None else rows, self.reference,
                              self.grids if grids is None else grids,
                              self.frequencies if frequencies is None else frequencies, self.args)[0]

    def test_reference_provenance_and_range(self):
        self.assertEqual(self.metadata["source_kind"], "publisher_figure_vector_digitization")
        self.assertEqual(len(self.reference), 101)
        self.assertLess(self.metadata["tick_fit_max_residual_THz"], .1)
        self.assertLess(self.frequencies[0], 250)
        self.assertGreater(self.frequencies[-1], 470)

    def test_reference_hash_is_newline_independent(self):
        source = verify.PACKAGE / "reference" / "fig4a_reference.csv"
        with tempfile.TemporaryDirectory() as temporary:
            lf = Path(temporary) / "reference_lf.csv"
            lf.write_text(source.read_text(encoding="utf-8").replace("\r\n", "\n"),
                          encoding="utf-8", newline="\n")
            self.assertNotEqual(verify.sha(source), verify.sha(lf))
            self.assertEqual(verify.canonical_text_sha(source), verify.canonical_text_sha(lf))

    def test_complete_identical_curves_pass(self):
        report = self.analyze()
        self.assertEqual(report["status"], "matched_within_tolerance")
        self.assertEqual(report["paper_metrics"]["T"]["max_abs"], 0)

    def test_complete_converged_but_wrong_curves_mismatch(self):
        for r in self.rows:
            for p in ("x", "y"):
                r[f"T_{p}"] *= .8
                r[f"T0_{p}"] *= .8
                r[f"A_{p}"] = 1 - r[f"T_{p}"] - r[f"R_{p}"]
        self.assertEqual(self.analyze()["status"], "mismatch")

    def test_partial_band_cannot_pass(self):
        frequencies = [f for f in self.frequencies if 350 < f < 380]
        self.assertEqual(self.analyze(frequencies=frequencies)["status"], "inconclusive")

    def test_missing_point_cannot_pass(self):
        self.assertFalse(self.analyze(rows=self.rows[:-1])["complete"])
        self.assertEqual(self.analyze(rows=self.rows[:-1])["status"], "inconclusive")

    def test_sparse_endpoints_cannot_pass(self):
        report = self.analyze(frequencies=[self.frequencies[0], self.frequencies[-1]])
        self.assertFalse(report["full_frequency_coverage"])

    def test_single_grid_cannot_pass(self):
        report = self.analyze(grids=[512])
        self.assertFalse(report["grid_converged"])
        self.assertEqual(report["status"], "inconclusive")

    def test_grid_disagreement_cannot_pass(self):
        self.rows[10]["T_x"] -= .1
        self.rows[10]["T_y"] -= .1
        self.rows[10]["T0_x"] -= .1
        self.rows[10]["T0_y"] -= .1
        self.assertFalse(self.analyze()["grid_converged"])

    def test_nonphysical_cannot_pass(self):
        self.rows[10]["A_x"] = -.05
        self.assertFalse(self.analyze()["passive"])
        self.assertEqual(self.analyze()["status"], "inconclusive")

    def test_nonfinite_is_reportable(self):
        self.rows[10]["T_x"] = float("nan")
        report = self.analyze()
        self.assertFalse(report["finite"])
        json.dumps(report, allow_nan=False)

    def test_wrong_order_cannot_pass(self):
        self.args.order = 2
        self.assertEqual(self.analyze()["status"], "inconclusive")

    def test_polarization_disagreement_cannot_pass(self):
        self.rows[10]["T_y"] += .02
        self.assertIn("x/y symmetry check failed", self.analyze()["reasons"])

    def test_resume_selection_excludes_old_frequencies_and_grids(self):
        extra = copy.deepcopy(self.rows[0])
        extra.update(grid=1024, frequency_THz=600)
        self.assertEqual(len(verify.select_rows(self.rows + [extra], self.grids, self.frequencies)), 202)

    def test_duplicate_checkpoint_rows_rejected(self):
        with self.assertRaises(ValueError):
            self.analyze(rows=self.rows + [self.rows[0]])

    def test_signature_detects_changed_physics(self):
        before = verify.make_signature(self.args, self.metadata)
        self.args.order = 13
        self.assertNotEqual(before, verify.make_signature(self.args, self.metadata))

    def test_frequency_inputs(self):
        self.assertEqual(verify.parse_frequencies("350:370:10,350", self.reference), [350., 360., 370.])
        for value in ("nan", "350:370:0", "-1"):
            with self.assertRaises(ValueError):
                verify.parse_frequencies(value, self.reference)

    def test_atomic_checkpoint_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.json"
            verify.atomic_json(path, {"rows": [1]})
            verify.atomic_json(path, {"rows": [1, 2]})
            self.assertEqual(json.loads(path.read_text())["rows"], [1, 2])
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_absorption_never_uses_zeroth_order_deficit(self):
        self.assertEqual(verify.power_column("A", "x", "zeroth"), "A_x")
        self.assertEqual(verify.power_column("T", "x", "zeroth"), "T0_x")
        self.assertEqual(verify.power_column("T", "x", "total"), "T_x")

    def test_wrong_total_order_definition_detected(self):
        for r in self.rows:
            if r["frequency_THz"] > 430:
                for p in ("x", "y"):
                    r[f"T_{p}"] = min(1., r[f"T0_{p}"] + .1)
        self.assertEqual(self.analyze()["status"], "matched_within_tolerance")
        self.args.paper_power = "total"
        self.assertEqual(self.analyze()["status"], "mismatch")


if __name__ == "__main__":
    unittest.main()
