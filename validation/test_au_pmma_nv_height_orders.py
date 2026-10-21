"""Check the requested case grid and keep numerical stability/passivity separate."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_au_pmma_nv_height_orders as sweep


class HeightOrderTests(unittest.TestCase):
    def test_requested_grid_has_51_unchanged_coating_cases(self):
        args = sweep.build_parser().parse_args([])
        orders, forwarded = sweep.forwarded_arguments(args)
        jobs = sweep.height.build_jobs(sweep.height.build_parser().parse_args(forwarded))
        self.assertEqual(orders, list(range(4, 37, 2)))
        self.assertEqual(len(jobs), 51)
        actual = set()
        for job in jobs:
            identity = job["identity"]
            g = identity["backend_identity"]["geometry"]
            self.assertEqual((identity["material"], identity["structure"], identity["factorization"]), ("Au", "pmma", "symmetric-nv"))
            self.assertEqual((identity["slices"], identity["grid"], identity["wavelength_nm"]), (210, 512, 700))
            self.assertEqual((g["period_nm"], g["tip_radius_nm"], g["base_radius_nm"]), (200, 5, 65))
            self.assertTrue(all(g[k] == 30 for k in ("metal_side_thickness_nm", "metal_top_thickness_nm", "metal_valley_thickness_nm")))
            actual.add((g["height_nm"], identity["order"]))
        self.assertEqual(actual, {(h, m) for h in (400, 300, 200) for m in orders})

    def rows(self):
        return [dict(height_nm=400., wavelength_nm=700., order=m, status="diagnostics_complete",
                     R_percent=10+.1*(m-32), T_percent=5., A_percent=85-.1*(m-32),
                     physical_result=True, diagnostic_checks_passed=True,
                     sample_checks_passed=False) for m in (32, 34, 36)]

    def test_half_percentage_point_tail_and_passivity_are_separate(self):
        rows = self.rows()
        result = sweep.tail_checks(rows, list(range(4, 37, 2)), .005)[0]
        self.assertTrue(result["RTA_tail_within_tolerance"])
        self.assertTrue(result["sampled_tail_checks_passed"])
        self.assertAlmostEqual(result["comparisons"][0]["delta_R_pp"], .2)
        rows[-1]["physical_result"] = False
        result = sweep.tail_checks(rows, list(range(4, 37, 2)), .005)[0]
        self.assertTrue(result["RTA_tail_within_tolerance"])
        self.assertFalse(result["physical_tail_passed"])
        self.assertFalse(result["sampled_tail_checks_passed"])
        rows = self.rows()
        rows[-1]["R_percent"] += 1.
        result = sweep.tail_checks(rows, list(range(4, 37, 2)), .005)[0]
        self.assertFalse(result["RTA_tail_within_tolerance"])
        self.assertAlmostEqual(result["comparisons"][-1]["delta_R_pp"], 1.2)

    def test_missing_or_failed_tail_cannot_pass(self):
        for rows in (self.rows()[::2], self.rows()):
            if len(rows) == 3:
                rows[1]["status"] = "oom"
            result = sweep.tail_checks(rows, list(range(4, 37, 2)), .005)[0]
            self.assertFalse(result["tail_complete"])
            self.assertFalse(result["sampled_tail_checks_passed"])
            self.assertIsNone(result["comparisons"][0]["maximum_change_pp"])

    def test_invalid_grid_and_no_output_dry_run(self):
        parser = sweep.build_parser()
        for argv in (("--step", "0"), ("--max-order", "35"), ("--grid", "128")):
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                sweep.forwarded_arguments(parser.parse_args(argv))
        with tempfile.TemporaryDirectory(prefix="_au_height_order_", dir=ROOT/"validation") as temporary:
            output = Path(temporary)/"results"
            with redirect_stdout(io.StringIO()):
                status = sweep.main(["--device", "cpu", "--start-order", "2", "--max-order", "4",
                    "--slices", "4", "--grid", "32", "--dry-run", "--output-root", str(output)])
            self.assertEqual(status, 0)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
