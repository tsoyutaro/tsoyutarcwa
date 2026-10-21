"""Validate changed geometry, source isolation, power parity and resumability."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_au_nv_geometry_diagnostics as sweep


class GeometryDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="_au_geometry_test_", dir=ROOT/"validation")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def argv(self, *extra):
        return ["--device", "cpu", "--orders", "Au:2", "--slices", "4", "--grids", "32",
                "--quadrature", "8", "--output-root", str(self.directory/"output"), "--no-plots", *extra]

    def test_nonoverlapping_geometry_preserves_material_and_film(self):
        args = sweep.build_parser().parse_args(self.argv())
        templates = deepcopy(sweep.diagnostic.build_jobs(args))
        originals = deepcopy((sweep.diagnostic.gold.METAL_GEOMETRY, sweep.diagnostic.gold.COATED_GEOMETRY))
        jobs = sweep.build_jobs(args)
        self.assertEqual(len(jobs), 2)
        for job, template in zip(jobs, templates):
            identity = deepcopy(job["identity"])
            sweep.diagnostic.verify_sources(identity)
            g = identity["backend_identity"].pop("geometry")
            expected = deepcopy(template["identity"])
            expected["backend_identity"].pop("geometry")
            identity["sources"].pop(Path(sweep.__file__).name)
            self.assertEqual(identity, expected)
            self.assertEqual((g["period_nm"], g["height_nm"], g["tip_radius_nm"]), (100, 200, 5))
            base = 47.5 if identity["structure"] == "metal" else 17.5
            self.assertEqual(g["base_radius_nm"], base)
            self.assertEqual(g["period_nm"]-2*(base+g.get("metal_side_thickness_nm", 0)), 5.)
            if identity["structure"] == "pmma":
                self.assertTrue(all(g[k] == 30 for k in ("metal_side_thickness_nm", "metal_top_thickness_nm", "metal_valley_thickness_nm")))
        self.assertEqual(originals, (sweep.diagnostic.gold.METAL_GEOMETRY, sweep.diagnostic.gold.COATED_GEOMETRY))

    def test_overlapping_or_invalid_geometry_rejected_before_output(self):
        for extra in (("--metal-base-radius-nm", "50"), ("--pmma-base-radius-nm", "20"),
                      ("--pmma-base-radius-nm", "4"), ("--period-nm", "nan"),
                      ("--height-nm", "20"), ("--metal-side-thickness-nm", "-1")):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                sweep.build_jobs(sweep.build_parser().parse_args(self.argv(*extra)))
        self.assertFalse((self.directory/"output").exists())

    def test_dry_run_writes_nothing(self):
        with redirect_stdout(io.StringIO()):
            self.assertEqual(sweep.main(self.argv("--dry-run")), 0)
        self.assertFalse((self.directory/"output").exists())

    def test_changed_period_cpu_powers_and_layer_depths_match_backend(self):
        import torch
        args = sweep.build_parser().parse_args(self.argv())
        jobs = sweep.build_jobs(args)
        originals = sweep.diagnostic.build_jobs(args)
        with torch.no_grad(), redirect_stdout(io.StringIO()):
            for job, template in zip(jobs, originals):
                identity = job["identity"]
                output = self.directory/job["directory"]
                output.mkdir(parents=True)
                actual = sweep.diagnostic.compute_case(identity, output)
                manual = deepcopy(template["identity"]["backend_identity"])
                g = manual["geometry"]
                g.update(period_nm=100., height_nm=200., tip_radius_nm=5.,
                         base_radius_nm=47.5 if identity["structure"] == "metal" else 17.5)
                expected = sweep.diagnostic.gold.compute_case(manual, 2, 700, torch.device("cpu"))
                for field in ("reflectance", "transmittance", "absorptance", "power_into_substrate"):
                    self.assertAlmostEqual(actual[field], expected[field], places=10)
                self.assertTrue(actual["diagnostic_checks_passed"])
                with (output/"layer_power.csv").open(encoding="utf-8-sig", newline="") as handle:
                    layers = list(csv.DictReader(handle))
                self.assertEqual(len(layers), 4 if identity["structure"] == "metal" else 5)
                self.assertAlmostEqual(float(layers[-1]["depth_bottom_nm"]), 200 if identity["structure"] == "metal" else 230)

    def test_oom_continues_resume_and_changed_radius_rejects_cache(self):
        calls = []
        def fake_run(command, check):
            path = Path(command[-1])
            identity = json.loads(path.read_text(encoding="utf-8"))["identity"]
            calls.append(identity["structure"])
            report = dict(identity=identity, signature=sweep.nv.identity_signature(identity))
            if identity["structure"] == "metal":
                report.update(status="oom", error="simulated CUDA OOM")
            else:
                report.update(status="diagnostics_complete", result=dict(physical_result=False,
                    minimum_layer_absorptance=-.001, negative_absorption_layers=1))
            sweep.nv.write_json(path.parent/"report.json", report)
            return SimpleNamespace(returncode=0)
        with patch.object(sweep.subprocess, "run", fake_run), redirect_stdout(io.StringIO()):
            self.assertEqual(sweep.main(self.argv()), 2)
            self.assertEqual(calls, ["metal", "pmma"])
            calls.clear()
            self.assertEqual(sweep.main(self.argv()), 2)
            self.assertEqual(calls, [])
        summary_path = self.directory/"output/geometry_summary.json"
        before = summary_path.read_bytes()
        summary = json.loads(before)
        self.assertEqual(summary["counts"], {"diagnostics_complete": 1, "oom": 1})
        self.assertEqual(summary["cases"][1]["total_pattern_height_nm"], 230.)
        self.assertEqual(summary["cases"][1]["minimum_layer_absorption_pp"], -.1)
        self.assertNotIn("refinement_checks", summary)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            sweep.main(self.argv("--pmma-base-radius-nm", "16"))
        self.assertEqual(raised.exception.code, 2)
        self.assertEqual(summary_path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
