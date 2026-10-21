"""Height-sweep geometry, reference provenance, fields, and failure isolation."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import csv
import io
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_au_nv_height_diagnostics as sweep


class HeightDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT/"validation"/f"_au_height_test_{uuid.uuid4().hex}"
        self.directory.mkdir()
        self.addCleanup(self.cleanup)

    def cleanup(self):
        resolved = self.directory.resolve()
        assert resolved.parent == (ROOT/"validation").resolve() and resolved.name.startswith("_au_height_test_")
        shutil.rmtree(resolved)

    def arguments(self, *extra):
        return sweep.build_parser().parse_args(["--device", "cpu", "--orders", "Au:2", "--slices", "4",
            "--grids", "32", "--quadrature", "8", "--reference-root", str(self.directory/"reference"),
            "--output-root", str(self.directory/"output"), "--no-plots", *extra])

    def test_default_eight_cases_change_only_height(self):
        args = sweep.build_parser().parse_args([])
        templates = sweep.diagnostic.build_jobs(args)
        before = deepcopy(templates)
        jobs = sweep.build_jobs(args)
        self.assertEqual(len(jobs), 8)
        self.assertEqual([j["identity"]["height_nm"] for j in jobs], [500, 500, 400, 400, 300, 300, 200, 200])
        for job in jobs:
            identity = deepcopy(job["identity"])
            self.assertEqual((identity["order"], identity["slices"], identity["grid"], identity["wavelength_nm"]), (36, 210, 512, 700))
            height = identity.pop("height_nm")
            identity["sources"].pop(Path(sweep.__file__).name)
            geometry = identity["backend_identity"]["geometry"]
            self.assertEqual(geometry["height_nm"], height)
            geometry["height_nm"] = 500.
            expected = next(j for j in templates if j["directory"] == job["reference_directory"])
            self.assertEqual(identity, expected["identity"])
            sweep.diagnostic.verify_sources(job["identity"])
        self.assertEqual(templates, before)
        self.assertEqual(sweep.diagnostic.gold.METAL_GEOMETRY["height_nm"], 500)

    def reference(self, job):
        identity = deepcopy(job["identity"])
        identity.pop("height_nm")
        identity["sources"].pop(Path(sweep.__file__).name)
        identity["backend_identity"]["geometry"]["height_nm"] = 500.
        source = self.directory/"reference"/job["reference_directory"]
        report = dict(status="diagnostics_complete", identity=identity, signature=sweep.nv.identity_signature(identity),
                      result=dict(physical_result=False, diagnostic_checks_passed=True))
        sweep.nv.write_json(source/"report.json", report)
        for name in sweep.REFERENCE_FILES:
            (source/name).write_bytes(b"complete diagnostic data")
        return source

    def test_compatible_nonphysical_reference_reuse_is_read_only(self):
        args = self.arguments("--structures", "metal")
        jobs = sweep.build_jobs(args)
        job = jobs[0]
        source = self.reference(job)
        previous = {p.name: p.read_bytes() for p in source.iterdir()}
        self.assertTrue(sweep.reuse_reference(job, args.reference_root, self.directory/"copy"))
        saved = json.loads((self.directory/"copy/report.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["identity"], job["identity"])
        self.assertFalse(saved["result"]["physical_result"])
        self.assertEqual(saved["signature"], sweep.nv.identity_signature(job["identity"]))
        self.assertEqual(saved["reused_reference"]["report_sha256"], sweep.nv.digest(source/"report.json"))
        self.assertTrue(all((source/name).read_bytes() == data for name, data in previous.items()))
        # The old 500 nm result cannot be used for another height.
        self.assertIsNone(sweep.read_reference(jobs[1], args.reference_root))

    def test_reference_rejects_changed_source_or_missing_internal_data(self):
        args = self.arguments("--structures", "metal", "--heights-nm", "500")
        job = sweep.build_jobs(args)[0]
        source = self.reference(job)
        report = json.loads((source/"report.json").read_text(encoding="utf-8"))
        report["identity"]["sources"]["rcwa_ext/nvm.py"] = "changed"
        report["signature"] = sweep.nv.identity_signature(report["identity"])
        sweep.nv.write_json(source/"report.json", report)
        self.assertIsNone(sweep.read_reference(job, args.reference_root))

    def test_pmma_reference_json_roundtrip_preserves_identity(self):
        args = self.arguments("--structures", "pmma", "--heights-nm", "500")
        job = sweep.build_jobs(args)[0]
        source = self.reference(job)
        self.assertIsNotNone(sweep.read_reference(job, args.reference_root))
        self.assertTrue(sweep.reuse_reference(job, args.reference_root, self.directory/"pmma_copy"))
        saved = json.loads((self.directory/"pmma_copy/report.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["signature"], sweep.nv.identity_signature(job["identity"]))
        source = self.reference(job)
        (source/"layer_power.csv").unlink()
        self.assertIsNone(sweep.read_reference(job, args.reference_root))

    def test_cpu_height_changes_propagation_and_keeps_metal_operators(self):
        import torch
        args = self.arguments("--heights-nm", "500,300")
        jobs = sweep.build_jobs(args)
        source_hashes = {p: sweep.nv.digest(ROOT/p) for p in jobs[0]["identity"]["sources"]}
        computed = {}
        minima = {}
        with torch.no_grad(), redirect_stdout(io.StringIO()):
            for job in jobs:
                identity = job["identity"]
                output = self.directory/job["directory"]
                output.mkdir(parents=True)
                actual = sweep.diagnostic.compute_case(identity, output)
                computed[identity["structure"], identity["height_nm"]] = actual
                self.assertTrue(actual["diagnostic_checks_passed"])
                with (output/"layer_power.csv").open(encoding="utf-8-sig", newline="") as handle:
                    layers = list(csv.DictReader(handle))
                self.assertAlmostEqual(float(layers[-1]["depth_bottom_nm"]), identity["height_nm"] + (30 if identity["structure"] == "pmma" else 0))
                self.assertEqual(actual["total_pattern_layers"], 5 if identity["structure"] == "pmma" else 4)
                if identity["structure"] == "metal":
                    with (output/"operator_diagnostics.csv").open(encoding="utf-8-sig", newline="") as handle:
                        minima[identity["height_nm"]] = [float(r["electric_loss_min_eigenvalue"]) for r in csv.DictReader(handle)]
                if identity["height_nm"] == 500:
                    expected = sweep.diagnostic.gold.compute_case(identity["backend_identity"], 2, 700, torch.device("cpu"))
                    for field in ("reflectance", "transmittance", "absorptance"):
                        self.assertAlmostEqual(actual[field], expected[field], places=10)
        for a, b in zip(minima[500], minima[300]):
            self.assertAlmostEqual(a, b, places=10)
        self.assertGreater(abs(computed["metal", 500]["reflectance"]-computed["metal", 300]["reflectance"]), 1e-5)
        self.assertTrue(all(sweep.nv.digest(ROOT/p) == digest for p, digest in source_hashes.items()))

    def test_oom_and_nonphysical_results_continue_and_resume(self):
        args = self.arguments("--structures", "metal")
        calls = []
        def fake_run(command, check):
            path = Path(command[-1])
            identity = json.loads(path.read_text(encoding="utf-8"))["identity"]
            calls.append(identity["height_nm"])
            report = dict(identity=identity, signature=sweep.nv.identity_signature(identity))
            if identity["height_nm"] == 400:
                report.update(status="oom", error="simulated CUDA OOM")
            else:
                report.update(status="diagnostics_complete", result=dict(physical_result=False))
            sweep.nv.write_json(path.parent/"report.json", report)
            return SimpleNamespace(returncode=2 if identity["height_nm"] == 400 else 0)
        argv = ["--device", "cpu", "--orders", "Au:2", "--structures", "metal", "--slices", "4",
                "--grids", "32", "--quadrature", "8", "--reference-root", str(args.reference_root),
                "--output-root", str(args.output_root), "--no-plots"]
        with patch.object(sweep.subprocess, "run", fake_run), redirect_stdout(io.StringIO()):
            self.assertEqual(sweep.main(argv), 2)
        self.assertEqual(calls, [500, 400, 300, 200])
        summary = json.loads((args.output_root/"height_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["counts"], {"diagnostics_complete": 3, "oom": 1})
        self.assertNotIn("refinement_checks", summary)
        self.assertFalse((args.output_root/"refinement_changes.csv").exists())
        calls.clear()
        with patch.object(sweep.subprocess, "run", fake_run), redirect_stdout(io.StringIO()):
            sweep.main(argv)
        self.assertEqual(calls, [])

    def test_invalid_geometry_and_dry_run_no_output(self):
        with self.assertRaises(ValueError):
            sweep.build_jobs(self.arguments("--heights-nm", "20"))
        with self.assertRaises(ValueError):
            sweep.build_jobs(self.arguments("--orders", "Ag:2"))
        with redirect_stdout(io.StringIO()):
            self.assertEqual(sweep.main(["--device", "cpu", "--dry-run", "--output-root", str(self.directory/"dry")]), 0)
        self.assertFalse((self.directory/"dry").exists())


if __name__ == "__main__":
    unittest.main()
