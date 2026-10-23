"""Check selective reuse, nonphysical continuation, and local OOM pruning."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_ag_nv_missing_orders as sweep


class CompletionTests(unittest.TestCase):
    def arguments(self, base, *extra):
        source = Path(base)/"source"
        source.mkdir(exist_ok=True)
        return ["--device", "cpu", "--slices", "4", "--grid", "64",
                "--start-order", "4", "--max-order", "8", "--wavelengths", "550",
                "--source-root", str(source), "--output-root", str(Path(base)/"output"),
                "--no-plots", *extra]

    def ok(self, identity, order, wave):
        coated = identity["structure"] == "pmma"
        row = dict(material="Ag", structure=identity["structure"], order=order,
            wavelength_nm=wave, slices=identity["slices"], grid=identity["grid"],
            total_pattern_layers=identity["slices"]+int(coated), layer_method="nvm",
            nv_lanczos_enabled=False, passivity_warning=False, reflectance=.2,
            transmittance=.1 if coated else 0., absorptance=.7 if coated else .8,
            power_into_substrate=.1 if coated else .2,
            motheye_absorptance=None if coated else .6,
            substrate_absorptance=None if coated else .2)
        return dict(status="ok", signature=sweep.nv.identity_signature(identity),
                    case=row, environment={"device": "test"})

    def worker_ok(self, plan, result, order, wave):
        return self.ok(json.loads(plan.read_text(encoding="utf-8"))["identity"], order, wave)

    def run_quiet(self, argv):
        with redirect_stdout(io.StringIO()):
            return sweep.main(argv)

    def test_default_grid_and_no_write_dry_run(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"validation") as base:
            argv = self.arguments(base, "--slices", "210", "--grid", "512",
                "--max-order", "36", "--wavelengths", "400,550,700")
            jobs, reports, refs, output = sweep.prepare(sweep.build_parser().parse_args(argv))
            self.assertEqual(len(jobs), 6)
            self.assertEqual(sum(len(j["orders"]) for j in jobs), 102)
            self.assertEqual(jobs[0]["identity"]["geometry"]["height_nm"], 500)
            self.assertEqual((reports, refs), ({}, {}))
            self.assertEqual(self.run_quiet(argv+["--dry-run"]), 0)
            self.assertFalse(output.exists())

    def test_reuse_nonphysical_reference_and_resume_without_recalculation(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"validation") as base:
            argv = self.arguments(base, "--structures", "metal")
            args = sweep.build_parser().parse_args(argv)
            jobs, _, _, _ = sweep.prepare(args)
            job = jobs[0]
            identity = deepcopy(job["identity"])
            identity["metal_csv"] = "/different/host/Ag.csv"
            failure = dict(status="calculation_failed", order=6, wavelength_nm=550.,
                error="ValueError: Nonphysical power result (not CUDA OOM)",
                power_diagnostics=dict(reflectance=1.1, transmittance=0., absorptance=-.1))
            source_file = args.source_root/"order_sweep"/job["directory"]/"checkpoint.json"
            sweep.nv.write_json(source_file, dict(plan={"identity": identity},
                signature=sweep.nv.identity_signature(identity),
                cases={sweep.nv.case_key(4, 550): self.ok(identity, 4, 550)["case"]}, failure=failure))
            before = source_file.read_bytes()
            with patch.object(sweep.backend, "launch_worker", side_effect=self.worker_ok) as worker:
                self.assertEqual(self.run_quiet(argv), 2)
                self.assertEqual([c.args[2] for c in worker.call_args_list], [8])
                self.assertEqual(self.run_quiet(argv), 2)
                self.assertEqual(worker.call_count, 1)
                self.assertEqual(self.run_quiet(argv+["--retry-failures"]), 0)
                self.assertEqual([c.args[2] for c in worker.call_args_list], [8, 6])
            self.assertEqual(source_file.read_bytes(), before)

    def test_new_nonphysical_continues_and_attempt_paths_are_unique(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"validation") as base:
            argv = self.arguments(base, "--structures", "metal")
            def worker(plan, result, order, wave):
                if order == 6:
                    return dict(status="calculation_failed", error="Nonphysical power result",
                        power_diagnostics=dict(reflectance=1.2, transmittance=0., absorptance=-.2))
                return self.worker_ok(plan, result, order, wave)
            with patch.object(sweep.backend, "launch_worker", side_effect=worker) as launch:
                self.assertEqual(self.run_quiet(argv), 2)
                self.assertEqual([c.args[2] for c in launch.call_args_list], [4, 6, 8])
                self.assertEqual(self.run_quiet(argv+["--retry-failures"]), 2)
                calls = launch.call_args_list
                self.assertEqual([c.args[2] for c in calls], [4, 6, 8, 6])
                self.assertNotEqual(calls[1].args[1], calls[3].args[1])
            summary = json.loads((Path(base)/"output/summary.json").read_text())
            self.assertEqual(summary["counts"], {"nonphysical": 1, "ok": 2})

    def test_oom_prunes_only_same_structure_and_wavelength(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"validation") as base:
            argv = self.arguments(base, "--wavelengths", "400,550")
            def worker(plan, result, order, wave):
                identity = json.loads(plan.read_text(encoding="utf-8"))["identity"]
                if identity["structure"] == "metal" and wave == 400 and order == 6:
                    return dict(status="oom", error="mock CUDA OOM")
                return self.ok(identity, order, wave)
            with patch.object(sweep.backend, "launch_worker", side_effect=worker) as launch:
                self.assertEqual(self.run_quiet(argv), 2)
                self.assertEqual(launch.call_count, 11)
                self.assertEqual(self.run_quiet(argv), 2)
                self.assertEqual(launch.call_count, 11)
            summary = json.loads((Path(base)/"output/summary.json").read_text())
            self.assertEqual(summary["counts"], {"ok": 10, "oom": 1, "skipped_oom": 1})

    def test_pmma_json_resume_and_changed_settings_are_rejected_before_launch(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"validation") as base:
            argv = self.arguments(base, "--structures", "pmma")
            with patch.object(sweep.backend, "launch_worker", side_effect=self.worker_ok) as launch:
                self.assertEqual(self.run_quiet(argv), 0)
                self.assertEqual(self.run_quiet(argv), 0)
                self.assertEqual(launch.call_count, 3)
            args = sweep.build_parser().parse_args(argv)
            jobs, _, _, output = sweep.prepare(args)
            identity = jobs[0]["identity"]
            shifted = deepcopy(identity)
            shifted["metal_csv"] = "/other/Ag.csv"
            shifted["pmma"]["path"] = "/other/PMMA.csv"
            self.assertEqual(sweep.fixed_signature(identity), sweep.fixed_signature(shifted))
            shifted["geometry"]["height_nm"] = 400
            self.assertNotEqual(sweep.fixed_signature(identity), sweep.fixed_signature(shifted))
            path = sweep.report_path(output, jobs[0], 4)
            report = json.loads(path.read_text())
            report["identity"]["metal_sha256_lf"] = "changed"
            sweep.nv.write_json(path, report)
            with self.assertRaises(ValueError):
                sweep.prepare(args)

    def test_nonfinite_failure_does_not_stop_other_orders(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"validation") as base:
            argv = self.arguments(base, "--structures", "metal")
            def worker(plan, result, order, wave):
                if order == 6:
                    return dict(status="calculation_failed", error="Nonphysical power result",
                        power_diagnostics=dict(reflectance="nan", transmittance=0., absorptance="nan"))
                return self.worker_ok(plan, result, order, wave)
            with patch.object(sweep.backend, "launch_worker", side_effect=worker) as launch:
                self.assertEqual(self.run_quiet(argv), 2)
                self.assertEqual([c.args[2] for c in launch.call_args_list], [4, 6, 8])

    def test_adding_galerkin_preserves_and_reuses_previous_nv_results(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"validation") as base:
            argv = self.arguments(base, "--structures", "metal")
            with patch.object(sweep.backend, "launch_worker", side_effect=self.worker_ok):
                self.assertEqual(self.run_quiet(argv), 0)
            args = sweep.build_parser().parse_args(argv)
            jobs, _, _, output = sweep.prepare(args)
            # A report from the earlier NV-only version has no outer method label.
            path = sweep.report_path(output, jobs[0], 4)
            old_report = json.loads(path.read_text())
            old_report.pop("factorization", None)
            sweep.nv.write_json(path, old_report)
            before = path.read_bytes()
            def worker(job, plan, result, order, wave):
                self.assertEqual(job["factorization"], "galerkin")
                return self.worker_ok(plan, result, order, wave)
            with patch.object(sweep, "launch_case", side_effect=worker) as launch:
                self.assertEqual(self.run_quiet(argv+["--factorizations", "symmetric-nv,galerkin"]), 0)
                self.assertEqual(launch.call_count, 3)
                self.assertEqual(self.run_quiet(argv+["--factorizations", "symmetric-nv,galerkin"]), 0)
                self.assertEqual(launch.call_count, 3)
            self.assertEqual(path.read_bytes(), before)
            args.factorizations = "symmetric-nv,galerkin"
            jobs, reports, _, _ = sweep.prepare(args)
            self.assertEqual(len({sweep.case_tag(j, m) for j in jobs for m in j["orders"]}), 6)
            self.assertNotEqual(jobs[0]["directory"], jobs[1]["directory"])
            self.assertEqual(len(reports), 6)
            self.assertNotIn("factorization", jobs[0]["identity"])
            self.assertEqual(jobs[1]["identity"]["factorization"], "galerkin")
            args.factorizations = "unrecognized"
            with self.assertRaises(ValueError):
                sweep.prepare(args)

    def test_oom_in_nv_does_not_prune_galerkin(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"validation") as base:
            argv = self.arguments(base, "--structures", "metal", "--factorizations", "symmetric-nv,galerkin")
            calls = []
            def worker(job, plan, result, order, wave):
                calls.append((job["factorization"], order))
                if job["factorization"] == "symmetric-nv" and order == 6:
                    return dict(status="oom", error="mock CUDA OOM")
                return self.worker_ok(plan, result, order, wave)
            with patch.object(sweep, "launch_case", side_effect=worker):
                self.assertEqual(self.run_quiet(argv), 2)
            self.assertEqual(calls, [("symmetric-nv", 4), ("symmetric-nv", 6),
                ("galerkin", 4), ("galerkin", 6), ("galerkin", 8)])
            summary = json.loads((Path(base)/"output/summary.json").read_text())
            self.assertEqual(summary["counts_by_factorization"]["galerkin"]["ok"], 3)
            self.assertEqual(summary["counts_by_factorization"]["symmetric-nv"]["skipped_oom"], 1)

    def test_galerkin_worker_rejects_changed_control_before_computation(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"validation") as base:
            args = sweep.build_parser().parse_args(self.arguments(base, "--structures", "metal",
                "--factorizations", "galerkin"))
            jobs, _, refs, _ = sweep.prepare(args)
            self.assertEqual(refs, {})
            identity = deepcopy(jobs[0]["identity"])
            identity["control_source_sha256"]["run_ag_nv_factorization_check.py"] = "changed"
            plan, result = Path(base)/"plan.json", Path(base)/"result.json"
            sweep.nv.write_json(plan, {"identity": identity})
            with patch.object(sweep.backend, "worker") as worker:
                self.assertEqual(sweep.galerkin_worker(plan, result, 4, 550), 1)
                worker.assert_not_called()
            self.assertEqual(json.loads(result.read_text())["status"], "calculation_failed")


if __name__ == "__main__":
    unittest.main()
