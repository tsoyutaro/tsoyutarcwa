"""Check physical equivalence, PMMA layer fields, and diagnostic job isolation."""
from contextlib import redirect_stderr, redirect_stdout
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
import run_metals_nv_diagnostics as driver


class MetalsDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT/"validation"/f"_metals_diagnostic_test_{uuid.uuid4().hex}"
        self.directory.mkdir()
        self.addCleanup(self.cleanup)

    def cleanup(self):
        target = self.directory.resolve()
        assert target.parent == (ROOT/"validation").resolve() and target.name.startswith("_metals_diagnostic_test_")
        shutil.rmtree(target)

    def arguments(self, *extra):
        return driver.build_parser().parse_args(["--device", "cpu", "--orders", "Cr:2", "--slices", "4",
            "--grids", "32", "--wavelengths", "550", "--quadrature", "8", "--no-plots",
            "--output-root", str(self.directory/"output"), *extra])

    def test_default_plan_has_48_existing_geometry_cases(self):
        args = driver.build_parser().parse_args(["--device", "cpu"])
        jobs = driver.build_jobs(args)
        self.assertEqual(len(jobs), 48)
        for job in jobs:
            identity, base = job["identity"], job["identity"]["backend_identity"]
            self.assertEqual(identity["order"], 24 if identity["material"] in ("Cr", "Fe", "Ni", "Ti", "V") else 36)
            self.assertEqual(base["geometry"]["period_nm"], 200)
            self.assertEqual(base["geometry"]["height_nm"], 500)
            self.assertEqual(base["geometry"]["base_radius_nm"], 65 if identity["structure"] == "pmma" else 95)
            driver.verify_sources(identity)
            if identity["material"] == "W":
                self.assertEqual(Path(base["metal_csv"]).name, "Werner(W).csv")

    def test_extra_materials_and_refinements_preserve_case_identity(self):
        single = driver.build_jobs(self.arguments("--structures", "metal"))[0]
        expanded = driver.build_jobs(self.arguments("--orders", "Cr:2,Au:2,Ag:2", "--slices", "4,6",
            "--grids", "32,48", "--wavelengths", "400,550,700", "--order-offsets", "0,2"))
        same = [j for j in expanded if j["directory"] == single["directory"]]
        self.assertEqual(same, [single])
        self.assertEqual(len(expanded), 144)

    def test_pmmas_fields_match_independent_internal_field_api(self):
        import torch
        from rcwa_solver_auto import AutoRCWA, GroupTheoryOptions, Lattice, NVMOptions, OutputSpec
        from studies.gold_motheye.converge import _mean_poynting_z, _zero_order_x_source
        from studies.pmma_metal_motheye.model import Geometry, PMMAIndex
        identity = driver.build_jobs(self.arguments("--structures", "pmma"))[0]["identity"]
        output = self.directory/"actual"
        output.mkdir()
        original_class = sys.modules["rcwa_solver_auto"].AutoRCWA
        with redirect_stdout(io.StringIO()), torch.no_grad():
            actual = driver.compute_case(identity, output)
            self.assertIs(sys.modules["rcwa_solver_auto"].AutoRCWA, original_class)
            base, q = identity["backend_identity"], identity["grid"]
            geom = Geometry(**base["geometry"])
            eps = driver.diagnostic.load_tabulated_nk(base["metal_csv"])(550)
            index = PMMAIndex(Path(base["pmma"]["path"]), base["pmma"]["policy"])(550)
            sim = AutoRCWA(freq=200/550, order=[2, 2], lattice=Lattice.triangular(1.), cascade="redheffer",
                outputs=OutputSpec(smatrix_size="half", fields="internal"),
                nvm=NVMOptions(grid=(q, q), use_lanczos=False),
                group_theory=GroupTheoryOptions(enabled=True, symmetry="d6", strict=True, polarization="x"),
                verify_cascade=False, dtype=torch.complex128, device=torch.device("cpu"))
            sim.add_input_layer(eps=1.)
            sim.add_output_layer(eps=index**2)
            sim.set_incident_angle(0., 0.)
            options = dict(centers=((.75, 3**.5/4),), nx=q, ny=q, use_lanczos=False)
            sim.add_layer_circle_nvm(30/200, 35/200, 1., eps, **options)
            for layer in geom.profile(4):
                d = (layer.bottom_depth_nm-layer.top_depth_nm)/200
                if layer.kind == "valley":
                    sim.add_layer_circle_nvm(d, layer.core_radius_nm/200, eps, index**2, **options)
                else:
                    sim.add_layer_circle_shell_nvm(d, layer.core_radius_nm/200, layer.outer_radius_nm/200,
                                                  1., eps, index**2, **options)
            sim.solve_global_smatrix()
            sim.source_planewave(amplitude=[1., 0.], direction="forward")
            source = _zero_order_x_source(sim)
            pin = _mean_poynting_z(source, sim.Vi, direction=1)
            expected_r = -_mean_poynting_z(sim.S[1]@source, sim.Vi, direction=-1)/pin
            expected_t = _mean_poynting_z(sim.S[0]@source, sim.Vo, direction=1)/pin
            self.assertAlmostEqual(actual["reflectance"], expected_r, places=10)
            self.assertAlmostEqual(actual["transmittance"], expected_t, places=10)
            self.assertAlmostEqual(actual["absorptance"], 1-expected_r-expected_t, places=10)
            self.assertEqual(actual["substrate_absorptance"], 0.)
            self.assertAlmostEqual(actual["pattern_absorptance"], actual["absorptance"], places=9)
            with (output/"layer_power.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["kind"], "top_cap")
            self.assertEqual(rows[-1]["kind"], "valley")
            self.assertAlmostEqual(float(rows[-1]["depth_bottom_nm"]), 530)
            for i, row in enumerate(rows):
                for z, field in ((0., "flux_top"), (float(sim.thickness[i].real), "flux_bottom")):
                    ex, ey, _, hx, hy, _ = sim._internal_fourier_fields(i, z)
                    flux = float(.5*torch.real(torch.sum(ex*hy.conj()-ey*hx.conj())))/pin
                    self.assertAlmostEqual(float(row[field]), flux, places=9)
            self.assertTrue(actual["diagnostic_checks_passed"])
            self.assertLess(actual["max_volume_flux_error"], 1e-7)

    def test_original_metal_and_gold_model_are_unchanged_and_ag_failure_retained(self):
        import torch
        args = self.arguments("--orders", "Au:2,Ag:2", "--structures", "metal")
        jobs = driver.build_jobs(args)
        sources = {p: driver.nv.digest(ROOT/p) for job in jobs for p in job["identity"]["sources"]}
        with redirect_stdout(io.StringIO()), torch.no_grad():
            for job in jobs:
                identity = job["identity"]
                output = self.directory/identity["material"]
                output.mkdir()
                actual = driver.compute_case(identity, output)
                base = identity["backend_identity"]
                expected = driver.BACKENDS[identity["material"]].compute_case(base, 2, 550, torch.device("cpu"))
                for field in ("reflectance", "transmittance", "absorptance", "power_into_substrate"):
                    self.assertAlmostEqual(actual[field], expected[field], places=10)
                self.assertEqual(actual["transmittance"], 0.)
                self.assertAlmostEqual(actual["pattern_absorptance"]+actual["substrate_absorptance"], actual["absorptance"], places=9)
                if identity["material"] == "Ag":
                    self.assertGreater(actual["reflectance"], 1.)
                    self.assertFalse(actual["physical_result"])
                    self.assertTrue(actual["diagnostic_checks_passed"])
                    self.assertGreater(actual["negative_absorption_layers"], 0)
        self.assertTrue(all(driver.nv.digest(ROOT/p) == digest for p, digest in sources.items()))

    def test_oom_does_not_stop_next_case_resume_and_source_guard(self):
        args = self.arguments("--structures", "metal", "--wavelengths", "400,550,700")
        calls = []
        def fake_run(command, check):
            plan = Path(command[-1])
            identity = json.loads(plan.read_text(encoding="utf-8"))["identity"]
            calls.append(identity["wavelength_nm"])
            report = dict(identity=identity, signature=driver.nv.identity_signature(identity))
            if identity["wavelength_nm"] == 400:
                report.update(status="oom", error="simulated CUDA OOM")
            else:
                report.update(status="diagnostics_complete", result=dict(physical_result=False))
            driver.nv.write_json(plan.parent/"report.json", report)
            return SimpleNamespace(returncode=2 if identity["wavelength_nm"] == 400 else 0)
        with patch.object(driver.subprocess, "run", fake_run), redirect_stdout(io.StringIO()):
            self.assertEqual(driver.main(self.argv(args)), 2)
        self.assertEqual(calls, [400, 550, 700])
        calls.clear()
        with patch.object(driver.subprocess, "run", fake_run), redirect_stdout(io.StringIO()):
            driver.main(self.argv(args))
        self.assertEqual(calls, [])
        first = driver.build_jobs(args)[0]
        path = args.output_root/first["directory"]/"report.json"
        saved = json.loads(path.read_text(encoding="utf-8"))
        saved["signature"] = "changed-source"
        driver.nv.write_json(path, saved)
        with patch.object(driver.subprocess, "run") as launch, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                driver.main(self.argv(args))
            launch.assert_not_called()

    @staticmethod
    def argv(args):
        return ["--device", args.device, "--orders", args.orders, "--slices", args.slices,
                "--grids", args.grids, "--wavelengths", args.wavelengths, "--structures", args.structures,
                "--quadrature", str(args.quadrature), "--no-plots", "--output-root", str(args.output_root)]

    def test_refinement_and_passivity_are_separate(self):
        args = self.arguments("--structures", "metal", "--slices", "4,6")
        jobs = driver.build_jobs(args)
        args.output_root.mkdir()
        for job in jobs:
            result = dict(reflectance=.1, transmittance=0., absorptance=.9, pattern_absorptance=.8,
                substrate_absorptance=.1, total_negative_loss_sector=-.01,
                physical_result=True, diagnostic_checks_passed=True)
            driver.nv.write_json(args.output_root/job["directory"]/"report.json",
                dict(status="diagnostics_complete", signature=driver.nv.identity_signature(job["identity"]), result=result))
        rows, summary = driver.export(jobs, args.output_root, args)
        self.assertTrue(all(r["physical_result"] for r in rows))
        self.assertTrue(all(not r["sample_checks_passed"] for r in rows))
        self.assertEqual(summary["refinement_checks"], 1)
        self.assertEqual(summary["refinement_checks_passed"], 0)

    def test_dry_run_and_grid_validation(self):
        with redirect_stdout(io.StringIO()):
            self.assertEqual(driver.main(self.argv(self.arguments())+["--dry-run"]), 0)
        self.assertFalse((self.directory/"output").exists())
        with self.assertRaises(ValueError):
            driver.build_jobs(self.arguments("--orders", "Cr:14"))


if __name__ == "__main__":
    unittest.main()
