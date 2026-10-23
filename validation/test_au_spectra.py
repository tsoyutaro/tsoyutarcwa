"""Check the Au spectrum grid, exact reference reuse, averages and continuation."""
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
import run_au_spectra as spectra


class AuSpectraTests(unittest.TestCase):
    def arguments(self, base, *extra):
        return ["--device", "cpu", "--order", "2", "--slices", "4", "--grid", "32",
            "--wavelengths", "400,550,700", "--reference-root", str(Path(base)/"source"),
            "--output-root", str(Path(base)/"output"), "--no-plots", *extra]

    def ok(self, job, reflectance=.2):
        coated = job["structure"] == "pmma"
        t = .1 if coated else 0.
        a = 1-reflectance-t
        row = dict(material="Au", structure=job["structure"], order=job["order"], wavelength_nm=job["wave"],
            slices=job["identity"]["slices"], grid=job["identity"]["grid"],
            total_pattern_layers=job["identity"]["slices"]+int(coated), layer_method="nvm",
            nv_lanczos_enabled=False, passivity_warning=False, reflectance=reflectance, transmittance=t,
            absorptance=a, power_into_substrate=.1, motheye_absorptance=None if coated else a-.1,
            substrate_absorptance=None if coated else .1)
        return dict(status="ok", signature=spectra.nv.identity_signature(job["identity"]), case=row, environment={"device":"test"})

    def run_quiet(self, argv):
        with redirect_stdout(io.StringIO()):
            return spectra.main(argv)

    def test_default_band_and_both_methods_have_244_conditions(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"validation") as base:
            args = spectra.build_parser().parse_args(["--factorizations", "symmetric-nv,galerkin",
                "--output-root", str(Path(base)/"output")])
            waves, jobs, current, _, output = spectra.prepare(args)
            self.assertEqual(waves, list(range(400, 701, 5)))
            self.assertEqual(len(jobs), 244)
            self.assertEqual(len({spectra.case_key(j) for j in jobs}), 244)
            self.assertEqual(current, {})
            for job in jobs:
                self.assertEqual((job["order"], job["identity"]["slices"], job["identity"]["grid"]), (36,210,512))
                g = job["identity"]["geometry"]
                self.assertEqual((g["period_nm"],g["height_nm"]), (200,500))
                self.assertEqual(len(job["identity"]["anchor_epsilon"]), 1)
                if job["structure"] == "pmma":
                    self.assertEqual(g["metal_side_thickness_nm"],30)
            self.assertFalse(output.exists())

    def test_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"validation") as base:
            self.assertEqual(self.run_quiet(self.arguments(base, "--dry-run")), 0)
            self.assertFalse((Path(base)/"output").exists())

    def test_nv_reference_is_not_recomputed_or_imported_as_galerkin(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"validation") as base:
            argv = self.arguments(base, "--structures", "metal", "--factorizations", "symmetric-nv,galerkin")
            args = spectra.build_parser().parse_args(argv)
            _, jobs, _, _, _ = spectra.prepare(args)
            original = deepcopy(jobs[0]["identity"])
            original.pop("factorization")
            original.pop("spectrum_sources")
            original["metal_csv"] = "/another/host/au_measured_nk.csv"
            original["start_order"] = 1
            original["step"] = 1
            source = args.reference_root/"order_sweep/metal/au/wl400/Nz4/checkpoint.json"
            spectra.nv.write_json(source, dict(plan={"identity":original}, signature=spectra.nv.identity_signature(original),
                cases={spectra.nv.case_key(2,400):self.ok(jobs[0])["case"]}))
            before = source.read_bytes()
            with patch.object(spectra, "launch", side_effect=lambda j,p,r:self.ok(j)) as launch:
                self.assertEqual(self.run_quiet(argv),0)
                self.assertEqual(launch.call_count,5)
                self.assertNotIn(("symmetric-nv",400),[(c.args[0]["method"],c.args[0]["wave"]) for c in launch.call_args_list])
                self.assertEqual(self.run_quiet(argv),0)
                self.assertEqual(launch.call_count,5)
            self.assertEqual(source.read_bytes(),before)

    def test_band_and_sample_means_are_distinct_and_invalid_band_has_no_mean(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"validation") as base:
            args = spectra.build_parser().parse_args(self.arguments(base, "--structures","pmma","--wavelengths","400,450,700"))
            waves,jobs,_,_,output = spectra.prepare(args)
            output.mkdir()
            reports = {spectra.case_key(j):spectra.decorate(j,self.ok(j,r)) for j,r in zip(jobs,(.2,.4,.8))}
            _,means,_ = spectra.export(waves,jobs,reports,output)
            self.assertAlmostEqual(means[0]["mean_R_band"],.55)
            self.assertAlmostEqual(means[0]["mean_R_sample"],1.4/3)
            last = jobs[-1]
            reports[spectra.case_key(last)] = spectra.decorate(last,dict(status="nonphysical",
                power_diagnostics=dict(reflectance=1.2,transmittance=.1,absorptance=-.3)))
            rows,means,_ = spectra.export(waves,jobs,reports,output)
            self.assertFalse(means[0]["complete"])
            self.assertIsNone(means[0]["mean_R_band"])
            self.assertAlmostEqual(rows[-1]["A_percent"],-30)

    def test_oom_and_nonphysical_results_continue_to_other_waves_and_method(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"validation") as base:
            argv = self.arguments(base,"--structures","metal","--factorizations","symmetric-nv,galerkin")
            def worker(job,plan,result):
                if job["method"] == "symmetric-nv" and job["wave"] == 400:
                    return dict(status="oom",error="mock CUDA OOM")
                if job["method"] == "symmetric-nv" and job["wave"] == 550:
                    return dict(status="calculation_failed",error="Nonphysical power result",
                        power_diagnostics=dict(reflectance=1.1,transmittance=0.,absorptance=-.1))
                return self.ok(job)
            with patch.object(spectra,"launch",side_effect=worker) as launch:
                self.assertEqual(self.run_quiet(argv),2)
                self.assertEqual(launch.call_count,6)
                self.assertEqual(self.run_quiet(argv),2)
                self.assertEqual(launch.call_count,6)
            summary = spectra.tables.read_json(Path(base)/"output/summary.json")
            self.assertEqual(summary["counts"],{"nonphysical":1,"ok":4,"oom":1})
            self.assertFalse(summary["spectra"][0]["complete"])
            self.assertTrue(summary["spectra"][1]["complete"])

    def test_modified_report_source_and_worker_source_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"validation") as base:
            argv = self.arguments(base,"--structures","metal")
            args = spectra.build_parser().parse_args(argv)
            with patch.object(spectra,"launch",side_effect=lambda j,p,r:self.ok(j)):
                self.assertEqual(self.run_quiet(argv),0)
            _,jobs,_,_,output = spectra.prepare(args)
            job=jobs[0]
            path=output/job["directory"]/"report.json"
            report=spectra.tables.read_json(path)
            report["identity"]["geometry"]["height_nm"]=400
            spectra.nv.write_json(path,report)
            with self.assertRaises(ValueError):
                spectra.prepare(args)
            identity=deepcopy(job["identity"])
            identity["spectrum_sources"][Path(spectra.__file__).name]="changed"
            plan,result=Path(base)/"plan.json",Path(base)/"result.json"
            spectra.nv.write_json(plan,{"identity":identity})
            with patch.object(spectra.backend,"worker") as launch:
                self.assertEqual(spectra.worker(plan,result,2,400),1)
                launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
