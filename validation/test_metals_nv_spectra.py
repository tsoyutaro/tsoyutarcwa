"""Check spectrum averages, incomplete bands, worker continuation and resume."""
from contextlib import redirect_stdout
from copy import deepcopy
import csv
import io
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import run_metals_nv_spectra as spectra


class SpectrumTests(unittest.TestCase):
    def setUp(self):
        self.output=ROOT/"validation"/f"_nv_spectrum_test_{uuid.uuid4().hex}"
        self.output.mkdir()
        self.addCleanup(self.cleanup)
        args=spectra.build_parser().parse_args(["--device","cpu","--orders","Cr:1,Al:1",
              "--wavelengths","400:410:5","--slices","2","--grid","32","--no-plots"])
        self.plan=spectra.build_plan(args)
        self.attempts=[]
        self.failures={}
        self.backends={}
        for symbol,original in spectra.BACKENDS.items():
            self.backends[symbol]=SimpleNamespace(launch_worker=self.fake_worker,validate_case=original.validate_case,
                                                  CoatedGeometry=original.CoatedGeometry)

    def cleanup(self):
        target=self.output.resolve()
        assert target.parent==(ROOT/"validation").resolve() and target.name.startswith("_nv_spectrum_test_")
        shutil.rmtree(target)

    def fake_worker(self,plan_path,result_path,order,wave):
        identity=spectra.read_json(plan_path)["identity"]
        key=(identity["material"],identity["structure"],wave)
        self.attempts.append(key)
        if key in self.failures:
            return dict(status=self.failures[key],wavelength_nm=wave,error="Injected test failure")
        r=.1+(wave-400)*.0001
        coated=identity["structure"]=="pmma"
        t=.05 if coated else 0.
        a=1-r-t
        case=dict(material=identity["material"],structure=identity["structure"],order=order,wavelength_nm=wave,
                  slices=identity["slices"],grid=identity["grid"],total_pattern_layers=identity["slices"]+int(coated),
                  reflectance=r,absorptance=a,transmittance=t,power_into_substrate=.05,
                  motheye_absorptance=None if coated else a-.05,substrate_absorptance=None if coated else .05,
                  passivity_warning=False,layer_method="nvm",nv_lanczos_enabled=False)
        return dict(status="ok",signature=spectra.nv.identity_signature(identity),case=case,
                    environment=dict(torch="test",device="cpu"))

    def run_batch(self):
        with patch.dict(spectra.BACKENDS,self.backends),redirect_stdout(io.StringIO()):
            return spectra.execute(self.plan,self.output)

    def test_exact_request_preflight(self):
        plan=spectra.build_plan(spectra.build_parser().parse_args([]))
        self.assertEqual(plan["wavelengths_nm"],list(range(400,701,5)))
        self.assertEqual(len(plan["jobs"]),16)
        self.assertEqual(len(plan["jobs"])*len(plan["wavelengths_nm"]),976)
        self.assertEqual(plan["orders"],dict(Cr=24,Fe=24,Ni=24,Ti=24,V=24,Al=36,Cu=36,W=36))
        for job in plan["jobs"]:
            identity=job["plan"]["identity"]
            self.assertEqual(identity["geometry"]["period_nm"],200)
            self.assertEqual(identity["geometry"]["height_nm"],500)
            self.assertEqual(identity["slices"],60)
            self.assertEqual(identity["grid"],512)
            if job["material"]=="W":
                self.assertEqual(Path(identity["metal_csv"]).name,"Werner(W).csv")
            if job["structure"]=="pmma":
                self.assertEqual(identity["geometry"]["metal_side_thickness_nm"],30)

    def test_arithmetic_and_trapezoid_means_are_separate(self):
        job=self.plan["jobs"][0]
        cases={str(i):dict(wavelength_nm=w,reflectance=r,transmittance=.05,absorptance=.95-r)
               for i,(w,r) in enumerate(zip((400.,405.,410.),(0.,0.,.9)))}
        mean=spectra.averages(job,cases,[400.,405.,410.])
        self.assertAlmostEqual(mean["mean_R_sample"],.3)
        self.assertAlmostEqual(mean["mean_R_band"],.225)
        self.assertAlmostEqual(mean["mean_R_band"]+mean["mean_A_band"]+mean["mean_T_band"],1)
        self.assertIsNone(mean["mean_moth_eye_A_band"])

    def test_nonuniform_wavelengths_use_correct_interval_weights(self):
        job=self.plan["jobs"][0]
        cases={str(i):dict(wavelength_nm=w,reflectance=r,transmittance=.1,absorptance=.9-r)
               for i,(w,r) in enumerate(zip((400.,500.,700.),(0.,.2,.8)))}
        mean=spectra.averages(job,cases,[400.,500.,700.])
        self.assertAlmostEqual(mean["mean_R_band"],110/300)
        self.assertAlmostEqual(mean["mean_R_sample"],1/3)

    def test_incomplete_band_has_no_final_means(self):
        job=self.plan["jobs"][0]
        cases={"one":dict(wavelength_nm=400.,reflectance=.1,absorptance=.9,transmittance=0.)}
        mean=spectra.averages(job,cases,[400.,405.,410.])
        self.assertFalse(mean["complete"])
        self.assertEqual(mean["completed_points"],1)
        self.assertTrue(all(v is None for k,v in mean.items() if k.startswith("mean_")))

    def test_oom_continues_other_wavelengths_and_structures(self):
        self.failures[("Cr","metal",405.)]="oom"
        states,summary=self.run_batch()
        self.assertEqual(summary["completed_cases"],11)
        self.assertEqual(summary["complete_spectra"],3)
        self.assertEqual(summary["failed_cases"],1)
        self.assertIn(("Cr","metal",410.),self.attempts)
        self.assertIn(("Cr","pmma",410.),self.attempts)
        self.assertIn(("Al","pmma",410.),self.attempts)
        with (self.output/"averages.csv").open(newline="") as stream:
            averages=list(csv.DictReader(stream))
        missing=next(r for r in averages if r["material"]=="Cr" and r["structure"]=="metal")
        self.assertEqual(missing["mean_R_band"],"")

    def test_resume_retries_only_failed_point_and_restores_means(self):
        self.failures[("Cr","metal",405.)]="oom"
        self.run_batch()
        before=len(self.attempts)
        self.failures.clear()
        _,summary=self.run_batch()
        self.assertEqual(len(self.attempts)-before,1)
        self.assertEqual(summary["completed_cases"],12)
        self.assertEqual(summary["complete_spectra"],4)
        self.assertEqual(summary["failed_cases"],0)
        self.assertEqual(summary["status"],"complete")

    def test_ordinary_error_is_not_mislabeled_as_oom(self):
        self.failures[("Al","pmma",405.)]="calculation_failed"
        _,summary=self.run_batch()
        self.assertEqual(summary["failed_cases"],1)
        self.assertIn(("Al","pmma",410.),self.attempts)
        with (self.output/"failed_cases.csv").open(newline="") as stream:
            failure=next(csv.DictReader(stream))
        self.assertEqual(failure["status"],"calculation_failed")

    def test_changed_batch_settings_reject_mixed_results(self):
        self.run_batch()
        self.plan=deepcopy(self.plan)
        self.plan["grid"]=768
        with self.assertRaisesRegex(ValueError,"Saved batch settings/source changed"):
            self.run_batch()

    def test_removing_a_material_reuses_results_and_keeps_excluded_files(self):
        self.run_batch()
        before=list(self.attempts)
        excluded=self.output/"metal/al/checkpoint.json"
        excluded_bytes=excluded.read_bytes()
        self.plan=deepcopy(self.plan)
        self.plan["orders"]={"Cr":1}
        self.plan["jobs"]=[j for j in self.plan["jobs"] if j["material"]=="Cr"]
        _,summary=self.run_batch()
        self.assertEqual(self.attempts,before)
        self.assertEqual(summary["completed_cases"],6)
        self.assertEqual(summary["requested_spectra"],2)
        self.assertEqual(excluded.read_bytes(),excluded_bytes)
        history=list((self.output/"batch_plan_history").glob("*.json"))
        self.assertEqual(len(history),1)
        self.assertIn("Al",spectra.read_json(history[0])["plan"]["orders"])

    def test_adding_a_material_runs_only_the_new_material(self):
        self.run_batch()
        before=len(self.attempts)
        args=spectra.build_parser().parse_args(["--device","cpu","--orders","Cr:1,Al:1,Fe:1",
            "--wavelengths","400:410:5","--slices","2","--grid","32"])
        self.plan=spectra.build_plan(args)
        _,summary=self.run_batch()
        self.assertEqual(len(self.attempts)-before,6)
        self.assertTrue(all(m=="Fe" for m,_,_ in self.attempts[before:]))
        self.assertEqual(summary["completed_cases"],18)

    def test_changed_driver_only_reuses_unchanged_solver_results(self):
        self.run_batch()
        before=list(self.attempts)
        self.plan=deepcopy(self.plan)
        self.plan["driver_sha256"]="updated-orchestration-only"
        _,summary=self.run_batch()
        self.assertEqual(self.attempts,before)
        self.assertEqual(summary["completed_cases"],12)

    def test_changed_order_rejected_before_any_worker_or_batch_update(self):
        self.run_batch()
        before=list(self.attempts)
        batch_bytes=(self.output/"batch_plan.json").read_bytes()
        args=spectra.build_parser().parse_args(["--device","cpu","--orders","Cr:2,Al:1",
            "--wavelengths","400:410:5","--slices","2","--grid","32"])
        self.plan=spectra.build_plan(args)
        with self.assertRaisesRegex(ValueError,"Saved settings/source differ"):
            self.run_batch()
        self.assertEqual(self.attempts,before)
        self.assertEqual((self.output/"batch_plan.json").read_bytes(),batch_bytes)

    def test_changed_solver_source_still_rejected(self):
        self.run_batch()
        before=list(self.attempts)
        self.plan=deepcopy(self.plan)
        self.plan["jobs"][0]["plan"]["identity"]["source_sha256"]["solver_changed.py"]="different"
        with self.assertRaisesRegex(ValueError,"Saved settings/source differ"):
            self.run_batch()
        self.assertEqual(self.attempts,before)

    def test_invalid_wavelength_range_is_rejected(self):
        for value in ("400:700:7","400:700:0","400,400,700","nan,700"):
            with self.assertRaises(ValueError):
                spectra.parse_wavelengths(value)


if __name__=="__main__":
    unittest.main()
