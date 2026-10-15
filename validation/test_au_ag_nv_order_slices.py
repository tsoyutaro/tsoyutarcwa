"""Exercise failure pruning/resume without allocating a GPU or changing physics."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import sys
import shutil
import uuid
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import run_au_ag_nv_order_slices as sweep


class GridTests(unittest.TestCase):
    def setUp(self):
        self.output = ROOT/"validation"/f"_nv_grid_test_{uuid.uuid4().hex}"
        self.output.mkdir()
        self.addCleanup(self.cleanup_output)
        args = sweep.build_parser().parse_args(["--device","cpu","--no-plots"])
        self.plan = sweep.build_plan(args)
        self.attempts = []
        self.failures = {}
        self.backends = {}
        for symbol,original in sweep.BACKENDS.items():
            self.backends[symbol] = SimpleNamespace(run_job=self.fake_run_job,validate_case=original.validate_case)

    def cleanup_output(self):
        target=self.output.resolve()
        assert target.parent==(ROOT/"validation").resolve() and target.name.startswith("_nv_grid_test_")
        shutil.rmtree(target)

    def fake_run_job(self, plan, output):
        identity = plan["identity"]
        checkpoint = output/"checkpoint.json"
        saved = sweep.load_json(checkpoint) if checkpoint.exists() else dict(cases={})
        saved.update(signature=sweep.nv.identity_signature(identity),plan=plan,status="running",failure=None)
        for order in range(identity["start_order"],plan["max_order"]+1,identity["step"]):
            wave = identity["wavelengths_nm"][0]
            key = sweep.nv.case_key(order,wave)
            if key in saved["cases"]:
                continue
            condition = (identity["material"],identity["structure"],identity["slices"],order,wave)
            self.attempts.append(condition)
            if condition in self.failures:
                saved.update(status=self.failures[condition],failure=dict(order=order,error="Injected test failure"))
                sweep.nv.write_json(checkpoint,saved)
                return dict(status=saved["status"],failure=saved["failure"])
            coated = identity["structure"]=="pmma"
            r = .01+order*.00001+identity["slices"]*.000001
            t = .02 if coated else 0.
            a = 1-r-t
            saved["cases"][key] = dict(material=identity["material"],structure=identity["structure"],
                order=order,wavelength_nm=wave,slices=identity["slices"],grid=identity["grid"],
                total_pattern_layers=identity["slices"]+int(coated),
                reflectance=r,transmittance=t,absorptance=a,
                power_into_substrate=.02,motheye_absorptance=None if coated else a-.02,
                substrate_absorptance=None if coated else .02,
                layer_method="nvm",nv_lanczos_enabled=False,passivity_warning=False)
            sweep.nv.write_json(checkpoint,saved)
        saved["status"]="requested_range_complete"
        sweep.nv.write_json(checkpoint,saved)
        return dict(status=saved["status"],failure=None)

    def run_grid(self, **kwargs):
        with patch.dict(sweep.BACKENDS,self.backends), redirect_stdout(io.StringIO()):
            return sweep.execute(self.plan,self.output,**kwargs)

    def test_preflight_has_exact_cartesian_grid_and_prior_materials(self):
        self.assertEqual(self.plan["orders"],[30,32,34,36])
        self.assertEqual(self.plan["slices"],[60,90,120])
        self.assertEqual(len(self.plan["jobs"])*len(self.plan["orders"]),48)
        for job in self.plan["jobs"]:
            identity=job["plan"]["identity"]
            self.assertEqual(identity["geometry"]["period_nm"],200)
            self.assertEqual(identity["geometry"]["height_nm"],500)
            self.assertEqual(identity["storage"],"reduced")
            if job["material"]=="Au":
                epsilon=identity["anchor_epsilon"]["700.0"]
                self.assertAlmostEqual(epsilon["real"],-16.500759)
            if job["structure"]=="pmma":
                self.assertEqual(identity["geometry"]["metal_side_thickness_nm"],30)

    def test_oom_prunes_only_dominated_conditions_and_continues(self):
        self.failures[("Au","metal",90,34,700.)]="oom"
        cases,summary=self.run_grid()
        self.assertEqual(summary["status"],"completed_with_oom_skips")
        self.assertEqual(summary["counts"]["ok"],44)
        self.assertEqual(summary["counts"]["oom"],1)
        self.assertEqual(summary["counts"]["skipped_dominated"],3)
        self.assertEqual(summary["counts"]["pending"],0)
        self.assertIn(("Au","metal",120,32,700.),self.attempts)
        self.assertNotIn(("Au","metal",90,36,700.),self.attempts)
        self.assertNotIn(("Au","metal",120,34,700.),self.attempts)
        self.assertIn(("Au","pmma",120,36,700.),self.attempts)
        self.assertIn(("Ag","metal",120,36,550.),self.attempts)
        self.assertIn(("Ag","pmma",120,36,550.),self.attempts)
        self.assertEqual(len(cases),44)

    def test_failure_at_first_condition_still_runs_other_structures(self):
        self.failures[("Au","metal",60,30,700.)]="oom"
        _,summary=self.run_grid()
        self.assertEqual(summary["counts"]["ok"],36)
        self.assertEqual(summary["counts"]["skipped_dominated"],11)
        self.assertEqual(sum(c[:2]==("Au","metal") for c in self.attempts),1)

    def test_resume_preserves_oom_frontier_without_retrying(self):
        self.failures[("Au","metal",90,34,700.)]="oom"
        self.run_grid()
        before=list(self.attempts)
        _,summary=self.run_grid()
        self.assertEqual(self.attempts,before)
        self.assertEqual(summary["counts"]["ok"],44)
        self.assertEqual(summary["counts"]["oom"],1)

    def test_explicit_retry_recovers_only_missing_conditions(self):
        self.failures[("Au","metal",90,34,700.)]="oom"
        self.run_grid()
        before=len(self.attempts)
        self.failures.clear()
        _,summary=self.run_grid(retry_failures=True)
        self.assertEqual(summary["status"],"requested_grid_complete")
        self.assertEqual(summary["counts"]["ok"],48)
        self.assertEqual(len(self.attempts)-before,4)

    def test_nonphysical_failure_is_distinct_from_oom_and_continues(self):
        self.failures[("Ag","metal",90,32,550.)]="calculation_failed"
        _,summary=self.run_grid()
        self.assertEqual(summary["status"],"completed_with_failures")
        self.assertEqual(summary["counts"]["calculation_failed"],1)
        self.assertEqual(summary["counts"]["oom"],0)
        self.assertIn(("Ag","pmma",120,36,550.),self.attempts)

    def test_changed_settings_reject_mixed_resume(self):
        self.run_grid()
        self.plan=deepcopy(self.plan)
        self.plan["tolerance"]*=2
        with self.assertRaisesRegex(ValueError,"Batch settings/source changed"):
            self.run_grid()

    def test_missing_points_are_not_compared_or_called_converged(self):
        self.failures[("Au","metal",90,34,700.)]="oom"
        self.run_grid()
        import csv
        with (self.output/"comparison_changes.csv").open(newline="") as stream:
            changes=list(csv.DictReader(stream))
        for change in changes:
            if change["material"]=="Au" and change["structure"]=="metal":
                if change["axis"]=="order" and change["fixed_slices"] in ("90","120"):
                    self.assertEqual((change["low"],change["high"]),("30","32"))
                if change["axis"]=="slices":
                    self.assertNotEqual((change["low"],change["high"]),("60","120"))


if __name__=="__main__":
    unittest.main()
