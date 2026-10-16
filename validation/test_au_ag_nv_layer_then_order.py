"""Validate automatic gating, checkpoint reuse and failure handling without CUDA."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_au_ag_nv_layer_then_order as workflow


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / "validation" / f"_nv_workflow_test_{uuid.uuid4().hex}"
        self.directory.mkdir()
        self.addCleanup(self.cleanup)
        self.output, self.reuse = self.directory / "output", self.directory / "reuse"
        self.args = workflow.build_parser().parse_args([
            "--device", "cpu", "--max-order", "8", "--slices", "180,210,240",
            "--reuse-root", str(self.reuse), "--output-root", str(self.output), "--no-plots"])
        self.plan = workflow.build_plan(self.args)
        self.attempts, self.failures, self.overrides = [], {}, {}
        self.original = dict(workflow.grid.BACKENDS)
        self.backends = {name: SimpleNamespace(
            run_job=self.fake_run_job, persist=backend.persist,
            validate_case=backend.validate_case, POWER_NAMES=backend.POWER_NAMES)
            for name, backend in self.original.items()}

    def cleanup(self):
        target = self.directory.resolve()
        assert target.parent == (ROOT / "validation").resolve() and target.name.startswith("_nv_workflow_test_")
        shutil.rmtree(target)

    def fake_run_job(self, plan, output):
        identity = plan["identity"]
        path = output / "checkpoint.json"
        saved = workflow.grid.load_json(path) if path.exists() else dict(cases={})
        saved.update(plan=plan, signature=workflow.grid.nv.identity_signature(identity),
                     status="running", failure=None, environment={"device": "cpu", "torch": "test"})
        wave = identity["wavelengths_nm"][0]
        for m in range(identity["start_order"], plan["max_order"] + 1, identity["step"]):
            key = workflow.grid.nv.case_key(m, wave)
            if key in saved["cases"]:
                continue
            condition = (identity["material"], identity["structure"], identity["slices"], m)
            self.attempts.append(condition)
            if condition in self.failures:
                saved.update(status=self.failures[condition], failure=dict(order=m, error="Injected failure"))
                return self.original[identity["material"]].persist(output, saved)
            coated = identity["structure"] == "pmma"
            r, t = self.overrides.get(condition, (.1 + identity["slices"] * .000001 + m * .000001,
                                                .01 if coated else 0.))
            a = 1 - r - t
            saved["cases"][key] = dict(material=identity["material"], structure=identity["structure"],
                wavelength_nm=wave, order=m, slices=identity["slices"], grid=identity["grid"],
                total_pattern_layers=identity["slices"] + int(coated), reflectance=r, transmittance=t,
                absorptance=a, motheye_absorptance=None if coated else a - .02,
                substrate_absorptance=None if coated else .02, power_into_substrate=t if coated else .02,
                layer_method="nvm", nv_lanczos_enabled=False, passivity_warning=False)
            self.original[identity["material"]].persist(output, saved)
        saved["status"] = "requested_range_complete"
        return self.original[identity["material"]].persist(output, saved)

    def run_workflow(self, **kwargs):
        with patch.dict(workflow.grid.BACKENDS, self.backends), redirect_stdout(io.StringIO()):
            return workflow.execute(self.plan, self.output, no_plots=True, **kwargs)

    def test_default_conditions_and_two_layer_refinements(self):
        plan = workflow.build_plan(workflow.build_parser().parse_args([]))
        self.assertEqual(plan["layer_plan"]["orders"], [36])
        self.assertEqual(plan["layer_plan"]["slices"], [180, 210, 240])
        self.assertEqual(plan["order_plan"]["orders"], list(range(4, 37, 2)))
        self.assertEqual(len(plan["order_plan"]["jobs"]), 4)
        self.assertEqual(plan["order_plan"]["slices"], [240])

    def test_passed_groups_sweep_and_reuse_the_highest_order(self):
        summary = self.run_workflow()
        self.assertEqual(summary["layer_gate_passed"], 4)
        self.assertEqual(len(self.attempts), 12 + 4 * 2)
        self.assertEqual(sum(nz == 240 and m == 8 for _, _, nz, m in self.attempts), 4)
        self.assertTrue(all(g["order_range_complete"] for g in summary["groups"]))
        self.assertTrue(all(g["order_tail_R_A_passed"] for g in summary["groups"]))

    def test_first_pair_failure_is_not_hidden_by_second_pair_pass(self):
        self.overrides[("Au", "pmma", 180, 8)] = (.10, .01)
        self.overrides[("Au", "pmma", 210, 8)] = (.12, .01)
        self.overrides[("Au", "pmma", 240, 8)] = (.1201, .01)
        summary = self.run_workflow()
        group = next(g for g in summary["groups"] if g["material"] == "Au" and g["structure"] == "pmma")
        self.assertEqual(group["layer_status"], "not_converged")
        self.assertEqual(group["order_status"], "skipped_layer_gate")
        self.assertNotIn(("Au", "pmma", 240, 4), self.attempts)
        self.assertIn(("Ag", "pmma", 240, 4), self.attempts)

    def test_A_is_checked_independently_of_R(self):
        self.overrides[("Ag", "pmma", 180, 8)] = (.1, .01)
        self.overrides[("Ag", "pmma", 210, 8)] = (.1, .02)
        self.overrides[("Ag", "pmma", 240, 8)] = (.1, .0201)
        summary = self.run_workflow()
        group = next(g for g in summary["groups"] if g["material"] == "Ag" and g["structure"] == "pmma")
        self.assertFalse(group["layer_gate_passed"])

    def test_layer_OOM_skips_higher_Nz_and_other_groups_continue(self):
        self.failures[("Au", "metal", 210, 8)] = "oom"
        summary = self.run_workflow()
        self.assertNotIn(("Au", "metal", 240, 8), self.attempts)
        self.assertNotIn(("Au", "metal", 240, 4), self.attempts)
        self.assertIn(("Ag", "pmma", 240, 4), self.attempts)
        self.assertEqual(summary["layer_gate_passed"], 3)

    def test_order_failure_stops_new_higher_orders_and_continues(self):
        self.failures[("Au", "metal", 240, 4)] = "calculation_failed"
        summary = self.run_workflow()
        self.assertNotIn(("Au", "metal", 240, 6), self.attempts)
        self.assertIn(("Ag", "pmma", 240, 6), self.attempts)
        group = next(g for g in summary["groups"] if g["material"] == "Au" and g["structure"] == "metal")
        self.assertEqual(group["order_status"], "calculation_failed")
        self.assertFalse(group["order_range_complete"])

    def test_setup_error_is_recorded_and_not_retried_implicitly(self):
        def setup_error(plan, output):
            identity = plan["identity"]
            if identity["structure"] == "metal" and identity["slices"] == 210:
                self.attempts.append(("Au", "metal", 210, 8))
                raise RuntimeError("Injected setup error")
            return self.fake_run_job(plan, output)
        with patch.object(self.backends["Au"], "run_job", side_effect=setup_error):
            first = self.run_workflow()
            before = list(self.attempts)
            self.run_workflow()
        self.assertEqual(before, self.attempts)
        self.assertEqual(next(g for g in first["groups"] if g["material"] == "Au" and
                              g["structure"] == "metal")["layer_status"], "job_setup_failed")
        self.assertEqual(self.run_workflow(retry_failures=True)["layer_gate_passed"], 4)

    def test_resume_and_explicit_retry_preserve_successful_cases(self):
        self.failures[("Au", "metal", 210, 8)] = "oom"
        self.run_workflow()
        before = list(self.attempts)
        self.run_workflow()
        self.assertEqual(self.attempts, before)
        self.failures.clear()
        summary = self.run_workflow(retry_failures=True)
        self.assertEqual(summary["layer_gate_passed"], 4)
        self.assertEqual(len(self.attempts) - len(before), 4)

    def test_external_reuse_checks_identity_and_preserves_provenance(self):
        job = self.plan["layer_plan"]["jobs"][0]
        path = self.reuse / "compatible"
        self.fake_run_job(job["plan"], path)
        other = deepcopy(job["plan"])
        other["identity"]["grid"] = 768
        self.fake_run_job(other, self.reuse / "wrong_grid")
        self.attempts.clear()
        summary = self.run_workflow()
        self.assertNotIn(("Au", "metal", 180, 8), self.attempts)
        self.assertTrue(any(e["source"] == str((path / "checkpoint.json").resolve())
                            for e in summary["reused_sources"]))
        self.assertFalse(any("wrong_grid" in e["source"] for e in summary["reused_sources"]))

    def test_conflicting_cached_results_are_recomputed(self):
        job = self.plan["layer_plan"]["jobs"][0]
        self.fake_run_job(job["plan"], self.reuse / "a")
        self.overrides[("Au", "metal", 180, 8)] = (.2, 0.)
        self.fake_run_job(job["plan"], self.reuse / "b")
        self.attempts.clear()
        summary = self.run_workflow()
        self.assertIn(("Au", "metal", 180, 8), self.attempts)
        self.assertTrue(any("Conflicting" in n["reason"] for n in summary["cache_notes"]))

    def test_changed_workflow_rejects_mixed_resume(self):
        self.run_workflow()
        self.plan = deepcopy(self.plan)
        self.plan["order_plan"]["tolerance"] = .004
        with self.assertRaisesRegex(ValueError, "Workflow settings/source changed"):
            self.run_workflow()

    def test_missing_results_cannot_pass_the_gate(self):
        passed, comparisons = workflow.layer_gate({}, [180, 210, 240], 36, 700., .005)
        self.assertFalse(passed)
        self.assertTrue(all(c["delta_R_pp"] is None and not c["passed"] for c in comparisons))

    def test_orders_only_runs_all_groups_at_one_Nz_without_claiming_layer_convergence(self):
        self.args.orders_only, self.args.slices = True, "210"
        self.plan = workflow.build_plan(self.args)
        self.assertEqual(self.plan["layer_plan"]["jobs"], [])
        summary = self.run_workflow()
        self.assertEqual(len(self.attempts), 12)
        self.assertEqual({nz for _, _, nz, _ in self.attempts}, {210})
        self.assertEqual(summary["layer_gate_passed"], 0)
        self.assertTrue(summary["orders_only"])
        self.assertTrue(all(g["order_range_complete"] and g["layer_status"] == "not_checked_orders_only"
                            and not g["layer_gate_passed"] for g in summary["groups"]))

    def test_orders_only_OOM_stops_new_higher_orders_and_continues_other_groups(self):
        self.args.orders_only, self.args.slices = True, "210"
        self.plan = workflow.build_plan(self.args)
        self.failures[("Au", "metal", 210, 6)] = "oom"
        summary = self.run_workflow()
        self.assertNotIn(("Au", "metal", 210, 8), self.attempts)
        self.assertIn(("Ag", "pmma", 210, 8), self.attempts)
        self.assertEqual(sum(g["order_range_complete"] for g in summary["groups"]), 3)

    def test_orders_only_reuses_previously_checked_high_order(self):
        job = next(j for j in self.plan["layer_plan"]["jobs"]
                   if j["material"] == "Au" and j["structure"] == "metal" and j["slices"] == 210)
        self.fake_run_job(job["plan"], self.reuse / "old_layer_check")
        self.attempts.clear()
        self.args.orders_only, self.args.slices = True, "210"
        self.plan = workflow.build_plan(self.args)
        self.run_workflow()
        self.assertEqual(len(self.attempts), 11)
        self.assertNotIn(("Au", "metal", 210, 8), self.attempts)

    def test_orders_only_requires_one_fixed_slice_count(self):
        self.args.orders_only = True
        with self.assertRaisesRegex(ValueError, "exactly one slice count"):
            workflow.build_plan(self.args)


if __name__ == "__main__":
    unittest.main()
