"""Recovery, independent OOM stopping, and convergence validity for the driver."""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from . import run_order_until_oom as sweep


def row(order, lattice="square", method="nv", wavelength=400., r=.01, t=None):
    t = 1 - r if t is None else t
    return {"order": order, "slices": 15, "grid": 64, "lattice": lattice, "method": method,
            "wavelength_nm": wavelength, "reflectance": r, "transmittance": t,
            "absorptance": 1-r-t, "energy_residual": r+t-1,
            "runtime_seconds": .1, "physical": True}


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name) / "results"
        self.args = sweep.parser().parse_args(["--device", "cpu", "--grid", "64",
            "--output-root", str(self.output), "--no-plots"])
        self.out, self.saved, self.keys = sweep.prepare(self.args)

    def run_group(self, key, outcomes):
        with patch.object(sweep, "launch", side_effect=outcomes) as launch, contextlib.redirect_stdout(io.StringIO()):
            sweep.run_series(self.out, self.saved, key, self.args)
        return launch.call_count

    def test_all_twelve_series_stop_independently_and_continue(self):
        self.assertEqual(len(self.keys), 12)
        for key in self.keys:
            group = self.saved["series"][key]
            completed = row(4, group["lattice"], group["method"], group["wavelength_nm"])
            calls = self.run_group(key, [{"status": "completed", "result": completed},
                                         {"status": "cuda_oom", "error": "allocation failed"}])
            self.assertEqual(calls, 2)
            self.assertEqual(group["status"], "stopped_cuda_oom")
            self.assertEqual(group["failed_order"], 6)
            self.assertEqual(set(group["cases"]), {"4"})
        report = sweep.save(self.out, self.saved)
        self.assertEqual(len(report["series"]), 12)
        self.assertTrue(all(g["first_oom_order"] == 6 for g in report["series"].values()))
        self.assertEqual(len(report["method_comparison_at_highest_common_order"]), 6)
        self.assertTrue((self.out / "square/nv/400nm/cases.csv").exists())

    def test_oom_is_skipped_on_resume_and_explicit_retry_restarts_failed_order(self):
        key = self.keys[0]
        self.run_group(key, [{"status": "completed", "result": row(4)}, {"status": "cpu_oom", "error": "oom"}])
        self.assertEqual(self.run_group(key, []), 0)
        self.args.retry_oom, self.args.max_order = True, 6
        self.assertEqual(self.run_group(key, [{"status": "completed", "result": row(6)}]), 1)
        self.assertEqual(set(self.saved["series"][key]["cases"]), {"4", "6"})
        self.assertEqual(self.saved["series"][key]["status"], "reached_max_order")
        self.assertIsNone(sweep.save(self.out, self.saved)["series"][key]["first_oom_order"])

    def test_capped_run_resumes_at_next_order_without_repeating_completed_cases(self):
        key = self.keys[0]
        self.args.max_order = 4
        self.run_group(key, [{"status": "completed", "result": row(4)}])
        self.args.max_order = 6
        self.assertEqual(self.run_group(key, [{"status": "completed", "result": row(6)}]), 1)
        self.assertEqual(set(self.saved["series"][key]["cases"]), {"4", "6"})

    def test_other_error_is_not_misreported_as_oom_and_next_series_runs(self):
        self.run_group(self.keys[0], [{"status": "error", "error": "eigensolver failure"}])
        group = sweep.save(self.out, self.saved)["series"][self.keys[0]]
        self.assertEqual(group["status"], "worker_error")
        self.assertIsNone(group["first_oom_order"])
        self.args.max_order = 4
        self.run_group(self.keys[1], [{"status": "completed", "result": row(4, wavelength=550.)}])
        self.assertEqual(self.saved["series"][self.keys[1]]["status"], "reached_max_order")

    def test_fixed_grid_guard_is_separate_from_oom(self):
        self.saved["plan"]["grid_max_order"] = 4
        self.run_group(self.keys[0], [{"status": "completed", "result": row(4)}])
        group = sweep.save(self.out, self.saved)["series"][self.keys[0]]
        self.assertEqual(group["status"], "stopped_grid_limit")
        self.assertIsNone(group["first_oom_order"])

    def test_interrupt_preserves_completed_result_and_propagates(self):
        key = self.keys[0]
        with self.assertRaises(KeyboardInterrupt):
            self.run_group(key, [{"status": "completed", "result": row(4)}, KeyboardInterrupt()])
        saved = json.loads((self.out / "checkpoint.json").read_text())
        self.assertEqual(saved["series"][key]["status"], "interrupted")
        self.assertIn("4", saved["series"][key]["cases"])

    def test_changed_grid_rejects_cache_but_larger_cap_is_allowed(self):
        sweep.save(self.out, self.saved)
        self.args.max_order = 12
        sweep.prepare(self.args)
        self.args.grid = 96
        with self.assertRaisesRegex(ValueError, "signature differs"):
            sweep.prepare(self.args)

    def test_missing_worker_result_is_error_even_if_process_was_killed(self):
        process = SimpleNamespace(returncode=-9, wait=lambda: None)
        with patch.object(sweep.subprocess, "Popen", return_value=process):
            outcome = sweep.launch(self.out, {"plan": self.saved["plan"]}, "killed")
        self.assertEqual(outcome["status"], "error")
        self.assertIn("not a confirmed OOM", outcome["error"])

    def test_oom_classification_is_strict(self):
        device = SimpleNamespace(type="cuda")
        self.assertEqual(sweep.is_oom(RuntimeError("CUDA out of memory. Tried to allocate"), None, device), "cuda_oom")
        self.assertIsNone(sweep.is_oom(RuntimeError("CUDA error: illegal memory access"), None, device))
        self.assertEqual(sweep.is_oom(MemoryError(), None, device), "cpu_oom")
        self.assertEqual(sweep.is_oom(RuntimeError("DefaultCPUAllocator: not enough memory"), None, device), "cpu_oom")
        self.assertIsNone(sweep.is_oom(RuntimeError("some memory-related numerical problem"), None, device))

    def test_no_false_convergence_from_drift_nonphysical_values_or_late_instability(self):
        stable = [row(4, r=.01), row(6, r=.01001), row(8, r=.01002)]
        self.assertEqual(sweep.assess(stable, .00025, 1e-6)["selected_order"], 6)
        drift = [row(2*i+4, r=.01+.0002*i) for i in range(5)]
        self.assertEqual(sweep.assess(drift, .00025, 1e-6)["selected_order"], 10)
        # An early candidate is rejected when many small steps accumulate.
        self.assertNotEqual(sweep.assess(drift, .00025, 1e-6)["selected_order"], 6)
        stable.append(row(10, r=.03))
        self.assertIsNone(sweep.assess(stable, .00025, 1e-6)["selected_order"])
        lossy = [row(m, t=.98) for m in (4, 6, 8)]
        self.assertEqual(sweep.assess(lossy, .00025, 1e-6)["nonphysical_orders"], [4, 6, 8])
        self.assertIsNone(sweep.assess(lossy, .00025, 1e-6)["selected_order"])


if __name__ == "__main__":
    unittest.main()
