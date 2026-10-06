"""Check partial-result safety on CUDA OOM, other errors, and resumption."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from studies.gold_motheye3 import converge, run_until_oom as probe


def row(order, reflectance=.7):
    return {"order": order, "slices": 140, "grid": 576, "wavelength_nm": 700.,
            "axis": "order", "value": order, "reflectance": reflectance,
            "power_into_substrate": .05, "motheye_absorptance": .95-reflectance,
            "substrate_absorptance": .05, "absorptance_total": 1-reflectance,
            "transmittance_far": 0., "symmetry_reduction": "D6-E1-source-row",
            "passivity_warning": False, "runtime_seconds": 1., "wall_seconds": 1.,
            "peak_cuda_allocated_bytes": 1024**3, "peak_cuda_reserved_bytes": 2*1024**3}


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.output = self.directory / "probe"
        sources = converge._source_hashes()
        sources["studies/gold_motheye3/run_memory_safe.py"] = converge._digest(probe.HERE / "run_memory_safe.py")
        material = probe.HERE / "data/au_measured_nk.csv"
        plan = {"version": 1, "axis": "order", "values": [4, 6], "wavelengths_nm": [700.],
                "fixed_numerics": {"order": 18, "slices": 140, "grid": 576},
                "geometry": copy.deepcopy(converge.GEOMETRY), "source_sha256": sources,
                "material": {"model": "measured_csv", "path": str(material),
                             "sha256": converge._digest(material)}, "tolerance": .005,
                "solver": {"cascade": "redheffer", "dtype": "complex128", "smatrix_size": "half",
                           "symmetry_reduction": "d6-source", "requested_device": "cuda",
                           "layer_storage": probe.STORAGE}}
        self.seed = self.directory / "seed.json"
        converge._write_json(self.seed, {"plan": plan, "signature": converge._signature(plan),
                                        "cases": {"4|700": row(4), "6|700": row(6)},
                                        "storage_parity": {"passed": True}})

    def run_main(self, outcomes=None, *extra):
        argv = [str(probe.__file__), "--seed-checkpoint", str(self.seed),
                "--output-dir", str(self.output), *extra]
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            if outcomes is None:
                return probe.main()
            with patch.object(probe, "launch", side_effect=outcomes) as mocked:
                return probe.main(), mocked.call_count

    def test_completed_case_retained_and_oom_exits_zero(self):
        original = self.seed.read_bytes()
        success = {"status": "completed", "order": 8, "result": row(8, .71),
                   "runtime_environment": None, "process_wall_seconds": 2.}
        failure = {"status": "cuda_oom", "order": 10, "error": "test CUDA OOM",
                   "process_wall_seconds": 3.}
        code, calls = self.run_main([success, failure])
        self.assertEqual((code, calls), (0, 2))
        saved = probe.read_checkpoint(self.output / "checkpoint.json")
        self.assertIn("8|700", saved["cases"])
        self.assertNotIn("10|700", saved["cases"])
        report = json.loads((self.output / "report.json").read_text())
        self.assertEqual(report["status"], "stopped_cuda_oom")
        self.assertEqual(report["highest_completed_order"], 8)
        self.assertEqual(report["first_oom_order"], 10)
        self.assertEqual(original, self.seed.read_bytes())
        svg = (self.output / "reflectance_vs_order_700nm.svg").read_text()
        self.assertIn("failed M=10 (no R value)", svg)
        self.assertTrue((self.output / "cases.csv").is_file())
        # A repeated invocation reports results instead of repeatedly crashing.
        with patch.object(probe, "launch") as mocked:
            self.assertEqual(self.run_main(), 0)
            mocked.assert_not_called()
        # An explicit retry starts at the failed order, never redoing M=8.
        retry = {"status": "completed", "order": 10, "result": row(10),
                 "runtime_environment": None, "process_wall_seconds": 2.}
        self.assertEqual(self.run_main([retry], "--retry-oom", "--max-order", "10"), (0, 1))
        self.assertIn("10|700", probe.read_checkpoint(self.output / "checkpoint.json")["cases"])

    def test_non_oom_is_an_error_with_partial_figures(self):
        self.assertEqual(self.run_main([{"status": "error", "order": 8,
                                        "error": "unexpected eigensolver failure"}]), (1, 1))
        report = json.loads((self.output / "report.json").read_text())
        self.assertEqual(report["status"], "worker_error")
        self.assertIsNone(report["first_oom_order"])
        self.assertTrue((self.output / "reflectance_vs_order_700nm.svg").exists())

    def test_prepare_and_report_need_no_gpu(self):
        self.assertEqual(self.run_main(None, "--prepare-only"), 0)
        self.assertEqual(self.run_main(None, "--report-only"), 0)

    def test_reject_changed_sources_before_new_solve(self):
        seed = json.loads(self.seed.read_text())
        seed["plan"]["source_sha256"]["rcwa_ext/asr.py"] = "changed"
        seed["signature"] = converge._signature(seed["plan"])
        converge._write_json(self.seed, seed)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as stopped:
            self.run_main()
        self.assertEqual(stopped.exception.code, 2)
        self.assertFalse((self.output / "checkpoint.json").exists())

    def test_worker_catches_torch_cuda_oom(self):
        class FakeOOM(RuntimeError):
            pass
        cuda = types.SimpleNamespace(is_available=lambda: True, get_device_name=lambda _: "test GPU",
                                     mem_get_info=lambda _: (123, 456),
                                     max_memory_allocated=lambda _: 333, max_memory_reserved=lambda _: 444)
        fake = types.ModuleType("torch")
        fake.__version__, fake.version, fake.cuda = "test", types.SimpleNamespace(cuda="test"), cuda
        fake.OutOfMemoryError = FakeOOM
        fake.device = lambda _: types.SimpleNamespace(type="cuda")
        seed = json.loads(self.seed.read_text())
        request, result = self.directory / "request.json", self.directory / "result.json"
        converge._write_json(request, {"plan": seed["plan"], "order": 8})
        module = types.ModuleType("studies.gold_motheye3.run_memory_safe")
        def fail(*args):
            raise FakeOOM("CUDA out of memory; test only")
        module.measure = fail
        with patch.dict(sys.modules, {"torch": fake, "studies.gold_motheye3.run_memory_safe": module}):
            self.assertEqual(probe.worker(request, result), 0)
        outcome = json.loads(result.read_text())
        self.assertEqual(outcome["status"], "cuda_oom")
        self.assertEqual(outcome["peak_cuda_allocated_bytes"], 333)

    def test_killed_worker_is_not_mislabeled_as_cuda_oom(self):
        class Killed:
            returncode = -9
            def wait(self):
                return self.returncode
        saved = json.loads(self.seed.read_text())
        self.output.mkdir()
        with patch.object(probe.subprocess, "Popen", return_value=Killed()):
            outcome = probe.launch(self.output, saved, 8)
        self.assertEqual(outcome["status"], "error")
        self.assertEqual(outcome["worker_returncode"], -9)


if __name__ == "__main__":
    unittest.main()
