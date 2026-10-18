"""Validate the diagnostic control, reference reuse, and failure isolation."""
from contextlib import redirect_stdout
from copy import deepcopy
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
import run_ag_nv_factorization_check as control


class FactorizationTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT/"validation"/f"_ag_factorization_test_{uuid.uuid4().hex}"
        self.directory.mkdir()
        self.addCleanup(self.cleanup)
        self.args = control.build_parser().parse_args(["--device", "cpu", "--order", "2",
            "--slices", "4", "--grids", "48", "--quadrature", "8",
            "--output-root", str(self.directory/"output"), "--no-plots"])
        self.jobs = control.build_jobs(self.args)

    def cleanup(self):
        target = self.directory.resolve()
        assert target.parent == (ROOT/"validation").resolve() and target.name.startswith("_ag_factorization_test_")
        shutil.rmtree(target)

    def reference(self):
        identity = deepcopy(self.jobs[0]["identity"])
        identity.pop("factorization")
        identity["sources"].pop("run_ag_nv_factorization_check.py")
        reference_root = self.directory/"reference"
        directory = reference_root/"Q48"
        directory.mkdir(parents=True)
        report = dict(status="diagnostics_complete", identity=identity,
                      result=dict(physical_result=False), signature="old-diagnostic-signature")
        control.nv.write_json(directory/"report.json", report)
        for name in control.RESULT_FILES:
            (directory/name).write_bytes(b"reference data")
        return reference_root, report

    def test_scalar_control_keeps_complex_permittivity_and_restores(self):
        import torch
        from rcwa_ext.nvm import CustomRCWA_NVM
        previous = CustomRCWA_NVM.__dict__["_symmetric_nvm_tensor"]
        eps = torch.diag(torch.tensor([1+0j, -13+.43j], dtype=torch.complex128))
        delta = torch.ones_like(eps)*complex(3, 4)
        projection = torch.eye(4, dtype=torch.complex128)
        with control.factorization_scope("galerkin"):
            actual = CustomRCWA_NVM._symmetric_nvm_tensor(eps, delta, projection)
            self.assertTrue(torch.equal(actual, torch.block_diag(eps, eps)))
        self.assertIs(CustomRCWA_NVM.__dict__["_symmetric_nvm_tensor"], previous)
        with self.assertRaises(RuntimeError):
            with control.factorization_scope("galerkin"):
                raise RuntimeError("deliberate test error")
        self.assertIs(CustomRCWA_NVM.__dict__["_symmetric_nvm_tensor"], previous)

    def test_actual_control_reconstructs_passive_power(self):
        import torch
        results = {}
        before = {path: control.nv.digest(ROOT/path) for path in self.jobs[0]["identity"]["sources"]}
        with redirect_stdout(io.StringIO()), torch.no_grad():
            for job in self.jobs:
                identity = job["identity"]
                output = self.directory/identity["factorization"]
                output.mkdir()
                with control.factorization_scope(identity["factorization"]):
                    results[identity["factorization"]] = control.diagnostic.compute_case(identity, output)
        nv_result, scalar = results["symmetric-nv"], results["galerkin"]
        self.assertGreater(nv_result["negative_absorption_layers"], 0)
        self.assertFalse(nv_result["physical_result"])
        self.assertEqual(scalar["negative_absorption_layers"], 0)
        self.assertTrue(scalar["physical_result"])
        self.assertTrue(scalar["diagnostic_checks_passed"])
        self.assertLess(scalar["max_boundary_residual"], 1e-10)
        self.assertLess(scalar["max_volume_flux_error"], 1e-7)
        # Sampling the now-unused normal field must not change the scalar
        # Fourier control: epsilon uses analytic circular coefficients.
        other = dict(self.jobs[1]["identity"], grid=32)
        other_output = self.directory/"galerkin_Q32"
        other_output.mkdir()
        with redirect_stdout(io.StringIO()), torch.no_grad(), control.factorization_scope("galerkin"):
            scalar_other_q = control.diagnostic.compute_case(other, other_output)
        self.assertAlmostEqual(scalar["reflectance"], scalar_other_q["reflectance"], places=10)
        self.assertTrue(all(control.nv.digest(ROOT/path) == digest for path, digest in before.items()))

    def test_matching_nonphysical_nv_reference_is_reused_with_provenance(self):
        reference, saved = self.reference()
        destination = self.directory/"copied"
        self.assertTrue(control.try_reference(self.jobs[0], reference, destination))
        result = json.loads((destination/"report.json").read_text(encoding="utf-8"))
        self.assertEqual(result["original_identity"], saved["identity"])
        self.assertEqual(result["original_signature"], saved["signature"])
        self.assertFalse(result["result"]["physical_result"])
        self.assertTrue(all((destination/name).read_bytes() == b"reference data" for name in control.RESULT_FILES))
        self.assertFalse(control.try_reference(self.jobs[1], reference, self.directory/"not_copied"))

    def test_reference_with_different_order_or_source_is_rejected(self):
        reference, saved = self.reference()
        for field in ("order", "source"):
            changed = deepcopy(saved)
            if field == "order":
                changed["identity"]["order"] = 3
            else:
                changed["identity"]["sources"]["rcwa_ext/nvm.py"] = "different"
            control.nv.write_json(reference/"Q48/report.json", changed)
            destination = self.directory/f"rejected_{field}"
            self.assertFalse(control.try_reference(self.jobs[0], reference, destination))
            self.assertFalse(destination.exists())

    def test_oom_does_not_stop_other_factorization(self):
        calls = []
        def fake_run(command, check):
            plan = Path(command[-1])
            identity = json.loads(plan.read_text(encoding="utf-8"))["identity"]
            name = identity["factorization"]
            calls.append(name)
            report = dict(status="oom" if name == "symmetric-nv" else "diagnostics_complete",
                identity=identity, signature=control.nv.identity_signature(identity))
            if name == "galerkin":
                report["result"] = dict(physical_result=True)
            control.nv.write_json(plan.parent/"report.json", report)
            return SimpleNamespace(returncode=2 if name == "symmetric-nv" else 0)
        argv = ["--device", "cpu", "--order", "2", "--slices", "4", "--grids", "48",
                "--output-root", str(self.directory/"batch"), "--no-plots", "--no-reuse-reference"]
        with patch.object(control.subprocess, "run", fake_run), redirect_stdout(io.StringIO()):
            self.assertEqual(control.main(argv), 2)
        self.assertEqual(calls, ["symmetric-nv", "galerkin"])

    def test_dry_run_writes_nothing(self):
        path = self.directory/"dry"
        with redirect_stdout(io.StringIO()):
            self.assertEqual(control.main(["--dry-run", "--output-root", str(path)]), 0)
        self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
