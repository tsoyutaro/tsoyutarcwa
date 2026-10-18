"""Check source-specific reconstruction against the native internal-field API."""
from contextlib import redirect_stdout
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
import run_ag_nv_m14_diagnostics as diagnostic


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT/"validation"/f"_ag_diagnostic_test_{uuid.uuid4().hex}"
        self.directory.mkdir()
        self.addCleanup(self.cleanup)
        self.args = diagnostic.build_parser().parse_args(["--device", "cpu", "--order", "2",
            "--slices", "4", "--grids", "32,48", "--quadrature", "8",
            "--output-root", str(self.directory), "--no-plots"])

    def cleanup(self):
        target = self.directory.resolve()
        assert target.parent == (ROOT/"validation").resolve() and target.name.startswith("_ag_diagnostic_test_")
        shutil.rmtree(target)

    def test_reconstruction_matches_independent_native_fields(self):
        import torch
        from rcwa_solver_auto import AutoRCWA, GroupTheoryOptions, Lattice, NVMOptions, OutputSpec
        from studies.gold_motheye.converge import _mean_poynting_z, _zero_order_x_source
        identity = diagnostic.build_identities(self.args)[0]
        out = self.directory/"actual"
        out.mkdir()
        with redirect_stdout(io.StringIO()), torch.no_grad():
            actual = diagnostic.compute_case(identity, out)
            eps = diagnostic.load_tabulated_nk(identity["ag_csv"])(550.)
            sim = AutoRCWA(freq=200/550, order=[2, 2], lattice=Lattice.triangular(1.),
                cascade="redheffer", outputs=OutputSpec(smatrix_size="half", fields="internal"),
                nvm=NVMOptions(grid=(32, 32), use_lanczos=False),
                group_theory=GroupTheoryOptions(enabled=True, symmetry="d6", strict=True, polarization="x"),
                verify_cascade=False, dtype=torch.complex128, device=torch.device("cpu"))
            sim.add_input_layer(eps=1.)
            sim.add_output_layer(eps=eps)
            sim.set_incident_angle(0., 0.)
            for i in range(4):
                radius = (5+90*(i+.5)/4)/200
                sim.add_layer_circle_nvm(500/4/200, radius, 1., eps,
                    centers=((.75, 3**.5/4),), nx=32, ny=32, use_lanczos=False)
            sim.solve_global_smatrix()
            sim.source_planewave(amplitude=[1., 0.], direction="forward")
            source = _zero_order_x_source(sim)
            pin = _mean_poynting_z(source, sim.Vi, direction=1)
            expected_r = -_mean_poynting_z(sim.S[1]@source, sim.Vi, direction=-1)/pin
            self.assertAlmostEqual(actual["reflectance"], expected_r, places=9)
            with (out/"layer_power.csv").open(encoding="utf-8-sig", newline="") as stream:
                powers = list(csv.DictReader(stream))
            for i, row in enumerate(powers):
                for z, key in ((0., "flux_top"), (500/4/200, "flux_bottom")):
                    ex, ey, _, hx, hy, _ = sim._internal_fourier_fields(i, z)
                    flux = float(.5*torch.real(torch.sum(ex*hy.conj()-ey*hx.conj())))/pin
                    self.assertAlmostEqual(float(row[key]), flux, places=8)
            self.assertLess(actual["max_volume_flux_error"], 1e-7)
            self.assertLess(actual["max_boundary_residual"], 1e-10)
            self.assertTrue(actual["diagnostic_checks_passed"])
            # This coarse setup deliberately produces nonphysical power. It
            # must be saved unchanged, so the batch can inspect the next Q.
            self.assertGreater(actual["reflectance"], 1.)
            self.assertFalse(actual["physical_result"])
            self.assertGreater(actual["negative_absorption_layers"], 0)
            self.assertLess(actual["total_negative_loss_sector"], 0.)

    def test_each_q_runs_after_oom_or_nonphysical_result(self):
        argv = ["--device", "cpu", "--order", "2", "--slices", "4", "--grids", "32,48,64",
                "--output-root", str(self.directory), "--no-plots"]
        calls = []
        def fake_run(command, check):
            plan = Path(command[-1])
            identity = json.loads(plan.read_text(encoding="utf-8"))["identity"]
            calls.append(identity["grid"])
            report = dict(identity=identity, signature=diagnostic.nv.identity_signature(identity))
            if identity["grid"] == 32:
                report.update(status="oom", error="simulated CUDA OOM")
            else:
                report.update(status="diagnostics_complete", result=dict(physical_result=False))
            diagnostic.nv.write_json(plan.parent/"report.json", report)
            return SimpleNamespace(returncode=2 if identity["grid"] == 32 else 0)
        with patch.object(diagnostic.subprocess, "run", fake_run), redirect_stdout(io.StringIO()):
            self.assertEqual(diagnostic.main(argv), 2)
        self.assertEqual(calls, [32, 48, 64])
        calls.clear()
        with patch.object(diagnostic.subprocess, "run", fake_run), redirect_stdout(io.StringIO()):
            diagnostic.main(argv)
        self.assertEqual(calls, [])

    def test_existing_q_can_be_reused_when_adding_grid(self):
        identities = diagnostic.build_identities(self.args)
        self.args.grids = "32,48,64"
        extended = diagnostic.build_identities(self.args)
        self.assertEqual(identities, extended[:2])

    def test_grid_below_nyquist_guard_is_rejected(self):
        self.args.order = 14
        self.args.grids = "32"
        with self.assertRaises(ValueError):
            diagnostic.build_identities(self.args)

    def test_dry_run_writes_nothing(self):
        out = self.directory/"dry"
        with redirect_stdout(io.StringIO()):
            code = diagnostic.main(["--device", "cpu", "--dry-run", "--output-root", str(out)])
        self.assertEqual(code, 0)
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
