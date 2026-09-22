"""Independent checks of pullback FFT signs and boundary elimination."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from paper_reproductions.weiss2009.compare_boundaries import direct_pullback, connect


class BoundaryTests(unittest.TestCase):
    def test_undeformed_pullback_is_identity(self):
        axis = torch.arange(32, dtype=torch.float64) / 32
        x, y = torch.meshgrid(axis, axis, indexing="ij")
        one, zero = torch.ones_like(x), torch.zeros_like(x)
        modes = torch.arange(-1, 2)
        kx, ky = torch.meshgrid(modes, modes, indexing="ij")
        mapping = SimpleNamespace(x=x, y=y, x_u=one, x_v=zero, y_u=zero, y_v=one)
        sim = SimpleNamespace(asr_mappings=[mapping], order_N=9,
            _dtype=torch.complex128, _device=torch.device("cpu"),
            order_x=modes, order_y=modes, Kx_norm_dn=kx.flatten(),
            Ky_norm_dn=ky.flatten(), omega=2 * torch.pi)
        self.assertTrue(torch.allclose(direct_pullback(sim), torch.eye(18, dtype=torch.complex128), atol=1e-13, rtol=0))
        # Constant translation fixes the sign of the physical plane-wave phase.
        mapping.x = x + 0.13
        expected = torch.diag(torch.exp(2j * torch.pi * kx.flatten().double() * 0.13)).repeat(2, 2)
        expected[:9, 9:] = 0
        expected[9:, :9] = 0
        self.assertTrue(torch.allclose(direct_pullback(sim), expected, atol=1e-13, rtol=0))

    def test_elimination_against_full_boundary_equations(self):
        torch.manual_seed(41)
        n = 4
        eye = torch.eye(n, dtype=torch.complex128)
        def matrix():
            return eye + 0.1 * torch.randn((n, n), dtype=torch.complex128)
        w, v, ex, hx = matrix(), matrix(), matrix(), matrix()
        phase = torch.exp(torch.linspace(-0.3, -0.1, n).to(torch.complex128) + 0.2j)
        incident = eye[:, :2]
        rr, tt, residuals, _, _ = connect(w, v, ex, hx, phase, incident)
        z = torch.zeros_like(eye)
        wp, vp = w * phase[None, :], v * phase[None, :]
        # Unknowns: internal forward at top, backward at bottom, reflected,
        # transmitted. Four independent E/H continuity equations.
        full = torch.cat([torch.cat(row, dim=1) for row in
            [(w, wp, -ex, z), (v, -vp, hx, z),
             (wp, w, z, -ex), (vp, -v, z, -hx)]], dim=0)
        rhs = torch.cat((ex @ incident, hx @ incident,
                         torch.zeros_like(incident), torch.zeros_like(incident)), dim=0)
        solution = torch.linalg.solve(full, rhs)
        self.assertTrue(torch.allclose(rr, solution[2*n:3*n], atol=1e-12, rtol=1e-12))
        self.assertTrue(torch.allclose(tt, solution[3*n:], atol=1e-12, rtol=1e-12))
        self.assertLess(max(residuals.values()), 1e-12)


if __name__ == "__main__":
    unittest.main()
