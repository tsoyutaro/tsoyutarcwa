"""Physical and boundary-normal checks for ellipse/rectangle NV layers."""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from Optimization.achromatic_quarter_wave.model import Config
from Optimization.achromatic_quarter_wave.nv_shapes import ShapeNVRCWA
from Optimization.achromatic_quarter_wave.nv_convergence import simulate_nv
from Optimization.achromatic_quarter_wave.run import save_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    cfg = Config(order=3, grid=128, layers=5)
    checks = []
    def check(name, callback):
        try:
            detail = callback()
            checks.append(dict(name=name, passed=True, details=detail))
        except Exception as exc:
            checks.append(dict(name=name, passed=False, error=repr(exc)))
        print(name, checks[-1]["passed"], flush=True)
    def normals():
        a, b = torch.tensor(.45, dtype=torch.float64), torch.tensor(.25, dtype=torch.float64)
        t = torch.linspace(-.99, .99, 51, dtype=torch.float64)
        nx, ny, _ = ShapeNVRCWA.normal_direction("rectangle", torch.full_like(t, a), t*b, a, b)
        assert float(ny.abs().max()) < 1e-13
        assert float((nx - 1).abs().max()) < 1e-13
        nx, ny, _ = ShapeNVRCWA.normal_direction("rectangle", t*a, torch.full_like(t, b), a, b)
        assert float(nx.abs().max()) < 1e-13
        assert float((ny - 1).abs().max()) < 1e-13
        theta = torch.linspace(0, 2*np.pi, 97, dtype=torch.float64)
        nx, ny, _ = ShapeNVRCWA.normal_direction("ellipse", a*torch.cos(theta), b*torch.sin(theta), a, b)
        tangent_dot = nx * (-a*torch.sin(theta)) + ny * (b*torch.cos(theta))
        assert float(tangent_dot.abs().max()) < 1e-13
        return "Normals are perpendicular to ellipse tangent and parallel to rectangle face normals."
    check("boundary_normals", normals)
    for shape in ("rectangle", "ellipse"):
        def fresnel(shape=shape):
            _, r = simulate_nv([[40, 60, 90], [20, 30, 45]], shape, 1., 550., cfg)
            np.testing.assert_allclose([r["Tx00"], r["Ty00"]], [.96, .96], atol=1e-8)
            np.testing.assert_allclose([r["Rx_total"], r["Ry_total"]], [.04, .04], atol=1e-8)
            return {k:r[k] for k in ("Tx00", "Rx_total", "energy_error")}
        check(shape + "_fresnel", fresnel)
        def equal_axes(shape=shape):
            _, r = simulate_nv([[20, 55, 90], [20, 55, 90]], shape, 2.5, 550., cfg)
            assert abs(r["retardance_deg"]) < 1e-7, r
            assert abs(r["Tx00"] - r["Ty00"]) < 1e-8, r
            assert r["cross_power_sum"] < 1e-12, r
            return dict(phase_deg=r["retardance_deg"], energy_error=r["energy_error"])
        check(shape + "_equal_axes", equal_axes)
        def swap(shape=shape):
            k = [[80, 87, 94], [25, 40, 52]]
            _, a = simulate_nv(k, shape, 2.5, 550., cfg)
            _, b = simulate_nv(k[::-1], shape, 2.5, 550., cfg)
            assert abs(a["retardance_deg"] + b["retardance_deg"]) < 1e-7, (a,b)
            assert abs(a["Tx00"] - b["Ty00"]) < 1e-8, (a,b)
            assert abs(a["stokes3_45"] + b["stokes3_45"]) < 1e-8, (a,b)
            assert max(a["energy_error"], b["energy_error"]) < 1e-8
            return dict(phase_deg=a["retardance_deg"], swapped_phase_deg=b["retardance_deg"], energy_error=a["energy_error"])
        check(shape + "_axis_swap", swap)
    def thin_layers():
        k = [[90, 90], [45, 45]]
        _, a = simulate_nv(k, "rectangle", 2.5, 550., replace(cfg, layers=1))
        _, b = simulate_nv(k, "rectangle", 2.5, 550., cfg)
        assert abs(a["retardance_deg"] - b["retardance_deg"]) < 1e-7, (a,b)
        assert abs(a["Tx00"] - b["Tx00"]) < 1e-8, (a,b)
        return "Splitting an unchanged slab into 5 identical layers preserves transmission and retardance."
    check("constant_slab_splitting", thin_layers)
    save_json(args.output, dict(passed=all(r["passed"] for r in checks), checks=checks))
    if not all(r["passed"] for r in checks):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
