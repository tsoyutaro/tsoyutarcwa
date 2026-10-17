"""Independent physical sanity checks; these do not establish RCWA convergence."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Optimization.achromatic_quarter_wave.geometry import (
    knots_from_logits, logits_from_knots, seeds, slices)
from Optimization.achromatic_quarter_wave.model import Config, evaluate


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cfg = Config(order=3, grid=128)
    checks = []
    def check(name, callback):
        try:
            detail = callback()
            checks.append(dict(name=name, passed=True, detail=detail))
        except Exception as exc:
            checks.append(dict(name=name, passed=False, error=str(exc)))
        print(name, checks[-1]["passed"], flush=True)
    def geometry():
        for p in [np.zeros(6), np.array([-4, 3, 2, 3, -4, 2])]:
            k = knots_from_logits(p, 3, cfg.half_min_nm, cfg.half_max_nm)
            np.testing.assert_allclose(logits_from_knots(k, 5., 95.), p, atol=1e-12)
            radii = slices(k, cfg)
            assert np.all(np.diff(radii, axis=0) >= 0)
            assert np.all((radii > 5) & (radii < 95))
        return "Softmax round trip, bounds and both monotone axes"
    check("monotone_geometry", geometry)
    for shape in ("ellipse", "rectangle"):
        def fresnel(shape=shape):
            r = evaluate(seeds(cfg)[0], shape, 1., 550., cfg)
            np.testing.assert_allclose([r["Tx00"], r["Ty00"]], [.96, .96], atol=2e-8)
            np.testing.assert_allclose([r["Rx_total"], r["Ry_total"]], [.04, .04], atol=2e-8)
            assert abs(r["retardance_deg"]) < 1e-7
            return {k: r[k] for k in ("Tx00", "Rx_total", "retardance_deg")}
        check(shape + "_air_to_glass_fresnel", fresnel)
        def isotropic(shape=shape):
            knots = [[15, 55, 90], [15, 55, 90]]
            r = evaluate(knots, shape, 2.5, 550., cfg)
            # Existing sequential x/y ASR factorization has small rotational
            # discretization errors at M=3. Raster controls preserve C4 exactly.
            phase_tol = .05 if shape == "rectangle" else 1e-7
            power_tol = 2e-4 if shape == "rectangle" else 1e-8
            assert abs(r["retardance_deg"]) < phase_tol, r
            assert abs(r["Tx00"] - r["Ty00"]) < power_tol, r
            return dict(phase_deg=r["retardance_deg"], cross_power=r["cross_power_sum"])
        check(shape + "_equal_axes_zero_retardance", isotropic)
        def swap(shape=shape):
            k = seeds(cfg)[1]
            a = evaluate(k, shape, 2.5, 550., cfg)
            b = evaluate(k[::-1], shape, 2.5, 550., cfg)
            phase_tol = .05 if shape == "rectangle" else 1e-7
            power_tol = 2e-4 if shape == "rectangle" else 1e-8
            assert abs(a["retardance_deg"] + b["retardance_deg"]) < phase_tol, (a, b)
            assert abs(a["Tx00"] - b["Ty00"]) < power_tol, (a, b)
            assert abs(a["stokes3_45"] + b["stokes3_45"]) < power_tol, (a, b)
            assert a["cross_power_sum"] < 1e-12
            assert a["energy_error"] < 2e-8
            if shape == "ellipse":
                assert abs(a["stokes3_45"]) > .95
            return dict(phase_a=a["retardance_deg"], phase_b=b["retardance_deg"],
                        stokes3=a["stokes3_45"], energy_error=a["energy_error"])
        check(shape + "_axis_swap_and_power", swap)
    def diffraction():
        c = replace(cfg, n_substrate=2.5)
        r = evaluate(seeds(c)[1], "rectangle", 2.5, 410., c)
        assert r["Tx_total"] > r["Tx00"] + 1e-6
        assert r["energy_error"] < 2e-8
        return dict(T00=r["Tx00"], T_total=r["Tx_total"], energy_error=r["energy_error"])
    check("propagating_diffraction_is_counted", diffraction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(checks=checks, passed=all(c["passed"] for c in checks)),
                                      indent=2, ensure_ascii=False), encoding="utf-8")
    if not all(c["passed"] for c in checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
