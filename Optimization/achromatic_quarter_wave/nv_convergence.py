"""NV-only circle equivalence and independent M/Q/z convergence sweeps.

From the 20260806 root:
  python Optimization/achromatic_quarter_wave/nv_convergence.py --help
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rcwa_ext import AutoRCWA, Circle, Lattice, LayerSpec, Material, NVMOptions, OutputSpec
from Optimization.achromatic_quarter_wave.geometry import slices
from Optimization.achromatic_quarter_wave.model import Config, scattering_metrics
from Optimization.achromatic_quarter_wave.nv_shapes import ShapeNVRCWA
from Optimization.achromatic_quarter_wave.run import save_csv, save_json


@torch.no_grad()
def simulate_nv(knots, shape, index, wavelength, cfg):
    cfg.validate()
    if not np.isfinite(index) or index <= 0 or not np.isfinite(wavelength) or wavelength <= 0:
        raise ValueError("Structure index and wavelength must be positive and finite.")
    # This study is normal incidence. Reject exact external diffraction cutoffs.
    for port_n in (cfg.n_input, cfg.n_substrate):
        for x in range(-cfg.order, cfg.order + 1):
            for y in range(-cfg.order, cfg.order + 1):
                if abs(port_n**2 - (wavelength*x/cfg.period_x_nm)**2
                       - (wavelength*y/cfg.period_y_nm)**2) < 1e-10:
                    raise ValueError("Wavelength is exactly at an external diffraction cutoff.")
    cls = AutoRCWA if shape == "circle" else ShapeNVRCWA
    scale = cfg.period_x_nm
    sim = cls(freq=scale / wavelength, order=[cfg.order, cfg.order],
              lattice=Lattice.rectangular(1., cfg.period_y_nm / scale),
              outputs=OutputSpec(smatrix_size="half", fields="none"),
              nvm=NVMOptions(grid=(cfg.grid, cfg.grid), use_lanczos=False),
              dtype=torch.complex128, device=cfg.device,
              cascade="redheffer", verify_cascade=False)
    sim.add_input_layer(eps=cfg.n_input**2)
    sim.add_output_layer(eps=cfg.n_substrate**2)
    sim.set_incident_angle(0., 0.)
    for a, b in slices(knots, cfg):
        if shape == "circle":
            if abs(a - b) > 1e-12:
                raise ValueError("Circle reference requires equal x/y half-widths in every layer.")
            sim.add_structured_layer(LayerSpec(
                cfg.height_nm / cfg.layers / scale, Circle(a / scale),
                background=Material(1.), inclusion=Material(index**2), method="nvm"))
        else:
            sim.add_layer_shape_nv(cfg.height_nm / cfg.layers / scale, shape,
                                  a / scale, b / scale, 1., index**2,
                                  nx=cfg.grid, ny=cfg.grid)
    sim.solve_global_smatrix()
    return sim, scattering_metrics(sim, cfg, wavelength)


def complex_t(row, pol):
    return complex(row[f"t{pol}{pol}_real"], row[f"t{pol}{pol}_imag"])


def phase_distance(a, b):
    return float(abs(np.angle(np.exp(1j * np.deg2rad(a - b)), deg=True)))


def differences(a, b):
    valid = a["phase_valid"] and b["phase_valid"]
    phase = phase_distance(a["retardance_deg"], b["retardance_deg"]) if valid else None
    individual = max(float(abs(np.angle(complex_t(a, p) * complex_t(b, p).conjugate(), deg=True)))
                     for p in ("x", "y")) if valid else None
    power = max(abs(a[k] - b[k]) for k in ("Tx00", "Ty00", "Tx_total", "Ty_total", "Rx_total", "Ry_total"))
    field = max(abs(complex_t(a, p) - complex_t(b, p)) for p in ("x", "y"))
    return dict(retardance_change_deg=phase, transmission_phase_change_deg=individual,
                power_change=float(power), complex_t_change=float(field))


def cache_job(out, key, callback):
    path = out / "cache" / (key + ".json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    start = time.perf_counter()
    value = callback()
    value["seconds"] = time.perf_counter() - start
    save_json(path, value)
    print(key, f"{value['seconds']:.2f} s", flush=True)
    return value


def fingerprint():
    files = [Path(__file__), Path(__file__).with_name("nv_shapes.py"),
             Path(__file__).with_name("model.py"), Path(__file__).with_name("geometry.py"),
             *sorted((ROOT / "rcwa_ext").glob("*.py"))]
    return {str(f.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}


def circle_check(args, base):
    profiles = [(f"cylinder_r{r:g}", [[r, r], [r, r]], 1) for r in args.radii]
    for layers in args.layers:
        profiles.append((f"taper_{layers}layers", [args.circle_knots, args.circle_knots], layers))
    rows = []
    for name, knots, layers in profiles:
        for m in args.orders:
            for q in args.grids:
                for w in args.wavelengths:
                    cfg = replace(base, order=m, grid=q, layers=layers)
                    key = f"equal_{name}_M{m}_Q{q}_w{w:g}"
                    def calculate(cfg=cfg, knots=knots, w=w):
                        ellipse, e = simulate_nv(knots, "ellipse", args.index, w, cfg)
                        circle, c = simulate_nv(knots, "circle", args.index, w, cfg)
                        diff = differences(e, c)
                        s_error = max(float((ellipse.S[i] - circle.S[i]).abs().max().cpu()) for i in (0, 1))
                        eps_error = max(float((a - b).abs().max().cpu()) for a, b in zip(ellipse.eps_conv, circle.eps_conv))
                        pq_error = max(float((a - b).abs().max().cpu())
                                       for first, second in ((ellipse.P, circle.P), (ellipse.Q, circle.Q))
                                       for a, b in zip(first, second))
                        energy = max(e["energy_error"], c["energy_error"])
                        isotropy = max(abs(e["retardance_deg"]), abs(c["retardance_deg"]))
                        passed = (diff["power_change"] <= args.equality_tol
                                  and diff["complex_t_change"] <= args.equality_tol
                                  and diff["transmission_phase_change_deg"] is not None
                                  and diff["transmission_phase_change_deg"] <= args.equality_phase_tol
                                  and isotropy <= args.equality_phase_tol
                                  and s_error <= args.equality_tol
                                  and eps_error <= args.operator_tol and pq_error <= args.operator_tol
                                  and energy <= args.energy_tol)
                        return dict(**diff, S_max_abs_error=s_error, eps_max_abs_error=eps_error,
                                    PQ_max_abs_error=pq_error, isotropic_retardance_max_deg=isotropy,
                                    energy_error=energy, passed=bool(passed),
                                    ellipse=e, circle=c)
                    value = cache_job(args.output, key, calculate)
                    rows.append(dict(profile=name, order=m, grid=q, layers=layers, wavelength_nm=w,
                                     **{k: v for k, v in value.items() if k not in ("ellipse", "circle")},
                                     ellipse_Tx00=value["ellipse"]["Tx00"], circle_Tx00=value["circle"]["Tx00"],
                                     ellipse_Rx=value["ellipse"]["Rx_total"], circle_Rx=value["circle"]["Rx_total"]))
                    save_csv(args.output / "circle_equivalence.csv", rows)
    summary = dict(passed=all(r["passed"] for r in rows), comparisons=len(rows),
                   max_S_error=max(r["S_max_abs_error"] for r in rows),
                   max_complex_t_error=max(r["complex_t_change"] for r in rows),
                   max_power_error=max(r["power_change"] for r in rows),
                   max_transmission_phase_error_deg=max(r["transmission_phase_change_deg"] for r in rows),
                   explanation="Equal-axis ellipse uses independent ellipse coefficients, normal field and layer assembly; no circle dispatch.")
    save_json(args.output / "circle_equivalence_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["passed"]:
        raise SystemExit(2)


def convergence(args, base):
    knots = [args.x_knots, args.y_knots]
    mmax, q0, qmax, z0, zmax = max(args.orders), min(args.grids), max(args.grids), min(args.layers), max(args.layers)
    schedule = [("orders", m, q0, z0) for m in args.orders]
    schedule += [("grids", mmax, q, z0) for q in args.grids]
    schedule += [("layers", mmax, qmax, z) for z in args.layers]
    schedule += [("orders_at_final_grid_layers", m, qmax, zmax) for m in args.orders[-2:]]
    rows = []
    for shape in args.shapes:
        for series, m, q, z in schedule:
            cfg = replace(base, order=m, grid=q, layers=z)
            for w in args.wavelengths:
                key = f"{shape}_M{m}_Q{q}_Z{z}_w{w:g}"
                def calculate(cfg=cfg, shape=shape, w=w):
                    _, result = simulate_nv(knots, shape, args.index, w, cfg)
                    return result
                row = cache_job(args.output, key, calculate)
                rows.append(dict(shape=shape, series=series, order=m, grid=q, layers=z, **row))
                save_csv(args.output / "nv_convergence.csv", rows)
    summaries = []
    for shape in args.shapes:
        for series, levels in [("orders", args.orders), ("grids", args.grids),
                               ("layers", args.layers), ("orders_at_final_grid_layers", args.orders[-2:])]:
            if len(levels) < 2:
                summaries.append(dict(shape=shape, series=series, status="insufficient_levels", passed=None))
                continue
            field = "order" if series.startswith("orders") else "grid" if series == "grids" else "layers"
            group = [r for r in rows if r["shape"] == shape and r["series"] == series]
            diffs = []
            for w in args.wavelengths:
                a = next(r for r in group if r[field] == levels[-2] and r["wavelength_nm"] == w)
                b = next(r for r in group if r[field] == levels[-1] and r["wavelength_nm"] == w)
                diffs.append(differences(a, b))
            valid = all(r["retardance_change_deg"] is not None for r in diffs)
            phase = max(r["retardance_change_deg"] for r in diffs) if valid else None
            tphase = max(r["transmission_phase_change_deg"] for r in diffs) if valid else None
            power = max(r["power_change"] for r in diffs)
            energy = max(r["energy_error"] for r in group)
            passed = bool(valid and phase <= args.phase_tol and tphase <= args.phase_tol
                          and power <= args.power_tol and energy <= args.energy_tol)
            summaries.append(dict(shape=shape, series=series, status="within_tolerance" if passed else "needs_refinement",
                                  passed=passed, compared_levels=levels[-2:],
                                  max_retardance_change_deg=phase, max_transmission_phase_change_deg=tphase,
                                  max_power_change=power, max_energy_error=energy))
    overall = all(r["passed"] is True for r in summaries)
    save_json(args.output / "convergence_summary.json", dict(
        within_requested_tolerances=overall, comparisons=summaries,
        note="Last-pair numerical convergence indicators at sampled wavelengths; not a proof over the continuous band."))
    save_csv(args.output / "convergence_summary.csv", [dict(
        shape=r["shape"], series=r["series"], status=r["status"],
        max_retardance_change_deg=r.get("max_retardance_change_deg"),
        max_transmission_phase_change_deg=r.get("max_transmission_phase_change_deg"),
        max_power_change=r.get("max_power_change"), max_energy_error=r.get("max_energy_error")) for r in summaries])
    print(json.dumps(summaries, indent=2), flush=True)
    if args.require_convergence and not overall:
        raise SystemExit(2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("circle-check", "converge"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--index", type=float, default=2.5)
    p.add_argument("--substrate-index", type=float, default=1.5)
    p.add_argument("--period", type=float, default=200.)
    p.add_argument("--height", type=float, default=500.)
    p.add_argument("--orders", type=int, nargs="+", default=[3, 5, 7])
    p.add_argument("--grids", type=int, nargs="+", default=[128, 256])
    p.add_argument("--layers", type=int, nargs="+", default=[15, 30])
    p.add_argument("--wavelengths", type=float, nargs="+", default=[400., 550., 700.])
    p.add_argument("--shapes", choices=("ellipse", "rectangle"), nargs="+", default=["rectangle", "ellipse"])
    p.add_argument("--x-knots", type=float, nargs="+", default=[90.5, 92.3, 94.1])
    p.add_argument("--y-knots", type=float, nargs="+", default=[48.2, 50., 51.8])
    p.add_argument("--radii", type=float, nargs="+", default=[30., 70.])
    p.add_argument("--circle-knots", type=float, nargs="+", default=[10., 50., 90.])
    p.add_argument("--phase-tol", type=float, default=.1, help="Convergence tolerance in degrees (retardance and co-polar phase)")
    p.add_argument("--power-tol", type=float, default=1e-3, help="Absolute power-fraction convergence tolerance")
    p.add_argument("--energy-tol", type=float, default=1e-7)
    p.add_argument("--equality-tol", type=float, default=1e-8, help="Circle comparison amplitude/power/S tolerance")
    p.add_argument("--equality-phase-tol", type=float, default=1e-6)
    p.add_argument("--operator-tol", type=float, default=1e-9)
    p.add_argument("--require-convergence", action="store_true")
    args = p.parse_args()
    for name in ("orders", "grids", "layers", "wavelengths"):
        values = getattr(args, name)
        if any(not np.isfinite(x) or x <= 0 for x in values) or len(set(values)) != len(values):
            p.error(f"--{name} values must be positive, finite and unique.")
        setattr(args, name, sorted(values))
    if args.threads < 1 or min(args.phase_tol, args.power_tol, args.energy_tol,
                               args.equality_tol, args.equality_phase_tol, args.operator_tol) <= 0:
        p.error("Thread count and all error tolerances must be positive.")
    if len(args.x_knots) != len(args.y_knots) or len(args.x_knots) < 2 or len(args.circle_knots) < 2:
        p.error("x/y knot arrays need the same length >=2; circle knots need length >=2.")
    torch.set_num_threads(args.threads)
    base = Config(period_x_nm=args.period, period_y_nm=args.period, height_nm=args.height,
                  n_substrate=args.substrate_index, device=args.device,
                  order=min(args.orders), grid=min(args.grids), layers=min(args.layers))
    base.validate()
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = dict(arguments={k: str(v.resolve()) if isinstance(v, Path) else v
                               for k, v in vars(args).items()}, source_sha256=fingerprint(),
                    materials="real constant n, air inclusion background, mu=1",
                    nv="analytic epsilon and inverse-epsilon; symmetric NV factorization; no Lanczos; no group reduction")
    path = args.output / "run_metadata.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != metadata:
        raise ValueError("Output settings or source code changed; use a new --output directory.")
    save_json(path, metadata)
    if args.mode == "circle-check":
        circle_check(args, base)
    else:
        convergence(args, base)


if __name__ == "__main__":
    main()
