"""Screen, optimize and verify tapered anisotropic RCWA structures.

Run from the project root: python Optimization/achromatic_quarter_wave/run.py --help
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import qmc
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Optimization.achromatic_quarter_wave.geometry import (
    knots_from_logits, logits_from_knots, seeds, slices)
from Optimization.achromatic_quarter_wave.model import Config, evaluate, metrics, objective


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def save_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def get_rows(knots, shape, index, wavelengths, cfg):
    return [evaluate(knots, shape, index, w, cfg) for w in wavelengths]


def design_payload(knots, shape, index, cfg, rows, provenance):
    return dict(shape=shape, structure_index=index, config=cfg.to_dict(),
                knots_nm=np.asarray(knots).tolist(), spectra=rows,
                metrics=metrics(rows, cfg), provenance=provenance,
                phase_convention="arg(txx*conj(tyy)), principal interval [-180,180] deg")


def export_design(path, design):
    save_json(path, design)
    cfg = Config(**design["config"])
    dz = cfg.height_nm / cfg.layers
    profile = [dict(layer=i + 1, z_top_nm=i * dz, z_bottom_nm=(i + 1) * dz,
                    half_x_nm=float(a), half_y_nm=float(b),
                    width_x_nm=float(2 * a), width_y_nm=float(2 * b),
                    gap_x_nm=float(cfg.period_x_nm - 2 * a),
                    gap_y_nm=float(cfg.period_y_nm - 2 * b))
               for i, (a, b) in enumerate(slices(design["knots_nm"], cfg))]
    save_csv(Path(path).with_suffix(".profile.csv"), profile)
    save_csv(Path(path).with_suffix(".spectrum.csv"), design["spectra"])


def label(index, shape):
    return "n" + str(float(index)).replace(".", "p") + "_" + shape


def screen(args, cfg):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    metadata = dict(config=cfg.to_dict(), wavelengths_nm=args.wavelengths,
                    shapes=args.shapes, indices=args.indices,
                    samples=args.samples, seed=args.seed)
    meta_path, history_path = out / "metadata.json", out / "history.jsonl"
    if meta_path.exists() and json.loads(meta_path.read_text(encoding="utf-8")) != metadata:
        raise ValueError("Existing output uses different settings; choose another --output.")
    save_json(meta_path, metadata)
    records = []
    if history_path.exists():
        for line in history_path.read_text(encoding="utf-8").splitlines():
            records.append(json.loads(line))
    done = {(r["shape"], r["structure_index"], r["candidate"]) for r in records}
    candidates = seeds(cfg)
    if args.samples:
        sampler = qmc.Sobol(2 * cfg.nodes, scramble=True, seed=args.seed)
        points = sampler.random_base2(int(np.ceil(np.log2(args.samples))))[:args.samples]
        candidates.extend(knots_from_logits(-3 + 6 * p, cfg.nodes,
                                           cfg.half_min_nm, cfg.half_max_nm)
                          for p in points)
    total = len(candidates) * len(args.indices) * len(args.shapes)
    print(f"{total} candidates; {len(args.wavelengths)} wavelengths each; "
          f"M={cfg.order}, slices={cfg.layers}, Q={cfg.grid}", flush=True)
    with history_path.open("a", encoding="utf-8") as f:
        for index in args.indices:
            for shape in args.shapes:
                for i, knots in enumerate(candidates):
                    if (shape, index, i) in done:
                        continue
                    started = time.perf_counter()
                    rows = get_rows(knots, shape, index, args.wavelengths, cfg)
                    d = design_payload(knots, shape, index, cfg, rows, "Sobol/seed screening")
                    d.update(candidate=i, seconds=time.perf_counter() - started)
                    f.write(json.dumps(d, ensure_ascii=False, allow_nan=False) + "\n")
                    f.flush()
                    records.append(d)
                    if i % 4 == 0 or i == len(candidates) - 1:
                        print(f"{label(index, shape)} {i+1}/{len(candidates)} "
                              f"phase={d['metrics']['max_retardance_deg']:.2f} deg "
                              f"minT={d['metrics']['min_T00']:.3f}", flush=True)
    summary, maxima = [], []
    for index in args.indices:
        for shape in args.shapes:
            selected = [d for d in records if d["shape"] == shape
                        and d["structure_index"] == index]
            qualified = [d for d in selected if d["metrics"]["passes_transmission"]]
            best_qwp = min(selected, key=lambda d: objective(d["spectra"], cfg, "qwp"))
            stem = label(index, shape)
            export_design(out / ("best_qwp_" + stem + ".json"), best_qwp)
            best_phase = max(qualified, key=lambda d: d["metrics"]["min_retardance_deg"]) if qualified else None
            if best_phase:
                export_design(out / ("best_maximin_phase_" + stem + ".json"), best_phase)
            summary.append(dict(case=stem, candidates=len(selected),
                                candidates_passing_all_samples=len(qualified),
                                best_maximin_phase_deg=best_phase["metrics"]["min_retardance_deg"] if best_phase else None,
                                best_qwp_error_deg=best_qwp["metrics"]["worst_qwp_error_deg"],
                                best_qwp_min_T00=best_qwp["metrics"]["min_T00"],
                                best_qwp_same_hand_circularity=best_qwp["metrics"]["worst_same_hand_circularity"]))
            for j, wavelength in enumerate(args.wavelengths):
                available = [d for d in selected if d["spectra"][j]["phase_valid"]]
                usable = [d for d in available if min(d["spectra"][j]["Tx00"],
                                                     d["spectra"][j]["Ty00"]) >= cfg.transmission_floor]
                for constrained, group in [(False, available), (True, usable)]:
                    if not group:
                        continue
                    best = max(group, key=lambda d: d["spectra"][j]["retardance_abs_deg"])
                    row = best["spectra"][j]
                    maxima.append(dict(case=stem, wavelength_nm=wavelength,
                                       transmission_constraint=constrained,
                                       maximum_observed_deg=row["retardance_abs_deg"],
                                       candidate=best["candidate"], Tx00=row["Tx00"], Ty00=row["Ty00"]))
                    if constrained:
                        export_design(out / f"max_phase_{stem}_{wavelength:g}nm.json", best)
    save_json(out / "summary.json", dict(cases=summary, per_wavelength_maxima=maxima,
              caveat="Observed maxima in a finite, low-order search; not global maxima or converged broadband claims."))
    save_csv(out / "summary.csv", summary)
    save_csv(out / "per_wavelength_maxima.csv", maxima)
    print(json.dumps(summary, indent=2), flush=True)


def optimize(args, cfg):
    if len(args.shapes) != 1 or len(args.indices) != 1:
        raise ValueError("optimize requires exactly one --shapes and one --indices value.")
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    initial = json.loads(args.design.read_text(encoding="utf-8")) if args.design else None
    if initial:
        if initial["shape"] != args.shapes[0] or initial["structure_index"] != args.indices[0]:
            raise ValueError("--design shape/index disagrees with the requested case.")
        knots = np.asarray(initial["knots_nm"])
        old_z, new_z = np.linspace(0, 1, knots.shape[1]), np.linspace(0, 1, cfg.nodes)
        knots = np.array([np.interp(new_z, old_z, a) for a in knots])
    else:
        knots = seeds(cfg)[1]
    x0 = np.clip(logits_from_knots(knots, cfg.half_min_nm, cfg.half_max_nm), -5., 5.)
    best, counter = [float("inf")], [0]
    history_path = out / "optimization_history.jsonl"
    if history_path.exists() and history_path.stat().st_size:
        raise ValueError("Optimization output already contains a run; choose a new --output.")
    def loss(x):
        k = knots_from_logits(x, cfg.nodes, cfg.half_min_nm, cfg.half_max_nm)
        rows = get_rows(k, args.shapes[0], args.indices[0], args.wavelengths, cfg)
        value = objective(rows, cfg, args.goal)
        counter[0] += 1
        d = design_payload(k, args.shapes[0], args.indices[0], cfg, rows,
                           f"differential_evolution: {args.goal}")
        d.update(evaluation=counter[0], objective=value)
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(d, allow_nan=False) + "\n")
        if value < best[0]:
            best[0] = value
            export_design(out / "best_design.json", d)
        if counter[0] % 10 == 0:
            print(f"evaluation={counter[0]} best_loss={best[0]:.6f}", flush=True)
        return value
    result = differential_evolution(loss, [(-5., 5.)] * len(x0), x0=x0,
                                    maxiter=args.maxiter, popsize=args.popsize,
                                    seed=args.seed, polish=False, workers=1,
                                    updating="immediate", tol=.003)
    save_json(out / "optimizer_status.json", dict(success=bool(result.success),
              message=str(result.message), evaluations=int(result.nfev),
              objective=float(result.fun), note="Optimizer termination is not a convergence proof for RCWA."))


def verify(args):
    if not args.design:
        raise ValueError("verify requires --design path/to/design.json")
    d = json.loads(args.design.read_text(encoding="utf-8"))
    base = Config(**d["config"])
    base.device = args.device
    # Change one discretization at a time, then refine z at the final M/Q.
    schedule = [(m, args.verify_grid, base.layers) for m in args.orders]
    schedule += [(max(args.orders), args.verify_grid * 2, base.layers),
                 (max(args.orders), args.verify_grid * 2, base.layers * 2)]
    schedule = list(dict.fromkeys(schedule))
    args.output.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for m, q, layers in schedule:
        cfg = replace(base, order=m, grid=q, layers=layers)
        for row in get_rows(d["knots_nm"], d["shape"], d["structure_index"], args.wavelengths, cfg):
            all_rows.append(dict(order=m, grid=q, layers=layers, **row))
        print(f"verified M={m}, Q={q}, layers={layers}", flush=True)
        save_csv(args.output / "convergence.csv", all_rows)
    last = [r for r in all_rows if (r["order"], r["grid"], r["layers"]) == schedule[-1]]
    previous = [r for r in all_rows if (r["order"], r["grid"], r["layers"]) == schedule[-2]]
    changes = [abs(float(np.angle(np.exp(1j * np.deg2rad(a["retardance_deg"] - b["retardance_deg"])), deg=True)))
               for a, b in zip(last, previous) if a["phase_valid"] and b["phase_valid"]]
    cfg = replace(base, order=schedule[-1][0], grid=schedule[-1][1], layers=schedule[-1][2])
    final = design_payload(d["knots_nm"], d["shape"], d["structure_index"], cfg,
                           last, "Fixed geometry, RCWA refinement sequence")
    final["last_z_refinement_max_phase_change_deg"] = max(changes) if changes else None
    export_design(args.output / "verified_design.json", final)
    print(json.dumps(final["metrics"], indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("screen", "optimize", "verify"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--shapes", nargs="+", choices=("ellipse", "rectangle"), default=["ellipse", "rectangle"])
    p.add_argument("--indices", nargs="+", type=float, default=[1.5, 2., 2.5])
    p.add_argument("--wavelengths", nargs="+", type=float, default=[400., 550., 700.])
    p.add_argument("--samples", type=int, default=32, help="Sobol candidates in addition to six physical seeds")
    p.add_argument("--nodes", type=int, default=3, help="Use 15 for independent monotone layer-scale degrees of freedom")
    p.add_argument("--order", type=int, default=3)
    p.add_argument("--layers", type=int, default=15)
    p.add_argument("--grid", type=int, default=128)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--period-x", type=float, default=200.)
    p.add_argument("--period-y", type=float, default=200.)
    p.add_argument("--height", type=float, default=500.)
    p.add_argument("--substrate-index", type=float, default=1.5)
    p.add_argument("--half-min", type=float, default=5.)
    p.add_argument("--half-max", type=float, default=95.)
    p.add_argument("--transmission-floor", type=float, default=.8)
    p.add_argument("--rectangle-method", choices=("asr-fr", "raster"), default="asr-fr")
    p.add_argument("--seed", type=int, default=20260806)
    p.add_argument("--goal", choices=("max-phase", "maximin-phase", "qwp"), default="qwp")
    p.add_argument("--design", type=Path)
    p.add_argument("--maxiter", type=int, default=20)
    p.add_argument("--popsize", type=int, default=5)
    p.add_argument("--orders", nargs="+", type=int, default=[3, 5, 7])
    p.add_argument("--verify-grid", type=int, default=256)
    args = p.parse_args()
    if (args.samples < 0 or args.threads < 1 or args.maxiter < 0 or args.popsize < 1
            or len(set(args.wavelengths)) != len(args.wavelengths)
            or any(w <= 0 for w in args.wavelengths)):
        p.error("Invalid sampling, threading, optimization or wavelength setting.")
    torch.set_num_threads(args.threads)
    cfg = Config(period_x_nm=args.period_x, period_y_nm=args.period_y,
                 height_nm=args.height, n_substrate=args.substrate_index,
                 half_min_nm=args.half_min, half_max_nm=args.half_max,
                 layers=args.layers, nodes=args.nodes, order=args.order, grid=args.grid,
                 device=args.device, rectangle_method=args.rectangle_method,
                 transmission_floor=args.transmission_floor)
    cfg.validate()
    if args.mode == "verify":
        verify(args)
    elif args.mode == "optimize":
        optimize(args, cfg)
    else:
        screen(args, cfg)


if __name__ == "__main__":
    main()
