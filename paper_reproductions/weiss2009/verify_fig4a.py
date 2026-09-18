"""Compute and compare the full Weiss 2009 Fig. 4(a) spectrum.

Default: 625 harmonics, both polarizations, grids 256/512, all 101 PDF
reference knots. Every point is checkpointed. --resume reuses only results
with the same physics, precision, cascade, reference and solver source hash.
--analyze-only produces plots/metrics without importing torch or torcwa.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
CHANNELS = ("T", "R", "A")
SCHEMA = "weiss2009-fig4a-verification-v2"
PHYSICS = {"period_um": .7, "radius_um": .15, "height_um": .05,
           "epsilon_background": 1., "epsilon_input": 1., "epsilon_output": 1.,
           "gold_epsilon_inf": 1., "gold_plasma_rad_s": 1.37e16,
           "gold_damping_rad_s": .85e14, "eta": .97,
           "incidence_degrees": 0., "profile": "weiss2009",
           "factorization": "Weiss symmetric", "conversion": "general-2d-T"}


def atomic_json(path, payload):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hash():
    digest = hashlib.sha256()
    paths = sorted((ROOT / "rcwa_ext").rglob("*.py")) + [PACKAGE / "reproduce.py", Path(__file__)]
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).replace("\\", "/").encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_reference():
    path = PACKAGE / "reference" / "fig4a_reference.csv"
    metadata = json.loads((path.parent / "fig4a_reference_metadata.json").read_text(encoding="utf-8"))
    if sha(path) != metadata["reference_csv_sha256"]:
        raise ValueError("Reference CSV differs from its extraction provenance.")
    data = np.genfromtxt(path, delimiter=",", names=True)
    if len(data) < 3 or not np.all(np.diff(data["frequency_THz"]) > 0):
        raise ValueError("Invalid reference frequency sequence.")
    return data, metadata


def parse_frequencies(value, reference):
    if value == "reference":
        return list(map(float, reference["frequency_THz"]))
    result = []
    for token in value.split(","):
        parts = [float(p) for p in token.split(":")]
        if not all(math.isfinite(p) for p in parts):
            raise ValueError("Frequency must be finite.")
        if len(parts) == 1:
            result += parts
        elif len(parts) == 3 and parts[2] > 0 and parts[1] >= parts[0]:
            start, stop, step = parts
            result.extend(start + i * step for i in range(int((stop-start)/step + 1e-9) + 1))
        else:
            raise ValueError("Use reference, comma-separated THz, or start:stop:step.")
    result = sorted(set(round(f, 10) for f in result))
    if not result or result[0] <= 0:
        raise ValueError("Frequencies must be positive.")
    return result


def row_key(row):
    return (int(row["grid"]), round(float(row["frequency_THz"]), 10))


def select_rows(rows, grids, frequencies):
    wanted = {(g, round(f, 10)) for g in grids for f in frequencies}
    selected = {}
    for row in rows:
        key = row_key(row)
        if key in selected:
            raise ValueError(f"Duplicate checkpoint row: {key}")
        if key in wanted:
            selected[key] = row
    return [selected[key] for key in sorted(selected)]


def error_metrics(delta, frequencies):
    i = int(np.argmax(np.abs(delta)))
    return {"max_abs": float(np.max(np.abs(delta))),
            "rmse": float(np.sqrt(np.mean(delta**2))),
            "mean_abs": float(np.mean(np.abs(delta))),
            "max_error_frequency_THz": float(frequencies[i])}


def power_column(channel, polarization, mode):
    """A is physical absorption regardless of the selected outgoing orders."""
    suffix = "0" if mode == "zeroth" and channel in ("T", "R") else ""
    return f"{channel}{suffix}_{polarization}"


def analyze(rows, reference, grids, frequencies, args):
    """Pure comparison logic; no solver import and no normalization of R/T/A."""
    rows = select_rows(rows, grids, frequencies)
    expected = len(grids) * len(frequencies)
    complete = len(rows) == expected
    required = [f"{c}_{p}" for p in ("x", "y") for c in (*CHANNELS, "T0", "R0")]
    finite = bool(rows) and all(math.isfinite(float(r[k])) for r in rows
                                for k in required + ["minimum_jacobian"])
    passive = finite and all(-args.physical_atol <= float(r[k]) <= 1 + args.physical_atol
                             for r in rows for k in required)
    passive = passive and all(float(r[f"{c}0_{p}"]) <= float(r[f"{c}_{p}"]) + args.physical_atol
                              for r in rows for p in ("x", "y") for c in ("T", "R"))
    jacobian = finite and all(float(r["minimum_jacobian"]) > 0 for r in rows)
    symmetry_error = max((abs(float(r[f"{c}_x"]) - float(r[f"{c}_y"]))
                          for r in rows for c in (*CHANNELS, "T0", "R0")), default=math.inf)
    order_ok = args.order == 12 and all(int(r["order"]) == 12 for r in rows)
    fref = reference["frequency_THz"]
    coverage = (len(frequencies) >= 2 and frequencies[0] <= fref[0] + 1e-8
                and frequencies[-1] >= fref[-1] - 1e-8
                and max(np.diff(frequencies)) <= args.max_frequency_step)
    summary = {"schema": SCHEMA, "status": "inconclusive", "rows": len(rows),
               "expected_rows": expected, "complete": complete,
               "full_frequency_coverage": bool(coverage), "paper_625_harmonics": order_ok,
               "finite": finite, "passive": passive, "positive_jacobian": jacobian,
               "max_polarization_difference": symmetry_error if math.isfinite(symmetry_error) else None,
               "paper_metrics": {}, "alternative_power_metrics": {},
               "paper_power": args.paper_power,
               "first_diffraction_threshold_THz": 299.792458 / PHYSICS["period_um"],
               "power_definition_note": "Default comparison uses zeroth-order T0/R0 and total "
               "absorption. This is inferred from the plotted loss of T+R+A above the first "
               "diffraction threshold; the paper caption does not specify the orders. "
               "Total-order metrics are also provided; the mode is never selected by best fit.",
               "grid_metrics": {}, "grid_converged": False,
               "reasons": [], "thresholds": {
                   "paper_max_abs": args.paper_atol, "paper_rmse": args.paper_rmse,
                   "grid_max_abs": args.grid_atol, "passivity_abs": args.physical_atol,
                   "polarization_abs": args.symmetry_atol,
                   "max_frequency_step_THz": args.max_frequency_step},
               "interpretation": "Agreement with digitized Fig. 4(a) within user-defined "
               "absolute tolerances. Not agreement with authors' raw data or a proof of "
               "Fourier-order convergence. A=1-R-T is not an independent conservation test."}
    comparisons, grid_rows = [], []
    finest = sorted([r for r in rows if int(r["grid"]) == max(grids)],
                    key=lambda r: float(r["frequency_THz"]))
    if finite and len(finest) >= 2:
        fs = np.array([float(r["frequency_THz"]) for r in finest])
        valid = (fref >= fs[0] - 1e-8) & (fref <= fs[-1] + 1e-8)
        ref_subset = reference[valid]
        if len(ref_subset):
            ff = ref_subset["frequency_THz"]
            simulated = {c: np.interp(ff, fs, [float(r[power_column(c, "x", args.paper_power)]) for r in finest])
                         for c in CHANNELS}
            alternative_mode = "total" if args.paper_power == "zeroth" else "zeroth"
            for c in CHANNELS:
                alternative = np.interp(ff, fs, [float(r[power_column(c, "x", alternative_mode)]) for r in finest])
                summary["alternative_power_metrics"][c] = error_metrics(alternative - ref_subset[c], ff)
            summary["alternative_power_mode"] = alternative_mode
            for c in CHANNELS:
                summary["paper_metrics"][c] = error_metrics(simulated[c] - ref_subset[c], ff)
            for i, f in enumerate(ff):
                record = {"frequency_THz": float(f)}
                for c in CHANNELS:
                    record.update({f"{c}_paper": float(ref_subset[c][i]),
                                   f"{c}_computed": float(simulated[c][i]),
                                   f"{c}_difference": float(simulated[c][i] - ref_subset[c][i])})
                comparisons.append(record)
        summary["sampled_extrema"] = {}
        for c, kind in (("T", "min"), ("R", "max"), ("A", "max")):
            values = np.array([float(r[power_column(c, "x", args.paper_power)]) for r in finest])
            idx = int(np.argmin(values) if kind == "min" else np.argmax(values))
            summary["sampled_extrema"][f"{c}_{kind}"] = {
                "frequency_THz": float(fs[idx]), "value": float(values[idx])}
    # Compare every common frequency on the two finest grids, both polarizations.
    # A single grid can never be labelled grid-converged.
    if finite and len(grids) >= 2:
        low, high = sorted(grids)[-2:]
        lookup = {row_key(r): r for r in rows}
        for f in frequencies:
            keys = [(g, round(f, 10)) for g in (low, high)]
            if all(k in lookup for k in keys):
                a, b = [lookup[k] for k in keys]
                grid_rows.append({"frequency_THz": f, "coarse_grid": low, "fine_grid": high,
                                  **{k: float(b[k]) - float(a[k]) for k in required}})
        if grid_rows:
            for k in required:
                summary["grid_metrics"][k] = error_metrics(
                    np.array([r[k] for r in grid_rows]),
                    np.array([r["frequency_THz"] for r in grid_rows]))
            summary["grid_converged"] = (len(grid_rows) == len(frequencies)
                and all(v["max_abs"] <= args.grid_atol for v in summary["grid_metrics"].values()))
    reasons = summary["reasons"]
    for ok, reason in [(complete, "incomplete requested sweep"),
                       (coverage, "partial or too sparse frequency coverage"),
                       (order_ok, "not the paper's 625 harmonics"),
                       (finite, "missing or nonfinite result"),
                       (passive, "passivity check failed"),
                       (jacobian, "nonpositive mapping Jacobian"),
                       (symmetry_error <= args.symmetry_atol, "x/y symmetry check failed"),
                       (summary["grid_converged"], "grid convergence not established")]:
        if not ok:
            reasons.append(reason)
    if not reasons and len(comparisons) == len(reference):
        matched = all(v["max_abs"] <= args.paper_atol and v["rmse"] <= args.paper_rmse
                      for v in summary["paper_metrics"].values())
        summary["status"] = "matched_within_tolerance" if matched else "mismatch"
        if not matched:
            reasons.append("paper curve error exceeds requested tolerance")
    return summary, comparisons, grid_rows, finest


def plot_results(output, reference, finest, comparisons, grid_rows, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"T": "#1769aa", "R": "#d95f02", "A": "#238b45"}
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True, constrained_layout=True)
    for c in CHANNELS:
        axes[0].plot(reference["frequency_THz"], reference[c], "--", color=colors[c],
                     lw=1.2, label=f"{c} paper (vector digitized)")
        if finest:
            axes[0].plot([r["frequency_THz"] for r in finest],
                         [r[power_column(c, "x", summary["paper_power"])] for r in finest],
                         color=colors[c], lw=1.4, label=f"{c} computed")
        if comparisons:
            axes[1].plot([r["frequency_THz"] for r in comparisons],
                         [r[f"{c}_difference"] for r in comparisons], color=colors[c], label=c)
        if grid_rows:
            axes[2].plot([r["frequency_THz"] for r in grid_rows],
                         [max(abs(r[power_column(c, "x", summary["paper_power"])]),
                              abs(r[power_column(c, "y", summary["paper_power"])])) for r in grid_rows],
                         color=colors[c], label=c)
    axes[0].set_ylabel("Power fraction")
    axes[0].legend(ncol=2, fontsize=8)
    axes[0].set_title("Weiss 2009 Fig. 4(a) | " + summary["status"].replace("_", " ")
                      + "\nT/R: " + summary["paper_power"] + " orders; A: total absorption")
    axes[1].set_ylabel("Computed - paper")
    for sign in (-1, 1):
        axes[1].axhline(sign * summary["thresholds"]["paper_max_abs"], color="gray", ls=":")
    axes[2].set_ylabel("Absolute grid difference")
    axes[2].set_xlabel("Frequency (THz)")
    axes[2].axhline(summary["thresholds"]["grid_max_abs"], color="gray", ls=":")
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(output / "fig4a_comparison.png", dpi=180)
    plt.close(fig)


def make_signature(args, reference_metadata):
    return {"schema": SCHEMA, "physics": PHYSICS, "order": args.order,
            "dtype": "complex128", "cascade": args.cascade,
            "solver_source_sha256": source_hash(),
            "reference_csv_sha256": reference_metadata["reference_csv_sha256"]}


def run(args):
    reference, metadata = load_reference()
    frequencies = parse_frequencies(args.frequencies, reference)
    grids = sorted(set(int(g) for g in args.grids.split(",")))
    if args.order < 1 or not grids or min(grids) < max(32, 4 * args.order + 4):
        raise ValueError("Require order>=1 and grids>=max(32,4*order+4).")
    for value in (args.paper_atol, args.paper_rmse, args.grid_atol, args.physical_atol,
                  args.symmetry_atol, args.max_frequency_step):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Tolerances and maximum frequency step must be finite and positive.")
    if args.threads < 1:
        raise ValueError("--threads must be positive.")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "fig4a_checkpoint.json"
    signature = make_signature(args, metadata)
    saved = {"signature": signature, "rows": []}
    if args.resume or args.analyze_only:
        if not checkpoint.exists():
            if args.analyze_only:
                raise ValueError("--analyze-only requires an existing checkpoint.")
        else:
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
            if saved["signature"] != signature:
                raise ValueError("Checkpoint physics/source/reference mismatch; use a new output directory.")
    elif checkpoint.exists():
        raise ValueError("Output already has a checkpoint; specify --resume or a new output directory.")
    report_path = output / "fig4a_report.json"
    if not args.analyze_only:
        # Invalidate an earlier 'matched' report before starting a new/extended run.
        atomic_json(report_path, {"schema": SCHEMA, "status": "running",
                                  "frequencies_THz": frequencies, "grids": grids})
        try:
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))
            import torch
            from paper_reproductions.weiss2009 import reproduce as solver
            torch.set_num_threads(args.threads)
            device = solver.select_device(args.device)
            available = {row_key(r) for r in saved["rows"]}
            for g in grids:
                for f in frequencies:
                    if (g, round(f, 10)) in available:
                        continue
                    print(f"run N={args.order} grid={g} f={f:.6f} THz", flush=True)
                    row = solver.simulate_scattering(
                        frequency_thz=f, order=args.order, period_um=PHYSICS["period_um"],
                        radius_um=PHYSICS["radius_um"], height_um=PHYSICS["height_um"],
                        epsilon=solver.gold_drude_epsilon(f), profile="weiss2009", grid=g,
                        dtype=torch.complex128, device=device, cascade=args.cascade,
                        both_polarizations=True)
                    row["device"] = str(device)
                    row["torch_version"] = torch.__version__
                    for key in [f"{c}_{p}" for p in ("x", "y") for c in (*CHANNELS, "T0", "R0")] + ["minimum_jacobian"]:
                        if not math.isfinite(float(row[key])):
                            raise RuntimeError(f"Nonfinite result at f={f}, grid={g}: {key}")
                    saved["rows"].append(row)
                    atomic_json(checkpoint, saved)
                    print(f"  T0={row['T0_x']:.8f} R0={row['R0_x']:.8f} A={row['A_x']:.8f} "
                          f"(total T={row['T_x']:.8f} R={row['R_x']:.8f})", flush=True)
        except BaseException as exc:
            atomic_json(report_path, {"schema": SCHEMA, "status": "interrupted_or_failed",
                                      "error": f"{type(exc).__name__}: {exc}",
                                      "completed_cached_rows": len(saved["rows"])})
            raise
    rows = select_rows(saved["rows"], grids, frequencies)
    summary, comparisons, grid_rows, finest = analyze(rows, reference, grids, frequencies, args)
    summary.update({"signature": signature, "reference": {
        k: metadata[k] for k in ("source_kind", "paper_doi", "figure", "source_pdf_sha256", "note")},
        "grids": grids, "frequencies_THz": frequencies})
    # Remove derived tables for which this run has no data; never show stale results.
    for name, data in [("fig4a_spectrum.csv", rows), ("fig4a_pointwise_errors.csv", comparisons),
                       ("fig4a_grid_errors.csv", grid_rows)]:
        path = output / name
        if data:
            write_csv(path, data)
        elif path.exists():
            path.unlink()
    try:
        plot_results(output, reference, finest, comparisons, grid_rows, summary)
    except Exception as exc:
        atomic_json(report_path, {"schema": SCHEMA, "status": "report_generation_failed",
                                  "error": f"{type(exc).__name__}: {exc}",
                                  "computed_rows": len(rows)})
        raise
    atomic_json(report_path, summary)
    lines = ["# Weiss 2009 Fig. 4(a) verification", "", f"Status: **{summary['status']}**", "",
             f"Computed rows: {len(rows)}/{summary['expected_rows']}; N={args.order}; grids={grids}.", "",
             "Reference: vector polylines extracted from the publisher PDF, not raw author data.", "",
             "| Curve | Max absolute error | RMSE |", "|---|---:|---:|"]
    for c, m in summary["paper_metrics"].items():
        lines.append(f"| {c} | {m['max_abs']:.8g} | {m['rmse']:.8g} |")
    lines += ["", "Reasons: " + ("; ".join(summary["reasons"]) or "all requested checks passed"),
              "", f"Paper T/R comparison: {args.paper_power}. A always uses all orders.",
              "", summary["power_definition_note"], "", summary["interpretation"],
              "", "See fig4a_report.json for tolerances and provenance."]
    (output / "fig4a_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{summary['status']}: {report_path}")
    if args.strict_exit_code and summary["status"] != "matched_within_tolerance":
        return 2 if summary["status"] == "mismatch" else 3
    return 0


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frequencies", default="reference", help="reference (default), or THz start:stop:step/list")
    p.add_argument("--order", type=int, default=12)
    p.add_argument("--grids", default="256,512", help="Compare the two finest requested grids over the full sweep")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--cascade", choices=("redheffer", "algo2a"), default="redheffer")
    p.add_argument("--paper-power", choices=("zeroth", "total"), default="zeroth",
                   help="T/R definition for PDF comparison; A always includes all orders")
    p.add_argument("--paper-atol", type=float, default=.02, help="Maximum absolute power difference from PDF curve")
    p.add_argument("--paper-rmse", type=float, default=.01)
    p.add_argument("--grid-atol", type=float, default=.005)
    p.add_argument("--physical-atol", type=float, default=1e-6)
    p.add_argument("--symmetry-atol", type=float, default=1e-6)
    p.add_argument("--max-frequency-step", type=float, default=3.)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--analyze-only", action="store_true")
    p.add_argument("--strict-exit-code", action="store_true", help="Exit 2 on mismatch, 3 on inconclusive")
    p.add_argument("--output-dir", type=Path, default=PACKAGE / "results" / "fig4a_verification")
    return p


if __name__ == "__main__":
    sys.exit(run(parser().parse_args()))
