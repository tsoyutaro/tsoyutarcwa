"""Plot saved square-lattice NV/ASR results; no optical recalculation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

from studies.dielectric_motheye_n25.audit import compatible
from studies.dielectric_motheye_n25.model import adjacent_error, mean_reflectance, physical

ROOT = Path(__file__).resolve().parent
COLORS = {"nv": "#2563eb", "asr": "#d97706"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path,
        default=ROOT / "studies/dielectric_motheye_n25/results/repaired_v3")
    args = parser.parse_args(argv)
    source = args.output_root.resolve()
    destination = source / "square_review"
    plan = read(source / "plan.json")
    pair = {m: read(source / "square" / m / "convergence.json") for m in COLORS}
    records = {m: next(r for r in reversed(g["history"]) if r["axis"] == "order")
               for m, g in pair.items()}
    fixed = {m: {k: r["fixed"][k] for k in ("slices", "grid")} for m, r in records.items()}
    if fixed["nv"] != fixed["asr"] or records["nv"]["candidates"] != records["asr"]["candidates"]:
        raise ValueError("An equal Nz/Q and common M scan is required for this plot.")
    orders = records["nv"]["candidates"]
    spectra = {m: {int(k): rows for k, rows in r["spectra"].items()} for m, r in records.items()}
    wavelengths = sorted({r["wavelength_nm"] for r in spectra["nv"][orders[0]]})
    if not all(sorted(r["wavelength_nm"] for r in spectra[m][order]) == wavelengths
               for m in COLORS for order in orders):
        raise ValueError("The order scans have different wavelengths.")
    def row_at(method, order, wavelength):
        return next(r for r in spectra[method][order] if r["wavelength_nm"] == wavelength)
    data = []
    for method in COLORS:
        for order in orders:
            for wavelength in wavelengths:
                row = row_at(method, order, wavelength)
                endpoint = row_at(method, orders[-1], wavelength)["reflectance"]
                data.append({"method": method, "order_M": order, "scalar_harmonics": (2*order+1)**2,
                    **fixed[method], "wavelength_nm": wavelength,
                    "reflectance_fraction": row["reflectance"], "reflectance_percent": 100*row["reflectance"],
                    "distance_to_own_largest_M_pp": 100*abs(row["reflectance"]-endpoint),
                    "runtime_seconds": row["runtime_seconds"]})
    maximum, metrics = adjacent_error(pair["nv"]["final_spectrum"], pair["asr"]["final_spectrum"])
    review = {"scope": "Saved initial-geometry forward results and baseline audit timings; "
              "gradient mesh validation is separate. No new optical solves.",
        "geometry": plan["assumptions"]["geometry"], "integration": plan.get("integration"),
        "fixed_order_scan": fixed["nv"], "orders": orders, "order_scan_wavelengths_nm": wavelengths,
        "comparison_tolerance": plan["tolerance"], "mesh_tolerance": plan.get("mesh_tolerance", plan["tolerance"]),
        "cross_method_max_abs_change": maximum, "cross_method_per_metric": metrics, "methods": {}}
    provenance = [source / "plan.json"]
    for method, group in pair.items():
        n = group["recommendation"]
        folder = source / "square" / method
        audit_path = folder / "audits" / f"M{n['order']}_Nz{n['slices']}_Q{n['grid']}" / "audit_checkpoint.json"
        tasks = read(audit_path)["tasks"]
        provenance.extend([folder / "convergence.json", folder / "checkpoint.json", audit_path])
        timing = {}
        for kind in ("forward", "gradient"):
            samples = [tasks[f"{kind}_{i}"] for i in range(plan["benchmark_repeats"])
                       if f"{kind}_{i}" in tasks]
            timing[kind] = {"samples_seconds": [r["seconds"] for r in samples],
                "median_seconds": statistics.median(r["seconds"] for r in samples) if samples else None,
                "max_peak_cuda_allocated_mib": max(r["peak_cuda_allocated_mb"] for r in samples) if samples else None}
        baseline = tasks.get("gradient_warmup", {})
        fd_checks = []
        for parameter, step in plan["gradient_audit"]["steps_nm"].items():
            ad = baseline.get("gradient_per_nm", {}).get(parameter)
            fds = []
            for scale in (1., .5):
                plus = tasks.get(f"fd_{parameter}_{scale}_plus")
                minus = tasks.get(f"fd_{parameter}_{scale}_minus")
                if plus and minus:
                    fds.append((plus["objective"]-minus["objective"])/(2*step*scale))
            fd_checks.append({"parameter": parameter, "autograd_per_nm": ad,
                "finite_differences_per_nm": fds, "passed": ad is not None and len(fds) == 2
                    and compatible(ad, fds[1], plan["gradient_audit"]["atol_per_nm"], plan["gradient_audit"]["rtol"])
                    and compatible(fds[0], fds[1], plan["gradient_audit"]["atol_per_nm"], plan["gradient_audit"]["rtol"])})
        rows = group["final_spectrum"]
        mesh = group.get("final_spectrum_mesh_validation") or {}
        cases = list(read(folder / "checkpoint.json")["cases"].values())
        review["methods"][method] = {"saved_status": group["status"], "numerical": n,
            "wavelength_count": len(rows), "mean_reflectance": mean_reflectance(rows),
            "forward_mesh_passed": bool(group["final_spectrum_complete"] and mesh.get("complete")
                and mesh.get("passed") and all(physical(r, plan["energy_tolerance"]) for r in rows)),
            "max_abs_energy_residual_final": max(abs(r["energy_residual"]) for r in rows),
            "max_abs_energy_residual_all_cases": max(abs(r["energy_residual"]) for r in cases),
            "timing": timing, "baseline_fd_checks": fd_checks,
            "gradient_mesh_passed": bool((group.get("audit") or {}).get("gradient_passed")),
            "gradient_error": (group.get("audit") or {}).get("gradient_error"),
            "saved_error": group.get("error")}
    review["nv_forward_speedup"] = (review["methods"]["asr"]["timing"]["forward"]["median_seconds"] /
                                     review["methods"]["nv"]["timing"]["forward"]["median_seconds"])
    review["nv_baseline_gradient_speedup"] = (review["methods"]["asr"]["timing"]["gradient"]["median_seconds"] /
                                             review["methods"]["nv"]["timing"]["gradient"]["median_seconds"])
    review["source_file_sha256"] = {p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in provenance}
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "review.json").write_text(json.dumps(review, indent=2), encoding="utf-8")
    with (destination / "reflectance_vs_order.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    def save(fig, name):
        for extension in ("png", "svg"):
            fig.savefig(destination / f"{name}.{extension}", dpi=200, facecolor="white")
        plt.close(fig)
    display_ws = [w for w in (400., 550., 700.) if w in wavelengths]
    if len(display_ws) != 3:
        raise ValueError("This three-panel figure requires 400, 550 and 700 nm in the saved scan.")
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.1), layout="constrained")
    for ax, wavelength in zip(axes, display_ws):
        for method in COLORS:
            ax.plot(orders, [100*row_at(method, m, wavelength)["reflectance"] for m in orders],
                    "o-" if method == "nv" else "s--", color=COLORS[method], label=method.upper())
        ax.set(title=f"{wavelength:g} nm", xlabel="Retained Fourier order M", ylabel="Reflectance R (%)", xticks=orders)
        ax.ticklabel_format(axis="y", useOffset=False, style="plain")
        ax.grid(alpha=.25); ax.legend()
    geometry = review["geometry"]
    fig.suptitle(f"Square-lattice n={geometry['index']:g} moth-eye: absolute reflectance\n"
                 f"P={geometry['period_nm']:g} nm, H={geometry['height_nm']:g} nm, "
                 f"tip/base={geometry['tip_radius_nm']:g}/{geometry['base_radius_nm']:g} nm; "
                 f"Nz={fixed['nv']['slices']}, Q={fixed['nv']['grid']}; normal x polarization",
                 fontsize=11)
    save(fig, "reflectance_vs_order")
    fig, ax = plt.subplots(figsize=(8.7, 4.7), layout="constrained")
    for method in COLORS:
        for wavelength, style in zip(display_ws, ("-", "--", ":")):
            ref = row_at(method, orders[-1], wavelength)["reflectance"]
            xs = [m for m in orders if abs(row_at(method, m, wavelength)["reflectance"]-ref) > 0]
            ax.semilogy(xs, [100*abs(row_at(method, m, wavelength)["reflectance"]-ref) for m in xs],
                marker="o" if method == "nv" else "s", ls=style, color=COLORS[method], label=f"{method.upper()} {wavelength:g} nm")
    ax.axhline(100*review["mesh_tolerance"], color="#b91c1c", ls="-.", label="self-refinement budget")
    ax.set(xlabel="Retained Fourier order M", ylabel=f"|R(M) - R({orders[-1]})| (percentage points)",
           xticks=orders, title=f"Distance to each method's own M={orders[-1]} result\nNot an error against an exact solution; zero endpoints omitted")
    ax.grid(alpha=.25); ax.legend(ncol=2, fontsize=9)
    save(fig, "distance_to_largest_order")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), layout="constrained")
    for ax, kind, title in zip(axes, ("forward", "gradient"), ("Forward objective", "Forward + AD objective")):
        for i, method in enumerate(COLORS):
            sample = review["methods"][method]["timing"][kind]
            value = sample["median_seconds"]
            ax.bar(i, value, color=COLORS[method], width=.6)
            ax.errorbar(i, value, yerr=[[value-min(sample['samples_seconds'])], [max(sample['samples_seconds'])-value]],
                        color="#334155", capsize=4, fmt="none")
            ax.text(i, max(sample['samples_seconds']) + .025*max(review['methods'][m]['timing'][kind]['median_seconds'] for m in COLORS),
                    f"{value:.1f} s", ha="center", va="bottom")
        ax.set(title=title, xticks=[0, 1], xticklabels=["NV", "ASR"], ylabel="Warm median time (s)")
        ax.margins(y=.2); ax.grid(axis="y", alpha=.2)
    conditions = "; ".join(f"{m.upper()}: M={r['numerical']['order']}, Nz={r['numerical']['slices']}, Q={r['numerical']['grid']}"
                            for m, r in review["methods"].items())
    fig.suptitle(f"Square; {conditions}; {len(plan['gradient_audit']['wavelengths_nm'])} audit wavelengths\n"
                 "3 measured runs (bars: median; whiskers: range). Gradient mesh validation incomplete.", fontsize=10)
    save(fig, "measured_cost")
    print(json.dumps({"forward_nv_speedup": review['nv_forward_speedup'],
                      "baseline_gradient_nv_speedup": review['nv_baseline_gradient_speedup'],
                      "cross_method_max_abs_change": maximum, "output": str(destination)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
