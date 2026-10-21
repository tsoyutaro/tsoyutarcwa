"""PMMA/Au order sweeps at several heights, retaining layer passivity diagnostics.

The existing height-diagnostic worker and memory-safe NV solver are unchanged.
Only M varies within each fixed-height scan. Tail checks do not certify Nz/Q
convergence, and numerical consistency alone does not establish passivity.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import run_au_nv_height_diagnostics as height

ROOT = Path(__file__).resolve().parent


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--heights-nm", default="400,300,200")
    p.add_argument("--wavelengths", default="700")
    p.add_argument("--start-order", type=int, default=4)
    p.add_argument("--step", type=int, default=2)
    p.add_argument("--max-order", type=int, default=36)
    p.add_argument("--slices", type=int, default=210)
    p.add_argument("--grid", type=int, default=512)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--quadrature", type=int, default=16)
    p.add_argument("--tolerance", type=float, default=.005, help="Adjacent R/T/A change, power fraction; .005 = .5 pp.")
    p.add_argument("--negative-loss-tolerance", type=float, default=1e-6)
    p.add_argument("--pmma-csv", type=Path)
    p.add_argument("--output-root", type=Path,
                   default=ROOT/"studies/au_pmma_nv_height_orders/results/P200_H400-200_Nz210_Q512_M4-36_700nm")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--retry-failures", action="store_true")
    return p


def forwarded_arguments(args):
    if (args.start_order < 1 or args.step < 1 or args.max_order < args.start_order
            or (args.max_order-args.start_order) % args.step or args.slices < 2
            or args.grid < max(32, 4*args.max_order+4)
            or not math.isfinite(args.tolerance) or args.tolerance <= 0):
        raise ValueError("Use positive orders/step, an aligned finite max-order, Nz>=2, Q>=max(32,4*Mmax+4), and positive tolerance.")
    orders = list(range(args.start_order, args.max_order+1, args.step))
    argv = ["--device", args.device, "--orders", f"Au:{args.start_order}",
        "--order-offsets", ",".join(str(m-args.start_order) for m in orders),
        "--structures", "pmma", "--factorizations", "symmetric-nv",
        "--heights-nm", args.heights_nm, "--wavelengths", args.wavelengths,
        "--slices", str(args.slices), "--grids", str(args.grid),
        "--cpu-threads", str(args.cpu_threads), "--quadrature", str(args.quadrature),
        "--negative-loss-tolerance", str(args.negative_loss_tolerance),
        "--output-root", str(args.output_root), "--no-reuse-reference", "--no-plots"]
    for flag, enabled in (("--dry-run", args.dry_run), ("--plot-only", args.plot_only),
                          ("--retry-failures", args.retry_failures)):
        if enabled:
            argv.append(flag)
    if args.pmma_csv is not None:
        argv += ["--pmma-csv", str(args.pmma_csv)]
    return orders, argv


def tail_checks(rows, orders, tolerance):
    tail = orders[-3:]
    groups = sorted({(r["height_nm"], r["wavelength_nm"]) for r in rows})
    result = []
    for h, wave in groups:
        by_order = {r["order"]: r for r in rows if (r["height_nm"], r["wavelength_nm"]) == (h, wave)}
        completed = len(tail) == 3 and all(by_order.get(m, {}).get("status") == "diagnostics_complete" for m in tail)
        changes = []
        for coarse, fine in zip(tail, tail[1:]):
            a, b = by_order.get(coarse, {}), by_order.get(fine, {})
            ready = a.get("status") == b.get("status") == "diagnostics_complete"
            change = dict(coarse=coarse, fine=fine, comparison_complete=ready)
            for field in ("R", "T", "A"):
                change[f"delta_{field}_pp"] = abs(b[f"{field}_percent"]-a[f"{field}_percent"]) if ready else None
            change["maximum_change_pp"] = max(change[f"delta_{f}_pp"] for f in ("R", "T", "A")) if ready else None
            changes.append(change)
        stable = bool(completed and all(c["maximum_change_pp"] <= 100*tolerance for c in changes))
        physical = bool(completed and all(by_order[m]["physical_result"] for m in tail))
        consistent = bool(completed and all(by_order[m]["diagnostic_checks_passed"] for m in tail))
        result.append(dict(height_nm=h, wavelength_nm=wave, tail_orders=tail,
            comparisons=changes, tail_complete=completed, RTA_tail_within_tolerance=stable,
            physical_tail_passed=physical, consistency_tail_passed=consistent,
            sampled_tail_checks_passed=stable and physical and consistent))
    return result


def plot_orders(rows, orders, output):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib unavailable; CSV/JSON data are intact.", flush=True)
        return
    heights = sorted({r["height_nm"] for r in rows}, reverse=True)
    waves = sorted({r["wavelength_nm"] for r in rows})
    colors = {wave: color for wave, color in zip(waves, ("#0072B2", "#D55E00", "#009E73"))}
    for h in heights:
        group = [r for r in rows if r["height_nm"] == h]
        first = group[0]
        for name, fields, labels in (
            ("RTA", ("R_percent", "T_percent", "A_percent"),
             ("Reflectance R (%)", "Transmittance T (%)", "Absorptance A (%)")),
            ("passivity", ("negative_absorption_layers", "minimum_layer_absorption_pp", "negative_loss_sector_pp"),
             ("Negative-absorption layers", "Minimum layer absorption (pp)", "Negative eigensector contribution (pp)"))):
            fig, axes = plt.subplots(1, 3, figsize=(14, 4.7))
            for wave in waves:
                selected = {r["order"]: r for r in group if r["wavelength_nm"] == wave}
                for ax, field, label in zip(axes, fields, labels):
                    # A failed or missing case breaks the line rather than linking across it.
                    values = [selected[m].get(field) if selected.get(m, {}).get("status") == "diagnostics_complete" else math.nan for m in orders]
                    ax.plot(orders, values, "o-", color=colors.get(wave), label=f"{wave:g} nm", ms=4)
                    if name == "RTA":
                        bad = [m for m in orders if selected.get(m, {}).get("status") == "diagnostics_complete" and selected[m]["negative_absorption_layers"]]
                        ax.scatter(bad, [selected[m][field] for m in bad], marker="x", color="#b2182b", s=60, linewidths=1.8, zorder=5)
                    ax.set(xlabel="Fourier truncation order M", ylabel=label)
                    ax.set_xticks(orders[::2] if len(orders) > 10 else orders)
                    ax.grid(alpha=.25)
            axes[0].legend()
            fig.suptitle(f"PMMA + Au | NV | P=200 nm, h={h:g} nm, Au=30 nm | Nz={first['slices']}+cap, Q={first['grid']}")
            fig.text(.06, .02, "Red crosses: negative integrated layer absorption. Fixed Nz/Q scan; no layer/grid convergence claim.", fontsize=9)
            fig.tight_layout(rect=(0, .06, 1, .93))
            for ext in ("png", "svg"):
                fig.savefig(output/f"order_{name}_H{h:g}_Nz{first['slices']}_Q{first['grid']}.{ext}", dpi=160)
            plt.close(fig)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        orders, forwarded = forwarded_arguments(args)
    except ValueError as error:
        parser.error(str(error))
    status = height.main(forwarded)
    if args.dry_run:
        return status
    output = args.output_root.resolve()
    summary = json.loads((output/"height_summary.json").read_text(encoding="utf-8"))
    checks = tail_checks(summary["cases"], orders, args.tolerance)
    height.nv.write_json(output/"order_tail_checks.json", dict(tolerance_pp=100*args.tolerance,
        checks=checks, interpretation="Last two adjacent refinements at fixed Nz/Q and sampled wavelengths only. Actual layer passivity and consistency are checked separately. This does not certify layer/grid convergence."))
    comparisons = [dict(height_nm=c["height_nm"], wavelength_nm=c["wavelength_nm"],
        RTA_tail_within_tolerance=c["RTA_tail_within_tolerance"], physical_tail_passed=c["physical_tail_passed"],
        consistency_tail_passed=c["consistency_tail_passed"], **pair) for c in checks for pair in c["comparisons"]]
    height.diagnostic.diagnostic.write_csv(output/"order_tail_changes.csv", comparisons)
    if not args.no_plots:
        plot_orders(summary["cases"], orders, output)
    for check in checks:
        print(f"h={check['height_nm']:g} nm, {check['wavelength_nm']:g} nm: "
              f"tail M={check['tail_orders']}; RTA stable={check['RTA_tail_within_tolerance']}; "
              f"physical={check['physical_tail_passed']}; consistency={check['consistency_tail_passed']}", flush=True)
    print(f"Order plots and tail checks: {output}", flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
