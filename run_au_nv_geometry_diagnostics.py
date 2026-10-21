"""Au/PMMA NV diagnostics with explicit period, height, radii and film thicknesses.

Uses the existing memory-safe solver and layer diagnostics unchanged. Geometry
changes describe different physical structures; they are not convergence tests.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys

import run_metals_nv_diagnostics as diagnostic

ROOT = Path(__file__).resolve().parent
nv = diagnostic.nv


def build_parser():
    p = diagnostic.build_parser()
    p.description = __doc__
    p.set_defaults(orders="Au:36", order_offsets="0", structures="both", wavelengths="700",
                   slices="210", grids="512", factorizations="symmetric-nv",
                   output_root=ROOT/"studies/au_nv_geometry_diagnostics/results/P100_H200_M36_Nz210_Q512_700nm_Au30")
    p.add_argument("--period-nm", type=float, default=100.)
    p.add_argument("--height-nm", type=float, default=200., help="Profile height, excluding the top metal cap.")
    p.add_argument("--tip-radius-nm", type=float, default=5., help="Au tip or PMMA core tip radius.")
    p.add_argument("--metal-base-radius-nm", type=float,
                   help="Au-only base radius; default is 0.475 times the period.")
    p.add_argument("--pmma-base-radius-nm", type=float,
                   help="PMMA core base radius; default is Au-only base radius minus side thickness.")
    p.add_argument("--metal-side-thickness-nm", type=float, default=30.)
    p.add_argument("--metal-top-thickness-nm", type=float, default=30.)
    p.add_argument("--metal-valley-thickness-nm", type=float, default=30.)
    return p


def build_jobs(args):
    if set(diagnostic.orders(args.orders)) != {"Au"}:
        raise ValueError("This geometry runner accepts Au only.")
    positive = (args.period_nm, args.height_nm, args.tip_radius_nm, args.metal_side_thickness_nm)
    nonnegative = (args.metal_top_thickness_nm, args.metal_valley_thickness_nm)
    if (any(not math.isfinite(x) or x <= 0 for x in positive)
            or any(not math.isfinite(x) or x < 0 for x in nonnegative)):
        raise ValueError("Period, height, tip and side thickness must be finite positive; top/valley thicknesses nonnegative.")
    metal_base = .475*args.period_nm if args.metal_base_radius_nm is None else args.metal_base_radius_nm
    pmma_base = metal_base-args.metal_side_thickness_nm if args.pmma_base_radius_nm is None else args.pmma_base_radius_nm
    jobs = deepcopy(diagnostic.build_jobs(args))
    for job in jobs:
        identity = job["identity"]
        g = identity["backend_identity"]["geometry"]
        g.update(period_nm=args.period_nm, height_nm=args.height_nm, tip_radius_nm=args.tip_radius_nm,
                 base_radius_nm=metal_base if identity["structure"] == "metal" else pmma_base)
        if identity["structure"] == "metal":
            if (not math.isfinite(metal_base) or metal_base < args.tip_radius_nm
                    or 2*metal_base >= args.period_nm):
                raise ValueError("Au-only radii require 0 < tip <= base < period/2.")
        else:
            from studies.pmma_metal_motheye.model import Geometry
            g.update(metal_side_thickness_nm=args.metal_side_thickness_nm,
                     metal_top_thickness_nm=args.metal_top_thickness_nm,
                     metal_valley_thickness_nm=args.metal_valley_thickness_nm)
            geom = Geometry(**g)
            geom.validate()
            geom.profile(identity["slices"])
        identity["sources"][Path(__file__).name] = nv.digest(__file__)
        job["directory"] = f"P{args.period_nm:g}_H{args.height_nm:g}/"+job["directory"]
    return jobs


def export(jobs, output, args):
    rows = diagnostic.collect(jobs, output, args.negative_loss_tolerance)
    for row, job in zip(rows, jobs):
        g = job["identity"]["backend_identity"]["geometry"]
        row.update({name: g.get(name) for name in ("period_nm", "height_nm", "tip_radius_nm", "base_radius_nm",
            "metal_side_thickness_nm", "metal_top_thickness_nm", "metal_valley_thickness_nm")})
        row["total_pattern_height_nm"] = g["height_nm"]+g.get("metal_top_thickness_nm", 0.)
        row["minimum_layer_absorption_pp"] = (None if row["minimum_layer_absorptance"] is None
                                               else 100*row["minimum_layer_absorptance"])
    diagnostic.diagnostic.write_csv(output/"geometry_summary.csv", rows)
    summary = dict(requested_cases=len(rows),
        counts={status: sum(r["status"] == status for r in rows) for status in sorted({r["status"] for r in rows})},
        physical_checks_passed=sum(bool(r["physical_result"]) for r in rows),
        consistency_checks_passed=sum(bool(r["diagnostic_checks_passed"]) for r in rows),
        strict_negative_loss_checks_passed=sum(bool(r["excited_negative_loss_passed"]) for r in rows),
        sample_checks_passed=sum(r["sample_checks_passed"] for r in rows),
        negative_loss_tolerance=args.negative_loss_tolerance, cases=rows,
        interpretation="Geometry changes are not convergence tests. Passivity, diagnostic consistency and the strict negative-eigensector warning are separate. Negative contributions are not R/T/A error estimates.")
    nv.write_json(output/"geometry_summary.json", summary)
    return rows, summary


def plot_comparison(rows, output):
    complete = [r for r in rows if r["status"] == "diagnostics_complete"]
    if not complete:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    keys = ("factorization", "order", "slices", "grid", "wavelength_nm", "period_nm", "height_nm")
    settings = sorted({tuple(r[k] for k in keys) for r in complete})
    for values in settings:
        group = [r for r in complete if tuple(r[k] for k in keys) == values]
        method, order, nz, q, wave, period, height = values
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, field, label in zip(axes, ("R_percent", "T_percent", "A_percent"),
                                   ("Reflectance R (%)", "Transmittance T (%)", "Absorptance A (%)")):
            for i, row in enumerate(group):
                ax.bar(i, row[field], color="#0072B2" if row["structure"] == "metal" else "#D55E00")
                if row["negative_absorption_layers"]:
                    ax.scatter(i, row[field], marker="x", s=65, color="#b2182b", zorder=5)
            ax.set_xticks(range(len(group)), ["Au only" if r["structure"] == "metal" else "PMMA + Au" for r in group])
            ax.set_ylabel(label)
            ax.grid(axis="y", alpha=.25)
        fig.suptitle(f"Au | P={period:g} nm, h={height:g} nm | {method} | {wave:g} nm | M={order}, Nz={nz}, Q={q}")
        fig.text(.06, .02, "Red crosses: negative integrated layer absorption. A single setting does not establish convergence.", fontsize=9)
        fig.tight_layout(rect=(0, .06, 1, .93))
        for ext in ("png", "svg"):
            fig.savefig(output/f"geometry_RTA_{method}_M{order}_Nz{nz}_Q{q}_{wave:g}nm.{ext}", dpi=160)
        plt.close(fig)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.worker:
        return diagnostic.worker(args.worker)
    try:
        jobs = build_jobs(args)
        output = args.output_root.resolve()
        for job in jobs:
            path = output/job["directory"]/"report.json"
            if path.exists() and json.loads(path.read_text(encoding="utf-8")).get("signature") != nv.identity_signature(job["identity"]):
                raise ValueError(f"Saved settings/source changed at {path}; use another --output-root.")
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    print(f"Au geometry diagnostics: P={args.period_nm:g} nm, h={args.height_nm:g} nm; "
          f"M={args.orders}; Nz={args.slices}; Q={args.grids}; wavelengths={args.wavelengths}; "
          f"{len(jobs)} conditions.\nOutput: {output}", flush=True)
    for structure in dict.fromkeys(j["identity"]["structure"] for j in jobs):
        g = next(j["identity"]["backend_identity"]["geometry"] for j in jobs if j["identity"]["structure"] == structure)
        print(f"  {structure}: {json.dumps(g, sort_keys=True)}", flush=True)
    if args.dry_run:
        print("Dry run: geometry/materials checked; no calculation or file changes.")
        return 0
    output.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        nv.write_json(output/"plan.json", dict(jobs=jobs, note="Explicit physical geometry; no original-geometry results are reused."))
        for index, job in enumerate(jobs):
            directory = output/job["directory"]
            directory.mkdir(parents=True, exist_ok=True)
            path = directory/"report.json"
            saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
            if saved and (saved["status"] == "diagnostics_complete" or not args.retry_failures):
                print(f"Reuse {job['directory']}: {saved['status']}", flush=True)
                continue
            nv.write_json(directory/"plan.json", dict(identity=job["identity"]))
            print(f"\n=== {index+1}/{len(jobs)}: {job['directory']} ===", flush=True)
            process = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()),
                                      "--worker", str(directory/"plan.json")], check=False)
            if not path.exists() or json.loads(path.read_text(encoding="utf-8")).get("signature") != nv.identity_signature(job["identity"]):
                nv.write_json(path, dict(status="calculation_failed", identity=job["identity"],
                    signature=nv.identity_signature(job["identity"]),
                    error=f"Worker exited {process.returncode} without a matching report."))
            export(jobs, output, args)
    rows, summary = export(jobs, output, args)
    if not args.no_plots:
        for row, job in zip(rows, jobs):
            if row["status"] == "diagnostics_complete":
                display = dict(job["identity"], material=f"Au, P={args.period_nm:g} nm, h={args.height_nm:g} nm")
                diagnostic.plot_case(output/job["directory"], display)
        plot_comparison(rows, output)
    print(f"Counts: {summary['counts']}; physical checks: {summary['physical_checks_passed']}/{len(rows)}; "
          f"consistency checks: {summary['consistency_checks_passed']}/{len(rows)}; "
          f"strict negative-loss checks: {summary['strict_negative_loss_checks_passed']}/{len(rows)}", flush=True)
    print(f"Results: {output/'geometry_summary.csv'}", flush=True)
    return 0 if all(r["status"] == "diagnostics_complete" for r in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
