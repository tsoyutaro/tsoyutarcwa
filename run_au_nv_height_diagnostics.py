"""Au height sweep using the existing memory-safe NV layer diagnostics.

Only the profile height changes. Radius endpoints, period, material tables,
coating thicknesses and numerical settings stay fixed. A height sweep changes
the physical structure and is not a numerical convergence test.
"""
from __future__ import annotations

from copy import deepcopy
import csv
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import run_metals_nv_diagnostics as diagnostic

ROOT = Path(__file__).resolve().parent
nv = diagnostic.nv
REFERENCE_FILES = diagnostic.control.RESULT_FILES


def build_parser():
    p = diagnostic.build_parser()
    p.description = __doc__
    p.set_defaults(orders="Au:36", order_offsets="0", structures="both", wavelengths="700",
                   slices="210", grids="512", factorizations="symmetric-nv",
                   output_root=ROOT/"studies/au_nv_height_diagnostics/results/M36_Nz210_Q512_700nm_H500-200")
    p.add_argument("--heights-nm", default="500,400,300,200")
    p.add_argument("--reference-root", type=Path,
        default=ROOT/"studies/metals_nv_diagnostics/results/Au_M36_Nz210_Q512_3wl2",
        help="Compatible completed 500 nm diagnostics can be reused, including nonphysical results.")
    p.add_argument("--no-reuse-reference", action="store_true")
    return p


def build_jobs(args):
    if set(diagnostic.orders(args.orders)) != {"Au"}:
        raise ValueError("This height-sweep runner accepts Au only.")
    heights = list(dict.fromkeys(float(x.strip()) for x in args.heights_nm.split(",")))
    if not heights or any(not math.isfinite(h) or h <= 0 for h in heights):
        raise ValueError("Use finite positive --heights-nm values.")
    templates = diagnostic.build_jobs(args)
    wrapper_hash = nv.digest(__file__)
    jobs = []
    for height in heights:
        for template in templates:
            job = deepcopy(template)
            identity = job["identity"]
            geometry = identity["backend_identity"]["geometry"]
            geometry["height_nm"] = height
            if identity["structure"] == "pmma":
                from studies.pmma_metal_motheye.model import Geometry
                geom = Geometry(**geometry)
                geom.validate()
                geom.profile(identity["slices"])
            identity["height_nm"] = height
            identity["sources"][Path(__file__).name] = wrapper_hash
            job["reference_directory"] = template["directory"]
            job["directory"] = f"H{height:g}/"+template["directory"]
            jobs.append(job)
    return jobs


def read_reference(job, root):
    """Reuse only a full, exactly compatible original diagnostic case."""
    directory = Path(root)/job["reference_directory"]
    report_path = directory/"report.json"
    if not report_path.exists():
        return None
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    expected = deepcopy(job["identity"])
    expected.pop("height_nm")
    expected["sources"].pop(Path(__file__).name)
    expected_signature = nv.identity_signature(expected)
    # JSON stores tuple-valued PMMA provenance points as lists. Compare their
    # canonical identities rather than Python container types.
    if (saved.get("status") != "diagnostics_complete" or not isinstance(saved.get("identity"), dict)
            or nv.identity_signature(saved["identity"]) != expected_signature
            or saved.get("signature") != expected_signature
            or any(not (directory/name).is_file() for name in REFERENCE_FILES)):
        return None
    return directory, saved


def reuse_reference(job, root, destination):
    found = read_reference(job, root)
    if found is None:
        return False
    source, saved = found
    destination.mkdir(parents=True, exist_ok=True)
    for name in REFERENCE_FILES:
        shutil.copy2(source/name, destination/name)
    report = dict(saved, identity=job["identity"], signature=nv.identity_signature(job["identity"]),
        reused_reference=dict(path=str(source.resolve()), original_signature=saved["signature"],
                              report_sha256=nv.digest(source/"report.json")))
    nv.write_json(destination/"report.json", report)
    return True


def collect(jobs, output, args):
    rows = diagnostic.collect(jobs, output, args.negative_loss_tolerance)
    for row, job in zip(rows, jobs):
        identity = job["identity"]
        geometry = identity["backend_identity"]["geometry"]
        row["height_nm"] = identity["height_nm"]
        row["total_pattern_height_nm"] = identity["height_nm"] + (
            geometry["metal_top_thickness_nm"] if identity["structure"] == "pmma" else 0.)
        minimum = row["minimum_layer_absorptance"]
        row["minimum_layer_absorption_pp"] = None if minimum is None else 100*minimum
        path = output/job["directory"]/"report.json"
        saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        row["reused_reference"] = bool(saved.get("reused_reference"))
    return rows


def export(jobs, output, args):
    rows = collect(jobs, output, args)
    diagnostic.diagnostic.write_csv(output/"height_summary.csv", rows)
    counts = {status: sum(r["status"] == status for r in rows) for status in sorted({r["status"] for r in rows})}
    summary = dict(requested_cases=len(rows), counts=counts,
        physical_checks_passed=sum(bool(r["physical_result"]) for r in rows),
        consistency_checks_passed=sum(bool(r["diagnostic_checks_passed"]) for r in rows),
        strict_negative_loss_checks_passed=sum(bool(r["excited_negative_loss_passed"]) for r in rows),
        sample_checks_passed=sum(r["sample_checks_passed"] for r in rows),
        reused_reference_cases=sum(r["reused_reference"] for r in rows),
        negative_loss_tolerance=args.negative_loss_tolerance, cases=rows,
        interpretation="Height changes the physical geometry. This is not a convergence scan. Passivity, flux/volume consistency, and the strict negative-eigensector warning are separate checks. Negative contributions are not R/T/A error estimates.")
    nv.write_json(output/"height_summary.json", summary)
    return rows, summary


def plot_height_comparison(rows, output):
    complete = [r for r in rows if r["status"] == "diagnostics_complete"]
    if not complete:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        print(f"Plots unavailable: {error}; CSV/JSON results are intact.", flush=True)
        return
    settings = sorted({(r["factorization"], r["order"], r["slices"], r["grid"], r["wavelength_nm"]) for r in complete})
    colors = {"metal": "#0072B2", "pmma": "#D55E00"}
    for method, order, nz, q, wave in settings:
        group = [r for r in complete if (r["factorization"], r["order"], r["slices"], r["grid"], r["wavelength_nm"]) == (method, order, nz, q, wave)]
        for name, fields, labels in (
            ("RTA", ("R_percent", "T_percent", "A_percent"),
             ("Reflectance R (%)", "Transmittance T (%)", "Absorptance A (%)")),
            ("passivity", ("negative_absorption_layers", "minimum_layer_absorption_pp", "negative_loss_sector_pp"),
             ("Negative-absorption layers", "Minimum layer absorption (pp)", "Negative eigensector contribution (pp)"))):
            fig, axes = plt.subplots(1, 3, figsize=(14, 4.7))
            for structure in ("metal", "pmma"):
                selected = sorted((r for r in group if r["structure"] == structure), key=lambda r:r["height_nm"])
                if not selected:
                    continue
                for ax, field, label in zip(axes, fields, labels):
                    ax.plot([r["height_nm"] for r in selected], [r[field] for r in selected], "o-",
                        color=colors[structure], label="Metal only" if structure == "metal" else "Metal-coated PMMA")
                    bad = [r for r in selected if r["negative_absorption_layers"]]
                    if name == "RTA" and bad:
                        ax.scatter([r["height_nm"] for r in bad], [r[field] for r in bad],
                                   color="#b2182b", marker="x", s=75, linewidths=2, zorder=5)
                    ax.set(xlabel="Profile height h (nm)", ylabel=label)
                    ax.grid(alpha=.25)
            axes[0].legend(fontsize=9)
            fig.suptitle(f"Au height dependence | {method} | {wave:g} nm | M={order}, Nz={nz}, Q={q}")
            fig.text(.06, .02, "Height sweep changes the geometry; it is not a convergence test. Red crosses indicate cases with negative integrated layer absorption.", fontsize=9)
            fig.tight_layout(rect=(0, .05, 1, .93))
            stem = f"height_{name}_{method}_M{order}_Nz{nz}_Q{q}_{wave:g}nm"
            for extension in ("png", "svg"):
                fig.savefig(output/f"{stem}.{extension}", dpi=160)
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
    print(f"Au height diagnostics: h={args.heights_nm} nm; M={args.orders}; Nz={args.slices}; "
          f"Q={args.grids}; wavelengths={args.wavelengths}; {len(jobs)} conditions.\nOutput: {output}", flush=True)
    if args.dry_run:
        reusable = 0 if args.no_reuse_reference else sum(read_reference(j, args.reference_root) is not None for j in jobs)
        print(f"Dry run: {reusable} exactly compatible reference cases available; no calculation or file changes.")
        print("Metal thicknesses and radius endpoints stay fixed. OOM/nonphysical cases do not stop other heights.")
        return 0
    output.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        nv.write_json(output/"plan.json", dict(jobs=jobs, note="Only profile height changes; coating thicknesses remain fixed."))
        for index, job in enumerate(jobs):
            directory = output/job["directory"]
            directory.mkdir(parents=True, exist_ok=True)
            path = directory/"report.json"
            saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
            if saved and (saved["status"] == "diagnostics_complete" or not args.retry_failures):
                print(f"Reuse {job['directory']}: {saved['status']}", flush=True)
                continue
            nv.write_json(directory/"plan.json", dict(identity=job["identity"]))
            if not args.no_reuse_reference and reuse_reference(job, args.reference_root, directory):
                print(f"Reuse compatible original result: {job['directory']}", flush=True)
            else:
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
                display = dict(job["identity"], material=f"Au, h={job['identity']['height_nm']:g} nm")
                diagnostic.plot_case(output/job["directory"], display)
        plot_height_comparison(rows, output)
    print(f"Counts: {summary['counts']}; physical checks: {summary['physical_checks_passed']}/{len(rows)}; "
          f"consistency checks: {summary['consistency_checks_passed']}/{len(rows)}; "
          f"strict negative-loss checks: {summary['strict_negative_loss_checks_passed']}/{len(rows)}", flush=True)
    print(f"Results: {output/'height_summary.csv'}", flush=True)
    return 0 if all(row["status"] == "diagnostics_complete" for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
