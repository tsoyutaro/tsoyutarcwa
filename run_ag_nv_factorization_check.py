"""Compare existing symmetric NV with ordinary Fourier/Galerkin factorization.

This is a diagnostic control, not a convergence certificate or a new NV fix.
The Galerkin control uses epsilon_xy=diag([epsilon],[epsilon]) and the same
epsilon_zz, geometry, D6 source row, complex128 and Redheffer reconstruction.
The selected tensor assembly is overridden only inside an isolated worker;
no existing solver or result file is modified.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

import run_ag_nv_m14_diagnostics as diagnostic

ROOT = Path(__file__).resolve().parent
nv = diagnostic.nv
LABELS = {"symmetric-nv": "Symmetric NV", "galerkin": "Ordinary Fourier (Galerkin)"}
RESULT_FILES = ("far_field.json", "layer_power.csv", "excited_modes.csv",
                "operator_diagnostics.csv", "mode_excitation_map.npz")


@contextmanager
def factorization_scope(name):
    if name not in LABELS:
        raise ValueError("Unknown factorization.")
    if name == "symmetric-nv":
        yield
        return
    import torch
    from rcwa_ext.nvm import CustomRCWA_NVM
    original = CustomRCWA_NVM.__dict__["_symmetric_nvm_tensor"]
    def scalar_convolution(eps_zz, delta, projection):
        # Conventional Fourier/Galerkin D=epsilon E in the transverse plane.
        # Do not clip loss eigenvalues or change any material permittivity.
        return torch.block_diag(eps_zz, eps_zz)
    CustomRCWA_NVM._symmetric_nvm_tensor = staticmethod(scalar_convolution)
    try:
        yield
    finally:
        CustomRCWA_NVM._symmetric_nvm_tensor = original


def build_parser():
    parser = diagnostic.build_parser()
    parser.description = __doc__
    parser.set_defaults(grids="512", output_root=ROOT/"studies/ag_nv_factorization_check/results/M14_Nz210_Q512_550nm")
    parser.add_argument("--factorizations", default="symmetric-nv,galerkin")
    parser.add_argument("--reference-root", type=Path,
        default=ROOT/"studies/ag_nv_m14_diagnostics/results/M14_Nz210_Q384-1024_550nm")
    parser.add_argument("--no-reuse-reference", action="store_true")
    return parser


def build_jobs(args):
    names = list(dict.fromkeys(value.strip() for value in args.factorizations.split(",")))
    if not names or any(name not in LABELS for name in names):
        raise ValueError("--factorizations must contain symmetric-nv and/or galerkin.")
    identities = diagnostic.build_identities(args)
    jobs = []
    for name in names:
        for source_identity in identities:
            identity = dict(source_identity, factorization=name)
            identity["sources"] = dict(source_identity["sources"])
            identity["sources"]["run_ag_nv_factorization_check.py"] = nv.digest(__file__)
            identity["method"] = LABELS[name]+"; D6 E1 x, complex128, same epsilon_zz and Redheffer"
            jobs.append(dict(identity=identity, directory=f"{name}/Q{identity['grid']}"))
    return jobs


def try_reference(job, reference_root, destination):
    identity = job["identity"]
    if identity["factorization"] != "symmetric-nv":
        return False
    source = reference_root/f"Q{identity['grid']}"
    report_path = source/"report.json"
    if not report_path.exists():
        return False
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    previous = saved.get("identity", {})
    fields = ("version", "material", "structure", "order", "wavelength_nm", "slices", "grid",
              "geometry", "material_sha256", "quadrature")
    if (saved.get("status") != "diagnostics_complete" or
            any(previous.get(key) != identity.get(key) for key in fields) or
            any(previous.get("sources", {}).get(key) != value for key, value in identity["sources"].items()
                if key != "run_ag_nv_factorization_check.py") or
            any(not (source/name).is_file() for name in RESULT_FILES)):
        return False
    destination.mkdir(parents=True, exist_ok=True)
    for name in RESULT_FILES:
        shutil.copy2(source/name, destination/name)
    # Preserve provenance: this is a copy of the old measured result, not a
    # claim that the new comparison wrapper computed the baseline again.
    report = dict(saved, signature=nv.identity_signature(identity), identity=identity,
                  reused_from=str(report_path.resolve()), original_identity=previous,
                  original_signature=saved.get("signature"))
    nv.write_json(destination/"plan.json", dict(identity=identity))
    nv.write_json(destination/"report.json", report)
    return True


def collect(output, jobs):
    rows = []
    for job in jobs:
        identity = job["identity"]
        path = output/job["directory"]/"report.json"
        saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else dict(status="pending")
        result = saved.get("result", saved.get("far_field", {}))
        rows.append(dict(factorization=identity["factorization"], grid=identity["grid"],
            order=identity["order"], slices=identity["slices"], wavelength_nm=identity["wavelength_nm"],
            status=saved["status"], reused_reference=bool(saved.get("reused_from")),
            R_percent=None if "reflectance" not in result else 100*result["reflectance"],
            A_percent=None if "absorptance" not in result else 100*result["absorptance"],
            motheye_A_percent=None if "motheye_absorptance" not in result else 100*result["motheye_absorptance"],
            substrate_A_percent=None if "substrate_absorptance" not in result else 100*result["substrate_absorptance"],
            negative_absorption_layers=result.get("negative_absorption_layers"),
            physical_result=result.get("physical_result"), diagnostic_checks_passed=result.get("diagnostic_checks_passed"),
            max_boundary_residual=result.get("max_boundary_residual"),
            max_volume_flux_error=result.get("max_volume_flux_error"), error=saved.get("error", "")))
    diagnostic.write_csv(output/"comparison.csv", rows)
    nv.write_json(output/"summary.json", dict(cases=rows,
        interpretation="An ordinary Fourier control isolates the NV transverse correction; physical bounds alone do not establish convergence or accuracy."))
    return rows


def plot(output, jobs, rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        print(f"Plots unavailable: {error}; CSV/JSON results are saved.", flush=True)
        return
    figures = output/"figures"
    figures.mkdir(exist_ok=True)
    for q in sorted({row["grid"] for row in rows}):
        selected = [row for row in rows if row["grid"] == q and row["R_percent"] is not None]
        if not selected:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.7))
        for ax, field, label in zip(axes, ("R_percent", "A_percent", "negative_absorption_layers"),
                                    ("Reflectance R (%)", "Total absorptance A (%)", "Layers with negative absorption")):
            for i, row in enumerate(selected):
                if row[field] is not None:
                    ax.bar(i, row[field], color="#0072b2" if row["factorization"] == "symmetric-nv" else "#009e73")
                    ax.annotate(f"{row[field]:.5g}", (i, row[field]), ha="center", va="bottom")
            ax.set_xticks(range(len(selected)), [LABELS[row["factorization"]].replace(" (", "\n(") for row in selected])
            ax.set_title(label)
            ax.axhline(0, color="black", lw=.6)
            ax.grid(axis="y", alpha=.2)
        first = selected[0]
        fig.suptitle(f"Ag metal: factorization control | {first['wavelength_nm']:g} nm | M={first['order']}, Nz={first['slices']}, Q={q}")
        fig.tight_layout(rect=(0, .03, 1, .93))
        for extension in ("png", "svg"):
            fig.savefig(figures/f"factorization_Q{q}.{extension}", dpi=160)
        plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
        any_layers = False
        for job in jobs:
            if job["identity"]["grid"] != q:
                continue
            path = output/job["directory"]/"layer_power.csv"
            if not path.exists():
                continue
            with path.open(encoding="utf-8-sig", newline="") as handle:
                powers = list(csv.DictReader(handle))
            name = job["identity"]["factorization"]
            color = "#0072b2" if name == "symmetric-nv" else "#009e73"
            for ax, field, scale, title in zip(axes,
                ("flux_bottom", "layer_absorptance", "electric_loss_min_eigenvalue"),
                (100., 100., 1.), ("Net downward flux (% of incident)", "Layer absorption (pp)", "Minimum electric-loss eigenvalue")):
                ax.plot([float(row["depth_mid_nm"]) for row in powers],
                        [scale*float(row[field]) for row in powers], label=LABELS[name], color=color)
                ax.set(xlabel="Depth (nm)", title=title)
                ax.set_yscale("symlog", linthresh=1e-3)
                ax.axhline(0, color="black", lw=.6)
                ax.grid(alpha=.2)
            any_layers = True
        if any_layers:
            axes[0].legend(fontsize=8)
            fig.suptitle(f"Ag metal | {first['wavelength_nm']:g} nm, M={first['order']}, Nz={first['slices']}, Q={q} | signed logarithmic vertical scales")
            fig.tight_layout(rect=(0, .03, 1, .93))
            for extension in ("png", "svg"):
                fig.savefig(figures/f"layer_comparison_Q{q}.{extension}", dpi=160)
        plt.close(fig)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.worker:
        identity = json.loads(args.worker.read_text(encoding="utf-8"))["identity"]
        with factorization_scope(identity["factorization"]):
            return diagnostic.worker(args.worker)
    try:
        jobs = build_jobs(args)
        output = args.output_root.resolve()
        for job in jobs:
            path = output/job["directory"]/"report.json"
            if path.exists() and not args.plot_only:
                if json.loads(path.read_text(encoding="utf-8")).get("signature") != nv.identity_signature(job["identity"]):
                    raise ValueError("Saved comparison settings/source changed; use a new --output-root.")
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Ag factorization diagnostic: M={args.order}, Nz={args.slices}, {args.wavelength_nm:g} nm; "
          f"{args.factorizations}; Q={args.grids}\nOutput: {output}", flush=True)
    if args.dry_run:
        print("Dry run: no computation or writes. Existing compatible NV reference can be reused.")
        return 0
    output.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        nv.write_json(output/"plan.json", dict(jobs=jobs, reference_root=str(args.reference_root.resolve())))
        for job in jobs:
            destination = output/job["directory"]
            report_path = destination/"report.json"
            if report_path.exists():
                saved = json.loads(report_path.read_text(encoding="utf-8"))
                if saved["status"] == "diagnostics_complete" or not args.retry_failures:
                    print(f"Reuse {job['directory']}: {saved['status']}", flush=True)
                    continue
            if (not args.no_reuse_reference and
                    try_reference(job, args.reference_root, destination)):
                print(f"Reuse existing NV reference: {job['directory']}", flush=True)
                continue
            destination.mkdir(parents=True, exist_ok=True)
            nv.write_json(destination/"plan.json", dict(identity=job["identity"]))
            print(f"\n=== {job['directory']} ===", flush=True)
            completed = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()),
                                       "--worker", str(destination/"plan.json")], check=False)
            if not report_path.exists():
                nv.write_json(report_path, dict(status="calculation_failed", identity=job["identity"],
                    signature=nv.identity_signature(job["identity"]), error=f"Worker exited {completed.returncode} without a report."))
            collect(output, jobs)
    rows = collect(output, jobs)
    if not args.no_plots:
        plot(output, jobs, rows)
    print(f"Saved: {output/'comparison.csv'}", flush=True)
    return 0 if all(row["status"] == "diagnostics_complete" for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
