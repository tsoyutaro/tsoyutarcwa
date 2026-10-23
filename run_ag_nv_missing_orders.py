"""Complete Ag NV orders and optionally compare ordinary Fourier/Galerkin.

The existing Ag worker/solver is unchanged. Compatible successful cases and
known nonphysical outcomes are imported read-only from the earlier workflow.
OOM prunes higher M only in the same factorization/structure/wavelength group.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import io
from itertools import product
import json
import math
from pathlib import Path
import subprocess
import sys
import traceback
import uuid

import run_cu_ag_al_motheye_order_nv as backend

ROOT = Path(__file__).resolve().parent
nv = backend.nv
STATUSES = {"ok", "nonphysical", "oom", "calculation_failed", "skipped_oom"}
LABELS = {"symmetric-nv": "NV", "galerkin": "Galerkin"}


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--structures", choices=("both", "metal", "pmma"), default="both")
    p.add_argument("--factorizations", default="symmetric-nv",
                   help="symmetric-nv, galerkin, or symmetric-nv,galerkin")
    p.add_argument("--wavelengths", default="400,550,700")
    p.add_argument("--slices", type=int, default=210)
    p.add_argument("--grid", type=int, default=512)
    p.add_argument("--start-order", type=int, default=4)
    p.add_argument("--step", type=int, default=2)
    p.add_argument("--max-order", type=int, default=36)
    p.add_argument("--tolerance", type=float, default=.005)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--source-root", type=Path,
        default=ROOT/"studies/au_ag_nv_order_slices/results/orders_only_Nz210_M4-36_Q512_3wl")
    p.add_argument("--output-root", type=Path,
        default=ROOT/"studies/ag_nv_order_completion/results/Nz210_Q512_M4-36_3wl")
    p.add_argument("--retry-failures", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--_galerkin-plan", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--_galerkin-result", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--_galerkin-order", type=int, help=argparse.SUPPRESS)
    p.add_argument("--_galerkin-wave", type=float, help=argparse.SUPPRESS)
    return p


def fixed_signature(identity):
    """Ignore scan choices and absolute data paths, retaining content hashes."""
    value = deepcopy(identity)
    for key in ("start_order", "step", "tolerance", "wavelengths_nm", "metal_csv"):
        value.pop(key, None)
    if value.get("pmma"):
        value["pmma"].pop("path", None)
    return nv.identity_signature(value)


def case_tag(job, order):
    # Preserve the earlier NV keys and directories for backward-compatible resume.
    prefix = "galerkin|" if job.get("factorization") == "galerkin" else ""
    return f"{prefix}{job['structure']}|wl={job['wave']:g}|M={order}"


def report_path(output, job, order):
    return output/job["directory"]/f"M{order}"/"report.json"


def validate_powers(powers):
    values = [float(powers[k]) for k in ("reflectance", "transmittance", "absorptance")]
    if not all(math.isfinite(v) for v in values) or abs(sum(values)-1) > 1e-8:
        raise ValueError("Nonfinite or inconsistent saved R/T/A.")


def reference_cases(job, source_root):
    if job.get("factorization") == "galerkin":
        return {}, "NV reference is not used as Galerkin data."
    path = source_root/"order_sweep"/job["directory"]/"checkpoint.json"
    if not path.exists():
        return {}, "No reference checkpoint."
    saved = json.loads(path.read_text(encoding="utf-8"))
    original = saved["plan"]["identity"]
    if saved.get("signature") != nv.identity_signature(original):
        raise ValueError(f"Reference signature does not match its saved identity: {path}")
    if fixed_signature(original) != fixed_signature(job["identity"]) or job["wave"] not in original["wavelengths_nm"]:
        return {}, "Reference source/material/fixed settings differ; no reference cases reused."
    audit = dict(checkpoint=str(path.resolve()), sha256=nv.digest(path), original_signature=saved["signature"])
    imported = {}
    for key, row in saved["cases"].items():
        backend.validate_case(row, original)
        if key != nv.case_key(int(row["order"]), float(row["wavelength_nm"])):
            raise ValueError(f"Invalid reference case key: {path}")
        if row["wavelength_nm"] == job["wave"] and int(row["order"]) in job["orders"]:
            imported[int(row["order"])] = dict(status="ok", case=row, environment=saved.get("environment"), reused_from=audit)
    failure = saved.get("failure") or {}
    if (failure.get("order") in job["orders"] and failure.get("wavelength_nm") == job["wave"]
            and "Nonphysical power result" in failure.get("error", "") and failure.get("power_diagnostics")):
        validate_powers(failure["power_diagnostics"])
        if failure["order"] not in imported:
            imported[failure["order"]] = dict(status="nonphysical", power_diagnostics=failure["power_diagnostics"],
                error=failure["error"], original_failure=failure, reused_from=audit)
    return imported, "Compatible reference."


def decorate(job, order, outcome):
    return dict(outcome, identity=job["identity"], signature=nv.identity_signature(job["identity"]),
                order=order, wavelength_nm=job["wave"], factorization=job.get("factorization", "symmetric-nv"))


def control_sources():
    return {name: nv.digest(ROOT/name) for name in
            (Path(__file__).name, "run_ag_nv_factorization_check.py")}


def galerkin_worker(plan_path, result_path, order, wave):
    """Use the existing Galerkin control inside this isolated process only."""
    try:
        identity = json.loads(plan_path.read_text(encoding="utf-8"))["identity"]
        if identity.get("factorization") != "galerkin" or identity.get("control_source_sha256") != control_sources():
            raise ValueError("Galerkin control/driver changed after preflight.")
        import run_ag_nv_factorization_check as control
        with control.factorization_scope("galerkin"):
            return backend.worker(plan_path, result_path, order, wave)
    except Exception as error:
        nv.write_json(result_path, dict(status="calculation_failed", order=order, wavelength_nm=wave,
            error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc()))
        return 1


def launch_case(job, plan_path, result_path, order, wave):
    if job["factorization"] == "symmetric-nv":
        return backend.launch_worker(plan_path, result_path, order, wave)
    completed = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()),
        "--_galerkin-plan", str(plan_path), "--_galerkin-result", str(result_path),
        "--_galerkin-order", str(order), "--_galerkin-wave", str(wave)], cwd=ROOT, check=False)
    if not result_path.exists():
        return dict(status="calculation_failed", error=f"Worker exited {completed.returncode} without a result.")
    try:
        response = json.loads(result_path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as error:
        return dict(status="calculation_failed", error=f"Invalid Galerkin worker result: {error}")
    expected = 0 if response.get("status") == "ok" else 2 if response.get("status") == "oom" else 1
    if completed.returncode != expected:
        return dict(status="calculation_failed", error="Galerkin worker status and exit code disagree.")
    return response


def prepare(args):
    waves = sorted(set(float(x.strip()) for x in args.wavelengths.split(",")))
    methods = list(dict.fromkeys(x.strip() for x in args.factorizations.split(",")))
    if not methods or any(m not in LABELS for m in methods):
        raise ValueError("Use symmetric-nv and/or galerkin for --factorizations.")
    if (not waves or any(not math.isfinite(w) or w <= 0 for w in waves)
            or args.slices < 2 or args.start_order < 1 or args.step < 1 or args.cpu_threads < 1
            or args.max_order < args.start_order or (args.max_order-args.start_order) % args.step
            or args.grid < max(32, 4*args.max_order+4)
            or not math.isfinite(args.tolerance) or args.tolerance <= 0):
        raise ValueError("Check waves, Nz>=2, aligned positive orders, Q>=max(32,4*Mmax+4), threads and tolerance.")
    if args.device not in ("cpu", "cuda") and not (args.device.startswith("cuda:") and args.device[5:].isdigit()):
        raise ValueError("Use cpu, cuda or cuda:<index>.")
    source, output = args.source_root.resolve(), args.output_root.resolve()
    if not source.is_dir():
        raise ValueError(f"Reference directory not found: {source}")
    if output == source or source in output.parents:
        raise ValueError("Use an output directory outside the read-only source-root.")
    orders = list(range(args.start_order, args.max_order+1, args.step))
    structures = ("metal", "pmma") if args.structures == "both" else (args.structures,)
    jobs, current, references = [], {}, {}
    for method, structure, wave in product(methods, structures, waves):
        options = backend.build_parser().parse_args([])
        for key in ("device", "slices", "grid", "start_order", "step", "max_order", "cpu_threads", "tolerance"):
            setattr(options, key, getattr(args, key))
        identity = deepcopy(backend.build_identity(options, "Ag", structure, [wave]))
        directory = f"{structure}/ag/wl{wave:g}/Nz{args.slices}"
        if method == "galerkin":
            identity.update(factorization=method, control_source_sha256=control_sources(),
                solver="Ordinary Fourier/Galerkin transverse convolution; D6 E1 x, complex128, Redheffer, half S")
            directory = f"galerkin/{directory}"
        job = dict(structure=structure, wave=wave, orders=orders, identity=identity,
            factorization=method, directory=directory)
        imported, note = reference_cases(job, source)
        job["reference_note"] = note
        jobs.append(job)
        for order in orders:
            tag = case_tag(job, order)
            if order in imported:
                references[tag] = decorate(job, order, imported[order])
            path = report_path(output, job, order)
            if path.exists():
                report = json.loads(path.read_text(encoding="utf-8"))
                if (report.get("signature") != nv.identity_signature(identity)
                        or nv.identity_signature(report.get("identity")) != nv.identity_signature(identity)
                        or report.get("order") != order or report.get("wavelength_nm") != wave or report.get("status") not in STATUSES):
                    raise ValueError(f"Saved settings/source changed or report is invalid: {path}; use a new output-root.")
                if report["status"] == "ok":
                    backend.validate_case(report["case"], identity)
                    if (report["case"]["order"], report["case"]["wavelength_nm"]) != (order, wave):
                        raise ValueError(f"Saved case has a different order/wavelength: {path}")
                elif report["status"] == "nonphysical":
                    validate_powers(report["power_diagnostics"])
                current[tag] = report
    return jobs, current, references, output


def export(jobs, reports, output):
    rows = []
    for job in jobs:
        for order in job["orders"]:
            report = reports.get(case_tag(job, order), {})
            powers = report.get("case", report.get("power_diagnostics", {}))
            row = dict(material="Ag", structure=job["structure"], factorization=job["factorization"],
                wavelength_nm=job["wave"], order=order,
                slices=job["identity"]["slices"], grid=job["identity"]["grid"],
                status=report.get("status", "pending"), reused_reference=bool(report.get("reused_from")),
                error=report.get("error", ""), directory=str(report_path(output, job, order).parent))
            for label, key in (("R", "reflectance"), ("T", "transmittance"), ("A", "absorptance")):
                row[f"{label}_percent"] = 100*float(powers[key]) if key in powers else None
            rows.append(row)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    nv.atomic_text(output/"all_cases.csv", stream.getvalue())
    counts = {name: sum(r["status"] == name for r in rows) for name in sorted({r["status"] for r in rows})}
    summary = dict(requested_cases=len(rows), counts=counts,
        counts_by_factorization={method: {status: sum(r["factorization"] == method and r["status"] == status for r in rows)
            for status in counts} for method in dict.fromkeys(r["factorization"] for r in rows)},
        reused_ok=sum(r["status"] == "ok" and r["reused_reference"] for r in rows),
        reused_nonphysical=sum(r["status"] == "nonphysical" and r["reused_reference"] for r in rows),
        interpretation="Nonphysical raw powers are retained and higher orders continue. OOM skips higher M only for the same factorization/structure/wavelength. Galerkin uses ordinary transverse convolution without the NV correction. Completion does not establish convergence or internal-field accuracy.")
    nv.write_json(output/"summary.json", summary)
    return rows, summary


def plot_results(rows, output):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    colors = ("#0072B2", "#D55E00", "#009E73")
    methods = list(dict.fromkeys(r["factorization"] for r in rows))
    for field, label, filename in (("R", "Reflectance R (%)", "reflectance"),
                                   ("T", "Transmittance T (%)", "transmittance"),
                                   ("A", "Absorptance A (%)", "absorptance")):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        for ax, structure, title in zip(axes, ("metal", "pmma"), ("Ag only", "PMMA + Ag (30 nm coating)")):
            group = [r for r in rows if r["structure"] == structure]
            for wave, color in zip(sorted({r["wavelength_nm"] for r in group}), colors):
                for method in methods:
                    points = sorted((r for r in group if r["wavelength_nm"] == wave and r["factorization"] == method), key=lambda r:r["order"])
                    style = "o-" if method == "symmetric-nv" else "s--"
                    ax.plot([r["order"] for r in points],
                        [r[f"{field}_percent"] if r[f"{field}_percent"] is not None else math.nan for r in points],
                        style, color=color, ms=4, label=f"{wave:g} nm | {LABELS[method]}")
            ax.set(title=title, xlabel="Fourier truncation order M")
            ax.grid(alpha=.25)
            if group:
                ax.legend(fontsize=8)
        axes[0].set_ylabel(label)
        first = rows[0]
        method_title = " / ".join(LABELS[m] for m in methods)
        fig.suptitle(f"Ag {method_title} | P=200 nm, h=500 nm | Nz={first['slices']} (+ cap for PMMA), Q={first['grid']}")
        fig.text(.06, .02, "Nonphysical values are retained; gaps are uncomputed/failed cases.", fontsize=9)
        fig.tight_layout(rect=(0, .06, 1, .92))
        for ext in ("png", "svg"):
            fig.savefig(output/f"Ag_{filename}_vs_order.{ext}", dpi=170)
        plt.close(fig)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args._galerkin_plan is not None:
        if any(v is None for v in (args._galerkin_result, args._galerkin_order, args._galerkin_wave)):
            parser.error("Incomplete private Galerkin worker arguments.")
        return galerkin_worker(args._galerkin_plan, args._galerkin_result, args._galerkin_order, args._galerkin_wave)
    try:
        jobs, reports, references, output = prepare(args)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    print(f"Ag {args.factorizations}: Nz={args.slices}, Q={args.grid}, M={args.start_order}:{args.step}:{args.max_order}; "
          f"{sum(len(j['orders']) for j in jobs)} requested cases.\nSource (read-only): {args.source_root.resolve()}\nOutput: {output}", flush=True)
    pending = 0
    for job in jobs:
        todo, reused = [], []
        for order in job["orders"]:
            saved = reports.get(case_tag(job, order), references.get(case_tag(job, order)))
            if not saved or (args.retry_failures and saved["status"] != "ok"):
                todo.append(order)
            else:
                reused.append((order, saved["status"]))
        pending += len(todo)
        print(f"  {LABELS[job['factorization']]} / {job['structure']} / {job['wave']:g} nm: calculate M={todo}; reuse={reused}\n    {job['reference_note']}", flush=True)
    print(f"New/retry cases requested: {pending}. Nonphysical outcomes continue; OOM prunes higher M in that method/structure/wavelength group only.", flush=True)
    if args.dry_run:
        print("Dry run: no calculation or file writes.")
        return 0
    output.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        nv.write_json(output/"plan.json", dict(jobs=jobs, source_root=str(args.source_root.resolve()), driver_sha256=nv.digest(__file__)))
        for job in jobs:
            oom_order = None
            plan_path = output/job["directory"]/"plan.json"
            nv.write_json(plan_path, dict(identity=job["identity"]))
            for order in job["orders"]:
                tag = case_tag(job, order)
                saved = reports.get(tag, references.get(tag))
                if saved and (saved["status"] == "ok" or not args.retry_failures):
                    reports[tag] = saved
                    nv.write_json(report_path(output, job, order), saved)
                    if saved["status"] == "oom":
                        oom_order = order
                    continue
                if oom_order is not None:
                    outcome = dict(status="skipped_oom", error=f"Higher M than OOM at M={oom_order} in the same factorization/structure/wavelength.")
                else:
                    print(f"solve: Ag/{job['structure']} {LABELS[job['factorization']]}, lambda={job['wave']:g} nm, M={order}, Nz={args.slices}, Q={args.grid}", flush=True)
                    # A fresh path prevents a crashed retry from reading an older result.
                    worker_path = output/job["directory"]/f"M{order}"/f"worker_{uuid.uuid4().hex}.json"
                    response = launch_case(job, plan_path, worker_path, order, job["wave"])
                    outcome = dict(response)
                    if response.get("status") not in ("ok", "oom", "calculation_failed"):
                        raise ValueError("Worker returned an unknown status.")
                    if response["status"] == "ok":
                        if response.get("signature") != nv.identity_signature(job["identity"]):
                            raise ValueError("Worker signature mismatch.")
                        backend.validate_case(response["case"], job["identity"])
                        if (response["case"]["order"], response["case"]["wavelength_nm"]) != (order, job["wave"]):
                            raise ValueError("Worker returned a different condition.")
                    elif "Nonphysical power result" in response.get("error", "") and response.get("power_diagnostics"):
                        try:
                            validate_powers(response["power_diagnostics"])
                        except (ValueError, KeyError, TypeError):
                            # Keep failed/nonfinite results separate from finite raw R/T/A.
                            outcome.pop("power_diagnostics", None)
                        else:
                            outcome["status"] = "nonphysical"
                            print("  Nonphysical powers retained; continue to the next order.", flush=True)
                    elif response["status"] == "oom":
                        oom_order = order
                reports[tag] = decorate(job, order, outcome)
                nv.write_json(report_path(output, job, order), reports[tag])
                export(jobs, reports, output)
            export(jobs, reports, output)
    rows, summary = export(jobs, reports, output)
    if not args.no_plots:
        plot_results(rows, output)
    print(f"Counts: {summary['counts']}\nSaved: {output/'all_cases.csv'}", flush=True)
    return 0 if all(r["status"] == "ok" for r in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
