"""Au-only and PMMA/Au spectra with independent NV and Galerkin workers.

Reuses the established Au material/geometry and memory-saving solver. Every
wavelength is checkpointed; failed/nonphysical/OOM cases do not stop other
conditions. Full-band means require every wavelength to pass validation.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys
import traceback
import uuid

import run_gold_motheye_order_nv as backend
import run_metals_nv_spectra as tables

ROOT = Path(__file__).resolve().parent
nv = backend.nv
LABELS = {"symmetric-nv": "NV", "galerkin": "Galerkin"}
STATUSES = {"ok", "nonphysical", "oom", "calculation_failed"}


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--factorizations", default="symmetric-nv")
    p.add_argument("--structures", choices=("both", "metal", "pmma"), default="both")
    p.add_argument("--wavelengths", default="400:700:5")
    p.add_argument("--order", type=int, default=36)
    p.add_argument("--slices", type=int, default=210)
    p.add_argument("--grid", type=int, default=512)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--gold-model", choices=("csv", "rakic-ld"), default="csv")
    p.add_argument("--gold-csv", type=Path)
    p.add_argument("--pmma-csv", type=Path)
    p.add_argument("--reference-root", type=Path,
        default=ROOT/"studies/au_ag_nv_order_slices/results/orders_only_Nz210_M4-36_Q512_3wl")
    p.add_argument("--output-root", type=Path,
        default=ROOT/"studies/au_spectra/results/M36_Nz210_Q512_400-700_61pt")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--retry-failures", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--_worker-plan", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--_worker-result", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--_worker-wave", type=float, help=argparse.SUPPRESS)
    p.add_argument("--_worker-order", type=int, help=argparse.SUPPRESS)
    return p


def driver_sources(method):
    paths = [Path(__file__), Path(tables.__file__)]
    if method == "galerkin":
        paths.append(ROOT/"run_ag_nv_factorization_check.py")
    return {p.name: nv.digest(p) for p in paths}


def case_key(job):
    return f"{job['method']}|{job['structure']}|wl={job['wave']:g}"


def validate_raw(powers):
    values = [float(powers[k]) for k in ("reflectance", "transmittance", "absorptance")]
    if any(not math.isfinite(v) for v in values) or abs(math.fsum(values)-1) > 1e-8:
        raise ValueError("Nonfinite or inconsistent raw powers.")


def legacy_signature(identity, wave):
    value = deepcopy(identity)
    for key in ("start_order", "step", "tolerance", "wavelengths_nm", "metal_csv", "factorization", "spectrum_sources"):
        value.pop(key, None)
    value["anchor_epsilon"] = {str(float(k)): v for k, v in value["anchor_epsilon"].items() if float(k) == wave}
    if not value["anchor_epsilon"]:
        raise ValueError("Reference is missing its wavelength permittivity.")
    if value.get("pmma"):
        value["pmma"].pop("path", None)
    return nv.identity_signature(value)


def reference_result(job, reference_root):
    if job["method"] != "symmetric-nv":
        return None
    path = reference_root/"order_sweep"/job["structure"]/"au"/f"wl{job['wave']:g}"/f"Nz{job['identity']['slices']}"/"checkpoint.json"
    if not path.is_file():
        return None
    saved = tables.read_json(path)
    original = saved["plan"]["identity"]
    if saved.get("signature") != nv.identity_signature(original):
        raise ValueError(f"Invalid reference identity signature: {path}")
    if job["wave"] not in original["wavelengths_nm"] or legacy_signature(original, job["wave"]) != legacy_signature(job["identity"], job["wave"]):
        return None
    row = saved["cases"].get(nv.case_key(job["order"], job["wave"]))
    if row is None:
        return None
    backend.validate_case(row, original)
    if (row["order"], row["wavelength_nm"]) != (job["order"], job["wave"]):
        raise ValueError(f"Reference has the wrong order/wavelength: {path}")
    return dict(status="ok", case=row, environment=saved.get("environment"),
        reused_from=dict(checkpoint=str(path.resolve()), sha256=nv.digest(path), original_signature=saved["signature"]))


def decorate(job, result):
    return dict(result, identity=job["identity"], signature=nv.identity_signature(job["identity"]),
        factorization=job["method"], order=job["order"], wavelength_nm=job["wave"])


def prepare(args):
    waves = tables.parse_wavelengths(args.wavelengths)
    methods = list(dict.fromkeys(x.strip() for x in args.factorizations.split(",")))
    if (not methods or any(m not in LABELS for m in methods) or args.order < 1 or args.slices < 2
            or args.grid < max(32, 4*args.order+4) or args.cpu_threads < 1):
        raise ValueError("Check method names, positive order/threads, Nz>=2, and Q>=max(32,4*M+4).")
    if args.device not in ("cpu", "cuda") and not (args.device.startswith("cuda:") and args.device[5:].isdigit()):
        raise ValueError("Use cpu, cuda, or cuda:<index>.")
    output, reference = args.output_root.resolve(), args.reference_root.resolve()
    if output == reference or reference in output.parents:
        raise ValueError("Output must be outside the read-only reference-root.")
    structures = ("metal", "pmma") if args.structures == "both" else (args.structures,)
    bases = {}
    for structure in structures:
        options = backend.build_parser().parse_args([])
        for key in ("device", "slices", "grid", "cpu_threads", "gold_model", "gold_csv", "pmma_csv"):
            setattr(options, key, getattr(args, key))
        options.start_order = options.max_order = args.order
        bases[structure] = deepcopy(backend.build_identity(options, "Au", structure, waves))
    jobs, reports, references = [], {}, {}
    for method in methods:
        sources = driver_sources(method)
        for structure in structures:
            for wave in waves:
                identity = deepcopy(bases[structure])
                identity.update(wavelengths_nm=[wave], factorization=method, spectrum_sources=sources,
                    anchor_epsilon={k: v for k, v in identity["anchor_epsilon"].items() if float(k) == wave})
                if method == "galerkin":
                    identity["solver"] = "Ordinary Fourier/Galerkin transverse convolution; D6 E1 x, complex128, Redheffer, half S"
                job = dict(method=method, structure=structure, order=args.order, wave=wave, identity=identity,
                    directory=f"{method}/{structure}/au/wl{wave:g}")
                jobs.append(job)
                key = case_key(job)
                prior = reference_result(job, reference)
                if prior is not None:
                    references[key] = decorate(job, prior)
                path = output/job["directory"]/"report.json"
                if path.exists():
                    report = tables.read_json(path)
                    signature = nv.identity_signature(identity)
                    if (report.get("signature") != signature or nv.identity_signature(report.get("identity")) != signature
                            or report.get("factorization") != method or report.get("status") not in STATUSES):
                        raise ValueError(f"Saved spectrum settings/source changed: {path}; use a new --output-root.")
                    if report["status"] == "ok":
                        backend.validate_case(report["case"], identity)
                        if (report["case"]["order"], report["case"]["wavelength_nm"]) != (args.order, wave):
                            raise ValueError(f"Saved result has a different order/wavelength: {path}")
                    elif report["status"] == "nonphysical":
                        validate_raw(report["power_diagnostics"])
                    reports[key] = report
    return waves, jobs, reports, references, output


def worker(plan_path, result_path, order, wave):
    try:
        identity = tables.read_json(plan_path)["identity"]
        method = identity["factorization"]
        if method not in LABELS or identity["spectrum_sources"] != driver_sources(method):
            raise ValueError("Spectrum driver/control source changed after preflight.")
        if method == "galerkin":
            import run_ag_nv_factorization_check as control
            context = control.factorization_scope(method)
        else:
            context = nullcontext()
        with context:
            return backend.worker(plan_path, result_path, order, wave)
    except Exception as error:
        nv.write_json(result_path, dict(status="calculation_failed", order=order, wavelength_nm=wave,
            error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc()))
        return 1


def launch(job, plan_path, result_path):
    completed = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()),
        "--_worker-plan", str(plan_path), "--_worker-result", str(result_path),
        "--_worker-order", str(job["order"]), "--_worker-wave", str(job["wave"])], cwd=ROOT, check=False)
    if not result_path.exists():
        return dict(status="calculation_failed", error=f"Worker exited {completed.returncode} without a result.")
    response = tables.read_json(result_path)
    expected = 0 if response.get("status") == "ok" else 2 if response.get("status") == "oom" else 1
    if completed.returncode != expected or response.get("status") not in ("ok", "oom", "calculation_failed"):
        return dict(status="calculation_failed", error="Worker exit code/status mismatch.")
    return response


def export(waves, jobs, reports, output):
    rows, means = [], []
    for job in jobs:
        report = reports.get(case_key(job), {})
        powers = report.get("case", report.get("power_diagnostics", {}))
        row = dict(material="Au", structure=job["structure"], factorization=job["method"], wavelength_nm=job["wave"],
            order=job["order"], slices=job["identity"]["slices"], grid=job["identity"]["grid"],
            status=report.get("status", "pending"), reused_reference=bool(report.get("reused_from")), error=report.get("error", ""))
        for short, field in tables.POWERS.items():
            row[f"{short}_percent"] = None if powers.get(field) is None else 100*float(powers[field])
        rows.append(row)
    for method, structure in dict.fromkeys((j["method"], j["structure"]) for j in jobs):
        group = [j for j in jobs if (j["method"], j["structure"]) == (method, structure)]
        valid = {case_key(j): reports[case_key(j)]["case"] for j in group if reports.get(case_key(j), {}).get("status") == "ok"}
        first = group[0]
        average_job = dict(material="Au", structure=structure, order=first["order"], plan={"identity": first["identity"]})
        mean = tables.averages(average_job, valid, waves)
        mean.update(method=LABELS[method], factorization=method,
            nonphysical_points=sum(reports.get(case_key(j), {}).get("status") == "nonphysical" for j in group))
        means.append(mean)
    tables.write_csv(output/"all_spectra.csv", rows, list(rows[0]))
    tables.write_csv(output/"averages.csv", means, list(means[0]))
    counts = {status: sum(r["status"] == status for r in rows) for status in sorted({r["status"] for r in rows})}
    summary = dict(requested_cases=len(rows), counts=counts, spectra=means,
        interpretation="Raw nonphysical powers are retained. Full-band means require all requested wavelengths to pass the existing power checks. NV/Galerkin agreement and a computed spectrum do not establish convergence.",
        mean_definition="Trapezoidal wavelength integral / bandwidth; arithmetic sample mean also saved. No spectral weighting.")
    nv.write_json(output/"summary.json", summary)
    return rows, means, summary


def plot(rows, means, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    for i, structure in enumerate(("metal", "pmma")):
        for k, (short, title) in enumerate((("R", "Reflectance"), ("T", "Transmittance"), ("A", "Absorptance"))):
            ax = axes[i, k]
            for method, color in (("symmetric-nv", "#0072B2"), ("galerkin", "#D55E00")):
                group = sorted((r for r in rows if r["structure"] == structure and r["factorization"] == method), key=lambda r:r["wavelength_nm"])
                if not group:
                    continue
                mean = next(m for m in means if m["structure"] == structure and m["factorization"] == method)
                average = f"mean {mean[f'mean_{short}_band_percent']:.3f}%" if mean["complete"] else f"mean unavailable ({mean['completed_points']}/{mean['expected_points']} valid)"
                ax.plot([r["wavelength_nm"] for r in group],
                    [r[f"{short}_percent"] if r[f"{short}_percent"] is not None else math.nan for r in group],
                    "o-" if method == "symmetric-nv" else "s--", ms=2.5, color=color, label=f"{LABELS[method]}: {average}")
            ax.set(title=f"{'Au only' if structure == 'metal' else 'PMMA + Au (30 nm)'} | {title}",
                xlabel="Wavelength (nm)", ylabel=f"{short} (%)")
            ax.grid(alpha=.25)
            if ax.lines:
                ax.legend(fontsize=8)
    first = rows[0]
    fig.suptitle(f"Au spectra | P=200 nm, h=500 nm | M={first['order']}, Nz={first['slices']} (+ cap for PMMA), Q={first['grid']}\nNormal incidence, x polarization; raw nonphysical values retained", fontsize=13)
    for ext in ("png", "svg"):
        fig.savefig(output/f"Au_spectra.{ext}", dpi=170)
    plt.close(fig)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args._worker_plan is not None:
        if any(v is None for v in (args._worker_result, args._worker_order, args._worker_wave)):
            parser.error("Incomplete private worker arguments.")
        return worker(args._worker_plan, args._worker_result, args._worker_order, args._worker_wave)
    try:
        waves, jobs, reports, references, output = prepare(args)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    print(f"Au {args.factorizations}: M={args.order}, Nz={args.slices}, Q={args.grid}; {len(waves)} wavelengths; {len(jobs)} cases.\nOutput: {output}", flush=True)
    for method, structure in dict.fromkeys((j["method"], j["structure"]) for j in jobs):
        group = [j for j in jobs if (j["method"], j["structure"]) == (method, structure)]
        reused = [j["wave"] for j in group if (saved := reports.get(case_key(j), references.get(case_key(j))))
            and (saved["status"] == "ok" or not args.retry_failures)]
        print(f"  {LABELS[method]} / {structure}: reuse {len(reused)}; new/retry {len(group)-len(reused)}; reused wavelengths={reused}", flush=True)
    if args.dry_run:
        print("Dry run: no calculation or writes.")
        return 0
    output.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        nv.write_json(output/"plan.json", dict(jobs=jobs, wavelengths_nm=waves))
        for job in jobs:
            key = case_key(job)
            saved = reports.get(key, references.get(key))
            if saved and (saved["status"] == "ok" or not args.retry_failures):
                reports[key] = saved
                nv.write_json(output/job["directory"]/"report.json", saved)
                continue
            directory = output/job["directory"]
            nv.write_json(directory/"plan.json", {"identity": job["identity"]})
            print(f"solve: Au/{job['structure']} {LABELS[job['method']]}, lambda={job['wave']:g} nm, M={job['order']}", flush=True)
            response = launch(job, directory/"plan.json", directory/f"worker_{uuid.uuid4().hex}.json")
            if response["status"] == "ok":
                if response.get("signature") != nv.identity_signature(job["identity"]):
                    raise ValueError("Worker signature mismatch.")
                backend.validate_case(response["case"], job["identity"])
                if (response["case"]["order"], response["case"]["wavelength_nm"]) != (job["order"], job["wave"]):
                    raise ValueError("Worker returned another condition.")
            elif "Nonphysical power result" in response.get("error", "") and response.get("power_diagnostics"):
                try:
                    validate_raw(response["power_diagnostics"])
                except (ValueError, KeyError, TypeError):
                    response.pop("power_diagnostics", None)
                else:
                    response["status"] = "nonphysical"
            reports[key] = decorate(job, response)
            nv.write_json(directory/"report.json", reports[key])
            export(waves, jobs, reports, output)
    rows, means, summary = export(waves, jobs, reports, output)
    if not args.no_plots:
        plot(rows, means, output)
    print(f"Counts: {summary['counts']}\nSaved: {output/'all_spectra.csv'}; {output/'averages.csv'}", flush=True)
    return 0 if all(r["status"] == "ok" for r in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
