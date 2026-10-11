"""Four-case NV/ASR convergence study and evidence-based optimization choice."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

from studies.shared.convergence_io import content_hash, signature_for, write_csv, write_json
from .model import (Geometry, Numerical, Study, adjacent_error, choose_candidate,
                    mean_reflectance, physical, release)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def parse_ints(value):
    values = sorted(set(int(v.strip()) for v in value.split(",")))
    if len(values) < 3 or any(v <= 0 for v in values):
        raise ValueError("Each axis requires at least three positive integer candidates.")
    return values


def parse_wavelengths(value):
    if ":" in value:
        if len(value.split(":")) != 3:
            raise ValueError("Wavelength ranges use start:stop:step (nm).")
        start, stop, step = map(float, value.split(":"))
        if not all(math.isfinite(v) for v in (start, stop, step)) or step <= 0 or stop < start:
            raise ValueError("Invalid wavelength range.")
        count = int(math.floor((stop - start) / step + 1e-10))
        if count > 10000:
            raise ValueError("Too many wavelength samples.")
        values = [start + i * step for i in range(count + 1)]
        if values[-1] < stop - 1e-9:
            values.append(stop)
    else:
        values = [float(v.strip()) for v in value.split(",")]
    if not values or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("Wavelengths must be finite and positive.")
    return tuple(sorted(set(values)))


def source_hashes():
    paths = [HERE / "model.py", ROOT / "rcwa_solver_auto.py",
             ROOT / "studies/pmma_gold_motheye/common.py"]
    paths += sorted((ROOT / "rcwa_ext").rglob("*.py"))
    hashes = {p.relative_to(ROOT).as_posix(): content_hash(p) for p in paths}
    spec = importlib.util.find_spec("torcwa")
    if spec and spec.origin:
        for p in sorted(Path(spec.origin).parent.glob("*.py")):
            hashes["installed_torcwa/" + p.name] = content_hash(p)
    return hashes


def write_profile(output, g, slices):
    rows = [{"slice": j + 1, "top_depth_nm": g.height_nm * j / slices,
             "bottom_depth_nm": g.height_nm * (j + 1) / slices,
             "radius_nm": g.radius(j, slices), "epsilon": g.index ** 2}
            for j in range(slices)]
    write_csv(output / "profile_layers.csv", rows)


def write_geometry(output, g, slices):
    write_profile(output, g, slices)
    from .plot_results import geometry_svg
    geometry_svg(g, output / "geometry.svg")


def scan_axis(study, axis, candidates, current, anchors, tolerance, energy_tolerance, cycle):
    spectra = {value: study.spectrum(replace(current, **{axis: value}), anchors) for value in candidates}
    selected, passed, comparisons = choose_candidate(candidates, spectra, tolerance, energy_tolerance)
    if passed:
        # Select the earliest fine end in a stable tail of at least two
        # intervals, checking it against every higher candidate as well.
        # A temporary plateau or cumulative drift cannot select a low order.
        for i in range(1, len(candidates) - 1):
            if all(c["passed"] for c in comparisons[i - 1:]) and all(
                adjacent_error(spectra[candidates[i]], spectra[finer])[0] <= tolerance
                for finer in candidates[i + 1:]
            ):
                selected = candidates[i]
                break
    return {"cycle": cycle, "axis": axis, "fixed": asdict(current),
            "candidates": candidates, "selected": selected, "converged": passed,
            "comparisons": comparisons, "spectra": {str(k): v for k, v in spectra.items()}}


def recommendation(reports, lattice, tolerance, gradient_atol, gradient_rtol, require_spectrum,
                   energy_tolerance=1e-6):
    from .audit import PARAMETERS, compatible
    pair = {name: reports.get(f"{lattice}/{name}") for name in ("nv", "asr")}
    result = {"lattice": lattice, "status": "insufficient_evidence",
              "derivative_free_method": None, "gradient_method": None}
    if any(r is None or r["status"] != "converged" for r in pair.values()):
        result["reason"] = "Both methods must pass their three-axis convergence and lossless checks."
        return result
    if any(r["tolerance"] > tolerance or r["energy_tolerance"] > energy_tolerance for r in pair.values()):
        result["reason"] = "A stored method was validated with a looser tolerance; rerun it at the requested accuracy."
        return result
    if require_spectrum and any(not r.get("final_spectrum_complete") for r in pair.values()):
        result["reason"] = "Requested full spectra are incomplete."
        return result
    spectra = {key: r["final_spectrum"] if require_spectrum else r["anchor_spectrum"] for key, r in pair.items()}
    try:
        error, per_metric = adjacent_error(spectra["nv"], spectra["asr"])
    except ValueError:
        result["reason"] = "Methods have different comparison wavelengths."
        return result
    result.update(cross_method_max_abs_change=error, cross_method_per_metric=per_metric,
                  tolerance=tolerance, comparison_scope="requested spectrum" if require_spectrum else "anchors only",
                  comparison_wavelength_range_nm=[spectra["nv"][0]["wavelength_nm"], spectra["nv"][-1]["wavelength_nm"]],
                  comparison_wavelength_count=len(spectra["nv"]),
                  converged_conditions={name: r.get("recommendation") for name, r in pair.items()})
    if not all(physical(row, energy_tolerance) for rows in spectra.values() for row in rows):
        result["reason"] = "Comparison spectra fail the requested lossless tolerance."
        return result
    if error > tolerance:
        result.update(status="methods_disagree", reason="Refine both methods; self-convergence alone is insufficient.")
        return result
    audits = {key: r.get("audit") for key, r in pair.items()}
    if any(not a or not a.get("forward_physical") for a in audits.values()):
        result["reason"] = "Warm, physical benchmark results are required for both methods."
        return result
    if audits["nv"]["settings"]["integration"] != audits["asr"]["settings"]["integration"]:
        result["reason"] = "Benchmarks use different devices or runtime settings."
        return result
    if audits["nv"]["settings"]["wavelengths"] != audits["asr"]["settings"]["wavelengths"]:
        result["reason"] = "Benchmarks cover different wavelength sets."
        return result

    def choice(costs):
        slower, faster = max(costs, key=costs.get), min(costs, key=costs.get)
        speedup = costs[slower] / costs[faster]
        return {"method": "comparable" if speedup < 1.15 else faster,
                "faster_measured": faster, "speedup": speedup,
                "median_seconds": costs, "timing_margin": "differences below 15% are treated as comparable"}

    result.update(status="forward_validated", derivative_free_method=choice(
        {name: a["forward"]["median_seconds"] for name, a in audits.items()}))
    eligible = {name: a for name, a in audits.items() if a.get("gradient_passed")
                and a["settings"]["atol_per_nm"] <= gradient_atol
                and a["settings"]["rtol"] <= gradient_rtol}
    if len(eligible) == 2:
        cross = {name: {"nv": audits["nv"]["gradient_per_nm"][name],
                        "asr": audits["asr"]["gradient_per_nm"][name],
                        "passed": compatible(audits["nv"]["gradient_per_nm"][name],
                                             audits["asr"]["gradient_per_nm"][name],
                                             gradient_atol, gradient_rtol)} for name in PARAMETERS}
        result["cross_method_gradients"] = cross
        if all(v["passed"] for v in cross.values()):
            result.update(status="gradient_validated", gradient_method=choice(
                {name: a["gradient"]["median_seconds"] for name, a in audits.items()}))
        else:
            result["gradient_reason"] = "Forward powers agree, but cross-method derivatives do not; increase gradient accuracy."
    elif len(eligible) == 1:
        method = next(iter(eligible))
        result["gradient_method"] = {"method": method, "reason": "Only this method passed the internal gradient audit; cross-method gradient agreement is unverified."}
        result["status"] = "one_method_gradient_candidate"
    else:
        result["gradient_reason"] = "Neither method has a passing gradient audit."
    return result


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="auto")
    p.add_argument("--lattices", default="hexagonal,square")
    p.add_argument("--methods", default="nv,asr")
    p.add_argument("--index", type=float, default=2.5)
    for name, value in (("period-nm", 200.), ("height-nm", 500.), ("tip-radius-nm", 5.),
                        ("base-radius-nm", 95.), ("profile-power", 1.), ("asr-circle-g", .03)):
        p.add_argument("--" + name, type=float, default=value)
    p.add_argument("--orders", default="2,4,6,8,10,12")
    p.add_argument("--slices", default="20,40,60,80")
    p.add_argument("--grids", default="128,256,384")
    p.add_argument("--anchor-wavelengths", default="400,430,436,495,505,550,700")
    p.add_argument("--spectrum-wavelengths", default="400:700:5")
    p.add_argument("--tolerance", type=float, default=.001)
    p.add_argument("--energy-tolerance", type=float, default=1e-6)
    p.add_argument("--max-cycles", type=int, default=4)
    p.add_argument("--run-final-spectrum", action="store_true")
    p.add_argument("--skip-gradient-audit", action="store_true")
    p.add_argument("--benchmark-repeats", type=int, default=3)
    p.add_argument("--fd-height-step-nm", type=float, default=.1)
    p.add_argument("--fd-radius-step-nm", type=float, default=.02)
    p.add_argument("--gradient-atol", type=float, default=1e-7)
    p.add_argument("--gradient-rtol", type=float, default=.05)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--no-symmetry", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--output-root", type=Path, default=HERE / "results")
    return p


def comparison_exit_code(reports, selected_keys, choices, lattices, *, require_gradients):
    selected = [reports[key] for key in selected_keys if key in reports]
    if any(r["status"] in {"calculation_failed", "interrupted"} for r in selected):
        return 1
    accepted = {"gradient_validated"} if require_gradients else {
        "forward_validated", "gradient_validated", "one_method_gradient_candidate"}
    complete = bool(selected) and all(r["status"] == "converged" for r in selected)
    return 0 if complete and all(choices[l]["status"] in accepted for l in lattices) else 2


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    output = args.output_root.resolve()
    if args.plot_only:
        from .plot_results import make_figures
        make_figures(output / "comparison.json")
        return 0
    try:
        geometry = Geometry(**{name: getattr(args, name) for name in Geometry.__dataclass_fields__})
        geometry.validate()
        axes = {"order": parse_ints(args.orders), "slices": parse_ints(args.slices), "grid": parse_ints(args.grids)}
        if min(axes["grid"]) < max(32, 4 * max(axes["order"]) + 4):
            raise ValueError("Every quadrature/projection grid must be >=max(32,4*max(M)+4).")
        lattices = list(dict.fromkeys(v.strip() for v in args.lattices.split(",")))
        methods = list(dict.fromkeys(v.strip() for v in args.methods.split(",")))
        if not lattices or any(v not in {"hexagonal", "square"} for v in lattices):
            raise ValueError("Lattices: hexagonal,square")
        if not methods or any(v not in {"nv", "asr"} for v in methods):
            raise ValueError("Methods: nv,asr")
        anchors, final_wavelengths = parse_wavelengths(args.anchor_wavelengths), parse_wavelengths(args.spectrum_wavelengths)
        for value in (args.tolerance, args.energy_tolerance, args.fd_height_step_nm, args.fd_radius_step_nm,
                      args.gradient_atol, args.gradient_rtol):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Tolerances and finite-difference steps must be positive and finite.")
        if min(args.max_cycles, args.benchmark_repeats, args.cpu_threads) <= 0:
            raise ValueError("Cycles, benchmark repeats and CPU threads must be positive.")
        if args.device not in {"auto", "cpu", "cuda"} and not (args.device.startswith("cuda:") and args.device[5:].isdigit()):
            raise ValueError("Device: auto, cpu, cuda, cuda:<index>")
        # Reject invalid perturbed geometries before any expensive solve.
        if not args.skip_gradient_audit:
            for name, step in (("height_nm", args.fd_height_step_nm), ("base_radius_nm", args.fd_radius_step_nm)):
                for sign in (-1, 1):
                    replace(geometry, **{name: getattr(geometry, name) + sign * step}).validate()
    except (ValueError, OSError) as error:
        parser.error(str(error))

    assumptions = {"geometry": asdict(geometry), "material": "lossless, nondispersive; epsilon=n^2, mu=1",
                   "substrate": "semi-infinite same material", "incidence": "normal x polarization",
                   "dtype": "complex128", "cascade": "redheffer", "smatrix_size": "half", "fields": "none",
                   "use_symmetry": not args.no_symmetry, "nv_lanczos_filter": False,
                   "asr_factorization_rules": True, "source_sha256": source_hashes()}
    signature = signature_for(assumptions)
    plan = {"assumptions": assumptions, "signature": signature, "lattices": lattices, "methods": methods,
            "candidate_axes": axes, "anchor_wavelengths_nm": anchors, "spectrum_wavelengths_nm": final_wavelengths,
            "tolerance": args.tolerance, "energy_tolerance": args.energy_tolerance,
            "max_cycles": args.max_cycles, "benchmark_repeats": args.benchmark_repeats,
            "gradient_audit": {"enabled": not args.skip_gradient_audit,
                "steps_nm": {"height_nm": args.fd_height_step_nm, "base_radius_nm": args.fd_radius_step_nm},
                "atol_per_nm": args.gradient_atol, "rtol": args.gradient_rtol},
            "requested_device": args.device, "output_root": str(output),
            "substrate_first_diffraction_threshold_nm": {"square": geometry.index * geometry.period_nm,
                "hexagonal": geometry.index * geometry.period_nm * math.sqrt(3) / 2},
            "flat_interface_reflectance": ((geometry.index - 1) / (geometry.index + 1)) ** 2,
            "base_fill_fraction": {l: geometry.fill_fraction(l) for l in lattices}}
    if args.dry_run:
        write_json(output / "plan.json", plan)
        write_geometry(output, geometry, axes["slices"][-2])
        print(json.dumps(plan, indent=2))
        print("Dry run only; no Torch import or optical solve.")
        return 0

    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available() or (device.index or 0) >= torch.cuda.device_count():
            parser.error("Explicit CUDA request cannot be satisfied. No CPU fallback; run on the GPU host.")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(args.cpu_threads)
    integration = {"device": str(device), "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                   "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                   "cpu_threads": args.cpu_threads}
    plan["integration"] = integration
    # Validate every requested checkpoint before overwriting any report.
    studies = {}
    try:
        for lattice in lattices:
            for method in methods:
                group = output / lattice / method
                key = f"{lattice}/{method}"
                studies[key] = Study(geometry=geometry, method=method, lattice=lattice, device=device,
                    use_symmetry=not args.no_symmetry, signature=signature_for({"physical": signature, "group": key}),
                    checkpoint=group / "checkpoint.json")
    except (ValueError, OSError) as error:
        parser.error(str(error))
    write_json(output / "plan.json", plan)
    write_geometry(output, geometry, axes["slices"][-2])
    print(f"Device: {device}; complex128; NV and matched-ASR; output: {output}", flush=True)
    print("R/T include all propagating orders. Lossless test: R+T=1; optimization objective: mean R.", flush=True)
    reports = {}
    interrupted = False
    for key, study in studies.items():
        group = study.checkpoint.parent
        report_path = group / "convergence.json"
        current = Numerical(**{name: values[-2] for name, values in axes.items()})
        report = {"status": "running", "physical_signature": study.signature,
                  "lattice": study.lattice, "method": study.method, "candidate_axes": axes,
                  "anchor_wavelengths_nm": anchors, "tolerance": args.tolerance,
                  "energy_tolerance": args.energy_tolerance, "recommendation": asdict(current),
                  "history": [], "anchor_spectrum": [], "final_spectrum": [],
                  "final_spectrum_complete": False, "audit": None, "integration": integration}
        write_json(report_path, report)
        write_csv(group / "spectrum.csv", [])
        try:
            converged = False
            for cycle in range(1, args.max_cycles + 1):
                start, statuses = current, []
                for axis in ("slices", "order", "grid"):
                    print(f"{key}: cycle {cycle}/{args.max_cycles}, scan {axis}", flush=True)
                    record = scan_axis(study, axis, axes[axis], current, anchors,
                                       args.tolerance, args.energy_tolerance, cycle)
                    report["history"].append(record)
                    current = replace(current, **{axis: record["selected"]})
                    report["recommendation"] = asdict(current)
                    statuses.append(record["converged"])
                    write_json(report_path, report)
                converged = current == start and all(statuses)
                if current == start:
                    break
            report["anchor_spectrum"] = study.spectrum(current, anchors)
            report["status"] = ("converged" if converged else "candidate_range_insufficient"
                                if not all(statuses) else "cycle_limit_reached")
            if not all(physical(r, args.energy_tolerance) for r in report["anchor_spectrum"]):
                report["status"] = "nonphysical_results"
            report["mean_anchor_reflectance"] = mean_reflectance(report["anchor_spectrum"])
            if report["status"] == "converged" and args.run_final_spectrum:
                report["status"] = "spectrum_running"
                for w in final_wavelengths:
                    report["final_spectrum"].append(study.evaluate(current, w))
                    write_json(report_path, report)
                report["final_spectrum_complete"] = True
                report["status"] = ("converged" if all(physical(r, args.energy_tolerance)
                    for r in report["final_spectrum"]) else "nonphysical_final_spectrum")
                report["mean_spectrum_reflectance"] = mean_reflectance(report["final_spectrum"])
            if report["status"] == "converged":
                from .audit import run_audit
                report["status"] = "audit_running"
                write_json(report_path, report)
                report["audit"] = run_audit(study, current,
                    {axis: replace(current, **{axis: values[-1]}) for axis, values in axes.items()},
                    anchors, output=group, repeats=args.benchmark_repeats,
                    steps={"height_nm": args.fd_height_step_nm, "base_radius_nm": args.fd_radius_step_nm},
                    atol=args.gradient_atol, rtol=args.gradient_rtol,
                    energy_tolerance=args.energy_tolerance, integration=integration,
                    check_gradients=not args.skip_gradient_audit)
                report["status"] = "converged"
        except (Exception, KeyboardInterrupt) as error:
            report["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "calculation_failed"
            report["error"] = f"{type(error).__name__}: {error}"
            interrupted = isinstance(error, KeyboardInterrupt)
            print(f"{key} stopped: {report['error']}; completed solves are saved.", flush=True)
        finally:
            report["completed_cases"] = len(study.cases)
            write_profile(group, geometry, current.slices)
            write_csv(group / "all_cases.csv", list(study.cases.values()))
            write_csv(group / "anchor_spectrum.csv", report["anchor_spectrum"])
            write_csv(group / "spectrum.csv", report["final_spectrum"])
            write_json(report_path, report)
            reports[key] = report
        release(device)
        if interrupted:
            break
    # Include other groups from earlier subset runs only if their physics matches.
    for lattice in ("hexagonal", "square"):
        for method in ("nv", "asr"):
            key = f"{lattice}/{method}"
            path = output / lattice / method / "convergence.json"
            if key not in reports and path.exists():
                saved = json.loads(path.read_text(encoding="utf-8"))
                if saved.get("physical_signature") == signature_for({"physical": signature, "group": key}):
                    reports[key] = saved
    choices = {l: recommendation(reports, l, args.tolerance, args.gradient_atol,
                args.gradient_rtol, args.run_final_spectrum, args.energy_tolerance) for l in ("hexagonal", "square")}
    summary = {"plan": plan, "groups": reports, "recommendations": choices,
               "limits": "Initial geometry only; finite refinement evidence, not an error bound. Gradient checks cover height and base radius, not arbitrary per-slice shapes. Recheck convergence after optimization."}
    write_json(output / "comparison.json", summary)
    if not args.no_plots:
        try:
            from .plot_results import make_figures
            make_figures(output / "comparison.json")
        except Exception as error:
            print(f"Plotting failed: {error}; numerical results are saved. Use --plot-only after fixing dependencies.")
    print(json.dumps(choices, indent=2))
    print(f"Comparison: {output / 'comparison.json'}")
    return comparison_exit_code(reports, studies, choices, lattices,
                                require_gradients=not args.skip_gradient_audit)


if __name__ == "__main__":
    raise SystemExit(main())
