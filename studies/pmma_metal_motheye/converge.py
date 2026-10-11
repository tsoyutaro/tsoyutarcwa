"""Three-axis convergence audit for a metal-coated PMMA hexagonal moth-eye array."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

from .model import (ROOT, Geometry, Numerical, PMMAIndex, Study, choose_candidate,
                    content_hash, load_tabulated_nk, physical, signature_for,
                    write_csv, write_json)

from studies.shared.transition_metals import select_material, default_wavelengths, provenance

HERE = Path(__file__).resolve().parent


def integers(value: str) -> list[int]:
    numbers = sorted(set(int(v.strip()) for v in value.split(",")))
    if len(numbers) < 3 or any(v <= 0 for v in numbers):
        raise ValueError("Each convergence axis requires at least three positive candidates.")
    return numbers


def wavelengths(value: str) -> tuple[float, ...]:
    if ":" in value:
        fields = value.split(":")
        if len(fields) != 3:
            raise ValueError("Wavelength ranges use start:stop:step (nm).")
        start, stop, step = map(float, fields)
        if not all(math.isfinite(v) for v in (start, stop, step)) or step <= 0 or stop < start:
            raise ValueError("Invalid wavelength range.")
        count = int(math.floor((stop - start) / step + 1e-10))
        if count > 10000:
            raise ValueError("Wavelength range exceeds 10001 samples.")
        values = [start + i * step for i in range(count + 1)]
        if values[-1] < stop - 1e-9:
            values.append(stop)
    else:
        values = [float(v.strip()) for v in value.split(",")]
    if not values or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("Wavelengths must be finite and positive.")
    return tuple(sorted(set(values)))


def source_hashes() -> dict:
    paths = [Path(__file__), HERE / "model.py", ROOT / "rcwa_solver_auto.py",
             ROOT / "studies/pmma_gold_motheye/common.py",
             ROOT / "studies/pmma_gold_motheye/geometry_preview.py",
             ROOT / "studies/shared/tabulated_nk.py", ROOT / "studies/shared/transition_metals.py",
             HERE / "plot_results.py"]
    paths += sorted((ROOT / "rcwa_ext").rglob("*.py"))
    return {p.relative_to(ROOT).as_posix(): content_hash(p) for p in paths}


def write_geometry(output: Path, geometry: Geometry, slices: int, material: str) -> None:
    """Export the very same midpoint slices used by the RCWA engine."""
    layers = geometry.profile(slices)
    rows = [{"layer": i + 1, "top_depth_nm": layer.top_depth_nm,
             "bottom_depth_nm": layer.bottom_depth_nm,
             "pmma_core_radius_nm": layer.core_radius_nm,
             "metal_outer_radius_nm": layer.outer_radius_nm,
             "kind": layer.kind, "outside_core": material if layer.kind == "valley"
             else f"{material} shell, then air"} for i, layer in enumerate(layers)]
    write_csv(output / "profile_layers.csv", rows)
    from .plot_results import geometry_svg
    geometry_svg(geometry, layers, output / "geometry.svg", material)


def scan_axis(study: Study, axis: str, candidates: list[int], current: Numerical,
              anchors: tuple[float, ...], tolerance: float, cycle: int) -> dict:
    spectra = {}
    for candidate in candidates:
        spectra[candidate] = study.spectrum(replace(current, **{axis: candidate}), anchors)
    selected, passed, comparisons = choose_candidate(candidates, spectra, tolerance)
    return {"cycle": cycle, "axis": axis, "fixed": asdict(current),
            "candidates": candidates, "selected": selected, "converged": passed,
            "comparisons": comparisons, "spectra": {str(k): v for k, v in spectra.items()}}


def spectrum_statistics(rows: list[dict]) -> dict:
    if not rows or not all(physical(row) for row in rows):
        raise ValueError("Spectrum statistics require physical R/T/A at every wavelength.")
    result = {}
    for name in ("reflectance", "transmittance", "absorptance"):
        values = [row[name] for row in rows]
        span = rows[-1]["wavelength_nm"] - rows[0]["wavelength_nm"]
        integral = sum(0.5 * (a[name] + b[name]) * (b["wavelength_nm"] - a["wavelength_nm"])
                       for a, b in zip(rows, rows[1:]))
        result[name] = {"sample_mean": sum(values) / len(values),
                        "wavelength_mean_trapezoidal": integral / span if span > 0 else values[0],
                        "min": min(values), "max": max(values)}
    return {"range_nm": [rows[0]["wavelength_nm"], rows[-1]["wavelength_nm"]],
            "samples": len(rows), "weighting": "uniform wavelength; no solar or source weighting",
            "powers": result}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0 etc.; explicit CUDA never falls back")
    p.add_argument("--orders", default="8,12,16,18,20")
    p.add_argument("--slices", default="40,60,80,100")
    p.add_argument("--grids", default="256,384,512")
    p.add_argument("--max-cycles", type=int, default=4)
    p.add_argument("--tolerance", type=float, default=0.005, help="absolute R/T/A change, 0.005 = 0.5 percentage points")
    p.add_argument("--anchor-wavelengths", help="Default: 400,550,700; W: 667,680,700")
    p.add_argument("--spectrum-wavelengths", help="Default: 400:700:5; W: 667:700:5")
    p.add_argument("--run-final-spectrum", action="store_true")
    p.add_argument("--material", required=True, help="Fe, Ni, Ti, W or V")
    p.add_argument("--metal-csv", type=Path, help="Optional replacement optical-constants CSV")
    p.add_argument("--pmma-csv", type=Path, default=ROOT / "studies/pmma_cr_motheye/data/Szczurowski.csv")
    p.add_argument("--pmma-shortwave-policy", choices=("linear-to-400", "error"), default="linear-to-400")
    p.add_argument("--radial-mapping", choices=("outer", "double"), default="outer")
    p.add_argument("--cascade", choices=("redheffer", "torcwa"), default="redheffer")
    p.add_argument("--no-symmetry", action="store_true")
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--period-nm", type=float, default=200.)
    p.add_argument("--height-nm", type=float, default=500.)
    p.add_argument("--tip-radius-nm", type=float, default=5.)
    p.add_argument("--base-radius-nm", type=float, default=65.)
    p.add_argument("--metal-side-thickness-nm", type=float, default=30.)
    p.add_argument("--metal-top-thickness-nm", type=float, default=30.)
    p.add_argument("--metal-valley-thickness-nm", type=float, default=30.)
    p.add_argument("--profile-power", type=float, default=1.)
    p.add_argument("--asr-circle-g", type=float, default=.03)
    p.add_argument("--output-root", type=Path, help="Default: studies/pmma_metal_motheye/results/<symbol>")
    p.add_argument("--dry-run", action="store_true", help="write plan and geometry without electromagnetic calculations")
    p.add_argument("--plot-only", action="store_true", help="replot saved report without solving")
    p.add_argument("--no-plots", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        material, args.metal_csv, metal = select_material(args.material, args.metal_csv)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    default_anchors, default_spectrum = default_wavelengths(material)
    args.anchor_wavelengths = args.anchor_wavelengths or default_anchors
    args.spectrum_wavelengths = args.spectrum_wavelengths or default_spectrum
    output = (args.output_root or HERE / "results" / material.lower()).resolve()
    report_path = output / "pmma_metal_motheye_convergence.json"
    if args.plot_only:
        from .plot_results import make_figures
        make_figures(report_path)
        return 0
    try:
        geometry = Geometry(**{name: getattr(args, name) for name in Geometry.__dataclass_fields__})
        geometry.validate()
        axes = {"order": integers(args.orders), "slices": integers(args.slices),
                "grid": integers(args.grids)}
        minimum_grid = max(32, 4 * max(axes["order"]) + 4)
        if min(axes["grid"]) < minimum_grid:
            raise ValueError(f"Each grid must be >= {minimum_grid} for the highest Fourier order.")
        if (not math.isfinite(args.tolerance) or args.tolerance <= 0
                or args.max_cycles <= 0 or args.cpu_threads <= 0):
            raise ValueError("Tolerance, max-cycles and cpu-threads must be positive.")
        anchors = wavelengths(args.anchor_wavelengths)
        final_wavelengths = wavelengths(args.spectrum_wavelengths)
        pmma = PMMAIndex(args.pmma_csv, args.pmma_shortwave_policy)
        for w in (*anchors, *(final_wavelengths if args.run_final_spectrum else ())):
            metal(w)
            pmma(w)
        requested = args.device
        if requested not in {"auto", "cpu", "cuda"} and not (
                requested.startswith("cuda:") and requested[5:].isdigit()):
            raise ValueError("Supported devices: auto, cpu, cuda, cuda:<index>.")
    except (ValueError, OSError) as error:
        parser.error(str(error))

    assumptions = {
        "model": f"PMMA core / {material} shell / air, triangular lattice, PMMA substrate",
        "material": material, "material_provenance": provenance(material, args.metal_csv),
        "geometry": asdict(geometry), "substrate": "semi-infinite, lossless dispersive PMMA",
        "incidence": "normal, x polarization", "dtype": "complex128",
        "radial_mapping": args.radial_mapping, "cascade": args.cascade,
        "symmetry": "disabled" if args.no_symmetry else "D6 source sector",
        "factorization_rules": True, "anchor_wavelengths_nm": anchors,
        "metal": {"csv": args.metal_csv.name, "sha256_lf": content_hash(args.metal_csv),
               "interpolation": "linear epsilon in wavelength_nm; epsilon=(n+i*k)^2; no extrapolation",
               "tabulated_range_nm": [metal.wavelength_nm[0], metal.wavelength_nm[-1]]},
        "pmma": pmma.provenance(args.pmma_csv), "source_sha256": source_hashes(),
        "convergence_rule": "last two refinements on every axis, physical powers, stable fixed configuration v1"}
    signature = signature_for(assumptions)
    plan = {"signature": signature, "assumptions": assumptions, "candidate_axes": axes,
            "tolerance": args.tolerance, "tolerance_percentage_points": args.tolerance * 100,
            "max_cycles": args.max_cycles, "requested_device": requested,
            "output_root": str(output), "final_spectrum_requested": args.run_final_spectrum,
            "spectrum_wavelengths_nm": final_wavelengths,
            "anchor_materials": [{"wavelength_nm": w, "pmma_n": pmma(w),
                                 "epsilon_metal_real": metal(w).real, "epsilon_metal_imag": metal(w).imag}
                                for w in anchors]}
    current = Numerical(order=axes["order"][-2], slices=axes["slices"][-2], grid=axes["grid"][-2])
    if args.dry_run:
        write_json(output / "plan.json", plan)
        write_geometry(output, geometry, current.slices, material)
        print(json.dumps(plan, indent=2))
        print(f"Dry run: no electromagnetic calculation. Geometry: {output / 'geometry.svg'}")
        return 0

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else torch.device(requested)
    if device.type == "cuda" and (not torch.cuda.is_available()
                                 or (device.index or 0) >= torch.cuda.device_count()):
        parser.error("Requested CUDA device is unavailable; run on the GPU host or explicitly choose --device cpu.")
    torch.set_num_threads(args.cpu_threads)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    try:
        study = Study(geometry=geometry, metal_csv=args.metal_csv, pmma_csv=args.pmma_csv,
                      pmma_policy=args.pmma_shortwave_policy, device=device,
                      radial_mapping=args.radial_mapping, cascade=args.cascade,
                      use_symmetry=not args.no_symmetry, signature=signature,
                      checkpoint=output / "checkpoint.json", material_name=material)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    integration = {"requested_device": requested, "device": str(device),
                   "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                   "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                   "host_preparation": "geometry, material interpolation and quadrature setup on CPU",
                   "gpu_calculation": "complex128 matrices, eigenproblems, interfaces, cascade and flux"}
    plan["integration"] = integration
    write_json(output / "plan.json", plan)
    write_geometry(output, geometry, current.slices, material)
    print(f"Output: {output}\nDevice: {device}; mapping={args.radial_mapping}; complex128", flush=True)
    print(f"R=reflected flux; T=flux into lossless PMMA; A=1-R-T ({material} absorption).", flush=True)
    if min(anchors) < pmma.wavelengths[0]:
        print(f"PMMA: {min(anchors):g}..{pmma.wavelengths[0]:g} nm uses the recorded shortwave extension.", flush=True)
    report = {"status": "running", "signature": signature, "assumptions": assumptions,
              "candidate_axes": axes, "tolerance": args.tolerance,
              "max_cycles": args.max_cycles, "recommendation": asdict(current),
              "history": [], "anchor_spectrum": [], "final_spectrum": [],
              "expected_spectrum_samples": len(final_wavelengths),
              "final_spectrum_complete": False,
              "spectrum_statistics": None, "integration": integration,
              "interpretation": f"A=1-R-T; lossless PMMA; absorption occurs in patterned {material} and valley coating; no metal substrate.",
              "limitations": "Convergence is checked at anchor wavelengths only; R+T+A=1 is an algebraic identity, not an independent absorption-volume check. Ideal radial shell and flat cap approximate deposition."}
    write_json(report_path, report)
    write_csv(output / "anchor_spectrum.csv", [])
    write_csv(output / "spectrum.csv", [])
    converged = False
    statuses = []
    failure = None
    try:
        for cycle in range(1, args.max_cycles + 1):
            start = current
            statuses = []
            for axis in ("slices", "order", "grid"):
                print(f"Cycle {cycle}/{args.max_cycles}: {axis}", flush=True)
                record = scan_axis(study, axis, axes[axis], current, anchors, args.tolerance, cycle)
                current = replace(current, **{axis: record["selected"]})
                report["history"].append(record)
                report["recommendation"] = asdict(current)
                statuses.append(record["converged"])
                write_json(report_path, report)
                print(f"  selected {axis}={record['selected']}; tail passed={record['converged']}", flush=True)
            converged = current == start and all(statuses)
            if current == start:
                break
        report["anchor_spectrum"] = study.spectrum(current, anchors)
        report["status"] = ("converged" if converged else "candidate_range_insufficient"
                            if not all(statuses) else "cycle_limit_reached")
        write_geometry(output, geometry, current.slices, material)
        write_csv(output / "anchor_spectrum.csv", report["anchor_spectrum"])
        if args.run_final_spectrum and converged:
            report["status"] = "final_spectrum_running"
            write_json(report_path, report)
            for w in final_wavelengths:
                report["final_spectrum"].append(study.evaluate(current, w))
                write_json(report_path, report)
            report["final_spectrum_complete"] = True
            if all(physical(row) for row in report["final_spectrum"]):
                report["status"] = "converged"
                report["spectrum_statistics"] = spectrum_statistics(report["final_spectrum"])
            else:
                report["status"] = "nonphysical_final_spectrum"
                converged = False
            write_csv(output / "spectrum.csv", report["final_spectrum"])
        elif args.run_final_spectrum:
            print("Final spectrum skipped because anchor convergence was not established.", flush=True)
    except (Exception, KeyboardInterrupt) as error:
        report["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "calculation_failed"
        report["error"] = f"{type(error).__name__}: {error}"
        failure = report["error"]
        converged = False
    finally:
        report["integration"]["completed_cases"] = len(study.cases)
        write_csv(output / "all_cases.csv", list(study.cases.values()))
        write_csv(output / "anchor_spectrum.csv", report["anchor_spectrum"])
        write_csv(output / "spectrum.csv", report["final_spectrum"])
        write_json(report_path, report)
    if not args.no_plots:
        from .plot_results import make_figures
        try:
            make_figures(report_path)
        except Exception as error:
            report["plot_error"] = f"{type(error).__name__}: {error}"
            write_json(report_path, report)
            print(f"Figure generation failed: {report['plot_error']}. Numerical results are saved; use --plot-only after fixing plotting dependencies.")
    print(json.dumps({"status": report["status"], "recommendation": asdict(current)}, indent=2))
    print(f"report: {report_path}")
    if failure:
        print(f"Calculation stopped: {failure}\nCompleted cases are saved. Rerun the same command to resume.")
        return 1
    return 0 if converged else 2


if __name__ == "__main__":
    raise SystemExit(main())
