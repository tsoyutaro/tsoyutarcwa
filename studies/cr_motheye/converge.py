"""Three-axis matched-ASR convergence study for Cr pillars on a Cr substrate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from studies.shared.tabulated_nk import load_tabulated_nk
from studies.gold_motheye.converge import (
    GeometryConfig, NumericalConfig, Study, _configuration_signature,
    _parse_int_list, _parse_wavelengths, _scan_axis, _validate_geometry,
)


def write_csv(path, rows):
    columns = ("wavelength_nm", "order", "slices", "grid", "epsilon_material_real",
               "epsilon_material_imag", "reflectance", "transmittance_far",
               "absorptance_total", "motheye_absorptance", "substrate_absorptance",
               "power_into_substrate", "passivity_warning", "reduced_dimension",
               "full_dimension", "symmetry_reduction", "symmetry_irrep", "runtime_seconds")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cr-csv", type=Path, default=PACKAGE / "data" / "Johnson.csv")
    parser.add_argument("--period-nm", type=float, default=200.0)
    parser.add_argument("--height-nm", type=float, default=500.0)
    parser.add_argument("--tip-radius-nm", type=float, default=5.0)
    parser.add_argument("--base-radius-nm", type=float, default=95.0)
    parser.add_argument("--profile-power", type=float, default=1.0)
    parser.add_argument("--substrate-mode", choices=("semi-infinite", "finite"), default="semi-infinite")
    parser.add_argument("--substrate-thickness-nm", type=float, default=200.0)
    parser.add_argument("--back-index", type=float, default=1.0)
    parser.add_argument("--asr-circle-g", type=float, default=0.03)
    parser.add_argument("--anchor-wavelengths", default="400,550,700")
    parser.add_argument("--orders", default="4,6,8,10,12,14,16,18,20")
    parser.add_argument("--slices", default="50,60,70,80,90,100")
    parser.add_argument("--grids", default="96,128,192,256")
    parser.add_argument("--tolerance", type=float, default=0.005,
                        help="Absolute power change; 0.005 means 0.5 percentage point.")
    parser.add_argument("--max-cycles", type=int, default=3)
    parser.add_argument("--cascade", choices=("redheffer", "algo2a"), default="redheffer")
    parser.add_argument("--symmetry-reduction", choices=("d6-source", "cs-source"), default="d6-source")
    parser.add_argument("--no-symmetry", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-prefix", type=Path, default=PACKAGE / "results" / "cr_motheye")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-final-spectrum", action="store_true")
    parser.add_argument("--spectrum-wavelengths", default="400:700:5")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    geometry = GeometryConfig(period_nm=args.period_nm, height_nm=args.height_nm,
                              tip_radius_nm=args.tip_radius_nm, base_radius_nm=args.base_radius_nm,
                              profile_power=args.profile_power, lattice="triangular",
                              substrate_mode=args.substrate_mode,
                              substrate_thickness_nm=args.substrate_thickness_nm,
                              back_index=args.back_index, asr_circle_g=args.asr_circle_g)
    _validate_geometry(geometry)
    axes = {name: _parse_int_list(getattr(args, name), name) for name in ("orders", "slices", "grids")}
    if min(map(len, axes.values())) < 3:
        parser.error("Each convergence axis needs at least three candidates.")
    minimum_grid = max(32, 4 * max(axes["orders"]) + 4)
    if min(axes["grids"]) < minimum_grid:
        parser.error(f"Every grid must be at least {minimum_grid} for the highest tested Fourier order.")
    if not math.isfinite(args.tolerance) or args.tolerance <= 0 or args.max_cycles <= 0:
        parser.error("Tolerance and max-cycles must be positive.")
    wavelengths = _parse_wavelengths(args.anchor_wavelengths)
    spectrum_wavelengths = _parse_wavelengths(args.spectrum_wavelengths)
    model = load_tabulated_nk(args.cr_csv)
    for w in (*wavelengths, *(spectrum_wavelengths if args.run_final_spectrum else ())):
        model(w)  # Reject out-of-range data before the expensive calculation.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    use_symmetry = not args.no_symmetry and geometry.substrate_mode == "semi-infinite"
    source_hashes = {}
    source_paths = [Path(__file__), ROOT / "studies/gold_motheye/converge.py",
                    ROOT / "studies/shared/tabulated_nk.py", ROOT / "rcwa_solver_auto.py"]
    source_paths += sorted((ROOT / "rcwa_ext").rglob("*.py"))
    for path in source_paths:
        content = path.read_bytes().replace(b"\r\n", b"\n")
        source_hashes[path.relative_to(ROOT).as_posix()] = hashlib.sha256(content).hexdigest()
    assumptions = {
        "material": "Cr", "material_model": "Johnson-Christy-1974-tabulated",
        "material_csv": args.cr_csv.name,
        "material_csv_sha256": hashlib.sha256(args.cr_csv.read_bytes()).hexdigest(),
        "interpolation": "linear epsilon in wavelength_nm; epsilon=(n+i*k)^2; no extrapolation",
        "tabulated_range_nm": [model.wavelength_nm[0], model.wavelength_nm[-1]],
        "geometry": asdict(geometry), "anchor_wavelengths": wavelengths,
        "cascade": args.cascade, "use_symmetry": use_symmetry,
        "symmetry_reduction": args.symmetry_reduction, "dtype": "complex128",
        "smatrix_size": "half", "source_sha256": source_hashes,
        "convergence_rule": "two_passing_steps_at_refinement_tail_and_passivity_v3",
    }
    signature = _configuration_signature(assumptions)
    plan = {"assumptions": assumptions, "signature": signature,
            "candidate_axes": {**axes, "tolerance": args.tolerance}, "max_cycles": args.max_cycles,
            "requested_device": args.device, "device": str(device),
            "cuda_available_here": torch.cuda.is_available(),
            "output_prefix": str(args.output_prefix.resolve()),
            "anchor_epsilon": {str(w): {"real": model(w).real, "imag": model(w).imag} for w in wavelengths}}
    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_name(prefix.name + "_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable. Select --device cpu or run on the GPU host.")
    if device.type not in {"cpu", "cuda"}:
        parser.error("Supported devices are cpu, cuda and auto.")
    study = Study(geometry=geometry, gold_epsilon=model, wavelengths=wavelengths,
                  cascade=args.cascade, use_symmetry=use_symmetry,
                  symmetry_reduction=args.symmetry_reduction, device=device,
                  checkpoint=prefix.with_name(prefix.name + "_checkpoint.json"),
                  signature=signature, material_name="Cr", reject_unphysical=True)
    current = NumericalConfig(order=axes["orders"][1], slices=axes["slices"][2], grid=axes["grids"][1])
    history = []
    converged = False
    for cycle in range(args.max_cycles):
        start = current
        statuses = []
        for axis, candidate_name in (("slices", "slices"), ("order", "orders"), ("grid", "grids")):
            selected, passed, record = _scan_axis(study, axis, axes[candidate_name], current, args.tolerance)
            values = asdict(current)
            values[axis] = selected
            current = NumericalConfig(**values)
            record["cycle"] = cycle + 1
            history.append(record)
            statuses.append(passed)
        converged = current == start and all(statuses)
        if current == start:
            break
    anchors = study.spectrum(current)
    status = "converged" if converged else "cycle_limit_reached" if all(statuses) else "candidate_range_insufficient"
    report = {"status": status,
              "signature": signature, "assumptions": assumptions,
              "candidate_axes": {**axes, "tolerance": args.tolerance,
                                 "criterion": "two passing refinements at the upper end, physical powers, and stable three-axis configuration"},
              "recommendation": asdict(current), "history": history, "anchor_spectrum": anchors,
              "interpretation": "Semi-infinite Cr: T_far=0; A_total=1-R; P_sub is substrate absorption; A_moth=1-R-P_sub.",
              "integration": {"torch": torch.__version__, "device": str(device),
                              "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                              "completed_cases": len(study.cases)}}
    report_path = prefix.with_name(prefix.name + "_convergence.json")
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    write_csv(prefix.with_name(prefix.name + "_all_cases.csv"), study.cases.values())
    write_csv(prefix.with_name(prefix.name + "_anchor_spectrum.csv"), anchors)
    if args.run_final_spectrum and converged:
        write_csv(prefix.with_name(prefix.name + "_spectrum.csv"), study.spectrum(current, spectrum_wavelengths))
    elif args.run_final_spectrum:
        print("Final spectrum skipped: convergence criterion was not met.")
    if not args.no_plots:
        from studies.cr_motheye.plot_results import make_figures
        make_figures(report_path)
    print(json.dumps({"status": report["status"], "recommendation": asdict(current)}, indent=2))
    print(f"report: {report_path.resolve()}")
    return 0 if converged else 2


if __name__ == "__main__":
    raise SystemExit(main())
