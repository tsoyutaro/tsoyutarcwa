"""Focused forward/shape-derivative regression for the 2026-10-12 ASR repair.

This checks the previously unstable square M=16 cases before the longer
four-case convergence study. Passing is not a full-spectrum convergence proof.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

from studies.dielectric_motheye_n25.compare import parse_wavelengths, source_hashes
from studies.dielectric_motheye_n25.model import (Geometry, Numerical, Study,
    adjacent_error, physical, release, solve)
from studies.shared.convergence_io import content_hash, signature_for, write_json

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda")
    p.add_argument("--order", type=int, default=16)
    p.add_argument("--slices", type=int, default=100)
    p.add_argument("--grids", default="384,512")
    p.add_argument("--wavelengths", default="550,700")
    p.add_argument("--tolerance", type=float, default=.001)
    p.add_argument("--energy-tolerance", type=float, default=1e-6)
    p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--output-root", type=Path,
                   default=ROOT / "studies/dielectric_motheye_n25/results/repair_validation_v3")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    try:
        grids = sorted(set(int(v) for v in args.grids.split(",")))
        wavelengths = parse_wavelengths(args.wavelengths)
        if len(grids) < 2 or min(grids) < max(32, 4 * args.order + 4):
            raise ValueError("Use at least two grids, each >= max(32,4*M+4).")
        if min(args.order, args.slices, args.cpu_threads) <= 0:
            raise ValueError("Order, slices and CPU threads must be positive.")
        if any(not math.isfinite(v) or v <= 0 for v in (args.tolerance, args.energy_tolerance)):
            raise ValueError("Tolerances must be positive and finite.")
        if args.device not in {"cuda", "cpu", "auto"} and not (
            args.device.startswith("cuda:") and args.device[5:].isdigit()):
            raise ValueError("Device: cuda, cuda:<index>, cpu, auto")
    except ValueError as error:
        p.error(str(error))
    g = Geometry()
    output = args.output_root.resolve()
    assumptions = {"geometry": asdict(g), "asr_profile": "local-radial",
        "fixed_nominal_interfaces": True, "symmetry": True, "dtype": "complex128",
        "source_sha256": source_hashes(), "validation_source_sha256": content_hash(Path(__file__))}
    plan = {"assumptions": assumptions, "order": args.order, "slices": args.slices,
        "grids": grids, "wavelengths_nm": wavelengths, "requested_device": args.device,
        "tolerance": args.tolerance, "energy_tolerance": args.energy_tolerance,
        "gradient_probe": {"order": 2, "slices": 4, "grid": 160, "wavelength_nm": 550.,
                           "parameter": "base_radius_nm", "fd_steps_nm": [.02, .01]},
        "scope": "critical square forward cases and small two-lattice discrete radius derivatives; "
                 "not full-spectrum or mesh-converged optimization validation"}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available() or (device.index or 0) >= torch.cuda.device_count():
            p.error("Explicit CUDA request cannot be satisfied; no CPU fallback.")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(args.cpu_threads)
    integration = {"device": str(device), "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda, "cpu_threads": args.cpu_threads,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
    signature = signature_for(assumptions)
    studies = {method: Study(geometry=g, method=method, lattice="square", device=device,
        use_symmetry=True, signature=signature_for({"physical": signature, "method": method}),
        checkpoint=output / method / "checkpoint.json") for method in ("asr", "nv")}
    plan["integration"] = integration
    write_json(output / "plan.json", plan)
    result = {"plan": plan, "status": "running", "forward_checks": [], "gradient_checks": []}
    report_path = output / "validation.json"
    write_json(report_path, result)
    try:
        spectra = {}
        for grid in grids:
            spectra[grid] = studies["asr"].spectrum(Numerical(args.order, args.slices, grid), wavelengths)
            release(device)
        reference = studies["nv"].spectrum(Numerical(args.order, args.slices, grids[-1]), wavelengths)
        for label, first, second in (
            [(f"quadrature Q{q} vs Q{grids[-1]}", spectra[q], spectra[grids[-1]]) for q in grids[:-1]]
            + [("ASR vs NV at finest grid", spectra[grids[-1]], reference)]
        ):
            error, per_metric = adjacent_error(first, second)
            passive = all(physical(row, args.energy_tolerance) for row in first + second)
            result["forward_checks"].append({"check": label, "max_abs_change": error,
                "per_metric": per_metric, "physical": passive,
                "passed": passive and error <= args.tolerance})
        result["asr_spectra"] = {str(q): rows for q, rows in spectra.items()}
        result["nv_spectrum"] = reference
        write_json(report_path, result)
        release(device)
        for lattice in ("hexagonal", "square"):
            print(f"{lattice}/asr: fixed-interface radius derivative probe", flush=True)
            base = torch.tensor(g.base_radius_nm, dtype=torch.float64, device=device, requires_grad=True)
            nominal, value = solve(550., Numerical(2, 4, 160), g,
                method="asr", lattice=lattice, device=device, parameters={"base_radius_nm": base})
            derivative = float(torch.autograd.grad(value, base)[0].detach().cpu())
            del value, base
            release(device)
            differences, rows = [], [nominal]
            for step in (.02, .01):
                plus, _ = solve(550., Numerical(2, 4, 160), replace(g, base_radius_nm=g.base_radius_nm+step),
                    method="asr", lattice=lattice, device=device, asr_reference_geometry=g)
                minus, _ = solve(550., Numerical(2, 4, 160), replace(g, base_radius_nm=g.base_radius_nm-step),
                    method="asr", lattice=lattice, device=device, asr_reference_geometry=g)
                rows.extend([plus, minus])
                differences.append((plus["reflectance"] - minus["reflectance"]) / (2 * step))
            passive = all(physical(row, args.energy_tolerance) for row in rows)
            passed = passive and math.isfinite(derivative) and all(
                abs(derivative-fd) <= 1e-7 + .001 * max(abs(derivative), abs(fd)) for fd in differences)
            result["gradient_checks"].append({"lattice": lattice, "autograd_per_nm": derivative,
                "finite_differences_per_nm": differences, "physical": passive, "passed": passed})
            write_json(report_path, result)
            release(device)
        result["status"] = "passed" if all(c["passed"] for c in
            result["forward_checks"] + result["gradient_checks"]) else "failed"
    except (Exception, KeyboardInterrupt) as error:
        result.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "calculation_failed",
                      error=f"{type(error).__name__}: {error}")
    finally:
        write_json(report_path, result)
        release(device)
    print(json.dumps({key: result.get(key) for key in
        ("status", "forward_checks", "gradient_checks", "error")}, indent=2))
    print(f"Report: {report_path}")
    return 0 if result["status"] == "passed" else 1 if result["status"] == "calculation_failed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
