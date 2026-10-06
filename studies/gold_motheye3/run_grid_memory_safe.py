"""Compare grids at a saved Fourier order with the verified layer-storage policy.

Default: M=30, Nz=140, 700 nm; grids 576,640,704. Reuses the saved
grid=576 optical case. --transforms also audits T_star at tip/middle/base
for each grid, using complex128 SVD. No RCWA source files change.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_SEED = HERE / "results" / "order_700_Nz140_grid576_memory_safe" / "checkpoint.json"


def prepare(args):
    from studies.gold_motheye3 import converge, diagnose_order_700
    from studies.gold_motheye3.show_results import read_checkpoint

    seed = read_checkpoint(args.seed_checkpoint)
    base = seed["plan"]
    if (base["wavelengths_nm"] != [700.]
            or base["solver"].get("layer_storage") != "D6-fixed-geometry-flux-only-release-v1"
            or not seed.get("storage_parity", {}).get("passed")):
        raise ValueError("Requires a 700 nm memory-safe order checkpoint with passed storage parity.")
    # Reuse the existing source/material/case/radius validation.
    identity, layers = diagnose_order_700.prepare(SimpleNamespace(
        checkpoint=args.seed_checkpoint, orders=str(args.order), layers=args.layers,
        wavelength_nm=700., device=args.device))
    grids = list(converge._integers(args.grids))
    if any(n % 2 or n < max(32, 4 * args.order + 4) for n in grids):
        raise ValueError("Use even grids >= max(32, 4*M+4).")
    if identity["grid"] not in grids or len(grids) < 3:
        raise ValueError("Include the seed grid and at least three grid values.")
    plan = copy.deepcopy(base)
    plan["axis"], plan["values"] = "grid", grids
    plan["fixed_numerics"]["order"] = args.order
    plan["solver"]["requested_device"] = args.device
    plan["seed_signature"] = seed["signature"]
    for name in ("run_grid_memory_safe.py", "diagnose_order_700.py", "diagnose_grid_700.py"):
        plan["source_sha256"][f"studies/gold_motheye3/{name}"] = converge._digest(HERE / name)
    plan["transform_diagnostics"] = {"enabled": args.transforms, "layers": identity["layers"],
                                     "svd_cuda_driver": "gesvd",
                                     "scope": "Selected-layer T_star SVD; no all-layer conditioning claim."}
    original_key = converge._key(args.order, 700.)
    row = seed["cases"][original_key]
    reused = {converge._key(identity["grid"], 700.): {
        **copy.deepcopy(row), "axis": "grid", "value": identity["grid"],
        "reused_from": {"signature": seed["signature"], "key": original_key,
                        "runtime_is_original_measurement": True}}}
    return plan, reused, layers, seed


def audit_layer(plan, grid, row, device, torch):
    from rcwa_solver_auto import ASROptions, AutoRCWA, GroupTheoryOptions, Lattice, OutputSpec
    from studies.gold_motheye3.diagnose_grid_700 import audit_map
    from studies.gold_motheye3.diagnose_order_700 import audit_transform
    from studies.shared.gold_dispersion import build_gold_model

    geometry = plan["geometry"]
    simulation = AutoRCWA(
        freq=geometry["period_nm"] / 700.,
        order=[plan["fixed_numerics"]["order"]] * 2, lattice=Lattice.triangular(1.),
        cascade=plan["solver"]["cascade"], outputs=OutputSpec(smatrix_size="half", fields="none"),
        asr=ASROptions(circle_G=geometry["asr_circle_g"], grid=(grid, grid), factorization_rules=True),
        group_theory=GroupTheoryOptions(enabled=True, symmetry="d6", strict=True, polarization="x"),
        verify_cascade=False, dtype=torch.complex128, device=device)
    epsilon = build_gold_model("csv", Path(plan["material"]["path"]))(700.)
    simulation.add_input_layer(eps=1., mu=1.)
    simulation.add_output_layer(eps=epsilon, mu=1.)
    simulation.set_incident_angle(0., 0.)
    radius = row["radius_nm"] / geometry["period_nm"]
    with torch.enable_grad():
        mapping = simulation.build_triangular_circle_asr_mapping(grid, grid, radius)
    with torch.no_grad():
        mask = simulation._periodic_circle_mask(mapping.x, mapping.y, radius)
    return {"layer": row["layer"], "radius_nm": row["radius_nm"],
            "map": audit_map(mapping, mask, radius), "transform": audit_transform(simulation, mapping),
            "completed": True}


def persist(output, saved):
    from studies.gold_motheye3 import converge

    report = converge._persist(output, saved)
    flat = []
    for grid in saved["plan"]["values"]:
        optical = saved["cases"].get(converge._key(grid,700.))
        if optical is None:
            continue
        for _, row in sorted(optical.get("transform_diagnostics", {}).items(), key=lambda p: int(p[0])):
            if not row.get("completed"):
                continue
            flat.append({"grid": grid, "layer": row["layer"], "radius_nm": row["radius_nm"],
                         "saved_R_percent": 100 * optical["reflectance"],
                         "jacobian_det_min": row["map"]["cartesian_jacobian_det_min"],
                         "gold_area_error_pp": row["map"]["gold_area_error_pp"],
                         **row["transform"], "diagnostic_runtime_seconds": row["runtime_seconds"]})
    if flat:
        with (output / "transform_conditions.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
    return report


def summarize(output, saved, report):
    from studies.gold_motheye3.show_results import print_results

    print_results(saved)
    if saved["plan"]["transform_diagnostics"]["enabled"]:
        print("\n grid layer        sigma_min      cond(T_star)       kappa*eps64")
        complete, expected = 0, len(saved["plan"]["values"]) * len(saved["plan"]["transform_diagnostics"]["layers"])
        for optical in sorted(saved["cases"].values(), key=lambda r: r["grid"]):
            for _, row in sorted(optical.get("transform_diagnostics", {}).items(), key=lambda p: int(p[0])):
                if not row.get("completed"):
                    print(f"grid={optical['grid']}|layer={row['layer']}: {row.get('error')}")
                    continue
                complete += 1
                t = row["transform"]
                condition, indicator = t["condition_number_2"], t["condition_times_float64_epsilon"]
                condition_text = "nonfinite" if condition is None else f"{condition:.7g}"
                indicator_text = "-" if indicator is None else f"{indicator:.3g}"
                print(f"{optical['grid']:5d} {row['layer']:5d} {t['smallest_singular_value']:16.6g} "
                      f"{condition_text:>17} {indicator_text:>18}")
        print(f"Selected-layer diagnostics: {complete}/{expected}; these do not certify R/T convergence.")
    print(f"status: {report['status']}")
    print(f"checkpoint: {output / 'checkpoint.json'}")
    print(f"figure: {output / 'convergence.svg'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-checkpoint", type=Path, default=DEFAULT_SEED)
    parser.add_argument("--order", type=int, default=30)
    parser.add_argument("--grids", default="576,640,704")
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--transforms", action="store_true", help="Also audit T_star for selected layers at each grid.")
    parser.add_argument("--layers", default="1,middle,last")
    parser.add_argument("--output-dir", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    args.seed_checkpoint = args.seed_checkpoint.resolve()
    try:
        plan, reused, layers, seed = prepare(args)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))
    from studies.gold_motheye3 import converge

    suffix = "_transform" if args.transforms else ""
    name = f"grid_700_M{args.order}_Nz{plan['fixed_numerics']['slices']}_memory_safe{suffix}"
    output = (args.output_dir or HERE / "results" / name).resolve()
    if args.seed_checkpoint.is_relative_to(output):
        parser.error("Use an output directory separate from the seed checkpoint.")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "checkpoint.json"
    saved = converge._load_checkpoint(path, plan)
    if not path.exists():
        saved["cases"] = reused
        saved["storage_parity_from_seed"] = {"passed": True,
            "reference_key": seed["storage_parity"]["reference_key"],
            "absolute_errors": seed["storage_parity"]["absolute_errors"]}
    converge._write_json(output / "plan.json", plan)
    missing = [g for g in plan["values"] if converge._key(g,700.) not in saved["cases"]]
    print(f"Grid plan: {plan['values']}; M={args.order}, Nz={plan['fixed_numerics']['slices']}; "
          f"{len(missing)} new optical solves; complex128, released D6 layer storage.", flush=True)
    if args.transforms:
        print(f"Additional T_star SVD: {len(plan['values'])*len(layers)} selected-layer cases.", flush=True)
    if args.prepare_only:
        print(f"plan: {output / 'plan.json'}")
        return 0
    if args.report_only:
        if not path.exists():
            parser.error("No saved grid checkpoint exists.")
        summarize(output, saved, persist(output, saved))
        return 0

    import torch
    from studies.gold_motheye3.run_memory_safe import measure

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    environment = {"torch": str(torch.__version__), "cuda": torch.version.cuda, "device_type": device.type,
                   "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
    if seed.get("runtime_environment") != environment or saved.get("runtime_environment") not in (None, environment):
        raise RuntimeError("Use the same runtime environment as the seed calculation.")
    saved["runtime_environment"] = environment
    converge._write_json(path, saved)
    persist(output, saved)
    for grid in plan["values"]:
        key = converge._key(grid,700.)
        if key not in saved["cases"]:
            numbers = converge._numerical(plan, grid)
            print(f"solve grid={grid}, M={args.order}, Nz={numbers['slices']}, wavelength=700 nm", flush=True)
            result = measure(700., numbers, plan, device, torch)
            saved["cases"][key] = {**result, "axis": "grid", "value": grid}
            converge._write_json(path, saved)
            persist(output, saved)
            peak = result["peak_cuda_allocated_bytes"]
            if peak is not None:
                print(f"peak CUDA tensor memory: {peak/2**30:.3f} GiB; wall time: {result['wall_seconds']:.2f} s", flush=True)
        if args.transforms:
            diagnostics = saved["cases"][key].setdefault("transform_diagnostics", {})
            for row in layers:
                if diagnostics.get(str(row["layer"]), {}).get("completed"):
                    continue
                print(f"audit grid={grid}|layer={row['layer']}, radius={row['radius_nm']:.4f} nm", flush=True)
                started = time.perf_counter()
                try:
                    measured = audit_layer(plan, grid, row, device, torch)
                except Exception as error:
                    measured = {"layer": row["layer"], "radius_nm": row["radius_nm"],
                                "completed": False, "error": str(error), "traceback": traceback.format_exc()}
                    print(f"Diagnostic failed: {error}", flush=True)
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    torch.cuda.empty_cache()
                measured["runtime_seconds"] = time.perf_counter() - started
                diagnostics[str(row["layer"])] = measured
                converge._write_json(path, saved)
                persist(output, saved)
    report = persist(output, saved)
    summarize(output, saved, report)
    failed = args.transforms and any(not saved["cases"][converge._key(g,700.)]
        .get("transform_diagnostics", {}).get(str(r["layer"]), {}).get("completed") for g in plan["values"] for r in layers)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
