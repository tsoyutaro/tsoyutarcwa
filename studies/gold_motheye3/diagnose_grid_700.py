"""Audit production ASR maps and optional modal transforms against saved grids.

Defaults: 700 nm, grids 448/512/576, tip/middle/base slices of the saved
100-slice profile. No optical eigensolve or full-stack R/T solve is performed.
--transforms adds the SVD of T_star = E^H T E used before D6 source reduction.
These diagnostics can locate numerical issues; they do not certify convergence.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_CHECKPOINT = HERE / "results" / "grid_M16_Nz100" / "checkpoint.json"
METRICS = ("reflectance", "power_into_substrate", "motheye_absorptance")


def prepare(args):
    from studies.gold_motheye3 import converge

    saved = json.loads(args.checkpoint.read_text(encoding="utf-8"))
    plan = saved["plan"]
    if saved.get("signature") != converge._signature(plan):
        raise ValueError("The grid checkpoint signature is invalid.")
    if (plan["axis"] != "grid" or plan["geometry"]["lattice"] != "triangular"
            or plan["solver"]["dtype"] != "complex128"
            or plan["solver"]["symmetry_reduction"] != "d6-source"):
        raise ValueError("Requires the triangular complex128 D6-source grid sweep.")
    current = converge._source_hashes()
    different = sorted(name for name in set(current) | set(plan["source_sha256"])
                       if current.get(name) != plan["source_sha256"].get(name))
    if different:
        raise ValueError("RCWA/study sources differ from the saved sweep: "
                         + ", ".join(different))
    csv_path = Path(plan["material"]["path"])
    if converge._digest(csv_path) != plan["material"]["sha256"]:
        raise ValueError("The measured Au CSV differs from the saved sweep.")
    grids = sorted(set(int(item.strip()) for item in args.grids.split(",")))
    count = int(plan["fixed_numerics"]["slices"])
    aliases = {"middle": (count + 1) // 2, "last": count}
    layers = (list(range(1, count + 1)) if args.layers.strip() == "all" else
              sorted(set(aliases[item.strip()] if item.strip() in aliases
                         else int(item.strip()) for item in args.layers.split(","))))
    order = int(plan["fixed_numerics"]["order"])
    if not grids or any(n % 2 or n < max(32, 4 * order + 4) for n in grids):
        raise ValueError("Use even grids >= max(32, 4*M+4).")
    if not layers or min(layers) < 1 or max(layers) > count:
        raise ValueError("Layer indices must be between 1 and the saved slice count.")
    geometry = plan["geometry"]
    if not all(math.isfinite(float(geometry[key])) for key in
               ("period_nm", "tip_radius_nm", "base_radius_nm", "profile_power")):
        raise ValueError("Nonfinite profile settings.")
    rows = []
    for layer in layers:
        coordinate = (layer - .5) / count
        radius = geometry["tip_radius_nm"] + (
            geometry["base_radius_nm"] - geometry["tip_radius_nm"]
        ) * coordinate ** geometry["profile_power"]
        if not 0 < radius < geometry["period_nm"] / 2:
            raise ValueError("This audit requires nonoverlapping circles with 0 < r < period/2.")
        for grid in grids:
            reference = saved["cases"].get(converge._key(grid, args.wavelength_nm))
            if reference is None:
                raise ValueError(f"A saved result is required: grid={grid}, "
                                 f"wavelength={args.wavelength_nm:g} nm.")
            rows.append({"grid": grid, "layer": layer, "radius_nm": radius,
                         "saved_result": {name: float(reference[name]) for name in METRICS}})
    identity = {"grid_checkpoint_signature": saved["signature"],
                "diagnostic_source_sha256": converge._digest(Path(__file__)),
                "grids": grids, "layers": layers, "order": order, "slices": count,
                "wavelength_nm": args.wavelength_nm, "geometry": geometry,
                "transforms": args.transforms, "requested_device": args.device,
                "symmetry_tolerance": 1e-8, "source_sha256": current,
                "material": plan["material"], "solver": plan["solver"]}
    return identity, rows


def audit_map(mapping, mask, radius):
    import numpy as np

    names = ("u", "v", "x", "y", "x_u", "x_v", "y_u", "y_v", "det_j")
    arrays = {name: getattr(mapping, name).detach().cpu().numpy() for name in names}
    inside = mask.detach().cpu().numpy().astype(bool)
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise RuntimeError("Nonfinite coordinates or map derivatives.")
    n = inside.shape[0]
    sine = math.sqrt(3) / 2
    u, v = np.meshgrid(arrays["u"], arrays["v"], indexing="ij")
    displacement = np.stack([arrays["x"] - (u + .5*v), arrays["y"] - sine*v])
    primitive = np.stack([np.stack([arrays["x_u"], arrays["x_v"]]),
                          np.stack([arrays["y_u"], arrays["y_v"]])])
    jacobian = np.einsum("abij,bc->acij", primitive,
                         np.array([[1., -.5/sine], [0., 1./sine]]))
    weight = arrays["det_j"] / sine
    analytic_area = math.pi * radius**2 / sine
    measured_area = float(np.mean(inside * weight))
    minimum_distance = np.full_like(arrays["x"], np.inf)
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            distance = (arrays["x"] - (.75+a+.5*b))**2 + (
                arrays["y"] - (.5*sine+sine*b))**2
            minimum_distance = np.minimum(minimum_distance, distance)
    # Near-boundary samples can change material labels at floating-point ties.
    boundary = np.abs(minimum_distance - radius**2) <= 256*np.finfo(float).eps
    result = {"cartesian_jacobian_det_min": float(weight.min()),
              "cartesian_jacobian_det_max": float(weight.max()),
              "nonpositive_jacobian_samples": int(np.count_nonzero(weight <= 0)),
              "unit_cell_area_ratio": float(weight.mean()),
              "analytic_gold_area_fraction": analytic_area,
              "quadrature_gold_area_fraction": measured_area,
              "gold_area_error_pp": 100 * (measured_area - analytic_area),
              "near_interface_samples": int(boundary.sum())}
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    transformations = [
        ("rotation60", (-j) % n, (i+j-n//2) % n,
         np.array([[.5, -sine], [sine, .5]])),
        ("reflection_x", (i+j-n//2) % n, (-j) % n,
         np.array([[1., 0.], [0., -1.]]))]
    peak = max(float(np.linalg.norm(jacobian, axis=(0, 1)).max()), 1.)
    norm = max(float(np.linalg.norm(jacobian)), np.finfo(float).tiny)
    for name, ii, jj, rotation in transformations:
        expected_map = np.einsum("ab,bij->aij", rotation, displacement)
        expected_j = np.einsum("ab,bcij,dc->adij", rotation, jacobian, rotation)
        difference = jacobian[:, :, ii, jj] - expected_j
        mismatch = inside[ii, jj] != inside
        near_either_interface = boundary | boundary[ii, jj]
        result[name] = {
            "map_max_absolute_error_period_units": float(
                np.linalg.norm(displacement[:, ii, jj] - expected_map, axis=0).max()),
            "jacobian_relative_max_error": float(
                np.linalg.norm(difference, axis=(0, 1)).max()) / peak,
            "jacobian_relative_frobenius_error": float(np.linalg.norm(difference)) / norm,
            "material_mask_mismatched_samples": int(mismatch.sum()),
            "material_mask_mismatch_fraction": float(mismatch.mean()),
            "material_mask_mismatches_away_from_interface": int(
                np.count_nonzero(mismatch & ~near_either_interface))}
    return result


def audit_transform(simulation, mapping):
    import torch

    with torch.no_grad():
        transform, _ = simulation._build_circle_conversion_matrices(mapping)
        embedding, _, _, _, _ = simulation._triangular_star_operators()
        projected = embedding.mH @ transform @ embedding
        if not bool(torch.isfinite(projected).all()):
            raise RuntimeError("Nonfinite triangular-star conversion matrix.")
        singular = torch.linalg.svdvals(projected)
        maximum, minimum = float(singular.max().cpu()), float(singular.min().cpu())
    condition = maximum / minimum if minimum > 0 else None
    if condition is not None and not math.isfinite(condition):
        condition = None
    return {"star_dimension": int(projected.shape[0]),
            "largest_singular_value": maximum, "smallest_singular_value": minimum,
            "condition_number_2": condition,
            "condition_number_is_nonfinite": condition is None,
            "condition_times_float64_epsilon": (
                condition * torch.finfo(torch.float64).eps if condition is not None else None)}


def case_key(row):
    return f"grid={row['grid']}|layer={row['layer']}"


def persist(output, document, rows):
    from studies.gold_motheye3.converge import _write_json

    _write_json(output / "diagnostics.json", document)
    _write_json(output / "plan.json", {"identity": document["identity"], "cases": rows,
                "remaining_cases": [case_key(row) for row in rows
                                    if not document["cases"].get(case_key(row), {}).get("completed")]})


def summarize(output, document):
    wavelength = document["identity"]["wavelength_nm"]
    print(f"\nSaved {wavelength:g} nm optical results (percent; these are not recalculated):")
    by_grid = {row["grid"]: row["saved_result"] for row in document["planned_cases"]}
    print(" grid         R       P_sub    A_pillar")
    for grid, values in sorted(by_grid.items()):
        print(f"{grid:5d} " + " ".join(f"{100*values[name]:11.6f}" for name in METRICS))
    print("\nASR diagnostics; area error in percentage points; indices start at the tip:")
    print(" grid layer radius_nm       J_min   area_error_pp  rot_J_rel   mask_flips     cond(T_star)")
    flat = []
    for row in document["planned_cases"]:
        case = document["cases"].get(case_key(row), {})
        if not case.get("completed"):
            print(f"{case_key(row)}: {case.get('error', 'not calculated')}")
            continue
        metrics = case["map"]
        transform = case.get("transform", {})
        condition = transform.get("condition_number_2")
        condition_text = ("not requested" if not transform else "nonfinite" if condition is None
                          else f"{condition:.6g}")
        rotation = metrics["rotation60"]
        print(f"{row['grid']:5d} {row['layer']:5d} {row['radius_nm']:9.4f} "
              f"{metrics['cartesian_jacobian_det_min']:11.5g} {metrics['gold_area_error_pp']:15.6g} "
              f"{rotation['jacobian_relative_max_error']:10.3g} "
              f"{rotation['material_mask_mismatched_samples']:12d} {condition_text:>16}")
        flat.append({"grid": row["grid"], "layer": row["layer"], "radius_nm": row["radius_nm"],
                     "jacobian_det_min": metrics["cartesian_jacobian_det_min"],
                     "nonpositive_jacobian_samples": metrics["nonpositive_jacobian_samples"],
                     "unit_cell_area_ratio": metrics["unit_cell_area_ratio"],
                     "gold_area_error_pp": metrics["gold_area_error_pp"],
                     "rotation_jacobian_relative_max_error": rotation["jacobian_relative_max_error"],
                     "rotation_mask_mismatched_samples": rotation["material_mask_mismatched_samples"],
                     "condition_number_2": condition, "runtime_seconds": case["runtime_seconds"]})
    if flat:
        with (output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
    print("These are numerical diagnostics, not an R/T convergence or physical validation pass.")
    print(f"report: {output / 'diagnostics.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--grids", default="448,512,576")
    parser.add_argument("--layers", default="1,middle,last",
                        help="One-based indices, middle,last, or all.")
    parser.add_argument("--wavelength-nm", type=float, default=700.)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--transforms", action="store_true", help="Add modal conversion and SVD.")
    parser.add_argument("--output-dir", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.wavelength_nm) or args.wavelength_nm <= 0:
        parser.error("The wavelength must be finite and positive.")
    label = f"{args.wavelength_nm:g}".replace(".", "p")
    suffix = "transform" if args.transforms else "map"
    output = (args.output_dir or HERE / "results" / f"grid_diagnostics_{label}_{suffix}").resolve()
    args.checkpoint = args.checkpoint.resolve()
    if output == args.checkpoint.parent or args.checkpoint.is_relative_to(output):
        parser.error("Use a diagnostic directory separate from the optical sweep.")
    path = output / "diagnostics.json"
    if args.report_only:
        summarize(output, json.loads(path.read_text(encoding="utf-8")))
        return 0
    try:
        identity, rows = prepare(args)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    output.mkdir(parents=True, exist_ok=True)
    document = (json.loads(path.read_text(encoding="utf-8")) if path.is_file() else
                {"identity": identity, "planned_cases": rows, "cases": {},
                 "scope": "Selected production maps and optional T_star SVD; no optical solve."})
    if document.get("identity") != identity:
        parser.error("Diagnostic settings/source differ. Use another --output-dir.")
    missing = [row for row in rows if not document["cases"].get(case_key(row), {}).get("completed")]
    persist(output, document, rows)
    print(f"Map audit: {len(rows)} cases; {len(missing)} remaining; "
          f"M={identity['order']}, Nz={identity['slices']}; transforms={args.transforms}", flush=True)
    if args.prepare_only:
        print(f"plan: {output / 'plan.json'}")
        return 0
    if not missing:
        summarize(output, document)
        return 0

    import torch
    from rcwa_solver_auto import ASROptions, AutoRCWA, GroupTheoryOptions, Lattice, OutputSpec
    from studies.gold_motheye.converge import GeometryConfig, _slice_radius_nm
    from studies.shared.gold_dispersion import build_gold_model

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    document["runtime_environment"] = {"torch": torch.__version__, "device": str(device),
                                        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
    geometry = GeometryConfig(**identity["geometry"])
    gold = build_gold_model("csv", Path(identity["material"]["path"]))
    epsilon = gold(args.wavelength_nm)
    for row in missing:
        start = time.perf_counter()
        print(f"audit {case_key(row)}, radius={row['radius_nm']:.4f} nm", flush=True)
        case = {**row, "completed": False}
        simulation = mapping = mask = None
        try:
            actual_radius = _slice_radius_nm(geometry, row["layer"] - 1, identity["slices"])
            if not math.isclose(actual_radius, row["radius_nm"], rel_tol=1e-12, abs_tol=1e-12):
                raise RuntimeError("The shared model profile differs from the diagnostic plan.")
            simulation = AutoRCWA(
                freq=geometry.period_nm / args.wavelength_nm,
                order=[identity["order"], identity["order"]], lattice=Lattice.triangular(1.),
                cascade=identity["solver"]["cascade"], outputs=OutputSpec(smatrix_size="half", fields="none"),
                asr=ASROptions(circle_G=geometry.asr_circle_g, grid=(row["grid"], row["grid"]),
                               factorization_rules=True),
                group_theory=GroupTheoryOptions(enabled=True, symmetry="d6", strict=True, polarization="x"),
                verify_cascade=False, dtype=torch.complex128, device=device)
            simulation.add_input_layer(eps=1., mu=1.)
            simulation.add_output_layer(eps=epsilon, mu=1.)
            simulation.set_incident_angle(0., 0.)
            radius = actual_radius / geometry.period_nm
            # The production map differentiates coordinates internally.
            with torch.enable_grad():
                mapping = simulation.build_triangular_circle_asr_mapping(row["grid"], row["grid"], radius)
            with torch.no_grad():
                mask = simulation._periodic_circle_mask(mapping.x, mapping.y, radius)
            case["map"] = audit_map(mapping, mask, radius)
            if args.transforms:
                case["transform"] = audit_transform(simulation, mapping)
            case["completed"] = True
        except Exception as error:
            case["error"] = str(error)
            case["traceback"] = traceback.format_exc()
            print(f"FAILED {case_key(row)}: {error}", flush=True)
        finally:
            simulation = mapping = mask = None
            gc.collect()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            case["runtime_seconds"] = time.perf_counter() - start
            document["cases"][case_key(row)] = case
            persist(output, document, rows)
    summarize(output, document)
    return int(any(not document["cases"].get(case_key(row), {}).get("completed") for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
