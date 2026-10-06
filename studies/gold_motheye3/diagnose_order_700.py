"""Audit T_star conditioning versus saved Fourier orders, without an R/T solve.

Default: M=16,18,20,22,24,26; tip/middle/base of the saved Nz=140 profile.
Uses production ASR mappings and T_star = E^H T E before D6 source reduction.
Records singular values, condition numbers and map diagnostics separately.
CUDA SVD uses gesvd. An audit is not an optical convergence certificate.
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
DEFAULT_CHECKPOINT = (HERE / "results" / "order_700_Nz140_grid576_memory_safe"
                      / "checkpoint.json")


def prepare(args):
    from studies.gold_motheye3 import converge
    from studies.gold_motheye3.show_results import key, read_checkpoint

    saved = read_checkpoint(args.checkpoint)
    plan = saved["plan"]
    if (plan["axis"] != "order" or plan["geometry"]["lattice"] != "triangular"
            or plan["solver"]["dtype"] != "complex128"
            or plan["solver"]["symmetry_reduction"] != "d6-source"):
        raise ValueError("Requires a triangular complex128 D6-source order checkpoint.")
    current = converge._source_hashes()
    driver = "studies/gold_motheye3/run_memory_safe.py"
    if driver in plan["source_sha256"]:
        current[driver] = converge._digest(ROOT / driver)
    different = sorted(name for name in set(current) | set(plan["source_sha256"])
                       if current.get(name) != plan["source_sha256"].get(name))
    if different:
        raise ValueError("Sources differ from the saved calculation: " + ", ".join(different))
    if converge._digest(Path(plan["material"]["path"])) != plan["material"]["sha256"]:
        raise ValueError("The measured Au CSV differs from the saved calculation.")
    orders = list(converge._integers(args.orders))
    count, grid = int(plan["fixed_numerics"]["slices"]), int(plan["fixed_numerics"]["grid"])
    aliases = {"middle": (count + 1) // 2, "last": count}
    layers = (list(range(1, count + 1)) if args.layers.strip() == "all" else
              sorted(set(aliases[item.strip()] if item.strip() in aliases else int(item.strip())
                         for item in args.layers.split(","))))
    if not layers or min(layers) < 1 or max(layers) > count:
        raise ValueError("Layer indices must be between 1 and the saved slice count.")
    if grid % 2 or grid < max(32, 4 * max(orders) + 4):
        raise ValueError("Requires an even grid >= max(32, 4*M+4).")
    geometry = plan["geometry"]
    if (not all(math.isfinite(float(geometry[name])) for name in
                ("period_nm", "height_nm", "tip_radius_nm", "base_radius_nm", "profile_power"))
            or geometry["period_nm"] <= 0 or geometry["profile_power"] <= 0):
        raise ValueError("Invalid physical geometry.")
    rows = []
    for order in orders:
        reference = saved["cases"].get(key(order, args.wavelength_nm))
        if reference is None:
            raise ValueError(f"No saved optical case at M={order}, {args.wavelength_nm:g} nm.")
        for layer in layers:
            radius = geometry["tip_radius_nm"] + (
                geometry["base_radius_nm"] - geometry["tip_radius_nm"]
            ) * ((layer - .5) / count) ** geometry["profile_power"]
            if not 0 < radius < geometry["period_nm"] / 2:
                raise ValueError("Requires nonoverlapping circles with 0 < r < period/2.")
            rows.append({"order": order, "layer": layer, "radius_nm": radius,
                         "saved_reflectance": float(reference["reflectance"]),
                         "saved_power_into_substrate": float(reference["power_into_substrate"]),
                         "saved_motheye_absorptance": float(reference["motheye_absorptance"])})
    identity = {"order_checkpoint_signature": saved["signature"],
                "diagnostic_source_sha256": converge._digest(Path(__file__)),
                "map_audit_source_sha256": converge._digest(HERE / "diagnose_grid_700.py"),
                "orders": orders, "layers": layers, "slices": count, "grid": grid,
                "wavelength_nm": args.wavelength_nm, "geometry": geometry,
                "source_sha256": current, "material": plan["material"], "solver": plan["solver"],
                "requested_device": args.device, "svd_cuda_driver": "gesvd",
                "transform_scope": "T_star = E^H T E before D6 source reduction"}
    return identity, rows


def singular_summary(largest, smallest, dimension):
    if not math.isfinite(largest) or not math.isfinite(smallest) or smallest < 0 or largest < smallest:
        raise ValueError("Invalid singular values.")
    condition = largest / smallest if smallest > 0 else None
    if condition is not None and not math.isfinite(condition):
        condition = None
    return {"star_dimension": dimension, "largest_singular_value": largest,
            "smallest_singular_value": smallest, "condition_number_2": condition,
            "condition_number_is_nonfinite": condition is None,
            "condition_times_float64_epsilon": condition * math.ulp(1.) if condition is not None else None}


def audit_transform(simulation, mapping):
    import torch

    with torch.no_grad():
        transform, _ = simulation._build_circle_conversion_matrices(mapping)
        embedding, _, _, _, _ = simulation._triangular_star_operators()
        projected = embedding.mH @ transform @ embedding
        if not bool(torch.isfinite(projected).all()):
            raise RuntimeError("Nonfinite T_star conversion matrix.")
        singular = (torch.linalg.svdvals(projected, driver="gesvd") if projected.is_cuda else
                    torch.linalg.svdvals(projected))
        return singular_summary(float(singular.max().cpu()), float(singular.min().cpu()),
                                int(projected.shape[0]))


def case_key(row):
    return f"M={row['order']}|layer={row['layer']}"


def persist(output, document):
    from studies.gold_motheye3.converge import _write_json

    flat, by_order = [], {}
    for row in document["planned_cases"]:
        case = document["cases"].get(case_key(row), {})
        if not case.get("completed"):
            continue
        mapping, transform = case["map"], case["transform"]
        record = {"order": row["order"], "layer": row["layer"], "radius_nm": row["radius_nm"],
                  "saved_R_percent": 100 * row["saved_reflectance"],
                  "jacobian_det_min": mapping["cartesian_jacobian_det_min"],
                  "gold_area_error_pp": mapping["gold_area_error_pp"],
                  "rotation_mask_mismatched_samples": mapping["rotation60"]["material_mask_mismatched_samples"],
                  **transform, "runtime_seconds": case["runtime_seconds"]}
        flat.append(record)
        by_order.setdefault(row["order"], []).append(record)
    summaries = []
    for order in document["identity"]["orders"]:
        rows = by_order.get(order, [])
        finite = [r for r in rows if r["condition_number_2"] is not None]
        maximum = max(finite, key=lambda r: r["condition_number_2"]) if finite else None
        summaries.append({"order": order, "completed_layer_count": len(rows),
                          "selected_layer_count": len(document["identity"]["layers"]),
                          "maximum_finite_condition_over_selected_layers": maximum["condition_number_2"] if maximum else None,
                          "layer_of_finite_maximum": maximum["layer"] if maximum else None,
                          "any_nonfinite_condition": any(r["condition_number_is_nonfinite"] for r in rows)})
    document["order_summary"] = summaries
    _write_json(output / "diagnostics.json", document)
    _write_json(output / "plan.json", {"identity": document["identity"],
                "planned_cases": document["planned_cases"],
                "remaining_cases": [case_key(r) for r in document["planned_cases"]
                    if not document["cases"].get(case_key(r), {}).get("completed")]})
    if flat:
        with (output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)


def summarize(output, document):
    print("\n M layer radius_nm        J_min       sigma_min       cond(T_star)        kappa*eps64")
    for row in document["planned_cases"]:
        case = document["cases"].get(case_key(row), {})
        if not case.get("completed"):
            print(f"{case_key(row)}: {case.get('error', 'not calculated')}")
            continue
        transform = case["transform"]
        condition, indicator = transform["condition_number_2"], transform["condition_times_float64_epsilon"]
        condition_text = "nonfinite" if condition is None else f"{condition:.7g}"
        indicator_text = "-" if indicator is None else f"{indicator:.3g}"
        print(f"{row['order']:2d} {row['layer']:5d} {row['radius_nm']:9.4f} "
              f"{case['map']['cartesian_jacobian_det_min']:12.5g} "
              f"{transform['smallest_singular_value']:15.6g} {condition_text:>18} {indicator_text:>18}")
    print("Conditioning diagnostics only; selected layers do not certify all layers or R/T convergence.")
    print(f"report: {output / 'diagnostics.json'}")
    print(f"table: {output / 'summary.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--orders", default="16,18,20,22,24,26")
    parser.add_argument("--layers", default="1,middle,last", help="One-based indices, middle,last, or all.")
    parser.add_argument("--wavelength-nm", type=float, default=700.)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--output-dir", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.wavelength_nm) or args.wavelength_nm <= 0:
        parser.error("The wavelength must be finite and positive.")
    args.checkpoint = args.checkpoint.resolve()
    output = (args.output_dir or HERE / "results" / "order_transform_diagnostics_700_Nz140_grid576").resolve()
    if args.checkpoint.is_relative_to(output):
        parser.error("Use a diagnostic directory separate from the optical checkpoint.")
    path = output / "diagnostics.json"
    if args.report_only:
        summarize(output, json.loads(path.read_text(encoding="utf-8")))
        return 0
    try:
        identity, rows = prepare(args)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))
    output.mkdir(parents=True, exist_ok=True)
    document = (json.loads(path.read_text(encoding="utf-8")) if path.exists() else
                {"identity": identity, "planned_cases": rows, "cases": {},
                 "scope": "Selected ASR maps and T_star SVD; no optical solve."})
    if document.get("identity") != identity:
        parser.error("Diagnostic settings/source differ. Use another --output-dir.")
    missing = [r for r in rows if not document["cases"].get(case_key(r), {}).get("completed")]
    persist(output, document)
    print(f"Transform audit: {len(rows)} cases; {len(missing)} remaining; "
          f"Nz={identity['slices']}, grid={identity['grid']}; layers={identity['layers']}", flush=True)
    if args.prepare_only:
        print(f"plan: {output / 'plan.json'}")
        return 0
    if not missing:
        summarize(output, document)
        return 0

    import torch
    from rcwa_solver_auto import ASROptions, AutoRCWA, GroupTheoryOptions, Lattice, OutputSpec
    from studies.gold_motheye.converge import GeometryConfig, _slice_radius_nm
    from studies.gold_motheye3.diagnose_grid_700 import audit_map
    from studies.shared.gold_dispersion import build_gold_model

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    environment = {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                   "device_type": device.type,
                   "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
    if document.get("runtime_environment") not in (None, environment):
        raise RuntimeError("Diagnostic runtime environment differs. Use another --output-dir.")
    document["runtime_environment"] = environment
    geometry = GeometryConfig(**identity["geometry"])
    epsilon = build_gold_model("csv", Path(identity["material"]["path"]))(args.wavelength_nm)
    for row in missing:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        print(f"audit {case_key(row)}, radius={row['radius_nm']:.4f} nm", flush=True)
        simulation = mapping = mask = None
        case = {**row, "completed": False}
        try:
            radius_nm = _slice_radius_nm(geometry, row["layer"] - 1, identity["slices"])
            if not math.isclose(radius_nm, row["radius_nm"], rel_tol=1e-12, abs_tol=1e-12):
                raise RuntimeError("The shared model radius differs from the diagnostic plan.")
            simulation = AutoRCWA(
                freq=geometry.period_nm / args.wavelength_nm, order=[row["order"], row["order"]],
                lattice=Lattice.triangular(1.), cascade=identity["solver"]["cascade"],
                outputs=OutputSpec(smatrix_size="half", fields="none"),
                asr=ASROptions(circle_G=geometry.asr_circle_g,
                               grid=(identity["grid"], identity["grid"]), factorization_rules=True),
                group_theory=GroupTheoryOptions(enabled=True, symmetry="d6", strict=True, polarization="x"),
                verify_cascade=False, dtype=torch.complex128, device=device)
            simulation.add_input_layer(eps=1., mu=1.)
            simulation.add_output_layer(eps=epsilon, mu=1.)
            simulation.set_incident_angle(0., 0.)
            radius = radius_nm / geometry.period_nm
            with torch.enable_grad():
                mapping = simulation.build_triangular_circle_asr_mapping(identity["grid"], identity["grid"], radius)
            with torch.no_grad():
                mask = simulation._periodic_circle_mask(mapping.x, mapping.y, radius)
            case["map"] = audit_map(mapping, mask, radius)
            case["transform"] = audit_transform(simulation, mapping)
            case["completed"] = True
        except Exception as error:
            case["error"], case["traceback"] = str(error), traceback.format_exc()
            print(f"FAILED {case_key(row)}: {error}", flush=True)
        finally:
            simulation = mapping = mask = None
            gc.collect()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
            case["runtime_seconds"] = time.perf_counter() - started
            document["cases"][case_key(row)] = case
            persist(output, document)
    summarize(output, document)
    return int(any(not document["cases"].get(case_key(row), {}).get("completed") for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
