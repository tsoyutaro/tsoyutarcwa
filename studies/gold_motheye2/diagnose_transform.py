"""Measure the ASR transform condition number for a fixed-slice order sweep.

The measured matrix is T_star = E^H T E, the triangular-star transform
inverted by the D6-source solver before its E1 source-row reduction. Its
2-norm condition number is diagnostic; it is not a convergence test by itself.
The singular-value decompositions are deliberately kept out of the timing
reported by converge.py.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_GOLD_CSV = HERE / "data" / "au_measured_nk.csv"
COLUMNS = (
    "order", "wavelength_nm", "slices", "grid", "layer_index",
    "radius_nm", "transform_star_dimension", "condition_number_2",
    "log10_condition_number_2", "reflectance", "solve_seconds",
    "condition_seconds",
)
SUMMARY_COLUMNS = (
    "order", "wavelength_nm", "slices", "grid", "reflectance",
    "max_condition_number_2", "layer_index_at_max", "solve_seconds",
    "condition_seconds",
)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (int(row["order"]), int(row["layer_index"]))))
    temporary.replace(path)


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    grouped: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(int(row["order"]), []).append(row)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for order in sorted(grouped):
            maximum = max(grouped[order], key=lambda row: float(row["condition_number_2"]))
            writer.writerow({
                "order": order,
                "wavelength_nm": maximum["wavelength_nm"],
                "slices": maximum["slices"],
                "grid": maximum["grid"],
                "reflectance": maximum["reflectance"],
                "max_condition_number_2": maximum["condition_number_2"],
                "layer_index_at_max": maximum["layer_index"],
                "solve_seconds": maximum["solve_seconds"],
                "condition_seconds": maximum["condition_seconds"],
            })
    temporary.replace(path)


def run_case(order: int, wavelength_nm: float, slices: int, grid: int,
             geometry: object, gold_model: object, device: object,
             cascade: str) -> tuple[float, float, float, int, list[float]]:
    import torch
    from rcwa_solver_auto import (
        ASROptions, AutoRCWA, Circle, GroupTheoryOptions, LayerSpec,
        Material, OutputSpec,
    )
    from studies.gold_motheye.converge import (
        _lattice, _mean_poynting_z, _slice_radius_nm, _zero_order_x_source,
    )

    start = time.perf_counter()
    epsilon_gold = gold_model(wavelength_nm)
    simulation = AutoRCWA(
        freq=geometry.period_nm / wavelength_nm,
        order=[order, order],
        lattice=_lattice(geometry),
        cascade=cascade,
        outputs=OutputSpec(smatrix_size="half", fields="none"),
        asr=ASROptions(
            circle_G=geometry.asr_circle_g,
            grid=(grid, grid),
            factorization_rules=True,
        ),
        group_theory=GroupTheoryOptions(
            enabled=True, symmetry="d6", strict=True, polarization="x",
        ),
        verify_cascade=False,
        dtype=torch.complex128,
        device=device,
    )
    simulation.add_input_layer(eps=1.0, mu=1.0)
    simulation.add_output_layer(eps=epsilon_gold, mu=1.0)
    simulation.set_incident_angle(0.0, 0.0)
    thickness = geometry.height_nm / slices / geometry.period_nm
    for layer in range(slices):
        radius_nm = _slice_radius_nm(geometry, layer, slices)
        simulation.add_structured_layer(
            LayerSpec(
                thickness=thickness,
                geometry=Circle(radius_nm / geometry.period_nm),
                background=Material(1.0, 1.0),
                inclusion=Material(epsilon_gold, 1.0),
                method="matched-asr",
                factorization_rules=True,
                label=f"moth-eye-{layer:03d}",
            )
        )
    simulation.solve_global_smatrix()
    if "Tf" not in simulation.computed_smatrix_blocks:
        raise RuntimeError("Forward transmission S block was not computed.")
    incident = _zero_order_x_source(simulation)
    reflected = simulation.S[1] @ incident
    transmitted = simulation.S[0] @ incident
    incident_flux = _mean_poynting_z(incident, simulation.Vi, direction=1)
    reflected_flux = _mean_poynting_z(reflected, simulation.Vi, direction=-1)
    transmitted_flux = _mean_poynting_z(transmitted, simulation.Vo, direction=1)
    if incident_flux <= 0:
        raise RuntimeError("Incident power flux is not positive.")
    reflectance = -reflected_flux / incident_flux
    power_into_substrate = transmitted_flux / incident_flux
    if (not math.isfinite(reflectance) or not 0 <= reflectance <= 1
            or not math.isfinite(power_into_substrate)
            or power_into_substrate <= 0):
        raise RuntimeError("Nonphysical result in transform diagnostic solve.")
    solve_seconds = time.perf_counter() - start

    if len(simulation.asr_T_matrices) != slices:
        raise RuntimeError(
            f"Expected {slices} ASR transforms, found {len(simulation.asr_T_matrices)}."
        )
    start = time.perf_counter()
    embedding, _, _, _, _ = simulation._triangular_star_operators()
    embedding_h = embedding.mH
    conditions = []
    with torch.no_grad():
        for layer, transform in enumerate(simulation.asr_T_matrices):
            projected = embedding_h @ transform @ embedding
            try:
                singular_values = torch.linalg.svdvals(projected)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"SVD failed for order={order}, layer={layer}."
                ) from exc
            condition = singular_values.max() / singular_values.min()
            conditions.append(float(condition.detach().cpu()))
    condition_seconds = time.perf_counter() - start
    return (reflectance, solve_seconds, condition_seconds,
            int(embedding.shape[1]), conditions)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", default="16,18,20",
                        help="Comma-separated Fourier orders; default: 16,18,20.")
    parser.add_argument("--slices", type=int, default=15)
    parser.add_argument("--wavelength-nm", type=float, default=700.0)
    parser.add_argument("--grid", type=int, default=256)
    parser.add_argument("--gold-csv", type=Path, default=DEFAULT_GOLD_CSV)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--cascade", choices=("redheffer", "algo2a"),
                        default="redheffer")
    args = parser.parse_args()
    try:
        orders = tuple(sorted({int(value.strip()) for value in args.orders.split(",")}))
    except ValueError as exc:
        parser.error(f"Invalid --orders: {exc}")
    if (not orders or min(orders) < 1 or args.slices < 1 or args.grid < 32
            or not math.isfinite(args.wavelength_nm) or args.wavelength_nm <= 0):
        parser.error("Require positive orders/slices/wavelength and grid >= 32.")

    import torch
    from studies.gold_motheye.converge import GeometryConfig, _slice_radius_nm
    from studies.shared.gold_dispersion import build_gold_model

    gold_csv = args.gold_csv.resolve()
    if not gold_csv.is_file():
        raise FileNotFoundError(f"Measured gold CSV not found: {gold_csv}")
    gold_model = build_gold_model("csv", gold_csv)
    gold_model(args.wavelength_nm)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot see a GPU.")

    geometry = GeometryConfig()
    wavelength_label = f"{args.wavelength_nm:g}".replace(".", "p")
    output_dir = (args.output_dir or HERE / "results" /
                  f"transform_condition_Nz{args.slices}_{wavelength_label}nm").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "transform_condition_numbers.csv"
    summary_path = output_dir / "transform_condition_summary.csv"
    metadata_path = output_dir / "transform_condition_metadata.json"
    config = {
        "study_version": "d6_star_transform_condition_v1",
        "geometry": asdict(geometry),
        "orders": orders,
        "wavelength_nm": args.wavelength_nm,
        "slices": args.slices,
        "grid": args.grid,
        "gold_csv_sha256": file_hash(gold_csv),
        "solver_source_sha256": file_hash(HERE.parent / "gold_motheye" / "converge.py"),
        "cascade": args.cascade,
        "matrix": "triangular_star_transverse_asr_transform",
        "condition_norm": "2",
        "dtype": "complex128",
    }
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    rows: list[dict[str, object]] = []
    if metadata_path.exists():
        saved = json.loads(metadata_path.read_text(encoding="utf-8"))
        if saved.get("signature") != signature:
            raise RuntimeError("Diagnostic settings differ; choose a new --output-dir.")
        if csv_path.exists():
            with csv_path.open(newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
        elif saved.get("completed_orders") != 0:
            raise RuntimeError("Diagnostic metadata reports completed orders without its CSV.")
    elif csv_path.exists():
        raise RuntimeError("Diagnostic CSV exists without metadata.")
    else:
        write_json(metadata_path, {**config, "signature": signature,
                                   "completed_orders": 0,
                                   "expected_orders": len(orders)})

    for order in orders:
        existing = [row for row in rows if int(row["order"]) == order]
        if len(existing) == args.slices and {
            int(row["layer_index"]) for row in existing
        } == set(range(args.slices)):
            print(f"skip: M={order} already measured", flush=True)
            continue
        if existing:
            raise RuntimeError(f"Partial or duplicate rows for M={order}; inspect {csv_path}.")
        print(f"diagnose: M={order}, Nz={args.slices}, "
              f"wavelength={args.wavelength_nm:g} nm", flush=True)
        reflectance, solve_seconds, condition_seconds, dimension, conditions = (
            run_case(order, args.wavelength_nm, args.slices, args.grid,
                     geometry, gold_model, device, args.cascade)
        )
        for layer, condition in enumerate(conditions):
            rows.append({
                "order": order,
                "wavelength_nm": args.wavelength_nm,
                "slices": args.slices,
                "grid": args.grid,
                "layer_index": layer,
                "radius_nm": _slice_radius_nm(geometry, layer, args.slices),
                "transform_star_dimension": dimension,
                "condition_number_2": condition,
                "log10_condition_number_2": (
                    math.log10(condition) if condition > 0 else float("nan")
                ),
                "reflectance": reflectance,
                "solve_seconds": solve_seconds,
                "condition_seconds": condition_seconds,
            })
        write_csv(csv_path, rows)
        write_summary(summary_path, rows)
        complete = sum(
            len([row for row in rows if int(row["order"]) == candidate]) == args.slices
            for candidate in orders
        )
        write_json(metadata_path, {**config, "signature": signature,
                                   "completed_orders": complete,
                                   "expected_orders": len(orders)})
        print(f"M={order}: R={100*reflectance:.4f}%, "
              f"max cond2={max(conditions):.4g}, "
              f"solve={solve_seconds:.1f}s, SVD={condition_seconds:.1f}s", flush=True)
    if rows:
        write_summary(summary_path, rows)
    print(f"conditions: {csv_path}")
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
