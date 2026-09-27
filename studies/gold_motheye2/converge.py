"""Fixed-grid, fixed-slice Fourier-order study for the gold moth-eye model.

The electromagnetic model is shared with studies.gold_motheye.converge.
This driver varies only Fourier order M. Grid=256 is fixed, and Nz is
fixed within a run (100 by default, configurable with --slices).
Results are checkpointed after each wavelength/order solve and plotted as SVG.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GRID = 256
DEFAULT_SLICES = 100
DEFAULT_GOLD_CSV = HERE / "data" / "au_measured_nk.csv"
DEFAULT_RESULTS = HERE / "results" / "order_sweep"
CASE_COLUMNS = (
    "order", "wavelength_nm", "slices", "grid", "reflectance",
    "transmittance_far", "absorptance_total", "motheye_absorptance",
    "substrate_absorptance", "power_into_substrate", "runtime_seconds",
    "passivity_warning", "reduced_dimension", "full_dimension",
)


def parse_positive_list(raw: str, *, integer: bool) -> tuple[int, ...] | tuple[float, ...]:
    converter = int if integer else float
    values = tuple(sorted({converter(item.strip()) for item in raw.split(",")}))
    if not values or any(not math.isfinite(float(value)) or value <= 0 for value in values):
        raise ValueError("List must contain finite positive comma-separated values.")
    return values


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configuration_signature(config: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")
    ).hexdigest()


def write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def case_key(order: int, wavelength_nm: float) -> str:
    return f"M={order}|wl={wavelength_nm:.12g}"


def sorted_cases(cases: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        cases.values(),
        key=lambda row: (int(row["order"]), float(row["wavelength_nm"])),
    )


def write_cases_csv(path: Path, cases: dict[str, dict[str, object]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CASE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted_cases(cases))
    temporary.replace(path)


def assess(
    cases: dict[str, dict[str, object]],
    orders: tuple[int, ...],
    wavelengths: tuple[float, ...],
    tolerance: float,
) -> dict[str, object]:
    complete = all(case_key(order, wavelength) in cases
                   for order in orders for wavelength in wavelengths)
    changes: dict[str, list[dict[str, float | int]]] = {}
    wavelength_pass: dict[str, bool] = {}
    for wavelength in wavelengths:
        entries: list[dict[str, float | int]] = []
        for coarse, fine in zip(orders, orders[1:]):
            first = cases.get(case_key(coarse, wavelength))
            second = cases.get(case_key(fine, wavelength))
            if first is None or second is None:
                continue
            difference = abs(float(second["reflectance"]) - float(first["reflectance"]))
            entries.append({
                "coarse_order": coarse, "fine_order": fine,
                "delta_reflectance": difference,
                "delta_reflectance_pp": 100.0 * difference,
            })
        label = f"{wavelength:g}"
        changes[label] = entries
        # The last two refinements must both pass; an earlier quiet interval
        # followed by a larger change is not convergence.
        wavelength_pass[label] = (
            complete and len(entries) >= 2
            and all(item["delta_reflectance"] <= tolerance for item in entries[-2:])
        )
    status = ("incomplete" if not complete else
              "converged_within_tested_orders" if all(wavelength_pass.values()) else
              "not_converged")
    return {
        "status": status,
        "completed_cases": len(cases),
        "expected_cases": len(orders) * len(wavelengths),
        "per_wavelength_pass": wavelength_pass,
        "adjacent_changes": changes,
    }


def finalize_completed_order_prefix(
    output_dir: Path, target_order: int, slices: int,
) -> None:
    """Remove only uncomputed trailing orders from a completed prefix."""
    from studies.gold_motheye2.plot_order import render

    csv_path = output_dir / "order_sweep.csv"
    metadata_path = output_dir / "order_sweep.json"
    checkpoint_path = output_dir / "order_checkpoint.json"
    if not metadata_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("Finalization needs existing metadata and checkpoint files.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    config_fields = (
        "study_version", "geometry", "orders", "wavelengths_nm",
        "fixed_grid", "fixed_slices", "gold_csv_sha256",
        "solver_source_sha256", "cascade", "dtype", "smatrix_size",
        "symmetry_reduction",
    )
    old_config = {name: metadata[name] for name in config_fields}
    old_orders = tuple(int(value) for value in old_config["orders"])
    wavelengths = tuple(float(value) for value in old_config["wavelengths_nm"])
    if int(old_config["fixed_slices"]) != slices or int(old_config["fixed_grid"]) != GRID:
        raise ValueError("Requested Nz/grid does not match the saved study.")
    if target_order not in old_orders:
        raise ValueError(f"M={target_order} is not in the saved order list.")
    orders = tuple(order for order in old_orders if order <= target_order)
    if len(orders) < 3:
        raise ValueError("At least three orders are required for convergence assessment.")
    config = {**old_config, "orders": orders}
    old_signature = configuration_signature(old_config)
    new_signature = configuration_signature(config)
    if checkpoint.get("signature") not in {old_signature, new_signature}:
        raise RuntimeError("Checkpoint settings do not match the saved study.")
    cases = checkpoint.get("cases")
    if not isinstance(cases, dict):
        raise ValueError("Invalid cases in order checkpoint.")
    expected = {case_key(order, wavelength)
                for order in orders for wavelength in wavelengths}
    if set(cases) != expected:
        missing = sorted(expected - set(cases))
        extra = sorted(set(cases) - expected)
        raise ValueError(
            f"Cannot finalize M={target_order}: {len(missing)} missing and "
            f"{len(extra)} outside-prefix cases."
        )
    for key, row in cases.items():
        if (key != case_key(int(row["order"]), float(row["wavelength_nm"]))
                or int(row["slices"]) != slices or int(row["grid"]) != GRID):
            raise ValueError(f"Invalid saved case: {key}")

    if old_orders != orders and checkpoint["signature"] == old_signature:
        backup_dir = output_dir / f"backup_before_finalize_M{old_orders[-1]}"
        backup_dir.mkdir(exist_ok=True)
        for path in (checkpoint_path, metadata_path, csv_path):
            if path.exists():
                backup = backup_dir / path.name
                if not backup.exists():
                    shutil.copy2(path, backup)
    write_json_atomic(checkpoint_path, {"signature": new_signature, "cases": cases})
    write_cases_csv(csv_path, cases)
    tolerance = float(metadata["tolerance"])
    assessment = assess(cases, orders, wavelengths, tolerance)
    write_json_atomic(metadata_path, {
        **config, **assessment,
        "gold_csv": metadata.get("gold_csv"),
        "tolerance": tolerance,
    })
    render(csv_path, metadata_path, output_dir)
    print(f"finalized through M={target_order}: "
          f"{assessment['completed_cases']}/{assessment['expected_cases']} cases, "
          f"status={assessment['status']}")
    print(f"results: {output_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_orders = "4,6,8,10,12,14,16,18,20"
    parser.add_argument("--orders", default=default_orders)
    parser.add_argument("--append-orders",
                        help="Append larger orders to an existing sweep, e.g. 22,24.")
    parser.add_argument("--finalize-through-order", type=int,
                        help="Keep a complete prefix through M and drop only uncomputed trailing orders; no GPU solve.")
    parser.add_argument("--slices", type=int, default=DEFAULT_SLICES,
                        help="Fixed number of height slices; default: 100.")
    parser.add_argument("--wavelengths", default="400,550,700",
                        help="Comma-separated vacuum wavelengths in nm.")
    parser.add_argument("--gold-csv", type=Path, default=DEFAULT_GOLD_CSV,
                        help="Combined wavelength_nm,n,k CSV.")
    parser.add_argument("--output-dir", type=Path,
                        help="Result directory; defaults to order_sweep_NzN for Nz != 100.")
    parser.add_argument("--tolerance", type=float, default=0.005,
                        help="Absolute reflectance change; 0.005 = 0.5 percentage point.")
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--cascade", choices=("redheffer", "algo2a"),
                        default="redheffer")
    parser.add_argument("--plot-only", action="store_true",
                        help="Regenerate graphs from saved CSV without importing PyTorch.")
    args = parser.parse_args()
    if args.append_orders and args.orders != default_orders:
        parser.error("Use either --orders or --append-orders, not both.")
    if args.finalize_through_order is not None and (
        args.append_orders or args.orders != default_orders or args.plot_only
    ):
        parser.error("--finalize-through-order cannot be combined with order selection or --plot-only.")
    if args.slices < 1:
        parser.error("--slices must be positive.")
    if not math.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error("--tolerance must be finite and positive.")
    default_results = (DEFAULT_RESULTS if args.slices == DEFAULT_SLICES else
                       DEFAULT_RESULTS.with_name(f"order_sweep_Nz{args.slices}"))
    output_dir = (args.output_dir or default_results).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "order_sweep.csv"
    metadata_path = output_dir / "order_sweep.json"
    checkpoint_path = output_dir / "order_checkpoint.json"
    if args.finalize_through_order is not None:
        finalize_completed_order_prefix(
            output_dir, args.finalize_through_order, args.slices
        )
        return 0
    try:
        orders = parse_positive_list(args.orders, integer=True)
        wavelengths = parse_positive_list(args.wavelengths, integer=False)
        if args.append_orders:
            if not metadata_path.exists():
                parser.error("--append-orders needs an existing order_sweep.json.")
            previous = json.loads(metadata_path.read_text(encoding="utf-8"))
            old_orders = tuple(int(value) for value in previous["orders"])
            added = parse_positive_list(args.append_orders, integer=True)
            if set(added).issubset(old_orders):
                orders = old_orders  # Resume an interrupted extension.
            elif added[0] > old_orders[-1]:
                orders = old_orders + added
            else:
                parser.error("New orders must all exceed the previous maximum M.")
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    if len(orders) < 3:
        parser.error("At least three orders are needed for two adjacent comparisons.")

    if args.plot_only:
        from studies.gold_motheye2.plot_order import render
        render(csv_path, metadata_path, output_dir)
        return 0

    import torch
    from studies.gold_motheye.converge import GeometryConfig, NumericalConfig, simulate_case
    from studies.shared.gold_dispersion import build_gold_model
    from studies.gold_motheye2.plot_order import render

    gold_csv = args.gold_csv.resolve()
    if not gold_csv.is_file():
        raise FileNotFoundError(f"Measured gold CSV not found: {gold_csv}")
    gold_model = build_gold_model("csv", gold_csv)
    for wavelength in wavelengths:
        gold_model(wavelength)  # fail before a GPU job if the table is out of range
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot see a GPU.")

    geometry = GeometryConfig()
    solver_source = HERE.parent / "gold_motheye" / "converge.py"
    config = {
        "study_version": "order_sweep_half_smatrix_v1",
        "geometry": asdict(geometry),
        "orders": orders,
        "wavelengths_nm": wavelengths,
        "fixed_grid": GRID,
        "fixed_slices": args.slices,
        "gold_csv_sha256": file_hash(gold_csv),
        "solver_source_sha256": file_hash(solver_source),
        "cascade": args.cascade,
        "dtype": "complex128",
        "smatrix_size": "half",
        "symmetry_reduction": "d6-source",
    }
    signature = configuration_signature(config)
    cases: dict[str, dict[str, object]] = {}
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("signature") != signature:
            if not args.append_orders or not metadata_path.exists():
                raise RuntimeError(
                    f"Checkpoint settings differ: {checkpoint_path}. "
                    "Choose a new --output-dir to keep studies separate."
                )
            previous = json.loads(metadata_path.read_text(encoding="utf-8"))
            old_orders = tuple(int(value) for value in previous["orders"])
            old_config = {**config, "orders": old_orders}
            if not (len(orders) > len(old_orders)
                    and orders[:len(old_orders)] == old_orders
                    and checkpoint.get("signature") == configuration_signature(old_config)):
                raise RuntimeError(
                    "Checkpoint settings differ. Only higher orders may be "
                    "appended; otherwise choose a new --output-dir."
                )
            print(f"extend order sweep: M={old_orders[-1]} -> {orders[-1]}; "
                  "reuse saved cases", flush=True)
        cases = checkpoint.get("cases", {})
        if not isinstance(cases, dict):
            raise ValueError("Invalid cases in order checkpoint.")
        for key, row in cases.items():
            order, wavelength = int(row["order"]), float(row["wavelength_nm"])
            if (key != case_key(order, wavelength) or order not in orders
                    or wavelength not in wavelengths
                    or int(row["slices"]) != args.slices
                    or int(row["grid"]) != GRID):
                raise ValueError(f"Case outside the requested order sweep: {key}")

    for order in orders:
        numerical = NumericalConfig(order=order, slices=args.slices, grid=GRID)
        for wavelength in wavelengths:
            key = case_key(order, wavelength)
            if key in cases:
                continue
            print(f"solve: M={order}, Nz={args.slices}, grid={GRID}, "
                  f"wavelength={wavelength:g} nm", flush=True)
            result = simulate_case(
                wavelength, numerical, geometry, gold_model,
                cascade=args.cascade, use_symmetry=True,
                symmetry_reduction="d6-source", device=device,
            )
            reflectance = float(result["reflectance"])
            if not math.isfinite(reflectance) or not -1e-6 <= reflectance <= 1 + 1e-6:
                raise RuntimeError(f"Nonphysical reflectance at {key}: {reflectance}")
            if result["passivity_warning"]:
                raise RuntimeError(f"Passivity warning at {key}; case not cached.")
            cases[key] = result
            write_json_atomic(checkpoint_path, {"signature": signature, "cases": cases})
            write_cases_csv(csv_path, cases)
            assessment = assess(cases, orders, wavelengths, args.tolerance)
            write_json_atomic(metadata_path, {**config, **assessment,
                                              "gold_csv": str(gold_csv),
                                              "tolerance": args.tolerance})

    write_cases_csv(csv_path, cases)
    assessment = assess(cases, orders, wavelengths, args.tolerance)
    write_json_atomic(metadata_path, {**config, **assessment,
                                      "gold_csv": str(gold_csv),
                                      "tolerance": args.tolerance})
    render(csv_path, metadata_path, output_dir)
    print(f"status: {assessment['status']}")
    print(f"results: {output_dir}")
    return 0 if assessment["status"] == "converged_within_tested_orders" else 2


if __name__ == "__main__":
    raise SystemExit(main())
