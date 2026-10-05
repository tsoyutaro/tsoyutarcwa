"""Au-coated PMMA moth-eye RCWA calculation using the attached optical tables.

The default model puts 30 nm Au on exposed flat valleys, with a separately
specified 30 nm radial sidewall shell and 30 nm top disk. Valley coverage may
be varied independently to test uncertainty in the deposited morphology.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

if __package__:
    from .geometry_preview import build_profile_layers, write_geometry_png, write_geometry_svg
else:
    from geometry_preview import build_profile_layers, write_geometry_png, write_geometry_svg


HERE = Path(__file__).resolve().parent
DEFAULT_SOLVER_ROOT = HERE.parents[1]


def read_pmma(path: Path) -> list[tuple[float, float]]:
    """The supplied Szczurowski table has wavelength in micrometres."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["wl", "n"]:
            raise ValueError(f"{path}: expected columns wl,n")
        data = [(1000.0 * float(row["wl"]), float(row["n"])) for row in reader]
    check_table(data, path)
    if any(n <= 0.0 for _, n in data):
        raise ValueError("PMMA refractive indices must be positive")
    return data


def read_gold(path: Path) -> list[tuple[float, float, float]]:
    """The supplied Au file contains separate wl,n and wl,k CSV blocks."""
    blocks: dict[str, list[tuple[float, float]]] = {}
    current: str | None = None
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if not row:
                continue
            cells = [cell.strip() for cell in row]
            if cells in (["wl", "n"], ["wl", "k"]):
                current = cells[1]
                if current in blocks:
                    raise ValueError(f"{path}:{line_number}: duplicate {current} block")
                blocks[current] = []
            elif len(cells) == 2 and current is not None:
                blocks[current].append((1000.0 * float(cells[0]), float(cells[1])))
            else:
                raise ValueError(f"{path}:{line_number}: unexpected row {row!r}")
    if set(blocks) != {"n", "k"}:
        raise ValueError(f"{path}: both wl,n and wl,k blocks are required")
    check_table(blocks["n"], path)
    check_table(blocks["k"], path)
    if len(blocks["n"]) != len(blocks["k"]):
        raise ValueError("Au n and k blocks have different lengths")
    data = []
    for (wn, n), (wk, k) in zip(blocks["n"], blocks["k"]):
        if not math.isclose(wn, wk, abs_tol=1e-8):
            raise ValueError("Au n and k wavelength grids differ")
        if n < 0.0 or k < 0.0:
            raise ValueError("Au n and k must be nonnegative")
        data.append((wn, n, k))
    return data


def check_table(data: list[tuple[float, ...]], path: Path) -> None:
    if len(data) < 2:
        raise ValueError(f"{path}: at least two rows are required")
    if any(not all(math.isfinite(value) for value in row) for row in data):
        raise ValueError(f"{path}: nonfinite value")
    if any(a[0] >= b[0] for a, b in zip(data, data[1:])):
        raise ValueError(f"{path}: wavelengths must increase strictly")


def interpolate(data: list[tuple[float, ...]], wavelength: float) -> tuple[float, ...]:
    grid = [row[0] for row in data]
    if wavelength < grid[0] or wavelength > grid[-1]:
        raise ValueError(
            f"{wavelength:g} nm is outside CSV range {grid[0]:g}–{grid[-1]:g} nm; "
            "extrapolation is disabled"
        )
    right = bisect.bisect_left(grid, wavelength)
    if right < len(data) and grid[right] == wavelength:
        return data[right][1:]
    left = right - 1
    fraction = (wavelength - grid[left]) / (grid[right] - grid[left])
    return tuple(
        a + fraction * (b - a)
        for a, b in zip(data[left][1:], data[right][1:])
    )


def parse_wavelengths(spec: str) -> list[float]:
    if ":" in spec:
        start, stop, step = (float(part) for part in spec.split(":"))
        if step <= 0 or stop < start:
            raise ValueError("invalid start:stop:step wavelength range")
        count = int(math.floor((stop - start) / step + 1e-10))
        values = [start + i * step for i in range(count + 1)]
        if values[-1] < stop - 1e-9:
            values.append(stop)
    else:
        values = [float(part) for part in spec.split(",")]
    if not values or any(not math.isfinite(w) or w <= 0 for w in values):
        raise ValueError("wavelengths must be finite and positive")
    return sorted(set(values))


def parse_orders(spec: str | None, single_order: int) -> list[int]:
    orders = [single_order] if spec is None else sorted({int(s) for s in spec.split(",")})
    if any(order <= 0 for order in orders):
        raise ValueError("orders must be positive")
    if spec is not None and len(orders) < 3:
        raise ValueError("--orders requires at least three distinct orders")
    return orders


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pmma-csv", type=Path, default=HERE / "data/Szczurowski.csv")
    parser.add_argument("--gold-csv", type=Path, default=HERE / "data/au_measured_nk.csv")
    parser.add_argument("--solver-root", type=Path, default=DEFAULT_SOLVER_ROOT)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results/measured_30nm")
    parser.add_argument("--period-nm", type=float, default=200.0)
    parser.add_argument("--height-nm", type=float, default=500.0)
    parser.add_argument("--wavelengths", default="450,550,700")
    parser.add_argument("--order", type=int, default=14)
    parser.add_argument("--orders", help="Optional order sweep, e.g. 10,12,14,16,18,20,22")
    parser.add_argument("--slices", type=int, default=32)
    parser.add_argument("--grid", type=int, default=256)
    parser.add_argument("--mapping", choices=("outer", "double"), default="outer")
    parser.add_argument("--valley-gold-nm", type=float, default=30.0,
                        help="Au height on exposed PMMA valleys; 0 reproduces the earlier bare valleys")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--tolerance", type=float, default=0.005)
    parser.add_argument("--retain-auxiliary", action="store_true",
                        help="Keep field-reconstruction tensors for memory/speed comparison")
    parser.add_argument("--check-only", action="store_true", help="Validate geometry/CSV without RCWA")
    parser.add_argument("--preview-only", action="store_true",
                        help="Write geometry SVG/CSV and exit before loading RCWA")
    args = parser.parse_args()

    if any(not math.isfinite(v) or v <= 0 for v in (args.period_nm, args.height_nm)):
        parser.error("period and height must be finite and positive")
    if args.slices <= 0 or args.grid <= 0 or args.tolerance <= 0:
        parser.error("slices, grid and tolerance must be positive")
    if not math.isfinite(args.valley_gold_nm) or not 0 <= args.valley_gold_nm <= args.height_nm:
        parser.error("--valley-gold-nm must be within [0, height-nm]")
    orders = parse_orders(args.orders, args.order)
    wavelengths = parse_wavelengths(args.wavelengths)
    pmma_table = read_pmma(args.pmma_csv)
    gold_table = read_gold(args.gold_csv)

    # PMMA core: 10 nm tip diameter, 130 nm base diameter. The Au shell is
    # 30 nm in the *radial* direction; the 30 nm high tip disk is 70 nm wide.
    tip_radius_nm, base_radius_nm, gold_thickness_nm = 5.0, 65.0, 30.0
    cap_diameter_nm = 2 * (tip_radius_nm + gold_thickness_nm)
    if 2 * (base_radius_nm + gold_thickness_nm) >= args.period_nm:
        parser.error("Au-coated bases touch/overlap; period must exceed 190 nm")
    profile_layers = build_profile_layers(
        height_nm=args.height_nm,
        tip_radius_nm=tip_radius_nm,
        base_radius_nm=base_radius_nm,
        gold_thickness_nm=gold_thickness_nm,
        profile_power=1.0,
        slices=args.slices,
        valley_gold_thickness_nm=args.valley_gold_nm,
    )
    material_values = []
    for wavelength in wavelengths:
        (pmma_n,) = interpolate(pmma_table, wavelength)
        gold_n, gold_k = interpolate(gold_table, wavelength)
        epsilon_gold = complex(gold_n, gold_k) ** 2
        material_values.append({
            "wavelength_nm": wavelength,
            "pmma_n": pmma_n,
            "gold_n": gold_n,
            "gold_k": gold_k,
            "gold_epsilon_real": epsilon_gold.real,
            "gold_epsilon_imag": epsilon_gold.imag,
        })
    settings = {
        "geometry_nm": {
            "period": args.period_nm,
            "height": args.height_nm,
            "pmma_tip_diameter": 10.0,
            "pmma_base_diameter": 130.0,
            "gold_radial_shell_thickness": gold_thickness_nm,
            "gold_top_disk_thickness": 30.0,
            "gold_top_disk_diameter": cap_diameter_nm,
            "minimum_base_gap_between_gold_shells": args.period_nm - 190.0,
            "gold_on_flat_valleys": args.valley_gold_nm > 0,
            "gold_valley_thickness": args.valley_gold_nm,
            "profile_power": 1.0,
            "substrate": "semi-infinite PMMA",
        },
        "numerics": {
            "orders": orders,
            "profile_slices": args.slices,
            "patterned_layers": len(profile_layers) + 1,
            "asr_grid": [args.grid, args.grid],
            "radial_mapping": args.mapping,
            "polarization": "x",
            "incidence": "normal from air",
            "lattice": "triangular",
            "cascade": "redheffer",
            "symmetry_reduction": "d6-source",
            "dtype": "complex128",
            "tolerance": args.tolerance,
            "device_request": args.device,
        },
        "material_csv": {
            "pmma": str(args.pmma_csv.resolve()),
            "pmma_sha256": file_sha256(args.pmma_csv),
            "gold": str(args.gold_csv.resolve()),
            "gold_sha256": file_sha256(args.gold_csv),
            "wavelength_unit_in_input": "micrometre",
            "interpolation": "linear in n and k; no extrapolation",
        },
        "materials_at_wavelengths": material_values,
        "solver_root": str(args.solver_root.resolve()),
    }
    if args.retain_auxiliary:
        settings["numerics"]["retain_auxiliary"] = True
    if args.check_only:
        print(json.dumps(settings, indent=2, ensure_ascii=False))
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    geometry_svg = args.output_dir / "geometry_preview.svg"
    write_geometry_svg(
        geometry_svg,
        layers=profile_layers,
        period_nm=args.period_nm,
        height_nm=args.height_nm,
        tip_radius_nm=tip_radius_nm,
        base_radius_nm=base_radius_nm,
        gold_thickness_nm=gold_thickness_nm,
        valley_gold_thickness_nm=args.valley_gold_nm,
    )
    write_geometry_png(
        args.output_dir / "geometry_preview.png",
        layers=profile_layers,
        period_nm=args.period_nm,
        height_nm=args.height_nm,
        tip_radius_nm=tip_radius_nm,
        base_radius_nm=base_radius_nm,
        gold_thickness_nm=gold_thickness_nm,
        valley_gold_thickness_nm=args.valley_gold_nm,
    )
    with (args.output_dir / "geometry_layers.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        columns = (
            "layer", "kind", "top_z_nm", "bottom_z_nm", "thickness_nm",
            "pmma_core_radius_nm", "gold_outer_radius_nm",
        )
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerow({
            "layer": 0, "kind": "top_disk", "top_z_nm": args.height_nm + gold_thickness_nm,
            "bottom_z_nm": args.height_nm, "thickness_nm": gold_thickness_nm,
            "pmma_core_radius_nm": "", "gold_outer_radius_nm": cap_diameter_nm / 2,
        })
        for index, layer in enumerate(profile_layers, start=1):
            writer.writerow({
                "layer": index,
                "kind": layer.kind,
                "top_z_nm": args.height_nm - layer.top_depth_nm,
                "bottom_z_nm": args.height_nm - layer.bottom_depth_nm,
                "thickness_nm": layer.bottom_depth_nm - layer.top_depth_nm,
                "pmma_core_radius_nm": layer.core_radius_nm,
                "gold_outer_radius_nm": (
                    "fills_cell_outside_pmma" if layer.kind == "valley"
                    else layer.outer_radius_nm
                ),
            })
    (args.output_dir / "settings.json").write_text(
        json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if args.preview_only:
        print(f"geometry preview: {geometry_svg}")
        return 0

    if not args.solver_root.is_dir():
        parser.error(f"solver root not found: {args.solver_root}")
    sys.path.insert(0, str(args.solver_root))
    try:
        import torch
        from studies.pmma_gold_motheye.common import (
            GeometryConfig,
            NumericalConfig,
            choose_candidate,
            simulate_case,
        )
    except ImportError as exc:
        parser.error(f"RCWA runtime is unavailable: {exc}")

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    settings["numerics"]["device_used"] = str(device)
    base_geometry = GeometryConfig(
        period_nm=args.period_nm,
        height_nm=args.height_nm,
        tip_radius_nm=tip_radius_nm,
        base_radius_nm=base_radius_nm,
        gold_thickness_nm=gold_thickness_nm,
        profile_power=1.0,
        pmma_index=1.49,  # Replaced by the CSV value at each wavelength.
        lattice="triangular",
        include_top_cap=True,
        asr_circle_g=0.03,
    )
    checkpoint_path = args.output_dir / "checkpoint.json"
    signature = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("signature") != signature:
            parser.error("checkpoint has different settings; select a new --output-dir")
        cases = checkpoint["cases"]
    else:
        cases = {}
    spectra = {}
    for order in orders:
        spectrum = []
        for material in material_values:
            wavelength = material["wavelength_nm"]
            key = f"M={order}|Nz={args.slices}|grid={args.grid}|wl={wavelength:.12g}"
            if key not in cases:
                geometry = replace(base_geometry, pmma_index=material["pmma_n"])
                numerical = NumericalConfig(order, args.slices, args.grid, args.mapping)
                epsilon_gold = complex(material["gold_n"], material["gold_k"]) ** 2
                print(f"solve: M={order}, Nz={args.slices}, wavelength={wavelength:g} nm", flush=True)
                result = simulate_case(
                    wavelength,
                    numerical,
                    geometry,
                    lambda _w, eps=epsilon_gold: eps,
                    cascade="redheffer",
                    use_symmetry=True,
                    symmetry_reduction="d6-source",
                    factorization_rules=True,
                    device=device,
                    valley_gold_thickness_nm=args.valley_gold_nm,
                    discard_auxiliary=not args.retain_auxiliary,
                )
                result.update({"pmma_n": material["pmma_n"], "gold_n": material["gold_n"], "gold_k": material["gold_k"]})
                cases[key] = result
                checkpoint_path.write_text(
                    json.dumps({"signature": signature, "cases": cases}, indent=2, allow_nan=False),
                    encoding="utf-8",
                )
            spectrum.append(cases[key])
        spectra[order] = spectrum

    if args.orders is None:
        selected_order, converged, comparisons = orders[0], False, []
        status = (
            "nonpassive_results"
            if any(case["passivity_warning"] for case in spectra[selected_order])
            else "single_order_unverified"
        )
    else:
        selected_order, converged, comparisons = choose_candidate(orders, spectra, args.tolerance)
        nonpassive = any(case["passivity_warning"] for spectrum in spectra.values() for case in spectrum)
        status = "converged" if converged else "nonpassive_results" if nonpassive else "candidate_range_insufficient"
    report = {
        "status": status,
        "order_convergence_only": args.orders is not None,
        "layer_and_grid_convergence_checked": False,
        "selected_order": selected_order,
        "comparisons": comparisons,
        "settings": settings,
        "selected_spectrum": spectra[selected_order],
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    columns = (
        "wavelength_nm", "order", "profile_slices", "total_pattern_layers",
        "valley_gold_thickness_nm", "grid", "pmma_n", "gold_n", "gold_k",
        "reflectance", "transmittance", "absorptance", "passivity_warning",
    )
    with (args.output_dir / "spectrum.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(spectra[selected_order])
    print(f"{status}: {args.output_dir / 'report.json'}")
    return 0 if status in {"converged", "single_order_unverified"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
