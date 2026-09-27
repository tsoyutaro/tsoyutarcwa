"""Sweep the number of height slices at fixed Fourier order and ASR grid.

The default study uses M=16, grid=256, Nz=10..100, and 400/550/700 nm.
Use --order 18 for an independent M=18 study. The gold material and physical
geometry match gold_motheye2/converge.py. Every completed case is checkpointed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studies.gold_motheye2.converge import (
    CASE_COLUMNS, file_hash, parse_positive_list, write_json_atomic,
)

GRID = 256
DEFAULT_SLICES = "10,15,20,30,40,50,60,70,80,90,100"
DEFAULT_GOLD_CSV = HERE / "data" / "au_measured_nk.csv"


def case_key(slices: int, wavelength_nm: float) -> str:
    return f"Nz={slices}|wl={wavelength_nm:.12g}"


def write_cases_csv(path: Path, cases: dict[str, dict[str, object]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CASE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(
            cases.values(),
            key=lambda row: (int(row["slices"]), float(row["wavelength_nm"])),
        ))
    temporary.replace(path)


def assess(
    cases: dict[str, dict[str, object]],
    slices: tuple[int, ...],
    wavelengths: tuple[float, ...],
    tolerance: float,
) -> dict[str, object]:
    complete = all(case_key(nz, wavelength) in cases
                   for nz in slices for wavelength in wavelengths)
    changes: dict[str, list[dict[str, float | int]]] = {}
    per_wavelength_pass: dict[str, bool] = {}
    for wavelength in wavelengths:
        entries: list[dict[str, float | int]] = []
        for coarse, fine in zip(slices, slices[1:]):
            first = cases.get(case_key(coarse, wavelength))
            second = cases.get(case_key(fine, wavelength))
            if first is None or second is None:
                continue
            delta = abs(float(second["reflectance"]) - float(first["reflectance"]))
            entries.append({"coarse_slices": coarse, "fine_slices": fine,
                            "delta_reflectance": delta,
                            "delta_reflectance_pp": 100.0 * delta})
        label = f"{wavelength:g}"
        changes[label] = entries
        # Require the two adjacent refinements at the actual upper end.
        last = entries[-2:]
        per_wavelength_pass[label] = bool(
            complete and len(last) == 2
            and last[0]["fine_slices"] == slices[-2]
            and last[1]["fine_slices"] == slices[-1]
            and all(float(item["delta_reflectance"]) <= tolerance for item in last)
        )
    status = ("incomplete" if not complete else
              "converged_within_tested_slices" if all(per_wavelength_pass.values())
              else "not_converged")
    return {"status": status, "completed_cases": len(cases),
            "expected_cases": len(slices) * len(wavelengths),
            "per_wavelength_pass": per_wavelength_pass,
            "adjacent_changes": changes}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", type=int, choices=(16, 18), default=16)
    parser.add_argument("--slices", default=DEFAULT_SLICES,
                        help="Comma-separated number of equal-height slices.")
    parser.add_argument("--wavelengths", default="400,550,700",
                        help="Comma-separated vacuum wavelengths in nm.")
    parser.add_argument("--gold-csv", type=Path, default=DEFAULT_GOLD_CSV)
    parser.add_argument("--output-dir", type=Path,
                        help="Default: results/slice_sweep_M16 or M18.")
    parser.add_argument("--tolerance", type=float, default=0.005,
                        help="Absolute change in R; 0.005 = 0.5 percentage point.")
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--cascade", choices=("redheffer", "algo2a"),
                        default="redheffer")
    parser.add_argument("--plot-only", action="store_true",
                        help="Regenerate SVGs from saved results without importing PyTorch.")
    args = parser.parse_args()
    try:
        slices = parse_positive_list(args.slices, integer=True)
        wavelengths = parse_positive_list(args.wavelengths, integer=False)
    except ValueError as error:
        parser.error(str(error))
    if len(slices) < 3:
        parser.error("At least three slice counts are required for two adjacent comparisons.")
    if not math.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error("--tolerance must be finite and positive.")
    output_dir = (args.output_dir or HERE / "results" / f"slice_sweep_M{args.order}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "slice_sweep.csv"
    metadata_path = output_dir / "slice_sweep.json"
    checkpoint_path = output_dir / "slice_checkpoint.json"

    if args.plot_only:
        from studies.gold_motheye2.plot_slices import render
        render(csv_path, metadata_path, output_dir)
        return 0

    import torch
    from studies.gold_motheye.converge import GeometryConfig, NumericalConfig, simulate_case
    from studies.shared.gold_dispersion import build_gold_model
    from studies.gold_motheye2.plot_slices import render

    gold_csv = args.gold_csv.resolve()
    if not gold_csv.is_file():
        raise FileNotFoundError(f"Measured gold CSV not found: {gold_csv}")
    gold_model = build_gold_model("csv", gold_csv)
    for wavelength in wavelengths:
        gold_model(wavelength)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot see a GPU.")

    geometry = GeometryConfig()
    config = {
        "study_version": "slice_sweep_half_smatrix_v1",
        "geometry": asdict(geometry),
        "fixed_order": args.order,
        "fixed_grid": GRID,
        "slice_counts": slices,
        "wavelengths_nm": wavelengths,
        "gold_csv_sha256": file_hash(gold_csv),
        "solver_source_sha256": file_hash(HERE.parent / "gold_motheye" / "converge.py"),
        "cascade": args.cascade,
        "dtype": "complex128",
        "smatrix_size": "half",
        "symmetry_reduction": "d6-source",
    }
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    cases: dict[str, dict[str, object]] = {}
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("signature") != signature:
            raise RuntimeError("Checkpoint settings differ; choose a new --output-dir.")
        cases = checkpoint.get("cases", {})

    def persist() -> None:
        write_cases_csv(csv_path, cases)
        assessment = assess(cases, slices, wavelengths, args.tolerance)
        write_json_atomic(metadata_path, {**config, **assessment,
                                          "gold_csv": str(gold_csv),
                                          "tolerance": args.tolerance})
        render(csv_path, metadata_path, output_dir)

    for nz in slices:
        numerical = NumericalConfig(order=args.order, slices=nz, grid=GRID)
        for wavelength in wavelengths:
            key = case_key(nz, wavelength)
            if key in cases:
                continue
            print(f"solve: M={args.order}, Nz={nz}, grid={GRID}, "
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
            persist()

    # Also reconstruct the CSV and figures if a previous job stopped just
    # after its final checkpoint was written.
    persist()
    print(f"status: {assess(cases, slices, wavelengths, args.tolerance)['status']}")
    print(f"results: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
