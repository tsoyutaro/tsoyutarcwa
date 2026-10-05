"""Checkpointed order, height-slice, and ASR-grid convergence for Au moth-eye.

The physical model and measured Au data match gold_motheye2. The checkpoint
signature includes all RCWA Python sources so results from a modified solver
cannot silently mix with older calculations. Preparation and reporting need
only the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studies.shared.gold_dispersion import build_gold_model


GEOMETRY = {
    "period_nm": 200.0,
    "height_nm": 500.0,
    "tip_radius_nm": 5.0,
    "base_radius_nm": 95.0,
    "profile_power": 1.0,
    "lattice": "triangular",
    "substrate_mode": "semi-infinite",
    "substrate_thickness_nm": 200.0,
    "back_index": 1.0,
    "asr_circle_g": 0.03,
}
DEFAULT_VALUES = {
    "order": "4,6,8,10,12,14,16,18,20",
    "slices": "10,15,20,30,40,50,60,70,80,90,100",
    "grid": "96,128,192,256,320",
}
METRICS = ("reflectance", "power_into_substrate", "motheye_absorptance")
CASE_COLUMNS = (
    "axis", "value", "wavelength_nm", "order", "slices", "grid",
    "reflectance", "transmittance_far", "power_into_substrate",
    "motheye_absorptance", "substrate_absorptance", "absorptance_total",
    "runtime_seconds", "reduced_dimension", "full_dimension", "symmetry_reduction",
)


def _integers(raw: str) -> tuple[int, ...]:
    values = tuple(sorted({int(item.strip()) for item in raw.split(",")}))
    if not values or any(value <= 0 for value in values):
        raise ValueError("Values must be positive comma-separated integers.")
    return values


def _wavelengths(raw: str) -> tuple[float, ...]:
    values = tuple(sorted({float(item.strip()) for item in raw.split(",")}))
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Wavelengths must be finite positive comma-separated values.")
    return values


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes() -> dict[str, str]:
    paths = [
        ROOT / "rcwa_solver_auto.py",
        ROOT / "studies" / "gold_motheye" / "converge.py",
        ROOT / "studies" / "shared" / "gold_dispersion.py",
        HERE / "converge.py",
        *(ROOT / "rcwa_ext").glob("*.py"),
    ]
    return {path.relative_to(ROOT).as_posix(): _digest(path)
            for path in sorted(paths)}


def _signature(plan: dict) -> str:
    return hashlib.sha256(json.dumps(plan, sort_keys=True,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _key(value: int, wavelength: float) -> str:
    return f"{value}|{wavelength:.12g}"


def _numerical(plan: dict, value: int) -> dict[str, int]:
    numbers = dict(plan["fixed_numerics"])
    numbers[plan["axis"]] = value
    return numbers


def _plan(args: argparse.Namespace) -> dict:
    values = _integers(args.values or DEFAULT_VALUES[args.axis])
    if len(values) < 3:
        raise ValueError("At least three values are needed for two upper-end comparisons.")
    wavelengths = _wavelengths(args.wavelengths)
    csv_path = args.gold_csv.resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Measured Au CSV not found: {csv_path}")
    gold = build_gold_model("csv", csv_path)
    for wavelength in wavelengths:
        gold(wavelength)
    if not math.isfinite(args.tolerance) or args.tolerance <= 0:
        raise ValueError("Tolerance must be finite and positive.")
    if min(args.order, args.slices, args.grid) <= 0:
        raise ValueError("Fixed M, Nz and grid must be positive.")
    return {
        "version": 1,
        "axis": args.axis,
        "values": list(values),
        "wavelengths_nm": list(wavelengths),
        "geometry": GEOMETRY,
        "fixed_numerics": {"order": args.order, "slices": args.slices,
                           "grid": args.grid},
        "material": {"model": "measured_csv", "path": str(csv_path),
                     "sha256": _digest(csv_path)},
        "solver": {"cascade": args.cascade, "dtype": "complex128",
                   "smatrix_size": "half", "symmetry_reduction": "d6-source",
                   "requested_device": args.device},
        "tolerance": args.tolerance,
        "source_sha256": _source_hashes(),
    }


def _load_checkpoint(path: Path, plan: dict) -> dict:
    if not path.exists():
        return {"signature": _signature(plan), "plan": plan, "cases": {}}
    saved = json.loads(path.read_text(encoding="utf-8"))
    old_plan = saved.get("plan")
    if not isinstance(old_plan, dict) or saved.get("signature") != _signature(old_plan):
        raise ValueError("Invalid checkpoint signature or missing saved plan.")
    if old_plan != plan:
        old_values = old_plan.get("values", ())
        previous = {key: value for key, value in old_plan.items() if key != "values"}
        requested = {key: value for key, value in plan.items() if key != "values"}
        if (previous != requested or len(plan["values"]) <= len(old_values)
                or list(plan["values"][:len(old_values)]) != list(old_values)):
            raise RuntimeError("Checkpoint uses different settings or RCWA code. "
                               "Use another --output-dir; only a higher-value "
                               "prefix extension can reuse saved results.")
        print(f"extend {plan['axis']} sweep: {old_values[-1]} -> {plan['values'][-1]}",
              flush=True)
        saved["plan"] = plan
        saved["signature"] = _signature(plan)
    cases = saved.get("cases")
    if not isinstance(cases, dict):
        raise ValueError("Invalid checkpoint cases.")
    for key, row in cases.items():
        value, wavelength = int(row["value"]), float(row["wavelength_nm"])
        if (key != _key(value, wavelength) or row.get("axis") != plan["axis"]
                or value not in plan["values"]
                or wavelength not in plan["wavelengths_nm"]
                or any(int(row[name]) != expected
                       for name, expected in _numerical(plan, value).items())):
            raise ValueError(f"Case outside current plan: {key}")
    return saved


def assess(plan: dict, cases: dict[str, dict]) -> dict:
    values = plan["values"]
    wavelengths = plan["wavelengths_nm"]
    changes = {}
    per_wavelength_pass = {}
    all_present = all(_key(value, wavelength) in cases
                      for value in values for wavelength in wavelengths)
    for wavelength in wavelengths:
        rows = []
        for low, high in zip(values, values[1:]):
            first, second = cases.get(_key(low, wavelength)), cases.get(_key(high, wavelength))
            if first is None or second is None:
                continue
            deltas = {name: abs(float(second[name]) - float(first[name]))
                      for name in METRICS}
            rows.append({"low": low, "high": high,
                         "absolute_changes": deltas,
                         "max_absolute_change": max(deltas.values()),
                         "passes_tolerance": max(deltas.values()) <= plan["tolerance"]})
        changes[f"{wavelength:g}"] = rows
        tail = rows[-2:]
        per_wavelength_pass[f"{wavelength:g}"] = bool(
            all_present and len(tail) == 2
            and tail[0]["high"] == values[-2] and tail[1]["high"] == values[-1]
            and all(row["passes_tolerance"] for row in tail))
    status = ("incomplete" if not all_present else
              "converged_within_tested_values" if all(per_wavelength_pass.values())
              else "not_converged")
    return {"status": status, "completed_cases": len(cases),
            "expected_cases": len(values) * len(wavelengths),
            "per_wavelength_pass": per_wavelength_pass,
            "adjacent_changes": changes,
            "criterion": "At every wavelength, all three metrics change by at most "
                         "tolerance across each of the final two adjacent steps."}


def _persist(output: Path, saved: dict) -> dict:
    plan, cases = saved["plan"], saved["cases"]
    rows = sorted(cases.values(), key=lambda row: (int(row["value"]),
                                                  float(row["wavelength_nm"])))
    temporary = output / "cases.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CASE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output / "cases.csv")
    report = {"signature": saved["signature"], "plan": plan,
              "runtime_environment": saved.get("runtime_environment"),
              **assess(plan, cases)}
    _write_json(output / "report.json", report)
    from studies.gold_motheye3.plot import render
    render(plan, cases, report, output / "convergence.svg")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis", choices=tuple(DEFAULT_VALUES), default="order")
    parser.add_argument("--values", help="Comma-separated values for selected axis; a strict upper extension resumes.")
    parser.add_argument("--wavelengths", default="400,550,700")
    parser.add_argument("--order", type=int, default=16, help="Fixed M for slices/grid sweeps.")
    parser.add_argument("--slices", type=int, default=100, help="Fixed Nz for order/grid sweeps.")
    parser.add_argument("--grid", type=int, default=256, help="Fixed ASR grid for order/slices sweeps.")
    parser.add_argument("--gold-csv", type=Path, default=HERE / "data" / "au_measured_nk.csv")
    parser.add_argument("--tolerance", type=float, default=0.005,
                        help="Absolute fraction: 0.005 means 0.5 percentage point.")
    parser.add_argument("--cascade", choices=("redheffer", "algo2a"), default="redheffer")
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--report-only", action="store_true",
                        help="Build CSV/report/SVG from saved cases without PyTorch.")
    args = parser.parse_args()
    if args.prepare_only and args.report_only:
        parser.error("Choose only one of --prepare-only and --report-only.")
    try:
        plan = _plan(args)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    fixed = plan["fixed_numerics"]
    name = (f"order_Nz{fixed['slices']}_grid{fixed['grid']}" if args.axis == "order"
            else f"slices_M{fixed['order']}_grid{fixed['grid']}" if args.axis == "slices"
            else f"grid_M{fixed['order']}_Nz{fixed['slices']}")
    output = (args.output_dir or HERE / "results" / name).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint.json"
    saved = _load_checkpoint(checkpoint_path, plan)
    _write_json(output / "plan.json", plan)
    if args.prepare_only:
        print(f"Prepared {len(plan['values']) * len(plan['wavelengths_nm'])} cases: "
              f"{output / 'plan.json'}")
        return 0
    if args.report_only:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No saved checkpoint: {checkpoint_path}")
        report = _persist(output, saved)
        print(f"{report['status']}: {report['completed_cases']}/{report['expected_cases']} cases")
        print(f"report: {output / 'report.json'}")
        return 0

    import torch
    from studies.gold_motheye.converge import GeometryConfig, NumericalConfig, simulate_case

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot see a GPU.")
    environment = {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                   "device_type": device.type}
    if (saved.get("runtime_environment") is not None
            and saved["runtime_environment"] != environment):
        raise RuntimeError("Checkpoint PyTorch/CUDA environment differs. Use another --output-dir.")
    saved["runtime_environment"] = environment
    gold = build_gold_model("csv", args.gold_csv.resolve())
    geometry = GeometryConfig(**GEOMETRY)
    for value in plan["values"]:
        numbers = _numerical(plan, value)
        numerical = NumericalConfig(**numbers)
        for wavelength in plan["wavelengths_nm"]:
            key = _key(value, wavelength)
            if key in saved["cases"]:
                continue
            print(f"solve {args.axis}={value}, M={numbers['order']}, "
                  f"Nz={numbers['slices']}, grid={numbers['grid']}, "
                  f"wavelength={wavelength:g} nm", flush=True)
            result = simulate_case(
                wavelength, numerical, geometry, gold,
                cascade=args.cascade, use_symmetry=True,
                symmetry_reduction="d6-source", device=device)
            metrics = (float(result[name]) for name in METRICS)
            if (result["passivity_warning"] or
                    not all(math.isfinite(item) and -1e-5 <= item <= 1 + 1e-5
                            for item in metrics) or
                    abs(float(result["transmittance_far"])) > 1e-10 or
                    result["symmetry_reduction"] != "D6-E1-source-row"):
                raise RuntimeError(f"Nonphysical or unexpected solver result at {key}: {result}")
            saved["cases"][key] = {**result, "axis": args.axis, "value": value}
            _write_json(checkpoint_path, saved)
            _persist(output, saved)
    report = _persist(output, saved)
    print(f"status: {report['status']}")
    print(f"report: {output / 'report.json'}")
    print(f"figure: {output / 'convergence.svg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
