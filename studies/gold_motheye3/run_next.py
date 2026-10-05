"""Inspect saved grid results or extend the Au moth-eye sweep to 512/576.

Use --report-only to print actual R/P/A and signed changes without a solve.
The default solve adds grids 512 and 576 to the existing M=16, Nz=100 sweep.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUTPUT = HERE / "results" / "grid_M16_Nz100"
BASE_VALUES = [96, 128, 192, 256, 320, 384, 448]
METRICS = ("reflectance", "power_into_substrate", "motheye_absorptance")


def read_checkpoint() -> dict:
    path = OUTPUT / "checkpoint.json"
    if not path.is_file():
        raise FileNotFoundError(f"Existing grid checkpoint is required: {path}")
    saved = json.loads(path.read_text(encoding="utf-8"))
    plan, cases = saved.get("plan", {}), saved.get("cases", {})
    if (plan.get("axis") != "grid"
            or plan.get("fixed_numerics") != {"order": 16, "slices": 100, "grid": 256}
            or plan.get("wavelengths_nm") != [400.0, 550.0, 700.0]
            or not isinstance(cases, dict)):
        raise ValueError("Expected the existing grid sweep at M=16, Nz=100, 400/550/700 nm.")
    from studies.gold_motheye3.converge import _signature
    if saved.get("signature") != _signature(plan):
        raise ValueError("Checkpoint signature does not match its saved plan.")
    return saved


def extended_values(saved: dict, added: list[int]) -> list[int]:
    current = saved["plan"]["values"]
    expected = {f"{value}|{wavelength:g}" for value in BASE_VALUES
                for wavelength in saved["plan"]["wavelengths_nm"]}
    if not expected.issubset(saved["cases"]):
        raise ValueError("The 21 cases through grid=448 must be completed before extension.")
    missing = sorted(set(added) - set(current))
    if missing and min(missing) <= max(current):
        raise ValueError("New grid values must exceed the saved maximum grid.")
    return sorted(set(current) | set(added))


def command(saved: dict, values: list[int], device: str, prepare: bool) -> list[str]:
    plan = saved["plan"]
    fixed = plan["fixed_numerics"]
    result = [sys.executable, str(HERE / "converge.py"),
              "--axis", "grid", "--values", ",".join(map(str, values)),
              "--order", str(fixed["order"]), "--slices", str(fixed["slices"]),
              "--grid", str(fixed["grid"]),
              "--wavelengths", ",".join(f"{value:g}" for value in plan["wavelengths_nm"]),
              "--tolerance", str(plan["tolerance"]),
              "--cascade", plan["solver"]["cascade"], "--device", device,
              "--gold-csv", plan["material"]["path"], "--output-dir", str(OUTPUT)]
    if prepare:
        result.append("--prepare-only")
    return result


def show_result(saved: dict) -> None:
    from studies.gold_motheye3.converge import assess
    plan, cases = saved["plan"], saved["cases"]
    report = assess(plan, cases)
    print(f"grid status: {report['status']} "
          f"({report['completed_cases']}/{report['expected_cases']} cases)")
    print(f"Tolerance: {100 * plan['tolerance']:g} percentage points; M=16, Nz=100")
    for wavelength in plan["wavelengths_nm"]:
        print(f"\n{wavelength:g} nm: values in percent; changes in percentage points")
        print(" grid       R(%)     P_sub(%)  A_pillar(%)    signed_dR     max_abs_change")
        previous = None
        for value in plan["values"]:
            row = cases.get(f"{value}|{wavelength:.12g}")
            if row is None:
                print(f"{value:5d}  not calculated")
                previous = None
                continue
            reflection, substrate, absorption = [100 * float(row[name]) for name in METRICS]
            if previous is None:
                change = "            -                  -"
            else:
                signed = 100 * (float(row["reflectance"]) - float(previous["reflectance"]))
                maximum = 100 * max(abs(float(row[name]) - float(previous[name]))
                                    for name in METRICS)
                change = f"{signed:13.6g} {maximum:18.6g}"
            print(f"{value:5d} {reflection:10.6f} {substrate:12.6f} {absorption:12.6f} {change}")
            previous = row
    print(f"\nSaved results: {OUTPUT}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"),
                        help="Default: the device setting saved in the checkpoint.")
    parser.add_argument("--add-grids", default="512,576")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    saved = read_checkpoint()
    if args.report_only:
        show_result(saved)
        return 0
    try:
        added = sorted(set(int(item.strip()) for item in args.add_grids.split(",")))
        if not added or any(value < 32 or value % 2 for value in added):
            raise ValueError("Use comma-separated even grids >= 32.")
        values = extended_values(saved, added)
    except ValueError as error:
        parser.error(str(error))
    missing = sum(f"{value}|{wavelength:.12g}" not in saved["cases"]
                  for value in values for wavelength in saved["plan"]["wavelengths_nm"])
    device = args.device or saved["plan"]["solver"]["requested_device"]
    print(f"Grid plan: {values}; {missing} cases to calculate.", flush=True)
    result = subprocess.run(command(saved, values, device, args.prepare_only), check=False)
    if not args.prepare_only:
        show_result(read_checkpoint())
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
