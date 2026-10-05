"""Extend the existing M=16 Au moth-eye grid sweep to 384 and 448.

Run from the project root with ``python3 studies/gold_motheye3/run_next.py
--device cuda``. This wrapper refuses to start a new sweep: the completed
96..320 checkpoint must already be present. It does not replace model, Au CSV,
or RCWA sources, and it keeps all previously computed cases.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "results" / "grid_M16_Nz100"
OLD_VALUES = [96, 128, 192, 256, 320]
NEW_VALUES = OLD_VALUES + [384, 448]
WAVELENGTHS = [400.0, 550.0, 700.0]
METRICS = ("reflectance", "power_into_substrate", "motheye_absorptance")


def validate_checkpoint(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Existing grid checkpoint is required: {path}. "
            "Place run_next.py in the same gold_motheye3 folder used for the first run."
        )
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    plan = checkpoint.get("plan", {})
    values = plan.get("values")
    expected = {f"{value}|{wavelength:g}"
                for value in OLD_VALUES for wavelength in WAVELENGTHS}
    cases = checkpoint.get("cases", {})
    if not isinstance(cases, dict) or not expected.issubset(cases):
        raise ValueError("The original 15 grid cases are not all saved in checkpoint.json.")
    if (plan.get("axis") != "grid"
            or values not in (OLD_VALUES, NEW_VALUES)
            or plan.get("wavelengths_nm") != WAVELENGTHS
            or plan.get("fixed_numerics") != {"order": 16, "slices": 100, "grid": 256}
            or plan.get("solver", {}).get("cascade") != "redheffer"
            or plan.get("solver", {}).get("symmetry_reduction") != "d6-source"
            or plan.get("tolerance") != 0.005):
        raise ValueError("The saved grid sweep differs from the expected M=16, Nz=100 setup.")
    return checkpoint


def command(device: str, action: str | None = None) -> list[str]:
    values = ",".join(str(value) for value in NEW_VALUES)
    result = [sys.executable, str(HERE / "converge.py"),
              "--axis", "grid", "--values", values,
              "--order", "16", "--slices", "100", "--grid", "256",
              "--wavelengths", "400,550,700", "--tolerance", "0.005",
              "--cascade", "redheffer", "--device", device]
    if action:
        result.append(action)
    return result


def show_result(report_path: Path) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    print(f"grid status: {report['status']} "
          f"({report['completed_cases']}/{report['expected_cases']} cases)")
    entries = report["adjacent_changes"].get("700", [])
    for entry in entries[-2:]:
        changes = entry["absolute_changes"]
        numbers = ", ".join(f"{name}={100 * changes[name]:.6g} pp"
                            for name in METRICS)
        print(f"700 nm, grid {entry['low']} -> {entry['high']}: {numbers}; "
              f"pass={entry['passes_tolerance']}")
    if report["status"] == "converged_within_tested_values":
        print("M=16 grid criterion passed. Check grid again at the final Fourier order.")
    elif report["status"] == "not_converged":
        print("Grid convergence is still unconfirmed at 700 nm.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.prepare_only and args.report_only:
        parser.error("Choose only one of --prepare-only and --report-only.")
    validate_checkpoint(OUTPUT / "checkpoint.json")
    action = "--prepare-only" if args.prepare_only else "--report-only" if args.report_only else None
    print("Extending the saved grid sweep from 320 to 384 and 448.", flush=True)
    result = subprocess.run(command(args.device, action), check=False)
    if result.returncode:
        if not action:
            print("Solve stopped; rebuilding a report from saved cases.", flush=True)
            subprocess.run(command(args.device, "--report-only"), check=False)
        return result.returncode
    if not args.prepare_only:
        show_result(OUTPUT / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
