"""Run the three independent Au moth-eye convergence sweeps in sequence."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.prepare_only and args.report_only:
        parser.error("Choose only one of --prepare-only and --report-only.")
    script = Path(__file__).with_name("converge.py")
    for axis in ("order", "slices", "grid"):
        command = [sys.executable, str(script), "--axis", axis,
                   "--device", args.device]
        if args.prepare_only:
            command.append("--prepare-only")
        if args.report_only:
            command.append("--report-only")
        print(f"running {axis} sweep", flush=True)
        result = subprocess.run(command, check=False)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
