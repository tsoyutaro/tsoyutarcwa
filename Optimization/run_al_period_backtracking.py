"""Run independent Al H500/Nz15 Optuna studies with Adam backtracking."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_al_p100_backtracking import (
    EXPECTED_TRAINING,
    ROOT,
    SOURCE_CASE,
    prepare_recovery_case,
    run_optimization,
)

PERIODS = (150, 200, 250, 300, 350)


def prepare_period(root, period, order=8):
    if order <= 0 or 512 < 4 * order + 4:
        raise ValueError("A positive order supported by grid 512 is required.")
    geometry = f"P{period}_H500_Nz15"
    source = root / SOURCE_CASE
    if period != 200:
        source = source / "geometry_cases" / geometry
    config_path = source / "config_hex_OFF_optuna30_M8_Q512.json"
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    expected = dict(EXPECTED_TRAINING, period_nm=period)
    differences = [key for key, value in expected.items() if config.get(key) != value]
    if differences:
        raise ValueError(f"Unexpected training conditions in {config_path}: {', '.join(differences)}")
    config["optimization_order"] = order
    if order > max(config["verification_orders"]):
        config["verification_orders"] = sorted({*config["verification_orders"], order})
        config["verification_grid"] = max(config["verification_grid"], 4 * order + 4)
    suffix = "" if order == 8 else f"_M{order}"
    relative_case = Path("Al") / f"al_motheye_{geometry}_backtracking_v1{suffix}"
    return prepare_recovery_case(root, relative_case, config)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period-nm", type=int, choices=PERIODS, required=True)
    parser.add_argument("--order", type=int, default=8, help="Fixed optimization order (default: 8)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Write the case and run manifest without starting optimization")
    parser.add_argument("--recover-lock", action="store_true",
                        help="Recover the run lock after its controller and worker have stopped")
    args = parser.parse_args(argv)
    try:
        case = prepare_period(ROOT, args.period_nm, args.order)
        output = case / "results" / f"hex_OFF_optuna30_backtracking_v1_M{args.order}_Q512"
        print(f"Al P={args.period_nm} H=500 nm: OFF M={args.order} Q=512, 30 trials; output: {output}",
              flush=True)
        return run_optimization(case, output, args)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
