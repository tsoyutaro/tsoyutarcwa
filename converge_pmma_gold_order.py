"""Convergence versus Fourier diffraction order for Au-coated PMMA moth-eye."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pmma_gold_motheye_common import (
    NumericalConfig,
    add_shared_arguments,
    parse_int_list,
    run_axis_convergence,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refine only the Fourier order M at fixed z-slice count."
    )
    add_shared_arguments(parser)
    parser.add_argument("--orders", default="2,3,4,5,6")
    parser.add_argument("--fixed-slices", type=int, default=32)
    parser.add_argument("--grid", type=int, default=192)
    parser.add_argument(
        "--output-prefix", type=Path, default=Path("pmma_gold_order")
    )
    args = parser.parse_args()
    candidates = parse_int_list(args.orders, "orders")
    fixed = NumericalConfig(order=candidates[0], slices=args.fixed_slices, grid=args.grid)
    report, path = run_axis_convergence(
        axis="order", candidates=candidates, fixed=fixed, args=args
    )
    print(
        json.dumps(
            {"status": report["status"], "recommendation": report["recommendation"]},
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"report: {path.resolve()}")
    return 0 if report["status"] == "converged" else 2


if __name__ == "__main__":
    raise SystemExit(main())
