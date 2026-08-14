"""Convergence versus z-slice count for Au-coated PMMA moth-eye."""

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
        description="Refine only the profile slice count Nz at fixed Fourier order."
    )
    add_shared_arguments(parser)
    parser.add_argument("--layers", default="8,12,16,24,32,48")
    parser.add_argument("--fixed-order", type=int, default=5)
    parser.add_argument("--grid", type=int, default=192)
    parser.add_argument(
        "--output-prefix", type=Path, default=Path("pmma_gold_layers")
    )
    args = parser.parse_args()
    candidates = parse_int_list(args.layers, "layers")
    fixed = NumericalConfig(order=args.fixed_order, slices=candidates[0], grid=args.grid)
    report, path = run_axis_convergence(
        axis="slices", candidates=candidates, fixed=fixed, args=args
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
