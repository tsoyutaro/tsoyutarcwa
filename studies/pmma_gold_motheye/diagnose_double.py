"""Small control matrix for diagnosing double-matched PMMA/Au calculations."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

_OUTPUTS_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_ROOT = Path(__file__).resolve().parent
if str(_OUTPUTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_OUTPUTS_ROOT))

from studies.pmma_gold_motheye.common import (
    GeometryConfig,
    NumericalConfig,
    simulate_case,
)
from studies.shared.gold_dispersion import gold_epsilon_rakic_ld


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wavelength", type=float, default=550.0)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--slices", type=int, default=8)
    parser.add_argument("--grid", type=int, default=96)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=_PACKAGE_ROOT / "results" / "diagnostics" / "double_controls.json",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    geometry = GeometryConfig()

    controls = (
        ("outer_d6_redheffer", "outer", True, "d6-source", True, "redheffer"),
        ("double_d6_redheffer", "double", True, "d6-source", True, "redheffer"),
        ("double_cs_redheffer", "double", True, "cs-source", True, "redheffer"),
        ("double_full_redheffer", "double", False, "d6-source", True, "redheffer"),
        ("double_d6_direct", "double", True, "d6-source", False, "redheffer"),
        ("double_d6_algo2a", "double", True, "d6-source", True, "algo2a"),
    )
    results: dict[str, dict[str, object]] = {}
    for name, mapping, symmetry, reduction, factorization, cascade in controls:
        print(f"solve diagnostic: {name}", flush=True)
        results[name] = simulate_case(
            args.wavelength,
            NumericalConfig(args.order, args.slices, args.grid, mapping),
            geometry,
            gold_epsilon_rakic_ld,
            cascade=cascade,
            use_symmetry=symmetry,
            symmetry_reduction=reduction,
            factorization_rules=factorization,
            device=device,
        )

    reference = results["double_full_redheffer"]
    for result in results.values():
        result["max_rta_difference_from_double_full"] = max(
            abs(float(result[key]) - float(reference[key]))
            for key in ("reflectance", "transmittance", "absorptance")
        )
    def rta_difference(left: str, right: str) -> float:
        return max(
            abs(float(results[left][key]) - float(results[right][key]))
            for key in ("reflectance", "transmittance", "absorptance")
        )

    comparisons = {
        "redheffer_vs_algo2a": rta_difference(
            "double_d6_redheffer", "double_d6_algo2a"
        ),
        "d6_vs_cs": rta_difference("double_d6_redheffer", "double_cs_redheffer"),
        "d6_vs_full_rectangular": rta_difference(
            "double_d6_redheffer", "double_full_redheffer"
        ),
        "generalized_li_vs_direct": rta_difference(
            "double_d6_redheffer", "double_d6_direct"
        ),
        "outer_vs_double": rta_difference(
            "outer_d6_redheffer", "double_d6_redheffer"
        ),
    }
    passivity = all(
        not bool(result["passivity_warning"]) for result in results.values()
    )
    finite = all(
        math.isfinite(float(result[key]))
        for result in results.values()
        for key in ("reflectance", "transmittance", "absorptance")
    )
    payload = {
        "passed": passivity and finite and comparisons["redheffer_vs_algo2a"] <= 2.0e-8,
        "purpose": (
            "Separate map, generalized-Li, D6/Cs source reduction, full basis, "
            "and Redheffer/algo2a effects before an expensive convergence run."
        ),
        "settings": {
            "wavelength_nm": args.wavelength,
            "order": args.order,
            "slices": args.slices,
            "grid": args.grid,
            "device": str(device),
        },
        "comparisons": comparisons,
        "strict_comparison": {
            "name": "redheffer_vs_algo2a",
            "limit": 2.0e-8,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"report: {args.output.resolve()}")
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
