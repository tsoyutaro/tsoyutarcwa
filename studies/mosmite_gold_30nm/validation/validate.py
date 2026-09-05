"""Dependency-light geometry and wiring checks for the 30 nm study."""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from pathlib import Path

_PACKAGE = Path(__file__).resolve().parents[1]
_OUTPUTS_ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integration", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    common = (_PACKAGE / "common.py").read_text(encoding="utf-8")
    run = (_PACKAGE / "run_spectrum.py").read_text(encoding="utf-8")
    converge = (_PACKAGE / "converge.py").read_text(encoding="utf-8")
    for source in (common, run, converge):
        ast.parse(source)

    period = 200.0
    slices = 32
    guard = 2.0
    radii = [5.0 + 70.0 * ((index + 0.5) / slices) for index in range(slices)]
    outer = [radius + 30.0 for radius in radii]
    limit = 0.5 * (period - guard)
    asr_count = sum(radius < limit for radius in outer)
    raster_count = slices - asr_count
    covering_radius = period / math.sqrt(3.0)
    checks = [
        {"name": "hybrid ASR/raster branch", "passed": "PERIODIC_AU_COALESCENCE" in common},
        {"name": "monotone double map requested", "passed": 'radial_mapping="double"' in common},
        {"name": "30 nm default coating", "passed": "gold_thickness_nm: float = 30.0" in common},
        {"name": "isolated and connected slices both present", "passed": asr_count > 0 and raster_count > 0},
        {"name": "residual air pockets remain at base", "passed": max(outer) < covering_radius},
        {"name": "independent convergence axes", "passed": '("order", "slices", "grid")' in converge},
        {"name": "R/T/A spectrum output", "passed": '"absorptance"' in common and "figure.savefig" in run},
    ]
    integration = None
    if args.integration:
        if str(_OUTPUTS_ROOT) not in sys.path:
            sys.path.insert(0, str(_OUTPUTS_ROOT))
        import torch

        from studies.mosmite_gold_30nm.common import (
            GeometryConfig,
            NumericalConfig,
            simulate_case,
        )
        from studies.shared.gold_dispersion import gold_epsilon_rakic_ld

        solved = simulate_case(
            550.0,
            NumericalConfig(order=1, slices=8, grid=48),
            GeometryConfig(),
            gold_epsilon_rakic_ld,
            device=torch.device(args.device),
        )
        integration_passed = bool(
            solved["asr_layers"] > 0
            and solved["coalesced_raster_layers"] > 0
            and all(
                math.isfinite(float(solved[name]))
                for name in ("reflectance", "transmittance", "absorptance")
            )
            and not solved["passivity_warning"]
        )
        integration = {
            "name": "mixed ASR/coalesced-raster S-matrix solve",
            "passed": integration_passed,
            "result": solved,
        }
        checks.append(integration)
    payload = {
        "passed": all(item["passed"] for item in checks),
        "passed_count": sum(bool(item["passed"]) for item in checks),
        "total_count": len(checks),
        "default_32_slice_diagnostics": {
            "asr_layers": asr_count,
            "coalesced_raster_layers": raster_count,
            "largest_midpoint_outer_radius_nm": max(outer),
            "triangular_covering_radius_nm": covering_radius,
        },
        "checks": checks,
    }
    if integration is not None:
        payload["integration"] = integration
    result = _PACKAGE / "validation" / "validation.json"
    result.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
