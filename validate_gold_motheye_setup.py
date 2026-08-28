"""Validate Au dispersion and optionally one minimal matched-ASR moth-eye case."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path

from gold_dispersion import gold_epsilon_rakic_ld, load_gold_csv


REFERENCE = {
    400.0: complex(-1.0611633694991678, 4.9206873818569425),
    550.0: complex(-5.371371338856619, 2.3581635829859637),
    700.0: complex(-13.754263491271207, 1.9104862668338753),
}


def core_checks() -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    for wavelength, reference in REFERENCE.items():
        value = gold_epsilon_rakic_ld(wavelength)
        error = abs(value - reference)
        checks.append(
            {
                "name": f"Rakic LD {wavelength:g} nm",
                "error": error,
                "limit": 2.0e-12,
                "passed": error <= 2.0e-12 and value.imag > 0.0,
            }
        )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "gold.csv"
        path.write_text(
            "wavelength_nm,n,k\n400,1.4,1.7\n700,0.25,3.7\n",
            encoding="utf-8",
        )
        table = load_gold_csv(path)
        midpoint = table(550.0)
        expected = 0.5 * (
            complex(1.4, 1.7) ** 2 + complex(0.25, 3.7) ** 2
        )
        error = abs(midpoint - expected)
        extrapolation_rejected = False
        try:
            table(399.0)
        except ValueError:
            extrapolation_rejected = True
        checks.append(
            {
                "name": "tabulated n,k interpolation and no extrapolation",
                "error": error,
                "limit": 2.0e-14,
                "passed": error <= 2.0e-14 and extrapolation_rejected,
            }
        )
    convergence_source = (Path(__file__).parent / "converge_gold_motheye.py").read_text(
        encoding="utf-8"
    )
    checks.append(
        {
            "name": "complete-D6 E1 source-row reduction wired by default",
            "error": 0.0,
            "limit": 0.0,
            "passed": (
                'symmetry_reduction: str = "d6-source"' in convergence_source
                and 'symmetry="d6" if symmetry_reduction == "d6-source"'
                in convergence_source
                and 'default="d6-source"' in convergence_source
            ),
        }
    )
    return checks


def integration_check(device: str) -> dict[str, object]:
    import torch

    from converge_gold_motheye import (
        GeometryConfig,
        NumericalConfig,
        simulate_case,
    )

    result = simulate_case(
        550.0,
        NumericalConfig(order=1, slices=2, grid=32),
        GeometryConfig(
            period_nm=200.0,
            height_nm=500.0,
            tip_radius_nm=10.0,
            base_radius_nm=90.0,
        ),
        gold_epsilon_rakic_ld,
        cascade="redheffer",
        use_symmetry=True,
        device=torch.device(device),
    )
    cs_result = simulate_case(
        550.0,
        NumericalConfig(order=1, slices=2, grid=32),
        GeometryConfig(
            period_nm=200.0,
            height_nm=500.0,
            tip_radius_nm=10.0,
            base_radius_nm=90.0,
        ),
        gold_epsilon_rakic_ld,
        cascade="redheffer",
        use_symmetry=True,
        symmetry_reduction="cs-source",
        device=torch.device(device),
    )
    reflectance = float(result["reflectance"])
    total = float(result["absorptance_total"])
    moth = float(result["motheye_absorptance"])
    substrate = float(result["substrate_absorptance"])
    partition_error = max(abs(reflectance + total - 1.0), abs(moth + substrate - total))
    d6_cs_error = max(
        abs(float(result[name]) - float(cs_result[name]))
        for name in (
            "reflectance",
            "absorptance_total",
            "motheye_absorptance",
            "substrate_absorptance",
        )
    )
    passed = (
        all(math.isfinite(value) for value in (reflectance, total, moth, substrate))
        and not bool(result["passivity_warning"])
        and partition_error <= 2.0e-9
        and int(result["reduced_dimension"]) == 3
        and int(cs_result["reduced_dimension"]) == 7
        and d6_cs_error <= 2.0e-4
    )
    return {
        "name": "minimal matched-ASR semi-infinite Au energy partition",
        "error": partition_error,
        "limit": 2.0e-9,
        "passed": passed,
        "expected_d6_e1_dimension": 3,
        "expected_cs_dimension": 7,
        "d6_cs_max_observable_error": d6_cs_error,
        "result": result,
        "cs_result": cs_result,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--integration", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    checks = core_checks()
    if args.integration:
        checks.append(integration_check(args.device))
    payload = {
        "passed": all(bool(check["passed"]) for check in checks),
        "passed_count": sum(bool(check["passed"]) for check in checks),
        "total_count": len(checks),
        "checks": checks,
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    print(text)
    if args.json:
        args.json.write_text(text, encoding="utf-8")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
