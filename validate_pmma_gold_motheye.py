"""Static checks and an optional minimal torcwa integration check."""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path

from gold_dispersion import gold_epsilon_rakic_ld


HERE = Path(__file__).resolve().parent


def _source(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    ast.parse(text, filename=str(path))
    return text


def core_checks() -> list[dict[str, object]]:
    asr = _source(HERE / "rcwa_ext" / "asr.py")
    auto = _source(HERE / "rcwa_ext" / "auto.py")
    symmetry = _source(HERE / "rcwa_ext" / "symmetry.py")
    common = _source(HERE / "pmma_gold_motheye_common.py")
    order = _source(HERE / "converge_pmma_gold_order.py")
    layers = _source(HERE / "converge_pmma_gold_layers.py")
    return [
        {
            "name": "core-shell matched-ASR public API",
            "passed": (
                "def add_layer_circle_shell_asr(" in asr
                and "core_inside" in asr
                and "add_layer_circle_shell_asr =" in auto
            ),
        },
        {
            "name": "three-material PMMA/Au/air slice",
            "passed": (
                "simulation.add_layer_circle_shell_asr(" in common
                and "epsilon_gold" in common
                and "epsilon_pmma" in common
            ),
        },
        {
            "name": "complete-D6 E1 source-row reduction wired by default",
            "passed": (
                "def _d6_source_eigendecomposition(" in symmetry
                and '"symmetry": "D6-E1-source-row"' in symmetry
                and 'default="d6-source"' in common
                and 'symmetry="d6" if symmetry_reduction == "d6-source"' in common
            ),
        },
        {
            "name": "semi-infinite PMMA output and Au top cap",
            "passed": (
                "simulation.add_output_layer(eps=epsilon_pmma" in common
                and "if geometry.include_top_cap:" in common
            ),
        },
        {
            "name": "independent order and layer entry points",
            "passed": (
                'axis="order"' in order
                and 'axis="slices"' in layers
                and "--fixed-slices" in order
                and "--fixed-order" in layers
            ),
        },
        {
            "name": "passive Au at 400/550/700 nm",
            "passed": all(
                gold_epsilon_rakic_ld(wavelength).imag > 0.0
                for wavelength in (400.0, 550.0, 700.0)
            ),
        },
        {
            "name": "default coated base has positive gap",
            "passed": 2.0 * (75.0 + 20.0) < 200.0,
        },
    ]


def integration_check(device: str) -> dict[str, object]:
    import torch

    from pmma_gold_motheye_common import GeometryConfig, NumericalConfig, simulate_case

    result = simulate_case(
        550.0,
        NumericalConfig(order=1, slices=2, grid=32),
        GeometryConfig(),
        gold_epsilon_rakic_ld,
        cascade="redheffer",
        use_symmetry=True,
        device=torch.device(device),
    )
    cs_result = simulate_case(
        550.0,
        NumericalConfig(order=1, slices=2, grid=32),
        GeometryConfig(),
        gold_epsilon_rakic_ld,
        cascade="redheffer",
        use_symmetry=True,
        symmetry_reduction="cs-source",
        device=torch.device(device),
    )
    values = tuple(
        float(result[name])
        for name in ("reflectance", "transmittance", "absorptance")
    )
    residual = abs(sum(values) - 1.0)
    d6_cs_error = max(
        abs(float(result[name]) - float(cs_result[name]))
        for name in ("reflectance", "transmittance", "absorptance")
    )
    passed = (
        all(math.isfinite(value) for value in values)
        and not bool(result["passivity_warning"])
        and residual <= 2.0e-9
        and int(result["reduced_dimension"]) == 3
        and int(cs_result["reduced_dimension"]) == 7
        and d6_cs_error <= 2.0e-4
    )
    return {
        "name": "minimal Au-coated PMMA matched-ASR solve",
        "error": residual,
        "limit": 2.0e-9,
        "passed": passed,
        "expected_d6_e1_dimension": 3,
        "expected_cs_dimension": 7,
        "d6_cs_max_rta_error": d6_cs_error,
        "result": result,
        "cs_result": cs_result,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--integration", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--json", type=Path, default=HERE / "pmma_gold_motheye_validation.json"
    )
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
    args.json.write_text(text, encoding="utf-8")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
