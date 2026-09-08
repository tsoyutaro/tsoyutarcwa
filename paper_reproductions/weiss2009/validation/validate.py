"""Static and numerical checks for the Weiss-2009 reproduction."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_OUTPUTS_ROOT = Path(__file__).resolve().parents[3]
if str(_OUTPUTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_OUTPUTS_ROOT))

from paper_reproductions.weiss2009.reproduce import (
    DIELECTRIC_EPSILON,
    DIELECTRIC_FREQUENCY_THZ,
    DIELECTRIC_HEIGHT_UM,
    DIELECTRIC_PERIOD_UM,
    DIELECTRIC_RADIUS_UM,
    INTERFACE_SLOPE,
    analytical_dielectric_rod_beta,
    gold_drude_epsilon,
    select_device,
    simulate_scattering,
)
from rcwa_ext import CustomRCWA_ASR_FR


def check(name: str, condition: bool, detail: object) -> dict[str, object]:
    return {"name": name, "passed": bool(condition), "detail": detail}


def mapping_checks(device: torch.device) -> list[dict[str, object]]:
    simulation = CustomRCWA_ASR_FR(
        DIELECTRIC_FREQUENCY_THZ / 299.792458,
        [1, 1],
        [DIELECTRIC_PERIOD_UM, DIELECTRIC_PERIOD_UM],
        dtype=torch.complex128,
        device=device,
        matched_asr_G=INTERFACE_SLOPE,
        matched_asr_profile="weiss2009",
        smatrix_size="half",
        store_mode_couplings=False,
    )
    displacement = DIELECTRIC_RADIUS_UM / math.sqrt(2.0)
    expected_breaks = torch.tensor(
        [
            0.0,
            DIELECTRIC_PERIOD_UM / 2.0 - displacement,
            DIELECTRIC_PERIOD_UM / 2.0 + displacement,
            DIELECTRIC_PERIOD_UM,
        ],
        dtype=torch.float64,
        device=device,
    )
    u, mapped, derivative, returned_breaks = simulation._weiss2009_asr_map(
        DIELECTRIC_PERIOD_UM, expected_breaks, 65536
    )
    interface_indices = [
        int(torch.argmin(torch.abs(u - expected_breaks[index])).cpu())
        for index in (1, 2)
    ]
    interface_slopes = [float(derivative[index].cpu()) for index in interface_indices]
    map2d = simulation.build_circle_asr_mapping(
        96, 96, DIELECTRIC_RADIUS_UM
    )
    identity = CustomRCWA_ASR_FR(
        DIELECTRIC_FREQUENCY_THZ / 299.792458,
        [1, 1],
        [DIELECTRIC_PERIOD_UM, DIELECTRIC_PERIOD_UM],
        dtype=torch.complex128,
        device=device,
        matched_asr_profile="identity",
        smatrix_size="half",
        store_mode_couplings=False,
    ).build_circle_asr_mapping(64, 64, DIELECTRIC_RADIUS_UM)
    return [
        check(
            "Eq. 41 fixed interface coordinates",
            bool(torch.allclose(returned_breaks, expected_breaks, atol=1.0e-13, rtol=0.0)),
            [float(value) for value in returned_breaks.cpu()],
        ),
        check(
            "Eq. 42 interface slope equals 1-eta",
            max(abs(value - INTERFACE_SLOPE) for value in interface_slopes) < 2.0e-4,
            interface_slopes,
        ),
        check(
            "one-dimensional map is finite and monotone",
            bool(torch.all(torch.isfinite(mapped))) and bool(torch.all(derivative > 0.0)),
            {
                "minimum_derivative": float(torch.min(derivative).cpu()),
                "maximum_derivative": float(torch.max(derivative).cpu()),
            },
        ),
        check(
            "two-dimensional matched-ASR Jacobian is positive",
            bool(torch.all(torch.isfinite(map2d.det_j))) and bool(torch.all(map2d.det_j > 0.0)),
            float(torch.min(map2d.det_j).cpu()),
        ),
        check(
            "identity profile leaves matched coordinates unstretched",
            bool(torch.equal(identity.u, identity.tu)) and bool(torch.equal(identity.v, identity.tv)),
            "u=tu and v=tv",
        ),
    ]


def numerical_checks(device: torch.device) -> list[dict[str, object]]:
    beta = analytical_dielectric_rod_beta()
    epsilon_gold = gold_drude_epsilon(370.0)
    rows = [
        simulate_scattering(
            frequency_thz=DIELECTRIC_FREQUENCY_THZ,
            order=1,
            period_um=DIELECTRIC_PERIOD_UM,
            radius_um=DIELECTRIC_RADIUS_UM,
            height_um=DIELECTRIC_HEIGHT_UM,
            epsilon=DIELECTRIC_EPSILON,
            profile=profile,
            grid=32,
            dtype=torch.complex128,
            device=device,
            cascade="redheffer",
            both_polarizations=True,
        )
        for profile in ("identity", "weiss2009")
    ]
    finite_rows = all(
        math.isfinite(float(row[key]))
        for row in rows
        for key in ("R_x", "T_x", "A_x", "R_y", "T_y", "A_y")
    )
    return [
        check(
            "analytical HE11 reference agrees with paper's rounded 25 1/um",
            25.0 < beta < 26.0,
            beta,
        ),
        check(
            "Drude gold has positive loss for exp(-i omega t)",
            epsilon_gold.imag > 0.0 and epsilon_gold.real < 0.0,
            [epsilon_gold.real, epsilon_gold.imag],
        ),
        check("low-order scattering is finite", finite_rows, rows),
        check(
            "symmetric factorization preserves x/y transmission",
            max(float(row["polarization_delta_T"]) for row in rows) < 1.0e-10,
            [float(row["polarization_delta_T"]) for row in rows],
        ),
    ]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    result.add_argument(
        "--output",
        type=Path,
        default=_PACKAGE_ROOT / "validation" / "results" / "validation.json",
    )
    return result


def main(args: argparse.Namespace) -> int:
    device = select_device(args.device)
    checks = mapping_checks(device) + numerical_checks(device)
    payload = {
        "paper_doi": "10.1364/OE.17.008051",
        "device": str(device),
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"wrote {args.output}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main(parser().parse_args()))
