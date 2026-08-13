"""Numerically validate radius/thickness autograd against central differences.

The scalar objective is the squared Frobenius norm of the forward-reflection
block.  Incidence is oblique for the full eigensolves to avoid symmetry-enforced
degeneracies; x/y sector tests use the normal-incidence symmetry reduction for
which they were derived.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torcwa

from rcwa_solver_auto import (
    ASROptions,
    AutoRCWA,
    Circle,
    GroupTheoryOptions,
    Lattice,
    LayerSpec,
    Material,
    NVMOptions,
    OutputSpec,
    UnsupportedCombinationError,
    _jinc,
)


@dataclass(frozen=True)
class Case:
    name: str
    method: str
    lattice: str
    polarization: str | None = None
    symmetry: str = "auto"


CASES = (
    Case("standard homogeneous thickness", "standard", "square"),
    Case("square NVM", "nvm", "square"),
    Case("triangular NVM", "nvm", "triangular"),
    Case("triangular NVM y sector", "nvm", "triangular", "y"),
    Case("oblique NVM x source sector", "nvm", "oblique", "x"),
    Case("triangular NVM complete D6", "nvm", "triangular", None, "d6"),
    Case("square matched-ASR", "matched-asr", "square"),
    Case("triangular matched-ASR", "matched-asr", "triangular"),
    Case("square matched-ASR x sector", "matched-asr", "square", "x"),
    Case("triangular matched-ASR y sector", "matched-asr", "triangular", "y"),
    Case(
        "triangular matched-ASR complete D6",
        "matched-asr",
        "triangular",
        None,
        "d6",
    ),
)


def _lattice(kind: str) -> Lattice:
    if kind == "square":
        return Lattice.square(1.0)
    if kind == "triangular":
        return Lattice.triangular(1.0)
    return Lattice.oblique((1.0, 0.0), (0.37, 1.07))


def objective(
    case: Case,
    radius: object,
    thickness: object,
    *,
    grid: int,
    order: int,
) -> torch.Tensor:
    sim = AutoRCWA(
        freq=1.0 / 1.55,
        order=[order, order],
        lattice=_lattice(case.lattice),
        outputs=OutputSpec(smatrix_size="half", fields="none"),
        asr=ASROptions(circle_G=0.08, grid=(grid, grid)),
        nvm=NVMOptions(grid=(grid, grid)),
        group_theory=GroupTheoryOptions(
            enabled=case.polarization is not None or case.symmetry == "d6",
            symmetry=case.symmetry,
            strict=case.polarization is not None or case.symmetry == "d6",
            polarization=case.polarization,
        ),
        verify_cascade=False,
        dtype=torch.complex128,
        device="cuda",
    )
    sim.add_input_layer(eps=1.0)
    sim.add_output_layer(eps=1.0)
    if case.polarization is None and case.symmetry != "d6":
        sim.set_incident_angle(0.13, 0.21)
    else:
        sim.set_incident_angle(0.0, 0.0)

    if case.method == "standard":
        sim.add_layer(thickness, eps=4.0, mu=1.0)
    else:
        sim.add_structured_layer(
            LayerSpec(
                thickness=thickness,
                geometry=Circle(radius),
                background=Material(1.0),
                inclusion=Material(4.0),
                method=case.method,
            )
        )
    sim.solve_global_smatrix()
    return torch.sum(torch.abs(sim.S[1]) ** 2)


def _scaled_error(autograd_value: float, finite_difference: float) -> float:
    return abs(autograd_value - finite_difference) / max(
        1.0, abs(autograd_value), abs(finite_difference)
    )


def internal_field_objective(
    case: Case,
    radius: object,
    thickness: object,
    *,
    grid: int,
    order: int,
) -> torch.Tensor:
    """Differentiable energy-like objective inside a reduced D6 layer."""
    sim = AutoRCWA(
        freq=1.0 / 1.55,
        order=[order, order],
        lattice=_lattice(case.lattice),
        outputs=OutputSpec(smatrix_size="quarter", fields="all"),
        asr=ASROptions(circle_G=0.08, grid=(grid, grid)),
        nvm=NVMOptions(grid=(grid, grid)),
        group_theory=GroupTheoryOptions(
            enabled=True,
            symmetry=case.symmetry,
            strict=True,
            polarization=case.polarization,
        ),
        verify_cascade=True,
        dtype=torch.complex128,
        device="cuda",
    )
    sim.add_input_layer(eps=1.0)
    sim.add_output_layer(eps=1.0)
    sim.set_incident_angle(0.0, 0.0)
    sim.add_structured_layer(
        LayerSpec(
            thickness=thickness,
            geometry=Circle(radius),
            background=Material(1.0),
            inclusion=Material(4.0),
            method=case.method,
        )
    )
    sim.solve_global_smatrix()
    zero_x = int(torch.nonzero(sim.order_x == 0)[0])
    zero_y = int(torch.nonzero(sim.order_y == 0)[0])
    harmonic = zero_x * len(sim.order_y) + zero_y
    source = torch.zeros(
        (2 * sim.order_N, 1), dtype=torch.complex128, device="cuda"
    )
    source[harmonic, 0] = 1.0
    sim.E_i = source
    sim.source_direction = "forward"
    fields = sim._fourier_fields(0, 0.5 * thickness)
    return sum(torch.sum(torch.abs(component) ** 2) for component in fields)


def check_internal_field_case(
    case: Case,
    *,
    grid: int,
    order: int,
    step: float,
    tolerance: float,
) -> dict[str, object]:
    radius = torch.tensor(0.22, dtype=torch.float64, requires_grad=True)
    thickness = torch.tensor(0.18, dtype=torch.float64, requires_grad=True)
    value = internal_field_objective(
        case, radius, thickness, grid=grid, order=order
    )
    radius_ad, thickness_ad = torch.autograd.grad(
        value, (radius, thickness)
    )
    radius_fd = (
        float(
            internal_field_objective(
                case, 0.22 + step, 0.18, grid=grid, order=order
            ).detach()
        )
        - float(
            internal_field_objective(
                case, 0.22 - step, 0.18, grid=grid, order=order
            ).detach()
        )
    ) / (2.0 * step)
    thickness_fd = (
        float(
            internal_field_objective(
                case, 0.22, 0.18 + step, grid=grid, order=order
            ).detach()
        )
        - float(
            internal_field_objective(
                case, 0.22, 0.18 - step, grid=grid, order=order
            ).detach()
        )
    ) / (2.0 * step)
    radius_ad_value = float(radius_ad)
    thickness_ad_value = float(thickness_ad)
    radius_error = _scaled_error(radius_ad_value, radius_fd)
    thickness_error = _scaled_error(thickness_ad_value, thickness_fd)
    return {
        "name": case.name + " internal fields",
        "objective": float(value.detach()),
        "radius": {
            "autograd": radius_ad_value,
            "finite_difference": radius_fd,
            "scaled_error": radius_error,
            "passed": math.isfinite(radius_ad_value)
            and abs(radius_ad_value) > 1.0e-10
            and radius_error <= tolerance,
        },
        "thickness": {
            "autograd": thickness_ad_value,
            "finite_difference": thickness_fd,
            "scaled_error": thickness_error,
            "passed": math.isfinite(thickness_ad_value)
            and abs(thickness_ad_value) > 1.0e-10
            and thickness_error <= tolerance,
        },
    }


def check_case(
    case: Case,
    *,
    grid: int,
    order: int,
    step: float,
    tolerance: float,
) -> dict[str, object]:
    radius = torch.tensor(0.22, dtype=torch.float64, requires_grad=True)
    thickness = torch.tensor(0.18, dtype=torch.float64, requires_grad=True)
    value = objective(case, radius, thickness, grid=grid, order=order)
    requested = (thickness,) if case.method == "standard" else (radius, thickness)
    gradients = torch.autograd.grad(value, requested)

    result: dict[str, object] = {
        "name": case.name,
        "method": case.method,
        "lattice": case.lattice,
        "polarization": case.polarization,
        "symmetry": case.symmetry,
        "objective": float(value.detach()),
        "parameters": {},
    }
    parameters = result["parameters"]
    assert isinstance(parameters, dict)

    if case.method != "standard":
        radius_fd = (
            float(
                objective(case, 0.22 + step, 0.18, grid=grid, order=order).detach()
            )
            - float(
                objective(case, 0.22 - step, 0.18, grid=grid, order=order).detach()
            )
        ) / (2.0 * step)
        radius_ad = float(gradients[0])
        radius_error = _scaled_error(radius_ad, radius_fd)
        parameters["radius"] = {
            "autograd": radius_ad,
            "finite_difference": radius_fd,
            "scaled_error": radius_error,
            "passed": math.isfinite(radius_ad)
            and abs(radius_ad) > 1.0e-10
            and radius_error <= tolerance,
        }
        thickness_gradient = gradients[1]
    else:
        thickness_gradient = gradients[0]

    thickness_fd = (
        float(objective(case, 0.22, 0.18 + step, grid=grid, order=order).detach())
        - float(objective(case, 0.22, 0.18 - step, grid=grid, order=order).detach())
    ) / (2.0 * step)
    thickness_ad = float(thickness_gradient)
    thickness_error = _scaled_error(thickness_ad, thickness_fd)
    parameters["thickness"] = {
        "autograd": thickness_ad,
        "finite_difference": thickness_fd,
        "scaled_error": thickness_error,
        "passed": math.isfinite(thickness_ad)
        and abs(thickness_ad) > 1.0e-10
        and thickness_error <= tolerance,
    }
    result["passed"] = all(bool(item["passed"]) for item in parameters.values())
    return result


def check_jinc(step: float, tolerance: float) -> dict[str, object]:
    x = torch.tensor(1.2, dtype=torch.float64, requires_grad=True)
    value = _jinc(x)
    derivative = float(torch.autograd.grad(value, x)[0])
    finite_difference = float(
        (_jinc(torch.tensor(1.2 + step)) - _jinc(torch.tensor(1.2 - step)))
        / (2.0 * step)
    )
    error = _scaled_error(derivative, finite_difference)
    return {
        "argument": 1.2,
        "autograd": derivative,
        "finite_difference": finite_difference,
        "scaled_error": error,
        "passed": math.isfinite(derivative) and error <= tolerance,
    }


def check_geometry_contract(grid: int, order: int) -> dict[str, object]:
    radius = torch.tensor(0.22, dtype=torch.float64, requires_grad=True)
    retained = Circle(radius).radius is radius
    rejected_hard_raster = False
    sim = AutoRCWA(
        freq=1.0 / 1.55,
        order=[order, order],
        lattice=Lattice.square(1.0),
        outputs=OutputSpec(smatrix_size="quarter", fields="none"),
        asr=ASROptions(grid=(grid, grid)),
        verify_cascade=False,
        dtype=torch.complex128,
        device="cuda",
    )
    sim.add_input_layer()
    sim.add_output_layer()
    sim.set_incident_angle(0.13, 0.21)
    try:
        sim.add_structured_layer(
            LayerSpec(
                thickness=0.18,
                geometry=Circle(radius),
                background=Material(1.0),
                inclusion=Material(4.0),
                method="standard",
            )
        )
    except UnsupportedCombinationError:
        rejected_hard_raster = True
    return {
        "circle_retains_tensor_identity": retained,
        "hard_raster_rejected_for_trainable_radius": rejected_hard_raster,
        "passed": retained and rejected_hard_raster,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--order", type=int, default=1)
    parser.add_argument("--step", type=float, default=1.0e-5)
    parser.add_argument("--tolerance", type=float, default=3.0e-6)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    case_results = [
        check_case(
            case,
            grid=args.grid,
            order=args.order,
            step=args.step,
            tolerance=args.tolerance,
        )
        for case in CASES
    ]
    internal_field_results = [
        check_internal_field_case(
            case,
            grid=args.grid,
            order=args.order,
            step=args.step,
            tolerance=args.tolerance,
        )
        for case in CASES
        if case.symmetry == "d6"
    ]
    jinc_result = check_jinc(args.step, args.tolerance)
    contract_result = check_geometry_contract(args.grid, args.order)
    passed = (
        all(bool(case["passed"]) for case in case_results)
        and all(
            bool(item["radius"]["passed"])
            and bool(item["thickness"]["passed"])
            for item in internal_field_results
        )
        and bool(jinc_result["passed"])
        and bool(contract_result["passed"])
    )
    payload = {
        "status": "passed" if passed else "failed",
        "torch_version": torch.__version__,
        "torcwa_version": getattr(torcwa, "__version__", "unknown"),
        "grid": args.grid,
        "order": args.order,
        "central_difference_step": args.step,
        "scaled_error_tolerance": args.tolerance,
        "cases": case_results,
        "internal_field_cases": internal_field_results,
        "jinc": jinc_result,
        "geometry_contract": contract_result,
    }
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    print(rendered)
    if args.json is not None:
        args.json.write_text(rendered + "\n", encoding="utf-8")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
