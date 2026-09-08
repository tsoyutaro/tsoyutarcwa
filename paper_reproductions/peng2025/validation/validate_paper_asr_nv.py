"""Focused regression checks for Peng--Zhang 2025 Eqs. (7)--(10)."""

from __future__ import annotations

import argparse
import cmath
import json
import math
import sys
from pathlib import Path

import torch

_VALIDATION_ROOT = Path(__file__).resolve().parent
_OUTPUTS_ROOT = Path(__file__).resolve().parents[3]
if str(_OUTPUTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_OUTPUTS_ROOT))

from paper_reproductions.peng2025.common import (
    Numerics,
    PaperGeometry,
    SilverDrude,
    select_device,
    simulate_matched_primitive,
)
from rcwa_solver_auto import ASROptions, AutoRCWA, Lattice, OutputSpec


def _simulation(device: torch.device, *, order: int = 1, g: float = 1.0e-3) -> AutoRCWA:
    simulation = AutoRCWA(
        freq=0.2,
        order=[order, order],
        lattice=Lattice.square(1.0),
        outputs=OutputSpec(smatrix_size="quarter", fields="none"),
        asr=ASROptions(G=g, grid=(96, 96)),
        verify_cascade=False,
        dtype=torch.complex128,
        device=device,
    )
    simulation.add_input_layer(eps=1.0)
    simulation.add_output_layer(eps=1.0)
    simulation.set_incident_angle(0.0, 0.0)
    return simulation


def _eq7_check(device: torch.device) -> dict[str, object]:
    simulation = _simulation(device)
    mapping = simulation.build_stepped_circle_asr_mapping(
        88, 88, 14.0 / 62.0, 30.0 / 62.0, staircase_grid=12
    )
    intervals_x = mapping.x_breaks.numel() - 1
    intervals_y = mapping.y_breaks.numel() - 1

    def equation_errors(
        physical: torch.Tensor, transformed: torch.Tensor
    ) -> tuple[float, float, float]:
        x0, x1 = physical[:-1], physical[1:]
        u0, u1 = transformed[:-1], transformed[1:]
        du, dx = u1 - u0, x1 - x0
        a1 = (u1 * x0 - u0 * x1) / du
        a2 = dx / du
        a3 = 1.0e-3 * du - dx
        mapped0 = a1 + a2 * u0
        mapped1 = a1 + a2 * u1
        slope = a2 + a3 / du
        break_error = torch.max(
            torch.maximum(torch.abs(mapped0 - x0), torch.abs(mapped1 - x1))
        )
        slope_error = torch.max(torch.abs(slope - 1.0e-3))
        expected_du = torch.pow(dx, 1.0 / 3.0)
        expected_du = expected_du / torch.sum(expected_du)
        allocation_error = torch.max(torch.abs(du - expected_du))
        return (
            float(break_error.cpu()),
            float(slope_error.cpu()),
            float(allocation_error.cpu()),
        )

    x_break_error, x_slope_error, x_allocation_error = equation_errors(
        mapping.x_breaks, mapping.u_breaks
    )
    y_break_error, y_slope_error, y_allocation_error = equation_errors(
        mapping.y_breaks, mapping.v_breaks
    )
    minimum_jacobian = float((mapping.f.min() * mapping.g.min()).cpu())
    return {
        "x_intervals": int(intervals_x),
        "y_intervals": int(intervals_y),
        "x_breakpoint_error": x_break_error,
        "y_breakpoint_error": y_break_error,
        "x_minimum_slope_error": x_slope_error,
        "y_minimum_slope_error": y_slope_error,
        "x_cube_root_allocation_error": x_allocation_error,
        "y_cube_root_allocation_error": y_allocation_error,
        "minimum_jacobian": minimum_jacobian,
        "passed": max(
            x_break_error,
            y_break_error,
            x_slope_error,
            y_slope_error,
            x_allocation_error,
            y_allocation_error,
        )
        < 1.0e-12
        and minimum_jacobian > 0.0,
    }


def _eq8_eq9_check(device: torch.device) -> dict[str, object]:
    simulation = _simulation(device, order=2)
    samples = 64
    axis = torch.arange(samples, dtype=torch.float64, device=device) / samples
    xx, yy = torch.meshgrid(axis, axis, indexing="ij")
    dx, dy = xx - 0.5, yy - 0.5
    radius = torch.sqrt(dx**2 + dy**2)
    epsilon = torch.where(
        radius < 0.3,
        torch.as_tensor(3.5 + 0.02j, dtype=torch.complex128, device=device),
        torch.as_tensor(1.0 + 0.0j, dtype=torch.complex128, device=device),
    )
    safe_radius = torch.clamp(radius, min=1.0e-14)
    normal_u = torch.where(radius > 1.0e-14, dx / safe_radius, torch.ones_like(dx))
    normal_v = torch.where(radius > 1.0e-14, dy / safe_radius, torch.zeros_like(dy))
    uu, uv, vu, vv, epsilon_matrix, inverse_rule = (
        simulation._peng_eq8_transverse_epsilon(epsilon, normal_u, normal_v)
    )
    # Literal transcription of the paper: Delta=E-inverse_rule and
    # D_i=(E-Delta*N_ii)E_i-Delta*N_ij*E_j.
    delta = epsilon_matrix - inverse_rule
    n_uu = simulation._material_conv(normal_u * normal_u)
    n_uv = simulation._material_conv(normal_u * normal_v)
    n_vv = simulation._material_conv(normal_v * normal_v)
    expected = (
        epsilon_matrix - torch.matmul(delta, n_uu),
        -torch.matmul(delta, n_uv),
        -torch.matmul(delta, n_uv),
        epsilon_matrix - torch.matmul(delta, n_vv),
    )
    errors = [
        float(torch.max(torch.abs(actual - reference)).cpu())
        for actual, reference in zip((uu, uv, vu, vv), expected)
    ]
    return {
        "maximum_literal_equation_error": max(errors),
        "component_errors": errors,
        "finite": all(
            bool(torch.all(torch.isfinite(value))) for value in (uu, uv, vu, vv)
        ),
        "passed": max(errors) < 1.0e-12,
    }


def _exact_rectangular_laurent_check(device: torch.device) -> dict[str, object]:
    simulation = _simulation(device, order=2)
    u_breaks = torch.tensor(
        [0.0, 0.2, 0.5, 0.8, 1.0], dtype=torch.float64, device=device
    )
    v_breaks = torch.tensor(
        [0.0, 0.25, 0.6, 1.0], dtype=torch.float64, device=device
    )
    values = torch.tensor(
        [
            [1.0, 2.0, 4.0],
            [3.0, 5.0, 7.0],
            [2.0, 6.0, 8.0],
            [1.5, 3.5, 9.0],
        ],
        dtype=torch.complex128,
        device=device,
    )
    matrix = simulation._piecewise_rectangular_conv(values, u_breaks, v_breaks)
    area_weights = (u_breaks[1:] - u_breaks[:-1])[:, None] * (
        v_breaks[1:] - v_breaks[:-1]
    )[None, :]
    expected_mean = torch.sum(values * area_weights)
    diagonal_error = float(
        torch.max(torch.abs(torch.diagonal(matrix) - expected_mean)).cpu()
    )
    hermitian_error = float(torch.max(torch.abs(matrix - matrix.mH)).cpu())

    def integral(harmonic: int, low: float, high: float) -> complex:
        if harmonic == 0:
            return high - low
        return (
            cmath.exp(-2.0j * math.pi * harmonic * high)
            - cmath.exp(-2.0j * math.pi * harmonic * low)
        ) / (-2.0j * math.pi * harmonic)

    manual = 0.0j
    values_cpu = values.real.cpu().tolist()
    u_cpu, v_cpu = u_breaks.cpu().tolist(), v_breaks.cpu().tolist()
    for i, row in enumerate(values_cpu):
        for j, value in enumerate(row):
            manual += value * integral(1, u_cpu[i], u_cpu[i + 1]) * integral(
                -1, v_cpu[j], v_cpu[j + 1]
            )
    order_x = [int(value) for value in simulation.order_x.cpu().tolist()]
    order_y = [int(value) for value in simulation.order_y.cpu().tolist()]
    row_index = order_x.index(1) * len(order_y) + order_y.index(-1)
    column_index = order_x.index(0) * len(order_y) + order_y.index(0)
    coefficient_error = abs(complex(matrix[row_index, column_index].cpu()) - manual)
    return {
        "diagonal_mean_error": diagonal_error,
        "real_field_hermitian_error": hermitian_error,
        "selected_coefficient_error": coefficient_error,
        "passed": max(diagonal_error, hermitian_error, coefficient_error) < 1.0e-12,
    }


def _idw_check(device: torch.device) -> dict[str, object]:
    simulation = _simulation(device)
    mapping = simulation.build_stepped_circle_asr_mapping(
        48, 48, 14.0 / 62.0, 30.0 / 62.0, staircase_grid=12
    )
    normal_u, normal_v = simulation._peng_idw_normal_field(
        mapping,
        14.0 / 62.0,
        30.0 / 62.0,
        boundary_samples=128,
        neighbors=16,
        power=2.0,
    )
    norm_error = float(
        torch.max(torch.abs(normal_u**2 + normal_v**2 - 1.0)).cpu()
    )
    finite = bool(torch.all(torch.isfinite(normal_u))) and bool(
        torch.all(torch.isfinite(normal_v))
    )
    return {
        "maximum_unit_norm_error": norm_error,
        "finite": finite,
        "passed": finite and norm_error < 1.0e-12,
    }


def _power_check(result: dict[str, object]) -> dict[str, object]:
    values = {
        key: float(result[key])
        for key in ("reflectance", "transmittance", "absorptance")
    }
    finite = all(math.isfinite(value) for value in values.values())
    balance_error = abs(sum(values.values()) - 1.0)
    return {
        **values,
        "finite": finite,
        "balance_error": balance_error,
        "passivity_warning": bool(result["passivity_warning"]),
        "factorization_scheme": result.get("backend_factorization_scheme"),
        "minimum_mapping_jacobian": result.get("minimum_mapping_jacobian"),
        "passed": finite
        and balance_error < 1.0e-10
        and not bool(result["passivity_warning"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=_VALIDATION_ROOT / "results" / "paper_asr_nv_validation.json",
    )
    args = parser.parse_args()
    device = select_device(args.device)
    geometry, drude = PaperGeometry(), SilverDrude()
    common = dict(
        order_x=1,
        order_y=1,
        grid_x=48,
        grid_y=48,
        asr_g=1.0e-3,
        staircase_grid=12,
        nv_boundary_samples=128,
        nv_neighbors=16,
        nv_power=2.0,
    )
    asr = simulate_matched_primitive(
        1.95,
        lattice_kind="square",
        geometry=geometry,
        drude=drude,
        numerics=Numerics(**common, solver="paper-asr"),
        device=device,
    )
    asr_nv = simulate_matched_primitive(
        1.95,
        lattice_kind="square",
        geometry=geometry,
        drude=drude,
        numerics=Numerics(**common, solver="paper-asr-nv"),
        device=device,
    )
    checks = {
        "peng_eq7_mapping": _eq7_check(device),
        "exact_rectangular_laurent_convolution": _exact_rectangular_laurent_check(
            device
        ),
        "peng_eq8_eq9_literal_matrix": _eq8_eq9_check(device),
        "periodic_idw_normal_field": _idw_check(device),
        "paper_asr_power_smoke": _power_check(asr),
        "paper_asr_nv_power_smoke": _power_check(asr_nv),
        "solver_dispatch": {
            "paper_asr": asr.get("backend_factorization_scheme"),
            "paper_asr_nv": asr_nv.get("backend_factorization_scheme"),
            "passed": asr.get("backend_factorization_scheme")
            == "peng-eq7-asr-exact-rectangular-laurent"
            and asr_nv.get("backend_factorization_scheme")
            == "peng-eq7-asr+eq8-10-nv-idw-paper-disclosed",
        },
    }
    report = {
        "passed": all(bool(check["passed"]) for check in checks.values()),
        "device": str(device),
        "checks": checks,
        "scope": (
            "Equation/dispatch/smoke validation only. Paper-scale order and "
            "staircase convergence remain a separate long calculation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
