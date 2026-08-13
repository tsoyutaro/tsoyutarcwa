"""Independent checks for the matched-coordinate circular ASR implementation.

The default checks require NumPy only and deliberately reimplement the map,
Jacobian, tensor density, symmetric factorization, and modal conversion instead
of importing the production module.  This catches sign/index mistakes that a
self-consistency test of the implementation would miss.

Run the full torcwa comparisons in an environment containing torch and torcwa::

    python validate_circle_matched_asr.py --integration

The integration run checks Redheffer/Li-2a parity, full/half/quarter parity,
and reports the convergence difference between matched-ASR and Cartesian NVM.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class Check:
    name: str
    error: float
    limit: float
    detail: str = ""

    @property
    def passed(self) -> bool:
        return math.isfinite(self.error) and self.error <= self.limit


def piecewise_asr(
    length: float,
    breaks: np.ndarray,
    samples: int,
    minimum_slope: float,
) -> tuple[np.ndarray, ...]:
    dx = np.diff(breaks)
    weights = np.cbrt(dx)
    du = length * weights / np.sum(weights)
    transformed_breaks = np.concatenate(([0.0], np.cumsum(du)))
    transformed_breaks[-1] = length
    u = np.arange(samples, dtype=float) * length / samples
    interval = np.searchsorted(transformed_breaks[1:-1], u, side="left")
    u0, u1 = transformed_breaks[interval], transformed_breaks[interval + 1]
    x0, x1 = breaks[interval], breaks[interval + 1]
    local_du, local_dx = u1 - u0, x1 - x0
    a1 = (u1 * x0 - u0 * x1) / local_du
    a2 = local_dx / local_du
    a3 = minimum_slope * local_du - local_dx
    phase = 2.0 * np.pi * (u - u0) / local_du
    mapped = a1 + a2 * u + a3 / (2.0 * np.pi) * np.sin(phase)
    derivative = a2 + a3 / local_du * np.cos(phase)
    return u, mapped, derivative, transformed_breaks


def matched_axis(
    main: np.ndarray,
    cross: np.ndarray,
    *,
    main_length: float,
    cross_center: float,
    circle_center: float,
    radius: float,
    main_minus: float,
    main_plus: float,
    cross_minus: float,
    cross_plus: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    main_grid, cross_grid = main[:, None], cross[None, :]
    central = (cross_grid >= cross_minus) & (cross_grid <= cross_plus)
    offset = cross_grid - cross_center
    root = np.sqrt(np.maximum(radius**2 - offset**2, 1.0e-30))
    curve_minus = np.where(central, circle_center - root, main_minus)
    curve_plus = np.where(central, circle_center + root, main_plus)
    derivative_minus = np.where(central, offset / root, 0.0)
    derivative_plus = np.where(central, -offset / root, 0.0)
    lower = main_grid < main_minus
    middle = (main_grid >= main_minus) & (main_grid <= main_plus)
    middle_width, upper_width = main_plus - main_minus, main_length - main_plus
    lower_weight = main_grid / main_minus
    plus_weight = (main_grid - main_minus) / middle_width
    minus_weight = 1.0 - plus_weight
    upper_curve_weight = (main_length - main_grid) / upper_width
    upper_end_weight = (main_grid - main_plus) / upper_width
    mapped = np.where(
        lower,
        lower_weight * curve_minus,
        np.where(
            middle,
            minus_weight * curve_minus + plus_weight * curve_plus,
            upper_curve_weight * curve_plus + upper_end_weight * main_length,
        ),
    )
    derivative_main = np.where(
        lower,
        curve_minus / main_minus,
        np.where(
            middle,
            (curve_plus - curve_minus) / middle_width,
            (main_length - curve_plus) / upper_width,
        ),
    )
    derivative_cross = np.where(
        lower,
        lower_weight * derivative_minus,
        np.where(
            middle,
            minus_weight * derivative_minus + plus_weight * derivative_plus,
            upper_curve_weight * derivative_plus,
        ),
    )
    return mapped, derivative_main, derivative_cross


def circle_map(
    lx: float,
    ly: float,
    radius: float,
    nx: int,
    ny: int,
    minimum_slope: float = 0.03,
) -> dict[str, np.ndarray]:
    displacement = radius / np.sqrt(2.0)
    bx = np.array([0.0, lx / 2 - displacement, lx / 2 + displacement, lx])
    by = np.array([0.0, ly / 2 - displacement, ly / 2 + displacement, ly])
    u, tu, fu, ub = piecewise_asr(lx, bx, nx, minimum_slope)
    v, tv, gv, vb = piecewise_asr(ly, by, ny, minimum_slope)
    x, x_tu, x_tv = matched_axis(
        tu,
        tv,
        main_length=lx,
        cross_center=ly / 2,
        circle_center=lx / 2,
        radius=radius,
        main_minus=bx[1],
        main_plus=bx[2],
        cross_minus=by[1],
        cross_plus=by[2],
    )
    yt, y_tv_t, y_tu_t = matched_axis(
        tv,
        tu,
        main_length=ly,
        cross_center=lx / 2,
        circle_center=ly / 2,
        radius=radius,
        main_minus=by[1],
        main_plus=by[2],
        cross_minus=bx[1],
        cross_plus=bx[2],
    )
    y, y_tv, y_tu = yt.T, y_tv_t.T, y_tu_t.T
    x_u, x_v = x_tu * fu[:, None], x_tv * gv[None, :]
    y_u, y_v = y_tu * fu[:, None], y_tv * gv[None, :]
    det_j = x_u * y_v - x_v * y_u
    return {
        "u": u,
        "v": v,
        "tu": tu,
        "tv": tv,
        "x": x,
        "y": y,
        "x_u": x_u,
        "x_v": x_v,
        "y_u": y_u,
        "y_v": y_v,
        "det_j": det_j,
        "matched_x_breaks": bx,
        "matched_y_breaks": by,
        "u_breaks": ub,
        "v_breaks": vb,
    }


def modal_conversion_fft(
    mapping: dict[str, np.ndarray],
    lx: float,
    ly: float,
    order_x: np.ndarray,
    order_y: np.ndarray,
    incident: tuple[float, float] = (0.17, -0.11),
) -> tuple[np.ndarray, np.ndarray]:
    u, v = mapping["u"][:, None], mapping["v"][None, :]
    x, y = mapping["x"], mapping["y"]
    weights = np.stack(
        (
            mapping["y_v"],
            -mapping["y_u"],
            -mapping["x_v"],
            mapping["x_u"],
            mapping["det_j"],
        )
    )
    px, py = np.meshgrid(order_x, order_y, indexing="ij")
    px, py = px.ravel(), py.ravel()
    kx = incident[0] + 2.0 * np.pi * px / lx
    ky = incident[1] + 2.0 * np.pi * py / ly
    n = len(kx)
    blocks = np.zeros((5, n, n), dtype=complex)
    incident_phase = np.exp(1j * (incident[0] * u + incident[1] * v))
    column_x = np.mod(order_x, len(mapping["u"]))
    column_y = np.mod(order_y, len(mapping["v"]))
    for row in range(n):
        physical_phase = np.exp(-1j * (kx[row] * x + ky[row] * y))
        spectra = np.fft.ifft2(weights * (incident_phase * physical_phase)[None])
        blocks[:, row] = spectra[:, column_x[:, None], column_y[None, :]].reshape(5, n)
    transform = np.block([[blocks[0], blocks[1]], [blocks[2], blocks[3]]])
    return transform, blocks[4]


def modal_conversion_brute(
    mapping: dict[str, np.ndarray],
    lx: float,
    ly: float,
    order_x: np.ndarray,
    order_y: np.ndarray,
    incident: tuple[float, float] = (0.17, -0.11),
) -> tuple[np.ndarray, np.ndarray]:
    u, v = mapping["u"][:, None], mapping["v"][None, :]
    x, y = mapping["x"], mapping["y"]
    weights = np.stack(
        (
            mapping["y_v"],
            -mapping["y_u"],
            -mapping["x_v"],
            mapping["x_u"],
            mapping["det_j"],
        )
    )
    px, py = np.meshgrid(order_x, order_y, indexing="ij")
    px, py = px.ravel(), py.ravel()
    kx = incident[0] + 2.0 * np.pi * px / lx
    ky = incident[1] + 2.0 * np.pi * py / ly
    n = len(kx)
    blocks = np.zeros((5, n, n), dtype=complex)
    for row in range(n):
        for column in range(n):
            phase = np.exp(
                1j
                * (
                    kx[column] * u
                    + ky[column] * v
                    - kx[row] * x
                    - ky[row] * y
                )
            )
            blocks[:, row, column] = np.mean(weights * phase[None], axis=(1, 2))
    return np.block([[blocks[0], blocks[1]], [blocks[2], blocks[3]]]), blocks[4]


def axis_toeplitz(field: np.ndarray, orders: np.ndarray, axis: int) -> np.ndarray:
    coefficients = np.fft.fft(field, axis=axis) / field.shape[axis]
    delta = orders[:, None] - orders[None, :]
    if axis == 1:
        return coefficients[:, delta]
    return coefficients[delta, :].transpose(2, 0, 1)


def assemble_matrix_samples(
    samples: np.ndarray,
    order_x: np.ndarray,
    order_y: np.ndarray,
    outer: str,
) -> np.ndarray:
    coefficients = np.fft.fft(samples, axis=0) / samples.shape[0]
    mx, my = len(order_x), len(order_y)
    result = np.zeros((mx * my, mx * my), dtype=complex)
    if outer == "u":
        delta = order_x[:, None] - order_x[None, :]
        for p in range(mx):
            for pp in range(mx):
                result[p * my : (p + 1) * my, pp * my : (pp + 1) * my] = coefficients[delta[p, pp]]
    else:
        delta = order_y[:, None] - order_y[None, :]
        for p in range(mx):
            for pp in range(mx):
                result[p * my : (p + 1) * my, pp * my : (pp + 1) * my] = coefficients[delta, p, pp]
    return result


def symmetric_factorization(
    values: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    order_x: np.ndarray,
    order_y: np.ndarray,
) -> tuple[np.ndarray, ...]:
    t11, t12, t21, t22 = values
    determinant = t11 * t22 - t12 * t21
    iv = np.broadcast_to(np.eye(len(order_y)), (t11.shape[0], len(order_y), len(order_y)))
    c22i = axis_toeplitz(1 / t22, order_y, 1)
    c12 = axis_toeplitz(t12 / t22, order_y, 1)
    c21 = axis_toeplitz(t21 / t22, order_y, 1)
    a22 = np.linalg.solve(c22i, iv)
    a21 = np.linalg.solve(c22i, c21)
    a12 = c12 @ a22
    a11 = axis_toeplitz(determinant / t22, order_y, 1) + c12 @ a21
    a11i = np.linalg.solve(a11, iv)
    r11 = np.linalg.solve(assemble_matrix_samples(a11i, order_x, order_y, "u"), np.eye(len(order_x) * len(order_y)))
    r12 = r11 @ assemble_matrix_samples(a11i @ a12, order_x, order_y, "u")

    iu = np.broadcast_to(np.eye(len(order_x)), (t11.shape[1], len(order_x), len(order_x)))
    c11i = axis_toeplitz(1 / t11, order_x, 0)
    c12 = axis_toeplitz(t12 / t11, order_x, 0)
    c21 = axis_toeplitz(t21 / t11, order_x, 0)
    b11 = np.linalg.solve(c11i, iu)
    b12 = np.linalg.solve(c11i, c12)
    b21 = c21 @ b11
    b22 = axis_toeplitz(determinant / t11, order_x, 0) + c21 @ b12
    b22i = np.linalg.solve(b22, iu)
    r22 = np.linalg.solve(assemble_matrix_samples(b22i, order_x, order_y, "v"), np.eye(len(order_x) * len(order_y)))
    r21 = r22 @ assemble_matrix_samples(b22i @ b21, order_x, order_y, "v")
    return r11, r12, r21, r22


def core_checks() -> list[Check]:
    checks: list[Check] = []
    mapping = circle_map(1.2, 1.6, 0.42, 96, 112)
    minimum_jacobian = float(np.min(mapping["det_j"]))
    checks.append(Check("orientation-preserving Jacobian", max(0.0, -minimum_jacobian), 0.0, f"min(detJ)={minimum_jacobian:.6e}"))
    h = mapping["det_j"]
    g11 = (mapping["x_v"] ** 2 + mapping["y_v"] ** 2) / h
    g12 = -(mapping["x_u"] * mapping["x_v"] + mapping["y_u"] * mapping["y_v"]) / h
    g22 = (mapping["x_u"] ** 2 + mapping["y_u"] ** 2) / h
    metric_error = float(np.max(np.abs(g11 * g22 - g12**2 - 1.0)))
    checks.append(Check("det(detJ J^-1 J^-T)=1", metric_error, 2.0e-9))

    bx, by = mapping["matched_x_breaks"], mapping["matched_y_breaks"]
    cross_y = np.linspace(by[1], by[2], 301)
    circle_errors = []
    for main_x in (bx[1], bx[2]):
        x, _, _ = matched_axis(
            np.array([main_x]), cross_y,
            main_length=1.2,
            cross_center=0.8,
            circle_center=0.6,
            radius=0.42,
            main_minus=bx[1], main_plus=bx[2],
            cross_minus=by[1], cross_plus=by[2],
        )
        circle_errors.append(np.max(np.abs((x - 0.6) ** 2 + (cross_y[None] - 0.8) ** 2 - 0.42**2)))
    checks.append(Check("matched coordinate lines lie on circle", float(max(circle_errors)), 2.0e-15))

    ox = np.arange(-1, 2)
    oy = np.arange(-1, 2)
    fft_t, fft_tz = modal_conversion_fft(mapping, 1.2, 1.6, ox, oy)
    brute_t, brute_tz = modal_conversion_brute(mapping, 1.2, 1.6, ox, oy)
    checks.append(Check("FFT T equals direct quadrature", float(np.max(np.abs(fft_t - brute_t))), 2.0e-12))
    checks.append(Check("FFT Tz equals direct quadrature", float(np.max(np.abs(fft_tz - brute_tz))), 2.0e-12))

    identity_map = circle_map(1.2, 1.6, 0.42, 96, 112)
    uu, vv = identity_map["u"][:, None], identity_map["v"][None, :]
    identity_map.update(
        x=np.broadcast_to(uu, (96, 112)),
        y=np.broadcast_to(vv, (96, 112)),
        x_u=np.ones((96, 112)), x_v=np.zeros((96, 112)),
        y_u=np.zeros((96, 112)), y_v=np.ones((96, 112)),
        det_j=np.ones((96, 112)),
    )
    identity_t, identity_tz = modal_conversion_fft(identity_map, 1.2, 1.6, ox, oy)
    n = len(ox) * len(oy)
    checks.append(Check("T identity-map limit", float(np.max(np.abs(identity_t - np.eye(2 * n)))), 2.0e-12))
    checks.append(Check("Tz identity-map limit", float(np.max(np.abs(identity_tz - np.eye(n)))), 2.0e-12))

    shape = (48, 56)
    constants = tuple(np.full(shape, value, dtype=float) for value in (2.0, 0.3, 0.3, 1.5))
    factorized = symmetric_factorization(constants, ox, oy)
    expected = tuple(value * np.eye(n) for value in (2.0, 0.3, 0.3, 1.5))
    factorization_error = max(float(np.max(np.abs(a - b))) for a, b in zip(factorized, expected))
    checks.append(Check("Eqs.29-36 constant-tensor limit", factorization_error, 2.0e-12))
    return checks


def integration_checks(order: int, grid: int) -> tuple[list[Check], dict[str, object]]:
    try:
        import torch
        from rcwa_solver_auto import ASROptions, AutoRCWA, Circle, Lattice, LayerSpec, Material, NVMOptions, OutputSpec
    except ImportError as exc:
        raise RuntimeError("--integration requires torch and torcwa beside rcwa_solver_auto.py") from exc

    def make(method: str, algorithm: str, size: str, selected_order: int) -> AutoRCWA:
        sim = AutoRCWA(
            freq=1 / 1.55,
            order=[selected_order, selected_order],
            lattice=Lattice.rectangular(1.0, 1.2),
            cascade=algorithm,
            outputs=OutputSpec(smatrix_size=size, fields="none"),
            asr=ASROptions(circle_G=0.03, grid=(grid, grid)),
            nvm=NVMOptions(grid=(grid, grid)),
            verify_cascade=False,
            compute_condition_numbers=True,
            dtype=torch.complex128,
            device="cpu",
        )
        sim.add_input_layer(eps=1.0)
        sim.add_output_layer(eps=1.0)
        sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
        chosen = sim.add_structured_layer(
            LayerSpec(
                thickness=0.18,
                geometry=Circle(0.25),
                background=Material(1.0),
                inclusion=Material(4.0),
                method=method,
                factorization_rules=True if method == "matched-asr" else None,
            )
        )
        if chosen != method:
            raise AssertionError(f"selected {chosen}, expected {method}")
        sim.solve_global_smatrix()
        return sim

    checks: list[Check] = []
    full_r = make("matched-asr", "redheffer", "full", order)
    full_l = make("matched-asr", "algo2a", "full", order)
    for index, name in enumerate(("Tf", "Rf", "Rb", "Tb")):
        checks.append(Check(f"matched ASR Li2a/Redheffer {name}", float(torch.max(torch.abs(full_r.S[index] - full_l.S[index]))), 2.0e-8))
    for algorithm, full in (("redheffer", full_r), ("algo2a", full_l)):
        half = make("matched-asr", algorithm, "half", order)
        quarter = make("matched-asr", algorithm, "quarter", order)
        checks.append(Check(f"{algorithm} half/full Tf", float(torch.max(torch.abs(half.S[0] - full.S[0]))), 2.0e-8))
        checks.append(Check(f"{algorithm} half/full Rf", float(torch.max(torch.abs(half.S[1] - full.S[1]))), 2.0e-8))
        checks.append(Check(f"{algorithm} quarter/full Rf", float(torch.max(torch.abs(quarter.S[1] - full.S[1]))), 2.0e-8))

    zero_x = int(torch.nonzero(full_r.order_x == 0, as_tuple=False)[0])
    zero_y = int(torch.nonzero(full_r.order_y == 0, as_tuple=False)[0])
    harmonic = zero_x * len(full_r.order_y) + zero_y
    source = torch.zeros(2 * full_r.order_N, dtype=torch.complex128)
    source[harmonic] = 1.0
    transmitted, reflected = full_r.S[0] @ source, full_r.S[1] @ source
    # With the selected wavelength only the zeroth order propagates and both
    # exterior media are vacuum, so the Euclidean zeroth-order amplitudes are
    # already power normalized.
    power = sum(
        float(torch.abs(vector[index]) ** 2)
        for vector in (transmitted, reflected)
        for index in (harmonic, full_r.order_N + harmonic)
    )
    checks.append(Check("lossless zeroth-order power conservation", abs(power - 1.0), 2.0e-10, f"R+T={power:.16f}"))

    convergence = []
    for selected_order in range(max(1, order - 2), order + 1):
        matched = make("matched-asr", "redheffer", "half", selected_order)
        nvm = make("nvm", "redheffer", "half", selected_order)
        source = torch.zeros(2 * matched.order_N, dtype=torch.complex128)
        zero_x = int(torch.nonzero(matched.order_x == 0, as_tuple=False)[0])
        zero_y = int(torch.nonzero(matched.order_y == 0, as_tuple=False)[0])
        source[zero_x * len(matched.order_y) + zero_y] = 1.0
        difference = max(
            float(torch.max(torch.abs((matched.S[index] - nvm.S[index]) @ source)))
            for index in (0, 1)
        )
        convergence.append({"order": selected_order, "matched_nvm_source_error": difference})
    diagnostics = {
        "matched_nvm_convergence": convergence,
        "T_condition_number": (
            None
            if full_r.asr_condition_numbers[-1] is None
            else float(full_r.asr_condition_numbers[-1])
        ),
    }
    return checks, diagnostics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--integration", action="store_true")
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--grid", type=int, default=128)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    checks = core_checks()
    diagnostics: dict[str, object] = {}
    if args.integration:
        extra, diagnostics = integration_checks(args.order, args.grid)
        checks.extend(extra)
    payload = {
        "passed": all(check.passed for check in checks),
        "checks": [asdict(check) | {"passed": check.passed} for check in checks],
        "diagnostics": diagnostics,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.json:
        args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
