"""Independent validation for triangular matched-ASR/NVM x/y sector solves.

The NumPy checks do not import the production module.  The integration checks
require torch, torcwa and rcwa_solver_auto.py on PYTHONPATH::

    python validate_triangular_matched_asr.py --integration --json result.json
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


def hermite(t, y0, slope0, y1, slope1, span):
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    return h00 * y0 + h10 * span * slope0 + h01 * y1 + h11 * span * slope1


def hex_radius(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    return np.maximum.reduce(
        (np.abs(q1 + 0.5 * q2), np.abs(0.5 * q1 + q2), np.abs(0.5 * (q1 - q2)))
    )


def central_map(q1, q2, radius=0.24, circle_g=0.08, period=1.0):
    q1, q2 = np.asarray(q1, float), np.asarray(q2, float)
    h = hex_radius(q1, q2)
    rho = h / radius
    active = rho > 1e-14
    safe = np.where(active, rho, 1.0)
    a, b = q1 / safe, q2 / safe
    norm = np.sqrt(np.maximum(a * a + b * b + a * b, 1e-30))
    circle_scale = np.where(active, radius / norm, 1.0)
    outer = 0.5 * period / radius
    ti = np.clip(rho, 0.0, 1.0)
    to = np.clip((rho - 1.0) / (outer - 1.0), 0.0, 1.0)
    inner = hermite(ti, 0.0, 1.0, circle_scale, circle_g, 1.0)
    outside = hermite(to, circle_scale, circle_g, outer, 1.0, outer - 1.0)
    radial = np.where(rho <= 1.0, inner, np.where(rho < outer, outside, rho))
    p1 = np.where(active, radial * a, q1)
    p2 = np.where(active, radial * b, q2)
    return p1, p2


def physical(q1, q2):
    return np.asarray(q1) + 0.5 * np.asarray(q2), (math.sqrt(3) / 2) * np.asarray(q2)


def core_checks() -> list[Check]:
    checks: list[Check] = []
    radius, circle_g = 0.24, 0.08
    angles = np.linspace(0.0, 2 * np.pi, 721, endpoint=False)
    dx, dy = np.cos(angles), np.sin(angles)
    d2 = dy / (math.sqrt(3) / 2)
    d1 = dx - 0.5 * d2
    scale = radius / hex_radius(d1, d2)
    q1, q2 = scale * d1, scale * d2
    p1, p2 = central_map(q1, q2, radius, circle_g)
    x, y = physical(p1, p2)
    checks.append(
        Check("hex interface maps to circle", float(np.max(np.abs(np.hypot(x, y) - radius))), 2e-13)
    )

    rng = np.random.default_rng(20260806)
    q1 = rng.uniform(-0.55, 0.55, 5000)
    q2 = rng.uniform(-0.55, 0.55, 5000)
    keep = hex_radius(q1, q2) < 0.495
    q1, q2 = q1[keep], q2[keep]
    p1, p2 = central_map(q1, q2, radius, circle_g)
    # 60-degree rotation (q1,q2)->(-q2,q1+q2).
    rp1, rp2 = central_map(-q2, q1 + q2, radius, circle_g)
    expected1, expected2 = -p2, p1 + p2
    rotation_error = max(np.max(np.abs(rp1 - expected1)), np.max(np.abs(rp2 - expected2)))
    checks.append(Check("D6 rotation equivariance", float(rotation_error), 3e-13))
    # Reflection in the local physical x axis.
    mp1, mp2 = central_map(q1 + q2, -q2, radius, circle_g)
    mirror_error = max(np.max(np.abs(mp1 - (p1 + p2))), np.max(np.abs(mp2 + p2)))
    checks.append(Check("D6 mirror equivariance", float(mirror_error), 3e-13))

    # Independent finite-difference Jacobian, away from piecewise joins.
    h = hex_radius(q1, q2)
    joins = np.minimum(np.abs(h - radius), np.abs(h - 0.5))
    sector_gap = np.min(
        np.stack(
            (
                np.abs(np.abs(q1 + 0.5 * q2) - np.abs(0.5 * q1 + q2)),
                np.abs(np.abs(q1 + 0.5 * q2) - np.abs(0.5 * (q1 - q2))),
                np.abs(np.abs(0.5 * q1 + q2) - np.abs(0.5 * (q1 - q2))),
            )
        ),
        axis=0,
    )
    sample = (joins > 2e-4) & (sector_gap > 2e-4)
    q1s, q2s = q1[sample][:1200], q2[sample][:1200]
    delta = 2e-7
    a1p, a2p = central_map(q1s + delta, q2s, radius, circle_g)
    a1m, a2m = central_map(q1s - delta, q2s, radius, circle_g)
    b1p, b2p = central_map(q1s, q2s + delta, radius, circle_g)
    b1m, b2m = central_map(q1s, q2s - delta, radius, circle_g)
    xap, yap = physical(a1p, a2p)
    xam, yam = physical(a1m, a2m)
    xbp, ybp = physical(b1p, b2p)
    xbm, ybm = physical(b1m, b2m)
    xu, yu = (xap - xam) / (2 * delta), (yap - yam) / (2 * delta)
    xv, yv = (xbp - xbm) / (2 * delta), (ybp - ybm) / (2 * delta)
    determinant = xu * yv - xv * yu
    checks.append(
        Check(
            "positive mapping Jacobian",
            float(max(0.0, -np.min(determinant))),
            1e-9,
            f"min(det J)={np.min(determinant):.6e}",
        )
    )

    orders = [(m, n) for m in range(-4, 5) for n in range(-4, 5) if max(abs(m), abs(n), abs(m-n)) <= 4]
    order_set = set(orders)
    closure_error = float(any((m, m-n) not in order_set or (n, n-m) not in order_set for m, n in orders))
    checks.append(Check("D6 Fourier-star closure", closure_error, 0.0, f"scalar dimension={len(orders)}"))

    star_index = {pair: position for position, pair in enumerate(orders)}
    scalar_reflection = np.zeros((len(orders), len(orders)))
    for column, (m, n) in enumerate(orders):
        scalar_reflection[star_index[(m, m - n)], column] = -1.0 if m % 2 else 1.0
    covariant_polar = np.array([[1.0, 0.0], [1.0, -1.0]])
    electric_reflection = np.kron(covariant_polar, scalar_reflection)
    vector_identity = np.eye(2 * len(orders))
    checks.append(
        Check(
            "D6-star vector mirror involution",
            float(np.linalg.norm(electric_reflection @ electric_reflection - vector_identity)),
            2e-13,
        )
    )
    projector_plus = 0.5 * (vector_identity + electric_reflection)
    projector_minus = 0.5 * (vector_identity - electric_reflection)
    sector_error = max(
        float(np.linalg.norm(projector_plus @ projector_minus)),
        float(np.linalg.norm(projector_plus + projector_minus - vector_identity)),
        float(abs(np.linalg.matrix_rank(projector_plus) - len(orders))),
        float(abs(np.linalg.matrix_rank(projector_minus) - len(orders))),
    )
    checks.append(
        Check(
            "D6-star x/y Cs-sector completeness",
            sector_error,
            2e-13,
            f"sector dimensions={np.linalg.matrix_rank(projector_plus)}/{np.linalg.matrix_rank(projector_minus)}",
        )
    )

    def rotate_index(pair):
        m, n = pair
        return m - n, m

    def mirror_index(pair):
        m, n = pair
        return m, m - n

    d6_electric = []
    d6_magnetic = []
    d6_scalar = []
    d6_electric_component = []
    d6_magnetic_component = []
    for reflected in (False, True):
        for power in range(6):
            scalar = np.zeros((len(orders), len(orders)))
            for column, pair in enumerate(orders):
                mapped = mirror_index(pair) if reflected else pair
                for _ in range(power):
                    mapped = rotate_index(mapped)
                exponent = pair[0] + pair[1] - mapped[0] - mapped[1]
                scalar[star_index[mapped], column] = -1.0 if exponent % 2 else 1.0
            angle = power * np.pi / 3.0
            rotation = np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )
            component = rotation @ np.diag([1.0, -1.0]) if reflected else rotation
            determinant = -1.0 if reflected else 1.0
            d6_scalar.append(scalar)
            d6_electric_component.append(component)
            d6_magnetic_component.append(determinant * component)
            d6_electric.append(np.kron(component, scalar))
            d6_magnetic.append(np.kron(determinant * component, scalar))

    characters = {}
    for label, rotation_sign, mirror_sign in (
        ("A1", 1, 1), ("A2", 1, -1),
        ("B1", -1, 1), ("B2", -1, -1),
    ):
        characters[label] = (
            1,
            [rotation_sign**power for power in range(6)]
            + [rotation_sign**power * mirror_sign for power in range(6)],
        )
    for harmonic in (1, 2):
        characters[f"E{harmonic}"] = (
            2,
            [2.0 * np.cos(harmonic * power * np.pi / 3.0) for power in range(6)]
            + [0.0] * 6,
        )
    irrep_matrices = {}
    for label, rotation_sign, mirror_sign in (
        ("A1", 1, 1), ("A2", 1, -1),
        ("B1", -1, 1), ("B2", -1, -1),
    ):
        irrep_matrices[label] = (
            [np.array([[rotation_sign**power]], float) for power in range(6)]
            + [
                np.array(
                    [[rotation_sign**power * mirror_sign]], float
                )
                for power in range(6)
            ]
        )
    standard_reflection = np.diag([1.0, -1.0])
    for harmonic in (1, 2):
        rotations = []
        for power in range(6):
            angle = harmonic * power * np.pi / 3.0
            rotations.append(
                np.array(
                    [
                        [np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)],
                    ]
                )
            )
        irrep_matrices[f"E{harmonic}"] = (
            rotations + [rotation @ standard_reflection for rotation in rotations]
        )

    projector_sum_e = np.zeros_like(d6_electric[0])
    projector_sum_h = np.zeros_like(d6_magnetic[0])
    previous_e = []
    d6_error = 0.0
    matrix_unit_error = 0.0
    d6_dimensions = {}
    solved_dimensions = {}
    matrix_unit_rows = {}
    for label in ("A1", "A2", "B1", "B2", "E1", "E2"):
        irrep_dimension, character = characters[label]
        projector_e = sum(
            irrep_dimension * value * operator / 12.0
            for value, operator in zip(character, d6_electric)
        )
        projector_h = sum(
            irrep_dimension * value * operator / 12.0
            for value, operator in zip(character, d6_magnetic)
        )
        projector_e = 0.5 * (projector_e + projector_e.T)
        projector_h = 0.5 * (projector_h + projector_h.T)
        rank_e = np.linalg.matrix_rank(projector_e, tol=1e-9)
        rank_h = np.linalg.matrix_rank(projector_h, tol=1e-9)
        d6_dimensions[label] = rank_e
        solved_dimensions[label] = rank_e
        d6_error = max(
            d6_error,
            float(np.linalg.norm(projector_e @ projector_e - projector_e)),
            float(np.linalg.norm(projector_h @ projector_h - projector_h)),
            float(abs(rank_e - rank_h)),
            float(rank_e % irrep_dimension),
        )
        for old in previous_e:
            d6_error = max(d6_error, float(np.linalg.norm(old @ projector_e)))
        previous_e.append(projector_e)
        projector_sum_e += projector_e
        projector_sum_h += projector_h
        if irrep_dimension == 2:
            representation = irrep_matrices[label]

            def matrix_unit(operators, row, column):
                return sum(
                    irrep_dimension * irrep[row, column] * operator / 12.0
                    for irrep, operator in zip(representation, operators)
                )

            e00, e11, e10 = (
                matrix_unit(d6_electric, 0, 0),
                matrix_unit(d6_electric, 1, 1),
                matrix_unit(d6_electric, 1, 0),
            )
            h00, h11, h10 = (
                matrix_unit(d6_magnetic, 0, 0),
                matrix_unit(d6_magnetic, 1, 1),
                matrix_unit(d6_magnetic, 1, 0),
            )
            solved_dimensions[label] = np.linalg.matrix_rank(e00, tol=1e-9)
            matrix_unit_rows[label] = (e00, e11, h00, h11)
            matrix_unit_error = max(
                matrix_unit_error,
                float(np.linalg.norm(e00 + e11 - projector_e)),
                float(np.linalg.norm(h00 + h11 - projector_h)),
                float(np.linalg.norm(e10.T @ e10 - e00)),
                float(np.linalg.norm(e10 @ e10.T - e11)),
                float(np.linalg.norm(h10.T @ h10 - h00)),
                float(np.linalg.norm(h10 @ h10.T - h11)),
                float(abs(2 * solved_dimensions[label] - rank_e)),
            )
    d6_error = max(
        d6_error,
        float(np.linalg.norm(projector_sum_e - vector_identity)),
        float(np.linalg.norm(projector_sum_h - vector_identity)),
    )
    checks.append(
        Check(
            "complete D6 character-projector resolution",
            d6_error,
            2e-12,
            "dimensions=" + ",".join(f"{key}:{value}" for key, value in d6_dimensions.items()),
        )
    )
    checks.append(
        Check(
            "D6 E-irrep matrix-unit row reconstruction",
            matrix_unit_error,
            3e-12,
            "solved dimensions="
            + ",".join(
                f"{key}:{value}" for key, value in solved_dimensions.items()
            ),
        )
    )
    e1_e00, e1_e11, _, _ = matrix_unit_rows["E1"]
    zero_harmonic = star_index[(0, 0)]
    source_x = np.zeros(2 * len(orders))
    source_y = np.zeros_like(source_x)
    source_x[zero_harmonic] = 1.0
    source_y[len(orders) + zero_harmonic] = 1.0
    expected_e1_row_dimension = 4 * (4 + 1) + 1
    source_row_error = max(
        float(abs(solved_dimensions["E1"] - expected_e1_row_dimension)),
        float(np.linalg.norm(source_x - e1_e00 @ source_x)),
        float(np.linalg.norm(source_y - e1_e11 @ source_y)),
        float(np.linalg.norm(e1_e11 @ source_x)),
        float(np.linalg.norm(e1_e00 @ source_y)),
    )
    checks.append(
        Check(
            "complete D6 E1 x/y source-row selection",
            source_row_error,
            3e-12,
            f"row dimension={solved_dimensions['E1']}",
        )
    )

    # Independent Maxwell-like intertwiner test.  Random P:H->E and Q:E->H
    # are Reynolds-averaged, then only one row of each E irrep is solved.
    size = vector_identity.shape[0]
    p_seed = rng.normal(size=(size, size))
    q_seed = rng.normal(size=(size, size))
    fast_action_error = 0.0
    scalar_size = len(orders)
    seed_blocks = p_seed.reshape(2, scalar_size, 2, scalar_size).transpose(
        0, 2, 1, 3
    )
    for scalar, electric_component, magnetic_component, electric, magnetic in zip(
        d6_scalar,
        d6_electric_component,
        d6_magnetic_component,
        d6_electric,
        d6_magnetic,
    ):
        inverse = np.argmax(np.abs(scalar), axis=1)
        phase = scalar[np.arange(scalar_size), inverse]
        spatial = seed_blocks[:, :, inverse, :][:, :, :, inverse]
        spatial = spatial * phase[None, None, :, None] * phase[None, None, None, :]
        fast = np.einsum(
            "ia,abxy,jb->ijxy",
            electric_component,
            spatial,
            magnetic_component,
        ).transpose(0, 2, 1, 3).reshape(size, size)
        fast_action_error = max(
            fast_action_error,
            float(np.linalg.norm(fast - electric @ p_seed @ magnetic.T)),
        )
    checks.append(
        Check(
            "D6 signed-permutation Reynolds action",
            fast_action_error,
            2e-11,
        )
    )
    p_d6 = sum(
        electric @ p_seed @ magnetic.T
        for electric, magnetic in zip(d6_electric, d6_magnetic)
    ) / 12.0
    q_d6 = sum(
        magnetic @ q_seed @ electric.T
        for electric, magnetic in zip(d6_electric, d6_magnetic)
    ) / 12.0
    modal_residual = 0.0
    reconstructed_electric = []
    for label in ("A1", "A2", "B1", "B2", "E1", "E2"):
        irrep_dimension, character = characters[label]
        representation = irrep_matrices[label]

        def unit(operators, row, column):
            return sum(
                irrep_dimension * irrep[row, column] * operator / 12.0
                for irrep, operator in zip(representation, operators)
            )

        e_projector = unit(d6_electric, 0, 0)
        h_projector = unit(d6_magnetic, 0, 0)
        e_values, e_vectors = np.linalg.eigh(0.5 * (e_projector + e_projector.T))
        h_values, h_vectors = np.linalg.eigh(0.5 * (h_projector + h_projector.T))
        e_basis = e_vectors[:, e_values > 0.5]
        h_basis = h_vectors[:, h_values > 0.5]
        p_sub = e_basis.T @ p_d6 @ h_basis
        q_sub = h_basis.T @ q_d6 @ e_basis
        kz2, e_sub = np.linalg.eig(p_sub @ q_sub)
        kz = np.sqrt(kz2.astype(complex))
        kz[np.abs(kz) < 1e-12] = 1e-12
        e_modes = e_basis @ e_sub
        h_modes = q_d6 @ e_modes / kz[None, :]
        scale = max(
            np.linalg.norm(e_modes * kz[None, :]),
            np.linalg.norm(h_modes * kz[None, :]),
            1e-30,
        )
        modal_residual = max(
            modal_residual,
            float(np.linalg.norm(p_d6 @ h_modes - e_modes * kz[None, :]) / scale),
            float(np.linalg.norm(q_d6 @ e_modes - h_modes * kz[None, :]) / scale),
        )
        reconstructed_electric.append(e_modes)
        if irrep_dimension == 2:
            e_partner = unit(d6_electric, 1, 0) @ e_modes
            h_partner = unit(d6_magnetic, 1, 0) @ h_modes
            modal_residual = max(
                modal_residual,
                float(
                    np.linalg.norm(
                        q_d6 @ e_partner - h_partner * kz[None, :]
                    )
                    / scale
                ),
            )
            reconstructed_electric.append(e_partner)
    all_electric = np.concatenate(reconstructed_electric, axis=1)
    modal_residual = max(
        modal_residual,
        float(abs(np.linalg.matrix_rank(all_electric, tol=1e-8) - size)),
    )
    checks.append(
        Check(
            "D6 matrix-unit Maxwell eigensolve reconstruction",
            modal_residual,
            2e-10,
            f"reconstructed dimension={all_electric.shape[1]}",
        )
    )

    # Independent multilayer field-coupling reconstruction.  Modal amplitudes
    # are defined as [a+(left), a-(right)], matching fields.py.
    modal_size = 5
    layer_count = 3
    identity_modal = np.eye(modal_size, dtype=complex)

    def random_invertible():
        return np.eye(modal_size) + 0.08 * rng.normal(
            size=(modal_size, modal_size)
        )

    input_v = random_invertible().astype(complex)
    output_v = random_invertible().astype(complex)
    layer_e = [random_invertible().astype(complex) for _ in range(layer_count)]
    layer_h = [random_invertible().astype(complex) for _ in range(layer_count)]
    layer_x = [
        np.diag(np.exp(-rng.uniform(0.05, 0.35, modal_size)))
        for _ in range(layer_count)
    ]

    def modal_basis(electric, magnetic):
        return np.block([[electric, electric], [magnetic, -magnetic]])

    left = []
    right = []
    zero_modal = np.zeros_like(identity_modal)
    for electric, magnetic, phase in zip(layer_e, layer_h, layer_x):
        basis = modal_basis(electric, magnetic)
        left.append(
            basis @ np.block([[identity_modal, zero_modal], [zero_modal, phase]])
        )
        right.append(
            basis @ np.block([[phase, zero_modal], [zero_modal, identity_modal]])
        )
    input_basis = modal_basis(identity_modal, input_v)
    output_basis = modal_basis(identity_modal, output_v)
    state_transfer = np.eye(2 * modal_size, dtype=complex)
    for left_matrix, right_matrix in zip(left, right):
        state_transfer = right_matrix @ np.linalg.solve(
            left_matrix, state_transfer
        )
    amplitude_transfer = np.linalg.solve(
        output_basis, state_transfer @ input_basis
    )
    t11 = amplitude_transfer[:modal_size, :modal_size]
    t12 = amplitude_transfer[:modal_size, modal_size:]
    t21 = amplitude_transfer[modal_size:, :modal_size]
    t22 = amplitude_transfer[modal_size:, modal_size:]
    rf = -np.linalg.solve(t22, t21)
    tf = t11 + t12 @ rf
    tb = np.linalg.solve(t22, identity_modal)
    rb = t12 @ tb

    forward_state = input_basis @ np.vstack((identity_modal, rf))
    backward_state = output_basis @ np.vstack((rb, identity_modal))
    field_coupling_error = 0.0
    for layer_index in range(layer_count):
        amplitudes = np.linalg.solve(left[layer_index], forward_state)
        forward_state = right[layer_index] @ amplitudes
    for layer_index in range(layer_count - 1, -1, -1):
        amplitudes = np.linalg.solve(right[layer_index], backward_state)
        backward_state = left[layer_index] @ amplitudes
    field_coupling_error = max(
        float(
            np.linalg.norm(
                forward_state - output_basis @ np.vstack((tf, zero_modal))
            )
        ),
        float(
            np.linalg.norm(
                backward_state - input_basis @ np.vstack((zero_modal, tb))
            )
        ),
    )
    checks.append(
        Check(
            "reduced bidirectional internal-field coupling",
            field_coupling_error,
            3e-11,
            f"layers={layer_count}, modal dimension={modal_size}",
        )
    )

    box_orders = [(m, n) for m in range(-4, 5) for n in range(-4, 5)]
    box_index = {pair: position for position, pair in enumerate(box_orders)}
    inversion = np.zeros((len(box_orders), len(box_orders)))
    for column, (m, n) in enumerate(box_orders):
        inversion[box_index[(-m, -n)], column] = 1.0
    vector_inversion = np.kron(np.eye(2), inversion)
    c2_source_projector = 0.5 * (
        np.eye(2 * len(box_orders)) + vector_inversion
    )
    zero_harmonic = box_index[(0, 0)]
    source_x = np.zeros(2 * len(box_orders)); source_x[zero_harmonic] = 1.0
    source_y = np.zeros(2 * len(box_orders)); source_y[len(box_orders) + zero_harmonic] = 1.0
    source_error = max(
        float(np.linalg.norm(c2_source_projector @ source_x - source_x)),
        float(np.linalg.norm(c2_source_projector @ source_y - source_y)),
    )
    checks.append(
        Check(
            "general-oblique C2 sector contains both x/y zero-order sources",
            source_error,
            2e-13,
        )
    )
    c2_rank = np.linalg.matrix_rank(c2_source_projector)
    checks.append(
        Check(
            "general-oblique C2 source-sector dimension reduction",
            0.0 if c2_rank < 2 * len(box_orders) else 1.0,
            0.0,
            f"dimension={c2_rank}/{2 * len(box_orders)}",
        )
    )
    return checks


def integration_checks(order: int, grid: int) -> tuple[list[Check], dict[str, object]]:
    import torch
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
    )

    torch.set_default_dtype(torch.float64)

    def make(
        method="matched-asr",
        algorithm="redheffer",
        size="half",
        pol=None,
        symmetry="auto",
        selected_order=order,
        factorization=True,
        fields="none",
        smatrix=True,
    ):
        sim = AutoRCWA(
            freq=1 / 1.55,
            order=[selected_order, selected_order],
            lattice=Lattice.triangular(1.0),
            cascade=algorithm,
            outputs=OutputSpec(
                smatrix=smatrix, smatrix_size=size, fields=fields
            ),
            asr=ASROptions(circle_G=0.08, grid=(grid, grid)),
            nvm=NVMOptions(grid=(grid, grid)),
            group_theory=GroupTheoryOptions(
                enabled=pol is not None or symmetry == "d6",
                symmetry=symmetry,
                strict=pol is not None or symmetry == "d6",
                polarization=pol,
            ),
            verify_cascade=fields != "none",
            compute_condition_numbers=True,
            dtype=torch.complex128,
            device="cuda",
        )
        sim.add_input_layer(eps=1.0)
        sim.add_output_layer(eps=1.0)
        sim.set_incident_angle(0.0, 0.0)
        sim.add_structured_layer(
            LayerSpec(
                thickness=0.18,
                geometry=Circle(0.24),
                background=Material(1.0),
                inclusion=Material(4.0),
                method=method,
                factorization_rules=factorization if method == "matched-asr" else None,
            )
        )
        sim.solve_global_smatrix()
        return sim

    checks: list[Check] = []
    full_r = make(size="full", algorithm="redheffer")
    full_l = make(size="full", algorithm="algo2a")
    for index, name in enumerate(("Tf", "Rf", "Rb", "Tb")):
        error = float(torch.max(torch.abs(full_r.S[index] - full_l.S[index])))
        checks.append(Check(f"triangular Li2a/Redheffer {name}", error, 3e-8))
    for algorithm, full in (("redheffer", full_r), ("algo2a", full_l)):
        half = make(size="half", algorithm=algorithm)
        quarter = make(size="quarter", algorithm=algorithm)
        checks.append(Check(f"{algorithm} half/full Tf", float(torch.max(torch.abs(half.S[0]-full.S[0]))), 3e-8))
        checks.append(Check(f"{algorithm} half/full Rf", float(torch.max(torch.abs(half.S[1]-full.S[1]))), 3e-8))
        checks.append(Check(f"{algorithm} quarter/full Rf", float(torch.max(torch.abs(quarter.S[1]-full.S[1]))), 3e-8))

    zero_x = int(torch.nonzero(full_r.order_x == 0, as_tuple=False)[0])
    zero_y = int(torch.nonzero(full_r.order_y == 0, as_tuple=False)[0])
    harmonic = zero_x * len(full_r.order_y) + zero_y
    for offset, label in ((0, "x"), (full_r.order_N, "y")):
        source = torch.zeros(2 * full_r.order_N, dtype=torch.complex128)
        source[offset + harmonic] = 1.0
        transmitted, reflected = full_r.S[0] @ source, full_r.S[1] @ source
        power = sum(float(torch.abs(vector[index]) ** 2) for vector in (transmitted, reflected) for index in (harmonic, full_r.order_N + harmonic))
        checks.append(Check(f"lossless power ({label})", abs(power-1.0), 5e-9, f"R+T={power:.15f}"))

    reduced: dict[tuple[str, str], AutoRCWA] = {}
    for algorithm in ("redheffer", "algo2a"):
        for pol in ("x", "y"):
            reduced[(algorithm, pol)] = make(algorithm=algorithm, size="half", pol=pol)
            source = torch.zeros(2 * full_r.order_N, dtype=torch.complex128)
            source[(0 if pol == "x" else full_r.order_N) + harmonic] = 1.0
            red = reduced[("redheffer", pol)] if ("redheffer", pol) in reduced else None
            if red is not None and algorithm == "algo2a":
                for block, label in ((0, "Tf"), (1, "Rf")):
                    error = float(torch.max(torch.abs((red.S[block]-reduced[(algorithm, pol)].S[block]) @ source)))
                    checks.append(Check(f"reduced {pol} Li2a/Redheffer {label}", error, 3e-8))
            diagnostic = reduced[(algorithm, pol)].group_theory_diagnostics[-1]
            checks.append(Check(f"reduced {pol} operator invariance", float(diagnostic["max_invariance_residual"]), 2e-10))
            checks.append(Check(f"reduced {pol} conversion invariance", float(diagnostic["conversion_residual"]), 2e-10))

    for pol in ("x", "y"):
        quarter = make(algorithm="redheffer", size="quarter", pol=pol)
        checks.append(
            Check(
                f"reduced {pol} quarter/half Rf",
                float(torch.max(torch.abs(quarter.S[1] - reduced[("redheffer", pol)].S[1]))),
                3e-8,
            )
        )
    direct_rule = make(pol="x", factorization=False)
    direct_diagnostic = direct_rule.group_theory_diagnostics[-1]
    checks.append(
        Check(
            "reduced direct-rule operator invariance",
            float(direct_diagnostic["max_invariance_residual"]),
            2e-10,
        )
    )

    # Independent full-star modal solve.  The sum of the two mirror-sector S
    # matrices must reproduce it because the x/y sectors span the D6 star.
    base = full_r
    tensors = base.asr_material_tensors[0]
    p_star, q_star = base._build_triangular_star_pq(
        *(tensors[name] for name in ("eps11", "eps12", "eps21", "eps22", "eps33", "mu11", "mu12", "mu21", "mu22", "mu33")),
        factorization_rules=True,
    )
    vector_embedding, *_ = base._triangular_star_operators()
    kz2, w_uv = base._eig(p_star @ q_star)
    kz = base._positive_kz(kz2)
    h_uv = base._magnetic_eigenvectors(p_star, q_star, w_uv, kz)
    t_star = vector_embedding.mH @ base.asr_T_matrices[0] @ vector_embedding
    w_cart, h_cart = t_star @ w_uv, t_star @ h_uv
    v_star = vector_embedding.mH @ base.Vf @ vector_embedding
    base._polarization_reference_v = v_star
    interface_in = base._reduced_interface_s(v_star, v_star, input_side=True)
    interface_out = base._reduced_interface_s(v_star, v_star, input_side=False)
    layer = {"electric": w_cart, "magnetic": h_cart, "kz": kz}
    saved_size = base.smatrix_size
    base.smatrix_size = "half"
    reference_star = base._polarized_redheffer_scattering([layer], interface_in, interface_out)
    base.smatrix_size = saved_size
    for block, label in ((0, "Tf"), (1, "Rf")):
        reference = vector_embedding @ reference_star[block] @ vector_embedding.mH
        sectors = reduced[("redheffer", "x")].S[block] + reduced[("redheffer", "y")].S[block]
        checks.append(
            Check(
                f"x+y sectors/full D6-star {label}",
                float(torch.max(torch.abs(sectors-reference))),
                2e-4,
                "The unreduced reference eigensolve mixes exactly degenerate D6 modes.",
            )
        )

    # Triangular NVM is assembled directly on the same D6-closed star.  The
    # Fourier inverse rules are recomputed after star restriction, so this is
    # not a rectangular inverse followed by a lossy projection.
    nvm_reduced: dict[tuple[str, str], AutoRCWA] = {}
    for algorithm in ("redheffer", "algo2a"):
        for pol in ("x", "y"):
            nvm_reduced[(algorithm, pol)] = make(
                method="nvm", algorithm=algorithm, size="half", pol=pol
            )
            diagnostic = nvm_reduced[(algorithm, pol)].group_theory_diagnostics[-1]
            checks.append(
                Check(
                    f"NVM reduced {pol} backend",
                    0.0
                    if diagnostic.get("backend") == "NVM"
                    and diagnostic.get("symmetry") == "D6-star/Cs(x-mirror)"
                    else 1.0,
                    0.0,
                )
            )
            checks.append(
                Check(
                    f"NVM reduced {pol} operator invariance",
                    float(diagnostic["max_invariance_residual"]),
                    2e-10,
                )
            )
            checks.append(
                Check(
                    f"NVM reduced {pol} conversion invariance",
                    float(diagnostic["conversion_residual"]),
                    2e-10,
                )
            )

    nvm_base = make(method="nvm", algorithm="redheffer", size="half")
    nvm_vector_embedding, *_ = nvm_base._triangular_star_operators()
    nvm_radius = torch.tensor(0.24, dtype=torch.float64)
    nvm_centers = nvm_base.layer_records[0].options["centers"]
    inverse_eps_rule = nvm_base._circle_toeplitz(
        nvm_radius,
        1.0,
        0.25,
        nvm_centers,
        use_lanczos=False,
        lanczos_power=2,
    )
    nvm_projection = nvm_base._projection_matrix(
        nvm_radius, nvm_centers, nx=grid, ny=grid
    )
    nvm_p_star, nvm_q_star = nvm_base._build_triangular_nvm_star_pq(
        nvm_base.eps_conv[0], inverse_eps_rule, nvm_projection
    )
    nvm_kz2, nvm_w_uv = nvm_base._eig(nvm_p_star @ nvm_q_star)
    nvm_kz = nvm_base._positive_kz(nvm_kz2)
    nvm_h_uv = nvm_base._magnetic_eigenvectors(
        nvm_p_star, nvm_q_star, nvm_w_uv, nvm_kz
    )
    identity_nvm = nvm_base._eye(nvm_base.order_N)
    zero_nvm = torch.zeros_like(identity_nvm)
    s_nvm, c_nvm = nvm_base.sin_zeta, nvm_base.cos_zeta
    nvm_transform = torch.cat(
        (
            torch.cat((identity_nvm, zero_nvm), dim=1),
            torch.cat(
                (
                    -(c_nvm / s_nvm) * identity_nvm,
                    (1.0 / s_nvm) * identity_nvm,
                ),
                dim=1,
            ),
        ),
        dim=0,
    )
    nvm_transform_star = (
        nvm_vector_embedding.mH @ nvm_transform @ nvm_vector_embedding
    )
    nvm_w_cart = nvm_transform_star @ nvm_w_uv
    nvm_h_cart = nvm_transform_star @ nvm_h_uv
    nvm_v_star = nvm_vector_embedding.mH @ nvm_base.Vf @ nvm_vector_embedding
    nvm_base._polarization_reference_v = nvm_v_star
    nvm_interface_in = nvm_base._reduced_interface_s(
        nvm_v_star, nvm_v_star, input_side=True
    )
    nvm_interface_out = nvm_base._reduced_interface_s(
        nvm_v_star, nvm_v_star, input_side=False
    )
    nvm_layer = {
        "electric": nvm_w_cart,
        "magnetic": nvm_h_cart,
        "kz": nvm_kz,
    }
    nvm_reference_star = nvm_base._polarized_redheffer_scattering(
        [nvm_layer], nvm_interface_in, nvm_interface_out
    )
    for block, label in ((0, "Tf"), (1, "Rf")):
        nvm_reference = (
            nvm_vector_embedding
            @ nvm_reference_star[block]
            @ nvm_vector_embedding.mH
        )
        nvm_sectors = (
            nvm_reduced[("redheffer", "x")].S[block]
            + nvm_reduced[("redheffer", "y")].S[block]
        )
        checks.append(
            Check(
                f"NVM x+y sectors/full D6-star {label}",
                float(torch.max(torch.abs(nvm_sectors - nvm_reference))),
                2e-4,
            )
        )
        for pol in ("x", "y"):
            source = torch.zeros(2 * nvm_base.order_N, dtype=torch.complex128)
            source[(0 if pol == "x" else nvm_base.order_N) + harmonic] = 1.0
            checks.append(
                Check(
                    f"NVM reduced/full-star {pol} {label}",
                    float(
                        torch.max(
                            torch.abs(
                                (nvm_reduced[("redheffer", pol)].S[block] - nvm_reference)
                                @ source
                            )
                        )
                    ),
                    2e-4,
                )
            )

    for pol in ("x", "y"):
        source = torch.zeros(2 * nvm_base.order_N, dtype=torch.complex128)
        source[(0 if pol == "x" else nvm_base.order_N) + harmonic] = 1.0
        for block, label in ((0, "Tf"), (1, "Rf")):
            checks.append(
                Check(
                    f"NVM reduced {pol} Li2a/Redheffer {label}",
                    float(
                        torch.max(
                            torch.abs(
                                (
                                    nvm_reduced[("algo2a", pol)].S[block]
                                    - nvm_reduced[("redheffer", pol)].S[block]
                                )
                                @ source
                            )
                        )
                    ),
                    3e-8,
                )
            )

        nvm_quarter = make(
            method="nvm", algorithm="redheffer", size="quarter", pol=pol
        )
        checks.append(
            Check(
                f"NVM reduced {pol} quarter/half Rf",
                float(
                    torch.max(
                        torch.abs(
                            (nvm_quarter.S[1] - nvm_reduced[("redheffer", pol)].S[1])
                            @ source
                        )
                    )
                ),
                3e-8,
            )
        )

    # Complete native-star D6: enumerate all six isotypic components, solve
    # one matrix-unit row for E1/E2, and recover the full-star response.
    complete_d6: dict[tuple[str, str], AutoRCWA] = {}
    references = {
        "matched-asr": (vector_embedding, reference_star),
        "nvm": (nvm_vector_embedding, nvm_reference_star),
    }
    for method in ("matched-asr", "nvm"):
        for algorithm in ("redheffer", "algo2a"):
            complete_d6[(method, algorithm)] = make(
                method=method,
                algorithm=algorithm,
                size="full",
                symmetry="d6",
            )
        red_complete = complete_d6[(method, "redheffer")]
        li_complete = complete_d6[(method, "algo2a")]
        diagnostic = red_complete.group_theory_diagnostics[-1]
        irrep_records = diagnostic["irreps"]
        labels = tuple(record["irrep"] for record in irrep_records)
        dimension_sum = sum(int(record["isotypic_dimension"]) for record in irrep_records)
        matrix_unit_rank_error = max(
            abs(
                int(record["solved_dimension"])
                * int(record["irrep_dimension"])
                - int(record["isotypic_dimension"])
            )
            for record in irrep_records
        )
        checks.append(
            Check(
                f"{method} complete D6 labels",
                0.0
                if labels == ("A1", "A2", "B1", "B2", "E1", "E2")
                else 1.0,
                0.0,
            )
        )
        checks.append(
            Check(
                f"{method} complete D6 dimension sum",
                float(abs(dimension_sum - int(diagnostic["star_vector_dimension"]))),
                0.0,
            )
        )
        checks.append(
            Check(
                f"{method} complete D6 matrix-unit ranks",
                float(matrix_unit_rank_error),
                0.0,
            )
        )
        checks.append(
            Check(
                f"{method} complete D6 projector completeness",
                float(diagnostic["projector_completeness"]),
                2e-10,
            )
        )
        for block, label in enumerate(("Tf", "Rf", "Rb", "Tb")):
            checks.append(
                Check(
                    f"{method} complete D6 Li2a/Redheffer {label}",
                    float(
                        torch.max(
                            torch.abs(
                                li_complete.S[block] - red_complete.S[block]
                            )
                        )
                    ),
                    3e-8,
                )
            )
        embedding, reference_blocks = references[method]
        for block, label in ((0, "Tf"), (1, "Rf")):
            reference = embedding @ reference_blocks[block] @ embedding.mH
            checks.append(
                Check(
                    f"{method} complete D6/full-star {label}",
                    float(torch.max(torch.abs(red_complete.S[block] - reference))),
                    2e-4,
                )
            )

        half = make(
            method=method,
            algorithm="redheffer",
            size="half",
            symmetry="d6",
        )
        quarter = make(
            method=method,
            algorithm="redheffer",
            size="quarter",
            symmetry="d6",
        )
        checks.append(
            Check(
                f"{method} complete D6 half/full Tf",
                float(torch.max(torch.abs(half.S[0] - red_complete.S[0]))),
                3e-8,
            )
        )
        checks.append(
            Check(
                f"{method} complete D6 half/full Rf",
                float(torch.max(torch.abs(half.S[1] - red_complete.S[1]))),
                3e-8,
            )
        )
        checks.append(
            Check(
                f"{method} complete D6 quarter/full Rf",
                float(torch.max(torch.abs(quarter.S[1] - red_complete.S[1]))),
                3e-8,
            )
        )

        # Internal/external fields must remain available for every public S
        # size and both cascade selections.  The quarter instance exposes Rf
        # only, but retains a private full field S matrix and both C directions.
        field_full = make(
            method=method,
            algorithm="redheffer",
            size="full",
            symmetry="d6",
            fields="all",
        )
        field_partial = make(
            method=method,
            algorithm="algo2a",
            size="quarter",
            symmetry="d6",
            fields="all",
        )
        field_sector = make(
            method=method,
            algorithm="redheffer",
            size="quarter",
            pol="x",
            fields="all",
        )
        field_only = make(
            method=method,
            algorithm="algo2a",
            size="quarter",
            symmetry="d6",
            fields="all",
            smatrix=False,
        )

        def set_source(sim, component, direction):
            source = torch.zeros(
                (2 * sim.order_N, 1), dtype=torch.complex128
            )
            source[(0 if component == "x" else sim.order_N) + harmonic, 0] = 1.0
            sim.E_i = source
            sim.source_direction = direction

        for direction, component in (("forward", "x"), ("backward", "y")):
            set_source(field_full, component, direction)
            set_source(field_partial, component, direction)
            for z_value in (0.0, 0.09, 0.18):
                reference_fields = field_full._fourier_fields(0, z_value)
                partial_fields = field_partial._fourier_fields(0, z_value)
                checks.append(
                    Check(
                        f"{method} D6 {direction} quarter/full internal fields z={z_value:g}",
                        max(
                            float(torch.max(torch.abs(a - b)))
                            for a, b in zip(reference_fields, partial_fields)
                        ),
                        5e-8,
                    )
                )
        set_source(field_full, "x", "forward")
        set_source(field_partial, "x", "forward")
        set_source(field_sector, "x", "forward")
        set_source(field_only, "x", "forward")
        for z_value in (0.0, 0.09, 0.18):
            complete_fields = field_full._fourier_fields(0, z_value)
            sector_fields = field_sector._fourier_fields(0, z_value)
            checks.append(
                Check(
                    f"{method} Cs/complete-D6 internal fields z={z_value:g}",
                    max(
                        float(torch.max(torch.abs(a - b)))
                        for a, b in zip(complete_fields, sector_fields)
                    ),
                    2e-4,
                )
            )
            field_only_values = field_only._fourier_fields(0, z_value)
            checks.append(
                Check(
                    f"{method} fields-only/internal reference z={z_value:g}",
                    max(
                        float(torch.max(torch.abs(a - b)))
                        for a, b in zip(complete_fields, field_only_values)
                    ),
                    5e-8,
                )
            )
        checks.append(
            Check(
                f"{method} fields-only public S disabled",
                max(float(torch.max(torch.abs(block))) for block in field_only.S),
                0.0,
            )
        )

        internal_left = field_partial._fourier_fields(0, 0.0)
        external_left = field_partial._fourier_fields(-1, 0.0)
        internal_right = field_partial._fourier_fields(0, 0.18)
        external_right = field_partial._fourier_fields(
            field_partial.layer_N, 0.0
        )
        tangential = (0, 1, 3, 4)
        checks.append(
            Check(
                f"{method} D6 input tangential-field continuity",
                max(
                    float(torch.max(torch.abs(internal_left[i] - external_left[i])))
                    for i in tangential
                ),
                5e-8,
            )
        )
        checks.append(
            Check(
                f"{method} D6 output tangential-field continuity",
                max(
                    float(torch.max(torch.abs(internal_right[i] - external_right[i])))
                    for i in tangential
                ),
                5e-8,
            )
        )
        electric_xy, magnetic_xy = field_partial.field_xy(
            0, [0.0, 0.2], [0.0, 0.2], z_prop=0.09
        )
        spatial_values = (*electric_xy, *magnetic_xy)
        spatial_error = 0.0 if all(
            bool(torch.all(torch.isfinite(value)))
            and tuple(value.shape) == (2, 2)
            for value in spatial_values
        ) else 1.0
        checks.append(
            Check(
                f"{method} D6 spatial field synthesis",
                spatial_error,
                0.0,
            )
        )
        star_count = embedding.shape[1] // 2
        scalar_embedding = embedding[
            : field_partial.order_N, :star_count
        ]
        outside_star = torch.sum(torch.abs(scalar_embedding), dim=1) == 0
        star_fields = field_partial._fourier_fields(0, 0.09)
        corner_error = (
            max(
                float(torch.max(torch.abs(component[outside_star])))
                for component in star_fields
            )
            if bool(torch.any(outside_star))
            else 0.0
        )
        checks.append(
            Check(
                f"{method} D6 internal fields contain no corner harmonics",
                corner_error,
                2e-10,
            )
        )

    # Source-specific complete D6: solve only the E1 matrix-unit row reached
    # by the requested zero-order x/y source.  Compare against both the
    # all-irrep complete-D6 solution and the older Cs mirror sector.
    d6_source: dict[tuple[str, str, str], AutoRCWA] = {}
    for method in ("matched-asr", "nvm"):
        cs_reference = reduced if method == "matched-asr" else nvm_reduced
        for pol in ("x", "y"):
            for algorithm in ("redheffer", "algo2a"):
                d6_source[(method, algorithm, pol)] = make(
                    method=method,
                    algorithm=algorithm,
                    size="half",
                    pol=pol,
                    symmetry="d6",
                )
            red_source = d6_source[(method, "redheffer", pol)]
            li_source = d6_source[(method, "algo2a", pol)]
            diagnostic = red_source.group_theory_diagnostics[-1]
            expected_dimension = order * (order + 1) + 1
            checks.append(
                Check(
                    f"{method} D6-E1 {pol} source-row label",
                    0.0
                    if diagnostic["symmetry"] == "D6-E1-source-row"
                    and diagnostic["irrep"] == "E1"
                    else 1.0,
                    0.0,
                )
            )
            checks.append(
                Check(
                    f"{method} D6-E1 {pol} source-row dimension",
                    float(
                        abs(
                            int(diagnostic["reduced_dimension"])
                            - expected_dimension
                        )
                    ),
                    0.0,
                )
            )
            checks.append(
                Check(
                    f"{method} D6-E1 {pol} source projection",
                    float(diagnostic["source_projection_residual"]),
                    2e-10,
                )
            )
            checks.append(
                Check(
                    f"{method} D6-E1 {pol} operator invariance",
                    float(diagnostic["max_invariance_residual"]),
                    2e-10,
                )
            )
            source = torch.zeros(
                2 * red_source.order_N, dtype=torch.complex128
            )
            source[
                (0 if pol == "x" else red_source.order_N) + harmonic
            ] = 1.0
            for block, label in ((0, "Tf"), (1, "Rf")):
                checks.append(
                    Check(
                        f"{method} D6-E1/all-irrep {pol} {label}",
                        float(
                            torch.max(
                                torch.abs(
                                    (
                                        red_source.S[block]
                                        - complete_d6[(method, "redheffer")].S[block]
                                    )
                                    @ source
                                )
                            )
                        ),
                        5e-8,
                    )
                )
                checks.append(
                    Check(
                        f"{method} D6-E1/Cs {pol} {label}",
                        float(
                            torch.max(
                                torch.abs(
                                    (
                                        red_source.S[block]
                                        - cs_reference[("redheffer", pol)].S[block]
                                    )
                                    @ source
                                )
                            )
                        ),
                        2e-4,
                    )
                )
                checks.append(
                    Check(
                        f"{method} D6-E1 Li2a/Redheffer {pol} {label}",
                        float(
                            torch.max(
                                torch.abs(
                                    (li_source.S[block] - red_source.S[block])
                                    @ source
                                )
                            )
                        ),
                        3e-8,
                    )
                )

        quarter_source = make(
            method=method,
            algorithm="redheffer",
            size="quarter",
            pol="x",
            symmetry="d6",
        )
        source_x = torch.zeros(
            2 * quarter_source.order_N, dtype=torch.complex128
        )
        source_x[harmonic] = 1.0
        checks.append(
            Check(
                f"{method} D6-E1 x quarter/half Rf",
                float(
                    torch.max(
                        torch.abs(
                            (
                                quarter_source.S[1]
                                - d6_source[(method, "redheffer", "x")].S[1]
                            )
                            @ source_x
                        )
                    )
                ),
                3e-8,
            )
        )

        complete_field = make(
            method=method,
            algorithm="redheffer",
            size="full",
            symmetry="d6",
            fields="all",
        )
        source_field = make(
            method=method,
            algorithm="algo2a",
            size="quarter",
            pol="x",
            symmetry="d6",
            fields="all",
        )
        source_column = source_x[:, None]
        for simulation in (complete_field, source_field):
            simulation.E_i = source_column
            simulation.source_direction = "forward"
        for z_value in (0.0, 0.09, 0.18):
            reference_fields = complete_field._fourier_fields(0, z_value)
            reduced_fields = source_field._fourier_fields(0, z_value)
            checks.append(
                Check(
                    f"{method} D6-E1 quarter/full internal fields z={z_value:g}",
                    max(
                        float(torch.max(torch.abs(a - b)))
                        for a, b in zip(reference_fields, reduced_fields)
                    ),
                    2e-4,
                )
            )

    # Orthogonal matched-ASR uses the existing C2v blocks and must agree with
    # an unreduced solve for a source in the selected sector.
    def make_square(
        pol=None,
        *,
        method="matched-asr",
        fields="none",
        size="half",
        algorithm="redheffer",
    ):
        sim = AutoRCWA(
            freq=1/1.55, order=[order, order], lattice=Lattice.square(1.0),
            cascade=algorithm,
            outputs=OutputSpec(smatrix_size=size, fields=fields),
            asr=ASROptions(circle_G=0.08, grid=(grid, grid)),
            nvm=NVMOptions(grid=(grid, grid)),
            group_theory=GroupTheoryOptions(
                enabled=pol is not None,
                strict=pol is not None,
                polarization=pol,
            ),
            verify_cascade=fields != "none", dtype=torch.complex128, device="cuda",
        )
        sim.add_input_layer(); sim.add_output_layer(); sim.set_incident_angle(0.0, 0.0)
        sim.add_structured_layer(LayerSpec(thickness=.18, geometry=Circle(.24), background=Material(1.0), inclusion=Material(4.0), method=method))
        sim.solve_global_smatrix()
        return sim

    square_full = make_square()
    square_harmonic = int(torch.nonzero(square_full.order_x == 0)[0]) * len(square_full.order_y) + int(torch.nonzero(square_full.order_y == 0)[0])
    for pol in ("x", "y"):
        square_reduced = make_square(pol)
        source = torch.zeros(2*square_full.order_N, dtype=torch.complex128)
        source[(0 if pol == "x" else square_full.order_N) + square_harmonic] = 1.0
        for block, label in ((0, "Tf"), (1, "Rf")):
            checks.append(Check(f"orthogonal reduced/full {pol} {label}", float(torch.max(torch.abs((square_reduced.S[block]-square_full.S[block])@source))), 5e-8))

    # The orthogonal C2v paths use a different sector construction from the
    # triangular Cs path.  Check both matched-ASR and NVM longitudinal fields,
    # and combine Li-2a with a quarter public S to exercise the private field S.
    for method in ("matched-asr", "nvm"):
        square_field_full = make_square(
            method=method, fields="all", size="full"
        )
        square_field_reduced = make_square(
            "x",
            method=method,
            fields="all",
            size="quarter",
            algorithm="algo2a",
        )
        source = torch.zeros(
            (2 * square_field_full.order_N, 1), dtype=torch.complex128
        )
        source[square_harmonic, 0] = 1.0
        for simulation in (square_field_full, square_field_reduced):
            simulation.E_i = source
            simulation.source_direction = "forward"
        for z_value in (0.0, 0.09, 0.18):
            reference_fields = square_field_full._fourier_fields(0, z_value)
            reduced_fields = square_field_reduced._fourier_fields(0, z_value)
            checks.append(
                Check(
                    f"orthogonal {method} C2v quarter/full internal fields z={z_value:g}",
                    max(
                        float(torch.max(torch.abs(a - b)))
                        for a, b in zip(reference_fields, reduced_fields)
                    ),
                    2e-4,
                )
            )
        for external_layer in (-1, square_field_reduced.layer_N):
            reference_fields = square_field_full._fourier_fields(
                external_layer, 0.0
            )
            reduced_fields = square_field_reduced._fourier_fields(
                external_layer, 0.0
            )
            checks.append(
                Check(
                    f"orthogonal {method} C2v quarter/full external fields layer={external_layer}",
                    max(
                        float(torch.max(torch.abs(a - b)))
                        for a, b in zip(reference_fields, reduced_fields)
                    ),
                    2e-4,
                )
            )

    convergence = []
    for selected_order in range(1, order + 1):
        matched = make(selected_order=selected_order)
        nvm = make(method="nvm", selected_order=selected_order)
        zx = int(torch.nonzero(matched.order_x == 0)[0]); zy = int(torch.nonzero(matched.order_y == 0)[0])
        h0 = zx * len(matched.order_y) + zy
        source = torch.zeros(2*matched.order_N, dtype=torch.complex128); source[h0] = 1.0
        difference = max(float(torch.max(torch.abs((matched.S[i]-nvm.S[i])@source))) for i in (0,1))
        convergence.append({"order": selected_order, "matched_nvm_source_error": difference})

    diagnostic_x = reduced[("redheffer", "x")].group_theory_diagnostics[-1]
    nvm_diagnostic_x = nvm_reduced[("redheffer", "x")].group_theory_diagnostics[-1]
    diagnostics = {
        "triangular_matched_nvm_convergence": convergence,
        "full_eigenproblem_dimension": 2 * full_r.order_N,
        "star_vector_dimension": diagnostic_x["star_vector_dimension"],
        "single_polarization_dimension": diagnostic_x["reduced_dimension"],
        "dimension_ratio": diagnostic_x["reduced_dimension"] / (2 * full_r.order_N),
        "T_condition_number": float(full_r.asr_condition_numbers[-1]),
        "nvm_star_vector_dimension": nvm_diagnostic_x["star_vector_dimension"],
        "nvm_single_polarization_dimension": nvm_diagnostic_x["reduced_dimension"],
        "nvm_dimension_ratio": nvm_diagnostic_x["reduced_dimension"] / (2 * nvm_base.order_N),
        "complete_d6": {
            method: complete_d6[(method, "redheffer")].group_theory_diagnostics[-1]
            for method in ("matched-asr", "nvm")
        },
        "d6_e1_source_rows": {
            method: {
                pol: d6_source[(method, "redheffer", pol)].group_theory_diagnostics[-1]
                for pol in ("x", "y")
            }
            for method in ("matched-asr", "nvm")
        },
    }
    return checks, diagnostics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--integration", action="store_true")
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--grid", type=int, default=128)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    checks = core_checks()
    diagnostics: dict[str, object] = {}
    if args.integration:
        extra, diagnostics = integration_checks(args.order, args.grid)
        checks.extend(extra)
    result = {
        "passed": all(item.passed for item in checks),
        "passed_count": sum(item.passed for item in checks),
        "total_count": len(checks),
        "checks": [{**asdict(item), "passed": item.passed} for item in checks],
        "diagnostics": diagnostics,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.json:
        args.json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
