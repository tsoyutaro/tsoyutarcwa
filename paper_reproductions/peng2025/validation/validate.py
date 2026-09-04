"""Fast validation for the Peng 2025 square/hexagonal execution scripts."""

from __future__ import annotations

import argparse
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
    annular_aperture_fill_fraction,
    rectangular_supercell_grid_y,
    rectangular_supercell_material,
    select_device,
    simulate_matched_primitive,
    simulate_rectangular_supercell,
    simulate_triangular_raster_primitive,
)
from rcwa_solver_auto import (
    ASROptions,
    AutoRCWA,
    GroupTheoryOptions,
    Lattice,
    OutputSpec,
)


def _check_power(result: dict[str, object]) -> dict[str, object]:
    values = {
        name: float(result[name])
        for name in ("reflectance", "transmittance", "absorptance")
    }
    finite = all(math.isfinite(value) for value in values.values())
    balance = sum(values.values())
    return {
        "finite": finite,
        "balance": balance,
        "balance_error": abs(balance - 1.0),
        "passivity_warning": bool(result["passivity_warning"]),
        "passed": finite
        and abs(balance - 1.0) <= 1.0e-10
        and not bool(result["passivity_warning"]),
    }


def _double_map_geometry(device: torch.device) -> dict[str, object]:
    """Check both mapped interfaces and positive Jacobians on both lattices."""

    errors: dict[str, float] = {}
    minimum_jacobians: dict[str, float] = {}
    fixed_masks: dict[str, bool] = {}
    grid = 96
    geometry = PaperGeometry()
    for name, lattice in (
        ("square", Lattice.square(1.0)),
        ("triangular", Lattice.triangular(1.0)),
    ):
        simulation = AutoRCWA(
            freq=0.2,
            order=[1, 1],
            lattice=lattice,
            outputs=OutputSpec(smatrix_size="quarter", fields="none"),
            asr=ASROptions(circle_G=3.0e-2, grid=(grid, grid)),
            verify_cascade=False,
            dtype=torch.complex128,
            device=device,
        )
        mapping = simulation.build_double_matched_circle_asr_mapping(
            grid,
            grid,
            geometry.inner_radius_um / geometry.period_um,
            geometry.outer_radius_um / geometry.period_um,
        )
        cosine = 0.5 if name == "triangular" else 0.0
        sine = 0.5 * math.sqrt(3.0) if name == "triangular" else 1.0
        center_x = 0.5 * (1.0 + cosine)
        center_y = 0.5 * sine
        interface_errors = []
        # Fixed support fractions 1/3 and 2/3 occur at offsets 16 and 32.
        for index, target in (
            (64, geometry.inner_radius_um / geometry.period_um),
            (80, geometry.outer_radius_um / geometry.period_um),
        ):
            radius = torch.sqrt(
                (mapping.x[index, 48] - center_x) ** 2
                + (mapping.y[index, 48] - center_y) ** 2
            )
            interface_errors.append(abs(float(radius.detach().cpu()) - target))
        errors[name] = max(interface_errors)
        minimum_jacobians[name] = float(mapping.det_j.min().detach().cpu())
        fixed_masks[name] = bool(mapping.matched_core_mask[64, 48]) and bool(
            mapping.matched_outer_mask[80, 48]
        )
    return {
        "maximum_interface_radius_error": errors,
        "minimum_jacobian": minimum_jacobians,
        "fixed_interface_masks": fixed_masks,
        "passed": max(errors.values()) < 1.0e-12
        and min(minimum_jacobians.values()) > 0.0
        and all(fixed_masks.values()),
    }


def _double_map_objective(
    core_radius: torch.Tensor,
    outer_radius: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Small dielectric S-matrix objective for radius-gradient validation."""

    simulation = AutoRCWA(
        freq=1.0 / 1.55,
        order=[1, 1],
        lattice=Lattice.square(1.0),
        outputs=OutputSpec(smatrix_size="half", fields="none"),
        asr=ASROptions(circle_G=8.0e-2, grid=(48, 48)),
        verify_cascade=False,
        dtype=torch.complex128,
        device=device,
    )
    simulation.add_input_layer(eps=1.0)
    simulation.add_output_layer(eps=1.0)
    simulation.set_incident_angle(0.07, 0.11)
    simulation.add_layer_circle_shell_asr(
        0.18,
        core_radius,
        outer_radius,
        1.0,
        2.5,
        4.0,
        nx=48,
        ny=48,
        factorization_rules=False,
        radial_mapping="double",
    )
    simulation.solve_global_smatrix()
    return torch.sum(torch.abs(simulation.S[1]) ** 2)


def _double_map_gradient(device: torch.device) -> dict[str, object]:
    core = torch.tensor(0.18, dtype=torch.float64, device=device, requires_grad=True)
    outer = torch.tensor(0.32, dtype=torch.float64, device=device, requires_grad=True)
    objective = _double_map_objective(core, outer, device=device)
    automatic = torch.autograd.grad(objective, (core, outer))
    step = 1.0e-5

    def fixed(core_value: float, outer_value: float) -> torch.Tensor:
        return _double_map_objective(
            torch.tensor(core_value, dtype=torch.float64, device=device),
            torch.tensor(outer_value, dtype=torch.float64, device=device),
            device=device,
        )

    finite_difference = (
        (fixed(0.18 + step, 0.32) - fixed(0.18 - step, 0.32)) / (2.0 * step),
        (fixed(0.18, 0.32 + step) - fixed(0.18, 0.32 - step)) / (2.0 * step),
    )
    ad = [float(value.detach().cpu()) for value in automatic]
    fd = [float(value.detach().cpu()) for value in finite_difference]
    scaled_errors = [
        abs(a - b) / max(1.0, abs(a), abs(b)) for a, b in zip(ad, fd)
    ]
    return {
        "objective": float(objective.detach().cpu()),
        "autograd": {"core_radius": ad[0], "outer_radius": ad[1]},
        "finite_difference": {"core_radius": fd[0], "outer_radius": fd[1]},
        "scaled_errors": {
            "core_radius": scaled_errors[0],
            "outer_radius": scaled_errors[1],
        },
        "passed": max(scaled_errors) < 2.0e-3
        and all(math.isfinite(value) for value in (*ad, *fd)),
    }


def _double_map_fields(device: torch.device) -> dict[str, object]:
    """Smoke-check six-component internal fields, including D6 E1 reduction."""

    finite: dict[str, bool] = {}
    for name, lattice, order, group in (
        (
            "square",
            Lattice.square(1.0),
            1,
            GroupTheoryOptions(enabled=False),
        ),
        (
            "triangular_d6_x",
            Lattice.triangular(1.0),
            4,
            GroupTheoryOptions(
                enabled=True, symmetry="d6", strict=True, polarization="x"
            ),
        ),
    ):
        simulation = AutoRCWA(
            freq=1.0 / 1.55,
            order=[order, order],
            lattice=lattice,
            outputs=OutputSpec(smatrix_size="quarter", fields="all"),
            asr=ASROptions(circle_G=8.0e-2, grid=(64, 64)),
            group_theory=group,
            verify_cascade=False,
            dtype=torch.complex128,
            device=device,
        )
        simulation.add_input_layer(eps=1.0)
        simulation.add_output_layer(eps=1.0)
        simulation.set_incident_angle(0.0, 0.0)
        simulation.add_layer_circle_shell_asr(
            0.18,
            0.18,
            0.32,
            1.0,
            2.5,
            4.0,
            nx=64,
            ny=64,
            factorization_rules=False,
            radial_mapping="double",
        )
        simulation.solve_global_smatrix()
        zero_x = int(torch.nonzero(simulation.order_x == 0)[0])
        zero_y = int(torch.nonzero(simulation.order_y == 0)[0])
        source = torch.zeros(
            (2 * simulation.order_N, 1), dtype=torch.complex128, device=device
        )
        source[zero_x * len(simulation.order_y) + zero_y, 0] = 1.0
        simulation.E_i = source
        simulation.source_direction = "forward"
        fourier_fields = simulation._fourier_fields(0, 0.09)
        electric_xy, magnetic_xy = simulation.field_xy(
            0, [0.0, 0.2], [0.0, 0.2], z_prop=0.09
        )
        finite[name] = all(
            bool(torch.all(torch.isfinite(component)))
            for component in (*fourier_fields, *electric_xy, *magnetic_xy)
        )
    return {"finite": finite, "passed": all(finite.values())}
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=_VALIDATION_ROOT / "results" / "peng2025_validation.json",
    )
    args = parser.parse_args()
    device = select_device(args.device)
    geometry = PaperGeometry()
    drude = SilverDrude()
    geometry.validate()
    drude.validate()

    grid_x = 48
    grid_y = rectangular_supercell_grid_y(grid_x)
    epsilon_silver = drude.epsilon(3.0)
    material = rectangular_supercell_material(
        geometry,
        epsilon_silver,
        grid_x=grid_x,
        grid_y=grid_y,
        device=device,
    )
    translated = torch.roll(material, shifts=(grid_x // 2, grid_y // 2), dims=(0, 1))
    hidden_translation_error = float(torch.max(torch.abs(material - translated)).cpu())
    aperture_value = torch.as_tensor(
        geometry.epsilon_aperture, dtype=material.dtype, device=device
    )
    sampled_fill = float(
        torch.mean((torch.abs(material - aperture_value) == 0).double()).cpu()
    )
    expected_fill = annular_aperture_fill_fraction(geometry, "triangular")
    fill_error = abs(sampled_fill - expected_fill)

    base = Numerics(
        order_x=1,
        order_y=1,
        grid_x=grid_x,
        grid_y=grid_x,
        asr_g=1.0e-3,
        cascade="redheffer",
        use_symmetry=False,
    )
    square = simulate_matched_primitive(
        1.95,
        lattice_kind="square",
        geometry=geometry,
        drude=drude,
        numerics=base,
        device=device,
    )
    triangular = simulate_matched_primitive(
        3.0,
        lattice_kind="triangular",
        geometry=geometry,
        drude=drude,
        numerics=base,
        device=device,
    )
    double_base = Numerics(
        order_x=1,
        order_y=1,
        grid_x=64,
        grid_y=64,
        asr_g=1.0e-3,
        cascade="redheffer",
        use_symmetry=False,
        shell_radial_mapping="double",
    )
    square_double = simulate_matched_primitive(
        3.0,
        lattice_kind="square",
        geometry=geometry,
        drude=drude,
        numerics=double_base,
        device=device,
    )
    square_double_algo2a = simulate_matched_primitive(
        3.0,
        lattice_kind="square",
        geometry=geometry,
        drude=drude,
        numerics=Numerics(
            order_x=1,
            order_y=1,
            grid_x=64,
            grid_y=64,
            asr_g=1.0e-3,
            cascade="algo2a",
            use_symmetry=False,
            shell_radial_mapping="double",
        ),
        device=device,
    )
    triangular_double = simulate_matched_primitive(
        3.0,
        lattice_kind="triangular",
        geometry=geometry,
        drude=drude,
        numerics=double_base,
        device=device,
    )
    triangular_double_d6 = simulate_matched_primitive(
        3.0,
        lattice_kind="triangular",
        geometry=geometry,
        drude=drude,
        numerics=Numerics(
            order_x=4,
            order_y=4,
            grid_x=64,
            grid_y=64,
            asr_g=1.0e-3,
            cascade="redheffer",
            use_symmetry=True,
            shell_radial_mapping="double",
        ),
        device=device,
    )
    # The extremely small D6 star at N=1 is not a physically converged metal
    # calculation and can show a small negative absorptance.  N=4 is still a
    # quick smoke case but is large enough for the passivity check to be useful.
    d6_numerics = Numerics(
        order_x=4,
        order_y=4,
        grid_x=64,
        grid_y=64,
        asr_g=1.0e-3,
        cascade="redheffer",
        use_symmetry=True,
    )
    triangular_d6 = simulate_matched_primitive(
        3.0,
        lattice_kind="triangular",
        geometry=geometry,
        drude=drude,
        numerics=d6_numerics,
        device=device,
    )
    triangular_full_d6_reference = simulate_matched_primitive(
        3.0,
        lattice_kind="triangular",
        geometry=geometry,
        drude=drude,
        numerics=Numerics(
            order_x=4,
            order_y=4,
            grid_x=64,
            grid_y=64,
            asr_g=1.0e-3,
            cascade="redheffer",
            use_symmetry=False,
        ),
        device=device,
    )
    triangular_raster = simulate_triangular_raster_primitive(
        3.0,
        geometry=geometry,
        drude=drude,
        numerics=base,
        device=device,
    )
    supercell = simulate_rectangular_supercell(
        3.0,
        geometry=geometry,
        drude=drude,
        numerics=Numerics(
            order_x=1,
            order_y=2,
            grid_x=grid_x,
            grid_y=grid_y,
            cascade="redheffer",
            use_symmetry=False,
        ),
        device=device,
    )
    d6_difference = max(
        abs(float(triangular_full_d6_reference[name]) - float(triangular_d6[name]))
        for name in ("reflectance", "transmittance", "absorptance")
    )
    folded_power = float(supercell["folded_reflection_power"]) + float(
        supercell["folded_transmission_power"]
    )
    lattice_difference = max(
        abs(float(triangular_raster[name]) - float(supercell[name]))
        for name in ("reflectance", "transmittance", "absorptance")
    )
    double_cascade_difference = max(
        abs(float(square_double[name]) - float(square_double_algo2a[name]))
        for name in ("reflectance", "transmittance", "absorptance")
    )
    checks = {
        "geometry_and_material_defaults": {
            "passed": geometry.period_um == 62.0
            and geometry.outer_radius_um == 30.0
            and geometry.inner_radius_um == 14.0
            and geometry.silver_thickness_um == 1.0
            and geometry.epsilon_aperture == complex(1.0, 0.0)
            and geometry.epsilon_pi == complex(3.5, 0.009)
        },
        "drude_passive": {
            "epsilon_at_3_thz": [epsilon_silver.real, epsilon_silver.imag],
            "passed": epsilon_silver.imag > 0.0,
        },
        "supercell_hidden_translation": {
            "maximum_material_error": hidden_translation_error,
            "passed": hidden_translation_error == 0.0,
        },
        "air_annular_aperture_fill_fraction": {
            "sampled": sampled_fill,
            "analytic": expected_fill,
            "absolute_error": fill_error,
            "passed": fill_error < 2.0e-2,
        },
        "square_smoke": _check_power(square),
        "triangular_smoke": _check_power(triangular),
        "double_matched_geometry": _double_map_geometry(device),
        "double_matched_radius_gradient": _double_map_gradient(device),
        "double_matched_internal_fields": _double_map_fields(device),
        "square_double_matched_smoke": _check_power(square_double),
        "double_matched_redheffer_vs_algo2a": {
            "maximum_RTA_difference": double_cascade_difference,
            "passed": double_cascade_difference < 1.0e-10,
        },
        "triangular_double_matched_smoke": _check_power(triangular_double),
        "triangular_double_matched_d6_smoke": _check_power(
            triangular_double_d6
        ),
        "triangular_d6_smoke": _check_power(triangular_d6),
        "triangular_full_d6_reference": _check_power(
            triangular_full_d6_reference
        ),
        "triangular_raster_smoke": _check_power(triangular_raster),
        "d6_vs_rectangular_truncation_diagnostic": {
            "maximum_RTA_difference": d6_difference,
            "basis_note": (
                "At equal index N=4 the D6 path uses a closed native star, "
                "whereas the unreduced path uses a rectangular box. "
                "Exact parity is not expected before separate basis convergence."
            ),
            "passed": math.isfinite(d6_difference),
        },
        "supercell_smoke": _check_power(supercell),
        "raster_lattice_equivalence": {
            "maximum_RTA_difference": lattice_difference,
            "passed": lattice_difference < 5.0e-3,
        },
        "forbidden_folded_orders": {
            "R_plus_T": folded_power,
            "passed": folded_power < 1.0e-8,
        },
    }
    passed = all(bool(check["passed"]) for check in checks.values())
    report = {
        "passed": passed,
        "device": str(device),
        "checks": checks,
        "note": (
            "This is an implementation smoke/invariance test, not a proof of "
            "high-order convergence or agreement with unavailable paper samples."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
