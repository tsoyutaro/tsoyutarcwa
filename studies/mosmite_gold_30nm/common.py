"""Hybrid matched-ASR/raster model for a 30 nm Au-coated MOSMITE film.

The nominal conformal coating is the periodic union of disks with radius
``r_core(z) + t_Au``.  Where that union is still a set of isolated disks, a
monotone double-matched map resolves both PMMA/Au and Au/air interfaces.  Near
the base the 30 nm coating coalesces across primitive-cell boundaries; those
slices are represented by a periodic nearest-image raster instead.  This is a
topology change, not a failure that can be repaired by allowing an overlapping
circle in the matched map.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import torch

_OUTPUTS_ROOT = Path(__file__).resolve().parents[2]
if str(_OUTPUTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_OUTPUTS_ROOT))

from rcwa_solver_auto import ASROptions, AutoRCWA, Lattice, OutputSpec
from studies.shared.gold_dispersion import build_gold_model


@dataclass(frozen=True)
class GeometryConfig:
    period_nm: float = 200.0
    height_nm: float = 500.0
    tip_radius_nm: float = 5.0
    base_radius_nm: float = 75.0
    gold_thickness_nm: float = 30.0
    profile_power: float = 1.0
    pmma_index: float = 1.49
    include_top_cap: bool = True
    asr_circle_g: float = 0.03
    # Switch before the isolated outer circle becomes tangent to its neighbour.
    # A small positive guard avoids an extremely compressed exterior ASR strip.
    asr_min_gap_nm: float = 2.0


@dataclass(frozen=True)
class NumericalConfig:
    order: int = 8
    slices: int = 48
    grid: int = 384


def parse_int_list(text: str, name: str) -> tuple[int, ...]:
    values = tuple(sorted({int(item.strip()) for item in text.split(",")}))
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{name} must contain positive comma-separated integers.")
    return values


def parse_wavelengths(text: str) -> tuple[float, ...]:
    if ":" in text:
        fields = tuple(float(item) for item in text.split(":"))
        if len(fields) != 3:
            raise ValueError("wavelength range must be start:stop:step")
        start, stop, step = fields
        if start <= 0.0 or stop < start or step <= 0.0:
            raise ValueError("invalid wavelength range")
        count = int(math.floor((stop - start) / step + 1.0e-12))
        values = tuple(start + index * step for index in range(count + 1))
        if values[-1] < stop - 1.0e-9:
            values += (stop,)
    else:
        values = tuple(float(item.strip()) for item in text.split(","))
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("wavelengths must be finite and positive")
    return tuple(sorted(set(values)))


def validate_geometry(config: GeometryConfig) -> None:
    for name, value in asdict(config).items():
        if name == "include_top_cap":
            continue
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if min(
        config.period_nm,
        config.height_nm,
        config.tip_radius_nm,
        config.base_radius_nm,
        config.gold_thickness_nm,
        config.profile_power,
        config.pmma_index,
    ) <= 0.0:
        raise ValueError("lengths, profile_power and pmma_index must be positive")
    if config.tip_radius_nm > config.base_radius_nm:
        raise ValueError("tip_radius_nm must not exceed base_radius_nm")
    if 2.0 * config.base_radius_nm >= config.period_nm:
        raise ValueError("PMMA cores overlap; require 2*base_radius_nm < period_nm")
    if config.asr_min_gap_nm < 0.0:
        raise ValueError("asr_min_gap_nm must be nonnegative")
    if not 0.0 < config.asr_circle_g < 1.0:
        raise ValueError("asr_circle_g must lie in (0,1)")


def slice_core_radius_nm(config: GeometryConfig, index: int, slices: int) -> float:
    coordinate = (index + 0.5) / slices
    return config.tip_radius_nm + (
        config.base_radius_nm - config.tip_radius_nm
    ) * coordinate**config.profile_power


def _periodic_nearest_radius(
    grid: int, *, device: torch.device
) -> torch.Tensor:
    """Distance to the nearest triangular-lattice site, in period units."""
    axis = torch.arange(grid, dtype=torch.float64, device=device) / grid
    u, v = torch.meshgrid(axis, axis, indexing="ij")
    candidates = []
    for shift_u in (-1, 0, 1):
        for shift_v in (-1, 0, 1):
            q1 = u - 0.5 - shift_u
            q2 = v - 0.5 - shift_v
            candidates.append(q1**2 + q2**2 + q1 * q2)
    return torch.sqrt(torch.clamp(torch.min(torch.stack(candidates), dim=0).values, min=0.0))


def periodic_union_material_grid(
    grid: int,
    core_radius: float,
    outer_radius: float,
    epsilon_pmma: complex | float,
    epsilon_gold: complex,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Three-material periodic union used after neighbouring Au shells merge."""
    radius = _periodic_nearest_radius(grid, device=device)
    core = radius <= core_radius
    coated = radius <= outer_radius
    one = torch.ones((grid, grid), dtype=torch.complex128, device=device)
    eps = torch.where(
        core,
        one * complex(epsilon_pmma),
        torch.where(coated, one * complex(epsilon_gold), one),
    )
    count = float(grid * grid)
    diagnostics = {
        "pmma_fraction": float(torch.count_nonzero(core).detach().cpu()) / count,
        "gold_fraction": float(torch.count_nonzero(coated & ~core).detach().cpu()) / count,
        "air_fraction": float(torch.count_nonzero(~coated).detach().cpu()) / count,
    }
    return eps, diagnostics


def _zero_order_x_source(simulation: AutoRCWA) -> torch.Tensor:
    zero_x = int(torch.nonzero(simulation.order_x == 0, as_tuple=False)[0, 0])
    zero_y = int(torch.nonzero(simulation.order_y == 0, as_tuple=False)[0, 0])
    harmonic = zero_x * len(simulation.order_y) + zero_y
    source = torch.zeros(
        2 * simulation.order_N,
        dtype=simulation._dtype,
        device=simulation._device,
    )
    source[harmonic] = 1.0
    return source


def _mean_poynting_z(
    electric: torch.Tensor,
    electric_to_magnetic: torch.Tensor,
    *,
    direction: int,
) -> float:
    magnetic = direction * (electric_to_magnetic @ electric)
    count = electric.numel() // 2
    ex, ey = electric[:count], electric[count:]
    hx, hy = magnetic[:count], magnetic[count:]
    flux = 0.5 * torch.real(torch.sum(ex * torch.conj(hy) - ey * torch.conj(hx)))
    return float(flux.detach().cpu())


def simulate_case(
    wavelength_nm: float,
    numerical: NumericalConfig,
    geometry: GeometryConfig,
    gold_epsilon: Callable[[float], complex],
    *,
    device: torch.device,
    cascade: str = "redheffer",
) -> dict[str, object]:
    """Solve one wavelength with automatic isolated/connected topology handling."""
    started = time.perf_counter()
    validate_geometry(geometry)
    if min(numerical.order, numerical.slices, numerical.grid) <= 0:
        raise ValueError("order, slices and grid must be positive")

    period = geometry.period_nm
    eps_gold = gold_epsilon(wavelength_nm)
    eps_pmma = geometry.pmma_index**2
    simulation = AutoRCWA(
        freq=period / wavelength_nm,
        order=[numerical.order, numerical.order],
        lattice=Lattice.triangular(1.0),
        cascade=cascade,
        outputs=OutputSpec(smatrix_size="half", fields="none"),
        asr=ASROptions(
            circle_G=geometry.asr_circle_g,
            grid=(numerical.grid, numerical.grid),
            factorization_rules=True,
        ),
        # A source-reduced basis cannot currently be mixed with raster layers.
        group_theory=False,
        verify_cascade=False,
        dtype=torch.complex128,
        device=device,
    )
    simulation.add_input_layer(eps=1.0, mu=1.0)
    simulation.add_output_layer(eps=eps_pmma, mu=1.0)
    simulation.set_incident_angle(0.0, 0.0)

    if geometry.include_top_cap:
        simulation.add_layer_circle_asr(
            geometry.gold_thickness_nm / period,
            (geometry.tip_radius_nm + geometry.gold_thickness_nm) / period,
            1.0,
            eps_gold,
            nx=numerical.grid,
            ny=numerical.grid,
            factorization_rules=True,
        )

    thickness = geometry.height_nm / numerical.slices / period
    isolated_limit_nm = 0.5 * (period - geometry.asr_min_gap_nm)
    asr_layers = 0
    raster_layers = 0
    raster_diagnostics: list[dict[str, float]] = []
    minimum_asr_jacobian = math.inf
    minimum_asr_slope = math.inf
    for index in range(numerical.slices):
        core_nm = slice_core_radius_nm(geometry, index, numerical.slices)
        outer_nm = core_nm + geometry.gold_thickness_nm
        if outer_nm < isolated_limit_nm:
            simulation.add_layer_circle_shell_asr(
                thickness,
                core_nm / period,
                outer_nm / period,
                1.0,
                eps_gold,
                eps_pmma,
                nx=numerical.grid,
                ny=numerical.grid,
                factorization_rules=True,
                radial_mapping="double",
            )
            record = simulation.layer_records[-1]
            minimum_asr_jacobian = min(
                minimum_asr_jacobian,
                float(record.options["minimum_mapping_jacobian"]),
            )
            minimum_asr_slope = min(
                minimum_asr_slope,
                float(record.options["effective_radial_slope"]),
            )
            asr_layers += 1
        else:
            eps_grid, fractions = periodic_union_material_grid(
                numerical.grid,
                core_nm / period,
                outer_nm / period,
                eps_pmma,
                eps_gold,
                device=device,
            )
            simulation.add_layer(thickness, eps=eps_grid, mu=1.0)
            record = simulation.layer_records[-1]
            record.reason = "PERIODIC_AU_COALESCENCE"
            record.shape = "periodic-union-core-shell-raster"
            record.options.update(
                {
                    "core_radius_nm": core_nm,
                    "outer_radius_nm": outer_nm,
                    **fractions,
                }
            )
            raster_diagnostics.append(fractions)
            raster_layers += 1

    simulation.solve_global_smatrix()
    incident = _zero_order_x_source(simulation)
    reflected = simulation.S[1] @ incident
    transmitted = simulation.S[0] @ incident
    incident_flux = _mean_poynting_z(incident, simulation.Vi, direction=1)
    reflected_flux = _mean_poynting_z(reflected, simulation.Vi, direction=-1)
    transmitted_flux = _mean_poynting_z(transmitted, simulation.Vo, direction=1)
    if incident_flux <= 0.0:
        raise RuntimeError("incident power flux is not positive")
    reflectance = -reflected_flux / incident_flux
    transmittance = transmitted_flux / incident_flux
    absorptance = 1.0 - reflectance - transmittance
    return {
        "wavelength_nm": wavelength_nm,
        "order": numerical.order,
        "slices": numerical.slices,
        "grid": numerical.grid,
        "reflectance": reflectance,
        "transmittance": transmittance,
        "absorptance": absorptance,
        "asr_layers": asr_layers,
        "coalesced_raster_layers": raster_layers,
        "top_cap_layers": int(geometry.include_top_cap),
        "minimum_asr_jacobian": None if math.isinf(minimum_asr_jacobian) else minimum_asr_jacobian,
        "minimum_asr_effective_slope": None if math.isinf(minimum_asr_slope) else minimum_asr_slope,
        "minimum_raster_air_fraction": (
            min(item["air_fraction"] for item in raster_diagnostics)
            if raster_diagnostics
            else None
        ),
        "passivity_warning": bool(
            reflectance < -1.0e-6
            or transmittance < -1.0e-6
            or absorptance < -1.0e-5
            or reflectance + transmittance > 1.0 + 1.0e-5
        ),
        "runtime_seconds": time.perf_counter() - started,
    }


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError("cannot write an empty result table")
    columns = tuple(materialized[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def add_arguments(parser) -> None:
    parser.add_argument("--period-nm", type=float, default=200.0)
    parser.add_argument("--height-nm", type=float, default=500.0)
    parser.add_argument("--tip-radius-nm", type=float, default=5.0)
    parser.add_argument("--base-radius-nm", type=float, default=75.0)
    parser.add_argument("--gold-thickness-nm", type=float, default=30.0)
    parser.add_argument("--profile-power", type=float, default=1.0)
    parser.add_argument("--pmma-index", type=float, default=1.49)
    parser.add_argument("--asr-circle-g", type=float, default=0.03)
    parser.add_argument("--asr-min-gap-nm", type=float, default=2.0)
    parser.add_argument("--no-top-cap", action="store_true")
    parser.add_argument("--gold-model", choices=("rakic-ld", "csv"), default="rakic-ld")
    parser.add_argument("--gold-csv", type=Path)
    parser.add_argument("--cascade", choices=("redheffer", "algo2a"), default="redheffer")
    parser.add_argument("--device", default="auto")


def configs_from_args(args) -> tuple[GeometryConfig, Callable[[float], complex], torch.device]:
    geometry = GeometryConfig(
        period_nm=args.period_nm,
        height_nm=args.height_nm,
        tip_radius_nm=args.tip_radius_nm,
        base_radius_nm=args.base_radius_nm,
        gold_thickness_nm=args.gold_thickness_nm,
        profile_power=args.profile_power,
        pmma_index=args.pmma_index,
        include_top_cap=not args.no_top_cap,
        asr_circle_g=args.asr_circle_g,
        asr_min_gap_nm=args.asr_min_gap_nm,
    )
    validate_geometry(geometry)
    gold = build_gold_model(args.gold_model, args.gold_csv)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    return geometry, gold, device

