"""Shared utilities for the Peng--Zhang 2025 coaxial-array studies.

The MI unit cell is modeled as a silver background, an air annular aperture
``r < rho < R``, and a central silver particle ``rho < r``.  The patterned
layer is placed on polyimide.  A semi-infinite PI output medium remains the
default because the article does not give the Fig. 2 PI thickness; a finite PI
layer followed by air can be selected explicitly.  Lengths are normalized by
the nearest-neighbour period before they enter torcwa.

The article does not state the numerical Drude constants for Ag or the finite
MI substrate thickness.  This module therefore uses a documented, overridable
Drude starting point and makes the output-stack assumption explicit.  These
assumptions must not be confused with digitized reference data from the paper.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

# Make direct execution of scripts below ``outputs/paper_reproductions/peng2025`` independent of
# the caller's working directory.  The reusable solver facade remains in the
# parent ``outputs`` directory.
_OUTPUTS_ROOT = Path(__file__).resolve().parents[2]
if str(_OUTPUTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_OUTPUTS_ROOT))

from rcwa_solver_auto import (
    ASROptions,
    AutoRCWA,
    GroupTheoryOptions,
    Lattice,
    OutputSpec,
)


PAPER_TITLE = (
    "Rigorous Coupled-Wave Analysis of Multilayer Metal-Insulator "
    "Aperture-Particle Composite Periodic Arrays"
)
PAPER_DOI = "10.1109/LAWP.2025.3543371"
C_UM_PER_PS = 299.792458


@dataclass(frozen=True)
class PaperGeometry:
    """Fig. 2 MI dimensions and the selected air-aperture interpretation."""

    period_um: float = 62.0
    outer_radius_um: float = 30.0
    inner_radius_um: float = 14.0
    silver_thickness_um: float = 1.0
    epsilon_aperture_real: float = 1.0
    epsilon_aperture_imag: float = 0.0
    epsilon_pi_real: float = 3.5
    epsilon_pi_imag: float = 0.009
    pi_thickness_um: float | None = None

    @property
    def epsilon_aperture(self) -> complex:
        """Relative permittivity of the air annulus in the Ag patterned layer."""

        return complex(self.epsilon_aperture_real, self.epsilon_aperture_imag)

    @property
    def epsilon_pi(self) -> complex:
        return complex(self.epsilon_pi_real, self.epsilon_pi_imag)

    def validate(self) -> None:
        positive = {
            "period_um": self.period_um,
            "outer_radius_um": self.outer_radius_um,
            "inner_radius_um": self.inner_radius_um,
            "silver_thickness_um": self.silver_thickness_um,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.pi_thickness_um is not None and (
            not math.isfinite(self.pi_thickness_um) or self.pi_thickness_um <= 0.0
        ):
            raise ValueError("pi_thickness_um must be finite and positive when set.")
        if self.inner_radius_um >= self.outer_radius_um:
            raise ValueError("inner_radius_um must be smaller than outer_radius_um.")
        if 2.0 * self.outer_radius_um >= self.period_um:
            raise ValueError(
                "The matched map requires a non-touching outer circle: 2R < p."
            )
        if self.epsilon_pi_imag < 0.0:
            raise ValueError("Passive exp(-i omega t) PI requires Im(epsilon) >= 0.")
        if self.epsilon_aperture_imag < 0.0:
            raise ValueError(
                "Passive exp(-i omega t) aperture requires Im(epsilon) >= 0."
            )
        if abs(self.epsilon_aperture) == 0.0:
            raise ValueError("epsilon_aperture must be nonzero.")
        material_components = {
            "epsilon_aperture_real": self.epsilon_aperture_real,
            "epsilon_aperture_imag": self.epsilon_aperture_imag,
            "epsilon_pi_real": self.epsilon_pi_real,
            "epsilon_pi_imag": self.epsilon_pi_imag,
        }
        if not all(math.isfinite(value) for value in material_components.values()):
            raise ValueError("Aperture and PI permittivities must be finite.")
        if abs(self.epsilon_pi) == 0.0:
            raise ValueError("epsilon_pi must be nonzero.")


@dataclass(frozen=True)
class SilverDrude:
    """Overridable Ag Drude parameters for the solver's exp(-i omega t) sign.

    The defaults ``omega_p=1.37e16 rad/s`` and ``gamma=2.73e13 rad/s`` are a
    common THz starting point.  Peng and Zhang only say "the Drude model" and
    do not list their constants, so these defaults are an explicit reproduction
    assumption rather than a value recovered from the article.
    """

    epsilon_infinity: float = 1.0
    plasma_rad_s: float = 1.37e16
    collision_rad_s: float = 2.73e13

    def validate(self) -> None:
        values = {
            "epsilon_infinity": self.epsilon_infinity,
            "plasma_rad_s": self.plasma_rad_s,
            "collision_rad_s": self.collision_rad_s,
        }
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("All Drude parameters must be finite.")
        if self.plasma_rad_s <= 0.0 or self.collision_rad_s <= 0.0:
            raise ValueError("Drude plasma and collision frequencies must be positive.")

    def epsilon(self, frequency_thz: float) -> complex:
        self.validate()
        frequency = float(frequency_thz)
        if not math.isfinite(frequency) or frequency <= 0.0:
            raise ValueError("frequency_thz must be finite and positive.")
        omega = 2.0 * math.pi * frequency * 1.0e12
        value = self.epsilon_infinity - self.plasma_rad_s**2 / (
            omega * (omega + 1.0j * self.collision_rad_s)
        )
        if value.imag <= 0.0:
            raise RuntimeError("The passive Ag Drude model produced nonpositive loss.")
        return complex(value)


@dataclass(frozen=True)
class Numerics:
    order_x: int
    order_y: int
    grid_x: int
    grid_y: int
    asr_g: float = 1.0e-3
    cascade: str = "redheffer"
    use_symmetry: bool = False
    shell_radial_mapping: str = "outer"
    solver: str = "matched-asr"

    def validate(self) -> None:
        if min(self.order_x, self.order_y) < 1:
            raise ValueError("Both Fourier truncation orders must be at least one.")
        if min(self.grid_x, self.grid_y) < 16:
            raise ValueError("Both material grids must be at least 16.")
        if not 0.0 < self.asr_g < 1.0:
            raise ValueError("asr_g must lie in (0, 1).")
        if self.cascade not in {"redheffer", "algo2a", "li-2a"}:
            raise ValueError("cascade must be redheffer or algo2a/li-2a.")
        if self.shell_radial_mapping not in {"outer", "double"}:
            raise ValueError("shell_radial_mapping must be 'outer' or 'double'.")
        if self.solver not in {"matched-asr", "nvm"}:
            raise ValueError("solver must be 'matched-asr' or 'nvm'.")


def normalized_frequency(frequency_thz: float, period_um: float) -> float:
    """Return torcwa's dimensionless ``period / vacuum wavelength``."""

    return period_um * frequency_thz / C_UM_PER_PS


def parse_float_list(text: str) -> tuple[float, ...]:
    """Parse ``a,b,c`` or inclusive ``start:stop:step`` values."""

    stripped = text.strip()
    if ":" in stripped:
        parts = tuple(float(item) for item in stripped.split(":"))
        if len(parts) != 3:
            raise ValueError("A range must have start:stop:step.")
        start, stop, step = parts
        if step <= 0.0 or stop < start:
            raise ValueError("Invalid inclusive range.")
        count = int(math.floor((stop - start) / step + 1.0e-10))
        values = tuple(start + index * step for index in range(count + 1))
        if values[-1] < stop - 1.0e-10:
            values += (stop,)
    else:
        values = tuple(float(item.strip()) for item in stripped.split(","))
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("Values must be finite and positive.")
    return values


def parse_int_list(text: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in text.split(","))
    if not values or any(value < 1 for value in values):
        raise ValueError("Orders must be positive comma-separated integers.")
    return values


def select_device(name: str) -> torch.device:
    normalized = name.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if normalized not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda.")
    return torch.device(normalized)


def _zero_order_x_source(simulation: AutoRCWA) -> torch.Tensor:
    zero_x = int(torch.nonzero(simulation.order_x == 0, as_tuple=False)[0, 0])
    zero_y = int(torch.nonzero(simulation.order_y == 0, as_tuple=False)[0, 0])
    harmonic = zero_x * len(simulation.order_y) + zero_y
    source = torch.zeros(
        2 * simulation.order_N,
        dtype=simulation._dtype,
        device=simulation._device,
    )
    # At normal incidence and azimuth zero, this is Cartesian x / torcwa p.
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


def _power_observables(simulation: AutoRCWA) -> dict[str, float]:
    source = _zero_order_x_source(simulation)
    reflected = simulation.S[1] @ source
    transmitted = simulation.S[0] @ source
    incident_flux = _mean_poynting_z(source, simulation.Vi, direction=1)
    reflected_flux = _mean_poynting_z(reflected, simulation.Vi, direction=-1)
    transmitted_flux = _mean_poynting_z(transmitted, simulation.Vo, direction=1)
    if incident_flux <= 0.0:
        raise RuntimeError("Incident power flux is not positive.")
    reflectance = -reflected_flux / incident_flux
    transmittance = transmitted_flux / incident_flux
    absorptance = 1.0 - reflectance - transmittance
    return {
        "reflectance": reflectance,
        "transmittance": transmittance,
        "absorptance": absorptance,
        "passivity_warning": bool(
            reflectance < -1.0e-6
            or transmittance < -1.0e-6
            or absorptance < -1.0e-5
            or reflectance + transmittance > 1.0 + 1.0e-5
        ),
    }


def _selected_order_power(
    simulation: AutoRCWA,
    selected: torch.Tensor,
    *,
    port: str,
) -> float:
    if selected.numel() == 0:
        return 0.0
    co = simulation.S_parameters(
        selected,
        direction="forward",
        port=port,
        polarization="pp",
        power_norm=True,
    )
    cross = simulation.S_parameters(
        selected,
        direction="forward",
        port=port,
        polarization="sp",
        power_norm=True,
    )
    return float(torch.sum(torch.abs(co) ** 2 + torch.abs(cross) ** 2).real.cpu())


def _base_result(
    simulation: AutoRCWA,
    frequency_thz: float,
    epsilon_silver: complex,
    numerics: Numerics,
    geometry: PaperGeometry,
    started: float,
) -> dict[str, object]:
    result: dict[str, object] = {
        "frequency_thz": float(frequency_thz),
        "wavelength_um": C_UM_PER_PS / float(frequency_thz),
        "epsilon_silver_real": epsilon_silver.real,
        "epsilon_silver_imag": epsilon_silver.imag,
        "epsilon_pi_real": geometry.epsilon_pi.real,
        "epsilon_pi_imag": geometry.epsilon_pi.imag,
        "pi_thickness_um": geometry.pi_thickness_um,
        "output_medium": "PI" if geometry.pi_thickness_um is None else "air",
        "epsilon_aperture_real": geometry.epsilon_aperture.real,
        "epsilon_aperture_imag": geometry.epsilon_aperture.imag,
        "order_x": numerics.order_x,
        "order_y": numerics.order_y,
        "harmonics": simulation.order_N,
        "full_modal_dimension": 2 * simulation.order_N,
        "grid_x": numerics.grid_x,
        "grid_y": numerics.grid_y,
        "cascade": numerics.cascade,
        "solver": numerics.solver,
        "symmetry_requested": numerics.use_symmetry,
        "shell_radial_mapping": numerics.shell_radial_mapping,
        "runtime_seconds": time.perf_counter() - started,
    }
    result.update(_power_observables(simulation))
    diagnostics = simulation.cascade_diagnostics
    result["reduced_dimension"] = diagnostics.get("reduced_dimension")
    if simulation.group_theory_diagnostics:
        group = simulation.group_theory_diagnostics[-1]
        result["symmetry_applied"] = group.get("applied")
        result["symmetry"] = group.get("symmetry")
        result["irrep"] = group.get("irrep")
    else:
        result["symmetry_applied"] = False
        result["symmetry"] = None
        result["irrep"] = None
    return result


def simulate_matched_primitive(
    frequency_thz: float,
    *,
    lattice_kind: str,
    geometry: PaperGeometry,
    drude: SilverDrude,
    numerics: Numerics,
    device: torch.device,
) -> dict[str, object]:
    """Simulate one centered MI cell with matched-ASR or analytic NVM."""

    geometry.validate()
    numerics.validate()
    normalized = lattice_kind.strip().lower()
    if normalized == "square":
        lattice = Lattice.square(1.0)
        symmetry = "auto"
    elif normalized in {"triangular", "hexagonal", "hex"}:
        normalized = "triangular"
        lattice = Lattice.triangular(1.0)
        symmetry = "d6"
    else:
        raise ValueError("lattice_kind must be square or triangular.")

    started = time.perf_counter()
    epsilon_silver = drude.epsilon(frequency_thz)
    simulation = AutoRCWA(
        freq=normalized_frequency(frequency_thz, geometry.period_um),
        order=[numerics.order_x, numerics.order_y],
        lattice=lattice,
        cascade=numerics.cascade,
        outputs=OutputSpec(smatrix_size="half", fields="none"),
        asr=ASROptions(
            circle_G=numerics.asr_g,
            grid=(numerics.grid_x, numerics.grid_y),
            factorization_rules=True,
        ),
        group_theory=GroupTheoryOptions(
            enabled=numerics.use_symmetry,
            symmetry=symmetry,
            strict=numerics.use_symmetry,
            polarization="x" if numerics.use_symmetry else None,
        ),
        verify_cascade=False,
        dtype=torch.complex128,
        device=device,
    )
    simulation.add_input_layer(eps=1.0, mu=1.0)
    simulation.add_output_layer(
        eps=geometry.epsilon_pi if geometry.pi_thickness_um is None else 1.0,
        mu=1.0,
    )
    simulation.set_incident_angle(0.0, 0.0)
    if normalized == "square":
        centers = ((0.5, 0.5),)
    else:
        centers = ((0.75, 0.25 * math.sqrt(3.0)),)
    if numerics.solver == "nvm":
        simulation.add_layer_circle_shell_nvm(
            geometry.silver_thickness_um / geometry.period_um,
            geometry.inner_radius_um / geometry.period_um,
            geometry.outer_radius_um / geometry.period_um,
            epsilon_silver,
            geometry.epsilon_aperture,
            epsilon_silver,
            centers=centers,
            nx=numerics.grid_x,
            ny=numerics.grid_y,
        )
    else:
        simulation.add_layer_circle_shell_asr(
            geometry.silver_thickness_um / geometry.period_um,
            geometry.inner_radius_um / geometry.period_um,
            geometry.outer_radius_um / geometry.period_um,
            epsilon_silver,
            geometry.epsilon_aperture,
            epsilon_silver,
            nx=numerics.grid_x,
            ny=numerics.grid_y,
            factorization_rules=True,
            radial_mapping=numerics.shell_radial_mapping,
        )
    if geometry.pi_thickness_um is not None:
        simulation.add_layer(
            geometry.pi_thickness_um / geometry.period_um,
            eps=geometry.epsilon_pi,
            mu=1.0,
        )
    simulation.solve_global_smatrix()
    result = _base_result(
        simulation,
        frequency_thz,
        epsilon_silver,
        numerics,
        geometry,
        started,
    )
    result.update(
        {
            "model": (
                "analytic-nvm-primitive"
                if numerics.solver == "nvm"
                else "matched-asr-primitive"
            ),
            "lattice": normalized,
            "cell_length_x_um": geometry.period_um,
            "cell_length_y_um": geometry.period_um,
            "sites_per_cell": 1,
            "factorization": (
                "analytic-concentric-NVM"
                if numerics.solver == "nvm"
                else "double-matched-ASR-generalized-Li"
                if numerics.shell_radial_mapping == "double"
                else "outer-matched-ASR-FR"
            ),
        }
    )
    return result


def rectangular_supercell_grid_y(grid_x: int) -> int:
    """Choose an even y grid with approximately square physical pixels."""

    if grid_x < 16:
        raise ValueError("grid_x must be at least 16.")
    # A multiple of four makes the hidden half-cell translation an exact roll.
    return max(16, 4 * int(round(math.sqrt(3.0) * grid_x / 4.0)))


def rectangular_supercell_material(
    geometry: PaperGeometry,
    epsilon_silver: complex,
    *,
    grid_x: int,
    grid_y: int,
    device: torch.device,
) -> torch.Tensor:
    """Rasterize the exact two-site orthogonal supercell of a triangular lattice.

    The normalized cell is ``1 x sqrt(3)``.  Its two sites are related by the
    hidden primitive translation ``(1/2, sqrt(3)/2)``.  Replicating this cell
    therefore gives exactly the same infinite lattice as the 60-degree
    triangular primitive cell; it is not a square approximation.
    """

    geometry.validate()
    if grid_x < 16 or grid_y < 16:
        raise ValueError("Supercell grids must be at least 16.")
    dtype = torch.float64
    height = math.sqrt(3.0)
    x = (torch.arange(grid_x, dtype=dtype, device=device) + 0.5) / grid_x
    y = (torch.arange(grid_y, dtype=dtype, device=device) + 0.5) * height / grid_y
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    centers = (
        (0.25, 0.25 * height),
        (0.75, 0.75 * height),
    )
    distances: list[torch.Tensor] = []
    for center_x, center_y in centers:
        for shift_x in (-1, 0, 1):
            for shift_y in (-1, 0, 1):
                distances.append(
                    (xx - center_x - shift_x) ** 2
                    + (yy - center_y - shift_y * height) ** 2
                )
    distance_squared = torch.min(torch.stack(distances), dim=0).values
    outer = geometry.outer_radius_um / geometry.period_um
    inner = geometry.inner_radius_um / geometry.period_um
    epsilon_ag = torch.as_tensor(
        epsilon_silver, dtype=torch.complex128, device=device
    )
    epsilon_aperture = torch.as_tensor(
        geometry.epsilon_aperture, dtype=torch.complex128, device=device
    )
    material = torch.full(
        (grid_x, grid_y), epsilon_ag, dtype=torch.complex128, device=device
    )
    # Open an air annulus in the Ag film, then restore its central Ag particle.
    material = torch.where(
        distance_squared <= outer**2, epsilon_aperture, material
    )
    material = torch.where(distance_squared <= inner**2, epsilon_ag, material)
    return material


def triangular_primitive_material(
    geometry: PaperGeometry,
    epsilon_silver: complex,
    *,
    grid_x: int,
    grid_y: int,
    device: torch.device,
) -> torch.Tensor:
    """Rasterize one coaxial site in the 60-degree primitive cell."""

    geometry.validate()
    if grid_x < 16 or grid_y < 16:
        raise ValueError("Primitive grids must be at least 16.")
    dtype = torch.float64
    u = (torch.arange(grid_x, dtype=dtype, device=device) + 0.5) / grid_x
    v = (torch.arange(grid_y, dtype=dtype, device=device) + 0.5) / grid_y
    uu, vv = torch.meshgrid(u, v, indexing="ij")
    cosine, sine = 0.5, 0.5 * math.sqrt(3.0)
    xx = uu + cosine * vv
    yy = sine * vv
    center_x = 0.5 * (1.0 + cosine)
    center_y = 0.5 * sine
    distances = []
    for shift_u in (-1, 0, 1):
        for shift_v in (-1, 0, 1):
            image_x = center_x + shift_u + cosine * shift_v
            image_y = center_y + sine * shift_v
            distances.append((xx - image_x) ** 2 + (yy - image_y) ** 2)
    distance_squared = torch.min(torch.stack(distances), dim=0).values
    outer = geometry.outer_radius_um / geometry.period_um
    inner = geometry.inner_radius_um / geometry.period_um
    epsilon_ag = torch.as_tensor(
        epsilon_silver, dtype=torch.complex128, device=device
    )
    epsilon_aperture = torch.as_tensor(
        geometry.epsilon_aperture, dtype=torch.complex128, device=device
    )
    material = torch.full(
        (grid_x, grid_y), epsilon_ag, dtype=torch.complex128, device=device
    )
    # Same three-region pattern as the matched primitive path.
    material = torch.where(
        distance_squared <= outer**2, epsilon_aperture, material
    )
    material = torch.where(distance_squared <= inner**2, epsilon_ag, material)
    return material


def _simulate_standard_raster(
    frequency_thz: float,
    *,
    lattice: Lattice,
    material: torch.Tensor,
    model: str,
    sites_per_cell: int,
    geometry: PaperGeometry,
    drude: SilverDrude,
    numerics: Numerics,
    device: torch.device,
    started: float,
) -> tuple[AutoRCWA, dict[str, object]]:
    epsilon_silver = drude.epsilon(frequency_thz)
    simulation = AutoRCWA(
        freq=normalized_frequency(frequency_thz, geometry.period_um),
        order=[numerics.order_x, numerics.order_y],
        lattice=lattice,
        cascade=numerics.cascade,
        outputs=OutputSpec(smatrix_size="half", fields="none"),
        group_theory=GroupTheoryOptions(enabled=False),
        verify_cascade=False,
        dtype=torch.complex128,
        device=device,
    )
    simulation.add_input_layer(eps=1.0, mu=1.0)
    simulation.add_output_layer(
        eps=geometry.epsilon_pi if geometry.pi_thickness_um is None else 1.0,
        mu=1.0,
    )
    simulation.set_incident_angle(0.0, 0.0)
    simulation.add_layer(
        geometry.silver_thickness_um / geometry.period_um,
        eps=material,
        mu=1.0,
    )
    if geometry.pi_thickness_um is not None:
        simulation.add_layer(
            geometry.pi_thickness_um / geometry.period_um,
            eps=geometry.epsilon_pi,
            mu=1.0,
        )
    simulation.solve_global_smatrix()
    result = _base_result(
        simulation,
        frequency_thz,
        epsilon_silver,
        numerics,
        geometry,
        started,
    )
    result.update(
        {
            "model": model,
            "solver": "standard-raster",
            "sites_per_cell": sites_per_cell,
            "factorization": "standard-raster-Laurent",
        }
    )
    return simulation, result


def simulate_triangular_raster_primitive(
    frequency_thz: float,
    *,
    geometry: PaperGeometry,
    drude: SilverDrude,
    numerics: Numerics,
    device: torch.device,
) -> dict[str, object]:
    """Standard-raster control calculation in the triangular primitive cell."""

    geometry.validate()
    numerics.validate()
    if numerics.use_symmetry:
        raise ValueError("The standard-raster control requires use_symmetry=False.")
    started = time.perf_counter()
    epsilon_silver = drude.epsilon(frequency_thz)
    material = triangular_primitive_material(
        geometry,
        epsilon_silver,
        grid_x=numerics.grid_x,
        grid_y=numerics.grid_y,
        device=device,
    )
    _simulation, result = _simulate_standard_raster(
        frequency_thz,
        lattice=Lattice.triangular(1.0),
        material=material,
        model="triangular-raster-primitive",
        sites_per_cell=1,
        geometry=geometry,
        drude=drude,
        numerics=numerics,
        device=device,
        started=started,
    )
    result.update(
        {
            "lattice": "triangular",
            "cell_length_x_um": geometry.period_um,
            "cell_length_y_um": geometry.period_um,
        }
    )
    return result


def simulate_rectangular_supercell(
    frequency_thz: float,
    *,
    geometry: PaperGeometry,
    drude: SilverDrude,
    numerics: Numerics,
    device: torch.device,
) -> dict[str, object]:
    """Simulate the triangular array in its exact orthogonal two-site cell."""

    geometry.validate()
    numerics.validate()
    if numerics.use_symmetry:
        raise ValueError(
            "The raster supercell path has no source-sector reduction; "
            "set use_symmetry=False."
        )
    started = time.perf_counter()
    epsilon_silver = drude.epsilon(frequency_thz)
    material = rectangular_supercell_material(
        geometry,
        epsilon_silver,
        grid_x=numerics.grid_x,
        grid_y=numerics.grid_y,
        device=device,
    )
    simulation, result = _simulate_standard_raster(
        frequency_thz,
        lattice=Lattice.rectangular(1.0, math.sqrt(3.0)),
        material=material,
        model="rectangular-two-site-supercell",
        sites_per_cell=2,
        geometry=geometry,
        drude=drude,
        numerics=numerics,
        device=device,
        started=started,
    )
    orders = torch.cartesian_prod(simulation.order_x, simulation.order_y)
    parity = torch.remainder(orders[:, 0] + orders[:, 1], 2)
    forbidden = orders[parity != 0]
    result.update(
        {
            "lattice": "rectangular-supercell-of-triangular",
            "cell_length_x_um": geometry.period_um,
            "cell_length_y_um": math.sqrt(3.0) * geometry.period_um,
            "folded_reflection_power": _selected_order_power(
                simulation, forbidden, port="reflection"
            ),
            "folded_transmission_power": _selected_order_power(
                simulation, forbidden, port="transmission"
            ),
        }
    )
    return result


def annular_aperture_fill_fraction(
    geometry: PaperGeometry, lattice: str
) -> float:
    """Analytic air-annulus area fraction per primitive cell."""

    area = math.pi * (
        geometry.outer_radius_um**2 - geometry.inner_radius_um**2
    )
    if lattice == "square":
        cell = geometry.period_um**2
    elif lattice in {"triangular", "rectangular-supercell"}:
        cell = 0.5 * math.sqrt(3.0) * geometry.period_um**2
    else:
        raise ValueError("Unknown lattice.")
    return area / cell


# Compatibility alias for result readers written before the material naming
# was corrected from a PI annulus to an air annulus.
annulus_fill_fraction = annular_aperture_fill_fraction


def write_rows(rows: Iterable[dict[str, object]], path: Path) -> None:
    values = list(rows)
    if not values:
        raise ValueError("Cannot write an empty result table.")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in values:
        for key in row:
            if key not in columns:
                columns.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(values)
    temporary.replace(path)


def write_metadata(
    path: Path,
    *,
    geometry: PaperGeometry,
    drude: SilverDrude,
    payload: dict[str, object],
) -> None:
    content = {
        "paper": {"title": PAPER_TITLE, "doi": PAPER_DOI},
        "geometry_and_material_model": asdict(geometry),
        "reproduction_assumptions": {
            "silver_drude": asdict(drude),
            "pattern_layer": "Ag background / air annulus / Ag central particle",
            "mi_substrate": (
                "semi-infinite PI"
                if geometry.pi_thickness_um is None
                else f"finite PI layer ({geometry.pi_thickness_um:g} um) / air output"
            ),
            "time_convention": "exp(-i omega t), passive Im(epsilon)>=0",
            "method_note": (
                "The primitive-cell solver selects an analytic concentric NVM "
                "or the project's outer-only/C2 double-boundary matched-ASR "
                "implementation. Neither is a bit-for-bit copy of the paper's "
                "stepped separable ASR plus interpolated NV field; agreement is "
                "assessed through convergence of the physical observables."
            ),
        },
        **payload,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2), encoding="utf-8")


def maximum_spectrum_difference(
    left: Sequence[dict[str, object]],
    right: Sequence[dict[str, object]],
) -> dict[str, float]:
    if len(left) != len(right):
        raise ValueError("Spectra have different lengths.")
    metrics = ("reflectance", "transmittance", "absorptance")
    result = {metric: 0.0 for metric in metrics}
    for first, second in zip(left, right):
        if not math.isclose(
            float(first["frequency_thz"]),
            float(second["frequency_thz"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("Spectra use different frequency grids.")
        for metric in metrics:
            result[metric] = max(
                result[metric], abs(float(first[metric]) - float(second[metric]))
            )
    result["maximum"] = max(result.values())
    return result


def numpy_column(rows: Sequence[dict[str, object]], name: str) -> np.ndarray:
    return np.asarray([float(row[name]) for row in rows], dtype=float)
