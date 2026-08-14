"""Shared model and convergence utilities for Au-coated PMMA moth-eye films.

The PMMA relief is discretized along z.  Each patterned slice contains a PMMA
circle, a concentric Au shell, and air.  The matched-coordinate map follows the
outer Au/air boundary.  An optional Au disk above the PMMA tip approximates the
top part of a conformal coating.  The substrate is semi-infinite PMMA.

Lengths are normalized by the in-plane period before they enter torcwa.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import torch

from gold_dispersion import build_gold_model
from rcwa_solver_auto import ASROptions, AutoRCWA, GroupTheoryOptions, Lattice, OutputSpec


@dataclass(frozen=True)
class GeometryConfig:
    period_nm: float = 200.0
    height_nm: float = 500.0
    tip_radius_nm: float = 5.0
    base_radius_nm: float = 75.0
    gold_thickness_nm: float = 20.0
    profile_power: float = 1.0
    pmma_index: float = 1.49
    lattice: str = "triangular"
    include_top_cap: bool = True
    asr_circle_g: float = 0.03


@dataclass(frozen=True)
class NumericalConfig:
    order: int
    slices: int
    grid: int


def parse_int_list(text: str, name: str) -> tuple[int, ...]:
    values = tuple(sorted({int(value.strip()) for value in text.split(",")}))
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{name} must contain positive comma-separated integers.")
    return values


def parse_wavelengths(text: str) -> tuple[float, ...]:
    stripped = text.strip()
    if ":" in stripped:
        fields = tuple(float(value) for value in stripped.split(":"))
        if len(fields) != 3:
            raise ValueError("Wavelength range must be start:stop:step.")
        start, stop, step = fields
        if step <= 0.0 or stop < start:
            raise ValueError("Invalid wavelength range.")
        count = int(math.floor((stop - start) / step + 1.0e-10))
        values = tuple(start + index * step for index in range(count + 1))
        if values[-1] < stop - 1.0e-9:
            values += (stop,)
    else:
        values = tuple(float(value.strip()) for value in stripped.split(","))
    if not values or any(not 0.0 < value < float("inf") for value in values):
        raise ValueError("Wavelengths must be finite and positive.")
    return tuple(sorted(set(values)))


def lattice_for(config: GeometryConfig) -> Lattice:
    if config.lattice == "triangular":
        return Lattice.triangular(1.0)
    if config.lattice == "square":
        return Lattice.square(1.0)
    raise ValueError("lattice must be triangular or square.")


def slice_core_radius_nm(
    config: GeometryConfig, index: int, slices: int
) -> float:
    """Midpoint radius, ordered from the air-side tip to the substrate."""
    coordinate = (index + 0.5) / slices
    return config.tip_radius_nm + (
        config.base_radius_nm - config.tip_radius_nm
    ) * coordinate**config.profile_power


def validate_geometry(config: GeometryConfig) -> None:
    positive = {
        "period_nm": config.period_nm,
        "height_nm": config.height_nm,
        "tip_radius_nm": config.tip_radius_nm,
        "base_radius_nm": config.base_radius_nm,
        "gold_thickness_nm": config.gold_thickness_nm,
        "profile_power": config.profile_power,
        "pmma_index": config.pmma_index,
    }
    for name, value in positive.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
    if config.tip_radius_nm > config.base_radius_nm:
        raise ValueError("tip_radius_nm must not exceed base_radius_nm.")
    outer_base = config.base_radius_nm + config.gold_thickness_nm
    if 2.0 * outer_base >= config.period_nm:
        raise ValueError(
            "The coated base circles overlap: require "
            "2*(base_radius_nm + gold_thickness_nm) < period_nm."
        )
    if not 0.0 < config.asr_circle_g < 1.0:
        raise ValueError("asr_circle_g must lie in (0,1).")


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
    cascade: str,
    use_symmetry: bool,
    device: torch.device,
) -> dict[str, object]:
    """Evaluate R, T and A for one wavelength and discretization."""
    started = time.perf_counter()
    validate_geometry(geometry)
    if numerical.order <= 0 or numerical.slices <= 0 or numerical.grid <= 0:
        raise ValueError("order, slices and grid must be positive.")

    period = geometry.period_nm
    epsilon_gold = gold_epsilon(wavelength_nm)
    epsilon_pmma = geometry.pmma_index**2
    simulation = AutoRCWA(
        freq=period / wavelength_nm,
        order=[numerical.order, numerical.order],
        lattice=lattice_for(geometry),
        cascade=cascade,
        outputs=OutputSpec(smatrix_size="half", fields="none"),
        asr=ASROptions(
            circle_G=geometry.asr_circle_g,
            grid=(numerical.grid, numerical.grid),
            factorization_rules=True,
        ),
        group_theory=GroupTheoryOptions(
            enabled=use_symmetry,
            symmetry="auto",
            strict=use_symmetry,
            polarization="x" if use_symmetry else None,
        ),
        verify_cascade=False,
        dtype=torch.complex128,
        device=device,
    )
    simulation.add_input_layer(eps=1.0, mu=1.0)
    simulation.add_output_layer(eps=epsilon_pmma, mu=1.0)
    simulation.set_incident_angle(0.0, 0.0)

    # Above the original PMMA tip, a gold disk approximates the conformal cap.
    if geometry.include_top_cap:
        simulation.add_layer_circle_asr(
            geometry.gold_thickness_nm / period,
            (geometry.tip_radius_nm + geometry.gold_thickness_nm) / period,
            1.0,
            epsilon_gold,
            nx=numerical.grid,
            ny=numerical.grid,
            factorization_rules=True,
        )

    layer_thickness = geometry.height_nm / numerical.slices / period
    for layer in range(numerical.slices):
        core_radius = slice_core_radius_nm(geometry, layer, numerical.slices)
        outer_radius = core_radius + geometry.gold_thickness_nm
        simulation.add_layer_circle_shell_asr(
            layer_thickness,
            core_radius / period,
            outer_radius / period,
            1.0,
            epsilon_gold,
            epsilon_pmma,
            nx=numerical.grid,
            ny=numerical.grid,
            factorization_rules=True,
        )

    simulation.solve_global_smatrix()
    incident = _zero_order_x_source(simulation)
    reflected = simulation.S[1] @ incident
    transmitted = simulation.S[0] @ incident
    incident_flux = _mean_poynting_z(incident, simulation.Vi, direction=1)
    reflected_flux = _mean_poynting_z(reflected, simulation.Vi, direction=-1)
    transmitted_flux = _mean_poynting_z(transmitted, simulation.Vo, direction=1)
    if incident_flux <= 0.0:
        raise RuntimeError("Incident power flux is not positive.")

    reflectance = -reflected_flux / incident_flux
    transmittance = transmitted_flux / incident_flux
    absorptance = 1.0 - reflectance - transmittance
    diagnostics = simulation.cascade_diagnostics
    return {
        "wavelength_nm": wavelength_nm,
        "order": numerical.order,
        "profile_slices": numerical.slices,
        "total_pattern_layers": numerical.slices + int(geometry.include_top_cap),
        "grid": numerical.grid,
        "epsilon_gold_real": epsilon_gold.real,
        "epsilon_gold_imag": epsilon_gold.imag,
        "epsilon_pmma": epsilon_pmma,
        "reflectance": reflectance,
        "transmittance": transmittance,
        "absorptance": absorptance,
        "passivity_warning": bool(
            reflectance < -1.0e-6
            or transmittance < -1.0e-6
            or absorptance < -1.0e-5
            or reflectance + transmittance > 1.0 + 1.0e-5
        ),
        "reduced_dimension": diagnostics.get("reduced_dimension"),
        "full_dimension": diagnostics.get("full_dimension", 2 * simulation.order_N),
        "runtime_seconds": time.perf_counter() - started,
    }


def _finite_metrics(result: dict[str, object]) -> dict[str, float]:
    values = {
        name: float(result[name])
        for name in ("reflectance", "transmittance", "absorptance")
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise RuntimeError("A convergence observable is NaN or infinity.")
    return values


def adjacent_error(
    coarse: Sequence[dict[str, object]], fine: Sequence[dict[str, object]]
) -> tuple[float, dict[str, float]]:
    if len(coarse) != len(fine):
        raise ValueError("Convergence spectra have different lengths.")
    per_metric: dict[str, float] = {}
    for left, right in zip(coarse, fine):
        left_values, right_values = _finite_metrics(left), _finite_metrics(right)
        for name in left_values:
            difference = abs(left_values[name] - right_values[name])
            per_metric[name] = max(per_metric.get(name, 0.0), difference)
    return max(per_metric.values()), per_metric


def choose_candidate(
    candidates: Sequence[int],
    spectra: dict[int, Sequence[dict[str, object]]],
    tolerance: float,
) -> tuple[int, bool, list[dict[str, object]]]:
    comparisons: list[dict[str, object]] = []
    for coarse, fine in zip(candidates, candidates[1:]):
        maximum, per_metric = adjacent_error(spectra[coarse], spectra[fine])
        comparisons.append(
            {
                "coarse": coarse,
                "fine": fine,
                "max_abs_change": maximum,
                "per_metric": per_metric,
                "passed": maximum <= tolerance,
            }
        )
    # One accidental close pair is not accepted.  Two consecutive refinement
    # steps must pass, and the middle value is the smallest recommendation.
    for first, second in zip(comparisons, comparisons[1:]):
        if bool(first["passed"]) and bool(second["passed"]):
            return int(first["fine"]), True, comparisons
    return int(candidates[-1]), False, comparisons


class Study:
    def __init__(
        self,
        *,
        geometry: GeometryConfig,
        gold_epsilon: Callable[[float], complex],
        wavelengths: Sequence[float],
        cascade: str,
        use_symmetry: bool,
        device: torch.device,
        checkpoint: Path,
        signature: str,
    ) -> None:
        self.geometry = geometry
        self.gold_epsilon = gold_epsilon
        self.wavelengths = tuple(wavelengths)
        self.cascade = cascade
        self.use_symmetry = use_symmetry
        self.device = device
        self.checkpoint = checkpoint
        self.signature = signature
        self.cases: dict[str, dict[str, object]] = {}
        if checkpoint.exists():
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            if payload.get("signature") != signature:
                raise RuntimeError(
                    f"Checkpoint {checkpoint} belongs to different settings; "
                    "use a different --output-prefix or remove that checkpoint."
                )
            stored = payload.get("cases", {})
            if isinstance(stored, dict):
                self.cases = stored

    @staticmethod
    def _key(wavelength: float, numerical: NumericalConfig) -> str:
        return (
            f"wl={wavelength:.12g}|M={numerical.order}|"
            f"Nz={numerical.slices}|grid={numerical.grid}"
        )

    def _write_checkpoint(self) -> None:
        self.checkpoint.write_text(
            json.dumps(
                {"signature": self.signature, "cases": self.cases},
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

    def evaluate(
        self, wavelength: float, numerical: NumericalConfig
    ) -> dict[str, object]:
        key = self._key(wavelength, numerical)
        if key not in self.cases:
            print(
                f"solve: lambda={wavelength:g} nm, M={numerical.order}, "
                f"Nz={numerical.slices}, grid={numerical.grid}",
                flush=True,
            )
            self.cases[key] = simulate_case(
                wavelength,
                numerical,
                self.geometry,
                self.gold_epsilon,
                cascade=self.cascade,
                use_symmetry=self.use_symmetry,
                device=self.device,
            )
            self._write_checkpoint()
        return self.cases[key]

    def spectrum(self, numerical: NumericalConfig) -> list[dict[str, object]]:
        return [self.evaluate(wavelength, numerical) for wavelength in self.wavelengths]


def configuration_signature(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_csv(path: Path, cases: Iterable[dict[str, object]]) -> None:
    rows = list(cases)
    columns = (
        "wavelength_nm",
        "order",
        "profile_slices",
        "total_pattern_layers",
        "grid",
        "epsilon_gold_real",
        "epsilon_gold_imag",
        "epsilon_pmma",
        "reflectance",
        "transmittance",
        "absorptance",
        "passivity_warning",
        "reduced_dimension",
        "full_dimension",
        "runtime_seconds",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_geometry(args) -> GeometryConfig:
    geometry = GeometryConfig(
        period_nm=args.period_nm,
        height_nm=args.height_nm,
        tip_radius_nm=args.tip_radius_nm,
        base_radius_nm=args.base_radius_nm,
        gold_thickness_nm=args.gold_thickness_nm,
        profile_power=args.profile_power,
        pmma_index=args.pmma_index,
        lattice=args.lattice,
        include_top_cap=not args.no_top_cap,
        asr_circle_g=args.asr_circle_g,
    )
    validate_geometry(geometry)
    return geometry


def add_shared_arguments(parser) -> None:
    parser.add_argument("--period-nm", type=float, default=200.0)
    parser.add_argument("--height-nm", type=float, default=500.0)
    parser.add_argument("--tip-radius-nm", type=float, default=5.0)
    parser.add_argument("--base-radius-nm", type=float, default=75.0)
    parser.add_argument("--gold-thickness-nm", type=float, default=20.0)
    parser.add_argument("--profile-power", type=float, default=1.0)
    parser.add_argument("--pmma-index", type=float, default=1.49)
    parser.add_argument("--lattice", choices=("triangular", "square"), default="triangular")
    parser.add_argument("--no-top-cap", action="store_true")
    parser.add_argument("--asr-circle-g", type=float, default=0.03)
    parser.add_argument("--gold-model", choices=("rakic-ld", "csv"), default="rakic-ld")
    parser.add_argument("--gold-csv", type=Path)
    parser.add_argument("--wavelengths", default="400,550,700")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=5.0e-3,
        help="Maximum absolute change in R, T or A (0.005 = 0.5 percentage point).",
    )
    parser.add_argument("--cascade", choices=("redheffer", "algo2a"), default="redheffer")
    parser.add_argument("--no-symmetry", action="store_true")
    parser.add_argument("--device", default="auto")


def run_axis_convergence(
    *,
    axis: str,
    candidates: Sequence[int],
    fixed: NumericalConfig,
    args,
) -> tuple[dict[str, object], Path]:
    if axis not in {"order", "slices"}:
        raise ValueError("axis must be order or slices.")
    if len(candidates) < 3:
        raise ValueError("At least three candidates are required.")
    if not math.isfinite(args.tolerance) or args.tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive.")

    geometry = make_geometry(args)
    wavelengths = parse_wavelengths(args.wavelengths)
    gold_model = build_gold_model(args.gold_model, args.gold_csv)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    csv_digest = None
    if args.gold_csv is not None:
        csv_digest = hashlib.sha256(args.gold_csv.read_bytes()).hexdigest()
    assumptions = {
        "geometry": asdict(geometry),
        "wavelengths_nm": wavelengths,
        "gold_model": args.gold_model,
        "gold_csv": str(args.gold_csv.resolve()) if args.gold_csv else None,
        "gold_csv_sha256": csv_digest,
        "cascade": args.cascade,
        "use_source_symmetry": not args.no_symmetry,
        "dtype": "complex128",
        "asr_statement": (
            "outer Au/air boundary matched exactly; concentric Au/PMMA "
            "boundary sampled in the transformed coordinates"
        ),
    }
    signature = configuration_signature(assumptions)
    prefix: Path = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    study = Study(
        geometry=geometry,
        gold_epsilon=gold_model,
        wavelengths=wavelengths,
        cascade=args.cascade,
        use_symmetry=not args.no_symmetry,
        device=device,
        checkpoint=prefix.with_name(prefix.name + "_checkpoint.json"),
        signature=signature,
    )

    spectra: dict[int, Sequence[dict[str, object]]] = {}
    for candidate in candidates:
        values = asdict(fixed)
        values[axis] = candidate
        spectra[candidate] = study.spectrum(NumericalConfig(**values))
    selected, converged, comparisons = choose_candidate(
        candidates, spectra, args.tolerance
    )
    selected_values = asdict(fixed)
    selected_values[axis] = selected
    recommendation = NumericalConfig(**selected_values)
    report = {
        "status": "converged" if converged else "candidate_range_insufficient",
        "axis": axis,
        "assumptions": assumptions,
        "fixed_numerics": asdict(fixed),
        "candidates": list(candidates),
        "tolerance": args.tolerance,
        "criterion": "two consecutive adjacent refinements below tolerance",
        "comparisons": comparisons,
        "recommendation": asdict(recommendation),
        "selected_spectrum": spectra[selected],
        "integration": {
            "torch": torch.__version__,
            "device": str(device),
            "completed_cases": len(study.cases),
        },
    }
    report_path = prefix.with_name(prefix.name + "_convergence.json")
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    write_csv(prefix.with_name(prefix.name + "_all_cases.csv"), study.cases.values())
    write_csv(prefix.with_name(prefix.name + "_selected_spectrum.csv"), spectra[selected])
    return report, report_path
