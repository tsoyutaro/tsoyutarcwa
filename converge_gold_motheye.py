"""Convergence study for Au moth-eye pillars on an Au substrate.

The tapered pillar is represented by centered circular matched-ASR slices.
The script alternately refines the z-slice count, Fourier order, and ASR
quadrature grid at representative wavelengths.  Results are checkpointed after
every solve because a metal/high-aspect-ratio convergence study can be long.

Default physical assumptions
----------------------------
* 60-degree triangular array, period 200 nm.
* Gold cone/frustum pillars in air, height 500 nm.
* Radius increases from 5 nm at the air-side tip to 95 nm at the Au substrate.
* Semi-infinite Au substrate.  Far-side T=0; power entering the substrate is
  reported separately from absorption in the patterned moth-eye region.
* Normal-incidence x polarization.  Cs source-sector reduction is used.
* Rakić Lorentz-Drude bulk-Au dispersion.

Use measured n,k data for a final fabricated sample with
``--gold-model csv --gold-csv your_data.csv``.
"""

from __future__ import annotations

import argparse
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
from rcwa_solver_auto import (
    ASROptions,
    AutoRCWA,
    Circle,
    GroupTheoryOptions,
    Lattice,
    LayerSpec,
    Material,
    OutputSpec,
)


@dataclass(frozen=True)
class GeometryConfig:
    period_nm: float = 200.0
    height_nm: float = 500.0
    tip_radius_nm: float = 5.0
    base_radius_nm: float = 95.0
    profile_power: float = 1.0
    lattice: str = "triangular"
    substrate_mode: str = "semi-infinite"
    substrate_thickness_nm: float = 200.0
    back_index: float = 1.0
    asr_circle_g: float = 0.03


@dataclass(frozen=True)
class NumericalConfig:
    order: int
    slices: int
    grid: int


def _parse_int_list(text: str, name: str) -> tuple[int, ...]:
    values = tuple(sorted({int(value.strip()) for value in text.split(",")}))
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{name} must contain positive comma-separated integers.")
    return values


def _parse_wavelengths(text: str) -> tuple[float, ...]:
    stripped = text.strip()
    if ":" in stripped:
        parts = tuple(float(value) for value in stripped.split(":"))
        if len(parts) != 3:
            raise ValueError("Wavelength range must be start:stop:step.")
        start, stop, step = parts
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


def _lattice(config: GeometryConfig) -> Lattice:
    if config.lattice == "triangular":
        return Lattice.triangular(1.0)
    if config.lattice == "square":
        return Lattice.square(1.0)
    raise ValueError("lattice must be triangular or square.")


def _slice_radius_nm(config: GeometryConfig, index: int, slices: int) -> float:
    # Layer order is air-side tip -> substrate-side base.
    coordinate = (index + 0.5) / slices
    return config.tip_radius_nm + (
        config.base_radius_nm - config.tip_radius_nm
    ) * coordinate**config.profile_power


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
    """Unit-cell averaged z flux; direction is +1 or -1."""
    magnetic = direction * (electric_to_magnetic @ electric)
    count = electric.numel() // 2
    ex, ey = electric[:count], electric[count:]
    hx, hy = magnetic[:count], magnetic[count:]
    flux = 0.5 * torch.real(
        torch.sum(ex * torch.conj(hy) - ey * torch.conj(hx))
    )
    return float(flux.detach().cpu())


def _validate_geometry(config: GeometryConfig) -> None:
    positive = {
        "period_nm": config.period_nm,
        "height_nm": config.height_nm,
        "tip_radius_nm": config.tip_radius_nm,
        "base_radius_nm": config.base_radius_nm,
        "profile_power": config.profile_power,
    }
    for name, value in positive.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
    if config.tip_radius_nm > config.base_radius_nm:
        raise ValueError("tip_radius_nm must not exceed base_radius_nm.")
    if 2.0 * config.base_radius_nm >= config.period_nm:
        raise ValueError(
            "matched-ASR requires 2*base_radius < period; leave a positive gap."
        )
    if config.substrate_mode not in {"semi-infinite", "finite"}:
        raise ValueError("substrate_mode must be semi-infinite or finite.")
    if config.substrate_mode == "finite" and config.substrate_thickness_nm <= 0.0:
        raise ValueError("A finite substrate needs positive thickness.")
    if config.back_index <= 0.0:
        raise ValueError("back_index must be positive.")
    if not 0.0 < config.asr_circle_g < 1.0:
        raise ValueError("asr_circle_g must lie in (0,1).")


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
    """Run one wavelength/configuration and return flux-normalized observables."""
    started = time.perf_counter()
    epsilon_gold = gold_epsilon(wavelength_nm)
    period = geometry.period_nm
    frequency = period / wavelength_nm
    semi_infinite = geometry.substrate_mode == "semi-infinite"
    symmetry_allowed = use_symmetry and semi_infinite
    simulation = AutoRCWA(
        freq=frequency,
        order=[numerical.order, numerical.order],
        lattice=_lattice(geometry),
        cascade=cascade,
        outputs=OutputSpec(smatrix_size="half", fields="none"),
        asr=ASROptions(
            circle_G=geometry.asr_circle_g,
            grid=(numerical.grid, numerical.grid),
            factorization_rules=True,
        ),
        group_theory=GroupTheoryOptions(
            enabled=symmetry_allowed,
            symmetry="auto",
            strict=symmetry_allowed,
            polarization="x" if symmetry_allowed else None,
        ),
        verify_cascade=False,
        dtype=torch.complex128,
        device=device,
    )
    simulation.add_input_layer(eps=1.0, mu=1.0)
    simulation.add_output_layer(
        eps=epsilon_gold if semi_infinite else geometry.back_index**2,
        mu=1.0,
    )
    simulation.set_incident_angle(0.0, 0.0)

    thickness = geometry.height_nm / numerical.slices / period
    for layer in range(numerical.slices):
        radius_nm = _slice_radius_nm(geometry, layer, numerical.slices)
        simulation.add_structured_layer(
            LayerSpec(
                thickness=thickness,
                geometry=Circle(radius_nm / period),
                background=Material(1.0, 1.0),
                inclusion=Material(epsilon_gold, 1.0),
                method="matched-asr",
                factorization_rules=True,
                label=f"moth-eye-{layer:03d}",
            )
        )
    if not semi_infinite:
        # Mixed homogeneous/structured stacks currently require the full basis.
        simulation.add_layer(
            geometry.substrate_thickness_nm / period,
            eps=epsilon_gold,
            mu=1.0,
        )
    simulation.solve_global_smatrix()

    incident = _zero_order_x_source(simulation)
    reflected = simulation.S[1] @ incident
    transmitted = simulation.S[0] @ incident
    incident_flux = _mean_poynting_z(incident, simulation.Vi, direction=1)
    reflected_flux = _mean_poynting_z(reflected, simulation.Vi, direction=-1)
    transmitted_flux = _mean_poynting_z(
        transmitted, simulation.Vo, direction=1
    )
    if not incident_flux > 0.0:
        raise RuntimeError("Incident power flux is not positive.")
    reflectance = -reflected_flux / incident_flux
    output_flux = transmitted_flux / incident_flux

    if semi_infinite:
        # No asymptotic transmitted port exists beyond an infinite lossy metal.
        transmission_far = 0.0
        substrate_absorptance = output_flux
        motheye_absorptance = 1.0 - reflectance - output_flux
        absorptance_total = 1.0 - reflectance
        convergence_metrics = {
            "reflectance": reflectance,
            "power_into_substrate": output_flux,
            "motheye_absorptance": motheye_absorptance,
        }
    else:
        transmission_far = output_flux
        substrate_absorptance = None
        motheye_absorptance = None
        absorptance_total = 1.0 - reflectance - transmission_far
        convergence_metrics = {
            "reflectance": reflectance,
            "transmittance": transmission_far,
            "absorptance": absorptance_total,
        }

    diagnostics = simulation.cascade_diagnostics
    return {
        "wavelength_nm": wavelength_nm,
        "order": numerical.order,
        "slices": numerical.slices,
        "grid": numerical.grid,
        "epsilon_gold_real": epsilon_gold.real,
        "epsilon_gold_imag": epsilon_gold.imag,
        "reflectance": reflectance,
        "transmittance_far": transmission_far,
        "absorptance_total": absorptance_total,
        "motheye_absorptance": motheye_absorptance,
        "substrate_absorptance": substrate_absorptance,
        "power_into_substrate": output_flux if semi_infinite else None,
        "convergence_metrics": convergence_metrics,
        "passivity_warning": bool(
            reflectance < -1.0e-6
            or absorptance_total < -1.0e-5
            or (
                semi_infinite
                and (
                    output_flux < -1.0e-6
                    or motheye_absorptance is None
                    or motheye_absorptance < -1.0e-5
                )
            )
        ),
        "reduced_dimension": diagnostics.get("reduced_dimension"),
        "full_dimension": diagnostics.get("full_dimension", 2 * simulation.order_N),
        "runtime_seconds": time.perf_counter() - started,
    }


def _finite_metrics(result: dict[str, object]) -> dict[str, float]:
    raw = result["convergence_metrics"]
    assert isinstance(raw, dict)
    values = {name: float(value) for name, value in raw.items()}
    if not all(math.isfinite(value) for value in values.values()):
        raise RuntimeError("A convergence observable is NaN or infinity.")
    return values


def _adjacent_error(
    coarse: Sequence[dict[str, object]], fine: Sequence[dict[str, object]]
) -> tuple[float, dict[str, float]]:
    if len(coarse) != len(fine):
        raise ValueError("Convergence spectra have different lengths.")
    per_metric: dict[str, float] = {}
    for left, right in zip(coarse, fine):
        left_metrics, right_metrics = _finite_metrics(left), _finite_metrics(right)
        if left_metrics.keys() != right_metrics.keys():
            raise ValueError("Convergence metrics disagree.")
        for name in left_metrics:
            difference = abs(left_metrics[name] - right_metrics[name])
            per_metric[name] = max(per_metric.get(name, 0.0), difference)
    return max(per_metric.values()), per_metric


def _choose_candidate(
    candidates: Sequence[int],
    spectra: dict[int, Sequence[dict[str, object]]],
    tolerance: float,
) -> tuple[int, bool, list[dict[str, object]]]:
    comparisons: list[dict[str, object]] = []
    for coarse, fine in zip(candidates, candidates[1:]):
        maximum, per_metric = _adjacent_error(spectra[coarse], spectra[fine])
        comparisons.append(
            {
                "coarse": coarse,
                "fine": fine,
                "max_abs_change": maximum,
                "per_metric": per_metric,
                "passed": maximum <= tolerance,
            }
        )
    # Require two consecutive refinement steps below tolerance.  Recommend the
    # middle value: it has both a converged incoming and outgoing comparison.
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
                    "move/delete it or select another --output-prefix."
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

    def evaluate(self, wavelength: float, numerical: NumericalConfig):
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

    def spectrum(
        self, numerical: NumericalConfig, wavelengths: Iterable[float] | None = None
    ) -> list[dict[str, object]]:
        selected = self.wavelengths if wavelengths is None else tuple(wavelengths)
        return [self.evaluate(wavelength, numerical) for wavelength in selected]


def _scan_axis(
    study: Study,
    axis: str,
    candidates: Sequence[int],
    current: NumericalConfig,
    tolerance: float,
) -> tuple[int, bool, dict[str, object]]:
    spectra: dict[int, Sequence[dict[str, object]]] = {}
    for candidate in candidates:
        values = asdict(current)
        values[axis] = candidate
        spectra[candidate] = study.spectrum(NumericalConfig(**values))
    selected, converged, comparisons = _choose_candidate(
        candidates, spectra, tolerance
    )
    return selected, converged, {
        "axis": axis,
        "fixed": asdict(current),
        "candidates": list(candidates),
        "comparisons": comparisons,
        "selected": selected,
        "converged": converged,
    }


def _write_csv(path: Path, cases: Iterable[dict[str, object]]) -> None:
    rows = list(cases)
    columns = (
        "wavelength_nm",
        "order",
        "slices",
        "grid",
        "epsilon_gold_real",
        "epsilon_gold_imag",
        "reflectance",
        "transmittance_far",
        "absorptance_total",
        "motheye_absorptance",
        "substrate_absorptance",
        "power_into_substrate",
        "passivity_warning",
        "reduced_dimension",
        "full_dimension",
        "runtime_seconds",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _configuration_signature(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--period-nm", type=float, default=200.0)
    parser.add_argument("--height-nm", type=float, default=500.0)
    parser.add_argument("--tip-radius-nm", type=float, default=5.0)
    parser.add_argument("--base-radius-nm", type=float, default=95.0)
    parser.add_argument("--profile-power", type=float, default=1.0)
    parser.add_argument("--lattice", choices=("triangular", "square"), default="triangular")
    parser.add_argument(
        "--substrate-mode", choices=("semi-infinite", "finite"), default="semi-infinite"
    )
    parser.add_argument("--substrate-thickness-nm", type=float, default=200.0)
    parser.add_argument("--back-index", type=float, default=1.0)
    parser.add_argument("--asr-circle-g", type=float, default=0.03)
    parser.add_argument("--gold-model", choices=("rakic-ld", "csv"), default="rakic-ld")
    parser.add_argument("--gold-csv", type=Path)
    parser.add_argument("--anchor-wavelengths", default="400,550,700")
    parser.add_argument("--orders", default="3,4,5,6,7")
    parser.add_argument("--slices", default="12,16,24,32,48")
    parser.add_argument("--grids", default="96,128,192,256")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=5.0e-3,
        help="Maximum absolute R/T/A change; 0.005 is 0.5 percentage point.",
    )
    parser.add_argument("--max-cycles", type=int, default=3)
    parser.add_argument("--cascade", choices=("redheffer", "algo2a"), default="redheffer")
    parser.add_argument("--no-symmetry", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-prefix", type=Path, default=Path("gold_motheye"))
    parser.add_argument("--run-final-spectrum", action="store_true")
    parser.add_argument("--spectrum-wavelengths", default="400:700:5")
    args = parser.parse_args()

    geometry = GeometryConfig(
        period_nm=args.period_nm,
        height_nm=args.height_nm,
        tip_radius_nm=args.tip_radius_nm,
        base_radius_nm=args.base_radius_nm,
        profile_power=args.profile_power,
        lattice=args.lattice,
        substrate_mode=args.substrate_mode,
        substrate_thickness_nm=args.substrate_thickness_nm,
        back_index=args.back_index,
        asr_circle_g=args.asr_circle_g,
    )
    _validate_geometry(geometry)
    orders = _parse_int_list(args.orders, "orders")
    slices = _parse_int_list(args.slices, "slices")
    grids = _parse_int_list(args.grids, "grids")
    if min(len(orders), len(slices), len(grids)) < 3:
        raise ValueError("Each convergence axis needs at least three candidates.")
    wavelengths = _parse_wavelengths(args.anchor_wavelengths)
    if not math.isfinite(args.tolerance) or args.tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive.")
    if args.max_cycles <= 0:
        raise ValueError("max-cycles must be positive.")
    if geometry.substrate_mode == "finite" and not args.no_symmetry:
        print(
            "note: finite Au film adds a homogeneous layer, so source-sector "
            "reduction is disabled for this mixed stack.",
            flush=True,
        )
    use_symmetry = not args.no_symmetry and geometry.substrate_mode == "semi-infinite"
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    gold_model = build_gold_model(args.gold_model, args.gold_csv)
    gold_csv_digest = None
    if args.gold_csv is not None:
        gold_csv_digest = hashlib.sha256(args.gold_csv.read_bytes()).hexdigest()
    signature_payload = {
        "geometry": asdict(geometry),
        "gold_model": args.gold_model,
        "gold_csv": str(args.gold_csv.resolve()) if args.gold_csv else None,
        "gold_csv_sha256": gold_csv_digest,
        "anchor_wavelengths": wavelengths,
        "cascade": args.cascade,
        "use_symmetry": use_symmetry,
        "dtype": "complex128",
    }
    signature = _configuration_signature(signature_payload)
    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = prefix.with_name(prefix.name + "_checkpoint.json")
    study = Study(
        geometry=geometry,
        gold_epsilon=gold_model,
        wavelengths=wavelengths,
        cascade=args.cascade,
        use_symmetry=use_symmetry,
        device=device,
        checkpoint=checkpoint,
        signature=signature,
    )

    current = NumericalConfig(
        order=orders[min(1, len(orders) - 1)],
        slices=slices[min(2, len(slices) - 1)],
        grid=grids[min(1, len(grids) - 1)],
    )
    history: list[dict[str, object]] = []
    all_converged = False
    for cycle in range(args.max_cycles):
        start = current
        axis_status: list[bool] = []
        for axis, candidates in (
            ("slices", slices),
            ("order", orders),
            ("grid", grids),
        ):
            selected, converged, record = _scan_axis(
                study, axis, candidates, current, args.tolerance
            )
            values = asdict(current)
            values[axis] = selected
            current = NumericalConfig(**values)
            record["cycle"] = cycle + 1
            history.append(record)
            axis_status.append(converged)
        fixed_point = current == start
        all_converged = fixed_point and all(axis_status)
        if fixed_point:
            break

    anchor_spectrum = study.spectrum(current)
    report = {
        "status": "converged" if all_converged else "candidate_range_insufficient",
        "signature": signature,
        "assumptions": signature_payload,
        "candidate_axes": {
            "orders": orders,
            "slices": slices,
            "grids": grids,
            "tolerance": args.tolerance,
            "criterion": "two consecutive adjacent refinements below tolerance",
        },
        "recommendation": asdict(current),
        "history": history,
        "anchor_spectrum": anchor_spectrum,
        "interpretation": (
            "For a semi-infinite Au substrate, transmittance_far=0, "
            "absorptance_total=1-R, power_into_substrate is eventually absorbed "
            "in the substrate, and motheye_absorptance=1-R-power_into_substrate."
        ),
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
    _write_csv(
        prefix.with_name(prefix.name + "_all_cases.csv"), study.cases.values()
    )
    _write_csv(
        prefix.with_name(prefix.name + "_anchor_spectrum.csv"), anchor_spectrum
    )

    if args.run_final_spectrum:
        spectrum_wavelengths = _parse_wavelengths(args.spectrum_wavelengths)
        final_spectrum = study.spectrum(current, spectrum_wavelengths)
        _write_csv(
            prefix.with_name(prefix.name + "_spectrum.csv"), final_spectrum
        )
        prefix.with_name(prefix.name + "_spectrum.json").write_text(
            json.dumps(final_spectrum, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )

    print(json.dumps({"status": report["status"], "recommendation": asdict(current)}, indent=2))
    print(f"report: {report_path.resolve()}")
    return 0 if all_converged else 2


if __name__ == "__main__":
    raise SystemExit(main())
