"""Reproduce Fig. 8 of Wang et al., Optics Express 30, 21295 (2022).

The simulation uses the paper's square metallic patch and the separable ASR
implementation in :mod:`rcwa_ext`.  Figure 8(a) contains zero-order and total
power, while Fig. 8(b) contains their differences after higher diffraction
orders open.  The paper's HFSS samples are not distributed with the article;
an optional CSV can be overlaid without inventing reference data.

Examples
--------
Paper settings (expensive; N=M=20 ASR requires a large dense eigensolve)::

    python reproduce_asr_fig8.py --study fig8 --device cuda

Fast integration/smoke case::

    python reproduce_asr_fig8.py --study smoke --device cpu --no-plot

Resume an interrupted sweep::

    python reproduce_asr_fig8.py --study fig8 --device cuda --resume
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from rcwa_ext import CustomRCWA_ASR_FR


PAPER_DOI = "10.1364/OE.459110"
C_MM_PER_NS = 299.792458
PERIOD_MM = 30.0
PATCH_FILL_X = 0.5
PATCH_FILL_Y = 0.5
PATCH_THICKNESS_MM = 0.01
METAL_EPS_TORCWA = 1.0 + 1.0e6j
ASR_G = 1.0e-3
FIG8_FREQUENCY_MIN_GHZ = 2.0
FIG8_FREQUENCY_MAX_GHZ = 18.0
FIG8_ASR_FR_ORDER = 8
FIG8_ASR_ORDER = 20
NUMERICS_REVISION = "fig8-v2-q-modes-separable-quadrature"

CSV_COLUMNS = (
    "study",
    "method",
    "factorization_rules",
    "frequency_GHz",
    "wavelength_mm",
    "order_x",
    "order_y",
    "harmonics",
    "modal_dimension",
    "grid_x",
    "grid_y",
    "quadrature_grid",
    "asr_G",
    "cascade",
    "dtype",
    "device",
    "propagating_orders_air",
    "R00",
    "T00",
    "R_total",
    "T_total",
    "delta_R",
    "delta_T",
    "A_total",
    "power_balance",
    "passivity_violation",
    "cond_T",
    "elapsed_s",
    "numerics_revision",
)


@dataclass(frozen=True)
class RunCase:
    method: str
    factorization_rules: bool
    frequency_ghz: float
    order: int


def inverse_mm_from_ghz(frequency_ghz: float) -> float:
    """Convert GHz (= ns^-1) to torcwa frequency in mm^-1."""
    return frequency_ghz / C_MM_PER_NS


def wavelength_mm(frequency_ghz: float) -> float:
    return C_MM_PER_NS / frequency_ghz


def first_grating_cutoff_ghz(period_mm: float = PERIOD_MM) -> float:
    """Normal-incidence Rayleigh frequency for the (±1,0)/(0,±1) orders."""
    return C_MM_PER_NS / period_mm


def propagating_order_count_air(frequency_ghz: float, order: int) -> int:
    """Count propagating reflected/transmitted orders in an air half-space."""
    ratio = wavelength_mm(frequency_ghz) / PERIOD_MM
    tolerance = 64.0 * np.finfo(float).eps
    return sum(
        (ratio * m) ** 2 + (ratio * n) ** 2 <= 1.0 + tolerance
        for m in range(-order, order + 1)
        for n in range(-order, order + 1)
    )


def _all_orders(simulation: CustomRCWA_ASR_FR) -> torch.Tensor:
    return torch.cartesian_prod(simulation.order_x, simulation.order_y)


def _tensor_scalar(value: torch.Tensor) -> float:
    return float(torch.real(value).detach().cpu().item())


def power_for_x_incidence(simulation: CustomRCWA_ASR_FR) -> dict[str, float | None]:
    """Power for normal-incidence Cartesian x polarization.

    With torcwa's azimuth set to zero, the incident p basis is Cartesian x.
    Both p and s output amplitudes must be summed for each selected order.
    ``power_norm=True`` makes the squared amplitudes power fractions and sets
    evanescent-order power to zero.
    """
    orders = _all_orders(simulation)
    zero = torch.tensor([[0, 0]], dtype=torch.int64, device=simulation._device)

    def port_power(selected: torch.Tensor, port: str) -> torch.Tensor:
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
        return torch.sum(torch.abs(co) ** 2 + torch.abs(cross) ** 2)

    r00 = port_power(zero, "reflection")
    t00 = port_power(zero, "transmission")
    r_total = port_power(orders, "reflection")
    t_total = port_power(orders, "transmission")
    balance = r_total + t_total
    absorptance = 1.0 - balance
    condition = simulation.asr_condition_numbers[-1]
    return {
        "R00": _tensor_scalar(r00),
        "T00": _tensor_scalar(t00),
        "R_total": _tensor_scalar(r_total),
        "T_total": _tensor_scalar(t_total),
        "delta_R": _tensor_scalar(r_total - r00),
        "delta_T": _tensor_scalar(t_total - t00),
        "A_total": _tensor_scalar(absorptance),
        "power_balance": _tensor_scalar(balance),
        "passivity_violation": _tensor_scalar(torch.clamp(-absorptance, min=0.0)),
        "cond_T": None if condition is None else _tensor_scalar(condition),
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def simulate_case(
    case: RunCase,
    *,
    grid: int,
    asr_g: float,
    cascade: str,
    dtype: torch.dtype,
    device: torch.device,
    compute_condition_number: bool,
    quadrature_grid: int,
    study: str,
) -> dict[str, object]:
    """Run one independent frequency point so large eigensystems are released."""
    frequency = inverse_mm_from_ghz(case.frequency_ghz)
    _synchronize(device)
    started = time.perf_counter()
    # _StableLinearAlgebraMixin deliberately takes freq/order/L positionally.
    simulation = CustomRCWA_ASR_FR(
        frequency,
        [case.order, case.order],
        [PERIOD_MM, PERIOD_MM],
        dtype=dtype,
        device=device,
        asr_G=asr_g,
        asr_quadrature_grid=quadrature_grid,
        cascade=cascade,
        smatrix_size="half",
        store_mode_couplings=False,
        verify_cascade=False,
        compute_condition_numbers=(
            compute_condition_number or not case.factorization_rules
        ),
        stable_eig_grad=False,
    )
    simulation.add_input_layer(eps=1.0, mu=1.0)
    simulation.add_output_layer(eps=1.0, mu=1.0)
    zero = torch.tensor(0.0, dtype=torch.float64, device=device)
    simulation.set_incident_angle(inc_ang=zero, azi_ang=zero)
    simulation.add_layer_metal_patch_asr(
        thickness=PATCH_THICKNESS_MM,
        eps_bg=1.0,
        eps_metal=METAL_EPS_TORCWA,
        fill_factor_x=PATCH_FILL_X,
        fill_factor_y=PATCH_FILL_Y,
        nx=grid,
        ny=grid,
        factorization_rules=case.factorization_rules,
    )

    simulation.solve_global_smatrix()
    powers = power_for_x_incidence(simulation)
    _synchronize(device)
    elapsed = time.perf_counter() - started

    harmonics = (2 * case.order + 1) ** 2
    result: dict[str, object] = {
        "study": study,
        "method": case.method,
        "factorization_rules": case.factorization_rules,
        "frequency_GHz": case.frequency_ghz,
        "wavelength_mm": wavelength_mm(case.frequency_ghz),
        "order_x": case.order,
        "order_y": case.order,
        "harmonics": harmonics,
        "modal_dimension": 2 * harmonics,
        "grid_x": grid,
        "grid_y": grid,
        "quadrature_grid": quadrature_grid,
        "asr_G": asr_g,
        "cascade": cascade,
        "dtype": str(dtype).removeprefix("torch."),
        "device": str(device),
        "propagating_orders_air": propagating_order_count_air(
            case.frequency_ghz, case.order
        ),
        "elapsed_s": elapsed,
        "numerics_revision": NUMERICS_REVISION,
    }
    result.update(powers)
    del simulation
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _method_specs(methods: str, asr_fr_order: int, asr_order: int) -> list[tuple[str, bool, int]]:
    specs: list[tuple[str, bool, int]] = []
    if methods in ("both", "asr-fr"):
        specs.append(("ASR-FR", True, asr_fr_order))
    if methods in ("both", "asr"):
        specs.append(("ASR", False, asr_order))
    return specs


def _frequency_values(args: argparse.Namespace) -> np.ndarray:
    if args.frequencies:
        values = np.asarray(
            [float(item.strip()) for item in args.frequencies.split(",")],
            dtype=float,
        )
    else:
        values = np.linspace(args.start_ghz, args.stop_ghz, args.points)
    if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("Frequencies must be a non-empty list of positive values.")
    return values


def build_cases(args: argparse.Namespace) -> list[RunCase]:
    if args.study == "fig8":
        frequencies = _frequency_values(args)
        specs = _method_specs(args.methods, args.asr_fr_order, args.asr_order)
        return [
            RunCase(method, factorization, float(frequency), order)
            for method, factorization, order in specs
            for frequency in frequencies
        ]
    if args.study == "convergence":
        specs = _method_specs(args.methods, 1, 1)
        return [
            RunCase(method, factorization, args.convergence_frequency_ghz, order)
            for method, factorization, _unused in specs
            for order in range(1, args.max_order + 1)
        ]
    # Two frequencies exercise the only-zero-order and multi-order regimes.
    specs = _method_specs(args.methods, 1, 1)
    return [
        RunCase(method, factorization, frequency, 1)
        for method, factorization, _unused in specs
        for frequency in (6.0, 12.0)
    ]


def _case_key_from_values(
    method: str,
    frequency: float,
    order: int,
    grid: int,
    quadrature_grid: int,
    asr_g: float,
    cascade: str,
    dtype: str,
) -> tuple[object, ...]:
    return (
        method,
        round(frequency, 12),
        order,
        grid,
        quadrature_grid,
        round(asr_g, 12),
        cascade,
        dtype,
        NUMERICS_REVISION,
    )


def _row_key(row: dict[str, object]) -> tuple[object, ...]:
    return _case_key_from_values(
        str(row["method"]),
        float(row["frequency_GHz"]),
        int(row["order_x"]),
        int(row["grid_x"]),
        int(row.get("quadrature_grid", 0)),
        float(row["asr_G"]),
        str(row["cascade"]),
        str(row["dtype"]),
    )


def read_csv(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(rows: Iterable[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})
    temporary.replace(path)


def _numeric_rows(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    numeric = {
        "frequency_GHz",
        "R00",
        "T00",
        "R_total",
        "T_total",
        "delta_R",
        "delta_T",
    }
    converted: list[dict[str, object]] = []
    for row in rows:
        item = dict(row)
        for key in numeric:
            item[key] = float(item[key])
        converted.append(item)
    return converted


def _load_hfss_csv(path: Path | None) -> list[dict[str, object]]:
    if path is None:
        return []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    required = {"frequency_GHz", "R00", "T00"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(
            "HFSS CSV must contain frequency_GHz,R00,T00 columns."
        )
    return _numeric_rows(rows)


def plot_fig8(
    rows: list[dict[str, object]], output: Path, hfss_csv: Path | None, show: bool
) -> None:
    import matplotlib.pyplot as plt

    values = _numeric_rows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    colors = {"R": "tab:blue", "T": "tab:orange"}
    line_styles = {"ASR-FR": "-", "ASR": "--"}
    for method in ("ASR-FR", "ASR"):
        selected = sorted(
            (row for row in values if row["method"] == method),
            key=lambda row: float(row["frequency_GHz"]),
        )
        if not selected:
            continue
        frequency = [float(row["frequency_GHz"]) for row in selected]
        for power in ("R", "T"):
            axes[0].plot(
                frequency,
                [float(row[f"{power}_total"]) for row in selected],
                color=colors[power],
                linestyle=line_styles[method],
                linewidth=1.8,
                label=f"{method} total {power}",
            )
            if method == "ASR-FR":
                axes[0].plot(
                    frequency,
                    [float(row[f"{power}00"]) for row in selected],
                    color=colors[power],
                    linestyle=":",
                    linewidth=1.5,
                    label=f"ASR-FR zero-order {power}",
                )
            # Fig. 8(b) is the difference between the ASR-FR zero-order and
            # ASR-FR total curves.  ASR differences remain available in CSV as
            # a diagnostic, but are not added to the paper-style panel.
            if method == "ASR-FR":
                axes[1].plot(
                    frequency,
                    [float(row[f"delta_{power}"]) for row in selected],
                    color=colors[power],
                    linestyle="-",
                    linewidth=1.8,
                    label=f"ASR-FR $\\Delta${power}",
                )

    hfss = _load_hfss_csv(hfss_csv)
    if hfss:
        for power, marker in (("R", "o"), ("T", "s")):
            axes[0].scatter(
                [float(row["frequency_GHz"]) for row in hfss],
                [float(row[f"{power}00"]) for row in hfss],
                s=22,
                marker=marker,
                facecolors="none",
                edgecolors=colors[power],
                label=f"HFSS zero-order {power}",
            )

    cutoff = first_grating_cutoff_ghz()
    for axis in axes:
        axis.axvline(cutoff, color="0.45", linewidth=1.0, linestyle="-.")
        axis.grid(True, alpha=0.25)
        axis.set_xlabel("Frequency (GHz)")
        axis.legend(fontsize=8, ncol=2)
    axes[0].set_ylabel("Power fraction")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_title("(a) Zero-order and total power")
    axes[1].set_ylabel("Total minus zero-order power")
    axes[1].set_ylim(-0.005, 0.26)
    axes[1].set_title("(b) Higher-order contribution")
    fig.suptitle(
        "Square metallic patch: $\\Lambda=30$ mm, $f_x=f_y=0.5$, "
        "$h=0.01$ mm"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    if show:
        plt.show()
    plt.close(fig)


def write_metadata(args: argparse.Namespace, rows: list[dict[str, object]], path: Path) -> None:
    metadata = {
        "paper": {
            "title": "2D rigorous coupled wave analysis with adaptive spatial resolution for a multilayer periodic structure",
            "doi": PAPER_DOI,
            "figure": 8,
        },
        "geometry": {
            "period_mm": [PERIOD_MM, PERIOD_MM],
            "patch_fill": [PATCH_FILL_X, PATCH_FILL_Y],
            "patch_thickness_mm": PATCH_THICKNESS_MM,
            "background_epsilon": 1.0,
            "metal_epsilon_torcwa_exp_minus_iwt": [
                METAL_EPS_TORCWA.real,
                METAL_EPS_TORCWA.imag,
            ],
        },
        "numerics": {
            "asr_G": args.asr_g,
            "asr_fr_order_fig8": args.asr_fr_order,
            "asr_order_fig8": args.asr_order,
            "grid": args.grid,
            "quadrature_grid": args.quadrature_grid,
            "cascade": args.cascade,
            "dtype": args.dtype,
            "device": args.device,
            "smatrix_size": "half",
            "polarization": "normal-incidence x (torcwa p at azimuth 0)",
        },
        "rayleigh_cutoff_GHz": first_grating_cutoff_ghz(),
        "numerics_revision": NUMERICS_REVISION,
        "rows": len(rows),
        "hfss_reference_data": (
            str(args.hfss_csv) if args.hfss_csv is not None else None
        ),
        "note": (
            "The article does not distribute the underlying HFSS samples. "
            "No HFSS values are synthesized by this script."
        ),
    }
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    return torch.device(name)


def _validate_arguments(args: argparse.Namespace) -> None:
    for name in ("asr_fr_order", "asr_order", "max_order"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be at least 1.")
    if args.grid < max(32, 4 * max(args.asr_fr_order, args.asr_order) + 4):
        if args.study == "fig8":
            raise ValueError(
                "Sampling grid is too small for the selected Fig. 8 order; "
                "use at least max(32, 4*order+4)."
            )
    if not 0.0 < args.asr_g < 1.0:
        raise ValueError("--asr-g must be in (0,1).")
    if args.quadrature_grid < args.grid:
        raise ValueError("--quadrature-grid must be at least --grid.")
    if args.passivity_tolerance < 0.0:
        raise ValueError("--passivity-tolerance must be nonnegative.")
    if args.study == "fig8" and not args.frequencies and args.points < 2:
        raise ValueError("--points must be at least 2.")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study", choices=("fig8", "convergence", "smoke"), default="fig8"
    )
    parser.add_argument(
        "--methods", choices=("both", "asr-fr", "asr"), default="both"
    )
    parser.add_argument("--asr-fr-order", type=int, default=FIG8_ASR_FR_ORDER)
    parser.add_argument("--asr-order", type=int, default=FIG8_ASR_ORDER)
    parser.add_argument("--max-order", type=int, default=12)
    parser.add_argument("--grid", type=int, default=256)
    parser.add_argument(
        "--quadrature-grid",
        type=int,
        default=4096,
        help=(
            "1-D ASR/T quadrature. 4096 is recommended for the paper's "
            "ill-conditioned non-FR N=M=20 reference."
        ),
    )
    parser.add_argument("--asr-g", type=float, default=ASR_G)
    parser.add_argument("--start-ghz", type=float, default=FIG8_FREQUENCY_MIN_GHZ)
    parser.add_argument("--stop-ghz", type=float, default=FIG8_FREQUENCY_MAX_GHZ)
    parser.add_argument("--points", type=int, default=33)
    parser.add_argument(
        "--frequencies",
        help="Comma-separated GHz values; overrides --start/--stop/--points.",
    )
    parser.add_argument("--convergence-frequency-ghz", type=float, default=6.0)
    parser.add_argument(
        "--cascade", choices=("redheffer", "algo2a"), default="redheffer"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--dtype", choices=("complex64", "complex128"), default="complex128"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("asr_fig8_results_v2")
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--condition-number",
        action="store_true",
        help=(
            "Also compute cond(T) for ASR-FR. The non-FR ASR reference always "
            "records the inexpensive exact separable cond(T)."
        ),
    )
    parser.add_argument(
        "--passivity-tolerance",
        type=float,
        default=5.0e-5,
        help="Reject a passive-material result if R+T exceeds 1 by more than this.",
    )
    parser.add_argument(
        "--plot", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--hfss-csv",
        type=Path,
        help="Optional external CSV with frequency_GHz,R00,T00 columns.",
    )
    return parser


def main(args: argparse.Namespace) -> list[dict[str, object]]:
    _validate_arguments(args)
    device = _select_device(args.device)
    dtype = torch.complex128 if args.dtype == "complex128" else torch.complex64
    cases = build_cases(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{args.study}_powers.csv"
    plot_path = args.output_dir / f"{args.study}.png"
    metadata_path = args.output_dir / f"{args.study}_metadata.json"

    if csv_path.exists() and not args.resume:
        raise FileExistsError(
            f"{csv_path} already exists. Use --resume or choose another --output-dir."
        )
    rows = read_csv(csv_path) if args.resume else []
    incompatible = [
        row
        for row in rows
        if row.get("numerics_revision", "") != NUMERICS_REVISION
    ]
    if incompatible:
        raise ValueError(
            "The existing CSV predates the stable Fig. 8 v2 formulation. "
            "Use a new --output-dir; do not resume the non-passive v1 rows."
        )
    completed = {
        _row_key(row)
        for row in rows
        if not (
            args.condition_number
            and str(row.get("cond_T", "")).strip() == ""
        )
    }

    largest_order = max(case.order for case in cases)
    modal_dimension = 2 * (2 * largest_order + 1) ** 2
    dense_mib = modal_dimension**2 * (16 if dtype == torch.complex128 else 8) / 2**20
    print(
        f"device={device}, dtype={args.dtype}, cases={len(cases)}, "
        f"largest modal matrix={modal_dimension}x{modal_dimension} "
        f"({dense_mib:.1f} MiB per dense matrix)"
    )
    if largest_order >= 16:
        warnings.warn(
            "The paper's ASR order 20 uses many simultaneous dense matrices; "
            "several GiB of free accelerator memory may be required.",
            RuntimeWarning,
            stacklevel=1,
        )

    for index, case in enumerate(cases, start=1):
        key = _case_key_from_values(
            case.method,
            case.frequency_ghz,
            case.order,
            args.grid,
            args.quadrature_grid,
            args.asr_g,
            args.cascade,
            args.dtype,
        )
        if key in completed:
            print(
                f"[{index}/{len(cases)}] skip {case.method} "
                f"{case.frequency_ghz:g} GHz N={case.order}"
            )
            continue
        print(
            f"[{index}/{len(cases)}] run  {case.method} "
            f"{case.frequency_ghz:g} GHz N=M={case.order}"
        )
        row = simulate_case(
            case,
            grid=args.grid,
            asr_g=args.asr_g,
            cascade=args.cascade,
            dtype=dtype,
            device=device,
            compute_condition_number=args.condition_number,
            quadrature_grid=args.quadrature_grid,
            study=args.study,
        )
        if float(row["A_total"]) < -args.passivity_tolerance:
            raise RuntimeError(
                f"Non-passive numerical result for {case.method} at "
                f"{case.frequency_ghz:g} GHz: R+T="
                f"{float(row['power_balance']):.8g}. No invalid curve was "
                "written. Increase --quadrature-grid, inspect cond_T, and "
                "run validate_asr_fig8.py --integration before continuing."
            )
        # Replace an old row with the same numerical key (for example when
        # --condition-number enriches a resumed run) instead of duplicating it.
        rows = [existing for existing in rows if _row_key(existing) != key]
        rows.append(row)
        rows.sort(key=lambda item: (str(item["method"]), float(item["frequency_GHz"]), int(item["order_x"])))
        write_csv(rows, csv_path)  # checkpoint after every expensive eigensolve
        completed.add(key)
        print(
            f"    R={float(row['R_total']):.8f}, T={float(row['T_total']):.8f}, "
            f"A={float(row['A_total']):.8f}, elapsed={float(row['elapsed_s']):.2f}s"
        )

    write_metadata(args, rows, metadata_path)
    if args.plot and args.study in ("fig8", "smoke"):
        plot_fig8(rows, plot_path, args.hfss_csv, args.show)
        print(f"wrote {plot_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {metadata_path}")
    return rows


if __name__ == "__main__":
    main(_parser().parse_args())
