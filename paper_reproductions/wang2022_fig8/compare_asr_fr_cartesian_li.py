"""Compare Wang-2022 ASR-FR with Cartesian Li factorization.

This is deliberately separate from :mod:`reproduce`: it is a numerical-method
comparison, not a claim that a new curve appears in the paper.  Both methods
use the Fig. 8 square metal patch, normal x-polarized incidence, and identical
rectangular Fourier truncation.  The default 6 GHz frequency follows Fig. 9.

Examples
--------
Practical first sweep::

    python -m paper_reproductions.wang2022_fig8.compare_asr_fr_cartesian_li \
        --orders 1,2,3,4,5,6,8,10,12 --device cuda

Extend the sweep after inspecting passivity and memory use::

    python -m paper_reproductions.wang2022_fig8.compare_asr_fr_cartesian_li \
        --orders 1:20 --device cuda --resume
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import warnings
from pathlib import Path

import torch

_PACKAGE_ROOT = Path(__file__).resolve().parent
_OUTPUTS_ROOT = Path(__file__).resolve().parents[2]
if str(_OUTPUTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_OUTPUTS_ROOT))

from rcwa_ext import CustomRCWA_ASR_FR
from paper_reproductions.wang2022_fig8.reproduce import (
    ASR_G,
    METAL_EPS_TORCWA,
    PATCH_FILL_X,
    PATCH_FILL_Y,
    PATCH_THICKNESS_MM,
    PERIOD_MM,
    first_grating_cutoff_ghz,
    inverse_mm_from_ghz,
    power_for_x_incidence,
    propagating_order_count_air,
    wavelength_mm,
)

METHODS = ("ASR-FR", "Cartesian-Li")
REVISION = "wang2022-asr-fr-vs-cartesian-li-v1"
CSV_COLUMNS = (
    "method",
    "frequency_GHz",
    "order",
    "harmonics",
    "modal_dimension",
    "factorization",
    "fourier_coefficients",
    "grid",
    "quadrature_grid",
    "asr_G",
    "cascade",
    "R00",
    "T00",
    "R_total",
    "T_total",
    "A_total",
    "power_balance",
    "passivity_violation",
    "abs_error_R",
    "abs_error_T",
    "successive_delta_R",
    "successive_delta_T",
    "elapsed_s",
    "device",
    "dtype",
    "revision",
)


def parse_orders(text: str) -> list[int]:
    """Parse comma-separated integers and inclusive ``start:stop`` ranges."""
    values: set[int] = set()
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            parts = token.split(":")
            if len(parts) not in (2, 3):
                raise ValueError(f"Invalid order range: {token!r}")
            start, stop = int(parts[0]), int(parts[1])
            step = int(parts[2]) if len(parts) == 3 else 1
            if step <= 0 or stop < start:
                raise ValueError(f"Invalid order range: {token!r}")
            values.update(range(start, stop + 1, step))
        else:
            values.add(int(token))
    orders = sorted(values)
    if not orders or orders[0] < 1:
        raise ValueError("--orders must contain positive integers.")
    return orders


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    return torch.device(name)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def simulate(
    method: str,
    order: int,
    *,
    frequency_ghz: float,
    grid: int,
    quadrature_grid: int,
    asr_g: float,
    cascade: str,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, object]:
    synchronize(device)
    started = time.perf_counter()
    simulation = CustomRCWA_ASR_FR(
        inverse_mm_from_ghz(frequency_ghz),
        [order, order],
        [PERIOD_MM, PERIOD_MM],
        dtype=dtype,
        device=device,
        asr_G=asr_g,
        asr_quadrature_grid=quadrature_grid,
        cascade=cascade,
        smatrix_size="half",
        store_mode_couplings=False,
        verify_cascade=False,
        compute_condition_numbers=False,
        stable_eig_grad=False,
    )
    simulation.add_input_layer(eps=1.0, mu=1.0)
    simulation.add_output_layer(eps=1.0, mu=1.0)
    zero = torch.tensor(0.0, dtype=torch.float64, device=device)
    simulation.set_incident_angle(inc_ang=zero, azi_ang=zero)
    if method == "ASR-FR":
        simulation.add_layer_metal_patch_asr(
            thickness=PATCH_THICKNESS_MM,
            eps_bg=1.0,
            eps_metal=METAL_EPS_TORCWA,
            fill_factor_x=PATCH_FILL_X,
            fill_factor_y=PATCH_FILL_Y,
            nx=grid,
            ny=grid,
            factorization_rules=True,
        )
        factorization = "ASR mixed inverse/direct rule"
        coefficients = "ASR separable quadrature"
    elif method == "Cartesian-Li":
        simulation.add_layer_rect_li(
            thickness=PATCH_THICKNESS_MM,
            eps_bg=1.0,
            eps_rect=METAL_EPS_TORCWA,
            fill_factor_x=PATCH_FILL_X,
            fill_factor_y=PATCH_FILL_Y,
            physical_grid=grid,
        )
        factorization = "Cartesian x-Toeplitz inverse/y-BTTB Li rule"
        coefficients = "analytic centered rectangle"
    else:
        raise ValueError(f"Unknown method: {method}")
    simulation.solve_global_smatrix()
    powers = power_for_x_incidence(simulation)
    synchronize(device)
    elapsed = time.perf_counter() - started
    harmonics = (2 * order + 1) ** 2
    row: dict[str, object] = {
        "method": method,
        "frequency_GHz": frequency_ghz,
        "order": order,
        "harmonics": harmonics,
        "modal_dimension": 2 * harmonics,
        "factorization": factorization,
        "fourier_coefficients": coefficients,
        "grid": grid,
        "quadrature_grid": quadrature_grid if method == "ASR-FR" else 0,
        "asr_G": asr_g,
        "cascade": cascade,
        "elapsed_s": elapsed,
        "device": str(device),
        "dtype": str(dtype).removeprefix("torch."),
        "revision": REVISION,
    }
    row.update(powers)
    del simulation
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def row_key(row: dict[str, object]) -> tuple[object, ...]:
    return (
        str(row["method"]),
        int(row["order"]),
        round(float(row["frequency_GHz"]), 12),
        int(row["grid"]),
        int(row["quadrature_grid"]),
        round(float(row["asr_G"]), 12),
        str(row["cascade"]),
        str(row["dtype"]),
        str(row["revision"]),
    )


def case_key(
    method: str, order: int, args: argparse.Namespace
) -> tuple[object, ...]:
    return (
        method,
        order,
        round(float(args.frequency_ghz), 12),
        args.grid,
        args.quadrature_grid if method == "ASR-FR" else 0,
        round(float(args.asr_g), 12),
        args.cascade,
        args.dtype,
        REVISION,
    )


def read_csv(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def enrich_convergence(
    rows: list[dict[str, object]],
    *,
    reference_method: str,
    reference_order: int | None,
    reference_r: float | None,
    reference_t: float | None,
    passivity_tolerance: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    numeric_rows = [dict(row) for row in rows]
    for row in numeric_rows:
        for key in (
            "frequency_GHz", "R00", "T00", "R_total", "T_total",
            "A_total", "power_balance", "passivity_violation", "elapsed_s",
            "asr_G",
        ):
            row[key] = float(row[key])
        for key in ("order", "harmonics", "modal_dimension", "grid", "quadrature_grid"):
            row[key] = int(row[key])

    if (reference_r is None) != (reference_t is None):
        raise ValueError("--reference-r and --reference-t must be supplied together.")
    if reference_r is None:
        candidates = [
            row
            for row in numeric_rows
            if row["method"] == reference_method
            and float(row["passivity_violation"]) <= passivity_tolerance
        ]
        if not candidates:
            raise ValueError(
                f"No passive rows are available for reference method "
                f"{reference_method}."
            )
        requested = [
            row
            for row in candidates
            if reference_order is not None
            and int(row["order"]) == reference_order
        ]
        source = (
            requested[0]
            if requested
            else max(candidates, key=lambda row: int(row["order"]))
        )
        reference_r = float(source["R_total"])
        reference_t = float(source["T_total"])
        reference = {
            "kind": (
                "requested passive internal reference"
                if requested
                else "highest-computed passive fallback reference"
            ),
            "method": reference_method,
            "order": int(source["order"]),
            "R_total": reference_r,
            "T_total": reference_t,
        }
    else:
        if not all(math.isfinite(value) for value in (reference_r, reference_t)):
            raise ValueError("External reference powers must be finite.")
        reference = {
            "kind": "user-supplied external reference",
            "method": None,
            "order": None,
            "R_total": reference_r,
            "T_total": reference_t,
        }

    for method in METHODS:
        selected = sorted(
            (row for row in numeric_rows if row["method"] == method),
            key=lambda row: int(row["order"]),
        )
        previous: dict[str, object] | None = None
        for row in selected:
            row["abs_error_R"] = abs(float(row["R_total"]) - reference_r)
            row["abs_error_T"] = abs(float(row["T_total"]) - reference_t)
            row["successive_delta_R"] = (
                "" if previous is None else abs(float(row["R_total"]) - float(previous["R_total"]))
            )
            row["successive_delta_T"] = (
                "" if previous is None else abs(float(row["T_total"]) - float(previous["T_total"]))
            )
            previous = row
    numeric_rows.sort(key=lambda row: (str(row["method"]), int(row["order"])))
    return numeric_rows, reference


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})
    temporary.replace(path)


def plot(
    rows: list[dict[str, object]],
    reference: dict[str, object],
    path: Path,
    passivity_tolerance: float,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), constrained_layout=True)
    styles = {"ASR-FR": ("o", "tab:blue"), "Cartesian-Li": ("s", "tab:orange")}
    for method in METHODS:
        selected = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: int(row["order"]),
        )
        if not selected:
            continue
        order = [int(row["order"]) for row in selected]
        marker, color = styles[method]
        label = "ASR-FR (paper)" if method == "ASR-FR" else "Cartesian Li"
        axes[0, 0].plot(order, [row["R_total"] for row in selected], marker=marker, color=color, label=label)
        axes[0, 1].plot(order, [row["T_total"] for row in selected], marker=marker, color=color, label=label)
        error = [max(float(row["abs_error_R"]), float(row["abs_error_T"]), 1e-16) for row in selected]
        axes[1, 0].semilogy(order, error, marker=marker, color=color, label=label)
        axes[1, 1].plot(order, [row["elapsed_s"] for row in selected], marker=marker, color=color, label=label)
        invalid = [
            row
            for row in selected
            if float(row["passivity_violation"]) > passivity_tolerance
        ]
        if invalid:
            axes[0, 0].scatter([row["order"] for row in invalid], [row["R_total"] for row in invalid], marker="x", s=70, color="red", zorder=5)
            axes[0, 1].scatter([row["order"] for row in invalid], [row["T_total"] for row in invalid], marker="x", s=70, color="red", zorder=5)

    axes[0, 0].axhline(float(reference["R_total"]), color="0.35", linestyle=":", linewidth=1)
    axes[0, 1].axhline(float(reference["T_total"]), color="0.35", linestyle=":", linewidth=1)
    axes[0, 0].set_title("(a) Total reflection")
    axes[0, 1].set_title("(b) Total transmission")
    axes[1, 0].set_title(
        "(c) max(|ΔR|, |ΔT|) vs "
        f"{reference.get('method') or 'external'} "
        f"N={reference.get('order') if reference.get('order') is not None else '-'}"
    )
    axes[1, 1].set_title("(d) Wall time")
    axes[0, 0].set_ylabel("Power fraction")
    axes[0, 1].set_ylabel("Power fraction")
    axes[1, 0].set_ylabel("Absolute error")
    axes[1, 1].set_ylabel("Time (s)")
    for axis in axes.flat:
        axis.set_xlabel("Fourier order N=M")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    frequency = float(rows[0]["frequency_GHz"])
    fig.suptitle(
        f"Wang-2022 square metal patch, {frequency:g} GHz: "
        "ASR-FR vs Cartesian Li"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--orders", default="1,2,3,4,5,6,8,10,12")
    result.add_argument("--frequency-ghz", type=float, default=6.0)
    result.add_argument("--grid", type=int, default=256)
    result.add_argument("--quadrature-grid", type=int, default=4096)
    result.add_argument("--asr-g", type=float, default=ASR_G)
    result.add_argument("--cascade", choices=("redheffer", "algo2a"), default="redheffer")
    result.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    result.add_argument("--dtype", choices=("complex64", "complex128"), default="complex128")
    result.add_argument("--reference-method", choices=METHODS, default="ASR-FR")
    result.add_argument(
        "--reference-order",
        type=int,
        default=8,
        help=(
            "Passive internal reference order (default: the paper's ASR-FR "
            "N=M=8). If absent from --orders, the highest passive computed "
            "order is used."
        ),
    )
    result.add_argument("--reference-r", type=float)
    result.add_argument("--reference-t", type=float)
    result.add_argument("--passivity-tolerance", type=float, default=5e-5)
    result.add_argument("--strict-passivity", action="store_true")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--no-plot", action="store_true")
    result.add_argument(
        "--output-dir",
        type=Path,
        default=_PACKAGE_ROOT / "results" / "asr_fr_vs_cartesian_li",
    )
    return result


def main(args: argparse.Namespace) -> list[dict[str, object]]:
    orders = parse_orders(args.orders)
    if args.frequency_ghz <= 0 or not math.isfinite(args.frequency_ghz):
        raise ValueError("--frequency-ghz must be finite and positive.")
    if args.grid < max(32, 4 * max(orders) + 4):
        raise ValueError("--grid must be at least max(32, 4*max(order)+4).")
    if args.quadrature_grid < args.grid:
        raise ValueError("--quadrature-grid must be at least --grid.")
    if args.passivity_tolerance < 0:
        raise ValueError("--passivity-tolerance must be nonnegative.")
    if not 0.0 < args.asr_g < 1.0:
        raise ValueError("--asr-g must be strictly between zero and one.")
    if (args.reference_r is None) != (args.reference_t is None):
        raise ValueError(
            "--reference-r and --reference-t must be supplied together."
        )
    if args.reference_order is not None and args.reference_order < 1:
        raise ValueError("--reference-order must be positive.")
    device = select_device(args.device)
    dtype = torch.complex128 if args.dtype == "complex128" else torch.complex64
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "convergence.csv"
    figure_path = args.output_dir / "convergence.png"
    metadata_path = args.output_dir / "convergence_metadata.json"
    if csv_path.exists() and not args.resume:
        raise FileExistsError(f"{csv_path} exists; use --resume or a new --output-dir.")
    rows = read_csv(csv_path) if args.resume else []
    if any(row.get("revision") != REVISION for row in rows):
        raise ValueError("Existing CSV has an incompatible numerical revision.")
    cases = [(method, order) for method in METHODS for order in orders]
    requested_keys = {
        case_key(method, order, args) for method, order in cases
    }
    # A narrowed resumed sweep must not inherit stale rows (and especially an
    # unstable high-order row) from an earlier, wider --orders selection.
    rows = [row for row in rows if row_key(row) in requested_keys]
    completed = {row_key(row) for row in rows}
    for index, (method, order) in enumerate(cases, start=1):
        key = case_key(method, order, args)
        if key in completed:
            print(f"[{index}/{len(cases)}] skip {method} N=M={order}")
            continue
        print(f"[{index}/{len(cases)}] run  {method} N=M={order}")
        row = simulate(
            method,
            order,
            frequency_ghz=args.frequency_ghz,
            grid=args.grid,
            quadrature_grid=args.quadrature_grid,
            asr_g=args.asr_g,
            cascade=args.cascade,
            dtype=dtype,
            device=device,
        )
        violation = float(row["passivity_violation"])
        if violation > args.passivity_tolerance:
            message = (
                f"{method} N={order} is non-passive by {violation:.3e}; "
                "the row is diagnostic and must not be treated as converged."
            )
            if args.strict_passivity:
                raise RuntimeError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=1)
        rows.append(row)
        temporary_rows, _temporary_reference = enrich_convergence(
            rows,
            reference_method=args.reference_method,
            reference_order=args.reference_order,
            reference_r=args.reference_r,
            reference_t=args.reference_t,
            passivity_tolerance=args.passivity_tolerance,
        )
        write_csv(temporary_rows, csv_path)
        rows = temporary_rows
        completed.add(key)
        print(
            f"    R={float(row['R_total']):.8f}, T={float(row['T_total']):.8f}, "
            f"A={float(row['A_total']):.8f}, time={float(row['elapsed_s']):.2f}s"
        )

    rows, reference = enrich_convergence(
        rows,
        reference_method=args.reference_method,
        reference_order=args.reference_order,
        reference_r=args.reference_r,
        reference_t=args.reference_t,
        passivity_tolerance=args.passivity_tolerance,
    )
    write_csv(rows, csv_path)
    metadata = {
        "purpose": "method comparison separate from the Fig. 8 reproduction",
        "paper_doi": "10.1364/OE.459110",
        "frequency_GHz": args.frequency_ghz,
        "wavelength_mm": wavelength_mm(args.frequency_ghz),
        "rayleigh_cutoff_GHz": first_grating_cutoff_ghz(),
        "propagating_orders_air": propagating_order_count_air(
            args.frequency_ghz, max(orders)
        ),
        "geometry": {
            "period_mm": [PERIOD_MM, PERIOD_MM],
            "patch_fill": [PATCH_FILL_X, PATCH_FILL_Y],
            "patch_thickness_mm": PATCH_THICKNESS_MM,
            "metal_epsilon_torcwa": [
                METAL_EPS_TORCWA.real, METAL_EPS_TORCWA.imag
            ],
        },
        "methods": {
            "ASR-FR": "paper ASR map plus mixed Fourier factorization",
            "Cartesian-Li": (
                "identity coordinates; analytic rectangle coefficients; "
                "directional Li inverse/direct factorization; no NV field"
            ),
        },
        "reference": reference,
        "reference_warning": (
            "The default ASR-FR N=M=8 row follows the paper's Fig. 8 setting "
            "but remains an internal comparison reference, not HFSS or an "
            "exact solution. Non-passive rows are never selected as a "
            "reference. Supply --reference-r and --reference-t when "
            "independent data are available."
        ),
        "orders": orders,
        "grid": args.grid,
        "quadrature_grid": args.quadrature_grid,
        "cascade": args.cascade,
        "device": str(device),
        "dtype": args.dtype,
        "revision": REVISION,
        "rows": len(rows),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if not args.no_plot:
        plot(
            rows,
            reference,
            figure_path,
            args.passivity_tolerance,
        )
        print(f"wrote {figure_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {metadata_path}")
    return rows


if __name__ == "__main__":
    main(parser().parse_args())
