"""Reproduce the square-lattice MI study of Peng and Zhang, Fig. 2.

The paper's stated dimensions are used: p=62 um, R=30 um, r=14 um, and
Ag thickness=1 um.  The selected material interpretation is air in the
annular aperture and PI with epsilon_PI=3.5+0.009j.  Incidence is normal
x/TM over 1--3 THz.  An analytic concentric NVM solver is the default
independent convergence route; the project's matched-ASR solver remains
selectable.  The missing Ag Drude constants and MI substrate thickness are
documented in the generated metadata rather than silently presented as paper
values.

Examples
--------
Fast installation and API check::

    python paper_reproductions/peng2025/reproduce_square.py --study smoke --device cpu

Paper-band calculation using the reported ASR-NV truncation rank 23::

    python paper_reproductions/peng2025/reproduce_square.py --study spectrum --device cuda

Convergence at the paper's 1.95 THz test frequency::

    python paper_reproductions/peng2025/reproduce_square.py --study convergence --device cuda
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__:
    from .common import (
        PaperGeometry,
        SilverDrude,
        Numerics,
        numpy_column,
        parse_float_list,
        parse_int_list,
        select_device,
        simulate_matched_primitive,
        write_metadata,
        write_rows,
    )
else:
    from common import (
        PaperGeometry,
        SilverDrude,
        Numerics,
        numpy_column,
        parse_float_list,
        parse_int_list,
        select_device,
        simulate_matched_primitive,
        write_metadata,
        write_rows,
    )


_PACKAGE_ROOT = Path(__file__).resolve().parent


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study",
        choices=("smoke", "spectrum", "convergence"),
        default="smoke",
        help="smoke is intentionally cheap; spectrum/convergence use paper-scale defaults.",
    )
    parser.add_argument(
        "--frequencies",
        help="THz values as comma list or inclusive start:stop:step.",
    )
    parser.add_argument("--order", type=int, help="Nx=Ny for spectrum/smoke.")
    parser.add_argument(
        "--orders",
        help="Comma-separated Nx=Ny values for convergence.",
    )
    parser.add_argument("--grid", type=int, help="Matched-ASR quadrature grid per axis.")
    parser.add_argument("--asr-g", type=float, default=1.0e-3)
    parser.add_argument(
        "--solver",
        choices=("nvm", "matched-asr"),
        default="nvm",
        help=(
            "nvm uses analytic Fourier coefficients for both concentric "
            "interfaces; matched-asr retains the coordinate-mapped backend."
        ),
    )
    parser.add_argument(
        "--radial-mapping",
        choices=("outer", "double"),
        default="outer",
        help="Match only the outer circle or both core-shell radii.",
    )
    parser.add_argument(
        "--pi-thickness-um",
        type=float,
        help=(
            "Finite PI thickness followed by air. Omit for a semi-infinite PI "
            "output; Fig. 2 does not state h2 numerically."
        ),
    )
    parser.add_argument(
        "--cascade", choices=("redheffer", "algo2a"), default="redheffer"
    )
    parser.add_argument(
        "--use-symmetry",
        action="store_true",
        help="Use the normal-incidence x-source C2v sector (not used in the paper).",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--silver-eps-infinity", type=float, default=1.0)
    parser.add_argument("--silver-plasma-rad-s", type=float, default=1.37e16)
    parser.add_argument("--silver-collision-rad-s", type=float, default=2.73e13)
    parser.add_argument(
        "--output-dir", type=Path, default=_PACKAGE_ROOT / "results" / "square"
    )
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--plot-all",
        action="store_true",
        help="Also draw reflection and absorption; the paper panels show T only.",
    )
    parser.add_argument(
        "--allow-nonpassive",
        action="store_true",
        help="Write a diagnostic plot even if a finite-order result violates passivity.",
    )
    parser.add_argument("--show", action="store_true")
    return parser


def _study_values(args: argparse.Namespace) -> tuple[tuple[float, ...], tuple[int, ...], int]:
    if args.study == "smoke":
        frequencies = parse_float_list(args.frequencies or "1.95")
        orders = (args.order or 1,)
        grid = args.grid or 48
    elif args.study == "spectrum":
        frequencies = parse_float_list(args.frequencies or "1.0:3.0:0.025")
        # Fig. 2(d) uses ASR-NV Nx=Ny=23.
        orders = (args.order or 23,)
        grid = args.grid or 256
    else:
        frequencies = parse_float_list(args.frequencies or "1.95")
        orders = parse_int_list(
            args.orders or "1,3,5,7,9,11,13,15,17,19,21,23"
        )
        grid = args.grid or 256
    if any(order < 1 for order in orders):
        raise ValueError("Every order must be positive.")
    return frequencies, orders, grid


def _plot(
    rows: list[dict[str, object]],
    path: Path,
    study: str,
    show: bool,
    *,
    plot_all: bool,
) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    if study == "convergence":
        selected = sorted(rows, key=lambda row: int(row["order_x"]))
        x = [int(row["order_x"]) for row in selected]
        transmission = numpy_column(selected, "transmittance")
        axis.plot(
            x,
            transmission,
            "o-",
            label="Transmission",
        )
        warning_x = [
            int(row["order_x"])
            for row in selected
            if bool(row["passivity_warning"])
        ]
        warning_t = [
            float(row["transmittance"])
            for row in selected
            if bool(row["passivity_warning"])
        ]
        if warning_x:
            axis.scatter(
                warning_x,
                warning_t,
                marker="x",
                s=80,
                linewidths=2,
                color="red",
                label="Nonpassive truncation",
                zorder=4,
            )
        if plot_all:
            axis.plot(
                x,
                numpy_column(selected, "reflectance"),
                "s-",
                label="Reflection",
            )
        axis.set_xlabel("Truncation rank $N_x=N_y$")
        axis.set_title("MI square lattice at 1.95 THz")
    else:
        selected = sorted(rows, key=lambda row: float(row["frequency_thz"]))
        x = numpy_column(selected, "frequency_thz")
        axis.plot(
            x, numpy_column(selected, "transmittance"), marker="o", label="Transmission"
        )
        if plot_all:
            axis.plot(
                x, numpy_column(selected, "reflectance"), marker="s", label="Reflection"
            )
            axis.plot(
                x, numpy_column(selected, "absorptance"), marker="^", label="Absorption"
            )
        axis.set_xlabel("Frequency (THz)")
        axis.set_title("Peng--Zhang MI coaxial cell, square lattice")
    axis.set_ylabel("Power fraction")
    axis.set_ylim(-0.03, 1.03)
    axis.grid(True, alpha=0.3)
    axis.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220)
    if show:
        plt.show()
    plt.close(figure)


def main() -> int:
    args = _parser().parse_args()
    frequencies, orders, grid = _study_values(args)
    if args.study == "convergence" and len(frequencies) != 1:
        raise ValueError("Convergence study accepts exactly one frequency.")
    if args.solver == "nvm" and args.radial_mapping != "outer":
        raise ValueError("--radial-mapping applies only to --solver matched-asr.")
    geometry = PaperGeometry(pi_thickness_um=args.pi_thickness_um)
    drude = SilverDrude(
        epsilon_infinity=args.silver_eps_infinity,
        plasma_rad_s=args.silver_plasma_rad_s,
        collision_rad_s=args.silver_collision_rad_s,
    )
    device = select_device(args.device)
    rows: list[dict[str, object]] = []
    cases = [(frequency, order) for order in orders for frequency in frequencies]
    for index, (frequency, order) in enumerate(cases, start=1):
        modal_dimension = 2 * (2 * order + 1) ** 2
        print(
            f"[{index}/{len(cases)}] square f={frequency:g} THz, "
            f"N={order}, full modal dimension={modal_dimension}"
        )
        result = simulate_matched_primitive(
            frequency,
            lattice_kind="square",
            geometry=geometry,
            drude=drude,
            numerics=Numerics(
                order_x=order,
                order_y=order,
                grid_x=grid,
                grid_y=grid,
                asr_g=args.asr_g,
                cascade=args.cascade,
                use_symmetry=args.use_symmetry,
                shell_radial_mapping=args.radial_mapping,
                solver=args.solver,
            ),
            device=device,
        )
        result["study"] = args.study
        rows.append(result)
        print(
            "  "
            f"R={float(result['reflectance']):.8f}, "
            f"T={float(result['transmittance']):.8f}, "
            f"A={float(result['absorptance']):.8f}, "
            f"time={float(result['runtime_seconds']):.2f} s"
        )
        # Preserve completed points in long sweeps.
        write_rows(rows, args.output_dir / "square_mi.csv")

    nonpassive = [row for row in rows if bool(row["passivity_warning"])]
    if nonpassive:
        print(
            "WARNING: "
            f"{len(nonpassive)}/{len(rows)} rows triggered the passivity diagnostic."
        )
    convergence_diagnostic = args.study == "convergence"
    figure_allowed = (
        not nonpassive or args.allow_nonpassive or convergence_diagnostic
    )
    if not args.no_plot and figure_allowed:
        _plot(
            rows,
            args.output_dir / "square_mi.png",
            args.study,
            args.show,
            plot_all=args.plot_all,
        )
    write_metadata(
        args.output_dir / "square_mi_metadata.json",
        geometry=geometry,
        drude=drude,
        payload={
            "study": args.study,
            "frequency_thz": list(frequencies),
            "orders": list(orders),
            "grid": [grid, grid],
            "asr_g": args.asr_g,
            "solver": args.solver,
            "radial_mapping": args.radial_mapping,
            "pi_thickness_um": args.pi_thickness_um,
            "cascade": args.cascade,
            "use_symmetry": args.use_symmetry,
            "passivity_warning_count": len(nonpassive),
            "figure_written": bool(
                not args.no_plot and figure_allowed
            ),
            "paper_comparison": {
                "figure": "Fig. 2(c,d)",
                "reported_asr_nv_spectrum_order": 23,
                "reference_curve_samples_available": False,
                "claim": (
                    "Compare curve shape and convergence qualitatively unless "
                    "the authors' exact Drude constants/reference samples are supplied."
                ),
            },
            "rows": len(rows),
        },
    )
    if not all(
        math.isfinite(float(row[name]))
        for row in rows
        for name in ("reflectance", "transmittance", "absorptance")
    ):
        raise RuntimeError("A power observable is NaN or infinity.")
    if nonpassive and not args.allow_nonpassive and not convergence_diagnostic:
        raise RuntimeError(
            "The requested truncation produced nonpassive power. CSV and metadata "
            "were retained, but no figure was generated. Increase --order/--grid "
            "or change the solver. Pass --allow-nonpassive only for debugging."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
