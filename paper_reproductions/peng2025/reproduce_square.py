"""Reproduce the square-lattice MI study of Peng and Zhang, Fig. 2.

The paper's stated dimensions are used: p=62 um, R=30 um, r=14 um, and
Ag thickness=1 um.  The selected material interpretation is air in the
annular aperture and PI with epsilon_PI=3.5+0.009j.  Incidence is normal
x/TM over 1--3 THz.  The default solver applies generalized normal-vector Li
factorization after the matched-coordinate transform; analytic concentric NVM
and ASR without the curved-interface normal factorization remain selectable.
The missing Ag Drude constants and MI substrate thickness are
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
import os
NUM_THREADS = os.environ.get("TARCWA_NUM_THREADS", "1")

for _name in (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ[_name] = NUM_THREADS

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__:
    from .common import (
        PaperGeometry,
        SilverDrude,
        Numerics,
        assess_order_convergence,
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
        assess_order_convergence,
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
    parser.add_argument(
        "--asr-g",
        type=float,
        default=3.0e-2,
        help=(
            "Minimum interface slope of this non-separable matched map. "
            "Default 0.03; the paper's 0.001 belongs to a different stepped "
            "separable map."
        ),
    )
    parser.add_argument(
        "--solver",
        choices=("matched-nvm", "nvm", "matched-asr"),
        default="matched-nvm",
        help=(
            "matched-nvm applies the normal-vector Li rule after ASR; nvm is "
            "the independent analytic-Fourier route; matched-asr omits the "
            "curved-interface normal factorization."
        ),
    )
    parser.add_argument(
        "--radial-mapping",
        choices=("auto", "outer", "double"),
        default="auto",
        help=(
            "auto (recommended) selects the monotone double-matched map for "
            "this concentric near-close-packed cell; outer is retained for "
            "diagnostics and may be rejected as ill-conditioned."
        ),
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
        "--output-dir",
        type=Path,
        help="Output directory. By default each solver/study gets its own directory.",
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
    parser.add_argument(
        "--convergence-window",
        type=int,
        default=3,
        help="Number of final orders used by the convergence classifier.",
    )
    parser.add_argument(
        "--convergence-tolerance",
        type=float,
        default=1.0e-2,
        help="Maximum R/T/A span in the final convergence window.",
    )
    parser.add_argument(
        "--relaxed-passivity-tolerance",
        type=float,
        default=2.0e-3,
        help="Largest residual passivity error allowed for provisional status.",
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
    output_dir = args.output_dir
    if output_dir is None:
        solver_tag = args.solver
        if args.solver.startswith("matched-"):
            mapping_tag = "double" if args.radial_mapping == "auto" else args.radial_mapping
            solver_tag = f"{solver_tag}_{mapping_tag}"
        output_dir = _PACKAGE_ROOT / "results" / f"square_{solver_tag}_{args.study}"
    if args.study == "convergence" and len(frequencies) != 1:
        raise ValueError("Convergence study accepts exactly one frequency.")
    if args.solver == "nvm" and args.radial_mapping not in {"auto", "outer"}:
        raise ValueError(
            "--radial-mapping applies only to --solver matched-asr/matched-nvm."
        )
    if args.solver.startswith("matched-") and args.radial_mapping == "outer":
        print(
            "MAP WARNING: the explicitly requested outer-only map can be poorly "
            f"conditioned for R/p=30/62 (requested G={args.asr_g:g}). The solver "
            "rejects a map whose minimum Jacobian is below its safety floor. Prefer "
            "--radial-mapping double (or omit the option)."
        )
    geometry = PaperGeometry(pi_thickness_um=args.pi_thickness_um)
    print(
        "REPRODUCTION ASSUMPTION: the article does not report the Ag Drude "
        "constants or a numerical Fig. 2 PI thickness h2. "
        + (
            "Using a semi-infinite PI output."
            if args.pi_thickness_um is None
            else f"Using the requested finite PI thickness {args.pi_thickness_um:g} um."
        )
    )
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
            + (
                ""
                if result.get("minimum_mapping_jacobian") is None
                else ", min(detJ)="
                f"{float(result['minimum_mapping_jacobian']):.3e}"
            )
        )
        # Preserve completed points in long sweeps.
        write_rows(rows, output_dir / "square_mi.csv")

    nonpassive = [row for row in rows if bool(row["passivity_warning"])]
    if nonpassive:
        print(
            "WARNING: "
            f"{len(nonpassive)}/{len(rows)} rows triggered the passivity diagnostic."
        )
    convergence_diagnostic = args.study == "convergence"
    convergence_assessment = None
    if convergence_diagnostic:
        convergence_assessment = assess_order_convergence(
            rows,
            window=args.convergence_window,
            tolerance=args.convergence_tolerance,
            relaxed_passivity_tolerance=args.relaxed_passivity_tolerance,
        )
        print(
            "convergence status: "
            f"{convergence_assessment['status']} "
            f"(tail orders={convergence_assessment.get('orders', [])})"
        )
    figure_allowed = (
        not nonpassive or args.allow_nonpassive or convergence_diagnostic
    )
    if not args.no_plot and figure_allowed:
        _plot(
            rows,
            output_dir / "square_mi.png",
            args.study,
            args.show,
            plot_all=args.plot_all,
        )
    write_metadata(
        output_dir / "square_mi_metadata.json",
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
            "radial_mapping_resolved": (
                None
                if args.solver == "nvm"
                else "double"
                if args.radial_mapping == "auto"
                else args.radial_mapping
            ),
            "pi_thickness_um": args.pi_thickness_um,
            "cascade": args.cascade,
            "use_symmetry": args.use_symmetry,
            "passivity_warning_count": len(nonpassive),
            "convergence_assessment": convergence_assessment,
            "figure_written": bool(
                not args.no_plot and figure_allowed
            ),
            "paper_comparison": {
                "figure": "Fig. 2(c,d)",
                "reported_asr_nv_spectrum_order": 23,
                "reference_curve_samples_available": False,
                "input_completeness": "underdetermined_from_paper",
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
