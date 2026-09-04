"""Compare a triangular MI array with its exact orthogonal supercell.

The native cell uses vectors ``a1=(a,0)`` and
``a2=(a/2,sqrt(3)*a/2)`` with one coaxial aperture-particle site.  The
orthogonal representation uses ``A1=(a,0)``, ``A2=(0,sqrt(3)*a)`` and two
sites.  These generate the same infinite structure.  A genuinely square cell
cannot represent the triangular lattice exactly because ``sqrt(3)`` is
irrational; "square-lattice supercell" is therefore interpreted here as the
orthogonal rectangular cell accepted by a Cartesian RCWA implementation.

Examples
--------
Fast equivalence smoke test::

    python paper_reproductions/peng2025/compare_hex_supercell.py --study smoke --device cpu

Frequency comparison with D6 x-source reduction on the native cell::

    python paper_reproductions/peng2025/compare_hex_supercell.py --study spectrum --order 8 \
        --use-d6 --device cuda
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
        C_UM_PER_PS,
        Numerics,
        PaperGeometry,
        SilverDrude,
        maximum_spectrum_difference,
        numpy_column,
        parse_float_list,
        parse_int_list,
        rectangular_supercell_grid_y,
        select_device,
        simulate_matched_primitive,
        simulate_rectangular_supercell,
        simulate_triangular_raster_primitive,
        write_metadata,
        write_rows,
    )
else:
    from common import (
        C_UM_PER_PS,
        Numerics,
        PaperGeometry,
        SilverDrude,
        maximum_spectrum_difference,
        numpy_column,
        parse_float_list,
        parse_int_list,
        rectangular_supercell_grid_y,
        select_device,
        simulate_matched_primitive,
        simulate_rectangular_supercell,
        simulate_triangular_raster_primitive,
        write_metadata,
        write_rows,
    )


_PACKAGE_ROOT = Path(__file__).resolve().parent


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study", choices=("smoke", "spectrum", "convergence"), default="smoke"
    )
    parser.add_argument(
        "--frequencies", help="THz values as comma list or inclusive start:stop:step."
    )
    parser.add_argument("--order", type=int, help="Native Nx=Ny truncation rank.")
    parser.add_argument("--orders", help="Native ranks for convergence.")
    parser.add_argument(
        "--super-order-x",
        type=int,
        help="Override supercell x rank; default equals native rank.",
    )
    parser.add_argument(
        "--super-order-y",
        type=int,
        help="Override supercell y rank; default is twice the native rank.",
    )
    parser.add_argument("--grid", type=int, help="Native grid and supercell x grid.")
    parser.add_argument("--super-grid-y", type=int, help="Override supercell y grid.")
    parser.add_argument("--asr-g", type=float, default=1.0e-3)
    parser.add_argument(
        "--radial-mapping",
        choices=("outer", "double"),
        default="outer",
        help="Matched primitive only: match the outer circle or both radii.",
    )
    parser.add_argument(
        "--cascade", choices=("redheffer", "algo2a"), default="redheffer"
    )
    parser.add_argument(
        "--use-d6",
        action="store_true",
        help="Use the native triangular D6 x-source sector; supercell remains full.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--silver-eps-infinity", type=float, default=1.0)
    parser.add_argument("--silver-plasma-rad-s", type=float, default=1.37e16)
    parser.add_argument("--silver-collision-rad-s", type=float, default=2.73e13)
    parser.add_argument("--comparison-tolerance", type=float, default=5.0e-2)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return exit code 2 when the requested comparison tolerance is missed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_PACKAGE_ROOT / "results" / "hex_supercell",
    )
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser


def _study_values(args: argparse.Namespace) -> tuple[tuple[float, ...], tuple[int, ...], int]:
    if args.study == "smoke":
        return (
            parse_float_list(args.frequencies or "1.95"),
            (args.order or 1,),
            args.grid or 48,
        )
    if args.study == "spectrum":
        return (
            parse_float_list(args.frequencies or "1.0:3.0:0.05"),
            (args.order or 8,),
            args.grid or 192,
        )
    return (
        parse_float_list(args.frequencies or "1.95"),
        parse_int_list(args.orders or "1,2,3,4,5,6,8,10"),
        args.grid or 192,
    )


def _plot(
    matched: list[dict[str, object]],
    primitive_raster: list[dict[str, object]],
    supercell: list[dict[str, object]],
    path: Path,
    study: str,
    show: bool,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    if study == "convergence":
        matched_sorted = sorted(matched, key=lambda row: int(row["order_x"]))
        raster_sorted = sorted(
            primitive_raster, key=lambda row: int(row["order_x"])
        )
        super_sorted = sorted(supercell, key=lambda row: int(row["native_order"]))
        x = [int(row["order_x"]) for row in matched_sorted]
        axes[0].plot(
            x, numpy_column(matched_sorted, "transmittance"), "o-", label="matched native"
        )
        axes[0].plot(
            x,
            numpy_column(raster_sorted, "transmittance"),
            "^-.",
            label="raster native",
        )
        axes[0].plot(
            x,
            numpy_column(super_sorted, "transmittance"),
            "s--",
            label="raster supercell",
        )
        lattice_differences = abs(
            numpy_column(raster_sorted, "transmittance")
            - numpy_column(super_sorted, "transmittance")
        )
        method_differences = abs(
            numpy_column(matched_sorted, "transmittance")
            - numpy_column(raster_sorted, "transmittance")
        )
        axes[1].semilogy(
            x, lattice_differences + 1.0e-16, "o-", label="lattice $|\\Delta T|$"
        )
        axes[1].semilogy(
            x, method_differences + 1.0e-16, "s--", label="method $|\\Delta T|$"
        )
        axes[0].set_xlabel("Native truncation rank")
        axes[1].set_xlabel("Native truncation rank")
    else:
        matched_sorted = sorted(matched, key=lambda row: float(row["frequency_thz"]))
        raster_sorted = sorted(
            primitive_raster, key=lambda row: float(row["frequency_thz"])
        )
        super_sorted = sorted(supercell, key=lambda row: float(row["frequency_thz"]))
        x = numpy_column(matched_sorted, "frequency_thz")
        axes[0].plot(
            x,
            numpy_column(matched_sorted, "transmittance"),
            marker="o",
            label="matched native T",
        )
        axes[0].plot(
            x,
            numpy_column(raster_sorted, "transmittance"),
            "-.",
            marker="^",
            label="raster native T",
        )
        axes[0].plot(
            x,
            numpy_column(super_sorted, "transmittance"),
            "--",
            marker="s",
            label="raster supercell T",
        )
        lattice_differences = abs(
            numpy_column(raster_sorted, "transmittance")
            - numpy_column(super_sorted, "transmittance")
        )
        method_differences = abs(
            numpy_column(matched_sorted, "transmittance")
            - numpy_column(raster_sorted, "transmittance")
        )
        axes[1].semilogy(
            x,
            lattice_differences + 1.0e-16,
            marker="o",
            label="lattice $|\\Delta T|$",
        )
        axes[1].semilogy(
            x,
            method_differences + 1.0e-16,
            "--",
            marker="s",
            label="method $|\\Delta T|$",
        )
        axes[1].semilogy(
            x,
            numpy_column(super_sorted, "folded_transmission_power") + 1.0e-16,
            marker="^",
            label="forbidden folded T",
        )
        axes[0].set_xlabel("Frequency (THz)")
        axes[1].set_xlabel("Frequency (THz)")
    axes[0].set_ylabel("Power fraction")
    axes[0].set_ylim(-0.03, 1.03)
    axes[1].set_ylabel("Absolute power / difference")
    axes[0].set_title("Equivalent triangular-lattice representations")
    axes[1].set_title("Numerical equivalence diagnostics")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220)
    if show:
        plt.show()
    plt.close(figure)


def main() -> int:
    args = _parser().parse_args()
    if args.comparison_tolerance < 0.0:
        raise ValueError("comparison-tolerance must be nonnegative.")
    frequencies, orders, grid = _study_values(args)
    if args.study == "convergence" and len(frequencies) != 1:
        raise ValueError("Convergence study accepts exactly one frequency.")
    geometry = PaperGeometry()
    drude = SilverDrude(
        epsilon_infinity=args.silver_eps_infinity,
        plasma_rad_s=args.silver_plasma_rad_s,
        collision_rad_s=args.silver_collision_rad_s,
    )
    device = select_device(args.device)
    super_grid_y = args.super_grid_y or rectangular_supercell_grid_y(grid)
    matched_rows: list[dict[str, object]] = []
    raster_rows: list[dict[str, object]] = []
    super_rows: list[dict[str, object]] = []
    cases = [(frequency, order) for order in orders for frequency in frequencies]
    for index, (frequency, order) in enumerate(cases, start=1):
        super_order_x = args.super_order_x or order
        super_order_y = args.super_order_y or 2 * order
        print(
            f"[{index}/{len(cases)}] f={frequency:g} THz: "
            f"native N=({order},{order}), super N=({super_order_x},{super_order_y})"
        )
        matched = simulate_matched_primitive(
            frequency,
            lattice_kind="triangular",
            geometry=geometry,
            drude=drude,
            numerics=Numerics(
                order_x=order,
                order_y=order,
                grid_x=grid,
                grid_y=grid,
                asr_g=args.asr_g,
                cascade=args.cascade,
                use_symmetry=args.use_d6,
                shell_radial_mapping=args.radial_mapping,
            ),
            device=device,
        )
        raster = simulate_triangular_raster_primitive(
            frequency,
            geometry=geometry,
            drude=drude,
            numerics=Numerics(
                order_x=order,
                order_y=order,
                grid_x=grid,
                grid_y=grid,
                asr_g=args.asr_g,
                cascade=args.cascade,
                use_symmetry=False,
            ),
            device=device,
        )
        supercell = simulate_rectangular_supercell(
            frequency,
            geometry=geometry,
            drude=drude,
            numerics=Numerics(
                order_x=super_order_x,
                order_y=super_order_y,
                grid_x=grid,
                grid_y=super_grid_y,
                asr_g=args.asr_g,
                cascade=args.cascade,
                use_symmetry=False,
            ),
            device=device,
        )
        matched["study"] = args.study
        matched["native_order"] = order
        raster["study"] = args.study
        raster["native_order"] = order
        supercell["study"] = args.study
        supercell["native_order"] = order
        matched_rows.append(matched)
        raster_rows.append(raster)
        super_rows.append(supercell)
        lattice_difference = abs(
            float(raster["transmittance"]) - float(supercell["transmittance"])
        )
        method_difference = abs(
            float(matched["transmittance"]) - float(raster["transmittance"])
        )
        print(
            f"  matched T={float(matched['transmittance']):.8f}; "
            f"raster native T={float(raster['transmittance']):.8f}; "
            f"raster super T={float(supercell['transmittance']):.8f}; "
            f"lattice |dT|={lattice_difference:.3e}; "
            f"method |dT|={method_difference:.3e}; "
            f"folded T={float(supercell['folded_transmission_power']):.3e}"
        )
        write_rows(
            [*matched_rows, *raster_rows, *super_rows],
            args.output_dir / "hex_vs_supercell.csv",
        )

    lattice_differences = maximum_spectrum_difference(raster_rows, super_rows)
    method_differences = maximum_spectrum_difference(matched_rows, raster_rows)
    passed = lattice_differences["maximum"] <= args.comparison_tolerance
    print(
        f"maximum lattice |d(R,T,A)|={lattice_differences['maximum']:.3e}; "
        f"maximum method |d(R,T,A)|={method_differences['maximum']:.3e}; "
        f"tolerance={args.comparison_tolerance:.3e}; passed={passed}"
    )
    if not args.no_plot:
        _plot(
            matched_rows,
            raster_rows,
            super_rows,
            args.output_dir / "hex_vs_supercell.png",
            args.study,
            args.show,
        )
    write_metadata(
        args.output_dir / "hex_vs_supercell_metadata.json",
        geometry=geometry,
        drude=drude,
        payload={
            "study": args.study,
            "frequencies_thz": list(frequencies),
            "native_orders": list(orders),
            "native_grid": [grid, grid],
            "matched_radial_mapping": args.radial_mapping,
            "supercell_grid": [grid, super_grid_y],
            "supercell_order_rule": {
                "x": args.super_order_x if args.super_order_x else "native_order",
                "y": args.super_order_y if args.super_order_y else "2*native_order",
            },
            "use_native_d6_x_source": args.use_d6,
            "lattice_identity": {
                "native_vectors": [[1.0, 0.0], [0.5, 0.5 * math.sqrt(3.0)]],
                "supercell_vectors": [[1.0, 0.0], [0.0, math.sqrt(3.0)]],
                "sites_per_supercell": 2,
                "hidden_translation": [0.5, 0.5 * math.sqrt(3.0)],
                "forbidden_supercell_orders": "m+n odd",
                "first_folded_air_cutoff_thz": C_UM_PER_PS
                / (math.sqrt(3.0) * geometry.period_um),
            },
            "comparison": {
                "lattice_maximum_absolute_differences": lattice_differences,
                "matched_vs_raster_maximum_absolute_differences": method_differences,
                "tolerance": args.comparison_tolerance,
                "passed": passed,
                "important_note": (
                    "The pass/fail comparison uses standard raster on both the "
                    "triangular primitive and rectangular supercell. The "
                    "matched primitive is reported separately so lattice "
                    "equivalence is not confused with factorization error."
                ),
            },
            "rows_per_model": len(matched_rows),
        },
    )
    if args.strict and not passed:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
