"""Compute and plot only the calculated Weiss 2009 Fig. 4(a) spectrum.

This is a calculation-only version of the verification script.

Output:
    - fig4a_spectrum_only.png : calculated R/T/A spectra only
    - fig4a_spectrum_only.csv : numerical calculated results

Important:
    - No digitized/published R/T/A values are plotted or compared.
    - The original Fig. 4(a) frequency knots may be used only as the x-axis
      sampling points. They are NOT used as reference y-values.
    - By default, T and R are zeroth-order power, while A is total physical
      absorption, matching the default definition used in the original
      verification script.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np


PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]

PHYSICS = {
    "period_um": 0.7,
    "radius_um": 0.15,
    "height_um": 0.05,
    "epsilon_background": 1.0,
    "epsilon_input": 1.0,
    "epsilon_output": 1.0,
    "gold_epsilon_inf": 1.0,
    "gold_plasma_rad_s": 1.37e16,
    "gold_damping_rad_s": 0.85e14,
    "eta": 0.97,
    "incidence_degrees": 0.0,
    "profile": "weiss2009",
    "factorization": "Weiss symmetric",
    "conversion": "general-2d-T",
}


def load_frequency_knots() -> np.ndarray:
    """Load only the frequency column used for Fig. 4(a) sampling.

    Published/reference R, T, A values are intentionally ignored.
    """
    path = PACKAGE / "reference" / "fig4a_reference.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Frequency reference file not found: {path}\n"
            "Use --frequencies to specify frequencies explicitly."
        )

    data = np.genfromtxt(path, delimiter=",", names=True)
    if "frequency_THz" not in data.dtype.names:
        raise ValueError("reference CSV does not contain frequency_THz.")

    frequencies = np.asarray(data["frequency_THz"], dtype=float)
    if len(frequencies) < 2 or not np.all(np.isfinite(frequencies)):
        raise ValueError("Invalid frequency sequence.")
    if not np.all(np.diff(frequencies) > 0):
        raise ValueError("Frequency sequence must be strictly increasing.")

    return frequencies


def parse_frequencies(value: str) -> list[float]:
    """Parse 'start:stop:step' or a comma-separated frequency list."""
    if value == "reference":
        return list(map(float, load_frequency_knots()))

    result: list[float] = []
    for token in value.split(","):
        parts = [float(p) for p in token.split(":")]

        if not all(math.isfinite(p) for p in parts):
            raise ValueError("Frequency must be finite.")

        if len(parts) == 1:
            result.append(parts[0])
        elif len(parts) == 3 and parts[2] > 0 and parts[1] >= parts[0]:
            start, stop, step = parts
            n = int((stop - start) / step + 1e-9) + 1
            result.extend(start + i * step for i in range(n))
        else:
            raise ValueError(
                "Use 'reference', comma-separated THz values, "
                "or 'start:stop:step'."
            )

    frequencies = sorted(set(round(f, 10) for f in result))
    if not frequencies or frequencies[0] <= 0:
        raise ValueError("Frequencies must be positive.")

    return frequencies


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_spectrum(output: Path, rows: list[dict[str, float]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frequency = [r["frequency_THz"] for r in rows]
    reflectance = [r["R"] for r in rows]
    transmittance = [r["T"] for r in rows]
    absorption = [r["A"] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(
        frequency,
        reflectance,
        lw=1.8,
        label="Reflectance R (zeroth order)",
    )
    ax.plot(
        frequency,
        transmittance,
        lw=1.8,
        label="Transmittance T (zeroth order)",
    )
    ax.plot(
        frequency,
        absorption,
        lw=1.8,
        label="Absorption A (total)",
    )

    ax.set_xlabel("Frequency (THz)", fontsize=12)
    ax.set_ylabel("Power fraction", fontsize=12)

    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=10)

    ax.set_title(
        "Weiss 2009 Fig. 4(a) | Calculated spectrum only",
        fontsize=13,
    )

    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    frequencies = parse_frequencies(args.frequencies)

    if args.order < 1:
        raise ValueError("--order must be >= 1.")
    if args.grid < max(32, 4 * args.order + 4):
        raise ValueError(
            f"--grid must be >= max(32, 4*order+4) "
            f"= {max(32, 4 * args.order + 4)}."
        )
    if args.threads < 1:
        raise ValueError("--threads must be positive.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import solver only; no paper/reference power data are used.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    import torch
    from paper_reproductions.weiss2009 import reproduce as solver

    torch.set_num_threads(args.threads)
    device = solver.select_device(args.device)

    rows: list[dict[str, float]] = []

    print(
        f"Running N={(2 * args.order + 1) ** 2} harmonics, "
        f"grid={args.grid}, dtype=complex128, device={device}"
    )
    print(f"Frequencies: {len(frequencies)} points")

    for i, frequency_thz in enumerate(frequencies, start=1):
        print(
            f"[{i:3d}/{len(frequencies):3d}] "
            f"f={frequency_thz:.6f} THz",
            flush=True,
        )

        result = solver.simulate_scattering(
            frequency_thz=frequency_thz,
            order=args.order,
            period_um=PHYSICS["period_um"],
            radius_um=PHYSICS["radius_um"],
            height_um=PHYSICS["height_um"],
            epsilon=solver.gold_drude_epsilon(frequency_thz),
            profile="weiss2009",
            grid=args.grid,
            dtype=torch.complex128,
            device=device,
            cascade=args.cascade,
            both_polarizations=True,
        )

        # Use x-polarization. At normal incidence the original script
        # checks x/y symmetry; this output intentionally contains only
        # the requested spectrum quantities.
        if args.paper_power == "zeroth":
            T = float(result["T0_x"])
            R = float(result["R0_x"])
        else:
            T = float(result["T_x"])
            R = float(result["R_x"])

        # A is physical absorption including all outgoing-order effects.
        A = float(result["A_x"])

        for name, value in {"T": T, "R": R, "A": A}.items():
            if not math.isfinite(value):
                raise RuntimeError(
                    f"Nonfinite {name} at f={frequency_thz} THz"
                )

        rows.append(
            {
                "frequency_THz": float(frequency_thz),
                "R": R,
                "T": T,
                "A": A,
            }
        )

        print(
            f"    R={R:.8f}, T={T:.8f}, A={A:.8f}",
            flush=True,
        )

    csv_path = output_dir / "fig4a_spectrum_only.csv"
    png_path = output_dir / "fig4a_spectrum_only.png"

    write_csv(csv_path, rows)
    plot_spectrum(png_path, rows)

    print()
    print("Finished.")
    print(f"CSV : {csv_path}")
    print(f"PNG : {png_path}")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Calculate and plot only the computed R/T/A spectrum "
            "for Weiss 2009 Fig. 4(a)."
        )
    )
    p.add_argument(
        "--frequencies",
        default="reference",
        help=(
            "reference = use only the original Fig. 4(a) frequency knots; "
            "or provide THz as start:stop:step / comma-separated list."
        ),
    )
    p.add_argument(
        "--order",
        type=int,
        default=12,
        help="RCWA order. N harmonics = (2*order+1)^2; default 12 -> 625.",
    )
    p.add_argument(
        "--grid",
        type=int,
        default=512,
        help="Spatial grid size; default 512.",
    )
    p.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=4,
    )
    p.add_argument(
        "--cascade",
        choices=("redheffer", "algo2a"),
        default="redheffer",
    )
    p.add_argument(
        "--paper-power",
        choices=("zeroth", "total"),
        default="zeroth",
        help=(
            "Definition of plotted T/R only. "
            "Default: zeroth-order T0/R0. "
            "A is always total physical absorption."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=PACKAGE / "results" / "fig4a_spectrum_only",
    )
    return p


if __name__ == "__main__":
    run(parser().parse_args())
