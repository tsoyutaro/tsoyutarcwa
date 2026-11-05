"""Compute Fig. 7 of Kikuta, Ohira, and Iwata (1997)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from ..vallius2002 import LayerSpec, PreparedStack


HERE = Path(__file__).resolve().parent
PERIOD = 0.4
DEPTH = 1.146
DUTY = 0.7
EPSILON = np.array([1.0, 4.0])
WIDTHS = PERIOD * np.array([1.0 - DUTY, DUTY])
WAVELENGTHS = np.linspace(0.8, 1.2, 201)


def emt_index(wavelength: float, *, tm: bool) -> float:
    """Fundamental Bloch mode of the infinite lamellar medium."""
    k0 = 2 * np.pi / wavelength
    weights = EPSILON[::-1] / EPSILON if tm else np.ones(2)
    fill = WIDTHS / PERIOD
    static_index = (
        1 / np.sqrt(fill @ (1 / EPSILON)) if tm else np.sqrt(fill @ EPSILON)
    )

    def residual(index: float) -> float:
        transverse_k_squared = k0**2 * (EPSILON - index**2)
        layer_phase = np.sqrt(transverse_k_squared + 0j) * WIDTHS
        # The Bloch trace at zero transverse Bloch momentum is 2.
        # For TM, continuity of H and (1/eps) dH/dx gives the epsilon ratios.
        # The sinc form keeps the expression finite at zero transverse k.
        trace_half = np.prod(np.cos(layer_phase)) - 0.5 * (
            weights @ transverse_k_squared
        ) * np.prod(WIDTHS * np.sinc(layer_phase / np.pi))
        return float(trace_half.real - 1)

    return brentq(residual, static_index, np.sqrt(EPSILON.max()))


def calculate(harmonics: int, device: str, backend: str) -> dict[str, np.ndarray]:
    layer = LayerSpec(
        thickness=DEPTH,
        breaks=(0.0, WIDTHS[0], PERIOD),
        epsilon=EPSILON,
        period=PERIOD,
    )
    spectra = {}
    for polarization in ("TE", "TM"):
        stack = PreparedStack(
            [layer], polarization=polarization, method="fmm", harmonics=harmonics,
            angle=0.0, epsilon_in=EPSILON[0], epsilon_out=EPSILON[1],
            diagnostics=False, device=device, backend=backend,
        )
        if polarization == "TE":
            print(f"RCWA: {stack.backend} / {stack.device} / {harmonics} harmonics")
        spectra[polarization] = stack.spectrum(WAVELENGTHS)

    te, tm = spectra["TE"], spectra["TM"]
    zeroth = int(np.flatnonzero(te["orders"] == 0)[0])
    # The H amplitude for forward TM in air and in the substrate converts
    # to the E transmission amplitude by a positive real impedance ratio.
    phase_difference = np.angle(te["t"][:, zeroth] * tm["t"][:, zeroth].conj())
    n_te = np.array([emt_index(wavelength, tm=False) for wavelength in WAVELENGTHS])
    n_tm = np.array([emt_index(wavelength, tm=True) for wavelength in WAVELENGTHS])
    return {
        "lambda_over_lambda0": WAVELENGTHS,
        "evaluation_lambda_over_lambda0": te["evaluation_wavelength"],
        "retardance_rcwa_deg": np.rad2deg(np.unwrap(phase_difference)),
        "retardance_emt_deg": 360 * DEPTH * (n_te - n_tm) / WAVELENGTHS,
        "retardance_quartz_approx_deg": 90 / WAVELENGTHS,
        "n_eff_te": n_te,
        "n_eff_tm": n_tm,
        "T0_te": te["T0"],
        "T0_tm": tm["T0"],
        "R_te": te["R"],
        "R_tm": tm["R"],
        # This empirical mapping follows the plotted markers in Fig. 7(b).
        "fig7b_te_label": tm["T0"]**2,
        "fig7b_tm_label": te["T0"]**2,
        "one_minus_R_minus_T_te": te["A"],
        "one_minus_R_minus_T_tm": tm["A"],
    }


def save_plot(path: Path, data: dict[str, np.ndarray]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = data["lambda_over_lambda0"]
    fig, (phase, power) = plt.subplots(
        2, 1, figsize=(6.2, 7.2), sharex=True, layout="constrained",
    )
    fig.suptitle("Kikuta et al. (1997)  |  Fig. 7 reproduction")

    phase.plot(x, data["retardance_rcwa_deg"], color="#21618c", marker="s",
               mfc="white", ms=4, markevery=10, label="Rigorous (RCWA)")
    phase.plot(x, data["retardance_emt_deg"], color="#9a4019", label="EMT")
    phase.plot(x, data["retardance_quartz_approx_deg"], "--", color="0.4",
               label="0th-order quartz (approx.)")
    phase.set(title="(a) Phase retardation", ylabel="Phase retardation (°)",
              ylim=(80, 100), yticks=np.arange(80, 101, 5))
    phase.legend(loc="upper right", frameon=False)

    power.plot(x, data["fig7b_te_label"], color="#21618c", marker="o", mfc="white",
               ms=4, markevery=10, label=r"Paper label TE: $T_{0,\mathrm{TM}}^2$")
    power.plot(x, data["fig7b_tm_label"], color="#9a4019", marker="o", ms=4,
               markevery=10, label=r"Paper label TM: $T_{0,\mathrm{TE}}^2$")
    power.set(title="(b) Transmittance curves as plotted in the paper",
              xlabel=r"Normalized wavelength $\lambda/\lambda_0$",
              ylabel="Transmittance (paper scale)", ylim=(0, 1.02),
              yticks=np.linspace(0, 1, 6), xlim=(0.8, 1.2),
              xticks=np.linspace(0.8, 1.2, 5))
    power.legend(loc="lower center", frameon=False)
    for ax in (phase, power):
        ax.grid(axis="y", color="0.88", linewidth=0.7)
        ax.set_axisbelow(True)
    phase.tick_params(labelbottom=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def report(data: dict[str, np.ndarray]) -> None:
    reference = np.genfromtxt(HERE / "fig7_markers.csv", delimiter=",", names=True)
    x = data["lambda_over_lambda0"]
    for name, unit in (
        ("retardance_rcwa_deg", " deg"),
        ("fig7b_te_label", ""),
        ("fig7b_tm_label", ""),
    ):
        difference = np.interp(reference["lambda_over_lambda0"], x, data[name]) - reference[name]
        print(f"{name}: max |calculated - digitized| = {np.max(np.abs(difference)):.4g}{unit}")
    for pol in ("te", "tm"):
        residual = np.max(np.abs(data[f"one_minus_R_minus_T_{pol}"]))
        print(f"{pol.upper()}: max |1 - R - T| = {residual:.3g}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:<index>")
    parser.add_argument("--backend", choices=("auto", "scipy", "torch"), default="auto")
    parser.add_argument("--harmonics", type=int, default=61,
                        help="odd number of retained modes (2M+1, default: 61)")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    if args.harmonics < 3 or args.harmonics % 2 == 0:
        parser.error("--harmonics must be an odd integer >= 3")
    try:
        data = calculate(args.harmonics, args.device, args.backend)
    except (RuntimeError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    report(data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output_dir / "fig7.csv", np.column_stack(list(data.values())),
               delimiter=",", header=",".join(data), comments="", fmt="%.12g")
    save_plot(args.output_dir / "fig7.png", data)
    print(f"Saved fig7.csv and fig7.png in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
