"""Plot the bundled preliminary run, keeping screening and convergence distinct."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=Path(__file__).parent / "results")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    m5 = load(args.results / "maxima_recheck_M5.csv")
    m7 = load(args.results / "maxima_recheck_M7_red.csv")
    m3 = load(args.results / "pilot_M3" / "per_wavelength_maxima.csv")
    qwp = load(args.results / "qwp_pilot_verification" / "convergence.csv")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = {"ellipse": "#dc7a2e", "rectangle": "#2368a2"}
    ax = axes[0, 0]
    x = np.arange(3)
    for offset, shape in [(-.19, "ellipse"), (.19, "rectangle")]:
        values = [abs(float(next(r for r in m5 if r["case"] == f"n{str(n).replace('.', 'p')}_{shape}"
                                and float(r["wavelength_nm"]) == 700)["retardance_deg"]))
                  for n in (1.5, 2.0, 2.5)]
        bars = ax.bar(x + offset, values, .36, label=shape, color=colors[shape])
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    ax.axhline(90, color="black", ls="--", lw=1)
    ax.set(xticks=x, xticklabels=["n=1.5", "n=2.0", "n=2.5"], ylim=(0, 110),
           ylabel="|retardance| (deg)", title="700 nm: screened candidates rechecked at M=5")
    ax.legend(frameon=False)
    ax = axes[0, 1]
    for shape in ("ellipse", "rectangle"):
        case = "n2p5_" + shape
        first = next(r for r in m3 if r["case"] == case and float(r["wavelength_nm"]) == 700
                     and r["transmission_constraint"] == "True")
        second = next(r for r in m5 if r["case"] == case and float(r["wavelength_nm"]) == 700)
        third = next(r for r in m7 if r["case"] == case)
        ax.plot([3, 5, 7], [float(first["maximum_observed_deg"]),
                           abs(float(second["retardance_deg"])),
                           abs(float(third["retardance_deg"]))], "o-", color=colors[shape], label=shape)
    ax.axhline(90, color="black", ls="--", lw=1)
    ax.set(xticks=[3, 5, 7], xlabel="Fourier truncation M", ylabel="|retardance| (deg)",
           title="Fixed geometry #4, n=2.5, 700 nm\nM=3: Q=128; M=5/7: Q=256")
    ax.legend(frameon=False)
    final_m = max(int(r["order"]) for r in qwp)
    final_q = max(int(r["grid"]) for r in qwp)
    final_l = max(int(r["layers"]) for r in qwp)
    final = [r for r in qwp if (int(r["order"]), int(r["grid"]), int(r["layers"]))
             == (final_m, final_q, final_l)]
    first = [r for r in qwp if (int(r["order"]), int(r["grid"]), int(r["layers"])) == (3, 128, 15)]
    if (args.results / "qwp_pilot_dense_M3.csv").exists():
        first = load(args.results / "qwp_pilot_dense_M3.csv")
    dense = (load(args.results / "qwp_pilot_dense_M5.csv")
             if (args.results / "qwp_pilot_dense_M5.csv").exists() else None)
    w = [float(r["wavelength_nm"]) for r in final]
    ax = axes[1, 0]
    ax.axhspan(85, 95, color="#e2e8e0", label="90 +/- 5 deg guide")
    ax.axhline(90, color="black", ls="--", lw=1)
    ax.plot([float(r["wavelength_nm"]) for r in first],
            [abs(float(r["retardance_deg"])) for r in first], "--", color="#8596a3", label="M=3, 15 slices (10 nm sampling)")
    if dense:
        ax.plot([float(r["wavelength_nm"]) for r in dense],
                [abs(float(r["retardance_deg"])) for r in dense], "-", color=colors["rectangle"],
                lw=1.5, label="M=5, 15 slices (10 nm sampling)")
    ax.plot(w, [abs(float(r["retardance_deg"])) for r in final], "o-", color=colors["rectangle"],
            label=f"M={final_m}, {final_l} slices")
    ax.set(xlabel="Wavelength (nm)", ylabel="|retardance| (deg)", ylim=(50, 105),
           title="One fixed rectangle design: short QWP optimization")
    ax.legend(frameon=False, fontsize=9)
    ax = axes[1, 1]
    ax.plot(w, [float(r["Tx00"]) for r in final], "o-", label="Tx, zero order", color="#2368a2")
    ax.plot(w, [float(r["Ty00"]) for r in final], "s-", label="Ty, zero order", color="#dc7a2e")
    if dense:
        dense_w = [float(r["wavelength_nm"]) for r in dense]
        ax.plot(dense_w, [float(r["Tx00"]) for r in dense], color="#2368a2", alpha=.45, lw=1)
        ax.plot(dense_w, [float(r["Ty00"]) for r in dense], color="#dc7a2e", alpha=.45, lw=1)
    ax.axhline(.8, color="black", ls="--", lw=1, label="screening floor")
    ax.set(xlabel="Wavelength (nm)", ylabel="Power transmittance", ylim=(.77, 1.01),
           title=f"Same candidate: M={final_m}, Q={final_q}\nLines: 15 slices, 10 nm; markers: {final_l} slices, 50 nm")
    ax.legend(frameon=False, fontsize=9)
    for ax in axes.ravel():
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
    fig.suptitle("Tapered anisotropic structures: preliminary RCWA results\n"
                 "200 x 200 nm period | 500 nm height | air to n=1.5 substrate", fontsize=14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    fig.savefig(args.output.with_suffix(".svg"))
    plt.close(fig)


if __name__ == "__main__":
    main()
