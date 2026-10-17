"""Plot NV convergence along M, normal-field grid Q, and z slice count."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    args = p.parse_args()
    with (args.input / "nv_convergence.csv").open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    meta = json.loads((args.input / "run_metadata.json").read_text(encoding="utf-8"))["arguments"]
    wls = sorted({float(r["wavelength_nm"]) for r in rows})
    colors = {w: plt.get_cmap("viridis")(.15 + .7*i / max(1, len(wls)-1)) for i,w in enumerate(wls)}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    panels = [("orders", "order", "Fourier truncation M", axes[0,0]),
              ("grids", "grid", "Normal-field grid Q x Q", axes[0,1]),
              ("layers", "layers", "Number of z slices", axes[1,0]),
              ("orders_at_final_grid_layers", "order", "Fourier truncation M", axes[1,1])]
    for series, variable, xlabel, ax in panels:
        for shape in meta["shapes"]:
            for w in wls:
                selected = sorted([r for r in rows if r["series"] == series and r["shape"] == shape
                                   and float(r["wavelength_nm"]) == w], key=lambda r:int(r[variable]))
                ax.plot([int(r[variable]) for r in selected],
                        [float(r["retardance_deg"]) for r in selected],
                        marker="o" if shape == "ellipse" else "s",
                        ls="--" if shape == "ellipse" else "-", color=colors[w],
                        label=f"{shape}, {w:g} nm")
        titles = {
            "orders": f"Order sweep: Q={min(meta['grids'])}, z slices={min(meta['layers'])}",
            "grids": f"Grid sweep: M={max(meta['orders'])}, z slices={min(meta['layers'])}",
            "layers": f"Slice sweep: M={max(meta['orders'])}, Q={max(meta['grids'])}",
            "orders_at_final_grid_layers": f"Order check: Q={max(meta['grids'])}, z slices={max(meta['layers'])}"}
        ax.set(xlabel=xlabel, ylabel="Signed retardance (deg)", title=titles[series])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"NV-only convergence | n={meta['index']:g}, period={meta['period']:g} nm, height={meta['height']:g} nm\n"
                 "Exact material Fourier coefficients; FFT grid is for the normal projector", fontsize=13)
    fig.savefig(args.input / "nv_convergence.png", dpi=160)
    fig.savefig(args.input / "nv_convergence.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
