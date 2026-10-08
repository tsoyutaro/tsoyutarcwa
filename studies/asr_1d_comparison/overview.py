"""One four-panel overview of the finite-reference differences in both studies."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path


def plot_overview(folders, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from .adapters import METRICS

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .2})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    exported, legends = [], {}
    for column, study in enumerate(("gold_grating_1d", "pmma_gold_grating_1d")):
        folder = Path(folders[study])
        plan = json.loads((folder / "plan.json").read_text(encoding="utf-8"))
        with (folder / "cases.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        methods = ["li", *(f"asr_r{r}" for r in plan["asr_ratios"])]
        for ip, pol in enumerate(("TE", "TM")):
            ax = axes[ip, column]
            for im, method in enumerate(methods):
                differences = []
                for order in plan["orders"]:
                    group = [r for r in rows if r["series"] == method
                             and int(r["order"]) == order and r["polarization"] == pol]
                    valid = (len(group) == len(plan["wavelengths_nm"]) and
                             all(r["passivity_ok"] == "True" for r in group) and
                             all(r.get("difference_from_finite_li_reference_"+k)
                                 for r in group for k in METRICS[study]))
                    delta = 100*max(float(r["difference_from_finite_li_reference_"+k])
                                    for r in group for k in METRICS[study]) if valid else None
                    differences.append(max(delta, 1e-12) if valid else math.nan)
                    exported.append(dict(study=study, polarization=pol, series=method,
                                         order=order, harmonics=2*order+1,
                                         max_finite_reference_difference_pp=delta,
                                         complete_and_passive=valid))
                    if not valid:
                        ax.plot(order, .035, "x", color="crimson", markersize=7,
                                transform=ax.get_xaxis_transform())
                label = "Cartesian Li" if method == "li" else f"ASR ({method[5:]}N internal)"
                line, = ax.plot(plan["orders"], differences, "o-", lw=1.8, ms=4,
                                color=f"C{im}", label=label)
                legends[label] = line
            ax.axhline(100*plan["tolerance"], color="gray", ls="--", lw=.8)
            ax.set_yscale("log")
            ax.set_xticks(plan["orders"])
            name = "Au grating" if column == 0 else "Au-coated PMMA grating"
            ax.set(title=f"{name} | {pol} | Nz={plan['slices']}",
                   xlabel="Maximum retained Fourier order M",
                   ylabel="Maximum difference (pp)")
    fig.suptitle("Li versus 1D ASR | N=2M+1 | normal incidence\n"
                 f"Difference from finite Li M={plan['reference_order']} reference; not an exact error", fontsize=12)
    legend = list(legends.values()) + [
        Line2D([0], [0], color="gray", ls="--", label=f"Tolerance: {100*plan['tolerance']:g} pp"),
        Line2D([0], [0], color="crimson", marker="x", ls="", label="Invalid at one or more wavelengths")]
    fig.legend(handles=legend, loc="lower center", ncol=2, fontsize=9)
    fig.text(.5, .105, f"Maximum over wavelengths and metrics | Au: R, P_sub, A_relief | PMMA: R, T, A | TM trace={plan.get('q_projection','direct')}",
             ha="center", fontsize=9)
    fig.subplots_adjust(top=.82, bottom=.22, hspace=.62, wspace=.37)
    fig.canvas.draw()
    fig.savefig(output / "convergence_overview.png", dpi=180)
    fig.savefig(output / "convergence_overview.svg")
    plt.close(fig)
    with (output / "overview.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)
