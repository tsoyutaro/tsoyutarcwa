"""Scientific figures: values, finite-reference differences and adjacent changes."""
from __future__ import annotations

import math


def plot_results(study, rows, references, plan, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from .adapters import METRICS
    metrics = METRICS[study]
    labels = ("Reflectance R", "Power into Au P_sub", "Relief absorption A_relief") if study == "gold_grating_1d" else (
        "Reflectance R", "Transmission into PMMA T", "Absorption A")
    methods = ["li", *(f"asr_r{ratio}" for ratio in plan["asr_ratios"])]
    method_labels = {"li": "Cartesian Li", **{f"asr_r{r}": f"ASR ({r}N internal)" for r in plan["asr_ratios"]}}
    line_styles = ["-", "--", ":", "-."]
    colors = plt.get_cmap("tab10").colors
    data = {(r["series"], r["order"], r["wavelength_nm"], r["polarization"]): r for r in rows}
    waves, orders = plan["wavelengths_nm"], plan["orders"]
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .2})
    context = (f"Nz={plan['slices']} fixed; N=2M+1; G={plan['G']:g}; {len(waves)} wavelengths"
               f"; TM trace={plan.get('q_projection','direct')}")

    def row(method, order, wave, pol):
        return data.get((method, order, wave, pol), {})

    def mark_invalid(ax, order):
        ax.plot(order, .035, marker="x", color="crimson", markersize=7,
                transform=ax.get_xaxis_transform(), clip_on=False)

    def finish(fig, name, title, legend):
        fig.suptitle(f"{study}\n{title}\n{context}", fontsize=12)
        fig.legend(handles=legend, loc="lower center", ncol=min(5, len(legend)), fontsize=9)
        fig.subplots_adjust(top=.78, bottom=.22, hspace=.65, wspace=.38)
        fig.canvas.draw()
        fig.savefig(output / f"{name}.png", dpi=180)
        fig.savefig(output / f"{name}.svg")
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ip, pol in enumerate(("TE", "TM")):
        for ik, metric in enumerate(metrics):
            ax = axes[ip, ik]
            for iw, wave in enumerate(waves):
                color = colors[iw % len(colors)]
                for im, method in enumerate(methods):
                    values = [row(method, n, wave, pol) for n in orders]
                    ax.plot(orders, [100*r[metric] if r.get("passivity_ok") else math.nan for r in values],
                            line_styles[im % len(line_styles)], color=color, marker="o", markersize=3)
                    for n, r in zip(orders, values):
                        if not r.get("passivity_ok"):
                            mark_invalid(ax, n)
                ref = references[str(wave)].get("polarizations", {}).get(pol, {}).get(metric)
                if ref is not None and math.isfinite(ref):
                    ax.axhline(100*ref, color=color, alpha=.25, lw=.7)
            ax.set(title=f"{pol}: {labels[ik]}", xlabel="Maximum retained Fourier order M", ylabel="Power (%)")
    legends = [Line2D([0], [0], color=colors[i % len(colors)], label=f"{w:g} nm") for i, w in enumerate(waves)]
    legends += [Line2D([0], [0], color="black", ls=line_styles[i % len(line_styles)], label=method_labels[m]) for i, m in enumerate(methods)]
    legends += [Line2D([0], [0], color="gray", alpha=.5, label=f"Horizontal: Li reference M={plan['reference_order']}"),
                Line2D([0], [0], color="crimson", marker="x", ls="", label="Invalid/failed calculation")]
    finish(fig, "values", "Matched geometry and dispersion", legends)

    for adjacent in (False, True):
        fig, axes = plt.subplots(2, 3, figsize=(13, 8))
        for ip, pol in enumerate(("TE", "TM")):
            for ik, metric in enumerate(metrics):
                ax = axes[ip, ik]
                for im, method in enumerate(methods):
                    xs = orders[1:] if adjacent else orders
                    values = []
                    for index, order in enumerate(xs):
                        differences = []
                        for wave in waves:
                            right = row(method, order, wave, pol)
                            if adjacent:
                                left = row(method, orders[index], wave, pol)
                                valid = right.get("passivity_ok") and left.get("passivity_ok")
                                difference = abs(right[metric]-left[metric]) if valid else None
                            else:
                                difference = right.get(f"difference_from_finite_li_reference_{metric}")
                            if difference is not None:
                                differences.append(difference)
                        if len(differences) == len(waves):
                            values.append(max(1e-12, 100*max(differences)))
                        else:
                            values.append(math.nan)
                            mark_invalid(ax, order)
                    ax.plot(xs, values, "o-", color=colors[im % len(colors)], label=method_labels[method], ms=4)
                ax.axhline(100*plan["tolerance"], color="gray", ls="--", lw=.8)
                ax.set_yscale("log")
                ax.set(title=f"{pol}: {labels[ik]}", xlabel="Maximum retained Fourier order M",
                       ylabel="Maximum over wavelengths (pp)")
        title = ("Adjacent-order change (not an absolute error)" if adjacent else
                 f"Difference from finite Li reference M={plan['reference_order']} (not an exact error)")
        legends = [Line2D([0], [0], color=colors[i % len(colors)], marker="o", label=method_labels[m]) for i, m in enumerate(methods)]
        legends += [Line2D([0], [0], color="gray", ls="--", label=f"Tolerance: {100*plan['tolerance']:g} pp"),
                    Line2D([0], [0], color="crimson", marker="x", ls="", label="Invalid at one or more wavelengths")]
        finish(fig, "adjacent_difference" if adjacent else "reference_difference", title, legends)
