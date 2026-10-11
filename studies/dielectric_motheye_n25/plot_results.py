"""Scientific comparison figures with explicit physical and numerical axes."""
from __future__ import annotations

import html
import json
import math
from pathlib import Path

COLORS = {"nv": "#2563eb", "asr": "#d97706"}


def geometry_svg(g, destination):
    elements = ['<rect width="100%" height="100%" fill="white"/>']

    def label(x, y, text, size=14):
        elements.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="#172235">{html.escape(text)}</text>')

    label(30, 30, f"Lossless n={g.index:g} moth-eye; P={g.period_nm:g} nm, H={g.height_nm:g} nm", 21)
    for k, lattice in enumerate(("hexagonal", "square")):
        x0, y0, scale = 45 + k * 445, 85, 250 / (3 * g.period_nm)
        label(x0, y0 - 15, lattice + f"; base fill={g.fill_fraction(lattice):.4f}", 16)
        for j in range(3):
            for i in range(3):
                x = x0 + (i + .5 * (j % 2) + .5) * g.period_nm * scale
                y = y0 + (j + .5) * g.period_nm * scale * (math.sqrt(3) / 2 if k == 0 else 1)
                elements.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{g.base_radius_nm * scale:.2f}" fill="#bfdbfe" stroke="#2563eb"/>')
        sx, sy, s = x0 + 140, 390, 185 / g.height_nm
        depths = [j / 50 for j in range(51)]
        points = [(sx + sign * (g.tip_radius_nm + (g.base_radius_nm - g.tip_radius_nm)
                   * z ** g.profile_power) * s, sy + g.height_nm * z * s)
                  for sign, zs in ((1, depths), (-1, depths[::-1])) for z in zs]
        elements.append('<polygon points="' + " ".join(f"{x:.2f},{y:.2f}" for x, y in points) + '" fill="#bfdbfe" stroke="#2563eb"/>')
        elements.append(f'<rect x="{sx - g.period_nm * s / 2:.2f}" y="{sy + 185}" width="{g.period_nm * s:.2f}" height="25" fill="#bfdbfe"/>')
        label(x0 + 5, sy - 15, "Side view: uncoated tapered dielectric pillar", 12)
        label(x0 + 15, sy + 240, f"tip/base radius={g.tip_radius_nm:g}/{g.base_radius_nm:g} nm", 13)
    label(30, 665, "Air above; same-material semi-infinite substrate. Both methods use identical midpoint z slices.")
    destination.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="950" height="690" viewBox="0 0 950 690">'
                           '<g font-family="Arial, sans-serif">' + "".join(elements) + '</g></svg>', encoding="utf-8")


def make_figures(summary_path: Path):
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install matplotlib to generate comparison PNG/SVG figures. JSON/CSV results are saved.")
        return
    output = summary_path.parent / "figures"
    output.mkdir(parents=True, exist_ok=True)
    groups = summary["groups"]
    index = summary["plan"]["assumptions"]["geometry"]["index"]

    def save(fig, stem):
        fig.tight_layout()
        for extension in ("png", "svg"):
            fig.savefig(output / f"{stem}.{extension}", dpi=180)
        plt.close(fig)

    for lattice in ("hexagonal", "square"):
        pair = {m: groups[f"{lattice}/{m}"] for m in COLORS if f"{lattice}/{m}" in groups}
        if not pair:
            continue
        for axis in ("order", "slices", "grid"):
            records = {m: next((r for r in reversed(report["history"]) if r["axis"] == axis), None)
                       for m, report in pair.items()}
            records = {m: r for m, r in records.items() if r is not None}
            if not records:
                continue
            xlabel = {"order": "Fourier order M", "slices": "Profile slices Nz",
                      "grid": "Q per axis (NV projection / ASR quadrature)"}[axis]
            fig, ax = plt.subplots(2, 1, figsize=(9, 7))
            for m, record in records.items():
                cs, spectra = record["candidates"], record["spectra"]
                ws = [row["wavelength_nm"] for row in spectra[str(cs[0])]]
                indices = sorted(set((0, min(range(len(ws)), key=lambda i: abs(ws[i] - 550)), len(ws) - 1)))
                for i, style in zip(indices, ("-", "--", ":")):
                    for target, metric in zip(ax, ("reflectance", "transmittance")):
                        target.plot(cs, [spectra[str(c)][i][metric] for c in cs], marker="o",
                                    linestyle=style, color=COLORS[m], label=f"{m.upper()} {ws[i]:g} nm")
            for target, label in zip(ax, ("R (fraction)", "T (fraction)")):
                target.set(xlabel=xlabel, ylabel=label)
                target.grid(alpha=.25)
                target.legend(fontsize=8, ncol=2)
            fixed = "; ".join(m.upper() + " " + str({k: v for k, v in r["fixed"].items() if k != axis})
                              for m, r in records.items())
            fig.suptitle(f"n={index:g} {lattice}: absolute powers\nFixed settings: {fixed}", fontsize=10)
            save(fig, f"absolute_{lattice}_{axis}")
            fig, ax = plt.subplots(2, 1, figsize=(8, 6.5))
            for m, record in records.items():
                comps = record["comparisons"]
                ax[0].plot([c["fine"] for c in comps], [100 * c["max_abs_change"] for c in comps],
                           "o-", color=COLORS[m], label=m.upper())
                ax[1].plot(record["candidates"], [sum(row["runtime_seconds"] for row in
                    record["spectra"][str(c)]) / len(record["spectra"][str(c)]) for c in record["candidates"]],
                    "o-", color=COLORS[m], label=m.upper())
            ax[0].axhline(100 * summary["plan"]["tolerance"], color="#dc2626", ls="--", label="tolerance")
            ax[0].set(ylabel="max adjacent |delta R/T/A| (pp)")
            ax[1].set(ylabel="Single solve time (s)")
            for target in ax:
                target.set_xlabel(xlabel)
                target.grid(alpha=.25)
                target.legend()
            fig.suptitle(f"n={index:g} {lattice}: refinement / observed solve cost")
            save(fig, f"changes_and_time_{lattice}_{axis}")

        fig, ax = plt.subplots(2, 1, figsize=(8, 6))
        any_spectrum = False
        for m, report in pair.items():
            rows = report.get("final_spectrum") or report.get("anchor_spectrum", [])
            if not rows:
                continue
            any_spectrum = True
            scope = "spectrum" if report.get("final_spectrum_complete") else "anchors/partial"
            marker = "o" if len(rows) < 4 else None
            ax[0].plot([r["wavelength_nm"] for r in rows], [r["reflectance"] for r in rows],
                       color=COLORS[m], marker=marker, label=f"{m.upper()} {scope}; {report['status']}")
            ax[1].plot([r["wavelength_nm"] for r in rows], [r["energy_residual"] for r in rows],
                       color=COLORS[m], marker=marker, label=m.upper())
        if any_spectrum:
            ax[0].set_ylabel("R (fraction)")
            ax[1].set_ylabel("R + T - 1")
            for target in ax:
                target.set_xlabel("Wavelength (nm)")
                target.grid(alpha=.25)
                target.legend(fontsize=8)
            fig.suptitle(f"n={index:g} {lattice}: reflection and lossless energy balance")
            save(fig, f"spectrum_{lattice}")
        else:
            plt.close(fig)
        audits = {m: r["audit"] for m, r in pair.items() if r.get("audit")}
        if audits:
            fig, ax = plt.subplots(figsize=(8, 4))
            for i, (m, a) in enumerate(audits.items()):
                ax.bar(i - .17, a["forward"]["median_seconds"], width=.32, color=COLORS[m], label=m.upper() + " forward")
                if a.get("gradient"):
                    ax.bar(i + .17, a["gradient"]["median_seconds"], width=.32, color=COLORS[m], alpha=.5, label=m.upper() + " forward+gradient")
            ax.set(xticks=range(len(audits)), xticklabels=[m.upper() for m in audits], ylabel="Warm median time per objective (s)")
            ax.legend(fontsize=8)
            fig.suptitle(f"n={index:g} {lattice}: converged-condition optimization cost")
            save(fig, f"optimization_cost_{lattice}")
            complete = {m: a for m, a in audits.items() if a.get("fd_checks") and a.get("refinement_checks")}
            if complete:
                fig, axes = plt.subplots(2, 1, figsize=(8.5, 6))
                for m, a in complete.items():
                    for parameter, target in zip(("height_nm", "base_radius_nm"), axes):
                        fd = next(r for r in a["fd_checks"] if r["parameter"] == parameter)
                        labels = ["AD", "FD h", "FD h/2"] + ["higher " + r["axis"] for r in a["refinement_checks"]]
                        values = [fd["autograd_per_nm"], fd["fd_h_per_nm"], fd["fd_half_h_per_nm"]]
                        values += [r["per_parameter"][parameter]["refined"] for r in a["refinement_checks"]]
                        target.plot(labels, values, "o-", color=COLORS[m], label=f"{m.upper()}; passed={a['gradient_passed']}")
                        target.set_ylabel("d mean R / d " + parameter + " (1/nm)")
                for target in axes:
                    target.grid(alpha=.25)
                    target.legend(fontsize=8)
                fig.suptitle(f"n={index:g} {lattice}: gradient accuracy / refinement stability")
                save(fig, f"gradients_{lattice}")
    print(f"Figures: {output}")
