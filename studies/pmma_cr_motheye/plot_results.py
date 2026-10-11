"""Absolute powers and refinement changes; SVG always, PNG when available."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

COLORS = ("#2563eb", "#d97706", "#16a34a", "#9333ea")
LABELS = {"reflectance": "R", "transmittance": "T", "absorptance": "A"}


def text(x, y, value, size=14, anchor="start", color="#182235"):
    return (f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" '
            f'text-anchor="{anchor}" fill="{color}">{html.escape(str(value))}</text>')


def svg(width, height, elements):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/>'
            '<g font-family="Arial, sans-serif">' + "".join(elements) + '</g></svg>')


def rectangle(x, y, w, h, fill):
    return f'<rect x="{x:.3f}" y="{y:.3f}" width="{w:.3f}" height="{h:.3f}" fill="{fill}"/>'


def geometry_svg(g, layers, path: Path):
    left, top = 235, 110
    scale_z = 430 / (g.height_nm + g.cr_top_thickness_nm)
    scale_x = scale_z
    cell_width = g.period_nm * scale_x
    center, cap = g.period_nm / 2, g.cr_top_thickness_nm
    bottom = top + (g.height_nm + cap) * scale_z
    e = [text(45, 35, "PMMA core with Cr coating: two-period cross-section", 21),
         text(45, 63, "Triangular lattice; air above; lossless semi-infinite PMMA substrate", 14),
         rectangle(left, bottom, 2 * cell_width, 65, "#dbeafe")]
    for cell in range(2):
        cell_left = left + cell * cell_width
        for layer in layers:
            y = top + (cap + layer.top_depth_nm) * scale_z
            h = (layer.bottom_depth_nm - layer.top_depth_nm) * scale_z
            outer = layer.outer_radius_nm
            if layer.kind == "valley":
                e.append(rectangle(cell_left, y, cell_width, h, "#71717a"))
            else:
                e.append(rectangle(cell_left + (center - outer) * scale_x, y,
                                   2 * outer * scale_x, h, "#71717a"))
            r = layer.core_radius_nm
            e.append(rectangle(cell_left + (center - r) * scale_x, y,
                               2 * r * scale_x, h, "#dbeafe"))
        if cap > 0:
            r = g.tip_radius_nm + g.cr_side_thickness_nm
            e.append(rectangle(cell_left + (center - r) * scale_x, top,
                               2 * r * scale_x, cap * scale_z, "#71717a"))
    e += [text(680, 145, "air"), text(405, bottom + 40, "PMMA substrate", anchor="middle"),
          text(405, bottom + 99, f"Period = {g.period_nm:g} nm", anchor="middle"),
          text(45, 670, f"Core H={g.height_nm:g} nm; tip/base radii={g.tip_radius_nm:g}/{g.base_radius_nm:g} nm"),
          text(45, 698, f"Cr radial shell / cap / valley = {g.cr_side_thickness_nm:g} / "
               f"{g.cr_top_thickness_nm:g} / {g.cr_valley_thickness_nm:g} nm; Nz={len(layers)}"),
          text(45, 725, "Cr: grey; PMMA: blue. Midpoint slices; no Cr beneath the PMMA core.", 13)]
    path.write_text(svg(850, 750, e), encoding="utf-8")


def panel_limits(series, tolerance=None):
    values = [y for _, _, ys in series for y in ys]
    if tolerance is not None:
        values.append(tolerance)
    lo, hi = min(values), max(values)
    padding = max((hi - lo) * .12, 1e-5 if hi < .01 else 1e-3)
    return lo - padding, hi + padding


def chart_svg(title, subtitle, xlabel, panels, destination: Path, footer=""):
    width, height = 1060, 160 + 255 * len(panels)
    e = [text(40, 35, title, 22), text(40, 61, subtitle, 13)]
    for j, panel in enumerate(panels):
        x0, y0, pw, ph = 108, 102 + 255 * j, 720, 168
        series = panel["series"]
        xs = [x for _, x, _ in series]
        xmin, xmax = min(min(x) for x in xs), max(max(x) for x in xs)
        if xmin == xmax:
            xmin, xmax = xmin - .5, xmax + .5
        ymin, ymax = panel_limits(series, panel.get("tolerance"))
        px = lambda x: x0 + (x - xmin) / (xmax - xmin) * pw
        py = lambda y: y0 + ph - (y - ymin) / (ymax - ymin) * ph
        e.append(text(12, y0 + 20, panel["label"], 17))
        for k in range(5):
            y = ymin + (ymax - ymin) * k / 4
            e.append(f'<path d="M{x0},{py(y):.2f}h{pw}" stroke="#e2e8f0" fill="none"/>')
            e.append(text(x0 - 10, py(y) + 5, f"{y:.5g}", 12, "end"))
        for x in sorted(set(v for group in xs for v in group)):
            if len(set(v for group in xs for v in group)) <= 12 or x in (xmin, xmax):
                e.append(text(px(x), y0 + ph + 22, f"{x:g}", 12, "middle"))
        e.append(f'<path d="M{x0},{y0}v{ph}h{pw}" fill="none" stroke="#64748b"/>')
        for i, (label, x, y) in enumerate(series):
            color = COLORS[i % len(COLORS)]
            points = " ".join(f"{px(a):.2f},{py(b):.2f}" for a, b in zip(x, y))
            e.append(f'<polyline points="{points}" stroke="{color}" fill="none" stroke-width="2"/>')
            if len(x) < 20:
                e.extend(f'<circle cx="{px(a):.2f}" cy="{py(b):.2f}" r="3.5" fill="{color}"/>'
                         for a, b in zip(x, y))
            e.append(text(850, y0 + 25 + i * 24, label, 13, color=color))
        if panel.get("tolerance") is not None:
            y = py(panel["tolerance"])
            e.append(f'<path d="M{x0},{y:.2f}h{pw}" stroke="#dc2626" stroke-dasharray="5 4"/>')
            e.append(text(850, y0 + ph - 6, f"tol={panel['tolerance']:g} pp", 12, color="#dc2626"))
        e.append(text(x0 + pw / 2, y0 + ph + 45, xlabel, 14, "middle"))
    e.append(text(40, height - 20, footer, 12))
    destination.write_text(svg(width, height, e), encoding="utf-8")


def chart_png(title, subtitle, xlabel, panels, destination, plt):
    fig, axes = plt.subplots(len(panels), 1, figsize=(8.5, 2.75 * len(panels) + .7), squeeze=False)
    for panel, ax in zip(panels, axes[:, 0]):
        for i, (label, xs, ys) in enumerate(panel["series"]):
            ax.plot(xs, ys, ".-" if len(xs) < 20 else "-", color=COLORS[i % len(COLORS)], label=label)
        if panel.get("tolerance") is not None:
            ax.axhline(panel["tolerance"], ls="--", color="#dc2626", label="tolerance")
        ax.set(xlabel=xlabel, ylabel=panel["label"])
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle(title + "\n" + subtitle, fontsize=11)
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def make_figures(report_path: Path) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    output = report_path.parent / "figures"
    output.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None
        print("Matplotlib is unavailable: SVG figures are still generated.")
    latest = {row["axis"]: row for row in report["history"]}
    # These are this writer's managed outputs. Do not leave an earlier run's
    # full spectrum or axis scan visible when the current report lacks it.
    unavailable = []
    for axis in ("order", "slices", "grid"):
        if axis not in latest:
            unavailable.extend((f"convergence_absolute_{axis}", f"convergence_delta_{axis}"))
    if not report.get("final_spectrum"):
        unavailable.append("spectrum")
    if not report.get("anchor_spectrum") or report.get("final_spectrum"):
        unavailable.append("anchor_spectrum")
    for stem in unavailable:
        for extension in (".svg", ".png"):
            (output / (stem + extension)).unlink(missing_ok=True)
    xlabel = {"order": "Retained Fourier order M", "slices": "PMMA profile slices Nz",
              "grid": "ASR quadrature grid per axis Q (Q x Q)"}
    for axis, record in latest.items():
        candidates = record["candidates"]
        spectra = record["spectra"]
        anchor = [row["wavelength_nm"] for row in spectra[str(candidates[0])]]
        fixed = {key: value for key, value in record["fixed"].items() if key != axis}
        subtitle = f"cycle={record['cycle']}; fixed={fixed}; status={report['status']}"
        panels = [{"label": LABELS[name] + " (fraction)", "series": [
            (f"{w:g} nm", candidates, [spectra[str(c)][i][name] for c in candidates])
            for i, w in enumerate(anchor)]} for name in LABELS]
        stem = output / f"convergence_absolute_{axis}"
        chart_svg("PMMA/Cr moth-eye: absolute powers", subtitle, xlabel[axis], panels,
                  stem.with_suffix(".svg"), "Raw R/T/A; no clipping; autoscaled y axes.")
        if plt:
            chart_png("PMMA/Cr moth-eye: absolute powers", subtitle, xlabel[axis], panels,
                      stem.with_suffix(".png"), plt)
        comparisons = record["comparisons"]
        panels = [{"label": "Change (pp)", "tolerance": 100 * report["tolerance"], "series": [
            ("max |delta " + LABELS[name] + "|", [c["fine"] for c in comparisons],
             [100 * c["per_metric"][name] for c in comparisons]) for name in LABELS]}]
        stem = output / f"convergence_delta_{axis}"
        chart_svg("PMMA/Cr moth-eye: adjacent refinement changes", subtitle, xlabel[axis], panels,
                  stem.with_suffix(".svg"), "Maximum over anchor wavelengths; 1 pp = 0.01 absolute power.")
        if plt:
            chart_png("PMMA/Cr moth-eye: refinement changes", subtitle, xlabel[axis], panels,
                      stem.with_suffix(".png"), plt)
    final = report.get("final_spectrum", [])
    rows = final if final else report.get("anchor_spectrum", [])
    if rows:
        panels = [{"label": "Power (fraction)", "series": [
            (LABELS[name], [r["wavelength_nm"] for r in rows], [r[name] for r in rows])
            for name in LABELS]}]
        subtitle = f"M/Nz/Q={report['recommendation']}; status={report['status']}"
        if final and not report.get("final_spectrum_complete", False):
            subtitle += " (partial spectrum)"
        stem = output / ("spectrum" if final else "anchor_spectrum")
        chart_svg("PMMA/Cr moth-eye: reflection / transmission / absorption", subtitle,
                  "Wavelength (nm)", panels, stem.with_suffix(".svg"),
                  "A=1-R-T; lossless semi-infinite PMMA substrate. Convergence checked at anchors.")
        if plt:
            chart_png("PMMA/Cr moth-eye: R / T / A", subtitle, "Wavelength (nm)", panels,
                      stem.with_suffix(".png"), plt)
    print(f"Figures: {output}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, default=Path(__file__).parent / "results")
    args = p.parse_args(argv)
    make_figures(args.output_root / "pmma_cr_motheye_convergence.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
