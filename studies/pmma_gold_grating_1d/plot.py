"""Plot saved sweeps and export signed changes; no torch or recalculation."""
from __future__ import annotations

import argparse
import csv
import html
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from studies.pmma_gold_grating_1d.common import METRICS, POLS, case_key, export_cases, read_json

COLORS = ("#1765ad", "#d45d12", "#17834b", "#965ab6", "#9b7b09", "#d34268", "#07848c")
LABELS = {"reflectance": "R (%)", "transmittance": "T (%)",
          "absorptance": "A (%)"}


def subtitle(report):
    axis, values, fixed = report["axis"], report["values"], report["fixed_numerics"]
    labels = {"order": "M", "slices": "Nz"}
    held = ", ".join(f"{labels[name]}={fixed[name]}"
                     for name in ("order", "slices") if name != axis)
    return (f"{labels[axis]}={values[0]}..{values[-1]} ({len(values)} points); "
            f"fixed: {held}; grid unused; M_y=0; {report['status']}")


def matplotlib_figure(checkpoint, report, output):
    """Use the standard scientific plotter when available; SVG fallback below."""
    os.environ.setdefault("MPLCONFIGDIR", str(output/".matplotlib"))
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        from matplotlib.ticker import ScalarFormatter
    except ImportError:
        return False
    axis, values = report["axis"], report["values"]
    fixed, waves = report["fixed_numerics"], report["wavelengths_nm"]
    fig, panels = plt.subplots(2, 3, figsize=(12.6, 8.5))
    for row, pol in enumerate(POLS):
        for column, metric in enumerate(METRICS):
            panel = panels[row, column]
            for i, wave in enumerate(waves):
                points = []
                for value in values:
                    case = checkpoint["cases"].get(case_key(dict(fixed, **{axis: value}), wave))
                    points.append(float("nan") if case is None else
                                  100*case["polarizations"][pol][metric])
                panel.plot(values, points, marker="o", markersize=3,
                           color=COLORS[i % len(COLORS)], label=f"{wave:g} nm")
            panel.set_title(f"{pol}: {LABELS[metric]}")
            panel.set_xlabel(axis)
            panel.set_ylabel(LABELS[metric])
            panel.yaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=True))
            panel.grid(True, alpha=0.25)
    handles, labels = panels[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.92),
               ncol=min(7, len(waves)), frameon=False)
    fig.suptitle(f"1D Au-coated PMMA grating / {axis} convergence", fontsize=18)
    fig.text(0.5, 0.94, subtitle(report)+"; separate y scales", ha="center", fontsize=9)
    fig.text(0.5, 0.018, "T enters the lossless PMMA substrate; A=1-R-T is absorption in Au.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.87))
    fig.savefig(output/"convergence.svg")
    fig.savefig(output/"convergence.png", dpi=180)
    plt.close(fig)
    return True


def plot_sweep(checkpoint, report, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    axis, values = report["axis"], report["values"]
    waves, fixed = report["wavelengths_nm"], report["fixed_numerics"]
    title = f"1D Au-coated PMMA grating: {axis} convergence / TE and TM"
    subheading = subtitle(report)+"; separate y scales"
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1260" height="875" viewBox="0 0 1260 875">',
           '<rect width="1260" height="875" fill="#f7f9fc"/>',
           '<g font-family="sans-serif" fill="#233246">',
           f'<text x="30" y="38" font-size="25">{html.escape(title)}</text>',
           f'<text x="30" y="66" font-size="14">{html.escape(subheading)}</text>']
    for i, wave in enumerate(waves):
        color = COLORS[i % len(COLORS)]
        x = 35+i*170
        svg += [f'<line x1="{x}" x2="{x+22}" y1="89" y2="89" stroke="{color}" stroke-width="3"/>',
                f'<text x="{x+29}" y="94" font-size="14">{wave:g} nm</text>']
    for row, pol in enumerate(POLS):
        for col, metric in enumerate(METRICS):
            px, py = 25+col*412, 115+row*364
            x0, y0, width, height = px+65, py+40, 318, 263
            series = []
            for wave in waves:
                points = []
                for value in values:
                    key = case_key(dict(fixed, **{axis: value}), wave)
                    case = checkpoint["cases"].get(key)
                    # Leave a break at missing points instead of inventing values.
                    points.append(None if case is None else
                                  (value, 100*case["polarizations"][pol][metric]))
                series.append(points)
            ys = [point[1] for points in series for point in points if point is not None]
            low, high = (min(ys), max(ys)) if ys else (0, 1)
            margin = max((high-low)*0.08, 0.005)
            low, high = low-margin, high+margin
            xmin, xmax = values[0], values[-1]
            if xmin == xmax:
                xmin, xmax = xmin-1, xmax+1
            def pos(value, y):
                return x0+(value-xmin)/(xmax-xmin)*width, y0+height-(y-low)/(high-low)*height
            svg += [f'<rect x="{px}" y="{py}" width="403" height="349" fill="white" stroke="#d9e2ef"/>',
                    f'<text x="{px+17}" y="{py+26}" font-size="18">{pol} / {LABELS[metric]}</text>']
            for tick in range(5):
                y = low+(high-low)*tick/4
                _, screen_y = pos(xmin, y)
                svg += [f'<line x1="{x0}" x2="{x0+width}" y1="{screen_y:.2f}" y2="{screen_y:.2f}" stroke="#dce5f1"/>',
                        f'<text x="{x0-8}" y="{screen_y+4:.2f}" text-anchor="end" font-size="11">{y:.4g}</text>']
            # Limit labels for long extended sweeps, retaining both endpoints.
            indices = sorted(set([0, len(values)-1, *range(0, len(values), max(1, (len(values)+8)//9))]))
            for index in indices:
                screen_x, _ = pos(values[index], low)
                svg.append(f'<text x="{screen_x:.2f}" y="{y0+height+19}" text-anchor="middle" font-size="11">{values[index]}</text>')
            svg.append(f'<text x="{x0+width/2}" y="{py+339}" text-anchor="middle" font-size="13">{axis}</text>')
            for i, points in enumerate(series):
                segment = []
                for point in points+[None]:
                    if point is None:
                        if segment:
                            coords = " ".join(f"{x:.3f},{y:.3f}" for x, y in segment)
                            svg.append(f'<polyline points="{coords}" fill="none" stroke="{COLORS[i % len(COLORS)]}" stroke-width="2"/>')
                            segment = []
                    else:
                        x, y = pos(*point)
                        segment.append((x, y))
                        svg.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="3" fill="{COLORS[i % len(COLORS)]}"/>')
    svg += ['<text x="30" y="859" font-size="13">T enters lossless PMMA; A=1-R-T is absorption in Au. Values are not a convergence guarantee.</text>', '</g></svg>']
    (output/"convergence.svg").write_text("\n".join(svg), encoding="utf-8")
    matplotlib_figure(checkpoint, report, output)
    fields = ("wavelength_nm", "polarization", "low", "high",
              "dR_pp", "dT_pp", "dA_pp", "max_abs_pp", "pass")
    with (output/"adjacent_changes.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fields)
        writer.writeheader()
        for wave, pairs in report["adjacent_changes"].items():
            for pair in pairs:
                if pair.get("missing"):
                    continue
                for pol in POLS:
                    delta = pair["signed_changes"][pol]
                    writer.writerow(dict(wavelength_nm=wave, polarization=pol,
                        low=pair["low"], high=pair["high"],
                        dR_pp=100*delta[METRICS[0]], dT_pp=100*delta[METRICS[1]],
                        dA_pp=100*delta[METRICS[2]],
                        max_abs_pp=100*max(abs(v) for v in delta.values()),
                        **{"pass": pair["passes_tolerance"]}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--report", type=Path, help="Defaults to report.json beside checkpoint.")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--png", action="store_true", help="Also render SVG to PNG using optional CairoSVG.")
    args = p.parse_args()
    checkpoint = read_json(args.checkpoint)
    report = read_json(args.report or args.checkpoint.parent/"report.json")
    if report.get("signature") and report["signature"] != checkpoint["signature"]:
        p.error("Report and checkpoint signatures differ.")
    output = args.output_dir or args.checkpoint.parent/"plots"
    reports = {report["axis"]: report} if "axis" in report else report.get("initial_sweeps", {})
    if not reports:
        p.error("No saved sweeps in this report; finish or resume the calculation first.")
    for axis, sweep in reports.items():
        folder = output/axis
        plot_sweep(checkpoint, sweep, folder)
        if args.png and not (folder/"convergence.png").exists():
            try:
                import cairosvg
            except ImportError:
                p.error("SVG was saved. --png requires optional CairoSVG; omit --png to use SVG.")
            cairosvg.svg2png(url=str(folder/"convergence.svg"), write_to=str(folder/"convergence.png"))
        print(folder/"convergence.svg")
    export_cases(checkpoint, output)


if __name__ == "__main__":
    main()
