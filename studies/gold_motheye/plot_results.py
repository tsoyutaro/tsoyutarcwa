#!/usr/bin/env python3
"""Make dependency-free SVG figures from gold_motheye convergence/spectrum output."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def text(x: float, y: float, value: object, *, size=16, fill="#182235",
         anchor="start", weight="normal") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
            f'fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">'
            f'{esc(value)}</text>')


def line(x1: float, y1: float, x2: float, y2: float, *, color="#cbd5e1",
         width=1, dash="") -> str:
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" '
            f'y2="{y2:.2f}" stroke="{color}" stroke-width="{width}"{extra}/>')


def circle(x: float, y: float, *, radius=4, fill="#2563eb") -> str:
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="{fill}"/>'


def svg_document(width: int, height: int, elements: list[str]) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'width="{width}" height="{height}" role="img">\n'
            '<rect width="100%" height="100%" fill="#ffffff"/>\n'
            '<g font-family="Arial, Helvetica, sans-serif">\n'
            + "\n".join(elements) + "\n</g>\n</svg>\n")


def plot_convergence(report: dict, destination: Path) -> None:
    history = report.get("history", [])
    if not history:
        raise ValueError("Convergence report has no history")
    tolerance_pp = 100 * float(report["candidate_axes"]["tolerance"])
    colors = ["#2563eb", "#7c3aed", "#dc2626", "#0891b2", "#a16207"]
    axes = [("slices", "Height slices Nz"), ("order", "Fourier order M"),
            ("grid", "ASR grid")]
    width, height = 1320, 790
    out = [text(48, 55, "Gold moth-eye | convergence", size=30, weight="bold"),
           text(48, 86, "Maximum adjacent change over all anchor wavelengths and convergence observables", size=17),
           text(48, 114, f"Criterion: two consecutive steps <= {tolerance_pp:g} percentage points", size=16)]
    status = report.get("status", "unknown")
    status_color = "#15803d" if status == "converged" else "#b91c1c"
    out.append(text(1270, 56, status.upper().replace("_", " "), size=17,
                    fill=status_color, anchor="end", weight="bold"))
    cycles = sorted({int(item["cycle"]) for item in history})
    for i, cycle in enumerate(cycles):
        color = colors[i % len(colors)]
        lx = 48 + i * 125
        out.extend([line(lx, 144, lx + 30, 144, color=color, width=3),
                    circle(lx + 15, 144, fill=color),
                    text(lx + 39, 149, f"Cycle {cycle}", size=14)])

    top, bottom = 210, 661
    log_min, log_max = -3.0, 2.0  # percentage points

    def y_for(value_pp: float) -> float:
        bounded = max(10 ** log_min, min(10 ** log_max, value_pp))
        return bottom - (math.log10(bounded) - log_min) / (log_max - log_min) * (bottom - top)

    for panel, (axis, axis_title) in enumerate(axes):
        left = 68 + panel * 425
        right = left + 344
        records = [item for item in history if item["axis"] == axis]
        latest = records[-1]
        candidates = [int(item["fine"]) for item in latest["comparisons"]]
        out.append(text(left, 191, axis_title, size=20, weight="bold"))
        for exponent in range(-3, 3):
            val = 10.0 ** exponent
            y = y_for(val)
            out.append(line(left, y, right, y, color="#e2e8f0"))
            out.append(text(left - 9, y + 5, f"{val:g}", size=12, anchor="end", fill="#475569"))
        out.extend([line(left, top, left, bottom, color="#64748b"),
                    line(left, bottom, right, bottom, color="#64748b")])
        y_tol = y_for(tolerance_pp)
        out.append(line(left, y_tol, right, y_tol, color="#111827", width=2, dash="7 5"))
        if panel == 2:
            out.append(text(right - 2, y_tol - 8, "tolerance", size=13, anchor="end"))
        for index, candidate in enumerate(candidates):
            x = left + (index + 0.5) * (right - left) / len(candidates)
            out.extend([line(x, bottom, x, bottom + 5, color="#64748b"),
                        text(x, bottom + 24, candidate, size=13, anchor="middle")])
        for record in records:
            color = colors[(int(record["cycle"]) - 1) % len(colors)]
            points = []
            comparison_by_fine = {int(c["fine"]): c for c in record["comparisons"]}
            for index, candidate in enumerate(candidates):
                if candidate not in comparison_by_fine:
                    continue
                x = left + (index + 0.5) * (right - left) / len(candidates)
                value_pp = 100 * float(comparison_by_fine[candidate]["max_abs_change"])
                points.append((x, y_for(value_pp)))
            if len(points) > 1:
                out.append('<polyline points="' + " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
                           + f'" fill="none" stroke="{color}" stroke-width="2.5"/>')
            out.extend(circle(x, y, fill=color) for x, y in points)
        judgement = "PASS" if latest["converged"] else "NOT CONVERGED"
        fill = "#15803d" if latest["converged"] else "#b91c1c"
        out.append(text(left, 720, f"Latest cycle: {judgement}", size=15, fill=fill, weight="bold"))
    out.append(text(48, 767, "Vertical axis: percentage points (log scale); each point compares the labeled value with the previous candidate.",
                    size=14, fill="#475569"))
    destination.write_text(svg_document(width, height, out), encoding="utf-8")


def read_spectrum(path: Path) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No spectrum rows: {path}")
    numeric = []
    for row in rows:
        converted = {}
        for key in ("wavelength_nm", "order", "slices", "grid",
                    "reflectance", "transmittance_far",
                    "absorptance_total", "motheye_absorptance", "substrate_absorptance"):
            raw = row.get(key, "")
            if raw not in (None, ""):
                value = float(raw)
                if not math.isfinite(value):
                    raise ValueError(f"Nonfinite value in {key}: {path}")
                converted[key] = value
        numeric.append(converted)
    numeric.sort(key=lambda item: item["wavelength_nm"])
    return numeric


def plot_spectrum(rows: list[dict[str, float]], report: dict, source: Path,
                  destination: Path) -> None:
    width, height = 1220, 740
    left, right, top, bottom = 105, 1135, 205, 630
    xs = [r["wavelength_nm"] for r in rows]
    xmin, xmax = min(xs), max(xs)
    margin = max(10.0, (xmax - xmin) * 0.04)
    if xmin == xmax:
        margin = 25.0
    xmin -= margin
    xmax += margin

    def x_for(wavelength: float) -> float:
        return left + (wavelength - xmin) / (xmax - xmin) * (right - left)

    def y_for(fraction: float) -> float:
        return bottom - fraction * (bottom - top)

    status = report.get("status", "unknown")
    sparse = len(rows) <= 3
    out = [text(55, 56, "Gold moth-eye | optical response", size=30, weight="bold"),
           text(55, 87, f"Source: {source.name}  |  {len(rows)} wavelengths  |  M={rows[0].get('order', report.get('recommendation', {}).get('order', '?'))}, "
                        f"Nz={report.get('recommendation', {}).get('slices', '?')}, grid={report.get('recommendation', {}).get('grid', '?')}",
                size=16),
           text(55, 116, "Anchor wavelengths only: points are not interpolated" if sparse
                else "Calculated wavelengths connected by lines", size=16,
                fill="#b45309" if sparse else "#475569")]
    if status != "converged":
        out.append(text(55, 144, "PROVISIONAL: numerical convergence criterion has not been met",
                        size=16, fill="#b91c1c", weight="bold"))

    series = [("reflectance", "R", "#1d4ed8"),
              ("transmittance_far", "T far", "#64748b"),
              ("absorptance_total", "A total", "#dc2626"),
              ("motheye_absorptance", "A moth-eye", "#ea580c"),
              ("substrate_absorptance", "A substrate", "#16a34a")]
    for index, (_, label, color) in enumerate(series):
        lx = 55 + index * 218
        out.extend([line(lx, 178, lx + 28, 178, color=color, width=3),
                    circle(lx + 14, 178, radius=3.5, fill=color),
                    text(lx + 36, 183, label, size=15)])
    for percent in range(0, 101, 20):
        y = y_for(percent / 100)
        out.extend([line(left, y, right, y, color="#e2e8f0"),
                    text(left - 14, y + 5, f"{percent}%", size=13, anchor="end", fill="#475569")])
    out.extend([line(left, top, left, bottom, color="#64748b"),
                line(left, bottom, right, bottom, color="#64748b")])
    ticks = sorted(set(xs)) if sparse else [xmin + (xmax - xmin) * i / 6 for i in range(7)]
    for wavelength in ticks:
        x = x_for(wavelength)
        out.extend([line(x, bottom, x, bottom + 6, color="#64748b"),
                    text(x, bottom + 27, f"{wavelength:g}", size=13, anchor="middle")])
    for key, _, color in series:
        points = [(x_for(row["wavelength_nm"]), y_for(row[key]))
                  for row in rows if key in row]
        if len(points) > 1 and not sparse:
            out.append('<polyline points="' + " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
                       + f'" fill="none" stroke="{color}" stroke-width="2.7"/>')
        out.extend(circle(x, y, radius=5 if sparse else 2.8, fill=color) for x, y in points)
    out.append(text((left + right) / 2, 700, "Wavelength (nm)", size=17, anchor="middle"))
    out.append(text(55, 722, "For a semi-infinite Au substrate, T far = 0 and A total = A moth-eye + A substrate.",
                    size=14, fill="#475569"))
    destination.write_text(svg_document(width, height, out), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path,
                        help="*_convergence.json produced by converge.py")
    parser.add_argument("--spectrum", type=Path,
                        help="Optional *_spectrum.csv; inferred if omitted")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    prefix = args.report.name.removesuffix("_convergence.json")
    if prefix == args.report.name:
        raise ValueError("Report filename must end in _convergence.json")
    spectrum = args.spectrum
    if spectrum is None:
        dense = args.report.parent / f"{prefix}_spectrum.csv"
        anchor = args.report.parent / f"{prefix}_anchor_spectrum.csv"
        spectrum = anchor
        if dense.exists():
            candidate_rows = read_spectrum(dense)
            recommended = report.get("recommendation", {})
            if len(candidate_rows) > 3 and all(
                int(row.get(key, -1)) == int(recommended.get(key, -2))
                for row in candidate_rows for key in ("order", "slices", "grid")
            ):
                spectrum = dense
    output_dir = args.output_dir or args.report.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    convergence_file = output_dir / "convergence.svg"
    spectrum_file = output_dir / "spectrum.svg"
    plot_convergence(report, convergence_file)
    plot_spectrum(read_spectrum(spectrum), report, spectrum, spectrum_file)
    print(f"Convergence status: {report.get('status', 'unknown')}")
    print(convergence_file)
    print(spectrum_file)


if __name__ == "__main__":
    main()
