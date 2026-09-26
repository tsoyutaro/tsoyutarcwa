"""Plot fixed-grid Fourier-order reflectance and runtime as portable SVG files."""

from __future__ import annotations

import csv
import html
import json
import math
from pathlib import Path

COLORS = ("#1d4ed8", "#ea580c", "#16a34a", "#7c3aed", "#0891b2")


def label(x: float, y: float, value: object, *, size=15, color="#182235",
          anchor="start", bold=False) -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
            f'font-size="{size}" font-weight="{"bold" if bold else "normal"}" '
            f'fill="{color}">{html.escape(str(value))}</text>')


def segment(x1: float, y1: float, x2: float, y2: float, *,
            color="#cbd5e1", width=1, dash="") -> str:
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{color}" stroke-width="{width}"{extra}/>')


def dot(x: float, y: float, color: str, radius=4.5) -> str:
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="{color}"/>'


def polyline(points: list[tuple[float, float]], color: str, width=2.7) -> str:
    if len(points) < 2:
        return ""
    coordinates = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return (f'<polyline points="{coordinates}" fill="none" '
            f'stroke="{color}" stroke-width="{width}"/>')


def write_svg(path: Path, width: int, height: int, contents: list[str]) -> None:
    data = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'width="{width}" height="{height}" role="img">\n'
            '<rect width="100%" height="100%" fill="white"/>\n'
            '<g font-family="Arial, Helvetica, sans-serif">\n'
            + "\n".join(item for item in contents if item) + "\n</g>\n</svg>\n")
    path.write_text(data, encoding="utf-8")


def read_cases(path: Path) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        raw = list(csv.DictReader(handle))
    if not raw:
        raise ValueError(f"No completed cases in {path}")
    rows = []
    seen: set[tuple[int, float]] = set()
    for item in raw:
        row = {key: float(item[key]) for key in
               ("order", "wavelength_nm", "slices", "grid",
                "reflectance", "runtime_seconds")}
        if any(not math.isfinite(value) for value in row.values()):
            raise ValueError("Nonfinite result in order_sweep.csv")
        key = (int(row["order"]), row["wavelength_nm"])
        if key in seen:
            raise ValueError(f"Duplicate result for {key}")
        seen.add(key)
        if not 0 <= row["reflectance"] <= 1 or row["runtime_seconds"] < 0:
            raise ValueError(f"Invalid reflectance or runtime for {key}")
        rows.append(row)
    return rows


def reflectance_figure(rows: list[dict[str, float]], meta: dict,
                       destination: Path) -> None:
    wavelengths = [float(item) for item in meta["wavelengths_nm"]]
    orders = [int(item) for item in meta["orders"]]
    tolerance_pp = 100 * float(meta["tolerance"])
    panel_width = 390
    width = max(1000, 80 + panel_width * len(wavelengths))
    height = 745
    top, bottom = 210, 585
    parts = [
        label(50, 55, "Reflectance convergence vs Fourier order", size=29, bold=True),
        label(50, 87, f"Fixed Nz={meta['fixed_slices']}, grid={meta['fixed_grid']}; "
              f"gold: measured n,k; tolerance {tolerance_pp:g} percentage points",
              size=17),
        label(50, 118, "Each panel uses its own vertical scale; markers are calculated orders.",
              size=15, color="#475569"),
        label(width - 50, 55, meta["status"].replace("_", " ").upper(),
              size=15, color="#b91c1c" if meta["status"] != "converged_within_tested_orders"
              else "#15803d", anchor="end", bold=True),
    ]
    for index, wavelength in enumerate(wavelengths):
        color = COLORS[index % len(COLORS)]
        left = 76 + index * panel_width
        right = left + 306
        selected = sorted((row for row in rows if row["wavelength_nm"] == wavelength),
                          key=lambda row: row["order"])
        parts.append(label(left, 185, f"{wavelength:g} nm", size=21, bold=True))
        if not selected:
            parts.append(label(left, 390, "No completed cases", color="#b91c1c"))
            continue
        values = [100 * row["reflectance"] for row in selected]
        padding = max(tolerance_pp, (max(values) - min(values)) * 0.18, 0.05)
        ymin = max(0.0, min(values) - padding)
        ymax = min(100.0, max(values) + padding)
        if ymax <= ymin:
            ymax = ymin + 1.0
        x_for = lambda order: left + (order - orders[0]) / max(1, orders[-1] - orders[0]) * (right - left)
        y_for = lambda percent: bottom - (percent - ymin) / (ymax - ymin) * (bottom - top)
        reference = values[-1]
        band_low = max(ymin, reference - tolerance_pp)
        band_high = min(ymax, reference + tolerance_pp)
        band_top = y_for(band_high)
        band_bottom = y_for(band_low)
        parts.append(f'<rect x="{left}" y="{band_top:.2f}" width="{right-left}" '
                     f'height="{max(0, band_bottom-band_top):.2f}" fill="#dbeafe" opacity="0.68"/>')
        for tick in range(5):
            value = ymin + (ymax - ymin) * tick / 4
            y = y_for(value)
            parts.extend([segment(left, y, right, y, color="#e2e8f0"),
                          label(left - 10, y + 5, f"{value:.2f}%", size=12,
                                color="#475569", anchor="end")])
        parts.extend([segment(left, top, left, bottom, color="#64748b"),
                      segment(left, bottom, right, bottom, color="#64748b")])
        for order in orders:
            x = x_for(order)
            parts.extend([segment(x, bottom, x, bottom + 5, color="#64748b"),
                          label(x, bottom + 24, order, size=12, anchor="middle")])
        points = [(x_for(row["order"]), y_for(100 * row["reflectance"]))
                  for row in selected]
        parts.append(polyline(points, color))
        parts.extend(dot(x, y, color) for x, y in points)
        changes = meta.get("adjacent_changes", {}).get(f"{wavelength:g}", [])
        if meta["status"] == "incomplete":
            verdict = f"Partial: {len(selected)}/{len(orders)} orders"
            verdict_color = "#b45309"
        elif len(changes) >= 2:
            last = [item["delta_reflectance_pp"] for item in changes[-2:]]
            passed = bool(meta["per_wavelength_pass"].get(f"{wavelength:g}", False))
            verdict = (f"Last two changes: {last[0]:.3f}, {last[1]:.3f} pp  "
                       f"{'PASS' if passed else 'FAIL'}")
            verdict_color = "#15803d" if passed else "#b91c1c"
        else:
            verdict = "Insufficient comparisons"
            verdict_color = "#b91c1c"
        parts.append(label(left, 646, verdict, size=13, color=verdict_color, bold=True))
    parts.append(label(width / 2, 707, "Fourier order M", size=17, anchor="middle"))
    parts.append(label(50, 730, "Blue band: +/- tolerance around the highest completed order in each panel.",
                       size=13, color="#475569"))
    write_svg(destination, width, height, parts)


def runtime_figure(rows: list[dict[str, float]], meta: dict,
                   destination: Path) -> None:
    orders = [int(item) for item in meta["orders"]]
    wavelengths = [float(item) for item in meta["wavelengths_nm"]]
    by_key = {(int(row["order"]), row["wavelength_nm"]): row for row in rows}
    complete_orders = [order for order in orders if
                       all((order, wavelength) in by_key for wavelength in wavelengths)]
    totals = {order: sum(by_key[(order, wavelength)]["runtime_seconds"]
                         for wavelength in wavelengths) for order in complete_orders}
    largest = max([row["runtime_seconds"] for row in rows] + list(totals.values()))
    scale = 60.0 if largest >= 120 else 1.0
    unit = "min" if scale == 60 else "s"
    ymax = max(1.0, largest / scale * 1.15)
    width, height = 1130, 700
    left, right, top, bottom = 102, 1050, 190, 585
    x_for = lambda order: left + (order - orders[0]) / max(1, orders[-1] - orders[0]) * (right-left)
    y_for = lambda seconds: bottom - (seconds / scale) / ymax * (bottom-top)
    parts = [
        label(52, 56, "Runtime vs Fourier order", size=29, bold=True),
        label(52, 88, f"Fixed Nz={meta['fixed_slices']}, grid={meta['fixed_grid']}; "
              "elapsed solve time recorded for each case", size=17),
        label(52, 116, "Total is the sum across wavelengths for complete orders only.",
              size=15, color="#475569"),
    ]
    legend = [("Total", "#111827")]
    legend += [(f"{wavelength:g} nm", COLORS[i % len(COLORS)])
               for i, wavelength in enumerate(wavelengths)]
    for index, (name, color) in enumerate(legend):
        x = 52 + index * 205
        parts.extend([segment(x, 154, x + 27, 154, color=color, width=3),
                      dot(x + 13, 154, color, radius=3.5),
                      label(x + 36, 159, name, size=14)])
    for tick in range(6):
        value = ymax * tick / 5
        y = bottom - tick / 5 * (bottom-top)
        parts.extend([segment(left, y, right, y, color="#e2e8f0"),
                      label(left - 13, y + 5, f"{value:.1f}", size=13,
                            color="#475569", anchor="end")])
    parts.extend([segment(left, top, left, bottom, color="#64748b"),
                  segment(left, bottom, right, bottom, color="#64748b"),
                  label(52, 180, f"Time ({unit})", size=15)])
    for order in orders:
        x = x_for(order)
        parts.extend([segment(x, bottom, x, bottom + 6, color="#64748b"),
                      label(x, bottom + 25, order, size=13, anchor="middle")])
    for index, wavelength in enumerate(wavelengths):
        color = COLORS[index % len(COLORS)]
        points = [(x_for(order), y_for(by_key[(order, wavelength)]["runtime_seconds"]))
                  for order in orders if (order, wavelength) in by_key]
        parts.append(polyline(points, color, width=2.1))
        parts.extend(dot(x, y, color, radius=3.8) for x, y in points)
    total_points = [(x_for(order), y_for(totals[order])) for order in complete_orders]
    parts.append(polyline(total_points, "#111827", width=3.2))
    parts.extend(dot(x, y, "#111827", radius=5) for x, y in total_points)
    parts.extend([label((left+right)/2, 654, "Fourier order M", size=17, anchor="middle"),
                  label(52, 682, "Lines connect measured points as guides; times depend on GPU, cache, and software environment.",
                        size=13, color="#475569")])
    write_svg(destination, width, height, parts)


def render(csv_path: Path, metadata_path: Path, output_dir: Path) -> tuple[Path, Path]:
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    rows = read_cases(csv_path)
    expected_slices = int(meta["fixed_slices"])
    expected_grid = int(meta["fixed_grid"])
    if any(int(row["slices"]) != expected_slices or int(row["grid"]) != expected_grid
           for row in rows):
        raise ValueError("CSV contains cases outside the fixed Nz/grid study.")
    output_dir.mkdir(parents=True, exist_ok=True)
    reflection = output_dir / "reflectance_vs_order.svg"
    runtime = output_dir / "runtime_vs_order.svg"
    reflectance_figure(rows, meta, reflection)
    runtime_figure(rows, meta, runtime)
    print(f"graphs: {reflection}, {runtime}")
    return reflection, runtime
