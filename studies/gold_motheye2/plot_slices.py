"""Plot Au moth-eye reflectance and runtime against height-slice count."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from studies.gold_motheye2.plot_order import (
    COLORS, dot, label, polyline, segment, write_svg,
)


def read_cases(path: Path, meta: dict) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        raw = list(csv.DictReader(handle))
    if not raw:
        raise ValueError(f"No completed cases in {path}")
    planned_slices = {int(value) for value in meta["slice_counts"]}
    planned_wavelengths = {float(value) for value in meta["wavelengths_nm"]}
    rows: list[dict[str, float]] = []
    seen: set[tuple[int, float]] = set()
    for item in raw:
        row = {key: float(item[key]) for key in
               ("order", "wavelength_nm", "slices", "grid",
                "reflectance", "runtime_seconds")}
        if any(not math.isfinite(value) for value in row.values()):
            raise ValueError("Nonfinite value in slice_sweep.csv")
        nz = int(row["slices"])
        wavelength = row["wavelength_nm"]
        key = (nz, wavelength)
        if (row["slices"] != nz or row["order"] != int(meta["fixed_order"])
                or row["grid"] != int(meta["fixed_grid"])
                or nz not in planned_slices or wavelength not in planned_wavelengths
                or key in seen or not 0 <= row["reflectance"] <= 1
                or row["runtime_seconds"] < 0):
            raise ValueError(f"Invalid or duplicate slice-sweep case: {key}")
        seen.add(key)
        rows.append(row)
    return rows


def reflectance_figure(rows: list[dict[str, float]], meta: dict,
                       destination: Path) -> None:
    slices = [int(value) for value in meta["slice_counts"]]
    wavelengths = [float(value) for value in meta["wavelengths_nm"]]
    panel_width = 410
    width = max(1050, 80 + panel_width * len(wavelengths))
    height = 730
    top, bottom = 190, 565
    tolerance_pp = 100 * float(meta["tolerance"])
    parts = [
        label(50, 52, "Reflectance vs height slices", size=29, bold=True),
        label(50, 82, f"Fixed M={meta['fixed_order']}, grid={meta['fixed_grid']}; "
              f"measured Au n,k; criterion {tolerance_pp:g} percentage points",
              size=17),
        label(50, 109, "Each wavelength panel has its own vertical scale.",
              size=14, color="#475569"),
        label(width - 50, 52, str(meta["status"]).replace("_", " ").upper(),
              size=14, anchor="end", bold=True,
              color="#15803d" if meta["status"] == "converged_within_tested_slices"
              else "#b45309" if meta["status"] == "incomplete" else "#b91c1c"),
    ]
    for index, wavelength in enumerate(wavelengths):
        color = COLORS[index % len(COLORS)]
        left = 76 + index * panel_width
        right = left + 326
        selected = sorted((row for row in rows if row["wavelength_nm"] == wavelength),
                          key=lambda row: row["slices"])
        parts.append(label(left, 158, f"{wavelength:g} nm", size=21, bold=True))
        if not selected:
            parts.append(label(left, 360, "No completed cases", color="#b91c1c"))
            continue
        values = [100 * row["reflectance"] for row in selected]
        padding = max(0.02, (max(values) - min(values)) * 0.16)
        ymin = max(0.0, min(values) - padding)
        ymax = min(100.0, max(values) + padding)
        if ymax <= ymin:
            ymax = ymin + 0.05
        x_for = lambda nz: left + (nz - slices[0]) / (slices[-1] - slices[0]) * (right-left)
        y_for = lambda percent: bottom - (percent - ymin) / (ymax - ymin) * (bottom-top)
        digits = 3 if ymax - ymin < 0.2 else 2
        for tick in range(5):
            value = ymin + tick * (ymax - ymin) / 4
            y = y_for(value)
            parts.extend((
                segment(left, y, right, y, color="#e2e8f0"),
                label(left - 9, y + 5, f"{value:.{digits}f}%", size=12,
                      color="#475569", anchor="end"),
            ))
        parts.extend((segment(left, top, left, bottom, color="#64748b"),
                      segment(left, bottom, right, bottom, color="#64748b")))
        for nz in slices:
            x = x_for(nz)
            parts.extend((segment(x, bottom, x, bottom + 5, color="#64748b"),
                          label(x, bottom + 23, nz, size=11, anchor="middle")))
        points = [(x_for(row["slices"]), y_for(100 * row["reflectance"]))
                  for row in selected]
        parts.append(polyline(points, color))
        parts.extend(dot(x, y, color) for x, y in points)
        changes = meta.get("adjacent_changes", {}).get(f"{wavelength:g}", [])
        if meta["status"] == "incomplete":
            verdict = f"Partial: {len(selected)}/{len(slices)} slice counts"
            verdict_color = "#b45309"
        elif len(changes) >= 2:
            last = changes[-2:]
            passed = bool(meta["per_wavelength_pass"].get(f"{wavelength:g}", False))
            verdict = (f"Last changes: {last[0]['delta_reflectance_pp']:.3f}, "
                       f"{last[1]['delta_reflectance_pp']:.3f} pp  "
                       f"{'PASS' if passed else 'FAIL'}")
            verdict_color = "#15803d" if passed else "#b91c1c"
        else:
            verdict = "Insufficient comparisons"
            verdict_color = "#b91c1c"
        parts.append(label(left, 625, verdict, size=13, color=verdict_color, bold=True))
    parts.extend((
        label(width / 2, 686, "Height slices Nz", size=17, anchor="middle"),
        label(50, 712, "Markers are calculated cases; lines connect them as guides.",
              size=13, color="#475569"),
    ))
    write_svg(destination, width, height, parts)


def runtime_figure(rows: list[dict[str, float]], meta: dict,
                   destination: Path) -> None:
    slices = [int(value) for value in meta["slice_counts"]]
    wavelengths = [float(value) for value in meta["wavelengths_nm"]]
    by_key = {(int(row["slices"]), row["wavelength_nm"]): row for row in rows}
    complete_slices = [nz for nz in slices if
                       all((nz, wavelength) in by_key for wavelength in wavelengths)]
    totals = {nz: sum(by_key[(nz, wavelength)]["runtime_seconds"]
                      for wavelength in wavelengths) for nz in complete_slices}
    largest = max([row["runtime_seconds"] for row in rows] + list(totals.values()))
    scale = 60.0 if largest >= 120 else 1.0
    unit = "min" if scale == 60 else "s"
    ymax = max(1.0, largest / scale * 1.15)
    width, height = 1130, 700
    left, right, top, bottom = 100, 1050, 190, 585
    x_for = lambda nz: left + (nz - slices[0]) / (slices[-1] - slices[0]) * (right-left)
    y_for = lambda seconds: bottom - (seconds / scale) / ymax * (bottom-top)
    parts = [
        label(52, 56, "Runtime vs height slices", size=29, bold=True),
        label(52, 88, f"Fixed M={meta['fixed_order']}, grid={meta['fixed_grid']}; "
              "elapsed solve time for each case", size=17),
        label(52, 116, "Total is summed only when all wavelengths are complete at that Nz.",
              size=15, color="#475569"),
    ]
    legend = [("Total", "#111827")]
    legend += [(f"{wavelength:g} nm", COLORS[i % len(COLORS)])
               for i, wavelength in enumerate(wavelengths)]
    for index, (name, color) in enumerate(legend):
        x = 52 + index * 205
        parts.extend((segment(x, 154, x + 27, 154, color=color, width=3),
                      dot(x + 13, 154, color, radius=3.5),
                      label(x + 36, 159, name, size=14)))
    for tick in range(6):
        value = ymax * tick / 5
        y = bottom - tick * (bottom-top) / 5
        parts.extend((segment(left, y, right, y, color="#e2e8f0"),
                      label(left - 12, y + 5, f"{value:.1f}", size=13,
                            color="#475569", anchor="end")))
    parts.extend((segment(left, top, left, bottom, color="#64748b"),
                  segment(left, bottom, right, bottom, color="#64748b"),
                  label(52, 180, f"Time ({unit})", size=15)))
    for nz in slices:
        x = x_for(nz)
        parts.extend((segment(x, bottom, x, bottom + 6, color="#64748b"),
                      label(x, bottom + 25, nz, size=12, anchor="middle")))
    for index, wavelength in enumerate(wavelengths):
        color = COLORS[index % len(COLORS)]
        points = [(x_for(nz), y_for(by_key[(nz, wavelength)]["runtime_seconds"]))
                  for nz in slices if (nz, wavelength) in by_key]
        parts.append(polyline(points, color, width=2.1))
        parts.extend(dot(x, y, color, radius=3.8) for x, y in points)
    points = [(x_for(nz), y_for(totals[nz])) for nz in complete_slices]
    parts.append(polyline(points, "#111827", width=3.2))
    parts.extend(dot(x, y, "#111827", radius=5) for x, y in points)
    parts.extend((
        label((left + right) / 2, 652, "Height slices Nz", size=17, anchor="middle"),
        label(52, 681, "GPU timing excludes queue wait and plot generation.",
              size=13, color="#475569"),
    ))
    write_svg(destination, width, height, parts)


def render(csv_path: Path, metadata_path: Path, output_dir: Path) -> tuple[Path, Path]:
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    rows = read_cases(csv_path, meta)
    output_dir.mkdir(parents=True, exist_ok=True)
    reflectance = output_dir / "reflectance_vs_slices.svg"
    runtime = output_dir / "runtime_vs_slices.svg"
    reflectance_figure(rows, meta, reflectance)
    runtime_figure(rows, meta, runtime)
    print(f"graphs: {reflectance}, {runtime}")
    return reflectance, runtime
