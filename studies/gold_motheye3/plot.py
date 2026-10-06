"""Dependency-free SVG figures for the Au moth-eye convergence sweep."""

from __future__ import annotations

import html
import math
from pathlib import Path

LABELS = {
    "reflectance": "Reflectance R (%)",
    "power_into_substrate": "Power into Au substrate (%)",
    "motheye_absorptance": "Pillar-region absorptance (%)",
}
COLORS = ("#1769aa", "#d05b18", "#15803d", "#7c3aed", "#0891b2")


def _text(x: float, y: float, value: object, *, size=15,
          color="#223047", anchor="start") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
            f'font-size="{size}" fill="{color}">{html.escape(str(value))}</text>')


def _line(x1: float, y1: float, x2: float, y2: float,
          color="#d9e2ec", width=1) -> str:
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" '
            f'y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"/>')


def _svg(path: Path, width: int, height: int, items: list[str]) -> None:
    content = (f'<svg xmlns="http://www.w3.org/2000/svg" '
               f'viewBox="0 0 {width} {height}" width="{width}" height="{height}">\n'
               '<rect width="100%" height="100%" fill="#f7f9fc"/>\n'
               '<g font-family="Arial, Helvetica, sans-serif">\n'
               + "\n".join(items) + "\n</g>\n</svg>\n")
    path.write_text(content, encoding="utf-8")


def _panel(items: list[str], *, left: float, top: float, width: float,
           height: float, values: tuple[int, ...], points: list[tuple[int, float]],
           color: str) -> None:
    right, bottom = left + width, top + height
    items.append(f'<rect x="{left-8:.1f}" y="{top-30:.1f}" '
                 f'width="{width+22:.1f}" height="{height+68:.1f}" '
                 'fill="white" stroke="#e2e8f0"/>')
    if not points:
        items.append(_text(left + 25, top + height / 2, "No saved cases", color="#a64b4b"))
        return
    magnitudes = [item[1] for item in points]
    minimum, maximum = min(magnitudes), max(magnitudes)
    spread = maximum - minimum
    padding = max(spread * 0.1, abs(maximum) * 0.005, 0.01)
    low, high = minimum - padding, maximum + padding
    x_min, x_max = min(values), max(values)

    def x_for(value: int) -> float:
        return left + (value - x_min) / (x_max - x_min) * width

    def y_for(value: float) -> float:
        return bottom - (value - low) / (high - low) * height

    for tick in range(3):
        value = low + (high - low) * tick / 2
        y = y_for(value)
        items.append(_line(left, y, right, y))
        items.append(_text(left - 10, y + 5, f"{value:.3g}", size=12,
                           color="#64748b", anchor="end"))
    items.extend((_line(left, top, left, bottom, "#64748b"),
                  _line(left, bottom, right, bottom, "#64748b")))
    for value in values:
        x = x_for(value)
        items.append(_line(x, bottom, x, bottom + 5, "#64748b"))
        items.append(_text(x, bottom + 22, value, size=11, color="#64748b",
                           anchor="middle"))
    coordinates = [(x_for(value), y_for(number)) for value, number in points]
    if len(coordinates) > 1:
        path = " ".join(f"{x:.2f},{y:.2f}" for x, y in coordinates)
        items.append(f'<polyline points="{path}" fill="none" '
                     f'stroke="{color}" stroke-width="2.6"/>')
    items.extend(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.7" '
                 f'fill="{color}"/>' for x, y in coordinates)


def render(plan: dict, cases: dict[str, dict], report: dict, destination: Path) -> None:
    values = tuple(int(value) for value in plan["values"])
    wavelengths = tuple(float(value) for value in plan["wavelengths_nm"])
    width = max(760, 90 + 390 * len(wavelengths))
    height = 1000
    numerical_labels = {"order": "M", "slices": "Nz", "grid": "grid"}
    fixed_text = ", ".join(f"{numerical_labels[name]}={value}"
                           for name, value in plan["fixed_numerics"].items()
                           if name != plan["axis"])
    geometry = plan["geometry"]
    items = [
        _text(45, 45, f"Au moth-eye convergence vs {plan['axis']}", size=27),
        _text(45, 76, f"Period={geometry['period_nm']:g} nm; height={geometry['height_nm']:g} nm; "
              f"fixed {fixed_text}; tolerance={100 * plan['tolerance']:g} pp", size=14,
              color="#52657b"),
        _text(45, 102, report["status"], size=15,
              color="#15803d" if report["status"] == "converged_within_tested_values"
              else "#a64b4b"),
    ]
    for column, wavelength in enumerate(wavelengths):
        left = 82 + column * 390
        color = COLORS[column % len(COLORS)]
        items.append(_text(left, 140, f"{wavelength:g} nm", size=20))
        for row_index, (metric, label) in enumerate(LABELS.items()):
            top = 194 + row_index * 260
            points = [(value, 100 * float(cases[f"{value}|{wavelength:.12g}"][metric]))
                      for value in values if f"{value}|{wavelength:.12g}" in cases]
            _panel(items, left=left, top=top, width=300, height=170,
                   values=values, points=points, color=color)
            items.append(_text(left, top - 7, label, size=15, color=color))
    items.append(_text(45, 972, "Each panel has its own y scale. Far-side T = 0 for the semi-infinite Au substrate.",
                       size=13, color="#52657b"))
    _svg(destination, width, height, items)

    runtime_items = [
        _text(45, 45, f"Au moth-eye runtime vs {plan['axis']}", size=29),
        _text(45, 76, f"{report['completed_cases']}/{report['expected_cases']} cases saved",
              size=15, color="#52657b"),
    ]
    runtime_height = 415
    for column, wavelength in enumerate(wavelengths):
        left = 82 + column * 390
        runtime_items.append(_text(left, 125, f"{wavelength:g} nm: seconds", size=17,
                                   color=COLORS[column % len(COLORS)]))
        points = [(value, float(cases[f"{value}|{wavelength:.12g}"]["runtime_seconds"]))
                  for value in values if f"{value}|{wavelength:.12g}" in cases]
        if any(not math.isfinite(number) or number < 0 for _, number in points):
            raise ValueError("Nonfinite or negative runtime in saved results.")
        _panel(runtime_items, left=left, top=175, width=300, height=170,
               values=values, points=points, color=COLORS[column % len(COLORS)])
    _svg(destination.with_name("runtime.svg"), width, runtime_height, runtime_items)
