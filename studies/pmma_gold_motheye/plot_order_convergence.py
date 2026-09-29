"""Plot saved reflectance/transmittance against Fourier order.

Usage: python3 plot_order_convergence.py [checkpoint.json ...] [--max-order 20]
SVG output needs only the standard library. PNG is also saved if Pillow is installed.
CSV input with wavelength_nm,order,reflectance,transmittance is supported.
No RCWA calculation is started.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = ImageDraw = ImageFont = None


DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parent
    / "results"
    / "measured_30nm_Nz100_order"
    / "checkpoint.json"
)
BLUE = "#1769aa"
ORANGE = "#c46a13"
RED = "#bb3344"
INK = "#1f2b3a"
MUTED = "#536579"
GRID = "#dce4ec"


class Figure:
    def __init__(self, width: int, height: int):
        self.width, self.height = width, height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#f7f9fc"/>',
        ]
        self.scale = 2
        self.image = Image.new("RGB", (width * 2, height * 2), "#f7f9fc") if Image else None
        self.draw = ImageDraw.Draw(self.image) if self.image else None

    def rect(self, x, y, w, h, fill="white", stroke=None):
        border = f' stroke="{stroke}"' if stroke else ""
        self.parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
            f'fill="{fill}"{border}/>'
        )
        if self.draw:
            s = self.scale
            self.draw.rectangle((x*s, y*s, (x+w)*s, (y+h)*s), fill=fill, outline=stroke)

    def line(self, x1, y1, x2, y2, color=GRID, width=1, dash=False):
        pattern = ' stroke-dasharray="5 4"' if dash else ""
        self.parts.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{color}" stroke-width="{width}"{pattern}/>'
        )
        if self.draw:
            s = self.scale
            if not dash:
                self.draw.line((x1*s, y1*s, x2*s, y2*s), fill=color, width=max(1, round(width*s)))
            else:
                length = math.hypot(x2-x1, y2-y1)
                steps = max(1, math.ceil(length / 9))
                for i in range(steps):
                    start = i / steps
                    end = min(1, (i + 0.56) / steps)
                    self.draw.line(
                        ((x1+(x2-x1)*start)*s, (y1+(y2-y1)*start)*s,
                         (x1+(x2-x1)*end)*s, (y1+(y2-y1)*end)*s),
                        fill=color, width=max(1, round(width*s)),
                    )

    def circle(self, x, y, r, fill):
        self.parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{fill}"/>')
        if self.draw:
            s = self.scale
            self.draw.ellipse(((x-r)*s, (y-r)*s, (x+r)*s, (y+r)*s), fill=fill)

    def text(self, x, y, value, size=14, color=INK, anchor="start", bold=False):
        weight = "700" if bold else "400"
        self.parts.append(
            f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial,DejaVu Sans,sans-serif" '
            f'font-size="{size}" font-weight="{weight}" fill="{color}" '
            f'text-anchor="{anchor}" dominant-baseline="hanging">{html.escape(str(value))}</text>'
        )
        if self.draw:
            s = self.scale
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", size*s)
            except OSError:
                font = ImageFont.load_default()
            pil_anchor = {"start": "lt", "middle": "mt", "end": "rt"}[anchor]
            self.draw.text((x*s, y*s), str(value), font=font, fill=color, anchor=pil_anchor)

    def save(self, svg_path: Path):
        svg_path.parent.mkdir(parents=True, exist_ok=True)
        svg_path.write_text("\n".join([*self.parts, "</svg>"]), encoding="utf-8")
        print(svg_path)
        if self.image:
            png_path = svg_path.with_suffix(".png")
            self.image.save(png_path)
            print(png_path)


def load_results(paths: list[Path], max_order: int):
    if len(paths) > 1 and any(path.suffix.lower() == ".csv" for path in paths):
        raise ValueError("multiple inputs must be checkpoints with companion settings.json files")
    cases = {}
    expected_orders: set[int] = set()
    expected_wavelengths = None
    common_settings = None
    for path in paths:
        if path.suffix.lower() == ".csv":
            with path.open(encoding="utf-8-sig", newline="") as handle:
                raw_cases = list(csv.DictReader(handle))
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            raw_cases = list(payload["cases"].values())
            settings_path = path.with_name("settings.json")
            if not settings_path.is_file():
                if len(paths) > 1:
                    raise ValueError(f"missing companion settings.json for {path}")
            else:
                settings = json.loads(settings_path.read_text(encoding="utf-8"))
                expected_orders.update(int(x) for x in settings["numerics"]["orders"])
                wavelengths_here = sorted(
                    float(x["wavelength_nm"])
                    for x in settings["materials_at_wavelengths"]
                )
                if expected_wavelengths is None:
                    expected_wavelengths = wavelengths_here
                elif wavelengths_here != expected_wavelengths:
                    raise ValueError("checkpoints have different wavelength lists")
                comparable = json.loads(json.dumps(settings))
                comparable["numerics"].pop("orders")
                if common_settings is None:
                    common_settings = comparable
                elif comparable != common_settings:
                    raise ValueError("checkpoints have different settings besides Fourier orders")
        for raw in raw_cases:
            order = int(raw["order"])
            if order > max_order:
                continue
            wavelength = float(raw["wavelength_nm"])
            reflectance = float(raw["reflectance"])
            transmittance = float(raw["transmittance"])
            if not all(map(math.isfinite, (wavelength, reflectance, transmittance))):
                raise ValueError(f"non-finite result at M={order}, wavelength={wavelength}")
            identity = (order, wavelength)
            values = {"R": reflectance, "T": transmittance}
            if identity in cases and cases[identity] != values:
                raise ValueError(f"conflicting result at M={order}, wavelength={wavelength}")
            cases[identity] = values
    if not cases:
        raise ValueError(f"no saved results through M={max_order}")
    orders = sorted(expected_orders or {m for m, _ in cases})
    orders = [m for m in orders if m <= max_order]
    wavelengths = expected_wavelengths or sorted({w for _, w in cases})
    complete = [m for m in orders if all((m, w) in cases for w in wavelengths)]
    incomplete = [m for m in orders if m not in complete]
    if incomplete:
        print(f"Incomplete orders excluded from plots: {incomplete}")
    if len(complete) < 2:
        raise ValueError("at least two complete orders are needed")
    return cases, complete, wavelengths


def axes(fig, x, y, width, height, orders, ymin, ymax, ylabels):
    left, top = x+64, y+42
    plot_width, plot_height = width-91, height-96
    fig.rect(x, y, width, height, "#ffffff", GRID)
    x_position = lambda m: (
        left + plot_width/2 if len(orders) == 1
        else left + (m-orders[0]) / (orders[-1]-orders[0]) * plot_width
    )
    y_position = lambda v: top + (ymax-v)/(ymax-ymin)*plot_height
    for value, label in ylabels:
        py = y_position(value)
        fig.line(left, py, left+plot_width, py)
        fig.text(left-8, py-7, label, 11, MUTED, "end")
    for order in orders:
        px = x_position(order)
        fig.line(px, top, px, top+plot_height, "#eef1f5")
        fig.text(px, top+plot_height+7, str(order), 12, MUTED, "middle")
    fig.line(left, top+plot_height, left+plot_width, top+plot_height, MUTED)
    fig.text(left+plot_width/2, top+plot_height+28, "Fourier order M", 12, MUTED, "middle")
    return x_position, y_position


def plot_values(cases, orders, wavelengths, prefix, source_name):
    width, height = 1300, 815
    fig = Figure(width, height)
    fig.text(40, 28, "Reflectance and transmittance vs Fourier order", 25, INK, bold=True)
    fig.text(40, 65, f"Source: {source_name}  |  Each panel has its own y scale", 13, MUTED)
    cell_w, cell_h, gap = 390, 330, 25
    for column, wavelength in enumerate(wavelengths):
        for row, (metric, color) in enumerate((("R", BLUE), ("T", ORANGE))):
            x, y = 40 + column*(cell_w+gap), 110 + row*(cell_h+22)
            values = [100*cases[(m, wavelength)][metric] for m in orders]
            span = max(values)-min(values)
            padding = max(0.0001, 0.15*span, 0.01*abs(sum(values)/len(values)))
            ymin, ymax = min(values)-padding, max(values)+padding
            ticks = [(ymin+i*(ymax-ymin)/4, f"{ymin+i*(ymax-ymin)/4:.4g}") for i in range(5)]
            xp, yp = axes(fig, x, y, cell_w, cell_h, orders, ymin, ymax, ticks)
            fig.text(x+15, y+10, f"{wavelength:g} nm  |  {metric} (%)", 17, color, bold=True)
            for first, second in zip(orders, orders[1:]):
                fig.line(xp(first), yp(100*cases[(first,wavelength)][metric]),
                         xp(second), yp(100*cases[(second,wavelength)][metric]), color, 2.5)
            for order, value in zip(orders, values):
                fig.circle(xp(order), yp(value), 4.2, color)
    fig.save(prefix.with_name(prefix.name + "_values.svg"))


def plot_deltas(cases, orders, wavelengths, prefix, tolerance, source_name):
    width, height = 1300, 470
    fig = Figure(width, height)
    fig.text(40, 26, "Change between adjacent Fourier orders", 25, INK, bold=True)
    fig.text(40, 63, f"Absolute change in R and T (fractions)  |  Reference tolerance = {tolerance:g}  |  Source: {source_name}", 13, MUTED)
    high_orders = orders[1:]
    for column, wavelength in enumerate(wavelengths):
        x, y, cell_w, cell_h = 40+column*415, 108, 390, 330
        deltas = {
            metric: [abs(cases[(b,wavelength)][metric]-cases[(a,wavelength)][metric])
                     for a,b in zip(orders, orders[1:])]
            for metric in ("R", "T")
        }
        positives = [v for values in deltas.values() for v in values if v > 0]
        smallest = min([tolerance, *positives])
        largest = max([tolerance, *positives])
        low_power = math.floor(math.log10(smallest))
        high_power = math.ceil(math.log10(largest))
        if low_power == high_power:
            low_power -= 1
            high_power += 1
        ymin, ymax = float(low_power), float(high_power)
        ticks = [(float(k), f"1e{k}") for k in range(low_power, high_power+1)]
        xp, yp = axes(fig, x, y, cell_w, cell_h, high_orders, ymin, ymax, ticks)
        fig.text(x+15, y+10, f"{wavelength:g} nm", 17, INK, bold=True)
        fig.line(xp(high_orders[0]), yp(math.log10(tolerance)),
                 xp(high_orders[-1]), yp(math.log10(tolerance)), RED, 1.7, dash=True)
        for metric, color in (("R", BLUE), ("T", ORANGE)):
            points = [(xp(order), yp(math.log10(value)))
                      for order, value in zip(high_orders, deltas[metric]) if value > 0]
            for (x1,y1),(x2,y2) in zip(points, points[1:]):
                fig.line(x1,y1,x2,y2,color,2.5)
            for px,py in points:
                fig.circle(px,py,4,color)
        fig.line(x+22,y+cell_h-15,x+42,y+cell_h-15,BLUE,2.5)
        fig.text(x+47,y+cell_h-23,"delta R",11,BLUE)
        fig.line(x+117,y+cell_h-15,x+137,y+cell_h-15,ORANGE,2.5)
        fig.text(x+142,y+cell_h-23,"delta T",11,ORANGE)
        fig.line(x+213,y+cell_h-15,x+233,y+cell_h-15,RED,1.7,True)
        fig.text(x+238,y+cell_h-23,"tolerance",11,RED)
    fig.save(prefix.with_name(prefix.name + "_deltas.svg"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path,
                        help="one or more checkpoint.json files, or one CSV")
    parser.add_argument("--max-order", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=0.005)
    parser.add_argument("--output-prefix", type=Path,
                        help="Path prefix; default is order_convergence_M20 beside input")
    args = parser.parse_args()
    if args.max_order <= 0 or not math.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error("--max-order and --tolerance must be positive")
    try:
        inputs = args.inputs or [DEFAULT_CHECKPOINT]
        cases, orders, wavelengths = load_results(inputs, args.max_order)
        if len(wavelengths) != 3:
            raise ValueError("the figure layout expects exactly three wavelengths")
        prefix = args.output_prefix or inputs[0].with_name(f"order_convergence_M{args.max_order}")
        source_name = inputs[0].name if len(inputs) == 1 else f"{len(inputs)} checkpoints"
        plot_values(cases, orders, wavelengths, prefix, source_name)
        plot_deltas(cases, orders, wavelengths, prefix, args.tolerance, source_name)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
