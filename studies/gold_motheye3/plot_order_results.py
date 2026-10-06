"""Plot saved reflectance versus Fourier order without a new RCWA solve.

Uses the Python standard library and the existing show_results.py validator.
Outputs a reflectance SVG and an adjacent-order absolute-change SVG.
The default checkpoint is the 700 nm, Nz=140, grid=576 memory-safe sweep.
"""
from __future__ import annotations

import argparse
import html
import math
from pathlib import Path

from show_results import key, read_checkpoint

HERE = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = (HERE / "results" / "order_700_Nz140_grid576_memory_safe"
                      / "checkpoint.json")


def collect(saved, wavelength, max_order=None):
    plan, cases = saved["plan"], saved["cases"]
    if plan["axis"] != "order":
        raise ValueError("Use an order-sweep checkpoint, not a slices/grid report.")
    if wavelength not in plan["wavelengths_nm"]:
        raise ValueError(f"No planned wavelength {wavelength:g} nm in this checkpoint.")
    orders = [m for m in plan["values"] if max_order is None or m <= max_order]
    points = [(m, 100 * float(cases[key(m, wavelength)]["reflectance"]))
              for m in orders if key(m, wavelength) in cases]
    if len(points) < 2:
        raise ValueError("At least two saved orders are required for this plot.")
    by_order = dict(points)
    changes = [(high, abs(by_order[high] - by_order[low]))
               for low, high in zip(orders, orders[1:])
               if low in by_order and high in by_order]
    return orders, points, changes


def text(x, y, value, size=16, anchor="start", color="#334155"):
    return (f'<text x="{x:g}" y="{y:g}" font-size="{size}" '
            f'text-anchor="{anchor}" fill="{color}">{html.escape(str(value))}</text>')


def render(path, title, subtitle, orders, points, ylabel, *, threshold=None,
           full_scale=False, zero_floor=False):
    width, height = 1040, 660
    left, top, plot_width, plot_height = 115, 130, 870, 400
    bottom = top + plot_height
    numbers = [y for _, y in points]
    if threshold is not None:
        numbers.append(threshold)
    low, high = min(numbers), max(numbers)
    padding = max((high - low) * .12, .05)
    low, high = low - padding, high + padding
    if full_scale:
        low, high = 0., 100.
    elif zero_floor:
        low, high = 0., max(high, .1)
    elif min(numbers) >= 0:
        low = max(0., low)
    x_min, x_max = min(orders), max(orders)

    def x_for(m):
        return left + (m - x_min) / (x_max - x_min) * plot_width

    def y_for(y):
        return bottom - (y - low) / (high - low) * plot_height

    items = [text(40, 42, title, size=28), text(40, 77, subtitle, size=15),
             f'<rect x="{left}" y="{top}" width="{plot_width}" '
             f'height="{plot_height}" fill="white" stroke="#cbd5e1"/>']
    for i in range(6):
        number = low + (high - low) * i / 5
        y = y_for(number)
        items += [f'<line x1="{left}" x2="{left+plot_width}" y1="{y:.3f}" '
                  f'y2="{y:.3f}" stroke="#e2e8f0"/>',
                  text(left - 12, y + 5, f"{number:.5g}", size=14, anchor="end")]
    tick_stride = max(1, math.ceil(len(orders) / 18))
    ticks = orders[::tick_stride]
    if ticks[-1] != orders[-1]:
        ticks.append(orders[-1])
    for m in ticks:
        x = x_for(m)
        items += [f'<line x1="{x:.3f}" x2="{x:.3f}" y1="{top}" '
                  f'y2="{bottom}" stroke="#edf2f7"/>',
                  text(x, bottom + 28, m, size=14, anchor="middle")]
    if threshold is not None:
        y = y_for(threshold)
        items += [f'<line x1="{left}" x2="{left+plot_width}" y1="{y:.3f}" '
                  f'y2="{y:.3f}" stroke="#b45309" stroke-dasharray="7 5"/>',
                  text(left + plot_width - 8, y - 9, f"Tolerance: {threshold:g} pp",
                       size=14, anchor="end", color="#b45309")]
    # Break connecting lines at missing planned orders; no interpolation.
    by_order = dict(points)
    segments, segment = [], []
    for m in orders:
        if m in by_order:
            segment.append((x_for(m), y_for(by_order[m])))
        elif segment:
            segments.append(segment)
            segment = []
    if segment:
        segments.append(segment)
    for segment in segments:
        if len(segment) > 1:
            coordinates = " ".join(f"{x:.3f},{y:.3f}" for x, y in segment)
            items.append(f'<polyline points="{coordinates}" fill="none" '
                         'stroke="#1769aa" stroke-width="2.5"/>')
    for m, y in points:
        items.append(f'<circle cx="{x_for(m):.3f}" cy="{y_for(y):.3f}" '
                     'r="4.5" fill="#1769aa"/>')
    items += [text(left + plot_width / 2, bottom + 65, "Fourier order M", anchor="middle"),
              f'<text transform="translate(30 {top+plot_height/2}) rotate(-90)" '
              f'text-anchor="middle" font-size="17">{html.escape(ylabel)}</text>',
              text(40, 625, "Saved data only. Connecting lines are guides. "
                   + ("Y axis: 0-100%." if full_scale else "Y axis: automatic range."), size=14)]
    document = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
                f'viewBox="0 0 {width} {height}">\n'
                '<rect width="100%" height="100%" fill="#f8fafc"/>\n'
                '<g font-family="Arial, Helvetica, sans-serif">\n'
                + "\n".join(items) + "\n</g>\n</svg>\n")
    path.write_text(document, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", nargs="?", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--wavelength", type=float, default=700.)
    parser.add_argument("--max-order", type=int)
    parser.add_argument("--full-scale", action="store_true", help="R axis spans 0-100 percent.")
    parser.add_argument("--output-dir", type=Path, help="Default: checkpoint folder.")
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    try:
        saved = read_checkpoint(checkpoint)
        orders, points, changes = collect(saved, args.wavelength, args.max_order)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    output = (args.output_dir or checkpoint.parent).resolve()
    output.mkdir(parents=True, exist_ok=True)
    fixed, geometry = saved["plan"]["fixed_numerics"], saved["plan"]["geometry"]
    subtitle = (f"Nz={fixed['slices']}; grid={fixed['grid']}; "
                f"period={geometry['period_nm']:g} nm; height={geometry['height_nm']:g} nm; "
                f"{len(points)}/{len(orders)} orders saved")
    suffix = f"{args.wavelength:g}nm"
    reflectance = output / f"reflectance_vs_order_{suffix}.svg"
    render(reflectance, f"Reflectance vs Fourier order at {args.wavelength:g} nm",
           subtitle, orders, points, "Reflectance R (%)", full_scale=args.full_scale)
    print(f"figure: {reflectance}")
    if changes:
        differences = output / f"reflectance_order_changes_{suffix}.svg"
        render(differences, f"Reflectance change between orders at {args.wavelength:g} nm",
               subtitle, orders, changes, "Absolute change in R (percentage points)",
               threshold=100 * saved["plan"]["tolerance"], zero_floor=True)
        print(f"changes: {differences}")
    print("      M         R (%)")
    for m, value in points:
        print(f"{m:7d} {value:13.7f}")
    print("No RCWA calculation or checkpoint update was performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
