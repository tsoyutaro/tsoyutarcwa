"""The preview and solver use exactly the same stepped Au/PMMA/air layers."""
from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Layer:
    top_depth_nm: float  # Downward from the uncoated PMMA tip; cap is negative.
    bottom_depth_nm: float
    core_width_nm: float
    outer_width_nm: float
    background: str
    kind: str


def validate_geometry(g):
    if any(not math.isfinite(float(v)) for v in g.values()):
        raise ValueError("Geometry values must be finite.")
    period, height = g["period_nm"], g["height_nm"]
    if min(period, height, g["profile_power"]) <= 0:
        raise ValueError("Period, PMMA height and profile power must be positive.")
    if not 0 < g["top_width_nm"] <= g["bottom_width_nm"] <= period:
        raise ValueError("Require 0 < PMMA top width <= bottom width <= period.")
    if any(g[k] < 0 for k in ("gold_side_thickness_x_nm", "gold_cap_thickness_nm", "gold_valley_thickness_nm")):
        raise ValueError("Gold thicknesses must be nonnegative.")
    if g["bottom_width_nm"]+2*g["gold_side_thickness_x_nm"] > period:
        raise ValueError("Side coatings overlap adjacent periods; reduce width or side thickness.")
    if not 0 <= g["gold_valley_thickness_nm"] <= height:
        raise ValueError("Valley gold thickness must be within the PMMA height.")
    if not 0 < g["gold_cap_width_nm"] <= period:
        raise ValueError("Gold cap width must be within (0, period].")


def build_layers(g, slices):
    validate_geometry(g)
    if not isinstance(slices, int) or isinstance(slices, bool) or slices <= 0:
        raise ValueError("PMMA slice count must be a positive integer.")
    height, valley = g["height_nm"], g["gold_valley_thickness_nm"]
    if 0 < valley < height:
        if slices < 2:
            raise ValueError("Partial valley coating needs at least two PMMA slices.")
        valley_slices = max(1, min(slices-1, round(slices*valley/height)))
    else:
        valley_slices = slices if valley == height else 0
    layers = []
    if g["gold_cap_thickness_nm"] > 0:
        layers.append(Layer(-g["gold_cap_thickness_nm"], 0., 0., g["gold_cap_width_nm"], "air", "cap"))
    regions = [(0., height-valley, slices-valley_slices, "shell"),
               (height-valley, height, valley_slices, "valley")]
    for top, bottom, count, kind in regions:
        for i in range(count):
            upper = top+(bottom-top)*i/count
            lower = top+(bottom-top)*(i+1)/count
            fraction = (upper+lower)/(2*height)
            core = g["top_width_nm"]+(g["bottom_width_nm"]-g["top_width_nm"])*fraction**g["profile_power"]
            layers.append(Layer(upper, lower, core,
                                g["period_nm"] if kind == "valley" else core+2*g["gold_side_thickness_x_nm"],
                                "gold" if kind == "valley" else "air", kind))
    return layers


def preview(config, path, slices):
    g = config["geometry"]
    layers = build_layers(g, slices)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.stem+"_layers.csv").open("w", encoding="utf-8", newline="") as handle:
        names = [*asdict(layers[0]), "thickness_nm"]
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for layer in layers:
            writer.writerow({**asdict(layer), "thickness_nm": layer.bottom_depth_nm-layer.top_depth_nm})
    p, h, cap = g["period_nm"], g["height_nm"], g["gold_cap_thickness_nm"]
    sx, sy, top_y = 820/(3*p), 410/(h+cap), 160
    base_y = top_y+(h+cap)*sy
    air, gold, pmma = "#e8f4fb", "#e2ad33", "#4b9bca"
    def rect(x, y, width, height, fill, stroke="none"):
        return f'<rect x="{x:.3f}" y="{y:.3f}" width="{width:.3f}" height="{height:.3f}" fill="{fill}" stroke="{stroke}"/>'
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1060" height="750" viewBox="0 0 1060 750">',
             '<rect width="1060" height="750" fill="#f8fafc"/>', '<g font-family="Arial, sans-serif" fill="#182536">',
             '<text x="40" y="42" font-size="24">Au-coated PMMA grating: x periodic, y invariant</text>',
             f'<text x="40" y="74" font-size="15">P={p:g} nm | PMMA H={h:g} nm | core width={g["top_width_nm"]:g}..{g["bottom_width_nm"]:g} nm</text>',
             f'<text x="40" y="99" font-size="15">Au side (x): {g["gold_side_thickness_x_nm"]:g} nm | cap: {cap:g} nm x {g["gold_cap_width_nm"]:g} nm | valley: {g["gold_valley_thickness_nm"]:g} nm</text>',
             f'<text x="40" y="124" font-size="15">{slices} PMMA profile layers + {int(cap > 0)} Au cap layer = {len(layers)} finite layers</text>',
             rect(90, top_y, 820, base_y-top_y, air), rect(90, base_y, 820, 65, pmma)]
    for i, (name, color) in enumerate((("PMMA", pmma), ("Au", gold), ("Air", air))):
        y = 190+i*30
        parts.extend([rect(935, y-12, 17, 17, color, "#75869a"),
                      f'<text x="960" y="{y+1}" font-size="13">{name}</text>'])
    for cell in range(3):
        center = 90+(cell+.5)*p*sx
        for layer in layers:
            y = top_y+(layer.top_depth_nm+cap)*sy
            thickness = (layer.bottom_depth_nm-layer.top_depth_nm)*sy
            parts.append(rect(center-layer.outer_width_nm*sx/2, y, layer.outer_width_nm*sx, thickness, gold))
            if layer.core_width_nm:
                parts.append(rect(center-layer.core_width_nm*sx/2, y, layer.core_width_nm*sx, thickness, pmma))
        parts.append(rect(center-p*sx/2, top_y, p*sx, base_y-top_y+65, "none", "#75869a"))
    parts.extend([f'<text x="330" y="{base_y+41:.1f}" font-size="19">Semi-infinite lossless PMMA substrate</text>',
                  '<text x="95" y="682" font-size="15">Infinite extrusion in y. TE: E along y; TM: E along x. Light enters from air above.</text>',
                  '<text x="95" y="710" font-size="15">Drawing shows the actual midpoint staircase. Horizontal/vertical scales differ.</text>',
                  '</g></svg>'])
    path.write_text("\n".join(parts), encoding="utf-8")
    return layers
