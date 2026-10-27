"""Pure-Python geometry shared by the RCWA stack and its preflight drawing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProfileLayer:
    top_depth_nm: float  # Depth measured downward from the uncoated PMMA tip.
    bottom_depth_nm: float
    core_radius_nm: float
    outer_radius_nm: float | None  # None means Au fills outside the PMMA core.
    kind: str  # "shell" or "valley"


def build_profile_layers(
    *,
    height_nm: float,
    tip_radius_nm: float,
    base_radius_nm: float,
    gold_thickness_nm: float,
    profile_power: float,
    slices: int,
    valley_gold_thickness_nm: float,
) -> list[ProfileLayer]:
    """Return the exact stepped geometry passed to the patterned RCWA layers."""
    dimensions = (
        height_nm, tip_radius_nm, base_radius_nm, gold_thickness_nm, profile_power
    )
    if any(not math.isfinite(value) or value <= 0 for value in dimensions):
        raise ValueError("geometry dimensions and profile power must be positive")
    if base_radius_nm < tip_radius_nm or slices <= 0:
        raise ValueError("invalid radii or profile slice count")
    if (
        not math.isfinite(valley_gold_thickness_nm)
        or not 0 <= valley_gold_thickness_nm <= height_nm
    ):
        raise ValueError("valley gold thickness must be within [0, height]")

    valley_start_depth = height_nm - valley_gold_thickness_nm
    if 0 < valley_gold_thickness_nm < height_nm:
        if slices < 2:
            raise ValueError("a partial valley coating needs at least two profile slices")
        valley_slices = max(
            1, min(slices - 1, round(slices * valley_gold_thickness_nm / height_nm))
        )
    else:
        valley_slices = slices if valley_gold_thickness_nm == height_nm else 0
    shell_slices = slices - valley_slices

    # Keep exactly `slices` profile layers and align one boundary with the valley Au.
    regions = (
        (0.0, valley_start_depth, shell_slices, "shell"),
        (valley_start_depth, height_nm, valley_slices, "valley"),
    )
    layers: list[ProfileLayer] = []
    for region_top, region_bottom, region_slices, kind in regions:
        for layer in range(region_slices):
            segment_top = region_top + (region_bottom - region_top) * layer / region_slices
            segment_bottom = region_top + (region_bottom - region_top) * (layer + 1) / region_slices
            midpoint_fraction = (segment_top + segment_bottom) / (2 * height_nm)
            core = tip_radius_nm + (base_radius_nm - tip_radius_nm) * (
                midpoint_fraction**profile_power
            )
            layers.append(
                ProfileLayer(
                    top_depth_nm=segment_top,
                    bottom_depth_nm=segment_bottom,
                    core_radius_nm=core,
                    outer_radius_nm=None if kind == "valley" else core + gold_thickness_nm,
                    kind=kind,
                )
            )
    return layers


def _layer_at_height(
    layers: list[ProfileLayer], height_nm: float, z_nm: float
) -> ProfileLayer:
    depth = height_nm - z_nm
    for layer in layers:
        if layer.top_depth_nm <= depth < layer.bottom_depth_nm:
            return layer
    return layers[-1]


def write_geometry_svg(
    path: Path,
    *,
    layers: list[ProfileLayer],
    period_nm: float,
    height_nm: float,
    tip_radius_nm: float,
    base_radius_nm: float,
    gold_thickness_nm: float,
    valley_gold_thickness_nm: float,
) -> None:
    """Draw the actual staircase stack, with two triangular-lattice plan views."""
    if not layers or period_nm <= 0 or not math.isfinite(period_nm):
        raise ValueError("invalid geometry for preview")
    cap_radius = tip_radius_nm + gold_thickness_nm
    air, pmma, gold = "#e8f4fb", "#4b9bca", "#e2ad33"
    ink, muted = "#182536", "#556779"
    panel_top, panel_bottom = 160.0, 690.0
    vertical_scale = (panel_bottom - panel_top) / (height_nm + gold_thickness_nm)
    lateral_scale = 1.0
    cell_center = 260.0
    cell_left = cell_center - period_nm * lateral_scale / 2

    def x(radius_nm: float) -> float:
        return cell_center + radius_nm * lateral_scale

    def y(z_nm: float) -> float:
        return panel_bottom - z_nm * vertical_scale

    def rect(x0: float, y0: float, width: float, height: float, fill: str, **extra) -> str:
        attrs = " ".join(f'{key.replace("_", "-")}="{value}"' for key, value in extra.items())
        return (
            f'<rect x="{x0:.3f}" y="{y0:.3f}" width="{width:.3f}" '
            f'height="{height:.3f}" fill="{fill}" {attrs}/>'
        )

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1240" height="820" '
        'viewBox="0 0 1240 820" role="img" '
        'aria-label="Au-coated PMMA moth-eye RCWA geometry preview">',
        rect(0, 0, 1240, 820, "#f8fafc"),
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#182536}'
        '.title{font-size:25px;font-weight:700}.head{font-size:17px;font-weight:700}'
        '.label{font-size:14px}.small{font-size:12px;fill:#556779}</style>',
        '<text class="title" x="55" y="58">PMMA moth-eye + Au — pre-solve geometry</text>',
        f'<text class="label" x="55" y="88">Triangular lattice P={period_nm:g} nm · '
        f'height={height_nm:g} nm · {len(layers)} profile layers + 1 top disk</text>',
        '<text class="head" x="130" y="132">Cross-section through one lattice site</text>',
        rect(cell_left, panel_top, period_nm * lateral_scale, panel_bottom - panel_top, air),
        rect(cell_left, panel_bottom, period_nm * lateral_scale, 68, pmma),
    ]
    for layer in layers:
        upper_z = height_nm - layer.top_depth_nm
        lower_z = height_nm - layer.bottom_depth_nm
        top_y, bottom_y = y(upper_z), y(lower_z)
        thickness = bottom_y - top_y
        core = layer.core_radius_nm
        if layer.kind == "valley":
            parts.append(rect(cell_left, top_y, period_nm * lateral_scale, thickness, gold))
        else:
            outer = layer.outer_radius_nm
            assert outer is not None
            parts.append(rect(x(-outer), top_y, (outer - core) * lateral_scale, thickness, gold))
            parts.append(rect(x(core), top_y, (outer - core) * lateral_scale, thickness, gold))
        parts.append(rect(x(-core), top_y, 2 * core * lateral_scale, thickness, pmma))
    parts.extend([
        rect(x(-cap_radius), y(height_nm + gold_thickness_nm),
             2 * cap_radius * lateral_scale, gold_thickness_nm * vertical_scale, gold),
        rect(cell_left, panel_top, period_nm * lateral_scale, panel_bottom - panel_top + 68,
             "none", stroke="#75869a", stroke_width="1.3"),
        f'<text class="small" x="128" y="{y(height_nm + gold_thickness_nm)-5:.1f}">'
        f'z={height_nm + gold_thickness_nm:g} nm</text>',
        f'<text class="small" x="85" y="{y(height_nm)+4:.1f}">z={height_nm:g} nm</text>',
        f'<text class="small" x="100" y="{panel_bottom+4:.1f}">z=0</text>',
        f'<text class="label" x="170" y="790">PMMA substrate</text>',
        f'<text class="small" x="385" y="195">Au top disk: '
        f'D={2*cap_radius:g}, t={gold_thickness_nm:g} nm</text>',
        f'<text class="small" x="385" y="575">Sidewall Au: radial '
        f'{gold_thickness_nm:g} nm</text>',
        f'<text class="small" x="385" y="605">Valley Au: '
        f'{valley_gold_thickness_nm:g} nm</text>',
        f'<text class="small" x="385" y="635">PMMA core: '
        f'D={2*tip_radius_nm:g} to {2*base_radius_nm:g} nm (stepped)</text>',
        '<text class="small" x="385" y="672">Horizontal and vertical scales differ.</text>',
    ])
    if valley_gold_thickness_nm > 0:
        valley_line = y(valley_gold_thickness_nm)
        parts.append(
            f'<line x1="{cell_left:.1f}" y1="{valley_line:.1f}" '
            f'x2="{cell_left+period_nm:.1f}" y2="{valley_line:.1f}" '
            'stroke="#854d0e" stroke-width="1.3" stroke-dasharray="4,4"/>'
        )

    def plan_view(cx: float, cy: float, z_nm: float, heading: str) -> None:
        layer = _layer_at_height(layers, height_nm, z_nm)
        scale = 0.82
        hex_radius = period_nm / math.sqrt(3) * scale
        half_width = period_nm * scale / 2
        points = [
            (cx, cy - hex_radius),
            (cx + half_width, cy - hex_radius / 2),
            (cx + half_width, cy + hex_radius / 2),
            (cx, cy + hex_radius),
            (cx - half_width, cy + hex_radius / 2),
            (cx - half_width, cy - hex_radius / 2),
        ]
        polygon = " ".join(f"{px:.2f},{py:.2f}" for px, py in points)
        background = gold if layer.kind == "valley" else air
        parts.append(f'<text class="head" x="{cx-110:.1f}" y="{cy-145:.1f}">{heading}</text>')
        parts.append(f'<text class="small" x="{cx-110:.1f}" y="{cy-125:.1f}">z={z_nm:g} nm above valley</text>')
        parts.append(
            f'<polygon points="{polygon}" fill="{background}" '
            'stroke="#75869a" stroke-width="1.5"/>'
        )
        if layer.outer_radius_nm is not None:
            parts.append(
                f'<circle cx="{cx:.2f}" cy="{cy:.2f}" '
                f'r="{layer.outer_radius_nm*scale:.2f}" fill="{gold}"/>'
            )
        parts.append(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" '
            f'r="{layer.core_radius_nm*scale:.2f}" fill="{pmma}"/>'
        )
        parts.append(
            f'<text class="small" x="{cx-110:.1f}" y="{cy+133:.1f}">'
            f'PMMA core radius {layer.core_radius_nm:.1f} nm</text>'
        )
        parts.append(
            f'<text class="small" x="{cx-110:.1f}" y="{cy+152:.1f}">'
            f'{"Au across exposed valley" if layer.kind == "valley" else "Au shell, then air"}</text>'
        )

    lower_view_z = valley_gold_thickness_nm / 2 if valley_gold_thickness_nm else min(15.0, height_nm / 4)
    plan_view(740, 360, lower_view_z, "Plan view: near valley")
    plan_view(1025, 360, height_nm * 0.5, "Plan view: mid-height")
    parts.extend([
        '<text class="small" x="652" y="585">Hexagon: one triangular-lattice Wigner–Seitz cell</text>',
        rect(680, 635, 21, 21, pmma),
        '<text class="label" x="710" y="651">PMMA</text>',
        rect(815, 635, 21, 21, gold),
        '<text class="label" x="845" y="651">Au</text>',
        rect(930, 635, 21, 21, air, stroke="#75869a"),
        '<text class="label" x="960" y="651">Air</text>',
        '<text class="small" x="650" y="700">Staircase boundaries match the RCWA layer input.</text>',
        '<text class="small" x="650" y="722">Au on the valley surrounds the PMMA core; it does not sit beneath it.</text>',
        '</svg>',
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def write_geometry_png(
    path: Path,
    *,
    layers: list[ProfileLayer],
    period_nm: float,
    height_nm: float,
    tip_radius_nm: float,
    base_radius_nm: float,
    gold_thickness_nm: float,
    valley_gold_thickness_nm: float,
) -> bool:
    """Write an optional raster companion; SVG remains available without Pillow."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return False

    air, pmma, gold = "#e8f4fb", "#4b9bca", "#e2ad33"
    ink, muted = "#182536", "#556779"
    image = Image.new("RGB", (1240, 820), "#f8fafc")
    draw = ImageDraw.Draw(image)
    font_dir = Path("C:/Windows/Fonts")

    def font(size: int, bold: bool = False):
        filename = "arialbd.ttf" if bold else "arial.ttf"
        try:
            return ImageFont.truetype(str(font_dir / filename), size)
        except OSError:
            return ImageFont.load_default()

    title_font, head_font = font(25, True), font(17, True)
    label_font, small_font = font(14), font(12)
    draw.text((55, 32), "PMMA moth-eye + Au - pre-solve geometry", font=title_font, fill=ink)
    draw.text(
        (55, 70),
        f"Triangular lattice P={period_nm:g} nm | height={height_nm:g} nm | "
        f"{len(layers)} profile layers + 1 top disk",
        font=label_font, fill=ink,
    )
    draw.text((130, 110), "Cross-section through one lattice site", font=head_font, fill=ink)
    center, y0 = 260.0, 690.0
    scale_z = 530.0 / (height_nm + gold_thickness_nm)
    left, right = center - period_nm / 2, center + period_nm / 2

    def y(z_nm: float) -> float:
        return y0 - z_nm * scale_z

    draw.rectangle((left, 160, right, y0), fill=air)
    draw.rectangle((left, y0, right, 758), fill=pmma)
    for layer in layers:
        top = y(height_nm - layer.top_depth_nm)
        bottom = y(height_nm - layer.bottom_depth_nm)
        core = layer.core_radius_nm
        if layer.kind == "valley":
            draw.rectangle((left, top, right, bottom), fill=gold)
        else:
            outer = layer.outer_radius_nm
            assert outer is not None
            draw.rectangle((center - outer, top, center - core, bottom), fill=gold)
            draw.rectangle((center + core, top, center + outer, bottom), fill=gold)
        draw.rectangle((center - core, top, center + core, bottom), fill=pmma)
    cap = tip_radius_nm + gold_thickness_nm
    draw.rectangle((center - cap, y(height_nm + gold_thickness_nm),
                    center + cap, y(height_nm)), fill=gold)
    draw.rectangle((left, 160, right, 758), outline="#75869a", width=2)
    if valley_gold_thickness_nm > 0:
        vy = y(valley_gold_thickness_nm)
        for xx in range(int(left), int(right), 9):
            draw.line((xx, vy, min(xx + 5, right), vy), fill="#854d0e", width=2)
    draw.text((120, 138), f"z={height_nm + gold_thickness_nm:g} nm", font=small_font, fill=muted)
    draw.text((83, y(height_nm) - 9), f"z={height_nm:g} nm", font=small_font, fill=muted)
    draw.text((115, y0 - 7), "z=0", font=small_font, fill=muted)
    draw.text((170, 770), "PMMA substrate", font=label_font, fill=ink)
    draw.text((385, 178), f"Au top disk: D={2*cap:g}, t={gold_thickness_nm:g} nm",
              font=small_font, fill=muted)
    draw.text((385, 558), f"Sidewall Au: radial {gold_thickness_nm:g} nm",
              font=small_font, fill=muted)
    draw.text((385, 588), f"Valley Au: {valley_gold_thickness_nm:g} nm",
              font=small_font, fill=muted)
    draw.text((385, 618),
              f"PMMA core: {2*tip_radius_nm:g} to {2*base_radius_nm:g} nm diameter",
              font=small_font, fill=muted)
    draw.text((385, 655), "Horizontal and vertical scales may differ.",
              font=small_font, fill=muted)

    def plan(cx: float, cy: float, z_nm: float, heading: str) -> None:
        layer = _layer_at_height(layers, height_nm, z_nm)
        scale = 0.82
        radius = period_nm / math.sqrt(3) * scale
        half_width = period_nm * scale / 2
        points = [
            (cx, cy-radius), (cx+half_width, cy-radius/2),
            (cx+half_width, cy+radius/2), (cx, cy+radius),
            (cx-half_width, cy+radius/2), (cx-half_width, cy-radius/2),
        ]
        draw.text((cx - 110, cy - 168), heading, font=head_font, fill=ink)
        draw.text((cx - 110, cy - 145), f"z={z_nm:g} nm above valley",
                  font=small_font, fill=muted)
        draw.polygon(points, fill=gold if layer.kind == "valley" else air)
        draw.line(points + [points[0]], fill="#75869a", width=2)
        if layer.outer_radius_nm is not None:
            r = layer.outer_radius_nm * scale
            draw.ellipse((cx-r, cy-r, cx+r, cy+r), fill=gold)
        r = layer.core_radius_nm * scale
        draw.ellipse((cx-r, cy-r, cx+r, cy+r), fill=pmma)
        draw.text((cx-110, cy+125), f"PMMA core radius {layer.core_radius_nm:.1f} nm",
                  font=small_font, fill=muted)
        draw.text((cx-110, cy+145),
                  "Au across exposed valley" if layer.kind == "valley" else "Au shell, then air",
                  font=small_font, fill=muted)

    lower_z = valley_gold_thickness_nm / 2 if valley_gold_thickness_nm else min(15.0, height_nm / 4)
    plan(740, 360, lower_z, "Plan view: near valley")
    plan(1025, 360, height_nm * 0.5, "Plan view: mid-height")
    draw.text((652, 570), "Hexagon: one triangular-lattice Wigner-Seitz cell",
              font=small_font, fill=muted)
    for xx, color, label in ((680, pmma, "PMMA"), (815, gold, "Au"), (930, air, "Air")):
        draw.rectangle((xx, 635, xx+21, 656), fill=color, outline="#75869a")
        draw.text((xx+30, 638), label, font=label_font, fill=ink)
    draw.text((650, 690), "Staircase boundaries match the RCWA layer input.",
              font=small_font, fill=muted)
    draw.text((650, 712), "Valley Au surrounds the PMMA core; it does not sit beneath it.",
              font=small_font, fill=muted)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return True
