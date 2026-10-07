"""One-dimensional structures in Figs. 5, 8 and 10 of Vallius (2002).

All lengths use the paper's normalized period d=1. Figure 5 lateral
dimensions are inferred from the drawing, not specified in its caption.
"""

from __future__ import annotations

from dataclasses import asdict
import math

from .solver import LayerSpec


PAPER_DOI = "10.1364/OE.10.000024"
FIGURES = {
    6: {"geometry": "metal", "polarization": "TE", "modes": (5, 10), "range": (0.95, 1.05)},
    7: {"geometry": "metal", "polarization": "TM", "modes": (12, 24), "range": (0.95, 1.05)},
    9: {"geometry": "checkerboard", "polarization": "TM", "modes": (19, 38), "range": (1.16, 1.18)},
    11: {"geometry": "cylinder", "polarization": "TE", "modes": (7, 14), "range": (0.90, 0.96)},
}


def periodic_patch(center: float, width: float, inside: complex, outside: complex):
    """Return exact physical intervals for a patch crossing the cell boundary."""
    if not 0 < width < 1:
        raise ValueError("Patch width must lie strictly between 0 and the period.")
    left = (center - width / 2) % 1.0
    right = (center + width / 2) % 1.0
    edges = sorted({0.0, left, right, 1.0})
    values = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        point = (lo + hi) / 2
        distance = ((point - center + 0.5) % 1.0) - 0.5
        values.append(complex(inside if abs(distance) < width / 2 else outside))
    return tuple(edges), tuple(values)


def metal_layers(widths=(0.3, 13 / 30, 17 / 30, 0.7), center=0.25):
    if len(widths) != 4:
        raise ValueError("Figure 5 requires four layer widths.")
    epsilon_metal = (0.1217 + 3.2966j) ** 2
    layers = []
    for j, width in enumerate(widths):
        breaks, epsilon = periodic_patch(center, float(width), epsilon_metal, 1.0)
        layers.append(LayerSpec(thickness=0.125, breaks=breaks, epsilon=epsilon,
                                period=1.0, name=f"metal_{j + 1}"))
    return layers


def checkerboard_layers():
    return [
        LayerSpec(thickness=10.0, breaks=(0.0, 0.5, 1.0), epsilon=(25.0, 2.25),
                  period=1.0, name="checkerboard_lower"),
        LayerSpec(thickness=10.0, breaks=(0.0, 0.5, 1.0), epsilon=(2.25, 25.0),
                  period=1.0, name="checkerboard_upper"),
    ]


def cylinder_layers(slices=120):
    """Midpoint staircase of the two staggered air cylinders in Fig. 10."""
    if isinstance(slices, bool) or int(slices) != slices or slices < 2:
        raise ValueError("slices must be an integer >= 2.")
    layers = []
    for j in range(slices):
        z = (j + 0.5) / slices
        cx, cz = (0.25, 0.25) if z < 0.5 else (0.75, 0.75)
        halfwidth = math.sqrt(max(0.25 ** 2 - (z - cz) ** 2, 0.0))
        breaks, epsilon = periodic_patch(cx, 2 * halfwidth, 1.0, 25.0)
        layers.append(LayerSpec(thickness=1.0 / slices, breaks=breaks,
                                epsilon=epsilon, period=1.0, name=f"cylinder_{j + 1}"))
    return layers


def geometry_for_figure(figure, *, slices=120, metal_widths=None):
    if figure not in FIGURES:
        raise ValueError(f"Unsupported figure: {figure}")
    if figure in (6, 7):
        return metal_layers() if metal_widths is None else metal_layers(metal_widths)
    if figure == 9:
        return checkerboard_layers()
    return cylinder_layers(slices)


def geometry_metadata(layers):
    """JSON-safe complete physical geometry, including the complex material."""
    result = []
    for layer in layers:
        item = asdict(layer)
        item["epsilon"] = [{"real": complex(v).real, "imag": complex(v).imag}
                           for v in layer.epsilon]
        result.append(item)
    return result
