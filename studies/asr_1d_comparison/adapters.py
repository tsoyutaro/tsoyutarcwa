"""Preserve each study's geometry, dispersion, incident side and port power."""
from __future__ import annotations

import copy
import gc
import math
import time
from pathlib import Path

from paper_reproductions.vallius2002.solver import LayerSpec, PreparedStack

ROOT = Path(__file__).resolve().parents[2]
STUDIES = ("gold_grating_1d", "pmma_gold_grating_1d")
METRICS = {
    "gold_grating_1d": ("reflectance", "power_into_substrate", "relief_absorptance"),
    "pmma_gold_grating_1d": ("reflectance", "transmittance", "absorptance"),
}


def load_study(study, config_path=None):
    if study not in STUDIES:
        raise ValueError(f"Unknown study: {study}")
    path = Path(config_path or ROOT / "studies" / study / "config.json").resolve()
    if study == "gold_grating_1d":
        from studies.gold_grating_1d.common import load_config
    else:
        from studies.pmma_gold_grating_1d.common import load_config
    config, materials = load_config(path)
    return config, materials, path


def nested_intervals(core_fill, outer_fill, core, shell, background):
    """Centered nested rectangles, with exact intervals in the unit period."""
    if not 0 <= core_fill <= outer_fill <= 1:
        raise ValueError("Require 0 <= core width <= outer width <= period")
    edges = sorted({0., 1., *(v for fill in (core_fill, outer_fill) if 0 < fill < 1
                            for v in ((1-fill)/2, (1+fill)/2))})
    values = []
    for left, right in zip(edges[:-1], edges[1:]):
        distance = abs((left+right)/2-.5)
        value = core if distance < core_fill/2 else (
            shell if distance < outer_fill/2 else background)
        values.append(complex(value))
    return tuple(edges), tuple(values)


def layer_specs(study, config, slices, materials):
    """Incident-side-first layers; all lengths are divided by the period."""
    if isinstance(slices, bool) or int(slices) != slices or slices < 1:
        raise ValueError("slices must be a positive integer")
    slices = int(slices)
    geometry = config["geometry"]
    period = geometry["period_nm"]
    result = []
    if study == "gold_grating_1d":
        gold = complex(materials)
        for index in range(slices):
            depth = (index+.5)/slices
            width = geometry["top_width_nm"] + (
                geometry["bottom_width_nm"]-geometry["top_width_nm"]) * depth**geometry["profile_power"]
            edges, epsilon = nested_intervals(0., width/period, gold, gold, 1.)
            result.append(LayerSpec(geometry["height_nm"]/slices/period, edges,
                                    epsilon, name=f"gold_{index+1}"))
        substrate = gold
    elif study == "pmma_gold_grating_1d":
        from studies.pmma_gold_grating_1d.geometry import build_layers
        pmma, gold = map(complex, materials)
        for index, layer in enumerate(build_layers(geometry, slices)):
            edges, epsilon = nested_intervals(layer.core_width_nm/period,
                                             layer.outer_width_nm/period,
                                             pmma, gold, gold if layer.background == "gold" else 1.)
            result.append(LayerSpec((layer.bottom_depth_nm-layer.top_depth_nm)/period,
                                    edges, epsilon, name=f"{layer.kind}_{index+1}"))
        substrate = pmma
    else:
        raise ValueError(f"Unknown study: {study}")
    return result, substrate


def scalar_observables(study, solution):
    r, p = float(solution["R"]), float(solution["T"])
    if study == "gold_grating_1d":
        # Include all interface-flux harmonics in absorbing Au. Evanescent
        # incident-air harmonics can transfer power into the lossy substrate.
        return dict(reflectance=r, power_into_substrate=p,
                    relief_absorptance=1-r-p, absorptance_total=1-r,
                    transmittance_far=0.)
    return dict(reflectance=r, transmittance=p, absorptance=1-r-p)


def scalar_case(study, config, slices, order, wavelength, materials, *, method="asr",
                oversampling=3, G=.001, quadrature=192, device="cpu", backend="auto",
                diagnostics=False):
    """TE and TM share the prepared material/map matrices, then solve separately."""
    if not math.isfinite(float(order)) or int(order) != order or order < 0:
        raise ValueError("order must be a nonnegative integer")
    layers, substrate = layer_specs(study, config, slices, materials)
    start = time.perf_counter()
    stack = PreparedStack(layers, method=method, harmonics=2*int(order)+1,
                          oversampling=oversampling if method == "asr" else 1,
                          G=G, quadrature=quadrature, epsilon_in=1., epsilon_out=substrate,
                          device=device, backend=backend, diagnostics=diagnostics,
                          retention="smallest_abs", q_projection="direct")
    polarizations, audit = {}, {}
    for polarization in ("TE", "TM"):
        stack.polarization = polarization
        solution = stack.solve(wavelength/config["geometry"]["period_nm"])
        polarizations[polarization] = scalar_observables(study, solution)
        audit[polarization] = dict(max_boundary_condition=(
            float(solution["max_boundary_condition"]) if diagnostics else None),
            cutoff_regularized=bool(solution["cutoff_regularized"]))
    result = dict(order=int(order), harmonics=stack.harmonics, slices=slices,
                  total_finite_layers=len(layers), wavelength_nm=float(wavelength),
                  method=method, oversampling=oversampling if method == "asr" else 1,
                  eigen_dimension=int(stack.internal_count), G=G if method == "asr" else None,
                  quadrature_actual=max(layer.quadrature_points for layer in stack.prepared_layers),
                  polarizations=polarizations, diagnostics=audit,
                  execution=dict(stack.execution), runtime_seconds=time.perf_counter()-start,
                  coefficient_method="adaptive quadrature" if method == "asr" else "analytic intervals")
    del stack
    gc.collect()
    return result


def existing_li_case(study, config, slices, order, wavelength, materials, device):
    """Call the actual study solver, forcing its existing analytic Li path."""
    local = copy.deepcopy(config)
    local["solver"]["fourier_coefficients"] = "analytic"
    if study == "gold_grating_1d":
        from studies.gold_grating_1d.solver import simulate
        result = simulate(local, dict(order=order, slices=slices,
                                     grid=config["fixed_numerics"]["grid"]),
                          wavelength, materials, device)
    else:
        from studies.pmma_gold_grating_1d.solver import simulate
        result = simulate(local, dict(order=order, slices=slices), wavelength, materials, device)
    result.update(method="li", oversampling=1, harmonics=2*order+1,
                  eigen_dimension=2*order+1)
    return result
