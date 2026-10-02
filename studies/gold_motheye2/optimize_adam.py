"""Adam pilot for a monotone Au moth-eye profile (no Optuna).

Run from the project root: python3 studies/gold_motheye2/optimize_adam.py --device cuda
Use --prepare-only to inspect the geometry before any electromagnetic solve.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VERSION = "au_adam_monotone_variable_endpoints_v2"
DEFAULT_GOLD_CSV = HERE / "data" / "au_measured_nk.csv"
DEFAULT_OUTPUT = HERE / "results" / "adam_variable_endpoints_Nz100_M8"
LOGIT_LIMIT = 10.0


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_data_hash(path: Path) -> str:
    """Match the Linux CSV bytes even when copied to Windows with CRLF."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def wavelengths_from_text(raw: str) -> tuple[float, ...]:
    if ":" in raw:
        start, stop, step = (float(part) for part in raw.split(":"))
        if not all(math.isfinite(value) for value in (start, stop, step)) or step <= 0 or stop <= start:
            raise ValueError("Invalid wavelength range.")
        count = int(math.floor((stop-start)/step+1e-10))
        values = [start+i*step for i in range(count+1)]
        if values[-1] < stop-1e-9:
            values.append(stop)
    else:
        values = [float(part) for part in raw.split(",")]
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Wavelengths must be finite and positive.")
    values = tuple(sorted(set(values)))
    if len(values) < 2:
        raise ValueError("At least two wavelengths are needed for a band average.")
    return values


def trapezoid_weights(wavelengths: tuple[float, ...]) -> tuple[float, ...]:
    span = wavelengths[-1] - wavelengths[0]
    if span <= 0:
        raise ValueError("Wavelengths must be distinct.")
    weights = []
    for index in range(len(wavelengths)):
        left = 0.0 if index == 0 else (wavelengths[index] - wavelengths[index-1]) / 2
        right = 0.0 if index == len(wavelengths)-1 else (wavelengths[index+1] - wavelengths[index]) / 2
        weights.append((left + right) / span)
    return tuple(weights)


def _softmax_numbers(logits: list[float]) -> list[float]:
    maximum = max(logits)
    numbers = [math.exp(value - maximum) for value in logits]
    total = sum(numbers)
    floor = min(1e-3, 0.1/len(logits))
    return [floor + (1-len(logits)*floor)*value/total for value in numbers]


def _sigmoid_number(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    number = math.exp(value)
    return number / (1.0 + number)


def initial_logits(config: dict) -> list[float]:
    """Initialize the trainable endpoints and shape to the existing cone."""
    geometry = config["geometry"]
    period = geometry["period_nm"]
    margin = config["diameter_margin_nm"]
    lower, upper = margin / 2, (period - margin) / 2
    tip = geometry["tip_radius_nm"]
    base = geometry["base_radius_nm"]
    tip_fraction = (tip - lower) / (upper - lower)
    span_fraction = (base - tip) / (upper - tip)
    if not (0 < tip_fraction < 1 and 0 < span_fraction < 1):
        raise ValueError("Initial cone endpoints do not fit inside the diameter bounds.")
    logit = lambda value: math.log(value / (1 - value))
    return [logit(tip_fraction), logit(span_fraction)] + [0.0] * config["segments"]


def endpoint_values(logits: list[float], config: dict) -> tuple[float, float]:
    period = config["geometry"]["period_nm"]
    margin = config["diameter_margin_nm"]
    lower, upper = margin / 2, (period - margin) / 2
    tip = lower + (upper - lower) * _sigmoid_number(logits[0])
    base = tip + (upper - tip) * _sigmoid_number(logits[1])
    return tip, base


def radius_values(logits: list[float], slices: int, config: dict) -> list[float]:
    """Sample strictly increasing piecewise-linear radii at layer midpoints."""
    tip, base = endpoint_values(logits, config)
    widths = _softmax_numbers(logits[2:])
    cumulative = [0.0]
    for width in widths:
        cumulative.append(cumulative[-1] + width)
    count = len(widths)
    result = []
    for index in range(slices):
        location = (index + 0.5) * count / slices
        segment = min(int(location), count - 1)
        fraction = location - segment
        result.append(tip + (base - tip) *
                      (cumulative[segment] + fraction * widths[segment]))
    return result


def radius_tensor(logits, slices: int, config: dict, torch):
    period = config["geometry"]["period_nm"]
    margin = config["diameter_margin_nm"]
    lower, upper = margin / 2, (period - margin) / 2
    tip = lower + (upper - lower) * torch.sigmoid(logits[0])
    base = tip + (upper - tip) * torch.sigmoid(logits[1])
    count = logits.numel() - 2
    floor = min(1e-3, 0.1/count)
    widths = floor + (1-count*floor)*torch.softmax(logits[2:], dim=0)
    cumulative = torch.cat((widths.new_zeros(1), torch.cumsum(widths, dim=0)))
    # Python indices are fixed by the layer positions. Tensor arithmetic keeps
    # the differentiable path from every radius to the control variables.
    radii = []
    for index in range(slices):
        location = (index + 0.5) * count / slices
        segment = min(int(location), count - 1)
        radii.append(tip + (base - tip) *
                     (cumulative[segment] + (location - segment) * widths[segment]))
    return radii


def save_profile(output: Path, name: str, logits: list[float], config: dict) -> None:
    geometry = config["geometry"]
    slices = config["slices"]
    height = geometry["height_nm"]
    tip, base = endpoint_values(logits, config)
    radii = radius_values(logits, slices, config)
    increasing = all(left < right for left, right in zip(radii, radii[1:]))
    if not increasing:
        raise RuntimeError(f"{name} profile does not increase strictly from top to bottom.")
    write_json(output / f"{name}_geometry.json", {
        "tip_diameter_nm": 2*tip, "base_diameter_nm": 2*base,
        "minimum_allowed_diameter_nm": config["diameter_margin_nm"],
        "maximum_allowed_diameter_nm": geometry["period_nm"]-config["diameter_margin_nm"],
        "strictly_increasing_layer_radii": increasing,
    })
    path = output / f"{name}_profile.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("layer_top_to_bottom", "z_midpoint_nm", "radius_nm", "diameter_nm"))
        for index, radius in enumerate(radii):
            writer.writerow((index+1, (index+0.5)*height/slices, radius, 2*radius))
    # Cross section of the actual 100 cylindrical slices, rather than an
    # interpolated outline that could hide the discretization.
    width, canvas_h = 720, 700
    cx, top, bottom = 360, 90, 630
    scale_x = 2.1
    layer_h = (bottom-top)/slices
    shapes = []
    for index, radius in enumerate(radii):
        x = cx-radius*scale_x
        shapes.append(f'<rect x="{x:.3f}" y="{top+index*layer_h:.3f}" '
                      f'width="{2*radius*scale_x:.3f}" height="{layer_h+0.05:.3f}" '
                      'fill="#d4aa00"/>')
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="700" viewBox="0 0 720 700">',
        '<rect width="720" height="700" fill="#f9fbfd"/>',
        f'<text x="24" y="34" font-size="22" fill="#233142">{name}: {slices}-layer Au profile</text>',
        f'<text x="24" y="60" font-size="14" fill="#445566">Period {geometry["period_nm"]:g} nm; height {height:g} nm; tip/base diameter {2*tip:.3f}/{2*base:.3f} nm</text>',
        '<rect x="160" y="630" width="400" height="28" fill="#b18b00"/>',
        *shapes,
        '<text x="570" y="114" font-size="15">air / tip</text>',
        '<text x="570" y="620" font-size="15">Au substrate</text>',
        '<text x="24" y="686" font-size="13" fill="#445566">Axial cross section; each rectangle is one constant-radius computational layer.</text>',
        '</svg>',
    ]
    (output / f"{name}_profile.svg").write_text("\n".join(lines), encoding="utf-8")


def _discard_reflectance_auxiliary(simulation):
    """Release diagnostics and field-reconstruction data in an R-only solve.

    Keep Cartesian/reduced modes, propagation and cascade state. Autograd
    keeps any tensors needed by a differentiable objective independently of
    these diagnostic containers.
    """
    if simulation.store_mode_couplings or simulation.field_regions != "none":
        raise RuntimeError("Cannot discard auxiliary data in a field-enabled solve.")
    for name in ("asr_mappings", "asr_material_tensors", "asr_T_matrices",
                 "asr_Tz_matrices", "asr_condition_numbers", "E_eigvec_uv",
                 "H_eigvec_uv"):
        getattr(simulation, name).clear()
    simulation._asr_slot_by_layer.clear()
    simulation._asr_field_context_by_layer.clear()
    simulation._physical_material_by_layer.clear()


def reflectance_tensor(wavelength_nm, radii, config, gold_model, device, torch,
                       *, discard_auxiliary=False):
    """Same measured-Au RCWA and flux definition as converge.py, with gradients."""
    from rcwa_solver_auto import (ASROptions, AutoRCWA, Circle, GroupTheoryOptions,
                                  Lattice, LayerSpec, Material, OutputSpec)
    from studies.gold_motheye.converge import _zero_order_x_source

    geometry = config["geometry"]
    period = geometry["period_nm"]
    epsilon_gold = gold_model(wavelength_nm)
    simulation = AutoRCWA(
        freq=period/wavelength_nm, order=[config["order"], config["order"]],
        lattice=Lattice.triangular(1.0), cascade="redheffer",
        outputs=OutputSpec(smatrix_size="half", fields="none"),
        asr=ASROptions(circle_G=geometry["asr_circle_g"],
                       grid=(config["grid"], config["grid"]), factorization_rules=True),
        group_theory=GroupTheoryOptions(enabled=True, symmetry="d6", strict=True,
                                        polarization="x"),
        verify_cascade=False, dtype=torch.complex128, device=device,
    )
    simulation.add_input_layer(eps=1.0, mu=1.0)
    simulation.add_output_layer(eps=epsilon_gold, mu=1.0)
    simulation.set_incident_angle(0.0, 0.0)
    thickness = geometry["height_nm"] / config["slices"] / period
    for index, radius_nm in enumerate(radii):
        simulation.add_structured_layer(LayerSpec(
            thickness=thickness, geometry=Circle(radius_nm/period),
            background=Material(1.0, 1.0), inclusion=Material(epsilon_gold, 1.0),
            method="matched-asr", factorization_rules=True,
            label=f"moth-eye-{index:03d}",
        ))
        if discard_auxiliary:
            _discard_reflectance_auxiliary(simulation)
    simulation.solve_global_smatrix()
    incident = _zero_order_x_source(simulation)
    reflected = simulation.S[1] @ incident

    def flux(electric, direction):
        magnetic = direction * (simulation.Vi @ electric)
        count = electric.numel() // 2
        return 0.5*torch.real(torch.sum(
            electric[:count]*torch.conj(magnetic[count:]) -
            electric[count:]*torch.conj(magnetic[:count])))

    incident_flux = flux(incident, 1)
    if float(incident_flux.detach().cpu()) <= 0:
        raise RuntimeError("Incident flux is not positive.")
    result = -flux(reflected, -1)/incident_flux
    if not bool(torch.isfinite(result).detach().cpu()):
        raise RuntimeError(f"Nonfinite reflectance at {wavelength_nm:g} nm.")
    return result


def evaluate(output: Path, checkpoint: dict, label: str, order: int,
             logits: list[float], config: dict, gold_model, device, torch) -> dict:
    key = f"{label}|M={order}|step={checkpoint['step']}"
    pending = checkpoint.get("pending")
    if pending is None:
        pending = {"key": key, "logits": logits, "values": {}}
        checkpoint["pending"] = pending
        write_json(output / "checkpoint.json", checkpoint)
    elif pending["key"] != key or pending["logits"] != logits:
        raise RuntimeError("Checkpoint has an unfinished different evaluation.")
    local_config = {**config, "order": order}
    for wavelength in config["wavelengths_nm"]:
        wl_key = f"{wavelength:g}"
        if wl_key in pending["values"]:
            continue
        # Matched-ASR differentiates its coordinate map internally even for a
        # fixed profile. no_grad() disables that Jacobian and breaks the solve.
        with torch.enable_grad():
            tensor = torch.tensor(logits, dtype=torch.float64, device=device)
            radii = radius_tensor(tensor, config["slices"], config, torch)
            result = reflectance_tensor(wavelength, radii, local_config,
                                        gold_model, device, torch)
            value = float(result.detach().cpu())
        if not (-1e-6 <= value <= 1+1e-6):
            raise RuntimeError(f"Nonphysical R={value:g} at {wavelength:g} nm, M={order}.")
        pending["values"][wl_key] = value
        write_json(output / "checkpoint.json", checkpoint)
        print(f"[{label} M={order}] {wavelength:g} nm R={value:.8f}", flush=True)
        del result, radii, tensor
        gc.collect()
    values = pending["values"]
    mean = sum(weight*values[f"{wavelength:g}"] for wavelength, weight
               in zip(config["wavelengths_nm"], config["weights"]))
    record = {"label": label, "order": order, "step": checkpoint["step"],
              "logits": logits, "mean_reflectance": mean, "values": values}
    checkpoint["evaluations"].append(record)
    checkpoint["pending"] = None
    write_json(output / "checkpoint.json", checkpoint)
    return record


def find_evaluation(checkpoint: dict, label: str, order: int,
                    logits: list[float], step: int | None = None) -> dict | None:
    for row in reversed(checkpoint["evaluations"]):
        if (row["label"] == label and row["order"] == order and
                row["logits"] == logits and (step is None or row["step"] == step)):
            return row
    return None


def save_results(output: Path, checkpoint: dict, config: dict) -> None:
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("step", "wavelength_nm", "reflectance",
                                                    "gradient_norm", "seconds"))
        writer.writeheader()
        writer.writerows(checkpoint["history"])
    with (output / "evaluations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("label", "order", "step", "wavelength_nm", "reflectance",
                         "band_mean_reflectance"))
        for row in checkpoint["evaluations"]:
            for wavelength in config["wavelengths_nm"]:
                writer.writerow((row["label"], row["order"], row["step"], wavelength,
                                 row["values"][f"{wavelength:g}"], row["mean_reflectance"]))
    best = checkpoint.get("best")
    if best:
        save_profile(output, "best", best["logits"], config)
    cone_logits = initial_logits(config)
    cone = find_evaluation(checkpoint, "cone", config["order"], cone_logits)
    verified_cone = find_evaluation(checkpoint, "verify_cone", config["verify_order"],
                                    cone_logits) if config["verify_order"] else None
    verified_best = find_evaluation(checkpoint, "verify_best", config["verify_order"],
                                    best["logits"]) if best and config["verify_order"] else None
    summary = {
        "status": "in_progress" if checkpoint["step"] < config["requested_steps"] else
                  "verified" if verified_cone and verified_best else "training_complete",
        "slices": config["slices"], "optimization_order": config["order"],
        "verification_order": config["verify_order"],
        "wavelengths_nm": config["wavelengths_nm"],
        "average_definition": "trapezoidal average over sampled wavelengths; fractions",
        "cone_mean_M_opt": cone["mean_reflectance"] if cone else None,
        "best_mean_M_opt": best["mean_reflectance"] if best else None,
        "best_step": best["step"] if best else None,
        "cone_tip_diameter_nm": 2*endpoint_values(cone_logits, config)[0],
        "cone_base_diameter_nm": 2*endpoint_values(cone_logits, config)[1],
        "best_tip_diameter_nm": 2*endpoint_values(best["logits"], config)[0] if best else None,
        "best_base_diameter_nm": 2*endpoint_values(best["logits"], config)[1] if best else None,
        "cone_mean_M_verify": verified_cone["mean_reflectance"] if verified_cone else None,
        "best_mean_M_verify": verified_best["mean_reflectance"] if verified_best else None,
        "verified_improvement": (verified_cone["mean_reflectance"]-
                                 verified_best["mean_reflectance"])
                                 if verified_cone and verified_best else None,
    }
    write_json(output / "summary.json", summary)
    if cone and best:
        panels = [("Adam order", cone, find_evaluation(checkpoint, "candidate",
                   config["order"], best["logits"]))]
        if best["logits"] == cone_logits:
            panels[0] = ("Adam order", cone, cone)
        if verified_cone and verified_best:
            panels.append(("Verification order", verified_cone, verified_best))
        draw_comparison(output / "comparison.svg", panels, config)
        draw_mean_history(output / "band_mean_history.svg", checkpoint, config)


def draw_comparison(path: Path, panels: list[tuple[str, dict, dict | None]], config: dict) -> None:
    wavelengths = config["wavelengths_nm"]
    width, height = 840, 350*len(panels)+50
    elements = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                '<rect width="100%" height="100%" fill="#f9fbfd"/>',
                '<text x="35" y="34" font-size="22" fill="#243447">Au moth-eye: cone and Adam profile</text>']
    for panel_index, (label, first, second) in enumerate(panels):
        top = 75+panel_index*350
        left, right, plot_top, plot_bottom = 85, 790, top+55, top+260
        rows = [first] + ([second] if second else [])
        maximum = max(float(row["values"][f"{wl:g}"]) for row in rows for wl in wavelengths)
        y_max = min(1.0, max(0.05, math.ceil(maximum*10)/10))
        for tick in range(6):
            y = plot_bottom - tick*(plot_bottom-plot_top)/5
            elements.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#d9e2ea"/>')
            elements.append(f'<text x="24" y="{y+5:.1f}" font-size="13" fill="#556677">{100*y_max*tick/5:.1f}%</text>')
        elements.append(f'<text x="{left}" y="{top+25}" font-size="19" fill="#243447">{label} M={first["order"]}</text>')
        for title, row, color in (("cone", first, "#1368aa"), ("Adam best", second, "#c26610")):
            if row is None:
                continue
            points = []
            for wl in wavelengths:
                x = left+(wl-wavelengths[0])/(wavelengths[-1]-wavelengths[0])*(right-left)
                y = plot_bottom-row["values"][f"{wl:g}"]/y_max*(plot_bottom-plot_top)
                points.append((x, y))
            elements.append('<polyline fill="none" stroke="'+color+'" stroke-width="2.8" points="'+
                            ' '.join(f'{x:.2f},{y:.2f}' for x, y in points)+'"/>')
            elements.extend(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>'
                            for x, y in points)
            legend_x = left if title == "cone" else left+320
            elements.append(f'<text x="{legend_x}" y="{top+302}" font-size="15" fill="{color}">'
                            f'{title}: mean {100*row["mean_reflectance"]:.3f}%</text>')
        tick_stride = max(1, math.ceil(len(wavelengths)/7))
        for index, wl in enumerate(wavelengths):
            if index % tick_stride and index != len(wavelengths)-1:
                continue
            x = left+(wl-wavelengths[0])/(wavelengths[-1]-wavelengths[0])*(right-left)
            elements.append(f'<text x="{x:.1f}" y="{plot_bottom+23}" text-anchor="middle" '
                            f'font-size="12" fill="#556677">{wl:g}</text>')
        elements.append(f'<text x="{right-5}" y="{plot_bottom+42}" text-anchor="end" '
                        'font-size="12" fill="#556677">wavelength (nm)</text>')
    elements.append('</svg>')
    path.write_text("\n".join(elements), encoding="utf-8")


def draw_mean_history(path: Path, checkpoint: dict, config: dict) -> None:
    records = sorted((row for row in checkpoint["evaluations"]
                      if row["order"] == config["order"] and row["label"] in {"cone", "candidate"}),
                     key=lambda row: row["step"])
    if not records:
        return
    left, right, top, bottom = 88, 765, 72, 330
    x_max = max(config["requested_steps"], records[-1]["step"], 1)
    values = [row["mean_reflectance"] for row in records]
    y_min = max(0.0, min(values)-0.01)
    y_max = min(1.0, max(values)+0.01)
    if y_max <= y_min:
        y_max = y_min+0.01
    point = lambda step, value: (
        left+(right-left)*step/x_max,
        bottom-(bottom-top)*(value-y_min)/(y_max-y_min),
    )
    elements = ['<svg xmlns="http://www.w3.org/2000/svg" width="820" height="405" viewBox="0 0 820 405">',
                '<rect width="820" height="405" fill="#f9fbfd"/>',
                '<text x="28" y="34" font-size="22" fill="#243447">Band-mean reflectance vs Adam updates</text>']
    for tick in range(6):
        fraction = tick/5
        y = bottom-(bottom-top)*fraction
        elements.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#d9e2ea"/>')
        elements.append(f'<text x="16" y="{y+5:.1f}" font-size="13" fill="#556677">{100*(y_min+(y_max-y_min)*fraction):.2f}%</text>')
    for step in (0, x_max//4, x_max//2, 3*x_max//4, x_max):
        x = left+(right-left)*step/x_max
        elements.append(f'<text x="{x:.1f}" y="{bottom+26}" text-anchor="middle" font-size="13" fill="#556677">{step}</text>')
    points = [point(row["step"], row["mean_reflectance"]) for row in records]
    elements.append('<polyline fill="none" stroke="#1368aa" stroke-width="2.5" points="'+
                    ' '.join(f'{x:.2f},{y:.2f}' for x,y in points)+'"/>')
    elements.extend(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="#1368aa"/>' for x,y in points)
    best_so_far = 1.0
    best_points = []
    for row in records:
        best_so_far = min(best_so_far, row["mean_reflectance"])
        best_points.append(point(row["step"], best_so_far))
    elements.append('<polyline fill="none" stroke="#c26610" stroke-width="2" stroke-dasharray="6 4" points="'+
                    ' '.join(f'{x:.2f},{y:.2f}' for x,y in best_points)+'"/>')
    elements.append('<text x="95" y="389" font-size="14" fill="#1368aa">full-band evaluation</text>')
    elements.append('<text x="360" y="389" font-size="14" fill="#c26610">best so far</text>')
    elements.append('</svg>')
    path.write_text("\n".join(elements), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--gold-csv", type=Path, default=DEFAULT_GOLD_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--wavelengths", default="400:700:50",
                        help="nm, e.g. 400:700:50 or 400,500,600,700")
    parser.add_argument("--slices", type=int, default=100)
    parser.add_argument("--order", type=int, default=8, help="Fourier order during Adam")
    parser.add_argument("--verify-order", type=int, default=16,
                        help="Final same-shape comparison order; 0 skips verification")
    parser.add_argument("--grid", type=int, default=256)
    parser.add_argument("--segments", type=int, default=8,
                        help="Number of monotone piecewise-linear radius segments")
    parser.add_argument("--diameter-margin-nm", type=float, default=0.1,
                        help="Keep both diameters strictly inside (0, period); default 0.1 nm")
    parser.add_argument("--steps", type=int, default=140,
                        help="Total Adam updates; raising this resumes an existing run")
    parser.add_argument("--eval-every-cycles", type=int, default=2,
                        help="Evaluate all wavelengths after this many wavelength cycles")
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--prepare-only", action="store_true",
                        help="Write cone geometry preview without importing torch")
    args = parser.parse_args()
    if (args.slices < 1 or args.order < 1 or args.verify_order < 0 or args.grid < 16 or
            args.segments < 2 or args.segments > args.slices or args.steps < 1 or
            args.eval_every_cycles < 1 or
            not math.isfinite(args.learning_rate) or
            args.learning_rate <= 0 or not math.isfinite(args.diameter_margin_nm) or
            not 0 < args.diameter_margin_nm < 10):
        parser.error("Check slices/order/verify-order/grid/segments/steps/eval-every-cycles/learning-rate/diameter-margin-nm.")
    try:
        wavelengths = wavelengths_from_text(args.wavelengths)
    except ValueError as error:
        parser.error(str(error))
    gold_csv = args.gold_csv.resolve()
    if not gold_csv.is_file():
        parser.error(f"Gold data not found: {gold_csv}")
    # Period and height remain fixed. Tip/base radii below are the initial cone,
    # while the optimization is free to move both within the diameter bounds.
    geometry = {
        "period_nm": 200.0, "height_nm": 500.0,
        "tip_radius_nm": 5.0, "base_radius_nm": 95.0,
        "lattice": "triangular", "substrate_mode": "semi-infinite",
        "asr_circle_g": 0.03,
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "version": VERSION, "geometry": geometry, "slices": args.slices,
        "order": args.order, "verify_order": args.verify_order,
        "grid": args.grid, "segments": args.segments,
        "eval_every_cycles": args.eval_every_cycles,
        "diameter_margin_nm": args.diameter_margin_nm,
        "wavelengths_nm": wavelengths, "weights": trapezoid_weights(wavelengths),
        "gold_csv": str(gold_csv), "gold_csv_sha256_lf": csv_data_hash(gold_csv),
        "core_sha256": file_hash(ROOT / "studies" / "gold_motheye" / "converge.py"),
        "asr_sha256": file_hash(ROOT / "rcwa_ext" / "asr.py"),
        "auto_sha256": file_hash(ROOT / "rcwa_ext" / "auto.py"),
        "learning_rate": args.learning_rate,
        "logit_limit": LOGIT_LIMIT,
        "objective": "normal-incidence x-polarized total reflectance, trapezoidal band mean",
        "requested_steps": args.steps,
    }
    signature_fields = {key: value for key, value in config.items()
                        if key not in {"requested_steps", "verify_order", "gold_csv",
                                       "eval_every_cycles"}}
    signature = hashlib.sha256(json.dumps(signature_fields, sort_keys=True).encode()).hexdigest()
    evaluation_interval = args.eval_every_cycles * len(wavelengths)
    checkpoint_path = output / "checkpoint.json"
    cone_logits = initial_logits(config)
    parameter_count = len(cone_logits)
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("signature") != signature:
            raise RuntimeError("Existing checkpoint settings differ. Use another --output-dir.")
        if args.steps < checkpoint["step"] and not args.prepare_only:
            raise RuntimeError("--steps is below the completed step count in this checkpoint.")
    else:
        checkpoint = {"signature": signature, "step": 0,
                      "logits": cone_logits, "m": [0.0]*parameter_count,
                      "v": [0.0]*parameter_count, "best": None,
                      "history": [], "evaluations": [], "pending": None}
    write_json(output / "config.json", config)
    save_profile(output, "cone", cone_logits, config)
    if args.prepare_only:
        print(f"Wrote cone_profile.csv/svg and config.json to {output}")
        return 0

    import torch
    from studies.shared.gold_dispersion import build_gold_model
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but PyTorch cannot see a GPU.")
    gold_model = build_gold_model("csv", gold_csv)
    for wavelength in wavelengths:
        gold_model(wavelength)
    print(f"Device={device}; Nz={args.slices}; Adam M={args.order}; "
          f"{len(wavelengths)} wavelengths; steps={args.steps}", flush=True)
    write_json(checkpoint_path, checkpoint)
    cone = find_evaluation(checkpoint, "cone", args.order, cone_logits)
    if cone is None:
        cone = evaluate(output, checkpoint, "cone", args.order, cone_logits,
                        config, gold_model, device, torch)
    if checkpoint["best"] is None:
        checkpoint["best"] = {"step": 0, "logits": cone_logits,
                              "mean_reflectance": cone["mean_reflectance"]}
        write_json(checkpoint_path, checkpoint)

    # Finish a wavelength-by-wavelength evaluation interrupted by walltime or
    # OOM before starting the next Adam update.
    pending = checkpoint.get("pending")
    if pending is not None:
        label, order_text, step_text = pending["key"].split("|")
        if int(step_text.removeprefix("step=")) != checkpoint["step"]:
            raise RuntimeError("Pending evaluation does not match checkpoint step.")
        resumed = evaluate(output, checkpoint, label, int(order_text.removeprefix("M=")),
                           pending["logits"], config, gold_model, device, torch)
        if label == "candidate" and resumed["mean_reflectance"] < checkpoint["best"]["mean_reflectance"]:
            checkpoint["best"] = {"step": checkpoint["step"],
                                  "logits": resumed["logits"],
                                  "mean_reflectance": resumed["mean_reflectance"]}
            write_json(checkpoint_path, checkpoint)
    # Also cover a stop after the last wavelength was saved, before the best
    # profile selection was written.
    current_candidate = find_evaluation(checkpoint, "candidate", args.order,
                                        checkpoint["logits"], checkpoint["step"])
    if (current_candidate is None and checkpoint["step"] > 0 and
            (checkpoint["step"] % evaluation_interval == 0 or checkpoint["step"] == args.steps)):
        current_candidate = evaluate(output, checkpoint, "candidate", args.order,
                                     checkpoint["logits"], config, gold_model, device, torch)
    if (current_candidate is not None and
            current_candidate["mean_reflectance"] < checkpoint["best"]["mean_reflectance"]):
        checkpoint["best"] = {"step": current_candidate["step"],
                              "logits": current_candidate["logits"],
                              "mean_reflectance": current_candidate["mean_reflectance"]}
        write_json(checkpoint_path, checkpoint)
    save_results(output, checkpoint, config)

    for step in range(checkpoint["step"]+1, args.steps+1):
        wavelength_index = (step-1) % len(wavelengths)
        wavelength = wavelengths[wavelength_index]
        started = time.perf_counter()
        logits = torch.tensor(checkpoint["logits"], dtype=torch.float64,
                              device=device, requires_grad=True)
        radii = radius_tensor(logits, args.slices, config, torch)
        result = reflectance_tensor(wavelength, radii, config, gold_model, device, torch)
        # Cycling all wavelengths once approximates the weighted band objective.
        (result * (len(wavelengths)*config["weights"][wavelength_index])).backward()
        if logits.grad is None or not bool(torch.isfinite(logits.grad).all().detach().cpu()):
            raise RuntimeError(f"Invalid shape gradient at step {step}.")
        gradient = [float(value) for value in logits.grad.detach().cpu().tolist()]
        norm = math.sqrt(sum(value*value for value in gradient))
        if norm > 1.0:
            gradient = [value/norm for value in gradient]
        # Adam, stored as plain JSON for transparent and reliable resume.
        m_old, v_old = checkpoint["m"], checkpoint["v"]
        m = [0.9*old + 0.1*g for old, g in zip(m_old, gradient)]
        v = [0.999*old + 0.001*g*g for old, g in zip(v_old, gradient)]
        updated = [value - args.learning_rate *
                   (mi/(1-0.9**step))/(math.sqrt(vi/(1-0.999**step))+1e-8)
                   for value, mi, vi in zip(checkpoint["logits"], m, v)]
        updated = [max(-LOGIT_LIMIT, min(LOGIT_LIMIT, value)) for value in updated]
        updated_radii = radius_values(updated, args.slices, config)
        if not all(left < right for left, right in zip(updated_radii, updated_radii[1:])):
            raise RuntimeError("Updated layer radii are not strictly increasing.")
        checkpoint.update({"step": step, "logits": updated, "m": m, "v": v})
        checkpoint["history"].append({"step": step, "wavelength_nm": wavelength,
                                      "reflectance": float(result.detach().cpu()),
                                      "gradient_norm": norm,
                                      "seconds": time.perf_counter()-started})
        write_json(checkpoint_path, checkpoint)
        print(f"step {step}/{args.steps}: {wavelength:g} nm R={float(result.detach().cpu()):.7f}, "
              f"|grad|={norm:.3g}", flush=True)
        del result, radii, logits
        gc.collect()
        if step % evaluation_interval == 0 or step == args.steps:
            row = evaluate(output, checkpoint, "candidate", args.order,
                           checkpoint["logits"], config, gold_model, device, torch)
            if row["mean_reflectance"] < checkpoint["best"]["mean_reflectance"]:
                checkpoint["best"] = {"step": step, "logits": checkpoint["logits"],
                                      "mean_reflectance": row["mean_reflectance"]}
                write_json(checkpoint_path, checkpoint)
            print(f"band mean: cone={cone['mean_reflectance']:.8f}, "
                  f"candidate={row['mean_reflectance']:.8f}, "
                  f"best={checkpoint['best']['mean_reflectance']:.8f}", flush=True)
            save_results(output, checkpoint, config)

    best = checkpoint["best"]
    if args.verify_order:
        verified_cone = find_evaluation(checkpoint, "verify_cone", args.verify_order, cone_logits)
        if verified_cone is None:
            verified_cone = evaluate(output, checkpoint, "verify_cone", args.verify_order,
                                     cone_logits, config, gold_model, device, torch)
        if find_evaluation(checkpoint, "verify_best", args.verify_order, best["logits"]) is None:
            if best["logits"] == cone_logits:
                checkpoint["evaluations"].append({**verified_cone, "label": "verify_best",
                                                  "logits": best["logits"]})
                write_json(checkpoint_path, checkpoint)
            else:
                evaluate(output, checkpoint, "verify_best", args.verify_order,
                         best["logits"], config, gold_model, device, torch)
    save_results(output, checkpoint, config)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
