"""Preparation, checkpointing, convergence criteria, and export (stdlib only)."""
from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import json
import math
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from studies.shared.gold_dispersion import build_gold_model

AXES = ("grid", "order", "slices")
POLS = ("TE", "TM")
METRICS = ("reflectance", "power_into_substrate", "relief_absorptance")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               allow_nan=False)+"\n", encoding="utf-8")
    temp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     allow_nan=False).encode()).hexdigest()


def integers(raw):
    values = [int(part) for part in raw.split(",")]
    return validate_values(values)


def validate_values(values):
    if (not isinstance(values, list) or not values or
        any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in values) or
        any(a >= b for a, b in zip(values, values[1:]))):
        raise ValueError("Numerical values must be positive, strictly increasing integers.")
    return values


def load_config(path):
    path = Path(path).resolve()
    config = read_json(path)
    geometry = config["geometry"]
    for name in ("period_nm", "height_nm", "profile_power"):
        if not math.isfinite(geometry[name]) or geometry[name] <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    top, bottom, period = (geometry[k] for k in
                           ("top_width_nm", "bottom_width_nm", "period_nm"))
    if not 0 < top <= bottom < period:
        raise ValueError("Require 0 < top_width <= bottom_width < period.")
    if config["solver"]["fourier_coefficients"] not in {"sampled", "analytic"}:
        raise ValueError("fourier_coefficients must be sampled or analytic.")
    if config["solver"]["dtype"] != "complex128" or config["solver"]["incidence"] != "normal":
        raise ValueError("This study supports complex128 and normal incidence only.")
    for name in ("tolerance", "passivity_tolerance"):
        if not math.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    waves = config["wavelengths_nm"]
    if (not waves or any(not math.isfinite(w) or w <= 0 for w in waves) or
        any(a >= b for a, b in zip(waves, waves[1:]))):
        raise ValueError("Wavelengths must be positive, finite and strictly increasing.")
    for axis in AXES:
        validate_values(config["search_values"][axis])
    validate_numbers(config["fixed_numerics"], sampled=config["solver"]["fourier_coefficients"] == "sampled")
    material_path = (path.parent/config["material"]["gold_csv"]).resolve()
    model = build_gold_model("csv", material_path)
    for wave in waves:
        eps = model(wave)
        if eps.imag <= 0:
            raise ValueError("Semi-infinite Au requires strictly positive absorption at every wavelength.")
    config["material"]["resolved_path"] = str(material_path)
    return config, model


def validate_numbers(numbers, *, sampled=True):
    for axis in AXES:
        validate_values([numbers[axis]])
    minimum = max(32, 4*numbers["order"]+4)
    if sampled and numbers["grid"] < minimum:
        raise ValueError(f"grid={numbers['grid']} is too small for M={numbers['order']}; need >= {minimum}.")


def signature_inputs(config):
    paths = [HERE/"solver.py", HERE/"common.py", HERE/"run_all.py",
             HERE/"converge.py", ROOT/"rcwa_solver_auto.py",
             ROOT/"studies/shared/gold_dispersion.py", *(ROOT/"rcwa_ext").glob("*.py")]
    if not (ROOT/"rcwa_ext/auto.py").is_file():
        raise ValueError("Put this folder under studies/ in the existing tsoyutarcwa repository.")
    sources = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(paths)}
    return {"geometry": config["geometry"], "solver": config["solver"],
            "formulation": "1D Cartesian Li; inverse normal/direct tangential rules",
            "source_sha256": sources,
            "gold_csv_sha256": hashlib.sha256(
                Path(config["material"]["resolved_path"]).read_bytes()).hexdigest(),
            "lateral_periodicity": "x only; M_y=0; y invariant",
            "substrate": "semi-infinite absorbing Au", "polarizations": list(POLS)}


def case_key(numbers, wavelength, *, analytic=False):
    if analytic:
        return f"analytic|{numbers['order']}|{numbers['slices']}|{wavelength:.12g}"
    return f"{numbers['order']}|{numbers['slices']}|{numbers['grid']}|{wavelength:.12g}"


def healthy(result, tolerance):
    try:
        return all(math.isfinite(result["polarizations"][p][m]) and
                   -tolerance <= result["polarizations"][p][m] <= 1+tolerance
                   for p in POLS for m in METRICS)
    except (KeyError, TypeError):
        return False


def compare(left, right, config):
    changes = {}
    for p in POLS:
        changes[p] = {m: right["polarizations"][p][m]-left["polarizations"][p][m]
                      for m in METRICS}
    maximum = max(abs(d) for metrics in changes.values() for d in metrics.values())
    valid = (healthy(left, config["passivity_tolerance"]) and
             healthy(right, config["passivity_tolerance"]))
    return {"signed_changes": changes, "max_absolute_change": maximum,
            "passes_tolerance": bool(valid and maximum <= config["tolerance"]),
            "passivity_bounds_pass": valid}


class Study:
    def __init__(self, config, model, output, device):
        self.config, self.model = config, model
        self.output, self.device = Path(output).resolve(), device
        self.output.mkdir(parents=True, exist_ok=True)
        self.path = self.output/"checkpoint.json"
        inputs = signature_inputs(config)
        self.signature = digest(inputs)
        if self.path.exists():
            self.checkpoint = read_json(self.path)
            if self.checkpoint.get("signature") != self.signature:
                raise ValueError("Saved geometry/material/solver sources differ. Use a new --output-dir.")
        else:
            self.checkpoint = {"signature": self.signature, "inputs": inputs,
                               "cases": {}, "errors": {}}
        self.simulate = None

    def save(self):
        write_json(self.path, self.checkpoint)

    def get(self, numbers, wave, *, calculate=True, analytic=False):
        key = case_key(numbers, wave, analytic=analytic)
        if key in self.checkpoint["cases"]:
            return self.checkpoint["cases"][key]
        if not calculate:
            return None
        local_config = self.config
        if analytic:
            local_config = copy.deepcopy(self.config)
            local_config["solver"]["fourier_coefficients"] = "analytic"
        method = local_config["solver"]["fourier_coefficients"]
        validate_numbers(numbers, sampled=method == "sampled")
        print(f"solve M={numbers['order']}, Nz={numbers['slices']}, grid={numbers['grid']}, "
              f"wavelength={wave:g} nm (TE and TM; {method} coefficients)", flush=True)
        try:
            if self.simulate is None:
                from solver import simulate
                self.simulate = simulate
            result = self.simulate(local_config, numbers, wave, self.model(wave), self.device)
            result["passivity_bounds_pass"] = healthy(result, self.config["passivity_tolerance"])
            if not result["passivity_bounds_pass"]:
                print("WARNING: passivity bounds failed; this case cannot pass convergence.", flush=True)
            self.checkpoint["cases"][key] = result
            self.checkpoint["errors"].pop(key, None)
            self.save()
            memory = result.get("peak_cuda_allocated_bytes")
            suffix = f"; peak CUDA tensors {memory/2**30:.3f} GiB" if memory is not None else ""
            print(f"saved: {result['runtime_seconds']:.2f} s{suffix}", flush=True)
            return result
        except Exception as exc:
            self.checkpoint["errors"][key] = {"error": str(exc), "traceback": traceback.format_exc()}
            self.save()
            raise

    def sweep(self, axis, values, fixed, *, calculate=True):
        validate_values(values)
        for value in values:
            numbers = dict(fixed, **{axis: value})
            validate_numbers(numbers, sampled=self.config["solver"]["fourier_coefficients"] == "sampled")
            for wave in self.config["wavelengths_nm"]:
                self.get(numbers, wave, calculate=calculate)
        report = self.sweep_report(axis, values, fixed)
        folder = self.output/f"{axis}_M{fixed['order']}_Nz{fixed['slices']}_grid{fixed['grid']}_values{values[0]}-{values[-1]}"
        folder.mkdir(exist_ok=True)
        write_json(folder/"report.json", report)
        from plot import plot_sweep
        plot_sweep(self.checkpoint, report, folder)
        print(f"{axis}: {report['status']}; report: {folder/'report.json'}", flush=True)
        return report

    def sweep_report(self, axis, values, fixed):
        changes, complete, tails = {}, True, []
        for wave in self.config["wavelengths_nm"]:
            rows = [self.get(dict(fixed, **{axis: value}), wave, calculate=False) for value in values]
            complete &= all(row is not None for row in rows)
            adjacent = []
            for i in range(1, len(values)):
                pair = {"low": values[i-1], "high": values[i]}
                if rows[i-1] is None or rows[i] is None:
                    pair.update(passes_tolerance=False, missing=True)
                else:
                    pair.update(compare(rows[i-1], rows[i], self.config))
                adjacent.append(pair)
            changes[f"{wave:g}"] = adjacent
            tails.append(len(adjacent) >= 2 and all(p["passes_tolerance"] for p in adjacent[-2:]))
        passed = bool(complete and all(tails))
        return {"signature": self.signature, "axis": axis, "values": values, "fixed_numerics": fixed,
                "coefficient_method": self.config["solver"]["fourier_coefficients"],
                "wavelengths_nm": self.config["wavelengths_nm"],
                "tolerance": self.config["tolerance"], "adjacent_changes": changes,
                "status": "converged_within_tested_values" if passed else (
                    "not_converged" if complete else "incomplete"),
                "criterion": "Both final adjacent steps pass every R/P_sub/A_relief metric at every wavelength for TE and TM.",
                "scope": "This axis at the stated fixed numerics; not absolute error or untested wavelengths."}


def geometry_svg(config, path, slices):
    g = config["geometry"]
    period, height = g["period_nm"], g["height_nm"]
    sx, sy, y0 = 800/(3*period), 420/height, 100
    body = ['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="670" viewBox="0 0 960 670">',
            '<rect width="960" height="670" fill="white"/>',
            '<g font-family="sans-serif" fill="#182738">',
            '<text x="40" y="35" font-size="23">Au trapezoidal grating: x periodic, y invariant</text>',
            f'<text x="40" y="65" font-size="15">period {period:g} nm | height {height:g} nm | top {g["top_width_nm"]:g} nm | bottom {g["bottom_width_nm"]:g} nm</text>',
            '<text x="430" y="90" font-size="15">air / incident light (downward)</text>']
    for cell in range(3):
        center = 80+(cell+0.5)*period*sx
        points = []
        for index in range(101):
            s = index/100
            width = g["top_width_nm"]+(g["bottom_width_nm"]-g["top_width_nm"])*s**g["profile_power"]
            points.append((center-width*sx/2, y0+s*height*sy))
        right = [(2*center-x, y) for x, y in reversed(points)]
        outline = " ".join(f"{x:.3f},{y:.3f}" for x, y in points+right)
        body.append(f'<polygon points="{outline}" fill="#d6ac38" stroke="#8b6c13"/>')
        for layer in range(slices):
            s = (layer+0.5)/slices
            width = g["top_width_nm"]+(g["bottom_width_nm"]-g["top_width_nm"])*s**g["profile_power"]
            body.append(f'<rect x="{center-width*sx/2:.3f}" y="{y0+layer*height*sy/slices:.3f}" width="{width*sx:.3f}" height="{height*sy/slices:.3f}" fill="none" stroke="#6c5514" stroke-opacity="0.25" stroke-width="0.5"/>')
    body.extend([f'<rect x="80" y="{y0+height*sy}" width="800" height="55" fill="#d6ac38"/>',
                 '<text x="315" y="553" font-size="17">semi-infinite Au substrate</text>',
                 f'<text x="80" y="605" font-size="15">Thin rectangles: {slices} midpoint-width slices of height {height/slices:.4g} nm.</text>',
                 '<text x="80" y="635" font-size="15">Infinite extrusion in y; TE: E along y, TM: E along x.</text>',
                 '</g></svg>'])
    Path(path).write_text("\n".join(body), encoding="utf-8")


def export_cases(checkpoint, output):
    names = ("wavelength_nm", "polarization", "order", "slices", "grid", *METRICS,
             "absorptance_total", "transmittance_far", "runtime_seconds",
             "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes",
             "coefficient_method", "grid_used", "maximum_fill_fraction_error_all_layers",
             "maximum_selected_electric_mode_condition", "maximum_selected_magnetic_mode_condition",
             "maximum_selected_reciprocal_epsilon_condition", "maximum_selected_eigen_residual")
    with (Path(output)/"cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, names)
        writer.writeheader()
        for key, case in sorted(checkpoint["cases"].items()):
            for p in POLS:
                row = {name: case.get(name, "") for name in names}
                row.update(case["polarizations"][p], polarization=p)
                for column, key in (("maximum_selected_electric_mode_condition", "electric_eigenvector_condition"),
                                    ("maximum_selected_magnetic_mode_condition", "magnetic_eigenvector_condition"),
                                    ("maximum_selected_reciprocal_epsilon_condition", "normal_epsilon_reciprocal_condition"),
                                    ("maximum_selected_eigen_residual", "relative_eigen_residual")):
                    row[column] = max((d[key] for d in case.get("selected_layer_diagnostics", [])), default="")
                writer.writerow({name: row.get(name, "") for name in names})


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", type=Path, default=HERE/"config.json")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--prepare-only", action="store_true", help="Write plan and geometry; do not import torch or solve.")
    p.add_argument("--report-only", action="store_true", help="Use saved cases without new solves.")
    return p
