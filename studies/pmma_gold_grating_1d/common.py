"""Configuration, signed checkpoints and R/T/A convergence without torch imports."""
from __future__ import annotations

import argparse
import csv
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
from studies.pmma_gold_grating_1d.geometry import build_layers, preview, validate_geometry
from studies.pmma_gold_grating_1d.materials import Materials

AXES = ("order", "slices")
POLS = ("TE", "TM")
METRICS = ("reflectance", "transmittance", "absorptance")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_values(values):
    if (not isinstance(values, list) or not values
            or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in values)
            or any(a >= b for a, b in zip(values, values[1:]))):
        raise ValueError("Use positive, strictly increasing integer values.")
    return values


def integers(raw):
    return validate_values([int(s.strip()) for s in raw.split(",")])


def validate_numbers(numbers, **unused):
    for axis in AXES:
        validate_values([numbers[axis]])


def load_config(path):
    path = Path(path).resolve()
    config = read_json(path)
    validate_geometry(config["geometry"])
    if config["solver"] != {"fourier_coefficients": "analytic", "dtype": "complex128", "incidence": "normal"}:
        raise ValueError("This study supports analytic coefficients, complex128 and normal incidence only.")
    for name in ("tolerance", "passivity_tolerance"):
        if not math.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    waves = config["wavelengths_nm"]
    if (not waves or any(not math.isfinite(w) or w <= 0 for w in waves)
            or any(a >= b for a, b in zip(waves, waves[1:]))):
        raise ValueError("Wavelengths must be finite, positive and increasing.")
    validate_numbers(config["fixed_numerics"])
    for axis in AXES:
        validate_values(config["search_values"][axis])
    for count in set(config["search_values"]["slices"]+[config["fixed_numerics"]["slices"]]):
        build_layers(config["geometry"], count)
    material = config["material"]
    for name in ("gold", "pmma"):
        material["resolved_"+name+"_path"] = str((path.parent/material[name+"_csv"]).resolve())
    model = Materials(material["resolved_pmma_path"], material["resolved_gold_path"],
                      material["pmma_shortwave_extension"])
    for wave in waves:
        model(wave)
    material["metadata"] = model.note
    return config, model


def signature_inputs(config):
    paths = [HERE/name for name in ("solver.py", "common.py", "materials.py", "geometry.py", "run_all.py", "converge.py")]
    paths += [ROOT/"rcwa_solver_auto.py", ROOT/"studies/gold_grating_1d/solver.py", *(ROOT/"rcwa_ext").glob("*.py")]
    sources = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    return {"geometry": config["geometry"], "solver": config["solver"], "source_sha256": sources,
            "material_sha256": {name: hashlib.sha256(Path(config["material"]["resolved_"+name+"_path"]).read_bytes()).hexdigest()
                                for name in ("pmma", "gold")},
            "pmma_shortwave_extension": config["material"]["pmma_shortwave_extension"],
            "pmma_loss": "k=0", "substrate": "semi-infinite lossless PMMA",
            "formulation": "1D Cartesian Li; inverse normal/direct tangential; analytic intervals",
            "lateral_periodicity": "x only; M_y=0; invariant y", "polarizations": list(POLS)}


def case_key(numbers, wavelength):
    return f"{numbers['order']}|{numbers['slices']}|{wavelength:.12g}"


def healthy(result, tolerance):
    try:
        return all(math.isfinite(result["polarizations"][p][m]) and
                   -tolerance <= result["polarizations"][p][m] <= 1+tolerance
                   for p in POLS for m in METRICS)
    except (KeyError, TypeError):
        return False


def compare(left, right, config):
    changes = {p: {m: right["polarizations"][p][m]-left["polarizations"][p][m] for m in METRICS} for p in POLS}
    maximum = max(abs(d) for metrics in changes.values() for d in metrics.values())
    valid = healthy(left, config["passivity_tolerance"]) and healthy(right, config["passivity_tolerance"])
    return {"signed_changes": changes, "max_absolute_change": maximum,
            "passes_tolerance": bool(valid and maximum <= config["tolerance"]), "passivity_bounds_pass": valid}


class Study:
    def __init__(self, config, model, output, device):
        self.config, self.model, self.device = config, model, device
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.path = self.output/"checkpoint.json"
        inputs = signature_inputs(config)
        self.signature = digest(inputs)
        if self.path.exists():
            self.checkpoint = read_json(self.path)
            if (self.checkpoint.get("signature") != self.signature
                    or digest(self.checkpoint["inputs"]) != self.signature):
                raise ValueError("Saved geometry/material/solver sources differ; use a new --output-dir.")
            for key, row in self.checkpoint["cases"].items():
                validate_numbers(row)
                if key != case_key(row, row["wavelength_nm"]):
                    raise ValueError("Invalid saved numerical case key.")
        else:
            self.checkpoint = {"signature": self.signature, "inputs": inputs, "cases": {}, "errors": {}}
        self.simulate = None

    def save(self):
        write_json(self.path, self.checkpoint)

    def get(self, numbers, wave, *, calculate=True):
        key = case_key(numbers, wave)
        if key in self.checkpoint["cases"]:
            return self.checkpoint["cases"][key]
        if not calculate:
            return None
        validate_numbers(numbers)
        build_layers(self.config["geometry"], numbers["slices"])
        print(f"solve M={numbers['order']}, Nz={numbers['slices']}, wavelength={wave:g} nm (TE/TM; analytic coefficients)", flush=True)
        try:
            if self.simulate is None:
                from studies.pmma_gold_grating_1d.solver import simulate
                self.simulate = simulate
            result = self.simulate(self.config, numbers, wave, self.model(wave), self.device)
            result["passivity_bounds_pass"] = healthy(result, self.config["passivity_tolerance"])
            result["pmma_index_source"] = ("short edge extrapolation" if wave < self.model.note["original_pmma_range_nm"][0]
                                           else "interpolation within original CSV")
            if not result["passivity_bounds_pass"]:
                print("WARNING: passivity bounds failed; this case cannot pass convergence.", flush=True)
            self.checkpoint["cases"][key] = result
            self.checkpoint["errors"].pop(key, None)
            self.save()
            memory = result.get("peak_cuda_allocated_bytes")
            suffix = "" if memory is None else f"; peak CUDA tensors {memory/2**30:.3f} GiB"
            print(f"saved: {result['runtime_seconds']:.2f} s{suffix}", flush=True)
            return result
        except Exception as error:
            self.checkpoint["errors"][key] = {"error": str(error), "traceback": traceback.format_exc()}
            self.save()
            raise

    def sweep_report(self, axis, values, fixed):
        changes, complete, tails = {}, True, []
        for wave in self.config["wavelengths_nm"]:
            rows = [self.get(dict(fixed, **{axis: value}), wave, calculate=False) for value in values]
            complete &= all(row is not None for row in rows)
            adjacent = []
            for i in range(1, len(values)):
                pair = {"low": values[i-1], "high": values[i]}
                pair.update(compare(rows[i-1], rows[i], self.config) if rows[i-1] is not None and rows[i] is not None
                            else {"passes_tolerance": False, "missing": True})
                adjacent.append(pair)
            changes[f"{wave:g}"] = adjacent
            tails.append(len(adjacent) >= 2 and all(p["passes_tolerance"] for p in adjacent[-2:]))
        return {"signature": self.signature, "axis": axis, "values": values, "fixed_numerics": fixed,
                "coefficient_method": "analytic", "grid_used": None, "wavelengths_nm": self.config["wavelengths_nm"],
                "tolerance": self.config["tolerance"], "adjacent_changes": changes,
                "status": "converged_within_tested_values" if complete and all(tails) else ("not_converged" if complete else "incomplete"),
                "criterion": "Both final adjacent steps pass every R/T/A metric at every wavelength for TE and TM.",
                "scope": "This axis at the stated fixed numerics; not absolute error or untested wavelengths."}

    def sweep(self, axis, values, fixed, *, calculate=True):
        validate_values(values)
        for value in values:
            for wave in self.config["wavelengths_nm"]:
                self.get(dict(fixed, **{axis: value}), wave, calculate=calculate)
        report = self.sweep_report(axis, values, fixed)
        folder = self.output/f"{axis}_M{fixed['order']}_Nz{fixed['slices']}_values{values[0]}-{values[-1]}"
        write_json(folder/"report.json", report)
        from studies.pmma_gold_grating_1d.plot import plot_sweep
        plot_sweep(self.checkpoint, report, folder)
        print(f"{axis}: {report['status']}; report: {folder/'report.json'}", flush=True)
        return report


def geometry_svg(config, path, slices):
    preview(config, path, slices)


def export_cases(checkpoint, output):
    names = ("wavelength_nm", "polarization", "order", "slices", "total_finite_layers", *METRICS,
             "runtime_seconds", "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes", "coefficient_method",
             "grid_used", "pmma_index_source", "epsilon_pmma", "epsilon_gold_real", "epsilon_gold_imag",
             "maximum_selected_reciprocal_epsilon_condition", "maximum_selected_eigen_residual")
    with (Path(output)/"cases.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for case in sorted(checkpoint["cases"].values(), key=lambda c: (c["order"], c["slices"], c["wavelength_nm"])):
            for pol in POLS:
                row = {name: case.get(name, "") for name in names}
                row.update(case["polarizations"][pol], polarization=pol)
                for column, key in (("maximum_selected_reciprocal_epsilon_condition", "normal_epsilon_reciprocal_condition"),
                                    ("maximum_selected_eigen_residual", "relative_eigen_residual")):
                    row[column] = max((d[key] for d in case.get("selected_layer_diagnostics", [])), default="")
                writer.writerow(row)


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", type=Path, default=HERE/"config.json")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--prepare-only", action="store_true", help="Write plan/geometry without torch or an optical solve.")
    p.add_argument("--report-only", action="store_true", help="Use only saved results.")
    return p
