"""Run independent physical checks for the Au/PMMA RCWA model.

Preparation needs only the Python standard library. Solves need PyTorch and
the same RCWA environment as the main studies.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
import traceback
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
STUDY = ROOT / "studies" / "pmma_gold_motheye"
if str(STUDY) not in sys.path:
    sys.path.insert(0, str(STUDY))

import run_pmma_gold_30nm as materials


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8")


def _plan() -> dict:
    config = json.loads((HERE / "cases.json").read_text(encoding="utf-8"))
    if config.get("version") != 1:
        raise ValueError("Unsupported cases.json version.")
    ids = [case["id"] for case in config["cases"]]
    if len(set(ids)) != len(ids):
        raise ValueError("Case IDs must be unique.")
    shape = config["geometry_nm"]
    number = config["numerics"]
    profile = materials.build_profile_layers(
        height_nm=shape["height"],
        tip_radius_nm=shape["pmma_tip_radius"],
        base_radius_nm=shape["pmma_base_radius"],
        gold_thickness_nm=shape["au_radial_thickness"],
        profile_power=1.0, slices=number["profile_slices"],
        valley_gold_thickness_nm=shape["au_valley_thickness"])
    radius_increases = all(a.core_radius_nm < b.core_radius_nm
                           for a, b in zip(profile, profile[1:]))
    valley_boundary_depth = shape["height"] - shape["au_valley_thickness"]
    valley_boundary_aligned = any(
        abs(layer.top_depth_nm - valley_boundary_depth) < 1e-9
        for layer in profile if layer.kind == "valley")
    outer_base_gap = shape["period"] - 2*(shape["pmma_base_radius"] +
                                           shape["au_radial_thickness"])
    if (len(profile) != number["profile_slices"] or not radius_increases or
            not valley_boundary_aligned or outer_base_gap <= 0):
        raise ValueError("Geometry/layer consistency check failed.")
    pmma_csv = ROOT / "studies" / "pmma_gold_motheye" / "data" / "Szczurowski.csv"
    gold_csv = ROOT / "studies" / "pmma_gold_motheye" / "data" / "au_measured_nk.csv"
    pmma = materials.read_pmma(pmma_csv)
    gold = materials.read_gold(gold_csv)
    values = {}
    for case in config["cases"]:
        wavelength = float(case["wavelength_nm"])
        (pmma_n,) = materials.interpolate(pmma, wavelength)
        gold_n, gold_k = materials.interpolate(gold, wavelength)
        values[case["id"]] = {"wavelength_nm": wavelength, "pmma_n": pmma_n,
                              "gold_n": gold_n, "gold_k": gold_k}
    source_hashes = {}
    for name, path in {
        "pmma_csv": pmma_csv, "gold_csv": gold_csv,
        "case_runner": HERE / "run.py",
        "spectrum_runner": STUDY / "run_pmma_gold_30nm.py",
        "common": ROOT / "studies" / "pmma_gold_motheye" / "common.py",
        "gold_gradient": ROOT / "studies" / "gold_motheye2" / "optimize_adam.py",
        "asr": ROOT / "rcwa_ext" / "asr.py",
        "asr_maps": ROOT / "rcwa_ext" / "asr_maps.py",
        "auto": ROOT / "rcwa_ext" / "auto.py",
        "scattering": ROOT / "rcwa_ext" / "scattering.py",
    }.items():
        source_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    geometry_check = {
        "pmma_height_nm": shape["height"],
        "top_au_disk_height_nm": shape["au_radial_thickness"],
        "total_height_nm": shape["height"] + shape["au_radial_thickness"],
        "top_au_disk_diameter_nm": 2*(shape["pmma_tip_radius"] +
                                       shape["au_radial_thickness"]),
        "profile_layers": len(profile), "patterned_layers_including_disk": len(profile)+1,
        "pmma_radius_strictly_increases_toward_base": radius_increases,
        "valley_gold_boundary_aligned": valley_boundary_aligned,
        "minimum_gap_between_au_shells_nm": outer_base_gap,
    }
    return {"configuration": config, "materials": values,
            "geometry_check": geometry_check,
            "source_sha256": source_hashes,
            "scope": "Fixed low-order physical checks; not Fourier, grid, or z-slice convergence."}


def _check_passivity(result: dict, *, lossless: bool = False) -> None:
    values = [float(result[name]) for name in ("reflectance", "transmittance", "absorptance")]
    if not all(math.isfinite(value) for value in values):
        raise AssertionError(f"Nonfinite R/T/A: {values}")
    if any(value < -1e-5 or value > 1.0 + 1e-5 for value in values):
        raise AssertionError(f"Passive power bounds violated: {values}")
    if lossless and abs(values[2]) > 1e-4:
        raise AssertionError(f"Lossless energy error: A={values[2]}")


def _run_case(case: dict, material: dict, config: dict, device, torch) -> dict:
    from rcwa_solver_auto import ASROptions, AutoRCWA, GroupTheoryOptions, Lattice, OutputSpec
    from studies.pmma_gold_motheye import common
    from studies.gold_motheye2 import optimize_adam as gold_gradient

    wavelength = material["wavelength_nm"]
    pmma_n = material["pmma_n"]
    epsilon_gold = complex(material["gold_n"], material["gold_k"]) ** 2
    numbers = config["numerics"]
    shape = config["geometry_nm"]
    numerical = common.NumericalConfig(
        order=numbers["fourier_order"], slices=numbers["profile_slices"],
        grid=numbers["asr_grid"], radial_mapping=numbers["mapping"])
    geometry = common.GeometryConfig(
        period_nm=shape["period"], height_nm=shape["height"],
        tip_radius_nm=shape["pmma_tip_radius"], base_radius_nm=shape["pmma_base_radius"],
        gold_thickness_nm=shape["au_radial_thickness"], pmma_index=pmma_n,
        lattice="triangular", include_top_cap=True, asr_circle_g=0.03)

    def relief(*, epsilon=epsilon_gold, symmetry=True, discard=True):
        return common.simulate_case(
            wavelength, numerical, geometry, lambda _w: epsilon,
            cascade="redheffer", use_symmetry=symmetry,
            symmetry_reduction="d6-source", factorization_rules=True,
            device=device, valley_gold_thickness_nm=shape["au_valley_thickness"],
            discard_auxiliary=discard)

    def measure(call):
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        value = call()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device)
        else:
            peak = None
        return value, {"wall_seconds": time.perf_counter() - started,
                       "peak_cuda_allocated_bytes": peak}

    kind = case["kind"]
    if kind == "fresnel":
        simulation = AutoRCWA(
            freq=shape["period"] / wavelength,
            order=[numbers["fourier_order"], numbers["fourier_order"]],
            lattice=Lattice.triangular(1.0), cascade="redheffer",
            outputs=OutputSpec(smatrix_size="half", fields="none"),
            asr=ASROptions(grid=(numbers["asr_grid"],) * 2),
            group_theory=GroupTheoryOptions(enabled=False),
            verify_cascade=False, dtype=torch.complex128, device=device)
        simulation.add_input_layer(eps=1.0, mu=1.0)
        simulation.add_output_layer(eps=pmma_n**2, mu=1.0)
        simulation.set_incident_angle(0.0, 0.0)
        # A uniform PMMA layer joins the semi-infinite PMMA substrate.
        simulation.add_layer(0.1, eps=pmma_n**2)
        simulation.solve_global_smatrix()
        incident = common._zero_order_x_source(simulation)
        incident_flux = common._mean_poynting_z(incident, simulation.Vi, direction=1)
        reflected_flux = common._mean_poynting_z(simulation.S[1] @ incident,
                                                   simulation.Vi, direction=-1)
        transmitted_flux = common._mean_poynting_z(simulation.S[0] @ incident,
                                                     simulation.Vo, direction=1)
        result = {"reflectance": -reflected_flux / incident_flux,
                  "transmittance": transmitted_flux / incident_flux}
        result["absorptance"] = 1 - result["reflectance"] - result["transmittance"]
        _check_passivity(result, lossless=True)
        reference = {"reflectance": ((1-pmma_n)/(1+pmma_n))**2,
                     "transmittance": 4*pmma_n/(1+pmma_n)**2}
        errors = {name: abs(result[name]-expected) for name, expected in reference.items()}
        if max(errors.values()) > 1e-7:
            raise AssertionError(f"Fresnel mismatch: {errors}")
        return {"computed": result, "fresnel_reference": reference, "absolute_errors": errors}

    if kind == "lossless_relief":
        result = relief(epsilon=complex(pmma_n**2))
        _check_passivity(result, lossless=True)
        return {"computed": result}

    if kind == "passive_relief":
        result = relief()
        _check_passivity(result)
        if result["absorptance"] <= 1e-4:
            raise AssertionError("Measured Au should absorb nonzero power.")
        return {"computed": result}

    if kind == "auxiliary_parity":
        retained, retained_cost = measure(lambda: relief(discard=False))
        released, released_cost = measure(lambda: relief(discard=True))
        _check_passivity(retained)
        _check_passivity(released)
        errors = {name: abs(retained[name]-released[name])
                  for name in ("reflectance", "transmittance", "absorptance")}
        if max(errors.values()) > 1e-8:
            raise AssertionError(f"Auxiliary-release R/T/A mismatch: {errors}")
        return {"retained": retained, "released": released, "absolute_errors": errors,
                "retained_cost": retained_cost, "released_cost": released_cost}

    if kind == "symmetry_parity":
        reduced = relief(symmetry=True)
        full = relief(symmetry=False)
        _check_passivity(reduced)
        _check_passivity(full)
        errors = {name: abs(reduced[name]-full[name])
                  for name in ("reflectance", "transmittance", "absorptance")}
        if max(errors.values()) > 1e-6:
            raise AssertionError(f"D6/full R/T/A mismatch: {errors}")
        return {"d6": reduced, "full": full, "absolute_errors": errors}

    if kind == "gradient_parity":
        gold_config = {"geometry": {"period_nm": shape["period"],
                                     "height_nm": shape["height"], "asr_circle_g": 0.03},
                       "order": numbers["fourier_order"], "slices": 3,
                       "grid": numbers["asr_grid"]}
        values = {}
        for discard in (False, True):
            def solve_gradient():
                radii = torch.tensor([10.0, 30.0, 70.0], dtype=torch.float64,
                                     device=device, requires_grad=True)
                reflection = gold_gradient.reflectance_tensor(
                    wavelength, radii, gold_config, lambda _w: epsilon_gold,
                    device, torch, discard_auxiliary=discard)
                derivative = torch.autograd.grad(reflection, radii)[0]
                if not bool(torch.isfinite(derivative).all().detach().cpu()):
                    raise AssertionError("Nonfinite radius gradient.")
                return {"reflectance": float(reflection.detach().cpu()),
                        "radius_gradient_per_nm": derivative.detach().cpu().tolist()}
            result, cost = measure(solve_gradient)
            values["released" if discard else "retained"] = {**result, **cost}
        r_error = abs(values["retained"]["reflectance"] -
                      values["released"]["reflectance"])
        gradient_error = max(abs(a-b) for a, b in zip(
            values["retained"]["radius_gradient_per_nm"],
            values["released"]["radius_gradient_per_nm"]))
        if r_error > 1e-10 or gradient_error > 1e-8:
            raise AssertionError(f"Gradient parity error: R={r_error}, dR/dr={gradient_error}")
        return {**values, "absolute_R_error": r_error,
                "maximum_radius_gradient_error": gradient_error}

    raise ValueError(f"Unknown case kind: {kind}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / "quick")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--case", action="append", dest="case_ids",
                        help="Run only the named case; may be repeated")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan = _plan()
    _write_json(output / "plan.json", plan)
    selected = [case for case in plan["configuration"]["cases"]
                if args.case_ids is None or case["id"] in args.case_ids]
    if not selected or (args.case_ids and set(args.case_ids) != {c["id"] for c in selected}):
        parser.error("Unknown or empty --case selection.")
    if args.prepare_only:
        print(f"Prepared {len(selected)} cases: {output / 'plan.json'}")
        return 0

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot see a GPU.")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cpu":
        torch.set_num_threads(2)
    signature = hashlib.sha256(json.dumps({"plan": plan, "device": str(device)},
                                          sort_keys=True).encode()).hexdigest()
    checkpoint_path = output / "checkpoint.json"
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("signature") != signature:
            raise RuntimeError("Existing checkpoint belongs to different settings or source code.")
    else:
        checkpoint = {"signature": signature, "device": str(device), "cases": {}}
    for case in selected:
        if checkpoint["cases"].get(case["id"], {}).get("passed"):
            print(f"SKIP {case['id']} (saved pass)", flush=True)
            continue
        started = time.perf_counter()
        try:
            details = _run_case(case, plan["materials"][case["id"]],
                                plan["configuration"], device, torch)
            record = {"passed": True, "details": details}
        except Exception as exc:
            record = {"passed": False, "error": str(exc),
                      "traceback": traceback.format_exc(limit=6)}
        record["runtime_seconds"] = time.perf_counter() - started
        checkpoint["cases"][case["id"]] = record
        _write_json(checkpoint_path, checkpoint)
        print(f"{'PASS' if record['passed'] else 'FAIL'} {case['id']} "
              f"({record['runtime_seconds']:.2f} s)", flush=True)
    report = {"passed": all(checkpoint["cases"].get(case["id"], {}).get("passed", False)
                            for case in selected),
              "selected_cases": [case["id"] for case in selected],
              "scope": plan["scope"], "device": str(device),
              "cases": {case["id"]: checkpoint["cases"].get(case["id"])
                        for case in selected}}
    _write_json(output / "report.json", report)
    print(f"report: {output / 'report.json'}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
