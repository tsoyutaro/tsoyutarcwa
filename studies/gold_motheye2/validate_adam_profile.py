"""Validate the saved best Adam profile while changing M, Nz or ASR grid.

Run from the project root. No Adam updates are performed. The continuous
piecewise-linear profile is resampled at each requested Nz, and each completed
optical solve is checkpointed. Matching training evaluations are reused.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import itertools
import json
import math
import sys
import time
from pathlib import Path


def file_hash(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def source_matches(saved, path):
    raw = path.read_bytes()
    lf = raw.replace(b"\r\n", b"\n")
    return saved in {hashlib.sha256(value).hexdigest()
                     for value in (raw, lf, lf.replace(b"\n", b"\r\n"))}


def integer_list(text, minimum):
    values = sorted(set(int(item) for item in text.split(",")))
    if not values or values[0] < minimum:
        raise ValueError(f"Values must be integers >= {minimum}.")
    return values


def case_key(label, order, slices, grid, wavelength):
    return f"{label}|M={order}|Nz={slices}|grid={grid}|wl={wavelength:.17g}"


def band_mean(values, wavelengths):
    if len(wavelengths) < 2:
        return None
    return sum((b-a)*(values[a]+values[b])/2
               for a,b in zip(wavelengths, wavelengths[1:])) / (wavelengths[-1]-wavelengths[0])


def comparisons(cases, orders, slices, grids, wavelengths, mean_tolerance, max_tolerance):
    result = []
    dimensions = {"order": orders, "slices": slices, "grid": grids}
    for axis, values in dimensions.items():
        other_axes = [key for key in dimensions if key != axis]
        for fixed in itertools.product(*(dimensions[key] for key in other_axes)):
            for label in ("cone", "best"):
                for before, after in zip(values, values[1:]):
                    a = dict(zip(other_axes, fixed), **{axis: before})
                    b = {**a, axis: after}
                    def collect(settings):
                        return {wl: cases[case_key(label, settings["order"], settings["slices"], settings["grid"], wl)]["reflectance"]
                                for wl in wavelengths
                                if case_key(label, settings["order"], settings["slices"], settings["grid"], wl) in cases}
                    left, right = collect(a), collect(b)
                    common = sorted(left.keys() & right.keys())
                    if not common:
                        continue
                    change, wl = max((abs(right[w]-left[w]), w) for w in common)
                    mean_a, mean_b = band_mean(left, common), band_mean(right, common)
                    delta = mean_b-mean_a if mean_a is not None else None
                    result.append({"label": label, "axis": axis, "before": a, "after": b,
                                   "wavelengths_nm": common, "max_absolute_delta": change,
                                   "max_delta_wavelength_nm": wl, "band_mean_before": mean_a,
                                   "band_mean_after": mean_b, "band_mean_delta": delta,
                                   "complete_requested_wavelengths": len(common) == len(wavelengths),
                                   "within_requested_tolerance_at_sampled_wavelengths":
                                   len(common) == len(wavelengths) and change <= max_tolerance
                                   and (delta is None or abs(delta) <= mean_tolerance)})
    return result


def main():
    location = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=location if (location/"rcwa_solver_auto.py").is_file() else Path.cwd())
    parser.add_argument("--run-dir", type=Path, help="Existing full-band Adam result directory")
    parser.add_argument("--output-dir", type=Path, help="Separate fixed-profile validation directory")
    parser.add_argument("--gold-csv", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--orders", default="18")
    parser.add_argument("--slices", default="100,120,140")
    parser.add_argument("--grids", default="256")
    parser.add_argument("--wavelengths", default="550,600,620,650,660,700")
    parser.add_argument("--mean-tolerance", type=float, default=0.001, help="Absolute reflectance fraction, on the requested wavelength range")
    parser.add_argument("--max-tolerance", type=float, default=0.005, help="Absolute reflectance fraction at each sampled wavelength")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if not (root/"rcwa_solver_auto.py").is_file():
        parser.error("Pass the tsoyutarcwa project root with --root.")
    sys.path.insert(0, str(root))
    from studies.gold_motheye2 import optimize_adam as shared
    try:
        orders = integer_list(args.orders, 1)
        slices = integer_list(args.slices, 1)
        grids = integer_list(args.grids, 16)
        wavelengths = shared.wavelengths_from_text(args.wavelengths)
    except ValueError as error:
        parser.error(str(error))
    if not all(math.isfinite(v) and v > 0 for v in (args.mean_tolerance, args.max_tolerance)):
        parser.error("Tolerances must be finite and positive.")
    run = (args.run_dir or root/"studies/gold_motheye2/results/adam_fullband_Nz100_M8").resolve()
    output = (args.output_dir or root/"studies/gold_motheye2/results/adam_fixed_profile_convergence").resolve()
    if output == run:
        parser.error("--output-dir must differ from the training --run-dir.")
    config = json.loads((run/"config.json").read_text(encoding="utf-8"))
    training = json.loads((run/"checkpoint.json").read_text(encoding="utf-8"))
    best = training.get("best")
    if best is None:
        parser.error("Training has no saved best profile.")
    best_logits = best["logits"]
    cone_logits = shared.initial_logits(config)
    if len(best_logits) != config["segments"]+2 or not all(math.isfinite(v) for v in best_logits):
        parser.error("Invalid saved profile parameters.")
    gold_csv = (args.gold_csv or root/"studies/gold_motheye2/data/au_measured_nk.csv").resolve()
    if not gold_csv.is_file() or shared.csv_data_hash(gold_csv) != config["gold_csv_sha256_lf"]:
        parser.error("Gold CSV must match the training data.")
    source_paths = list((root/"rcwa_ext").glob("*.py")) + [
        root/"rcwa_solver_auto.py", root/"studies/gold_motheye2/optimize_adam.py",
        root/"studies/gold_motheye/converge.py", root/"studies/shared/gold_dispersion.py"]
    identity = {"schema": 1, "training_signature": training["signature"],
                "geometry": config["geometry"], "segments": config["segments"],
                "diameter_margin_nm": config["diameter_margin_nm"],
                "best_logits": best_logits, "cone_logits": cone_logits,
                "gold_csv_sha256_lf": shared.csv_data_hash(gold_csv),
                "solver_sources_sha256_lf": {str(path.relative_to(root)).replace("\\", "/"): file_hash(path) for path in source_paths}}
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output/"checkpoint.json"
    if checkpoint_path.exists():
        document = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if document.get("identity") != identity:
            raise RuntimeError("Validation profile, material or solver differs. Use a new --output-dir.")
    else:
        document = {"identity": identity, "cases": {}}
    cases = document["cases"]
    def add(label, order, nz, grid, wavelength, value, origin, seconds=None):
        if not math.isfinite(value) or not -1e-6 <= value <= 1+1e-6:
            raise RuntimeError(f"Invalid reflectance: {value}.")
        key = case_key(label, order, nz, grid, wavelength)
        if key not in cases:
            cases[key] = {"label": label, "order": order, "slices": nz, "grid": grid,
                          "wavelength_nm": wavelength, "reflectance": value,
                          "origin": origin, "seconds": seconds}

    predecessor_sources = {"core_sha256": root/"studies/gold_motheye/converge.py",
                           "asr_sha256": root/"rcwa_ext/asr.py", "auto_sha256": root/"rcwa_ext/auto.py",
                           "profile_source_sha256": root/"studies/gold_motheye2/optimize_adam.py"}
    seed_compatible = all(source_matches(config.get(key, ""), path) for key,path in predecessor_sources.items())
    seeded_before = len(cases)
    if seed_compatible:
        for path in sorted(run.glob("dense_validation_M*_*.json")):
            saved = json.loads(path.read_text(encoding="utf-8"))
            stamp = saved.get("identity", {})
            if stamp.get("training_signature") != training["signature"] or stamp.get("best_logits") != best_logits:
                continue
            for label in ("cone", "best"):
                for wavelength, value in saved.get("cases", {}).get(label, {}).items():
                    add(label, int(stamp["order"]), config["slices"], config["grid"], float(wavelength), float(value), "training dense validation")
        for row in training.get("evaluations", []):
            labels = [label for label,logits in (("cone",cone_logits),("best",best_logits)) if row.get("logits") == logits]
            for label in labels:
                for wavelength,value in row.get("values", {}).items():
                    add(label, int(row["order"]), config["slices"], config["grid"], float(wavelength), float(value), "training evaluation")
    planned = [(label,m,n,g,w) for label in ("cone", "best")
               for m,n,g in itertools.product(orders,slices,grids) for w in wavelengths]
    missing = [row for row in planned if case_key(*row) not in cases]
    for nz in slices:
        local = {**config, "slices": nz}
        shared.save_profile(output, f"fixed_best_Nz{nz}", best_logits, local)
        shared.save_profile(output, f"fixed_cone_Nz{nz}", cone_logits, local)
    def persist():
        shared.write_json(checkpoint_path, document)
        fields = ("label", "order", "slices", "grid", "wavelength_nm", "reflectance", "origin", "seconds")
        with (output/"reflectance.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(sorted(cases.values(), key=lambda row: tuple(row[k] for k in fields[:5])))
        shared.write_json(output/"comparisons.json", {"mean_tolerance": args.mean_tolerance, "max_tolerance": args.max_tolerance,
                          "comparisons": comparisons(cases,orders,slices,grids,wavelengths,args.mean_tolerance,args.max_tolerance)})
        shared.write_json(output/"plan.json", {"best_training_step": best["step"], "orders": orders, "slices": slices, "grids": grids,
                          "wavelengths_nm": wavelengths, "requested_cases": len(planned),
                          "missing_cases": [case_key(*row) for row in planned if case_key(*row) not in cases],
                          "training_results_reusable": seed_compatible})
    persist()
    print(f"Fixed best profile: step={best['step']}; {len(planned)} requested cases, {len(missing)} new solves; {len(cases)-seeded_before} imported cases.", flush=True)
    print(f"Results: {output}", flush=True)
    if args.prepare_only:
        return 0
    if not missing:
        print("All requested cases are cached; no optical solves are needed.", flush=True)
        return 0
    import torch
    from studies.shared.gold_dispersion import build_gold_model
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot see a GPU.")
    gold_model = build_gold_model("csv", gold_csv)
    for label,m,n,g,w in missing:
        print(f"solve: {case_key(label,m,n,g,w)}", flush=True)
        started = time.perf_counter()
        local = {**config, "order": m, "slices": n, "grid": g}
        logits = best_logits if label == "best" else cone_logits
        with torch.enable_grad():
            parameters = torch.tensor(logits, dtype=torch.float64, device=device)
            radii = shared.radius_tensor(parameters, n, local, torch)
            reflectance = shared.reflectance_tensor(w, radii, local, gold_model, device, torch)
            value = float(reflectance.detach().cpu())
        del parameters, radii, reflectance
        gc.collect()
        add(label,m,n,g,w,value,"fixed-profile solve",time.perf_counter()-started)
        persist()
        print(f"{case_key(label,m,n,g,w)} R={value:.9f}", flush=True)
    print("Requested validation complete. See comparisons.json; means cover only the requested wavelength range.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
