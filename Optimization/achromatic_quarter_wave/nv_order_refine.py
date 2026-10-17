"""Fixed-Q, fixed-z NV order refinement, reusing a verified experiment cache.

The reference experiment supplies the material, periods, height and profile.
This driver does not change the NV operators or reinterpret a fixed Q/z axis
as an unperformed convergence test. Only the requested M comparison is judged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Optimization.achromatic_quarter_wave.model import Config
from Optimization.achromatic_quarter_wave.nv_convergence import differences, fingerprint, simulate_nv
from Optimization.achromatic_quarter_wave.run import save_csv, save_json


def verify_sources(expected):
    """Accept CRLF/LF serialization only, never a change to Python source text."""
    current = fingerprint()
    if set(current) != set(expected):
        raise ValueError("Reference source inventory differs from the current NV implementation.")
    canonical, serialized_differences = {}, []
    for name in sorted(current):
        data = (ROOT / name).read_bytes()
        lf = data.replace(b"\r\n", b"\n")
        hashes = {hashlib.sha256(x).hexdigest() for x in (data, lf, lf.replace(b"\n", b"\r\n"))}
        if expected[name] not in hashes:
            raise ValueError(f"Reference physics source differs: {name}. Run a new reference experiment.")
        canonical[name] = hashlib.sha256(lf).hexdigest()
        if expected[name] != current[name]:
            serialized_differences.append(name)
    return canonical, serialized_differences


def verified_reuse_directories(directories, identity):
    """Reuse refinement caches only when their physical calculation agrees."""
    verified = []
    for directory in directories:
        directory = directory.resolve()
        path = directory / "refinement_metadata.json"
        if not path.is_file():
            raise ValueError(f"Reuse directory must contain refinement_metadata.json: {directory}")
        metadata = json.loads(path.read_text(encoding="utf-8"))
        previous = metadata["identity"]
        for key in ("model", "shape", "grid", "layers", "device", "normalized_source_sha256"):
            if previous.get(key) != identity[key]:
                raise ValueError(f"Reuse calculation differs in {key}: {directory}")
        if metadata.get("dtype") != "complex128":
            raise ValueError(f"Reuse calculation must use complex128: {directory}")
        verified.append((directory, previous["orders"], previous["wavelengths"]))
    return verified


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--shape", choices=("rectangle", "ellipse"), required=True)
    p.add_argument("--orders", type=int, nargs="+", default=[11, 13])
    p.add_argument("--grid", type=int, default=1024)
    p.add_argument("--layers", type=int, default=30)
    p.add_argument("--wavelengths", type=float, nargs="+", required=True)
    p.add_argument("--device", choices=("cpu", "cuda"))
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--phase-tol", type=float)
    p.add_argument("--power-tol", type=float)
    p.add_argument("--reuse", type=Path, nargs="+", default=[],
                   help="Reuse compatible earlier refinement outputs after checking their physical settings and source hashes")
    p.add_argument("--cache-only", action="store_true", help="Read cached results only; never initialize CUDA or solve RCWA")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if (len(set(args.orders)) != len(args.orders) or len(args.orders) < 2
            or min(args.orders) < 1 or min(args.grid, args.layers, args.threads) < 1
            or len(set(args.wavelengths)) != len(args.wavelengths)
            or any(not np.isfinite(w) or w <= 0 for w in args.wavelengths)):
        p.error("Require at least two unique positive orders and positive grid, layers, threads and wavelengths.")
    args.orders, args.wavelengths = sorted(args.orders), sorted(args.wavelengths)
    reference = args.reference.resolve()
    meta = json.loads((reference / "run_metadata.json").read_text(encoding="utf-8"))
    a = meta["arguments"]
    if a["mode"] != "converge" or args.shape not in a["shapes"]:
        raise ValueError("Reference must be a convergence experiment containing the requested shape.")
    canonical, eol_changes = verify_sources(meta["source_sha256"])
    device = args.device or a["device"]
    if device != a["device"] and not args.cache_only:
        raise ValueError("Reference and calculation devices differ; use the reference device.")
    phase_tol = a["phase_tol"] if args.phase_tol is None else args.phase_tol
    power_tol = a["power_tol"] if args.power_tol is None else args.power_tol
    if min(phase_tol, power_tol) <= 0:
        p.error("Error tolerances must be positive.")
    model = {key:a[key] for key in ("index", "substrate_index", "period", "height", "x_knots", "y_knots")}
    identity = dict(model=model, shape=args.shape, orders=args.orders, grid=args.grid,
                    layers=args.layers, wavelengths=args.wavelengths, device=device,
                    phase_tol=phase_tol, power_tol=power_tol, energy_tol=a["energy_tol"],
                    normalized_source_sha256=canonical,
                    driver_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest())
    reuse = verified_reuse_directories(args.reuse, identity)
    out = args.output.resolve()
    if (out / "run_metadata.json").exists():
        raise ValueError("Use a separate refinement output directory; keep the reference experiment unchanged.")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "refinement_metadata.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8"))["identity"] != identity:
        raise ValueError("Refinement settings or physics sources changed; choose a new --output.")
    save_json(path, dict(identity=identity, reference_directory=str(reference),
                         reuse_directories=[str(directory) for directory, _, _ in reuse],
                         reference_files_differing_only_by_newlines=eol_changes,
                         dtype="complex128", scope="Order convergence at fixed Q and fixed z slices"))
    torch.set_num_threads(args.threads)
    rows, reused, computed = [], 0, 0
    for m in args.orders:
        for w in args.wavelengths:
            key = f"{args.shape}_M{m}_Q{args.grid}_Z{args.layers}_w{w:g}"
            own, previous = out / "cache" / (key + ".json"), reference / "cache" / (key + ".json")
            candidates = [(own, "output_cache"), (previous, "reference_cache")]
            candidates.extend((directory / "cache" / (key + ".json"), "refinement_cache")
                              for directory, orders, wavelengths in reuse
                              if m in orders and w in wavelengths)
            cached = next(((path, label) for path, label in candidates if path.is_file()), None)
            if cached is not None:
                source, origin = cached
                result = json.loads(source.read_text(encoding="utf-8"))
                elapsed = 0.
                reused += 1
            else:
                if args.cache_only:
                    raise FileNotFoundError(f"Missing {key}; --cache-only forbids a new RCWA calculation.")
                cfg = Config(period_x_nm=a["period"], period_y_nm=a["period"], height_nm=a["height"],
                             n_substrate=a["substrate_index"], device=device,
                             order=m, grid=args.grid, layers=args.layers)
                start = time.perf_counter()
                simulation, result = simulate_nv([a["x_knots"], a["y_knots"]], args.shape, a["index"], w, cfg)
                # Metrics are CPU scalars. Release this wavelength's modal/S
                # arrays before constructing the next large GPU simulation.
                del simulation
                elapsed = time.perf_counter() - start
                result["seconds"] = elapsed
                save_json(own, result)
                source = own
                origin = "computed"
                computed += 1
            if abs(result["wavelength_nm"] - w) > 1e-9:
                raise ValueError(f"Cached wavelength disagrees with filename: {key}")
            rows.append(dict(shape=args.shape, order=m, grid=args.grid, layers=args.layers,
                             origin=origin, cache_source=str(source), new_calculation_seconds=elapsed, **result))
            save_csv(out / "order_spectra.csv", rows)
            print(key, origin, f"{elapsed:.2f} s", flush=True)
    comparisons = []
    for w in args.wavelengths:
        first = next(r for r in rows if r["order"] == args.orders[-2] and r["wavelength_nm"] == w)
        last = next(r for r in rows if r["order"] == args.orders[-1] and r["wavelength_nm"] == w)
        diff = differences(first, last)
        valid = diff["retardance_change_deg"] is not None
        energy = max(first["energy_error"], last["energy_error"])
        passed = bool(valid and diff["retardance_change_deg"] <= phase_tol
                      and diff["transmission_phase_change_deg"] <= phase_tol
                      and diff["power_change"] <= power_tol and energy <= a["energy_tol"])
        comparisons.append(dict(wavelength_nm=w, order_from=args.orders[-2], order_to=args.orders[-1],
                                status="within_tolerance" if passed else "needs_refinement",
                                passed=passed, **diff, energy_error=energy,
                                retardance_from_deg=first["retardance_deg"],
                                retardance_to_deg=last["retardance_deg"]))
    save_csv(out / "order_comparisons.csv", comparisons)
    summary = dict(order_within_tolerances=all(r["passed"] for r in comparisons),
                   fixed_grid=args.grid, fixed_layers=args.layers, reused_results=reused,
                   new_calculations=computed, comparisons=comparisons,
                   note="Only order convergence is assessed. Q/z convergence and continuous-band accuracy are not certified here.")
    save_json(out / "order_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
