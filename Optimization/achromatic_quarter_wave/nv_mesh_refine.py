"""Fixed-order NV grid/layer checks, reusing verified existing calculations.

Vary Q at the smallest requested layer count, then vary the layer count at
the largest requested Q. The numerical NV implementation is unchanged.
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
from Optimization.achromatic_quarter_wave.nv_convergence import differences, simulate_nv
from Optimization.achromatic_quarter_wave.nv_order_refine import verify_sources
from Optimization.achromatic_quarter_wave.run import save_csv, save_json


def reuse_sources(directories, identity):
    sources = []
    for directory in directories:
        directory = directory.resolve()
        path = directory / "refinement_metadata.json"
        if not path.is_file():
            raise ValueError(f"Reuse directory must contain refinement_metadata.json: {directory}")
        metadata = json.loads(path.read_text(encoding="utf-8"))
        previous = metadata["identity"]
        for key in ("model", "shape", "device", "normalized_source_sha256"):
            if previous.get(key) != identity[key]:
                raise ValueError(f"Reuse calculation differs in {key}: {directory}")
        if metadata.get("dtype") != "complex128":
            raise ValueError(f"Reuse calculation must use complex128: {directory}")
        sources.append((directory, previous))
    return sources


def covers(settings, order, grid, layers, wavelength):
    """Require the key to be covered by the source's declared settings."""
    grids = settings.get("grids", [settings.get("grid")])
    zlevels = settings.get("layer_counts", [settings.get("layers")])
    return (order in settings["orders"] and grid in grids and layers in zlevels
            and wavelength in settings["wavelengths"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--reuse", type=Path, nargs="+", default=[])
    p.add_argument("--shape", choices=("ellipse", "rectangle"), required=True)
    p.add_argument("--order", type=int, default=15)
    p.add_argument("--grids", type=int, nargs="+", default=[1024, 2048])
    p.add_argument("--layers", type=int, nargs="+", default=[30])
    p.add_argument("--wavelengths", type=float, nargs="+", required=True)
    p.add_argument("--device", choices=("cpu", "cuda"))
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--cache-only", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    for name in ("grids", "layers", "wavelengths"):
        values = getattr(args, name)
        if len(set(values)) != len(values) or any(not np.isfinite(v) or v <= 0 for v in values):
            p.error(f"--{name} values must be positive, finite and unique.")
        setattr(args, name, sorted(values))
    if min(args.order, args.threads) < 1 or max(len(args.grids), len(args.layers)) < 2:
        p.error("Require positive order/threads and at least two grid or layer levels.")
    reference = args.reference.resolve()
    meta = json.loads((reference / "run_metadata.json").read_text(encoding="utf-8"))
    a = meta["arguments"]
    if a["mode"] != "converge" or args.shape not in a["shapes"]:
        raise ValueError("Reference must be a convergence experiment containing the requested shape.")
    canonical, eol_changes = verify_sources(meta["source_sha256"])
    device = args.device or a["device"]
    if device != a["device"]:
        raise ValueError("Reference and calculation devices must agree; cache-only needs no CUDA initialization.")
    identity = dict(
        model={key: a[key] for key in ("index", "substrate_index", "period", "height", "x_knots", "y_knots")},
        shape=args.shape, orders=[args.order], grids=args.grids, layer_counts=args.layers,
        wavelengths=args.wavelengths, device=device,
        phase_tol=a["phase_tol"], power_tol=a["power_tol"], energy_tol=a["energy_tol"],
        normalized_source_sha256=canonical,
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
    )
    extra = reuse_sources(args.reuse, identity)
    out = args.output.resolve()
    if (out / "run_metadata.json").exists():
        raise ValueError("Keep the original convergence experiment unchanged; use a new --output.")
    path = out / "refinement_metadata.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8"))["identity"] != identity:
        raise ValueError("Refinement settings or source changed; use a new --output.")
    out.mkdir(parents=True, exist_ok=True)
    save_json(path, dict(identity=identity, reference_directory=str(reference),
                         reuse_directories=[str(d) for d, _ in extra], dtype="complex128",
                         reference_files_differing_only_by_newlines=eol_changes,
                         scope="Grid/layer changes at one fixed Fourier order"))
    torch.set_num_threads(args.threads)
    reference_settings = dict(orders=a["orders"], grids=a["grids"],
                              layer_counts=a["layers"], wavelengths=a["wavelengths"])
    sources = [(out, identity, "output_cache"), (reference, reference_settings, "reference_cache")]
    sources += [(d, settings, "refinement_cache") for d, settings in extra]
    schedule = list(dict.fromkeys([(q, args.layers[0]) for q in args.grids]
                                 + [(args.grids[-1], z) for z in args.layers]))
    rows, reused, computed = [], 0, 0
    for grid, layers in schedule:
        for w in args.wavelengths:
            key = f"{args.shape}_M{args.order}_Q{grid}_Z{layers}_w{w:g}"
            cached = next(((d / "cache" / (key + ".json"), origin)
                           for d, settings, origin in sources
                           if covers(settings, args.order, grid, layers, w)
                           and (d / "cache" / (key + ".json")).is_file()), None)
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
                             order=args.order, grid=grid, layers=layers)
                start = time.perf_counter()
                simulation, result = simulate_nv([a["x_knots"], a["y_knots"]], args.shape, a["index"], w, cfg)
                del simulation
                elapsed = time.perf_counter() - start
                result["seconds"] = elapsed
                source = out / "cache" / (key + ".json")
                save_json(source, result)
                origin = "computed"
                computed += 1
            if abs(result["wavelength_nm"] - w) > 1e-9:
                raise ValueError(f"Cached wavelength disagrees with filename: {key}")
            rows.append(dict(shape=args.shape, order=args.order, grid=grid, layers=layers,
                             origin=origin, cache_source=str(source), new_calculation_seconds=elapsed, **result))
            save_csv(out / "mesh_spectra.csv", rows)
            print(key, origin, f"{elapsed:.2f} s", flush=True)
    comparisons = []
    for axis, levels in (("grid", args.grids), ("layers", args.layers)):
        if len(levels) < 2:
            continue  # A fixed axis is deliberately not assessed.
        for w in args.wavelengths:
            def row_at(level):
                q, z = (level, args.layers[0]) if axis == "grid" else (args.grids[-1], level)
                return next(r for r in rows if r["grid"] == q and r["layers"] == z and r["wavelength_nm"] == w)
            first, last = row_at(levels[-2]), row_at(levels[-1])
            diff = differences(first, last)
            energy = max(first["energy_error"], last["energy_error"])
            passed = bool(diff["retardance_change_deg"] is not None
                          and diff["retardance_change_deg"] <= a["phase_tol"]
                          and diff["transmission_phase_change_deg"] <= a["phase_tol"]
                          and diff["power_change"] <= a["power_tol"] and energy <= a["energy_tol"])
            comparisons.append(dict(axis=axis, wavelength_nm=w, level_from=levels[-2], level_to=levels[-1],
                                    fixed_order=args.order,
                                    fixed_layers=args.layers[0] if axis == "grid" else None,
                                    fixed_grid=args.grids[-1] if axis == "layers" else None,
                                    status="within_tolerance" if passed else "needs_refinement",
                                    passed=passed, **diff, energy_error=energy,
                                    retardance_from_deg=first["retardance_deg"], retardance_to_deg=last["retardance_deg"]))
    summary = dict(requested_mesh_checks_within_tolerances=all(r["passed"] for r in comparisons),
                   fixed_order=args.order, assessed_axes=sorted({r["axis"] for r in comparisons}),
                   reused_results=reused, new_calculations=computed, comparisons=comparisons,
                   note="Only requested grid/layer last-pair changes are assessed; order and continuous-band accuracy are not certified.")
    save_csv(out / "mesh_comparisons.csv", comparisons)
    save_json(out / "mesh_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
