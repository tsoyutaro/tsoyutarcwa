"""Continue the saved 700 nm study while releasing unused D6 layer storage.

Default: reuse M=12..20 and calculate M=22,24 at Nz=100, grid=576.
--axis slices compares Nz=100,120,140 at M=18 and the same grid.
--slices 140 compares M=16,18,20 on a new 140-slice order sweep.
First reproduce the highest completed order with reduced storage and check
R/P_sub/A against its saved values. No RCWA equations or source files change.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_SEED = HERE / "results" / "order_700_Nz100_grid576" / "checkpoint.json"
METRICS = ("reflectance", "power_into_substrate", "motheye_absorptance")
LEGACY_LISTS = ("P", "Q", "eps_conv", "mu_conv", "E_eigvec", "H_eigvec")
AUXILIARY_LISTS = ("asr_mappings", "asr_material_tensors", "asr_T_matrices",
                   "asr_Tz_matrices", "asr_condition_numbers", "E_eigvec_uv", "H_eigvec_uv")
AUXILIARY_DICTS = ("_asr_slot_by_layer", "_asr_field_context_by_layer", "_physical_material_by_layer")
PARITY_TOLERANCE = 1e-8  # fraction; 0.000001 percentage point


def discard_unused_layer_storage(simulation):
    """Only for an all-D6-source, fields-disabled, fixed-geometry solve.

Reduced electric/magnetic modes, kz, thickness, ports and cascade data stay
alive. Preserve legacy list indices with None slots for completed layers.
"""
    if simulation.store_mode_couplings or simulation.field_regions != "none":
        raise RuntimeError("Storage release requires a fields-disabled solve.")
    if (getattr(simulation, "group_theory_symmetry", None) != "d6"
            or getattr(simulation, "polarization_reduction", None) != "x"
            or len(simulation._polarized_layers) != simulation.layer_N
            or simulation.group_theory_diagnostics[-1].get("symmetry") != "D6-E1-source-row"):
        raise RuntimeError("Storage release requires every layer to use the D6 E1 source row.")
    # Validate all containers before mutation. The layer has already installed
    # independent reduced modes in _polarized_layers before returning here.
    for name in LEGACY_LISTS:
        if len(getattr(simulation, name)) != simulation.layer_N:
            raise RuntimeError(f"Unexpected layer-storage layout: {name}")
    for name in AUXILIARY_LISTS + AUXILIARY_DICTS:
        if not hasattr(simulation, name):
            raise RuntimeError(f"Required auxiliary container is missing: {name}")
    for name in LEGACY_LISTS:
        getattr(simulation, name)[-1] = None
    for name in AUXILIARY_LISTS + AUXILIARY_DICTS:
        getattr(simulation, name).clear()


def prepare(args):
    from studies.gold_motheye3 import converge

    seed = json.loads(args.seed_checkpoint.read_text(encoding="utf-8"))
    base = seed["plan"]
    # Also validate case numerics and the original signature.
    converge._load_checkpoint(args.seed_checkpoint, base)
    if (base["axis"] != "order" or base["wavelengths_nm"] != [700.]
            or base["fixed_numerics"]["slices"] != 100 or base["fixed_numerics"]["grid"] != 576
            or base["solver"]["symmetry_reduction"] != "d6-source"
            or base["solver"]["dtype"] != "complex128"
            or base["geometry"]["lattice"] != "triangular"
            or base["geometry"]["substrate_mode"] != "semi-infinite"):
        raise ValueError("Requires the saved 700 nm order sweep at Nz=100, grid=576, D6-source.")
    if base["source_sha256"] != converge._source_hashes():
        raise ValueError("RCWA/study sources differ from the seed calculation.")
    if converge._digest(Path(base["material"]["path"])) != base["material"]["sha256"]:
        raise ValueError("The Au CSV differs from the seed calculation.")
    if base["geometry"] != converge.GEOMETRY or not seed["cases"]:
        raise ValueError("Requires the same physical geometry and at least one completed reference.")
    default_values = ("100,120,140" if args.axis == "slices" else
                      "22,24" if args.slices == 100 else "16,18,20")
    requested = converge._integers(args.values or default_values)
    plan = copy.deepcopy(base)
    plan["axis"] = args.axis
    plan["values"] = (sorted(set(base["values"]) | set(requested))
                       if args.axis == "order" and args.slices == 100 else list(requested))
    if len(plan["values"]) < 3:
        raise ValueError("At least three values are needed for two adjacent comparisons.")
    if args.axis == "slices":
        plan["fixed_numerics"]["order"] = args.order
    else:
        plan["fixed_numerics"]["slices"] = args.slices
    plan["solver"]["requested_device"] = args.device or base["solver"]["requested_device"]
    plan["solver"]["layer_storage"] = "D6-fixed-geometry-flux-only-release-v1"
    plan["source_sha256"]["studies/gold_motheye3/run_memory_safe.py"] = converge._digest(Path(__file__))
    plan["seed_signature"] = seed["signature"]
    plan["parity_absolute_tolerance"] = PARITY_TOLERANCE
    # Transfer only physically and numerically identical saved cases. Their
    # old runtimes remain explicitly marked as reused rather than remeasured.
    cases = {}
    for value in plan["values"]:
        expected = converge._numerical(plan, value)
        for old_key, row in seed["cases"].items():
            if all(row[name] == number for name, number in expected.items()):
                cases[converge._key(value, row["wavelength_nm"])] = {
                    **row, "axis": args.axis, "value": value,
                    "reused_from": {"signature": seed["signature"], "key": old_key,
                                    "runtime_is_original_measurement": True}}
    reference_key = max(seed["cases"], key=lambda k: seed["cases"][k]["order"])
    reference = {"key": reference_key, "case": seed["cases"][reference_key]}
    return plan, cases, reference


def low_memory_case(wavelength, numbers, plan, device):
    from studies.gold_motheye import converge as shared
    from studies.shared.gold_dispersion import build_gold_model

    original_class = shared.AutoRCWA

    class FluxOnlyRCWA(original_class):
        def add_structured_layer(self, spec):
            method = super().add_structured_layer(spec)
            discard_unused_layer_storage(self)
            return method

    shared.AutoRCWA = FluxOnlyRCWA
    try:
        return shared.simulate_case(
            wavelength, shared.NumericalConfig(**numbers), shared.GeometryConfig(**plan["geometry"]),
            build_gold_model("csv", Path(plan["material"]["path"])),
            cascade=plan["solver"]["cascade"], use_symmetry=True,
            symmetry_reduction="d6-source", device=device)
    finally:
        shared.AutoRCWA = original_class


def measure(wavelength, numbers, plan, device, torch):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    result = low_memory_case(wavelength, numbers, plan, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak = int(torch.cuda.max_memory_allocated(device))
        reserved = int(torch.cuda.max_memory_reserved(device))
    else:
        peak = reserved = None
    if (result["passivity_warning"]
            or result["symmetry_reduction"] != "D6-E1-source-row"
            or any(not math.isfinite(float(result[name]))
                   or not -1e-5 <= float(result[name]) <= 1+1e-5 for name in METRICS)):
        raise RuntimeError(f"Nonphysical or unexpected solver result: {result}")
    return {**result, "wall_seconds": time.perf_counter()-started,
            "peak_cuda_allocated_bytes": peak, "peak_cuda_reserved_bytes": reserved}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis", choices=("order", "slices"), default="order")
    parser.add_argument("--values", help="Default: add 22,24 for order; 100,120,140 for slices.")
    parser.add_argument("--order", type=int, default=18, help="Fixed M for the slices sweep.")
    parser.add_argument("--slices", type=int, default=100,
                        help="Fixed Nz for the order sweep; Nz!=100 defaults to M=16,18,20.")
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"))
    parser.add_argument("--seed-checkpoint", type=Path, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.order <= 0 or args.slices <= 0:
        parser.error("Order and slices must be positive.")
    args.seed_checkpoint = args.seed_checkpoint.resolve()
    try:
        plan, reused, reference = prepare(args)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))
    from studies.gold_motheye3 import converge

    fixed = plan["fixed_numerics"]
    name = (f"order_700_Nz{fixed['slices']}_grid{fixed['grid']}_memory_safe" if args.axis == "order"
            else f"slices_700_M{fixed['order']}_grid{fixed['grid']}_memory_safe")
    output = (args.output_dir or HERE / "results" / name).resolve()
    if args.seed_checkpoint.is_relative_to(output):
        parser.error("Use an output directory separate from the seed checkpoint.")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoint.json"
    saved = converge._load_checkpoint(checkpoint, plan)
    if not checkpoint.exists():
        saved["cases"] = reused
    converge._write_json(output / "plan.json", plan)
    missing = [v for v in plan["values"] if converge._key(v,700.) not in saved["cases"]]
    print(f"Storage policy: release unused layer arrays after each D6 layer; complex128.", flush=True)
    print(f"{args.axis} plan: {plan['values']}; {len(missing)} new solves; "
          f"reference check first (M={reference['case']['order']}).", flush=True)
    if args.prepare_only:
        print(f"plan: {output / 'plan.json'}")
        return 0

    import torch

    requested_device = plan["solver"]["requested_device"]
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if requested_device == "auto" else requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    environment = {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                   "device_type": device.type,
                   "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
    if saved.get("runtime_environment") not in (None, environment):
        raise RuntimeError("The memory-safe checkpoint environment differs. Use another --output-dir.")
    saved["runtime_environment"] = environment
    if not saved.get("storage_parity", {}).get("passed"):
        old = reference["case"]
        numbers = {name: old[name] for name in ("order", "slices", "grid")}
        print(f"verify saved {reference['key']} with reduced layer storage", flush=True)
        checked = measure(old["wavelength_nm"], numbers, plan, device, torch)
        errors = {name: abs(float(checked[name])-float(old[name])) for name in METRICS}
        passed = max(errors.values()) <= PARITY_TOLERANCE
        saved["storage_parity"] = {"passed": passed, "reference_key": reference["key"],
                                    "reference": old, "recomputed": checked,
                                    "absolute_errors": errors, "tolerance": PARITY_TOLERANCE}
        if not passed:
            converge._write_json(output / "storage_parity.json", saved["storage_parity"])
            raise RuntimeError(f"Storage parity failed; no new cases were calculated: {errors}")
        converge._write_json(output / "storage_parity.json", saved["storage_parity"])
        converge._write_json(checkpoint, saved)
        print(f"PASS storage parity: max absolute difference={max(errors.values()):.3g}", flush=True)
        if checked["peak_cuda_allocated_bytes"] is not None:
            print(f"Reference peak CUDA tensor memory: "
                  f"{checked['peak_cuda_allocated_bytes']/2**30:.3f} GiB", flush=True)
    for value in missing:
        numbers = converge._numerical(plan, value)
        print(f"solve {args.axis}={value}, M={numbers['order']}, Nz={numbers['slices']}, "
              f"grid={numbers['grid']}, wavelength=700 nm", flush=True)
        result = measure(700., numbers, plan, device, torch)
        saved["cases"][converge._key(value,700.)] = {**result,"axis":args.axis,"value":value}
        converge._write_json(checkpoint, saved)
        converge._persist(output, saved)
        peak = result["peak_cuda_allocated_bytes"]
        if peak is not None:
            print(f"peak CUDA tensor memory: {peak/2**30:.3f} GiB; "
                  f"wall time: {result['wall_seconds']:.2f} s", flush=True)
    report = converge._persist(output, saved)
    print(f"status: {report['status']} ({report['completed_cases']}/{report['expected_cases']} cases)")
    print(f"report: {output / 'report.json'}")
    print(f"checkpoint: {checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
