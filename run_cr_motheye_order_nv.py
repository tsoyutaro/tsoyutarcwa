"""Sweep Cr Fourier order with the normal-vector method (NV/NVM).

One fresh subprocess per wavelength prevents GPU tensors from accumulating
between cases. Geometry, layer count, normal-vector grid, wavelength anchors,
Lanczos filtering (off), D6 E1 source reduction and complex128 are fixed within
a run. Only M changes. Unused NV layer arrays are released after each layer,
and the Redheffer cascade creates one layer S matrix at a time. No matched-ASR
layer or adaptive coordinate transform is used. Existing ASR scripts remain
separate. An optional same-NV reference checkpoint can check storage parity.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import fields, is_dataclass, replace
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import traceback
import uuid

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from studies.shared.tabulated_nk import load_tabulated_nk

GEOMETRY = dict(period_nm=200.0, height_nm=500.0, tip_radius_nm=5.0,
                base_radius_nm=95.0, profile_power=1.0, lattice="triangular",
                substrate_mode="semi-infinite", substrate_thickness_nm=200.0,
                back_index=1.0)
COLUMNS = ("wavelength_nm", "order", "slices", "grid", "epsilon_material_real",
           "epsilon_material_imag", "reflectance", "transmittance_far",
           "absorptance_total", "motheye_absorptance", "substrate_absorptance",
           "power_into_substrate", "passivity_warning", "reduced_dimension",
           "full_dimension", "symmetry_reduction", "runtime_seconds",
           "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes", "storage_policy",
           "discarded_auxiliary_storage_bytes", "post_layer_retained_tensor_peak_bytes",
           "reference_parity_max_absolute_error", "layer_method", "nv_lanczos_enabled",
           "normal_vector_grid_nx", "normal_vector_grid_ny", "nv_factorization_rule")
METRICS = ("reflectance", "absorptance_total", "motheye_absorptance",
           "substrate_absorptance", "power_into_substrate")
PARITY_TOLERANCE = 1e-8  # Power fraction; 0.000001 percentage point.
LEGACY_LISTS = ("P", "Q", "eps_conv", "mu_conv", "E_eigvec", "H_eigvec")
AUXILIARY_LISTS = ("asr_mappings", "asr_material_tensors", "asr_T_matrices",
                   "asr_Tz_matrices", "asr_condition_numbers", "E_eigvec_uv", "H_eigvec_uv")
AUXILIARY_DICTS = ("_asr_slot_by_layer", "_asr_field_context_by_layer", "_physical_material_by_layer",
                   "_nvm_eps_tensor_by_layer", "_reduced_longitudinal_context_by_layer")
STORAGE_POLICY = "NV-per-layer-D6-release-and-streamed-Redheffer-v1"


def tensor_storage_bytes(items):
    """Unique live tensor storage, excluding allocator cache and autograd graphs."""
    storages = {}
    def visit(value):
        if hasattr(value, "untyped_storage"):
            storage = value.untyped_storage()
            storages[(str(value.device), storage.data_ptr())] = storage.nbytes()
        elif is_dataclass(value) and not isinstance(value, type):
            for field in fields(value):
                visit(getattr(value, field.name))
        elif isinstance(value, dict):
            for entry in value.values():
                visit(entry)
        elif isinstance(value, (tuple, list)):
            for entry in value:
                visit(entry)
    visit(items)
    return sum(storages.values())


def discard_unused_layer_storage(simulation):
    """Keep independent reduced E/H/kz, thickness, ports and diagnostics only.

    Discard full-space operators and NV field-recovery tensors after the NV
    layer has installed independent reduced modes. Preserve all port and
    reduced-cascade data. This is for fixed geometry without fields/gradients.
    """
    if simulation.store_mode_couplings or simulation.field_regions != "none":
        raise RuntimeError("Storage release requires fields disabled.")
    if (getattr(simulation, "group_theory_symmetry", None) != "d6"
            or getattr(simulation, "polarization_reduction", None) != "x"
            or len(simulation._polarized_layers) != simulation.layer_N
            or simulation.group_theory_diagnostics[-1].get("symmetry") != "D6-E1-source-row"
            or simulation.group_theory_diagnostics[-1].get("backend") != "NVM"
            or simulation.layer_records[-1].method != "nvm"):
        raise RuntimeError("Storage release requires every layer to use the NV D6 E1 source row.")
    if any(t.requires_grad for layer in simulation._polarized_layers for t in layer.values()):
        raise RuntimeError("Storage release is only for fixed-geometry, non-gradient calculations.")
    for name in LEGACY_LISTS:
        if len(getattr(simulation, name)) != simulation.layer_N:
            raise RuntimeError(f"Unexpected layer-storage layout: {name}")
    for name in AUXILIARY_LISTS + AUXILIARY_DICTS:
        if not hasattr(simulation, name):
            raise RuntimeError(f"Required auxiliary container is missing: {name}")
    discarded = tensor_storage_bytes([getattr(simulation, name)
                                     for name in LEGACY_LISTS + AUXILIARY_LISTS + AUXILIARY_DICTS])
    for name in LEGACY_LISTS:
        getattr(simulation, name)[-1] = None
    for name in AUXILIARY_LISTS + AUXILIARY_DICTS:
        getattr(simulation, name).clear()
    return discarded


def nv_class(original_class, grid):
    from rcwa_solver_auto import NVMOptions
    class NVRCWA(original_class):
        def __init__(self, *args, **kwargs):
            kwargs["nvm"] = NVMOptions(grid=(grid, grid), use_lanczos=False, lanczos_power=2)
            super().__init__(*args, **kwargs)

        def add_structured_layer(self, spec):
            # Reuse the shared Cr geometry/ports/flux calculation, explicitly
            # selecting NVM instead of its default matched-ASR layer method.
            selected = super().add_structured_layer(replace(spec, method="nvm", factorization_rules=None))
            if selected != "nvm" or self.layer_records[-1].method != "nvm":
                raise RuntimeError("The NV runner unexpectedly selected another layer method.")
            return selected
    return NVRCWA


def memory_safe_class(original_class):
    class FluxOnlyRCWA(original_class):
        def add_structured_layer(self, spec):
            selected = super().add_structured_layer(spec)
            released = discard_unused_layer_storage(self)
            self._discarded_auxiliary_storage_bytes = getattr(self, "_discarded_auxiliary_storage_bytes", 0) + released
            retained = tensor_storage_bytes(vars(self))
            self._post_layer_retained_tensor_peak_bytes = max(
                getattr(self, "_post_layer_retained_tensor_peak_bytes", 0), retained)
            return selected

        def _polarized_redheffer_scattering(self, layers, input_interface, output_interface, *, force_full=False):
            # Same reverse-order half-S recurrence as rcwa_ext/reduced.py.
            # Build and consume one layer S instead of retaining all Nz S's.
            if force_full or self.store_mode_couplings or self.smatrix_size != "half":
                raise RuntimeError("Memory-safe Cr cascade requires half-S output and no fields.")
            if len(layers) != len(self.thickness):
                raise RuntimeError("Reduced layers and thicknesses disagree.")
            identity = self._eye(layers[0]["electric"].shape[0])
            transmission, reflection = output_interface[:2]
            def prepend(left):
                nonlocal transmission, reflection
                tf_left, rf_left, rb_left, tb_left = left
                reflection_new = rf_left + tb_left @ self._solve(
                    identity - reflection @ rb_left, reflection @ tf_left)
                transmission = transmission @ self._solve(
                    identity - rb_left @ reflection, tf_left)
                reflection = reflection_new
            for index in range(len(layers) - 1, -1, -1):
                layer = layers[index]
                layer_s = self._reduced_layer_smatrix(layer["electric"], layer["magnetic"],
                    layer["kz"], self.thickness[index], self._polarization_reference_v)
                prepend(layer_s)
                del layer_s
            prepend(input_interface)
            zero = 0 * identity
            return [transmission, reflection, zero, zero]
    return FluxOnlyRCWA


def nv_case(wave, numerical, geometry, model, device, *, storage="reduced"):
    from studies.gold_motheye import converge as shared
    original_class = shared.AutoRCWA
    metrics = {}
    nv_solver = nv_class(original_class, numerical.grid)
    solver = memory_safe_class(nv_solver) if storage == "reduced" else nv_solver
    class ObservedNVRCWA(solver):
        def solve_global_smatrix(self):
            super().solve_global_smatrix()
            if (len(self.layer_records) != numerical.slices
                    or any(r.method != "nvm" for r in self.layer_records)
                    or any(d.get("backend") != "NVM" for d in self.group_theory_diagnostics)):
                raise RuntimeError("Not every layer used the NV backend.")
            metrics.update(discarded_auxiliary_storage_bytes=getattr(self, "_discarded_auxiliary_storage_bytes", 0),
                post_layer_retained_tensor_peak_bytes=getattr(self, "_post_layer_retained_tensor_peak_bytes", tensor_storage_bytes(vars(self))),
                layer_method="nvm", nv_lanczos_enabled=False,
                normal_vector_grid_nx=numerical.grid, normal_vector_grid_ny=numerical.grid,
                nv_factorization_rule="symmetric-nv-anticommutator")
    # This substitution is scoped to an independent, single-case subprocess.
    shared.AutoRCWA = ObservedNVRCWA
    try:
        row = shared.simulate_case(wave, numerical, geometry, model, cascade="redheffer",
            use_symmetry=True, symmetry_reduction="d6-source", device=device, material_name="Cr")
    finally:
        shared.AutoRCWA = original_class
    row.update(storage_policy=STORAGE_POLICY if storage == "reduced" else "NV-original-full-layer-storage", **metrics)
    return row


def prepare_reference(path, identity):
    saved = json.loads(path.read_text(encoding="utf-8"))
    base = saved["plan"]["identity"]
    if (saved["signature"] != identity_signature(base)
            or saved["plan"]["signature"] != saved["signature"]):
        raise ValueError("Reference checkpoint signature is inconsistent.")
    for name in ("solver", "geometry", "slices", "grid", "material_sha256_lf", "nvm"):
        if base[name] != identity[name]:
            raise ValueError(f"Reference checkpoint differs in {name}.")
    # The original driver has a different hash. Require every mathematical
    # solver/material source to match, while allowing this new storage driver.
    for name, expected in identity["source_sha256"].items():
        if name.startswith(("rcwa_ext/", "studies/")) or name == "rcwa_solver_auto.py":
            if base["source_sha256"].get(name) != expected:
                raise ValueError(f"Reference solver source differs: {name}")
    cases = saved["cases"]
    for key, row in cases.items():
        validate_case(row, base)
        if key != case_key(int(row["order"]), float(row["wavelength_nm"])):
            raise ValueError("Reference case key is inconsistent.")
    complete = [m for m in {int(r["order"]) for r in cases.values()}
                if all(case_key(m, w) in cases for w in identity["wavelengths_nm"])]
    order = max(complete, default=None)
    if order is None or identity["start_order"] != order:
        raise ValueError(f"Set --start-order to the reference's last complete order ({order}).")
    return {"path": str(path), "checkpoint_sha256_lf": digest(path), "signature": saved["signature"],
            "order": order, "absolute_tolerance": PARITY_TOLERANCE,
            "cases": {case_key(order, w): cases[case_key(order, w)] for w in identity["wavelengths_nm"]},
            "environment": saved.get("environment")}


def compare_reference(row, reference):
    if reference is None:
        return None
    old = reference["cases"].get(case_key(int(row["order"]), float(row["wavelength_nm"])))
    if old is None:
        return None
    errors = {name: abs(float(row[name]) - float(old[name])) for name in METRICS}
    if any(not math.isfinite(v) for v in errors.values()) or max(errors.values()) > reference["absolute_tolerance"]:
        raise ValueError(f"Storage parity failed; higher orders will not be computed: {errors}")
    return {"passed": True, "absolute_errors": errors, "tolerance": reference["absolute_tolerance"]}


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def identity_signature(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()


def case_key(order, wave):
    return f"M={order}|wl={wave:.12g}"


def source_hashes():
    paths = [Path(__file__), ROOT / "studies/gold_motheye/converge.py",
             ROOT / "studies/shared/tabulated_nk.py", ROOT / "rcwa_solver_auto.py"]
    paths += sorted((ROOT / "rcwa_ext").rglob("*.py"))
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}


def validate_case(row, identity):
    order, wave = int(row["order"]), float(row["wavelength_nm"])
    if (int(row["slices"]) != identity["slices"] or int(row["grid"]) != identity["grid"]
            or wave not in identity["wavelengths_nm"] or order < identity["start_order"]
            or (order - identity["start_order"]) % identity["step"]):
        raise ValueError("Saved case conditions disagree with the sweep identity.")
    if (row.get("layer_method") != "nvm" or row.get("nv_lanczos_enabled") is not False
            or row.get("normal_vector_grid_nx") != identity["grid"]
            or row.get("normal_vector_grid_ny") != identity["grid"]):
        raise ValueError("Saved case is not the requested fixed-grid NV calculation.")
    names = ("reflectance", "transmittance_far", "absorptance_total",
             "motheye_absorptance", "substrate_absorptance", "power_into_substrate")
    powers = {name: float(row[name]) for name in names}
    if row["passivity_warning"] or any(not math.isfinite(v) or v < -1e-5 or v > 1 + 1e-5 for v in powers.values()):
        raise ValueError("Nonphysical power result; this is not classified as CUDA OOM.")
    if abs(powers["reflectance"] + powers["transmittance_far"] + powers["absorptance_total"] - 1) > 1e-8:
        raise ValueError("Saved power balance is inconsistent.")


def persist(output, saved):
    identity = saved["plan"]["identity"]
    rows = sorted(saved["cases"].values(), key=lambda r: (r["order"], r["wavelength_nm"]))
    complete_orders = sorted(m for m in {int(r["order"]) for r in rows}
                             if all(case_key(m, w) in saved["cases"] for w in identity["wavelengths_nm"]))
    saved["last_complete_order"] = max(complete_orders, default=None)
    write_json(output / "checkpoint.json", saved)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=COLUMNS, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(output / "all_cases.csv", stream.getvalue())
    report = {"status": saved["status"], "signature": saved["signature"],
              "fixed_slices": identity["slices"], "fixed_grid": identity["grid"],
              "wavelengths_nm": identity["wavelengths_nm"], "completed_cases": len(rows),
              "complete_orders": complete_orders, "last_complete_order": saved["last_complete_order"],
              "failure": saved.get("failure"), "requested_max_order": saved["plan"]["max_order"],
              "method": "NV (NVM), symmetric normal-vector factorization, Lanczos off",
              "interpretation": "A fixed-Nz/fixed-NV-grid order sweep, not a three-axis convergence certification."}
    reference = saved["plan"]["identity"].get("reference")
    if reference is not None:
        parity = {key: saved["cases"][key].get("reference_parity")
                  for key in reference["cases"] if key in saved["cases"]}
        memory = {}
        for key, old in reference["cases"].items():
            if key not in saved["cases"]:
                continue
            new = saved["cases"][key]
            old_peak, new_peak = old.get("peak_cuda_allocated_bytes"), new.get("peak_cuda_allocated_bytes")
            memory[key] = {"original_peak_cuda_allocated_bytes": old_peak,
                "memory_safe_peak_cuda_allocated_bytes": new_peak,
                "cuda_peak_reduction_percent": 100 * (1 - new_peak / old_peak)
                    if old_peak is not None and old_peak > 0 and new_peak is not None else None}
        report["reference_parity"] = {"order": reference["order"], "cases": parity,
            "passed_all_wavelengths": len(parity) == len(reference["cases"]) and all(p and p["passed"] for p in parity.values()),
            "memory_comparison": memory}
        write_json(output / "storage_parity.json", report["reference_parity"])
    write_json(output / "report.json", report)


def initialize_device(torch_module, requested):
    device = torch_module.device(requested)
    if device.type == "cuda":
        if not torch_module.cuda.is_available():
            raise ValueError("Requested CUDA device is unavailable.")
        index = device.index if device.index is not None else torch_module.cuda.current_device()
        if not 0 <= index < torch_module.cuda.device_count():
            raise ValueError("Requested CUDA device is unavailable.")
        # CUDA management APIs require an explicit index even when the solver
        # accepts the generic device name "cuda".
        device = torch_module.device("cuda", index)
        torch_module.cuda.set_device(device)
        torch_module.cuda.reset_peak_memory_stats(device)
    return device


def worker(plan_path, result_path, order, wave):
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    identity = plan["identity"]
    torch = None
    try:
        if source_hashes() != identity["source_sha256"] or digest(identity["cr_csv"]) != identity["material_sha256_lf"]:
            raise ValueError("Source or material changed after the plan was written.")
        import torch
        from studies.gold_motheye.converge import GeometryConfig, NumericalConfig
        torch.set_num_threads(identity["cpu_threads"])
        device = initialize_device(torch, identity["device"])
        model = load_tabulated_nk(identity["cr_csv"])
        numerical = NumericalConfig(order, identity["slices"], identity["grid"])
        geometry = GeometryConfig(**identity["geometry"])
        row = nv_case(wave, numerical, geometry, model, device, storage=identity["storage"])
        validate_case(row, identity)
        parity = compare_reference(row, identity.get("reference"))
        if parity is not None:
            row.update(reference_parity=parity, reference_parity_max_absolute_error=max(parity["absolute_errors"].values()))
            print(f"  PASS storage parity: max absolute power difference={row['reference_parity_max_absolute_error']:.3g}", flush=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        row.update(peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
                   peak_cuda_reserved_bytes=int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None)
        environment = {"torch": str(torch.__version__), "device": str(device), "cuda": torch.version.cuda,
                       "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
        write_json(result_path, {"status": "ok", "signature": plan["signature"], "case": row, "environment": environment})
        print(f"  R={100*row['reflectance']:.6f}%, A={100*row['absorptance_total']:.6f}%", flush=True)
        if device.type == "cuda":
            print(f"  peak CUDA allocated={row['peak_cuda_allocated_bytes']/2**30:.3f} GiB, reserved={row['peak_cuda_reserved_bytes']/2**30:.3f} GiB", flush=True)
        return 0
    except Exception as error:
        oom = torch is not None and isinstance(error, torch.cuda.OutOfMemoryError) and identity["device"].startswith("cuda")
        failure = {"status": "oom" if oom else "calculation_failed", "signature": plan["signature"],
                   "order": order, "wavelength_nm": wave, "slices": identity["slices"], "grid": identity["grid"],
                   "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
        if "row" in locals():
            failure["power_diagnostics"] = {name: float(row[name]) if math.isfinite(float(row[name]))
                else repr(float(row[name])) for name in METRICS}
        write_json(result_path, failure)
        print(failure["error"], file=sys.stderr, flush=True)
        return 2 if oom else 1


def launch_worker(plan_path, result_path, order, wave):
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--_worker-plan", str(plan_path),
               "--_worker-result", str(result_path), "--_worker-order", str(order), "--_worker-wave", str(wave)]
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if not result_path.exists():
        return {"status": "calculation_failed", "order": order, "wavelength_nm": wave,
                "error": f"Worker exited {completed.returncode} without a result. Inspect its terminal output."}
    result = json.loads(result_path.read_text(encoding="utf-8"))
    expected = 0 if result["status"] == "ok" else 2 if result["status"] == "oom" else 1
    if completed.returncode != expected:
        return {"status": "calculation_failed", "order": order, "wavelength_nm": wave,
                "error": f"Worker result and exit code disagree ({completed.returncode})."}
    return result


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda", help="cuda, cuda:0 etc., or cpu")
    p.add_argument("--storage", choices=("reduced", "full"), default="reduced",
                   help="reduced: release unused layer data and stream S matrices; full: original reference path")
    p.add_argument("--reference-checkpoint", type=Path,
                   help="Optional NV checkpoint only (never ASR); recompute its last complete M first")
    p.add_argument("--slices", type=int, default=60, help="Fixed Nz; no layer-count scan")
    p.add_argument("--grid", type=int, default=512, help="Fixed Q x Q sampling grid for the NV normal field")
    p.add_argument("--start-order", type=int, default=4)
    p.add_argument("--step", type=int, default=2)
    p.add_argument("--max-order", type=int, default=36, help="Default 36; 0: until OOM or the fixed-grid validity limit")
    p.add_argument("--wavelengths", default="400,550,700")
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--cr-csv", type=Path, default=ROOT / "studies/cr_motheye/data/Johnson.csv")
    p.add_argument("--output-dir", type=Path, default=ROOT / "studies/cr_motheye/results/nv_order_Nz60_Q512")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--_worker-plan", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--_worker-result", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--_worker-order", type=int, help=argparse.SUPPRESS)
    p.add_argument("--_worker-wave", type=float, help=argparse.SUPPRESS)
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args._worker_plan is not None:
        return worker(args._worker_plan, args._worker_result, args._worker_order, args._worker_wave)
    try:
        if min(args.slices, args.grid, args.start_order, args.step, args.cpu_threads) <= 0 or args.max_order < 0:
            raise ValueError("Slices, grid, start-order, step and threads must be positive; max-order must be >=0.")
        if args.max_order and args.max_order < args.start_order:
            raise ValueError("max-order must be 0 or >= start-order.")
        if args.grid < max(32, 4 * args.start_order + 4):
            raise ValueError("grid must be >= max(32, 4*M+4) for the starting order.")
        if args.device != "cpu" and args.device != "cuda" and not (args.device.startswith("cuda:") and args.device[5:].isdigit()):
            raise ValueError("Use cpu, cuda, or cuda:<index>.")
        waves = sorted(set(float(v.strip()) for v in args.wavelengths.split(",")))
        if not waves or any(not math.isfinite(w) or w <= 0 for w in waves):
            raise ValueError("Use positive finite comma-separated wavelengths in nm.")
        csv_path = args.cr_csv.resolve()
        model = load_tabulated_nk(csv_path)
        for wave in waves:
            model(wave)
        identity = {"geometry": GEOMETRY, "slices": args.slices, "grid": args.grid, "wavelengths_nm": waves,
                    "start_order": args.start_order, "step": args.step, "device": args.device,
                    "cpu_threads": args.cpu_threads, "cr_csv": str(csv_path), "material_sha256_lf": digest(csv_path),
                    "source_sha256": source_hashes(), "solver": "NV (NVM), D6-source, complex128, Redheffer",
                    "nvm": {"grid": [args.grid, args.grid], "use_lanczos": False, "lanczos_power": 2,
                            "factorization_rule": "symmetric-nv-anticommutator"},
                    "scheduling": "one independent process per wavelength and order; stop at first CUDA OOM"}
        identity.update(storage=args.storage,
            storage_policy=STORAGE_POLICY if args.storage == "reduced" else "NV-original-full-layer-storage")
        if args.reference_checkpoint is not None:
            identity["reference"] = prepare_reference(args.reference_checkpoint.resolve(), identity)
        signature = identity_signature(identity)
        output = args.output_dir.resolve()
        if args.reference_checkpoint is not None and args.reference_checkpoint.resolve().is_relative_to(output):
            raise ValueError("Use an output directory separate from the reference checkpoint.")
        plan = {"identity": identity, "signature": signature, "max_order": args.max_order,
                "fixed_grid_maximum_order": (args.grid - 4) // 4}
        checkpoint_path = output / "checkpoint.json"
        if checkpoint_path.exists():
            saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if saved["signature"] != signature:
                raise ValueError("Checkpoint source/material/conditions differ; use a new --output-dir.")
            for key, row in saved["cases"].items():
                validate_case(row, identity)
                parity = compare_reference(row, identity.get("reference"))
                if parity is not None and row.get("reference_parity") != parity:
                    raise ValueError("Saved reference parity is inconsistent.")
                if key != case_key(int(row["order"]), float(row["wavelength_nm"])):
                    raise ValueError("Checkpoint case key is inconsistent.")
        else:
            saved = {"signature": signature, "cases": {}}
        saved.update(plan=plan, status="running", failure=None)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / ("plan_preview.json" if args.dry_run else "plan.json"), plan)
    print(f"Fixed Nz={args.slices}, Q={args.grid}; M={args.start_order}, step={args.step}; wavelengths={waves}", flush=True)
    print(f"Output: {output}; resume {len(saved['cases'])} saved cases", flush=True)
    print(f"Method: NV (NVM); normal grid={args.grid} x {args.grid}; Lanczos off; D6-source, complex128, Redheffer.", flush=True)
    print(f"Storage: {identity['storage_policy']}", flush=True)
    if identity.get("reference"):
        print(f"Recompute reference M={identity['reference']['order']} at all wavelengths before higher orders; absolute power tolerance={PARITY_TOLERANCE:g}.", flush=True)
    if args.dry_run:
        print("Dry run: no electromagnetic calculation; existing checkpoint unchanged.", flush=True)
        return 0
    persist(output, saved)
    order = args.start_order
    try:
        while not args.max_order or order <= args.max_order:
            if args.grid < max(32, 4 * order + 4):
                saved.update(status="fixed_grid_limit", failure={"order": order,
                             "error": f"Q={args.grid} is below required 4*M+4={4*order+4}; use another output-dir for a larger fixed grid."})
                persist(output, saved)
                print("Stopped at the fixed-grid validity limit; CUDA OOM was not observed.", flush=True)
                return 3
            for wave in waves:
                key = case_key(order, wave)
                if key in saved["cases"]:
                    continue
                print(f"solve: lambda={wave:g} nm, M={order}, Nz={args.slices}, grid={args.grid}", flush=True)
                result_path = output / "worker_results" / f"M{order}_wl{wave:g}_{uuid.uuid4().hex}.json"
                result = launch_worker(output / "plan.json", result_path, order, wave)
                if result["status"] != "ok":
                    saved.update(status=result["status"], failure=result)
                    persist(output, saved)
                    print(f"Stopped: {saved['status']}; last order complete at all wavelengths: {saved['last_complete_order']}", flush=True)
                    print(f"Saved cases: {output / 'all_cases.csv'}", flush=True)
                    return 2 if saved["status"] == "oom" else 1
                if result.get("signature") != signature:
                    raise ValueError("Worker signature disagrees with the plan.")
                if saved.get("environment") not in (None, result["environment"]):
                    raise ValueError("Worker runtime environment changed; use another output-dir.")
                row = result["case"]
                validate_case(row, identity)
                parity = compare_reference(row, identity.get("reference"))
                if parity is not None and row.get("reference_parity") != parity:
                    raise ValueError("Worker reference parity is inconsistent.")
                if case_key(int(row["order"]), float(row["wavelength_nm"])) != key:
                    raise ValueError("Worker returned the wrong case.")
                saved["environment"] = result["environment"]
                saved["cases"][key] = row
                persist(output, saved)
            order += args.step
    except KeyboardInterrupt:
        saved.update(status="interrupted", failure={"order": order, "error": "KeyboardInterrupt"})
        persist(output, saved)
        print("Interrupted; completed cases are saved.", flush=True)
        return 130
    except Exception as error:
        saved.update(status="calculation_failed", failure={"order": order, "error": f"{type(error).__name__}: {error}"})
        persist(output, saved)
        raise
    saved["status"] = "requested_range_complete"
    persist(output, saved)
    print(f"Requested range complete; last complete M={saved['last_complete_order']}. This is not a convergence certification.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
