"""Warm timings, CUDA allocation and checked shape derivatives at a fixed order."""
from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import asdict, replace
from pathlib import Path

from studies.shared.convergence_io import content_hash, signature_for, write_json
from .model import physical, release, solve, synchronize, trapezoid_weights

PARAMETERS = ("height_nm", "base_radius_nm")


def compatible(a, b, atol, rtol):
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= atol + rtol * max(abs(a), abs(b))


def objective(study, numerical, wavelengths, geometry=None, *, gradient=False):
    import torch
    geometry = geometry or study.geometry
    value, gradients = 0., dict.fromkeys(PARAMETERS, 0.)
    weights = trapezoid_weights(wavelengths)
    rows = []
    # Accumulate each wavelength's gradient separately. This is exactly the
    # gradient of the weighted sum, with only one wavelength graph in memory.
    for w, weight in zip(wavelengths, weights):
        tensors = {name: torch.tensor(getattr(geometry, name), dtype=torch.float64,
                                      device=study.device, requires_grad=True) for name in PARAMETERS} if gradient else None
        row, reflected = solve(w, numerical, geometry, method=study.method, lattice=study.lattice,
                               device=study.device, use_symmetry=study.use_symmetry, parameters=tensors,
                               square_asr_profile=study.square_asr_profile, checkpoint_layers=study.checkpoint_layers,
                               asr_reference_geometry=study.geometry)
        rows.append(row)
        value += weight * row["reflectance"]
        if gradient:
            derivatives = torch.autograd.grad(reflected, tuple(tensors.values()))
            for name, derivative in zip(PARAMETERS, derivatives):
                number = float(derivative.detach().cpu())
                if not math.isfinite(number):
                    raise RuntimeError(f"Nonfinite {name} autograd derivative.")
                gradients[name] += weight * number
            del derivatives
        del reflected, tensors
    return {"objective": value, "gradient_per_nm": gradients if gradient else None,
            "max_energy_residual": max(abs(row["energy_residual"]) for row in rows),
            "rows": rows}


def timed_objective(study, numerical, wavelengths, *, gradient):
    import torch
    release(study.device)
    synchronize(study.device)
    if study.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(study.device)
    start = time.perf_counter()
    result = objective(study, numerical, wavelengths, gradient=gradient)
    synchronize(study.device)
    result["seconds"] = time.perf_counter() - start
    result["peak_cuda_allocated_mb"] = torch.cuda.max_memory_allocated(study.device) / 2 ** 20 if study.device.type == "cuda" else None
    result["peak_cuda_reserved_mb"] = torch.cuda.max_memory_reserved(study.device) / 2 ** 20 if study.device.type == "cuda" else None
    release(study.device)
    return result


def run_audit(study, numerical, refinements, wavelengths, *, output: Path,
              repeats, steps, atol, rtol, energy_tolerance, integration,
              check_gradients=True):
    settings = {"physical_signature": study.signature, "numerical": asdict(numerical),
                "refinements": {k: asdict(v) for k, v in refinements.items()},
                "wavelengths": wavelengths, "steps_nm": steps, "atol_per_nm": atol,
                "rtol": rtol, "energy_tolerance": energy_tolerance,
                "repeats": repeats, "integration": integration,
                "check_gradients": check_gradients,
                "checkpoint_layers": study.checkpoint_layers,
                "audit_source_sha256": content_hash(Path(__file__))}
    identity = signature_for(settings)
    progress_path = output / "audit_checkpoint.json"
    progress = {"signature": identity, "tasks": {}}
    if progress_path.exists():
        saved = json.loads(progress_path.read_text(encoding="utf-8"))
        if saved["signature"] == identity:
            progress = saved
    tasks = progress["tasks"]

    def cached(name, calculate):
        if name not in tasks:
            print(f"{study.lattice}/{study.method}: audit {name}", flush=True)
            tasks[name] = calculate()
            write_json(progress_path, progress)
        return tasks[name]

    def summarize(samples):
        memory = [r["peak_cuda_allocated_mb"] for r in samples if r["peak_cuda_allocated_mb"] is not None]
        return {"median_seconds": statistics.median(r["seconds"] for r in samples),
                "samples_seconds": [r["seconds"] for r in samples],
                "max_peak_cuda_allocated_mb": max(memory) if memory else None}

    # Warmups are separate from the repeated measurements, and are checkpointed.
    cached("forward_warmup", lambda: timed_objective(study, numerical, wavelengths, gradient=False))
    forward = [cached(f"forward_{i}", lambda: timed_objective(study, numerical, wavelengths, gradient=False))
               for i in range(repeats)]
    result = {"settings": settings, "objective_definition": "trapezoidal wavelength average of R over audit wavelengths",
              "forward": summarize(forward), "gradient": None, "gradient_passed": False,
              "coverage": "height and base radius only; not a proof for all per-slice design variables"}
    forward_physical = all(physical(row, energy_tolerance) for s in forward for row in s["rows"])
    result["forward_physical"] = forward_physical
    if not check_gradients:
        result["gradient_audit_skipped"] = True
        write_json(output / "audit.json", result)
        return result
    try:
        baseline = cached("gradient_warmup", lambda: timed_objective(study, numerical, wavelengths, gradient=True))
        repeated = [cached(f"gradient_{i}", lambda: timed_objective(study, numerical, wavelengths, gradient=True))
                    for i in range(repeats)]
        fd_checks, refinement_checks = [], []
        for parameter in PARAMETERS:
            reference = getattr(study.geometry, parameter)
            fds = []
            for scale in (1., .5):
                step = steps[parameter] * scale
                plus_geometry = replace(study.geometry, **{parameter: reference + step})
                minus_geometry = replace(study.geometry, **{parameter: reference - step})
                plus_geometry.validate()
                minus_geometry.validate()
                plus = cached(f"fd_{parameter}_{scale}_plus", lambda: objective(study, numerical, wavelengths, plus_geometry))
                minus = cached(f"fd_{parameter}_{scale}_minus", lambda: objective(study, numerical, wavelengths, minus_geometry))
                fds.append((plus["objective"] - minus["objective"]) / (2 * step))
            ad = baseline["gradient_per_nm"][parameter]
            fd_checks.append({"parameter": parameter, "autograd_per_nm": ad,
                              "fd_h_per_nm": fds[0], "fd_half_h_per_nm": fds[1],
                              "passed": compatible(ad, fds[1], atol, rtol)
                              and compatible(fds[0], fds[1], atol, rtol)})
        for axis, refined in refinements.items():
            refined_result = cached(f"refined_gradient_{axis}", lambda: timed_objective(study, refined, wavelengths, gradient=True))
            per_parameter = {name: {"baseline": baseline["gradient_per_nm"][name],
                "refined": refined_result["gradient_per_nm"][name],
                "passed": compatible(baseline["gradient_per_nm"][name],
                                     refined_result["gradient_per_nm"][name], atol, rtol)} for name in PARAMETERS}
            refinement_checks.append({"axis": axis, "numerical": asdict(refined),
                                      "per_parameter": per_parameter,
                                      "passed": all(v["passed"] for v in per_parameter.values())})
        # All auxiliary FD/refinement solves must also remain lossless/physical.
        all_physical = all(physical(row, energy_tolerance) for t in tasks.values() for row in t.get("rows", []))
        result.update(gradient=summarize(repeated), objective=baseline["objective"],
                      gradient_per_nm=baseline["gradient_per_nm"], fd_checks=fd_checks,
                      refinement_checks=refinement_checks,
                      auxiliary_physical=all_physical,
                      gradient_passed=forward_physical and all_physical
                      and all(c["passed"] for c in fd_checks + refinement_checks))
    except Exception as error:
        result["gradient_error"] = f"{type(error).__name__}: {error}"
        error.__traceback__ = None
        release(study.device)
    write_json(output / "audit.json", result)
    return result
