from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

from io_utils import content_hash, signature_for, write_csv, write_json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def wavelengths(text):
    if isinstance(text, list):
        values = [float(v) for v in text]
    elif ":" in text:
        start, stop, step = map(float, text.split(":"))
        if step <= 0 or stop < start or not all(math.isfinite(v) for v in (start, stop, step)):
            raise ValueError("Invalid wavelength range.")
        count = int(round((stop-start)/step))
        if count > 10000 or abs(start+count*step-stop) > 1e-8:
            raise ValueError("Wavelength range must have an integral number of intervals, <=10000.")
        values = [start+i*step for i in range(count+1)]
    else:
        values = [float(v) for v in text.split(",")]
    if not values or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("Wavelengths must be positive and finite.")
    return sorted(set(values))


def load_config(path, smoke=False):
    c = json.loads(path.read_text(encoding="utf-8-sig"))
    if smoke:
        c.update(optimization_order=1, optimization_grid=32, verification_orders=[1, 2], verification_grid=48,
                 optimization_wavelengths_nm="550", verification_wavelengths_nm="550,600",
                 audit_wavelengths_nm="550", audit_directions=1, trials=2, adam_steps=2,
                 patience=2, pruner_startup_trials=10, gradient_audit=True)
    if c["layers"] < 2 or c["index"] <= 0 or c["height_nm"] <= 0:
        raise ValueError("Invalid layer/material/height settings.")
    if not 0 < c["radius_min_nm"] < c["baseline_tip_nm"] < c["baseline_base_nm"] < c["radius_max_nm"] < c["period_nm"]/2:
        raise ValueError("Require 0 < radius_min < baseline_tip < baseline_base < radius_max < period/2.")
    if not 0 < (c["layers"]+1)*c["softmax_gap_floor"] < 1:
        raise ValueError("Invalid softmax gap floor.")
    if not c["lattices"] or any(v not in ("square", "hexagonal") for v in c["lattices"]):
        raise ValueError("Lattices must be square and/or hexagonal.")
    if c["objective_weighting"] not in ("arithmetic", "trapezoidal"):
        raise ValueError("Unknown objective weighting.")
    for name in ("optimization_wavelengths_nm", "verification_wavelengths_nm", "audit_wavelengths_nm"):
        c[name] = wavelengths(c[name])
    for name in ("optimization_order", "optimization_grid", "verification_grid", "trials", "adam_steps", "patience",
                 "scheduler_patience", "cpu_threads", "audit_directions", "pruner_interval_steps"):
        if isinstance(c[name], bool) or not isinstance(c[name], int) or c[name] <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if any(not isinstance(m, int) or m <= 0 for m in c["verification_orders"]):
        raise ValueError("Verification orders must be positive integers.")
    for name in ("learning_rate_min", "learning_rate_max", "audit_fd_step", "energy_tolerance",
                 "gradient_atol", "gradient_rtol", "verification_tolerance", "logit_limit"):
        if not math.isfinite(c[name]) or c[name] <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    if c["learning_rate_min"] > c["learning_rate_max"] or not 0 < c["scheduler_factor"] < 1:
        raise ValueError("Invalid learning rate or scheduler factor.")
    if not 0 <= c["gradient_cosine_min"] <= 1 or c["min_improvement"] < 0:
        raise ValueError("Invalid gradient/early-stop tolerance.")
    if c["optimization_grid"] < max(32, 4*c["optimization_order"]+4):
        raise ValueError("Optimization grid undersamples the retained order.")
    if c["verification_grid"] < max(32, 4*max(c["verification_orders"])+4):
        raise ValueError("Verification grid undersamples the retained order.")
    from .materials import prepare_material
    c = prepare_material(c, ROOT)
    if not c["radius_min_nm"] < c["init_top_min_nm"] < c["init_top_max_nm"] < c["radius_max_nm"]:
        raise ValueError("Initial-radius search interval is invalid.")
    return c


def sources():
    files = list(HERE.glob("*.py")) + list((ROOT/"rcwa_ext").glob("*.py"))
    files += [ROOT/"run.py", ROOT/"rcwa_solver_auto.py", ROOT/"io_utils.py", ROOT/"runtime_policy.py"]
    files += [ROOT/"d6_nvm.py"]
    files += list((ROOT/"data").glob("*.csv"))
    result = {p.relative_to(ROOT).as_posix(): content_hash(p) for p in files}
    spec = importlib.util.find_spec("torcwa")
    if spec and spec.origin:
        result.update({"installed_torcwa/"+p.name: content_hash(p) for p in Path(spec.origin).parent.glob("*.py")})
    return result


def conditions(c, orders=None):
    # Keep a same-Q order comparison; then vary Q at fixed verification order.
    items = [(c["optimization_order"], c["optimization_grid"])]
    for order in orders or c["verification_orders"]:
        if c["optimization_grid"] >= 4*order+4:
            items.append((order, c["optimization_grid"]))
        items.append((order, c["verification_grid"]))
    return sorted(set(items))


def torch_save(path, value):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def device_setup(requested, threads):
    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if requested == "auto" else torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available() or (device.index or 0) >= torch.cuda.device_count():
            raise RuntimeError("Requested CUDA device is unavailable; no CPU fallback.")
        device = torch.device("cuda", device.index if device.index is not None else torch.cuda.current_device())
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats(device)
    torch.set_num_threads(threads)
    return device


def runtime_environment(device):
    import torch
    import platform
    import torcwa
    import importlib.metadata
    return {"python": platform.python_version(), "torch": str(torch.__version__), "torcwa": importlib.metadata.version("torcwa"),
            "cuda": torch.version.cuda, "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "gpu_total_bytes": int(torch.cuda.get_device_properties(device).total_memory) if device.type == "cuda" else None,
            "optuna": importlib.metadata.version("optuna")}
