"""Compare NV powers and peak GPU allocation in separate processes."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from Optimization.dielectric_motheye_n25.common import device_setup, load_config, write_json
from Optimization.dielectric_motheye_n25.geometry import baseline_logits, radii_tensor


def worker(request, result_path):
    import torch
    from Optimization.dielectric_motheye_n25.physics import solve_train, solve_verify
    try:
        c = request["config"]
        device = device_setup(request["device"], c["cpu_threads"])
        radii = radii_tensor(torch.tensor(baseline_logits(c), dtype=torch.float64, device=device), c)
        if request["path"] == "fast":
            with torch.no_grad():
                value, row = solve_train(radii, c, request["lattice"], request["wavelength_nm"], request["order"], request["grid"], device)
            del value
        else:
            row = solve_verify(radii, c, request["lattice"], request["wavelength_nm"], request["order"], request["grid"], device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        result = {"status": "completed", "powers": row,
                  "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
                  "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None}
    except Exception as error:
        result = {"status": "oom" if isinstance(error, (torch.OutOfMemoryError, MemoryError)) or "cuda out of memory" in str(error).lower() else "error",
                  "error": str(error), "traceback": traceback.format_exc()}
    write_json(result_path, result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=HERE/"config.json")
    p.add_argument("--device", default="cuda")
    p.add_argument("--orders", default="8,16")
    p.add_argument("--grid", type=int, default=512)
    p.add_argument("--wavelength-nm", type=float, default=550.)
    p.add_argument("--output", type=Path, default=HERE/"results/memory_parity.json")
    p.add_argument("--worker-request", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.worker_request:
        worker(json.loads(args.worker_request.read_text()), args.worker_result)
        return 0
    c = load_config(args.config)
    orders = sorted(set(int(v) for v in args.orders.split(",")))
    if min(orders) <= 0 or args.grid < max(32,4*max(orders)+4) or args.wavelength_nm <= 0:
        p.error("Invalid order/grid/wavelength.")
    rows = []
    for lattice in c["lattices"]:
        for order in orders:
            row = {"lattice": lattice, "order": order, "grid": args.grid, "wavelength_nm": args.wavelength_nm}
            for method in ("fast", "streamed"):
                directory = args.output.parent/(args.output.stem+"_workers")/f"{lattice}_M{order}_{method}"
                req, res = directory/"request.json", directory/"result.json"
                request = {"config": c, "device": args.device, "path": method, "lattice": lattice,
                           "order": order, "grid": args.grid, "wavelength_nm": args.wavelength_nm}
                write_json(req, request)
                res.unlink(missing_ok=True)
                with (directory/"worker.log").open("w", encoding="utf-8") as log:
                    process = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                        "--worker-request", str(req), "--worker-result", str(res)], stdout=log, stderr=subprocess.STDOUT)
                row[method] = json.loads(res.read_text()) if res.exists() and process.returncode == 0 else {"status": "error", "error": "Worker exited without a valid result; not confirmed OOM."}
            if all(row[k]["status"] == "completed" for k in ("fast", "streamed")):
                row["max_abs_power_difference"] = max(abs(row["fast"]["powers"][k]-row["streamed"]["powers"][k]) for k in ("reflectance", "transmittance"))
                row["passed"] = row["max_abs_power_difference"] <= 1e-9
            else:
                row["passed"] = False
            rows.append(row)
            write_json(args.output, {"cases": rows, "complete": len(rows) == len(c["lattices"])*len(orders),
                                    "passed": all(v["passed"] for v in rows),
                                    "scope": "Forward-power parity and per-process PyTorch CUDA peak memory at the requested wavelengths; no gradient convergence claim."})
            print(f"{lattice} M={order}: parity passed={row['passed']}", flush=True)
    return 0 if all(row["passed"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
