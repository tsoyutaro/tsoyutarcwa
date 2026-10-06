"""Increase saved Au moth-eye Fourier orders until a CUDA allocation fails.

Each new order runs in a separate process. A caught CUDA OOM is a normal
stopping condition: completed cases, CSV and figures are saved, then exit 0.
Other failures remain errors. The default seed is the 700 nm/Nz140/grid576
memory-safe order sweep. Geometry, ASR equations and complex128 are inherited.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for directory in (ROOT, HERE):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from studies.gold_motheye3 import converge
from studies.gold_motheye3.show_results import read_checkpoint

DEFAULT_SEED = HERE / "results/order_700_Nz140_grid576_memory_safe/checkpoint.json"
DEFAULT_OUTPUT = HERE / "results/order_700_Nz140_grid576_until_oom"
STORAGE = "D6-fixed-geometry-flux-only-release-v1"
DRIVER = "studies/gold_motheye3/run_until_oom.py"


def validate(saved):
    """Require unchanged equations/data and the already verified storage policy."""
    plan = saved["plan"]
    solver = plan["solver"]
    if (plan["axis"] != "order" or plan["wavelengths_nm"] != [700.]
            or plan["geometry"] != converge.GEOMETRY
            or plan["material"]["model"] != "measured_csv"
            or solver.get("layer_storage") != STORAGE
            or solver["dtype"] != "complex128"
            or solver["symmetry_reduction"] != "d6-source"
            or solver["smatrix_size"] != "half"
            or solver["cascade"] != "redheffer"):
        raise ValueError("Requires the same 700 nm complex128 D6 memory-safe Au order study.")
    parity = saved.get("storage_parity", saved.get("storage_parity_from_seed", {}))
    if not parity.get("passed") or len(saved["cases"]) < 2:
        raise ValueError("Requires passed storage parity and at least two completed seed orders.")
    current = converge._source_hashes()
    current["studies/gold_motheye3/run_memory_safe.py"] = converge._digest(HERE / "run_memory_safe.py")
    if DRIVER in plan["source_sha256"]:
        current[DRIVER] = converge._digest(Path(__file__))
    changed = sorted(name for name in set(current) | set(plan["source_sha256"])
                     if current.get(name) != plan["source_sha256"].get(name))
    if changed:
        raise ValueError("Sources differ from the saved calculation: " + ", ".join(changed))
    if converge._digest(Path(plan["material"]["path"])) != plan["material"]["sha256"]:
        raise ValueError("The Au CSV differs from the saved calculation.")
    for row in saved["cases"].values():
        if (row.get("passivity_warning") or row["symmetry_reduction"] != "D6-E1-source-row"
                or any(not -1e-5 <= float(row[name]) <= 1+1e-5 for name in converge.METRICS)
                or abs(float(row["transmittance_far"])) > 1e-10):
            raise ValueError("Seed contains a nonphysical or unexpected solver result.")


def next_order(saved):
    plan = saved["plan"]
    missing = [m for m in plan["values"] if converge._key(m, 700.) not in saved["cases"]]
    if missing:
        return min(missing)
    probe = plan["capacity_probe"]
    return max(max(row["order"] for row in saved["cases"].values()) + probe["step"],
               probe["start_order"])


def prepare(args, output):
    checkpoint = output / "checkpoint.json"
    if checkpoint.exists():
        saved = read_checkpoint(checkpoint)
        validate(saved)
        probe = saved["plan"].get("capacity_probe")
        if not probe or probe["worker_policy"] != "process-per-order-v1":
            raise ValueError("This output is not a run_until_oom checkpoint. Use another --output-dir.")
        if args.step != probe["step"] or (args.start_order is not None
                                         and args.start_order != probe["start_order"]):
            raise ValueError("Start order/step differs from the saved probe. Use another --output-dir.")
        if args.device != saved["plan"]["solver"]["requested_device"]:
            raise ValueError("Device differs from the saved probe. Use another --output-dir.")
        return saved
    seed_path = args.seed_checkpoint.resolve()
    if seed_path.is_relative_to(output):
        raise ValueError("Use an output directory separate from the seed checkpoint.")
    seed = read_checkpoint(seed_path)
    validate(seed)
    highest = max(row["order"] for row in seed["cases"].values())
    start = args.start_order if args.start_order is not None else highest + args.step
    if start <= highest:
        raise ValueError(f"Start order must exceed the highest completed seed order M={highest}.")
    saved = copy.deepcopy(seed)
    saved.pop("storage_parity", None)
    saved["storage_parity_from_seed"] = copy.deepcopy(
        seed.get("storage_parity", seed.get("storage_parity_from_seed")))
    plan = saved["plan"]
    plan["values"] = sorted(row["order"] for row in saved["cases"].values())
    plan["seed_signature"] = seed["signature"]
    plan["solver"]["requested_device"] = args.device
    plan["source_sha256"][DRIVER] = converge._digest(Path(__file__))
    plan["capacity_probe"] = {"start_order": start, "step": args.step,
                              "worker_policy": "process-per-order-v1"}
    for key, row in saved["cases"].items():
        row["reused_from"] = {"signature": seed["signature"], "key": key,
                              "runtime_is_original_measurement": True}
    saved["signature"] = converge._signature(plan)
    saved["capacity_probe"] = {"status": "ready", "seed_checkpoint": str(seed_path),
                                "attempts": []}
    output.mkdir(parents=True, exist_ok=True)
    converge._write_json(output / "seed_checkpoint.json", seed)
    return saved


def worker(request_path, result_path):
    """One process owns every CUDA allocation for one order, including on OOM."""
    request = json.loads(request_path.read_text(encoding="utf-8"))
    plan, order = request["plan"], request["order"]
    started = time.perf_counter()
    torch, device, environment = None, None, None
    try:
        import torch
        from studies.gold_motheye3.run_memory_safe import measure

        device = torch.device(plan["solver"]["requested_device"])
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable.")
        environment = {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                       "device_type": device.type,
                       "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
        if request.get("runtime_environment") not in (None, environment):
            raise RuntimeError("PyTorch/CUDA/GPU differs from the seed calculation.")
        result = measure(700., converge._numerical(plan, order), plan, device, torch)
        outcome = {"status": "completed", "order": order, "result": result}
    except Exception as error:
        cuda_oom = (torch is not None and device is not None and device.type == "cuda"
                    and isinstance(error, torch.OutOfMemoryError))
        outcome = {"status": "cuda_oom" if cuda_oom else "error", "order": order,
                   "error_type": type(error).__name__, "error": str(error)}
        if not cuda_oom:
            outcome["traceback"] = traceback.format_exc()
            print(outcome["traceback"], file=sys.stderr, flush=True)
        if cuda_oom:
            try:
                free, total = torch.cuda.mem_get_info(device)
                outcome.update(peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
                               peak_cuda_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
                               cuda_free_bytes=int(free), cuda_total_bytes=int(total))
            except Exception:
                pass  # The process exit still releases its CUDA context.
    outcome["runtime_environment"] = environment
    outcome["process_wall_seconds"] = time.perf_counter() - started
    converge._write_json(result_path, outcome)
    return 0 if outcome["status"] in ("completed", "cuda_oom") else 1


def launch(output, saved, order):
    workers = output / "workers"
    workers.mkdir(exist_ok=True)
    request_path, result_path = workers / f"M{order}_request.json", workers / f"M{order}_result.json"
    if result_path.exists():
        result_path.unlink()  # A killed worker must not be mistaken for an old success.
    converge._write_json(request_path, {"plan": saved["plan"], "order": order,
                                       "runtime_environment": saved.get("runtime_environment")})
    started = time.perf_counter()
    with (workers / f"M{order}.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                    "--worker-request", str(request_path),
                                    "--worker-result", str(result_path)],
                                   stdout=log, stderr=subprocess.STDOUT)
        try:
            process.wait()
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if not result_path.exists():
        return {"status": "error", "order": order, "worker_returncode": process.returncode,
                "error": "Worker exited without a result; see its log. This is not a confirmed CUDA OOM.",
                "process_wall_seconds": time.perf_counter() - started}
    outcome = json.loads(result_path.read_text(encoding="utf-8"))
    if (outcome.get("order") != order
            or (process.returncode != 0 and outcome.get("status") != "error")):
        return {"status": "error", "order": order, "worker_returncode": process.returncode,
                "error": "Invalid worker result/exit status; see its log."}
    outcome["process_wall_seconds"] = time.perf_counter() - started
    return outcome


def record(saved, outcome):
    probe = saved["capacity_probe"]
    probe["attempts"].append({k: v for k, v in outcome.items() if k != "result"})
    if outcome["status"] == "completed":
        order = outcome["order"]
        row = {**outcome["result"], "axis": "order", "value": order,
               "process_wall_seconds": outcome["process_wall_seconds"]}
        # The worker uses the existing measure() checks; also validate transfer numerics.
        if any(row[name] != number for name, number in converge._numerical(saved["plan"], order).items()):
            raise ValueError("Worker returned different numerical settings.")
        saved["cases"][converge._key(order, 700.)] = row
        saved["runtime_environment"] = outcome["runtime_environment"]
        probe["status"] = "running"
        probe.pop("failed_order", None)
    else:
        probe["status"] = "stopped_cuda_oom" if outcome["status"] == "cuda_oom" else "worker_error"
        probe["failed_order"] = outcome["order"]


def persist(output, saved):
    converge._write_json(output / "checkpoint.json", saved)
    converge._write_json(output / "plan.json", saved["plan"])
    probe = saved["capacity_probe"]
    completed_plan = copy.deepcopy(saved["plan"])
    completed_plan["values"] = sorted(row["order"] for row in saved["cases"].values())
    convergence = converge.assess(completed_plan, saved["cases"])
    report = {"signature": saved["signature"], "plan": saved["plan"],
              "runtime_environment": saved.get("runtime_environment"),
              "status": probe["status"], "highest_completed_order": max(completed_plan["values"]),
              "first_oom_order": probe.get("failed_order") if probe["status"] == "stopped_cuda_oom" else None,
              "capacity_probe": probe, "convergence_on_completed_orders": convergence,
              "scope": "Memory capacity for this geometry/device/process policy; not an accuracy guarantee."}
    converge._write_json(output / "report.json", report)
    columns = list(converge.CASE_COLUMNS) + ["wall_seconds", "process_wall_seconds",
              "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes", "peak_cuda_allocated_GiB"]
    with (output / "cases.csv.tmp").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(saved["cases"].values(), key=lambda r: r["order"]):
            peak = row.get("peak_cuda_allocated_bytes")
            writer.writerow({**row, "peak_cuda_allocated_GiB": None if peak is None else peak / 2**30})
    (output / "cases.csv.tmp").replace(output / "cases.csv")
    from studies.gold_motheye3.plot import render as render_all
    from plot_order_results import collect, render

    render_all(saved["plan"], saved["cases"], {**convergence, "status": probe["status"]},
               output / "convergence.svg")
    orders, points, changes = collect(saved, 700.)
    fixed = saved["plan"]["fixed_numerics"]
    subtitle = (f"700 nm; Nz={fixed['slices']}; grid={fixed['grid']}; "
                f"{probe['status']}; highest completed M={report['highest_completed_order']}")
    if probe.get("failed_order"):
        subtitle += f"; failed M={probe['failed_order']} (no R value)"
    render(output / "reflectance_vs_order_700nm.svg", "Au moth-eye: reflectance vs Fourier order",
           subtitle, orders, points, "Reflectance R (%)")
    if changes:
        render(output / "reflectance_order_changes_700nm.svg", "Adjacent-order reflectance changes",
               subtitle, orders, changes, "Absolute change in R (percentage points)",
               threshold=100 * saved["plan"]["tolerance"], zero_floor=True)
    memory = [(r["order"], r["peak_cuda_allocated_bytes"] / 2**30)
              for r in sorted(saved["cases"].values(), key=lambda r: r["order"])
              if r.get("peak_cuda_allocated_bytes") is not None]
    if memory:
        render(output / "peak_memory_vs_order.svg", "CUDA tensor peak memory vs Fourier order",
               subtitle, orders, memory, "Peak allocated CUDA tensors (GiB)", zero_floor=True)
    return report


def summarize(output, saved, report):
    print("\n M          R(%)       peak tensors(GiB)      runtime(s)")
    for row in sorted(saved["cases"].values(), key=lambda r: r["order"]):
        peak = row.get("peak_cuda_allocated_bytes")
        memory = "-" if peak is None else f"{peak / 2**30:.3f}"
        print(f"{row['order']:3d} {100 * row['reflectance']:13.7f} {memory:>23} {row['runtime_seconds']:15.2f}")
    print(f"status: {report['status']}")
    print(f"Highest completed tested order: M={report['highest_completed_order']}")
    if report["first_oom_order"] is not None:
        print(f"CUDA OOM at M={report['first_oom_order']}; completed results retained; normal exit.")
    print(f"Convergence of completed orders: {report['convergence_on_completed_orders']['status']}")
    print(f"report: {output / 'report.json'}")
    print(f"figure: {output / 'reflectance_vs_order_700nm.svg'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed-checkpoint", type=Path, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-order", type=int, help="Default: highest completed seed order + step.")
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--max-order", type=int, help="Optional stopping cap; default: continue until CUDA OOM.")
    parser.add_argument("--retry-oom", action="store_true", help="Retry the failed order after freeing GPU memory.")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--worker-request", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_request:
        if args.worker_result is None:
            parser.error("Missing worker result path.")
        return worker(args.worker_request, args.worker_result)
    if args.step <= 0 or any(v is not None and v <= 0 for v in (args.start_order, args.max_order)):
        parser.error("Orders and step must be positive.")
    if args.prepare_only and args.report_only:
        parser.error("Choose --prepare-only or --report-only.")
    if args.device == "cpu" and args.max_order is None and not (args.report_only or args.prepare_only):
        parser.error("CPU verification requires --max-order; this tool probes CUDA OOM.")
    output = args.output_dir.resolve()
    try:
        if args.report_only:
            saved = read_checkpoint(output / "checkpoint.json")
            if "capacity_probe" not in saved:
                raise ValueError("Use the run_until_oom output checkpoint.")
        else:
            saved = prepare(args, output)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))
    report = persist(output, saved)
    fixed = saved["plan"]["fixed_numerics"]
    print(f"700 nm; Nz={fixed['slices']}; grid={fixed['grid']}; complex128; released D6 storage.", flush=True)
    print(f"Each new order gets a fresh process; step={args.step}; next M={next_order(saved)}.", flush=True)
    if args.report_only:
        summarize(output, saved, report)
        return 0
    if args.prepare_only:
        print(f"Prepared; no RCWA solve: {output / 'plan.json'}")
        return 0
    if report["status"] == "stopped_cuda_oom" and not args.retry_oom:
        summarize(output, saved, report)
        print("Already stopped on OOM. Use --retry-oom to retry that order.")
        return 0
    try:
        while True:
            order = next_order(saved)
            if args.max_order is not None and order > args.max_order:
                saved["capacity_probe"]["status"] = "max_order_reached"
                report = persist(output, saved)
                break
            if order not in saved["plan"]["values"]:
                saved["plan"]["values"].append(order)
                saved["signature"] = converge._signature(saved["plan"])
            saved["capacity_probe"]["status"] = "running"
            persist(output, saved)
            print(f"solve M={order}, Nz={fixed['slices']}, grid={fixed['grid']}, wavelength=700 nm", flush=True)
            outcome = launch(output, saved, order)
            record(saved, outcome)
            report = persist(output, saved)
            if outcome["status"] != "completed":
                if outcome["status"] == "error":
                    print(f"Worker error: {outcome['error']}; see workers/M{order}.log", flush=True)
                break
            row = outcome["result"]
            peak = row.get("peak_cuda_allocated_bytes")
            memory = "n/a" if peak is None else f"{peak / 2**30:.3f} GiB"
            print(f"saved M={order}: R={100 * row['reflectance']:.7f}%; "
                  f"peak CUDA tensors={memory}; compute={row['wall_seconds']:.2f} s", flush=True)
    except KeyboardInterrupt:
        saved["capacity_probe"]["status"] = "interrupted"
        report = persist(output, saved)
        summarize(output, saved, report)
        return 130
    summarize(output, saved, report)
    return 1 if report["status"] == "worker_error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
