"""Compare dense and TE/TM-separated RCWA in fresh processes, without checkpoints."""
from __future__ import annotations

import argparse
import copy
import json
import statistics
import subprocess
import sys
from pathlib import Path

from common import HERE, load_config, read_json, write_json


def peak_cpu_working_set():
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)]+[
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = (
            wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),
                                          ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return counters.PeakWorkingSetSize
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if sys.platform == "darwin" else peak*1024)
    except ImportError:
        return None


def retained_tensor_storage(sim):
    import torch
    seen_objects, seen_storage = set(), set()
    def visit(value):
        if id(value) in seen_objects:
            return 0
        seen_objects.add(id(value))
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            key = (str(value.device), storage.data_ptr(), storage.nbytes())
            if key in seen_storage:
                return 0
            seen_storage.add(key)
            return storage.nbytes()
        if isinstance(value, dict):
            return sum(visit(item) for item in value.values())
        if isinstance(value, (tuple, list)):
            return sum(visit(item) for item in value)
        return 0
    return visit(vars(sim))


def worker(args):
    import torch
    import solver
    torch.set_num_threads(args.threads)
    config, model = load_config(args.config)
    config["solver"]["fourier_coefficients"] = "analytic"
    config["solver"]["polarization_separated"] = args.variant == "separated"
    # Warm up small eigensolves/FFTs before the timed calculation.
    solver.simulate(config, {"order": 2, "slices": 2, "grid": 32},
                    args.wave, model(args.wave), args.device)
    original_factory = solver._simulation
    captured = []
    def capture(*factory_args, **factory_kwargs):
        sim = original_factory(*factory_args, **factory_kwargs)
        captured.append(sim)
        return sim
    solver._simulation = capture
    numbers = {"order": args.order, "slices": args.slices,
               "grid": max(32, 4*args.order+4)}
    row = solver.simulate(config, numbers, args.wave, model(args.wave), args.device)
    row.update(variant=args.variant,
               retained_tensor_storage_bytes=retained_tensor_storage(captured[0]),
               peak_cpu_working_set_bytes=peak_cpu_working_set(),
               cpu_threads=torch.get_num_threads())
    write_json(args.worker_output, row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE/"config.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--order", type=int, default=80)
    parser.add_argument("--slices", type=int, default=420)
    parser.add_argument("--wavelengths", default="550")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output-dir", type=Path,
                        default=HERE/"results"/"polarization_benchmark")
    parser.add_argument("--variant", choices=("dense", "separated"), help=argparse.SUPPRESS)
    parser.add_argument("--wave", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.order, args.slices, args.repeats, args.threads) < 1:
        parser.error("order, slices, repeats, and threads must be positive.")
    if args.variant is not None:
        worker(args)
        return 0
    waves = [float(part) for part in args.wavelengths.split(",")]
    config, model = load_config(args.config)
    for wave in waves:
        model(wave)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for repeat in range(args.repeats):
        variants = ("dense", "separated") if repeat % 2 == 0 else ("separated", "dense")
        for wave in waves:
            for variant in variants:
                destination = args.output_dir/f"{variant}_{wave:g}_{repeat+1}.json"
                command = [sys.executable, str(Path(__file__).resolve()),
                           "--config", str(args.config.resolve()), "--device", args.device,
                           "--order", str(args.order), "--slices", str(args.slices),
                           "--threads", str(args.threads), "--variant", variant,
                           "--wave", str(wave), "--worker-output", str(destination)]
                print(f"{variant}: M={args.order}, Nz={args.slices}, {wave:g} nm, repeat={repeat+1}",
                      flush=True)
                subprocess.run(command, check=True, creationflags=(
                    subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0))
                row = read_json(destination)
                rows.append(row)
                print(f"  {row['runtime_seconds']:.3f} s; retained tensors "
                      f"{row['retained_tensor_storage_bytes']/2**20:.2f} MiB", flush=True)
    summary = {"device": args.device, "cpu_threads": args.threads,
               "order": args.order, "slices": args.slices, "repeats": args.repeats,
               "coefficient_method": "analytic", "comparisons": {},
               "scope": "Fresh-process runtime medians; CPU peak includes interpreter/libraries; retained tensor storage is measured after solving, not peak allocation. Both polarizations are included. This is not an M/Nz convergence study."}
    maximum_error = 0.0
    for wave in waves:
        variants = {variant: [row for row in rows if row["variant"] == variant
                              and row["wavelength_nm"] == wave]
                    for variant in ("dense", "separated")}
        times = {variant: statistics.median(row["runtime_seconds"] for row in group)
                 for variant, group in variants.items()}
        reference = variants["dense"][0]["polarizations"]
        error = max(abs(row["polarizations"][pol][metric]-reference[pol][metric])
                    for row in variants["separated"] for pol in ("TE", "TM")
                    for metric in ("reflectance", "power_into_substrate", "relief_absorptance"))
        maximum_error = max(maximum_error, error)
        comparison = {"median_runtime_seconds": times,
                      "speedup": times["dense"]/times["separated"],
                      "maximum_absolute_observable_difference": error,
                      "retained_tensor_storage_bytes": {
                          variant: max(row["retained_tensor_storage_bytes"] for row in group)
                          for variant, group in variants.items()},
                      "peak_cpu_working_set_bytes": {
                          variant: max((row["peak_cpu_working_set_bytes"] or 0) for row in group)
                          for variant, group in variants.items()}}
        if args.device.startswith("cuda"):
            comparison["peak_cuda_allocated_bytes"] = {
                variant: max(row["peak_cuda_allocated_bytes"] for row in group)
                for variant, group in variants.items()}
        summary["comparisons"][f"{wave:g}"] = comparison
    summary.update(passed=maximum_error <= 1e-8,
                   parity_absolute_tolerance=1e-8,
                   maximum_absolute_observable_difference=maximum_error)
    write_json(args.output_dir/"report.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
