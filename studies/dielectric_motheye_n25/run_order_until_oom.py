"""Sweep Fourier order at fixed Nz/Q; each lattice/method/wavelength stops on OOM.

No optical equations are changed. Each order owns a separate process so an OOM
cannot leave a CUDA context or live layer arrays in the next calculation.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studies.shared.convergence_io import content_hash, signature_for, write_csv, write_json
from studies.dielectric_motheye_n25.compare import source_hashes, write_geometry
from studies.dielectric_motheye_n25.model import Geometry, Numerical, POWERS, physical

DEFAULT_OUTPUT = HERE / "results/order_Nz15_Q512_until_oom"
POLICY = "n25-fixed-layer-independent-series-process-per-order-v1"


def series_key(lattice, method, wavelength):
    return f"{lattice}/{method}/{wavelength:g}nm"


def is_oom(error, torch, device):
    """Only positively identified allocation failures count as OOM."""
    if isinstance(error, MemoryError):
        return "cpu_oom"
    oom_type = getattr(torch, "OutOfMemoryError", ()) if torch is not None else ()
    if oom_type and isinstance(error, oom_type):
        return "cuda_oom" if device is not None and device.type == "cuda" else "cpu_oom"
    message = str(error).lower()
    if device is not None and device.type == "cuda" and "cuda out of memory" in message:
        return "cuda_oom"
    if "defaultcpuallocator" in message and any(
            phrase in message for phrase in ("not enough memory", "can't allocate memory", "out of memory")):
        return "cpu_oom"
    return None


def runtime(torch, device, threads):
    return {"python": sys.version.split()[0], "torch": str(torch.__version__),
            "cuda": torch.version.cuda, "device": str(device), "cpu_threads": threads,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "gpu_total_bytes": int(torch.cuda.get_device_properties(device).total_memory)
            if device.type == "cuda" else None}


def worker(request_path, result_path):
    request = json.loads(request_path.read_text(encoding="utf-8"))
    plan = request["plan"]
    started = time.perf_counter()
    torch, device, environment = None, None, None
    try:
        import torch
        import torcwa  # Fail preflight before creating twelve apparent OOM series.
        requested = plan["requested_device"]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if requested == "auto" else torch.device(requested)
        if device.type == "cuda":
            if not torch.cuda.is_available() or (device.index or 0) >= torch.cuda.device_count():
                raise RuntimeError("Requested CUDA device is unavailable; no CPU fallback.")
            if device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.cuda.reset_peak_memory_stats(device)
        torch.set_num_threads(plan["cpu_threads"])
        environment = runtime(torch, device, plan["cpu_threads"])
        if request.get("runtime_environment") not in (None, environment):
            raise RuntimeError("Python/PyTorch/CUDA/GPU differs from the saved run; use a new output root.")
        if request.get("preflight"):
            outcome = {"status": "ready"}
        else:
            from studies.dielectric_motheye_n25.model import solve
            numerical = Numerical(request["order"], plan["slices"], plan["grid"])
            # Do not wrap ASR in no_grad/inference_mode: spatial Jacobians need
            # autograd even though no design variables require gradients here.
            row, tensor = solve(request["wavelength_nm"], numerical, Geometry(**plan["geometry"]),
                                method=request["method"], lattice=request["lattice"], device=device,
                                use_symmetry=plan["use_symmetry"], square_asr_profile="local-radial")
            del tensor
            row["physical"] = physical(row, plan["energy_tolerance"])
            row["peak_cuda_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
            row["peak_cuda_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None
            outcome = {"status": "completed", "result": row}
    except Exception as error:
        status = is_oom(error, torch, device)
        outcome = {"status": status or "error", "error_type": type(error).__name__, "error": str(error)}
        if not status:
            outcome["traceback"] = traceback.format_exc()
            print(outcome["traceback"], file=sys.stderr, flush=True)
        if device is not None and device.type == "cuda" and torch is not None:
            try:
                outcome["peak_cuda_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
                outcome["peak_cuda_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
            except Exception:
                pass
    outcome.update(runtime_environment=environment, process_wall_seconds=time.perf_counter() - started,
                   order=request.get("order"), lattice=request.get("lattice"),
                   method=request.get("method"), wavelength_nm=request.get("wavelength_nm"))
    write_json(result_path, outcome)
    return 1 if outcome["status"] == "error" else 0


def launch(output, request, tag):
    folder = output / "workers" / tag
    folder.mkdir(parents=True, exist_ok=True)
    req, res = folder / "request.json", folder / "result.json"
    if res.exists():
        res.unlink()
    write_json(req, request)
    started = time.perf_counter()
    with (folder / "worker.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                    "--worker-request", str(req), "--worker-result", str(res)],
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
    if not res.exists():
        return {"status": "error", "worker_returncode": process.returncode,
                "error": "Worker exited without a result. This is not a confirmed OOM; see worker.log.",
                "process_wall_seconds": time.perf_counter() - started}
    outcome = json.loads(res.read_text(encoding="utf-8"))
    mismatch = any(outcome.get(k) != request.get(k) for k in ("order", "lattice", "method", "wavelength_nm"))
    if mismatch or (process.returncode != 0 and outcome.get("status") != "error"):
        return {"status": "error", "worker_returncode": process.returncode,
                "error": "Worker result identity or exit code is inconsistent; see worker.log."}
    if outcome["status"] == "completed":
        row = outcome["result"]
        if any(row.get(k) != request.get(k) for k in ("order", "lattice", "method", "wavelength_nm")):
            return {"status": "error", "error": "Optical row identity differs from its request."}
    return outcome


def assess(rows, tolerance, energy_tolerance):
    rows = sorted(rows, key=lambda r: r["order"])
    changes = []
    for coarse, fine in zip(rows, rows[1:]):
        delta = {name: abs(fine[name] - coarse[name]) for name in POWERS}
        maximum = max(delta.values())
        valid = physical(coarse, energy_tolerance) and physical(fine, energy_tolerance)
        changes.append({"coarse": coarse["order"], "fine": fine["order"], "per_metric": delta,
                        "max_abs_change": maximum, "physical": valid, "passed": valid and maximum <= tolerance})
    selected = None
    # At least two stable adjacent intervals plus agreement with every higher
    # point. This rejects temporary plateaus and accumulated small drift.
    for i in range(1, len(rows) - 1):
        if all(c["passed"] for c in changes[i - 1:]) and all(
                max(abs(other[name] - rows[i][name]) for name in POWERS) <= tolerance
                for other in rows[i + 1:]):
            selected = rows[i]["order"]
            break
    return {"order_converged_at_fixed_Nz_Q": selected is not None,
            "selected_order": selected, "adjacent_changes": changes,
            "highest_completed_order": rows[-1]["order"] if rows else None,
            "nonphysical_orders": [r["order"] for r in rows if not physical(r, energy_tolerance)]}


def prepare(args):
    geometry = Geometry()
    geometry.validate()
    hashes = source_hashes()
    hashes[Path(__file__).relative_to(ROOT).as_posix()] = content_hash(Path(__file__))
    physics = {"policy": POLICY, "geometry": asdict(geometry), "slices": 15,
               "grid": args.grid, "use_symmetry": not args.no_symmetry,
               "asr_profile": "local-radial", "asr_factorization_rules": True,
               "nv_lanczos_filter": False, "dtype": "complex128", "cascade": "redheffer",
               "smatrix_size": "half", "fields": "none", "incidence": "normal, x polarized",
               "substrate": "semi-infinite n=2.5", "start_order": args.start_order,
               "step": args.step, "source_sha256": hashes,
               "requested_device": args.device, "cpu_threads": args.cpu_threads}
    signature = signature_for(physics)
    output = args.output_root.resolve()
    path = output / "checkpoint.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("signature") != signature:
            raise ValueError("Saved geometry/numerics/device/source signature differs; use a new --output-root.")
    else:
        saved = {"signature": signature, "plan": physics, "runtime_environment": None, "series": {}}
    saved["plan"].update(tolerance=args.tolerance, energy_tolerance=args.energy_tolerance,
                         grid_max_order=(args.grid - 4) // 4,
                         limits="Only Fourier order at Nz=15 and fixed Q is assessed; no layer/grid or full-band convergence claim.")
    requested = []
    for lattice in args.lattices.split(","):
        for method in args.methods.split(","):
            for wavelength in (400., 550., 700.):
                key = series_key(lattice, method, wavelength)
                requested.append(key)
                saved["series"].setdefault(key, {"lattice": lattice, "method": method,
                    "wavelength_nm": wavelength, "status": "pending", "cases": {}, "attempts": []})
    write_json(output / "plan.json", saved["plan"])
    write_geometry(output, geometry, 15)
    return output, saved, requested


def save(output, saved):
    write_json(output / "checkpoint.json", saved)
    all_rows, reports = [], {}
    plan = saved["plan"]
    for key, group in saved["series"].items():
        rows = sorted(group["cases"].values(), key=lambda r: r["order"])
        enriched = []
        for index, row in enumerate(rows):
            item = dict(row)
            item["physical"] = physical(row, plan["energy_tolerance"])
            item["delta_R_abs"] = abs(row["reflectance"] - rows[index - 1]["reflectance"]) if index else None
            item["delta_T_abs"] = abs(row["transmittance"] - rows[index - 1]["transmittance"]) if index else None
            enriched.append(item)
        group_report = {k: v for k, v in group.items() if k != "cases"}
        group_report.update(assess(rows, plan["tolerance"], plan["energy_tolerance"]))
        group_report["first_oom_order"] = group.get("failed_order") if group["status"] in ("stopped_cuda_oom", "stopped_cpu_oom") else None
        group_report["completed_cases"] = len(rows)
        group_report["rows"] = enriched
        reports[key] = group_report
        directory = output / key
        write_csv(directory / "cases.csv", enriched)
        write_json(directory / "report.json", group_report)
        all_rows.extend(enriched)
    cross = []
    for lattice in ("square", "hexagonal"):
        for wavelength in (400., 550., 700.):
            nv = saved["series"].get(series_key(lattice, "nv", wavelength), {}).get("cases", {})
            asr = saved["series"].get(series_key(lattice, "asr", wavelength), {}).get("cases", {})
            common = sorted(set(nv) & set(asr), key=int)
            if common:
                order = common[-1]
                a, b = nv[order], asr[order]
                cross.append({"lattice": lattice, "wavelength_nm": wavelength, "highest_common_order": int(order),
                    "delta_R_abs": abs(a["reflectance"] - b["reflectance"]),
                    "delta_T_abs": abs(a["transmittance"] - b["transmittance"]),
                    "both_physical": physical(a, plan["energy_tolerance"]) and physical(b, plan["energy_tolerance"]),
                    "meaning": "Same-order snapshot, not an independent exact reference."})
    report = {"plan": plan, "runtime_environment": saved.get("runtime_environment"),
              "series": reports, "method_comparison_at_highest_common_order": cross}
    write_json(output / "report.json", report)
    write_csv(output / "all_cases.csv", all_rows)
    write_csv(output / "method_comparison.csv", cross)
    return report


def make_figures(output, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    colors = {"nv": "#1769aa", "asr": "#d56318"}
    for metric, label in (("reflectance", "R (%)"), ("transmittance", "T (%)"),
                          ("delta_R_abs", "Adjacent |delta R| (percentage points)"),
                          ("runtime_seconds", "Solve time (s)"), ("energy_residual", "|R+T-1|")):
        fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), squeeze=False)
        for i, lattice in enumerate(("square", "hexagonal")):
            for j, wavelength in enumerate((400., 550., 700.)):
                ax = axes[i, j]
                for method in ("nv", "asr"):
                    group = report["series"].get(series_key(lattice, method, wavelength))
                    if group is None:
                        continue
                    rows = group["rows"]
                    available = [r for r in rows if r.get(metric) is not None]
                    scale = 100 if metric in ("reflectance", "transmittance", "delta_R_abs") else 1
                    values = [scale * (max(abs(r[metric]), 1e-16) if metric == "energy_residual" else r[metric]) for r in available]
                    label_line = method.upper()
                    if group["first_oom_order"] is not None:
                        label_line += f" (OOM M={group['first_oom_order']})"
                        ax.axvline(group["first_oom_order"], color=colors[method], ls=":", alpha=.5)
                    ax.plot([r["order"] for r in available], values, "o-", color=colors[method], label=label_line, ms=3)
                    invalid = [r for r in available if not r["physical"]]
                    if invalid:
                        ax.scatter([r["order"] for r in invalid],
                                   [scale * (max(abs(r[metric]), 1e-16) if metric == "energy_residual" else r[metric]) for r in invalid],
                                   marker="x", color="crimson", zorder=5)
                if metric == "delta_R_abs":
                    ax.axhline(100 * report["plan"]["tolerance"], color="gray", ls="--", lw=1)
                    ax.set_yscale("symlog", linthresh=1e-6)
                elif metric == "energy_residual":
                    ax.set_yscale("log")
                    ax.axhline(report["plan"]["energy_tolerance"], color="gray", ls="--", lw=1)
                ax.set(title=f"{lattice}, {wavelength:g} nm", xlabel="Retained order M", ylabel=label)
                ax.grid(alpha=.22)
                if ax.get_legend_handles_labels()[0]:
                    ax.legend(fontsize=8)
        fig.suptitle(f"n=2.5 moth-eye; Nz=15, fixed Q={report['plan']['grid']}; red crosses: nonphysical")
        fig.tight_layout()
        for extension in ("png", "svg"):
            fig.savefig(figures / f"{metric}_vs_order.{extension}", dpi=160)
        plt.close(fig)


def run_series(output, saved, key, args):
    group, plan = saved["series"][key], saved["plan"]
    if group["status"] in ("stopped_cuda_oom", "stopped_cpu_oom") and not args.retry_oom:
        print(f"{key}: previous OOM at M={group['failed_order']}; skipped (use --retry-oom to retry).", flush=True)
        return
    group.pop("failed_order", None)
    group.pop("error", None)
    order = plan["start_order"]
    while str(order) in group["cases"]:
        order += plan["step"]
    group["status"] = "running"
    save(output, saved)
    try:
        while True:
            if args.max_order is not None and order > args.max_order:
                group["status"] = "reached_max_order"
                break
            if order > plan["grid_max_order"]:
                group["status"] = "stopped_grid_limit"
                group["error"] = "Fixed Q would undersample this order. Increase Q in a NEW output root. This is not OOM."
                break
            print(f"{key}: M={order}, Nz=15, Q={plan['grid']}", flush=True)
            request = {"plan": plan, "runtime_environment": saved["runtime_environment"], "order": order,
                       **{k: group[k] for k in ("lattice", "method", "wavelength_nm")}}
            outcome = launch(output, request, f"{key}/M{order}")
            group["attempts"].append({"order": order, **{k: v for k, v in outcome.items()
                                     if k not in ("result", "traceback", "runtime_environment")}})
            if outcome["status"] == "completed":
                row = outcome["result"]
                group["cases"][str(order)] = row
                print(f"  R={row['reflectance']:.9g}, T={row['transmittance']:.9g}, "
                      f"R+T-1={row['energy_residual']:.3g}; {row['runtime_seconds']:.2f}s" +
                      (" [NONPHYSICAL]" if not physical(row, plan["energy_tolerance"]) else ""), flush=True)
                order += plan["step"]
                save(output, saved)
                continue
            group["failed_order"] = order
            group["error"] = outcome.get("error", "Worker failed.")
            group["status"] = "stopped_" + outcome["status"] if outcome["status"] in ("cuda_oom", "cpu_oom") else "worker_error"
            print(f"  {group['status']} at M={order}: {group['error']}; continuing with the next series.", flush=True)
            break
    except KeyboardInterrupt:
        group["status"] = "interrupted"
        raise
    finally:
        save(output, saved)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda", help="cuda (default), cuda:<index>, cpu or auto")
    p.add_argument("--grid", type=int, default=512, help="Fixed NV projection/ASR quadrature grid Q")
    p.add_argument("--start-order", type=int, default=4)
    p.add_argument("--step", type=int, default=2)
    p.add_argument("--max-order", type=int, help="Optional finite cap; default: continue until OOM or fixed-Q limit")
    p.add_argument("--lattices", default="square,hexagonal")
    p.add_argument("--methods", default="nv,asr")
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--tolerance", type=float, default=.00025, help="R/T/A absolute change; 0.00025 = 0.025 percentage points")
    p.add_argument("--energy-tolerance", type=float, default=1e-6)
    p.add_argument("--no-symmetry", action="store_true")
    p.add_argument("--retry-oom", action="store_true")
    p.add_argument("--prepare-only", "--dry-run", dest="prepare_only", action="store_true")
    p.add_argument("--report-only", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--worker-request", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.worker_request:
        if not args.worker_result:
            p.error("Worker result path is required.")
        return worker(args.worker_request, args.worker_result)
    if args.report_only:
        output = args.output_root.resolve()
        saved = json.loads((output / "checkpoint.json").read_text(encoding="utf-8"))
        report = save(output, saved)
        if not args.no_plots:
            make_figures(output, report)
        print(f"Report: {output / 'report.json'}")
        return 0
    try:
        for name, allowed in (("lattices", {"square", "hexagonal"}), ("methods", {"nv", "asr"})):
            values = list(dict.fromkeys(v.strip() for v in getattr(args, name).split(",")))
            if not values or any(v not in allowed for v in values):
                raise ValueError(f"Invalid {name}: {values}")
            setattr(args, name, ",".join(values))
        if min(args.start_order, args.step, args.cpu_threads) <= 0:
            raise ValueError("Start order, step and CPU threads must be positive.")
        if args.grid < max(32, 4 * args.start_order + 4):
            raise ValueError("Fixed grid must be >=max(32,4*start_order+4).")
        if args.max_order is not None and args.max_order < args.start_order:
            raise ValueError("Maximum order must be >=start_order.")
        if not all(math.isfinite(v) and v > 0 for v in (args.tolerance, args.energy_tolerance)):
            raise ValueError("Tolerances must be finite and positive.")
        if args.device not in ("cuda", "cpu", "auto") and not (args.device.startswith("cuda:") and args.device[5:].isdigit()):
            raise ValueError("Device must be cuda, cuda:<index>, cpu or auto.")
        output, saved, requested = prepare(args)
    except (ValueError, OSError) as error:
        p.error(str(error))
    report = save(output, saved)
    if args.prepare_only:
        print(f"Prepared {len(requested)} independent series; Nz=15, Q={args.grid}; no optical solve.")
        print(f"Output: {output}")
        return 0
    preflight = launch(output, {"plan": saved["plan"], "preflight": True,
                               "runtime_environment": saved.get("runtime_environment")}, "preflight")
    if preflight["status"] != "ready":
        print(f"Preflight failed: {preflight.get('error')}; see {output / 'workers/preflight/worker.log'}", flush=True)
        return 1
    saved["runtime_environment"] = preflight["runtime_environment"]
    save(output, saved)
    print(f"Runtime: {saved['runtime_environment']}; output: {output}", flush=True)
    interrupted = False
    try:
        for key in requested:
            run_series(output, saved, key, args)
            if not args.no_plots:
                try:
                    make_figures(output, save(output, saved))
                except Exception as error:
                    print(f"Plotting failed: {error}; saved data intact. Use --report-only.", flush=True)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        report = save(output, saved)
    print(json.dumps({k: {"status": report['series'][k]['status'],
                               "highest_completed_order": report['series'][k]['highest_completed_order'],
                               "first_oom_order": report['series'][k]['first_oom_order'],
                               "selected_order_at_fixed_Nz_Q": report['series'][k]['selected_order']}
                      for k in requested}, indent=2), flush=True)
    print(f"Report: {output / 'report.json'}", flush=True)
    if interrupted:
        return 130
    statuses = [saved["series"][k]["status"] for k in requested]
    if "worker_error" in statuses:
        return 1
    return 2 if "stopped_grid_limit" in statuses else 0


if __name__ == "__main__":
    raise SystemExit(main())
