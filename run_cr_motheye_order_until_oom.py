"""Increase Cr Fourier order at fixed Nz/Q, saving cases until the first CUDA OOM.

One fresh subprocess per wavelength prevents GPU tensors from accumulating
between cases. This uses the existing Cr matched-ASR solve without changing
its equations, layer storage, precision or symmetry reduction.
"""
from __future__ import annotations

import argparse
import csv
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
                back_index=1.0, asr_circle_g=0.03)
COLUMNS = ("wavelength_nm", "order", "slices", "grid", "epsilon_material_real",
           "epsilon_material_imag", "reflectance", "transmittance_far",
           "absorptance_total", "motheye_absorptance", "substrate_absorptance",
           "power_into_substrate", "passivity_warning", "reduced_dimension",
           "full_dimension", "symmetry_reduction", "runtime_seconds",
           "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes")


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
              "interpretation": "An order sweep and memory limit measurement, not a three-axis convergence certification."}
    write_json(output / "report.json", report)


def worker(plan_path, result_path, order, wave):
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    identity = plan["identity"]
    torch = None
    try:
        if source_hashes() != identity["source_sha256"] or digest(identity["cr_csv"]) != identity["material_sha256_lf"]:
            raise ValueError("Source or material changed after the plan was written.")
        import torch
        from studies.gold_motheye.converge import GeometryConfig, NumericalConfig, simulate_case
        torch.set_num_threads(identity["cpu_threads"])
        device = torch.device(identity["device"])
        if device.type == "cuda":
            if not torch.cuda.is_available() or (device.index or 0) >= torch.cuda.device_count():
                raise ValueError("Requested CUDA device is unavailable.")
            torch.cuda.set_device(device)
            torch.cuda.reset_peak_memory_stats(device)
        model = load_tabulated_nk(identity["cr_csv"])
        row = simulate_case(wave, NumericalConfig(order, identity["slices"], identity["grid"]),
                            GeometryConfig(**identity["geometry"]), model, cascade="redheffer",
                            use_symmetry=True, symmetry_reduction="d6-source", device=device, material_name="Cr")
        validate_case(row, identity)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        row.update(peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
                   peak_cuda_reserved_bytes=int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None)
        environment = {"torch": str(torch.__version__), "device": str(device), "cuda": torch.version.cuda,
                       "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
        write_json(result_path, {"status": "ok", "signature": plan["signature"], "case": row, "environment": environment})
        print(f"  R={100*row['reflectance']:.6f}%, A={100*row['absorptance_total']:.6f}%", flush=True)
        return 0
    except Exception as error:
        oom = torch is not None and isinstance(error, torch.cuda.OutOfMemoryError) and identity["device"].startswith("cuda")
        failure = {"status": "oom" if oom else "calculation_failed", "signature": plan["signature"],
                   "order": order, "wavelength_nm": wave, "slices": identity["slices"], "grid": identity["grid"],
                   "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
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
    p.add_argument("--slices", type=int, default=60, help="Fixed Nz; no layer-count scan")
    p.add_argument("--grid", type=int, default=512, help="Fixed Q for the entire order sweep")
    p.add_argument("--start-order", type=int, default=16)
    p.add_argument("--step", type=int, default=2)
    p.add_argument("--max-order", type=int, default=0, help="0: continue until OOM or the fixed-grid validity limit")
    p.add_argument("--wavelengths", default="400,550,700")
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--cr-csv", type=Path, default=ROOT / "studies/cr_motheye/data/Johnson.csv")
    p.add_argument("--output-dir", type=Path, default=ROOT / "studies/cr_motheye/results/order_Nz60_Q512_until_oom")
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
                    "source_sha256": source_hashes(), "solver": "existing matched-ASR, D6-source, complex128, Redheffer",
                    "scheduling": "one independent process per wavelength and order; stop at first CUDA OOM"}
        signature = identity_signature(identity)
        output = args.output_dir.resolve()
        plan = {"identity": identity, "signature": signature, "max_order": args.max_order,
                "fixed_grid_maximum_order": (args.grid - 4) // 4}
        checkpoint_path = output / "checkpoint.json"
        if checkpoint_path.exists():
            saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if saved["signature"] != signature:
                raise ValueError("Checkpoint source/material/conditions differ; use a new --output-dir.")
            for key, row in saved["cases"].items():
                validate_case(row, identity)
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
