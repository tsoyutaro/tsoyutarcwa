"""NV-only Optuna/Adam driver; GPU training M8, streamed verification M16+."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import signal
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
# Optional workspace-only dependency directory used by local validation.
if os.environ.get("MOTHEYE_DEPENDENCY_PATH"):
    sys.path.append(os.environ["MOTHEYE_DEPENDENCY_PATH"])

from optimizer.common import (conditions, device_setup, load_config,
    runtime_environment, signature_for, sources, write_csv, write_json)
from optimizer.geometry import baseline_logits, profile_rows, radii_values
from runtime_policy import (check_runtime, execution_sources, persist_manifest,
    reconcile_manifest, record_runtime)


def storage_and_pruner(folder, config):
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
    path = str(folder/"optuna_journal.log")
    storage = JournalStorage(JournalFileBackend(path, lock_obj=JournalFileOpenLock(path)))
    pruner = optuna.pruners.MedianPruner(n_startup_trials=config["pruner_startup_trials"],
        n_warmup_steps=config["pruner_warmup_steps"], interval_steps=config["pruner_interval_steps"])
    return storage, pruner


def is_cuda_oom(error, device):
    """CUDA allocation can fail before device_setup returns a device object."""
    import torch
    message = " ".join(str(error).lower().split())
    return ((isinstance(error, torch.OutOfMemoryError) and device is not None and device.type == "cuda")
            or "cuda out of memory" in message or "cuda error: out of memory" in message)


def worker(request_path, result_path):
    request = json.loads(request_path.read_text(encoding="utf-8"))
    c, environment = request["config"], None
    device = None
    started = time.perf_counter()
    try:
        import torch
        device = device_setup(request["device"], c["cpu_threads"])
        torch.manual_seed(c["seed"]+request.get("trial_number", 0))
        environment = runtime_environment(device)
        check_runtime(request.get("runtime_environment"), environment)
        if request["mode"] == "preflight":
            outcome = {"status": "ready"}
        elif request["mode"] == "train":
            import optuna
            from optimizer.engine import train
            folder = Path(request["study_folder"])
            storage, pruner = storage_and_pruner(folder, c)
            study = optuna.load_study(study_name=request["study_name"], storage=storage, pruner=pruner)
            trial = optuna.trial.Trial(study, request["trial_id"])
            outcome = train(request, device, trial)
        elif request["mode"] in ("analyze_spectrum", "analyze_timing"):
            from optimizer.postprocess import spectrum_worker, timing_worker
            outcome = (timing_worker if request["mode"] == "analyze_timing" else spectrum_worker)(request, device)
        elif request["mode"] == "audit":
            from optimizer.audit import check
            outcome = check(request, device)
        elif request["mode"] == "verify":
            from optimizer.physics import solve_verify
            radii = torch.tensor(radii_values(request["logits"], c), dtype=torch.float64, device=device)
            row = solve_verify(radii, c, request["lattice"], request["wavelength_nm"], request["order"], request["grid"], device)
            outcome = {"status": "completed", "result": {"lattice": request["lattice"],
                "wavelength_nm": request["wavelength_nm"], "order": request["order"], "grid": request["grid"], **row}}
        else:
            raise ValueError("Unknown worker mode.")
    except (Exception, KeyboardInterrupt) as error:
        import torch
        message = str(error).lower()
        confirmed = is_cuda_oom(error, device)
        cpu_oom = (isinstance(error, MemoryError) or isinstance(error, torch.OutOfMemoryError) and device is not None and device.type == "cpu"
                   or "defaultcpuallocator" in message and "not enough memory" in message)
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else (
            "cuda_oom" if confirmed else "cpu_oom" if cpu_oom else "error")
        outcome = {"status": status, "error_type": type(error).__name__, "error": str(error),
                   "traceback": traceback.format_exc()}
        print(outcome["traceback"], file=sys.stderr, flush=True)
    outcome.update(request_signature=signature_for(request), runtime_environment=environment,
                   process_wall_seconds=time.perf_counter()-started)
    if device is not None and device.type == "cuda":
        try:
            import torch
            outcome.update(peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
                           peak_cuda_reserved_bytes=int(torch.cuda.max_memory_reserved(device)))
        except Exception:
            pass
    write_json(result_path, outcome)
    return 1 if outcome["status"] == "error" else 0


def launch(folder, request):
    folder.mkdir(parents=True, exist_ok=True)
    req, res = folder/"request.json", folder/"result.json"
    if res.exists():
        res.unlink()
    write_json(req, request)
    with (folder/"worker.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
            "--worker-request", str(req), "--worker-result", str(res)], stdout=log, stderr=subprocess.STDOUT)
        try:
            process.wait()
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if not res.exists():
        return {"status": "error", "error": "Worker exited without a result; not a confirmed OOM.", "returncode": process.returncode}
    result = json.loads(res.read_text(encoding="utf-8"))
    if result.get("request_signature") != signature_for(request) or process.returncode != 0 and result["status"] != "error":
        return {"status": "error", "error": "Worker result identity/exit code differs."}
    return result


def save_sampler(path, sampler, version):
    temporary = path.with_name(path.name+".tmp")
    with temporary.open("wb") as file:
        pickle.dump({"optuna_version": version, "sampler": sampler}, file)
    temporary.replace(path)


def optimize_lattice(folder, request_base, deadline):
    import optuna
    c, lattice = request_base["config"], request_base["lattice"]
    storage, pruner = storage_and_pruner(folder, c)
    sampler_path = folder/"sampler.pkl"
    if sampler_path.exists():
        # Only load a sampler created by this program in this signed run folder.
        with sampler_path.open("rb") as file:
            restored = pickle.load(file)
        if restored["optuna_version"] != optuna.__version__:
            raise ValueError("Optuna version differs from saved sampler.")
        sampler = restored["sampler"]
    else:
        sampler = optuna.samplers.TPESampler(seed=c["seed"], multivariate=True)
    name = "motheye_nv_off_"+lattice
    study = optuna.create_study(study_name=name, direction="minimize", storage=storage,
                               sampler=sampler, pruner=pruner, load_if_exists=True)
    if not study.trials:
        study.enqueue_trial({"lr": .03})
    while True:
        terminal = [t for t in study.trials if t.state.is_finished()]
        active = [t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]
        if len(active) > 1:
            raise RuntimeError("Multiple running trials violate the single-controller policy.")
        if deadline is not None and time.time() >= deadline:
            return study, "paused"
        if not active and len(terminal) >= c["trials"]:
            return study, "completed"
        if active:
            trial = optuna.trial.Trial(study, active[0]._trial_id)
            print(f"Resuming {lattice} trial {trial.number}", flush=True)
        else:
            trial = study.ask()
        params = {"lr": trial.suggest_float("lr", c["learning_rate_min"], c["learning_rate_max"], log=True)}
        save_sampler(sampler_path, study.sampler, optuna.__version__)
        trial_folder = folder/"trials"/f"trial_{trial.number:04d}"
        request = {**request_base, "mode": "train", "params": params, "trial_number": trial.number,
                   "trial_id": trial._trial_id, "study_name": name, "study_folder": str(folder),
                   "trial_folder": str(trial_folder), "deadline": deadline}
        print(f"{lattice} trial {trial.number}: {params}", flush=True)
        outcome = launch(trial_folder/"worker", request)
        write_json(trial_folder/"outcome.json", outcome)
        status = outcome["status"]
        if status in ("cuda_oom", "cpu_oom", "paused", "interrupted"):
            return study, status  # Keep RUNNING so this exact Adam checkpoint resumes.
        if status == "completed":
            for key in ("best_logits", "radii_nm", "best_step", "completed_steps"):
                trial.set_user_attr(key, outcome[key])
            study.tell(trial, outcome["best_loss"])
        elif status == "pruned":
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
        else:
            trial.set_user_attr("error", outcome.get("error"))
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            print(f"trial {trial.number} failed: {outcome.get('error')}; retained in journal.", flush=True)
        save_sampler(sampler_path, study.sampler, optuna.__version__)
        export_trials(folder, study)


def export_trials(folder, study):
    rows = [{"trial_number": t.number, "state": t.state.name, "objective": t.value,
             **t.params, "best_step": t.user_attrs.get("best_step"), "error": t.user_attrs.get("error")}
            for t in study.trials]
    write_csv(folder/"trials.csv", rows)


def verification(folder, base, logits, label, deadline, orders, retry_oom):
    from optimizer.physics import weights
    c = base["config"]
    shape_id = signature_for({"logits": logits, "signature": base["signature"]})[:16]
    output = folder/"verification"/f"{label}_{shape_id}"
    waves = sorted(set(c["verification_wavelengths_nm"]+c["optimization_wavelengths_nm"]+c["audit_wavelengths_nm"]))
    reports = []
    for order, grid in conditions(c, orders):
        path = output/f"M{order}_Q{grid}.json"
        identity = signature_for({"shape": shape_id, "order": order, "grid": grid, "waves": waves})
        saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"signature": identity, "cases": {}, "status": "pending"}
        if saved["signature"] != identity:
            raise ValueError("Verification identity differs.")
        if saved["status"] in ("cuda_oom", "cpu_oom") and not retry_oom:
            reports.append({"order": order, "grid": grid, "path": str(path), **saved})
            continue
        for wave in waves:
            key = f"{wave:.17g}"
            if key in saved["cases"]:
                continue
            if deadline is not None and time.time() >= deadline:
                saved["status"] = "paused"
                break
            print(f"verify {base['lattice']}/{label} M={order} Q={grid} lambda={wave:g}", flush=True)
            request = {**base, "mode": "verify", "logits": logits, "wavelength_nm": wave, "order": order, "grid": grid}
            outcome = launch(output/"workers"/f"M{order}_Q{grid}_wl{wave:g}", request)
            if outcome["status"] != "completed":
                saved.update(status=outcome["status"], failed_wavelength_nm=wave, error=outcome.get("error"))
                write_json(path, saved)
                break
            saved["cases"][key] = {**outcome["result"], "process_wall_seconds": outcome["process_wall_seconds"],
                "peak_cuda_allocated_bytes": outcome.get("peak_cuda_allocated_bytes"),
                "peak_cuda_reserved_bytes": outcome.get("peak_cuda_reserved_bytes")}
            saved["status"] = "running"
            write_json(path, saved)
        else:
            saved["status"] = "completed"
        rows = sorted(saved["cases"].values(), key=lambda r: r["wavelength_nm"])
        saved["requested_wavelength_count"] = len(waves)
        saved["complete"] = len(rows) == len(waves) and saved["status"] == "completed"
        if saved["complete"]:
            for name, selected in (("optimization", c["optimization_wavelengths_nm"]), ("detailed", c["verification_wavelengths_nm"])):
                saved[f"{name}_mean_reflectance"] = sum(weight*saved["cases"][f"{wave:.17g}"]["reflectance"]
                    for wave, weight in zip(selected, weights(selected, c["objective_weighting"])))
        write_json(path, saved)
        write_csv(path.with_suffix(".csv"), rows)
        reports.append({"order": order, "grid": grid, "path": str(path), **saved})
    comparisons = []
    complete = [r for r in reports if r.get("complete")]
    for i, a in enumerate(complete):
        for b in complete[i+1:]:
            if (a["order"] == b["order"]) == (a["grid"] == b["grid"]):
                continue  # Vary exactly one numerical axis in each comparison.
            change = max(abs(a["cases"][f"{w:.17g}"]["reflectance"]-b["cases"][f"{w:.17g}"]["reflectance"]) for w in waves)
            comparisons.append({"axis": "grid" if a["order"] == b["order"] else "order",
                                "coarse": [a["order"], a["grid"]], "fine": [b["order"], b["grid"]],
                                "max_abs_R_change": change, "passed": change <= c["verification_tolerance"]})
    return {"shape_id": shape_id, "logits": logits, "radii_nm": radii_values(logits, c), "conditions": reports,
            "comparisons": comparisons, "complete": all(r.get("complete") for r in reports),
            "limits": "Finite NV order/grid comparison for this stepped profile, not an exact-reference error bound."}


def audit_profile(folder, base, logits, label, skip, retry_oom=False):
    if skip or not base["config"]["gradient_audit"]:
        return {"status": "skipped", "passed": False}
    identity = signature_for({"base": base["signature"], "logits": logits, "label": label})[:16]
    path = folder/"audits"/f"{label}_{identity}"/"result.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if not retry_oom or saved["status"] not in ("cuda_oom", "cpu_oom"):
            return saved
    outcome = launch(path.parent, {**base, "mode": "audit", "logits": logits})
    # Audit failures are evidence; they do not silently change requested M8.
    print(f"{base['lattice']} {label} gradient audit: status={outcome['status']}, passed={outcome.get('passed')}", flush=True)
    return outcome


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-root", type=Path)
    p.add_argument("--mode", choices=("all", "optimize", "verify", "audit", "plot", "convergence", "timing", "spectrum"), default="optimize")
    p.add_argument("--lattices", help="Optional square,hexagonal subset")
    p.add_argument("--verification-orders", help="For example 8,16,20,24,28; optimization stays M8")
    p.add_argument("--timeout-hours", type=float, default=24.)
    p.add_argument("--skip-audit", action="store_true")
    p.add_argument("--checkpoint-layers", action="store_true", help="Recompute layer modes in backward; trades speed for gradient memory")
    p.add_argument("--retry-oom", action="store_true")
    p.add_argument("--recover-lock", action="store_true", help="Use only after confirming the previous controller/job has stopped")
    p.add_argument("--prepare-only", "--dry-run", dest="prepare_only", action="store_true")
    p.add_argument("--smoke", action="store_true", help="Small real Optuna/Adam test; uses M1/M2 and smaller grids")
    p.add_argument("--wavelengths", help="Postprocessing wavelengths; e.g. 400:700:3")
    p.add_argument("--orders", help="Convergence orders, comma separated")
    p.add_argument("--grids", help="Convergence grids, comma separated")
    p.add_argument("--order", type=int, help="Spectrum/timing Fourier order")
    p.add_argument("--grid", type=int, help="Spectrum/timing NVM grid")
    p.add_argument("--repeats", type=int, default=3, help="Warm timing repetitions")
    p.add_argument("--forward-only", action="store_true", help="Timing without gradient")
    p.add_argument("--analysis-backend", choices=("auto","matrix","reference"), default="auto")
    p.add_argument("--profile-json", type=Path, help="Optional saved logits for postprocessing")
    p.add_argument("--worker-request", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
    return p


def main(argv=None):
    p, args = parser(), None
    args = p.parse_args(argv)
    if args.mode == "all":
        args.mode = "optimize"  # Legacy alias: always stop before verification.
    def interrupted(signum, frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, interrupted)
    if args.worker_request:
        if not args.worker_result:
            p.error("Missing worker result path.")
        return worker(args.worker_request, args.worker_result)
    if args.config is None or args.output_root is None:
        p.error("Controller requires --config and --output-root (or use a case run.py).")
    output = args.output_root.resolve()
    if args.mode == "plot":
        from optimizer.plots import make_plots
        make_plots(output)
        return 0
    try:
        c = load_config(args.config, args.smoke)
        if args.checkpoint_layers:
            c["checkpoint_layers"] = True
        lattices = args.lattices.split(",") if args.lattices else c["lattices"]
        if any(l not in c["lattices"] for l in lattices):
            raise ValueError("Requested lattice is absent from config.")
        orders = sorted(set(int(v) for v in args.verification_orders.split(","))) if args.verification_orders else c["verification_orders"]
        if not orders or min(orders) <= 0 or c["verification_grid"] < 4*max(orders)+4:
            raise ValueError("Invalid verification orders or insufficient verification grid.")
        if args.timeout_hours < 0:
            raise ValueError("Timeout hours must be nonnegative; 0 means no limit.")
        identity = {"version": 1, "config": c, "source_sha256": execution_sources(sources()), "requested_device": args.device}
        manifest_path = output/"manifest.json"
        original_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
        previous, migration = reconcile_manifest(original_manifest, identity)
        signature = previous["signature"]  # Keep existing Adam/shape/audit identities.
    except (ValueError, OSError) as error:
        p.error(str(error))
    output.mkdir(parents=True, exist_ok=True)
    lock = output/"controller.lock"
    if lock.exists() and args.recover_lock:
        (lock/"owner.json").unlink(missing_ok=True)
        lock.rmdir()
    try:
        lock.mkdir()
    except FileExistsError:
        p.error("Another controller holds this run. After its job has stopped, use --recover-lock.")
    write_json(lock/"owner.json", {"pid": os.getpid(), "host": socket.gethostname()})
    status, reports = "running", {}
    deadline = time.time()+args.timeout_hours*3600 if args.timeout_hours else None
    try:
        if args.prepare_only:
            persist_manifest(manifest_path, original_manifest, previous)
            write_json(output/"resolved_config.json", c)
            write_csv(output/"baseline_profile.csv", profile_rows(baseline_logits(c), c))
            status = "prepared"
            print(f"Prepared NV-only {lattices}; Adam M={c['optimization_order']} Q={c['optimization_grid']}; verification {conditions(c, orders)}.")
            print(f"GPU policy: same-model CUDA/MIG changes allowed; software/configuration/physics sources remain checked. Output: {output}")
            return 0
        preflight = launch(output/"preflight", {"mode": "preflight", "config": c, "device": args.device,
            "runtime_environment": previous.get("runtime_environment")})
        if preflight["status"] != "ready":
            raise RuntimeError(f"Preflight failed: {preflight.get('error')}")
        record_runtime(previous, preflight["runtime_environment"])
        persist_manifest(manifest_path, original_manifest, previous)
        write_json(output/"resolved_config.json", c)
        write_csv(output/"baseline_profile.csv", profile_rows(baseline_logits(c), c))
        if migration:
            print(f"Controller source upgrade recorded ({migration['kind']}); existing checkpoint signature preserved.", flush=True)
        if args.mode in ("convergence", "timing", "spectrum"):
            from optimizer.postprocess import run as postprocess
            status = postprocess(args,c,output,signature,lattices,launch,previous['runtime_environment'])
            return 0 if status == 'completed' else 1
        for lattice in lattices:
            folder = output/lattice
            folder.mkdir(exist_ok=True)
            base = {"config": c, "lattice": lattice, "device": args.device, "signature": signature,
                    "runtime_environment": previous["runtime_environment"]}
            group_path = folder/"report.json"
            group = json.loads(group_path.read_text(encoding="utf-8")) if group_path.exists() else {"lattice": lattice}
            if args.mode == "audit":
                group["initial_gradient_audit"] = audit_profile(folder, base, baseline_logits(c), "initial", args.skip_audit, args.retry_oom)
                write_json(group_path, group)
            if args.mode in ("all", "optimize"):
                study, status = optimize_lattice(folder, base, deadline)
                export_trials(folder, study)
                if status != "completed":
                    group["status"] = status
                    write_json(group_path, group)
                    break
            else:
                if args.mode == "audit" and not (folder/"optuna_journal.log").exists():
                    group["status"] = "initial_audit_only"
                    write_json(group_path, group)
                    reports[lattice] = group
                    continue
                import optuna
                storage, pruner = storage_and_pruner(folder, c)
                study = optuna.load_study(study_name="motheye_nv_off_"+lattice, storage=storage, pruner=pruner)
            completed = [t for t in study.trials if t.state.name == "COMPLETE"]
            if not completed:
                group["status"] = "no_completed_trials"
                write_json(group_path, group)
                status = "error"
                continue
            best = study.best_trial
            best_logits = best.user_attrs["best_logits"]
            group["best"] = {"trial_number": best.number, "objective": best.value, "params": best.params,
                "logits": best_logits, "radii_nm": radii_values(best_logits, c),
                "trial_folder": str(folder/"trials"/f"trial_{best.number:04d}")}
            write_csv(folder/"optimized_profile.csv", profile_rows(best_logits, c))
            if args.mode in ("all", "verify"):
                group["verification"] = {"baseline": verification(folder, base, baseline_logits(c), "baseline", deadline, orders, args.retry_oom),
                    "best": verification(folder, base, best_logits, "best", deadline, orders, args.retry_oom)}
            if args.mode in ("all", "audit"):
                group["final_gradient_audit"] = audit_profile(folder, base, best_logits, "best", args.skip_audit, args.retry_oom)
            group["status"] = "verification_partial" if group.get("verification") and not all(
                item["complete"] for item in group["verification"].values()) else "completed"
            group["scientific_scope"] = f"NV-only; optimized finite {c['layers']}-layer profile at M={c['optimization_order']}. Finite verification and sampled gradient checks are recorded separately; no global-optimum or publication-ready claim."
            write_json(group_path, group)
            reports[lattice] = group
        status = status if status in ("paused", "cuda_oom", "cpu_oom", "error") else (
            "verification_partial" if any(g.get("status") == "verification_partial" for g in reports.values()) else "completed")
    except KeyboardInterrupt:
        status = "interrupted"
    except Exception as error:
        status = "error"
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        write_json(output/"controller_error.json", {"error": str(error), "traceback": traceback.format_exc()})
    finally:
        if status != "prepared" and args.mode not in ("convergence", "timing", "spectrum"):
            write_json(output/"workflow.json", {"status": status, "requested_lattices": lattices, "signature": signature,
                                              "reports": {l: str(output/l/"report.json") for l in lattices}})
        (lock/"owner.json").unlink(missing_ok=True)
        lock.rmdir()
    if args.mode == "optimize":
        print(f"OFF optimization status={status}; verification not run; output: {output}",flush=True)
        return {"completed":0,"paused":2,"interrupted":130,"cuda_oom":3,"cpu_oom":3}.get(status,1)
    try:
        from optimizer.plots import make_plots
        make_plots(output)
    except Exception as error:
        print(f"Plots not generated: {error}; numerical data are saved.", flush=True)
    print(f"Workflow status={status}; output: {output}", flush=True)
    return {"completed": 0, "paused": 2, "verification_partial": 2, "cuda_oom": 3, "cpu_oom": 3, "interrupted": 130}.get(status, 1)


if __name__ == "__main__":
    raise SystemExit(main())
