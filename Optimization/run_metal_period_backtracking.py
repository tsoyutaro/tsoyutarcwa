"""Prepare or resume independent metal studies with fixed-order Adam backtracking.

The numerical solver and each existing experiment remain unchanged. Period,
height, order and grid select separate configs/results. Repeating a command
resumes its Optuna journal and Adam checkpoint through the original run.py.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys

from run_al_p100_backtracking import prepare_recovery_case

ROOT = Path(__file__).resolve().parent
AL_SOURCE = Path("Al/al_motheye_P200_H500_Nz15")
CU_CASE = Path("Cu/cu_motheye_P200_H500_Nz15_backtracking_v1")
AU_CASE = Path("Au/au_motheye_P200_H500_Nz15_backtracking_v1")
BACKTRACKING_CASES = {"Cu": CU_CASE, "Au": AU_CASE}
DEFAULT_ORDERS = {"Al": 8, "Cu": 14, "Au": 20}
POLICY = "validated-adam-backtracking-v1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_new_config(path, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)
            handle.write("\n")
    except FileExistsError:
        if read(path) != config:
            raise ValueError(f"A different configuration already exists: {path}")


def validate_saved(config, expected, path):
    changed = [key for key, value in expected.items() if config.get(key) != value]
    if changed:
        raise ValueError(f"Saved configuration differs ({', '.join(changed)}): {path}. "
                         "Preserve that experiment and select different conditions.")


@dataclass
class Plan:
    material: str
    case: Path
    config_path: Path
    config: dict
    output: Path
    import_source: Path | None = None


def prepare(root, material, *, period_nm, height_nm=500., order=None, grid=512,
            layers=15, trials=30, adam_steps=500, run_tag=None):
    order = DEFAULT_ORDERS[material] if order is None else order
    if (not all(math.isfinite(v) and v > 0 for v in (period_nm, height_nm))
            or period_nm <= 2.022 or layers < 2 or min(order, grid, trials, adam_steps) <= 0
            or grid < max(32, 4 * order + 4)):
        raise ValueError("Positive dimensions/steps/trials, at least two layers and a sufficient grid are required.")
    if run_tag is not None and (not run_tag or run_tag in (".", "..") or Path(run_tag).name != run_tag
                    or any(c in run_tag for c in "/\\")):
        raise ValueError("run-tag must be one directory name.")
    label = f"P{period_nm:.15g}_H{height_nm:.15g}_Nz{layers}"
    defaults = grid == 512 and layers == 15 and trials == 30 and adam_steps == 500
    expected = dict(material=material, period_nm=period_nm, height_nm=height_nm,
                    layers=layers, lattices=["hexagonal"], normal_mode="off",
                    optimization_order=order, optimization_grid=grid, trials=trials,
                    adam_steps=adam_steps, adam_recovery_policy=POLICY,
                    nonphysical_max_retries=4, nonphysical_lr_factor=.5,
                    rcwa_acceleration="auto")
    if material == "Al":
        suffix = "" if order == 8 else f"_M{order}"
        if grid != 512:
            suffix += f"_Q{grid}"
        if trials != 30:
            suffix += f"_T{trials}"
        if adam_steps != 500:
            suffix += f"_S{adam_steps}"
        relative_case = Path("Al") / f"al_motheye_{label}_backtracking_v1{suffix}"
        case = root / relative_case
        config_path = case / "config.json"
    else:
        case = root / BACKTRACKING_CASES[material]
        condition = case if (period_nm, height_nm, layers) == (200., 500., 15) else case / "geometry_cases" / label
        if adam_steps != 500:
            condition = condition / f"adam_steps_{adam_steps}"
        config_path = condition / f"config_hex_OFF_optuna{trials}_backtracking_M{order}_Q{grid}.json"

    import_source = None
    legacy_al = material == "Al" and defaults and (period_nm, height_nm, order) == (100., 500., 8)
    source_run = root / AL_SOURCE / "geometry_cases/P100_H500_Nz15/results/hex_OFF_optuna30_auto_v1_M8_Q512"
    if config_path.is_file():
        config = read(config_path)
        validate_saved(config, expected, config_path)
    else:
        from_legacy_manifest = legacy_al and (source_run / "manifest.json").is_file()
        if from_legacy_manifest:
            config = dict(read(source_run / "manifest.json")["config"])
        elif material == "Al":
            source = root / AL_SOURCE
            if (period_nm, height_nm, layers) != (200., 500., 15):
                source = source / "geometry_cases" / label
            legacy_config = source / "config_hex_OFF_optuna30_M8_Q512.json"
            config = read(legacy_config if legacy_config.is_file() else root / AL_SOURCE / "config.json")
        else:
            config = read(case / "config.json")
        config.pop("_metal_csv_path", None)
        config.pop("material_provenance", None)
        config.pop("initial_logits", None)
        config.pop("initial_profile_provenance", None)
        config.update(expected)
        config["case_id"] = f"{material.lower()}_motheye_{label}_backtracking_v1" + (suffix if material == "Al" else "")
        if not from_legacy_manifest:
            config.update(radius_min_nm=.001, radius_max_nm=period_nm / 2 - .001,
                          baseline_tip_nm=.025 * period_nm, baseline_base_nm=.475 * period_nm,
                          init_top_min_nm=1., init_top_max_nm=min(99., period_nm / 2 - .011))
        if order > max(config["verification_orders"]):
            config["verification_orders"] = sorted({*config["verification_orders"], order})
        config["verification_grid"] = max(config["verification_grid"], 32, 4 * max(config["verification_orders"]) + 4)
        if material == "Al":
            prepare_recovery_case(root, relative_case, config)
            config = read(config_path)
        else:
            write_new_config(config_path, config)
        validate_saved(config, expected, config_path)

    tag = run_tag or f"hex_OFF_optuna{trials}_backtracking_v1"
    condition = case if material == "Al" else config_path.parent
    output = condition / "results" / f"{tag}_M{order}_Q{grid}"
    # Keep the established imported-trial studies as the default resume target.
    legacy_tag = None
    if run_tag is None and legacy_al:
        legacy_tag = "hex_OFF_optuna30_backtracking_from10_v1_M8_Q512"
    elif (run_tag is None and material == "Cu" and defaults
          and (period_nm, height_nm, order) == (200., 500., 14)):
        legacy_tag = "hex_OFF_optuna30_backtracking_from15_v1_M14_Q512"
    elif (run_tag is None and material == "Au" and defaults
          and (period_nm, height_nm, order) == (200., 500., 20)):
        legacy_tag = "hex_OFF_optuna30_backtracking_from5_v1_M20_Q512"
    if legacy_tag:
        legacy = case / "results" / legacy_tag
        # Au's imported-from-5 study is the established source of the published
        # spectra. Its older interrupted standard study remains separate.
        if material != "Au" and (output / "manifest.json").exists() and (legacy / "manifest.json").exists():
            raise ValueError(f"Two existing studies match these conditions. Select --run-tag explicitly: {output}, {legacy}")
        if (legacy / "manifest.json").exists() or (legacy_al and (source_run / "manifest.json").exists()
                                                  and not (output / "manifest.json").exists()):
            output = legacy
            if legacy_al and not (legacy / "manifest.json").exists():
                import_source = source_run
    return Plan(material, case, config_path, config, output, import_source)


def finished(plan):
    workflow = plan.output / "workflow.json"
    group = plan.output / "hexagonal/report.json"
    return (workflow.is_file() and group.is_file()
            and read(workflow).get("status") == "completed"
            and read(group).get("status") == "completed"
            and len(read(group).get("best", {}).get("logits", [])) == plan.config["layers"] + 1)


def pid_alive(pid):
    if os.name == "nt":
        # Windows os.kill(pid, 0) can terminate a process. Query it instead.
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.GetExitCodeProcess.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() != 87  # Invalid PID is not running.
        try:
            code = wintypes.DWORD()
            return not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def check_lock(lock, recover):
    if not lock.exists():
        return
    owner_path = lock / "owner.json"
    owner = read(owner_path) if owner_path.is_file() else {}
    if (owner.get("host") == socket.gethostname() and isinstance(owner.get("pid"), int)
            and owner["pid"] > 0):
        if pid_alive(owner["pid"]):
            raise RuntimeError(f"The lock owner is still alive: {lock}, pid={owner['pid']}")
    if not recover:
        raise RuntimeError(f"A previous lock remains: {lock}. After its controller and workers stop, use --recover-lock.")
    # For another host, --recover-lock asserts that its old job has stopped.


@contextmanager
def launcher_lock(output, recover):
    # The legacy importer publishes the output directory atomically and requires
    # it not to exist beforehand. Keep the wrapper lock beside that directory.
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_name(output.name + ".period_launcher.lock")
    check_lock(lock, recover)
    if lock.exists():
        (lock / "owner.json").unlink(missing_ok=True)
        lock.rmdir()
    lock.mkdir()
    try:
        (lock / "owner.json").write_text(json.dumps({"pid": os.getpid(), "host": socket.gethostname()}), encoding="utf-8")
        yield
    finally:
        (lock / "owner.json").unlink(missing_ok=True)
        lock.rmdir()


def command(plan, args):
    cmd = [sys.executable, str(plan.case / "run.py"), "--config", str(plan.config_path),
           "--output-root", str(plan.output), "--device", args.device,
           "--mode", "optimize", "--lattices", "hexagonal", "--timeout-hours", "0"]
    if args.recover_lock:
        cmd.append("--recover-lock")
    return cmd


def main(argv=None, *, root=ROOT, executor=subprocess.run):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materials", "--material", required=True, help="Comma-separated materials from Al,Cu,Au")
    periods = parser.add_mutually_exclusive_group(required=True)
    periods.add_argument("--period-nm", type=float, help="One nearest-neighbor period in nm")
    periods.add_argument("--periods", help="Comma-separated periods in nm, executed in this order")
    heights = parser.add_mutually_exclusive_group()
    heights.add_argument("--height-nm", type=float, help="One fixed height in nm (default: 500)")
    heights.add_argument("--heights", help="Comma-separated heights in nm, executed in this order")
    parser.add_argument("--order", type=int, help="Fixed order. Defaults: Al=8, Cu=14, Au=20")
    parser.add_argument("--grid", type=int, default=512)
    parser.add_argument("--layers", type=int, default=15)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--adam-steps", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-tag", help="Separate result-study prefix. Omit to resume established studies")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare configs and print commands without importing trials or running RCWA")
    parser.add_argument("--recover-lock", action="store_true", help="Assert previous controllers/workers have stopped; live local lock owners are refused")
    args = parser.parse_args(argv)
    materials = [x.strip() for x in args.materials.split(",")]
    if len(set(materials)) != len(materials) or any(m not in DEFAULT_ORDERS for m in materials):
        parser.error("Select distinct materials from Al,Cu,Au")
    try:
        selected_periods = [args.period_nm] if args.period_nm is not None else [float(x.strip()) for x in args.periods.split(",")]
        selected_heights = ([float(x.strip()) for x in args.heights.split(",")] if args.heights is not None
                            else [500. if args.height_nm is None else args.height_nm])
        if (len(set(selected_periods)) != len(selected_periods)
                or len(set(selected_heights)) != len(selected_heights)):
            raise ValueError("Select distinct periods and heights.")
        if not all(math.isfinite(v) and v > 0 for v in selected_periods + selected_heights):
            raise ValueError("Periods and heights must be finite and positive.")
        for material in materials:
            for period in selected_periods:
                for height in selected_heights:
                    plan = prepare(root, material, period_nm=period, height_nm=height,
                                   order=args.order, grid=args.grid, layers=args.layers,
                                   trials=args.trials, adam_steps=args.adam_steps, run_tag=args.run_tag)
                    print(f"\n{material}: P={period:g}, H={height:g} nm, "
                          f"M={plan.config['optimization_order']} Q={args.grid}, output={plan.output}", flush=True)
                    cmd = command(plan, args)
                    print(shlex.join(cmd), flush=True)
                    if args.prepare_only:
                        if plan.import_source:
                            print(f"Will import trials 0-9 and retry trial 10 from {plan.import_source}", flush=True)
                        continue
                    check_lock(plan.output / "controller.lock", args.recover_lock)
                    with launcher_lock(plan.output, args.recover_lock):
                        if finished(plan):
                            controller = plan.output / "controller.lock"
                            if controller.exists() and args.recover_lock:
                                check_lock(controller, True)
                                (controller / "owner.json").unlink(missing_ok=True)
                                controller.rmdir()
                            print("Already completed; proceeding to the next condition.", flush=True)
                            continue
                        if plan.import_source:
                            imported = executor([sys.executable, str(plan.case / "import_completed_trials.py"),
                                                 "--source-run", str(plan.import_source), "--output-root", str(plan.output),
                                                 "--restart-trial", "10", "--device", args.device], cwd=plan.case)
                            if imported.returncode:
                                return imported.returncode if imported.returncode > 0 else 128 - imported.returncode
                        completed = executor(cmd, cwd=plan.case)
                        if completed.returncode:
                            return completed.returncode if completed.returncode > 0 else 128 - completed.returncode
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Stopped: {error}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
