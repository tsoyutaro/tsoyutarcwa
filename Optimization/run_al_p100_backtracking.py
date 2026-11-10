"""Reuse Al P100/H500 M8 trials 0-9 and retry trial 10 with Adam backtracking."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
SOURCE_CASE = Path("Al/al_motheye_P200_H500_Nz15")
SOURCE_RUN = SOURCE_CASE / "geometry_cases/P100_H500_Nz15/results/hex_OFF_optuna30_auto_v1_M8_Q512"
RECOVERY_CASE = Path("Al/al_motheye_P100_H500_Nz15_backtracking_v1")
RECOVERY_CODE = Path("Cu/cu_motheye_P200_H500_Nz15_backtracking_v1")
RECOVERY_RUN = Path("results/hex_OFF_optuna30_backtracking_from10_v1_M8_Q512")
RECOVERY_FILES = ("optimizer/common.py", "optimizer/driver.py", "optimizer/engine.py",
                  "import_completed_trials.py")
EXPECTED_TRAINING = {
    "material": "Al",
    "period_nm": 100.0,
    "height_nm": 500.0,
    "layers": 15,
    "lattices": ["hexagonal"],
    "normal_mode": "off",
    "optimization_order": 8,
    "optimization_grid": 512,
    "trials": 30,
}


def prepare_case(root):
    source_run = root / SOURCE_RUN
    manifest = json.loads((source_run / "manifest.json").read_text(encoding="utf-8-sig"))
    config = dict(manifest["config"])
    differences = [key for key, value in EXPECTED_TRAINING.items() if config.get(key) != value]
    if differences:
        raise ValueError("Unexpected source training conditions: " + ", ".join(differences))
    config.pop("_metal_csv_path", None)
    config.pop("material_provenance", None)
    config.update(case_id=RECOVERY_CASE.name,
                  adam_recovery_policy="validated-adam-backtracking-v1",
                  nonphysical_max_retries=4,
                  nonphysical_lr_factor=0.5)
    destination = root / RECOVERY_CASE
    if not destination.exists():
        source_case = root / SOURCE_CASE
        with tempfile.TemporaryDirectory(prefix=".al-p100-", dir=destination.parent) as temporary:
            stage = Path(temporary) / "case"
            stage.mkdir()
            for name in ("data", "optimizer", "rcwa_ext"):
                shutil.copytree(source_case / name, stage / name,
                                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            for name in ("run.py", "d6_nvm.py", "rcwa_solver_auto.py", "io_utils.py", "runtime_policy.py"):
                shutil.copyfile(source_case / name, stage / name)
            for name in RECOVERY_FILES:
                shutil.copyfile(root / RECOVERY_CODE / name, stage / name)
            (stage / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            try:
                stage.rename(destination)
            except OSError:
                if not destination.is_dir():
                    raise
    saved = json.loads((destination / "config.json").read_text(encoding="utf-8-sig"))
    if saved != config:
        raise ValueError(f"Existing recovery configuration differs: {destination / 'config.json'}")
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prepare-only", action="store_true", help="Import and queue trial 10 only")
    parser.add_argument("--recover-lock", action="store_true",
                        help="Recover the target lock after its controller and worker have stopped")
    args = parser.parse_args(argv)
    try:
        case = prepare_case(ROOT)
        output = case / RECOVERY_RUN
        imported = subprocess.run(
            [sys.executable, str(case / "import_completed_trials.py"),
             "--source-run", str(ROOT / SOURCE_RUN), "--output-root", str(output),
             "--restart-trial", "10", "--device", args.device],
            cwd=case)
        if imported.returncode or args.prepare_only:
            return imported.returncode
        command = [sys.executable, str(case / "run.py"), "--output-root", str(output),
                   "--device", args.device, "--mode", "optimize", "--lattices", "hexagonal",
                   "--timeout-hours", "0"]
        if args.recover_lock:
            command.append("--recover-lock")
        return subprocess.run(command, cwd=case).returncode
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
