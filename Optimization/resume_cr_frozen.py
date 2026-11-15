"""Resume the existing Cr study with the exact sources recorded in its manifest."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
import uuid

from off_order_controller import POLICY, RETRY_EXIT_CODE
from run_metal_off_optuna30 import prepare

OPTIMIZATION = Path(__file__).resolve().parent
ROOT = OPTIMIZATION.parent


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def content_hash(data):
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def installed_torcwa():
    spec = importlib.util.find_spec("torcwa")
    if spec is None or spec.origin is None:
        raise ValueError("The saved torcwa package must be installed in this Python environment.")
    return Path(spec.origin).parent


def relative_path(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe source path in manifest: {name}")
    return Path(*path.parts)


def original_source(name, expected, revisions):
    relative = relative_path(name)
    path = ROOT / relative
    if path.is_file():
        data = path.read_bytes()
        if content_hash(data) == expected:
            return data, None
    git_root = Path(subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "--show-toplevel"], text=True).strip())
    git_path = (ROOT.relative_to(git_root) / relative).as_posix()
    if git_path not in revisions:
        revisions[git_path] = subprocess.check_output(
            ["git", "-C", str(git_root), "log", "--format=%H", "--", git_path],
            text=True).splitlines()
    for commit in revisions[git_path]:
        result = subprocess.run(["git", "-C", str(git_root), "show", f"{commit}:{git_path}"],
                                capture_output=True)
        if result.returncode == 0 and content_hash(result.stdout) == expected:
            return result.stdout, commit
    raise ValueError(f"Cannot recover the saved source from local Git history: {name}")


def verify_snapshot(snapshot, saved):
    package = installed_torcwa()
    for name, expected in saved["source_sha256"].items():
        relative = relative_path(name)
        path = package / relative.name if relative.parts[0] == "installed_torcwa" else snapshot / relative
        if not path.is_file() or content_hash(path.read_bytes()) != expected:
            raise ValueError(f"Saved source hash does not match: {name}")
    csv = snapshot / "Optimization/cr_motheye/data/Johnson.csv"
    if content_hash(csv.read_bytes()) != saved["config"]["material_provenance"]["csv_sha256_lf"]:
        raise ValueError("Saved Cr dispersion does not match the snapshot.")


def prepare_snapshot(saved):
    expected_identity = {key: saved[key] for key in ("version", "config", "source_sha256", "requested_device")}
    if signature(expected_identity) != saved["signature"]:
        raise ValueError("The saved manifest signature is inconsistent.")
    package = installed_torcwa()
    files, recovered = {}, {}
    revisions = {}
    for name, expected in saved["source_sha256"].items():
        relative = relative_path(name)
        if relative.parts[0] == "installed_torcwa":
            if content_hash((package / relative.name).read_bytes()) != expected:
                raise ValueError(f"Installed package differs from the saved experiment: {name}")
            continue
        data, commit = original_source(name, expected, revisions)
        files[name] = data
        if commit is not None:
            recovered[name] = commit
    csv = Path(saved["config"]["material_csv"])
    csv_bytes = csv.read_bytes()
    if content_hash(csv_bytes) != saved["config"]["material_provenance"]["csv_sha256_lf"]:
        raise ValueError(f"The original Cr dispersion file differs: {csv}")
    parent = OPTIMIZATION / "frozen_resume"
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / f"cr_{saved['signature'][:16]}"
    if not destination.exists():
        with tempfile.TemporaryDirectory(prefix=".cr-sources-", dir=parent) as temporary:
            stage = Path(temporary) / "sources"
            stage.mkdir()
            for name, data in files.items():
                path = stage / relative_path(name)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            for name in ("Optimization/__init__.py", "studies/__init__.py", "studies/shared/__init__.py"):
                path = stage / name
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, path)
            data_path = stage / "Optimization/cr_motheye/data/Johnson.csv"
            data_path.parent.mkdir(parents=True, exist_ok=True)
            data_path.write_bytes(csv_bytes)
            (stage / "source_recovery.json").write_text(
                json.dumps(dict(original_signature=saved["signature"], recovered_from_git=recovered),
                           indent=2) + "\n", encoding="utf-8")
            verify_snapshot(stage, saved)
            try:
                stage.rename(destination)
            except OSError:
                if not destination.is_dir():
                    raise
    verify_snapshot(destination, saved)
    print(f"Verified frozen source snapshot: {destination}", flush=True)
    for name, commit in recovered.items():
        print(f"Recovered {name} from {commit[:12]}", flush=True)
    return destination


def internal_check(request_path):
    request = read(request_path)
    sys.path.insert(0, request["snapshot"])
    dependency = os.environ.get("RCWA_OPT_DEPENDENCY_PATH")
    if dependency:
        sys.path.append(dependency)
    from Optimization.cr_motheye.common import load_config, sources
    identity = dict(version=1, config=load_config(Path(request["config"])),
                    source_sha256=sources(), requested_device=request["device"])
    manifest = Path(request["output"]) / "manifest.json"
    if manifest.exists():
        saved = read(manifest)
        differences = [key for key in identity if saved.get(key) != identity[key]]
        if differences:
            raise ValueError("Frozen resume still differs in: " + ", ".join(differences))
        if saved["signature"] != signature(identity):
            raise ValueError("Frozen resume signature differs.")
    print("Configuration, source hashes and device identity passed.", flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period-nm", type=float, default=150.)
    parser.add_argument("--max-order", type=int, default=40)
    parser.add_argument("--recover-lock", action="store_true",
                        help="Recover the run lock after its previous controller and workers have stopped")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Recover frozen sources and check identity without running optimization")
    parser.add_argument("--internal-check", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.internal_check:
            return internal_check(args.internal_check)
        if not math.isfinite(args.period_nm) or args.period_nm <= 0 or args.max_order < 8:
            parser.error("A positive period and max-order >= 8 are required.")
        _, _, initial_output = prepare("Cr", period_nm=args.period_nm, height_nm=500.,
                                       grid=512, order=8, run_tag="hex_OFF_optuna30_auto_v1")
        saved = read(initial_output / "manifest.json")
        snapshot = prepare_snapshot(saved)
        device = saved["requested_device"]
        env = os.environ.copy()
        if env.get("MOTHEYE_DEPENDENCY_PATH") and not env.get("RCWA_OPT_DEPENDENCY_PATH"):
            env["RCWA_OPT_DEPENDENCY_PATH"] = env["MOTHEYE_DEPENDENCY_PATH"]
        for order in range(8, args.max_order + 1, 2):
            _, config, output = prepare("Cr", period_nm=args.period_nm, height_nm=500.,
                                        grid=512, order=order, run_tag="hex_OFF_optuna30_auto_v1")
            with tempfile.TemporaryDirectory(prefix=".cr-check-", dir=snapshot.parent) as temporary:
                request = Path(temporary) / "request.json"
                request.write_text(json.dumps(dict(snapshot=str(snapshot), config=str(config),
                                                   output=str(output), device=device)), encoding="utf-8")
                checked = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                          "--internal-check", str(request)], env=env)
            if checked.returncode or args.prepare_only:
                return checked.returncode
            attempt = uuid.uuid4().hex
            command = [sys.executable, str(OPTIMIZATION / "off_order_controller.py"),
                       "--case-dir", str(snapshot / "Optimization/cr_motheye"),
                       "--attempt-id", attempt, "--", "--config", str(config),
                       "--device", device, "--mode", "optimize", "--lattices", "hexagonal",
                       "--timeout-hours", "0", "--output-root", str(output)]
            if args.recover_lock:
                command.append("--recover-lock")
            print(f"Cr frozen resume: P={args.period_nm:g}, H=500, M={order}, Q=512; {output}", flush=True)
            code = subprocess.run(command, env=env).returncode
            if code < 0:
                return 128 - code
            if code != RETRY_EXIT_CODE:
                return code
            marker = read(output / "order_retry.json")
            if (marker.get("policy") != POLICY or marker.get("attempt_id") != attempt
                    or marker.get("status") != "retry_higher_order" or marker.get("order") != order
                    or marker.get("grid") != 512 or marker.get("normal_mode") != "off"):
                raise ValueError("Unconfirmed order-retry result; stopped.")
            print(f"Confirmed nonphysical result at M={order}; requesting the next order.", flush=True)
        print(f"Maximum order {args.max_order} reached.", file=sys.stderr)
        return RETRY_EXIT_CODE
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr, flush=True)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
