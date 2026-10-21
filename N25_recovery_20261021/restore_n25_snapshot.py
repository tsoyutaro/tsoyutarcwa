"""Restore the Git-tracked pre-incident files into a NEW result directory."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import sys


DEFAULT_OUTPUT = "Optimization/dielectric_motheye_n25/results/NV_Optuna_Adam_M8_verifyM16_recovered_20261021_0909"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def content_digest(files):
    return digest(json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def source_digest(path):
    return digest(path.read_bytes().replace(b"\r\n", b"\n"))


def safe_relative(name):
    path = PurePosixPath(name)
    if (path.is_absolute() or not path.parts or any(part in ("..", ".") for part in path.parts)
            or "\\" in name or ":" in name):
        raise ValueError("Unsafe snapshot path: "+name)
    return Path(*path.parts)


def validate_snapshot(bundle):
    index = json.loads((bundle/"snapshot_index.json").read_text(encoding="utf-8"))
    if index.get("format_version") != 2 or index.get("storage") != "plain_files":
        raise ValueError("Expected the Git version of the recovery files (format 2).")
    if content_digest(index["files"]) != index["snapshot_content_sha256"]:
        raise ValueError("Snapshot inventory checksum differs.")
    snapshot = bundle/"snapshot"
    for name, expected in index["files"].items():
        path = snapshot/safe_relative(name)
        if path.is_symlink() or not path.is_file():
            raise ValueError("Missing or linked snapshot file: "+name)
        if digest(path.read_bytes()) != expected:
            raise ValueError("Snapshot file checksum differs: "+name)
    return index, snapshot


def validate_project(project, index):
    if not (project/"run_n25_nv_optuna_adam.py").is_file():
        raise ValueError("Project root must contain run_n25_nv_optuna_adam.py.")
    for relative, expected in index["expected_project_sources_sha256_lf"].items():
        path = project/safe_relative(relative)
        if not path.is_file() or source_digest(path) != expected:
            raise ValueError("Project source differs or is missing; update the project first: "+relative)


def restore(project, target, bundle):
    index, snapshot = validate_snapshot(bundle)
    project, target = project.resolve(), target.resolve()
    target.relative_to(project)
    if target.exists():
        raise ValueError("Destination already exists; no files were overwritten. Choose a NEW --output-root.")
    # These checks finish before any result directory is created. The recovery
    # tool never changes the existing results, project sources or journal.
    validate_project(project,index)
    target.mkdir(parents=True, exist_ok=False)
    for name, expected in index["files"].items():
        data = (snapshot/safe_relative(name)).read_bytes()
        if digest(data) != expected:
            raise ValueError("Snapshot changed while copying: "+name)
        destination = target/safe_relative(name)
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(data)
    record = {"status":"restored_not_started", "restored_utc":datetime.now(timezone.utc).isoformat(),
        "source_snapshot":index["snapshot"], "snapshot_content_sha256":index["snapshot_content_sha256"],
        "snapshot_storage":"Git-tracked plain files", "original_result_directory_modified":False,
        "project_sources_modified":False,
        "limits":"Restores the shared snapshot; later progress on the original GPU job is not included."}
    (target/"recovery_provenance.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
    command = "python run_n25_nv_optuna_adam.py --device cuda --output-root "+shlex.quote(str(target))+" --timeout-hours 0"
    print(json.dumps({"restored":str(target),"snapshot":index["snapshot"]},indent=2))
    print("After both old jobs have stopped, use ONE controller only:")
    print(command)
    return target


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root",type=Path,default=Path.cwd())
    p.add_argument("--output-root",type=Path,default=Path(DEFAULT_OUTPUT))
    p.add_argument("--inspect-only",action="store_true")
    args = p.parse_args(argv)
    bundle = Path(__file__).resolve().parent
    try:
        if args.inspect_only:
            index,_ = validate_snapshot(bundle)
            print(json.dumps(index["snapshot"],indent=2))
            return 0
        target = args.output_root if args.output_root.is_absolute() else args.project_root/args.output_root
        restore(args.project_root,target,bundle)
    except (ValueError,OSError,KeyError) as error:
        p.error(str(error))
    return 0


if __name__ == "__main__":
    sys.exit(main())
