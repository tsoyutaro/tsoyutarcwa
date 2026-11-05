"""Seed the recovery study with completed M20 trials and queue legacy trial 5."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import shutil
import sys
import uuid

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if os.environ.get("MOTHEYE_DEPENDENCY_PATH"):
    sys.path.append(os.environ["MOTHEYE_DEPENDENCY_PATH"])

import optuna
from optimizer.common import load_config, signature_for, sources, write_csv, write_json
from optimizer.driver import export_trials, storage_and_pruner
from optimizer.geometry import profile_rows, radii_values
from runtime_policy import execution_sources, reconcile_manifest

POLICY = "import-five-completed-and-retry-trial-five-v1"
STUDY = "motheye_nv_hexagonal"
CONTROLLER_SOURCES = {
    "optimizer/engine.py", "optimizer/common.py", "optimizer/driver.py",
    "run.py", "runtime_policy.py",
}
TRAINING_SETTINGS = (
    "period_nm", "height_nm", "layers", "radius_min_nm", "radius_max_nm",
    "softmax_gap_floor", "normal_mode", "lattices", "optimization_order",
    "optimization_grid", "optimization_wavelengths_nm", "objective_weighting",
    "material", "material_model", "substrate_mode", "passivity_tolerance",
    "checkpoint_layers", "adam_steps", "patience", "min_improvement",
    "learning_rate_min", "learning_rate_max", "init_top_min_nm", "init_top_max_nm",
    "scheduler_patience", "scheduler_factor", "logit_limit", "seed",
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_source(source, config):
    if (source / "controller.lock").exists():
        raise ValueError("The source M20 experiment is locked; finish that job before importing.")
    manifest_path = source / "manifest.json"
    manifest = read_json(manifest_path)
    identity = {key: manifest[key] for key in ("version", "config", "source_sha256", "requested_device")}
    if signature_for(identity) != manifest["signature"]:
        raise ValueError("The source experiment manifest/signature is inconsistent.")
    old_config = manifest["config"]
    differences = [key for key in TRAINING_SETTINGS if old_config.get(key) != config.get(key)]
    if differences:
        raise ValueError("Training conditions differ from the completed source trials: " + ", ".join(differences))
    old_sources = manifest.get("active_source_sha256", manifest["source_sha256"])
    current_sources = execution_sources(sources())
    changed_model = [key for key in sorted(set(old_sources) | set(current_sources))
                     if key not in CONTROLLER_SOURCES and old_sources.get(key) != current_sources.get(key)]
    if changed_model:
        raise ValueError("Physics, material data or installed torcwa sources differ: " + ", ".join(changed_model))
    old_material = old_config.get("material_provenance", {}).get("csv_sha256_lf")
    if old_material != config.get("material_provenance", {}).get("csv_sha256_lf"):
        raise ValueError("The source material-data checksum differs.")

    journal = source / "hexagonal/optuna_journal.log"
    sampler_path = source / "hexagonal/sampler.pkl"
    if not journal.is_file() or not sampler_path.is_file():
        raise ValueError("The source journal or sampler checkpoint is missing.")
    checksums = {str(path.relative_to(source)): checksum(path)
                 for path in (manifest_path, journal, sampler_path)}
    storage, pruner = storage_and_pruner(source / "hexagonal", old_config)
    study = optuna.load_study(study_name=STUDY, storage=storage, pruner=pruner)
    trials = study.get_trials(deepcopy=True)
    if len(trials) < 6 or any(trials[n].number != n or trials[n].state.name != "COMPLETE" for n in range(5)):
        raise ValueError("Source trials 0 through 4 must all be COMPLETE.")
    failed = trials[5]
    if failed.number != 5 or failed.state.name != "FAIL" or failed.user_attrs.get("error") != "Invalid shared power flux.":
        raise ValueError("Source trial 5 must be the recorded nonphysical-power failure.")
    expected_distributions = {
        "lr": optuna.distributions.FloatDistribution(config["learning_rate_min"], config["learning_rate_max"], log=True),
        "init_top": optuna.distributions.FloatDistribution(config["init_top_min_nm"], config["init_top_max_nm"]),
        "delta_ratio": optuna.distributions.FloatDistribution(0., 1.),
    }
    for trial in trials[:6]:
        if trial.distributions != expected_distributions:
            raise ValueError(f"Source trial {trial.number} uses different parameter distributions.")

    best_payloads = []
    for trial in trials[:5]:
        path = source / "hexagonal/trials" / f"trial_{trial.number:04d}" / "best.json"
        payload = read_json(path)
        logits = payload.get("best_logits")
        if (payload.get("status") != "completed" or payload.get("params") != trial.params
                or not isinstance(logits, list) or len(logits) != config["layers"] + 1
                or not all(math.isfinite(value) for value in logits)
                or logits != trial.user_attrs.get("best_logits")
                or not math.isfinite(trial.value)
                or abs(payload["best_loss"] - trial.value) > 1e-12):
            raise ValueError(f"Source best shape/value for trial {trial.number} is inconsistent.")
        radii = radii_values(logits, config)
        if any(not config["radius_min_nm"] < radius < config["radius_max_nm"] for radius in radii):
            raise ValueError("An imported radius violates the target geometry bounds.")
        best_payloads.append(payload)
        checksums[str(path.relative_to(source))] = checksum(path)

    # The sampler is an existing, trusted checkpoint from the user's own run.
    with sampler_path.open("rb") as handle:
        sampler = pickle.load(handle)
    if sampler.get("optuna_version") != optuna.__version__:
        raise ValueError(
            f"Source sampler uses Optuna {sampler.get('optuna_version')}; "
            f"current environment uses {optuna.__version__}. "
            "Run the import and recovery study with the source Optuna version.")
    if not isinstance(sampler.get("sampler"), optuna.samplers.TPESampler):
        raise ValueError("The source sampler must be TPESampler.")
    for relative, expected in checksums.items():
        if checksum(source / relative) != expected:
            raise ValueError("The source changed during import preparation; retry after its job has stopped.")
    return manifest, trials[:5], failed, best_payloads, checksums


def prepare_import(source, output, device):
    source = source.resolve()
    output = output.resolve()
    if source == output:
        raise ValueError("Choose a separate target directory for the recovery study.")
    config = load_config(HERE / "config.json")
    manifest, completed, failed, payloads, source_checksums = read_source(source, config)
    identity = dict(version=1, config=config, source_sha256=execution_sources(sources()), requested_device=device)
    target_manifest, _ = reconcile_manifest(None, identity)
    import_identity = dict(policy=POLICY, source_run=str(source), source_checksums=source_checksums,
                           target_signature=target_manifest["signature"], completed_trials=list(range(5)), retry_trial=5)
    marker = output / "import_completed_trials.json"
    if output.exists():
        if marker.is_file() and read_json(marker).get("identity") == import_identity:
            print(f"Completed trials 0-4 have already been imported into {output}.")
            return output
        raise ValueError("The target already exists with different data; choose a new --output-root.")

    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_name(output.name + ".import.lock")
    lock.mkdir()
    stage = output.parent / (".seed-" + uuid.uuid4().hex[:12])
    try:
        if output.exists():
            raise ValueError("The target was created by another process; import was stopped.")
        stage.mkdir()
        folder = stage / "hexagonal"
        folder.mkdir()
        storage, pruner = storage_and_pruner(folder, config)
        study = optuna.create_study(study_name=STUDY, direction="minimize", storage=storage, pruner=pruner)
        for old_trial, old_payload in zip(completed, payloads):
            attributes = copy.deepcopy(old_trial.user_attrs)
            attributes.update(
                termination_reason="imported_completed_trial", numerical_early_stop=False,
                recovery_attempts=0, origin_policy="legacy-adam", origin_trial_number=old_trial.number,
                origin_run_root=str(source), origin_signature=manifest["signature"])
            study.add_trial(optuna.trial.create_trial(
                state=optuna.trial.TrialState.COMPLETE, value=old_trial.value,
                params=old_trial.params, distributions=old_trial.distributions,
                user_attrs=attributes, intermediate_values=old_trial.intermediate_values))
            destination = folder / "trials" / f"trial_{old_trial.number:04d}"
            destination.mkdir(parents=True)
            payload = dict(old_payload, imported_from=str(source), origin_policy="legacy-adam")
            write_json(destination / "best.json", payload)
            write_json(destination / "outcome.json", payload)
            write_csv(destination / "best_profile.csv", profile_rows(payload["best_logits"], config))

        study.enqueue_trial(failed.params, user_attrs=dict(
            origin_run_root=str(source), origin_trial_number=5,
            restart_reason="retry_legacy_nonphysical_trial_with_backtracking"))
        queued = study.get_trials(deepcopy=True)[5]
        if queued.number != 5 or queued.state.name != "WAITING":
            raise RuntimeError("The original trial 5 was not queued at the expected trial number.")
        shutil.copyfile(source / "hexagonal/sampler.pkl", folder / "sampler.pkl")
        export_trials(folder, study)
        best = study.best_trial
        write_json(folder / "report.json", dict(lattice="hexagonal", status="imported_completed_trials", best=dict(
            trial_number=best.number, objective=best.value, params=best.params,
            logits=best.user_attrs["best_logits"], radii_nm=radii_values(best.user_attrs["best_logits"], config),
            trial_folder=str(output / "hexagonal/trials" / f"trial_{best.number:04d}"))))
        write_json(stage / "manifest.json", target_manifest)
        write_json(stage / "resolved_config.json", config)
        write_json(stage / "workflow.json", dict(status="prepared", signature=target_manifest["signature"],
            requested_lattices=["hexagonal"], reports=dict(hexagonal=str(output / "hexagonal/report.json"))))
        write_json(stage / "import_completed_trials.json", dict(
            identity=import_identity, utc=datetime.now(timezone.utc).isoformat(),
            imported_best_objective=best.value, queued_trial5_params=failed.params,
            trial5_adam_start="step 0 from the original initial shape",
            target_policy="validated-adam-backtracking-v1"))
        for relative, expected in source_checksums.items():
            if checksum(source / relative) != expected:
                raise ValueError("The source changed while preparing the imported study.")
        if output.exists():
            raise ValueError("The target was created by another process; the staged data is preserved.")
        stage.rename(output)
    finally:
        lock.rmdir()
    print(f"Imported COMPLETE trials 0-4. Queued trial 5 with params {failed.params}.")
    print(f"Best imported mean R={best.value:.12g}; target={output}")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=(
        HERE.parent / "au_motheye_P200_H500_Nz15/results/hex_OFF_optuna30_auto_v1_M20_Q512"))
    parser.add_argument("--output-root", type=Path, default=(
        HERE / "results/hex_OFF_optuna30_backtracking_from5_v1_M20_Q512"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    try:
        prepare_import(args.source_run, args.output_root, args.device)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
