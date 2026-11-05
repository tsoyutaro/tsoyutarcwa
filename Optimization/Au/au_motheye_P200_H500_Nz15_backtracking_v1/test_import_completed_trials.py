"""Import integrity, queued trial numbering and safe repeated preparation."""
from pathlib import Path
import copy
import json
import os
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if os.environ.get("MOTHEYE_DEPENDENCY_PATH"):
    sys.path.append(os.environ["MOTHEYE_DEPENDENCY_PATH"])

import optuna
import import_completed_trials as migration
from optimizer.geometry import initial_logits, radii_values


class ImportTests(unittest.TestCase):
    def setUp(self):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.source = self.folder / "legacy"
        self.target = self.folder / "recovered"
        config = migration.load_config(HERE / "config.json")
        self.config = config
        identity = dict(version=1, config=config,
                        source_sha256=migration.execution_sources(migration.sources()), requested_device="cuda:0")
        migration.write_json(self.source / "manifest.json", dict(signature=migration.signature_for(identity), **identity))
        (self.source / "hexagonal").mkdir()
        self.storage, pruner = migration.storage_and_pruner(self.source / "hexagonal", config)
        study = optuna.create_study(study_name=migration.STUDY, storage=self.storage, pruner=pruner, direction="minimize")
        self.distributions = {
            "lr": optuna.distributions.FloatDistribution(.01, 1., log=True),
            "init_top": optuna.distributions.FloatDistribution(1., 99.),
            "delta_ratio": optuna.distributions.FloatDistribution(0., 1.),
        }
        for number in range(6):
            params = dict(lr=.03, init_top=8. + number, delta_ratio=.8)
            if number == 5:
                study.add_trial(optuna.trial.create_trial(
                    state=optuna.trial.TrialState.FAIL, params=params, distributions=self.distributions,
                    user_attrs=dict(error="Invalid shared power flux.")))
                self.failed_params = params
                continue
            logits = initial_logits(params["init_top"], 90., config)
            payload = dict(status="completed", trial_number=number, params=params,
                           best_loss=.3 - .02 * number, best_logits=logits, best_step=10,
                           completed_steps=111, radii_nm=radii_values(logits, config))
            study.add_trial(optuna.trial.create_trial(
                state=optuna.trial.TrialState.COMPLETE, value=payload["best_loss"],
                params=params, distributions=self.distributions,
                user_attrs={key: payload[key] for key in ("best_logits", "radii_nm", "best_step", "completed_steps")},
                intermediate_values={0: .4, 10: payload["best_loss"]}))
            migration.write_json(self.source / "hexagonal/trials" / f"trial_{number:04d}" / "best.json", payload)
        sampler = dict(optuna_version=optuna.__version__, sampler=optuna.samplers.TPESampler(seed=42, multivariate=True))
        with (self.source / "hexagonal/sampler.pkl").open("wb") as handle:
            pickle.dump(sampler, handle)

    def target_trials(self):
        storage, pruner = migration.storage_and_pruner(self.target / "hexagonal", self.config)
        return optuna.load_study(study_name=migration.STUDY, storage=storage, pruner=pruner).get_trials()

    def test_completed_values_shapes_and_sampler_are_preserved(self):
        journal = self.source / "hexagonal/optuna_journal.log"
        before = migration.checksum(journal)
        migration.prepare_import(self.source, self.target, "cpu")
        trials = self.target_trials()
        self.assertEqual([t.number for t in trials], list(range(6)))
        self.assertEqual([t.state.name for t in trials], ["COMPLETE"] * 5 + ["WAITING"])
        self.assertEqual([t.value for t in trials[:5]], [.3 - .02 * n for n in range(5)])
        self.assertEqual(migration.checksum(journal), before)
        self.assertEqual(migration.checksum(self.source / "hexagonal/sampler.pkl"),
                         migration.checksum(self.target / "hexagonal/sampler.pkl"))
        for number, trial in enumerate(trials[:5]):
            source = migration.read_json(self.source / "hexagonal/trials" / f"trial_{number:04d}" / "best.json")
            target = migration.read_json(self.target / "hexagonal/trials" / f"trial_{number:04d}" / "best.json")
            self.assertEqual(source["best_logits"], target["best_logits"])
            self.assertEqual(trial.user_attrs["origin_policy"], "legacy-adam")
        storage, pruner = migration.storage_and_pruner(self.target / "hexagonal", self.config)
        study = optuna.load_study(study_name=migration.STUDY, storage=storage, pruner=pruner)
        trial = study.ask()
        self.assertEqual(trial.number, 5)
        params = dict(lr=trial.suggest_float("lr", .01, 1., log=True),
                      init_top=trial.suggest_float("init_top", 1., 99.),
                      delta_ratio=trial.suggest_float("delta_ratio", 0., 1.))
        self.assertEqual(params, self.failed_params)

    def test_repeated_import_is_idempotent(self):
        migration.prepare_import(self.source, self.target, "cpu")
        journal = self.target / "hexagonal/optuna_journal.log"
        before = migration.checksum(journal)
        migration.prepare_import(self.source, self.target, "cpu")
        self.assertEqual(migration.checksum(journal), before)
        self.assertEqual(len(self.target_trials()), 6)

    def test_existing_unrelated_output_is_preserved(self):
        self.target.mkdir()
        marker = self.target / "keep.txt"
        marker.write_text("existing data", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "already exists"):
            migration.prepare_import(self.source, self.target, "cpu")
        self.assertEqual(marker.read_text(encoding="utf-8"), "existing data")

    def test_partial_preparation_does_not_publish_incomplete_output(self):
        with patch.object(migration, "export_trials", side_effect=RuntimeError("injected interrupted preparation")):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                migration.prepare_import(self.source, self.target, "cpu")
        self.assertFalse(self.target.exists())
        self.assertFalse(self.target.with_name(self.target.name + ".import.lock").exists())
        migration.prepare_import(self.source, self.target, "cpu")
        self.assertEqual(len(self.target_trials()), 6)

    def test_training_condition_change_is_rejected(self):
        path = self.source / "manifest.json"
        manifest = migration.read_json(path)
        manifest["config"]["height_nm"] = 400.
        identity = {key: manifest[key] for key in ("version", "config", "source_sha256", "requested_device")}
        manifest["signature"] = migration.signature_for(identity)
        migration.write_json(path, manifest)
        with self.assertRaisesRegex(ValueError, "Training conditions differ"):
            migration.prepare_import(self.source, self.target, "cpu")
        self.assertFalse(self.target.exists())

    def test_sampler_version_mismatch_is_rejected(self):
        path = self.source / "hexagonal/sampler.pkl"
        with path.open("rb") as handle:
            sampler = pickle.load(handle)
        sampler["optuna_version"] = "different-version"
        with path.open("wb") as handle:
            pickle.dump(sampler, handle)
        with self.assertRaisesRegex(ValueError, "Source sampler uses Optuna"):
            migration.prepare_import(self.source, self.target, "cpu")
        self.assertFalse(self.target.exists())

    def test_controller_starts_at_five_then_advances_to_six(self):
        import torch
        from optimizer import driver, engine
        migration.prepare_import(self.source, self.target, "cpu")
        observed = []
        evaluations = 0
        source_before = migration.checksum(self.source / "hexagonal/optuna_journal.log")

        def evaluate(logits, *args, **kwargs):
            nonlocal evaluations
            evaluations += 1
            if evaluations > 1:
                raise engine.NonphysicalResult("injected failure while retrying trial 5")
            return dict(objective=.2 + .01 * float(logits.detach().square().sum()),
                        gradient=.02 * logits.detach(), seconds=.01)

        def launch(folder, request):
            if request["mode"] == "preflight":
                return dict(status="ready", runtime_environment=driver.runtime_environment(torch.device("cpu")))
            observed.append((request["trial_number"], request["params"]))
            if request["trial_number"] == 6:
                raise KeyboardInterrupt()
            self.assertEqual(request["trial_number"], 5)
            storage, pruner = driver.storage_and_pruner(Path(request["study_folder"]), request["config"])
            study = optuna.load_study(study_name=migration.STUDY, storage=storage, pruner=pruner)
            trial = optuna.trial.Trial(study, request["trial_id"])
            return engine.train(request, torch.device("cpu"), trial=trial, evaluator=evaluate)

        arguments = ["--config", str(HERE / "config.json"), "--output-root", str(self.target),
                     "--device", "cpu", "--mode", "optimize", "--lattices", "hexagonal", "--timeout-hours", "0"]
        with patch.object(driver, "launch", side_effect=launch):
            self.assertEqual(driver.main(arguments), 130)
        self.assertEqual([number for number, _ in observed], [5, 6])
        self.assertEqual(observed[0][1], self.failed_params)
        trials = self.target_trials()
        self.assertEqual([trial.state.name for trial in trials], ["COMPLETE"] * 6 + ["RUNNING"])
        self.assertEqual(trials[5].user_attrs["termination_reason"], "nonphysical_retry_exhausted")
        self.assertEqual([trial.value for trial in trials[:5]], [.3 - .02 * n for n in range(5)])
        self.assertEqual(migration.checksum(self.source / "hexagonal/optuna_journal.log"), source_before)
        target_journal = self.target / "hexagonal/optuna_journal.log"
        target_before = migration.checksum(target_journal)
        migration.prepare_import(self.source, self.target, "cpu")
        self.assertEqual(migration.checksum(target_journal), target_before)


if __name__ == "__main__":
    unittest.main()
