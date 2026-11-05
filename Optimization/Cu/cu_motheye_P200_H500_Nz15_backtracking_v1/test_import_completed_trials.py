"""Preserve Cu's COMPLETE/PRUNED history and start recovery at trial 15."""
from pathlib import Path
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
import torch
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
        self.config = migration.load_config(HERE / "config.json")
        identity = dict(version=1, config=self.config,
                        source_sha256=migration.execution_sources(migration.sources()), requested_device="cuda:0")
        self.signature = migration.signature_for(identity)
        migration.write_json(self.source / "manifest.json", dict(signature=self.signature, **identity))
        (self.source / "hexagonal").mkdir()
        storage, pruner = migration.storage_and_pruner(self.source / "hexagonal", self.config)
        study = optuna.create_study(study_name=migration.STUDY, storage=storage, pruner=pruner, direction="minimize")
        self.distributions = {
            "lr": optuna.distributions.FloatDistribution(.01, 1., log=True),
            "init_top": optuna.distributions.FloatDistribution(1., 99.),
            "delta_ratio": optuna.distributions.FloatDistribution(0., 1.),
        }
        self.values = []
        for number in range(16):
            params = dict(lr=.03, init_top=8. + number, delta_ratio=.8)
            if number == 15:
                self.failed_params = dict(lr=.06437390180032174,
                    init_top=43.114360071905416, delta_ratio=.3500107737300064)
                study.add_trial(optuna.trial.create_trial(
                    state=optuna.trial.TrialState.FAIL, params=self.failed_params,
                    distributions=self.distributions, user_attrs=dict(error="Invalid shared power flux.")))
                continue
            logits = initial_logits(params["init_top"], 90., self.config)
            value = .05 if number == 7 else .3 - .01 * number
            pruned = number in (12, 13)
            if pruned:
                value = .001 * (number - 11)
            self.values.append(value)
            payload = dict(status="pruned" if pruned else "completed", trial_number=number, params=params,
                           best_loss=value, best_logits=logits, best_step=10,
                           completed_steps=111, radii_nm=radii_values(logits, self.config))
            attributes = {} if pruned else {key: payload[key] for key in
                ("best_logits", "radii_nm", "best_step", "completed_steps")}
            state = optuna.trial.TrialState.PRUNED if pruned else optuna.trial.TrialState.COMPLETE
            study.add_trial(optuna.trial.create_trial(
                state=state, value=value, params=params, distributions=self.distributions,
                user_attrs=attributes, intermediate_values={0: .4, 10: value}))
            folder = self.source / "hexagonal/trials" / f"trial_{number:04d}"
            if pruned:
                migration.write_json(folder / "outcome.json", dict(status="pruned", best_loss=value, next_step=111))
                torch.save(dict(signature=self.signature, params=params, best_loss=value,
                    best_logits=logits, best_step=10, next_step=111), folder / "adam_checkpoint.pt")
            else:
                migration.write_json(folder / "best.json", payload)
        sampler = dict(optuna_version=optuna.__version__, sampler=optuna.samplers.TPESampler(seed=42, multivariate=True))
        with (self.source / "hexagonal/sampler.pkl").open("wb") as handle:
            pickle.dump(sampler, handle)

    def target_trials(self):
        storage, pruner = migration.storage_and_pruner(self.target / "hexagonal", self.config)
        return optuna.load_study(study_name=migration.STUDY, storage=storage, pruner=pruner).get_trials()

    def test_values_pruning_shapes_and_sampler_are_preserved(self):
        journal = self.source / "hexagonal/optuna_journal.log"
        before = migration.checksum(journal)
        migration.prepare_import(self.source, self.target, "cpu")
        trials = self.target_trials()
        self.assertEqual([trial.number for trial in trials], list(range(16)))
        expected_states = ["PRUNED" if n in (12, 13) else "COMPLETE" for n in range(15)] + ["WAITING"]
        self.assertEqual([trial.state.name for trial in trials], expected_states)
        self.assertEqual([trial.value for trial in trials[:15]], self.values)
        self.assertEqual(migration.checksum(journal), before)
        self.assertEqual(migration.checksum(self.source / "hexagonal/sampler.pkl"),
                         migration.checksum(self.target / "hexagonal/sampler.pkl"))
        for number, trial in enumerate(trials[:15]):
            best = migration.read_json(self.target / "hexagonal/trials" / f"trial_{number:04d}" / "best.json")
            self.assertEqual(trial.user_attrs["best_logits"], best["best_logits"])
            self.assertEqual(trial.intermediate_values, {0: .4, 10: self.values[number]})
        # A smaller PRUNED value must not silently become a completed optimum.
        group = migration.read_json(self.target / "hexagonal/report.json")
        self.assertEqual(group["best"]["trial_number"], 7)
        self.assertEqual(group["best"]["objective"], .05)
        storage, pruner = migration.storage_and_pruner(self.target / "hexagonal", self.config)
        study = optuna.load_study(study_name=migration.STUDY, storage=storage, pruner=pruner)
        trial = study.ask()
        self.assertEqual(trial.number, 15)
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
        self.assertEqual(len(self.target_trials()), 16)

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
        self.assertEqual(len(self.target_trials()), 16)

    def test_pruned_checkpoint_mismatch_is_rejected(self):
        path = self.source / "hexagonal/trials/trial_0012/adam_checkpoint.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        checkpoint["best_loss"] = .8
        torch.save(checkpoint, path)
        with self.assertRaisesRegex(ValueError, "best shape/value"):
            migration.prepare_import(self.source, self.target, "cpu")
        self.assertFalse(self.target.exists())

    def test_training_condition_change_is_rejected(self):
        path = self.source / "manifest.json"
        manifest = migration.read_json(path)
        manifest["config"]["height_nm"] = 400.
        identity = {key: manifest[key] for key in ("version", "config", "source_sha256", "requested_device")}
        manifest["signature"] = migration.signature_for(identity)
        migration.write_json(path, manifest)
        with self.assertRaisesRegex(ValueError, "Training conditions differ"):
            migration.prepare_import(self.source, self.target, "cpu")

    def test_sampler_version_mismatch_is_rejected(self):
        path = self.source / "hexagonal/sampler.pkl"
        with path.open("rb") as handle:
            sampler = pickle.load(handle)
        sampler["optuna_version"] = "different-version"
        with path.open("wb") as handle:
            pickle.dump(sampler, handle)
        with self.assertRaisesRegex(ValueError, "Source sampler uses Optuna"):
            migration.prepare_import(self.source, self.target, "cpu")

    def test_controller_starts_at_fifteen_then_advances_to_sixteen(self):
        from optimizer import driver, engine
        migration.prepare_import(self.source, self.target, "cpu")
        observed = []
        evaluations = 0
        source_before = migration.checksum(self.source / "hexagonal/optuna_journal.log")

        def evaluate(logits, *args, **kwargs):
            nonlocal evaluations
            evaluations += 1
            if evaluations > 1:
                raise engine.NonphysicalResult("injected failure while retrying trial 15")
            return dict(objective=.2 + .01 * float(logits.detach().square().sum()),
                        gradient=.02 * logits.detach(), seconds=.01)

        def launch(folder, request):
            if request["mode"] == "preflight":
                return dict(status="ready", runtime_environment=driver.runtime_environment(torch.device("cpu")))
            observed.append((request["trial_number"], request["params"]))
            if request["trial_number"] == 16:
                raise KeyboardInterrupt()
            self.assertEqual(request["trial_number"], 15)
            storage, pruner = driver.storage_and_pruner(Path(request["study_folder"]), request["config"])
            study = optuna.load_study(study_name=migration.STUDY, storage=storage, pruner=pruner)
            trial = optuna.trial.Trial(study, request["trial_id"])
            return engine.train(request, torch.device("cpu"), trial=trial, evaluator=evaluate)

        arguments = ["--config", str(HERE / "config.json"), "--output-root", str(self.target),
                     "--device", "cpu", "--mode", "optimize", "--lattices", "hexagonal", "--timeout-hours", "0"]
        with patch.object(driver, "launch", side_effect=launch):
            self.assertEqual(driver.main(arguments), 130)
        self.assertEqual([number for number, _ in observed], [15, 16])
        self.assertEqual(observed[0][1], self.failed_params)
        trials = self.target_trials()
        self.assertEqual(trials[15].state.name, "COMPLETE")
        self.assertEqual(trials[16].state.name, "RUNNING")
        self.assertEqual(trials[15].user_attrs["termination_reason"], "nonphysical_retry_exhausted")
        self.assertEqual([trial.value for trial in trials[:15]], self.values)
        self.assertEqual(migration.checksum(self.source / "hexagonal/optuna_journal.log"), source_before)
        target_before = migration.checksum(self.target / "hexagonal/optuna_journal.log")
        migration.prepare_import(self.source, self.target, "cpu")
        self.assertEqual(migration.checksum(self.target / "hexagonal/optuna_journal.log"), target_before)


if __name__ == "__main__":
    unittest.main()
