"""Launcher regressions. All experiments live in temporary directories; no RCWA runs."""
import copy
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_metal_period_backtracking as runner


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get("MOTHEYE_TEST_TMPDIR"))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.al_base = runner.read(runner.ROOT / runner.AL_SOURCE / "config.json")
        self.cu_base = runner.read(runner.ROOT / runner.CU_CASE / "config.json")
        self.au_base = runner.read(runner.ROOT / runner.AU_CASE / "config.json")
        for relative, config in ((runner.AL_SOURCE, self.al_base), (runner.CU_CASE, self.cu_base),
                                 (runner.AU_CASE, self.au_base)):
            case = self.root / relative
            for folder in ("data", "optimizer", "rcwa_ext"):
                (case / folder).mkdir(parents=True, exist_ok=True)
            for name in ("run.py", "d6_nvm.py", "rcwa_solver_auto.py", "io_utils.py", "runtime_policy.py"):
                (case / name).write_text("# isolated test fixture\n", encoding="utf-8")
            (case / "config.json").write_text(json.dumps(config), encoding="utf-8")
        for name in ("optimizer/common.py", "optimizer/driver.py", "optimizer/engine.py", "import_completed_trials.py"):
            (self.root / runner.CU_CASE / name).write_text("# recovery fixture\n", encoding="utf-8")

    def run_main(self, args, executor):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return runner.main(args, root=self.root, executor=executor)

    def plan(self, material="Cu", period=150, **kwargs):
        return runner.prepare(self.root, material, period_nm=period, **kwargs)

    def test_five_periods_have_separate_configs_and_correct_radius_bounds(self):
        outputs = set()
        for material, order in (("Al", 8), ("Cu", 14)):
            for period in (100, 150, 250, 300, 350):
                p = self.plan(material, period)
                self.assertEqual(p.config["optimization_order"], order)
                self.assertEqual(p.config["period_nm"], period)
                self.assertEqual(p.config["height_nm"], 500)
                self.assertEqual(p.config["normal_mode"], "off")
                self.assertEqual(p.config["nonphysical_lr_factor"], .5)
                self.assertAlmostEqual(p.config["radius_max_nm"], period / 2 - .001)
                self.assertLess(p.config["init_top_max_nm"], p.config["radius_max_nm"])
                outputs.add(p.output)
        self.assertEqual(len(outputs), 10)

    def test_existing_m14_al_results_are_not_reused_by_m8(self):
        old = self.plan("Al", 150, order=14)
        old.output.mkdir(parents=True)
        checkpoint = old.output / "adam_checkpoint.pt"
        checkpoint.write_bytes(b"keep M14 checkpoint")
        new = self.plan("Al", 150, order=8)
        self.assertNotEqual(old.case, new.case)
        self.assertNotEqual(old.output, new.output)
        self.assertEqual(checkpoint.read_bytes(), b"keep M14 checkpoint")

    def test_existing_config_and_checkpoint_are_preserved_on_resume(self):
        p = self.plan()
        cfg = runner.read(p.config_path)
        cfg["verification_orders"] = [14, 16, 20]
        p.config_path.write_text(json.dumps(cfg), encoding="utf-8")
        trial = p.output / "hexagonal/trials/trial_0015"
        trial.mkdir(parents=True)
        checkpoint = trial / "adam_checkpoint.pt"
        checkpoint.write_bytes(b"saved Adam moments and step 123")
        journal = p.output / "hexagonal/optuna_journal.log"
        journal.write_text("existing journal", encoding="utf-8")
        before_config = p.config_path.read_bytes()
        calls = []
        def execute(cmd, **kwargs):
            calls.append(cmd)
            self.assertEqual(Path(cmd[cmd.index("--output-root") + 1]), p.output)
            self.assertEqual(checkpoint.read_bytes(), b"saved Adam moments and step 123")
            return SimpleNamespace(returncode=0)
        self.assertEqual(self.run_main(["--materials", "Cu", "--period-nm", "150"], execute), 0)
        self.assertEqual(len(calls), 1)
        self.assertIn("--mode", calls[0])
        self.assertEqual(calls[0][calls[0].index("--mode") + 1], "optimize")
        self.assertEqual(before_config, p.config_path.read_bytes())
        self.assertEqual(journal.read_text(), "existing journal")

    def test_completed_condition_is_skipped_and_next_period_runs(self):
        p = self.plan(period=100)
        group = p.output / "hexagonal/report.json"
        group.parent.mkdir(parents=True)
        group.write_text(json.dumps({"status": "completed", "best": {"logits": [0.] * 16}}))
        (p.output / "workflow.json").write_text(json.dumps({"status": "completed"}))
        calls = []
        def execute(cmd, **kwargs):
            calls.append(cmd)
            return SimpleNamespace(returncode=0)
        self.assertEqual(self.run_main(["--materials", "Cu", "--periods", "100,150"], execute), 0)
        self.assertEqual(len(calls), 1)
        self.assertIn("P150_H500", " ".join(calls[0]))

    def test_recover_lock_refuses_a_live_local_controller(self):
        p = self.plan()
        lock = p.output / "controller.lock"
        lock.mkdir(parents=True)
        (lock / "owner.json").write_text(json.dumps({"host": socket.gethostname(), "pid": os.getpid()}))
        def execute(*a, **k):
            self.fail("Must not launch over a live controller")
        self.assertEqual(self.run_main(["--materials", "Cu", "--period-nm", "150", "--recover-lock"], execute), 2)
        self.assertTrue((lock / "owner.json").exists())

    def test_stale_lock_requires_explicit_recovery_and_flag_is_forwarded(self):
        p = self.plan()
        lock = p.output / "controller.lock"
        lock.mkdir(parents=True)
        (lock / "owner.json").write_text(json.dumps({"host": socket.gethostname(), "pid": 999999}))
        calls = []
        def execute(cmd, **kwargs):
            calls.append(cmd)
            return SimpleNamespace(returncode=0)
        with patch.object(runner, "pid_alive", return_value=False):
            self.assertEqual(self.run_main(["--materials", "Cu", "--period-nm", "150"], execute), 2)
            self.assertEqual(self.run_main(["--materials", "Cu", "--period-nm", "150", "--recover-lock"], execute), 0)
        self.assertEqual(len(calls), 1)
        self.assertIn("--recover-lock", calls[0])

    def test_configuration_change_is_rejected_without_rewriting(self):
        p = self.plan()
        config = runner.read(p.config_path)
        config["optimization_order"] = 12
        p.config_path.write_text(json.dumps(config))
        before = p.config_path.read_bytes()
        with self.assertRaisesRegex(ValueError, "optimization_order"):
            self.plan()
        self.assertEqual(before, p.config_path.read_bytes())

    def test_interrupt_releases_only_launcher_lock_and_same_command_can_resume(self):
        p = self.plan()
        trial = p.output / "hexagonal/trials/trial_0003"
        trial.mkdir(parents=True)
        saved = trial / "adam_checkpoint.pt"
        saved.write_bytes(b"step 200")
        def interrupted(*a, **k):
            raise KeyboardInterrupt()
        args = ["--materials", "Cu", "--period-nm", "150"]
        self.assertEqual(self.run_main(args, interrupted), 130)
        self.assertFalse(p.output.with_name(p.output.name + ".period_launcher.lock").exists())
        self.assertEqual(saved.read_bytes(), b"step 200")
        self.assertEqual(self.run_main(args, lambda *a, **k: SimpleNamespace(returncode=0)), 0)

    def test_legacy_al_p100_imported_study_remains_resume_target(self):
        p = self.plan("Al", 100)
        source = self.root / runner.AL_SOURCE / "geometry_cases/P100_H500_Nz15/results/hex_OFF_optuna30_auto_v1_M8_Q512"
        source.mkdir(parents=True)
        (source / "manifest.json").write_text(json.dumps({"config": p.config}))
        imported = p.case / "results/hex_OFF_optuna30_backtracking_from10_v1_M8_Q512"
        imported.mkdir(parents=True)
        (imported / "manifest.json").write_text("{}")
        resumed = self.plan("Al", 100)
        self.assertEqual(resumed.output, imported)
        self.assertIsNone(resumed.import_source)

    def test_legacy_al_import_is_scheduled_only_before_target_exists(self):
        source = self.root / runner.AL_SOURCE / "geometry_cases/P100_H500_Nz15/results/hex_OFF_optuna30_auto_v1_M8_Q512"
        source.mkdir(parents=True)
        config = copy.deepcopy(self.al_base)
        config.update(period_nm=100, radius_max_nm=49.999, baseline_tip_nm=2.5, baseline_base_nm=47.5,
                      init_top_max_nm=49.989)
        (source / "manifest.json").write_text(json.dumps({"config": config}))
        plan = self.plan("Al", 100)
        self.assertEqual(plan.import_source, source)
        calls = []
        def execute(cmd, **kwargs):
            calls.append(cmd)
            if cmd[1].endswith("import_completed_trials.py"):
                # The actual importer rejects even an empty pre-created output.
                self.assertFalse(plan.output.exists())
                plan.output.mkdir()
                (plan.output / "manifest.json").write_text("{}")
            return SimpleNamespace(returncode=0)
        self.assertEqual(self.run_main(["--materials", "Al", "--period-nm", "100"], execute), 0)
        self.assertEqual(len(calls), 2)
        self.assertTrue(calls[0][1].endswith("import_completed_trials.py"))
        self.assertIn("--restart-trial", calls[0])

    def test_existing_cu_p200_h500_imported_study_is_selected(self):
        case = self.root / runner.CU_CASE
        imported = case / "results/hex_OFF_optuna30_backtracking_from15_v1_M14_Q512"
        imported.mkdir(parents=True)
        (imported / "manifest.json").write_text("{}")
        p = self.plan("Cu", 200)
        self.assertEqual(p.output, imported)
        self.assertIsNone(p.import_source)

    def test_existing_cu_height_config_is_reused_without_changing_search_settings(self):
        p = self.plan("Cu", 200, height_nm=400)
        c = runner.read(p.config_path)
        c["learning_rate_max"] = .7
        p.config_path.write_text(json.dumps(c))
        old = p.config_path.read_bytes()
        resumed = self.plan("Cu", 200, height_nm=400)
        self.assertEqual(resumed.output, p.output)
        self.assertEqual(resumed.config["learning_rate_max"], .7)
        self.assertEqual(old, p.config_path.read_bytes())

    def test_recover_completed_condition_removes_stale_controller_lock(self):
        p = self.plan()
        group = p.output / "hexagonal/report.json"
        group.parent.mkdir(parents=True)
        group.write_text(json.dumps({"status": "completed", "best": {"logits": [0.] * 16}}))
        (p.output / "workflow.json").write_text(json.dumps({"status": "completed"}))
        lock = p.output / "controller.lock"
        lock.mkdir()
        (lock / "owner.json").write_text(json.dumps({"host": socket.gethostname(), "pid": 999999}))
        with patch.object(runner, "pid_alive", return_value=False):
            code = self.run_main(["--materials", "Cu", "--period-nm", "150", "--recover-lock"],
                                 lambda *a, **k: self.fail("Completed study must not run again"))
        self.assertEqual(code, 0)
        self.assertFalse(lock.exists())

    def test_prepare_only_never_starts_a_worker(self):
        self.assertEqual(self.run_main(["--materials", "Al,Cu", "--periods", "100,150", "--prepare-only"],
                                      lambda *a, **k: self.fail("Worker started")), 0)

    def test_custom_dimensions_and_order_use_separate_settings(self):
        p = self.plan("Al", 250, height_nm=400, layers=10, order=12, grid=1024, trials=4, adam_steps=20)
        for key, value in {"height_nm": 400, "layers": 10, "optimization_order": 12,
                           "optimization_grid": 1024, "trials": 4, "adam_steps": 20}.items():
            self.assertEqual(p.config[key], value)
        self.assertIn("P250_H400_Nz10", str(p.output))


    def test_au_five_heights_use_separate_m20_configs(self):
        outputs = set()
        for height in (100, 200, 300, 400, 500):
            p = self.plan("Au", 200, height_nm=height)
            self.assertEqual(p.config["material"], "Au")
            self.assertEqual(p.config["material_csv"], self.au_base["material_csv"])
            self.assertEqual(p.config["optimization_order"], 20)
            self.assertEqual(p.config["height_nm"], height)
            self.assertEqual(p.config["period_nm"], 200)
            self.assertEqual(p.config["adam_recovery_policy"], runner.POLICY)
            outputs.add(p.output)
        self.assertEqual(len(outputs), 5)

    def test_au_height_batch_skips_completed_established_h500_study(self):
        case = self.root / runner.AU_CASE
        imported = case / "results/hex_OFF_optuna30_backtracking_from5_v1_M20_Q512"
        (imported / "hexagonal").mkdir(parents=True)
        (imported / "manifest.json").write_text("{}")
        (imported / "workflow.json").write_text(json.dumps({"status": "completed"}))
        (imported / "hexagonal/report.json").write_text(json.dumps({"status": "completed", "best": {"logits": [0.] * 16}}))
        older = case / "results/hex_OFF_optuna30_backtracking_v1_M20_Q512"
        (older / "hexagonal").mkdir(parents=True)
        (older / "manifest.json").write_text("{}")
        (older / "workflow.json").write_text(json.dumps({"status": "interrupted"}))
        journal = older / "hexagonal/optuna_journal.log"
        journal.write_text("preserve the interrupted study")
        calls = []
        def execute(cmd, **kwargs):
            calls.append(cmd)
            return SimpleNamespace(returncode=0)
        self.assertEqual(self.run_main(["--materials", "Au", "--period-nm", "200",
                                       "--heights", "100,200,300,400,500"], execute), 0)
        self.assertEqual(len(calls), 4)
        for cmd, height in zip(calls, (100, 200, 300, 400)):
            self.assertIn(f"P200_H{height}_Nz15", " ".join(cmd))
        self.assertEqual(journal.read_text(), "preserve the interrupted study")
        self.assertEqual(self.plan("Au", 200).output, imported)

    def test_au_existing_height_config_and_state_are_preserved(self):
        p = self.plan("Au", 200, height_nm=300)
        trial = p.output / "hexagonal/trials/trial_0007"
        trial.mkdir(parents=True)
        checkpoint = trial / "adam_checkpoint.pt"
        checkpoint.write_bytes(b"Au: step 321, saved moments")
        config_before = p.config_path.read_bytes()
        code = self.run_main(["--materials", "Au", "--period-nm", "200", "--height-nm", "300", "--order", "20"],
                             lambda *a, **k: SimpleNamespace(returncode=0))
        self.assertEqual(code, 0)
        self.assertEqual(p.config_path.read_bytes(), config_before)
        self.assertEqual(checkpoint.read_bytes(), b"Au: step 321, saved moments")

    def test_invalid_height_in_batch_starts_no_condition(self):
        code = self.run_main(["--materials", "Au", "--period-nm", "200", "--heights", "100,nan"],
                             lambda *a, **k: self.fail("Invalid batch must not start"))
        self.assertEqual(code, 2)
        self.assertFalse((self.root / runner.AU_CASE / "geometry_cases").exists())


if __name__ == "__main__":
    unittest.main()
