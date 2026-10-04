"""Test saved-profile initialization, fresh Adam state and resume guards.

These tests check optimizer state flow; they do not run an optical solver.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studies.gold_motheye2 import optimize_adam_fullband as full

SOURCE = ROOT / "studies/gold_motheye2/results/adam_fullband_Nz100_M8"


class StopBeforeGradient(Exception):
    pass


class InitialProfileTests(unittest.TestCase):
    def setUp(self):
        self.saved = json.loads((SOURCE/"config.json").read_text())
        self.source_checkpoint = json.loads((SOURCE/"checkpoint.json").read_text())
        self.current = copy.deepcopy(self.saved)
        self.current.update(learning_rate=.01, wavelengths_nm=list(range(400,701,25)))

    def test_best_parameters_only_with_zero_moments(self):
        seed = full._load_initial_profile(SOURCE, self.current)
        checkpoint = full._initial_checkpoint("new-stage", seed["logits"])
        self.assertEqual(checkpoint["logits"], self.source_checkpoint["best"]["logits"])
        self.assertEqual(checkpoint["step"], 0)
        self.assertFalse(any(checkpoint["m"]))
        self.assertFalse(any(checkpoint["v"]))
        self.assertIsNone(checkpoint["best"])
        self.assertEqual(checkpoint["evaluations"], [])
        checkpoint["logits"][0] += .01
        self.assertEqual(seed["logits"], self.source_checkpoint["best"]["logits"])

    def test_rejects_incompatible_shape_and_material(self):
        for key, value in (("segments", self.current["segments"]+1),
                           ("gold_csv_sha256_lf", "changed-material"),
                           ("diameter_margin_nm", .2)):
            current = copy.deepcopy(self.current)
            current[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                full._load_initial_profile(SOURCE, current)

    def test_rejects_bad_source_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            (run/"config.json").write_text(json.dumps(self.saved))
            checkpoint = copy.deepcopy(self.source_checkpoint)
            checkpoint["signature"] = "wrong-config"
            (run/"checkpoint.json").write_text(json.dumps(checkpoint))
            with self.assertRaises(ValueError):
                full._load_initial_profile(run, self.current)

    def argv(self, output, prepare=False):
        args = ["optimize_adam_fullband.py", "--device", "cpu",
                "--initial-run-dir", str(SOURCE), "--output-dir", str(output),
                "--wavelengths", "400:700:25", "--learning-rate", ".01",
                "--steps", "12", "--verify-order", "0"]
        return args + (["--prepare-only"] if prepare else [])

    def test_protects_source_output(self):
        before = {name: hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()
                  for name in ("config.json", "checkpoint.json")}
        with patch.object(sys, "argv", self.argv(SOURCE, True)):
            with self.assertRaises(SystemExit) as stopped:
                full.main()
        self.assertEqual(stopped.exception.code, 2)
        self.assertEqual(before, {name: hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()
                                  for name in before})

    def test_initial_shape_is_evaluated_before_first_update_and_resume_preserves_state(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/"stage"
            seed_logits = self.source_checkpoint["best"]["logits"]
            labels = []
            def evaluate(target, checkpoint, label, order, logits, config, *unused):
                labels.append(label)
                # An evaluator double makes the state-flow requirement
                # independent of the source run's old numerical objective.
                value = .30 if label == "cone" else .21
                row = {"label": label, "order": order, "step": checkpoint["step"],
                       "logits": list(logits), "mean_reflectance": value,
                       "values": {f"{w:g}": value for w in config["wavelengths_nm"]}}
                checkpoint["evaluations"].append(row)
                return row
            def gradient(target, checkpoint, config, *unused):
                self.assertEqual(labels, ["cone", "candidate"])
                self.assertEqual(checkpoint["best"]["logits"], seed_logits)
                self.assertEqual(checkpoint["best"]["mean_reflectance"], .21)
                self.assertEqual(len(config["wavelengths_nm"]), 13)
                self.assertFalse(any(checkpoint["m"]))
                self.assertFalse(any(checkpoint["v"]))
                raise StopBeforeGradient
            torch_stub = SimpleNamespace(device=lambda name: SimpleNamespace(type=name))
            dispersion_stub = SimpleNamespace(build_gold_model=lambda *args: lambda wl: 4.)
            with (patch.object(sys, "argv", self.argv(output)),
                  patch.dict(sys.modules, {"torch": torch_stub,
                      "studies.shared.gold_dispersion": dispersion_stub}),
                  patch.object(full.shared, "evaluate", side_effect=evaluate),
                  patch.object(full.shared, "save_results"),
                  patch.object(full, "_full_gradient_step", side_effect=gradient)):
                with self.assertRaises(StopBeforeGradient):
                    full.main()
            checkpoint_path = output/"checkpoint.json"
            checkpoint = json.loads(checkpoint_path.read_text())
            self.assertEqual(checkpoint["best"]["mean_reflectance"], .21)
            checkpoint.update(step=1, m=[.17]*10, v=[.02]*10)
            checkpoint["logits"][0] += .001
            full.shared.write_json(checkpoint_path, checkpoint)
            with patch.object(sys, "argv", self.argv(output, True)):
                self.assertEqual(full.main(), 0)
            self.assertEqual(json.loads(checkpoint_path.read_text()), checkpoint)
            changed_args = self.argv(output, True)
            changed_args[changed_args.index("--learning-rate")+1] = ".02"
            with patch.object(sys, "argv", changed_args), self.assertRaises(RuntimeError):
                full.main()


if __name__ == "__main__":
    unittest.main()
