"""Guard against mismatched geometry, incomplete sweeps and false convergence."""
from __future__ import annotations

import contextlib
import copy
import io
import tempfile
import unittest
from pathlib import Path

from studies.pmma_gold_grating_1d.common import HERE, Study, compare, load_config
from studies.pmma_gold_grating_1d.geometry import build_layers
from studies.pmma_gold_grating_1d.run_all import run_search


class StudyTests(unittest.TestCase):
    def setUp(self):
        self.config, self.model = load_config(HERE/"config.json")
        self.config["wavelengths_nm"] = [550]
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name)

    def test_exact_valley_interface_and_profile_layer_count(self):
        for count in (32, 100, 140):
            layers = build_layers(self.config["geometry"], count)
            self.assertEqual(len(layers), count+1)
            self.assertEqual(layers[0].core_width_nm, 0.)
            self.assertEqual(layers[0].outer_width_nm, 70.)
            self.assertEqual(layers[0].bottom_depth_nm-layers[0].top_depth_nm, 30.)
            valleys = [layer for layer in layers if layer.kind == "valley"]
            self.assertAlmostEqual(sum(layer.bottom_depth_nm-layer.top_depth_nm for layer in valleys), 30.)
            self.assertEqual(valleys[0].top_depth_nm, 470.)
            self.assertTrue(all(layer.background == "gold" and layer.outer_width_nm == 200 for layer in valleys))
            self.assertAlmostEqual(sum(layer.bottom_depth_nm-layer.top_depth_nm for layer in layers[1:]), 500.)
        g = dict(self.config["geometry"], gold_valley_thickness_nm=0.)
        self.assertTrue(all(layer.kind != "valley" for layer in build_layers(g, 100)))

    def test_pmma_extension_is_bounded_and_explicit(self):
        eps, _ = self.model(400.)
        self.assertGreater(eps.real, 2.)
        self.assertEqual(eps.imag, 0.)
        self.assertIn("derived_point_nm_n", self.model.note)
        with self.assertRaises(ValueError):
            self.model(399.)

    def test_absorption_changes_must_pass_too(self):
        left = {"polarizations": {p: {"reflectance": .2, "transmittance": .2, "absorptance": .6}
                                   for p in ("TE", "TM")}}
        right = copy.deepcopy(left)
        right["polarizations"]["TM"] = {"reflectance": .203, "transmittance": .203, "absorptance": .594}
        self.assertFalse(compare(left, right, self.config)["passes_tolerance"])

    def test_incomplete_data_do_not_recommend_settings(self):
        study = Study(self.config, self.model, self.output, "cpu")
        with contextlib.redirect_stdout(io.StringIO()):
            report = run_search(study, {"order": [2, 4, 6], "slices": [2, 4, 6]}, calculate=False, max_rounds=6)
        self.assertEqual(report["status"], "incomplete")
        self.assertNotIn("recommended_numerics", report)

    def test_joint_candidate_recheck_and_resume(self):
        study = Study(self.config, self.model, self.output, "cpu")
        calls = []
        def simulated(config, numbers, wave, materials, device):
            calls.append((numbers["order"], numbers["slices"]))
            interaction = .02 if numbers["order"] < 8 and numbers["slices"] < 8 else 0
            r = .2+interaction
            return dict(numbers, wavelength_nm=wave, runtime_seconds=0.,
                        polarizations={p: {"reflectance": r, "transmittance": .3, "absorptance": .7-r}
                                       for p in ("TE", "TM")})
        study.simulate = simulated
        values = {"order": [2, 4, 6, 8], "slices": [2, 4, 6, 8]}
        with contextlib.redirect_stdout(io.StringIO()):
            report = run_search(study, values, calculate=True, max_rounds=6)
        self.assertEqual(report["recommended_numerics"], {"order": 8, "slices": 8})
        self.assertGreater(len(report["verification_rounds"]), 1)
        resumed = Study(self.config, self.model, self.output, "cpu")
        with contextlib.redirect_stdout(io.StringIO()):
            repeated = run_search(resumed, values, calculate=False, max_rounds=6)
        self.assertEqual(repeated["recommended_numerics"], report["recommended_numerics"])
        changed = copy.deepcopy(self.config)
        changed["geometry"]["gold_valley_thickness_nm"] = 0.
        with self.assertRaisesRegex(ValueError, "differ"):
            Study(changed, self.model, self.output, "cpu")


if __name__ == "__main__":
    unittest.main()
