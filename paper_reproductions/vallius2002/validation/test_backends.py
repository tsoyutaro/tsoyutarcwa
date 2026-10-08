"""Device selection, physical power checks and optional actual CUDA tests."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ..devices import resolve_execution
from ..solver import LayerSpec, PreparedStack
from .test_solver import slab_power

try:
    import torch
except ImportError:
    torch = None


class SelectionTests(unittest.TestCase):
    def test_cpu_needs_no_torch(self):
        with patch("importlib.import_module", side_effect=ImportError("absent")):
            self.assertEqual(resolve_execution("cpu")["backend"], "scipy")
            self.assertEqual(resolve_execution("auto")["device"], "cpu")
            with self.assertRaisesRegex(RuntimeError, "PyTorch is required"):
                resolve_execution("cuda")

    def test_auto_selects_visible_cuda_and_respects_explicit_cpu(self):
        cuda = SimpleNamespace(is_available=lambda: True, current_device=lambda: 1,
                               device_count=lambda: 2, get_device_name=lambda index: "test GPU")
        fake = SimpleNamespace(__version__="test", version=SimpleNamespace(cuda="test"),
                               cuda=cuda, empty=lambda *args, **kwargs: None, complex128="test")
        with patch("importlib.import_module", return_value=fake):
            execution = resolve_execution("auto")
            self.assertEqual((execution["device"], execution["backend"]), ("cuda:1", "torch"))
            self.assertEqual(resolve_execution("cuda:0")["device"], "cuda:0")
            self.assertEqual(resolve_execution("cpu")["device"], "cpu")
            with self.assertRaisesRegex(RuntimeError, "out of range"):
                resolve_execution("cuda:2")

    def test_explicit_cuda_cannot_silently_fall_back(self):
        fake = SimpleNamespace(__version__="test+cpu", version=SimpleNamespace(cuda=None),
                               cuda=SimpleNamespace(is_available=lambda: False))
        with patch("importlib.import_module", return_value=fake):
            self.assertEqual(resolve_execution("auto")["backend"], "scipy")
            with self.assertRaisesRegex(RuntimeError, "CUDA was requested but is unavailable"):
                resolve_execution("cuda")

    def test_invalid_device_or_backend_is_rejected(self):
        for device in ("gpu", "cuda:-1", "mps"):
            with self.subTest(device=device), self.assertRaises(ValueError):
                resolve_execution(device)
        with self.assertRaises(ValueError):
            resolve_execution("cuda", "scipy")


class TensorPhysicalChecks:
    device = "cpu"

    def stack(self, layers, **options):
        return PreparedStack(layers, device=self.device, backend="torch", **options)

    def test_homogeneous_passive_and_unequal_media_match_fresnel(self):
        for polarization in ("TE", "TM"):
            for epsilon, epsilon_out in ((6.25, 2.25), (3.2 + 0.8j, 1.0)):
                for method in ("fmm", "asr"):
                    with self.subTest(polarization=polarization, epsilon=epsilon, method=method):
                        result = self.stack([LayerSpec(0.16, (0, 1), (epsilon,))],
                                            polarization=polarization, method=method,
                                            harmonics=7, epsilon_out=epsilon_out).solve(0.91)
                        expected = slab_power(0.91, 0.16, epsilon, epsilon_out=epsilon_out)
                        np.testing.assert_allclose([result[k] for k in ("T", "R", "A")], expected,
                                                   atol=2e-10, rtol=2e-10)

    def test_lossless_fmm_conserves_energy(self):
        layers = [LayerSpec(0.21, (0, 0.37, 1), (4, 1)),
                  LayerSpec(0.13, (0, 0.19, 0.64, 1), (1, 2.25, 1))]
        for polarization in ("TE", "TM"):
            result = self.stack(layers, polarization=polarization, harmonics=17).solve(0.72)
            self.assertAlmostEqual(result["T"] + result["R"], 1, delta=2e-9)

    def test_galerkin_tm_three_material_interfaces_remain_passive(self):
        layers = [LayerSpec(.12, (0,.24,.48,.72,1), (1,-9.3875+1.5292j,2.22,1)),
                  LayerSpec(.09, (0,.41,1), (3+.2j,1))]
        options = dict(polarization="TM", method="asr", harmonics=17, oversampling=4,
                       q_projection="galerkin", epsilon_out=2.25)
        for angle in (0., .12):
            for wavelength in (.72, 3.):
                result = self.stack(layers, angle=angle, **options).solve(wavelength)
                self.assertGreaterEqual(result["R"], -2e-9)
                self.assertGreaterEqual(result["T"], -2e-9)
                self.assertLessEqual(result["R"] + result["T"], 1+2e-9)
                reference = PreparedStack(layers, angle=angle, **options,
                                          device="cpu", backend="scipy").solve(wavelength)
                np.testing.assert_allclose([result[k] for k in ("R","T")],
                                           [reference[k] for k in ("R","T")], atol=2e-8, rtol=2e-7)

    def test_oblique_projection_retention_and_laurent_match_scipy(self):
        layers = [LayerSpec(0.23, (0, 0.3, 1), (4, 1))]
        for polarization in ("TE", "TM"):
            for retention in ("smallest_abs", "physical"):
                for q_projection in ("direct", "laurent", "galerkin"):
                    options = dict(polarization=polarization, method="asr", harmonics=11,
                                   angle=0.17, retention=retention, q_projection=q_projection)
                    cpu = PreparedStack(layers, **options, device="cpu", backend="scipy").solve(0.89)
                    tensor = self.stack(layers, **options).solve(0.89)
                    for key in ("t", "r", "T_orders", "R_orders"):
                        np.testing.assert_allclose(tensor[key], cpu[key], atol=2e-8, rtol=2e-7)

    def test_cutoff_spectrum_metadata_and_double_device_storage(self):
        stack = self.stack([LayerSpec(0.23, (0, 0.4, 1), (4, 1))], method="asr", harmonics=7)
        result = stack.spectrum([0.97, 1.0, 1.01])
        self.assertTrue(result["cutoff_regularized"][1])
        self.assertEqual(result["backend"], "torch")
        self.assertEqual(str(stack.prepared_layers[0].a.device), result["device"])
        self.assertEqual(stack.prepared_layers[0].a.dtype, torch.complex128)
        self.assertGreater(result["evaluation_wavelength"][1], 1)
        with self.assertRaisesRegex(ValueError, "exact external Rayleigh cutoff"):
            stack.solve(1, cutoff_shift=0)
        with self.assertRaises(ValueError):
            stack.solve(float("nan"))


@unittest.skipIf(torch is None, "optional PyTorch is not installed")
class TorchCPUTests(TensorPhysicalChecks, unittest.TestCase):
    device = "cpu"


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA is unavailable on this host")
class CUDATests(TensorPhysicalChecks, unittest.TestCase):
    device = "cuda"


if __name__ == "__main__":
    unittest.main()
