"""Independent QZ comparison and production-trace checks for device diagnostics."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from paper_reproductions.vallius2002.solver import LayerSpec, _PreparedLayer
from paper_reproductions.vallius2002.torch_backend import TorchPreparedLayer
from .diagnose_passivity import (audit_layer, audit_layer_torch, interface_audit_torch,
                                production_torch_modes)


class DeviceDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.lossy = LayerSpec(.22, (0., .21, .58, 1.), (-7.+1.2j, 2.25, 1.))
        self.lossless = LayerSpec(.22, (0., .43, 1.), (2.25, 1.))
        self.k = 2*np.pi/.93

    def check_layer(self, spec, device):
        layer = TorchPreparedLayer(spec, 'asr', 9, 27, 192, .001, device=device)
        reference = _PreparedLayer(spec, 'asr', 9, 27, 192, .001)
        cpu, _ = audit_layer(reference, self.k, 0)
        with torch.no_grad():
            row, traces = audit_layer_torch(layer, self.k, 0)
            w, labels, gamma = traces
            self.assertEqual(w.device, torch.device(device))
            self.assertEqual(labels['galerkin'].device, w.device)
            self.assertEqual(gamma.device, w.device)
            self.assertEqual(row['modal_trace_source'], 'TorchPreparedLayer.modes')
            self.assertLess(row['generalized_eigen_residual'], 1e-10)
            self.assertLess(row['production_field_projection_relative_error'], 1e-12)
            self.assertLess(row['projections']['galerkin']['power_metric_relative_error'], 1e-10)
            for key in ('field_generator_loss', 'companion_generator_loss'):
                self.assertGreaterEqual(row['projections']['galerkin'][key]['relative_minimum'], -1e-10)
            np.testing.assert_allclose(row['projected_field_condition'], cpu['projected_field_condition'], rtol=2e-7)
            for key in ('full_inverse_metric_loss', 'full_operator_loss',
                        'reduced_negative_metric_loss', 'reduced_operator_loss'):
                np.testing.assert_allclose(row[key]['operator_scale'], cpu[key]['operator_scale'], rtol=2e-7)
        return layer, traces

    def test_lossy_tensor_path_matches_independent_qz(self):
        self.check_layer(self.lossy, 'cpu')

    def test_lossless_cholesky_path_matches_independent_qz(self):
        layer, _ = self.check_layer(self.lossless, 'cpu')
        with torch.no_grad():
            *_, eigenproblem, _ = production_torch_modes(layer, self.k)
        self.assertEqual(eigenproblem, 'cholesky_eigh')

    def test_recording_preserves_production_modes_and_restores_eigensolver(self):
        layer = TorchPreparedLayer(self.lossy, 'asr', 9, 27, 192, .001, device='cpu')
        original = torch.linalg.eig
        with torch.no_grad():
            expected = layer.modes(self.k, 0., 'TM', q_projection='galerkin')
            w, v, gamma, *_ = production_torch_modes(layer, self.k)
        self.assertIs(torch.linalg.eig, original)
        for actual, reference in zip((w, v, gamma), expected):
            torch.testing.assert_close(actual, reference, rtol=0., atol=0.)

    def test_device_interface_preserves_fields_and_power(self):
        _, left = self.check_layer(self.lossy, 'cpu')
        _, right = self.check_layer(self.lossless, 'cpu')
        with torch.no_grad():
            for label in ('direct', 'galerkin'):
                result = interface_audit_torch(left, right, label, 1234)
                self.assertLess(result['relative_field_continuity_residual'], 1e-11)
                self.assertLess(result['relative_interface_power_jump'], 1e-11)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable on this host')
    def test_cuda_diagnostics_keep_dense_inputs_on_device(self):
        functions = ('eig', 'eigh', 'eigvalsh', 'svdvals', 'solve', 'matrix_norm')
        originals = {name: getattr(torch.linalg, name) for name in functions}
        calls = []

        def wrapper(name):
            def invoke(*args, **kwargs):
                for value in args:
                    if isinstance(value, torch.Tensor):
                        self.assertTrue(value.is_cuda, f'{name} received a CPU tensor')
                calls.append(name)
                return originals[name](*args, **kwargs)
            return invoke

        # Prepare reference data before instrumentation so the independent
        # SciPy/CPU validation is not counted as a GPU diagnostic operation.
        layers = [TorchPreparedLayer(spec, 'asr', 9, 27, 192, .001, device='cuda:0')
                  for spec in (self.lossy, self.lossless)]
        from contextlib import ExitStack
        with torch.no_grad(), ExitStack() as contexts:
            for name in functions:
                contexts.enter_context(patch.object(torch.linalg, name, side_effect=wrapper(name)))
            rows_and_traces = [audit_layer_torch(layer, self.k, i) for i, layer in enumerate(layers)]
            result = interface_audit_torch(rows_and_traces[0][1], rows_and_traces[1][1], 'galerkin', 1234)
            self.assertLess(result['relative_interface_power_jump'], 1e-10)
        self.assertTrue(set(functions).issubset(set(calls)))


if __name__ == '__main__':
    unittest.main()
