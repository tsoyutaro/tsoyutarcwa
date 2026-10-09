"""Numerical parity, bounded layer lifetimes, and checkpoint resume regression."""
from __future__ import annotations

import copy
import gc
import json
import tempfile
import unittest
import weakref
from argparse import Namespace
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
try:
    from threadpoolctl import threadpool_limits
except ModuleNotFoundError as exc:
    if exc.name != 'threadpoolctl':
        raise
    threadpool_limits = None

from paper_reproductions.vallius2002.devices import resolve_execution
from paper_reproductions.vallius2002.solver import LayerSpec, PreparedStack
from paper_reproductions.vallius2002.torch_backend import TorchPreparedLayer
from .adapters import load_study, scalar_case
from .streaming_gpu import _prepare_mode_stack, scalar_case_streamed


class ParityChecks:
    device = 'cpu'

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.limits = threadpool_limits(limits=1) if threadpool_limits is not None else nullcontext()
        cls.limits.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.limits.__exit__(None,None,None)

    def test_full_complex_fields_lossy_lossless_oblique_and_cutoff(self):
        for epsilon in (4., -9.3+1.5j):
            layers = [LayerSpec(.11,(0,.2,.45,.7,1),(1,epsilon,2.25,1)),
                      LayerSpec(.07,(0,.2,.45,.7,1),(1,epsilon,2.25,1)),
                      LayerSpec(.08,(0,1),(2.25,))]
            for angle,wave in ((0.,.93),(.12,.88),(0.,1.)):
                for projection in ('direct','laurent','galerkin'):
                    options = dict(method='asr',harmonics=9,oversampling=4,
                                   q_projection=projection,epsilon_out=2.25,
                                   diagnostics=True,angle=angle)
                    resident = PreparedStack(layers,quadrature=192,device=self.device,
                                             backend='torch',**options)
                    with torch.no_grad():
                        streamed = _prepare_mode_stack(layers,wave,quadrature=192,
                            execution=resolve_execution(self.device,'torch'),**options)
                    for pol in ('TE','TM'):
                        resident.polarization=streamed.polarization=pol
                        expected,actual=resident.solve(wave),streamed.solve(wave)
                        for key in ('r','t','R_orders','T_orders'):
                            np.testing.assert_allclose(actual[key],expected[key],atol=2e-11,rtol=2e-10)
                        self.assertEqual(actual['cutoff_regularized'],expected['cutoff_regularized'])
                        self.assertEqual(actual['linear_solves'],expected['linear_solves'])
                        self.assertEqual(streamed.prepared_layers[0],streamed.prepared_layers[1])

    def test_actual_studies_observables_and_metadata_match(self):
        for study in ('gold_grating_1d','pmma_gold_grating_1d'):
            config,model,_=load_study(study)
            for wave in (650.,700.):
                options=dict(oversampling=6,quadrature=256,device=self.device,
                             backend='torch',diagnostics=True)
                expected=scalar_case(study,config,6,4,wave,model(wave),**options)
                actual=scalar_case_streamed(study,config,6,4,wave,model(wave),**options)
                for pol in ('TE','TM'):
                    for metric,value in expected['polarizations'][pol].items():
                        self.assertAlmostEqual(actual['polarizations'][pol][metric],value,delta=2e-10)
                for key in ('harmonics','eigen_dimension','total_finite_layers','quadrature_actual'):
                    self.assertEqual(actual[key],expected[key])
                self.assertEqual(actual['execution']['linear_algebra_device'],actual['execution']['device'])
                self.assertEqual(actual['execution']['large_prepared_layers_live_limit'],1)

    def test_one_large_prepared_layer_and_compact_modal_storage(self):
        active=peak=0
        def release():
            nonlocal active
            active-=1
        class TrackedLayer(TorchPreparedLayer):
            def __init__(self,*args,**kwargs):
                nonlocal active,peak
                super().__init__(*args,**kwargs)
                active+=1;peak=max(peak,active)
                weakref.finalize(self,release)
        layers=[LayerSpec(.01,(0,.15+i*.01,1),(3+.2j,1)) for i in range(18)]
        with patch('paper_reproductions.vallius2002.torch_backend.TorchPreparedLayer',TrackedLayer):
            with torch.no_grad():
                stack=_prepare_mode_stack(layers,.92,quadrature=192,
                    execution=resolve_execution(self.device,'torch'),
                    method='asr',harmonics=7,oversampling=6,q_projection='galerkin')
        gc.collect()
        self.assertEqual(peak,1)
        self.assertEqual(active,0)
        for layer in stack.prepared_layers:
            for traces in layer.traces.values():
                for value in traces:
                    self.assertEqual(value.untyped_storage().nbytes(),value.numel()*value.element_size())
                    self.assertEqual(value.dtype,torch.complex128)
                    self.assertEqual(value.device.type,'cuda' if self.device.startswith('cuda') else 'cpu')
        with self.assertRaisesRegex(ValueError,'prepared wavelength'):
            stack.solve(.93)


class TorchCPUChecks(ParityChecks,unittest.TestCase):
    device='cpu'


@unittest.skipUnless(torch.cuda.is_available(),'CUDA is unavailable on this host')
class CUDAChecks(ParityChecks,unittest.TestCase):
    device='cuda'


class ResumeChecks(unittest.TestCase):
    def test_nine_completed_m64_cases_survive_and_only_failed_case_is_retried(self):
        from .compare import fingerprint, main
        study='pmma_gold_grating_1d'
        config,model,_=load_study(study)
        inputs,signature=fingerprint(study,config,Namespace(G=.001,quadrature=4096,q_projection='galerkin'))
        observed={p:dict(reflectance=.2,transmittance=.3,absorptance=.5) for p in ('TE','TM')}
        valid=dict(runtime_seconds=0.,polarizations=observed,
                   diagnostics={p:dict(max_boundary_condition=1.) for p in ('TE','TM')})
        cases={f'li|300|{order}|{wave}':copy.deepcopy(valid)
               for order in (64,120,128) for wave in (650,700)}
        cases.update({f'asr_r{r}|300|64|{wave}':copy.deepcopy(valid)
                      for r in (6,7) for wave in (650,700)})
        failed='asr_r7|300|64|700'
        cases[failed]=dict(error='OutOfMemoryError: previous failure',polarizations={})
        preserved={key:copy.deepcopy(value) for key,value in cases.items() if key != failed}
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path=Path(directory)/study/'checkpoint.json';path.parent.mkdir()
            path.write_text(json.dumps(dict(signature=signature,inputs=inputs,cases=cases)))
            with patch('studies.asr_1d_comparison.streaming_gpu.scalar_case_streamed',return_value=valid) as retry:
                with patch('studies.asr_1d_comparison.adapters.existing_li_case',side_effect=AssertionError('recomputed cached Li')):
                    with patch('studies.asr_1d_comparison.compare.export'):
                        main(['--study',study,'--orders','64','--asr-ratios','6,7',
                              '--reference-order','128','--reference-check-order','120',
                              '--wavelengths','650,700','--quadrature','4096',
                              '--device','cpu','--backend','torch','--output-root',directory])
            retry.assert_called_once()
            self.assertEqual(retry.call_args.args[4],700.)
            self.assertEqual(retry.call_args.kwargs['oversampling'],7)
            saved=json.loads(path.read_text())
            self.assertEqual(saved['signature'],signature)
            self.assertEqual(saved['inputs'],inputs)
            self.assertEqual({key:saved['cases'][key] for key in preserved},preserved)
            self.assertNotIn('error',saved['cases'][failed])


if __name__=='__main__':
    unittest.main()
