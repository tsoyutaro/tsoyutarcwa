"""Cold CUDA allocator regression; real GPU tests run in fresh processes."""
from __future__ import annotations

import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from ..model import Geometry
from ..solver import solve


class StopBeforeTensorAllocation(Exception):
    pass


class CUDAInitializationTests(unittest.TestCase):
    def test_first_solve_initializes_before_resetting_allocator_stats(self):
        # Simulate the allocator's cold-start precondition on a CPU-only host.
        # The original code raises Invalid device argument before MatrixFunction.
        for device in ('cuda:0', 'cuda'):
            with self.subTest(device=device):
                state = {'initialized': False}

                def initialize():
                    state['initialized'] = True

                def reset_stats(selected):
                    # PyTorch's index-free spelling calls current_device(),
                    # which lazily initializes; cuda:0 bypasses that path.
                    if torch.device(selected).index is None:
                        initialize()
                    if not state['initialized']:
                        raise RuntimeError('Invalid device argument')
                    self.assertEqual(torch.device(selected),torch.device(device))

                with patch('torch.cuda.is_available',return_value=True), \
                     patch('torch.cuda.init',side_effect=initialize), \
                     patch('torch.cuda.reset_peak_memory_stats',side_effect=reset_stats), \
                     patch('studies.gold_motheye_fff2d.solver.MatrixFunction',side_effect=StopBeforeTensorAllocation):
                    with self.assertRaises(StopBeforeTensorAllocation):
                        solve(Geometry(),550.,4.,order=0,device=device)

    def test_cpu_does_not_initialize_cuda_or_access_cuda_stats(self):
        with patch('torch.cuda.init',side_effect=AssertionError('CPU touched CUDA')), \
             patch('torch.cuda.reset_peak_memory_stats',side_effect=AssertionError('CPU touched CUDA')), \
             patch('studies.gold_motheye_fff2d.solver.MatrixFunction',side_effect=StopBeforeTensorAllocation):
            with self.assertRaises(StopBeforeTensorAllocation):
                solve(Geometry(),550.,4.,order=0,device='cpu')

    def test_unavailable_cuda_is_not_silently_replaced_with_cpu(self):
        with patch('torch.cuda.is_available',return_value=False), \
             patch('torch.cuda.init') as initialize:
            with self.assertRaisesRegex(ValueError,'CUDA was requested but is unavailable'):
                solve(Geometry(),550.,4.,order=0,device='cuda:0')
            initialize.assert_not_called()

    def test_failed_validation_cache_is_retried(self):
        from ..run import main
        failed = dict(passed=False,device='cpu',ordering='symmetric',checks=[])
        with TemporaryDirectory(prefix='fff_cuda_cache_',dir=Path.cwd()) as output, \
             patch('studies.gold_motheye_fff2d.validate.checks',return_value=failed) as validate, \
             redirect_stdout(StringIO()):
            args = ['--device','cpu','--orders','1','--slices','1','--grid','32',
                    '--wavelengths','550','--methods','rcwa-fff','--output-dir',output]
            self.assertEqual(main(args),1)
            self.assertEqual(main(args),1)
            self.assertEqual(validate.call_count,2,'A cached startup failure must not permanently block retries')


@unittest.skipUnless(torch.cuda.is_available(),'Actual CUDA is unavailable on this host')
class FreshProcessCUDATests(unittest.TestCase):
    def test_first_solve_on_each_device_spelling_matches_fresnel(self):
        code = r'''
import cmath,json,sys
import torch
from studies.gold_motheye_fff2d.model import Geometry
from studies.gold_motheye_fff2d.solver import solve
torch.set_num_threads(1)
assert not torch.cuda.is_initialized(), 'CUDA must be cold before first solve'
n=1.8+.2j
g=Geometry(height_nm=80,tip_radius_nm=30,base_radius_nm=60)
r=solve(g,550.,n*n,order=0,slices=4,grid=32,device=sys.argv[1],
        homogeneous=True,epsilon_out=2.25,volume_absorption=True)
p=cmath.exp(2j*cmath.pi*n*80/550)
r01,r12=(1-n)/(1+n),(n-1.5)/(n+1.5)
ref_r=abs((r01+r12*p*p)/(1+r01*r12*p*p))**2
ref_t=1.5*abs(2/(1+n)*2*n/(n+1.5)*p/(1+r01*r12*p*p))**2
error=max(abs(r.R-ref_r),abs(r.P_sub-ref_t),abs(r.absorption_volume-(1-ref_r-ref_t)))
assert error<2e-8,(sys.argv[1],error)
assert r.metadata['peak_cuda_bytes'] is not None
print(json.dumps(dict(device=sys.argv[1],max_fresnel_error=error)))
'''
        for device in ('cuda:0','cuda'):
            with self.subTest(device=device):
                result = subprocess.run([sys.executable,'-c',code,device],capture_output=True,text=True,timeout=180)
                self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
                print(result.stdout.strip(),flush=True)


if __name__ == '__main__':
    unittest.main()
