import argparse
import cmath
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

STUDY = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('mosmite', STUDY/'run_spectrum.py')
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)
args = model.parser().parse_args([])
sys.path.insert(0, str(args.solver_root))
from rcwa_ext import AutoRCWA, Lattice, OutputSpec
torch.set_num_threads(2)
checks = []

def record(name, error, tolerance):
    assert error < tolerance, (name, error, tolerance)
    checks.append({'name': name, 'max_error': float(error), 'tolerance': tolerance})
    print(name, error, flush=True)

n = 1.57
r0 = ((1-n)/(1+n))**2
value = model.finite_sample(model.interface(1., n), n, [(n, 54800., True)], 600., 16)
record('Lossless incoherent slab versus analytic multiple reflections',
       max(abs(value[0]-2*r0/(1+r0)), abs(value[1]-(1-r0)/(1+r0)), abs(value[2])), 1e-12)

n = 1.57+.012j
d, wl = 1234., 600.
r01 = (1-n)/(1+n)
r12 = -r01
p = cmath.exp(2j*np.pi*n*d/wl)
denom = 1+r01*r12*p*p
reference = (abs((r01+r12*p*p)/denom)**2, abs((2/(1+n))*(2*n/(n+1))*p/denom)**2)
value = model.finite_sample(model.interface(1., n), n, [(n, d, False)], wl, 1)
record('Absorbing coherent slab versus independent Airy formula',
       max(abs(value[i]-reference[i]) for i in (0, 1)), 1e-12)

layers = [(1.57333+1.32e-6j, 54800., True), (1.5+0j, 20000., True),
          (model.silica_n(550.), 1e6, True)]
top = model.interface(1., layers[0][0])
coherent = model.finite_sample(top, layers[0][0], layers, 550., 1)
sim = AutoRCWA(freq=1/550., order=[1,1], lattice=Lattice.square(100.),
    group_theory=False, outputs=OutputSpec(smatrix_size='full', fields='none'),
    dtype=torch.complex128, device='cpu')
sim.add_input_layer(eps=1.); sim.add_output_layer(eps=1.)
sim.set_incident_angle(0., 0.)
for index, thickness, _ in layers:
    sim.add_layer(thickness, eps=index**2)
sim.solve_global_smatrix()
i = sim.order_N//2
direct = [float(abs(sim.S[k][i,i])**2) for k in [1,0]]
record('Finite coherent multilayer versus full homogeneous RCWA',
       max(abs(coherent[i]-direct[i]) for i in [0,1]), 1e-9)

avg16 = model.finite_sample(top, layers[0][0], layers, 550., 16)
avg32 = model.finite_sample(top, layers[0][0], layers, 550., 32)
record('Three-layer independent phase quadrature 16 versus 32',
       max(abs(avg16[i]-avg32[i]) for i in [0,1,2]), 1e-12)

pet, _ = model.load_nk(STUDY/'data/PET_Zhang_400_800nm.csv')
record('Frozen PET dataset at 600 nm', abs(pet(600.)-(1.56811+1.29e-6j)), 1e-12)
try:
    pet(900.)
except ValueError:
    checks.append({'name': 'PET extrapolation rejected', 'passed': True})
else:
    raise AssertionError('Extrapolation was not rejected')

args.height_nm = 0.
args.order, args.slices, args.grid = 2, 1, 128
flat = model.surface_rcwa(args, 600., pet(600.))
expected = model.interface(1., pet(600.))
record('Zero-height D6 patterned RCWA versus complex Fresnel amplitudes',
       max(abs(a-b) for a,b in zip(flat, expected)), 1e-9)

args.residual_um = .3
nr, ns = 1.49+.0003j, pet(600.)
residual = model.surface_rcwa(args, 600., nr, n_substrate=ns)
r01, r12 = (1-nr)/(1+nr), (nr-ns)/(nr+ns)
p = cmath.exp(2j*np.pi*nr*300./600.)
denom = 1+r01*r12*p*p
expected = ((2/(1+nr))*(2*nr/(nr+ns))*p/denom,
            (r01+r12*p*p)/denom,
            (-r12-r01*p*p)/denom,
            (2*ns/(ns+nr))*(2*nr/(nr+1))*p/denom)
record('Different residual resin and PET versus asymmetric Airy amplitudes',
       max(abs(a-b) for a,b in zip(residual, expected)), 1e-9)
args.residual_um = 0.

args.height_nm = 200.
args.order, args.slices = 3, 20
sx = model.surface_rcwa(args, 600., pet(600.))
args.polarization = 'y'
sy = model.surface_rcwa(args, 600., pet(600.))
record('D6 x/y equality including reverse amplitudes', max(abs(a-b) for a,b in zip(sx,sy)), 1e-9)

args.polarization = 'x'
args.method = 'matched-asr'
sa = model.surface_rcwa(args, 600., pet(600.))
assert all(np.isfinite(value) for value in sa)
checks.append({'name': 'Optional matched-ASR backend executes', 'passed': True})

Path(__file__).with_suffix('.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
print('All checks passed', flush=True)

