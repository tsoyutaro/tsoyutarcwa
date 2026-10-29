"""Production objective: shared geometry/material coefficients and one geometry VJP."""
from __future__ import annotations
import cmath
import copy
from collections import OrderedDict
import math
from types import SimpleNamespace
import time
import torch
from torch.utils.checkpoint import checkpoint
from ..auto import AutoRCWA
from ..config import GroupTheoryOptions,Lattice,NVMOptions,OutputSpec
from . import matrix
from .ports import CompactPort,free_coefficients


def sync(device):
    if torch.device(device).type=='cuda': torch.cuda.synchronize(device)


def make_sim(args,wave,index):
    sim=AutoRCWA(freq=args.period/wave,order=[args.order]*2,
        lattice=Lattice.square(1.) if args.lattice=='square' else Lattice.triangular(1.),
        nvm=NVMOptions(grid=(args.grid,args.grid),use_lanczos=False),
        outputs=OutputSpec(smatrix_size='half',fields='none'),cascade='redheffer',
        group_theory=GroupTheoryOptions(enabled=True,symmetry='c2v' if args.lattice=='square' else 'd6',polarization='x',strict=True),
        dtype=torch.complex128,device=args.device,verify_cascade=False,acceleration='reference')
    sim.add_input_layer(eps=1.)
    sim.add_output_layer(eps=index**2)
    sim.set_incident_angle(0.,0.)
    return sim

_BACKEND=SimpleNamespace(make_sim=make_sim,assembly=matrix,CompactPort=CompactPort,free_coefficients=free_coefficients)


class SharedEvaluator:
    backend_name = 'rcwa-shared-v3'

    def __init__(self, model):
        self.model = model
        self._cache = OrderedDict()

    def indices(self, config, waves):
        return [cmath.sqrt(complex(self.model.epsilon_at(config,w))) for w in waves]

    def cache(self, c, lattice, waves, order, grid, device):
        if lattice not in ('square', 'hexagonal'):
            raise ValueError('Only square and hexagonal normal-incidence circle cases are supported.')
        indices = self.indices(c, waves)
        key = (lattice, order, grid, c['period_nm'], c['height_nm'], c['layers'],
               tuple(waves), tuple(indices), str(device))
        if key not in self._cache:
            backend = _BACKEND
            args = SimpleNamespace(order=order, grid=grid, layers=c['layers'],
                                   period=c['period_nm'], height=c['height_nm'],
                                   device=str(device), material=self.model.material,lattice=lattice)
            with torch.no_grad():
                template = backend.make_sim(args, waves[0], indices[0])
                plan = backend.assembly.plan_for(template)
                unit = copy.copy(template)
                factor = torch.real(template.freq)
                unit.K1_norm, unit.K2_norm = template.K1_norm*factor, template.K2_norm*factor
                unit.Kx_norm_dn, unit.Ky_norm_dn = template.Kx_norm_dn*factor, template.Ky_norm_dn*factor
                p0, q2 = backend.free_coefficients(unit, plan)
                ports = [backend.CompactPort(backend.make_sim(args, w, n), plan,
                              c['height_nm']/(c['layers']*c['period_nm']), c['layers'])
                         for w, n in zip(waves, indices)]
            self._cache[key] = dict(args=args, backend=backend, template=template, plan=plan,
                unit=unit, p0=p0, q2=q2, ports=ports, indices=indices,
                identity=template._eye(plan.scalar_dimension),
                shared_geometry_only=not all(n==indices[0] for n in indices))
            while len(self._cache)>2:
                self._cache.popitem(last=False)
        self._cache.move_to_end(key)
        return self._cache[key]

    def geometry_pair(self, radius, cache):
        args, b, t, p = cache['args'], cache['backend'], cache['template'], cache['plan']
        if cache['shared_geometry_only']:
            eps, _, normal = b.assembly.material_inputs(t, radius, 2., args.grid, p)
            return (eps-cache['identity'])/3., normal
        inputs = b.assembly.material_inputs(t, radius, cache['indices'][0], args.grid, p)
        left, right, _ = b.assembly.candidate_pq(cache['unit'], *inputs, p)
        return left-cache['p0'], right-cache['q2']

    def response(self, cache, leaves, wave, index, port, checkpointed):
        def layer_modes(a, b):
            if cache['shared_geometry_only']:
                eps = cache['identity']+(index**2-1)*a
                rule = cache['identity']+(1/index**2-1)*a
                p, q, _ = cache['backend'].assembly.candidate_pq(
                    port, eps, rule, b, cache['plan'])
            else:
                scale = (wave/cache['args'].period)**2
                p, q = cache['p0']+scale*a, b+scale*cache['q2']
            squared, electric = port._eig(p@q)
            kz = port._positive_kz(squared)
            magnetic = port._magnetic_eigenvectors(p, q, electric, kz)
            return electric, magnetic, kz
        modes = []
        for j in range(cache['args'].layers):
            pair = leaves[2*j:2*j+2]
            if checkpointed:
                electric, magnetic, kz = checkpoint(layer_modes, *pair,
                    use_reentrant=False, preserve_rng_state=False)
            else:
                electric, magnetic, kz = layer_modes(*pair)
            modes.append(dict(electric=electric, magnetic=magnetic, kz=kz))
        transmission, reflection, _, _ = port._polarized_redheffer_scattering(
            modes, port.input_interface, port.output_interface)
        t, r = transmission@port.source, reflection@port.source
        R = .5*torch.real(r.conj()@(port.flux_in@r))/port.incident_flux
        T = .5*torch.real(t.conj()@(port.flux_out@t))/port.incident_flux
        return R, T

    def spectrum(self, normalized_radii, c, waves, weights, cache, gradient):
        values = normalized_radii.detach().clone().requires_grad_(gradient)
        checkpointed = bool(c['checkpoint_layers'] and gradient)
        def build(radius):
            return self.geometry_pair(radius, cache)
        originals = []
        for radius in values:
            pair = checkpoint(build, radius, use_reentrant=False, preserve_rng_state=False) if checkpointed else build(radius)
            originals.extend(pair)
        active = [i for i,x in enumerate(originals) if x.requires_grad] if gradient else []
        leaves = tuple(x.detach().requires_grad_(gradient and i in active) for i,x in enumerate(originals))
        covectors = [torch.zeros_like(leaves[i]) for i in active] if gradient else None
        rows = []
        total = 0.
        for wave, weight, index, port in zip(waves, weights, cache['indices'], cache['ports']):
            R, T = self.response(cache, leaves, wave, index, port, checkpointed)
            if gradient:
                derivatives = torch.autograd.grad(weight*R, tuple(leaves[i] for i in active))
                for accumulator, derivative in zip(covectors, derivatives):
                    accumulator.add_(derivative)
                del derivatives
            rv, tv = float(R.detach().cpu()), float(T.detach().cpu())
            metal = c.get('substrate_mode')=='semi-infinite'
            tol=c.get('passivity_tolerance',c.get('energy_tolerance',1e-6))
            powers=(rv,tv,1-rv,1-rv-tv) if metal else (rv,tv)
            if (not all(math.isfinite(v) and -tol<=v<=1+tol for v in powers)
                    or not metal and abs(rv+tv-1)>tol):
                raise self.model.physics.NonphysicalResult('Invalid shared power flux.')
            row = dict(wavelength_nm=wave, reflectance=rv,
                retained_scalar_harmonics=cache['plan'].scalar_dimension,
                eigen_dimension=cache['plan'].reduced_dimension,
                storage=self.backend_name, checkpoint_layers=checkpointed)
            if metal:
                eps=index**2
                row.update(power_into_substrate=tv,transmittance_far=0.,absorptance_total=1-rv,
                    motheye_absorptance=1-rv-tv,substrate_absorptance=tv,passivity_warning=False,
                    epsilon_material_real=eps.real,epsilon_material_imag=eps.imag,
                    absorption_evaluation='inferred from surface power flux; no independent volume-loss audit')
            else:
                row.update(transmittance=tv,energy_residual=rv+tv-1)
            if self.model.material == 'PMMA':
                row.update(material_index=float(index.real),pmma_shortwave_extrapolated=
                    wave<c['pmma_samples_nm'][0][0])
            rows.append(row)
            total += weight*rv
            del R,T
        derivative = (torch.autograd.grad(tuple(originals[i] for i in active), values,
            grad_outputs=tuple(covectors))[0].detach() if active else torch.zeros_like(values)) if gradient else None
        return total, derivative, rows

    def __call__(self, logits, config, lattice, wavelengths, order, grid, device, *, gradient):
        device = torch.device(device)
        sync(device)
        started = time.perf_counter()
        waves = [float(w) for w in wavelengths]
        if not waves:
            raise ValueError('Empty objective wavelength list.')
        weights = self.model.physics.weights(waves, config['objective_weighting'])
        if abs(sum(weights)-1) > 1e-12:
            raise ValueError('Objective weights must sum to one.')
        cache = self.cache(config, lattice, waves, order, grid, device)
        with torch.set_grad_enabled(gradient):
            radii = self.model.geometry.radii_tensor(logits, config)/config['period_nm']
            value, radius_gradient, rows = self.spectrum(radii, config, waves, weights, cache, gradient)
            derivative = torch.autograd.grad(radii, logits, grad_outputs=radius_gradient)[0].detach() if gradient else None
        if not math.isfinite(value) or gradient and not bool(torch.isfinite(derivative).all()):
            raise self.model.physics.NonphysicalResult('Nonfinite shared objective/logits gradient.')
        sync(device)
        return dict(objective=value, gradient=derivative, rows=rows,
                    seconds=time.perf_counter()-started, backend=self.backend_name)

