"""Evaluator compatible with the frozen production Adam engine.

Returns the SAME objective/gradient/rows interface as production physics.objective.
The radius-to-logits VJP uses the original monotone-softmax geometry, including
the nm-to-period normalization. Only first derivatives are supported.
"""
from __future__ import annotations
import copy
import importlib
import math
from pathlib import Path
from types import SimpleNamespace
import time
import torch
from torch.utils.checkpoint import checkpoint
import benchmark_square as square
import hex_core as hexagonal


def load_model(material):
    if material not in ('n2.5', 'PMMA'):
        raise ValueError('Only the validated lossless n2.5 and PMMA materials are supported.')
    prefix = 'Optimization.dielectric_motheye_n25' if material == 'n2.5' else 'pmma_case.optimizer'
    modules = {name: importlib.import_module(prefix+'.'+name)
               for name in ('common', 'geometry', 'physics', 'engine')}
    root = Path(__file__).resolve().parent
    if not all(Path(m.__file__).resolve().is_relative_to(root) for m in modules.values()):
        raise RuntimeError('Production module resolved outside the frozen validation package.')
    path = root/('Optimization/dielectric_motheye_n25/config.json'
                 if material == 'n2.5' else 'pmma_case/config.json')
    return SimpleNamespace(material=material, config_path=path, **modules)


def sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


class FastEvaluator:
    backend_name = 'spectral-shared-production-contract-v1'

    def __init__(self, model):
        self.model = model
        self._cache = {}

    def indices(self, config, waves):
        if self.model.material == 'PMMA':
            from pmma_case.optimizer.materials import material_index
            return [material_index(config, wave) for wave in waves]
        if config.get('material_model', 'constant') != 'constant' or config['index'] != 2.5:
            raise ValueError('n2.5 evaluator requires constant lossless index 2.5.')
        return [float(config['index'])]*len(waves)

    def cache(self, c, lattice, waves, order, grid, device):
        if lattice not in ('square', 'hexagonal'):
            raise ValueError('Only square and hexagonal normal-incidence circle cases are supported.')
        indices = self.indices(c, waves)
        key = (lattice, order, grid, c['period_nm'], c['height_nm'], c['layers'],
               tuple(waves), tuple(indices), str(device))
        if key not in self._cache:
            backend = square if lattice == 'square' else hexagonal
            args = SimpleNamespace(order=order, grid=grid, layers=c['layers'],
                                   period=c['period_nm'], height=c['height_nm'],
                                   device=str(device), material=self.model.material)
            with torch.no_grad():
                template = backend.make_sim(args, waves[0], indices[0])
                plan = backend.assembly.GeometryContext(template)
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
                identity=template._eye(plan.scalar_dimension))
        return self._cache[key]

    def geometry_pair(self, radius, cache):
        args, b, t, p = cache['args'], cache['backend'], cache['template'], cache['plan']
        if self.model.material == 'PMMA':
            eps, _, normal = b.assembly.material_inputs(t, radius, 2., args.grid, p)
            return (eps-cache['identity'])/3., normal
        inputs = b.assembly.material_inputs(t, radius, cache['indices'][0], args.grid, p)
        left, right, _ = b.assembly.candidate_pq(cache['unit'], *inputs, p)
        return left-cache['p0'], right-cache['q2']

    def response(self, cache, leaves, wave, index, port, checkpointed):
        def layer_modes(a, b):
            if self.model.material == 'PMMA':
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
        leaves = tuple(x.detach().requires_grad_(gradient) for x in originals)
        covectors = [torch.zeros_like(x) for x in leaves] if gradient else None
        rows = []
        total = 0.
        for wave, weight, index, port in zip(waves, weights, cache['indices'], cache['ports']):
            R, T = self.response(cache, leaves, wave, index, port, checkpointed)
            if gradient:
                derivatives = torch.autograd.grad(weight*R, leaves)
                for accumulator, derivative in zip(covectors, derivatives):
                    accumulator.add_(derivative)
                del derivatives
            rv, tv = float(R.detach().cpu()), float(T.detach().cpu())
            if (not math.isfinite(rv+tv) or min(rv,tv) < -1e-7 or max(rv,tv) > 1+1e-7
                    or abs(rv+tv-1) > c['energy_tolerance']):
                raise self.model.physics.NonphysicalResult('Invalid lossless shared power.')
            row = dict(wavelength_nm=wave, reflectance=rv, transmittance=tv,
                energy_residual=rv+tv-1, retained_scalar_harmonics=cache['plan'].scalar_dimension,
                eigen_dimension=cache['plan'].reduced_dimension,
                storage=self.backend_name, checkpoint_layers=checkpointed)
            if self.model.material == 'PMMA':
                row.update(material_index=index, pmma_shortwave_extrapolated=
                    wave < c['pmma_samples_nm'][0][0])
            rows.append(row)
            total += weight*rv
            del R,T
        derivative = torch.autograd.grad(tuple(originals), values,
            grad_outputs=tuple(covectors))[0].detach() if gradient else None
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

