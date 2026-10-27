"""Hexagonal D6 adapter for the common CPU/CUDA validation driver.

Frozen full-space RCWA is the reference, native D6 is the block path, and
spectral sharing uses the same native D6 operators as the block path.
"""
from __future__ import annotations
from pathlib import Path
import torch
import block_assembly as assembly
import d6_nvm as d6
import hex_spectral_cpu as frozen_hex
from benchmark_square import load_pmma, pmma_index, wavelengths, compare

HERE = Path(__file__).resolve().parent


def make_sim(args, wave, index):
    return assembly.make_sim('hexagonal', args.order, args.grid, index, wave,
                             args.period, args.device)


class CompactPort(frozen_hex.CompactPort):
    def __init__(self, sim, plan, depth, layers):
        super().__init__(sim, plan, depth, layers)
        # Material algebra must use the Cartesian wavevector at THIS wavelength.
        self.Kx_norm_dn = sim.Kx_norm_dn.detach()
        self.Ky_norm_dn = sim.Ky_norm_dn.detach()


free_coefficients = frozen_hex.free_coefficients


def sequential(args, waves, indices, radii, context, variant, gradient=True):
    total = torch.zeros_like(radii) if gradient else None
    spectrum = []
    for wave, index in zip(waves, indices):
        sim = make_sim(args, wave, index)
        values = radii.detach().clone().requires_grad_(gradient)
        R, T, transmitted, reflected = assembly.optical_response(
            sim, values, index, args.height, args.period, args.grid,
            'reference' if variant == 'reference' else 'candidate', context)
        if gradient:
            total += torch.autograd.grad(R/len(waves), values)[0]
        spectrum.append([wave, float(R.detach()), float(T.detach())])
        del sim, values, R, T, transmitted, reflected
    return dict(spectrum=spectrum, gradient=total.detach().cpu().tolist() if gradient else None)


def shared(args, waves, indices, radii, template, context, ports, unit, p0, q2,
           gradient=True):
    values = radii.detach().clone().requires_grad_(gradient)
    identity = template._eye(context.scalar_dimension)
    originals = []
    for radius in values:
        if args.material == 'PMMA':
            temporary, _, normal = d6.native_inputs(template, radius, 2., args.grid, context)
            originals.extend(((temporary-identity)/3., normal))
        else:
            inputs = d6.native_inputs(template, radius, indices[0], args.grid, context)
            p, q, _ = d6.reduced_pq(unit, *inputs, context)
            originals.extend((p-p0, q-q2))
    leaves = tuple(x.detach().requires_grad_(gradient) for x in originals)
    accumulated = [torch.zeros_like(x) for x in leaves] if gradient else None
    spectrum = []
    for wave, index, port in zip(waves, indices, ports):
        if args.material == 'PMMA':
            ps, qs = [], []
            for layer in range(args.layers):
                disk, normal = leaves[2*layer:2*layer+2]
                eps = identity+(index**2-1)*disk
                rule = identity+(1/index**2-1)*disk
                p, q, _ = d6.reduced_pq(port, eps, rule, normal, context)
                ps.append(p)
                qs.append(q)
            zero = port._eye(context.reduced_dimension)*0
            R, T = port.response(ps, qs, zero, zero, 1.)
        else:
            R, T = port.response(leaves[0::2], leaves[1::2], p0, q2,
                                 (wave/args.period)**2)
        if gradient:
            derivatives = torch.autograd.grad(R/len(waves), leaves)
            for target, derivative in zip(accumulated, derivatives):
                target.add_(derivative)
            del derivatives
        spectrum.append([wave, float(R.detach()), float(T.detach())])
        if args.material == 'PMMA':
            del ps, qs, p, q, eps, rule
        del R, T
    derivative = torch.autograd.grad(tuple(originals), values,
                                   grad_outputs=tuple(accumulated))[0] if gradient else None
    return dict(spectrum=spectrum,
        gradient=derivative.detach().cpu().tolist() if derivative is not None else None,
        geometry_builds=args.layers, spectral_graphs_retained_at_once=1,
        material_inverse_or_PQ_shared_across_wavelengths=args.material != 'PMMA')

