"""Validate merged-shell D6 operators against full native-star equations."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from rcwa_ext import AutoRCWA

from geometry import nearest_radius, material_codes
from memory import connect_full
from model import parser, surface_rcwa
from native_d6 import surface_native_d6
from native_d6_union import make_geometry, NativeUnionPort, scalar_pq


def dense_operators(port, eps):
    """Unreduced Cartesian Maxwell equations on the same Fourier star."""
    n = eps.shape[0]
    identity = port._eye(n)
    zero = torch.zeros_like(identity)
    kx, ky = torch.diag(port.kx), torch.diag(port.ky)
    left = torch.cat((kx, ky), dim=0)
    p = torch.cat((torch.cat((zero, identity), 1), torch.cat((-identity, zero), 1)), 0)
    p = p + left @ port._solve(eps, torch.cat((ky, -kx), dim=1))
    q = torch.cat((torch.cat((zero, -eps), 1), torch.cat((eps, zero), 1)), 0)
    q = q + left @ torch.cat((-ky, kx), dim=1)
    return p, q


def full_admittance(port, eps):
    view = SimpleNamespace(Kx_norm_dn=port.kx, Ky_norm_dn=port.ky,
                           _dtype=port._dtype, _device=port._device)
    kz = port._positive_kz(eps-port.kx**2-port.ky**2)
    return AutoRCWA._cartesian_e_to_h(view, kz)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    cli.add_argument('--order', type=int, default=3)
    cli.add_argument('--grid', type=int, default=96)
    opts = cli.parse_args()
    torch.set_num_threads(2)
    errors = []

    def check(name, error, tolerance=1e-9):
        if not np.isfinite(error) or error > tolerance:
            raise AssertionError((name, error, tolerance))
        errors.append(error)
        print(f'PASS {name}: {error:.3g}', flush=True)

    with torch.no_grad():
        for side in (8, 12):
            args = parser('evaporation').parse_args([
                '--native-d6', '--side-gold-nm', str(side), '--order', str(opts.order),
                '--grid', str(opts.grid), '--slices', '20', '--device', opts.device])
            shapes = make_geometry(args, args.slices, args.grid)
            shapes.prepare(args, opts.order)
            port = NativeUnionPort(shapes, 100/550)
            plan = shapes.plan
            zero = int(torch.nonzero((plan.m == 0) & (plan.n == 0), as_tuple=True)[0].item())
            for layer in shapes.layers:
                if not layer.kind.startswith('coalesced'):
                    continue
                eps, _ = shapes.materials(layer, 2.4, -10+2j)
                # Independently use the shared rectangular FFT convolution
                # and select its native star, before taking any inverse.
                code = material_codes(layer, nearest_radius(opts.grid)*args.period_nm)
                values = np.where(code == 1, 2.4, np.where(code == 2, -10+2j, 1.))
                full = shapes.template._material_conv(torch.as_tensor(
                    values, dtype=port._dtype, device=port._device))
                reference = full.index_select(0, plan.indices).index_select(1, plan.indices)
                check(f'{side} nm union Fourier inputs', float((reference-eps).abs().max()))
                p_full, q_full = dense_operators(port, reference)
                p, q = scalar_pq(port, eps)
                p_expected = plan.electric.mH @ p_full @ plan.magnetic
                q_expected = plan.magnetic.mH @ q_full @ plan.electric
                check(f'{side} nm direct E1 P/Q',
                      max(float((p-p_expected).abs().max()), float((q-q_expected).abs().max())), 1e-8)
                depth = layer.thickness_nm / args.period_nm
                v_ref, v_out = full_admittance(port, 1.), full_admittance(port, 2.4)
                squared, electric = port._eig(p_full @ q_full)
                kz = port._positive_kz(squared)
                magnetic = port._magnetic_eigenvectors(p_full, q_full, electric, kz)
                full_layer = port._reduced_layer_smatrix(
                    electric, magnetic, kz, torch.as_tensor(depth, dtype=port._dtype, device=port._device), v_ref)
                full_s = connect_full(port, port._reduced_interface_s(v_ref, v_ref, input_side=True), full_layer)
                full_s = connect_full(port, full_s, port._reduced_interface_s(v_ref, v_out, input_side=False))
                reduced_layer = port.layer_scattering(eps, None, shapes, depth)
                reduced_s = connect_full(port, port._reduced_interface_s(
                    port.reference_v, port.reference_v, input_side=True), reduced_layer)
                reduced_s = connect_full(port, reduced_s, port._reduced_interface_s(
                    port.reference_v, port.admittance(2.4), input_side=False))
                check(f'{side} nm all four complex S blocks',
                      max(float((a[zero, zero] - torch.vdot(shapes.source, b@shapes.source)).abs())
                          for a, b in zip(full_s, reduced_s)))
            indices = {'PET': 1.+0j, 'resin': np.sqrt(2.4)+0j,
                       'Au_top': 1.8+0j, 'Au_side': 1.8+0j, 'Au_valley': 1.8+0j}
            for wl in (400., 550., 700.):
                result, _ = surface_rcwa(args, wl, indices, shapes=shapes)
                r, t = abs(result[1][0, 0])**2, abs(result[0][0, 0])**2
                check(f'{side} nm lossless R+T at {wl:g} nm', abs(1-r-t), 1e-8)

        # The existing 3.6 nm entry still dispatches to its original solver.
        args = parser('evaporation').parse_args([
            '--native-d6', '--order', str(opts.order), '--grid', str(opts.grid),
            '--slices', '12', '--device', opts.device])
        shapes = make_geometry(args, args.slices, args.grid)
        indices = {'PET': 1.57+0j, 'resin': 1.57+0j,
                   'Au_top': .5+4j, 'Au_side': .5+4j, 'Au_valley': .5+4j}
        before, _ = surface_native_d6(args, 550., indices, args.order, args.slices, shapes)
        after, _ = surface_rcwa(args, 550., indices, shapes=shapes)
        check('3.6 nm entry parity', max(float(np.abs(a-b).max()) for a, b in zip(before, after)))
    print(f'All {len(errors)} checks passed; maximum absolute error {max(errors):.3g}.')


if __name__ == '__main__':
    main()
