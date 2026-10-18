"""Geometry, independent optical identities, and optional small RCWA checks."""
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np

from geometry import build_layers, cap_count, material_codes, nearest_radius, side_thickness
from model import HERE, load_materials, parser, surface_rcwa, validate_args
from stack import finite_sample, interface, silica_n


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--integration', action='store_true')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--output-dir', type=Path, default=HERE/'results/validation')
    opts = p.parse_args()
    checks = []
    def check(name, error=0., tolerance=1e-10):
        if not math.isfinite(float(error)) or error > tolerance:
            raise AssertionError((name, error, tolerance))
        checks.append({'name': name, 'passed': True, 'error': float(error), 'tolerance': tolerance})
        print('PASS '+name+f' ({error:.3g})', flush=True)
    for mode in ['evaporation', 'sputter']:
        args = parser(mode).parse_args([])
        validate_args(args)
        layers = build_layers(args)
        body = [a for a in layers if a.z_top_nm <= args.height_nm+1e-9]
        cap = [a for a in layers if a.z_bottom_nm >= args.height_nm-1e-9]
        check(mode+': 150 relief plus 23 cap layers', abs(len(body)-150)+abs(len(cap)-23))
        check(mode+': relief height remains 200 nm', abs(sum(a.thickness_nm for a in body)-200.))
        check(mode+': total height is 230 nm', abs(sum(a.thickness_nm for a in layers)-230.))
        check(mode+': valley boundary preserved', min(abs(a.z_bottom_nm-30.) for a in layers))
        if mode == 'evaporation':
            check('Evaporation radial thickness', abs(side_thickness(args)-3.6))
            r = np.linspace(.013, 65.013, 1279)
            surface_height = np.where(r <= 25., 200., np.maximum(0., (49.-r)/.12))
            errors = 0
            for layer in layers:
                z = (layer.z_bottom_nm+layer.z_top_nm)/2
                expected = np.where(z <= surface_height, 1,
                                    np.where(z < surface_height+30., 2, 0))
                errors += np.count_nonzero(material_codes(layer, r) != expected)
            check('Directional geometry equals exact 30 nm vertical surface lift', errors)
            check('Directional coating has no isolated-shell overlap', sum(a.kind.startswith('coalesced') for a in layers))
        else:
            assert any(a.kind == 'coalesced-shell' for a in layers)
            check('Sputter uses periodic union where shells meet')
            assert all(a.kind == 'valley' for a in layers if a.z_top_nm <= 30.+1e-9)
            check('Valley retains resin cores inside surrounding Au')
    d = nearest_radius(64)
    check('Nearest-image metric at center and across cell boundaries',
          max(abs(d[32,32]), abs(d[0,0]-.5), abs(d[16,16]-math.sqrt(3)/4), abs(d[0,32]-.5)))
    # Independent exact Airy slab, with absorbing n and lossless external ports.
    n, wl, thickness = 1.57+.012j, 600., 1234.
    rf = (1-n)/(1+n)
    phase = np.exp(2j*np.pi*n*thickness/wl)
    den = 1-rf*rf*phase*phase
    r = (rf-rf*phase*phase)/den
    t = 4*n/(1+n)**2*phase/den
    blocks = [np.eye(2)*s for s in interface(1., n)]
    actual = finite_sample(blocks, n, [(n, thickness, False)], wl, 1)
    check('Jones stack agrees with absorbing Airy slab', max(abs(actual['R']-abs(r)**2), abs(actual['T']-abs(t)**2)))
    n = 1.57
    reflection = ((1-n)/(1+n))**2
    actual = finite_sample([np.eye(2)*s for s in interface(1., n)], n, [(n, 54800., True)], wl, 16)
    check('Incoherent dielectric slab agrees with exact multiple-reflection formula',
          max(abs(actual['R']-2*reflection/(1+reflection)), abs(actual['T']-(1-reflection)/(1+reflection))))
    # An arbitrary unitary polarization rotation must conserve total power.
    angle = .43
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    zero = np.zeros((2, 2), dtype=np.complex128)
    actual = finite_sample([rotation, zero, zero, rotation.T], 1., [(1., 1000., True)], wl, 16)
    check('Jones connection includes cross-polarized transmitted power', abs(actual['T']-1.)+abs(actual['R']))
    assert actual['cross_T'] > .1
    check('Cross-polarized channel remains explicit')
    args = parser('evaporation').parse_args([])
    functions, _ = load_materials(args, [400., 600., 800.])
    try:
        functions['PET'](900.)
    except ValueError:
        check('Material extrapolation rejected')
    else:
        raise AssertionError('Extrapolation accepted')
    if opts.integration:
        import torch
        sys.path.insert(0, str(args.solver_root))
        from rcwa_ext import AutoRCWA, Circle, GroupTheoryOptions, Lattice, LayerSpec, Material, NVMOptions, OutputSpec
        torch.set_num_threads(2)
        for mode in ['evaporation', 'sputter']:
            args = parser(mode).parse_args([])
            args.device, args.order, args.slices, args.grid = opts.device, 2, 12, 96
            functions, _ = load_materials(args, [600.])
            indices = {key: f(600.) for key, f in functions.items()}
            blocks, diagnostic = surface_rcwa(args, 600., indices)
            tail = [(indices['PET'], 54800., True), (1.5+0j, 20000., True), (silica_n(600.), 1e6, True)]
            actual = finite_sample(blocks, indices['PET'], tail, 600., 16)
            assert all(-1e-8 <= actual[key] <= 1+1e-8 for key in ['R', 'T', 'A']) and actual['A'] > .05
            check(mode+': passive Au shell/valley/cap RCWA and finite stack')
            refined = finite_sample(blocks, indices['PET'], tail, 600., 32)
            check(mode+': phase quadrature 16 versus 32', max(abs(actual[k]-refined[k]) for k in ['R', 'T', 'A']))
            if mode == 'sputter':
                assert diagnostic['coalesced_raster_layers'] > 0 and diagnostic['basis'] == 'full-rectangular'
                check('Sputter integration exercises the coalesced raster backend')
            else:
                blocks_y, _ = surface_rcwa(args, 600., indices, source_pol='y')
                check('D6 source x/y equality for Au core-shell layers', max(np.max(abs(a-b)) for a,b in zip(blocks, blocks_y)), 1e-8)
            lossless = {key: complex(value.real, 0) for key, value in indices.items()}
            for key in ['Au_top', 'Au_side', 'Au_valley']:
                lossless[key] = 1.8+0j
            blocks, _ = surface_rcwa(args, 600., lossless)
            tail = [(lossless['PET'], 54800., True), (1.5+0j, 20000., True), (silica_n(600.), 1e6, True)]
            actual = finite_sample(blocks, lossless['PET'], tail, 600., 16)
            check(mode+': independent lossless energy balance', abs(actual['A']), 1e-8)
        # Removing all gold must recover an independently assembled bare relief.
        args = parser('evaporation').parse_args([])
        args.device, args.order, args.slices, args.grid = opts.device, 2, 12, 96
        args.top_gold_nm = args.valley_gold_nm = args.side_gold_nm = 0.
        functions, _ = load_materials(args, [600.])
        indices = {key: f(600.) for key, f in functions.items()}
        blocks, _ = surface_rcwa(args, 600., indices)
        sim = AutoRCWA(freq=100./600., order=[2,2], lattice=Lattice.triangular(1.),
            outputs=OutputSpec(smatrix_size='full', fields='none'), nvm=NVMOptions(grid=(96,96)),
            group_theory=GroupTheoryOptions(enabled=True, symmetry='d6', polarization='x', strict=True),
            dtype=torch.complex128, device=opts.device)
        sim.add_input_layer(eps=1.); sim.add_output_layer(eps=indices['PET']**2)
        sim.set_incident_angle(0.,0.)
        with torch.no_grad():
            for j in range(12):
                radius=(25.+24.*(j+.5)/12)/100.
                sim.add_structured_layer(LayerSpec(200./12/100., Circle(radius), Material(1.), Material(indices['resin']**2), method='nvm'))
            sim.solve_global_smatrix()
        i = sim.order_N//2
        check('Zero-Au limit agrees with independently assembled uncoated RCWA',
              max(abs(block[0,0]-complex(s[i,i].detach().cpu())) for block,s in zip(blocks, sim.S)), 1e-9)
    opts.output_dir.mkdir(parents=True, exist_ok=True)
    (opts.output_dir/'checks.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    print(f'All {len(checks)} checks passed.', flush=True)


if __name__ == '__main__':
    main()
