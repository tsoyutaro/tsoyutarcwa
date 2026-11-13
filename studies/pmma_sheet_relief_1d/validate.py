"""Independent Fresnel/Airy checks, energy conservation, and symmetry checks."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from model import HERE, Materials, geometry, layers, save_json, solver_root
from optics import admittance, average_sheet, phase_sample, propagating


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--solver-root', type=Path)
    p.add_argument('--output-dir', type=Path, default=HERE/'results/validation')
    args = p.parse_args()
    solver_root(args.solver_root)
    import torch
    from solver import local_surface
    torch.set_num_threads(2)
    checks = []
    def record(name, error, tolerance):
        checks.append(dict(name=name, error=float(error), tolerance=tolerance, passed=bool(error <= tolerance)))
        print(name, error, 'PASS' if error <= tolerance else 'FAIL', flush=True)

    g, wavelength, npmma, nw = geometry(), 550., 1.5, 1.333
    orders, rf, tf = local_surface(g, [], wavelength, npmma, npmma, nw, 16)
    z = int(np.where(orders == 0)[0][0])
    expected_r = (npmma-nw)/(npmma+nw)
    expected_t = 2*npmma/(npmma+nw)
    record('flat PMMA-water amplitude vs Fresnel', max(abs(rf[:, z, z]-expected_r).max(), abs(tf[:, z, z]-expected_t).max()), 1e-10)
    r01, r12 = ((1-npmma)/(1+npmma))**2, expected_r**2
    r_expected = r01+(1-r01)**2*r12/(1-r01*r12)
    t_expected = (1-r01)*(1-r12)/(1-r01*r12)
    error = 0.
    for j, pol in enumerate(['s', 'p']):
        r, t, phase = average_sheet(rf[j], tf[j], orders, wavelength, g['period_nm'], npmma, nw, pol,
            points=256, max_points=4096, tolerance=1e-7)
        error = max(error, abs(r.sum()-r_expected), abs(t.sum()-t_expected))
    record('phase average vs analytic incoherent planar sheet', error, 2e-7)

    ng = Materials().gold(wavelength)
    film = [dict(z_top_nm=0., z_bottom_nm=50., core_width_nm=0., outer_width_nm=g['period_nm'])]
    orders, rf, tf = local_surface(geometry(gold=50), film, wavelength, npmma, ng, nw, 16)
    ph = np.exp(2j*np.pi*ng*50/wavelength)
    ra, rb = (npmma-ng)/(npmma+ng), (ng-nw)/(ng+nw)
    r_exact = (ra+rb*ph**2)/(1+ra*rb*ph**2)
    t_exact = (2*npmma/(npmma+ng))*(2*ng/(ng+nw))*ph/(1+ra*rb*ph**2)
    record('uniform Au 50nm amplitudes vs independent Airy formula',
           max(abs(rf[:, z, z]-r_exact).max(), abs(tf[:, z, z]-t_exact).max()), 1e-9)

    # An arbitrary local incoming PMMA order must conserve flux in a lossless relief.
    g = geometry(period=3920.)
    orders, rf, tf = local_surface(g, layers(g, 40), wavelength, npmma, npmma, 1., 18)
    kx = orders*wavelength/g['period_nm']
    energy_error = symmetry_error = 0.
    for j, pol in enumerate(['s', 'p']):
        yp, yw = admittance(npmma, kx, pol), admittance(nw, kx, pol)
        ypflux = np.where(propagating(npmma, kx), yp.real, 0)
        ywflux = np.where(propagating(nw, kx), yw.real, 0)
        flux = (abs(rf[j])**2*ypflux[:, None]).sum(axis=0)+(abs(tf[j])**2*ywflux[:, None]).sum(axis=0)
        mask = propagating(npmma, kx)
        energy_error = max(energy_error, abs(flux[mask]/ypflux[mask]-1).max())
        symmetry_error = max(symmetry_error, abs(rf[j]-rf[j, ::-1, ::-1]).max(), abs(tf[j]-tf[j, ::-1, ::-1]).max())
        phases = np.random.default_rng(17).uniform(0, 2*np.pi, (32, len(set(abs(orders[mask])))))
        r, t = phase_sample(rf[j], tf[j], orders, wavelength, g['period_nm'], npmma, nw, pol, phases)
        record(f'full sheet phase samples conserve energy ({pol})', abs(r.sum()+t.sum()-1), 1e-9)
        record(f'+/- orders retain exact degeneracy ({pol})', max(abs(r-r[::-1]).max(), abs(t-t[::-1]).max()), 1e-10)
    record('lossless local grating all propagating incoming channels', energy_error, 1e-8)
    record('local scattering preserves mirror symmetry', symmetry_error, 1e-8)

    # Exactly a vacuum Rayleigh cutoff: ensure the auxiliary reference is regular.
    g = geometry()
    orders, rf, tf = local_surface(g, layers(g, 24), 620., npmma, npmma, nw, 16)
    kx = orders*620/g['period_nm']
    for j, pol in enumerate(['s', 'p']):
        r, t, _ = average_sheet(rf[j], tf[j], orders, 620., g['period_nm'], npmma, nw, pol,
            points=128, max_points=1024, tolerance=1e-3)
        record(f'exact air cutoff 620nm: finite and zero grazing flux ({pol})',
               max(abs(r[abs(orders) == 6]).max(), abs(r.sum()+t.sum()-1)), 1e-8)

    g = geometry(gold=50.)
    record('directional Au geometry side-normal thickness', abs(g['gold_normal_nm']-29.74215027051473), 1e-12)
    save_json(args.output_dir/'checks.json', dict(passed=all(r['passed'] for r in checks), checks=checks))
    return 0 if all(r['passed'] for r in checks) else 1


if __name__ == '__main__':
    raise SystemExit(main())
