"""Normal illumination: air / incoherent thick PMMA / 1D relief / water."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from model import (HERE, Materials, geometry, geometry_for_args, layers, parse_wavelengths, save_csv,
                   save_json, solver_root)
from optics import admittance, angles, average_sheet, order_rows, propagating


def parser(coating=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--solver-root', type=Path)
    p.add_argument('--coating', choices=['uncoated', 'gold', 'both'], default=coating or 'uncoated')
    p.add_argument('--voids', choices=['air', 'water', 'both'], default='both')
    p.add_argument('--wavelengths', default='400:700:5', help='nm: start:stop:step or comma list')
    p.add_argument('--periods', default='3720,3920', help='nm; scaled mode sets base=period, tip=height=period*15/37.2')
    p.add_argument('--geometry-mode',choices=['scaled','legacy'],default='scaled',help='scaled: base=period, tip=height=period*15/37.2')
    p.add_argument('--base-width', type=float, help='legacy geometry only')
    p.add_argument('--tip-width', type=float, help='legacy geometry only')
    p.add_argument('--height', type=float, help='legacy geometry only')
    p.add_argument('--gold-nm', type=float, default=50., help='Directional deposition displacement in z, not side-normal thickness')
    p.add_argument('--order', type=int, default=24, help='M; retained orders -M..+M, My=0')
    p.add_argument('--slices', type=int, default=120, help='maximum layer thickness = height/slices')
    p.add_argument('--phase-points', type=int, default=512)
    p.add_argument('--phase-max', type=int, default=65536)
    p.add_argument('--phase-tol', type=float, default=2e-4)
    p.add_argument('--device', default='cpu',help='cpu or cuda[:index]')
    p.add_argument('--acceleration',choices=['fast','reference'],default='fast')
    p.add_argument('--phase-acceleration',choices=['auto','prepared','reference'],default='auto')
    p.add_argument('--geometry-cache',choices=['on','off'],default='on')
    p.add_argument('--cache-dir',type=Path,help='Shared content-addressed point cache (default results/v2/cache)')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--allow-unconverged-phase', action='store_true', help='Save provisional results and exit successfully')
    p.add_argument('--preview-only', action='store_true')
    p.add_argument('--no-plot', action='store_true')
    p.add_argument('--no-preview', action='store_true', help='Skip geometry rendering for repeated convergence runs')
    return p


def source_signature(root):
    paths = [*HERE.glob('*.py'), root/'studies/gold_grating_1d/solver.py', *sorted((root/'rcwa_ext').glob('*.py'))]
    return hashlib.sha256(b''.join(p.read_bytes() for p in sorted(paths))).hexdigest()


def calculate(args, g, coating, void, wavelength, material, stack=None):
    from solver import local_surface,local_surface_fast,clear_plan_cache
    started = time.perf_counter()
    n_pmma, pmma_source = material.pmma(wavelength)
    n_gold = material.gold(wavelength) if coating == 'gold' else complex(n_pmma)
    min_order = int(np.floor(n_pmma*g['period_nm']/wavelength))
    if args.order < min_order:
        raise ValueError(f'M={args.order} omits propagating PMMA orders up to {min_order}; increase --order.')
    stack = layers(g, args.slices) if stack is None else stack
    acceleration=getattr(args,'acceleration','reference')
    if getattr(args,'geometry_cache','on')=='off':clear_plan_cache()
    surface=local_surface_fast if acceleration=='fast' else local_surface
    orders, rf, tf = surface(g, stack, wavelength, n_pmma, n_gold,
                                   1. if void == 'air' else 1.333, args.order, args.device)
    kx = orders*wavelength/g['period_nm']
    zero = int(np.where(orders == 0)[0][0])
    summaries, rows, local = [], [], []
    for j, pol in enumerate(['s', 'p']):
        phase_acceleration=getattr(args,'phase_acceleration','auto')
        if phase_acceleration=='auto':phase_acceleration='reference' if acceleration=='reference' else 'prepared'
        r, t, phase = average_sheet(rf[j], tf[j], orders, wavelength, g['period_nm'], n_pmma, 1.333, pol,
            points=args.phase_points, max_points=args.phase_max, tolerance=args.phase_tol,acceleration=phase_acceleration)
        a = float(1-r.sum()-t.sum())
        cone = abs(kx) <= np.sin(np.radians(20))+1e-12
        if min(r.min(), t.min(), a) < -1e-7:
            raise RuntimeError(f'Passivity failure at {wavelength:g} nm ({coating}/{void}/{pol}): A={a:g}.')
        labels = dict(coating=coating, void=void, period_nm=g['period_nm'], polarization=pol)
        summaries.append(dict(labels, wavelength_nm=float(wavelength), order=args.order, slices=args.slices,
            actual_layers=len(stack), n_PMMA=n_pmma, PMMA_source=pmma_source,
            Au_n=n_gold.real if coating == 'gold' else '', Au_k=n_gold.imag if coating == 'gold' else '',
            R=float(r.sum()), T=float(t.sum()), A_total=a,
            R_0=float(r[zero]), R_diffraction=float(r.sum()-r[zero]),
            R_le_20deg=float(r[cone].sum()), R_gt_20deg=float(r[~cone].sum()),
            R_max_propagating_m=int(np.max(abs(orders[propagating(1., kx)]))),
            T_max_propagating_m=int(np.max(abs(orders[propagating(1.333, kx)]))),
            PMMA_max_propagating_m=int(np.max(abs(orders[propagating(n_pmma, kx)]))),
            phase_points_per_scramble=phase['points_per_scramble'], phase_scrambles=2,
            phase_error=phase['maximum_change'], phase_converged=phase['converged'],
            runtime_seconds=time.perf_counter()-started,
            discretization_converged=False,geometry_mode=getattr(args,'geometry_mode','explicit'),
            base_width_nm=g['base_width_nm'],tip_width_nm=g['tip_width_nm'],height_nm=g['height_nm'],
            gold_vertical_nm=g['gold_vertical_nm'],acceleration=acceleration,
            eigen_dimension_per_polarization=args.order+1 if acceleration=='fast' else 2*args.order+1,
            phase_acceleration=phase_acceleration))
        rows.extend(order_rows(r, t, orders, wavelength, g['period_nm'], n_pmma, 1.333, **labels))
        # Diagnostic response at the local PMMA port before the front-surface feedback.
        yp = admittance(n_pmma, kx, pol)
        yw = admittance(1.333, kx, pol)
        rl = abs(rf[j, :, zero])**2*np.where(propagating(n_pmma, kx), yp.real, 0)/yp[zero].real
        tl = abs(tf[j, :, zero])**2*np.where(propagating(1.333, kx), yw.real, 0)/yp[zero].real
        local.extend([dict(labels, wavelength_nm=float(wavelength), m=int(m),
                           R_PMMA_single_pass=float(rv), T_water_single_pass=float(tv),
                           A_local_total=float(1-rl.sum()-tl.sum()))
                      for m, rv, tv in zip(orders, rl, tl)])
    return dict(summary=summaries, orders=rows, local=local)


def plot_results(output, summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups = sorted(set((r['coating'], r['period_nm'], r['void']) for r in summaries))
    for coating, period, void in groups:
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
        for pol, style in [('s', '-'), ('p', '--')]:
            rr = sorted([r for r in summaries if (r['coating'], r['period_nm'], r['void'], r['polarization']) ==
                         (coating, period, void, pol)], key=lambda r: r['wavelength_nm'])
            wl = [r['wavelength_nm'] for r in rr]
            for key, color in [('R', '#255d98'), ('T', '#188376'), ('A_total', '#b66c15')]:
                axes[0].plot(wl, [100*r[key] for r in rr], style, color=color, label=pol+': '+key)
            axes[1].plot(wl, [100*r['R_le_20deg'] for r in rr], style, color='#255d98', label=pol+': <=20 deg')
            axes[1].plot(wl, [100*r['R_gt_20deg'] for r in rr], style, color='#a34462', label=pol+': >20 deg')
            axes[2].plot(wl, [r['phase_error'] for r in rr], style, label=pol)
        for ax in axes:
            ax.set_xlabel('Vacuum wavelength (nm)')
            ax.grid(alpha=.25)
            ax.legend(fontsize=7)
        axes[0].set_ylabel('Incident power (%)')
        axes[1].set_ylabel('Reflected power (%)')
        axes[2].set_ylabel('Phase quadrature change (fraction)')
        fig.suptitle(f'{coating}; P={period/1000:g} um; {void} voids / water; provisional M,Nz')
        fig.tight_layout()
        fig.savefig(output/f'spectrum_{coating}_P{period:g}_{void}.png', dpi=170)
        plt.close(fig)


def main(argv=None, coating=None):
    args = parser(coating).parse_args(argv)
    root = solver_root(args.solver_root)
    import torch
    from solver import plan_cache_info
    from fast_solver import canonical_device
    args.device=str(canonical_device(args.device))
    torch.set_num_threads(args.threads)
    if args.order < 1 or args.slices < 1 or args.threads < 1 or not np.isfinite(args.phase_tol) or args.phase_tol <= 0:
        raise ValueError('Positive order, slices, threads and phase tolerance required.')
    periods = [float(v) for v in args.periods.split(',')]
    coatings = ['uncoated', 'gold'] if args.coating == 'both' else [args.coating]
    voids = ['air', 'water'] if args.voids == 'both' else [args.voids]
    for c in coatings:
        for period in periods:geometry_for_args(args,period,c)
    output = (args.output_dir or HERE/'results/v2'/args.geometry_mode/args.coating).resolve()
    output.mkdir(parents=True, exist_ok=True)
    material = Materials()
    signature = source_signature(root)
    material_identity={k:v['sha256'] for k,v in material.metadata().items()}
    settings = dict(vars(args), solver_root=str(root), output_dir=str(output),
                    material=material.metadata(), source_sha256=signature,
                    version=2,cache_dir=str(args.cache_dir or HERE/'results/v2/cache'),
                    pmma_ratio=[37.2,15,15],base_tracks_period=args.geometry_mode=='scaled',gold_thickness_scales_with_period=False,
                    geometries={f'{c}/P{p:g}':geometry_for_args(args,p,c) for c in coatings for p in periods},
                    torch=str(torch.__version__), method='1D Li; exact mirror-even reduction; shared analytic interval plan; prepared Sobol phase average',
                    output_index=1.333, incident_index=1., incidence_deg=0.,
                    phase_assumption='independent uniform one-way phases for each |m|; +/-m share the same phase',
                    absorption='total absorption only; outgoing-order absorption is not uniquely defined')
    settings['solver_root'] = str(root)
    settings['output_dir'] = str(output)
    if args.solver_root is not None:
        settings['solver_root'] = str(args.solver_root)
    if args.cache_dir is not None:settings['cache_dir']=str(args.cache_dir)
    if (output/'settings.json').exists():
        previous=json.loads((output/'settings.json').read_text(encoding='utf-8'))
        if previous.get('version')!=2 or previous.get('geometries')!=settings['geometries']:
            raise ValueError('Output contains a different geometry/version. Choose a new --output-dir.')
    save_json(output/'settings.json', settings)
    if not args.no_preview:
        from preview import preview_all
        preview_all(output/'geometry', periods, args.base_width, args.tip_width, args.height, args.gold_nm, args.slices,geometry_mode=args.geometry_mode)
    if args.preview_only:
        return 0
    summaries, allorders, local = [], [], []
    phase_failed = False
    point_cache_hits=0
    for name in ['spectra.csv','diffraction_orders.csv','local_single_pass_orders.csv']:
        if (output/name).exists():(output/name).unlink()
    def append_csv(path,rows):
        import csv
        first=not path.exists()
        with path.open('a',encoding='utf-8-sig' if first else 'utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            if first:writer.writeheader()
            writer.writerows(rows)
    for coating_name in coatings:
        for period in periods:
            g = geometry_for_args(args,period,coating_name)
            stack=layers(g,args.slices)
            for void in voids:
                for wave in parse_wavelengths(args.wavelengths):
                    cache_parameters = dict(g=g, void=void, coating=coating_name, wavelength=float(wave),
                        order=args.order, slices=args.slices, phase_points=args.phase_points,
                        phase_max=args.phase_max, phase_tol=args.phase_tol, source=signature,material=material_identity,
                        geometry_mode=args.geometry_mode,acceleration=args.acceleration,phase_acceleration=args.phase_acceleration,
                        device=args.device,torch=str(torch.__version__),numpy=str(np.__version__))
                    key = hashlib.sha256(json.dumps(cache_parameters, sort_keys=True).encode()).hexdigest()
                    cache = (args.cache_dir or HERE/'results/v2/cache')/f'{key}.json'
                    if cache.exists():
                        result = json.loads(cache.read_text(encoding='utf-8'))
                        point_cache_hits+=1
                    else:
                        result = calculate(args, g, coating_name, void, float(wave), material,stack=stack)
                        save_json(cache, result)
                    summaries.extend(result['summary'])
                    allorders.extend(result['orders'])
                    local.extend(result['local'])
                    phase_failed |= any(not r['phase_converged'] for r in result['summary'])
                    append_csv(output/'spectra.csv', result['summary'])
                    append_csv(output/'diffraction_orders.csv', result['orders'])
                    append_csv(output/'local_single_pass_orders.csv', result['local'])
                    print(f'{coating_name} P={period:g} {void} {wave:g} nm: '+
                          ', '.join(f"{r['polarization']} R={r['R']:.6f} T={r['T']:.6f} A={r['A_total']:.3g} phase={r['phase_error']:.2g}" for r in result['summary']), flush=True)
    if not args.no_plot:
        plot_results(output, summaries)
    save_json(output/'completion.json', dict(completed_cases=len(summaries), phase_converged=not phase_failed,
        discretization_converged=False, status='preliminary_spectrum_pending_order_and_layer_convergence',
        point_cache_hits=point_cache_hits,geometry_plan_cache=plan_cache_info()))
    return 2 if phase_failed and not args.allow_unconverged_phase else 0


if __name__ == '__main__':
    sys.exit(main())
