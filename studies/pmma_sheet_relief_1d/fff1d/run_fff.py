"""Normal illumination: air / incoherent thick PMMA / 1D relief / water."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import (HERE, Materials, geometry_for_args, layers, parse_wavelengths, save_csv,
                   save_json, solver_root)
from optics import admittance, angles, average_sheet, order_rows, propagating
from run import parser as legacy_parser, source_signature as legacy_signature
from fff1d.operators import angle_parameters


def parser(coating=None):
    p = legacy_parser(coating)
    p.add_argument('--polarization', choices=['s', 'p', 'both'], default='p')
    p.add_argument('--theta-deg',type=float,help='FFF projection angle; 90 recovers lamellar RCWA, not incidence angle')
    p.add_argument('--gil-nm',type=float,default=0.,help='total horizontal GIL width in nm')
    p.add_argument('--gil-interfaces',choices=['inner','outer','both'],default='both')
    p.add_argument('--gil-quadrature',type=int,default=16)
    p.add_argument('--transfer-norm',type=float,default=4.)
    p.add_argument('--symmetry',choices=['even','full'],default='even')
    p.add_argument('--section-sampling',choices=['midpoint','entrance'],default='midpoint')
    p.add_argument('--passivity-tolerance',type=float,default=1e-7)
    p.set_defaults(periods='3720',wavelengths='700',slices=960,coating=coating or 'gold',phase_tol=5e-5,phase_max=262144)
    return p


def source_signature(root, polarization='both'):
    original = legacy_signature(root)
    paths = sorted(Path(__file__).resolve().parent.glob('*.py'))
    return hashlib.sha256(original.encode()+b''.join(p.read_bytes() for p in paths)).hexdigest()


def selected_polarizations(value):
    return ['s', 'p'] if value == 'both' else [value]


def calculate(args, g, coating, void, wavelength, material, stack=None):
    from fff1d.solver import local_surface_fff
    from fff1d.operators import clear_fff_cache
    started = time.perf_counter()
    n_pmma, pmma_source = material.pmma(wavelength)
    n_gold = material.gold(wavelength) if coating == 'gold' else complex(n_pmma)
    min_order = int(np.floor(n_pmma*g['period_nm']/wavelength))
    if args.order < min_order:
        raise ValueError(f'M={args.order} omits propagating PMMA orders up to {min_order}; increase --order.')
    stack = layers(g, args.slices) if stack is None else stack
    acceleration=getattr(args,'acceleration','reference')
    if getattr(args,'geometry_cache','on')=='off':clear_fff_cache()
    summaries, rows, local = [], [], []
    for pol in selected_polarizations(args.polarization):
        orders,rf,tf,diagnostic=local_surface_fff(g,stack,wavelength,n_pmma,n_gold,
            1. if void=='air' else 1.333,args.order,args.device,theta_deg=args.theta_deg,
            gil_nm=args.gil_nm,gil_interfaces=args.gil_interfaces,gil_quadrature=args.gil_quadrature,
            transfer_norm=args.transfer_norm,return_diagnostics=True,polarization=pol,symmetry=args.symmetry)
        j=0;kx=orders*wavelength/g['period_nm'];zero=int(np.where(orders==0)[0][0])
        phase_acceleration=getattr(args,'phase_acceleration','auto')
        if phase_acceleration=='auto':phase_acceleration='reference' if acceleration=='reference' else 'prepared'
        r, t, phase = average_sheet(rf[j], tf[j], orders, wavelength, g['period_nm'], n_pmma, 1.333, pol,
            points=args.phase_points, max_points=args.phase_max, tolerance=args.phase_tol,acceleration=phase_acceleration)
        a = float(1-r.sum()-t.sum())
        cone = abs(kx) <= np.sin(np.radians(20))+1e-12
        passive=bool(min(r.min(),t.min(),a)>=-args.passivity_tolerance)
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
            eigen_dimension_per_polarization=diagnostic['generator_dimension'],
            phase_acceleration=phase_acceleration,method='rcwa-fff-paper-left',
            theta_deg=diagnostic['theta_deg'],gil_nm=args.gil_nm,gil_interfaces=args.gil_interfaces,
            gil_quadrature=args.gil_quadrature,symmetry=args.symmetry,section_sampling=args.section_sampling,
            passivity_valid=passive,max_local_subsections=diagnostic['max_local_subsections']))
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
    return dict(summary=summaries, orders=rows, local=local,
                calculation_origin='rcwa_fff')


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
            if not rr:
                continue
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
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser(coating).parse_args(argv)
    if args.acceleration!='fast':raise ValueError('RCWA-FFF uses its own solver; --acceleration must be fast.')
    if not np.isfinite(args.gil_nm) or args.gil_nm<0 or args.gil_quadrature<2:
        raise ValueError('Require gil-nm >=0 and gil-quadrature >=2.')
    if not np.isfinite(args.passivity_tolerance) or args.passivity_tolerance<0:
        raise ValueError('Require a finite nonnegative passivity tolerance.')
    root = solver_root(args.solver_root)
    import torch
    from fff1d.operators import cache_info as plan_cache_info
    from fast_solver import canonical_device
    args.device=str(canonical_device(args.device))
    torch.set_num_threads(args.threads)
    if args.device=='cpu':
        try:
            from threadpoolctl import threadpool_limits
            thread_limit_guard=threadpool_limits(limits=args.threads)
        except ImportError:
            pass
    if args.order < 1 or args.slices < 1 or args.threads < 1 or not np.isfinite(args.phase_tol) or args.phase_tol <= 0:
        raise ValueError('Positive order, slices, threads and phase tolerance required.')
    periods = [float(v) for v in args.periods.split(',')]
    coatings = ['uncoated', 'gold'] if args.coating == 'both' else [args.coating]
    voids = ['air', 'water'] if args.voids == 'both' else [args.voids]
    for c in coatings:
        for period in periods:angle_parameters(geometry_for_args(args,period,c),args.theta_deg)
    output = (args.output_dir or HERE/'results/fff1d'/args.geometry_mode/args.coating/args.polarization).resolve()
    output.mkdir(parents=True, exist_ok=True)
    material = Materials()
    signature = source_signature(root, args.polarization)
    original_signature = legacy_signature(root)
    material_identity={k:v['sha256'] for k,v in material.metadata().items()}
    settings = dict(vars(args), solver_root=str(root), output_dir=str(output),
                    material=material.metadata(), source_sha256=signature,
                    version=3,cache_dir=str(args.cache_dir or HERE/'results/fff1d/cache'),
                    pmma_ratio=[37.2,15,15],base_tracks_period=args.geometry_mode=='scaled',gold_thickness_scales_with_period=False,
                    geometries={f'{c}/P{p:g}':geometry_for_args(args,p,c) for c in coatings for p in periods},
                    torch=str(torch.__version__), method='RCWA-FFF paper-left; frozen-section exponential; Redheffer cascade',
                    reference_legacy_source_sha256=original_signature,
                    normal_field='constant c^2; signed cs on the two sides of the periodic trapezoid',
                    gil_definition='linear complex index; centered total horizontal width; transverse metal interfaces only',
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
        relevant=['polarization','theta_deg','gil_nm','gil_interfaces','gil_quadrature','symmetry','section_sampling','transfer_norm']
        if previous.get('version')!=3 or previous.get('geometries')!=settings['geometries'] or any(previous.get(k)!=settings[k] for k in relevant):
            raise ValueError('Output contains a different geometry/version. Choose a new --output-dir.')
    save_json(output/'settings.json', settings)
    if not args.no_preview:
        from preview import preview_all
        preview_all(output/'geometry', periods, args.base_width, args.tip_width, args.height, args.gold_nm, args.slices,geometry_mode=args.geometry_mode)
    if args.preview_only:
        return 0
    summaries, allorders, local = [], [], []
    phase_failed = False
    passivity_failed = False
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
            if args.section_sampling=='entrance':
                from model import width
                stack=[dict(row,core_width_nm=width(row['z_top_nm']+1e-9,g),
                    outer_width_nm=width(row['z_top_nm']-g['gold_vertical_nm']+1e-9,g)) for row in stack]
            save_csv(output/'actual_layers'/f'{coating_name}_P{period:g}.csv',stack)
            for void in voids:
                for wave in parse_wavelengths(args.wavelengths):
                    cache_parameters = dict(g=g, void=void, coating=coating_name, wavelength=float(wave),
                        order=args.order, slices=args.slices, phase_points=args.phase_points,
                        phase_max=args.phase_max, phase_tol=args.phase_tol, source=signature,material=material_identity,
                        geometry_mode=args.geometry_mode,acceleration=args.acceleration,phase_acceleration=args.phase_acceleration,
                        device=args.device,torch=str(torch.__version__),numpy=str(np.__version__),
                        polarization=args.polarization,theta_deg=args.theta_deg,gil_nm=args.gil_nm,
                        gil_interfaces=args.gil_interfaces,gil_quadrature=args.gil_quadrature,
                        symmetry=args.symmetry,section_sampling=args.section_sampling,transfer_norm=args.transfer_norm,
                        passivity_tolerance=args.passivity_tolerance)
                    key = hashlib.sha256(json.dumps(cache_parameters, sort_keys=True).encode()).hexdigest()
                    cache = (args.cache_dir or HERE/'results/fff1d/cache')/f'{key}.json'
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
                    passivity_failed |= any(not r['passivity_valid'] for r in result['summary'])
                    append_csv(output/'spectra.csv', result['summary'])
                    append_csv(output/'diffraction_orders.csv', result['orders'])
                    append_csv(output/'local_single_pass_orders.csv', result['local'])
                    print(f'{coating_name} P={period:g} {void} {wave:g} nm: '+
                          ', '.join(f"{r['polarization']} R={r['R']:.6f} T={r['T']:.6f} A={r['A_total']:.3g} phase={r['phase_error']:.2g}" for r in result['summary']), flush=True)
    if not args.no_plot:
        plot_results(output, summaries)
    save_json(output/'completion.json', dict(completed_cases=len(summaries), phase_converged=not phase_failed,
        discretization_converged=False, status='preliminary_spectrum_pending_order_and_layer_convergence',
        point_cache_hits=point_cache_hits,
        polarization=args.polarization,geometry_plan_cache=plan_cache_info()))
    # A finite nonpassive truncation is retained and excluded from convergence;
    # it is never clipped or silently made passive.
    completion=json.loads((output/'completion.json').read_text(encoding='utf-8'))
    completion.update(passivity_valid=not passivity_failed,method='rcwa-fff-paper-left')
    save_json(output/'completion.json',completion)
    return 2 if phase_failed and not args.allow_unconverged_phase else 0


if __name__ == '__main__':
    sys.exit(main())
