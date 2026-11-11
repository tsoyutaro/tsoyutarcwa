"""Au-coated finite MOSMITE stack shared by evaporation and sputter entries."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np

from geometry import build_layers, cap_count, material_codes, nearest_radius, side_thickness, write_geometry
from memory import POLICY, PARITY_TOL, compare_reference, memory_safe_class, tensor_storage_bytes
from normalize_references import bare_mirror_r, bare_quartz_t, write_referenced_spectrum
from stack import finite_sample, load_nk, save_csv, silica_n, wavelengths, THEORY
from mosmite_phase_average import PHASE_AVERAGE_THEORY, phase_average

HERE = Path(__file__).resolve().parent


def parser(mode):
    p = argparse.ArgumentParser(description=__doc__+' Mode: '+mode)
    p.set_defaults(mode=mode)
    p.add_argument('--solver-root', type=Path, default=HERE.parents[1])
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--wavelengths', default='400:800:5')
    p.add_argument('--progress-every', type=int, default=10,
                   help='Refresh the cumulative spectra plots and print progress every N wavelengths.')
    p.add_argument('--mirror-nk-csv', type=Path,
                   help='Also save mirror- and quartz-referenced results at each progress update.')
    p.add_argument('--period-nm', type=float, default=100.)
    p.add_argument('--height-nm', type=float, default=200.)
    p.add_argument('--base-diameter-nm', type=float, default=98.)
    p.add_argument('--tip-diameter-nm', type=float, default=50.)
    p.add_argument('--top-gold-nm', type=float, default=30.)
    p.add_argument('--valley-gold-nm', type=float, default=30.)
    p.add_argument('--side-gold-nm', type=float,
                   help='RADIAL thickness; default directional 3.6 nm / phenomenological sputter 8 nm.')
    p.add_argument('--slices', type=int, default=150, help='Layers in the 200 nm polymer relief, INCLUDING the valley region.')
    p.add_argument('--cap-slices', type=int,
                   help='Additional cap layers; default ceil(slices*top_gold_nm/height_nm).')
    p.add_argument('--film-total-um', type=float, default=55., help='Uncoated film thickness; gold is additional.')
    p.add_argument('--residual-um', type=float, default=0.)
    p.add_argument('--adhesive-um', type=float, default=20., help='Set 0 for direct PET-glass contact.')
    p.add_argument('--adhesive-n', type=float, default=1.50)
    p.add_argument('--adhesive-k', type=float, default=0.)
    p.add_argument('--glass-um', type=float, default=1000.)
    p.add_argument('--pet-nk-csv', type=Path, default=HERE/'data/PET_Zhang_400_800nm.csv')
    p.add_argument('--resin-nk-csv', type=Path)
    p.add_argument('--adhesive-nk-csv', type=Path)
    p.add_argument('--gold-csv', type=Path, default=HERE/'data/Au_Johnson_1972.csv')
    for region in ['top', 'side', 'valley']:
        p.add_argument('--'+region+'-gold-csv', type=Path, help='Optional region-specific n,k; otherwise use --gold-csv.')
    p.add_argument('--order', type=int, default=7)
    p.add_argument('--grid', type=int, default=512)
    p.add_argument('--layer-storage', choices=['full', 'reduced'], default='full',
                   help='Reduced: release unused layer arrays and stream all four S blocks; no change to precision or physics.')
    p.add_argument('--reuse-geometry', action='store_true',
                   help='Reuse fixed Fourier shape coefficients, normal projectors and raster masks across wavelengths.')
    p.add_argument('--native-d6', action='store_true',
                   help='Build P/Q directly on D6 E1 and stream the full two-sided S. Includes geometry reuse.')
    p.add_argument('--resume', action='store_true',
                   help='Reuse completed wavelengths from the same output directory after checking settings and sources.')
    p.add_argument('--reference-dir', type=Path,
                   help='Optional completed original run at identical conditions; require R/T/A and polarization parity within 1e-8.')
    p.add_argument('--basis', choices=['auto', 'full'], default='auto',
                   help='Auto: D6 source only if all layers are eligible; otherwise full rectangular harmonics.')
    p.add_argument('--coherence', choices=['incoherent', 'coherent', 'phase-average'], default='incoherent',
                   help='Analytic intensity balance, fixed coherent phase, or legacy phase quadrature.')
    p.add_argument('--coherent-layers', default='',
                   help='Keep fixed phase in selected uniform layers: PET,adhesive,glass.')
    p.add_argument('--phase-points', type=int, default=16,
                   help='Phase samples per selected layer in phase-average mode (default: 16).')
    p.add_argument('--convergence-tol', type=float, default=1e-4)
    p.add_argument('--check-convergence', action='store_true')
    p.add_argument('--check-wavelengths', default='400,600,800')
    p.add_argument('--preview-only', action='store_true', help='Write exact layer table, geometry plot and settings; skip RCWA.')
    p.add_argument('--output-dir', type=Path)
    return p


def validate_args(args):
    if args.progress_every <= 0:
        raise ValueError('--progress-every must be positive.')
    if args.resume and args.preview_only:
        raise ValueError('--resume cannot be combined with --preview-only.')
    if {v.strip().lower() for v in args.coherent_layers.split(',') if v.strip()} - {'pet', 'adhesive', 'glass'}:
        raise ValueError('Unknown --coherent-layers entry; use PET,adhesive,glass.')
    for name, value in vars(args).items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f'{name} must be finite.')
    if min(args.height_nm, args.period_nm, args.glass_um,
           args.film_total_um, args.adhesive_n, args.convergence_tol) <= 0:
        raise ValueError('Lengths, adhesive n and tolerance must be positive.')
    if not 0 < args.tip_diameter_nm <= args.base_diameter_nm < args.period_nm:
        raise ValueError('Require 0 < tip <= base < period.')
    if min(args.top_gold_nm, args.valley_gold_nm, args.residual_um, args.adhesive_um, args.adhesive_k) < 0:
        raise ValueError('Negative thickness or extinction coefficient.')
    if args.valley_gold_nm > args.height_nm or side_thickness(args) < 0:
        raise ValueError('Invalid valley or side thickness.')
    if min(args.slices, args.order, args.grid, args.threads, args.phase_points) <= 0:
        raise ValueError('Numerical sizes must be positive.')
    if args.cap_slices is not None and args.cap_slices <= 0:
        raise ValueError('--cap-slices must be positive.')
    if args.grid < 4*args.order+2:
        raise ValueError('--grid is too small for this Fourier order.')
    if args.film_total_um*1000-args.height_nm-args.residual_um*1000 <= 0:
        raise ValueError('The uncoated film thickness is smaller than its relief/residual.')
    layers = build_layers(args)
    if args.native_d6:
        if args.basis != 'auto' or any(layer.kind.startswith('coalesced') for layer in layers):
            raise ValueError('--native-d6 requires --basis auto and nonoverlapping centered circles.')
        if args.grid < max(32, 4*args.order+4):
            raise ValueError('--native-d6 requires grid >= max(32, 4*order+4).')


def load_materials(args, wls):
    functions, provenance = {}, {}
    functions['PET'], provenance['PET'] = load_nk(args.pet_nk_csv)
    if args.resin_nk_csv:
        functions['resin'], provenance['resin'] = load_nk(args.resin_nk_csv)
    else:
        functions['resin'] = functions['PET']
        provenance['resin'] = {'assumption': 'PET n,k proxy for unknown surface resin'}
    if args.adhesive_um == 0:
        provenance['adhesive'] = {'assumption': 'omitted: direct PET-glass contact'}
    elif args.adhesive_nk_csv:
        functions['adhesive'], provenance['adhesive'] = load_nk(args.adhesive_nk_csv)
    else:
        functions['adhesive'] = lambda wl: complex(args.adhesive_n, args.adhesive_k)
        provenance['adhesive'] = {'assumption': 'constant n,k', 'n': args.adhesive_n, 'k': args.adhesive_k}
    functions['silica'] = silica_n
    provenance['silica'] = {'source': 'https://doi.org/10.1364/JOSA.55.001205', 'k': 0.}
    for role in ['top', 'side', 'valley']:
        file = getattr(args, role+'_gold_csv') or args.gold_csv
        functions['Au_'+role], provenance['Au_'+role] = load_nk(file)
    for wl in wls:
        for evaluate in functions.values():
            evaluate(wl)
    max_n = max(functions[key](wl).real for key in ('PET', 'resin', 'adhesive', 'silica')
                if key in functions for wl in wls)
    if min(wls) <= math.sqrt(3)/2*args.period_nm*max_n:
        raise ValueError('A nonzero diffraction order may propagate in the thick layers.')
    return functions, provenance


def surface_rcwa(args, wl, indices, *, order=None, slices=None, grid=None, source_pol='x', shapes=None):
    m, nz, ng = order or args.order, slices or args.slices, grid or args.grid
    if args.native_d6:
        from native_d6 import NativeD6Geometry, surface_native_d6
        if shapes is None:
            shapes = NativeD6Geometry(args, nz, ng)
        return surface_native_d6(args, wl, indices, m, nz, shapes)

    import torch
    from rcwa_ext import (AutoRCWA, Circle, GroupTheoryOptions, Lattice,
                          LayerSpec, Material, NVMOptions, OutputSpec)
    layers = shapes.layers if shapes is not None else build_layers(args, nz)
    coalesced = sum(a.kind.startswith('coalesced') for a in layers)
    reduced = args.basis == 'auto' and coalesced == 0
    storage = getattr(args, 'layer_storage', 'full')
    solver_class = memory_safe_class(AutoRCWA) if storage == 'reduced' else AutoRCWA
    if shapes is not None:
        solver_class = shapes.solver_class(solver_class)
        initial_shape_counts = shapes.counts()
    sim = solver_class(freq=args.period_nm/wl, order=[m, m], lattice=Lattice.triangular(1.),
        cascade='redheffer', outputs=OutputSpec(smatrix_size='full', fields='none'),
        nvm=NVMOptions(grid=(ng, ng)),
        group_theory=GroupTheoryOptions(enabled=True, strict=True, symmetry='d6',
                                       polarization=source_pol) if reduced else False,
        dtype=torch.complex128, device=args.device)
    sim.add_input_layer(eps=1.)
    sim.add_output_layer(eps=indices['PET']**2)
    sim.set_incident_angle(0., 0.)
    center = ((.75, math.sqrt(3)/4),)
    distances = nearest_radius(ng)*args.period_nm if coalesced and shapes is None else None
    with torch.no_grad():
        for layer in layers:
            thickness = layer.thickness_nm/args.period_nm
            eps_resin = indices['resin']**2
            eps_gold = indices['Au_'+layer.gold_role]**2
            if layer.kind.startswith('coalesced'):
                code = shapes.raster(layer) if shapes is not None else material_codes(layer, distances)
                values = np.where(code == 1, eps_resin, np.where(code == 2, eps_gold, 1.+0j))
                sim.add_layer(thickness, eps=torch.as_tensor(values, dtype=torch.complex128, device=args.device))
            elif layer.kind == 'shell':
                sim.add_layer_circle_shell_nvm(thickness, layer.core_radius_nm/args.period_nm,
                    layer.outer_radius_nm/args.period_nm, 1., eps_gold, eps_resin,
                    centers=center, nx=ng, ny=ng)
            else:
                radius = layer.outer_radius_nm if layer.kind == 'cap' else layer.core_radius_nm
                bg = eps_gold if layer.kind == 'valley' else 1.
                inclusion = eps_gold if layer.kind == 'cap' else eps_resin
                sim.add_structured_layer(LayerSpec(thickness, Circle(radius=radius/args.period_nm),
                    background=Material(bg), inclusion=Material(inclusion), method='nvm'))
            if storage == 'reduced':
                sim.release_completed_layer()
        if args.residual_um > 0:
            sim.add_structured_layer(LayerSpec(args.residual_um*1000/args.period_nm,
                Circle(radius=.25), background=Material(indices['resin']**2),
                inclusion=Material(indices['resin']**2), method='nvm'))
            if storage == 'reduced':
                sim.release_completed_layer()
        retained_before_solve = tensor_storage_bytes(vars(sim))
        sim.solve_global_smatrix()
        zx = int(torch.nonzero(sim.order_x == 0)[0, 0])
        zy = int(torch.nonzero(sim.order_y == 0)[0, 0])
        zero = zx*len(sim.order_y)+zy
        polarizations = [zero, zero+sim.order_N]
        if reduced:
            index = polarizations[0 if source_pol == 'x' else 1]
            scattering = [np.eye(2, dtype=np.complex128)*complex(block[index, index].detach().cpu())
                          for block in sim.S]
        else:
            take = torch.tensor(polarizations, dtype=torch.long, device=args.device)
            scattering = [block.index_select(0, take).index_select(1, take).detach().cpu().numpy()
                          for block in sim.S]
    diagnostics = {'basis': 'D6-star-source' if reduced else 'full-rectangular',
        'harmonics': 1+3*m*(m+1) if reduced else (2*m+1)**2,
        'core_layers': nz, 'cap_layers': cap_count(args, nz),
        'total_pattern_layers': len(layers), 'coalesced_raster_layers': coalesced,
        'layer_storage': storage,
        'storage_policy': POLICY if storage == 'reduced' else 'original-full-storage',
        'retained_before_solve_bytes': retained_before_solve,
        'post_layer_retained_tensor_peak_bytes': getattr(sim, 'post_layer_retained_tensor_peak_bytes', None),
        'unpolarized': 'D6 x=y equality' if reduced else 'Jones x/y average including cross polarization'}
    if shapes is not None:
        keys = ('geometry_disk_builds', 'geometry_normal_builds', 'geometry_raster_builds')
        diagnostics.update({key: after-before for key, before, after in
                            zip(keys, initial_shape_counts, shapes.counts())})
    return scattering, diagnostics


def evaluate_case(args, functions, wl, m, nz, grid, *, shapes=None):
    import torch
    gc.collect()
    if args.device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    indices = {key: f(wl) for key, f in functions.items()}
    scattering, diagnostic = surface_rcwa(args, wl, indices, order=m, slices=nz, grid=grid, shapes=shapes)
    pet_thickness = args.film_total_um*1000-args.height_nm-args.residual_um*1000
    coherent = {v.strip().lower() for v in args.coherent_layers.split(',') if v.strip()}
    tail = [(indices['PET'], pet_thickness, 'pet' not in coherent)]
    if args.adhesive_um > 0:
        tail.append((indices['adhesive'], args.adhesive_um*1000, 'adhesive' not in coherent))
    tail.append((indices['silica'], args.glass_um*1000, 'glass' not in coherent))
    phase = {}
    if args.coherence == 'phase-average':
        result, error = phase_average(scattering, indices['PET'], tail, wl,
                                      points=args.phase_points, tolerance=args.convergence_tol)
        phase['phase_average_error'] = error
    else:
        result = finite_sample(scattering, indices['PET'], tail, wl, coherence=args.coherence)
    if not all(math.isfinite(result[key]) and -2e-6 <= result[key] <= 1+2e-6 for key in ['R', 'T', 'A']):
        raise RuntimeError(f'Nonphysical R/T/A at {wl} nm: {result}')
    if args.device == 'cuda':
        torch.cuda.synchronize()
    return {'wavelength_nm': float(wl), 'order': m, 'slices': nz, 'grid': grid,
        **result, 'coherence': args.coherence, **phase,
        'xy_power_difference': max(abs(result['R_x']-result['R_y']), abs(result['T_x']-result['T_y'])),
        **diagnostic, 'seconds': time.perf_counter()-started,
        'peak_cuda_allocated_bytes': int(torch.cuda.max_memory_allocated()) if args.device == 'cuda' else None,
        'peak_cuda_reserved_bytes': int(torch.cuda.max_memory_reserved()) if args.device == 'cuda' else None}


def plot_spectra(args, output, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.1), sharex=True)
    for ax, key, color in zip(axes, ['R', 'T', 'A'], ['#275d9f', '#197b60', '#b47713']):
        ax.plot([r['wavelength_nm'] for r in rows], [100*r[key] for r in rows], color=color)
        ax.set_ylabel(key+' (%)')
        ax.grid(alpha=.25)
    axes[-1].set_xlabel('Wavelength (nm)')
    fig.suptitle('Au-coated MOSMITE / adhesive / silica / air: '+args.mode)
    fig.tight_layout()
    pending = output/'spectra.png.tmp'
    fig.savefig(pending, format='png', dpi=180)
    plt.close(fig)
    pending.replace(output/'spectra.png')


def main(mode):
    args = parser(mode).parse_args()
    if args.native_d6:
        args.reuse_geometry = True
        args.layer_storage = 'reduced'
    args.solver_root = args.solver_root.resolve()
    validate_args(args)
    wls = wavelengths(args.wavelengths)
    functions, provenance = load_materials(args, wls)
    mirror, quartz = None, None
    if args.mirror_nk_csv:
        args.mirror_nk_csv = args.mirror_nk_csv.resolve()
        mirror = bare_mirror_r(args.mirror_nk_csv, wls)
        quartz = bare_quartz_t(wls)
    output = (args.output_dir or HERE/'results'/mode/('convergence' if args.check_convergence else 'reference')).resolve()
    if args.reference_dir and args.reference_dir.resolve() == output:
        raise ValueError('--reference-dir and --output-dir must differ; preserve the original results.')
    layers = build_layers(args)
    meta = {'arguments': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        'geometry': {'mode': mode, 'side_radial_nm': side_thickness(args), 'core_layers': args.slices,
            'cap_layers': cap_count(args), 'total_pattern_layers': len(layers),
            'coalesced_raster_layers': sum(a.kind.startswith('coalesced') for a in layers),
            'height_including_gold_nm': args.height_nm+args.top_gold_nm,
            'uniform_PET_equivalent_nm': args.film_total_um*1000-args.height_nm-args.residual_um*1000,
            'top_cap_profile': 'linear taper to original tip radius' if mode == 'evaporation' else 'flat disk, radius=tip_radius+side_radial'},
        'material_provenance': provenance,
        'assumptions': ['Normal incidence; isotropic substrate/resin; unpolarized input.',
            'Core+valley is 150 layers by default; top cap is additional (default 23).',
            'Gold is a continuous local isotropic material; side island formation is not represented.',
            'Evaporation defaults to ideal normal directional deposition with no shadowing.',
            'Sputter defaults to a phenomenological 8 nm RADIAL shell, not a deposition transport calculation.',
            'Periodic union is used when Au shells coalesce; it is not replaced by an isolated or clipped circle.',
            'Uniform PET/adhesive/glass use the selected coherence model.',
            'Resin optical constants default to the PET proxy; adhesive n,k default to constants.',
            'Uncoated film remains 55 um; gold is added outside that thickness.',
            'Glass rear-surface reflection is included. No diffuse scattering or instrument convolution.'],
        'thick_layer_theory': THEORY,
        'optics_sha256': hashlib.sha256((HERE.parent/'mosmite_optics.py').read_bytes()).hexdigest(),
        'total_points': len(wls), 'completed_points': 0,
        'status': 'prepared'}
    if args.coherence == 'phase-average':
        meta['thick_layer_theory'] = PHASE_AVERAGE_THEORY
        meta['phase_average_sha256'] = hashlib.sha256((HERE.parent/'mosmite_phase_average.py').read_bytes()).hexdigest()
    for name in ['model.py', 'geometry.py', 'stack.py', 'memory.py', 'run_evaporation.py', 'run_sputter.py']:
        meta.setdefault('study_sha256', {})[name] = hashlib.sha256((HERE/name).read_bytes()).hexdigest()
    if args.reuse_geometry:
        meta['study_sha256']['shape_cache.py'] = hashlib.sha256((HERE/'shape_cache.py').read_bytes()).hexdigest()
    if args.native_d6:
        meta['study_sha256']['native_d6.py'] = hashlib.sha256((HERE/'native_d6.py').read_bytes()).hexdigest()
        native_source = args.solver_root/'rcwa_ext'/'acceleration'/'d6.py'
        meta['native_d6_sha256'] = hashlib.sha256(native_source.read_bytes()).hexdigest()
    if args.mirror_nk_csv:
        meta['study_sha256']['normalize_references.py'] = hashlib.sha256(
            (HERE/'normalize_references.py').read_bytes()).hexdigest()
    if (args.solver_root/'rcwa_ext').is_dir():
        meta['solver_sha256'] = {str(p.relative_to(args.solver_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted((args.solver_root/'rcwa_ext').glob('*.py'))}
    restored = []
    if args.resume:
        from resume import load_resume
        restored = load_resume(output, meta, wls)
        meta.update(completed_points=len(restored), resumed_points=len(restored))
        meta['study_sha256']['resume.py'] = hashlib.sha256((HERE/'resume.py').read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    write_geometry(args, output)
    materials = []
    for wl in wls:
        row = {'wavelength_nm': float(wl)}
        for key, function in functions.items():
            value = function(wl)
            row.update({key+'_n': value.real, key+'_k': value.imag})
        materials.append(row)
    save_csv(output/'materials.csv', materials)
    def write_meta():
        pending = output/'settings.json.tmp'
        pending.write_text(json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        pending.replace(output/'settings.json')

    def write_progress(rows):
        plot_spectra(args, output, rows)
        if mirror is not None:
            current_wls = np.array([row['wavelength_nm'] for row in rows])
            positions = np.searchsorted(wls, current_wls)
            powers = np.array([(row['R'], row['T']) for row in rows])
            write_referenced_spectrum(output, current_wls, powers, quartz[positions], mirror[positions])
        print(f'Progress: {len(rows)}/{len(wls)} ({len(rows) / len(wls):.1%}) '
              f'through {rows[-1]["wavelength_nm"]:g} nm. Updated spectra in {output}', flush=True)
    if args.reference_dir:
        # Reject incompatible saved conditions before expensive RCWA solves.
        compare_reference(args, meta, [])
    write_meta()
    if args.preview_only:
        print(f'Prepared {mode}: {len(layers)} pattern layers; side={side_thickness(args):g} nm; {output}', flush=True)
        return
    try:
        if not (args.solver_root/'rcwa_ext').is_dir():
            raise FileNotFoundError('--solver-root must contain the 20260806 rcwa_ext directory.')
        sys.path.insert(0, str(args.solver_root))
        import torch
        torch.set_num_threads(args.threads)
        if args.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable in this Python environment.')
        meta.update({'status': 'running', 'torch': torch.__version__})
        write_meta()
        rows, checks = restored, []
        completed_rows = {row['wavelength_nm']: row for row in restored}
        cache = {(row['wavelength_nm'], args.order, args.slices, args.grid): row for row in restored}
        if args.resume:
            remaining = [float(wl) for wl in wls if float(wl) not in completed_rows]
            print(f'Resume: {len(rows)}/{len(wls)} wavelengths restored; '
                  + (f'next wavelength {remaining[0]:g} nm.' if remaining else 'all wavelengths are saved.'),
                  flush=True)
            if rows:
                write_progress(rows)
        shape_config, shapes = None, None
        if args.native_d6:
            from native_d6 import NativeD6Geometry as GeometryCache
        elif args.reuse_geometry:
            from shape_cache import ShapeCache as GeometryCache

        def calculate(wl, m, nz, ng):
            nonlocal shape_config, shapes
            key = (float(wl), m, nz, ng)
            if key not in cache:
                if args.reuse_geometry and (m, nz, ng) != shape_config:
                    # Only one set of geometry tensors may occupy the GPU.
                    shape_config = (m, nz, ng)
                    shapes = GeometryCache(args, nz, ng)
                cache[key] = evaluate_case(args, functions, wl, m, nz, ng, shapes=shapes)
            return cache[key]
        for wl in wls:
            if float(wl) in completed_rows:
                continue
            row = calculate(wl, args.order, args.slices, args.grid)
            completed_rows[float(wl)] = row
            rows = [completed_rows[float(value)] for value in wls if float(value) in completed_rows]
            completed = len(rows)
            pending = output/'spectra.csv.tmp'
            save_csv(pending, rows)
            pending.replace(output/'spectra.csv')
            print(f'{mode} {wl:g} nm R={row["R"]:.8f} T={row["T"]:.8f} A={row["A"]:.8f} '
                  f'{row["basis"]} raster={row["coalesced_raster_layers"]}', flush=True)
            if args.reuse_geometry:
                print(f'  geometry builds: disks={row["geometry_disk_builds"]}, '
                      f'normal={row["geometry_normal_builds"]}, raster={row["geometry_raster_builds"]}; '
                      f'{row["seconds"]:.2f} s', flush=True)
            if row['peak_cuda_allocated_bytes'] is not None:
                print(f'  peak CUDA allocated={row["peak_cuda_allocated_bytes"]/2**30:.3f} GiB; '
                      f'storage={row["layer_storage"]}', flush=True)
            meta['completed_points'] = completed
            write_meta()
            if completed % args.progress_every == 0 or completed == len(wls):
                write_progress(rows)
        if args.reference_dir:
            parity = compare_reference(args, meta, rows)
            save_csv(output/'storage_parity.csv', parity)
            meta['storage_parity'] = {'passed': all(r['passed'] for r in parity),
                'reference_dir': str(args.reference_dir.resolve()), 'tolerance': PARITY_TOL,
                'max_absolute_error': max(r['max_absolute_error'] for r in parity)}
            write_meta()
            if not meta['storage_parity']['passed']:
                raise RuntimeError('Storage parity failed: inspect storage_parity.csv.')
            print(f'PASS storage parity: max absolute power difference='
                  f'{meta["storage_parity"]["max_absolute_error"]:.3g}', flush=True)
        if args.check_convergence:
            configs = [(args.order, args.slices, args.grid),
                       (args.order+2, args.slices, args.grid),
                       (args.order+2, 2*args.slices, args.grid),
                       (args.order+2, 2*args.slices, 2*args.grid)]
            for wl in wavelengths(args.check_wavelengths):
                if not min(wls) <= wl <= max(wls):
                    continue
                previous, baseline = None, None
                for m, nz, ng in configs:
                    row = calculate(wl, m, nz, ng)
                    if baseline is None:
                        baseline = row
                    keys = ['R', 'T', 'A', 'R_x', 'T_x', 'R_y', 'T_y']
                    delta = 0. if previous is None else max(abs(row[k]-previous[k]) for k in keys)
                    total = max(abs(row[k]-baseline[k]) for k in keys)
                    checks.append({'wavelength_nm': float(wl), 'order': m, 'slices': nz,
                        'cap_layers': row['cap_layers'], 'grid': ng, 'R': row['R'], 'T': row['T'], 'A': row['A'],
                        'max_delta': delta, 'baseline_delta': total,
                        'xy_power_difference': row['xy_power_difference'], 'passed': delta <= args.convergence_tol,
                        'baseline_passed': total <= args.convergence_tol,
                        'xy_passed': row['xy_power_difference'] <= args.convergence_tol})
                    save_csv(output/'convergence.csv', checks)
                    print(f'check {wl:g} M={m} Nz={nz} grid={ng}: delta={delta:.4g}', flush=True)
                    previous = row
            if not checks:
                raise ValueError('No convergence-check wavelengths within requested range.')
            # Also require the refined D6 physical x/y degeneracy in full-basis calculations.
            final_checks = [checks[i] for i in range(3, len(checks), 4)]
            meta['convergence_passed'] = (all(c['passed'] for c in checks)
                and all(c['baseline_delta'] <= args.convergence_tol for c in final_checks)
                and all(c['xy_power_difference'] <= args.convergence_tol for c in final_checks))
        meta.update({'status': 'completed', 'completed_points': len(rows)})
        if args.check_convergence and not meta['convergence_passed']:
            raise RuntimeError('Convergence criteria not met: inspect convergence.csv; increase order/slices/grid.')
        write_meta()
        print(f'Saved: {output}', flush=True)
    except BaseException as exc:
        meta.update({'status': 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                     'error': type(exc).__name__+': '+str(exc)})
        write_meta()
        raise
