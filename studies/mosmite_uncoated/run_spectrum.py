"""Uncoated MOSMITE: local RCWA plus finite PET/adhesive/silica/air stack.

Normal incidence, isotropic materials and a D6 triangular lattice only.
The thick-layer tail is connected as electric-field scattering amplitudes,
then intensities are averaged over independent round-trip phases. This avoids
assigning an isolated incident/reflected power to an absorbing internal port.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
PET_SOURCE = ("https://raw.githubusercontent.com/polyanskiy/"
    "refractiveindex.info-database/main/database/data/organic/"
    "%28C10H8O4%29n%20-%20polyethylene%20terephthalate/nk/Zhang.yml")


def wavelengths(text):
    if ':' not in text:
        values = np.array([float(v) for v in text.split(',')])
    else:
        start, stop, step = map(float, text.split(':'))
        if step <= 0 or stop < start:
            raise ValueError('Use start:stop:positive_step in nm.')
        values = np.arange(start, stop + step * 1e-7, step)
    if len(values) == 0 or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError('Wavelengths must be finite and positive.')
    return np.unique(values)


def load_nk(path):
    """Read wavelength_nm,n,k or wavelength_um,n,k; no extrapolation."""
    path = Path(path).resolve()
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if 'wavelength_nm' in fields:
            key, factor = 'wavelength_nm', 1.
        elif 'wavelength_um' in fields:
            key, factor = 'wavelength_um', 1000.
        else:
            raise ValueError(f'{path}: require wavelength_nm or wavelength_um header.')
        data = np.array([[factor*float(row[key]), float(row['n']),
                          float(row['k'])] for row in reader])
    if (data.ndim != 2 or data.shape[0] < 2 or not np.all(np.isfinite(data))
            or np.any(data[:, 1] <= 0) or np.any(data[:, 2] < 0)):
        raise ValueError(f'{path}: invalid passive n,k table.')
    data = data[np.argsort(data[:, 0])]
    if np.any(np.diff(data[:, 0]) <= 0):
        raise ValueError('Wavelengths must be unique.')

    def evaluate(wl):
        if not data[0, 0]-1e-8 <= wl <= data[-1, 0]+1e-8:
            raise ValueError(f'{wl} nm outside {path} range; extrapolation disabled.')
        return complex(np.interp(wl, data[:, 0], data[:, 1]),
                       np.interp(wl, data[:, 0], data[:, 2]))

    return evaluate, {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                      'range_nm': [float(data[0, 0]), float(data[-1, 0])],
                      'interpolation': 'linear n and k separately; no extrapolation'}


def silica_n(wl):
    lam2 = (wl / 1000.)**2
    if not 210 <= wl <= 3710:
        raise ValueError('Malitson fused-silica range is 210--3710 nm.')
    return complex(math.sqrt(1 + .6961663*lam2/(lam2-.0684043**2)
        + .4079426*lam2/(lam2-.1162414**2)
        + .8974794*lam2/(lam2-9.896161**2)), 0.)


def interface(n_left, n_right):
    """Return electric-field (tf, rf, rb, tb), at normal incidence."""
    denom = n_left + n_right
    rf = (n_left - n_right) / denom
    return 2*n_left/denom, rf, -rf, 2*n_right/denom


def prepend(left, right_r, right_t):
    tf, rf, rb, tb = left
    denominator = 1 - rb*right_r
    return rf + tb*right_r*tf/denominator, right_t*tf/denominator


def bulk_layers(args, wl, pet, adhesive):
    """Only existing layers enter both the amplitude stack and phase average."""
    pet_nm = args.film_total_um*1000-args.height_nm-args.residual_um*1000
    layers = [(pet(wl), pet_nm, True)]
    if args.adhesive_um > 0:
        layers.append((adhesive(wl), args.adhesive_um*1000, True))
    if args.glass_um > 0:
        layers.append((silica_n(wl), args.glass_um*1000, True))
    return layers


def stack_description(args):
    parts = ['air', 'MOSMITE/PET']
    if args.adhesive_um > 0:
        parts.append('adhesive')
    if args.glass_um > 0:
        parts.append('silica')
    return ' / '.join(parts+['air'])


def finite_sample(surface, port_index, layers, wl, phase_points):
    """layers: (complex n, thickness_nm, phase_average_bool), from top down.

    A phase point is an extra SINGLE-pass phase in [0, pi). Thus there are
    exactly phase_points distinct ROUND-trip phases for each averaged layer.
    Phase variation preserves nominal attenuation and averages interference.
    Final and incident external media are both air: R=|r|^2, T=|t|^2.
    """
    count = sum(bool(layer[2]) for layer in layers) if phase_points > 1 else 0
    if count:
        axes = np.meshgrid(*([np.arange(phase_points)*np.pi/phase_points]*count),
                           indexing='ij')
        phases = [axis.ravel() for axis in axes]
    else:
        phases = []
    size = phase_points**count
    r = np.zeros(size, dtype=np.complex128)
    t = np.ones(size, dtype=np.complex128)
    n_right = 1.+0j
    phase_slot = count - 1
    for index, thickness_nm, average in reversed(layers):
        r, t = prepend(interface(index, n_right), r, t)
        extra_phase = phases[phase_slot] if average and count else 0.
        if average and count:
            phase_slot -= 1
        propagation = np.exp(2j*np.pi*index*thickness_nm/wl + 1j*extra_phase)
        r, t = r*propagation**2, t*propagation
        n_right = index
    r, t = prepend(interface(port_index, n_right), r, t)
    r, t = prepend(surface, r, t)
    reflectance = float(np.mean(np.abs(r)**2))
    transmittance = float(np.mean(np.abs(t)**2))
    return reflectance, transmittance, 1-reflectance-transmittance


def surface_rcwa(args, wl, n_resin, *, n_substrate=None, order=None, slices=None, grid=None):
    import torch
    from rcwa_ext import (AutoRCWA, Lattice, NVMOptions, ASROptions,
                          GroupTheoryOptions, OutputSpec, Circle, LayerSpec, Material)
    m, nz, ng = order or args.order, slices or args.slices, grid or args.grid
    sim = AutoRCWA(freq=args.period_nm/wl, order=[m, m],
        lattice=Lattice.triangular(1.), cascade='redheffer',
        outputs=OutputSpec(smatrix_size='full', fields='none'),
        nvm=NVMOptions(grid=(ng, ng)), asr=ASROptions(grid=(ng, ng)),
        group_theory=GroupTheoryOptions(enabled=True, strict=True,
                                       symmetry='d6', polarization=args.polarization),
        dtype=torch.complex128, device=args.device)
    sim.add_input_layer(eps=1.)
    sim.add_output_layer(eps=(n_resin if n_substrate is None else n_substrate)**2)
    sim.set_incident_angle(0., 0.)
    # The matched-coordinate map differentiates its coordinate functions even
    # for fixed geometry. NVM does not need that autograd context.
    with torch.set_grad_enabled(args.method == 'matched-asr'):
        for j in range(nz):
            fraction = (j+.5)/nz
            diameter = args.tip_diameter_nm + (
                args.base_diameter_nm-args.tip_diameter_nm)*fraction**args.profile_power
            sim.add_structured_layer(LayerSpec(
                thickness=args.height_nm/nz/args.period_nm,
                geometry=Circle(radius=diameter/2/args.period_nm),
                background=Material(1.), inclusion=Material(n_resin**2),
                method=args.method))
        if args.residual_um > 0:
            # Equal inclusion/background is a homogeneous layer in the same
            # D6 basis. Keeping it inside RCWA retains near-field coupling to
            # the PET interface even for an extremely thin residual layer.
            sim.add_structured_layer(LayerSpec(
                thickness=args.residual_um*1000/args.period_nm,
                geometry=Circle(radius=.25), background=Material(n_resin**2),
                inclusion=Material(n_resin**2), method=args.method))
        sim.solve_global_smatrix()
        zero_x = int(torch.nonzero(sim.order_x == 0)[0, 0])
        zero_y = int(torch.nonzero(sim.order_y == 0)[0, 0])
        zero = zero_x*len(sim.order_y)+zero_y
        i = zero + (sim.order_N if args.polarization == 'y' else 0)
        coefficients = tuple(complex(block[i, i].detach().cpu()) for block in sim.S)
    return coefficients


def save_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_spectrum(output, rows, description='air / film / adhesive / silica / air'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    wl = [row['wavelength_nm'] for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.5), sharex=True)
    for ax, key, label, color in zip(axes, ['R', 'T'],
            ['Reflectance (%)', 'Transmittance (%)'], ['#275d9f', '#197b60']):
        ax.plot(wl, [100*row[key] for row in rows], color=color, linewidth=1.8)
        ax.set_ylabel(label)
        ax.grid(alpha=.25)
    axes[-1].set_xlabel('Wavelength (nm)')
    fig.suptitle('Uncoated MOSMITE: '+description)
    fig.tight_layout()
    fig.savefig(output/'spectra.png', dpi=180)
    plt.close(fig)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--solver-root', type=Path, default=HERE.parents[1])
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--wavelengths', default='400:800:5', help='nm; start:stop:step or comma list')
    p.add_argument('--pet-nk-csv', type=Path, default=HERE/'data/PET_Zhang_400_800nm.csv')
    p.add_argument('--resin-nk-csv', type=Path, help='Omit to use the PET optical proxy.')
    p.add_argument('--adhesive-nk-csv', type=Path, help='Optional measured adhesive dispersion.')
    p.add_argument('--adhesive-n', type=float, default=1.50)
    p.add_argument('--adhesive-k', type=float, default=0.)
    p.add_argument('--period-nm', type=float, default=100.)
    p.add_argument('--height-nm', type=float, default=200.)
    p.add_argument('--base-diameter-nm', type=float, default=98.)
    p.add_argument('--tip-diameter-nm', type=float, default=50.)
    p.add_argument('--profile-power', type=float, default=1.)
    p.add_argument('--film-total-um', type=float, default=55.,
                   help='Total including relief and any specified residual layer.')
    p.add_argument('--residual-um', type=float, default=0.,
                   help='Explicit uniform surface-resin layer; default is optically merged into PET.')
    p.add_argument('--adhesive-um', type=float, default=20., help='Set 0 to omit the adhesive layer.')
    p.add_argument('--glass-um', type=float, default=1000., help='Set 0 to omit glass; the last remaining layer borders air.')
    p.add_argument('--order', type=int, default=7)
    p.add_argument('--slices', type=int, default=100)
    p.add_argument('--grid', type=int, default=256)
    p.add_argument('--method', choices=['nvm', 'matched-asr'], default='nvm')
    p.add_argument('--polarization', choices=['x', 'y'], default='x',
                   help='D6 and normal incidence imply identical x/y intensity spectra.')
    p.add_argument('--coherence', choices=['phase-average', 'coherent'], default='phase-average')
    p.add_argument('--phase-points', type=int, default=16)
    p.add_argument('--check-convergence', action='store_true',
                   help='Compare M+2, then Nz*2, then grid*2 at 400,600,800 nm in range.')
    p.add_argument('--check-wavelengths', default='400,600,800',
                   help='Convergence probes in nm, comma list or start:stop:step.')
    p.add_argument('--convergence-tol', type=float, default=1e-4,
                   help='Absolute R/T/A fraction tolerance; 0.005 means 0.5 percentage point, not 0.5 percent relative.')
    p.add_argument('--output-dir', type=Path, default=HERE/'results/reference')
    return p


def main():
    args = parser().parse_args()
    for name, value in vars(args).items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f'{name} must be finite.')
    if min(args.order, args.slices, args.grid, args.threads, args.phase_points) <= 0:
        raise ValueError('Numerical sizes must be positive.')
    if (not 0 <= args.tip_diameter_nm <= args.base_diameter_nm < args.period_nm
            or args.period_nm <= 0 or args.height_nm < 0 or args.profile_power <= 0):
        raise ValueError('Require 0 <= tip <= base < period and nonnegative height.')
    pet_nm = args.film_total_um*1000 - args.height_nm - args.residual_um*1000
    if min(pet_nm, args.adhesive_n, args.convergence_tol) <= 0 or min(
            args.residual_um, args.adhesive_k, args.adhesive_um, args.glass_um) < 0:
        raise ValueError('Invalid layer thicknesses or adhesive index.')
    if not (args.solver_root/'rcwa_ext').is_dir():
        raise FileNotFoundError('Set --solver-root to the 20260806 directory containing rcwa_ext/.')
    sys.path.insert(0, str(args.solver_root.resolve()))
    import torch
    torch.set_num_threads(args.threads)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but this Python has no available CUDA device.')
    pet, pet_provenance = load_nk(args.pet_nk_csv)
    if args.resin_nk_csv:
        resin, resin_provenance = load_nk(args.resin_nk_csv)
    else:
        resin, resin_provenance = pet, {'assumption': 'PET n,k optical proxy for uncharacterized surface resin'}
    if args.adhesive_nk_csv:
        adhesive, adhesive_provenance = load_nk(args.adhesive_nk_csv)
    else:
        adhesive = lambda wl: complex(args.adhesive_n, args.adhesive_k)
        adhesive_provenance = {'assumption': 'constant n,k; adhesive dispersion is unknown',
                              'n': args.adhesive_n, 'k': args.adhesive_k}
    wls = wavelengths(args.wavelengths)
    # Every nonzero diffraction order must be evanescent in the thick layers.
    active_materials = [pet, resin]
    if args.adhesive_um > 0:
        active_materials.append(adhesive)
    if args.glass_um > 0:
        active_materials.append(silica_n)
    max_n = max(abs(f(w).real) for w in wls for f in active_materials)
    if min(wls) <= math.sqrt(3)/2*args.period_nm*max_n:
        raise ValueError('A higher order may propagate: the scalar thick-layer tail is ineligible.')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    phase_points = args.phase_points if args.coherence == 'phase-average' else 1
    metadata = {'arguments': {key: str(value) if isinstance(value, Path) else value
                             for key, value in vars(args).items()},
        'stack_description': stack_description(args),
        'stack_nm': {'pattern': args.height_nm, 'explicit_residual': args.residual_um*1000,
                     'PET_or_optically_merged_resin_and_PET': pet_nm,
                     'adhesive': args.adhesive_um*1000, 'silica': args.glass_um*1000},
        'material_provenance': {'PET': pet_provenance, 'PET_source': PET_SOURCE,
             'PET_paper': 'https://doi.org/10.1364/AO.383831', 'resin': resin_provenance,
             'adhesive': adhesive_provenance,
             'silica': ('Malitson 1965 https://doi.org/10.1364/JOSA.55.001205; k=0'
                        if args.glass_um > 0 else 'omitted: --glass-um 0')},
        'interpretation': ['Reference model, not a calibrated MOSMITE material measurement.',
             '55 um includes the 0.2 um relief: default uniform optical thickness is 54.8 um.',
             'No independently known residual thickness; default resin and PET are optically merged.',
             'Only the front surface is patterned; the last existing uniform layer borders air.',
             'Only positive-thickness layers are included in the interfaces and phase average.',
             'Existing PET/adhesive/glass layers are phase averaged unless coherent; residual stays coherent.',
             'D6 triangular lattice, normal incidence, scalar isotropic n,k; x=y=unpolarized.',
             'Pattern is coherent; absorbing internal ports are connected as amplitudes.',
             'No diffuse scattering, anisotropy, instrument convolution or parameter fitting.'],
        'torch': torch.__version__, 'solver_root': str(args.solver_root.resolve()),
        'harmonics': 1+3*args.order*(args.order+1),
        'D6_source_dimension': 1+args.order*(args.order+1)}
    hashes = {str(path.relative_to(args.solver_root)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted((args.solver_root/'rcwa_ext').glob('*.py'))}
    metadata['solver_sha256'] = hashes
    metadata['study_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (output/'settings.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    rows, material_rows, convergence = [], [], []
    for wl in wls:
        started = time.perf_counter()
        npet, nresin = pet(wl), resin(wl)
        layers = bulk_layers(args, wl, pet, adhesive)
        surface = surface_rcwa(args, wl, nresin, n_substrate=npet)
        result = finite_sample(surface, npet, layers, wl, phase_points)
        phase_error = 0.
        if phase_points > 1:
            doubled = finite_sample(surface, npet, layers, wl, 2*phase_points)
            phase_error = max(abs(a-b) for a, b in zip(result, doubled))
        if phase_error > args.convergence_tol:
            raise RuntimeError(f'Phase average not converged at {wl}: {phase_error}; increase --phase-points.')
        if min(result) < -2e-6 or max(result) > 1+2e-6:
            raise RuntimeError(f'Unphysical R,T,A at {wl}: {result}')
        row = {'wavelength_nm': float(wl), 'R': result[0], 'T': result[1], 'A': result[2],
               'phase_average_error': phase_error, 'seconds': time.perf_counter()-started}
        rows.append(row)
        matrow = {'wavelength_nm': float(wl)}
        material_indices = [('PET', npet), ('resin', nresin)]
        if args.adhesive_um > 0:
            material_indices.append(('adhesive', adhesive(wl)))
        if args.glass_um > 0:
            material_indices.append(('silica', silica_n(wl)))
        for name, index in material_indices:
            matrow.update({name+'_n': index.real, name+'_k': index.imag})
        material_rows.append(matrow)
        save_csv(output/'spectra.csv', rows)
        save_csv(output/'materials.csv', material_rows)
        print(f'{wl:g} nm  R={result[0]:.8f}  T={result[1]:.8f}  A={result[2]:.8f}', flush=True)
    if args.check_convergence:
        for wl in wavelengths(args.check_wavelengths):
            if not wls[0] <= wl <= wls[-1]:
                continue
            nresin = resin(wl)
            layers = bulk_layers(args, wl, pet, adhesive)
            configs = [(args.order, args.slices, args.grid),
                       (args.order+2, args.slices, args.grid),
                       (args.order+2, 2*args.slices, args.grid),
                       (args.order+2, 2*args.slices, 2*args.grid)]
            previous, baseline = None, None
            for m, nz, grid in configs:
                cached = next((row for row in rows if row['wavelength_nm'] == wl), None)
                if previous is None and cached is not None:
                    value = (cached['R'], cached['T'], cached['A'])
                else:
                    s = surface_rcwa(args, wl, nresin, n_substrate=pet(wl), order=m, slices=nz, grid=grid)
                    value = finite_sample(s, pet(wl), layers, wl, phase_points)
                    if phase_points > 1:
                        doubled = finite_sample(s, pet(wl), layers, wl, 2*phase_points)
                        if max(abs(a-b) for a,b in zip(value, doubled)) > args.convergence_tol:
                            raise RuntimeError(f'Phase average not converged at refined case {wl}: increase --phase-points.')
                if not all(math.isfinite(v) and -2e-6 <= v <= 1+2e-6 for v in value):
                    raise RuntimeError(f'Unphysical refined R,T,A at {wl}: {value}')
                if baseline is None:
                    baseline = value
                delta_rt = 0. if previous is None else max(abs(a-b) for a, b in zip(value[:2], previous[:2]))
                delta = 0. if previous is None else max(abs(a-b) for a, b in zip(value, previous))
                baseline_delta = max(abs(a-b) for a,b in zip(value, baseline))
                convergence.append({'wavelength_nm': wl, 'order': m, 'slices': nz, 'grid': grid,
                    'R': value[0], 'T': value[1], 'A': value[2], 'max_delta_RT': delta_rt,
                    'max_delta_RTA': delta, 'baseline_delta_RTA': baseline_delta,
                    'passed': delta <= args.convergence_tol,
                    'baseline_passed': baseline_delta <= args.convergence_tol})
                save_csv(output/'convergence.csv', convergence)
                print(f'check {wl:g} M={m} Nz={nz} grid={grid} delta={delta:.3g}', flush=True)
                previous = value
        if convergence:
            save_csv(output/'convergence.csv', convergence)
            metadata['convergence_passed'] = all(row['passed'] and row['baseline_passed'] for row in convergence)
            (output/'settings.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
        else:
            raise ValueError('No convergence probes inside the requested wavelength range.')
    plot_spectrum(output, rows, stack_description(args))
    if convergence and not metadata['convergence_passed']:
        raise RuntimeError('Convergence tolerance not met; inspect convergence.csv and increase numerical settings.')
    print(f'Saved: {output}', flush=True)


if __name__ == '__main__':
    main()
