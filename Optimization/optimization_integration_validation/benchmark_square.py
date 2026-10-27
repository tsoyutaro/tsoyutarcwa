"""Square-lattice C2v matrix assembly and wavelength-sharing validation.

Three measured paths: frozen production, block assembly, block + sharing.
Lossless, normal x incidence, centred circles; arithmetic-mean reflectance
and derivatives w.r.t. all normalized radii. This does not edit production.
"""
from __future__ import annotations
import argparse
import bisect
import copy
import csv
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time
import torch
import block_assembly as assembly
from rcwa_ext.reduced import _ReducedScatteringMixin
from rcwa_ext.scattering import _StableLinearAlgebraMixin

HERE = Path(__file__).resolve().parent


def load_pmma(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['wl', 'n']:
            raise ValueError('Expected CSV columns wl,n; wavelength in micrometres.')
        rows = [(float(row['wl'])*1000, float(row['n'])) for row in reader]
    if len(rows) < 2 or any(b[0] <= a[0] for a, b in zip(rows, rows[1:])):
        raise ValueError('CSV wavelengths must be strictly increasing.')
    return rows


def pmma_index(rows, wave):
    if wave < 400 or wave > rows[-1][0]:
        raise ValueError('PMMA wavelength outside supported interval.')
    j = bisect.bisect_left([row[0] for row in rows], wave)
    if j == 0:
        if rows[0][0] > 405:
            raise ValueError('Only the existing short extrapolation to 400 nm is allowed.')
        a, b = rows[:2]
    elif j < len(rows) and rows[j][0] == wave:
        return rows[j][1]
    else:
        a, b = rows[j-1:j+1]
    return a[1] + (wave-a[0])*(b[1]-a[1])/(b[0]-a[0])


def wavelengths(text):
    if ':' in text:
        first, last, step = map(float, text.split(':'))
        if step <= 0 or last < first:
            raise ValueError('Use first:last:positive_step.')
        values = [first+i*step for i in range(int((last-first)/step+1e-8)+1)]
    else:
        values = list(map(float, text.split(',')))
    if not values or any(x <= 0 for x in values) or len(values) != len(set(values)):
        raise ValueError('Wavelengths must be positive and distinct.')
    return values


def make_sim(args, wave, index):
    return assembly.make_sim('square', args.order, args.grid, index, wave,
                             args.period, args.device)


class CompactPort(_ReducedScatteringMixin, _StableLinearAlgebraMixin):
    """C2v port operators and full retained-order Poynting flux in the sector."""
    def __init__(self, sim, context, depth, layers):
        self._dtype, self._device = sim._dtype, sim._device
        self.stable_eig_grad = getattr(sim, 'stable_eig_grad', False)
        self.omega = sim.omega
        self.smatrix_size = 'half'
        self.thickness = [torch.as_tensor(depth, dtype=self._dtype,
                                         device=self._device)]*layers
        self.K1_norm, self.K2_norm = sim.K1_norm, sim.K2_norm
        self.sin_zeta, self.cos_zeta = sim.sin_zeta, sim.cos_zeta
        e, h = context.electric, context.magnetic
        with torch.no_grad():
            self._polarization_reference_v = h.mH @ sim.Vf @ e
            vi, vo = h.mH @ sim.Vi @ e, h.mH @ sim.Vo @ e
            self.input_interface = self._reduced_interface_s(
                self._polarization_reference_v, vi, input_side=True)
            self.output_interface = self._reduced_interface_s(
                self._polarization_reference_v, vo, input_side=False)
            self.source = e.mH @ assembly.source(sim)
            n = sim.order_N
            pairing = e.mH @ torch.cat((h[n:], -h[:n]), dim=0)
            self.flux_in, self.flux_out = pairing @ vi, pairing @ vo
            self.incident_flux = .5*torch.real(self.source.conj() @
                                              (self.flux_in @ self.source))

    def response(self, ps, qs, p0=None, q2=None, scale=1.):
        modes = []
        for left, right in zip(ps, qs):
            p, q = (p0+scale*left, right+scale*q2) if p0 is not None else (left, right)
            squared, electric = self._eig(p @ q)
            kz = self._positive_kz(squared)
            magnetic = self._magnetic_eigenvectors(p, q, electric, kz)
            modes.append({'electric': electric, 'magnetic': magnetic, 'kz': kz})
        transmission, reflection, _, _ = self._polarized_redheffer_scattering(
            modes, self.input_interface, self.output_interface)
        t, r = transmission @ self.source, reflection @ self.source
        R = .5*torch.real(r.conj() @ (self.flux_in @ r))/self.incident_flux
        T = .5*torch.real(t.conj() @ (self.flux_out @ t))/self.incident_flux
        return R, T


def free_coefficients(unit, context):
    n = context.scalar_dimension
    e, h = context.electric, context.magnetic
    p0 = e.mH @ torch.cat((h[n:], -h[:n]), dim=0)
    kx, ky = torch.diagonal(unit.K1_norm)[:, None], torch.diagonal(unit.K2_norm)[:, None]
    ex, ey = e[:n], e[n:]
    q2 = h.mH @ torch.cat((-kx*ky*ex+kx*kx*ey,
                           -ky*ky*ex+ky*kx*ey), dim=0)
    return p0, q2


def sequential(args, waves, indices, radii, context, variant, gradient=True):
    """Production and block-only paths use identical full simulation/flux APIs."""
    started = time.perf_counter()
    total = torch.zeros_like(radii) if gradient else None
    spectrum = []
    stages = dict(initialization=0., layers_and_eig=0., cascade_and_flux=0., backward=0.)
    for wave, index in zip(waves, indices):
        begin = time.perf_counter()
        sim = make_sim(args, wave, index)
        stages['initialization'] += time.perf_counter()-begin
        values = radii.detach().clone().requires_grad_(gradient)
        begin = time.perf_counter()
        depth = args.height/(args.layers*args.period)
        for radius in values:
            if variant == 'reference':
                sim.add_layer_circle_nvm(depth, radius, 1., index**2,
                    (assembly.center_of(sim),), nx=args.grid, ny=args.grid)
            else:
                assembly.add_candidate_layer(sim, radius, index, depth, args.grid, context)
        stages['layers_and_eig'] += time.perf_counter()-begin
        begin = time.perf_counter()
        incident = assembly.source(sim)
        transmitted, reflected = sim._solve_polarization_reduced_smatrix(incident_source=incident)
        incoming = assembly.flux(incident, sim.Vi, 1.)
        R = -assembly.flux(reflected, sim.Vi, -1.)/incoming
        T = assembly.flux(transmitted, sim.Vo, 1.)/incoming
        stages['cascade_and_flux'] += time.perf_counter()-begin
        if gradient:
            begin = time.perf_counter()
            total += torch.autograd.grad(R/len(waves), values)[0]
            stages['backward'] += time.perf_counter()-begin
        spectrum.append([wave, float(R.detach()), float(T.detach())])
        del sim, values, R, T, transmitted, reflected
    return dict(seconds=time.perf_counter()-started, spectrum=spectrum,
                gradient=total.tolist() if gradient else None, stages_seconds=stages)


def shared(args, waves, indices, radii, template, context, ports, unit, p0, q2,
           gradient=True):
    started = time.perf_counter()
    values = radii.detach().clone().requires_grad_(gradient)
    originals = []
    identity = template._eye(context.scalar_dimension)
    for radius in values:
        if args.material == 'PMMA':
            eps, _, normal = assembly.material_inputs(template, radius, 2., args.grid, context)
            originals.extend(((eps-identity)/3., normal))
        else:
            inputs = assembly.material_inputs(template, radius, indices[0], args.grid, context)
            p, q, _ = assembly.candidate_pq(unit, *inputs, context)
            originals.extend((p-p0, q-q2))
    prepared = time.perf_counter()
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
                p, q, _ = assembly.candidate_pq(port, eps, rule, normal, context)
                ps.append(p); qs.append(q)
            R, T = port.response(ps, qs)
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
    spectral_done = time.perf_counter()
    derivative = torch.autograd.grad(tuple(originals), values,
        grad_outputs=tuple(accumulated))[0] if gradient else None
    ended = time.perf_counter()
    return dict(seconds=ended-started, spectrum=spectrum,
                gradient=derivative.tolist() if derivative is not None else None,
                stages_seconds=dict(geometry_preparation=prepared-started,
                                    all_wavelengths_and_adjoint=spectral_done-prepared,
                                    geometry_vjp=ended-spectral_done),
                geometry_builds=args.layers, spectral_graphs_retained_at_once=1,
                material_inverse_or_PQ_shared_across_wavelengths=args.material != 'PMMA')


def compare(reference, candidate):
    a, b = (torch.tensor(x['spectrum'], dtype=torch.float64)[:, 1:]
            for x in (reference, candidate))
    ga, gb = (torch.tensor(x['gradient'], dtype=torch.float64)
              for x in (reference, candidate))
    finite = all(torch.isfinite(x).all().item() for x in (a, b, ga, gb))
    power, grad = float((a-b).abs().max()), float((ga-gb).abs().max())
    energy = max(float((x.sum(1)-1).abs().max()) for x in (a, b))
    bounded = bool(((a >= -1e-8) & (a <= 1+1e-8)).all() and
                   ((b >= -1e-8) & (b <= 1+1e-8)).all())
    passed = finite and bounded and power < 1e-8 and energy < 1e-6 and bool(
        torch.allclose(ga, gb, atol=1e-7, rtol=2e-5))
    return dict(passed=passed, power_max_abs=power, gradient_max_abs=grad,
                energy_max_abs=energy, finite=finite, lossless_bounds=bounded)


def matrix_check(args, template, context, index):
    with torch.no_grad():
        radius = torch.tensor(.27, dtype=torch.float64, device=args.device)
        inputs = assembly.material_inputs(template, radius, index, args.grid, context)
        p_ref, q_ref, _, _ = assembly.reference_pq(template, *inputs, context)
        p, q, diagnostics = assembly.candidate_pq(template, *inputs, context)
        errors = {key: assembly.relative_error(a, b) for key, a, b in
                  [('P', p, p_ref), ('Q', q, q_ref), ('PQ', p@q, p_ref@q_ref)]}
        timings = {}
        for name, function in [('reference', assembly.reference_pq),
                               ('block', assembly.candidate_pq)]:
            samples = []
            for _ in range(args.assembly_repeats):
                begin = time.perf_counter()
                answer = function(template, *inputs, context)
                samples.append(time.perf_counter()-begin)
                del answer
            timings[name] = dict(samples_seconds=samples,
                                 median_seconds=statistics.median(samples))
    return dict(passed=max(errors.values()) < 2e-9, errors=errors, **diagnostics,
                timing=timings,
                timing_scope='one-layer assembly only, input Fourier/normal and backward excluded')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--order', type=int, default=8)
    parser.add_argument('--grid', type=int, default=512)
    parser.add_argument('--layers', type=int, default=15)
    parser.add_argument('--material', choices=['n1.5', 'n2.0', 'n2.5', 'PMMA'], default='n2.5')
    parser.add_argument('--period', type=float, default=200.)
    parser.add_argument('--height', type=float, default=500.)
    parser.add_argument('--wavelengths', default='400:700:10')
    parser.add_argument('--device', choices=['cpu'], default='cpu')
    parser.add_argument('--cpu-threads', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--assembly-repeats', type=int, default=5)
    parser.add_argument('--check-fd', action='store_true')
    parser.add_argument('--csv', type=Path, default=HERE/'data/Szczurowski.csv')
    parser.add_argument('--output', type=Path, default=HERE/'results/report.json')
    args = parser.parse_args()
    if min(args.order, args.layers, args.repeats, args.assembly_repeats, args.cpu_threads) < 1:
        parser.error('Orders, layers, repeats and threads must be positive.')
    torch.set_num_threads(args.cpu_threads)
    probe = torch.ones(1, dtype=torch.float64, requires_grad=True)
    torch.autograd.grad(probe.square(), probe, grad_outputs=torch.ones_like(probe))
    waves = wavelengths(args.wavelengths)
    rows = load_pmma(args.csv) if args.material == 'PMMA' else None
    indices = [pmma_index(rows, w) if rows else float(args.material[1:]) for w in waves]
    template = make_sim(args, waves[0], indices[0])
    context = assembly.GeometryContext(template)
    unit = copy.copy(template)
    factor = torch.real(template.freq)
    unit.K1_norm, unit.K2_norm = template.K1_norm*factor, template.K2_norm*factor
    with torch.no_grad():
        p0, q2 = free_coefficients(unit, context)
    radii = torch.linspace(.10, .39, args.layers, dtype=torch.float64)
    report = dict(status='running', lattice='square', symmetry='c2v', material=args.material,
        order=args.order, grid=args.grid, layers=args.layers, period_nm=args.period,
        height_nm=args.height, radii_over_period=radii.tolist(), wavelengths_nm=waves,
        indices=indices, device=args.device, cpu_threads=args.cpu_threads,
        python=platform.python_version(), torch=str(torch.__version__), platform=platform.platform(),
        scalar_dimension=context.scalar_dimension, reduced_eigen_dimension=context.reduced_dimension,
        production_code_modified=False, timing_repetitions=args.repeats,
        timing_scope='forward plus arithmetic-mean reflectance radius gradient; per wave = total/Nlambda',
        common_basis_python_and_VJP_startup_excluded=True,
        source_sha256={p.relative_to(HERE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(HERE.rglob('*.py'))},
        pmma_csv_sha256=hashlib.sha256(args.csv.read_bytes()).hexdigest() if rows else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    def log(message):
        print(message, flush=True)
    report['matrix'] = matrix_check(args, template, context, indices[0])
    save(); log('PASS matrix' if report['matrix']['passed'] else 'FAIL matrix')
    begin = time.perf_counter()
    ports = [CompactPort(make_sim(args, w, n), context,
                         args.height/(args.layers*args.period), args.layers)
             for w, n in zip(waves, indices)]
    report['port_preparation_seconds'] = time.perf_counter()-begin
    runs = {name: [] for name in ('reference', 'block', 'shared')}
    # Alternate path order on repeated measurements to reduce systematic ordering bias.
    for repeat in range(args.repeats):
        order = ('reference', 'block', 'shared') if repeat % 2 == 0 else ('shared', 'block', 'reference')
        for variant in order:
            log(f'START {variant} repeat={repeat+1}/{args.repeats}')
            if variant == 'shared':
                value = shared(args, waves, indices, radii, template, context, ports, unit, p0, q2)
            else:
                value = sequential(args, waves, indices, radii, context, variant)
            runs[variant].append(value)
            report['runs'] = runs
            save(); log(f'DONE {variant}: {value["seconds"]:.3f} s')
    report['comparisons'] = []
    for i in range(args.repeats):
        for variant in ('block', 'shared'):
            report['comparisons'].append(dict(repeat=i+1, variant=variant,
                **compare(runs['reference'][i], runs[variant][i])))
    report['finite_difference'] = None
    if args.check_fd:
        direction = torch.linspace(-.7, .9, args.layers, dtype=torch.float64)
        direction /= torch.linalg.vector_norm(direction)
        h = 1e-5
        with torch.no_grad():
            plus = shared(args, waves, indices, radii+h*direction, template, context, ports, unit, p0, q2, False)
            minus = shared(args, waves, indices, radii-h*direction, template, context, ports, unit, p0, q2, False)
        fd = (statistics.mean(x[1] for x in plus['spectrum'])-
              statistics.mean(x[1] for x in minus['spectrum']))/(2*h)
        ad = float(torch.tensor(runs['shared'][0]['gradient'], dtype=torch.float64) @ direction)
        report['finite_difference'] = dict(step=h, autodiff=ad, finite_difference=fd,
            abs_error=abs(ad-fd), passed=abs(ad-fd) < 1e-6+3e-4*abs(fd))
    report['dispersion_negative_control'] = None
    if rows and max(indices)-min(indices) > 1e-6:
        n550 = pmma_index(rows, 550.)
        j = max(range(len(waves)), key=lambda i: abs(indices[i]-n550))
        with torch.no_grad():
            wrong = sequential(args, [waves[j]], [n550], radii, context, 'block', False)
        error = max(abs(a-b) for a, b in zip(wrong['spectrum'][0][1:],
                      runs['reference'][0]['spectrum'][j][1:]))
        report['dispersion_negative_control'] = dict(wavelength_nm=waves[j],
            actual_index=indices[j], wrong_fixed_index=n550,
            power_difference=error, passed=error > 1e-8)
    times = {name: statistics.median(x['seconds'] for x in values)
             for name, values in runs.items()}
    report['timing'] = dict(seconds=times,
        mean_seconds_per_wavelength={name: value/len(waves) for name, value in times.items()},
        shared_with_port_preparation_seconds=times['shared']+report['port_preparation_seconds'],
        shared_with_port_preparation_seconds_per_wavelength=(times['shared']+report['port_preparation_seconds'])/len(waves),
        block_speedup=times['reference']/times['block'],
        shared_speedup_warm=times['reference']/times['shared'],
        shared_speedup_with_port_preparation=times['reference']/(times['shared']+report['port_preparation_seconds']))
    passed = report['matrix']['passed'] and all(x['passed'] for x in report['comparisons'])
    passed = passed and all(report[k] is None or report[k]['passed']
                            for k in ('finite_difference', 'dispersion_negative_control'))
    report['status'] = 'passed' if passed else 'failed'
    save()
    log(json.dumps({k: report[k] for k in ('status', 'comparisons', 'finite_difference', 'timing')}, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
