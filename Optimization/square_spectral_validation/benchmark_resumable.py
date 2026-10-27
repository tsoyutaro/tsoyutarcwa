"""Progress/checkpoint driver using unchanged benchmark_square physics.

Reference/block paths save after each wavelength. Wall and process CPU times
are both recorded; heavily interrupted samples are retained as excluded
attempts and repeated. Resume requires identical settings and source hashes.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import time
import torch
import benchmark_square as core


class ProgressPort(core.CompactPort):
    def response(self, *args, **kwargs):
        print(f'SHARED wavelength={self.wave:g} nm n={self.index:.9f}', flush=True)
        return super().response(*args, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--order', type=int, default=8)
    parser.add_argument('--grid', type=int, default=512)
    parser.add_argument('--layers', type=int, default=15)
    parser.add_argument('--material', choices=['n1.5', 'n2.0', 'n2.5', 'PMMA'], default='PMMA')
    parser.add_argument('--wavelengths', default='400:700:10')
    parser.add_argument('--period', type=float, default=200.)
    parser.add_argument('--height', type=float, default=500.)
    parser.add_argument('--assembly-repeats', type=int, default=5)
    parser.add_argument('--csv', type=Path, default=core.HERE/'data/Szczurowski.csv')
    parser.add_argument('--output', type=Path, default=core.HERE/'results/pmma_progress.json')
    args = parser.parse_args()
    args.device = 'cpu'
    if min(args.order, args.layers, args.assembly_repeats) < 1:
        parser.error('Order, layers and assembly repeats must be positive.')
    torch.set_num_threads(1)
    probe = torch.ones(1, dtype=torch.float64, requires_grad=True)
    torch.autograd.grad(probe.square(), probe, grad_outputs=torch.ones_like(probe))
    waves = core.wavelengths(args.wavelengths)
    rows = core.load_pmma(args.csv) if args.material == 'PMMA' else None
    indices = [core.pmma_index(rows, w) if rows else float(args.material[1:]) for w in waves]
    hashes = {p.relative_to(core.HERE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(core.HERE.rglob('*.py'))}
    settings = dict(order=args.order, grid=args.grid, layers=args.layers, material=args.material,
                    period_nm=args.period, height_nm=args.height, wavelengths_nm=waves,
                    indices=indices, source_sha256=hashes,
                    pmma_csv_sha256=hashlib.sha256(args.csv.read_bytes()).hexdigest() if rows else None)
    checkpoint = args.output.with_suffix('.checkpoint.json')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text(encoding='utf-8'))
        if state['settings'] != settings:
            raise RuntimeError('Checkpoint differs from settings, source or data; use a new output path.')
    else:
        state = dict(settings=settings, reference=[], block=[], excluded_timing_attempts=[])
    def save_state():
        temporary = checkpoint.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2), encoding='utf-8')
        temporary.replace(checkpoint)
    def measure(label, function):
        for attempt in range(1, 4):
            wall0, cpu0 = time.perf_counter(), time.process_time()
            value = function()
            wall, cpu = time.perf_counter()-wall0, time.process_time()-cpu0
            timing = dict(wall_seconds=wall, process_cpu_seconds=cpu,
                          wall_minus_cpu_seconds=wall-cpu)
            # Single CPU-thread kernels: a long wall/CPU gap indicates a
            # suspended or heavily descheduled measurement, not solver work.
            if wall-cpu > max(10., .10*cpu):
                state['excluded_timing_attempts'].append(dict(label=label, attempt=attempt, **timing))
                save_state()
                print(f'INTERRUPTED {label}: wall={wall:.2f}s CPU={cpu:.2f}s; repeating', flush=True)
                continue
            print(f'DONE {label}: wall={wall:.2f}s CPU={cpu:.2f}s', flush=True)
            return value, timing
        raise RuntimeError('Three interrupted timing attempts; checkpoint saved for later resume.')
    template = core.make_sim(args, waves[0], indices[0])
    context = core.assembly.GeometryContext(template)
    radii = torch.linspace(.10, .39, args.layers, dtype=torch.float64)
    if 'matrix' not in state:
        state['matrix'] = core.matrix_check(args, template, context, indices[0])
        save_state()
    for variant in ('reference', 'block'):
        for j in range(len(state[variant]), len(waves)):
            wave, index = waves[j], indices[j]
            print(f'START {variant} {j+1}/{len(waves)}: {wave:g} nm n={index:.9f}', flush=True)
            value, timing = measure(f'{variant} {wave:g}nm', lambda: core.sequential(
                args, [wave], [index], radii, context, variant))
            state[variant].append(dict(value=value, timing=timing))
            save_state()
    unit = copy.copy(template)
    factor = torch.real(template.freq)
    unit.K1_norm, unit.K2_norm = template.K1_norm*factor, template.K2_norm*factor
    with torch.no_grad():
        p0, q2 = core.free_coefficients(unit, context)
    def prepare_ports():
        ports = []
        for wave, index in zip(waves, indices):
            port = ProgressPort(core.make_sim(args, wave, index), context,
                                args.height/(args.layers*args.period), args.layers)
            port.wave, port.index = wave, index
            ports.append(port)
        return ports
    if 'shared' not in state:
        ports, prep = measure('port preparation', prepare_ports)
        value, timing = measure('shared all wavelengths', lambda: core.shared(
            args, waves, indices, radii, template, context, ports, unit, p0, q2))
        state['shared'], state['shared_timing'], state['port_preparation'] = value, timing, prep
        save_state()
    def aggregate(variant):
        samples = state[variant]
        return dict(seconds=sum(x['timing']['wall_seconds'] for x in samples),
            process_cpu_seconds=sum(x['timing']['process_cpu_seconds'] for x in samples),
            spectrum=[x['value']['spectrum'][0] for x in samples],
            gradient=[sum(x['value']['gradient'][i] for x in samples)/len(waves)
                      for i in range(args.layers)],
            stages_seconds={k: sum(x['value']['stages_seconds'][k] for x in samples)
                            for k in samples[0]['value']['stages_seconds']})
    runs = dict(reference=[aggregate('reference')], block=[aggregate('block')], shared=[state['shared']])
    comparisons = [dict(repeat=1, variant=v, **core.compare(runs['reference'][0], runs[v][0]))
                   for v in ('block', 'shared')]
    negative = None
    if rows and max(indices)-min(indices) > 1e-6:
        fixed = core.pmma_index(rows, 550.)
        j = max(range(len(waves)), key=lambda i: abs(indices[i]-fixed))
        with torch.no_grad():
            wrong = core.sequential(args, [waves[j]], [fixed], radii, context, 'block', False)
        error = max(abs(a-b) for a,b in zip(wrong['spectrum'][0][1:], runs['reference'][0]['spectrum'][j][1:]))
        negative = dict(wavelength_nm=waves[j], actual_index=indices[j], wrong_fixed_index=fixed,
                        power_difference=error, passed=error > 1e-8)
    times = {k: v[0]['seconds'] for k,v in runs.items()}
    prep = state['port_preparation']['wall_seconds']
    passed = state['matrix']['passed'] and all(x['passed'] for x in comparisons)
    passed = passed and (negative is None or negative['passed'])
    result = dict(**settings, status='passed' if passed else 'failed', lattice='square', symmetry='c2v',
        device='cpu', cpu_threads=1, python=platform.python_version(), torch=str(torch.__version__),
        platform=platform.platform(), radii_over_period=radii.tolist(),
        scalar_dimension=context.scalar_dimension, reduced_eigen_dimension=context.reduced_dimension,
        production_code_modified=False, timing_repetitions=1, matrix=state['matrix'], runs=runs,
        comparisons=comparisons, finite_difference=None, dispersion_negative_control=negative,
        port_preparation_seconds=prep,
        timing_scope='forward plus mean-R radius gradient; sequential sum of per-wave wall times; checkpoint I/O excluded',
        common_basis_python_and_VJP_startup_excluded=True,
        timing_guard='exclude and retry if wall-CPU > max(10 seconds, 10% CPU); all attempts retained',
        excluded_timing_attempts=state['excluded_timing_attempts'],
        timing=dict(seconds=times, mean_seconds_per_wavelength={k:v/len(waves) for k,v in times.items()},
            shared_with_port_preparation_seconds=times['shared']+prep,
            shared_with_port_preparation_seconds_per_wavelength=(times['shared']+prep)/len(waves),
            block_speedup=times['reference']/times['block'], shared_speedup_warm=times['reference']/times['shared'],
            shared_speedup_with_port_preparation=times['reference']/(times['shared']+prep)))
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(dict(status=result['status'], comparisons=comparisons, timing=result['timing']), indent=2), flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
