"""CUDA-aware validation of native D6 assembly and spectral sharing.

Only centred, lossless circular dielectric/PMMA layers at normal x incidence.
All reported timings synchronize the device. CUDA CPU idle time is legitimate.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time
import torch
import hex_core as core


def configure_device(parser, requested):
    try:
        device = torch.device(requested)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    if device.type not in ('cpu', 'cuda'):
        parser.error('Use cpu, cuda, or cuda:N.')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            parser.error(f'CUDA unavailable: torch={torch.__version__}, built CUDA={torch.version.cuda}. '
                         'Use a CUDA-enabled PyTorch environment on an NVIDIA GPU.')
        index = device.index if device.index is not None else 0
        if index >= torch.cuda.device_count():
            parser.error(f'CUDA device index {index} is unavailable.')
        torch.cuda.set_device(index)
        device = torch.device('cuda', index)
    elif device.index is not None:
        parser.error('Use cpu without an index.')
    return device


class Timer:
    def __init__(self, device):
        self.device = device

    def sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def measure(self, label, function):
        self.sync()
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        wall0, cpu0 = time.perf_counter(), time.process_time()
        value = function()
        self.sync()
        wall, cpu = time.perf_counter()-wall0, time.process_time()-cpu0
        timing = dict(wall_seconds=wall, process_cpu_seconds=cpu,
                      wall_minus_cpu_seconds=wall-cpu,
                      cuda_synchronized=self.device.type == 'cuda')
        if self.device.type == 'cuda':
            timing.update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                          peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(self.device))
        print(f'DONE {label}: {wall:.6f}s (CPU {cpu:.6f}s)', flush=True)
        return value, timing


def measured(timer, label, function):
    value, timing = timer.measure(label, function)
    # CPU-oriented stage clocks inside the frozen core are NOT CUDA timings.
    value.pop('stages_seconds', None)
    value['seconds'], value['timing'] = timing['wall_seconds'], timing
    return value


class ProgressPort(core.CompactPort):
    def response(self, *args, **kwargs):
        print(f'SHARED wavelength={self.wave:g} nm', flush=True)
        return super().response(*args, **kwargs)


def check_matrix(args, template, context, index, timer):
    with torch.no_grad():
        radius = torch.tensor(.27, dtype=torch.float64, device=args.device)
        inputs = core.assembly.material_inputs(template, radius, index, args.grid, context)
        ref_inputs = core.assembly.material_inputs_reference(template, radius, index, args.grid, context)
        input_errors = {key: core.assembly.relative_error(a, b) for key, a, b in zip(
            ('eps', 'inverse_rule', 'normal_projection'), inputs, ref_inputs)}
        p_ref, q_ref, _, _ = core.assembly.reference_pq(template, *ref_inputs, context)
        p, q, diagnostic = core.assembly.candidate_pq(template, *inputs, context)
        errors = {key: core.assembly.relative_error(a, b) for key, a, b in
                  [('P', p, p_ref), ('Q', q, q_ref), ('PQ', p@q, p_ref@q_ref)]}
        del p_ref, q_ref, p, q
        timings = {}
        for label, function in [('reference', core.assembly.reference_pq),
                                 ('block', core.assembly.candidate_pq)]:
            prepared_inputs = ref_inputs if label == 'reference' else inputs
            warmup = function(template, *prepared_inputs, context)
            timer.sync()
            del warmup
            samples = []
            for repeat in range(args.assembly_repeats):
                answer, timing = timer.measure(f'matrix {label} {repeat+1}',
                                               lambda: function(template, *prepared_inputs, context))
                samples.append(timing)
                del answer
            timings[label] = dict(median_seconds=statistics.median(x['wall_seconds'] for x in samples),
                                  samples=samples)
    return dict(passed=max(errors.values()) < 2e-9 and max(input_errors.values()) < 2e-12,
                errors=errors, native_input_errors=input_errors, **diagnostic, timing=timings,
                timing_scope='one-layer assembly; no backward, geometry-input preparation or eig')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--order', type=int, default=8)
    parser.add_argument('--grid', type=int, default=512)
    parser.add_argument('--layers', type=int, default=15)
    parser.add_argument('--material', choices=['n1.5', 'n2.0', 'n2.5', 'PMMA'], default='n2.5')
    parser.add_argument('--wavelengths', default='400:700:10')
    parser.add_argument('--period', type=float, default=200.)
    parser.add_argument('--height', type=float, default=500.)
    parser.add_argument('--cpu-threads', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--assembly-repeats', type=int, default=5)
    parser.add_argument('--check-fd', action='store_true')
    parser.add_argument('--compare-cpu', action='store_true',
                        help='Also compare to the original CPU reference; use small cases.')
    parser.add_argument('--csv', type=Path, default=core.HERE/'data/Szczurowski.csv')
    parser.add_argument('--output', type=Path, default=core.HERE/'results/device_report.json')
    args = parser.parse_args()
    if min(args.order, args.layers, args.cpu_threads, args.repeats, args.assembly_repeats) < 1:
        parser.error('Order, layers, CPU threads, and repeats must be positive.')
    device = configure_device(parser, args.device)
    args.device = str(device)
    if args.compare_cpu and device.type != 'cuda':
        parser.error('--compare-cpu requires --device cuda:N.')
    torch.set_num_threads(args.cpu_threads)
    timer = Timer(device)
    probe = torch.ones(1, dtype=torch.float64, device=device, requires_grad=True)
    torch.autograd.grad(probe.square(), probe, grad_outputs=torch.ones_like(probe))
    timer.sync()
    waves = core.wavelengths(args.wavelengths)
    rows = core.load_pmma(args.csv) if args.material == 'PMMA' else None
    indices = [core.pmma_index(rows, w) if rows else float(args.material[1:]) for w in waves]
    gpu = None
    if device.type == 'cuda':
        properties = torch.cuda.get_device_properties(device)
        gpu = dict(name=properties.name, index=device.index, total_memory=properties.total_memory,
                   capability=list(torch.cuda.get_device_capability(device)),
                   uuid=str(getattr(properties, 'uuid', 'unavailable')))
    runtime = dict(python=platform.python_version(), torch=str(torch.__version__),
                   cuda_runtime=torch.version.cuda, platform=platform.platform(), device=args.device,
                   cpu_threads=args.cpu_threads, gpu=gpu, dtype='complex128')
    settings = dict(order=args.order, grid=args.grid, layers=args.layers, material=args.material,
        period_nm=args.period, height_nm=args.height, wavelengths_nm=waves, indices=indices,
        runtime=runtime, repeats=args.repeats, check_fd=args.check_fd, compare_cpu=args.compare_cpu,
        assembly_repeats=args.assembly_repeats,
        source_sha256={p.relative_to(core.HERE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(core.HERE.rglob('*.py'))},
        pmma_csv_sha256=hashlib.sha256(args.csv.read_bytes()).hexdigest() if rows else None)
    checkpoint = args.output.with_suffix('.checkpoint.json')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text(encoding='utf-8'))
        if state['settings'] != settings:
            raise RuntimeError('Checkpoint settings, runtime, GPU, sources or data differ; use a new output path.')
    else:
        state = dict(settings=settings, samples={name: [] for name in ('reference', 'block', 'shared')})
    def save():
        temporary = checkpoint.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2), encoding='utf-8')
        temporary.replace(checkpoint)
    template = core.make_sim(args, waves[0], indices[0])
    context = core.assembly.GeometryContext(template)
    radii = torch.linspace(.10, .39, args.layers, dtype=torch.float64, device=device)
    print('Untimed one-wavelength forward/backward warmup.', flush=True)
    warmup = core.sequential(args, [waves[0]], [indices[0]], radii, context, 'reference')
    timer.sync()
    del warmup
    if 'matrix' not in state:
        state['matrix'] = check_matrix(args, template, context, indices[0], timer)
        save()
    for repeat in range(args.repeats):
        for variant in ('reference', 'block'):
            samples = state['samples'][variant]
            while len(samples) <= repeat:
                samples.append([])
            for j in range(len(samples[repeat]), len(waves)):
                wave, index = waves[j], indices[j]
                print(f'START {variant} repeat={repeat+1} wave={j+1}/{len(waves)} ({wave:g} nm)', flush=True)
                value = measured(timer, f'{variant} {wave:g}nm', lambda: core.sequential(
                    args, [wave], [index], radii, context, variant))
                samples[repeat].append(value)
                save()
    unit = copy.copy(template)
    factor = torch.real(template.freq)
    unit.K1_norm, unit.K2_norm = template.K1_norm*factor, template.K2_norm*factor
    unit.Kx_norm_dn = template.Kx_norm_dn*factor
    unit.Ky_norm_dn = template.Ky_norm_dn*factor
    with torch.no_grad():
        p0, q2 = core.free_coefficients(unit, context)
    def prepare_ports():
        ports = []
        for wave, index in zip(waves, indices):
            port = ProgressPort(core.make_sim(args, wave, index), context,
                                args.height/(args.layers*args.period), args.layers)
            port.wave = wave
            ports.append(port)
        return ports
    ports, port_timing = timer.measure('port preparation', prepare_ports)
    state['port_preparation'] = port_timing
    while len(state['samples']['shared']) < args.repeats:
        value = measured(timer, 'shared all wavelengths', lambda: core.shared(
            args, waves, indices, radii, template, context, ports, unit, p0, q2))
        state['samples']['shared'].append(value)
        save()
    def aggregate(variant, repeat):
        samples = state['samples'][variant][repeat]
        result = dict(seconds=sum(x['seconds'] for x in samples),
                      spectrum=[x['spectrum'][0] for x in samples],
                      gradient=[sum(x['gradient'][i] for x in samples)/len(waves)
                                for i in range(args.layers)])
        if device.type == 'cuda':
            result['peak_cuda_allocated_bytes'] = max(x['timing']['peak_cuda_allocated_bytes'] for x in samples)
        return result
    runs = {v: [aggregate(v, i) for i in range(args.repeats)] for v in ('reference', 'block')}
    runs['shared'] = state['samples']['shared']
    comparisons = [dict(repeat=i+1, variant=v, **core.compare(runs['reference'][i], runs[v][i]))
                   for i in range(args.repeats) for v in ('block', 'shared')]
    per_wave = [dict(repeat=i+1, wavelength_nm=a['spectrum'][0][0], **core.compare(a, b))
                for i in range(args.repeats)
                for a, b in zip(state['samples']['reference'][i], state['samples']['block'][i])]
    finite_difference = None
    if args.check_fd:
        direction = torch.linspace(-.7, .9, args.layers, dtype=torch.float64, device=device)
        direction /= torch.linalg.vector_norm(direction)
        h = 1e-5
        with torch.no_grad():
            plus = core.shared(args, waves, indices, radii+h*direction, template, context, ports, unit, p0, q2, False)
            minus = core.shared(args, waves, indices, radii-h*direction, template, context, ports, unit, p0, q2, False)
        finite = (statistics.mean(x[1] for x in plus['spectrum'])-
                  statistics.mean(x[1] for x in minus['spectrum']))/(2*h)
        automatic = float(torch.tensor(runs['shared'][0]['gradient'], dtype=torch.float64) @ direction.cpu())
        finite_difference = dict(step=h, finite_difference=finite, autodiff=automatic,
            abs_error=abs(finite-automatic), passed=abs(finite-automatic) < 1e-6+3e-4*abs(finite))
    cpu_comparisons = None
    if args.compare_cpu:
        print('Comparing to independent CPU reference (excluded from GPU timings).', flush=True)
        cpu_args = copy.copy(args)
        cpu_args.device = 'cpu'
        cpu_template = core.make_sim(cpu_args, waves[0], indices[0])
        cpu_context = core.assembly.GeometryContext(cpu_template)
        reference_cpu = core.sequential(cpu_args, waves, indices, radii.cpu(), cpu_context, 'reference')
        cpu_comparisons = [dict(variant=v, **core.compare(reference_cpu, runs[v][0]))
                           for v in ('reference', 'block', 'shared')]
    negative = None
    if rows and max(indices)-min(indices) > 1e-6:
        fixed = core.pmma_index(rows, 550.)
        j = max(range(len(waves)), key=lambda i: abs(indices[i]-fixed))
        with torch.no_grad():
            wrong = core.sequential(args, [waves[j]], [fixed], radii, context, 'block', False)
        error = max(abs(a-b) for a,b in zip(wrong['spectrum'][0][1:], runs['reference'][0]['spectrum'][j][1:]))
        negative = dict(wavelength_nm=waves[j], actual_index=indices[j], wrong_fixed_index=fixed,
                        power_difference=error, passed=error > 1e-8)
    passed = state['matrix']['passed'] and all(x['passed'] for x in comparisons+per_wave)
    passed = passed and (finite_difference is None or finite_difference['passed'])
    passed = passed and (cpu_comparisons is None or all(x['passed'] for x in cpu_comparisons))
    passed = passed and (negative is None or negative['passed'])
    times = {v: statistics.median(x['seconds'] for x in values) for v, values in runs.items()}
    prep = port_timing['wall_seconds']
    report = dict(**settings, status='passed' if passed else 'failed', lattice='hexagonal', symmetry='d6',
        cuda_execution_performed=device.type == 'cuda', production_code_modified=False,
        scope='normal x incidence, centred lossless circular layers/substrate; mean R and radius gradient',
        timing_scope='synchronized end-to-end wall time; per-wave times summed; checkpoint I/O excluded',
        cuda_cpu_gap_rejection_enabled=False,
        common_basis_python_and_VJP_startup_excluded=True, one_wavelength_warmup_excluded=True,
        scalar_dimension=context.scalar_dimension,
        reduced_eigen_dimension=context.reduced_dimension, matrix=state['matrix'], runs=runs,
        comparisons=comparisons, per_wavelength_block_comparisons=per_wave,
        finite_difference=finite_difference, cpu_reference_comparisons=cpu_comparisons,
        dispersion_negative_control=negative, port_preparation_timing=port_timing,
        timing=dict(seconds=times, mean_seconds_per_wavelength={k:v/len(waves) for k,v in times.items()},
            shared_with_port_preparation_seconds=times['shared']+prep,
            shared_with_port_preparation_seconds_per_wavelength=(times['shared']+prep)/len(waves),
            block_speedup=times['reference']/times['block'],
            shared_speedup_with_port_preparation=times['reference']/(times['shared']+prep)))
    timer.sync()
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('status', 'cuda_execution_performed', 'comparisons',
                                          'finite_difference', 'cpu_reference_comparisons', 'timing')}, indent=2), flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())

