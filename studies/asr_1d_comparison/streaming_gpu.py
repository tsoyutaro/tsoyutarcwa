"""Bound ASR layer residency while reusing the production tensor kernels.

For one wavelength, prepare one material profile, compute both polarizations,
and retain only compact W/V/gamma traces on the selected torch device. The
unchanged production scattering solver then joins those traces. No dense
linear algebra is moved to the host, and the numerical source fingerprint of
existing comparison checkpoints stays valid. The additional residency
controller hash is recorded in each new case's execution metadata.
"""
from __future__ import annotations

import gc
import hashlib
import math
import time
from pathlib import Path

from paper_reproductions.vallius2002.devices import resolve_execution
from paper_reproductions.vallius2002.solver import LayerSpec, PreparedStack
from .adapters import layer_specs, scalar_observables


def release_unused_cuda(device):
    """Release completed-case references and unused allocator blocks only."""
    gc.collect()
    if str(device).startswith('cuda'):
        import torch
        with torch.cuda.device(device):
            torch.cuda.empty_cache()


class _ModalLayer:
    """Compact production traces for one fixed wavelength and profile."""
    def __init__(self, prepared, traces, k, alpha0, retention, q_projection):
        self.rayleigh_orders = prepared.rayleigh_orders.detach().clone()
        self.internal_count = prepared.internal_count
        self.quadrature_points = prepared.quadrature_points
        self.traces = traces
        self.k, self.alpha0 = k, alpha0
        self.retention, self.q_projection = retention, q_projection

    def modes(self, k, alpha0, polarization, retention, q_projection):
        if (k != self.k or alpha0 != self.alpha0 or retention != self.retention
                or q_projection != self.q_projection):
            raise ValueError('Modal traces can only be reused for their prepared wavelength and options')
        return self.traces[polarization]


def _prepare_modal_layer(spec, stack, quadrature, k, alpha0):
    from paper_reproductions.vallius2002.torch_backend import TorchPreparedLayer
    prepared = TorchPreparedLayer(spec, stack.method, stack.harmonics,
                                  stack.internal_count, quadrature, stack.G,
                                  device=stack.device)
    traces = {}
    for polarization in ('TE', 'TM'):
        # Compact independent storage also prevents a view from retaining a
        # larger internal eigenvector or coefficient allocation.
        traces[polarization] = tuple(value.detach().clone() for value in prepared.modes(
            k, alpha0, polarization, stack.retention, stack.q_projection))
    return _ModalLayer(prepared, traces, k, alpha0, stack.retention, stack.q_projection)


def _prepare_mode_stack(layers, wavelength, *, quadrature, execution, **options):
    """Validate using PreparedStack, then store only wavelength-specific modes.

    A homogeneous zero-thickness placeholder runs the existing argument and
    cutoff validation without preparing dense coefficients or linear algebra.
    It is replaced with the actual layer sequence before any solve.
    """
    layers = tuple(layers)
    if not layers:
        raise ValueError('at least one layer is required')
    period = layers[0].period
    if any(abs(layer.period-period) > 1e-12*period for layer in layers):
        raise ValueError('all layers must share one physical period')
    placeholder = LayerSpec(0., (0., period), (options.get('epsilon_in', 1.),), period=period)
    stack = PreparedStack([placeholder], quadrature=quadrature,
                          device='cpu', backend='scipy', **options)
    stack.layers = layers
    stack.prepared_layers = []
    stack.execution = dict(execution)
    stack.device, stack.backend = execution['device'], execution['backend']
    if stack.backend != 'torch':
        raise ValueError('Streaming modal preparation requires the torch backend')
    _, _, k, alpha0, _ = stack._wavelength_parameters(wavelength, 1e-12)
    profiles = {}
    for spec in layers:
        key = (spec.breaks, spec.epsilon, spec.period)
        if key not in profiles:
            profiles[key] = _prepare_modal_layer(spec, stack, quadrature, k, alpha0)
        stack.prepared_layers.append(profiles[key])
    stack.execution.update(
        layer_storage='streamed_production_modal_traces',
        coefficient_preparation_device='cpu', linear_algebra_device=stack.device,
        unique_profiles=len(profiles), large_prepared_layers_live_limit=1,
        residency_controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return stack


def _modal_storage_bytes(stack):
    unique = {id(layer): layer for layer in stack.prepared_layers}
    return sum(value.numel()*value.element_size() for layer in unique.values()
               for traces in layer.traces.values() for value in traces)


def scalar_case_streamed(study, config, slices, order, wavelength, materials, *, method='asr',
                         oversampling=3, G=.001, quadrature=192, device='cuda', backend='auto',
                         diagnostics=False, q_projection='galerkin'):
    """One-wavelength TE/TM comparison using at most one large prepared layer."""
    if not math.isfinite(float(order)) or int(order) != order or order < 0:
        raise ValueError('order must be a nonnegative integer')
    execution = resolve_execution(device, backend)
    if execution['backend'] != 'torch':
        from .adapters import scalar_case
        return scalar_case(study, config, slices, order, wavelength, materials, method=method,
                           oversampling=oversampling, G=G, quadrature=quadrature, device=device,
                           backend=backend, diagnostics=diagnostics, q_projection=q_projection)
    import torch
    layers, substrate = layer_specs(study, config, slices, materials)
    start = time.perf_counter()
    stack = None
    release_unused_cuda(execution['device'])
    if execution['device'].startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(execution['device'])
    try:
        with torch.no_grad():
            stack = _prepare_mode_stack(layers, wavelength/config['geometry']['period_nm'],
                method=method, harmonics=2*int(order)+1,
                oversampling=oversampling if method == 'asr' else 1,
                quadrature=quadrature, G=G, epsilon_in=1., epsilon_out=substrate,
                diagnostics=diagnostics, retention='smallest_abs', q_projection=q_projection,
                execution=execution)
            polarizations, audit = {}, {}
            for polarization in ('TE', 'TM'):
                stack.polarization = polarization
                solution = stack.solve(wavelength/config['geometry']['period_nm'])
                polarizations[polarization] = scalar_observables(study, solution)
                audit[polarization] = dict(max_boundary_condition=(
                    float(solution['max_boundary_condition']) if diagnostics else None),
                    cutoff_regularized=bool(solution['cutoff_regularized']))
            memory = dict(retained_modal_bytes=_modal_storage_bytes(stack))
            if execution['device'].startswith('cuda'):
                memory.update(peak_allocated_bytes=int(torch.cuda.max_memory_allocated(stack.device)),
                              peak_reserved_bytes=int(torch.cuda.max_memory_reserved(stack.device)))
            result = dict(order=int(order), harmonics=stack.harmonics, slices=slices,
                total_finite_layers=len(layers), wavelength_nm=float(wavelength),
                method=method, oversampling=oversampling if method == 'asr' else 1,
                eigen_dimension=int(stack.internal_count), G=G if method == 'asr' else None,
                q_projection=q_projection if method == 'asr' else 'direct',
                quadrature_actual=max(layer.quadrature_points for layer in stack.prepared_layers),
                polarizations=polarizations, diagnostics=audit, execution=dict(stack.execution),
                memory=memory, runtime_seconds=time.perf_counter()-start,
                coefficient_method='adaptive quadrature' if method == 'asr' else 'analytic intervals')
            return result
    finally:
        stack = None
        release_unused_cuda(execution['device'])
