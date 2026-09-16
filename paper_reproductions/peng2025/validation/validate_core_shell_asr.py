"""Physical and numerical regression tests for core-shell ASR corrections."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from rcwa_solver_auto import ASROptions, AutoRCWA, GroupTheoryOptions, Lattice, OutputSpec
from paper_reproductions.peng2025.common import _power_observables


def simulation(order=3, g=0.001, symmetry=False, backend=AutoRCWA, eps_out=1):
    sim = backend(
        freq=0.4, order=[order, order], lattice=Lattice.square(1),
        asr=ASROptions(G=g, circle_G=g),
        group_theory=GroupTheoryOptions(enabled=symmetry, polarization='x' if symmetry else None),
        outputs=OutputSpec(smatrix_size='half', fields='none'),
        verify_cascade=False, dtype=torch.complex128, device='cpu',
    )
    sim.add_input_layer(eps=1)
    sim.add_output_layer(eps=eps_out)
    sim.set_incident_angle(0, 0)
    return sim


def run():
    checks = {}
    sim = simulation()
    mapping = sim.build_stepped_circle_asr_mapping(256, 256, 14/62, 30/62, staircase_grid=64)
    physical = torch.linspace(0, 1, 10001, dtype=torch.float64)
    coordinate, derivative = sim._inverse_piecewise_asr_axis(physical, mapping.x_breaks, mapping.u_breaks)
    interval = torch.bucketize(physical, mapping.x_breaks[1:-1])
    u0, u1 = mapping.u_breaks[interval], mapping.u_breaks[interval + 1]
    x0, x1 = mapping.x_breaks[interval], mapping.x_breaks[interval + 1]
    local = (coordinate - u0)/(u1 - u0)
    recovered = x0 + (x1-x0)*local + (sim.asr_G*(u1-u0)-(x1-x0))/(2*math.pi)*torch.sin(2*math.pi*local)
    error = float((recovered-physical).abs().max())
    checks['inverse_roundtrip'] = dict(maximum_error=error, passed=error < 2e-14 and bool((derivative > 0).all()))
    te, th, _ = sim._peng_conversion_matrices(mapping)
    n = sim.order_N
    zero, identity = torch.zeros((n, n), dtype=torch.complex128), torch.eye(n, dtype=torch.complex128)
    flux = torch.cat((torch.cat((zero, identity), 1), torch.cat((-identity, zero), 1)), 0)
    error = float(torch.linalg.vector_norm(te.mH @ flux @ th - flux) / torch.linalg.vector_norm(flux))
    checks['interface_flux_duality'] = dict(relative_error=error, passed=error < 1e-12)

    sim = simulation(g=0.1)
    mapping = sim.build_double_matched_circle_asr_mapping(128, 128, 14/62, 30/62)
    indices = (-torch.arange(128)) % 128
    errors = []
    for field, sign in ((mapping.x_u, 1), (mapping.y_v, 1), (mapping.x_v, -1), (mapping.y_u, -1)):
        errors.extend(float((field - sign*field.index_select(axis, indices)).abs().max()) for axis in (0, 1))
    checks['double_map_mirror_parity'] = dict(maximum_error=max(errors), passed=max(errors) < 1e-12)

    def paper(grid, uniform=False, symmetry=False):
        sim = simulation(order=4, symmetry=symmetry)
        sim.add_layer_circle_shell_peng_asr(
            .15, .18, .35, 2.25 if uniform else 1., 2.25 if uniform else 2.,
            2.25 if uniform else 9., nx=grid, ny=grid, staircase_grid=32,
        )
        sim.solve_global_smatrix()
        return _power_observables(sim)

    a, b, reduced = paper(128), paper(256), paper(128, symmetry=True)
    keys = ('reflectance', 'transmittance', 'absorptance')
    error = max(abs(a[k]-b[k]) for k in keys)
    checks['asr_quadrature_grid_invariance'] = dict(maximum_RTA_difference=error, grid128=a, grid256=b, passed=error < 1e-11)
    error = max(abs(a[k]-reduced[k]) for k in keys)
    checks['asr_full_vs_c2v'] = dict(maximum_RTA_difference=error, passed=error < 1e-10)

    uniform = paper(128, uniform=True)
    reference = simulation(order=4)
    reference.add_layer(.15, eps=2.25)
    reference.solve_global_smatrix()
    reference = _power_observables(reference)
    error = max(abs(uniform[k]-reference[k]) for k in keys)
    checks['uniform_slab_reference'] = dict(maximum_RTA_difference=error, asr=uniform, reference=reference, passed=error < 1e-10)

    # Independent Cartesian raster baseline for a resolved dielectric shell.
    # This is a finite-order comparison, unlike the exact slab identity above.
    cartesian, covariant = simulation(order=8, g=.5), simulation(order=8, g=.5)
    mapping = cartesian.build_stepped_circle_asr_mapping(384, 384, .18, .35, staircase_grid=16)
    epsilon = torch.where(mapping.region_grid == 2, 9., torch.where(mapping.region_grid == 1, 2., 1.))
    cartesian.add_layer(.15, eps=epsilon.repeat_interleave(24, 0).repeat_interleave(24, 1))
    covariant.add_layer_circle_shell_peng_asr(
        .15, .18, .35, 1., 2., 9., nx=384, ny=384, staircase_grid=16,
        normal_vector_factorization=True,
    )
    cartesian.solve_global_smatrix()
    covariant.solve_global_smatrix()
    a, b = _power_observables(cartesian), _power_observables(covariant)
    error = max(abs(a[k]-b[k]) for k in keys)
    checks['covariant_nv_dielectric_reference'] = dict(
        maximum_RTA_difference=error, tolerance=.001, cartesian=a, covariant=b,
        passed=error < .001 and not b['passivity_warning'],
    )

    from paper_reproductions.peng2025.adaptive_ports import _AdaptivePortRCWA
    # Resolve both independent discretizations further than the N=8 NV
    # smoke above; at N=8 their difference was 1.38e-3, at N=12 <1e-3.
    reference = simulation(order=12, g=.5)
    reference.add_layer(.15, eps=epsilon.repeat_interleave(24, 0).repeat_interleave(24, 1))
    reference.solve_global_smatrix()
    a = _power_observables(reference)
    adaptive = simulation(order=12, g=.5, backend=_AdaptivePortRCWA)
    adaptive.add_layer_circle_shell_peng_asr(
        .15, .18, .35, 1., 2., 9., nx=384, ny=384, staircase_grid=16,
    )
    adaptive.solve_global_smatrix()
    c = _power_observables(adaptive)
    error = max(abs(a[k]-c[k]) for k in keys)
    checks['shared_adaptive_dielectric_reference'] = dict(
        maximum_RTA_difference=error, tolerance=.001, cartesian=a, adaptive=c,
        passed=error < .001 and not c['passivity_warning'],
    )
    pair = []
    for symmetry in (False, True):
        adaptive = simulation(order=4, symmetry=symmetry, backend=_AdaptivePortRCWA)
        adaptive.add_layer_circle_shell_peng_asr(
            .15, .18, .35, 1., 2., 9., nx=128, ny=128, staircase_grid=32,
        )
        adaptive.solve_global_smatrix()
        pair.append(_power_observables(adaptive))
    error = max(abs(pair[0][k]-pair[1][k]) for k in keys)
    checks['shared_adaptive_full_vs_c2v'] = dict(maximum_RTA_difference=error, passed=error < 1e-10)

    # A zero-thickness pattern must leave only the ordinary air/PI interface.
    adaptive = simulation(order=4, backend=_AdaptivePortRCWA, eps_out=3.5)
    adaptive.add_layer_circle_shell_peng_asr(
        0., 14/62, 30/62, -2e5+4e5j, 1., -2e5+4e5j,
        nx=128, ny=128, staircase_grid=64,
    )
    adaptive.solve_global_smatrix()
    a = _power_observables(adaptive)
    expected = ((math.sqrt(3.5)-1)/(math.sqrt(3.5)+1))**2
    error = max(abs(a['reflectance']-expected), abs(a['transmittance']-(1-expected)))
    checks['shared_adaptive_zero_thickness_fresnel'] = dict(maximum_RTA_difference=error, passed=error < 1e-9)

    # Compare the batched homogeneous-port solve to the old dense algebra.
    port = simulation(order=2)
    port.set_incident_angle(.31, .47)
    medium = port._cartesian_e_to_h(port._positive_kz(2.5 * 1.2 - port.Kx_norm_dn**2 - port.Ky_norm_dn**2), mu=1.2)
    inverse = torch.linalg.solve(port.Vf + medium, torch.eye(2*port.order_N, dtype=torch.complex128))
    delta = port.Vf - medium
    expected = (2*inverse@medium, -inverse@delta, inverse@delta, 2*inverse@port.Vf)
    actual = port._interface_s(medium, input_side=True)
    error = max(float((x-y).abs().max()) for x,y in zip(actual, expected))
    checks['batched_port_dense_reference'] = dict(maximum_error=error, passed=error < 1e-12)

    from paper_reproductions.peng2025.common import Numerics, PaperGeometry, SilverDrude, simulate_matched_primitive
    silver = simulate_matched_primitive(
        1.95, lattice_kind='square', geometry=PaperGeometry(), drude=SilverDrude(),
        numerics=Numerics(16, 16, 256, 256, asr_g=.001, solver='paper-asr', use_symmetry=True),
        device=torch.device('cpu'),
    )
    checks['silver_core_shell_passivity'] = dict(
        **{k: silver[k] for k in keys},
        passed=not silver['passivity_warning'] and all(math.isfinite(silver[k]) for k in keys),
    )
    return dict(passed=all(c['passed'] for c in checks.values()), checks=checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent/'results/core_shell_asr.json')
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
