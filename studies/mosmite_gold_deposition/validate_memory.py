"""Compare full complex S, finite-sample powers and retained tensor storage."""
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np

from model import HERE, load_materials, parser, surface_rcwa
from memory import memory_safe_class, tensor_storage_bytes
from stack import finite_sample, silica_n


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--output-dir', type=Path, default=HERE/'results/memory_validation')
    opts = p.parse_args()
    sys.path.insert(0, str(HERE.parents[1]))
    import torch
    from rcwa_ext import AutoRCWA, GroupTheoryOptions, Lattice, NVMOptions, OutputSpec
    torch.set_num_threads(2)
    checks, storage = [], []
    def check(name, error, tolerance=2e-9):
        if not math.isfinite(error) or error > tolerance:
            raise AssertionError((name, error, tolerance))
        checks.append(dict(name=name, max_absolute_error=float(error), tolerance=tolerance, passed=True))
        print(f'PASS {name}: {error:.3g}', flush=True)

    # Compare every entry of every bidirectional S block, not only R+T+A.
    # Full-basis test mixes a raster layer with analytic shell layers.
    for d6 in [True, False]:
        matrices, retained = [], []
        for reduced_storage in [False, True]:
            cls = memory_safe_class(AutoRCWA) if reduced_storage else AutoRCWA
            sim = cls(freq=1/6, order=[2, 2], lattice=Lattice.triangular(1.),
                cascade='redheffer', outputs=OutputSpec(smatrix_size='full', fields='none'),
                nvm=NVMOptions(grid=(64, 64)),
                group_theory=GroupTheoryOptions(enabled=True, strict=True, symmetry='d6', polarization='x') if d6 else False,
                dtype=torch.complex128, device=opts.device)
            sim.add_input_layer(eps=1.)
            sim.add_output_layer(eps=(1.57+1e-6j)**2)
            sim.set_incident_angle(0., 0.)
            with torch.no_grad():
                for j in range(6):
                    if j == 2 and not d6:
                        u = torch.arange(64, device=opts.device)/64
                        eps = torch.where((u[:, None] < .4) & (u[None, :] < .7),
                                          2.3+.01j, -8.+1.3j).to(torch.complex128)
                        sim.add_layer(.08, eps=eps)
                    else:
                        r = .27+.03*j
                        sim.add_layer_circle_shell_nvm(.08, r, r+.036, 1., -8.+1.3j,
                            2.46+1e-5j, centers=((.75, math.sqrt(3)/4),), nx=64, ny=64)
                    if reduced_storage:
                        sim.release_completed_layer()
                retained.append(tensor_storage_bytes(vars(sim)))
                if reduced_storage:
                    assert all(value is None for value in sim.P + sim.Q + sim.E_eigvec + sim.H_eigvec)
                    assert not sim._physical_material_by_layer and not sim._nvm_eps_tensor_by_layer
                sim.solve_global_smatrix()
                matrices.append([s.detach().cpu().numpy().copy() for s in sim.S])
            if reduced_storage:
                try:
                    sim.release_completed_layer()  # grad mode is enabled here
                except RuntimeError:
                    check(f'{d6=}: unsafe storage release rejected', 0.)
                else:
                    raise AssertionError('Unsafe storage release was accepted.')
            del sim
        check(f'{d6=}: all four complex S matrices agree',
              max(float(np.max(np.abs(a-b))) for a, b in zip(*matrices)))
        assert retained[1] < retained[0]
        storage.append(dict(test='D6' if d6 else 'full-raster-shell',
                            original_bytes=retained[0], reduced_bytes=retained[1],
                            reduced_fraction=retained[1]/retained[0]))

    for mode, basis, nz, m, grid, wls, residual in [
            ('evaporation', 'auto', 12, 3, 96, [400., 600., 800.], 0.),
            ('sputter', 'auto', 12, 2, 96, [400., 600., 800.], 0.),
            ('evaporation', 'full', 12, 2, 96, [600.], 1.),
            ('evaporation', 'auto', 150, 3, 128, [600.], 0.),
            ('sputter', 'auto', 150, 2, 96, [600.], 0.)]:
        args = parser(mode).parse_args([])
        args.device, args.basis, args.slices = opts.device, basis, nz
        args.order, args.grid, args.residual_um = m, grid, residual
        functions, _ = load_materials(args, wls)
        for wl in wls:
            indices = {k: f(wl) for k, f in functions.items()}
            blocks, powers, sizes = [], [], []
            for policy in ['full', 'reduced']:
                args.layer_storage = policy
                s, diag = surface_rcwa(args, wl, indices)
                blocks.append(s)
                tail = [(indices['PET'], 54800.-1000.*residual, True),
                        (1.5+0j, 20000., True), (silica_n(wl), 1e6, True)]
                powers.append(finite_sample(s, indices['PET'], tail, wl, 16))
                sizes.append(diag['retained_before_solve_bytes'])
            label = f'{mode} {basis} Nz={nz} M={m} {wl:g}nm residual={residual:g}'
            check(label+' Jones amplitudes', max(float(np.max(abs(a-b))) for a, b in zip(*blocks)))
            check(label+' finite-stack R/T/A and cross polarization',
                  max(abs(powers[0][key]-powers[1][key]) for key in powers[0]), 1e-8)
            assert sizes[1] < sizes[0]
            storage.append(dict(test=label, original_bytes=sizes[0], reduced_bytes=sizes[1],
                                reduced_fraction=sizes[1]/sizes[0]))
    opts.output_dir.mkdir(parents=True, exist_ok=True)
    report = dict(device=opts.device, torch=str(torch.__version__), checks=checks,
                  retained_tensor_storage=storage,
                  limitation='Retained storage excludes temporary workspaces and allocator cache; CPU tests do not measure CUDA peak or establish a maximum feasible order.')
    (opts.output_dir/'memory_checks.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'All {len(checks)} checks passed.', flush=True)


if __name__ == '__main__':
    main()
