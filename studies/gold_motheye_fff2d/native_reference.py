"""Independent adapter to the existing rcwa_ext eigenmode NV implementation.

No projection/tensor/propagation code from this study is used by the reference.
All units are nm. A rectangular Fourier set, no Lanczos filter and no symmetry
reduction are required for a like-for-like comparison with AutoRCWA.
"""
from __future__ import annotations
import numpy as np
import torch


def compare_native(geometry, wavelength_nm, epsilon, *, order=1, slices=4, grid=48,
                   device='cpu', polarization='x', epsilon_out=None, method='slice-nvm'):
    from rcwa_ext import AutoRCWA, NVMOptions, Lattice, OutputSpec, GroupTheoryOptions
    from .solver import solve, flux_orders
    from .model import Operators, harmonics
    if geometry.gil_nm or geometry.harmonic_shape != 'rectangular':
        raise ValueError('Native reference requires sharp GIL=0 and a rectangular Fourier set')
    lattice = Lattice.triangular(geometry.period_nm) if geometry.lattice=='triangular' else Lattice.square(geometry.period_nm)
    sim = AutoRCWA(freq=1/wavelength_nm,order=[order,order],lattice=lattice,
                   outputs=OutputSpec(smatrix_size='full',fields='none'),
                   nvm=NVMOptions(grid=(grid,grid),use_lanczos=False),
                   group_theory=GroupTheoryOptions(enabled=False),
                   dtype=torch.complex128,device=device)
    eps_out = epsilon if epsilon_out is None else epsilon_out
    sim.add_input_layer(eps=1.)
    sim.add_output_layer(eps=eps_out)
    sim.set_incident_angle(0.,0.)
    for i in range(slices):
        sim.add_layer_circle_nvm(thickness=geometry.height_nm/slices,
            radius=geometry.radius((i+.5)*geometry.height_nm/slices),eps_bg=1.,eps_cyl=epsilon,
            centers=((0.,0.),),nx=grid,ny=grid,use_lanczos=False)
    sim.solve_global_smatrix()
    actual = solve(geometry,wavelength_nm,epsilon,order=order,slices=slices,grid=grid,
                   device=device,polarization=polarization,method=method,epsilon_out=eps_out)
    n = sim.order_N
    source = torch.zeros((2*n,1),dtype=torch.complex128,device=device)
    source[n//2+(n if polarization=='y' else 0),0] = 1.
    native_r,native_t = sim.S[1]@source,sim.S[0]@source
    our_r,our_t = actual.S.r@source,actual.S.t@source
    # rcwa_ext V uses Cartesian H for the same electric port basis.
    incoming = float(flux_orders(source,sim.Vi@source).sum())
    rp = flux_orders(native_r,sim.Vi@native_r)/incoming
    tp = flux_orders(native_t,sim.Vo@native_t)/incoming
    # Quantify backend differences in analytical disk/Bessel evaluation;
    # do not modify either implementation just to force agreement.
    z = geometry.height_nm*(1-.5/slices)
    radius = torch.tensor(geometry.radius(z),dtype=torch.float64,device=device)
    e_native = sim._circle_toeplitz(radius,1.,epsilon,((0.,0.),),use_lanczos=False,lanczos_power=2)
    eta_native = sim._circle_toeplitz(radius,1.,1/epsilon,((0.,0.),),use_lanczos=False,lanczos_power=2)
    ops = Operators(geometry,harmonics(order,geometry.lattice,'rectangular'),grid)
    e,eta = ops.material(z,epsilon)
    pn = sim._projection_matrix(radius,((0.,0.),),grid,grid).detach().cpu().numpy()
    ps = ops.projections(z,tilted=False)
    ps = np.block([[ps[a][b] for b in range(2)] for a in range(2)])
    return dict(R=actual.R,P_sub=actual.P_sub,native_R=float(rp.sum()),native_P_sub=float(tp.sum()),
                epsilon_matrix_error=float(np.max(abs(e-e_native.detach().cpu().numpy()))),
                inverse_epsilon_matrix_error=float(np.max(abs(eta-eta_native.detach().cpu().numpy()))),
                projection_matrix_error=float(np.max(abs(pn-ps))),
                max_complex_amplitude_error=max(float((our_r-native_r).abs().max()),float((our_t-native_t).abs().max())),
                max_order_power_error=max(float(np.max(abs(actual.R_orders-rp.detach().cpu().numpy().ravel()))),
                                          float(np.max(abs(actual.T_orders-tp.detach().cpu().numpy().ravel())))),
                lattice=geometry.lattice,method=method,polarization=polarization,
                reference='rcwa_ext.AutoRCWA.add_layer_circle_nvm, no Lanczos, no symmetry reduction')
