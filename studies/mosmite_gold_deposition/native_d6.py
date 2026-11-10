"""Direct D6-E1 NVM for centered circular layers and full two-sided S."""
import math

import numpy as np
import torch

from rcwa_ext import AutoRCWA, Lattice, NVMOptions
from rcwa_ext.acceleration import d6
from rcwa_ext.reduced import _ReducedScatteringMixin
from rcwa_ext.scattering import _StableLinearAlgebraMixin

from geometry import cap_count
from memory import connect_full
from shape_cache import ShapeCache


class NativeD6Geometry(ShapeCache):
    def __init__(self, args, slices, grid):
        super().__init__(args, slices, grid)
        if args.basis != 'auto' or any(layer.kind.startswith('coalesced') for layer in self.layers):
            raise ValueError('--native-d6 requires --basis auto and nonoverlapping centered circles.')

    def prepare(self, args, order):
        # The template provides geometry and the symmetry group without
        # initializing any wavelength-dependent rectangular port matrices.
        template = AutoRCWA(freq=1., order=[order, order], lattice=Lattice.triangular(1.),
                            nvm=NVMOptions(normal_mode='off'),
                            dtype=torch.complex128, device=args.device)
        self.template = template
        self.plan = d6.D6Plan(template, keep_full_basis=False)
        m, n = self.plan.m, self.plan.n
        self.star_p_index = m[:, None] - m[None, :] + 2 * order
        self.star_q_index = n[:, None] - n[None, :] + 2 * order
        self.centers = (d6.center_of(template),)
        self.projection = d6.native_projection(template, .25, self.grid, self.plan)
        self.normal_projectors['native'] = self.projection
        self.normal_image = d6.projected_electric(self.projection, self.plan)
        zero = int(torch.nonzero((m == 0) & (n == 0), as_tuple=True)[0].item())
        self.source = self.plan.electric[zero].conj()

    def disk(self, radius):
        return self.disk_symbol(self.template, radius, self.centers)[self.star_p_index, self.star_q_index]

    def materials(self, layer, eps_resin, eps_gold):
        if layer.kind == 'valley':
            bg, inclusion, radius = eps_gold, eps_resin, layer.core_radius_nm
        elif layer.kind == 'bare':
            bg, inclusion, radius = 1., eps_resin, layer.core_radius_nm
        else:
            bg, inclusion, radius = 1., eps_gold, layer.outer_radius_nm
        disk = self.disk(radius / self.period_nm)
        rule = disk * (1 / inclusion - 1 / bg)
        eps = disk.mul_(inclusion - bg)
        eps.diagonal().add_(bg)
        rule.diagonal().add_(1 / bg)
        if layer.kind == 'shell':
            inner = self.disk(layer.core_radius_nm / self.period_nm)
            eps.add_(inner, alpha=eps_resin - inclusion)
            rule.add_(inner, alpha=1 / eps_resin - 1 / inclusion)
        return eps, rule


class NativePort(_ReducedScatteringMixin, _StableLinearAlgebraMixin):
    def __init__(self, geometry, frequency):
        template = geometry.template
        self._dtype, self._device = template._dtype, template._device
        self.stable_eig_grad = template.stable_eig_grad
        freq = torch.as_tensor(frequency, dtype=self._dtype, device=self._device)
        self.omega = 2 * math.pi * freq
        self.plan = geometry.plan
        m, n = torch.meshgrid(template.order_x, template.order_y, indexing='ij')
        self.Kx_norm_dn = m.flatten() * (1 / freq)
        self.Ky_norm_dn = (n.flatten() * (1 / freq)
                           - template.cos_zeta * self.Kx_norm_dn) / template.sin_zeta
        self.kx = self.Kx_norm_dn[self.plan.indices]
        self.ky = self.Ky_norm_dn[self.plan.indices]
        self.reference_v = self.admittance(1.)

    def admittance(self, eps):
        # Apply the diagonal Cartesian plane-wave admittance to the E1
        # electric basis without constructing a full Cartesian matrix.
        kx, ky = self.kx, self.ky
        kz = self._positive_kz(eps - kx**2 - ky**2)
        safe = torch.where(kz.abs() < 1e-12, kz + (1e-10 + 1e-10j), kz)
        ex, ey = self.plan.electric.chunk(2, dim=0)
        xy = (kx * ky / safe)[:, None]
        hx = -xy * ex - (safe + ky**2 / safe)[:, None] * ey
        hy = (safe + kx**2 / safe)[:, None] * ex + xy * ey
        return self.plan.magnetic.mH @ torch.cat((hx, hy), dim=0)

    def layer_scattering(self, eps, rule, geometry, thickness):
        p, q, _ = d6.reduced_pq(self, eps, rule, geometry.projection, self.plan,
                                normal_image=geometry.normal_image)
        kz_squared, electric = self._eig(p @ q)
        kz = self._positive_kz(kz_squared)
        magnetic = self._magnetic_eigenvectors(p, q, electric, kz)
        order = torch.argsort(kz.real - 1e-6 * kz.imag, descending=True)
        depth = torch.as_tensor(thickness, dtype=self._dtype, device=self._device)
        return self._reduced_layer_smatrix(electric[:, order], magnetic[:, order],
                                           kz[order], depth, self.reference_v)


@torch.no_grad()
def surface_native_d6(args, wl, indices, order, slices, shapes):
    initial_counts = shapes.counts()
    if not hasattr(shapes, 'plan'):
        shapes.prepare(args, order)
    port = NativePort(shapes, args.period_nm / wl)
    eps_resin = indices['resin']**2
    scattering = port._reduced_interface_s(port.reference_v, port.reference_v, input_side=True)
    for layer in shapes.layers:
        eps, rule = shapes.materials(layer, eps_resin, indices['Au_' + layer.gold_role]**2)
        layer_s = port.layer_scattering(eps, rule, shapes, layer.thickness_nm / args.period_nm)
        scattering = connect_full(port, scattering, layer_s)
        del eps, rule, layer_s
    if args.residual_um > 0:
        identity = port._eye(shapes.plan.scalar_dimension)
        residual = port.layer_scattering(identity * eps_resin, identity / eps_resin, shapes,
                                         args.residual_um * 1000 / args.period_nm)
        scattering = connect_full(port, scattering, residual)
    exit_v = port.admittance(indices['PET']**2)
    scattering = connect_full(port, scattering,
                              port._reduced_interface_s(port.reference_v, exit_v, input_side=False))
    # Return only the zero-order Jones blocks needed by the finite PET/glass
    # stack. D6 makes the x/y blocks equal and removes cross polarization.
    source = shapes.source
    amplitudes = torch.stack([torch.vdot(source, block @ source) for block in scattering]).cpu().numpy()
    jones = [np.eye(2, dtype=np.complex128) * amplitude for amplitude in amplitudes]
    counts = dict(zip(('geometry_disk_builds', 'geometry_normal_builds', 'geometry_raster_builds'),
                      (after - before for before, after in zip(initial_counts, shapes.counts()))))
    return jones, {
        'basis': 'D6-native-E1', 'harmonics': shapes.plan.scalar_dimension,
        'eigen_dimension': shapes.plan.reduced_dimension, 'rectangular_material_matrices_assembled': False,
        'core_layers': slices, 'cap_layers': cap_count(args, slices),
        'total_pattern_layers': len(shapes.layers), 'coalesced_raster_layers': 0,
        'layer_storage': 'reduced', 'storage_policy': 'native-d6-streamed-full-S-v1',
        'retained_before_solve_bytes': None, 'post_layer_retained_tensor_peak_bytes': None,
        'unpolarized': 'D6 x=y equality', **counts,
    }
