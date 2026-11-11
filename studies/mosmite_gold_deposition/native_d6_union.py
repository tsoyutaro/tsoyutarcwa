"""D6-E1 for periodic unions of touching Au shells.

Merged layers use the scalar Fourier formulation of the existing raster
path. Isolated circular layers retain the analytic NVM formulation.
"""
import numpy as np
import torch

from rcwa_ext.acceleration import d6

from geometry import build_layers, cap_count, material_codes
from memory import connect_full
from native_d6 import NativeD6Geometry, NativePort, surface_native_d6
from shape_cache import ShapeCache


def make_geometry(args, slices, grid):
    if any(layer.kind.startswith('coalesced') for layer in build_layers(args, slices)):
        return NativeUnionGeometry(args, slices, grid)
    return NativeD6Geometry(args, slices, grid)


class NativeUnionGeometry(NativeD6Geometry):
    def __init__(self, args, slices, grid):
        ShapeCache.__init__(self, args, slices, grid)
        if args.basis != 'auto' or grid % 2:
            raise ValueError('D6 union layers require --basis auto and an even --grid.')
        self.raster_symbols = {}

    def raster(self, layer):
        if layer not in self.rasters:
            if self.distances is None:
                # Integer lattice distances give exactly identical samples
                # on each D6 orbit, including circle-boundary ties.
                axis = 2 * np.arange(self.grid, dtype=np.int64) - self.grid
                u, v = np.meshgrid(axis, axis, indexing='ij')
                nearest = np.full(u.shape, np.iinfo(np.int64).max, dtype=np.int64)
                for a in (-1, 0, 1):
                    for b in (-1, 0, 1):
                        du, dv = u - 2*self.grid*a, v - 2*self.grid*b
                        nearest = np.minimum(nearest, du*du + du*dv + dv*dv)
                self.distances = np.sqrt(nearest) * self.period_nm / (2*self.grid)
            self.rasters[layer] = material_codes(layer, self.distances)
        return self.rasters[layer]

    def materials(self, layer, eps_resin, eps_gold):
        if not layer.kind.startswith('coalesced'):
            return super().materials(layer, eps_resin, eps_gold)
        if layer not in self.raster_symbols:
            code = self.raster(layer)
            bound = 2 * int(self.template.order[0])
            orders = torch.arange(-bound, bound+1, device=self.template._device) % self.grid
            symbols = []
            for material in (1, 2):
                mask = torch.as_tensor(code == material, dtype=self.template._dtype,
                                       device=self.template._device)
                coefficients = torch.fft.fft2(mask) / self.grid**2
                symbols.append(coefficients[orders[:, None], orders[None, :]])
            self.raster_symbols[layer] = symbols
        core, gold = self.raster_symbols[layer]
        eps = ((eps_resin-1) * core[self.star_p_index, self.star_q_index]
               + (eps_gold-1) * gold[self.star_p_index, self.star_q_index])
        eps.diagonal().add_(1)
        return eps, None


def scalar_pq(port, eps):
    """Scalar Fourier Maxwell operators, assembled directly in E1."""
    plan = port.plan
    inverse = d6.D6BlockInverse(port, eps, plan)
    n, dimension = plan.scalar_dimension, plan.reduced_dimension
    kx, ky = port.kx[:, None], port.ky[:, None]
    hx, hy = plan.magnetic_pair[:n], plan.magnetic_pair[n:]
    longitudinal = inverse.apply(ky*hx - kx*hy)
    p_image = torch.cat((kx*longitudinal + hy, ky*longitudinal - hx), dim=0)
    ex, ey = plan.electric_pair[:n], plan.electric_pair[n:]
    q_image = torch.cat((-kx*ky*ex + kx*kx*ey - eps@ey,
                         eps@ex - ky*ky*ex + ky*kx*ey), dim=0)
    p = .5 * (plan.electric.mH@p_image[:, :dimension]
              + plan.electric_partner.mH@p_image[:, dimension:])
    q = .5 * (plan.magnetic.mH@q_image[:, :dimension]
              + plan.magnetic_partner.mH@q_image[:, dimension:])
    return p, q


class NativeUnionPort(NativePort):
    def layer_scattering(self, eps, rule, geometry, thickness):
        if rule is not None:
            return super().layer_scattering(eps, rule, geometry, thickness)
        p, q = scalar_pq(self, eps)
        squared, electric = self._eig(p@q)
        kz = self._positive_kz(squared)
        magnetic = self._magnetic_eigenvectors(p, q, electric, kz)
        order = torch.argsort(kz.real - 1e-6*kz.imag, descending=True)
        depth = torch.as_tensor(thickness, dtype=self._dtype, device=self._device)
        return self._reduced_layer_smatrix(electric[:, order], magnetic[:, order],
                                           kz[order], depth, self.reference_v)


@torch.no_grad()
def surface(args, wl, indices, order, slices, shapes):
    if not isinstance(shapes, NativeUnionGeometry):
        return surface_native_d6(args, wl, indices, order, slices, shapes)
    initial = shapes.counts()
    if not hasattr(shapes, 'plan'):
        shapes.prepare(args, order)
    port = NativeUnionPort(shapes, args.period_nm/wl)
    eps_resin = indices['resin']**2
    scattering = port._reduced_interface_s(port.reference_v, port.reference_v, input_side=True)
    for layer in shapes.layers:
        eps, rule = shapes.materials(layer, eps_resin, indices['Au_'+layer.gold_role]**2)
        layer_s = port.layer_scattering(eps, rule, shapes, layer.thickness_nm/args.period_nm)
        scattering = connect_full(port, scattering, layer_s)
        del eps, rule, layer_s
    if args.residual_um > 0:
        identity = port._eye(shapes.plan.scalar_dimension)
        residual = port.layer_scattering(identity*eps_resin, identity/eps_resin, shapes,
                                         args.residual_um*1000/args.period_nm)
        scattering = connect_full(port, scattering, residual)
    exit_v = port.admittance(indices['PET']**2)
    scattering = connect_full(port, scattering,
                              port._reduced_interface_s(port.reference_v, exit_v, input_side=False))
    amplitudes = torch.stack([torch.vdot(shapes.source, block@shapes.source)
                              for block in scattering]).cpu().numpy()
    jones = [np.eye(2, dtype=np.complex128)*value for value in amplitudes]
    counts = dict(zip(('geometry_disk_builds', 'geometry_normal_builds', 'geometry_raster_builds'),
                      (after-before for before, after in zip(initial, shapes.counts()))))
    return jones, {
        'basis': 'D6-native-E1-union', 'harmonics': shapes.plan.scalar_dimension,
        'eigen_dimension': shapes.plan.reduced_dimension,
        'rectangular_material_matrices_assembled': False,
        'core_layers': slices, 'cap_layers': cap_count(args, slices),
        'total_pattern_layers': len(shapes.layers),
        'coalesced_raster_layers': sum(layer.kind.startswith('coalesced') for layer in shapes.layers),
        'layer_storage': 'reduced', 'storage_policy': 'native-d6-streamed-full-S-v1',
        'retained_before_solve_bytes': None, 'post_layer_retained_tensor_peak_bytes': None,
        'unpolarized': 'D6 x=y equality', **counts,
    }
