"""Fixed MOSMITE geometry for one order, slice count, and raster grid."""

import math

import torch

from rcwa_ext.config import _TWO_PI
from rcwa_ext.nvm import _jinc

from geometry import build_layers, material_codes, nearest_radius


class ShapeCache:
    def __init__(self, args, slices, grid):
        self.layers = build_layers(args, slices)
        self.normal_projectors = {}
        self.disk_symbols = {}
        self.rasters = {}
        self.distances = None
        self.grid = grid
        self.period_nm = args.period_nm

    def raster(self, layer):
        if layer not in self.rasters:
            if self.distances is None:
                self.distances = nearest_radius(self.grid) * self.period_nm
            self.rasters[layer] = material_codes(layer, self.distances)
        return self.rasters[layer]

    def counts(self):
        return len(self.disk_symbols), len(self.normal_projectors), len(self.rasters)

    def _prepare_reciprocal_grid(self, sim):
        mx, my = (int(order) for order in sim.order)
        self.p_index = (sim.order_x[:, None] - sim.order_x[None, :] + 2 * mx)[:, None, :, None]
        self.q_index = (sim.order_y[:, None] - sim.order_y[None, :] + 2 * my)[None, :, None, :]
        lx, ly = (float(length) for length in sim.L)
        p = torch.arange(-2 * mx, 2 * mx + 1, dtype=torch.float64, device=sim._device)
        q = torch.arange(-2 * my, 2 * my + 1, dtype=torch.float64, device=sim._device)
        self.dx = (p * (_TWO_PI / lx))[:, None]
        self.dy = (q[None, :] * (_TWO_PI / ly) - self.dx * sim.cos_zeta) / sim.sin_zeta
        self.norm = torch.sqrt(self.dx**2 + self.dy**2)
        self.area = lx * ly * sim.sin_zeta
        self.lanczos_buffer = min(_TWO_PI / lx, _TWO_PI / ly)

    def circle(self, sim, radius, eps_bg, eps_cyl, centers, *, use_lanczos, lanczos_power):
        # The convolution is Toeplitz: only Fourier-order differences matter.
        # Keep (4*Mx+1)*(4*My+1) entries per disk, not order_N**2 entries.
        key = (float(radius), tuple(centers), use_lanczos, lanczos_power)
        symbol = self.disk_symbols.get(key)
        if symbol is None:
            if not self.disk_symbols:
                self._prepare_reciprocal_grid(sim)
            phase = torch.zeros_like(self.norm, dtype=sim._dtype)
            for cx, cy in centers:
                phase += torch.exp(-1j * (self.dx * cx + self.dy * cy))
            fill = math.pi * key[0]**2 / self.area
            symbol = fill * _jinc(self.norm * key[0]) * phase
            if use_lanczos:
                z = 3.831705970 * self.norm / (self.norm.max() + self.lanczos_buffer)
                symbol *= _jinc(z)**lanczos_power
            self.disk_symbols[key] = symbol
        matrix = symbol[self.p_index, self.q_index].reshape(sim.order_N, sim.order_N)
        matrix.mul_(eps_cyl - eps_bg)
        matrix.diagonal().add_(eps_bg)
        return matrix

    def solver_class(self, original):
        cache = self

        class FixedShapeRCWA(original):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                if self.normal_mode != 'off':
                    raise RuntimeError('Fixed MOSMITE shape reuse requires NVM normal_mode=off.')
                # Reuse the existing NVM projector cache, including its grid
                # and center keys. Its unsmoothed normal is radius independent.
                self._off_normal_cache = cache.normal_projectors

            def _circle_toeplitz(self, radius, eps_bg, eps_cyl, centers, *, use_lanczos, lanczos_power):
                return cache.circle(self, radius, eps_bg, eps_cyl, centers,
                                    use_lanczos=use_lanczos, lanczos_power=lanczos_power)

        return FixedShapeRCWA
