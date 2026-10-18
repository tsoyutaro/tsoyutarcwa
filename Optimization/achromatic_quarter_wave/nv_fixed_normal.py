"""Diagnostic NV extension with a normal field independent of its FFT grid.

The reference grid defines the legacy regularization widths once. Increasing
the actual sampling grid then samples the same continuous projector field.
At actual grid == reference grid this reproduces the legacy projector.
"""
from __future__ import annotations

import numpy as np
import torch

from rcwa_ext import Lattice, NVMOptions, OutputSpec
from rcwa_ext.config import _as_float
from rcwa_ext.nvm import _smoothstep
from Optimization.achromatic_quarter_wave.geometry import slices
from Optimization.achromatic_quarter_wave.model import scattering_metrics
from Optimization.achromatic_quarter_wave.nv_shapes import ShapeNVRCWA


class FixedNormalNVRCWA(ShapeNVRCWA):
    def __init__(self, *args, normal_reference_grid, **kwargs):
        if (isinstance(normal_reference_grid, bool)
                or not isinstance(normal_reference_grid, int) or normal_reference_grid < 1):
            raise ValueError("normal_reference_grid must be a positive integer.")
        self.normal_reference_grid = normal_reference_grid
        super().__init__(*args, **kwargs)

    def shape_projection(self, shape, half_x, half_y, nx, ny):
        minimum = max(32, 4 * max(int(v) for v in self.order) + 4)
        if min(nx, ny) < minimum:
            raise ValueError(f"NV normal grid requires Q >= {minimum}.")
        lx, ly = _as_float(self.L[0]), _as_float(self.L[1])
        # Only this resolution defines the continuous field; nx/ny sample it.
        fixed_resolution = max(lx, ly) / self.normal_reference_grid
        edge_width = 3 * fixed_resolution
        edge_gap = min(lx - 2 * _as_float(half_x), ly - 2 * _as_float(half_y))
        if edge_gap < edge_width:
            raise ValueError("Fixed normal transition reaches the pillar gap; increase --normal-reference-grid.")
        x = torch.arange(nx, dtype=torch.float64, device=self._device) * (lx / nx) - lx / 2
        y = torch.arange(ny, dtype=torch.float64, device=self._device) * (ly / ny) - ly / 2
        dx, dy = torch.meshgrid(x, y, indexing="ij")
        nearest = torch.sqrt(dx**2 + dy**2)
        second = torch.sqrt(torch.minimum((lx - dx.abs())**2 + dy**2,
                                          dx**2 + (ly - dy.abs())**2))
        vx, vy, magnitude = self.normal_direction(shape, dx, dy, half_x, half_y)
        center_radius = torch.maximum(torch.minimum(half_x, half_y) / 2,
                                      half_x.new_tensor(fixed_resolution))
        weight = (_smoothstep(nearest, 0., center_radius)
                  * _smoothstep(second - nearest, 0., edge_width))
        weight = torch.where(magnitude > 1e-14, weight, torch.zeros_like(weight))
        pxx = .5 + (vx**2 - .5) * weight
        pyy = .5 + (vy**2 - .5) * weight
        pxy = vx * vy * weight
        count_y = len(self.order_y)
        px = self.order_x.repeat_interleave(count_y)
        py = self.order_y.repeat(len(self.order_x))
        ix, iy = px[:, None] - px[None, :], py[:, None] - py[None, :]
        def conv(field):
            fft = torch.fft.fft2(field.to(self._dtype)) / (nx * ny)
            result = fft[ix, iy]
            return .5 * (result + result.mH)
        xx, yy, xy = conv(pxx), conv(pyy), conv(pxy)
        return torch.cat((torch.cat((xx, xy), 1), torch.cat((xy.mH, yy), 1)), 0)


@torch.no_grad()
def simulate_fixed_nv(knots, shape, index, wavelength, cfg, normal_reference_grid):
    cfg.validate()
    if shape not in ("ellipse", "rectangle"):
        raise ValueError("Fixed-normal diagnostics support ellipse and rectangle.")
    if not np.isfinite(index) or index <= 0 or not np.isfinite(wavelength) or wavelength <= 0:
        raise ValueError("Structure index and wavelength must be positive and finite.")
    for port_n in (cfg.n_input, cfg.n_substrate):
        for x in range(-cfg.order, cfg.order + 1):
            for y in range(-cfg.order, cfg.order + 1):
                if abs(port_n**2 - (wavelength*x/cfg.period_x_nm)**2
                       - (wavelength*y/cfg.period_y_nm)**2) < 1e-10:
                    raise ValueError("Wavelength is exactly at an external diffraction cutoff.")
    scale = cfg.period_x_nm
    sim = FixedNormalNVRCWA(
        freq=scale / wavelength, order=[cfg.order, cfg.order],
        lattice=Lattice.rectangular(1., cfg.period_y_nm / scale),
        outputs=OutputSpec(smatrix_size="half", fields="none"),
        nvm=NVMOptions(grid=(cfg.grid, cfg.grid), use_lanczos=False),
        normal_reference_grid=normal_reference_grid,
        dtype=torch.complex128, device=cfg.device,
        cascade="redheffer", verify_cascade=False)
    sim.add_input_layer(eps=cfg.n_input**2)
    sim.add_output_layer(eps=cfg.n_substrate**2)
    sim.set_incident_angle(0., 0.)
    for a, b in slices(knots, cfg):
        sim.add_layer_shape_nv(cfg.height_nm / cfg.layers / scale, shape,
                              a / scale, b / scale, 1., index**2,
                              nx=cfg.grid, ny=cfg.grid)
        sim.layer_records[-1].options.update(
            normal_reference_grid=normal_reference_grid,
            normal_edge_width_normalized=3 * max(1., cfg.period_y_nm / scale) / normal_reference_grid)
    sim.solve_global_smatrix()
    return sim, scattering_metrics(sim, cfg, wavelength)
