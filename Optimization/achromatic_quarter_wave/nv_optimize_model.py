"""Differentiable OFF-NVM response of the fifteen-layer anisotropic taper."""

from __future__ import annotations

import math

import torch

from rcwa_ext import Lattice, NVMOptions, OutputSpec
from rcwa_ext.config import _as_float
from rcwa_ext.nvm import _jinc

from .model import _flux, scattering_metrics
from .nv_off_convergence import UnsmoothedShapeNVRCWA


CUTOFF_SHIFT = 1e-8


class InvalidResponse(RuntimeError):
    """Numerically invalid optical response or gradient for one trial."""


class DifferentiableShapeNVM(UnsmoothedShapeNVRCWA):
    @staticmethod
    def normal_direction(shape, dx, dy, half_x, half_y):
        vx, vy = dx / half_x**2, dy / half_y**2
        if shape == "rectangle":
            vx = vx * (1 - (dy / half_y)**2)
            vy = vy * (1 - (dx / half_x)**2)
        # vector_norm defines a zero derivative at the origin. The inherited
        # OFF projector uses I/2 there and at undefined rectangle corners.
        length = torch.linalg.vector_norm(torch.stack((vx, vy)), dim=0)
        safe = length.clamp_min(1e-14)
        return vx / safe, vy / safe, length

    def shape_convolution(self, shape, half_x, half_y, background, inclusion):
        if shape != "ellipse":
            return super().shape_convolution(shape, half_x, half_y, background, inclusion)
        # The same analytic ellipse coefficients as ShapeNVRCWA. Use a norm
        # with a finite derivative for the exactly zero Fourier difference.
        lx, ly = _as_float(self.L[0]), _as_float(self.L[1])
        gx = (2 * math.pi / lx * self.order_x.to(torch.float64)).repeat_interleave(len(self.order_y))
        gy = (2 * math.pi / ly * self.order_y.to(torch.float64)).repeat(len(self.order_x))
        dgx, dgy = gx[:, None] - gx[None, :], gy[:, None] - gy[None, :]
        argument = torch.linalg.vector_norm(torch.stack((half_x * dgx, half_y * dgy)), dim=0)
        transform = math.pi * half_x * half_y / (lx * ly) * _jinc(argument)
        translation = torch.exp(-0.5j * (dgx * lx + dgy * ly))
        return background * self._eye(self.order_N) + (inclusion - background) * transform * translation


def half_widths(logits, cfg):
    """Return (2, 15) layer-center dimensions in nm, ordered from tip to base."""
    gaps = torch.softmax(torch.cat((logits, torch.zeros_like(logits[:, :1])), dim=1), dim=1)
    # Keep the monotonic gaps positive even if Adam makes logits very uneven.
    floor = 1e-6
    gaps = floor + (1 - gaps.shape[1] * floor) * gaps
    return cfg.half_min_nm + (cfg.half_max_nm - cfg.half_min_nm) * gaps.cumsum(dim=1)[:, :-1]


def evaluation_wavelength(wavelength, cfg):
    """Evaluate the long-wavelength limit at an exact external Rayleigh cutoff."""
    for n in (cfg.n_input, cfg.n_substrate):
        for m in range(cfg.order + 1):
            for k in range(cfg.order + 1):
                transverse = (wavelength * m / cfg.period_x_nm)**2 + (wavelength * k / cfg.period_y_nm)**2
                if math.isclose(n**2, transverse, rel_tol=1e-12):
                    return wavelength * (1 + CUTOFF_SHIFT)
    return wavelength


def solve_profile(widths, shape, wavelength, cfg):
    """Use a matched, homogeneous exit port with the same index as the pillars."""
    evaluated = evaluation_wavelength(wavelength, cfg)
    scale = cfg.period_x_nm
    sim = DifferentiableShapeNVM(
        freq=scale / evaluated, order=[cfg.order, cfg.order],
        lattice=Lattice.rectangular(1., cfg.period_y_nm / scale),
        outputs=OutputSpec(smatrix_size="half", fields="none"),
        nvm=NVMOptions(grid=(cfg.grid, cfg.grid), use_lanczos=False, normal_mode="off"),
        dtype=torch.complex128, device=cfg.device, cascade="redheffer", verify_cascade=False,
    )
    # This only stabilizes the eigensolver's backward rule near degeneracy.
    sim.stable_eig_grad = torch.is_grad_enabled()
    sim.add_input_layer(eps=cfg.n_input**2)
    sim.add_output_layer(eps=cfg.n_substrate**2)
    sim.set_incident_angle(0., 0.)
    for a, b in widths.T:
        sim.add_layer_shape_nv(cfg.height_nm / (cfg.layers * scale), shape,
                               a / scale, b / scale, 1., cfg.n_substrate**2,
                               nx=cfg.grid, ny=cfg.grid)
    sim.solve_global_smatrix()
    return sim, evaluated


def phase_error(t_x, t_y):
    return torch.angle(t_x * t_y.conj() * -1j)


def wavelength_loss(widths, shape, wavelength, cfg, goal):
    sim, _ = solve_profile(widths, shape, wavelength, cfg)
    count = sim.order_N
    zero = count // 2
    orders = [zero, count + zero]
    if goal == "reflectance":
        source = sim._eye(2 * count)[:, orders]
        reflected = sim.S[1][:, orders]
        return (_flux(reflected, sim.Vi, count) / _flux(source, sim.Vi, count)).mean()
    t_x, t_y = sim.S[0][zero, zero], sim.S[0][count + zero, count + zero]
    if min(t_x.detach().abs().square(), t_y.detach().abs().square()) <= 1e-10:
        raise InvalidResponse("Co-polar transmission is too small to define retardance")
    return phase_error(t_x, t_y).square()


@torch.no_grad()
def evaluate(widths, shape, wavelengths, cfg):
    rows = []
    for wavelength in wavelengths:
        sim, evaluated = solve_profile(widths, shape, wavelength, cfg)
        row = scattering_metrics(sim, cfg, wavelength)
        delta = row["retardance_deg"]
        rows.append(dict(**row, evaluated_wavelength_nm=evaluated,
                         target_error_deg=math.remainder(delta - 90, 360) if delta is not None else None))
    return rows
