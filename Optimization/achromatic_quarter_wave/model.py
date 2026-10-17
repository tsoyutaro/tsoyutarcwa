"""Normal-incidence Jones matrices and power from the existing rcwa_ext.

The existing circle-only polarization reduction is intentionally not enabled.
Ellipse: standard raster RCWA. Rectangle: ASR-FR, or raster for cross-checking.
No effective-medium approximation is used to produce the reported spectra.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
import torch
from rcwa_ext import (ASROptions, AutoRCWA, Homogeneous, Lattice, LayerSpec, Material,
                      OutputSpec, Raster, Rectangle)

from .geometry import slices


@dataclass
class Config:
    period_x_nm: float = 200.0
    period_y_nm: float = 200.0
    height_nm: float = 500.0
    n_input: float = 1.0
    n_substrate: float = 1.5
    half_min_nm: float = 5.0
    half_max_nm: float = 95.0
    layers: int = 15
    nodes: int = 3
    order: int = 3
    grid: int = 128
    device: str = "cpu"
    rectangle_method: str = "asr-fr"
    transmission_floor: float = 0.8

    def validate(self):
        positive = (self.period_x_nm, self.period_y_nm, self.height_nm,
                    self.n_input, self.n_substrate)
        if not all(math.isfinite(v) and v > 0 for v in positive):
            raise ValueError("Periods, height and port indices must be positive.")
        if not (0 < self.half_min_nm < self.half_max_nm
                < min(self.period_x_nm, self.period_y_nm) / 2):
            raise ValueError("Require 0 < half_min < half_max < min(periods)/2.")
        if self.layers < 1 or self.nodes < 2 or self.order < 1:
            raise ValueError("Require layers>=1, nodes>=2 and order>=1.")
        if self.grid < max(8, 4 * self.order + 2):
            raise ValueError("Raster grid is too small for retained Fourier orders.")
        if not 0 < self.transmission_floor <= 1:
            raise ValueError("transmission_floor must lie in (0,1].")
        if self.rectangle_method not in ("asr-fr", "raster"):
            raise ValueError("rectangle_method must be asr-fr or raster.")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available in this Python environment.")

    def to_dict(self):
        return asdict(self)


def _flux(e, admittance, count):
    h = admittance @ e
    return .5 * torch.real((e[:count] * h[count:].conj()
                           - e[count:] * h[:count].conj()).sum(dim=0))


@torch.no_grad()
def evaluate(knots, shape, index, wavelength_nm, cfg):
    cfg.validate()
    if shape not in ("ellipse", "rectangle"):
        raise ValueError("shape must be ellipse or rectangle.")
    if not math.isfinite(index) or index <= 0:
        raise ValueError("Structure index must be finite and positive.")
    if not math.isfinite(wavelength_nm) or wavelength_nm <= 0:
        raise ValueError("Wavelength must be finite and positive.")
    # A port exactly at a Rayleigh cutoff has a singular modal admittance.
    for port_n in (cfg.n_input, cfg.n_substrate):
        for mx in range(-cfg.order, cfg.order + 1):
            for my in range(-cfg.order, cfg.order + 1):
                transverse = (wavelength_nm * mx / cfg.period_x_nm)**2 + (
                    wavelength_nm * my / cfg.period_y_nm)**2
                if abs(port_n**2 - transverse) < 1e-10:
                    raise ValueError("Wavelength is exactly at a port diffraction cutoff.")
    scale = cfg.period_x_nm
    sim = AutoRCWA(freq=scale / wavelength_nm, order=[cfg.order, cfg.order],
                   lattice=Lattice.rectangular(1., cfg.period_y_nm / scale),
                   outputs=OutputSpec(smatrix_size="half", fields="none"),
                   asr=ASROptions(grid=(cfg.grid, cfg.grid)),
                   cascade="redheffer", verify_cascade=False,
                   dtype=torch.complex128, device=cfg.device)
    sim.add_input_layer(eps=cfg.n_input**2)
    sim.add_output_layer(eps=cfg.n_substrate**2)
    sim.set_incident_angle(0., 0.)
    axis = (torch.arange(cfg.grid, dtype=torch.float64, device=cfg.device)
            + .5) / cfg.grid - .5
    x, y = torch.meshgrid(axis * cfg.period_x_nm,
                          axis * cfg.period_y_nm, indexing="ij")
    for ax, ay in slices(knots, cfg):
        if index == 1.:
            # No physical interface: avoid an unnecessary ASR coordinate map.
            geometry, method = Homogeneous(Material(1.)), "standard"
        elif shape == "rectangle" and cfg.rectangle_method == "asr-fr":
            geometry = Rectangle((2 * ax / scale, 2 * ay / scale))
            method = "asr-fr"
        else:
            inside = ((x / ax)**2 + (y / ay)**2 <= 1 if shape == "ellipse"
                      else (x.abs() <= ax) & (y.abs() <= ay))
            eps = 1. + (index**2 - 1.) * inside.to(torch.float64)
            geometry, method = Raster(eps), "standard"
        sim.add_structured_layer(LayerSpec(
            cfg.height_nm / cfg.layers / scale, geometry,
            background=Material(1.), inclusion=Material(index**2), method=method))
    sim.solve_global_smatrix()
    return scattering_metrics(sim, cfg, wavelength_nm)


def scattering_metrics(sim, cfg, wavelength_nm):
    """Power, Jones phase and Stokes extraction shared by all RCWA methods."""
    n = sim.order_N
    zero = n // 2
    source = torch.zeros((2 * n, 2), dtype=torch.complex128, device=cfg.device)
    source[zero, 0] = source[n + zero, 1] = 1.
    transmitted = sim.S[0] @ source
    reflected = sim.S[1] @ source
    incoming = _flux(source, sim.Vi, n)
    T = (_flux(transmitted, sim.Vo, n) / incoming).cpu().numpy()
    R = (_flux(reflected, sim.Vi, n) / incoming).cpu().numpy()
    jones = transmitted[[zero, n + zero]].cpu().numpy()
    t00 = cfg.n_substrate / cfg.n_input * np.sum(abs(jones)**2, axis=0)
    error = max(abs(T + R - 1))
    if not np.all(np.isfinite(jones)) or error > 2e-6:
        raise RuntimeError(f"Nonfinite fields or lossless power defect: {error:.3g}")
    phase_valid = bool(min(abs(jones[0, 0])**2, abs(jones[1, 1])**2) > 1e-10)
    delta = float(np.angle(jones[0, 0] * jones[1, 1].conjugate(), deg=True))
    e45 = jones @ np.array([1., 1.]) / np.sqrt(2)
    norm45 = float(np.vdot(e45, e45).real)
    stokes3 = float(2 * np.imag(e45[0] * e45[1].conjugate()) / norm45) if norm45 > 1e-14 else 0.
    t45 = cfg.n_substrate / cfg.n_input * norm45
    cross = cfg.n_substrate / cfg.n_input * (abs(jones[1, 0])**2 + abs(jones[0, 1])**2)
    return dict(wavelength_nm=float(wavelength_nm), phase_valid=phase_valid,
                retardance_deg=delta if phase_valid else None,
                retardance_abs_deg=abs(delta) if phase_valid else None,
                Tx00=float(t00[0]), Ty00=float(t00[1]),
                Tx_total=float(T[0]), Ty_total=float(T[1]),
                Rx_total=float(R[0]), Ry_total=float(R[1]),
                T45_00=t45, stokes3_45=stokes3, circularity_45=abs(stokes3),
                cross_power_sum=float(cross), energy_error=float(error),
                txx_real=float(jones[0, 0].real), txx_imag=float(jones[0, 0].imag),
                tyy_real=float(jones[1, 1].real), tyy_imag=float(jones[1, 1].imag),
                txy_real=float(jones[0, 1].real), txy_imag=float(jones[0, 1].imag),
                tyx_real=float(jones[1, 0].real), tyx_imag=float(jones[1, 0].imag))


def metrics(rows, cfg):
    valid = all(r["phase_valid"] for r in rows)
    floor = min(min(r["Tx00"], r["Ty00"]) for r in rows)
    retardances = [r["retardance_abs_deg"] for r in rows] if valid else []
    signed_s3 = [r["stokes3_45"] for r in rows]
    # Preserve one handedness across the band; |S3| alone would hide flips.
    same_hand = max(min(signed_s3), min(-x for x in signed_s3))
    return dict(min_T00=floor, passes_transmission=bool(floor >= cfg.transmission_floor),
                max_retardance_deg=max(retardances) if valid else None,
                min_retardance_deg=min(retardances) if valid else None,
                worst_qwp_error_deg=max(abs(x - 90) for x in retardances) if valid else None,
                worst_same_hand_circularity=same_hand,
                worst_energy_error=max(r["energy_error"] for r in rows))


def objective(rows, cfg, goal):
    m = metrics(rows, cfg)
    if m["max_retardance_deg"] is None:
        return 1e6
    penalty = 10 * max(0., cfg.transmission_floor - m["min_T00"])
    if goal == "max-phase":
        return -m["max_retardance_deg"] / 180 + penalty
    if goal == "maximin-phase":
        return -m["min_retardance_deg"] / 180 + penalty
    return (m["worst_qwp_error_deg"] / 90
            + .25 * (1 - m["worst_same_hand_circularity"]) + penalty)
