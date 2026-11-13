"""Reuse the project's streaming, polarization-separated Cartesian Li RCWA."""
from __future__ import annotations

import numpy as np
import torch

from rcwa_ext.config import GroupTheoryOptions, Lattice, OutputSpec
from studies.gold_grating_1d.solver import Grating1D


class SheetRelief1D(Grating1D):
    def _kvectors(self):
        result = super()._kvectors()
        # The auxiliary reference is arbitrary. eps=2 avoids the artificial
        # vacuum-reference singularity at lambda=period/|m| (e.g. 620 nm).
        qref = self._positive_kz(2.-self.Kx_norm_dn**2)
        self.Vf = self._cartesian_e_to_h(qref, mu=1.)
        self.Sin[:] = self._interface_s(self.Vi, input_side=True)
        self.Sout[:] = self._interface_s(self.Vo, input_side=False)
        return result

    def indicator(self, fill):
        identity = self._eye(self.order_N)
        if fill <= 0:
            return torch.zeros_like(identity)
        if fill >= 1:
            return identity
        delta = (self.order_x[:, None]-self.order_x[None, :]).to(torch.float64)
        return fill*torch.sinc(delta*fill)*torch.exp(-1j*torch.pi*delta)

    def add_profile_layer(self, layer, period, eps_pmma, eps_gold, eps_void):
        core = self.indicator(layer['core_width_nm']/period)
        outer = self.indicator(layer['outer_width_nm']/period)
        identity = self._eye(self.order_N)
        direct = eps_void*identity+(eps_gold-eps_void)*outer+(eps_pmma-eps_gold)*core
        reciprocal = identity/eps_void+(1/eps_gold-1/eps_void)*outer+(1/eps_pmma-1/eps_gold)*core
        thickness = torch.as_tensor((layer['z_bottom_nm']-layer['z_top_nm'])/period,
                                    dtype=torch.float64, device=self._device)
        self._add_ridge_separated(thickness, direct, reciprocal, audit=False)


def local_surface(g, layer_list, wavelength, n_pmma, n_gold, void, order, device='cpu'):
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; select --device cpu.')
    period = g['period_nm']
    sim = SheetRelief1D(freq=period/wavelength, order=[order, 0], lattice=Lattice.square(1.),
        cascade='redheffer', outputs=OutputSpec(smatrix_size='half', fields='none'),
        group_theory=GroupTheoryOptions(enabled=False), dtype=torch.complex128, device=device)
    with torch.no_grad():
        sim.add_input_layer(eps=n_pmma**2)
        sim.add_output_layer(eps=1.333**2)  # water surface is at the relief/Au tip plane
        sim.set_incident_angle(0., 0.)
        for layer in layer_list:
            sim.add_profile_layer(layer, period, n_pmma**2, n_gold**2, void**2)
        blocks = sim.solve_polarized_smatrix()
        # Only forward Tf and Rf are needed; avoid Cartesian public S materialization.
        tf, rf = (b.detach().cpu().numpy() for b in blocks[:2])
        orders = sim.order_x.detach().cpu().numpy().astype(int)
    if not np.all(np.isfinite(tf)) or not np.all(np.isfinite(rf)):
        raise RuntimeError('Nonfinite local scattering matrix.')
    return orders, rf, tf
