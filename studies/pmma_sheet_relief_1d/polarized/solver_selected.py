"""One-polarization mirror-even RCWA; the other polarization is never eigensolved."""
import math
import numpy as np
import torch
from fast_solver import EvenCascade, get_plan


class SelectedCascade(EvenCascade):
    def __init__(self, plan, wavelength, n_pmma, polarization):
        self.polarization = polarization
        self._device, self._dtype = plan.device, torch.complex128
        self.order_N = plan.order + 1
        self.omega = 2 * math.pi * plan.period / wavelength
        self.k = torch.arange(self.order_N, dtype=torch.float64, device=plan.device) * wavelength / plan.period
        self.identity = self._eye(self.order_N)
        if polarization == 'p':
            self.odd_identity = self._eye(plan.order)
        j = 0 if polarization == 's' else 1
        yf = self.admittance(2.)[j:j+1]
        yp = self.admittance(n_pmma * n_pmma)[j:j+1]
        yw = self.admittance(1.333**2)[j:j+1]
        self.reference_y = yf
        self.Vf = None  # The borrowed layer routine obtains yf through the override below.
        diag = torch.diag_embed
        self.stack = [diag(2*yp/(yp+yf)), diag((yp-yf)/(yp+yf)),
                      diag((yf-yp)/(yp+yf)), diag(2*yf/(yp+yf))]
        self.output = [diag(2*yf/(yf+yw)), diag((yf-yw)/(yf+yw)),
                       diag((yw-yf)/(yf+yw)), diag(2*yw/(yf+yw))]

    def _polarized_admittance(self, port):
        return self.reference_y

    def add_selected(self, direct_even, reciprocal_even, direct_odd, dz):
        if self.polarization == 's':
            operator = direct_even - torch.diag(self.k*self.k)
            eigenvalues, electric = self._eig(operator[None, :, :])
            kz = self._positive_kz(eigenvalues)
            magnetic = electric*kz[:, None, :]
        else:
            # Kx maps even fields to odd fields: retain the odd inverse block.
            normal = self._solve(reciprocal_even, self.identity)
            inverse_odd = self._solve(direct_odd, self.odd_identity)
            p_tm = self.identity.clone()
            p_tm[1:, 1:] -= self.k[1:, None]*inverse_odd*self.k[None, 1:]
            eigenvalues, electric = self._eig((p_tm@normal)[None, :, :])
            kz = self._positive_kz(eigenvalues)
            magnetic = self._magnetic_eigenvectors(p_tm, normal, electric[0], kz[0])[None, :, :]
        self.stack = self._polarized_connect(self.stack,
            self._polarized_layer_smatrix(electric, magnetic, kz, dz))


def local_surface_selected(g, layer_list, wavelength, n_pmma, n_gold, void, order,
                           device='cpu', polarization='p'):
    if polarization not in ('s', 'p'):
        raise ValueError('Select exactly s or p for the single-polarization solver.')
    plan = get_plan(layer_list, g['period_nm'], order, device)
    with torch.no_grad():
        sim = SelectedCascade(plan, wavelength, n_pmma, polarization)
        ep, eg, ev = complex(n_pmma)**2, complex(n_gold)**2, complex(void)**2
        for j, dz in enumerate(plan.dz):
            ec, oc = plan.blocks(plan.coefficients[j, 0])
            eo, oo = (ec, oc) if plan.same[j] else plan.blocks(plan.coefficients[j, 1])
            direct = ev*sim.identity+(eg-ev)*eo+(ep-eg)*ec if polarization == 's' else None
            reciprocal = odd = None
            if polarization == 'p':
                reciprocal = sim.identity/ev+(1/eg-1/ev)*eo+(1/ep-1/eg)*ec
                odd = ev*sim.odd_identity+(eg-ev)*oo+(ep-eg)*oc
            sim.add_selected(direct, reciprocal, odd, dz)
        tf, rf = [q.detach().cpu().numpy() for q in sim.solve()]
    u = plan.basis
    tf = u[None, :, :]@tf@u.T[None, :, :]
    rf = u[None, :, :]@rf@u.T[None, :, :]
    if not np.all(np.isfinite(tf)) or not np.all(np.isfinite(rf)):
        raise RuntimeError('Nonfinite single-polarization scattering matrix.')
    return plan.orders.copy(), rf, tf
