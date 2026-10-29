"""Compact ports and affine wavevector coefficients from validated backends."""
import torch
from ..reduced import _ReducedScatteringMixin
from ..scattering import _StableLinearAlgebraMixin
from . import legacy as assembly
from .legacy import source
class CompactPort(_ReducedScatteringMixin,_StableLinearAlgebraMixin):
    """Frozen small boundary matrices; material-independent within a step."""
    def __init__(self,sim,plan,depth,layers):
        self._dtype,self._device = sim._dtype,sim._device
        self.stable_eig_grad = getattr(sim,'stable_eig_grad',False)
        self.omega = sim.omega
        self.K1_norm,self.K2_norm=sim.K1_norm,sim.K2_norm
        self.Kx_norm_dn,self.Ky_norm_dn=sim.Kx_norm_dn,sim.Ky_norm_dn
        self.sin_zeta,self.cos_zeta=sim.sin_zeta,sim.cos_zeta
        self.smatrix_size = 'half'
        self.thickness = [torch.tensor(depth,dtype=self._dtype,device=self._device)]*layers
        e,h = plan.full_electric,plan.full_magnetic
        with torch.no_grad():
            self._polarization_reference_v = h.mH@sim.Vf@e
            vi,vo = h.mH@sim.Vi@e,h.mH@sim.Vo@e
            self.input_interface = self._reduced_interface_s(self._polarization_reference_v,vi,input_side=True)
            self.output_interface = self._reduced_interface_s(self._polarization_reference_v,vo,input_side=False)
            self.source = e.mH@source(sim)
            count = sim.order_N
            symplectic_h = torch.cat((h[count:],-h[:count]),dim=0)
            pairing = e.mH@symplectic_h
            self.flux_in,self.flux_out = pairing@vi,pairing@vo
            self.incident_flux = .5*torch.real(self.source.conj()@(self.flux_in@self.source))

    def response(self,p_coeff,q_coeff,p0,q2,scale):
        modes = []
        for p2,q0 in zip(p_coeff,q_coeff):
            p,q = p0+scale*p2,q0+scale*q2
            squared,electric = self._eig(p@q)
            kz = self._positive_kz(squared)
            magnetic = self._magnetic_eigenvectors(p,q,electric,kz)
            modes.append({'electric':electric,'magnetic':magnetic,'kz':kz})
        transmission,reflection,_,_ = self._polarized_redheffer_scattering(modes,self.input_interface,self.output_interface)
        t,r = transmission@self.source,reflection@self.source
        R = .5*torch.real(r.conj()@(self.flux_in@r))/self.incident_flux
        T = .5*torch.real(t.conj()@(self.flux_out@t))/self.incident_flux
        return R,T

def hex_free_coefficients(unit,plan):
    n,d = plan.scalar_dimension,plan.reduced_dimension
    hx,hy = plan.magnetic_pair[:n],plan.magnetic_pair[n:]
    p_image = torch.cat((hy,-hx),dim=0)
    p0 = .5*(plan.electric.mH@p_image[:,:d]+plan.electric_partner.mH@p_image[:,d:])
    kx = unit.Kx_norm_dn.index_select(0,plan.indices)[:,None]
    ky = unit.Ky_norm_dn.index_select(0,plan.indices)[:,None]
    ex,ey = plan.electric_pair[:n],plan.electric_pair[n:]
    q_image = torch.cat((-kx*ky*ex+kx*kx*ey,-ky*ky*ex+ky*kx*ey),dim=0)
    q2 = .5*(plan.magnetic.mH@q_image[:,:d]+plan.magnetic_partner.mH@q_image[:,d:])
    return p0,q2

def square_free_coefficients(unit, context):
    n = context.scalar_dimension
    e, h = context.electric, context.magnetic
    p0 = e.mH @ torch.cat((h[n:], -h[:n]), dim=0)
    kx, ky = torch.diagonal(unit.K1_norm)[:, None], torch.diagonal(unit.K2_norm)[:, None]
    ex, ey = e[:n], e[n:]
    q2 = h.mH @ torch.cat((-kx*ky*ex+kx*kx*ey,
                           -ky*ky*ex+ky*kx*ey), dim=0)
    return p0, q2

def free_coefficients(unit,plan):
    return (hex_free_coefficients if plan.triangular else square_free_coefficients)(unit,plan)

