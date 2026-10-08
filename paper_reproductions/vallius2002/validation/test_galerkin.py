"""Physical regression checks for the TM Galerkin boundary trace."""
import unittest
import numpy as np
from scipy import linalg
from scipy.special import roots_legendre
from threadpoolctl import threadpool_limits

from ..solver import (LayerSpec, PreparedStack, _Diagnostics, _interface, _star,
                      _propagate, _sqrt_outgoing)


def imaginary(a):
    return (a-a.conj().T)/(2j)


def right_solve(a,b):
    return linalg.solve(b.T,a.T,check_finite=False).T


class GalerkinPhysicalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.limits=threadpool_limits(limits=1)

    @classmethod
    def tearDownClass(cls):
        cls.limits.restore_original_limits()

    def stack(self,layers,**kwargs):
        return PreparedStack(layers,polarization='TM',method='asr',harmonics=9,
                             oversampling=4,q_projection='galerkin',device='cpu',backend='scipy',**kwargs)

    def test_lossless_different_maps_conserve_propagating_power(self):
        layers=[LayerSpec(.21,(0,.37,1),(4,1)),
                LayerSpec(.13,(0,.19,.64,1),(1,2.25,1))]
        for angle in (0.,.17):
            for wavelength in (.72,1.21):
                result=self.stack(layers,angle=angle,epsilon_out=2.25).solve(wavelength)
                self.assertAlmostEqual(result['R']+result['T'],1.,delta=2e-9)

    def test_legacy_projection_creates_gain_before_interface_and_galerkin_removes_it(self):
        shell=LayerSpec(1/120,(0,.3245,.4745,.5255,.6755,1),
                        (1,-9.3875+1.5292j,2.22,-9.3875+1.5292j,1))
        stack=PreparedStack([shell],polarization='TM',method='asr',harmonics=17,oversampling=4,
                            q_projection='galerkin',device='cpu',backend='scipy')
        layer=stack.prepared_layers[0];k=2*np.pi/3
        for mode in ('direct','galerkin'):
            w,v,gamma=layer.modes(k,0,'TM',q_projection=mode)
            self.assertGreater(float(gamma.imag.min()),0)
            generators=(right_solve(w*gamma[None,:],v),right_solve(v*gamma[None,:],w))
            minimum=min(float(linalg.eigvalsh(imaginary(a)).min()) for a in generators)
            if mode=='direct':
                self.assertLess(minimum,-1.)
            else:
                for a in generators:
                    self.assertGreaterEqual(linalg.eigvalsh(imaginary(a)).min(),-1e-10*max(1,linalg.norm(a,2)))

    def test_passive_multilayer_cannot_amplify_any_propagating_two_port_excitation(self):
        layers=[LayerSpec(.12,(0,.24,.48,.72,1),(1,-5+1j,2.25,1)),
                LayerSpec(.09,(0,.41,1),(3+.2j,1))]
        stack=self.stack(layers,epsilon_out=2.25,angle=.12)
        _,_,k,alpha0,_=stack._wavelength_parameters(.72,1e-12)
        wl,vl,gin=stack._external_modes(k,alpha0,stack.epsilon_in)
        wo,vo,gout=stack._external_modes(k,alpha0,stack.epsilon_out)
        d=_Diagnostics(False);s=None
        for spec,p in zip(stack.layers,stack.prepared_layers):
            w,v,g=p.modes(k,alpha0,'TM',q_projection='galerkin')
            interface=_interface(wl,vl,w,v,d)
            s=interface if s is None else _star(s,interface,d)
            s=_propagate(s,np.exp(1j*g*spec.thickness));wl,vl=w,v
        s=_star(s,_interface(wl,vl,wo,vo,d),d)
        full=np.block([[s[0],s[1]],[s[2],s[3]]])
        flux=np.concatenate(((gin/stack.epsilon_in).real,(gout/stack.epsilon_out).real))
        keep=np.flatnonzero(flux>1e-10)
        powers=np.sqrt(flux[keep])
        propagating=full[np.ix_(keep,keep)]*powers[:,None]/powers[None,:]
        self.assertLessEqual(linalg.svdvals(propagating)[0],1+2e-9)

    def test_absorption_matches_independent_integral_of_internal_dissipation(self):
        spec=LayerSpec(.05,(0,.35,1),(-5+1j,1))
        stack=self.stack([spec],epsilon_out=2.25)
        result=stack.solve(1.3)
        p=stack.prepared_layers[0];k=2*np.pi/1.3
        alpha=2*np.pi*p.orders
        operator=k*k*p.f-alpha[:,None]*p.inverse_a*alpha[None,:]
        values,e=linalg.eig(operator,p.b,check_finite=False)
        keep=np.argsort(np.abs(values),kind='stable')[:stack.harmonics]
        e=e[:,keep];gamma=_sqrt_outgoing(values[keep])
        scales=np.maximum(linalg.norm(p.K0@e,axis=0),1e-30);e=e/scales[None,:]
        w,v,g=p.modes(k,0,'TM',q_projection='galerkin')
        _,vin,_=stack._external_modes(k,0,1)
        incident=np.zeros(stack.harmonics,complex);incident[stack.harmonics//2]=1
        field=incident+result['r'];companion=vin@(incident-result['r'])
        coefficients=linalg.solve(np.block([[w,w],[v,-v]]),np.concatenate((field,companion)))
        a,b=np.split(coefficients,2)
        nodes,weights=roots_legendre(80);z=(nodes+1)*spec.thickness/2
        loss=0.
        for pos,weight in zip(z,weights):
            plus=a*np.exp(1j*gamma*pos);minus=b*np.exp(-1j*gamma*pos)
            h=e@(plus+minus);dz=e@(gamma*(plus-minus))
            density=(np.vdot(dz,-imaginary(p.b)@dz)+np.vdot(h,imaginary(operator)@h)).real
            self.assertGreaterEqual(density,-1e-10)
            loss+=weight*density*spec.thickness/2/k
        self.assertAlmostEqual(loss,result['A'],delta=2e-8)


if __name__=='__main__':unittest.main()
