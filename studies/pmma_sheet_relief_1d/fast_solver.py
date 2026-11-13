"""Exact mirror-even Li RCWA and wavelength-independent interval plans.

TM needs the odd block of epsilon^{-1}, because Kx maps even to odd. Both
parities of that inverse are accounted for; only odd incident fields are omitted.
"""
from collections import OrderedDict
import math
import numpy as np
import torch
from studies.gold_grating_1d.solver import Grating1D

_PLANS=OrderedDict()
_MAX_BYTES=128*1024*1024
_HITS=_MISSES=0


def canonical_device(device):
    d=torch.device(device)
    if d.type not in ('cpu','cuda'):raise ValueError('Use cpu or cuda[:index].')
    if d.type=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no CPU fallback.')
        d=torch.device('cuda',torch.cuda.current_device() if d.index is None else d.index)
    return d


class IntervalPlan:
    def __init__(self,layer_list,period,order,device):
        self.order=order;self.device=device;self.period=period
        self.orders=np.arange(-order,order+1,dtype=int)
        self.rows=tuple(tuple(float(row[k]) for k in ('z_top_nm','z_bottom_nm','core_width_nm','outer_width_nm')) for row in layer_list)
        fills=np.array([[r[2]/period,r[3]/period] for r in self.rows],dtype=float).reshape(-1,2)
        if np.any(fills<0) or np.any(fills>1):raise ValueError('Interval outside periodic cell.')
        k=np.arange(2*order+1)
        c=fills[:,:,None]*np.sinc(fills[:,:,None]*k)*(1-2*(k%2))
        c=np.where((fills==1)[:,:,None],(k==0)[None,None,:],c)
        self.coefficients=torch.as_tensor(c,dtype=torch.float64,device=device)
        m=torch.arange(order+1,device=device)
        self.diff=abs(m[:,None]-m[None,:]);self.sum=m[:,None]+m[None,:]
        v=torch.ones(order+1,dtype=torch.float64,device=device);v[0]=1/math.sqrt(2)
        self.even_weight=v[:,None]*v[None,:]
        self.same=[r[2]==r[3] for r in self.rows]
        self.dz=[(r[1]-r[0])/period for r in self.rows]
        u=np.zeros((2*order+1,order+1));u[order,0]=1
        for j in range(1,order+1):u[order-j,j]=u[order+j,j]=1/math.sqrt(2)
        self.basis=u
        self.bytes=sum(x.numel()*x.element_size() for x in [self.coefficients,self.diff,self.sum,self.even_weight])+u.nbytes

    def blocks(self,c):
        return ((c[self.diff]+c[self.sum])*self.even_weight,
                c[self.diff[1:,1:]]-c[self.sum[1:,1:]])


def get_plan(layer_list,period,order,device):
    global _HITS,_MISSES
    device=canonical_device(device)
    rows=tuple(tuple(float(row[k]) for k in ('z_top_nm','z_bottom_nm','core_width_nm','outer_width_nm')) for row in layer_list)
    key=(float(period),int(order),str(device),rows)
    if key in _PLANS:
        _HITS+=1;_PLANS.move_to_end(key);return _PLANS[key]
    _MISSES+=1;plan=IntervalPlan(layer_list,period,order,device)
    while _PLANS and sum(p.bytes for p in _PLANS.values())+plan.bytes>_MAX_BYTES:_PLANS.popitem(last=False)
    if plan.bytes<=_MAX_BYTES:_PLANS[key]=plan
    return plan


def clear_plan_cache():
    global _HITS,_MISSES
    _PLANS.clear();_HITS=_MISSES=0


def plan_cache_info():
    return dict(hits=_HITS,misses=_MISSES,plans=len(_PLANS),bytes=sum(p.bytes for p in _PLANS.values()),limit_bytes=_MAX_BYTES)


class EvenCascade:
    _eye=Grating1D._eye
    _solve=Grating1D._solve
    _eig=Grating1D._eig
    _positive_kz=staticmethod(Grating1D._positive_kz)
    _magnetic_eigenvectors=Grating1D._magnetic_eigenvectors
    _polarized_admittance=Grating1D._polarized_admittance
    _polarized_connect=Grating1D._polarized_connect
    _polarized_layer_smatrix=Grating1D._polarized_layer_smatrix

    def __init__(self,plan,wavelength,n_pmma):
        self._device=plan.device;self._dtype=torch.complex128
        self.order_N=plan.order+1;self.omega=2*math.pi*plan.period/wavelength
        self.k=torch.arange(self.order_N,dtype=torch.float64,device=plan.device)*wavelength/plan.period
        self.identity=self._eye(self.order_N);self.odd_identity=self._eye(plan.order)
        yf=self.admittance(2.);yp=self.admittance(n_pmma*n_pmma);yw=self.admittance(1.333**2)
        z=torch.zeros_like(self.identity)
        self.Vf=torch.cat((torch.cat((z,-torch.diag(yf[0])),1),torch.cat((torch.diag(yf[1]),z),1)),0)
        diag=torch.diag_embed
        self.stack=[diag(2*yp/(yp+yf)),diag((yp-yf)/(yp+yf)),diag((yf-yp)/(yp+yf)),diag(2*yf/(yp+yf))]
        self.output=[diag(2*yf/(yf+yw)),diag((yf-yw)/(yf+yw)),diag((yw-yf)/(yf+yw)),diag(2*yw/(yf+yw))]

    def admittance(self,eps):
        q=self._positive_kz(torch.as_tensor(eps,dtype=self._dtype,device=self._device)-self.k**2)
        safe=torch.where(abs(q)<1e-12,q+(1e-10+1e-10j),q)
        return torch.stack((safe,safe+self.k**2/safe))

    def add(self,direct_even,reciprocal_even,direct_odd,dz):
        normal=self._solve(reciprocal_even,self.identity)
        inverse_odd=self._solve(direct_odd,self.odd_identity)
        p_tm=self.identity.clone()
        p_tm[1:,1:]-=self.k[1:,None]*inverse_odd*self.k[None,1:]
        q_te=direct_even-torch.diag(self.k*self.k)
        eigenvalues,electric=self._eig(torch.stack((q_te,p_tm@normal)))
        kz=self._positive_kz(eigenvalues)
        magnetic=torch.stack((electric[0]*kz[0][None,:],self._magnetic_eigenvectors(p_tm,normal,electric[1],kz[1])))
        self.stack=self._polarized_connect(self.stack,self._polarized_layer_smatrix(electric,magnetic,kz,dz))

    def solve(self):
        return self._polarized_connect(self.stack,self.output,forward_only=True)[:2]


def local_surface_fast(g,layer_list,wavelength,n_pmma,n_gold,void,order,device='cpu'):
    plan=get_plan(layer_list,g['period_nm'],order,device)
    with torch.no_grad():
        sim=EvenCascade(plan,wavelength,n_pmma)
        ep,eg,ev=complex(n_pmma)**2,complex(n_gold)**2,complex(void)**2
        for j,dz in enumerate(plan.dz):
            ec,oc=plan.blocks(plan.coefficients[j,0])
            eo,oo=(ec,oc) if plan.same[j] else plan.blocks(plan.coefficients[j,1])
            direct=ev*sim.identity+(eg-ev)*eo+(ep-eg)*ec
            reciprocal=sim.identity/ev+(1/eg-1/ev)*eo+(1/ep-1/eg)*ec
            odd=ev*sim.odd_identity+(eg-ev)*oo+(ep-eg)*oc
            sim.add(direct,reciprocal,odd,dz)
        tf,rf=[q.detach().cpu().numpy() for q in sim.solve()]
    u=plan.basis
    # These embedded matrices contain only the even-incidence response.
    tf=u[None,:,:]@tf@u.T[None,:,:];rf=u[None,:,:]@rf@u.T[None,:,:]
    if not np.all(np.isfinite(tf)) or not np.all(np.isfinite(rf)):raise RuntimeError('Nonfinite even scattering matrix.')
    return plan.orders.copy(),rf,tf
