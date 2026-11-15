"""Analytic 1D material coefficients and the paper's ordered FFF normal projection."""
from collections import OrderedDict
import math
import numpy as np
import torch
from numpy.polynomial.legendre import leggauss

_PLANS = OrderedDict()
_LIMIT = 128*1024*1024
_HITS = _MISSES = 0


def angle_parameters(g, theta_deg=None):
    angle = g['slope_deg'] if theta_deg is None else float(theta_deg)
    if not math.isfinite(angle) or not 0 <= angle <= 90:
        raise ValueError('Projection angle must be finite and in [0,90] degrees.')
    # Exact endpoints are required for an exact algebraic RCWA reduction.
    if angle == 90:
        return angle, 0., 0.
    if angle == 0:
        return angle, 1., 0.
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    return angle, c*c, c*s


class Transition:
    def __init__(self, radius, period, delta, inner, outer, quadrature):
        self.inner, self.outer = inner, outer
        # Delta is total width along x, centered on the sharp interface.
        # Near a cell seam or axis, use the part inside this half-cell.
        lo, hi = max(0., radius-delta/2), min(period/2, radius+delta/2)
        xs, ws = leggauss(quadrature)
        self.parts = []
        for a,b,label in [(lo,radius,inner),(radius,hi,outer)]:
            if b <= a:
                continue
            nodes = (a+b)/2+(b-a)/2*xs
            weights = (b-a)/2*ws
            fractions = (nodes-(radius-delta/2))/delta
            self.parts.append((nodes, weights, fractions, label))

    def correction(self, indices, period, harmonic):
        eps = np.zeros(len(harmonic),complex)
        eta = np.zeros(len(harmonic),complex)
        na, nb = indices[self.inner], indices[self.outer]
        parity = 1-2*(harmonic % 2)
        for nodes,weights,f,label in self.parts:
            n = (1-f)*na+f*nb
            if np.any(abs(n)<1e-14):
                raise ValueError('GIL index reaches zero; reciprocal permittivity is singular.')
            base = indices[label]**2
            kernel = np.cos(2*np.pi/period*harmonic[:,None]*nodes[None,:])*weights[None,:]
            eps += 2/period*parity*(kernel@(n*n-base))
            eta += 2/period*parity*(kernel@(1/(n*n)-1/base))
        return eps, eta


class Plan:
    def __init__(self,g,stack,order,device,gil_nm,gil_interfaces,gil_quadrature):
        from fast_solver import get_plan
        if gil_interfaces not in ('inner','outer','both'):
            raise ValueError('GIL interfaces must be inner, outer, or both.')
        if not math.isfinite(gil_nm) or gil_nm < 0 or gil_quadrature < 2:
            raise ValueError('Require GIL >=0 and quadrature >=2.')
        if gil_nm and not g['gold_vertical_nm']:
            raise ValueError('Metal-interface GIL requires an Au coating.')
        self.base=get_plan(stack,g['period_nm'],order,device)
        self.order,self.period=order,g['period_nm']
        self.harmonic=np.arange(2*order+1)
        m=np.arange(order+1)[:,None];n=np.arange(1,order+1)[None,:]
        def sign_coeff(k):
            safe=np.where(k==0,1,k)
            return np.where(k==0,0,(1-(-1.)**k)/(1j*np.pi*safe))
        cross=sign_coeff(m-n)-sign_coeff(m+n)
        cross[0]/=math.sqrt(2)
        self.cross=torch.as_tensor(cross,dtype=torch.complex128,device=self.base.device)
        self.transitions=[]
        for row in stack:
            ri,ro=row['core_width_nm']/2,row['outer_width_nm']/2
            transitions=[]
            active_inner=0<ri<self.period/2 and ri<ro
            active_outer=0<ro<self.period/2 and ri<ro
            if gil_nm and active_inner and active_outer and gil_interfaces=='both' and ro-ri<gil_nm-1e-10:
                raise ValueError('Inner/outer GIL regions overlap across the Au film; reduce --gil-nm.')
            if gil_nm and active_inner and gil_interfaces in ('inner','both'):
                transitions.append(Transition(ri,self.period,gil_nm,'pmma','gold',gil_quadrature))
            if gil_nm and active_outer and gil_interfaces in ('outer','both'):
                transitions.append(Transition(ro,self.period,gil_nm,'gold','void',gil_quadrature))
            self.transitions.append(transitions)
        self.bytes=self.base.bytes+self.cross.numel()*self.cross.element_size()+sum(a.nbytes for row in self.transitions for t in row for part in t.parts for a in part[:3])

    def material(self,j,n_pmma,n_gold,n_void):
        ep,eg,ev=complex(n_pmma)**2,complex(n_gold)**2,complex(n_void)**2
        core,outer=self.base.coefficients[j]
        unit=torch.zeros_like(core,dtype=torch.complex128);unit[0]=1.
        e=ev*unit+(eg-ev)*outer+(ep-eg)*core
        eta=unit/ev+(1/eg-1/ev)*outer+(1/ep-1/eg)*core
        if self.transitions[j]:
            indices=dict(pmma=complex(n_pmma),gold=complex(n_gold),void=complex(n_void))
            de=np.zeros(len(self.harmonic),complex);di=de.copy()
            for transition in self.transitions[j]:
                a,b=transition.correction(indices,self.period,self.harmonic);de+=a;di+=b
            e=e+torch.as_tensor(de,dtype=torch.complex128,device=self.base.device)
            eta=eta+torch.as_tensor(di,dtype=torch.complex128,device=self.base.device)
        return e,eta


def get_fff_plan(g,stack,order,device,gil_nm=0.,gil_interfaces='both',gil_quadrature=16):
    global _HITS,_MISSES
    rows=tuple(tuple(float(r[k]) for k in ('z_top_nm','z_bottom_nm','core_width_nm','outer_width_nm')) for r in stack)
    key=(g['period_nm'],g['gold_vertical_nm'],rows,int(order),str(device),float(gil_nm),gil_interfaces,int(gil_quadrature))
    if key in _PLANS:
        _HITS+=1;_PLANS.move_to_end(key);return _PLANS[key]
    _MISSES+=1;plan=Plan(g,stack,order,device,gil_nm,gil_interfaces,gil_quadrature)
    while _PLANS and sum(x.bytes for x in _PLANS.values())+plan.bytes>_LIMIT:
        _PLANS.popitem(last=False)
    if plan.bytes<=_LIMIT:_PLANS[key]=plan
    return plan


def clear_fff_cache():
    global _HITS,_MISSES
    _PLANS.clear();_HITS=_MISSES=0


def cache_info():
    return dict(hits=_HITS,misses=_MISSES,plans=len(_PLANS),bytes=sum(p.bytes for p in _PLANS.values()),limit_bytes=_LIMIT)
