"""Direct scalar-block inverses/PQ with guarded legacy fallback."""
from __future__ import annotations
import cmath
from collections import OrderedDict
import math
import torch
from ..config import LayerRecord, _real_parameter_tensor
from . import legacy, d6
_SQUARE_PLANS = OrderedDict()


def plan_for(sim):
    if sim.lattice_kind == 'triangular':
        return d6.get_plan(sim)
    key = (tuple(int(x) for x in sim.order), tuple(float(x) for x in sim.L),
           str(sim._dtype), str(sim._device))
    if key not in _SQUARE_PLANS:
        plan = legacy.GeometryContext(sim)
        plan.full_electric = plan.full_basis(plan.electric)
        plan.full_magnetic = plan.full_basis(plan.magnetic)
        _SQUARE_PLANS[key] = plan
        while len(_SQUARE_PLANS) > 4:
            _SQUARE_PLANS.popitem(last=False)
    _SQUARE_PLANS.move_to_end(key)
    return _SQUARE_PLANS[key]


def material_inputs(sim, radius, index, grid, plan):
    return (d6.native_inputs if plan.triangular else legacy.material_inputs)(sim,radius,index,grid,plan)


def candidate_pq(sim, eps, rule, projection, plan):
    return (d6.reduced_pq if plan.triangular else legacy.candidate_pq)(sim,eps,rule,projection,plan)


def try_add_circle_layer(sim, thickness, radius, eps_bg, eps_cyl, centers, *,
                         core_radius=None, eps_core=None, use_lanczos=False, nx=256, ny=256):
    eligible = (sim.lattice_kind in ('square','triangular') and sim.use_group_theory
        and sim.group_theory_symmetry == ('d6' if sim.lattice_kind=='triangular' else 'c2v')
        and sim.polarization_reduction=='x' and sim.field_regions=='none'
        and not sim.store_mode_couplings and sim.smatrix_size=='half'
        and core_radius is None and eps_core is None and not use_lanczos and nx==ny
        and sim._dtype==torch.complex128 and int(sim.order[0])==int(sim.order[1])
        and abs(float(sim.L[0])-float(sim.L[1]))<1e-12
        and hasattr(sim,'inc_ang') and abs(float(sim.inc_ang))<1e-8 and len(centers)==1)
    if not eligible:
        return False
    if any(torch.is_tensor(x) and x.requires_grad for x in (eps_bg,eps_cyl)):
        return False
    bg,eps = complex(eps_bg),complex(eps_cyl)
    if bg!=1 or not math.isfinite(eps.real+eps.imag) or abs(eps)<1e-14 or eps.imag<0:
        return False
    if max(abs(float(a)-b) for a,b in zip(centers[0],legacy.center_of(sim)))>1e-12:
        return False
    r,value = _real_parameter_tensor('radius',radius,dtype=torch.float64,device=sim._device,allow_zero=False)
    if not 0<value<.5*min(float(x) for x in sim.L):
        return False
    depth,_ = _real_parameter_tensor('thickness',thickness,dtype=sim._dtype,device=sim._device,allow_zero=True)
    plan = plan_for(sim)
    p,q,info = candidate_pq(sim,*material_inputs(sim,r,cmath.sqrt(eps),nx,plan),plan)
    index = sim.layer_N
    kz = d6.install_layer(sim,p,q,plan,depth)
    for name in ('eps_conv','mu_conv','P','Q','E_eigvec','H_eigvec'):
        getattr(sim,name).append(None)
    sim.kz_norm.append(kz)
    version='d6-native-e1-pair-v1' if plan.triangular else 'c2v-direct-block-v1'
    diagnostic=dict(layer=index,requested=True,applied=True,symmetry='D6' if plan.triangular else 'C2v',
        backend='NVM',polarization='x',reduced_dimension=plan.reduced_dimension,
        full_dimension=2*sim.order_N,implementation=version,**{k:v for k,v in info.items() if k!='implementation'})
    sim.group_theory_diagnostics.append(diagnostic)
    sim.layer_records.append(LayerRecord(index=index,method='nvm',shape='circle',lattice=sim.lattice_kind,
        reason='DIRECT_SYMMETRY_BLOCK_NVM',options=dict(radius=value,grid=(nx,ny),
            group_theory=diagnostic,factorization_rule='symmetric-nv-anticommutator',implementation=version)))
    sim._nvm_eps_tensor=None
    return True
