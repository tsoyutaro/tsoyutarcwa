"""D6-specific, source-row NVM assembly for centered, lossless circles.

The twelve-term Reynolds average is evaluated exactly through the two E1
matrix-unit partner rows. Scalar E1/E2 multiplicity inverses are shared between
their two rows. Native-star inputs bypass rectangular inverse/P/Q assembly.
No production module is monkey-patched.
"""
from __future__ import annotations
from collections import OrderedDict
import math
import torch
from rcwa_solver_auto import AutoRCWA
from rcwa_ext.config import LayerRecord, _real_parameter_tensor
from rcwa_ext.nvm import _jinc, _smoothstep

_PLANS = OrderedDict()
VERSION = 'd6-native-e1-pair-v1'


def relative_error(a,b):
    return float((torch.linalg.vector_norm(a-b)/torch.clamp(torch.linalg.vector_norm(b),min=1e-30)).detach())


def center_of(sim):
    return (float(sim.L[0])/2+float(sim.L[1])*sim.cos_zeta/2,
            float(sim.L[1])*sim.sin_zeta/2)


class D6Plan:
    """Geometry-only objects reused across radii, wavelengths and layers."""
    triangular = True
    def __init__(self,sim):
        embedding,group = sim._triangular_star_group_operators()
        self.full_count = sim.order_N
        self.scalar_dimension = embedding.shape[1]//2
        self.indices = torch.argmax(torch.abs(embedding[:self.full_count,:self.scalar_dimension]),dim=0)
        ox,oy = torch.meshgrid(sim.order_x,sim.order_y,indexing='ij')
        self.m = ox.reshape(-1).index_select(0,self.indices)
        self.n = oy.reshape(-1).index_select(0,self.indices)
        self.scalar_rows,self.scalar_bases,self.labels = [],[],[]
        for label,matrices in sim._d6_irrep_matrices().items():
            first = sim._projector_range(sim._d6_matrix_unit(group,matrices,representation='scalar',row=0,column=0))
            if not first.shape[1]:
                continue
            rows = [first]
            for row in range(1,matrices[0].shape[0]):
                rows.append(sim._d6_matrix_unit(group,matrices,representation='scalar',row=row,column=0)@first)
            self.scalar_rows.append(tuple(rows))
            self.scalar_bases.append(torch.cat(rows,dim=1))
            self.labels.append(label)
        all_scalar = torch.cat(self.scalar_bases,dim=1)
        if all_scalar.shape[1] != self.scalar_dimension or relative_error(all_scalar@all_scalar.mH,sim._eye(self.scalar_dimension))>1e-9:
            raise RuntimeError('D6 scalar rows do not form a complete orthogonal basis.')
        matrices = sim._d6_irrep_matrices()['E1']
        self.electric = sim._projector_range(sim._d6_matrix_unit(group,matrices,representation='electric',row=0,column=0))
        self.magnetic = sim._projector_range(sim._d6_matrix_unit(group,matrices,representation='magnetic',row=0,column=0))
        self.electric_partner = sim._d6_matrix_unit(group,matrices,representation='electric',row=1,column=0)@self.electric
        self.magnetic_partner = sim._d6_matrix_unit(group,matrices,representation='magnetic',row=1,column=0)@self.magnetic
        self.reduced_dimension = self.electric.shape[1]
        self.electric_pair = torch.cat((self.electric,self.electric_partner),dim=1)
        self.magnetic_pair = torch.cat((self.magnetic,self.magnetic_partner),dim=1)
        identity = sim._eye(2*self.reduced_dimension)
        if max(relative_error(self.electric_pair.mH@self.electric_pair,identity),
               relative_error(self.magnetic_pair.mH@self.magnetic_pair,identity))>1e-9:
            raise RuntimeError('E1 partner rows are not mutually orthonormal.')
        self.full_electric = self.full_basis(self.electric)
        self.full_magnetic = self.full_basis(self.magnetic)
        # Dense group operators are deliberately not retained by the plan.

    def scalar(self,matrix):
        return matrix.index_select(0,self.indices).index_select(1,self.indices)

    def vector(self,matrix):
        indices = torch.cat((self.indices,self.indices+self.full_count))
        return matrix.index_select(0,indices).index_select(1,indices)

    def full_basis(self,basis):
        indices = torch.cat((self.indices,self.indices+self.full_count))
        result = basis.new_zeros((2*self.full_count,basis.shape[1]))
        return result.index_copy(0,indices,basis)


def get_plan(sim):
    device = str(sim._device)
    if torch.device(sim._device).type == 'cuda':
        device = f'cuda:{torch.cuda.current_device() if torch.device(sim._device).index is None else torch.device(sim._device).index}'
    key = (tuple(int(x) for x in sim.order),tuple(float(x) for x in sim.L),
           sim.zeta_deg,str(sim._dtype),device)
    if key not in _PLANS:
        _PLANS[key] = D6Plan(sim)
        while len(_PLANS)>4:
            _PLANS.popitem(last=False)
    _PLANS.move_to_end(key)
    return _PLANS[key]


def clear_plan_cache():
    _PLANS.clear()


class D6BlockInverse:
    """One multiplicity inverse per irrep, shared by its partner rows."""
    def __init__(self,sim,matrix,plan):
        self.entries,self.sizes,self.row_counts = [],[],[]
        self.leakage = 0.
        for rows in plan.scalar_rows:
            first = rows[0]
            small = first.mH@(matrix@first)
            for basis in rows:
                self.leakage = max(self.leakage,relative_error(matrix@basis,basis@small))
            size = small.shape[0]
            self.entries.append((rows,sim._solve(small,sim._eye(size))))
            self.sizes.append(size)
            self.row_counts.append(len(rows))
        if self.leakage>2e-9:
            raise ValueError(f'D6 multiplicity invariance failed: {self.leakage:.3e}')

    def apply(self,value):
        return sum(basis@(inverse@(basis.mH@value))
                   for rows,inverse in self.entries for basis in rows)


def native_inputs(sim,radius,index,grid,plan):
    """Same circle coefficients and sampled normal field, native star only."""
    lx,ly = float(sim.L[0]),float(sim.L[1])
    s,c = sim.sin_zeta,sim.cos_zeta
    gx = plan.m.to(torch.float64)*(2*math.pi/lx)
    gy = (plan.n.to(torch.float64)*(2*math.pi/ly)-c*gx)/s
    dx,dy = gx[:,None]-gx[None,:],gy[:,None]-gy[None,:]
    cx,cy = center_of(sim)
    disk = (math.pi*radius**2/(lx*ly*s))*_jinc(torch.sqrt(dx**2+dy**2)*radius)*torch.exp(-1j*(dx*cx+dy*cy))
    identity = sim._eye(plan.scalar_dimension)
    eps = identity+(index**2-1)*disk
    inverse_rule = identity+(1/index**2-1)*disk

    minimum = max(32,4*int(sim.order[0])+4,4*int(sim.order[1])+4)
    if grid<minimum:
        raise ValueError(f'Grid {grid} must be at least {minimum}.')
    xi = torch.arange(grid,dtype=torch.float64,device=sim._device)/grid*lx
    eta = torch.arange(grid,dtype=torch.float64,device=sim._device)/grid*ly
    u,v = torch.meshgrid(xi,eta,indexing='ij')
    x,y = u+c*v,s*v
    dxs,dys = [],[]
    # Use the exact same image search and tie handling as the frozen reference.
    centre = torch.tensor([cx,cy],dtype=torch.float64,device=sim._device)
    a1 = torch.tensor([lx,0.],dtype=torch.float64,device=sim._device)
    a2 = torch.tensor([ly*c,ly*s],dtype=torch.float64,device=sim._device)
    for i in range(-2,3):
        for j in range(-2,3):
            image = centre+i*a1+j*a2
            dxs.append(x-image[0]); dys.append(y-image[1])
    dxs,dys = torch.stack(dxs),torch.stack(dys)
    distance,indices = torch.topk(dxs**2+dys**2,k=2,dim=0,largest=False)
    nearest = indices[0]
    dx = torch.gather(dxs,0,nearest[None])[0]
    dy = torch.gather(dys,0,nearest[None])[0]
    r1,r2 = torch.sqrt(torch.clamp(distance[0],min=0.)),torch.sqrt(torch.clamp(distance[1],min=0.))
    safe = torch.clamp(r1,min=1e-14)
    nx,ny = dx/safe,dy/safe
    resolution = max(lx/grid,ly/grid)
    taper = torch.maximum(radius/2,radius.new_tensor(resolution))
    weight = _smoothstep(r1,0.,taper)*_smoothstep(r2-r1,0.,3*resolution)
    normal_fields = (.5+(nx**2-.5)*weight,nx*ny*weight,.5+(ny**2-.5)*weight)
    dm = plan.m[:,None]-plan.m[None,:]
    dn = plan.n[:,None]-plan.n[None,:]
    blocks = []
    for field in normal_fields:
        coefficients = torch.fft.fft2(field.to(sim._dtype))/(grid*grid)
        block = coefficients[dm,dn]
        blocks.append(.5*(block+block.mH))
    xx,xy,yy = blocks
    projection = torch.cat((torch.cat((xx,xy),dim=1),torch.cat((xy.mH,yy),dim=1)),dim=0)
    return eps,inverse_rule,projection


def reduced_pq(sim,eps,inverse_rule,projection,plan):
    """Exact E1 Reynolds average through two partner rows, evaluated in batch."""
    eps_inverse,rule_inverse = D6BlockInverse(sim,eps,plan),D6BlockInverse(sim,inverse_rule,plan)
    n,d = plan.scalar_dimension,plan.reduced_dimension
    kx = sim.Kx_norm_dn.index_select(0,plan.indices)[:,None]
    ky = sim.Ky_norm_dn.index_select(0,plan.indices)[:,None]
    h1,h2 = plan.magnetic_pair[:n],plan.magnetic_pair[n:]
    inverse = eps_inverse.apply(ky*h1-kx*h2)
    p_image = torch.cat((kx*inverse+h2,ky*inverse-h1),dim=0)

    def delta(value):
        return rule_inverse.apply(value)-eps@value

    e1,e2 = plan.electric_pair[:n],plan.electric_pair[n:]
    de1,de2 = delta(e1),delta(e2)
    px = projection[:n,:n]@e1+projection[:n,n:]@e2
    py = projection[n:,:n]@e1+projection[n:,n:]@e2
    tx = eps@e1+.5*(delta(px)+projection[:n,:n]@de1+projection[:n,n:]@de2)
    ty = eps@e2+.5*(delta(py)+projection[n:,:n]@de1+projection[n:,n:]@de2)
    q_image = torch.cat((-kx*ky*e1+kx*kx*e2-ty,tx-ky*ky*e1+ky*kx*e2),dim=0)
    p = .5*(plan.electric.mH@p_image[:,:d]+plan.electric_partner.mH@p_image[:,d:])
    q = .5*(plan.magnetic.mH@q_image[:,:d]+plan.magnetic_partner.mH@q_image[:,d:])
    return p,q,{'implementation':VERSION,'scalar_inverse_block_sizes':eps_inverse.sizes,
        'scalar_irrep_row_counts':eps_inverse.row_counts,
        'scalar_invariance_error':max(eps_inverse.leakage,rule_inverse.leakage),
        'full_scalar_inverse_assembled':False,'full_PQ_assembled':False,
        'rectangular_material_matrices_assembled':False,'hex_reynolds_operations':0,
        'hex_partner_rows':2,'full_convolution_and_normal_inputs_retained':False,
        'native_star_normal_projection_retained':True}


def install_layer(sim,p,q,plan,depth):
    squared,electric = sim._eig(p@q)
    kz = sim._positive_kz(squared)
    magnetic = sim._magnetic_eigenvectors(p,q,electric,kz)
    order = torch.argsort(kz.real-1e-6*kz.imag,descending=True)
    kz,electric,magnetic = kz[order],electric[:,order],magnetic[:,order]
    sim._install_polarization_basis(plan.full_electric,plan.full_magnetic,1e-8)
    sim._polarized_layers.append({'electric':electric,'magnetic':magnetic,'kz':kz})
    sim.thickness.append(torch.as_tensor(depth,dtype=sim._dtype,device=sim._device))
    sim.layer_N += 1
    return kz


class D6FastRCWA(AutoRCWA):
    """Drop-in source-response path; other requests use the original solver."""
    def add_layer_circle_nvm(self,thickness,radius,eps_bg,eps_cyl,centers=((0.,0.),),*,
                             core_radius=None,eps_core=None,use_lanczos=False,lanczos_power=2,nx=256,ny=256):
        eligible = (self.lattice_kind=='triangular' and self.use_group_theory
            and self.group_theory_symmetry=='d6' and self.polarization_reduction=='x'
            and self.field_regions=='none' and not self.store_mode_couplings
            and core_radius is None and eps_core is None and not use_lanczos and nx==ny
            and complex(eps_bg)==1 and complex(eps_cyl).imag==0 and complex(eps_cyl).real>0
            and self._dtype==torch.complex128)
        if not eligible:
            return super().add_layer_circle_nvm(thickness,radius,eps_bg,eps_cyl,centers,
                core_radius=core_radius,eps_core=eps_core,use_lanczos=use_lanczos,
                lanczos_power=lanczos_power,nx=nx,ny=ny)
        if abs(float(self.inc_ang))>1e-8 or len(centers)!=1 or max(abs(a-b) for a,b in zip(centers[0],center_of(self)))>1e-12:
            raise ValueError('D6 fast path requires normal incidence and one centered circle.')
        r,r_value = _real_parameter_tensor('radius',radius,dtype=torch.float64,device=self._device,allow_zero=False)
        if r_value>=.5*min(float(x) for x in self.L):
            raise ValueError('Periodic circles overlap or touch.')
        depth,_ = _real_parameter_tensor('thickness',thickness,dtype=self._dtype,device=self._device,allow_zero=True)
        plan = get_plan(self)
        inputs = native_inputs(self,r,math.sqrt(complex(eps_cyl).real),nx,plan)
        p,q,info = reduced_pq(self,*inputs,plan)
        index = self.layer_N
        kz = install_layer(self,p,q,plan,depth)
        # The fields-disabled contract retains only reduced modes and ports.
        for name in ('eps_conv','mu_conv','P','Q','E_eigvec','H_eigvec'):
            getattr(self,name).append(None)
        self.kz_norm.append(kz)
        self.group_theory_diagnostics.append({'layer':index,'requested':True,'applied':True,
            'symmetry':'D6-E1-partner-pair','backend':'NVM','polarization':'x',
            'reduced_dimension':plan.reduced_dimension,'full_dimension':2*self.order_N,**info})
        self.layer_records.append(LayerRecord(index=index,method='nvm',shape='circle',
            lattice='triangular',reason='D6_NATIVE_SOURCE_ROW',options={'radius':r_value,
                'grid':(nx,ny),'group_theory':self.group_theory_diagnostics[-1],
                'factorization_rule':'symmetric-nv-anticommutator','implementation':VERSION}))
        self._nvm_eps_tensor = None
