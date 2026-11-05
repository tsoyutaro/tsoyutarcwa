"""Experimental NVM assembly. Does not patch or replace production code.

Scalar convolution and normal-projector inputs are still assembled by the
frozen solver. Full scalar inverses, NVM epsilon tensors and full P/Q are not
assembled by the candidate. Hexagonal operators use the SAME native D6 star
and Reynolds average as the reference; they never restrict a box inverse.
"""
from __future__ import annotations
import math
import torch
from ..auto import AutoRCWA
from ..config import GroupTheoryOptions, Lattice, NVMOptions, OutputSpec


def relative_error(actual, expected):
    return float((torch.linalg.vector_norm(actual-expected) /
                  torch.clamp(torch.linalg.vector_norm(expected), min=1e-30)).detach().cpu())


def make_sim(lattice, order, grid, index, wavelength_nm, period_nm, device):
    sim = AutoRCWA(freq=period_nm/wavelength_nm, order=[order, order],
        lattice=Lattice.square(1.) if lattice == 'square' else Lattice.triangular(1.),
        nvm=NVMOptions(grid=(grid, grid), use_lanczos=False),
        outputs=OutputSpec(smatrix_size='half', fields='none'), cascade='redheffer',
        group_theory=GroupTheoryOptions(enabled=True,
            symmetry='c2v' if lattice == 'square' else 'd6', polarization='x', strict=True),
        dtype=torch.complex128, device=device, verify_cascade=False)
    sim.add_input_layer(eps=1.)
    sim.add_output_layer(eps=index**2)
    sim.set_incident_angle(0., 0.)
    return sim


def center_of(sim):
    return (float(sim.L[0])/2 + float(sim.L[1])*sim.cos_zeta/2,
            float(sim.L[1])*sim.sin_zeta/2)


def check_supported(sim, radius, center):
    if abs(float(sim.inc_ang)) > 1e-8:
        raise ValueError('Prototype requires normal incidence.')
    if sim.lattice_kind not in ('square', 'rectangular', 'triangular'):
        raise ValueError('Prototype requires square or triangular lattice.')
    if abs(float(sim.L[0])-float(sim.L[1])) > 1e-12:
        raise ValueError('Prototype requires equal lattice lengths.')
    if sim.polarization_reduction != 'x':
        raise ValueError('Prototype is validated for x polarization only.')
    expected = center_of(sim)
    if max(abs(a-b) for a,b in zip(center, expected)) > 1e-12:
        raise ValueError('Prototype requires the circle at the primitive-cell centre.')
    if not 0 < float(radius.detach()) < .5*float(sim.L[0]):
        raise ValueError('Prototype requires a nonzero non-overlapping circle.')


class GeometryContext:
    """Material-independent bases; can be reused across radii/wavelengths."""
    def __init__(self, sim):
        self.triangular = sim.lattice_kind == 'triangular'
        self.scalar_bases = []
        self.labels = []
        self.group = None
        n = sim.order_N
        if not self.triangular:
            even_x, odd_x = sim._one_dimensional_parity_bases(sim.order_x, 2*math.pi/float(sim.L[0]), center_of(sim)[0])
            even_y, odd_y = sim._one_dimensional_parity_bases(sim.order_y, 2*math.pi/float(sim.L[1]), center_of(sim)[1])
            for label, bx, by in [('ee',even_x,even_y),('eo',even_x,odd_y),('oe',odd_x,even_y),('oo',odd_x,odd_y)]:
                self.labels.append(label)
                self.scalar_bases.append(sim._kronecker(bx,by))
            self.electric, self.magnetic = sim._c2v_source_bases(center_of(sim),'x')
            self.embedding = None
            self.indices = None
        else:
            embedding, group = sim._triangular_star_group_operators()
            count = embedding.shape[1]//2
            self.embedding = embedding
            self.indices = torch.argmax(torch.abs(embedding[:n,:count]),dim=0)
            self.group = group
            for label,(dim,characters) in sim._d6_character_table().items():
                projector = sum((dim*complex(character).conjugate()/len(group))*entry['scalar']
                                for character,entry in zip(characters,group))
                basis = sim._projector_range(projector)
                if basis.shape[1]:
                    self.labels.append(label)
                    self.scalar_bases.append(basis)
            matrices = sim._d6_irrep_matrices()['E1']
            self.electric = sim._projector_range(sim._d6_matrix_unit(group,matrices,representation='electric',row=0,column=0))
            self.magnetic = sim._projector_range(sim._d6_matrix_unit(group,matrices,representation='magnetic',row=0,column=0))
        self.scalar_dimension = self.scalar_bases[0].shape[0]
        self.reduced_dimension = self.electric.shape[1]
        combined = torch.cat(self.scalar_bases,dim=1)
        identity = torch.eye(self.scalar_dimension,dtype=sim._dtype,device=sim._device)
        if combined.shape[1] != self.scalar_dimension or relative_error(combined@combined.mH,identity)>1e-9:
            raise RuntimeError('Scalar symmetry bases are not a complete orthogonal decomposition.')

    def scalar(self, matrix):
        if self.indices is None:
            return matrix
        return matrix.index_select(0,self.indices).index_select(1,self.indices)

    def vector(self, matrix):
        if self.indices is None:
            return matrix
        n = matrix.shape[0]//2
        indices = torch.cat((self.indices,self.indices+n))
        return matrix.index_select(0,indices).index_select(1,indices)

    def full_basis(self, basis):
        return self.embedding@basis if self.embedding is not None else basis


class BlockInverse:
    """Invert scalar invariant blocks, never reconstruct the full inverse."""
    def __init__(self, sim, matrix, context):
        self.bases = context.scalar_bases
        self.blocks = []
        self.leakage = 0.
        self.sizes = []
        for basis in self.bases:
            image = matrix@basis
            reduced = basis.mH@image
            self.leakage = max(self.leakage,relative_error(image,basis@reduced))
            size = reduced.shape[0]
            self.sizes.append(size)
            self.blocks.append(sim._solve(reduced,sim._eye(size)))
        if self.leakage > 2e-9:
            raise ValueError(f'Scalar matrix is not invariant: leakage={self.leakage:.3e}')

    def apply(self, value):
        return sum(basis@(inverse@(basis.mH@value)) for basis,inverse in zip(self.bases,self.blocks))


def material_inputs(sim, radius, index, grid, context):
    center = center_of(sim)
    check_supported(sim,radius,center)
    args = dict(use_lanczos=False,lanczos_power=2)
    eps = sim._circle_toeplitz(radius,1.,index**2,(center,),**args)
    inverse_rule = sim._circle_toeplitz(radius,1.,1/index**2,(center,),**args)
    projection = sim._projection_matrix(radius,(center,),nx=grid,ny=grid)
    return context.scalar(eps),context.scalar(inverse_rule),context.vector(projection)


def reference_pq(sim, eps, inverse_rule, projection, context):
    """Original full-size NVM algebra, on the same Fourier set as candidate."""
    n = eps.shape[0]
    identity = sim._eye(n)
    delta = sim._solve(inverse_rule,identity)-eps
    tensor = sim._symmetric_nvm_tensor(eps,delta,projection)
    a,b,c,d = tensor[:n,:n],tensor[:n,n:],tensor[n:,:n],tensor[n:,n:]
    inverse = sim._solve(eps,identity)
    k1 = context.scalar(sim.K1_norm)
    k2 = context.scalar(sim.K2_norm)
    s,z = sim.sin_zeta,sim.cos_zeta
    p = torch.cat((torch.cat(((k1@inverse@k2-z*identity)/s,(identity-k1@inverse@k1)/s),dim=1),
                   torch.cat(((k2@inverse@k2-identity)/s,(z*identity-k2@inverse@k1)/s),dim=1)),dim=0)
    q = torch.cat((torch.cat(((-k1@k2-s*c+z*d)/s,(k1@k1-d)/s),dim=1),
                   torch.cat(((s*s*a-s*z*(b+c)+z*z*d-k2@k2)/s,(k2@k1+s*b-z*d)/s),dim=1)),dim=0)
    if context.triangular:
        transform = torch.kron(torch.tensor([[1.,0.],[-z/s,1/s]],dtype=sim._dtype,device=sim._device),identity)
        # Use the frozen solver's actual Cartesian transform and Reynolds average.
        _,_,_,p,q,_,_,_ = sim._d6_symmetrized_cartesian_pq(p,q,transform)
    e,h = context.electric,context.magnetic
    return e.mH@p@h,h.mH@q@e,p,q


def candidate_pq(sim, eps, inverse_rule, projection, context):
    """Direct reduced P/Q via block-inverse actions and NVM anticommutator."""
    eps_inverse = BlockInverse(sim,eps,context)
    rule_inverse = BlockInverse(sim,inverse_rule,context)
    n = eps.shape[0]
    k1 = torch.diagonal(context.scalar(sim.K1_norm))[:,None]
    k2 = torch.diagonal(context.scalar(sim.K2_norm))[:,None]
    s,c = sim.sin_zeta,sim.cos_zeta

    def delta(value):
        return rule_inverse.apply(value)-eps@value

    def tensor(a,b,value):
        normal = projection[a*n:(a+1)*n,b*n:(b+1)*n]
        answer = .5*(delta(normal@value)+normal@delta(value))
        return answer+eps@value if a == b else answer

    def p_cart(value):
        h1,h2 = value[:n],c*value[:n]+s*value[n:]
        inverse = eps_inverse.apply(k2*h1-k1*h2)
        out1 = (k1*inverse-c*h1+h2)/s
        out2 = (k2*inverse-h1+c*h2)/s
        return torch.cat((out1,(-c*out1+out2)/s),dim=0)

    def q_cart(value):
        e1,e2 = value[:n],c*value[:n]+s*value[n:]
        if not context.triangular:
            dx1,dx2 = delta(e1),delta(e2)
            px = projection[:n,:n]@e1+projection[:n,n:]@e2
            py = projection[n:,:n]@e1+projection[n:,n:]@e2
            tx = eps@e1+.5*(delta(px)+projection[:n,:n]@dx1+projection[:n,n:]@dx2)
            ty = eps@e2+.5*(delta(py)+projection[n:,:n]@dx1+projection[n:,n:]@dx2)
            out1 = -k1*k2*e1+k1*k1*e2-ty
            out2 = tx-k2*k2*e1+k2*k1*e2
        else:
            t00,t01,t10,t11 = (tensor(a,b,e1) for a,b in ((0,0),(0,1),(1,0),(1,1)))
            t01_2,t11_2 = tensor(0,1,e2),tensor(1,1,e2)
            out1 = (-k1*k2*e1+k1*k1*e2-s*t10+c*t11-t11_2)/s
            out2 = (s*s*t00-s*c*(t01+t10)+c*c*t11-k2*k2*e1+k2*k1*e2+s*t01_2-c*t11_2)/s
        return torch.cat((out1,(-c*out1+out2)/s),dim=0)

    e,h = context.electric,context.magnetic
    if not context.triangular:
        p,q = e.mH@p_cart(h),h.mH@q_cart(e)
    else:
        # Project the SAME twelve-operation Reynolds average without forming
        # 2*Nstar square P/Q. This intentionally favors a transparent oracle
        # over a final optimized E1-row implementation.
        p = e.new_zeros((e.shape[1],h.shape[1]))
        q = h.new_zeros((h.shape[1],e.shape[1]))
        for entry in context.group:
            electric = entry['electric'].mH@e
            magnetic = entry['magnetic'].mH@h
            p = p+electric.mH@p_cart(magnetic)/len(context.group)
            q = q+magnetic.mH@q_cart(electric)/len(context.group)
    return p,q,{'scalar_inverse_block_sizes':eps_inverse.sizes,
                'scalar_invariance_error':max(eps_inverse.leakage,rule_inverse.leakage),
                'full_scalar_inverse_assembled':False,'full_PQ_assembled':False,
                'full_convolution_and_normal_inputs_retained':True,
                'hex_reynolds_operations':12 if context.triangular else 0}


def add_candidate_layer(sim, radius, index, depth, grid, context):
    eps,rule,projection = material_inputs(sim,radius,index,grid,context)
    p,q,diagnostics = candidate_pq(sim,eps,rule,projection,context)
    kz_squared,electric = sim._eig(p@q)
    kz = sim._positive_kz(kz_squared)
    magnetic = sim._magnetic_eigenvectors(p,q,electric,kz)
    sim._install_polarization_basis(context.full_basis(context.electric),context.full_basis(context.magnetic),1e-8)
    sim._polarized_layers.append({'electric':electric,'magnetic':magnetic,'kz':kz})
    sim.thickness.append(torch.as_tensor(depth,dtype=sim._dtype,device=sim._device))
    sim.layer_N += 1
    return diagnostics


def source(sim):
    zero_x = int(torch.nonzero(sim.order_x == 0,as_tuple=False)[0,0])
    zero_y = int(torch.nonzero(sim.order_y == 0,as_tuple=False)[0,0])
    value = torch.zeros(2*sim.order_N,dtype=sim._dtype,device=sim._device)
    value[zero_x*len(sim.order_y)+zero_y] = 1.
    return value


def flux(electric, matrix, direction):
    magnetic = direction*(matrix@electric)
    n = electric.numel()//2
    return .5*torch.real((electric[:n]*magnetic[n:].conj()-electric[n:]*magnetic[:n].conj()).sum())


def optical_response(sim, radii, index, height_nm, period_nm, grid, variant, context):
    depth = height_nm/(len(radii)*period_nm)
    for radius in radii:
        if variant == 'reference':
            sim.add_layer_circle_nvm(depth,radius,1.,index**2,(center_of(sim),),nx=grid,ny=grid)
        else:
            add_candidate_layer(sim,radius,index,depth,grid,context)
    incident = source(sim)
    transmitted,reflected = sim._solve_polarization_reduced_smatrix(incident_source=incident)
    incoming = flux(incident,sim.Vi,1.)
    r,t = -flux(reflected,sim.Vi,-1.)/incoming,flux(transmitted,sim.Vo,1.)/incoming
    return r,t,transmitted,reflected
