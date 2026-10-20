"""Full-vector 2D-periodic FFF, including Nz and mixed xz/yz constitutive blocks.

F=[Ex,Ey,Z0 Hx,Z0 Hy], dF/dz=i*k0*K*F, time convention exp(-i omega t).
CPU uses SciPy's local matrix exponential; CUDA uses torch.matrix_exp.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
import numpy as np
from scipy.linalg import expm
import torch

from .model import Geometry, Operators, harmonics

METHODS = ("rcwa-fff", "dm-fff", "slice-nvm", "direct")


class Backend:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Only cpu and cuda devices are supported")

    def array(self, value):
        return torch.as_tensor(value, dtype=torch.complex128, device=self.device)

    def eye(self, n):
        return torch.eye(n, dtype=torch.complex128, device=self.device)

    def exp(self, matrix):
        if self.device.type == "cpu":
            return self.array(expm(matrix.detach().numpy()))
        return torch.linalg.matrix_exp(matrix)

    def numpy(self, value):
        return value.detach().cpu().numpy()


def block(rows):
    return torch.cat([torch.cat(row, dim=1) for row in rows], dim=0)


@dataclass
class SMatrix:
    r: torch.Tensor
    tr: torch.Tensor
    t: torch.Tensor
    rr: torch.Tensor


def identity_s(eye):
    return SMatrix(0*eye, eye, eye, 0*eye)


def star(a, b):
    eye = torch.eye(a.r.shape[0], dtype=a.r.dtype, device=a.r.device)
    solved = torch.linalg.solve(eye-a.rr@b.r, torch.cat((a.t, a.rr@b.tr), dim=1))
    u, v = solved.chunk(2, dim=1)
    return SMatrix(a.r+a.tr@b.r@u, a.tr@(b.tr+b.r@v), b.t@u, b.rr+b.t@v)


def interface(vl, vr):
    eye = torch.eye(vl.shape[0], dtype=vl.dtype, device=vl.device)
    joined = torch.linalg.solve(vl+vr, torch.cat((vl-vr, 2*vr), dim=1))
    r, tr = joined.chunk(2, dim=1)
    return SMatrix(r, tr, eye+r, tr-eye)


def _left_j(matrix):
    first, second = matrix.chunk(2, dim=0)
    return torch.cat((-second, first), dim=0)


def _right_j(matrix):
    first, second = matrix.chunk(2, dim=1)
    return torch.cat((second, -first), dim=1)


def propagation_s(p):
    size = p.shape[0]//2
    ee, eh, he, hh = p[:size,:size], p[:size,size:], p[size:,:size], p[size:,size:]
    jhe, ehj, jhhj = _left_j(he), _right_j(eh), _left_j(_right_j(hh))
    aa = .5*(ee-jhe+ehj-jhhj)
    ab = .5*(ee-jhe-ehj+jhhj)
    ba = .5*(ee+jhe+ehj+jhhj)
    bb = .5*(ee+jhe-ehj-jhhj)
    eye = torch.eye(size, dtype=p.dtype, device=p.device)
    x, y = torch.linalg.solve(bb, torch.cat((ba, eye), dim=1)).chunk(2, dim=1)
    return SMatrix(-x, y, aa-ab@x, ab@y)


def constitutive(e, eta, projections, backend, ordering="symmetric"):
    identity = backend.eye(e.shape[0])
    delta = torch.linalg.solve(eta, identity)-e
    tensor = [[None]*3 for _ in range(3)]
    for a in range(3):
        for b in range(3):
            p = backend.array(projections[a][b])
            correction = .5*(delta@p+p@delta) if ordering == "symmetric" else delta@p
            tensor[a][b] = correction+e if a == b else correction
    return tensor


def generator_from_tensor(c, kx, ky, eye):
    """Eliminate Ez and Hz from the six Maxwell equations, preserving order."""
    xx, xy, xz = c[0]
    yx, yy, yz = c[1]
    zx, zy, zz = c[2]
    ix = torch.diag(kx); iy = torch.diag(ky)
    zx_s, zy_s, ix_s, iy_s = torch.linalg.solve(zz, torch.cat((zx,zy,ix,iy),dim=1)).chunk(4,dim=1)
    dxx, dxy = xx-xz@zx_s, xy-xz@zy_s
    dyx, dyy = yx-yz@zx_s, yy-yz@zy_s
    k = block([
        [-kx[:,None]*zx_s, -kx[:,None]*zy_s, kx[:,None]*iy_s, eye-kx[:,None]*ix_s],
        [-ky[:,None]*zx_s, -ky[:,None]*zy_s, ky[:,None]*iy_s-eye, -ky[:,None]*ix_s],
        [-torch.diag(kx*ky)-dyx, torch.diag(kx*kx)-dyy, -yz@iy_s, yz@ix_s],
        [dxx-torch.diag(ky*ky), dxy+torch.diag(ky*kx), xz@iy_s, -xz@ix_s],
    ])
    # Ez = recovery @ [Ex,Ey,hx,hy].
    recovery = torch.cat((-zx_s,-zy_s,iy_s,-ix_s),dim=1)
    return k, recovery


def uniform_v(epsilon, kx, ky, backend):
    q = torch.sqrt(backend.array(epsilon)-kx*kx-ky*ky)
    q = torch.where((q.imag < 0) | ((q.imag.abs() < 1e-14) & (q.real < 0)), -q, q)
    if bool((q.abs() < 1e-10).any()):
        raise ValueError("A retained exterior order is at a Rayleigh cutoff; shift wavelength slightly")
    return block([[torch.diag(-ky*kx/q), torch.diag(-q-ky*ky/q)],
                  [torch.diag(q+kx*kx/q), torch.diag(kx*ky/q)]]), q


def flux_orders(electric, magnetic):
    ex, ey = electric.chunk(2,dim=0); hx, hy = magnetic.chunk(2,dim=0)
    return .5*torch.real(ex.conj()*hy-ey.conj()*hx)


class MatrixFunction:
    def __init__(self, geometry, order, grid, wavelength_nm, epsilon, backend,
                 method="rcwa-fff", ordering="symmetric", homogeneous=False):
        self.geometry, self.backend, self.method = geometry, backend, method
        self.orders = harmonics(order, geometry.lattice)
        self.operators = Operators(geometry, self.orders, grid)
        self.eye = backend.eye(len(self.orders))
        g = self.orders @ geometry.reciprocal.T
        self.kx, self.ky = backend.array(g[:,0]*wavelength_nm/(2*np.pi)), backend.array(g[:,1]*wavelength_nm/(2*np.pi))
        self.epsilon, self.ordering, self.homogeneous = epsilon, ordering, homogeneous
        self.cache = {}

    def __call__(self, z):
        key = float(z)
        if key in self.cache:
            return self.cache[key]
        e, eta = (self.backend.array(x) for x in self.operators.material(z,self.epsilon,self.homogeneous))
        if self.method == "direct":
            zero = 0*e
            c = [[e,zero,zero],[zero,e,zero],[zero,zero,e]]
        else:
            projection = self.operators.projections(z, tilted=self.method != "slice-nvm")
            c = constitutive(e,eta,projection,self.backend,self.ordering)
        k, recovery = generator_from_tensor(c,self.kx,self.ky,self.eye)
        value = (k,recovery,c)
        if len(self.cache) >= 3:
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = value
        return value


def rk4(function, start, dz, k0, step_nm, eye):
    steps = max(1, math.ceil(dz/step_nm))
    step = dz/steps
    p = eye.clone()
    for j in range(steps):
        z = start+j*step
        m0 = 1j*k0*function(z)[0]
        mh = 1j*k0*function(z+.5*step)[0]
        m1 = 1j*k0*function(z+step)[0]
        a = m0@p
        b = mh@(p+.5*step*a)
        c = mh@(p+.5*step*b)
        d = m1@(p+step*c)
        p = p+step/6*(a+2*b+2*c+d)
    return p


@dataclass
class Result:
    R: float
    P_sub: float
    A_moth: float
    A_total: float
    T_far: float
    R_orders: np.ndarray
    T_orders: np.ndarray
    metadata: dict
    S: SMatrix
    V_in: torch.Tensor
    V_out: torch.Tensor
    zero: int
    absorption_volume: float | None = None
    absorption_error: float | None = None


@torch.no_grad()
def solve(geometry: Geometry, wavelength_nm: float, epsilon: complex, *, order=2,
          slices=60, grid=128, method="rcwa-fff", device="cpu", polarization="x",
          ordering="symmetric", dm_step_nm=.5, n_in=1., epsilon_out=None,
          homogeneous=False, volume_absorption=False, quadrature=5, loss_diagnostics=False):
    started = time.perf_counter()
    if method not in METHODS or ordering not in ("symmetric","paper-left"):
        raise ValueError("Unknown method or factorization ordering")
    if not math.isfinite(wavelength_nm) or wavelength_nm <= 0 or slices < 1 or dm_step_nm <= 0:
        raise ValueError("Require positive wavelength, slices, and DM step")
    if n_in <= 0 or not math.isfinite(n_in) or not np.isfinite(epsilon) or epsilon.imag < 0:
        raise ValueError("Require a real positive input index and passive finite epsilon")
    if polarization not in ("x","y"):
        raise ValueError("Polarization must be x or y; normal incidence is used")
    backend = Backend(device)
    if backend.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(backend.device)
    function = MatrixFunction(geometry,order,grid,wavelength_nm,epsilon,backend,method,ordering,homogeneous)
    count = len(function.orders)
    eye = backend.eye(count)
    eye2, eye4 = backend.eye(2*count), backend.eye(4*count)
    j = block([[0*eye,-eye],[eye,0*eye]])
    eps_out = epsilon if epsilon_out is None else complex(epsilon_out)
    vi, _ = uniform_v(n_in*n_in,function.kx,function.ky,backend)
    vo, _ = uniform_v(eps_out,function.kx,function.ky,backend)
    sections = [(i*geometry.height_nm/slices,(i+1)*geometry.height_nm/slices) for i in range(slices)]
    blocks = [interface(vi,j)]
    total = blocks[0]
    k0 = 2*np.pi/wavelength_nm
    # Keep each user's section and its frozen midpoint geometry. Subdivide only
    # the local exponential/cascade to avoid growing evanescent transfer entries.
    max_g = np.linalg.norm(function.orders@geometry.reciprocal.T,axis=1).max()
    growth_bound = max_g+k0*max(abs(np.sqrt(complex(epsilon))),n_in,abs(np.sqrt(eps_out)))
    local_splits = 0
    for a,b in sections:
        dz = b-a
        if method == "dm-fff":
            p = rk4(function,a,dz,k0,dm_step_nm,eye4)
            layer_s = propagation_s(p)
        else:
            k = function(.5*(a+b))[0]
            splits = max(1,math.ceil(growth_bound*dz/2.))
            local_splits += splits
            thin = propagation_s(backend.exp(1j*k0*dz/splits*k))
            layer_s = identity_s(eye2)
            for _ in range(splits):
                layer_s = star(layer_s,thin)
        total = star(total,layer_s)
        if volume_absorption:
            blocks.append(layer_s)
    exit_s = interface(j,vo)
    total = star(total,exit_s)
    zero = int(np.flatnonzero((function.orders == 0).all(axis=1))[0])
    incident = backend.array(np.zeros((2*count,1),complex))
    incident[zero+(count if polarization == "y" else 0)] = 1
    incoming = float(flux_orders(incident,vi@incident).sum())
    reflected, transmitted = total.r@incident, total.t@incident
    ro = backend.numpy(flux_orders(reflected,vi@reflected)/incoming).reshape(-1)
    to = backend.numpy(flux_orders(transmitted,vo@transmitted)/incoming).reshape(-1)
    r, p_sub = float(ro.sum()),float(to.sum())
    if not np.isfinite([r,p_sub]).all():
        raise FloatingPointError("Non-finite scattering power")
    metadata = dict(method=method,order=int(order),scalar_harmonics=count,field_dimension=4*count,
                    slices=slices,grid=grid,wavelength_nm=wavelength_nm,epsilon_real=epsilon.real,
                    epsilon_imag=epsilon.imag,ordering=ordering,polarization=polarization,
                    dm_step_nm=dm_step_nm,device=str(backend.device),dtype="complex128",
                    elapsed_s=time.perf_counter()-started,local_exponential_subsections=local_splits,
                    harmonic_shape="D6-closed hexagon" if geometry.lattice=="triangular" else "square",
                    peak_cuda_bytes=torch.cuda.max_memory_allocated(backend.device) if backend.device.type=="cuda" else None)
    t_far = 0. if eps_out.imag > 1e-12 else p_sub
    result = Result(r,p_sub,1-r-p_sub,1-r-t_far,t_far,ro,to,metadata,total,vi,vo,zero)
    if loss_diagnostics:
        samples = []
        for fraction in (.1,.5,.9):
            tensor = block(function(geometry.height_nm*fraction)[2])
            loss = (tensor-tensor.mH)/(2j)
            minimum = float(torch.linalg.eigvalsh(loss).min())
            samples.append(dict(z_nm=geometry.height_nm*fraction,loss_min_eigenvalue=minimum))
        result.metadata['constitutive_loss_samples'] = samples
        result.metadata['constitutive_loss_min_eigenvalue'] = min(x['loss_min_eigenvalue'] for x in samples)
    if volume_absorption:
        blocks.append(exit_s)
        suffixes = [None]*len(blocks)
        suffix = identity_s(eye2)
        for i in range(len(blocks)-1,-1,-1):
            suffixes[i] = suffix
            suffix = star(blocks[i],suffix)
        prefix = blocks[0]
        nodes,weights = np.polynomial.legendre.leggauss(quadrature)
        absorbed = 0.
        boundary_residual = 0.
        for index,(a,b) in enumerate(sections):
            right = suffixes[index]
            forward = torch.linalg.solve(eye2-prefix.rr@right.r,prefix.t@incident)
            backward = right.r@forward
            state = torch.cat((forward+backward,j@(forward-backward)),dim=0)
            dz = b-a
            frozen = function(.5*(a+b)) if method != "dm-fff" else None
            for node,weight in zip(nodes,weights):
                offset = dz*.5*(node+1)
                if frozen is None:
                    p = rk4(function,a,offset,k0,dm_step_nm,eye4)
                    _,recovery,c = function(a+offset)
                else:
                    k,recovery,c = frozen
                    p = backend.exp(1j*k0*offset*k)
                f = p@state
                electric = torch.cat((f[:2*count],recovery@f),dim=0)
                displacement = block(c)@electric
                density = float(torch.imag((electric.conj()*displacement).sum()))
                absorbed += k0/(2*incoming)*dz*.5*float(weight)*density
            if index == 0:
                expected = torch.cat((incident+reflected,vi@(incident-reflected)),dim=0)
                boundary_residual = float((state-expected).abs().max())
            prefix = star(prefix,blocks[index+1])
        result.absorption_volume = absorbed
        result.absorption_error = abs(absorbed-result.A_moth)
        if not np.isfinite([result.absorption_volume,result.absorption_error]).all():
            raise FloatingPointError("Non-finite independently integrated absorption")
        result.metadata['input_boundary_residual'] = boundary_residual
    result.metadata['elapsed_s'] = time.perf_counter()-started
    return result
