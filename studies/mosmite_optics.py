"""Analytic coherent/incoherent optics at normal incidence; no phase quadrature.

The scalar limit is the intensity balance of Byrnes arXiv:1603.02720 Sec6.2,
Eqs25--30. Coherent Jones blocks come from RCWA or analytic uniform layers.
Incoherent multiple reflections are summed with a linear system in optical
coherency space. This keeps polarization coherence within a beam, including
cross polarization, while removing interference between distinct path passes.
External input/output media are air. Internal amplitude ports use Re(n) flux
normalization; T is never replaced by1-R for an absorbing incident medium.
"""
from __future__ import annotations
import math
import numpy as np

THEORY={
    'method':'analytic-intensity-scattering-cascade',
    'implementation':'Jones coherent blocks and coherency-space intensity balance; no phase samples or wavelength smoothing',
    'references':[
        'https://arxiv.org/html/1603.02720v5#S6.SS2',
        'https://doi.org/10.1364/AO.44.007532',
        'https://doi.org/10.1364/AO.41.003978'],
    'assumption':'Each designated incoherent layer loses path phase information. This is not inferred from its thickness alone.'}

def _matrix(value):
    x=np.asarray(value,dtype=np.complex128)
    if not np.isfinite(x).all():raise ValueError('Nonfinite Jones amplitude.')
    if x.ndim==0:return np.eye(2,dtype=np.complex128)*x
    if x.shape!=(2,2):raise ValueError('Expected a finite scalar or2x2 Jones matrix.')
    return x

def identity(size=2):
    e=np.eye(size,dtype=np.complex128);z=np.zeros_like(e)
    return e,z,z,e

def cascade(left,right):
    """Connect two amplitude or intensity-map scattering blocks, (Tf,Rf,Rb,Tb).

    X=Tf_left*input+Rb_left*Y; Y=Rf_right*X+Tb_right*input_right.
    Solve these equations exactly; the solve sums all repeated round trips.
    """
    at,ar,ab,az=left;bt,br,bb,bz=right;e=np.eye(at.shape[0],dtype=np.complex128)
    forward=np.linalg.solve(e-ab@br,at)
    reverse=np.linalg.solve(e-br@ab,bz)
    return bt@forward, ar+az@br@forward, bb+bt@ab@reverse, az@reverse

def interface(n_left,n_right):
    q=n_left+n_right
    return tuple(_matrix(v) for v in (2*n_left/q,(n_left-n_right)/q,(n_right-n_left)/q,2*n_right/q))

def propagation(index,thickness_nm,wl):
    p=np.exp(2j*np.pi*index*thickness_nm/wl);z=np.zeros((2,2),dtype=np.complex128)
    return p*np.eye(2),z,z,p*np.eye(2)

def _map(jones):
    # vec(J C J^H)=(J* tensor J) vec(C), with column-major vectorization.
    return np.kron(jones.conj(),jones)

def power_block(amplitudes,n_left,n_right):
    left,right=float(complex(n_left).real),float(complex(n_right).real)
    if min(left,right)<=0:raise ValueError('Scalar ports must have positive real admittance at normal incidence.')
    tf,rf,rb,tb=amplitudes
    return (_map(tf*math.sqrt(right/left)),_map(rf),_map(rb),_map(tb*math.sqrt(left/right)))

def _layers(layers,wl):
    if not math.isfinite(float(wl)) or wl<=0:raise ValueError('Wavelength must be positive and finite.')
    result=[]
    for n,d,inc in layers:
        n=complex(n);d=float(d)
        if not all(math.isfinite(v) for v in (n.real,n.imag,d)) or n.real<=0 or n.imag<0 or d<0:
            raise ValueError('Invalid passive layer index or thickness.')
        # An absent layer has no interface or coherence-breaking operation.
        if d>0:result.append((n,d,bool(inc)))
    return result

def _result(block):
    tf,rf,_,_=block
    powers=[]
    for j in range(2):
        incident=np.zeros((2,2),dtype=np.complex128);incident[j,j]=1.
        reflected=(rf@incident.reshape(4,order='F')).reshape((2,2),order='F')
        transmitted=(tf@incident.reshape(4,order='F')).reshape((2,2),order='F')
        powers.append((float(np.trace(reflected).real),float(np.trace(transmitted).real),
                       float(reflected[1-j,1-j].real),float(transmitted[1-j,1-j].real)))
    rx,tx,cx,dx=powers[0];ry,ty,cy,dy=powers[1]
    return dict(R=(rx+ry)/2,T=(tx+ty)/2,A=1-(rx+ry+tx+ty)/2,
                R_x=rx,T_x=tx,R_y=ry,T_y=ty,cross_R=(cx+cy)/2,cross_T=(dx+dy)/2)

def finite_sample(surface,port_index,layers,wl,*,coherence='incoherent'):
    """Return R,T,A and Jones x/y diagnostics, with air on both external sides.

    layers=(complex n,thickness_nm,incoherent_bool), ordered front-to-back.
    False flags allow coherent thin interlayers in an otherwise incoherent stack.
    coherence='coherent' retains the fixed optical phase in every layer.
    """
    if coherence not in ('incoherent','coherent'):raise ValueError('Use incoherent or coherent; phase averaging has been removed.')
    layers=_layers(layers,wl);n_current=complex(port_index)
    if not np.isfinite(n_current) or n_current.real<=0 or n_current.imag<0:raise ValueError('Invalid substrate port.')
    amplitude=tuple(_matrix(v) for v in surface);flux=identity(4);n_left=1.+0j
    for index,d,incoherent in layers:
        amplitude=cascade(amplitude,interface(n_current,index));n_current=index
        if coherence=='incoherent' and incoherent:
            flux=cascade(flux,power_block(amplitude,n_left,index))
            attenuation=math.exp(-4*math.pi*index.imag*d/wl)
            e=np.eye(4,dtype=np.complex128);z=np.zeros_like(e)
            flux=cascade(flux,(attenuation*e,z,z,attenuation*e))
            amplitude=identity(2);n_left=index
        else:amplitude=cascade(amplitude,propagation(index,d,wl))
    amplitude=cascade(amplitude,interface(n_current,1.+0j))
    flux=cascade(flux,power_block(amplitude,n_left,1.+0j))
    return _result(flux)
