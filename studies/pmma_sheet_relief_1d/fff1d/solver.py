"""Paper Eqs. (4)-(11), ordered products, frozen-section RCWA-FFF.

Local exponentials are equivalent to diagonalizing the paper's constant section
generator. Only short local transfers are formed; global propagation uses S matrices.
"""
import math
import numpy as np
import torch
from scipy.linalg import expm,solve as scipy_solve
from fast_solver import canonical_device
from fff1d.operators import get_fff_plan,angle_parameters


class Backend:
    def __init__(self,device='cpu'):
        self.device=canonical_device(device)
    def array(self,a):return torch.as_tensor(a,dtype=torch.complex128,device=self.device)
    def eye(self,n):return torch.eye(n,dtype=torch.complex128,device=self.device)
    def solve(self,a,b):
        if self.device.type=='cpu':return self.array(scipy_solve(a.detach().numpy(),b.detach().numpy()))
        return torch.linalg.solve(a,b)
    def exp(self,a):
        if self.device.type=='cpu':return self.array(expm(a.detach().numpy()))
        return torch.linalg.matrix_exp(a)


def star(a,b,backend):
    tf,rf,rb,tb=a;uf,vf,vb,ub=b;n=tf.shape[0]
    x=backend.solve(backend.eye(n)-rb@vf,torch.cat((tf,rb@ub),dim=1))
    y=backend.solve(backend.eye(n)-vf@rb,torch.cat((vf@tf,ub),dim=1))
    return [uf@x[:,:n],rf+tb@y[:,:n],vb+uf@x[:,n:],tb@y[:,n:]]


def interface(yl,yr):
    diag=torch.diag
    return [diag(2*yl/(yl+yr)),diag((yl-yr)/(yl+yr)),diag((yr-yl)/(yl+yr)),diag(2*yr/(yl+yr))]


def identity_s(n,backend):
    i=backend.eye(n);z=torch.zeros_like(i)
    return [i,z,z,i]


def power_s(s,n,backend):
    result=identity_s(s[0].shape[0],backend)
    while n:
        if n&1:result=star(result,s,backend)
        n//=2
        if n:s=star(s,s,backend)
    return result


def amplitude_generator(g,y):
    n=len(y);a,b,c,d=g[:n,:n],g[:n,n:],g[n:,:n],g[n:,n:]
    plus=a+b*y[None,:];minus=a-b*y[None,:]
    hplus=(c+d*y[None,:])/y[:,None];hminus=(c-d*y[None,:])/y[:,None]
    return torch.cat((torch.cat(((plus+hplus)/2,(minus+hminus)/2),1),
                      torch.cat(((plus-hplus)/2,(minus-hminus)/2),1)),0)


def local_s(g,h,wavelength,y,backend,transfer_norm=4.):
    amplitude=amplitude_generator(g,y)
    norm=float(torch.linalg.matrix_norm(amplitude,ord=float('inf')).item())
    if not math.isfinite(norm):raise RuntimeError('Nonfinite RCWA-FFF section generator.')
    length=2*math.pi*h/wavelength
    subdivisions=max(1,math.ceil(length*norm/transfer_norm))
    transfer=backend.exp(1j*(length/subdivisions)*amplitude)
    if not bool(torch.all(torch.isfinite(transfer))):raise RuntimeError('Nonfinite local RCWA-FFF exponential.')
    n=len(y);a,b,c,d=transfer[:n,:n],transfer[:n,n:],transfer[n:,:n],transfer[n:,n:]
    x,z=backend.solve(d,torch.cat((c,backend.eye(n)),1)).chunk(2,dim=1)
    section=[a-b@x,-x,b@z,z]
    return (power_s(section,subdivisions,backend) if subdivisions>1 else section),subdivisions


def generator(e,eta,plan,wavelength,c2,cs,backend):
    """Even Ex/Hy sector; all odd intermediates required by Kx/Q2 are retained."""
    E,O=plan.base.blocks(e);etae,etao=plan.base.blocks(eta)
    n=plan.order+1;I=backend.eye(n);Io=backend.eye(plan.order)
    V=backend.solve(etae,I);Vo=backend.solve(etao,Io)
    D,Do=E-V,O-Vo
    Cxx=V+c2*D;Czz=O-c2*Do
    P=cs*plan.cross
    Cxz=D@P;Czx=Do@P.conj().T
    k=torch.arange(n,dtype=torch.float64,device=backend.device)*wavelength/plan.period
    K=torch.zeros((n,plan.order),dtype=torch.complex128,device=backend.device)
    K[1:]=torch.diag(k[1:].to(torch.complex128))
    zx,kk=backend.solve(Czz,torch.cat((Czx,K.T),dim=1)).chunk(2,dim=1)
    return torch.cat((torch.cat((-K@zx,I-K@kk),1),
                      torch.cat((Cxx-Cxz@zx,-Cxz@kk),1)),0)


def full_generator(E,eta,Cc2,Ccs,k,backend):
    """Unreduced paper Eq. (7), also supporting nonconstant normal fields."""
    I=backend.eye(E.shape[0]);V=backend.solve(eta,I);delta=E-V
    Q1=delta@Cc2;Q2=delta@Ccs
    Czz=E-Q1;Cxx=V+Q1;K=torch.diag(k.to(torch.complex128))
    zx,kk=backend.solve(Czz,torch.cat((Q2,K),1)).chunk(2,dim=1)
    return torch.cat((torch.cat((-K@zx,I-K@kk),1),
                      torch.cat((Cxx-Q2@zx,-Q2@kk),1)),0)


def admittance(epsilon,k,backend,polarization='p'):
    q=torch.sqrt(backend.array(epsilon)-k*k)
    q=torch.where((q.imag<0)|((q.imag.abs()<=1e-14)&(q.real<0)),-q,q)
    q=torch.where(q.abs()<1e-12,q+(1e-10+1e-10j),q)
    return epsilon/q if polarization=='p' else q


@torch.no_grad()
def local_surface_fff(g,stack,wavelength,n_pmma,n_gold,void,order,device='cpu',
        theta_deg=None,gil_nm=0.,gil_interfaces='both',gil_quadrature=16,
        transfer_norm=4.,return_diagnostics=False,polarization='p',symmetry='even'):
    if polarization not in ('s','p'):raise ValueError('Select s or p.')
    if symmetry not in ('even','full'):raise ValueError('Symmetry must be even or full.')
    if not math.isfinite(transfer_norm) or transfer_norm<=0:raise ValueError('Positive finite --transfer-norm required.')
    backend=Backend(device)
    plan=get_fff_plan(g,stack,order,backend.device,gil_nm,gil_interfaces,gil_quadrature)
    angle,c2,cs=angle_parameters(g,theta_deg)
    kept=np.arange(order+1) if symmetry=='even' else plan.base.orders
    k=torch.as_tensor(kept,dtype=torch.float64,device=backend.device)*wavelength/g['period_nm']
    if symmetry=='full':
        differences=plan.base.orders[:,None]-plan.base.orders[None,:]
        diff=torch.as_tensor(abs(differences),dtype=torch.int64,device=backend.device)
        safe=np.where(differences==0,1,differences)
        sign=np.where(differences==0,0,(1-(-1.)**differences)/(1j*np.pi*safe))
        Ccs=cs*backend.array(sign);Cc2=c2*backend.eye(2*order+1)
    yf=admittance(2.,k,backend,polarization)
    cascade=interface(admittance(n_pmma**2,k,backend,polarization),yf)
    substeps=[]
    for j,row in enumerate(stack):
        e,eta=plan.material(j,n_pmma,n_gold,void)
        if symmetry=='full':
            E,Et=e[diff],eta[diff]
            if polarization=='p':G=full_generator(E,Et,Cc2,Ccs,k,backend)
            else:
                I=backend.eye(2*order+1);Z=torch.zeros_like(I)
                G=torch.cat((torch.cat((Z,I),1),torch.cat((E-torch.diag(k*k),Z),1)),0)
        elif polarization=='p':
            G=generator(e,eta,plan,wavelength,c2,cs,backend)
        else:
            E,_=plan.base.blocks(e);I=backend.eye(order+1);Z=torch.zeros_like(I)
            G=torch.cat((torch.cat((Z,I),1),torch.cat((E-torch.diag(k*k),Z),1)),0)
        section,count=local_s(G,row['z_bottom_nm']-row['z_top_nm'],wavelength,yf,backend,transfer_norm)
        cascade=star(cascade,section,backend);substeps.append(count)
    cascade=star(cascade,interface(yf,admittance(1.333**2,k,backend,polarization)),backend)
    rf=cascade[1].detach().cpu().numpy();tf=cascade[0].detach().cpu().numpy()
    if symmetry=='even':
        u=plan.base.basis;rf=u@rf@u.T;tf=u@tf@u.T
    if not np.all(np.isfinite(rf)) or not np.all(np.isfinite(tf)):
        raise RuntimeError('Nonfinite RCWA-FFF scattering response.')
    result=(plan.base.orders.copy(),rf[None,:,:],tf[None,:,:])
    if return_diagnostics:
        return (*result,dict(method='rcwa-fff-paper-left',theta_deg=angle,
            generator_dimension=2*len(kept),symmetry=symmetry,actual_layers=len(stack),max_local_subsections=max(substeps),
            total_local_subsections=sum(substeps),gil_nm=gil_nm,gil_interfaces=gil_interfaces,
            gil_quadrature=gil_quadrature,transfer_norm=transfer_norm))
    return result
