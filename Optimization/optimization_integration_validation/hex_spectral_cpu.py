"""Proof of constant-index geometry/PQ reuse across wavelengths.

The published D6 v1 remains the reference. No production code is changed.
Only the scalar objective and its first derivative are supported here.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import time
import torch
import d6_nvm as d6
from rcwa_solver_auto import GroupTheoryOptions,Lattice,NVMOptions,OutputSpec
from rcwa_ext.reduced import _ReducedScatteringMixin
from rcwa_ext.scattering import _StableLinearAlgebraMixin


def make_sim(args,wave):
    sim = d6.D6FastRCWA(freq=args.period/wave,order=[args.order]*2,lattice=Lattice.triangular(1.),
        nvm=NVMOptions(grid=(args.grid,args.grid),use_lanczos=False),
        outputs=OutputSpec(smatrix_size='half',fields='none'),cascade='redheffer',
        group_theory=GroupTheoryOptions(enabled=True,symmetry='d6',polarization='x',strict=True),
        dtype=torch.complex128,device=args.device,verify_cascade=False)
    sim.add_input_layer(eps=1.)
    sim.add_output_layer(eps=args.index**2)
    sim.set_incident_angle(0.,0.)
    return sim


def source(sim):
    x = int(torch.nonzero(sim.order_x==0,as_tuple=False)[0,0])
    y = int(torch.nonzero(sim.order_y==0,as_tuple=False)[0,0])
    result = torch.zeros(2*sim.order_N,dtype=sim._dtype,device=sim._device)
    result[x*len(sim.order_y)+y] = 1.
    return result


def flux(electric,matrix,direction):
    magnetic = direction*(matrix@electric)
    n = electric.numel()//2
    return .5*torch.real((electric[:n]*magnetic[n:].conj()-electric[n:]*magnetic[:n].conj()).sum())


def sync(args):
    if args.device.startswith('cuda'):
        torch.cuda.synchronize(args.device)


class CompactPort(_ReducedScatteringMixin,_StableLinearAlgebraMixin):
    """Frozen small boundary matrices; material-independent within a step."""
    def __init__(self,sim,plan,depth,layers):
        self._dtype,self._device = sim._dtype,sim._device
        self.stable_eig_grad = getattr(sim,'stable_eig_grad',False)
        self.omega = sim.omega
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


def free_coefficients(unit,plan):
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


def current(args,waves,radii):
    start = time.perf_counter()
    gradient = torch.zeros_like(radii)
    spectrum = []
    stages = {'initialization':0.,'layers':0.,'cascade_and_flux':0.,'backward':0.}
    for wave in waves:
        begin = time.perf_counter()
        sim = make_sim(args,wave)
        stages['initialization'] += time.perf_counter()-begin
        values = radii.detach().clone().requires_grad_(True)
        begin = time.perf_counter()
        for radius in values:
            sim.add_layer_circle_nvm(args.height/(args.layers*args.period),radius,1.,args.index**2,(d6.center_of(sim),),nx=args.grid,ny=args.grid)
        stages['layers'] += time.perf_counter()-begin
        begin = time.perf_counter()
        incident = source(sim)
        transmitted,reflected = sim._solve_polarization_reduced_smatrix(incident_source=incident)
        incoming = flux(incident,sim.Vi,1.)
        R,T = -flux(reflected,sim.Vi,-1.)/incoming,flux(transmitted,sim.Vo,1.)/incoming
        stages['cascade_and_flux'] += time.perf_counter()-begin
        begin = time.perf_counter()
        gradient += torch.autograd.grad(R/len(waves),values)[0]
        stages['backward'] += time.perf_counter()-begin
        spectrum.append([wave,float(R.detach()),float(T.detach())])
        del sim,values,R,T,transmitted,reflected
    sync(args)
    return {'seconds':time.perf_counter()-start,'gradient':gradient.cpu().tolist(),'spectrum':spectrum,'stages_seconds':stages}


def reused(args,waves,radii,template,plan,ports,unit,p0,q2):
    values = radii.detach().clone().requires_grad_(True)
    start = time.perf_counter()
    p_coeff,q_coeff = [],[]
    for radius in values:
        inputs = d6.native_inputs(template,radius,args.index,args.grid,plan)
        p1,q1,_ = d6.reduced_pq(unit,*inputs,plan)
        p_coeff.append(p1-p0)
        q_coeff.append(q1-q2)
    prepared_at = time.perf_counter()
    originals = tuple(p_coeff+q_coeff)
    leaves = tuple(value.detach().requires_grad_(True) for value in originals)
    accumulators = [torch.zeros_like(value) for value in leaves]
    spectrum = []
    for wave,port in zip(waves,ports):
        R,T = port.response(leaves[:args.layers],leaves[args.layers:],p0,q2,(wave/args.period)**2)
        derivatives = torch.autograd.grad(R/len(waves),leaves)
        for target,derivative in zip(accumulators,derivatives):
            target.add_(derivative)
        spectrum.append([wave,float(R.detach()),float(T.detach())])
        del R,T,derivatives
    spectra_at = time.perf_counter()
    gradient = torch.autograd.grad(originals,values,grad_outputs=tuple(accumulators))[0]
    sync(args)
    ended = time.perf_counter()
    return {'seconds':ended-start,'gradient':gradient.cpu().tolist(),'spectrum':spectrum,
        'stages_seconds':{'geometry_coefficient_preparation':prepared_at-start,
                         'all_wavelengths_and_adjoint':spectra_at-prepared_at,
                         'one_geometry_vjp':ended-spectra_at},
        'geometry_builds':args.layers,'geometry_builds_current':args.layers*len(waves),
        'spectral_graphs_retained_at_once':1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--order',type=int,default=8)
    parser.add_argument('--grid',type=int,default=512)
    parser.add_argument('--layers',type=int,default=3)
    parser.add_argument('--index',type=float,default=2.5)
    parser.add_argument('--period',type=float,default=200.)
    parser.add_argument('--height',type=float,default=500.)
    parser.add_argument('--wavelengths',default='400,550,700')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--output',type=Path,default=Path('results/report.json'))
    args = parser.parse_args()
    if args.device.startswith('cuda'):
        parser.error('This proof script is restricted to CPU; GPU profiling is separate work.')
    torch.set_num_threads(1)
    # Initialize PyTorch's vector-Jacobian-product support before timing.
    # Some installs lazily import symbolic-shape dependencies on first use.
    probe = torch.ones(1,dtype=torch.float64,requires_grad=True)
    torch.autograd.grad(probe.square(),probe,grad_outputs=torch.ones_like(probe))
    if ':' in args.wavelengths:
        first,last,step = map(float,args.wavelengths.split(':'))
        waves = [first+i*step for i in range(int(round((last-first)/step))+1)]
    else:
        waves = list(map(float,args.wavelengths.split(',')))
    template = make_sim(args,waves[0])
    plan = d6.get_plan(template)
    unit = copy.copy(template)
    factor = torch.real(template.freq)
    unit.Kx_norm_dn = template.Kx_norm_dn*factor
    unit.Ky_norm_dn = template.Ky_norm_dn*factor
    with torch.no_grad():
        p0,q2 = free_coefficients(unit,plan)
    startup = time.perf_counter()
    ports = []
    for wave in waves:
        sim = make_sim(args,wave)
        ports.append(CompactPort(sim,plan,args.height/(args.layers*args.period),args.layers))
        del sim
    port_preparation_seconds = time.perf_counter()-startup
    radii = torch.linspace(.10,.39,args.layers,dtype=torch.float64,device=args.device)
    reference = current(args,waves,radii)
    candidate = reused(args,waves,radii,template,plan,ports,unit,p0,q2)
    original_spectrum = torch.tensor(reference['spectrum'],dtype=torch.float64)
    actual_spectrum = torch.tensor(candidate['spectrum'],dtype=torch.float64)
    original_gradient = torch.tensor(reference['gradient'],dtype=torch.float64)
    actual_gradient = torch.tensor(candidate['gradient'],dtype=torch.float64)
    errors = {'power_max_abs':float(torch.max(torch.abs(actual_spectrum[:,1:]-original_spectrum[:,1:]))),
              'gradient_max_abs':float(torch.max(torch.abs(actual_gradient-original_gradient))),
              'energy_max_abs':float(torch.max(torch.abs(actual_spectrum[:,1:].sum(1)-1)))}
    passed = (errors['power_max_abs']<1e-8 and errors['energy_max_abs']<1e-6
              and torch.allclose(actual_gradient,original_gradient,atol=1e-7,rtol=2e-5))
    result = {'status':'passed' if passed else 'failed','scope':'constant index; hexagonal; arithmetic objective; CPU first-derivative proof only',
        'timing_repetitions':1,'autograd_vjp_initialization_excluded':True,
        'order':args.order,'grid':args.grid,'layers':args.layers,'index':args.index,'wavelengths':waves,
        'python':platform.python_version(),'torch':str(torch.__version__),'device':args.device,
        'reference':reference,'candidate':candidate,'errors':errors,
        'warm_speedup':reference['seconds']/candidate['seconds'],
        'port_preparation_seconds':port_preparation_seconds,
        'cold_speedup_excluding_common_basis':reference['seconds']/(candidate['seconds']+port_preparation_seconds),
        'production_code_modified':False,
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','order','grid','layers','errors','warm_speedup','cold_speedup_excluding_common_basis')},indent=2))
    return 0 if passed else 1


if __name__=='__main__':
    raise SystemExit(main())
