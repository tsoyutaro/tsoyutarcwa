"""Validate geometry-only sharing with tabulated PMMA dispersion.

At every wavelength both the structured layers AND the substrate use n(lambda).
The material matrices, their inverses and P/Q are recomputed per wavelength.
Only the disk's Fourier geometry and the sampled normal projector are shared.
"""
from __future__ import annotations
import argparse
import bisect
import copy
import csv
import hashlib
import json
from pathlib import Path
import platform
import time
import torch
import benchmark as common
import d6_nvm as d6

HERE = Path(__file__).resolve().parent


def samples_from(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames!=['wl','n']:
            raise ValueError('PMMA CSV must have wl,n columns; wavelength in micrometres.')
        samples = [(1000*float(row['wl']),float(row['n'])) for row in reader]
    if len(samples)<2 or any(b[0]<=a[0] for a,b in zip(samples,samples[1:])):
        raise ValueError('Wavelength samples must be strictly increasing.')
    return samples


def index_at(samples,wave):
    if wave<400 or wave>samples[-1][0]:
        raise ValueError('Wavelength outside allowed PMMA interval.')
    j = bisect.bisect_left([row[0] for row in samples],wave)
    if j==0:
        if samples[0][0]>405:
            raise ValueError('Only the existing first-segment extrapolation down to 400 nm is allowed.')
        a,b = samples[:2]
    elif j<len(samples) and samples[j][0]==wave:
        return samples[j][1]
    else:
        a,b = samples[j-1:j+1]
    return a[1]+(wave-a[0])*(b[1]-a[1])/(b[0]-a[0])


def wave_args(args,index):
    result = copy.copy(args)
    result.index = index
    return result


def current(args,waves,indices,radii):
    begin = time.perf_counter()
    spectrum = []
    gradient = torch.zeros_like(radii)
    for wave,index in zip(waves,indices):
        value = common.current(wave_args(args,index),[wave],radii)
        spectrum.extend(value['spectrum'])
        gradient += torch.tensor(value['gradient'],dtype=radii.dtype,device=radii.device)/len(waves)
    return {'seconds':time.perf_counter()-begin,'spectrum':spectrum,'gradient':gradient.cpu().tolist()}


def reused(args,waves,indices,radii,template,plan,ports,gradient=True):
    started = time.perf_counter()
    values = radii.detach().clone().requires_grad_(gradient)
    originals = []
    # Dummy contrast is only a way of obtaining the material-free disk Fourier
    # geometry. It is never used as PMMA permittivity or as the substrate index.
    identity = template._eye(plan.scalar_dimension)
    for radius in values:
        temporary_eps,_,normal = d6.native_inputs(template,radius,2.,args.grid,plan)
        disk = (temporary_eps-identity)/3.
        originals.extend((disk,normal))
    prepared = time.perf_counter()
    leaves = tuple(value.detach().requires_grad_(gradient) for value in originals)
    accumulated = [torch.zeros_like(value) for value in leaves] if gradient else None
    spectrum = []
    for wave,index,port in zip(waves,indices,ports):
        p_values,q_values = [],[]
        for layer in range(args.layers):
            disk,normal = leaves[2*layer:2*layer+2]
            eps = identity+(index**2-1)*disk
            rule = identity+(1/index**2-1)*disk
            p,q,_ = d6.reduced_pq(port,eps,rule,normal,plan)
            p_values.append(p); q_values.append(q)
        zero = port._eye(plan.reduced_dimension)*0
        R,T = port.response(p_values,q_values,zero,zero,1.)
        if gradient:
            derivatives = torch.autograd.grad(R/len(waves),leaves)
            for target,derivative in zip(accumulated,derivatives):
                target.add_(derivative)
        spectrum.append([wave,float(R.detach()),float(T.detach())])
        del p_values,q_values,R,T
    spectral_done = time.perf_counter()
    derivative = torch.autograd.grad(tuple(originals),values,grad_outputs=tuple(accumulated))[0] if gradient else None
    ended = time.perf_counter()
    return {'seconds':ended-started,'spectrum':spectrum,
        'gradient':derivative.cpu().tolist() if derivative is not None else None,
        'stages_seconds':{'geometry_preparation':prepared-started,
                         'all_dispersive_wavelengths_and_adjoint':spectral_done-prepared,
                         'geometry_vjp':ended-spectral_done},
        'geometry_builds':args.layers,'material_matrix_sets':args.layers*len(waves),
        'material_inverse_or_PQ_shared_across_wavelengths':False,
        'spectral_graphs_retained_at_once':1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--order',type=int,default=8)
    parser.add_argument('--grid',type=int,default=512)
    parser.add_argument('--layers',type=int,default=5)
    parser.add_argument('--period',type=float,default=200.)
    parser.add_argument('--height',type=float,default=500.)
    parser.add_argument('--wavelengths',default='400:700:10')
    parser.add_argument('--device',choices=['cpu'],default='cpu')
    parser.add_argument('--csv',type=Path,default=HERE/'data/Szczurowski.csv')
    parser.add_argument('--check-fd',action='store_true')
    parser.add_argument('--output',type=Path,default=Path('results/pmma_report.json'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    probe = torch.ones(1,dtype=torch.float64,requires_grad=True)
    torch.autograd.grad(probe.square(),probe,grad_outputs=torch.ones_like(probe))
    if ':' in args.wavelengths:
        first,last,step = map(float,args.wavelengths.split(':'))
        waves = [first+i*step for i in range(int(round((last-first)/step))+1)]
    else:
        waves = list(map(float,args.wavelengths.split(',')))
    samples = samples_from(args.csv)
    indices = [index_at(samples,wave) for wave in waves]
    if max(indices)-min(indices)<1e-6:
        parser.error('Dispersion validation requires wavelengths with different tabulated indices.')
    template = common.make_sim(wave_args(args,indices[0]),waves[0])
    plan = d6.get_plan(template)
    start = time.perf_counter()
    ports = []
    for wave,index in zip(waves,indices):
        sim = common.make_sim(wave_args(args,index),wave)
        port = common.CompactPort(sim,plan,args.height/(args.layers*args.period),args.layers)
        # All scalar material algebra is performed at the correct wavevector.
        port.Kx_norm_dn = sim.Kx_norm_dn.detach()
        port.Ky_norm_dn = sim.Ky_norm_dn.detach()
        ports.append(port)
    port_seconds = time.perf_counter()-start
    del sim
    radii = torch.linspace(.10,.39,args.layers,dtype=torch.float64)
    reference = current(args,waves,indices,radii)
    print('Dispersive reference completed',flush=True)
    candidate = reused(args,waves,indices,radii,template,plan,ports)
    ref_spec = torch.tensor(reference['spectrum'],dtype=torch.float64)
    got_spec = torch.tensor(candidate['spectrum'],dtype=torch.float64)
    ref_grad = torch.tensor(reference['gradient'],dtype=torch.float64)
    got_grad = torch.tensor(candidate['gradient'],dtype=torch.float64)
    errors = {'power_max_abs':float(torch.max(torch.abs(got_spec[:,1:]-ref_spec[:,1:]))),
              'gradient_max_abs':float(torch.max(torch.abs(got_grad-ref_grad))),
              'energy_max_abs':float(torch.max(torch.abs(got_spec[:,1:].sum(1)-1)))}
    passed = (errors['power_max_abs']<1e-8 and errors['energy_max_abs']<1e-6
              and bool(torch.allclose(got_grad,ref_grad,atol=1e-7,rtol=2e-5)))
    finite = None
    if args.check_fd:
        direction = torch.arange(1,args.layers+1,dtype=torch.float64)*torch.tensor([(-1.)**i for i in range(args.layers)])
        direction /= torch.linalg.vector_norm(direction)
        h = 1e-5
        with torch.no_grad():
            plus = reused(args,waves,indices,radii+h*direction,template,plan,ports,gradient=False)
            minus = reused(args,waves,indices,radii-h*direction,template,plan,ports,gradient=False)
        mean = lambda report:sum(row[1] for row in report['spectrum'])/len(waves)
        fd = (mean(plus)-mean(minus))/(2*h)
        ad = float(got_grad@direction)
        finite = {'autodiff':ad,'central_difference':fd,'abs_error':abs(ad-fd),'step':h}
        finite['passed'] = abs(ad-fd)<1e-6+3e-4*abs(fd)
        passed = passed and finite['passed']
    # An independent failure control: using a constant n(550 nm) must NOT
    # reproduce the dispersive result at the short-wavelength endpoint.
    fixed_index = index_at(samples,550.)
    fixed = common.current(wave_args(args,fixed_index),[waves[0]],radii)
    fixed_error = max(abs(fixed['spectrum'][0][j]-reference['spectrum'][0][j]) for j in (1,2))
    negative = {'constant_index_used':fixed_index,'wavelength_nm':waves[0],
                'power_error_if_dispersion_ignored':fixed_error,'passed':fixed_error>1e-8}
    passed = passed and negative['passed']
    result = {'status':'passed' if passed else 'failed','scope':'PMMA dispersive geometry-only sharing; hexagonal; CPU first derivative',
        'order':args.order,'grid':args.grid,'layers':args.layers,'period_nm':args.period,'height_nm':args.height,
        'wavelengths':waves,'indices':indices,'structured_layers_and_substrate_use_same_n_lambda':True,
        'material_csv':str(args.csv),'csv_sha256':hashlib.sha256(args.csv.read_bytes()).hexdigest(),
        'interpolation':'linear n versus wavelength','shortwave_policy':'first segment linear extrapolation to 400 nm only',
        'extrapolated_wavelengths':[w for w in waves if w<samples[0][0]],'extinction_k_assumed':0.,
        'reference':reference,'candidate':candidate,'errors':errors,'finite_difference':finite,'negative_control':negative,
        'port_preparation_seconds':port_seconds,'warm_speedup':reference['seconds']/candidate['seconds'],
        'cold_speedup_excluding_common_basis':reference['seconds']/(candidate['seconds']+port_seconds),
        'timing_repetitions':1,'python':platform.python_version(),'torch':str(torch.__version__),
        'production_code_modified':False,
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({key:result[key] for key in ('status','layers','errors','finite_difference','negative_control','warm_speedup')},indent=2))
    return 0 if passed else 1


if __name__=='__main__':
    raise SystemExit(main())
