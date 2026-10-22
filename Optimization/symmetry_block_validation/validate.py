"""Compare a frozen production NVM solver against experimental block assembly."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time
import traceback
import torch
import block_assembly as assembly

HERE = Path(__file__).resolve().parent


def pmma_index(wavelength_nm):
    with (HERE/'data/Szczurowski.csv').open(encoding='utf-8-sig',newline='') as stream:
        rows = [(1000*float(row['wl']),float(row['n'])) for row in csv.DictReader(stream)]
    if not 400 <= wavelength_nm <= rows[-1][0]:
        raise ValueError('PMMA wavelength outside supported range.')
    a,b = rows[:2]
    for first,second in zip(rows,rows[1:]):
        if first[0] <= wavelength_nm <= second[0]:
            a,b = first,second
            break
    return a[1]+(wavelength_nm-a[0])*(b[1]-a[1])/(b[0]-a[0])


def index_of(material,wave):
    return pmma_index(wave) if material == 'PMMA' else float(material[1:])


def synchronize(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)


def timing(function,device,repeats):
    with torch.no_grad():
        answer = function()
        synchronize(device)
        del answer
        seconds,extra = [],[]
        for _ in range(repeats):
            if str(device).startswith('cuda'):
                torch.cuda.reset_peak_memory_stats(device)
                initial = torch.cuda.memory_allocated(device)
            start = time.perf_counter()
            answer = function()
            synchronize(device)
            seconds.append(time.perf_counter()-start)
            if str(device).startswith('cuda'):
                extra.append(torch.cuda.max_memory_allocated(device)-initial)
            del answer
    return {'median_seconds':statistics.median(seconds),'samples_seconds':seconds,
            'peak_cuda_extra_bytes':max(extra) if extra else None,
            'scope':'assembly only; shared input matrices and reusable geometry bases excluded; no backward'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',choices=['quick','standard'],default='quick')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--orders',type=int,nargs='+')
    parser.add_argument('--lattices',choices=['square','hexagonal'],nargs='+',default=['square','hexagonal'])
    parser.add_argument('--grid',type=int)
    parser.add_argument('--matrix-only',action='store_true')
    parser.add_argument('--benchmark-repeats',type=int,default=0)
    parser.add_argument('--cpu-threads',type=int,default=1)
    parser.add_argument('--output',type=Path,default=HERE/'validation_results')
    args = parser.parse_args()
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        parser.error('CUDA is not available in this Python environment. Use --device cpu or install compatible GPU PyTorch.')
    orders = args.orders or ([1,2] if args.suite == 'quick' else [4,8])
    if any(m < 1 for m in orders) or args.benchmark_repeats < 0:
        parser.error('Orders must be >=1 and benchmark repeats >=0.')
    torch.set_num_threads(args.cpu_threads)
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    report = {'status':'running','suite':args.suite,'device':args.device,
        'orders':orders,'lattices':args.lattices,'torch':torch.__version__,
        'python':sys.version,'platform':platform.platform(),'cuda_version':torch.version.cuda,
        'gpu':torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else None,
        'candidate_scope':'scalar-block inverse actions and direct source-sector P/Q; scalar convolution and full normal-projector inputs retained',
        'tolerances':{'matrix_relative':2e-9,'scalar_invariance_relative':2e-9,
            'field_relative':2e-7,'power_absolute':1e-8,'lossless_energy_absolute':1e-6,
            'gradient_atol':1e-7,'gradient_rtol':2e-5,'finite_difference_atol':1e-6,'finite_difference_rtol':3e-4},
        'source_sha256':{p.relative_to(HERE).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(HERE.rglob('*.py'))},
        'pmma_sha256':hashlib.sha256((HERE/'data/Szczurowski.csv').read_bytes()).hexdigest(),
        'cases':[],'failures':[],'production_code_modified':False}
    contexts = {}

    def save():
        (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')

    def record(label,function):
        start = time.perf_counter()
        try:
            data = function()
            data.update(label=label,status='passed',wall_seconds=time.perf_counter()-start)
            report['cases'].append(data)
            print('PASS '+label,flush=True)
        except Exception as exc:
            report['cases'].append({'label':label,'status':'failed','error':str(exc),
                                    'traceback':traceback.format_exc(),'wall_seconds':time.perf_counter()-start})
            report['failures'].append(label)
            print('FAIL '+label+': '+str(exc),flush=True)
        save()

    def setup(lattice,m,index,wave):
        grid = args.grid or max(48 if args.suite == 'quick' else 128,4*m+4)
        sim = assembly.make_sim(lattice,m,grid,index,wave,200.,args.device)
        key = (lattice,m)
        if key not in contexts:
            contexts[key] = assembly.GeometryContext(sim)
        return sim,contexts[key],grid

    def matrix_case(lattice,m,radius=.27,material='n2.5',wave=550.):
        index = index_of(material,wave)
        sim,context,grid = setup(lattice,m,index,wave)
        r = torch.tensor(radius,dtype=torch.float64,device=args.device)
        with torch.no_grad():
            inputs = assembly.material_inputs(sim,r,index,grid,context)
            p_ref,q_ref,full_p,full_q = assembly.reference_pq(sim,*inputs,context)
            p,q,diag = assembly.candidate_pq(sim,*inputs,context)
            errors = {'P':assembly.relative_error(p,p_ref),'Q':assembly.relative_error(q,q_ref),
                      'PQ':assembly.relative_error(p@q,p_ref@q_ref)}
            if max(errors.values()) > report['tolerances']['matrix_relative']:
                raise AssertionError(errors)
            if lattice == 'square':
                leakage = max(assembly.relative_error(full_p@context.magnetic,context.electric@p_ref),
                              assembly.relative_error(full_q@context.electric,context.magnetic@q_ref))
                if leakage > 2e-9:
                    raise AssertionError(f'Vector source-sector invariance failed: {leakage}')
            else:
                leakage = max(assembly.relative_error(full_p@context.magnetic,context.electric@p_ref),
                              assembly.relative_error(full_q@context.electric,context.magnetic@q_ref))
                if leakage > 2e-9:
                    raise AssertionError(f'D6 E1-row invariance failed: {leakage}')
            result = {'type':'matrix','lattice':lattice,'order':m,'grid':grid,'material':material,
                'wavelength_nm':wave,'radius_over_period':radius,'errors':errors,
                'vector_invariance_error':leakage,'scalar_harmonics':context.scalar_dimension,
                'reference_PQ_dimension':full_p.shape[0],'candidate_PQ_dimension':p.shape[0],**diag}
            if args.benchmark_repeats:
                del p_ref,q_ref,full_p,full_q,p,q
                result['reference_timing'] = timing(lambda:assembly.reference_pq(sim,*inputs,context),args.device,args.benchmark_repeats)
                result['candidate_timing'] = timing(lambda:assembly.candidate_pq(sim,*inputs,context),args.device,args.benchmark_repeats)
        return result

    def response(lattice,m,material,wave,radii,variant,gradient):
        index = index_of(material,wave)
        sim,context,grid = setup(lattice,m,index,wave)
        values = torch.tensor(radii,dtype=torch.float64,device=args.device,requires_grad=gradient)
        start = time.perf_counter()
        r,t,transmitted,reflected = assembly.optical_response(sim,values,index,500.,200.,grid,variant,context)
        grad = torch.autograd.grad(r,values)[0] if gradient else None
        synchronize(args.device)
        result = {'R':float(r.detach()),'T':float(t.detach()),
            'transmitted':transmitted.detach(),'reflected':reflected.detach(),
            'gradient':grad.detach() if grad is not None else None,
            'forward_backward_seconds':time.perf_counter()-start}
        return result

    def optical_case(lattice,m,material,wave):
        radii = [.10,.23,.39]
        ref = response(lattice,m,material,wave,radii,'reference',True)
        got = response(lattice,m,material,wave,radii,'candidate',True)
        tol = report['tolerances']
        power_error = max(abs(got[key]-ref[key]) for key in ('R','T'))
        field_error = max(assembly.relative_error(got[key],ref[key]) for key in ('transmitted','reflected'))
        energy = max(abs(value['R']+value['T']-1) for value in (ref,got))
        if power_error > tol['power_absolute'] or field_error > tol['field_relative'] or energy > tol['lossless_energy_absolute']:
            raise AssertionError({'power':power_error,'field':field_error,'energy':energy})
        if min(ref['R'],ref['T'],got['R'],got['T']) < -1e-8 or max(ref['R'],ref['T'],got['R'],got['T']) > 1+1e-8:
            raise AssertionError('Lossless powers outside [0,1].')
        if not torch.all(torch.isfinite(got['gradient'])) or not torch.allclose(got['gradient'],ref['gradient'],atol=tol['gradient_atol'],rtol=tol['gradient_rtol']):
            raise AssertionError({'reference_gradient':ref['gradient'].tolist(),'candidate_gradient':got['gradient'].tolist()})
        direction = torch.tensor([.3,-.5,.8],dtype=torch.float64,device=args.device)
        direction = direction/torch.linalg.vector_norm(direction)
        h = 2e-5
        plus = [r+h*float(d) for r,d in zip(radii,direction)]
        minus = [r-h*float(d) for r,d in zip(radii,direction)]
        with torch.no_grad():
            finite = (response(lattice,m,material,wave,plus,'candidate',False)['R']-
                      response(lattice,m,material,wave,minus,'candidate',False)['R'])/(2*h)
        autodiff = float(got['gradient']@direction)
        fd_error = abs(finite-autodiff)
        if fd_error > tol['finite_difference_atol']+tol['finite_difference_rtol']*abs(finite):
            raise AssertionError({'finite_difference':finite,'autodiff':autodiff,'error':fd_error})
        return {'type':'optics_and_gradient','lattice':lattice,'order':m,'material':material,
                'wavelength_nm':wave,'radii_over_period':radii,'height_nm':500.,'period_nm':200.,
                'reference_R':ref['R'],'candidate_R':got['R'],'reference_T':ref['T'],'candidate_T':got['T'],
                'power_max_abs_error':power_error,'field_max_relative_error':field_error,
                'energy_max_abs_error':energy,'reference_gradient':ref['gradient'].tolist(),
                'candidate_gradient':got['gradient'].tolist(),'gradient_max_abs_error':float(torch.max(torch.abs(ref['gradient']-got['gradient']))),
                'directional_autodiff':autodiff,'directional_finite_difference':finite,'finite_difference_abs_error':fd_error,
                'reference_forward_backward_seconds':ref['forward_backward_seconds'],
                'candidate_forward_backward_seconds':got['forward_backward_seconds']}

    def negative_case(lattice):
        sim,context,grid = setup(lattice,min(orders),2.5,550.)
        r = torch.tensor(.27,dtype=torch.float64,device=args.device)
        eps,_,_ = assembly.material_inputs(sim,r,2.5,grid,context)
        first,second = context.scalar_bases[:2]
        bad = eps+.01*(first[:,:1]@second[:,:1].mH+second[:,:1]@first[:,:1].mH)
        rejected = []
        try:
            assembly.BlockInverse(sim,bad,context)
        except ValueError:
            rejected.append('symmetry_broken_scalar_matrix')
        displaced = list(assembly.center_of(sim))
        displaced[0] += .03
        try:
            assembly.check_supported(sim,r,displaced)
        except ValueError:
            rejected.append('off_centre_geometry')
        sim.set_incident_angle(.1,0.)
        try:
            assembly.check_supported(sim,r,assembly.center_of(sim))
        except ValueError:
            rejected.append('oblique_incidence')
        if len(rejected) != 3:
            raise AssertionError(rejected)
        return {'type':'negative_controls','lattice':lattice,'rejected':rejected}

    save()
    for lattice in args.lattices:
        for m in orders:
            record(f'matrix/{lattice}/M{m}/n2.5/r0.27',lambda l=lattice,o=m:matrix_case(l,o))
        for radius,material,wave in [(.005,'n1.5',400.),(.495,'PMMA',700.)]:
            record(f'matrix/{lattice}/M{min(orders)}/{material}/r{radius}',
                   lambda l=lattice,r=radius,mat=material,w=wave:matrix_case(l,min(orders),r,mat,w))
        record(f'negative_controls/{lattice}',lambda l=lattice:negative_case(l))
    if not args.matrix_only:
        optical = [('n1.5',400.),('n2.5',550.),('PMMA',700.)]
        if args.suite == 'standard':
            optical += [('n2.0',550.),('PMMA',400.),('PMMA',550.)]
        for lattice in args.lattices:
            for material,wave in optical:
                record(f'optics/{lattice}/M{max(orders)}/{material}/{wave:g}nm',
                       lambda l=lattice,mat=material,w=wave:optical_case(l,max(orders),mat,w))
    report['status'] = 'passed' if not report['failures'] else 'failed'
    report['passed_count'] = sum(row['status'] == 'passed' for row in report['cases'])
    report['failed_count'] = len(report['failures'])
    save()
    fields = ['label','status','type','lattice','order','material','wavelength_nm','power_max_abs_error',
              'field_max_relative_error','energy_max_abs_error','gradient_max_abs_error','finite_difference_abs_error','wall_seconds','error']
    with (output/'summary.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(report['cases'])
    print(f"{report['status'].upper()}: {report['passed_count']} passed, {report['failed_count']} failed; report={output/'report.json'}",flush=True)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
