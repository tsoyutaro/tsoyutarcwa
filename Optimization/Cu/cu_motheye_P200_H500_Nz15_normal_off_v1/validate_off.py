"""Finite-difference checks of the actual OFF production objective in logits."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import sys
import traceback

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
if os.environ.get('MOTHEYE_DEPENDENCY_PATH'):sys.path.append(os.environ['MOTHEYE_DEPENDENCY_PATH'])
from optimizer import common,physics
from runtime_policy import execution_sources


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=HERE/'config_hex_trial1_M12_Q1024_off.json')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--order',type=int)
    p.add_argument('--grid',type=int)
    p.add_argument('--wavelengths',default='400,550,700')
    p.add_argument('--profile-json',type=Path)
    p.add_argument('--fd-step',type=float,default=1e-4)
    p.add_argument('--fd-atol',type=float,default=1e-6)
    p.add_argument('--fd-rtol',type=float,default=2e-3)
    p.add_argument('--fd-backend',choices=['reference','auto'],default='reference')
    p.add_argument('--directions-only',action='store_true',help='Deterministic directions instead of all components')
    p.add_argument('--direction-count',type=int,default=3)
    p.add_argument('--skip-reference-gradient',action='store_true',help='Omit reference autodiff parity; recorded as unchecked')
    a=p.parse_args()
    if a.direction_count<1:p.error('--direction-count must be positive.')
    if any(not math.isfinite(v) or v<=0 for v in [a.fd_step,a.fd_atol,a.fd_rtol]):p.error('FD parameters must be positive and finite.')
    c=common.load_config(a.config);waves=common.wavelengths(a.wavelengths)
    order=a.order or c['optimization_order'];grid=a.grid or c['optimization_grid']
    if order<1 or grid<max(32,4*order+4):p.error('Invalid order/grid.')
    import torch
    from rcwa_ext.normal_fields import DEFINITION
    if DEFINITION!=c['normal_field_definition']:raise ValueError('OFF definition differs from configuration.')
    device=common.device_setup(a.device,c['cpu_threads']);runtime=common.runtime_environment(device)
    logits=c['initial_logits']
    if a.profile_json:
        data=json.loads(a.profile_json.read_text(encoding='utf-8-sig'))
        logits=data.get('logits',data.get('best_logits',data.get('best',{}).get('logits')))
    if not isinstance(logits,list) or len(logits)!=c['layers']+1:raise ValueError('Profile dimensions differ.')
    x=torch.tensor(logits,dtype=torch.float64,device=device,requires_grad=True)
    sources=execution_sources(common.sources())
    ident=common.signature_for(dict(config=c,sources=sources,runtime=runtime,logits=logits,order=order,grid=grid,waves=waves,arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}))[:16]
    output=HERE/'validation'/ident/'report.json'
    report=dict(status='running',normal_field_mode='off',normal_field_definition=DEFINITION,
        runtime_environment=runtime,cuda_execution_performed=device.type=='cuda',source_sha256=sources,
        order=order,grid=grid,layers=c['layers'],wavelengths_nm=waves,initial_logits=logits,
        finite_difference_backend=a.fd_backend,full_component_fd=not a.directions_only,reference_gradient_checked=not a.skip_reference_gradient,
        full_training_wavelength_coverage=waves==c['optimization_wavelengths_nm'],finite_differences=[])
    def save():common.write_json(output,report)
    def evaluate(value,backend,gradient=False,checkpoint=False):
        return physics.objective(value,{**c,'rcwa_acceleration':backend,'checkpoint_layers':checkpoint},'hexagonal',waves,order,grid,device,gradient=gradient)
    def close(u,v):return abs(u-v)<=a.fd_atol+a.fd_rtol*max(abs(u),abs(v))
    try:
        print(f'OFF gradient check: M={order} Q={grid}, {c["layers"]} layers, {len(waves)} wavelengths',flush=True)
        print('  shared autodiff',flush=True)
        auto=evaluate(x,'auto',True)
        print('  reference autodiff' if not a.skip_reference_gradient else '  reference autodiff omitted by explicit flag',flush=True)
        reference=None if a.skip_reference_gradient else evaluate(x,'reference',True)
        print('  layer-checkpoint autodiff',flush=True)
        checkpoint=evaluate(x,'auto',True,True)
        g=auto['gradient']
        def errors(other):
            power=max(abs(u[k]-v[k]) for u,v in zip(auto['rows'],other['rows']) for k in ['reflectance','power_into_substrate'])
            return dict(loss_abs=abs(auto['objective']-other['objective']),power_max_abs=power,
                gradient_max_abs=float((g-other['gradient']).abs().max().cpu()),
                passed=abs(auto['objective']-other['objective'])<1e-8 and power<1e-8 and torch.allclose(g,other['gradient'],atol=1e-7,rtol=2e-5))
        report.update(objective=auto['objective'],gradient_logits=g.cpu().tolist(),rows=auto['rows'],
            reference_parity=errors(reference) if reference is not None else dict(passed=None,checked=False),checkpoint_parity=errors(checkpoint),gauge_gradient_abs_sum=abs(float(g.sum().cpu())))
        directions=[]
        if a.directions_only:
            gen=torch.Generator(device='cpu').manual_seed(c['seed']+557)
            for i in range(a.direction_count):
                d=torch.randn(x.numel(),generator=gen,dtype=torch.float64);d-=d.mean()
                directions.append((f'random_{i}',(d/d.norm()).to(device)))
        else:
            for i in range(x.numel()):
                d=torch.zeros_like(x);d[i]=1;directions.append((f'logit_{i}',d))
        for i,(label,direction) in enumerate(directions):
            values=[]
            for h in [a.fd_step,a.fd_step/2]:
                plus=evaluate(x.detach()+h*direction,a.fd_backend)['objective']
                minus=evaluate(x.detach()-h*direction,a.fd_backend)['objective']
                values.append((plus-minus)/(2*h))
            ad=float(torch.dot(g,direction).cpu())
            row=dict(label=label,autodiff=ad,fd_h=values[0],fd_half_h=values[1],
                abs_error=abs(ad-values[1]),passed=close(ad,values[1]) and close(values[0],values[1]))
            report['finite_differences'].append(row);save()
            print(f'FD {i+1}/{len(directions)} {label}: abs_error={row["abs_error"]:.3g} passed={row["passed"]}',flush=True)
        same=execution_sources(common.sources())==sources
        passed=(a.skip_reference_gradient or report['reference_parity']['passed']) and report['checkpoint_parity']['passed'] and all(v['passed'] for v in report['finite_differences']) and report['gauge_gradient_abs_sum']<1e-7 and same
        report.update(status='passed' if passed else 'failed',source_unchanged=same,
            fd_step=a.fd_step,fd_atol=a.fd_atol,fd_rtol=a.fd_rtol,
            max_fd_abs_error=max(v['abs_error'] for v in report['finite_differences']),
            scope='Derivative consistency at one shape and fixed numerical resolution; not Fourier/grid convergence or whole-band proof.')
    except Exception as error:
        report.update(status='failed',error=str(error),traceback=traceback.format_exc())
        print(report['traceback'],file=sys.stderr,flush=True)
    save();common.write_json(HERE/'latest_off_validation.json',dict(report=str(output),status=report['status']))
    print(json.dumps({k:report[k] for k in ['status','order','grid','layers','cuda_execution_performed','max_fd_abs_error','error'] if k in report},indent=2),flush=True)
    print('Saved '+str(output),flush=True)
    return 0 if report['status']=='passed' else 1

if __name__=='__main__':raise SystemExit(main())
