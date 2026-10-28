"""Read-only v2 diagnostics, including a no-NV-correction Galerkin control.

Reports raw fluxes even when passivity fails, and compares reference, matrix,
and shared paths. It never resumes or rewrites an Optuna/Adam experiment.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def power_record(r,p,tolerance):
    values=dict(reflectance=r,power_into_substrate=p,absorptance_total=1-r,motheye_absorptance=1-r-p)
    violations={k:repr(v) for k,v in values.items() if not math.isfinite(v) or v < -tolerance or v > 1+tolerance}
    return dict(**{k:v if math.isfinite(v) else None for k,v in values.items()},
                finite=all(math.isfinite(v) for v in values.values()),passive=not violations,violations=violations)


def json_safe(value):
    if isinstance(value,float) and not math.isfinite(value):return repr(value)
    if isinstance(value,dict):return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [json_safe(v) for v in value]
    return value


def compare(a,b):
    if 'error' in a or 'error' in b or not a['finite'] or not b['finite']:
        return dict(comparable=False,passed=False,max_power_abs_error=None)
    error=max(abs(a[k]-b[k]) for k in ('reflectance','power_into_substrate'))
    return dict(comparable=True,passed=error<=1e-8,max_power_abs_error=error)



def galerkin_pq(port,eps,plan):
    """Standard Cartesian Maxwell P/Q, with no NV normal-field correction.

    Uses the same epsilon convolution but not the inverse-rule/normal tensor.
    This control is for convergence diagnostics, not a replacement optimizer.
    """
    import torch
    from rcwa_ext.acceleration.d6 import D6BlockInverse
    from rcwa_ext.acceleration.legacy import BlockInverse
    inverse=(D6BlockInverse if plan.triangular else BlockInverse)(port,eps,plan)
    n=plan.scalar_dimension
    kx,ky=port.Kx_norm_dn,port.Ky_norm_dn
    if plan.triangular:
        kx=kx.index_select(0,plan.indices);ky=ky.index_select(0,plan.indices)
    kx,ky=kx[:,None],ky[:,None]
    electric,magnetic=plan.electric,plan.magnetic
    hx,hy=magnetic[:n],magnetic[n:]
    longitudinal=inverse.apply(ky*hx-kx*hy)
    p_image=torch.cat((hy+kx*longitudinal,-hx+ky*longitudinal),dim=0)
    ex,ey=electric[:n],electric[n:]
    curl=-ky*ex+kx*ey
    q_image=torch.cat((-eps@ey+kx*curl,eps@ex+ky*curl),dim=0)
    return electric.mH@p_image,magnetic.mH@q_image


def galerkin_response(cache,disks,index,port):
    import torch
    plan=cache['plan'];modes=[]
    for disk in disks:
        eps=cache['identity']+(index**2-1)*disk
        p,q=galerkin_pq(port,eps,plan)
        squared,electric=port._eig(p@q);kz=port._positive_kz(squared)
        magnetic=port._magnetic_eigenvectors(p,q,electric,kz)
        modes.append(dict(electric=electric,magnetic=magnetic,kz=kz))
    transmission,reflection,_,_=port._polarized_redheffer_scattering(modes,port.input_interface,port.output_interface)
    t,r=transmission@port.source,reflection@port.source
    R=.5*torch.real(r.conj()@(port.flux_in@r))/port.incident_flux
    T=.5*torch.real(t.conj()@(port.flux_out@t))/port.incident_flux
    return R,T


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-dir',type=Path,required=True)
    parser.add_argument('--run-root',type=Path,default=Path('results/hex_trial1_v3'))
    parser.add_argument('--trial',type=int,default=0)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--lattice',choices=['square','hexagonal'],default='hexagonal')
    parser.add_argument('--order',type=int)
    parser.add_argument('--grid',type=int)
    parser.add_argument('--wavelengths')
    parser.add_argument('--state',choices=['current','best'],default='current')
    parser.add_argument('--methods',nargs='+',choices=['shared','reference','matrix','galerkin'],default=['shared','reference','matrix'])
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    case=args.case_dir.resolve()
    sys.path.insert(0,str(case))
    import torch
    from optimizer import common,geometry,physics
    from optimizer.materials import epsilon_at,wavelength_config
    from optimizer.helpers import zero_order_x_source
    from runtime_policy import execution_sources
    from rcwa_ext.acceleration.spectral import SharedEvaluator
    folder=args.run_root if args.run_root.is_absolute() else case/args.run_root
    trial=folder/args.lattice/'trials'/f'trial_{args.trial:04d}'
    checkpoint=trial/'adam_checkpoint.pt'
    checksum=sha(checkpoint)
    config=json.loads((folder/'resolved_config.json').read_text(encoding='utf-8-sig'))
    if config.get('material_model')!='metal_csv' or config.get('substrate_mode')!='semi-infinite':
        parser.error('This diagnostic targets standalone v3 metal/metal-substrate cases.')
    manifest=folder/'manifest.json'
    if manifest.exists():
        saved=json.loads(manifest.read_text(encoding='utf-8-sig'))
        if saved.get('source_sha256')!=execution_sources(common.sources()):
            raise ValueError('Experiment source identity differs. Use the same v3 sources as this run; do not edit its manifest.')
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    values=state['logits'] if args.state=='current' else state['best_logits']
    if values is None:raise ValueError('No saved valid best shape; use --state current.')
    device=common.device_setup(args.device,config.get('cpu_threads',1))
    logits=torch.as_tensor(values,dtype=torch.float64,device=device)
    order=args.order or config['optimization_order'];grid=args.grid or config['optimization_grid']
    if order<1 or grid<max(32,4*order+4):parser.error('Invalid order/grid.')
    waves=common.wavelengths(args.wavelengths) if args.wavelengths else config['optimization_wavelengths_nm']
    report=dict(status='running',diagnostic_version=2,galerkin_is_diagnostic_only=True,material=config.get('material'),lattice=args.lattice,
        device=str(device),cuda_execution_performed=device.type=='cuda',torch=str(torch.__version__),
        gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        checkpoint=str(checkpoint),checkpoint_sha256=checksum,state=args.state,
        next_step=state['next_step'],successful_gradient_evaluations=len(state['history']),
        last_history=state['history'][-1] if state['history'] else None,
        original_order=config['optimization_order'],evaluation_order=order,grid=grid,
        wavelengths_nm=waves,logits=logits.detach().cpu().tolist(),radii_nm=geometry.radii_values(logits.detach().cpu().tolist(),config),
        passivity_tolerance=config.get('passivity_tolerance',1e-6),rows=[])
    settings_id=hashlib.sha256(json.dumps(dict(version=2,order=order,grid=grid,state=args.state,waves=waves,methods=args.methods),sort_keys=True).encode()).hexdigest()[:8]
    output=args.output or folder/'diagnostics'/f'power_flux_v2_M{order}_Q{grid}_{args.state}_{checksum[:12]}_{settings_id}.json'
    if not output.resolve().is_relative_to((folder/'diagnostics').resolve()):
        parser.error('Diagnostic output must stay within run-root/diagnostics.')
    def save():
        output.parent.mkdir(parents=True,exist_ok=True)
        tmp=output.with_name(output.name+'.tmp')
        tmp.write_text(json.dumps(json_safe(report),indent=2,allow_nan=False)+'\n',encoding='utf-8')
        tmp.replace(output)
    def sync():
        if device.type=='cuda':torch.cuda.synchronize(device)
    def flux(electric,medium,direction):
        magnetic=direction*(medium@electric);n=electric.numel()//2
        return .5*torch.real((electric[:n]*magnetic[n:].conj()-electric[n:]*magnetic[:n].conj()).sum())
    def raw_regular(radius,wave,backend):
        cc=wavelength_config({**config,'rcwa_acceleration':backend},wave)
        sim=physics.make_sim(cc,args.lattice,wave,order,grid,device)
        depth=cc['height_nm']/(cc['layers']*cc['period_nm'])
        for r in radius:sim.add_structured_layer(physics.layer_spec(r,cc,depth))
        incident=zero_order_x_source(sim)
        transmitted,reflected=sim._solve_polarization_reduced_smatrix(incident_source=incident)
        incoming=flux(incident,sim.Vi,1)
        return float((-flux(reflected,sim.Vi,-1)/incoming).cpu()),float((flux(transmitted,sim.Vo,1)/incoming).cpu())
    tol=report['passivity_tolerance']
    with torch.no_grad():
        radii=geometry.radii_tensor(logits,config)
        cache=leaves=evaluator=galerkin_disks=None
        if 'shared' in args.methods or 'galerkin' in args.methods:
            model=SimpleNamespace(material=config.get('material','metal'),physics=physics,
                geometry=geometry,epsilon_at=epsilon_at)
            evaluator=SharedEvaluator(model)
            cache=evaluator.cache(config,args.lattice,waves,order,grid,device)
            if 'shared' in args.methods:
                leaves=tuple(v for r in radii/config['period_nm'] for v in evaluator.geometry_pair(r,cache))
            if 'galerkin' in args.methods:
                from rcwa_ext.acceleration.legacy import center_of
                template=cache['template'];plan=cache['plan']
                galerkin_disks=[]
                for radius in radii/config['period_nm']:
                    eps_full=template._circle_toeplitz(radius,1.,4.,(center_of(template),),use_lanczos=False,lanczos_power=2)
                    galerkin_disks.append((plan.scalar(eps_full)-cache['identity'])/3.)
        for i,wave in enumerate(waves):
            row=dict(wavelength_nm=wave,epsilon_real=epsilon_at(config,wave).real,
                epsilon_imag=epsilon_at(config,wave).imag,methods={})
            for method in args.methods:
                print(f'{i+1}/{len(waves)} {wave:g}nm {method}',flush=True)
                sync();started=time.perf_counter()
                try:
                    if method=='shared':
                        r,p=evaluator.response(cache,leaves,wave,cache['indices'][i],cache['ports'][i],False)
                        r,p=float(r.cpu()),float(p.cpu())
                    elif method=='galerkin':
                        r,p=galerkin_response(cache,galerkin_disks,cache['indices'][i],cache['ports'][i])
                        r,p=float(r.cpu()),float(p.cpu())
                    else:r,p=raw_regular(radii,wave,method)
                    sync();record=power_record(r,p,tol)
                    record['seconds']=time.perf_counter()-started
                    print(f'  R={r:.12g} P_sub={p:.12g} A_moth={1-r-p:.12g} passive={record["passive"]}',flush=True)
                except Exception as error:
                    if isinstance(error,torch.OutOfMemoryError):raise
                    record=dict(error=type(error).__name__+': '+str(error))
                row['methods'][method]=record
            if 'reference' in row['methods']:
                row['comparisons']={k:compare(row['methods']['reference'],v) for k,v in row['methods'].items() if k in ('shared','matrix')}
            if 'galerkin' in row['methods']:
                nvm=next((k for k in ('shared','reference','matrix') if k in row['methods']),None)
                if nvm is not None:
                    delta=compare(row['methods'][nvm],row['methods']['galerkin'])
                    row['factorization_difference']=dict(nvm_method=nvm,max_power_abs_difference=delta['max_power_abs_error'],convergence_not_tested=True)
            report['rows'].append(row);save()
    assert sha(checkpoint)==checksum,'Checkpoint changed during diagnosis; stop concurrent optimization before comparing.'
    report.update(status='diagnostic_complete',checkpoint_unchanged=True,
        nonpassive_wavelengths_by_method={method:[r['wavelength_nm'] for r in report['rows'] if not r['methods'][method].get('passive',False)] for method in args.methods},
        reference_comparisons_passed=all(c['passed'] for r in report['rows'] for c in r.get('comparisons',{}).values()) if 'reference' in args.methods and len(set(args.methods)&{'shared','reference','matrix'})>1 else None)
    save()
    print(json.dumps({k:report[k] for k in ['status','material','next_step','successful_gradient_evaluations','nonpassive_wavelengths_by_method','reference_comparisons_passed','checkpoint_unchanged']},indent=2),flush=True)
    print('Saved '+str(output),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
