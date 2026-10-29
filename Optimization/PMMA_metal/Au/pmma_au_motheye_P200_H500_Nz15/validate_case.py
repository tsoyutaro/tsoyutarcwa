"""Validate this independent PMMA-core/metal-shell optimization case."""
from pathlib import Path
import argparse
import cmath
import json
import math
import sys
import traceback

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='cpu')
    p.add_argument('--suite',choices=['small','initial'],default='small')
    p.add_argument('--order',type=int)
    p.add_argument('--grid',type=int)
    p.add_argument('--with-gradient',action='store_true')
    p.add_argument('--output',type=Path,default=HERE/'validation_report.json')
    args=p.parse_args()
    report=dict(status='running',suite=args.suite,device=args.device,cuda_execution_performed=False,checks=[])
    try:
        import torch
        from optimizer import common,geometry,physics
        from optimizer.materials import metal_epsilon_at,pmma_epsilon_at
        c=common.load_config(HERE/'config_hex_trial1_M8_Q512.json')
        device=common.device_setup(args.device,1)
        report.update(material=c['material'],cuda_execution_performed=device.type=='cuda',
                      material_provenance=c['material_provenance'])
        if args.suite=='initial':
            m=args.order or c['optimization_order'];q=args.grid or c['optimization_grid']
            x=torch.tensor(geometry.baseline_logits(c),dtype=torch.float64,device=device,requires_grad=args.with_gradient)
            print(f"START {c['material']}: initial M={m} Q={q} Nz={c['layers']}, {len(c['optimization_wavelengths_nm'])} wavelengths",flush=True)
            result=physics.objective(x,c,'hexagonal',c['optimization_wavelengths_nm'],m,q,device,gradient=args.with_gradient)
            report.update(order=m,grid=q,layers=c['layers'],wavelengths_nm=c['optimization_wavelengths_nm'],
                mean_reflectance=result['mean_reflectance'],mean_transmittance=result['mean_transmittance'],
                mean_absorptance=result['mean_absorptance'],seconds=result['seconds'],
                initial_profiles=geometry.profile_rows(x,c),gradient_checked=args.with_gradient,
                gradient_norm=float(result['gradient'].norm().cpu()) if args.with_gradient else None)
            report['checks'].append(dict(name='initial_full_band_physical',passed=True))
        else:
            c={**c,'layers':3,'parameter_count':8}
            m=args.order or 2;q=args.grid or 48;waves=[400.,550.,700.]
            x=torch.tensor(geometry.baseline_logits(c),dtype=torch.float64,device=device,requires_grad=True)
            reference=physics.objective(x,{**c,'rcwa_acceleration':'reference'},'hexagonal',waves,m,q,device,gradient=True)
            shared=physics.objective(x,c,'hexagonal',waves,m,q,device,gradient=True)
            power_error=max(abs(a[k]-b[k]) for a,b in zip(reference['rows'],shared['rows']) for k in ('reflectance','transmittance','absorptance_total'))
            gradient_error=float((reference['gradient']-shared['gradient']).abs().max().cpu())
            assert power_error<1e-8 and torch.allclose(reference['gradient'],shared['gradient'],atol=1e-7,rtol=2e-5)
            assert abs(shared['objective']-(1-shared['mean_absorptance']))<1e-12
            report['checks'].append(dict(name='native_reference_RTA_and_all_logits_gradient',passed=True,
                max_power_abs_error=power_error,max_gradient_abs_error=gradient_error))
            for backend in ('reference','auto'):
                cp=physics.objective(x,{**c,'rcwa_acceleration':backend,'checkpoint_layers':True},'hexagonal',waves,m,q,device,gradient=True)
                base=reference if backend=='reference' else shared
                assert abs(cp['objective']-base['objective'])<1e-9
                assert torch.allclose(cp['gradient'],base['gradient'],atol=1e-7,rtol=2e-5)
            report['checks'].append(dict(name='checkpoint_OFF_ON_reference_and_shared',passed=True))
            width=c['layers']+1
            for block,label in [(0,'PMMA_core'),(1,'metal_outer')]:
                d=torch.zeros_like(x);v=torch.sin(torch.arange(1,width+1,dtype=x.dtype,device=device));v-=v.mean()
                d[block*width:(block+1)*width]=v/v.norm();h=1e-5
                a=physics.objective(x.detach()+h*d,{**c,'rcwa_acceleration':'reference'},'hexagonal',waves,m,q,device,gradient=False)['objective']
                b=physics.objective(x.detach()-h*d,{**c,'rcwa_acceleration':'reference'},'hexagonal',waves,m,q,device,gradient=False)['objective']
                fd=(a-b)/(2*h);ad=float((shared['gradient']*d).sum().cpu())
                assert abs(fd-ad)<1e-6+.005*max(abs(fd),abs(ad))
                assert float(shared['gradient'][block*width:(block+1)*width].norm().cpu())>1e-10
                report['checks'].append(dict(name='independent_reference_FD_'+label,passed=True,finite_difference=fd,autodiff=ad,abs_error=abs(fd-ad)))
            # A finite perturbation of each physical radius checks the nm/period chain rule.
            radii=geometry.radii_tensor(x,c).detach()
            _,dr,_=physics._evaluator.evaluate_radii(radii,c,'hexagonal',waves,m,q,device,gradient=True)
            for col,label in [(0,'core_radius_nm'),(1,'outer_radius_nm')]:
                d=torch.zeros_like(radii);d[1,col]=1.;h=1e-3
                a=physics._evaluator.evaluate_radii(radii+h*d,c,'hexagonal',waves,m,q,device,gradient=False)[0]
                b=physics._evaluator.evaluate_radii(radii-h*d,c,'hexagonal',waves,m,q,device,gradient=False)[0]
                fd=(a-b)/(2*h);ad=float(dr[1,col].cpu())
                assert abs(fd-ad)<1e-6+.005*max(abs(fd),abs(ad))
                report['checks'].append(dict(name='physical_radius_FD_'+label,passed=True,abs_error=abs(fd-ad)))
            original=physics.metal_epsilon_at
            try:
                physics.metal_epsilon_at=lambda config,w:complex(pmma_epsilon_at(config,w))
                for backend in ('reference','auto'):
                    control=physics.objective(x,{**c,'rcwa_acceleration':backend},'hexagonal',waves,m,q,device,gradient=True)
                    assert abs(control['objective']-1)<1e-8 and float(control['gradient'].norm().cpu())<1e-7
            finally:physics.metal_epsilon_at=original
            report['checks'].append(dict(name='lossless_control_R_plus_T_equals_1_and_zero_total_gradient',passed=True))
            # Independent normal-incidence Fresnel slab on the actual lossless PMMA substrate.
            wave=550.;em=complex(metal_epsilon_at(c,wave));ec=complex(pmma_epsilon_at(c,wave));h_nm=20.
            with torch.no_grad():
                sim=physics.make_sim(c,'hexagonal',wave,m,q,device)
                center=(physics.d6.center_of(sim),)
                sim.add_layer_circle_shell_nvm(h_nm/c['period_nm'],.1,.3,em,em,em,centers=center,nx=q,ny=q,use_lanczos=False)
                t,r=sim._solve_polarization_reduced_smatrix(incident_source=physics.source(sim))
                _,slab=physics.powers(sim,t,r,c,wave)
            n0=1.;n1=cmath.sqrt(em);n2=cmath.sqrt(ec);phase=cmath.exp(1j*2*math.pi*n1*h_nm/wave)
            r01=(n0-n1)/(n0+n1);r12=(n1-n2)/(n1+n2)
            rr=(r01+r12*phase**2)/(1+r01*r12*phase**2)
            tt=(2*n0/(n0+n1))*(2*n1/(n1+n2))*phase/(1+r01*r12*phase**2)
            rf=abs(rr)**2;tf=n2.real*abs(tt)**2
            err=max(abs(slab['reflectance']-rf),abs(slab['transmittance']-tf),abs(slab['absorptance_total']-(1-rf-tf)))
            assert err<1e-8
            report['checks'].append(dict(name='uniform_metal_film_on_PMMA_vs_Fresnel',passed=True,max_power_abs_error=err))
            report.update(order=m,grid=q,layers=3,wavelengths_nm=waves,
                          mean_absorptance=shared['mean_absorptance'])
        report['status']='passed'
    except Exception as error:
        report.update(status='nonphysical' if type(error).__name__=='NonphysicalResult' else 'failed',
                      error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
