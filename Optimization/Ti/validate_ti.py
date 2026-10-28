"""Check Ti material loading, reference/shared gradients, and optional M8 initial state."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys

ROOT=Path(__file__).resolve().parent
CASE=ROOT/'ti_motheye_P200_H500_Nz15'
sys.path.insert(0,str(CASE))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='cpu')
    p.add_argument('--check-m8-initial',action='store_true')
    p.add_argument('--output',type=Path,default=ROOT/'verification_cpu.json')
    args=p.parse_args()
    import torch
    from optimizer import common,geometry,physics
    from optimizer.materials import epsilon_at
    device=common.device_setup(args.device,1)
    c=common.load_config(CASE/'config_hex_trial1_M8_Q512.json')
    report=dict(status='running',device=str(device),cuda_execution_performed=device.type=='cuda',
                material='Ti',material_provenance=c['material_provenance'],checks=[])
    for n in (5,10,15,20):
        case=ROOT/f'ti_motheye_P200_H500_Nz{n}'
        config=json.loads((case/'config.json').read_text(encoding='utf-8-sig'))
        test=json.loads((case/'config_hex_trial1_M8_Q512.json').read_text(encoding='utf-8-sig'))
        assert config['material']=='Ti' and config['layers']==n and config['material_csv']=='data/Ti.csv'
        assert config['optimization_order']==8 and config['optimization_grid']==512
        assert test['trials']==1 and test['adam_steps']==500 and test['lattices']==['hexagonal']
        assert hashlib.sha256((case/'data/Ti.csv').read_bytes()).hexdigest()=='1b13b2ee9306617d9dd5a1a0e2746d7c6bd880324c7a607f50853b8e6e819216'
    report['checks'].append(dict(name='four_independent_cases_and_csv',passed=True))
    eps=[epsilon_at(c,w) for w in c['optimization_wavelengths_nm']]
    assert all(math.isfinite(z.real) and math.isfinite(z.imag) and z.imag>0 for z in eps)
    assert max(z.real for z in eps)!=min(z.real for z in eps)
    report['checks'].append(dict(name='31_wavelengths_passive_dispersive_material',passed=True))
    small={**c,'layers':5}
    x=torch.tensor(geometry.baseline_logits(small),dtype=torch.float64,device=device,requires_grad=True)
    reference=physics.objective(x,{**small,'rcwa_acceleration':'reference'},'hexagonal',[400.,550.,700.],2,48,device,gradient=True)
    shared=physics.objective(x,{**small,'rcwa_acceleration':'auto'},'hexagonal',[400.,550.,700.],2,48,device,gradient=True)
    power_error=max(abs(a[k]-b[k]) for a,b in zip(reference['rows'],shared['rows']) for k in ('reflectance','power_into_substrate'))
    gradient_error=float((reference['gradient']-shared['gradient']).abs().max().cpu())
    assert power_error<1e-8 and torch.allclose(reference['gradient'],shared['gradient'],atol=1e-7,rtol=2e-5)
    report['checks'].append(dict(name='M2_Q48_Nz5_reference_shared_loss_gradient',passed=True,
                                power_max_abs_error=power_error,gradient_max_abs_error=gradient_error))
    d=torch.sin(torch.arange(1,x.numel()+1,dtype=torch.float64,device=device))
    d=d/torch.linalg.vector_norm(d)
    h=1e-5
    values=[physics.objective(x.detach()+sign*h*d,small,'hexagonal',[400.,550.,700.],2,48,device,gradient=False)['objective'] for sign in (-1,1)]
    fd=(values[1]-values[0])/(2*h)
    ad=float((shared['gradient']*d).sum().cpu())
    error=abs(fd-ad)
    assert error<=1e-6+.01*max(abs(fd),abs(ad))
    report['checks'].append(dict(name='independent_finite_difference_in_logits',passed=True,
                                finite_difference=fd,autodiff=ad,abs_error=error))
    print('PASS material, four cases, small reference/shared and finite difference',flush=True)
    if args.check_m8_initial:
        profile=geometry.radii_values(geometry.baseline_logits(c),c)
        top=profile[0]
        ratio=(profile[-1]-profile[0])/(c['radius_max_nm']-profile[0])
        bottom=min(c['radius_max_nm']-1e-7,max(top,top+ratio*(c['radius_max_nm']-top)))
        logits=torch.tensor(geometry.initial_logits(top,bottom,c),dtype=torch.float64,device=device,requires_grad=True)
        print('START production M8 Q512 Nz15: 31 wavelengths forward and gradient',flush=True)
        forward=physics.objective(logits,c,'hexagonal',c['optimization_wavelengths_nm'],8,512,device,gradient=False)
        gradient=physics.objective(logits,c,'hexagonal',c['optimization_wavelengths_nm'],8,512,device,gradient=True)
        error=max(abs(a[k]-b[k]) for a,b in zip(forward['rows'],gradient['rows']) for k in ('reflectance','power_into_substrate'))
        assert error<1e-8 and bool(torch.isfinite(gradient['gradient']).all())
        report['checks'].append(dict(name='production_M8_Q512_Nz15_initial_31_wavelengths',passed=True,
            objective=gradient['objective'],forward_gradient_power_max_abs_error=error,
            gradient_norm=float(torch.linalg.vector_norm(gradient['gradient']).cpu()),
            forward_seconds=forward['seconds'],gradient_seconds=gradient['seconds'],
            minimum_motheye_absorptance=min(r['motheye_absorptance'] for r in gradient['rows']),
            reflectance_range=[min(r['reflectance'] for r in gradient['rows']),max(r['reflectance'] for r in gradient['rows'])]))
        print('PASS production M8 initial forward/gradient, all 31 wavelengths physical',flush=True)
    report['status']='passed'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
