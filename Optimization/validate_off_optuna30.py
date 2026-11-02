"""Validate OFF defaults, gradients, checkpoints and the original Optuna search.

This is a small-order connection test, not a production-order convergence test.
CUDA is used only when explicitly requested and available, without CPU fallback.
"""
from __future__ import annotations
import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
for variable in ('MOTHEYE_DEPENDENCY_PATH','RCWA_OPT_DEPENDENCY_PATH'):
    if os.environ.get(variable):sys.path.append(os.environ[variable])

def package(name,folder):
    spec=importlib.util.spec_from_file_location(name,folder/'__init__.py',submodule_search_locations=[str(folder)])
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module

def model(material,relative):
    case=HERE/relative
    if material=='Cr':name='Optimization.cr_motheye'
    else:
        sys.path.insert(0,str(case))
        name='off_connection_'+material
        package(name,case/'optimizer')
    parts=('common','geometry','physics','engine','run' if material=='Cr' else 'driver')
    return case,*(importlib.import_module(name+'.'+part) for part in parts)

def check_projectors(device,order,grid):
    import torch
    from rcwa_solver_auto import AutoRCWA,NVMOptions,Lattice,GroupTheoryOptions,OutputSpec
    from rcwa_ext.acceleration import d6
    frozen_off=package('frozen_off_projectors',HERE/'Al/al_motheye_P200_H500_Nz15_normal_off_v1/rcwa_ext')
    frozen_on=package('frozen_on_projectors',HERE/'optimization_integration_validation/rcwa_ext')
    results=[]
    assert NVMOptions().normal_mode=='off'
    for lattice in ('square','hexagonal'):
        for mode,frozen in [('off',frozen_off),('on',frozen_on)]:
            sims=[]
            for module,is_new in [(sys.modules['rcwa_solver_auto'],True),(frozen,False)]:
                opts={'grid':(grid,grid),'use_lanczos':False}
                if is_new:opts['normal_mode']=mode
                sim=module.AutoRCWA(freq=.4,order=[order]*2,
                    lattice=module.Lattice.square(1.) if lattice=='square' else module.Lattice.triangular(1.),
                    nvm=module.NVMOptions(**opts),
                    group_theory=module.GroupTheoryOptions(enabled=True,symmetry='c2v' if lattice=='square' else 'd6',polarization='x',strict=True),
                    outputs=module.OutputSpec(smatrix_size='half',fields='none'),
                    dtype=torch.complex128,device=device,**({'acceleration':'reference'} if is_new else {}))
                sim.add_input_layer(eps=1.);sim.add_output_layer(eps=2.5**2);sim.set_incident_angle(0.,0.)
                sims.append(sim)
            center=d6.center_of(sims[0]);radius=torch.tensor(.23,dtype=torch.float64,device=device,requires_grad=True)
            p=[s._projection_matrix(radius,(center,),nx=grid,ny=grid) for s in sims]
            error=float((p[0]-p[1]).abs().max().detach().cpu())
            assert error<1e-12,(lattice,mode,error)
            if mode=='off':assert not p[0].requires_grad
            if lattice=='hexagonal':
                plan=d6.get_plan(sims[0])
                a=d6.native_inputs(sims[0],radius,2.5,grid,plan)
                assert float((a[2]-plan.vector(p[1])).abs().max().detach().cpu())<1e-12
                if mode=='off':
                    old_d6=importlib.import_module(frozen.__name__+'.acceleration.d6')
                    b=old_d6.native_inputs(sims[1],radius,2.5,grid,old_d6.get_plan(sims[1]))
                    assert max(float((x-y).abs().max().detach().cpu()) for x,y in zip(a,b))<1e-12
            results.append(dict(lattice=lattice,normal_mode=mode,frozen_projector_max_abs=error,passed=True))
    return results

def check_model(material,relative,device,order,grid,work):
    import torch
    case,common,geometry,physics,engine,driver=model(material,relative)
    c=common.load_config(case/'config.json')
    assert c['normal_mode']=='off'
    c.update(optimization_order=order,optimization_grid=grid,optimization_wavelengths_nm=[400.,550.,700.],
             adam_steps=2,trials=3,checkpoint_layers=False)
    x=torch.tensor(geometry.baseline_logits(c),dtype=torch.float64,device=device,requires_grad=True)
    direction=torch.sin(torch.arange(1,x.numel()+1,dtype=x.dtype,device=device));direction-=direction.mean();direction/=direction.norm()
    results=[]
    production_height=c['height_nm']
    for lattice in (['hexagonal'] if material=='Cr' else ['square','hexagonal']):
        # Thin square samples avoid low-order metallic resonances in this
        # additional regression. Requested hexagonal cases keep H=500 nm.
        c={**c,'height_nm':50. if lattice=='square' else production_height}
        print(f'CHECK {material}/{lattice}: reference, shared, checkpoint and FD',flush=True)
        waves=c['optimization_wavelengths_nm']
        ref=physics.objective(x,{**c,'rcwa_acceleration':'reference'},lattice,waves,order,grid,device,gradient=True)
        shared=physics.objective(x,c,lattice,waves,order,grid,device,gradient=True)
        cp=physics.objective(x,{**c,'checkpoint_layers':True},lattice,waves,order,grid,device,gradient=True)
        assert abs(ref['objective']-shared['objective'])<1e-10
        assert torch.allclose(ref['gradient'],shared['gradient'],atol=1e-8,rtol=1e-6)
        assert torch.allclose(cp['gradient'],shared['gradient'],atol=1e-8,rtol=1e-6)
        h=1e-5
        fd_values=[physics.objective(x.detach()+sign*h*direction,{**c,'rcwa_acceleration':'reference'},lattice,waves,order,grid,device,gradient=False)['objective'] for sign in (1,-1)]
        fd=(fd_values[0]-fd_values[1])/(2*h)
        ad=float(torch.dot(shared['gradient'],direction).cpu())
        assert abs(fd-ad)<1e-7+1e-4*max(abs(fd),abs(ad)),(material,lattice,fd,ad)
        # Reuse one evaluator across OFF/ON/OFF without reusing the other field's ports.
        evaluator=physics._spectral_evaluator
        off=evaluator.cache(c,lattice,waves,order,grid,device)
        on=evaluator.cache({**c,'normal_mode':'on'},lattice,waves,order,grid,device)
        assert off is not on and off['template'].normal_mode=='off' and on['template'].normal_mode=='on'
        assert evaluator.cache(c,lattice,waves,order,grid,device) is off
        radii=geometry.radii_tensor(x.detach(),c)
        streamed=physics.solve_verify(radii,{**c,'rcwa_acceleration':'reference'},lattice,550.,order,grid,device)
        assert abs(streamed['reflectance']-ref['rows'][1]['reflectance'])<1e-10
        results.append(dict(lattice=lattice,height_nm=c['height_nm'],objective=shared['objective'],gradient_max_abs=float((ref['gradient']-shared['gradient']).abs().max().cpu()),finite_difference_abs_error=abs(fd-ad),passed=True))

    c={**c,'height_nm':production_height}
    print(f'CHECK {material}: original Optuna lr/init_top/delta_ratio search',flush=True)
    requests=[]
    def fake_worker(folder,request):
        params=request['params'];requests.append(dict(params))
        top=params['init_top'];bottom=min(c['radius_max_nm']-1e-7,top+params['delta_ratio']*(c['radius_max_nm']-top))
        logits=geometry.initial_logits(top,bottom,c)
        return dict(status='completed',best_loss=.1+params['lr'],best_logits=logits,
                    radii_nm=geometry.radii_values(logits,c),best_step=0,completed_steps=2)
    study_folder=work/material/'optuna';study_folder.mkdir(parents=True)
    with patch.object(driver,'launch',side_effect=fake_worker):
        study,status=driver.optimize_lattice(study_folder,dict(config=c,lattice='hexagonal'),None)
    assert status=='completed' and len(study.trials)==3
    assert all(set(p)=={'lr','init_top','delta_ratio'} for p in requests)
    assert len({(p['init_top'],p['delta_ratio']) for p in requests})>1

    print(f'CHECK {material}: actual Adam, interruption and resume',flush=True)
    params={'lr':.003,'init_top':8.,'delta_ratio':(92.-8.)/(c['radius_max_nm']-8.)}
    base=dict(config=c,lattice='hexagonal',params=params,signature='small-off-connection-validation',trial_number=0,deadline=None)
    full=engine.train({**base,'trial_folder':str(work/material/'full')},device)
    interrupted={**base,'trial_folder':str(work/material/'resumed')}
    count=0
    def stop_after_one(*args,**kwargs):
        nonlocal count
        if count==1:raise KeyboardInterrupt()
        count+=1
        return physics.objective(*args,**kwargs)
    try:engine.train(interrupted,device,evaluator=stop_after_one)
    except KeyboardInterrupt:pass
    else:raise AssertionError('The test did not interrupt')
    resumed=engine.train(interrupted,device)
    assert full['status']==resumed['status']=='completed'
    assert abs(full['best_loss']-resumed['best_loss'])<1e-12
    assert torch.allclose(torch.tensor(full['best_logits']),torch.tensor(resumed['best_logits']),atol=1e-10,rtol=0)
    return dict(material=material,status='passed',cases=results,optuna_parameter_sets=requests,adam_resume_passed=True)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='cpu')
    p.add_argument('--order',type=int,default=2)
    p.add_argument('--grid',type=int,default=48)
    p.add_argument('--materials',default='Al,Ti,Cr,Cu,Au')
    p.add_argument('--output',type=Path,default=HERE/'results/off_optuna30_connection.json')
    a=p.parse_args(argv)
    import torch
    from run_metal_off_optuna30 import CASES
    requested=a.materials.split(',')
    if any(m not in CASES for m in requested):p.error('Unknown material')
    device=torch.device(a.device)
    if device.type=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('Requested CUDA is unavailable; no CPU fallback.')
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32=False
    torch.set_num_threads(1)
    report=dict(status='running',device=str(device),cuda_execution_performed=device.type=='cuda',order=a.order,grid=a.grid,layers=15,production_order_convergence_test=False)
    try:
        report['projectors']=check_projectors(device,a.order,a.grid)
        report['materials']=[]
        with tempfile.TemporaryDirectory(prefix='off-optuna30-') as directory:
            for material in requested:
                report['materials'].append(check_model(material,CASES[material][0],device,a.order,a.grid,Path(directory)))
        report['status']='passed'
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=report['status'],cuda_execution_performed=report['cuda_execution_performed'],output=str(a.output)),indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
