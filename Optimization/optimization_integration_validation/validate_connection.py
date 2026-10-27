"""Validate production objective, logits VJP, Adam and checkpoint integration.

Uses frozen, byte-identical production geometry/physics/engine components.
Original active experiments are never modified. This is not an Optuna search
or a Fourier-order convergence proof. Cr is intentionally unsupported.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import platform
import time
import traceback
import torch
from benchmark_device import configure_device
from fast_objective import FastEvaluator, load_model, sync
from source_integrity import POLICY as SOURCE_HASH_POLICY, require_frozen_sources, text_sha256

ROOT = Path(__file__).resolve().parent
TOL = dict(loss_atol=1e-8, gradient_atol=1e-7, gradient_rtol=2e-5,
           logits_atol=2e-6, logits_rtol=2e-6, resume_logits_atol=2e-7,
           state_atol=1e-8, state_rtol=2e-5, fd_atol=1e-6, fd_rtol=3e-4)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def value_comparison(a, b, label):
    loss_error = abs(a['objective']-b['objective'])
    gradient_error = None
    passed = math.isfinite(a['objective']+b['objective']) and loss_error <= TOL['loss_atol']
    if a['gradient'] is not None and b['gradient'] is not None:
        x,y = a['gradient'].detach().cpu(), b['gradient'].detach().cpu()
        gradient_error = float((x-y).abs().max())
        passed = passed and bool(torch.isfinite(x).all() and torch.isfinite(y).all()
            and torch.allclose(x,y,atol=TOL['gradient_atol'],rtol=TOL['gradient_rtol']))
    else:
        passed = passed and a['gradient'] is None and b['gradient'] is None
    assert len(a['rows']) == len(b['rows'])
    power_error = max(abs(x[k]-y[k]) for x,y in zip(a['rows'],b['rows'])
                      for k in ('reflectance','transmittance'))
    metadata_passed = all(x['wavelength_nm']==y['wavelength_nm']
        and x['retained_scalar_harmonics']==y['retained_scalar_harmonics']
        and x['eigen_dimension']==y['eigen_dimension'] for x,y in zip(a['rows'],b['rows']))
    if 'material_index' in a['rows'][0]:
        metadata_passed = metadata_passed and all(abs(x['material_index']-y['material_index'])<1e-14
                                                  for x,y in zip(a['rows'],b['rows']))
    passed = passed and power_error <= 1e-8 and metadata_passed
    result = dict(label=label,passed=bool(passed),loss_abs_error=loss_error,
                  logits_gradient_max_abs_error=gradient_error,power_max_abs_error=power_error,
                  metadata_passed=metadata_passed)
    if not passed:
        raise AssertionError(result)
    return result


def max_tensor_error(a,b,atol,rtol):
    a,b = torch.as_tensor(a,dtype=torch.float64).cpu(),torch.as_tensor(b,dtype=torch.float64).cpu()
    assert a.shape==b.shape and bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    error=float((a-b).abs().max()) if a.numel() else 0.
    if not torch.allclose(a,b,atol=atol,rtol=rtol):
        raise AssertionError(dict(max_abs_error=error,atol=atol,rtol=rtol))
    return error


def tree_check(a,b,path='state'):
    if torch.is_tensor(a):
        if path.endswith('.step'):
            assert torch.equal(a.cpu(),b.cpu()),path
            return 0.
        return max_tensor_error(a,b,TOL['state_atol'],TOL['state_rtol'])
    if isinstance(a,dict):
        assert a.keys()==b.keys(),path
        return max([tree_check(a[k],b[k],path+'.'+str(k)) for k in a] or [0.])
    if isinstance(a,(tuple,list)):
        assert len(a)==len(b),path
        return max([tree_check(x,y,path+'.'+str(i)) for i,(x,y) in enumerate(zip(a,b))] or [0.])
    if isinstance(a,float):
        # ReduceLROnPlateau stores +inf as its mode_worse sentinel.
        if not math.isfinite(a) or not math.isfinite(b):
            assert a == b and math.isinf(a), path
            return 0.
        assert abs(a-b)<=TOL['state_atol']+TOL['state_rtol']*abs(a),path
        return abs(a-b)
    assert a==b,path
    return 0.


def train_comparison(folder_a,folder_b,result_a,result_b,resume=False):
    a=torch.load(folder_a/'adam_checkpoint.pt',map_location='cpu',weights_only=True)
    b=torch.load(folder_b/'adam_checkpoint.pt',map_location='cpu',weights_only=True)
    assert a['next_step']==b['next_step']
    assert result_a['completed_steps']==result_b['completed_steps']==a['next_step']
    tolerance=TOL['resume_logits_atol'] if resume else TOL['logits_atol']
    logits_error=max_tensor_error(a['logits'],b['logits'],tolerance,TOL['logits_rtol'])
    best_error=max_tensor_error(a['best_logits'],b['best_logits'],tolerance,TOL['logits_rtol'])
    radius_error=max_tensor_error(result_a['radii_nm'],result_b['radii_nm'],1e-4,2e-6)
    assert abs(result_a['best_loss']-result_b['best_loss'])<=TOL['loss_atol']
    assert a['best_step']==b['best_step'] and a['stale']==b['stale']
    optimizer_error=tree_check(a['optimizer'],b['optimizer'])
    scheduler_error=tree_check(a['scheduler'],b['scheduler'])
    assert len(a['history'])==len(b['history'])
    history_error=max([abs(x['objective']-y['objective']) for x,y in zip(a['history'],b['history'])] or [0.])
    assert history_error<=TOL['loss_atol']
    assert [x['learning_rate'] for x in a['history']]==[x['learning_rate'] for x in b['history']]
    return dict(passed=True,final_logits_max_abs_error=logits_error,
                best_logits_max_abs_error=best_error,radii_nm_max_abs_error=radius_error,
                history_loss_max_abs_error=history_error,optimizer_state_max_abs_error=optimizer_error,
                scheduler_state_max_abs_error=scheduler_error,completed_steps=a['next_step'])


def run_case(args,material,lattice,folder,source_hashes,profile):
    model=load_model(material)
    c=model.common.load_config(model.config_path)
    c.update(layers=args.layers,optimization_order=args.order,optimization_grid=args.grid,
             optimization_wavelengths_nm=args.waves,adam_steps=args.steps,
             patience=args.steps+5,checkpoint_layers=False)
    device=torch.device(args.device)
    fast=FastEvaluator(model)
    reference=model.physics.objective
    x=torch.tensor(model.geometry.baseline_logits(c),dtype=torch.float64,device=device,requires_grad=True)
    identity=dict(material=material,lattice=lattice,config=c,sources=source_hashes,
                  tol=TOL,device=args.device,steps=args.steps,profile=profile)
    tests=[]
    print(f'CASE {material}/{lattice}: loss, all logits gradients, layer checkpoint OFF/ON',flush=True)
    parity={}
    for checkpointed in (False,True):
        cc={**c,'checkpoint_layers':checkpointed}
        print(f'  reference checkpoint={checkpointed}',flush=True)
        a=reference(x,cc,lattice,args.waves,args.order,args.grid,device,gradient=True)
        print(f'  shared checkpoint={checkpointed}',flush=True)
        b=fast(x,cc,lattice,args.waves,args.order,args.grid,device,gradient=True)
        tests.append(value_comparison(a,b,f'reference/shared checkpoint={checkpointed}'))
        parity[checkpointed]=(a,b)
    tests.append(value_comparison(parity[False][0],parity[True][0],'reference checkpoint OFF/ON'))
    tests.append(value_comparison(parity[False][1],parity[True][1],'shared checkpoint OFF/ON'))
    gauge=float(parity[False][1]['gradient'].sum().detach().cpu())
    assert abs(gauge)<1e-7
    profiles={}
    boundary=model.geometry.initial_logits(c['radius_min_nm']+.001*(c['radius_max_nm']-c['radius_min_nm']),
        c['radius_max_nm']-.001*(c['radius_max_nm']-c['radius_min_nm']),c)
    profiles['near_bounds']=boundary
    if profile is not None:
        if len(profile['logits'])!=args.layers+1:
            raise ValueError('Profile logits must have layers+1 entries; use matching --layers.')
        profiles['supplied_best']=profile['logits']
    for label,values in profiles.items():
        z=torch.tensor(values,dtype=torch.float64,device=device,requires_grad=True)
        a=reference(z,c,lattice,args.waves,args.order,args.grid,device,gradient=True)
        b=fast(z,c,lattice,args.waves,args.order,args.grid,device,gradient=True)
        tests.append(value_comparison(a,b,label))
    print(f'CASE {material}/{lattice}: independent production-forward finite differences in logits',flush=True)
    direction=torch.linspace(-.7,.9,args.layers+1,dtype=torch.float64,device=device)
    direction-=direction.mean()
    direction/=torch.linalg.vector_norm(direction)
    fd_waves=args.fd_waves
    g=fast(x,c,lattice,fd_waves,args.order,args.grid,device,gradient=True)['gradient']
    ad=float((g@direction).detach().cpu())
    differences=[]
    for h in (1e-4,5e-5):
        with torch.no_grad():
            plus=reference(x.detach()+h*direction,c,lattice,fd_waves,args.order,args.grid,device,gradient=False)['objective']
            minus=reference(x.detach()-h*direction,c,lattice,fd_waves,args.order,args.grid,device,gradient=False)['objective']
        differences.append((plus-minus)/(2*h))
    close=lambda a,b:abs(a-b)<=TOL['fd_atol']+TOL['fd_rtol']*max(abs(a),abs(b))
    fd=dict(passed=close(ad,differences[-1]) and close(*differences),wavelengths_nm=fd_waves,
            autodiff=ad,fd_h=differences[0],fd_half_h=differences[1],
            abs_error=abs(ad-differences[-1]),full_objective_coverage=fd_waves==args.waves)
    assert fd['passed'],fd
    # Verify a second supported objective weighting without changing the physics.
    if len(args.waves)>1:
        cc={**c,'objective_weighting':'trapezoidal'}
        print('  reference trapezoidal weighting',flush=True)
        a=reference(x,cc,lattice,args.waves,args.order,args.grid,device,gradient=True)
        print('  shared trapezoidal weighting',flush=True)
        b=fast(x,cc,lattice,args.waves,args.order,args.grid,device,gradient=True)
        tests.append(value_comparison(a,b,'trapezoidal weights'))
    train_c={**c,'checkpoint_layers':True}
    params=dict(lr=.03,init_top=8.,delta_ratio=.8)
    request=dict(config=train_c,lattice=lattice,params=params,trial_number=0)
    def request_for(backend,directory):
        return dict(request,trial_folder=str(directory),
                    signature=digest(dict(identity,backend=backend,params=params)))
    ref_dir,fast_dir,resume_dir=(folder/v for v in ('reference','shared','shared_resumed'))
    print(f'CASE {material}/{lattice}: original Adam engine, {args.steps} steps, reference/shared',flush=True)
    sync(device);start=time.perf_counter()
    ref_result=model.engine.train(request_for('reference',ref_dir),device,evaluator=reference)
    sync(device);ref_seconds=time.perf_counter()-start
    sync(device);start=time.perf_counter()
    fast_result=model.engine.train(request_for(FastEvaluator.backend_name,fast_dir),device,evaluator=fast)
    sync(device);fast_seconds=time.perf_counter()-start
    trajectory=train_comparison(ref_dir,fast_dir,ref_result,fast_result)
    print(f'CASE {material}/{lattice}: interrupt after step 1, recreate evaluator and resume',flush=True)
    resume_request=request_for(FastEvaluator.backend_name,resume_dir)
    if not (resume_dir/'adam_checkpoint.pt').exists():
        calls=0
        first=FastEvaluator(model)
        def interrupt(*a,**kw):
            nonlocal calls
            if calls==1:
                raise KeyboardInterrupt('Controlled validation interruption after one saved Adam update.')
            calls+=1
            return first(*a,**kw)
        try:
            model.engine.train(resume_request,device,evaluator=interrupt)
            raise AssertionError('Expected controlled interruption.')
        except KeyboardInterrupt:
            checkpoint_state=torch.load(resume_dir/'adam_checkpoint.pt',map_location='cpu',weights_only=True)
            assert checkpoint_state['next_step']==1
    resumed=model.engine.train(resume_request,device,evaluator=FastEvaluator(model))
    resume_test=train_comparison(fast_dir,resume_dir,fast_result,resumed,resume=True)
    before=(resume_dir/'adam_checkpoint.pt').read_bytes()
    try:
        model.engine.train({**resume_request,'signature':'incompatible-backend'},device,evaluator=reference)
        raise AssertionError('Wrong checkpoint identity was accepted.')
    except ValueError as error:
        assert 'identity' in str(error)
    assert before==(resume_dir/'adam_checkpoint.pt').read_bytes()
    best=torch.tensor(fast_result['best_logits'],dtype=torch.float64,device=device)
    print(f'CASE {material}/{lattice}: original streaming verification of the best shape',flush=True)
    with torch.no_grad():
        radii=model.geometry.radii_tensor(best,train_c)
        rows=[dict(wavelength_nm=w,**model.physics.solve_verify(
            radii,train_c,lattice,w,args.order,args.grid,device)) for w in args.waves]
    observed=sum(w*row['reflectance'] for w,row in zip(
        model.physics.weights(args.waves,train_c['objective_weighting']),rows))
    best_shared=fast(best,train_c,lattice,args.waves,args.order,args.grid,device,gradient=False)
    verification=value_comparison(dict(objective=observed,gradient=None,rows=rows),
        best_shared,'original same-order streaming verification/shared best shape')
    assert abs(observed-fast_result['best_loss'])<=TOL['loss_atol']
    return dict(status='passed',material=material,lattice=lattice,order=args.order,grid=args.grid,layers=args.layers,
        objective_wavelengths_nm=args.waves,fd_wavelengths_nm=fd_waves,steps=args.steps,
        model_config=c,parity=tests,softmax_gauge_gradient_sum=gauge,finite_difference=fd,
        adam_trajectory=trajectory,checkpoint_resume=resume_test,
        wrong_backend_checkpoint_rejected=True,best_loss_shape_abs_error=abs(observed-fast_result['best_loss']),
        reference_train_wall_seconds=ref_seconds,shared_train_wall_seconds=fast_seconds,
        frozen_production_copy_verified=True,optuna_controller_tested=False,
        same_order_streaming_verification=verification,
        order_convergence_tested=False,checkpoint_layers_enabled_in_training=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--suite',choices=['quick','production'],default='quick')
    parser.add_argument('--materials',nargs='+',choices=['n2.5','PMMA'],default=['n2.5','PMMA'])
    parser.add_argument('--lattices',nargs='+',choices=['square','hexagonal'],default=['square','hexagonal'])
    parser.add_argument('--order',type=int)
    parser.add_argument('--grid',type=int)
    parser.add_argument('--layers',type=int)
    parser.add_argument('--steps',type=int)
    parser.add_argument('--wavelengths')
    parser.add_argument('--fd-wavelengths',default='400,550,700')
    parser.add_argument('--profile-json',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'results/integration_check')
    args=parser.parse_args()
    args.device=str(configure_device(parser,args.device))
    torch.set_num_threads(1)
    quick=args.suite=='quick'
    args.order=args.order if args.order is not None else (2 if quick else 8)
    args.grid=args.grid if args.grid is not None else (48 if quick else 512)
    args.layers=args.layers if args.layers is not None else (5 if quick else 15)
    args.steps=args.steps if args.steps is not None else (2 if quick else 3)
    if min(args.order,args.layers,args.steps)<2 or args.grid<max(32,4*args.order+4):
        parser.error('Use order>=2, layers>=2, steps>=2 and a sufficient NVM grid.')
    args.waves=load_model('n2.5').common.wavelengths(args.wavelengths or ('400,550,700' if quick else '400:700:10'))
    args.fd_waves=load_model('n2.5').common.wavelengths(args.fd_wavelengths)
    profile=None
    if args.profile_json:
        raw=args.profile_json.read_bytes()
        data=json.loads(raw)
        values=data.get('best_logits',data.get('logits'))
        if values is None:
            parser.error('Profile JSON must contain best_logits or logits.')
        profile=dict(path=str(args.profile_json.resolve()),sha256=hashlib.sha256(raw).hexdigest(),logits=values)
    integrity=require_frozen_sources(ROOT)
    print(f'SOURCE INTEGRITY PASSED: {integrity["checked_files"]} files; '
          f'{len(integrity["newline_or_bom_only_files"])} newline/BOM-only differences',flush=True)
    source_hashes={p.relative_to(ROOT).as_posix():text_sha256(p)
                   for p in sorted(ROOT.rglob('*.py')) if '__pycache__' not in p.parts and 'results' not in p.parts}
    spec=importlib.util.find_spec('torcwa')
    if not spec or not spec.origin:
        raise RuntimeError('torcwa is required.')
    source_hashes.update({'installed_torcwa/'+p.name:text_sha256(p)
        for p in sorted(Path(spec.origin).parent.glob('*.py'))})
    runtime=dict(device=args.device,torch=str(torch.__version__),python=platform.python_version(),
                 torcwa=importlib.metadata.version('torcwa'),
                 cuda=torch.version.cuda,platform=platform.platform(),
                 gpu=torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else None)
    if args.device.startswith('cuda'):
        runtime['gpu_uuid']=str(getattr(torch.cuda.get_device_properties(args.device),'uuid','unavailable'))
    config=dict(suite=args.suite,order=args.order,grid=args.grid,layers=args.layers,steps=args.steps,
                waves=args.waves,fd_waves=args.fd_waves,materials=args.materials,lattices=args.lattices,
                runtime=runtime,sources=source_hashes,source_hash_policy=SOURCE_HASH_POLICY,profile=profile,tolerances=TOL)
    out=args.output.resolve()
    if any(out.is_relative_to(ROOT/p) for p in ('Optimization','pmma_case','studies','rcwa_ext')):
        parser.error('Output must not be inside the frozen source directories.')
    identity=out/'validation_identity.json'
    if out.exists() and any(out.iterdir()) and not identity.exists():
        parser.error('Use a new empty output folder for integration validation.')
    out.mkdir(parents=True,exist_ok=True)
    if identity.exists():
        if json.loads(identity.read_text())!=config:
            parser.error('Existing validation settings/source/runtime differ; use a new output folder.')
    else:
        write_json(identity,config)
    summary=dict(status='running',configuration=config,cuda_execution_performed=args.device.startswith('cuda'),
                 production_fast_path_enabled=False,source_integrity=integrity,cases=[])
    for material in args.materials:
        for lattice in args.lattices:
            folder=out/(material.replace('.','')+'_'+lattice)
            folder.mkdir(exist_ok=True)
            path=folder/'report.json'
            if path.exists() and json.loads(path.read_text()).get('status')=='passed':
                result=json.loads(path.read_text())
                print(f'REUSE passed case {material}/{lattice}',flush=True)
            else:
                try:
                    result=run_case(args,material,lattice,folder,source_hashes,profile)
                except Exception as error:
                    result=dict(status='failed',material=material,lattice=lattice,error=str(error),
                                traceback=traceback.format_exc())
                write_json(path,result)
            summary['cases'].append(result)
            write_json(out/'report.json',summary)
            print(f'{result["status"].upper()} {material}/{lattice}',flush=True)
    summary['status']='passed' if all(c['status']=='passed' for c in summary['cases']) else 'failed'
    write_json(out/'report.json',summary)
    print(json.dumps(dict(status=summary['status'],cuda_execution_performed=summary['cuda_execution_performed'],
                         output=str(out),cases=[dict(material=c['material'],lattice=c['lattice'],status=c['status'])
                                               for c in summary['cases']]),indent=2),flush=True)
    return 0 if summary['status']=='passed' else 1


if __name__=='__main__':
    raise SystemExit(main())

