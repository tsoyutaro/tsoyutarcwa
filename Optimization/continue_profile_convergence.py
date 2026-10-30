"""Evaluate frozen OFF-optimized profiles with original ON sources across orders.

No optimization is started. External profile/config hashes and source identities
are recorded. Nonphysical conditions advance to the next order; confirmed OOM
stops the remaining material conditions by default.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback

TOOL='rcwa-frozen-off-profile-on-convergence-v1'
OOM={'cuda_oom','cpu_oom'}
CACHEABLE={'completed','nonphysical','cuda_oom','cpu_oom'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def numbers(text,integer=False):
    if ':' in text:
        a,b,h=map(float,text.split(':'))
        if not all(math.isfinite(x) for x in (a,b,h)) or h<=0 or b<a:
            raise ValueError('Invalid inclusive range: '+text)
        n=round((b-a)/h)
        if n>10000 or abs(a+n*h-b)>1e-8:raise ValueError('Nonintegral range intervals.')
        result=[a+i*h for i in range(n+1)]
    else:result=list(map(float,text.split(',')))
    if not result or any(not math.isfinite(x) or x<=0 for x in result):raise ValueError('Numbers must be positive and finite.')
    if integer and any(x!=int(x) for x in result):raise ValueError('Orders/grids must be integers.')
    return sorted(set(map(int,result))) if integer else sorted(set(result))


def import_case(case):
    sys.path.insert(0,str(case))
    if os.environ.get('MOTHEYE_DEPENDENCY_PATH'):
        sys.path.append(os.environ['MOTHEYE_DEPENDENCY_PATH'])
    from optimizer import common,physics
    import runtime_policy
    return common,physics,runtime_policy


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def profile_inputs(request,config):
    if not request.get('profile_json') or not request.get('profile_config'):
        raise ValueError('Both profile_json and profile_config are required.')
    profile_path=Path(request['profile_json']).resolve()
    config_path=Path(request['profile_config']).resolve()
    data=read(profile_path);origin=read(config_path)
    if origin.get('normal_field_mode')!='off':
        raise ValueError('The external profile must come from a documented OFF case.')
    keys=('material','material_model','substrate_mode','layers','period_nm','height_nm',
          'radius_min_nm','radius_max_nm','softmax_gap_floor')
    for key in keys:
        if config.get(key)!=origin.get(key):
            raise ValueError('External profile geometry/material mismatch: '+key)
    if config['material_provenance'].get('csv_sha256_lf')!=origin.get('material_provenance',{}).get('csv_sha256_lf'):
        raise ValueError('External profile and ON evaluation material CSV differ.')
    logits=data.get('best_logits',data.get('logits',data.get('best',{}).get('logits')))
    if (not isinstance(logits,list) or len(logits)!=config['layers']+1 or
            any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in logits)):
        raise ValueError('Expected finite layers+1 logits in the external profile.')
    return logits,dict(profile_json=str(profile_path),profile_json_sha256=file_sha(profile_path),
        profile_config=str(config_path),profile_config_sha256=file_sha(config_path),
        profile_origin_normal_field_mode='off',
        profile_sha256=digest(dict(logits=logits)))


def check_profile_inputs(context):
    origin=context['profile_origin']
    for path_key,hash_key in [('profile_json','profile_json_sha256'),('profile_config','profile_config_sha256')]:
        if file_sha(origin[path_key])!=origin[hash_key]:
            raise ValueError('Frozen external profile input changed: '+origin[path_key])


def training_files(run,lattice):
    paths=[run/'manifest.json',run/'resolved_config.json',run/'workflow.json',run/lattice/'report.json']
    paths+=list((run/lattice/'trials').glob('*/best.json'))
    paths+=list((run/lattice/'trials').glob('*/adam_checkpoint.pt'))
    paths+=list((run/lattice).glob('optuna_journal*'))
    return {str(path):file_sha(path) for path in paths if path.is_file()}


def prepare_child(request):
    case=Path(request['case_dir'])
    common,physics,policy=import_case(case)
    config=common.load_config(Path(request['config']))
    if config.get('material')!=request['material']:
        raise ValueError('Material label differs from configuration.')
    if config.get('normal_field_mode')=='off':
        raise ValueError('case_dir/config must select the original ON solver case.')
    from rcwa_ext.acceleration import d6
    normal_source=inspect.getsource(d6.native_inputs)
    if '_smoothstep(' not in normal_source:
        raise ValueError('Expected original smoothstep ON normal-field sources.')
    previous=read(Path(request['run_root'])/'manifest.json')
    sources=policy.execution_sources(common.sources())
    checked,_=policy.reconcile_manifest(previous,dict(version=1,config=config,source_sha256=sources,
                                                      requested_device=request['device']))
    logits,origin=profile_inputs(request,config)
    if request['lattice'] not in config['lattices']:
        raise ValueError('Lattice is absent from configuration.')
    device=common.device_setup(request['device'],config['cpu_threads'])
    environment=common.runtime_environment(device)
    policy.check_runtime(previous.get('runtime_environment'),environment)
    return dict(status='ready',context=dict(tool=TOOL,tool_sha256=file_sha(__file__),
        material=request['material'],case_dir=str(case),run_root=request['run_root'],
        config=config,source_sha256=sources,training_signature=checked['signature'],
        runtime_environment=environment,device=str(device),lattice=request['lattice'],
        logits=logits,waves=request['waves'],backend=request['backend'],profile_origin=origin,
        evaluator_normal_field_mode='on',normal_field_definition='legacy-smoothstep-grid-coupled',
        normal_field_input_source_sha256=hashlib.sha256(normal_source.encode()).hexdigest(),
        protected_training_files_sha256=training_files(Path(request['run_root']),request['lattice'])))



def worker_child(request):
    context=read(request['context'])
    check_profile_inputs(context)
    common,physics,policy=import_case(Path(context['case_dir']))
    if context['tool_sha256']!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError('Analysis tool source changed during this run.')
    if policy.execution_sources(common.sources())!=context['source_sha256']:
        raise ValueError('Production sources/material CSV changed during this run.')
    device=common.device_setup(context['device'],context['config']['cpu_threads'])
    environment=common.runtime_environment(device)
    if environment!=context['runtime_environment']:
        raise ValueError('Runtime/hardware changed; restart the analysis to use a new identity.')
    import torch
    x=torch.tensor(context['logits'],dtype=torch.float64,device=device)
    c={**context['config'],'rcwa_acceleration':context['backend']}
    result=physics.objective(x,c,context['lattice'],context['waves'],request['order'],request['grid'],device,gradient=False)
    check_profile_inputs(context)
    if len(result['rows'])!=len(context['waves']):raise ValueError('Incomplete wavelength results.')
    return dict(status='completed',mean_R=result['objective'],rows=result['rows'],
        rcwa_seconds=result['seconds'],runtime_environment=environment,
        peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated(device)) if device.type=='cuda' else None)


def child_main(request_file,output):
    request=read(request_file)
    started=time.perf_counter()
    try:
        result=prepare_child(request) if request['mode']=='prepare' else worker_child(request)
    except (Exception,KeyboardInterrupt) as error:
        message=str(error);lower=message.lower()
        from_types=type(error).__name__
        if isinstance(error,KeyboardInterrupt):status='interrupted'
        elif ('cuda out of memory' in lower or 'cuda error: out of memory' in lower or
              from_types=='OutOfMemoryError' and request.get('device','').startswith('cuda')):status='cuda_oom'
        elif (isinstance(error,MemoryError) or from_types=='OutOfMemoryError' or
              'defaultcpuallocator' in lower and 'not enough memory' in lower):status='cpu_oom'
        elif from_types=='NonphysicalResult':status='nonphysical'
        else:status='setup_error' if request['mode']=='prepare' else 'error'
        result=dict(status=status,error_type=from_types,error=message,traceback=traceback.format_exc())
        print(result['traceback'],file=sys.stderr,flush=True)
    result['worker_seconds']=time.perf_counter()-started
    write(output,result)
    return 0


def launch(folder,request):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    req,res=folder/'request.json',folder/'result.json'
    write(req,request)
    # Prevent a killed worker from being mistaken for an earlier successful run.
    if res.exists():res.unlink()
    with (folder/'worker.log').open('w',encoding='utf-8') as log:
        process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--internal-request',str(req),
                                  '--internal-result',str(res)],stdout=log,stderr=subprocess.STDOUT)
        try:process.wait()
        except BaseException:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            raise
    if not res.exists():
        return dict(status='worker_lost',returncode=process.returncode,
                    error='Worker returned no result; OOM is not confirmed.')
    result=read(res)
    if process.returncode!=0:
        return dict(status='worker_lost',returncode=process.returncode,
                    error='Worker exited abnormally; OOM is not confirmed.',child_result=result)
    return result


def drive_conditions(orders,grids,execute,cache=None,oom_policy='stop-material',retry_failures=False,progress=None):
    cache=cache or {};records=[];oom_points=[];stop_reason=None
    for m in orders:
        skip_order=None
        for q in grids:
            old=cache.get((m,q))
            if old and old['status']=='completed':
                row={**old,'cache_hit':True}
            elif stop_reason:
                row=dict(order=m,grid=q,status='skipped',reason=stop_reason)
            elif skip_order:
                row=dict(order=m,grid=q,status='skipped',reason=skip_order)
            elif any(m>=a and q>=b for a,b in oom_points):
                row=dict(order=m,grid=q,status='skipped',reason='dominated_by_oom_condition')
            else:
                if old and old['status'] in CACHEABLE and not retry_failures:
                    row={**old,'cache_hit':True}
                else:
                    row=dict(order=m,grid=q,**execute(m,q),cache_hit=False)
            records.append(row)
            status=row['status']
            if status=='nonphysical':skip_order='nonphysical_at_same_order'
            elif status in OOM:
                oom_points.append((m,q))
                if oom_policy=='stop-material':stop_reason='remaining_material_conditions_skipped_after_oom'
            elif status=='worker_lost':stop_reason='remaining_material_conditions_skipped_after_worker_loss'
            elif status in ('error','setup_error'):stop_reason='remaining_material_conditions_skipped_after_error'
            elif status=='interrupted':
                if progress:progress(records)
                raise KeyboardInterrupt()
            if progress:progress(records)
    return records


def power_difference(a,b):
    if len(a['rows'])!=len(b['rows']):raise ValueError('Wavelength count mismatch.')
    errors=[]
    for x,y in zip(a['rows'],b['rows']):
        if x['wavelength_nm']!=y['wavelength_nm']:raise ValueError('Wavelength grid mismatch.')
        keys=('reflectance','power_into_substrate') if 'power_into_substrate' in x else ('reflectance','transmittance')
        errors.extend(abs(x[k]-y[k]) for k in keys)
    return max(errors)


def aggregate(records,tolerance):
    good=[r for r in records if r['status']=='completed']
    comparisons=[];reference=None;checks=[];converged=None
    if good:
        ref=max(good,key=lambda r:(r['order'],r['grid']))
        reference=dict(order=ref['order'],grid=ref['grid'])
        for r in good:
            error=power_difference(r,ref)
            comparisons.append(dict(order=r['order'],grid=r['grid'],max_power_abs_difference_from_reference=error,
                is_reference=r is ref,within_tolerance=None if r is ref else error<=tolerance))
        lower_order=[r for r in good if r['grid']==ref['grid'] and r['order']<ref['order']]
        lower_grid=[r for r in good if r['order']==ref['order'] and r['grid']<ref['grid']]
        for axis,candidates,key in [('order',lower_order,'order'),('grid',lower_grid,'grid')]:
            if candidates:
                previous=max(candidates,key=lambda r:r[key])
                error=power_difference(previous,ref)
                checks.append(dict(axis=axis,coarse=[previous['order'],previous['grid']],
                    fine=[ref['order'],ref['grid']],max_power_abs_difference=error,passed=error<=tolerance))
        if len(checks)==2:converged=all(c['passed'] for c in checks)
    return dict(comparison_reference=reference,reference_selection='highest_valid_order_then_grid',
        comparison=comparisons,convergence_checks=checks,numerical_convergence_passed=converged,
        tolerance=tolerance,scope='Finite comparisons of completed physical conditions only; missing axes give null.')


def persist(folder,context,records,state='running'):
    check_profile_inputs(context)
    summary=aggregate(records,context['config']['verification_tolerance'])
    order_pass=next((c['passed'] for c in summary['convergence_checks'] if c['axis']=='order'),None)
    grid_pass=next((c['passed'] for c in summary['convergence_checks'] if c['axis']=='grid'),None)
    preserved=all(Path(p).exists() and file_sha(p)==h for p,h in context['protected_training_files_sha256'].items())
    if not preserved:raise RuntimeError('A protected original optimization record changed.')
    report=dict(status=state,material=context['material'],requested_device=context['device'],
        cuda_execution_performed=context['device'].startswith('cuda'),training_signature=context['training_signature'],
        frozen_logits=context['logits'],wavelengths_nm=context['waves'],counts=dict(Counter(r['status'] for r in records)),
        records=records,profile_origin=context['profile_origin'],evaluator_normal_field_mode=context['evaluator_normal_field_mode'],
        normal_field_definition=context['normal_field_definition'],protected_training_files_unchanged=preserved,
        order_convergence_passed=order_pass,grid_convergence_passed=grid_pass,**summary)
    write(folder/'report.json',report)
    compared={(r['order'],r['grid']):r for r in summary['comparison']}
    fields=['material','order','grid','status','mean_R','max_power_abs_difference_from_reference',
            'within_tolerance','worker_seconds','rcwa_seconds','cache_hit','reason','error']
    with (folder/'convergence.csv.tmp').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in records:
            row={**r,**compared.get((r['order'],r['grid']),{}),'material':context['material']}
            writer.writerow({key:row.get(key) for key in fields})
    (folder/'convergence.csv.tmp').replace(folder/'convergence.csv')
    return report


def defaults(root):
    specs=[]
    for material,low,off_order,config,on_run in [
        ('Al','al',8,'config_hex_trial1.json','hex_trial1_v3'),
        ('Ti','ti',8,'config_hex_trial1_M8_Q512.json','hex_trial1_M8_Q512_v3'),
        ('Cu','cu',12,'config_hex_trial1_M12.json','hex_trial1_M12_v3'),
        ('Au','au',14,'config_hex_trial1_M12.json','hex_trial1_M12_v3')]:
        case=root/material/f'{low}_motheye_P200_H500_Nz15'
        profile_run=root/material/f'{low}_motheye_P200_H500_Nz15_normal_off_v1'/'results'/f'hex_trial1_M{off_order}_Q1024_off_v1'
        specs.append(dict(material=material,case_dir=str(case),config=str(case/config),
            run_root=str(case/'results'/on_run),profile_json=str(profile_run/'hexagonal/trials/trial_0000/best.json'),
            profile_config=str(profile_run/'resolved_config.json')))
    return specs


def acquire(lock):
    lock.mkdir()
    write(lock/'owner.json',dict(tool=TOOL,pid=os.getpid(),host=socket.gethostname()))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--materials',default='Al,Ti,Cu,Au')
    p.add_argument('--orders',default='20,24,28,30,32,34')
    p.add_argument('--grids',default='1024')
    p.add_argument('--wavelengths',default='400,550,700')
    p.add_argument('--lattice',choices=['square','hexagonal'],default='hexagonal')
    p.add_argument('--backend',choices=['auto','matrix','reference'],default='auto')
    p.add_argument('--oom-policy',choices=['stop-material','dominated'],default='stop-material')
    p.add_argument('--retry-failures',action='store_true')
    p.add_argument('--cases-file',type=Path)
    p.add_argument('--internal-request',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--internal-result',type=Path,help=argparse.SUPPRESS)
    args=p.parse_args(argv)
    if args.internal_request:return child_main(args.internal_request,args.internal_result)
    root=args.root.resolve();orders=numbers(args.orders,True);grids=numbers(args.grids,True);waves=numbers(args.wavelengths)
    if any(q<max(32,4*m+4) for m in orders for q in grids):p.error('A requested grid undersamples a retained order.')
    specs=read(args.cases_file) if args.cases_file else defaults(root)
    requested=args.materials.split(',');by_name={x['material']:x for x in specs}
    if any(n not in by_name for n in requested):p.error('A requested material has no case specification.')
    batch=dict(status='running',materials={},oom_policy=args.oom_policy,orders=orders,grids=grids,waves=waves)
    batch_path=root/'same_OFF_shape_ON_convergence'/'summary.json'
    def interrupted(signum,frame):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupted)
    try:
        for name in requested:
            spec={**by_name[name]}
            for key in ('case_dir','config','run_root','profile_json','profile_config'):spec[key]=str(Path(spec[key]).resolve())
            run=Path(spec['run_root']);lock=run/'controller.lock'
            held=False
            try:
                if not (run/'manifest.json').exists():raise ValueError('Saved experiment manifest is missing: '+str(run))
                acquire(lock);held=True
                base=run/args.lattice/'postprocess/profile_convergence'
                preparation=launch(base/'preparation',dict(mode='prepare',**spec,device=args.device,
                    lattice=args.lattice,waves=waves,backend=args.backend))
                if preparation['status']!='ready':
                    batch['materials'][name]=dict(status='setup_error',preparation=preparation)
                    print(name+': setup_error; continuing to next material',flush=True)
                    continue
                context=preparation['context']
                identity=dict(context=context,orders=orders,grids=grids,oom_policy=args.oom_policy)
                folder=base/digest(identity)[:16];folder.mkdir(parents=True,exist_ok=True)
                ctx=folder/'context.json';write(ctx,context)
                report_path=folder/'report.json'
                prior=read(report_path).get('records',[]) if report_path.exists() else []
                cache={(r['order'],r['grid']):r for r in prior}
                def progress(records):persist(folder,context,records)
                def execute(m,q):
                    print(f'{name}: START M={m} Q={q}, {len(waves)} wavelengths',flush=True)
                    result=launch(folder/f'M{m}_Q{q}',dict(mode='evaluate',context=str(ctx),order=m,grid=q,device=context['device']))
                    print(f"{name}: DONE M={m} Q={q}: {result['status']}",flush=True)
                    return result
                records=drive_conditions(orders,grids,execute,cache,args.oom_policy,args.retry_failures,progress)
                report=persist(folder,context,records,'completed_with_skips' if any(r['status']!='completed' for r in records) else 'completed')
                write(run/args.lattice/'latest_profile_convergence.json',dict(report=str(report_path),status=report['status']))
                batch['materials'][name]=dict(status=report['status'],report=str(report_path),
                    counts=report['counts'],comparison_reference=report['comparison_reference'],
                    order_convergence_passed=report['order_convergence_passed'],
                    grid_convergence_passed=report['grid_convergence_passed'],
                    numerical_convergence_passed=report['numerical_convergence_passed'])
                print(name+': '+json.dumps(batch['materials'][name]),flush=True)
            except FileExistsError:
                batch['materials'][name]=dict(status='busy',error='A controller lock exists; it was not removed.')
                print(name+': busy; continuing to next material',flush=True)
            except KeyboardInterrupt:raise
            except Exception as error:
                batch['materials'][name]=dict(status='setup_error',error=str(error),traceback=traceback.format_exc())
                print(name+': '+str(error)+'; continuing to next material',flush=True)
            finally:
                if held:
                    (lock/'owner.json').unlink(missing_ok=True);lock.rmdir()
                write(batch_path,batch)
        batch['status']='completed_with_skips' if any(r['status']!='completed' for r in batch['materials'].values()) else 'completed'
    except KeyboardInterrupt:
        batch['status']='interrupted';write(batch_path,batch)
        print('Interrupted; recorded results can be reused on the next run.',flush=True)
        return 130
    write(batch_path,batch)
    print('Saved '+str(batch_path),flush=True)
    return 1 if any(r['status'] in ('setup_error','busy') for r in batch['materials'].values()) else 0


if __name__=='__main__':
    raise SystemExit(main())
