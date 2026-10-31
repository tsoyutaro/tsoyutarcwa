"""Cross-evaluate saved ON/OFF optimal profiles, reusing matching completed rows.

The physics sources and original optimization records remain unchanged.
Only missing wavelength rows are calculated. OOM stops this material's
remaining conditions, then processing continues with the next material.
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
TOOL='rcwa-cross-profile-cache-v1'
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


def acquire(lock):
    lock.mkdir()
    write(lock/'owner.json',dict(tool=TOOL,pid=os.getpid(),host=socket.gethostname()))


# Standalone controller body, combined with the existing verified utility functions.
MODEL_KEYS = ('material','material_model','substrate_mode','layers','period_nm','height_nm',
              'radius_min_nm','radius_max_nm','softmax_gap_floor','index',
              'normal_field_mode','normal_field_definition','normal_reference_grid')
DIRECTIONS = {'ON_to_OFF':('on','off'), 'OFF_to_ON':('off','on')}


def normal_mode(config):
    return 'off' if config.get('normal_field_mode') == 'off' else 'on'


def model_matches(config, target):
    return (all(config.get(k) == target.get(k) for k in MODEL_KEYS)
            and config.get('material_provenance',{}).get('csv_sha256_lf')
                == target.get('material_provenance',{}).get('csv_sha256_lf'))


def profile_inputs(request, config):
    data=read(request['profile_json']);origin=read(request['profile_config'])
    if normal_mode(origin) != request['profile_mode']:
        raise ValueError('External profile mode differs from the requested direction.')
    for key in MODEL_KEYS:
        if key.startswith('normal_'):continue
        if config.get(key) != origin.get(key):
            raise ValueError('External profile geometry/material mismatch: '+key)
    if config['material_provenance']['csv_sha256_lf'] != origin['material_provenance']['csv_sha256_lf']:
        raise ValueError('External profile material CSV differs.')
    logits=data.get('best_logits',data.get('logits'))
    if logits is None:logits=(data.get('best') or {}).get('logits')
    if (not isinstance(logits,list) or len(logits)!=config['layers']+1
            or any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in logits)):
        raise ValueError('Expected finite layers+1 external logits.')
    return logits,dict(profile_json=str(Path(request['profile_json']).resolve()),
        profile_json_sha256=file_sha(request['profile_json']),
        profile_config=str(Path(request['profile_config']).resolve()),
        profile_config_sha256=file_sha(request['profile_config']),
        profile_origin_normal_field_mode=request['profile_mode'],profile_sha256=digest(dict(logits=logits)))


def prepare_child(request):
    case=Path(request['case_dir']);common,physics,policy=import_case(case)
    config=common.load_config(Path(request['config']))
    if config.get('material') != request['material'] or normal_mode(config)!=request['evaluator_mode']:
        raise ValueError('Evaluator material or normal-field mode differs.')
    if request['lattice']!='hexagonal':raise ValueError('This cross-profile controller targets hexagonal cases.')
    from rcwa_ext.acceleration import d6
    normal_source=inspect.getsource(d6.native_inputs)
    expected='_smoothstep(' if request['evaluator_mode']=='on' else 'unsmoothed_projection('
    if expected not in normal_source:raise ValueError('Unexpected evaluator normal-field implementation.')
    previous=read(Path(request['run_root'])/'manifest.json')
    sources=policy.execution_sources(common.sources())
    checked,_=policy.reconcile_manifest(previous,dict(version=1,config=config,source_sha256=sources,
                                                     requested_device=request['device']))
    logits,origin=profile_inputs(request,config)
    device=common.device_setup(request['device'],config['cpu_threads'])
    runtime=common.runtime_environment(device);policy.check_runtime(previous.get('runtime_environment'),runtime)
    return dict(status='ready',context=dict(tool=TOOL,tool_sha256=file_sha(__file__),
        direction=request['direction'],material=request['material'],case_dir=str(case),run_root=request['run_root'],
        config=config,source_sha256=sources,training_signature=checked['signature'],runtime_environment=runtime,
        device=str(device),lattice=request['lattice'],waves=request['waves'],backend=request['backend'],
        logits=logits,profile_origin=origin,evaluator_normal_field_mode=request['evaluator_mode'],
        normal_field_definition='legacy-smoothstep-grid-coupled' if request['evaluator_mode']=='on'
                                else config['normal_field_definition'],
        normal_field_input_source_sha256=hashlib.sha256(normal_source.encode()).hexdigest(),
        protected_training_files_sha256=training_files(Path(request['run_root']),request['lattice'])))


def worker_child(request):
    context=read(request['context']);check_profile_inputs(context)
    common,physics,policy=import_case(Path(context['case_dir']))
    if file_sha(__file__)!=context['tool_sha256']:raise ValueError('Analysis tool changed.')
    if policy.execution_sources(common.sources())!=context['source_sha256']:
        raise ValueError('Frozen evaluation sources or material CSV changed.')
    device=common.device_setup(context['device'],context['config']['cpu_threads'])
    runtime=common.runtime_environment(device)
    if runtime!=context['runtime_environment']:raise ValueError('Runtime changed during the analysis.')
    import torch
    waves=request['waves'];x=torch.tensor(context['logits'],dtype=torch.float64,device=device)
    result=physics.objective(x,{**context['config'],'rcwa_acceleration':context['backend']},
        context['lattice'],waves,request['order'],request['grid'],device,gradient=False)
    check_profile_inputs(context)
    if len(result['rows'])!=len(waves):raise ValueError('Incomplete wavelength rows.')
    return dict(status='completed',rows=result['rows'],mean_R=result['objective'],rcwa_seconds=result['seconds'],
        runtime_environment=runtime,cuda_execution_performed=device.type=='cuda',
        peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated(device)) if device.type=='cuda' else None)


def weighted_mean(rows, kind):
    if kind=='arithmetic' or len(rows)==1:return sum(r['reflectance'] for r in rows)/len(rows)
    if kind!='trapezoidal':raise ValueError('Unknown objective weighting.')
    span=rows[-1]['wavelength_nm']-rows[0]['wavelength_nm']
    return sum((b['wavelength_nm']-a['wavelength_nm'])*(a['reflectance']+b['reflectance'])/2
               for a,b in zip(rows,rows[1:]))/span


def cache_index(context, orders, grids):
    """Import completed numerical rows only when model, shape and sources match."""
    run=Path(context['run_root']);base=run/context['lattice']/'postprocess'
    result={};wanted={(m,q) for m in orders for q in grids};waves=set(context['waves'])
    source_digests={}
    def insert(order,grid,rows,path,source_kind,metadata_paths=()):
        if (order,grid) not in wanted:return
        if path not in source_digests:source_digests[path]=file_sha(path)
        proof=[dict(path=str(p),sha256=file_sha(p)) for p in metadata_paths]
        for row in rows:
            wave=row.get('wavelength_nm')
            if wave not in waves:continue
            r,p=row.get('reflectance'),row.get('power_into_substrate')
            if not isinstance(r,(int,float)) or not isinstance(p,(int,float)) or not all(map(math.isfinite,(r,p))):
                raise ValueError('Nonfinite completed cache row: '+str(path))
            if min(r,p)<-1e-7 or r>1+1e-7 or r+p>1+1e-7 or row.get('passivity_warning',False):
                raise ValueError('Nonphysical completed cache row: '+str(path))
            key=(order,grid,wave)
            if key in result:
                previous=result[key]['row']
                if max(abs(previous[k]-row[k]) for k in ('reflectance','power_into_substrate'))>1e-8:
                    raise ValueError('Conflicting equivalent cache rows: '+str(key))
                continue
            result[key]=dict(row=row,origin=dict(path=str(path),sha256=source_digests[path],
                source_kind=source_kind,metadata=proof))
    def valid(config,logits,lattice,backend='auto'):
        return (logits==context['logits'] and lattice==context['lattice']
                and model_matches(config,context['config']) and backend==context['backend'])

    for category in ('convergence_continue','profile_convergence','cross_profile_convergence'):
        for path in sorted((base/category).glob('**/report.json')):
            cp=path.with_name('context.json')
            if not cp.exists():continue
            report,ctx=read(path),read(cp)
            if ctx.get('source_sha256')!=context['source_sha256']:continue
            if not valid(ctx['config'],report.get('frozen_logits'),ctx.get('lattice'),ctx.get('backend','auto')):continue
            for record in report.get('records',[]):
                if record.get('status')=='completed':
                    insert(record['order'],record['grid'],record['rows'],path,category,(cp,))

    # A worker can finish just before the parent is interrupted. Its completed
    # rows remain reusable even when the aggregate report was not saved yet.
    for path in sorted((base/'cross_profile_convergence').glob('**/M*_Q*/result.json')):
        reqpath=path.with_name('request.json');cp=path.parent.parent/'context.json'
        if not reqpath.exists() or not cp.exists():continue
        outcome,req,ctx=read(path),read(reqpath),read(cp)
        if outcome.get('status')!='completed' or req.get('mode')!='evaluate':continue
        if ctx.get('tool')!=TOOL or ctx.get('source_sha256')!=context['source_sha256']:continue
        if not valid(ctx['config'],ctx.get('logits'),ctx.get('lattice'),ctx.get('backend','auto')):continue
        if sorted(row['wavelength_nm'] for row in outcome['rows'])!=sorted(req['waves']):continue
        insert(req['order'],req['grid'],outcome['rows'],path,'completed_cross_worker',(cp,reqpath))

    for category in ('spectrum','convergence'):
        for path in sorted((base/category).glob('*/report.json')):
            report=read(path)
            if report.get('status')!='completed' or report.get('source_profile_logits')!=context['logits']:continue
            for outcome in report.get('outcomes',[]):
                reqpath=path.parent/f"M{outcome['order']}_Q{outcome['grid']}"/'request.json'
                if not reqpath.exists() or outcome.get('status')!='completed':continue
                req=read(reqpath)
                if (req.get('signature')!=context['training_signature'] or outcome.get('request_signature')!=digest(req)
                        or not valid(req['config'],req.get('logits'),req.get('lattice'),req['config'].get('rcwa_acceleration','auto'))):continue
                if req['order']!=outcome['order'] or req['grid']!=outcome['grid']:continue
                insert(req['order'],req['grid'],outcome['rows'],path,category,(reqpath,))

    if context['evaluator_normal_field_mode']=='off':
        for path in sorted((base/'off_old_new_comparison').glob('*/report.json')):
            report=read(path)
            if report.get('status')!='completed' or report.get('source_sha256')!=context['source_sha256']:continue
            if report.get('normal_field_definition')!=context['normal_field_definition']:continue
            for label in ('old','new'):
                if report.get(label+'_logits')!=context['logits']:continue
                reqpath=path.parent/label/'request.json';sp=path.with_name(label+'_spectrum.json')
                if not reqpath.exists():continue
                req=read(reqpath)
                if not valid(req['config'],req.get('logits'),req.get('lattice'),req['config'].get('rcwa_acceleration','auto')):continue
                if req['order']!=report['order'] or req['grid']!=report['grid']:continue
                if not sp.exists():
                    sp=path.parent/label/'result.json'
                if not sp.exists():
                    rows=[dict(wavelength_nm=row['wavelength_nm'],reflectance=row[label+'_reflectance'],
                               power_into_substrate=row[label+'_power_into_substrate'],
                               motheye_absorptance=row[label+'_motheye_absorptance'],
                               absorptance_total=row[label+'_absorptance_total']) for row in report['rows']]
                    if report.get('production_files_unchanged') is not True:continue
                    insert(report['order'],report['grid'],rows,path,'off_old_new_comparison_summary',(reqpath,))
                    continue
                spectrum=read(sp)
                if spectrum.get('status')!='completed' or spectrum.get('request_signature')!=digest(req):continue
                expected={row['wavelength_nm']:row[label+'_reflectance'] for row in report['rows']}
                if any(abs(row['reflectance']-expected[row['wavelength_nm']])>1e-12 for row in spectrum['rows']):
                    raise ValueError('Old/new comparison cache identity differs.')
                insert(report['order'],report['grid'],spectrum['rows'],sp,'off_old_new_comparison',(path,reqpath))
    return result


def condition_plan(context, orders, grids, index):
    return [dict(order=m,grid=q,reused_wavelengths_nm=[w for w in context['waves'] if (m,q,w) in index],
                 missing_wavelengths_nm=[w for w in context['waves'] if (m,q,w) not in index])
            for m in orders for q in grids]


def merge_rows(context,m,q,index,new_result=None):
    supplied={r['wavelength_nm']:r for r in (new_result or {}).get('rows',[])}
    rows=[];origins=[]
    for w in context['waves']:
        if (m,q,w) in index:
            entry=index[m,q,w];rows.append(entry['row']);origins.append(dict(wavelength_nm=w,**entry['origin']))
        else:rows.append(supplied[w])
    return dict(status='completed',mean_R=weighted_mean(rows,context['config']['objective_weighting']),rows=rows,
        worker_seconds=(new_result or {}).get('worker_seconds',0.),rcwa_seconds=(new_result or {}).get('rcwa_seconds',0.),
        cache_hit=not supplied,reused_wavelength_points=len(origins),computed_wavelength_points=len(supplied),
        reused_sources=origins,cuda_execution_performed=(new_result or {}).get('cuda_execution_performed',False),
        runtime_environment=(new_result or {}).get('runtime_environment',context['runtime_environment']))


def persist(folder,context,records,state='running'):
    check_profile_inputs(context)
    preserved=all(Path(p).is_file() and file_sha(p)==h for p,h in context['protected_training_files_sha256'].items())
    if not preserved:raise RuntimeError('An original optimization record changed.')
    summary=aggregate(records,context['config']['verification_tolerance'])
    axes={c['axis']:c['passed'] for c in summary['convergence_checks']}
    report=dict(status=state,direction=context['direction'],material=context['material'],
        requested_device=context['device'],cuda_execution_performed=any(r.get('cuda_execution_performed',False) for r in records),
        training_signature=context['training_signature'],frozen_logits=context['logits'],profile_origin=context['profile_origin'],
        evaluator_normal_field_mode=context['evaluator_normal_field_mode'],normal_field_definition=context['normal_field_definition'],
        protected_training_files_unchanged=preserved,wavelengths_nm=context['waves'],
        counts=dict(Counter(r['status'] for r in records)),records=records,
        reused_completed_conditions=sum(r['status']=='completed' and r.get('cache_hit',False) for r in records),
        reused_wavelength_points=sum(r.get('reused_wavelength_points',0) for r in records),
        computed_wavelength_points=sum(r.get('computed_wavelength_points',0) for r in records),
        order_convergence_passed=axes.get('order'),grid_convergence_passed=axes.get('grid'),**summary)
    write(folder/'report.json',report)
    with (folder/'convergence.csv.tmp').open('w',encoding='utf-8',newline='') as f:
        fields=['order','grid','status','mean_R','cache_hit','reused_wavelength_points','computed_wavelength_points','rcwa_seconds','reason','error']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for record in records:writer.writerow({k:record.get(k) for k in fields})
    (folder/'convergence.csv.tmp').replace(folder/'convergence.csv')
    return report


def defaults(root):
    result=[]
    for material,low,off_order,on_config,on_suffix in [
        ('Al','al',8,'config_hex_trial1.json','hex_trial1_v3'),
        ('Ti','ti',8,'config_hex_trial1_M8_Q512.json','hex_trial1_M8_Q512_v3'),
        ('Cu','cu',12,'config_hex_trial1_M12.json','hex_trial1_M12_v3'),
        ('Au','au',14,'config_hex_trial1_M12.json','hex_trial1_M12_v3')]:
        on=root/material/f'{low}_motheye_P200_H500_Nz15'
        off=on.with_name(on.name+'_normal_off_v1')
        cases={'on':(on,on/on_config,on/'results'/on_suffix),
               'off':(off,off/f'config_hex_trial1_M{off_order}_Q1024_off.json',off/'results'/f'hex_trial1_M{off_order}_Q1024_off_v1')}
        for direction,(profile_mode,evaluator_mode) in DIRECTIONS.items():
            case,config,run=cases[evaluator_mode];origin_run=cases[profile_mode][2]
            result.append(dict(direction=direction,material=material,profile_mode=profile_mode,evaluator_mode=evaluator_mode,
                case_dir=str(case),config=str(config),run_root=str(run),
                profile_json=str(origin_run/'hexagonal/report.json'),profile_config=str(origin_run/'resolved_config.json')))
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--output-root',type=Path)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--materials',default='Al,Ti,Cu,Au')
    p.add_argument('--directions',default='ON_to_OFF,OFF_to_ON')
    p.add_argument('--orders',default='20:40:2')
    p.add_argument('--grids',default='512,1024')
    p.add_argument('--wavelengths',default='400,550,700')
    p.add_argument('--lattice',choices=['hexagonal'],default='hexagonal')
    p.add_argument('--backend',choices=['auto','matrix','reference'],default='auto')
    p.add_argument('--oom-policy',choices=['stop-material'],default='stop-material')
    p.add_argument('--plan-only',action='store_true',help='Check sources and cached rows; do not run RCWA.')
    p.add_argument('--cases-file',type=Path)
    p.add_argument('--internal-request',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--internal-result',type=Path,help=argparse.SUPPRESS)
    args=p.parse_args(argv)
    if args.internal_request:return child_main(args.internal_request,args.internal_result)
    root=args.root.resolve();out=(args.output_root or root/'cross_profile_convergence_v1').resolve()
    orders=numbers(args.orders,True);grids=numbers(args.grids,True);waves=numbers(args.wavelengths)
    if any(q<max(32,4*m+4) for m in orders for q in grids):p.error('Grid undersamples an order.')
    specs=read(args.cases_file) if args.cases_file else defaults(root)
    by_key={(s['direction'],s['material']):s for s in specs}
    directions=args.directions.split(',');materials=args.materials.split(',')
    if any(d not in DIRECTIONS for d in directions):p.error('Unknown direction.')
    if any((d,m) not in by_key for d in directions for m in materials):p.error('Missing case specification.')
    batch=dict(status='running',plan_only=args.plan_only,directions=directions,orders=orders,grids=grids,
               wavelengths_nm=waves,cases={})
    summary_path=out/('plan_summary.json' if args.plan_only else 'summary.json')
    def interrupted(signum,frame):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupted)
    try:
        for direction in directions:
            for material in materials:
                label=direction+'/'+material;spec={**by_key[direction,material]}
                for k in ('case_dir','config','run_root','profile_json','profile_config'):spec[k]=str(Path(spec[k]).resolve())
                lock=Path(spec['run_root'])/'controller.lock';held=False
                try:
                    if not (Path(spec['run_root'])/'manifest.json').is_file():raise ValueError('Evaluator manifest missing.')
                    acquire(lock);held=True
                    preparation=launch(out/direction/material/'preparation',dict(mode='prepare',**spec,
                        device=args.device,lattice=args.lattice,waves=waves,backend=args.backend))
                    if preparation['status']!='ready':raise RuntimeError('Preparation failed: '+str(preparation))
                    context=preparation['context'];index=cache_index(context,orders,grids)
                    plan=condition_plan(context,orders,grids,index)
                    reusable=sum(not x['missing_wavelengths_nm'] for x in plan)
                    missing=sum(len(x['missing_wavelengths_nm']) for x in plan)
                    print(f'{label}: REUSE {reusable}/{len(plan)} conditions; {len(index)} wavelength rows; COMPUTE {missing} missing wavelength rows',flush=True)
                    write(out/direction/material/'plan.json',dict(direction=direction,material=material,
                        reusable_conditions=reusable,reused_wavelength_points=len(index),missing_wavelength_points=missing,conditions=plan))
                    if args.plan_only:
                        batch['cases'][label]=dict(status='planned',reusable_conditions=reusable,
                            reused_wavelength_points=len(index),missing_wavelength_points=missing)
                        continue
                    folder=Path(spec['run_root'])/args.lattice/'postprocess/cross_profile_convergence'/direction/digest(dict(context=context,orders=orders,grids=grids))[:16]
                    folder.mkdir(parents=True,exist_ok=True);ctxpath=folder/'context.json';write(ctxpath,context)
                    cache={}
                    for item in plan:
                        if not item['missing_wavelengths_nm']:
                            m,q=item['order'],item['grid'];cache[m,q]=dict(order=m,grid=q,**merge_rows(context,m,q,index))
                    def execute(m,q):
                        todo=[w for w in waves if (m,q,w) not in index]
                        print(f'{label}: START M={m} Q={q}, missing wavelengths={todo}',flush=True)
                        result=launch(folder/f'M{m}_Q{q}',dict(mode='evaluate',context=str(ctxpath),order=m,grid=q,
                            waves=todo,device=context['device']))
                        print(f"{label}: DONE M={m} Q={q}: {result['status']}",flush=True)
                        if result['status']=='completed':
                            completed=merge_rows(context,m,q,index,result)
                            completed.pop('cache_hit',None)  # drive_conditions assigns this flag.
                            return completed
                        return result
                    records=drive_conditions(orders,grids,execute,cache,oom_policy='stop-material',
                        progress=lambda records:persist(folder,context,records))
                    report=persist(folder,context,records,'completed_with_skips' if any(r['status']!='completed' for r in records) else 'completed')
                    pointer=Path(spec['run_root'])/args.lattice/f'latest_cross_{direction}.json'
                    write(pointer,dict(report=str(folder/'report.json'),status=report['status']))
                    batch['cases'][label]={k:report[k] for k in ('status','counts','reused_completed_conditions',
                        'reused_wavelength_points','computed_wavelength_points','order_convergence_passed','grid_convergence_passed','numerical_convergence_passed')}
                    batch['cases'][label]['report']=str(folder/'report.json')
                    print(label+': '+json.dumps(batch['cases'][label]),flush=True)
                except FileExistsError:
                    batch['cases'][label]=dict(status='busy',error='Controller lock exists; not removed.')
                    print(label+': busy; continuing.',flush=True)
                except KeyboardInterrupt:raise
                except Exception as error:
                    batch['cases'][label]=dict(status='setup_error',error=str(error),traceback=traceback.format_exc())
                    print(label+': '+str(error)+'; continuing.',flush=True)
                finally:
                    if held:(lock/'owner.json').unlink(missing_ok=True);lock.rmdir()
                    write(summary_path,batch)
        batch['status']='completed_with_skips' if any(x['status'] not in ('completed','planned') for x in batch['cases'].values()) else 'completed'
    except KeyboardInterrupt:
        batch['status']='interrupted';write(summary_path,batch)
        print('Interrupted; completed wavelength rows can be reused next time.',flush=True)
        return 130
    write(summary_path,batch);print('Saved '+str(summary_path),flush=True)
    return 1 if any(x['status'] in ('setup_error','busy') for x in batch['cases'].values()) else 0


if __name__=='__main__':
    raise SystemExit(main())
