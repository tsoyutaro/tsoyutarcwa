"""Isolate normal-field sampling and smoothing for saved hexagonal metal shapes.

Depends on the existing sibling continue_convergence.py. Production files are
never edited. A single verified assignment is replaced in memory in each worker.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
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
import textwrap
import time
import traceback
from types import SimpleNamespace

import continue_convergence as saved

TOOL='rcwa-normal-grid-isolation-v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixed_normal_function(original, reference_grid):
    """Freeze all resolution-based normal-field tapers; retain the sample grid."""
    source=textwrap.dedent(inspect.getsource(original))
    tree=ast.parse(source)
    expected=ast.parse('max(lx/grid,ly/grid)',mode='eval').body
    found=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='resolution':
            if ast.dump(node.value)!=ast.dump(expected):
                raise ValueError('Unexpected normal-field resolution expression; no patch applied.')
            found.append(node)
    if len(found)!=1:raise ValueError('Expected exactly one normal-field resolution assignment.')
    found[0].value=ast.parse('max(lx/_diagnostic_reference_grid,ly/_diagnostic_reference_grid)',mode='eval').body
    ast.fix_missing_locations(tree)
    namespace=dict(original.__globals__)
    namespace['_diagnostic_reference_grid']=int(reference_grid)
    exec(compile(tree,'<fixed-normal-field-diagnostic>','exec'),namespace)
    return namespace[original.__name__],hashlib.sha256(source.encode()).hexdigest()


def power_row(wave,r,p,tolerance):
    raw={'reflectance':r,'power_into_substrate':p,'absorptance_total':1-r,'motheye_absorptance':1-r-p}
    valid=all(math.isfinite(v) and -tolerance<=v<=1+tolerance for v in raw.values())
    return dict(wavelength_nm=wave,**{k:v if math.isfinite(v) else None for k,v in raw.items()},passive=valid)


def evaluate(context,grid,reference_grid):
    common,physics,policy=saved.import_case(Path(context['case_dir']))
    if policy.execution_sources(common.sources())!=context['source_sha256']:
        raise ValueError('Production sources/material data changed.')
    device=common.device_setup(context['device'],context['config']['cpu_threads'])
    if common.runtime_environment(device)!=context['runtime_environment']:
        raise ValueError('Runtime/hardware changed; restart to create a new analysis identity.')
    import torch
    from optimizer import geometry
    from optimizer.materials import epsilon_at
    from rcwa_ext.acceleration import d6
    from rcwa_ext.acceleration.spectral import SharedEvaluator
    c=context['config'];waves=context['waves'];order=context['order']
    original=d6.native_inputs
    patched,source_sha=fixed_normal_function(original,reference_grid)
    if source_sha!=context['normal_input_source_sha256']:raise ValueError('Normal-field routine changed.')
    def sync():
        if device.type=='cuda':torch.cuda.synchronize(device)
    sync();started=time.perf_counter()
    rows=[]
    try:
        d6.native_inputs=patched
        with torch.no_grad():
            x=torch.tensor(context['logits'],dtype=torch.float64,device=device)
            radii=geometry.radii_tensor(x,c)
            model=SimpleNamespace(material=c['material'],physics=physics,geometry=geometry,epsilon_at=epsilon_at)
            evaluator=SharedEvaluator(model)
            cache=evaluator.cache(c,'hexagonal',waves,order,grid,device)
            leaves=tuple(v for radius in radii/c['period_nm'] for v in evaluator.geometry_pair(radius,cache))
            for i,wave in enumerate(waves):
                r,p=evaluator.response(cache,leaves,wave,cache['indices'][i],cache['ports'][i],False)
                sync();r=float(r.cpu());p=float(p.cpu())
                row=power_row(wave,r,p,c.get('passivity_tolerance',1e-6));rows.append(row)
                print(f'{wave:g} nm: R={r:.12g} P_sub={p:.12g} A_moth={1-r-p:.12g} passive={row["passive"]}',flush=True)
    finally:
        d6.native_inputs=original
    sync()
    if policy.execution_sources(common.sources())!=context['source_sha256']:
        raise ValueError('Production sources/material data changed during evaluation.')
    finite=all(row['reflectance'] is not None for row in rows)
    return dict(status='completed' if all(r['passive'] for r in rows) else 'nonphysical',rows=rows,
        mean_R=sum(r['reflectance'] for r in rows)/len(rows) if finite else None,
        rcwa_seconds=time.perf_counter()-started,
        peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated(device)) if device.type=='cuda' else None)


def child(request,output):
    request=saved.read(request);started=time.perf_counter()
    try:
        if request['diagnostic_tool_sha256']!=sha(__file__):raise ValueError('Diagnostic script changed during this run.')
        if request['mode']=='prepare':
            result=saved.prepare_child(request)
            c=result['context']['config']
            if c.get('material_model')!='metal_csv' or c.get('substrate_mode')!='semi-infinite':
                raise ValueError('This version supports standalone metal pillars on the same metal substrate.')
            from rcwa_ext.acceleration import d6
            _,normal_sha=fixed_normal_function(d6.native_inputs,1024)
            result['context'].update(diagnostic_tool=TOOL,diagnostic_tool_sha256=sha(__file__),
                                     normal_input_source_sha256=normal_sha,order=request['order'])
        else:
            context=saved.read(request['context'])
            if context['diagnostic_tool_sha256']!=sha(__file__):raise ValueError('Diagnostic script changed.')
            result=evaluate(context,request['grid'],request['normal_reference_grid'])
    except (Exception,KeyboardInterrupt) as error:
        lower=str(error).lower();kind=type(error).__name__
        status='interrupted' if isinstance(error,KeyboardInterrupt) else (
            'cuda_oom' if 'cuda out of memory' in lower or kind=='OutOfMemoryError' and request['device'].startswith('cuda') else
            'cpu_oom' if isinstance(error,MemoryError) or kind=='OutOfMemoryError' else
            'setup_error' if request['mode']=='prepare' else 'error')
        result=dict(status=status,error_type=kind,error=str(error),traceback=traceback.format_exc())
        print(result['traceback'],file=sys.stderr,flush=True)
    result['worker_seconds']=time.perf_counter()-started
    saved.write(output,result)
    return 0


def launch(folder,request):
    folder.mkdir(parents=True,exist_ok=True)
    req,res=folder/'request.json',folder/'result.json'
    saved.write(req,{**request,'diagnostic_tool_sha256':sha(__file__)})
    if res.exists():res.unlink()
    with (folder/'worker.log').open('w',encoding='utf-8') as log:
        process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--internal-request',str(req),'--internal-result',str(res)],stdout=log,stderr=subprocess.STDOUT)
        try:process.wait()
        except BaseException:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            raise
    if not res.exists() or process.returncode!=0:
        return dict(status='worker_lost',returncode=process.returncode,error='No successful worker completion; OOM is not confirmed.')
    return saved.read(res)


def difference(a,b):
    if a['status']!='completed' or b['status']!='completed':return None
    result={}
    for key in ['reflectance','power_into_substrate','motheye_absorptance']:
        pairs=list(zip(a['rows'],b['rows']))
        if not pairs or any(x['wavelength_nm']!=y['wavelength_nm'] for x,y in pairs):raise ValueError('Wavelength grid differs.')
        x,y=max(pairs,key=lambda p:abs(p[0][key]-p[1][key]))
        result[key]=dict(max_abs_difference=abs(x[key]-y[key]),wavelength_nm=x['wavelength_nm'],coarse=x[key],fine=y[key])
    return result


def comparisons(records,grids,references,tolerance):
    lookup={(r['grid'],r['normal_reference_grid']):r for r in records}
    sampling=[];smoothing=[]
    def append(target,coarse,fine,**labels):
        if coarse not in lookup or fine not in lookup:return
        delta=difference(lookup[coarse],lookup[fine])
        error=max(delta[k]['max_abs_difference'] for k in ['reflectance','power_into_substrate']) if delta else None
        target.append(dict(**labels,coarse=list(coarse),fine=list(fine),details=delta,
                           max_power_abs_difference=error,passed=error<=tolerance if error is not None else None))
    for ref in references:
        for lo,hi in zip(grids,grids[1:]):append(sampling,(lo,ref),(hi,ref),fixed_normal_reference_grid=ref)
    for grid in grids:
        for lo,hi in zip(references,references[1:]):append(smoothing,(grid,lo),(grid,hi),fixed_sample_grid=grid)
    return dict(sampling_comparisons=sampling,smoothing_comparisons=smoothing,
                scope='Finite comparisons at fixed diffraction order; no claim of a converged physical solution.')


def protected(run):
    paths=[p for folder in [run,run/'hexagonal'] for p in folder.glob('*') if p.is_file()]
    paths+=list((run/'hexagonal/trials').glob('*/adam_checkpoint.pt'))
    return {str(p):sha(p) for p in paths}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case-dir',type=Path)
    p.add_argument('--run-root',type=Path,default=Path('results/hex_trial1_v3'))
    p.add_argument('--config',type=Path)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--order',type=int,default=30)
    p.add_argument('--grids',default='1024,2048')
    p.add_argument('--normal-reference-grids',default='1024,2048')
    p.add_argument('--wavelengths',default='400,550,700')
    p.add_argument('--retry-failures',action='store_true')
    p.add_argument('--internal-request',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--internal-result',type=Path,help=argparse.SUPPRESS)
    args=p.parse_args(argv)
    if args.internal_request:return child(args.internal_request,args.internal_result)
    if not args.case_dir:p.error('--case-dir is required.')
    case=args.case_dir.resolve();run=(args.run_root if args.run_root.is_absolute() else case/args.run_root).resolve()
    config=(args.config if args.config and args.config.is_absolute() else case/args.config if args.config else run/'resolved_config.json').resolve()
    grids=saved.numbers(args.grids,True);refs=saved.numbers(args.normal_reference_grids,True);waves=saved.numbers(args.wavelengths)
    if args.order<1 or any(q<max(32,4*args.order+4) for q in grids):p.error('Invalid diffraction order/sample grid.')
    if len(grids)<2 or len(refs)<2:p.error('Specify at least two sample grids and two normal reference grids.')
    if not (run/'manifest.json').exists():p.error('Saved experiment manifest is missing.')
    material=saved.read(config)['material'];base=run/'hexagonal/postprocess/normal_grid'
    lock=run/'controller.lock';held=False;folder=None;records=[];context=None;original_hashes={}
    def interrupted(signum,frame):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupted)
    def persist(status):
        report=dict(tool=TOOL,status=status,material=material,order=args.order,wavelengths_nm=waves,
            sample_grids=grids,normal_reference_grids=refs,counts=dict(Counter(r['status'] for r in records)),records=records,
            smoothing_parameter_nm={str(ref):3*context['config']['period_nm']/ref for ref in refs},
            production_file_hashes_before=original_hashes,
            production_files_unchanged=all(Path(k).exists() and sha(k)==v for k,v in original_hashes.items()),
            tolerance=context['config']['verification_tolerance'],cuda_execution_performed=context['device'].startswith('cuda'),
            **comparisons(records,grids,refs,context['config']['verification_tolerance']))
        saved.write(folder/'report.json',report)
        return report
    try:
        try:lock.mkdir()
        except FileExistsError:
            p.exit(2,'Controller lock exists: '+str(lock)+'; verify the previous job before clearing a stale lock.\n')
        held=True;saved.write(lock/'owner.json',dict(tool=TOOL,pid=os.getpid(),host=socket.gethostname()))
        original_hashes=protected(run)
        preparation=launch(base/'preparation',dict(mode='prepare',case_dir=str(case),run_root=str(run),config=str(config),
            material=material,lattice='hexagonal',waves=waves,backend='auto',device=args.device,order=args.order))
        if preparation['status']!='ready':
            raise RuntimeError('Preparation failed: '+str(preparation.get('error',preparation)))
        context=preparation['context']
        folder=base/saved.digest(dict(context=context,grids=grids,references=refs))[:16]
        folder.mkdir(parents=True,exist_ok=True);saved.write(folder/'context.json',context)
        prior=saved.read(folder/'report.json')['records'] if (folder/'report.json').exists() else []
        cache={(r['grid'],r['normal_reference_grid']):r for r in prior}
        stop=False
        for grid in grids:
            for ref in refs:
                item=dict(grid=grid,normal_reference_grid=ref,smoothing_parameter_nm=3*context['config']['period_nm']/ref,
                          equivalent_to_legacy_tapers=(grid==ref))
                old=cache.get((grid,ref))
                if old and (old['status']=='completed' or not args.retry_failures and old['status'] in ['nonphysical','cuda_oom','cpu_oom']):
                    item={**old,'cache_hit':True}
                elif stop:item.update(status='skipped',reason='remaining_conditions_skipped_after_oom_or_worker_loss')
                else:
                    print(f'START Q={grid} normal_ref={ref} smoothing={item["smoothing_parameter_nm"]:.9g} nm ({len(waves)} wavelengths)',flush=True)
                    result=launch(folder/f'Q{grid}_N{ref}',dict(mode='evaluate',context=str(folder/'context.json'),grid=grid,normal_reference_grid=ref,device=context['device']))
                    item.update(result,cache_hit=False)
                    print(f'DONE Q={grid} normal_ref={ref}: {item["status"]}',flush=True)
                records.append(item)
                if item['status'] in ['cuda_oom','cpu_oom','worker_lost']:stop=True
                persist('running')
        report=persist('completed' if all(r['status']=='completed' for r in records) else 'completed_with_skips')
        if not report['production_files_unchanged']:raise RuntimeError('A protected production file changed during diagnosis.')
        saved.write(run/'hexagonal/latest_normal_grid.json',dict(report=str(folder/'report.json'),status=report['status']))
        print(json.dumps({k:report[k] for k in ['status','counts','production_files_unchanged','sampling_comparisons','smoothing_comparisons']},indent=2),flush=True)
        print('Saved '+str(folder/'report.json'),flush=True)
        return 0 if report['status']=='completed' else 1
    except KeyboardInterrupt:
        if folder is not None and context is not None:persist('interrupted')
        print('Interrupted; completed conditions can be reused.',flush=True)
        return 130
    finally:
        if held:(lock/'owner.json').unlink(missing_ok=True);lock.rmdir()


if __name__=='__main__':raise SystemExit(main())
