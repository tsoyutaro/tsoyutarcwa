"""Explicit post-optimization commands; no training is started here."""
from __future__ import annotations
import json
import statistics
import time
from pathlib import Path
from .common import signature_for,write_csv,write_json,wavelengths


def spectrum_worker(request,device):
    import torch
    from .physics import objective
    x=torch.tensor(request['logits'],dtype=torch.float64,device=device)
    result=objective(x,request['config'],request['lattice'],request['waves'],request['order'],request['grid'],device,gradient=False)
    return dict(status='completed',rows=result['rows'],objective=result['objective'],seconds=result['seconds'])


def timing_worker(request,device):
    import torch
    from . import physics
    def sync():
        if device.type=='cuda':torch.cuda.synchronize(device)
    x=torch.tensor(request['logits'],dtype=torch.float64,device=device,requires_grad=True)
    results={};observed={}
    for backend in ['reference','auto']:
        c={**request['config'],'rcwa_acceleration':backend}
        physics._spectral_evaluator=None
        def evaluate():
            sync();started=time.perf_counter()
            result=physics.objective(x,c,request['lattice'],request['waves'],request['order'],request['grid'],device,gradient=request['gradient'])
            sync()
            return time.perf_counter()-started,result
        cold,result=evaluate()
        observed[backend]=result
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        samples=[evaluate()[0] for _ in range(request['repeats'])]
        median=statistics.median(samples)
        results[backend]=dict(cold_seconds=cold,warm_seconds=samples,warm_median_seconds=median,
            warm_median_seconds_per_wavelength=median/len(request['waves']),
            peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated(device)) if device.type=='cuda' else None)
    a,b=observed['reference'],observed['auto']
    loss_error=abs(a['objective']-b['objective'])
    keys=['reflectance','power_into_substrate' if request['config'].get('substrate_mode')=='semi-infinite' else 'transmittance']
    power_error=max(abs(x[k]-y[k]) for x,y in zip(a['rows'],b['rows']) for k in keys)
    gradient_error=float((a['gradient']-b['gradient']).abs().max().detach().cpu()) if request['gradient'] else None
    passed=loss_error<1e-8 and power_error<1e-8 and (not request['gradient'] or torch.allclose(a['gradient'],b['gradient'],atol=1e-7,rtol=2e-5))
    if not passed:raise ValueError(f'Timing parity failed: loss={loss_error}, power={power_error}, gradient={gradient_error}')
    return dict(status='completed',passed=True,warmup_excluded=True,gradient=request['gradient'],
        wavelengths_nm=request['waves'],repeats=request['repeats'],timings=results,
        warm_speedup=results['reference']['warm_median_seconds']/results['auto']['warm_median_seconds'],
        loss_abs_error=loss_error,power_max_abs_error=power_error,logits_gradient_max_abs_error=gradient_error)


def _best(folder,profile):
    path=profile or folder/'report.json'
    data=json.loads(path.read_text(encoding='utf-8-sig'))
    logits=data.get('best_logits',data.get('logits'))
    if logits is None:logits=data.get('best',{}).get('logits')
    if logits is None:raise ValueError('No completed best shape; run optimization first or specify --profile-json.')
    return logits


def _plot(path,rows,metal):
    try:import matplotlib.pyplot as plt
    except ImportError:return 'matplotlib unavailable; CSV and JSON are saved'
    fig,ax=plt.subplots(figsize=(7,4))
    wave=[r['wavelength_nm'] for r in rows]
    keys=['reflectance','absorptance_total','power_into_substrate'] if metal else ['reflectance','transmittance']
    for key in keys:ax.plot(wave,[r[key] for r in rows],label=key)
    ax.set(xlabel='Wavelength (nm)',ylabel='Power fraction',ylim=(-.01,1.01))
    ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(path,dpi=180);plt.close(fig)
    return str(path)


def run(args,c,output,signature,lattices,launch,runtime):
    mode=args.mode
    waves=wavelengths(args.wavelengths or ('400,550,700' if mode=='convergence' else ('550' if mode=='timing' else c['verification_wavelengths_nm'])))
    orders=sorted(set(map(int,args.orders.split(',')))) if args.orders else c['verification_orders']
    grids=sorted(set(map(int,args.grids.split(',')))) if args.grids else sorted(set([c['optimization_grid'],c['verification_grid']]))
    order=args.order or (max(orders) if mode=='spectrum' else c['optimization_order'])
    grid=args.grid or (c['verification_grid'] if mode=='spectrum' else c['optimization_grid'])
    pairs=[(m,q) for m in orders for q in grids] if mode=='convergence' else [(order,grid)]
    if any(m<1 or q<max(32,4*m+4) for m,q in pairs):raise ValueError('Invalid analysis order/grid.')
    if args.repeats<1:raise ValueError('repeats must be positive.')
    reports={}
    for lattice in lattices:
        logits=_best(output/lattice,args.profile_json)
        if len(logits)!=c['layers']+1:raise ValueError('Best shape layer count differs.')
        base=dict(config={**c,'rcwa_acceleration':args.analysis_backend},lattice=lattice,device=args.device,
            signature=signature,runtime_environment=runtime,logits=logits,waves=waves,
            gradient=not args.forward_only,repeats=args.repeats)
        ident=signature_for(dict(base,analysis=mode,pairs=pairs))[:16]
        folder=output/lattice/'postprocess'/mode/ident
        folder.mkdir(parents=True,exist_ok=True)
        outcomes=[]
        for m,q in pairs:
            print(f'{mode}: {lattice} M={m} Q={q}, {len(waves)} wavelengths',flush=True)
            request={**base,'mode':'analyze_timing' if mode=='timing' else 'analyze_spectrum','order':m,'grid':q}
            result=launch(folder/f'M{m}_Q{q}',request)
            if result['status']!='completed':
                write_json(folder/'report.json',dict(status=result['status'],failure=result))
                raise RuntimeError(f'{mode} worker failed: {result.get("error",result["status"])}')
            outcomes.append(dict(order=m,grid=q,**result))
        report=dict(status='completed',mode=mode,lattice=lattice,training_signature=signature,
            source_profile_logits=logits,wavelengths_nm=waves,conditions=pairs,
            requested_device=args.device,cuda_execution_performed=runtime['device'].startswith('cuda'),outcomes=outcomes)
        if mode=='spectrum':
            write_csv(folder/'spectrum.csv',outcomes[0]['rows'])
            report['spectrum_csv']=str(folder/'spectrum.csv')
            report['plot']=_plot(folder/'spectrum.png',outcomes[0]['rows'],c.get('substrate_mode')=='semi-infinite')
        elif mode=='convergence':
            highest=next(o for o in outcomes if (o['order'],o['grid'])==(max(orders),max(grids)))
            key='power_into_substrate' if c.get('substrate_mode')=='semi-infinite' else 'transmittance'
            rows=[]
            for result in outcomes:
                error=max(abs(a[k]-b[k]) for a,b in zip(result['rows'],highest['rows']) for k in ['reflectance',key])
                rows.append(dict(order=result['order'],grid=result['grid'],max_power_abs_difference_from_highest=error,
                    within_tolerance=error<=c['verification_tolerance'],worker_seconds=result['process_wall_seconds'],rcwa_seconds=result['seconds']))
            needed=[]
            if len(orders)>1:needed.append((orders[-2],max(grids)))
            if len(grids)>1:needed.append((max(orders),grids[-2]))
            report.update(highest_reference=dict(order=max(orders),grid=max(grids)),comparison=rows,
                tolerance=c['verification_tolerance'],order_convergence_tested=len(orders)>1,grid_convergence_tested=len(grids)>1,
                numerical_convergence_passed=all(r['within_tolerance'] for r in rows if (r['order'],r['grid']) in needed) if len(needed)==2 else None,
                scope='Same optimized geometry, finite order/grid comparisons; not a proof of infinite-resolution convergence')
            write_csv(folder/'convergence.csv',rows)
        else:report.update(outcomes[0])
        write_json(folder/'report.json',report)
        write_json(output/lattice/f'latest_{mode}.json',dict(report=str(folder/'report.json'),status='completed'))
        reports[lattice]=str(folder/'report.json')
        print(f'Saved {folder}',flush=True)
    write_json(output/f'analysis_workflow_{mode}.json',dict(status='completed',reports=reports))
    return 'completed'
