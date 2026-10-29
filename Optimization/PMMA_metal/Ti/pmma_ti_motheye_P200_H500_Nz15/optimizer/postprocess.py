"""Explicit postprocessing of dual-radius PMMA-core/metal-shell profiles."""
from __future__ import annotations
import json
import math
import statistics
import time
from pathlib import Path
from .common import signature_for,write_csv,write_json,wavelengths

POWER_KEYS=('reflectance','transmittance','absorptance_total')


def mean_metrics(rows,c,waves):
    from .physics import weights
    if len(rows)!=len(waves):raise ValueError('Incomplete spectrum.')
    for row,wave in zip(rows,waves):
        if row['wavelength_nm']!=wave:raise ValueError('Wavelength order differs.')
        if any(not math.isfinite(row[k]) for k in POWER_KEYS):raise ValueError('Nonfinite R/T/A in postprocessing.')
    w=weights(waves,c['objective_weighting'])
    return {name:sum(a*r[key] for a,r in zip(w,rows)) for name,key in (
        ('mean_reflectance','reflectance'),('mean_transmittance','transmittance'),
        ('mean_absorptance','absorptance_total'))}


def spectrum_worker(request,device):
    import torch
    from .physics import objective
    x=torch.tensor(request['logits'],dtype=torch.float64,device=device)
    result=objective(x,request['config'],request['lattice'],request['waves'],request['order'],request['grid'],device,gradient=False)
    metrics=mean_metrics(result['rows'],request['config'],request['waves'])
    if abs(result['objective']-(metrics['mean_reflectance']+metrics['mean_transmittance']))>1e-9:
        raise ValueError('Objective is inconsistent with weighted mean R+T.')
    return dict(status='completed',rows=result['rows'],objective=result['objective'],
                objective_name='mean_R_plus_T',seconds=result['seconds'],**metrics)


def parse_timing_backends(value):
    values=value.split(',') if isinstance(value,str) else list(value)
    result=list(dict.fromkeys(str(x).strip() for x in values))
    if not result or any(x not in ('auto','reference') for x in result):
        raise ValueError('timing-backends must be auto, reference, or reference,auto.')
    return result


def timing_worker(request,device):
    import torch
    from . import physics
    def sync():
        if device.type=='cuda':torch.cuda.synchronize(device)
    backends=parse_timing_backends(request.get('timing_backends','auto'))
    x=torch.tensor(request['logits'],dtype=torch.float64,device=device,requires_grad=request['gradient'])
    results={};observed={}
    for backend in backends:
        c={**request['config'],'rcwa_acceleration':backend}
        for name in ('_evaluator','_spectral_evaluator'):
            if hasattr(physics,name):setattr(physics,name,None)
        def evaluate():
            sync();started=time.perf_counter()
            result=physics.objective(x,c,request['lattice'],request['waves'],request['order'],request['grid'],device,gradient=request['gradient'])
            sync()
            elapsed=time.perf_counter()-started
            if not math.isfinite(elapsed) or elapsed<=0 or not math.isfinite(result['objective']):
                raise ValueError('Nonfinite timing/objective.')
            mean_metrics(result['rows'],c,request['waves'])
            if request['gradient'] and (result['gradient'] is None or not bool(torch.isfinite(result['gradient']).all())):
                raise ValueError('Nonfinite logits gradient during timing.')
            return elapsed,result
        cold,result=evaluate();observed[backend]=result
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        samples=[evaluate()[0] for _ in range(request['repeats'])]
        median=statistics.median(samples)
        results[backend]=dict(cold_seconds=cold,warm_seconds=samples,warm_median_seconds=median,
            warm_median_seconds_per_wavelength=median/len(request['waves']),
            peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated(device)) if device.type=='cuda' else None,
            objective=result['objective'],objective_name='mean_R_plus_T',
            **mean_metrics(result['rows'],c,request['waves']))
    parity=None;loss_error=power_error=gradient_error=speedup=None
    if 'reference' in observed and 'auto' in observed:
        a,b=observed['reference'],observed['auto']
        loss_error=abs(a['objective']-b['objective'])
        power_error=max(abs(u[k]-v[k]) for u,v in zip(a['rows'],b['rows']) for k in POWER_KEYS)
        gradient_error=float((a['gradient']-b['gradient']).abs().max().detach().cpu()) if request['gradient'] else None
        parity=loss_error<1e-8 and power_error<1e-8 and (
            not request['gradient'] or torch.allclose(a['gradient'],b['gradient'],atol=1e-7,rtol=2e-5))
        if not parity:raise ValueError(f'Timing parity failed: loss={loss_error}, R/T/A={power_error}, gradient={gradient_error}')
        speedup=results['reference']['warm_median_seconds']/results['auto']['warm_median_seconds']
    return dict(status='completed',passed=True,timing_finite=True,parity_passed=parity,
        parity_checked=parity is not None,warmup_excluded=True,gradient=request['gradient'],
        objective_name='mean_R_plus_T',wavelengths_nm=request['waves'],repeats=request['repeats'],
        timing_backends=backends,timings=results,warm_speedup=speedup,
        loss_abs_error=loss_error,power_max_abs_error=power_error,logits_gradient_max_abs_error=gradient_error)


def _best(folder,profile):
    path=profile or folder/'report.json'
    data=json.loads(path.read_text(encoding='utf-8-sig'))
    logits=data.get('best_logits',data.get('logits'))
    if logits is None:logits=data.get('best',{}).get('logits')
    if logits is None:raise ValueError('No saved best dual-radius shape; run optimization or specify --profile-json.')
    return logits


def draw_profile(ax,radii,c):
    import matplotlib.pyplot as plt
    dz=c['height_nm']/c['layers']
    for i,(core,outer) in enumerate(radii):
        ax.add_patch(plt.Rectangle((-outer,i*dz),2*outer,dz,facecolor='#d59a42',edgecolor='#855b1f',linewidth=.45))
        ax.add_patch(plt.Rectangle((-core,i*dz),2*core,dz,facecolor='#b8dbef',edgecolor='#34718f',linewidth=.45))
    p=c['period_nm'];h=c['height_nm']
    ax.add_patch(plt.Rectangle((-p/2,h),p,.10*h,facecolor='#b8dbef',edgecolor='#34718f',linewidth=.6))
    ax.text(0,h*1.055,'PMMA substrate',ha='center',va='center',fontsize=8)
    ax.axvline(0,color='gray',linewidth=.5,linestyle=':')
    ax.set(xlim=(-p*.55,p*.55),ylim=(h*1.11,0),xlabel='Position from center (nm)',ylabel='Depth (nm)',
           title=f"PMMA core / {c.get('material','metal')} shell; {c['layers']} layers")


def _plot(path,rows,c,logits):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:return 'matplotlib unavailable; CSV and JSON are saved'
    from .geometry import radii_values
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),gridspec_kw={'width_ratios':[1.5,1]})
    waves=[r['wavelength_nm'] for r in rows]
    for key,label in zip(POWER_KEYS,('R','T into PMMA','A = 1 - R - T')):
        axes[0].plot(waves,[r[key] for r in rows],label=label)
    axes[0].set(xlabel='Wavelength (nm)',ylabel='Power fraction',ylim=(-.01,1.01))
    axes[0].legend();axes[0].grid(alpha=.25)
    draw_profile(axes[1],radii_values(logits,c),c)
    fig.tight_layout();fig.savefig(path,dpi=180);plt.close(fig)
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
        if len(logits)!=2*(c['layers']+1):raise ValueError('Best shape must contain 2*(layers+1) logits.')
        if any(not math.isfinite(x) for x in logits):raise ValueError('Nonfinite saved logits.')
        base=dict(config={**c,'rcwa_acceleration':args.analysis_backend},lattice=lattice,device=args.device,
            signature=signature,runtime_environment=runtime,logits=logits,waves=waves,
            gradient=not args.forward_only,repeats=args.repeats)
        if mode=='timing':base['timing_backends']=parse_timing_backends(getattr(args,'timing_backends','auto'))
        ident=signature_for(dict(base,analysis=mode,pairs=pairs))[:16]
        folder=output/lattice/'postprocess'/mode/ident;folder.mkdir(parents=True,exist_ok=True)
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
            objective_name='mean_R_plus_T',optimization_goal='maximize mean absorptance = 1 - mean(R+T)',
            source_profile_logits=logits,wavelengths_nm=waves,conditions=pairs,
            requested_device=args.device,cuda_execution_performed=runtime['device'].startswith('cuda'),outcomes=outcomes)
        if mode=='spectrum':
            write_csv(folder/'spectrum.csv',outcomes[0]['rows'])
            report['spectrum_csv']=str(folder/'spectrum.csv')
            report['plot']=_plot(folder/'spectrum.png',outcomes[0]['rows'],c,logits)
            report.update({key:outcomes[0][key] for key in ('mean_reflectance','mean_transmittance','mean_absorptance')})
        elif mode=='convergence':
            highest=next(o for o in outcomes if (o['order'],o['grid'])==(max(orders),max(grids)))
            rows=[]
            for result in outcomes:
                error=max(abs(a[k]-b[k]) for a,b in zip(result['rows'],highest['rows']) for k in POWER_KEYS)
                rows.append(dict(order=result['order'],grid=result['grid'],max_power_abs_difference_from_highest=error,
                    within_tolerance=error<=c['verification_tolerance'],worker_seconds=result['process_wall_seconds'],
                    rcwa_seconds=result['seconds'],mean_absorptance=result['mean_absorptance'],
                    mean_reflectance=result['mean_reflectance'],mean_transmittance=result['mean_transmittance']))
            needed=[]
            if len(orders)>1:needed.append((orders[-2],max(grids)))
            if len(grids)>1:needed.append((max(orders),grids[-2]))
            report.update(highest_reference=dict(order=max(orders),grid=max(grids)),comparison=rows,
                compared_power_keys=list(POWER_KEYS),tolerance=c['verification_tolerance'],
                order_convergence_tested=len(orders)>1,grid_convergence_tested=len(grids)>1,
                numerical_convergence_passed=all(r['within_tolerance'] for r in rows if (r['order'],r['grid']) in needed) if len(needed)==2 else None,
                scope='Fixed dual-radius profile; finite R/T/A order/grid comparison, not an exact-reference bound.')
            write_csv(folder/'convergence.csv',rows)
        else:report.update(outcomes[0])
        write_json(folder/'report.json',report)
        write_json(output/lattice/f'latest_{mode}.json',dict(report=str(folder/'report.json'),status='completed'))
        reports[lattice]=str(folder/'report.json');print(f'Saved {folder}',flush=True)
    write_json(output/f'analysis_workflow_{mode}.json',dict(status='completed',reports=reports))
    return 'completed'
