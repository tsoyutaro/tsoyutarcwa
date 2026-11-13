"""Before/after checks of every requested even response and measured speed."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
from scipy.stats import qmc
from model import HERE,Materials,geometry,layers,save_json,save_csv,solver_root


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--solver-root',type=Path)
    p.add_argument('--device',default='cpu')
    p.add_argument('--suite',choices=['quick','standard'],default='standard')
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--skip-timing',action='store_true')
    a=p.parse_args();solver_root(a.solver_root)
    import torch
    from solver import local_surface,local_surface_fast,clear_plan_cache,plan_cache_info
    from fast_solver import canonical_device,get_plan
    from optics import phase_sample_reference,PreparedSheet,average_sheet,admittance,propagating
    from reference_optics import average_sheet as old_average
    from run import calculate,parser
    device=canonical_device(a.device);torch.set_num_threads(2)
    a.device=str(device)
    out=a.output_dir or HERE/'validation'/('acceleration_'+a.device.replace(':','_'))
    mat=Materials();checks=[];details=[]
    def record(name,error,tol):
        checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(error<=tol)))
        print(name,f'{error:.3g}',checks[-1]['passed'],flush=True)
    cases=[]
    for mode,period in ([('scaled',3720.)] if a.suite=='quick' else [('scaled',3720.),('scaled',3920.),('legacy',3920.)]):
        for t in [0.,50.]:
            for void in [1.,1.333]:
                for wave in ([550.] if a.suite=='quick' else [400.,550.,700.]):
                    cases.append((mode,period,t,void,wave,16 if a.suite=='quick' else 24,30 if a.suite=='quick' else 120))
    if a.suite=='standard':
        cases += [('scaled',1860.,t,v,550.,24,120) for t in [0.,50.] for v in [1.,1.333]]
        cases += [('scaled',p,t,v,w,24,120) for p,w in [(3720.,620.),(3920.,560.)] for t in [0.,50.] for v in [1.,1.333]]
        cases += [('scaled',3720.,50.,v,w,40,120) for v in [1.,1.333] for w in [400.,550.,700.]]
        cases += [('scaled',3920.,50.,v,550.,24,240) for v in [1.,1.333]]
        cases += [('scaled',100.,t,v,550.,8,30) for t in [0.,50.] for v in [1.,1.333]]
    max_amp=max_power=max_local=max_adaptive=0.
    for num,(mode,period,t,void,wave,m,nz) in enumerate(cases,1):
        g=geometry(period=period,gold=t) if mode=='scaled' else geometry(period,3720.,1500.,1500.,t)
        stack=layers(g,nz);n=mat.pmma(wave)[0];ng=mat.gold(wave) if t else n
        orders,rr,tr=local_surface(g,stack,wave,n,ng,void,m,a.device)
        _,rf,tf=local_surface_fast(g,stack,wave,n,ng,void,m,a.device)
        u=get_plan(stack,period,m,a.device).basis
        amp=max(np.max(abs((rr-rf)@u)),np.max(abs((tr-tf)@u)))
        kx=orders*wave/period;zero=m
        result_error=local_error=adaptive_error=0.
        for j,pol in enumerate(['s','p']):
            d=len(set(abs(orders[propagating(n,kx)])))
            phases=qmc.Sobol(d,scramble=True,seed=1729).random_base2(10)*2*np.pi
            ref=np.array(phase_sample_reference(rr[j],tr[j],orders,wave,period,n,1.333,pol,phases))
            value=np.array(PreparedSheet(rf[j],tf[j],orders,wave,period,n,1.333,pol).sample(phases))
            result_error=max(result_error,float(np.max(abs(ref-value))),float(np.max(abs(ref.sum(axis=1)-value.sum(axis=1)))))
            for old,new,index in [(rr,rf,n),(tr,tf,1.333)]:
                y=admittance(index,kx,pol);yin=admittance(n,[0.],pol)[0].real
                weight=np.where(propagating(index,kx),y.real,0)/yin
                local_error=max(local_error,float(np.max(abs(abs(old[j,:,zero])**2-abs(new[j,:,zero])**2)*weight)))
            if mode=='scaled' and period==3720. and m==24 and nz==120 and wave in [400.,550.,700.]:
                ar,at,meta=old_average(rr[j],tr[j],orders,wave,period,n,1.333,pol,points=512,max_points=65536,tolerance=2e-4)
                br,bt,newmeta=average_sheet(rf[j],tf[j],orders,wave,period,n,1.333,pol,points=512,max_points=65536,tolerance=2e-4)
                adaptive_error=max(adaptive_error,float(np.max(abs(ar-br))),float(np.max(abs(at-bt))),abs((ar.sum()+at.sum())-(br.sum()+bt.sum())))
                if meta['points_per_scramble']!=newmeta['points_per_scramble']:
                    raise AssertionError('Adaptive quadrature stopped at a different sample count.')
        max_amp=max(max_amp,amp);max_power=max(max_power,result_error);max_local=max(max_local,local_error);max_adaptive=max(max_adaptive,adaptive_error)
        details.append(dict(mode=mode,period_nm=period,gold_nm=t,void_index=void,wavelength_nm=wave,order=m,slices=nz,
            amplitude_even_sector_max_error=float(amp),fixed_phase_Rm_Tm_and_totals_max_error=result_error,
            local_Rm_Tm_max_error=local_error,adaptive_Rm_Tm_max_error=adaptive_error))
        print(f'{num}/{len(cases)} {mode} P{period:g} Au{t:g} nvoid{void:g} wl{wave:g} M{m} Nz{nz}: power {result_error:.2g}',flush=True)
        save_csv(out/'comparison_cases.csv',details)
    record('all even-sector local amplitudes',max_amp,1e-8)
    record('all fixed-phase R_m/T_m and R/T totals',max_power,1e-9)
    record('local single-pass R_m/T_m',max_local,1e-9)
    record('old vs new adaptive phase average',max_adaptive,1e-9)
    ratio_error=0.
    for period in [100.,200.,1860.,3720.,3920.,7440.]:
        g=geometry(period=period,gold=50.)
        ratio_error=max(ratio_error,abs(g['base_width_nm']/period-1),abs(g['tip_width_nm']/period-25/62),abs(g['height_nm']/period-25/62),abs(g['gap_nm']))
        assert g['gold_vertical_nm']==50.
    record('scaled ratio and no root gap',ratio_error,1e-12)
    g=geometry(period=3720.)
    record('reference period retains exactly 3720/1500/1500 nm',max(abs(g[k]-v) for k,v in [('base_width_nm',3720),('tip_width_nm',1500),('height_nm',1500)]),0.)
    clear_plan_cache();g=geometry(gold=50.);stack=layers(g,120)
    for w in [400.,550.,700.]:get_plan(stack,3720.,24,a.device)
    info=plan_cache_info();assert info['misses']==1 and info['hits']==2,info
    record('shared geometry plan reused across three wavelengths',0.,0.)
    timings={}
    def sync():
        if device.type=='cuda':torch.cuda.synchronize(device)
    if not a.skip_timing:
        if a.repeats<2:raise ValueError('Use at least two timing repeats.')
        # Warm kernels before measuring the cold-geometry three-wave pass.
        n=mat.pmma(550.)[0];ng=mat.gold(550.)
        local_surface(g,stack,550.,n,ng,1.,24,a.device);local_surface_fast(g,stack,550.,n,ng,1.,24,a.device)
        for name,fn in [('reference',local_surface),('fast',local_surface_fast)]:
            if name=='fast':clear_plan_cache()
            values=[]
            for repeat in range(a.repeats+1):
                sync();start=time.perf_counter()
                for w in [400.,550.,700.]:fn(g,stack,w,mat.pmma(w)[0],mat.gold(w),1.,24,a.device)
                sync();values.append(time.perf_counter()-start)
            timings[name]=dict(first_three_wave_seconds=values[0],warm_three_wave_seconds=values[1:],median_warm_seconds=float(np.median(values[1:])))
        timings['local_speedup']=timings['reference']['median_warm_seconds']/timings['fast']['median_warm_seconds']
        for name in ['reference','fast']:
            args=parser().parse_args(['--acceleration',name,'--device',a.device,'--order','24','--slices','120','--phase-max','65536'])
            values=[]
            for _ in range(a.repeats):
                sync();start=time.perf_counter();calculate(args,g,'gold','air',550.,mat,stack=stack);sync()
                values.append(time.perf_counter()-start)
            timings[name]['whole_case_550nm_seconds']=values
            timings[name]['median_whole_case_550nm_seconds']=float(np.median(values))
        timings['whole_case_speedup']=timings['reference']['median_whole_case_550nm_seconds']/timings['fast']['median_whole_case_550nm_seconds']
    report=dict(passed=all(c['passed'] for c in checks),cases=len(cases),polarization_cases=2*len(cases),
        device=a.device,torch=str(torch.__version__),cuda_execution_performed=device.type=='cuda',
        checks=checks,case_csv='comparison_cases.csv',timings=timings,geometry_plan_cache=info,
        source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(HERE.glob('*.py'))},
        limitation='Floating-point equality within stated tolerances, for tested cases. GPU equivalence/timing only if this report says cuda_execution_performed=true.')
    save_json(out/'validation.json',report)
    print(json.dumps(dict(passed=report['passed'],cases=len(cases),timings=timings),indent=2),flush=True)
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
