"""Independent checks of the RCWA-FFF reduction, symmetry and graded-index integrals."""
from pathlib import Path
import argparse,json,math,sys
import numpy as np
from scipy.stats import qmc

STUDY=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(STUDY))
from model import Materials,geometry,layers,save_json,solver_root


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='cpu')
    p.add_argument('--suite',choices=['quick','standard'],default='standard')
    p.add_argument('--solver-root',type=Path)
    p.add_argument('--output-dir',type=Path)
    a=p.parse_args();solver_root(a.solver_root)
    import torch
    from fast_solver import local_surface_fast
    from fff1d.solver import local_surface_fff
    from fff1d.operators import Transition
    from optics import PreparedSheet,propagating
    torch.set_num_threads(2)
    try:
        from threadpoolctl import threadpool_limits
        guard=threadpool_limits(limits=2)
    except ImportError:pass
    out=a.output_dir or STUDY/'validation/fff1d'/a.device.replace(':','_')
    mat=Materials();checks=[];cases=[]
    def check(name,error,tolerance):
        checks.append(dict(name=name,error=float(error),tolerance=tolerance,passed=bool(error<=tolerance)))
        print(name,f'{error:.3g}',checks[-1]['passed'],flush=True)
    reduction=[]
    for period in ([3720.] if a.suite=='quick' else [3720.,3920.]):
        for gold in [0.,50.]:
            for void in [1.,1.333]:
                for wave in ([700.] if a.suite=='quick' else [400.,550.,700.]):
                    reduction.append((period,gold,void,wave,16,30))
    if a.suite=='standard':
        reduction += [(3720.,50.,v,620.,16,30) for v in [1.,1.333]]
        reduction += [(3720.,50.,v,700.,32,120) for v in [1.,1.333]]
    amplitude=power=0.
    for num,(period,gold,void,wave,m,nz) in enumerate(reduction,1):
        g=geometry(period=period,gold=gold);stack=layers(g,nz)
        n=mat.pmma(wave)[0];ng=mat.gold(wave) if gold else n
        orders,rf,tf=local_surface_fast(g,stack,wave,n,ng,void,m,a.device)
        err=flux=0.
        for j,pol in enumerate(['s','p']):
            _,r,t=local_surface_fff(g,stack,wave,n,ng,void,m,a.device,theta_deg=90.,polarization=pol)
            err=max(err,float(np.max(abs(r[0]-rf[j]))),float(np.max(abs(t[0]-tf[j]))))
            kx=orders*wave/period;dim=len(set(abs(orders[propagating(n,kx)])))
            phases=qmc.Sobol(dim,scramble=True,seed=1729).random_base2(10)*2*np.pi
            before=np.array(PreparedSheet(rf[j],tf[j],orders,wave,period,n,1.333,pol).sample(phases))
            after=np.array(PreparedSheet(r[0],t[0],orders,wave,period,n,1.333,pol).sample(phases))
            flux=max(flux,float(np.max(abs(before-after))),float(np.max(abs(before.sum(axis=1)-after.sum(axis=1)))))
        amplitude=max(amplitude,err);power=max(power,flux)
        cases.append(dict(check='theta90',period_nm=period,gold_nm=gold,void_index=void,wavelength_nm=wave,
            order=m,slices=nz,maximum_amplitude_difference=err,maximum_power_difference=flux))
        save_json(out/'cases.json',cases)
        print(f'reduction {num}/{len(reduction)} P{period:g} Au{gold:g} n{void:g} wl{wave:g}: {err:.2g}',flush=True)
    check('theta=90, GIL=0 vs original even RCWA all amplitudes',amplitude,1e-9)
    check('theta=90 all per-order R/T and R/T totals',power,1e-9)
    # Independent full Fourier calculation checks the nonzero xz coupling and reduction.
    symmetric=0.
    for delta,scope in [(0.,'both'),(1.,'inner'),(1.,'outer'),(1.,'both')]:
        g=geometry(gold=50.);stack=layers(g,12);n=mat.pmma(700.)[0];ng=mat.gold(700.);m=8
        orders,r,t=local_surface_fff(g,stack,700.,n,ng,1.333,m,a.device,gil_nm=delta,gil_interfaces=scope)
        _,fr,ft=local_surface_fff(g,stack,700.,n,ng,1.333,m,a.device,gil_nm=delta,gil_interfaces=scope,symmetry='full')
        u=np.zeros((2*m+1,m+1));u[m,0]=1
        for j in range(1,m+1):u[m-j,j]=u[m+j,j]=1/math.sqrt(2)
        e=max(np.max(abs((fr[0]-r[0])@u)),np.max(abs((ft[0]-t[0])@u)))
        symmetric=max(symmetric,float(e))
    check('profile normals and GIL: even vs full Fourier response',symmetric,1e-8)
    # GIL integral at harmonic zero has an independent closed form.
    indices=dict(a=1.49+0j,b=.16+5.26j)
    delta,period=1.,3720.
    transition=Transition(1000.,period,delta,'a','b',32)
    e,h=transition.correction(indices,period,np.arange(65))
    na,nb=indices['a'],indices['b']
    e0=-delta/(3*period)*(na-nb)**2
    h0=2*delta/period*(1/(na*nb)-.5*(1/na**2+1/nb**2))
    check('GIL DC epsilon integral vs exact linear-complex-index formula',abs(e[0]-e0),1e-12)
    check('GIL DC inverse-epsilon integral vs exact formula',abs(h[0]-h0),1e-12)
    gq=0.
    for delta in [.5,1.,2.]:
        t16=Transition(1000.,period,delta,'a','b',16)
        t32=Transition(1000.,period,delta,'a','b',32)
        ea,ha=t16.correction(indices,period,np.arange(129));eb,hb=t32.correction(indices,period,np.arange(129))
        gq=max(gq,float(np.max(abs(ea-eb))),float(np.max(abs(ha-hb))))
    check('GIL quadrature 16 vs 32: all coefficients through harmonic 128',gq,1e-10)
    # Uniform film limit: an independent Airy formula, at every projection angle.
    g=geometry(gold=50.);stack=[dict(z_top_nm=0.,z_bottom_nm=50.,core_width_nm=0.,outer_width_nm=g['period_nm'])]
    n=mat.pmma(700.)[0];ng=mat.gold(700.);k0=2*np.pi/700.;outn=1.333
    r12=(n-ng)/(n+ng);r23=(ng-outn)/(ng+outn)
    den=1+r12*r23*np.exp(2j*k0*ng*50)
    analytic_r=(r12+r23*np.exp(2j*k0*ng*50))/den
    analytic_t=(2*n/(n+ng))*(2*ng/(ng+outn))*np.exp(1j*k0*ng*50)/den
    airy=0.
    for angle in [0.,53.49855887949368,90.]:
        for pol in ['s','p']:
            _,r,t=local_surface_fff(g,stack,700.,n,ng,1.,8,a.device,theta_deg=angle,polarization=pol)
            airy=max(airy,abs(r[0,8,8]-analytic_r),abs(t[0,8,8]-analytic_t))
    check('uniform Au 50nm independent Airy amplitudes at all angles',airy,1e-9)
    report=dict(passed=all(c['passed'] for c in checks),checks=checks,reduction_cases=len(reduction),
        device=a.device,torch=str(torch.__version__),cuda_execution_performed=a.device.startswith('cuda'),
        method='paper-left ordered FFF, local exponentials and Redheffer',
        limitation='Finite-discretization implementation tests, not convergence certification for Au sheet relief.')
    save_json(out/'validation.json',report)
    print(json.dumps(report,indent=2),flush=True)
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
