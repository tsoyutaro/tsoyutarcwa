"""Independent power, passive-loss and internal-field checks for NV/ASR.

Run with --solver-root to reproduce the same checks using an older solver.
Energy conservation is a physical consistency check, not a convergence test.
"""
from __future__ import annotations
import argparse
import cmath
import json
import math
from pathlib import Path
import sys
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--solver-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--json', type=Path, required=True)
    parser.add_argument('--allow-failures', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0, str(args.solver_root.resolve()))
    import torch
    from rcwa_ext import (ASROptions, NVMOptions, AutoRCWA, Circle, Rectangle,
                          Lattice, LayerSpec, Material, OutputSpec, GroupTheoryOptions)
    torch.set_num_threads(2)
    checks = []
    def check(name, fn):
        try:
            details = fn()
            checks.append({'name': name, 'passed': True, 'details': details})
        except Exception as exc:
            checks.append({'name': name, 'passed': False, 'error': str(exc),
                           'traceback': traceback.format_exc(limit=3)})
        print(name, checks[-1]['passed'], flush=True)
    def close(value, expected, limit):
        assert abs(value-expected) < limit, (value, expected, limit)
    def setup(*, order=3, grid=128, wavelength=.45, lattice='square', pol=None,
              fields=False, angle=0., azimuth=0., period=.3, eps_out=2.25,
              algorithm='redheffer'):
        cell=Lattice.triangular(period) if lattice=='triangular' else Lattice.square(period)
        sim=AutoRCWA(freq=1/wavelength, order=[order, order], lattice=cell,
                     outputs=OutputSpec(smatrix_size='full', fields='all' if fields else 'none'),
                     cascade=algorithm,
                     asr=ASROptions(grid=(grid, grid)), nvm=NVMOptions(grid=(grid, grid)),
                     group_theory=GroupTheoryOptions(enabled=pol is not None,
                         polarization=pol, symmetry='d6' if lattice=='triangular' else 'auto', strict=True),
                     dtype=torch.complex128, device=args.device)
        sim.add_input_layer(eps=1.); sim.add_output_layer(eps=eps_out)
        sim.set_incident_angle(angle, azimuth)
        return sim
    def add(sim, method, eps=4., thickness=.16, radius=.085, rectangle=False, background=1.):
        geometry=Rectangle((.16, .1)) if rectangle else Circle(radius)
        sim.add_structured_layer(LayerSpec(thickness, geometry,
            background=Material(background), inclusion=Material(eps), method=method))
    def power(sim, pol='x'):
        n=sim.order_N
        source=torch.zeros((2*n,1), dtype=sim._dtype, device=sim._device)
        source[(0 if pol=='x' else n)+n//2,0]=1.
        def flux(e,v):
            h=v@e
            return float(.5*torch.real((e[:n]*h[n:].conj()-e[n:]*h[:n].conj()).sum()).detach())
        incoming=flux(source,sim.Vi)
        r=flux(sim.S[1]@source,sim.Vi)/incoming
        t=flux(sim.S[0]@source,sim.Vo)/incoming
        return {'R':r,'T':t,'A':1-r-t}
    methods=[('nvm',False),('matched-asr',False),('asr-fr',True)]
    for method,rectangle in methods:
        for name,angle,azimuth,multilayer in [('normal',0.,0.,False),
                ('oblique',.47,.54,False),('multilayer',0.,0.,True)]:
            def lossless(method=method, rectangle=rectangle, angle=angle,
                         azimuth=azimuth,multilayer=multilayer):
                sim=setup(angle=angle,azimuth=azimuth)
                add(sim,method,rectangle=rectangle)
                if multilayer:add(sim,method,eps=7.29,thickness=.09,radius=.065,rectangle=rectangle)
                sim.solve_global_smatrix(); result=power(sim)
                close(result['A'],0.,2e-9)
                return result
            check(f'{method}: {name} lossless power', lossless)
        def film(method=method,rectangle=rectangle):
            index=1.73+.17j; thickness=.16; wavelength=.6
            sim=setup(wavelength=wavelength)
            add(sim,method,eps=index**2,background=index**2,thickness=thickness,rectangle=rectangle)
            sim.solve_global_smatrix(); result=power(sim)
            r01=(1-index)/(1+index); r12=(index-1.5)/(index+1.5)
            phase=cmath.exp(2j*math.pi*index*thickness/wavelength)
            denominator=1+r01*r12*phase**2
            r=(r01+r12*phase**2)/denominator
            t=(2/(1+index))*(2*index/(index+1.5))*phase/denominator
            reference={'R':abs(r)**2,'T':1.5*abs(t)**2}
            reference['A']=1-reference['R']-reference['T']
            error=max(abs(result[k]-reference[k]) for k in result)
            assert error<1e-4 and result['A']>.1, (result,reference,error)
            return {'actual':result,'analytic_fresnel':reference,'max_error':error}
        check(f'{method}: complex homogeneous film keeps true loss',film)
        def metal(method=method,rectangle=rectangle):
            sim=setup(); add(sim,method,eps=(.25+3j)**2,rectangle=rectangle)
            sim.solve_global_smatrix(); result=power(sim)
            assert all(math.isfinite(v) and -1e-9<=v<=1+1e-9 for v in result.values()),result
            assert result['A']>.01,result
            return result
        check(f'{method}: passive metal absorption',metal)

    for method in ('nvm','matched-asr'):
        def hexagonal(method=method):
            values=[]
            for pol in ('x','y'):
                sim=setup(lattice='triangular',pol=pol)
                add(sim,method);sim.solve_global_smatrix();result=power(sim,pol)
                close(result['A'],0.,2e-9);values.append(result)
            error=max(abs(values[0][k]-values[1][k]) for k in values[0])
            assert error<1e-9,error
            return {'x':values[0],'y':values[1],'max_difference':error}
        check(f'{method}: D6 x/y power equality',hexagonal)
        def channels(method=method):
            sim=setup(order=2,period=1.,wavelength=1/1.4)
            add(sim,method,thickness=.4,radius=.23);sim.solve_global_smatrix()
            n=sim.order_N;eye=sim._eye(n);zero=torch.zeros_like(eye)
            c=torch.cat((torch.cat((zero,eye),1),torch.cat((-eye,zero),1)),0)
            def metric(v):return (c@v+(c@v).mH)*.25
            qi,qo=metric(sim.Vi),metric(sim.Vo)
            eigen,basis=torch.linalg.eigh(qi); keep=eigen>1e-8
            normalized=basis[:,keep]/eigen[keep].sqrt()
            defect=normalized.mH@(sim.S[1].mH@qi@sim.S[1]+sim.S[0].mH@qo@sim.S[0]-qi)@normalized
            error=float(defect.abs().max());assert error<1e-8,error
            return {'open_incident_channels':int(keep.sum()),'power_gram_error':error}
        check(f'{method}: all open-channel superpositions conserve power',channels)

    def thick_fields():
        sim=setup(period=1.,wavelength=2.,pol='x',fields=True)
        for radius,thickness in ((.28,.9),(.21,.6)):
            add(sim,'matched-asr',eps=(.25+3j)**2,radius=radius,thickness=thickness)
        sim.solve_global_smatrix();sim.source_planewave(amplitude=[1.,0.],direction='forward')
        residual=float(sim._reduced_field_boundary_residual)
        assert residual<2e-9,residual
        axis=torch.linspace(.01,.99,16,dtype=torch.float64,device=args.device)
        for layer,depth in ((0,.45),(1,.3)):
            e,h=sim.field_xy(layer,axis,axis,z_prop=depth)
            assert all(bool(torch.isfinite(component).all()) for component in (*e,*h))
        return {'boundary_residual':residual,'six_component_fields_finite':True,'power':power(sim)}
    check('Thick absorbing multilayer: stable bidirectional internal fields',thick_fields)

    report={'passed':all(c['passed'] for c in checks), 'solver_root':str(args.solver_root.resolve()),
            'device':args.device,'dtype':'complex128','torch_version':torch.__version__,
            'check_count':len(checks),'passed_count':sum(c['passed'] for c in checks),'checks':checks,
            'scope':'Physical consistency at fixed order, not Fourier/grid convergence or experimental validation.'}
    args.json.parent.mkdir(parents=True,exist_ok=True)
    args.json.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return 0 if report['passed'] or args.allow_failures else 1

if __name__=='__main__':
    raise SystemExit(main())
