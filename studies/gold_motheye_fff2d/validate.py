"""Independent Fresnel, Maxwell, flux and dissipated-power checks."""
from __future__ import annotations

import argparse
import cmath
import json
from pathlib import Path
import sys
import traceback
import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    from studies.gold_motheye_fff2d.model import Geometry
    from studies.gold_motheye_fff2d.solver import Backend, MatrixFunction, block, constitutive, generator_from_tensor, solve
else:
    from .model import Geometry
    from .solver import Backend, MatrixFunction, block, constitutive, generator_from_tensor, solve


def checks(device="cpu",ordering="symmetric"):
    report = []
    def solve_case(*args,**kwargs):
        return solve(*args,ordering=ordering,**kwargs)
    def check(name,fn):
        try:
            details = fn()
            report.append(dict(name=name,passed=True,details=details))
        except Exception as exc:
            report.append(dict(name=name,passed=False,error=str(exc),traceback=traceback.format_exc(limit=4)))
        print(f"{name}: {'PASS' if report[-1]['passed'] else 'FAIL'}",flush=True)

    def close(actual,expected,tolerance):
        error = abs(actual-expected)
        if error > tolerance:
            raise AssertionError(f"actual={actual}, expected={expected}, error={error}, limit={tolerance}")
        return error

    g = Geometry(height_nm=80,tip_radius_nm=30,base_radius_nm=60)
    for method in ("rcwa-fff","dm-fff"):
        def film(method=method):
            n = 1.8+.2j
            r = solve_case(g,550.,n*n,order=0,slices=4,grid=32,method=method,device=device,
                      homogeneous=True,epsilon_out=2.25,volume_absorption=True)
            p = cmath.exp(2j*np.pi*n*g.height_nm/550)
            r01,r12 = (1-n)/(1+n),(n-1.5)/(n+1.5)
            ref_r = abs((r01+r12*p*p)/(1+r01*r12*p*p))**2
            ref_t = 1.5*abs(2/(1+n)*2*n/(n+1.5)*p/(1+r01*r12*p*p))**2
            error = max(close(r.R,ref_r,2e-8),close(r.P_sub,ref_t,2e-8),
                        close(r.absorption_volume,1-ref_r-ref_t,2e-8))
            return dict(R=r.R,T=r.P_sub,independent_volume_A=r.absorption_volume,max_error=error)
        check(f"{method}: absorbing planar film vs Fresnel/Fabry-Perot",film)

    results = {}
    for method in ("direct","slice-nvm","rcwa-fff","dm-fff"):
        def lossless(method=method):
            r = solve_case(g,550.,4.,order=1,slices=6,grid=48,method=method,device=device,
                      epsilon_out=2.25,dm_step_nm=.5)
            results[method] = r
            error = close(r.R+r.P_sub,1.,2e-8)
            return dict(R=r.R,T=r.P_sub,energy_error=error)
        check(f"{method}: lossless tapered 2D array conserves power",lossless)

    def tilt_and_tensor():
        f = MatrixFunction(g,1,48,550.,4.,Backend(device),ordering=ordering)
        _,_,c = f(40.)
        matrix = block(c)
        hermitian_error = float((matrix-matrix.mH).abs().max())
        close(hermitian_error,0.,1e-11)
        mixed = max(float(c[0][2].abs().max()),float(c[1][2].abs().max()))
        if mixed < 1e-4:
            raise AssertionError("Tilted normals did not generate xz/yz coupling")
        return dict(hermitian_error=hermitian_error,mixed_xz_yz_max=mixed)
    check("Full three-component normal: nonzero mixed blocks and lossless Hermiticity",tilt_and_tensor)

    def tm_reduction():
        from paper_reproductions.mohamad2020.geometries import sinusoidal_aluminium
        from paper_reproductions.mohamad2020.solver import PropagationMatrix, convolution
        f = PropagationMatrix(sinusoidal_aluminium(),632.8,40.,2,128,True)
        z = 90.
        e = f.geometry.sample(f.x,z)
        c,s = f.geometry.tangent(f.x)
        normal = (-s,np.zeros_like(s),c)
        projection = [[convolution(normal[a]*normal[b],2) for b in range(3)] for a in range(3)]
        backend = Backend(device)
        tensor = constitutive(backend.array(convolution(e,2)),backend.array(convolution(1/e,2)),projection,backend,"paper-left")
        k,_ = generator_from_tensor(tensor,backend.array(f.alpha),backend.array(0*f.alpha),backend.eye(5))
        selected = torch.tensor([*range(5),*range(15,20)],device=backend.device)
        reduced = backend.numpy(k[selected[:,None],selected[None,:]])
        error = float(abs(reduced-f(z)).max())
        close(error,0.,2e-10)
        return dict(max_matrix_error=error,note="paper-left reduces to the 1D paper; default uses symmetric ordering")
    check("Vector Maxwell operator reduces to original 1D TM equations",tm_reduction)

    def isotropy_reciprocity():
        x = results["rcwa-fff"]
        y = solve_case(g,550.,4.,order=1,slices=6,grid=48,device=device,epsilon_out=2.25,polarization="y")
        difference = max(close(x.R,y.R,2e-9),close(x.P_sub,y.P_sub,2e-9))
        count = x.S.t.shape[0]//2
        ids = torch.tensor([x.zero,x.zero+count],device=x.S.t.device)
        tf = x.S.t[ids[:,None],ids[None,:]]
        tb = x.S.tr[ids[:,None],ids[None,:]]
        error = float((np.sqrt(1.5)*tf-tb.T/np.sqrt(1.5)).abs().max())
        close(error,0.,2e-9)
        return dict(xy_power_error=difference,zero_order_reciprocity_error=error)
    check("D6 normal-incidence x/y equality and port-normalized reciprocity",isotropy_reciprocity)

    def open_channels():
        wide = Geometry(period_nm=900,height_nm=100,tip_radius_nm=100,base_radius_nm=200)
        r = solve_case(wide,550.,4.,order=1,slices=6,grid=48,device=device,epsilon_out=2.25)
        b = Backend(device); n = r.S.r.shape[0]//2; eye = b.eye(n)
        cross = block([[0*eye,eye],[-eye,0*eye]])
        qi = (cross@r.V_in+(cross@r.V_in).mH)/4
        qo = (cross@r.V_out+(cross@r.V_out).mH)/4
        value,vectors = torch.linalg.eigh(qi)
        keep = value > 1e-9
        normalized = vectors[:,keep]/value[keep].sqrt()
        defect = normalized.mH@(r.S.r.mH@qi@r.S.r+r.S.t.mH@qo@r.S.t-qi)@normalized
        error = float(defect.abs().max())
        close(error,0.,2e-8)
        return dict(open_incident_channels=int(keep.sum()),power_gram_error=error)
    check("Lossless patterned array: arbitrary open-channel superpositions",open_channels)

    def volume():
        r = solve_case(g,550.,(1.8+.12j)**2,order=1,slices=6,grid=48,device=device,
                  epsilon_out=2.25,volume_absorption=True)
        if r.A_moth <= 0 or r.absorption_volume <= 0:
            raise AssertionError("Passive patterned medium produced negative absorption")
        close(r.absorption_error,0.,2e-7)
        close(r.metadata['input_boundary_residual'],0.,2e-9)
        return dict(A_flux=r.A_moth,A_volume=r.absorption_volume,error=r.absorption_error)
    check("Patterned lossy 3D-normal medium: independent volume loss vs boundary flux",volume)

    def refinement():
        reference = solve_case(g,550.,(1.8+.2j)**2,order=0,slices=4,grid=32,device=device,
                          homogeneous=True,epsilon_out=2.25)
        errors = []
        for step in (10.,5.):
            r = solve_case(g,550.,(1.8+.2j)**2,order=0,slices=4,grid=32,device=device,
                      method="dm-fff",homogeneous=True,epsilon_out=2.25,dm_step_nm=step)
            errors.append(max(abs(r.R-reference.R),abs(r.P_sub-reference.P_sub)))
        if not errors[1] < errors[0]/8:
            raise AssertionError(f"RK4 step refinement failed: {errors}")
        return dict(step_nm=[10,5],errors=errors)
    check("DM-FFF RK4 step refinement against exact constant generator",refinement)
    return dict(passed=all(x['passed'] for x in report),check_count=len(report),
                passed_count=sum(x['passed'] for x in report),checks=report,device=str(device),ordering=ordering,
                scope="Low-order implementation/physical consistency; not convergence of the Au moth-eye sweep")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--ordering',choices=('symmetric','paper-left'),default='symmetric')
    parser.add_argument('--json',type=Path,default=Path(__file__).parent/'results/physics_cpu.json')
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    result = checks(args.device,args.ordering)
    args.json.parent.mkdir(parents=True,exist_ok=True)
    args.json.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f"{result['passed_count']}/{result['check_count']} checks passed; {args.json}")
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
