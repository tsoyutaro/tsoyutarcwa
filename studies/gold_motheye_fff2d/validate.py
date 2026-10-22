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
        if not np.isfinite(error) or error > tolerance:
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
                      method="dm-fff",homogeneous=True,epsilon_out=2.25,dm_step_nm=step,dm_integrator="rk4")
            errors.append(max(abs(r.R-reference.R),abs(r.P_sub-reference.P_sub)))
        if not errors[1] < errors[0]/8:
            raise AssertionError(f"RK4 step refinement failed: {errors}")
        return dict(step_nm=[10,5],errors=errors)
    check("DM-FFF RK4 step refinement against exact constant generator",refinement)
    def vertical_reduction():
        errors = []
        for lattice in ('square','triangular'):
            for shape in ('auto','rectangular'):
                cylinder = Geometry(height_nm=80,tip_radius_nm=60,base_radius_nm=60,
                                    lattice=lattice,harmonic_shape=shape)
                b = Backend(device)
                f = MatrixFunction(cylinder,1,48,550.,(.25+3j)**2,b,method='rcwa-fff',ordering=ordering)
                nv = MatrixFunction(cylinder,1,48,550.,(.25+3j)**2,b,method='slice-nvm',ordering=ordering)
                k_error = float((f(40.)[0]-nv(40.)[0]).abs().max())
                a = solve_case(cylinder,550.,(.25+3j)**2,order=1,slices=4,grid=48,device=device)
                v = solve_case(cylinder,550.,(.25+3j)**2,order=1,slices=4,grid=48,device=device,method='slice-nvm')
                s_error = max(float((x-y).abs().max()) for x,y in zip(vars(a.S).values(),vars(v.S).values()))
                power_error = max(float(abs(a.R_orders-v.R_orders).max()),float(abs(a.T_orders-v.T_orders).max()))
                close(max(k_error,s_error,power_error),0.,1e-11)
                errors.append(dict(lattice=lattice,shape=shape,K_error=k_error,S_error=s_error,order_power_error=power_error))
        return errors
    check('Vertical cylinders: 3D normal reduces to NV in K, all S blocks and every order',vertical_reduction)

    for lattice in ('square','triangular'):
        for profile in ('cylinder','cone'):
            def native(lattice=lattice,profile=profile):
                from studies.gold_motheye_fff2d.native_reference import compare_native
                from studies.shared.gold_dispersion import DEFAULT_GOLD_CSV, load_gold_csv
                cylinder = profile=='cylinder'
                geometry = Geometry(height_nm=80,tip_radius_nm=60 if cylinder else 30,base_radius_nm=60,
                                    lattice=lattice,harmonic_shape='rectangular')
                details = compare_native(geometry,550.,load_gold_csv(DEFAULT_GOLD_CSV)(550.) if cylinder else 4.,
                                         device=device,method='rcwa-fff' if cylinder else 'slice-nvm',
                                         epsilon_out=None if cylinder else 2.25)
                # SciPy j1 and torch.special.bessel_j1 differ in analytical
                # disk coefficients by ~1e-8 for these arguments. Retain both
                # independent backends and test below the sweep's 1e-6 bound.
                close(details['max_complex_amplitude_error'],0.,1e-7)
                close(details['max_order_power_error'],0.,1e-8)
                close(details['projection_matrix_error'],0.,1e-12)
                return details
            check(f'Independent rcwa_ext NV: {lattice} {profile} complex amplitudes and order powers',native)

    def continuous_integrator():
        from scipy.integrate import solve_ivp
        from studies.gold_motheye_fff2d.solver import propagation_s
        geom = Geometry(height_nm=35,tip_radius_nm=30,base_radius_nm=60)
        backend = Backend(device)
        fun = MatrixFunction(geom,0,48,550.,(1.8+.2j)**2,backend,ordering=ordering)
        k0 = 2*np.pi/550
        def rhs(z,y):
            return (1j*k0*backend.numpy(fun(z)[0])@y.reshape(4,4)).ravel()
        independent = solve_ivp(rhs,(0,35),np.eye(4,dtype=complex).ravel(),method='DOP853',rtol=2e-12,atol=2e-14)
        if not independent.success:
            raise AssertionError(independent.message)
        reference = propagation_s(backend.array(independent.y[:,-1].reshape(4,4)))
        from studies.gold_motheye_fff2d.integration import dm_section, diagnostics
        stats = diagnostics()
        computed = dm_section(fun,0.,35.,k0,35.,backend,'cf4-adaptive',1e-9,1e-12,1e-5,stats)
        error = max(float((a-b).abs().max()) for a,b in zip(vars(computed).values(),vars(reference).values()))
        close(error,0.,1e-8)
        if stats['rejected_steps']==0:
            raise AssertionError('The error controller was not exercised')
        try:
            dm_section(fun,0.,35.,k0,35.,backend,'cf4-adaptive',1e-15,1e-17,35.,diagnostics())
        except FloatingPointError:
            minimum_step_guard = True
        else:
            raise AssertionError('A failed local error test was silently accepted')
        return dict(max_S_error_vs_DOP853=error,minimum_step_guard=minimum_step_guard,**stats)
    check('Adaptive CF4: varying noncommuting Maxwell operator vs independent DOP853',continuous_integrator)

    def gil_limit():
        from dataclasses import replace
        from studies.gold_motheye_fff2d.model import Operators, harmonics
        base = Geometry(height_nm=80,tip_radius_nm=30,base_radius_nm=60)
        orders = harmonics(1)
        epsilon = (.25+3j)**2
        sharp = Operators(base,orders,48).material(40.,epsilon)
        smooth = Operators(replace(base,gil_nm=1e-7),orders,48).material(40.,epsilon)
        error = max(float(abs(a-b).max()) for a,b in zip(sharp,smooth))
        close(error,0.,1e-7)
        return dict(max_matrix_error_at_delta_1e_7_nm=error,
                    placement=replace(base,gil_nm=2.).gil_side,total_width_nm=2.)
    check('Centered GIL approaches the sharp analytical disk as width tends to zero',gil_limit)
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
