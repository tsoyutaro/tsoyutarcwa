"""Compare production smoothstep ON/OFF with Schuster et al. (2007), Fig. 10.

No production file is modified. One constant-z layer, square rectangular
Fourier truncation (-M..M)^2; no Lanczos filter, ASR, or radius optimization.
The paper plots R_(0,0), not the sum of reflected diffraction orders.
"""
from __future__ import annotations

import argparse
import ast
import csv
import gc
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch


def numbers(value):
    if ':' not in value:
        return [int(v) for v in value.split(',')]
    a, b, *step = [int(v) for v in value.split(':')]
    return list(range(a, b+1, step[0] if step else 1))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def load_solver(name, path):
    spec = importlib.util.spec_from_file_location(name, path/'__init__.py',
                                                submodule_search_locations=[str(path)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.CustomRCWA_NVM


def check_only_projection_differs(on, off):
    def methods(path):
        tree = ast.parse((path/'nvm.py').read_text(encoding='utf-8-sig'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name=='CustomRCWA_NVM')
        return {n.name: ast.dump(n, include_attributes=False) for n in cls.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    a, b = methods(on), methods(off)
    changed = [k for k in a.keys() | b.keys() if a.get(k)!=b.get(k)]
    if changed != ['_projection_matrix']:
        raise RuntimeError(f'Expected only normal projector difference: {changed}')
    common = ['config.py', 'symmetry.py', 'reduced.py', 'scattering.py']
    for file in common:
        if (on/file).read_text(encoding='utf-8-sig') != (off/file).read_text(encoding='utf-8-sig'):
            raise RuntimeError('Other production algebra differs: '+file)
    return dict(changed_class_methods=changed, identical_shared_modules=common,
                factorization='production symmetric anticommutator, both ON and OFF')


def schuster_tensor(eps, delta, projection):
    """Paper Eq.(7): E_ab = delta_ab*[eps] + delta@[N_a*N_b].

    An additional reference check; this never replaces the production formula
    in the main ON/OFF comparison.
    """
    n = eps.shape[0]
    result = torch.block_diag(eps, eps)
    for a in range(2):
        for b in range(2):
            result[a*n:(a+1)*n, b*n:(b+1)*n] += delta @ projection[a*n:(a+1)*n, b*n:(b+1)*n]
    return result


def synchronize(device):
    if device.type=='cuda':
        torch.cuda.synchronize(device)


def observable(sim, tx, rx, polarization):
    n = sim.order_N
    m = int(sim.order[0]); z = m*(2*m+1)+m
    r0 = float((abs(rx[z])**2+abs(rx[n+z])**2).cpu())
    t0 = float((1.5*(abs(tx[z])**2+abs(tx[n+z])**2)).cpu())
    # Direct propagating-order Poynting flux. Exact Rayleigh-cutoff orders
    # have zero normal flux and are excluded, avoiding division by kz=0.
    def propagating_flux(electric, epsilon):
        kx, ky = sim.Kx_norm_dn.real, sim.Ky_norm_dn.real
        squared = epsilon-kx*kx-ky*ky
        mask = squared>1e-12
        q = torch.sqrt(squared[mask])
        ex, ey = electric[:n][mask], electric[n:][mask]
        ez = -(kx[mask]*ex+ky[mask]*ey)/q
        hx = ky[mask]*ez-q*ey
        hy = q*ex-kx[mask]*ez
        return float(torch.real((ex*hy.conj()-ey*hx.conj()).sum()).cpu())
    r = propagating_flux(rx, 1.)
    t = propagating_flux(tx, 2.25)
    return dict(polarization=polarization, R00=r0, T00=t0, total_R=r, total_T=t,
                absorption=1-r-t, passive=bool(r>=-1e-7 and t>=-1e-7 and r+t<=1+1e-7),
                r00_real=float(rx[z].real.cpu()), r00_imag=float(rx[z].imag.cpu()))


@torch.no_grad()
def solve(cls, m, grid, device, *, full=False, wavelength_nm=500.):
    synchronize(device); start=time.perf_counter()
    sim=cls(1000./wavelength_nm, [m,m], [1.,1.], dtype=torch.complex128,
            device=device, use_group_theory=True, group_theory_symmetry='c2v',
            polarization_reduction=None if full else 'x', group_theory_strict=True,
            smatrix_size='half', enable_fields=False, verify_cascade=False)
    sim.add_input_layer(eps=1.)
    sim.add_output_layer(eps=2.25)
    sim.set_incident_angle(0.,0.)
    sim.add_layer_circle_nvm(.05, .25, (1.75+1.5j)**2, 1., centers=((.5,.5),),
                             use_lanczos=False, nx=grid, ny=grid)
    z=m*(2*m+1)+m
    source=torch.zeros(2*sim.order_N, dtype=torch.complex128, device=device)
    source[z]=1.
    if full:
        sim.solve_global_smatrix()
        src_y=torch.zeros_like(source); src_y[sim.order_N+z]=1.
        sources={'x':source, 'y':src_y, 'diagonal':(source+src_y)/math.sqrt(2.)}
        result=[observable(sim, sim.S[0]@s, sim.S[1]@s, label) for label,s in sources.items()]
        dim=2*sim.order_N
    else:
        tx,rx=sim.solve_polarization_source(source)
        result=[observable(sim,tx,rx,'x')]
        dim=sim._polarization_bases[0].shape[1]
    synchronize(device)
    elapsed=time.perf_counter()-start
    del sim
    gc.collect()
    return dict(order=m, grid=grid, wavelength_nm=wavelength_nm, seconds=elapsed,
                eigensolve_dimension=dim, full_two_polarizations=full, results=result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--source-root',type=Path,help='Snapshot directory with ON/rcwa_ext and OFF/rcwa_ext')
    p.add_argument('--output',type=Path,default=Path('results'))
    p.add_argument('--device',default='cpu')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--orders',type=numbers,default=numbers('4:13'))
    p.add_argument('--grids',type=numbers,default=[512,1024])
    p.add_argument('--validate',action='store_true')
    p.add_argument('--paper-equation-check',action='store_true')
    a=p.parse_args()
    torch.set_num_threads(a.threads)
    device=torch.device(a.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable.')
    if a.source_root:
        paths={mode:a.source_root/mode/'rcwa_ext' for mode in ['ON','OFF']}
    else:
        base=a.repo_root/'Optimization/Al'
        paths={'ON':base/'al_motheye_P200_H500_Nz15/rcwa_ext',
               'OFF':base/'al_motheye_P200_H500_Nz15_normal_off_v1/rcwa_ext'}
    parity=check_only_projection_differs(paths['ON'],paths['OFF'])
    sources={mode:{str(f.relative_to(path)):digest(f) for f in sorted(path.rglob('*.py'))}
             for mode,path in paths.items()}
    classes={mode:load_solver('schuster_fig10_'+mode.lower(),path) for mode,path in paths.items()}
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    manifest=dict(paper='Schuster et al., JOSA A 24, 2880-2890 (2007), Fig.10',
        doi='10.1364/JOSAA.24.002880', observable='reflected zeroth-order diffraction efficiency R00',
        parameters=dict(period_x_nm=1000.,period_y_nm=1000.,circle_diameter_nm=500.,
        layer_thickness_nm=50.,wavelength_nm=500.,n_metal=[1.75,1.5],n_input=1.,n_substrate=1.5,
        geometry='air circular hole in lossy metal film on glass',incidence='normal',
        truncation='rectangular -M..M in both directions',lanczos=False),
        torch_version=torch.__version__,device=str(device),cuda_execution_performed=device.type=='cuda',
        threads=a.threads,source_paths={k:str(v) for k,v in paths.items()},sources=sources,algebra_check=parity)
    if (out/'manifest.json').exists():
        previous=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
        for field in ['device','torch_version','threads','parameters','sources']:
            if previous.get(field)!=manifest.get(field):
                raise RuntimeError(f'Output context changed ({field}); choose a new --output directory.')
    write_json(out/'manifest.json',manifest)
    rows=[]
    for m in a.orders:
        for grid in a.grids:
            for mode,cls in classes.items():
                f=out/'points'/f'{mode}_M{m}_Q{grid}.json'
                if f.exists():
                    record=json.loads(f.read_text(encoding='utf-8'))
                    if record.get('sources')!=sources[mode]:
                        raise RuntimeError('Cached point has different source: '+str(f))
                else:
                    record=solve(cls,m,grid,device)
                    record.update(mode=mode,sources=sources[mode])
                    write_json(f,record)
                obs=record['results'][0]
                rows.append(dict(mode=mode,order=m,grid=grid,seconds=record['seconds'],**obs))
                print(f'{mode} M={m} Q={grid}: R00={obs["R00"]:.12f}, Rtotal={obs["total_R"]:.9f}, '
                      f'passive={obs["passive"]}, {record["seconds"]:.3f}s',flush=True)
    fields=list(rows[0])
    with (out/'production_ON_OFF.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
    validation=[]
    def validation_solve(cls,mode,m,grid,*,full=False,wavelength_nm=500.):
        key=f'{mode}_M{m}_Q{grid}_full{int(full)}_w{wavelength_nm:.12f}'
        path=out/'validation_points'/(key+'.json')
        if path.exists():
            cached=json.loads(path.read_text(encoding='utf-8'))
            if cached.get('sources')!=sources[mode]:
                raise RuntimeError('Validation source mismatch: '+key)
            return cached['calculation']
        calculation=solve(cls,m,grid,device,full=full,wavelength_nm=wavelength_nm)
        write_json(path,dict(sources=sources[mode],calculation=calculation))
        return calculation
    if a.validate:
        for mode,cls in classes.items():
            for m in [4,13]:
                if m not in a.orders:continue
                full=validation_solve(cls,mode,m,1024,full=True)
                reference=next(r['R00'] for r in rows if r['mode']==mode and r['order']==m and r['grid']==1024)
                error=max(abs(x['R00']-reference) for x in full['results'])
                # This geometry is exactly at a Rayleigh cutoff (lambda=P/2).
                # Resolve parity well below the ~4e-5 figure-readout accuracy;
                # retain the stricter diagnostic rather than hiding roundoff.
                validation.append(dict(kind='full_x_y_diagonal_vs_x_sector',mode=mode,error=error,
                                       tolerance=1e-7,passed=error<1e-7,
                                       strict_1e9_passed=error<1e-9,**full))
                print(f'VALIDATE {mode} M={m}: polarization/full-sector error={error:.3g}',flush=True)
            for m in [4,8,13]:
                if m not in a.orders:continue
                fine=validation_solve(cls,mode,m,2048)
                reference=next(r['R00'] for r in rows if r['mode']==mode and r['order']==m and r['grid']==1024)
                error=abs(fine['results'][0]['R00']-reference)
                validation.append(dict(kind='Q1024_to_Q2048',mode=mode,error=error,tolerance=1e-5,passed=error<1e-5,**fine))
                print(f'VALIDATE {mode} M={m}: grid error={error:.3g}',flush=True)
            if 13 in a.orders:
                reference=next(r['R00'] for r in rows if r['mode']==mode and r['order']==13 and r['grid']==1024)
                for sign in [-1,1]:
                    near=validation_solve(cls,mode,13,1024,wavelength_nm=500.*(1+sign*1e-8))
                    error=abs(near['results'][0]['R00']-reference)
                    validation.append(dict(kind='Rayleigh_cutoff_nearby_wavelength',mode=mode,error=error,
                                           tolerance=1e-5,passed=error<1e-5,**near))
            # Away from the cutoff, also check full/source-sector parity at
            # a stricter tolerance. This is a diagnostic, not paper data.
            reduced=validation_solve(cls,mode,4,1024,wavelength_nm=501.)
            full=validation_solve(cls,mode,4,1024,full=True,wavelength_nm=501.)
            error=max(abs(v['R00']-reduced['results'][0]['R00']) for v in full['results'])
            validation.append(dict(kind='off_cutoff_full_sector_parity',mode=mode,error=error,
                                   tolerance=1e-11,passed=error<1e-11,**full))
        write_json(out/'validation.json',validation)
    if a.paper_equation_check:
        check=[]
        class PaperEquationOFF(classes['OFF']):
            _symmetric_nvm_tensor=staticmethod(schuster_tensor)
        for m in a.orders:
            r=solve(PaperEquationOFF,m,1024,device)
            check.append(r)
            print(f'PAPER EQ7 OFF M={m}: R00={r["results"][0]["R00"]:.12f}',flush=True)
        write_json(out/'paper_eq7_OFF.json',check)
    for mode,path in paths.items():
        assert sources[mode]=={str(f.relative_to(path)):digest(f) for f in sorted(path.rglob('*.py'))}
    result=dict(status='completed',points=len(rows),all_passive=all(r['passive'] for r in rows),
                production_sources_unchanged=True,
                validation_passed=all(v['passed'] for v in validation) if validation else None,
                main_polarization='x; square symmetry makes R00 identical for normalized [1,1] (tested separately)')
    write_json(out/'run_summary.json',result)
    print(json.dumps(result,indent=2))
    return 0 if result['all_passive'] and result['validation_passed'] is not False else 1


if __name__=='__main__':
    raise SystemExit(main())
