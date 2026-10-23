"""Order, slice, normal-grid and DM-step convergence for full-vector gold FFF."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(name,'1')
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

import numpy as np
import torch
from studies.shared.gold_dispersion import DEFAULT_GOLD_CSV, load_gold_csv
from studies.gold_motheye_fff2d.model import Geometry, harmonics
from studies.gold_motheye_fff2d.solver import METHODS, solve
from studies.gold_motheye_fff2d.integration import DM_INTEGRATORS

HERE = Path(__file__).resolve().parent
METRICS = ('R','P_sub','A_moth')
FLUX_CHECKS = ('R','P_sub','A_moth','A_total')


def values(text, integer=False):
    result = []
    for part in text.split(','):
        bits = part.strip().split(':')
        if len(bits) == 1:
            result.append(float(bits[0]))
        elif len(bits) == 3:
            start,stop,step = map(float,bits)
            if step <= 0 or stop < start:
                raise argparse.ArgumentTypeError('Use increasing start:stop:positive_step')
            count = int(math.floor((stop-start)/step+1e-9))+1
            if count > 10000:
                raise argparse.ArgumentTypeError('Too many requested points')
            result.extend(start+i*step for i in range(count))
        else:
            raise argparse.ArgumentTypeError('Use comma-separated values or start:stop:step')
    if not result or any(not math.isfinite(x) or x <= 0 for x in result):
        raise argparse.ArgumentTypeError('Values must be finite and positive')
    if integer and any(x != int(x) for x in result):
        raise argparse.ArgumentTypeError('Expected integers')
    return sorted(set(int(x) if integer else float(x) for x in result))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()


def write_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def physical(row,tolerance):
    bounds = {key:bool(math.isfinite(row[key]) and -tolerance <= row[key] <= 1+tolerance)
              for key in FLUX_CHECKS}
    if row.get('absorption_error') is not None:
        bounds['independent_volume_absorption'] = bool(
            row['absorption_error'] <= tolerance and row['absorption_volume'] >= -tolerance)
    if row.get('constitutive_loss_min_eigenvalue') is not None:
        material_scale = max(1.,abs(complex(row['epsilon_real'],row['epsilon_imag'])))
        bounds['constitutive_passivity_at_sampled_depths'] = bool(
            row['constitutive_loss_min_eigenvalue'] >= -tolerance*material_scale)
    return dict(passed=all(bounds.values()),checks=bounds,
                flux_bounds_passed=all(bounds[key] for key in FLUX_CHECKS),
                failed_checks=[key for key,value in bounds.items() if not value],
                scope='Passive flux bounds, optional volume loss, optional constitutive passivity at sampled depths. A_total=1-R-T_far is a definition, not an energy test.')


def flux_bounds_passed(row):
    """Read actual flux checks, including from checkpoints made before this fix."""
    checks = row.get('physical',{}).get('checks',{})
    return row.get('status')!='FAILED' and all(checks.get(key,False) for key in FLUX_CHECKS)


def make_report(rows,args,requested,physics):
    current = [x for x in rows if x['method'] in args.methods and x['polarization'] in args.polarizations
               and x['wavelength_nm'] in args.wavelengths and x['axis_value'] in requested]
    groups = []
    for method in args.methods:
        for wavelength in args.wavelengths:
            for pol in args.polarizations:
                curve = [x for x in current if x['method']==method and x['wavelength_nm']==wavelength and x['polarization']==pol]
                curve.sort(key=lambda x:requested.index(x['axis_value']))
                complete = len(curve)==len(requested) and all(x['status']!='FAILED' for x in curve)
                differences = []
                for a,b in zip(curve[:-1],curve[1:]):
                    if a['status']=='FAILED' or b['status']=='FAILED':
                        continue
                    errors = {key:abs(b[key]-a[key]) for key in METRICS}
                    differences.append(dict(from_value=a['axis_value'],to_value=b['axis_value'],
                                            errors=errors,max_change=max(errors.values())))
                enough = len(differences)>=2
                passive = complete and all(flux_bounds_passed(x) for x in curve)
                all_checks = complete and all(x['physical']['passed'] for x in curve)
                numerical = bool(complete and enough and all(x['max_change']<=args.tolerance for x in differences[-2:]))
                converged = bool(numerical and all_checks)
                groups.append(dict(method=method,wavelength_nm=wavelength,polarization=pol,
                                   complete=complete,passive_flux_bounds=passive,
                                   physical_checks_passed=all_checks,
                                   numerically_converged_at_fixed_other_parameters=numerical,
                                   converged_at_fixed_other_parameters=converged,differences=differences))
    physical_passed = bool(current and all(x['status']!='FAILED' and x['physical']['passed'] for x in current))
    suite_passed = physics.get('passed') if physics is not None else None
    convergence = bool(groups and all(x['converged_at_fixed_other_parameters'] for x in groups))
    flux_passed = bool(current and all(flux_bounds_passed(x) for x in current))
    loss_flagged = sum(x.get('physical',{}).get('checks',{}).get('constitutive_passivity_at_sampled_depths') is False for x in current)
    comparisons = []
    for value in requested:
        for wavelength in args.wavelengths:
            for pol in args.polarizations:
                matched = {x['method']:x for x in current if x['axis_value']==value and x['wavelength_nm']==wavelength
                           and x['polarization']==pol and x['status']!='FAILED'}
                baseline = matched.get('slice-nvm')
                if baseline is None:
                    continue
                for method,row in matched.items():
                    if method == 'slice-nvm':
                        continue
                    comparisons.append(dict(axis_value=value,wavelength_nm=wavelength,polarization=pol,
                        baseline='slice-nvm',method=method,
                        differences={key:row[key]-baseline[key] for key in METRICS},
                        max_order_power_difference=max(float(np.max(abs(np.array(row[key])-np.array(baseline[key]))))
                                                       for key in ('R_orders','P_sub_orders')),
                        both_physical=row['physical']['passed'] and baseline['physical']['passed']))
    return dict(axis=args.axis,values=requested,case_count=len(current),method_comparisons=comparisons,
                failed_cases=sum(x['status']=='FAILED' for x in current),
                unphysical_cases=sum(x['status']=='UNPHYSICAL' for x in current),
                all_requested_cases_complete=all(x['complete'] for x in groups),
                passive_flux_bounds_passed=flux_passed,physical_checks_passed=physical_passed,
                constitutive_loss_flagged_cases=loss_flagged,independent_physics_suite_passed=suite_passed,
                axis_numerically_converged=bool(groups and all(x['numerically_converged_at_fixed_other_parameters'] for x in groups)),
                axis_converged=convergence,
                passed=bool(convergence and physical_passed and suite_passed is True),
                tolerance_absolute=args.tolerance,physical_tolerance=args.physics_tolerance,curves=groups,
                limitations=[
                    'Order convergence at fixed slices/grid does not establish slice/grid/DM-step convergence.',
                    'No monotonic convergence is assumed; only the final two complete refinement differences are tested.',
                    'Passive R/A bounds alone do not validate a solution. The independent suite is low-order.',
                    'An indefinite loss operator flags potential gain directions; it does not by itself show gain for the chosen incident wave.',
                    'GIL changes the modeled structure; compare separate runs rather than treating it as a neutral numerical filter.',
                    'GPU path is implemented but must be validated on the requested GPU before trusting high-order results.',
                ])


def tables_and_plots(output,rows,report,args):
    columns = ['axis_value','method','wavelength_nm','polarization','order','scalar_harmonics','field_dimension',
               'slices','grid','dm_step_nm','R','P_sub','A_moth','A_total','T_far',
               'absorption_volume','absorption_error','elapsed_s','peak_cuda_bytes','status','error']
    columns.insert(-4,'constitutive_loss_min_eigenvalue')
    columns.extend(['flux_bounds_passed','physical_failed_checks'])
    with (output/'cases.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=columns,extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            checks = row.get('physical',{}).get('checks',{})
            writer.writerow(dict(row,flux_bounds_passed=flux_bounds_passed(row),
                                 physical_failed_checks=';'.join(key for key,value in checks.items() if not value)))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,3,figsize=(15,4.5))
    for curve in report['curves']:
        selected = sorted((x for x in rows if x['method']==curve['method'] and x['wavelength_nm']==curve['wavelength_nm']
                           and x['polarization']==curve['polarization'] and x['axis_value'] in report['values']),
                          key=lambda x:report['values'].index(x['axis_value']))
        for ax,key in zip(axes,METRICS):
            ax.plot([x['axis_value'] for x in selected],[x.get(key,np.nan) for x in selected],
                    'o-',ms=3,label=f"{curve['method']} {curve['wavelength_nm']:g}nm {curve['polarization']}")
            bad = [x for x in selected if x['status']=='UNPHYSICAL']
            if bad:
                ax.scatter([x['axis_value'] for x in bad],[x[key] for x in bad],marker='x',c='red',s=55,zorder=5)
    for ax,title in zip(axes,('Reflectance R','Power entering Au substrate','Absorption in moth-eye region')):
        ax.set(xlabel=args.axis,ylabel='Fraction of incident power',title=title)
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(output/'convergence.png',dpi=170); plt.close(fig)
    fig,ax = plt.subplots(figsize=(8,4.5))
    for curve in report['curves']:
        differences = curve['differences']
        ax.plot([x['to_value'] for x in differences],
                [max(x['max_change'],1e-16) for x in differences],'o-',ms=3,
                label=f"{curve['method']} {curve['wavelength_nm']:g}nm {curve['polarization']}")
    ax.axhline(args.tolerance,color='black',ls='--',label='Absolute tolerance')
    ax.set(xlabel=args.axis,ylabel='Maximum adjacent change of R, P_sub, A_moth',yscale='log')
    ax.grid(alpha=.25); ax.legend(fontsize=6); fig.tight_layout()
    fig.savefig(output/'changes.png',dpi=170); plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--axis',choices=('order','slices','grid','dm-step'),default='order')
    parser.add_argument('--values',help='Axis values; comma-separated or start:stop:step')
    parser.add_argument('--orders',default='1:5:1',help='Order-axis values (D6 hexagonal shell index for triangular lattice)')
    parser.add_argument('--order',type=int,default=3,help='Fixed Fourier order for non-order axes')
    parser.add_argument('--slices',type=int,default=60)
    parser.add_argument('--grid',type=int,default=128)
    parser.add_argument('--dm-step-nm',type=float,default=.5)
    parser.add_argument('--dm-integrator',choices=DM_INTEGRATORS,default='cf4-adaptive')
    parser.add_argument('--dm-rtol',type=float,default=1e-6)
    parser.add_argument('--dm-atol',type=float,default=1e-9)
    parser.add_argument('--dm-min-step-nm',type=float,default=1e-5)
    parser.add_argument('--methods',default='rcwa-fff,slice-nvm',help='Comma-separated: '+','.join(METHODS))
    parser.add_argument('--polarizations',default='x')
    parser.add_argument('--wavelengths',default='400,550,700')
    parser.add_argument('--device',default='auto',help='auto, cpu, cuda, or cuda:0')
    parser.add_argument('--gold-csv',type=Path,default=DEFAULT_GOLD_CSV)
    parser.add_argument('--period-nm',type=float,default=200.)
    parser.add_argument('--height-nm',type=float,default=500.)
    parser.add_argument('--tip-radius-nm',type=float,default=5.)
    parser.add_argument('--base-radius-nm',type=float,default=95.)
    parser.add_argument('--profile',choices=('cone','cylinder'),default='cone')
    parser.add_argument('--cylinder-radius-nm',type=float,default=60.)
    parser.add_argument('--lattice',choices=('triangular','square'),default='triangular')
    parser.add_argument('--harmonic-shape',choices=('auto','rectangular'),default='auto')
    parser.add_argument('--gil-nm',type=float,default=0.)
    parser.add_argument('--gil-rings',type=int,default=12)
    parser.add_argument('--gil-side',choices=('centered','dielectric','metal'),default='centered')
    parser.add_argument('--ordering',choices=('symmetric','paper-left'),default='symmetric')
    parser.add_argument('--tolerance',type=float,default=.0005)
    parser.add_argument('--physics-tolerance',type=float,default=1e-6)
    parser.add_argument('--volume-absorption',action='store_true',help='Independent volume-loss integration at every case; uses more memory')
    parser.add_argument('--absorption-quadrature',type=int,default=7,help='Gauss-Legendre points per section for volume absorption')
    parser.add_argument('--loss-diagnostics',action='store_true',help='Test constitutive passivity at 10%, 50%, 90% depth; negative loss eigenvalues flag UNPHYSICAL')
    parser.add_argument('--skip-physics',action='store_true',help='Skip the independent low-order suite; report.passed will not be true')
    parser.add_argument('--require-convergence',action='store_true',help='Return exit code 2 if the requested axis has not converged')
    parser.add_argument('--stop-on-failure',action='store_true')
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--report-only',action='store_true')
    parser.add_argument('--output-dir',type=Path)
    args = parser.parse_args(argv)
    try:
        args.methods = list(dict.fromkeys(x.strip() for x in args.methods.split(',')))
        args.polarizations = list(dict.fromkeys(x.strip() for x in args.polarizations.split(',')))
        if not set(args.methods)<=set(METHODS) or not set(args.polarizations)<={'x','y'}:
            raise ValueError('Unknown method or polarization')
        requested = values(args.values or (args.orders if args.axis=='order' else
                           {'slices':'30,60,90,120','grid':'64,128,256','dm-step':'.25,.5,1'}[args.axis]),args.axis!='dm-step')
        if args.axis=='dm-step':
            requested.sort(reverse=True)
            if args.methods!=['dm-fff']:
                raise ValueError('--axis dm-step requires --methods dm-fff')
        args.wavelengths = values(args.wavelengths)
        if args.order<0 or args.slices<1 or args.grid<16 or args.threads<1 or args.dm_step_nm<=0 or args.absorption_quadrature<2:
            raise ValueError('Invalid order, slices, grid, threads or DM step')
        if not all(math.isfinite(x) and x>0 for x in (args.tolerance,args.physics_tolerance,args.dm_step_nm,
                                                     args.dm_rtol,args.dm_atol,args.dm_min_step_nm)):
            raise ValueError('Tolerances and DM step must be finite and positive')
        tip,base = ((args.cylinder_radius_nm,)*2 if args.profile=='cylinder' else
                    (args.tip_radius_nm,args.base_radius_nm))
        geometry = Geometry(args.period_nm,args.height_nm,tip,base,args.lattice,args.gil_nm,args.gil_rings,
                            args.gil_side,args.harmonic_shape)
        model = load_gold_csv(args.gold_csv)
        for wavelength in args.wavelengths:
            model(wavelength)
        max_order = max(requested) if args.axis=='order' else args.order
        min_grid = min(requested) if args.axis=='grid' else args.grid
        if min_grid < max(16,4*max_order+2):
            raise ValueError(f'Normal grid must be >= {max(16,4*max_order+2)} for these orders')
        device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
        if torch.device(device).type=='cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA requested but unavailable')
    except (ValueError, OSError, argparse.ArgumentTypeError) as error:
        parser.error(str(error))
    torch.set_num_threads(args.threads)
    output = (args.output_dir or HERE/'results'/args.axis).resolve()
    source_paths = [HERE/name for name in ('model.py','solver.py','integration.py','native_reference.py','run.py','validate.py')]
    source_paths.append(ROOT/'studies/shared/gold_dispersion.py')
    identity = dict(axis=args.axis,geometry=geometry.metadata(),ordering=args.ordering,
                    fixed_order=None if args.axis=='order' else args.order,
                    fixed_slices=None if args.axis=='slices' else args.slices,
                    fixed_grid=None if args.axis=='grid' else args.grid,
                    fixed_dm_step=None if args.axis=='dm-step' else args.dm_step_nm,
                    dm_integrator=args.dm_integrator,dm_rtol=args.dm_rtol,dm_atol=args.dm_atol,
                    dm_min_step_nm=args.dm_min_step_nm,
                    volume_absorption=args.volume_absorption,
                    absorption_quadrature=args.absorption_quadrature,
                    loss_diagnostics=args.loss_diagnostics,
                    material_sha256=hashlib.sha256(args.gold_csv.read_bytes()).hexdigest(),material_interpolation='shared linear epsilon',
                    sources={str(path.relative_to(ROOT)).replace('\\','/'):digest(path) for path in source_paths})
    signature = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    count = len(harmonics(max_order,args.lattice,args.harmonic_shape)); dimension = 4*count
    plan = dict(identity=identity,signature=signature,requested_values=requested,wavelengths_nm=args.wavelengths,
                methods=args.methods,polarizations=args.polarizations,device=device,
                max_scalar_harmonics=count,max_field_dimension=dimension,
                one_dense_complex_matrix_gib=dimension**2*16/2**30,
                indicative_twenty_matrix_gib=20*dimension**2*16/2**30,
                memory_note='Indicative dense-matrix footprint, not a guaranteed peak or allocation promise; volume fields store layer S matrices.',
                gold_csv=str(args.gold_csv.resolve()),python=sys.version,torch=torch.__version__,platform=platform.platform())
    print(json.dumps(plan,indent=2),flush=True)
    if args.dry_run:
        return 0
    output.mkdir(parents=True,exist_ok=True)
    checkpoint = output/'checkpoint.json'
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text(encoding='utf-8'))
        if state['signature']!=signature:
            parser.error('Checkpoint geometry/material/code/numerical settings differ; use a new output-dir')
    else:
        state = dict(signature=signature,identity=identity,cases=[])
    write_json(output/'plan.json',plan)
    physics = None
    if not args.skip_physics:
        if args.report_only:
            if (output/'physics_validation.json').exists():
                physics = json.loads((output/'physics_validation.json').read_text(encoding='utf-8'))
        elif (output/'physics_validation.json').exists():
            physics = json.loads((output/'physics_validation.json').read_text(encoding='utf-8'))
            if (physics.get('device')!=str(device) or physics.get('ordering')!=args.ordering
                    or physics.get('sources')!=identity['sources'] or not physics.get('passed')):
                physics = None
        if physics is None and not args.report_only:
            from studies.gold_motheye_fff2d.validate import checks
            physics = checks(device,args.ordering)
            physics['sources'] = identity['sources']
            write_json(output/'physics_validation.json',physics)
    if physics is not None and not physics['passed'] and not args.report_only:
        print('Independent validation failed; see physics_validation.json. Sweep was not started.',flush=True)
        return 1
    keys = {(x['method'],x['wavelength_nm'],x['polarization'],x['axis_value']) for x in state['cases'] if x['status']!='FAILED'}
    stop = False
    if not args.report_only:
        for value in requested:
            for method in args.methods:
                for wavelength in args.wavelengths:
                    for polarization in args.polarizations:
                        key = (method,wavelength,polarization,value)
                        if key in keys:
                            continue
                        if stop:
                            break
                        order = int(value) if args.axis=='order' else args.order
                        slices = int(value) if args.axis=='slices' else args.slices
                        grid = int(value) if args.axis=='grid' else args.grid
                        dm_step = float(value) if args.axis=='dm-step' else args.dm_step_nm
                        row = dict(axis_value=value,method=method,wavelength_nm=wavelength,polarization=polarization,
                                   order=order,slices=slices,grid=grid,dm_step_nm=dm_step,status='OK',error='')
                        started = time.perf_counter()
                        try:
                            result = solve(geometry,wavelength,model(wavelength),order=order,slices=slices,grid=grid,
                                           method=method,device=device,polarization=polarization,ordering=args.ordering,
                                           dm_step_nm=dm_step,volume_absorption=args.volume_absorption,loss_diagnostics=args.loss_diagnostics,
                                           dm_integrator=args.dm_integrator,dm_rtol=args.dm_rtol,dm_atol=args.dm_atol,
                                           dm_min_step_nm=args.dm_min_step_nm,
                                           quadrature=args.absorption_quadrature)
                            row.update(result.metadata)
                            row.update(R=result.R,P_sub=result.P_sub,A_moth=result.A_moth,A_total=result.A_total,T_far=result.T_far,
                                       absorption_volume=result.absorption_volume,absorption_error=result.absorption_error,
                                       R_orders=result.R_orders.tolist(),P_sub_orders=result.T_orders.tolist(),
                                       harmonic_indices=harmonics(order,args.lattice,args.harmonic_shape).tolist())
                            row['physical'] = physical(row,args.physics_tolerance)
                            if not row['physical']['passed']:
                                row['status']='UNPHYSICAL'
                            del result
                        except Exception as error:
                            row.update(status='FAILED',error=f'{type(error).__name__}: {error}',physical=dict(passed=False))
                            if isinstance(error,torch.cuda.OutOfMemoryError):
                                stop = True
                        row['elapsed_s'] = time.perf_counter()-started
                        row['created_utc'] = datetime.now(timezone.utc).isoformat()
                        state['cases'] = [x for x in state['cases'] if (x['method'],x['wavelength_nm'],x['polarization'],x['axis_value'])!=key]
                        state['cases'].append(row)
                        write_json(checkpoint,state)
                        report = make_report(state['cases'],args,requested,physics)
                        write_json(output/'report.json',report)
                        print(f"{method} {args.axis}={value:g}, {wavelength:g}nm {polarization}: {row['status']} "
                              f"R={row.get('R')} P_sub={row.get('P_sub')} A_moth={row.get('A_moth')} {row['elapsed_s']:.2f}s",flush=True)
                        if row['status']=='UNPHYSICAL':
                            failed = [key for key,value in row['physical']['checks'].items() if not value]
                            print(f"  failed_checks={','.join(failed)}; flux_bounds={flux_bounds_passed(row)}; "
                                  f"loss_min_eigenvalue={row.get('constitutive_loss_min_eigenvalue')}",flush=True)
                        if device.startswith('cuda'):
                            torch.cuda.empty_cache()
                        if args.stop_on_failure and row['status']!='OK':
                            stop=True
    report = make_report(state['cases'],args,requested,physics)
    write_json(output/'report.json',report)
    tables_and_plots(output,state['cases'],report,args)
    print(f"Saved {output}; axis_converged={report['axis_converged']}; "
          f"passive_flux_bounds={report['passive_flux_bounds_passed']}; physical_checks={report['physical_checks_passed']}; "
          f"constitutive_loss_flagged_cases={report['constitutive_loss_flagged_cases']}; "
          f"axis_numerically_converged={report['axis_numerically_converged']}; "
          f"independent_physics={report['independent_physics_suite_passed']}")
    if not report['physical_checks_passed'] or report['independent_physics_suite_passed'] is False:
        return 1
    return 2 if args.require_convergence and not report['passed'] else 0


if __name__=='__main__':
    raise SystemExit(main())
