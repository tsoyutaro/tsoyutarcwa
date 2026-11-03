"""Run saved metal Optuna best shapes through convergence and then a spectrum."""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import continue_convergence as cc
from run_metal_off_optuna30 import CASES

ROOT=Path(__file__).resolve().parent


def specification(material,args,override=None):
    relative,default_order=CASES[material]
    order=args.optimization_order or default_order
    case=ROOT/relative
    config=case/f'config_hex_OFF_optuna{args.trials}_M{order}_Q{args.training_grid}.json'
    run=case/'results'/f'{args.run_tag}_M{order}_Q{args.training_grid}'
    if override is not None:
        case,config,run=(Path(override[k]).resolve() for k in ('case_dir','config','run_root'))
    if (run/'controller.lock').exists():
        raise ValueError('The optimization/analysis controller lock exists; wait for that job to finish.')
    if cc.read(run/'workflow.json').get('status')!='completed':
        raise ValueError('Optimization has not completed in '+str(run))
    c=cc.read(config)
    if (c.get('material')!=material or c.get('normal_mode')!='off'
            or c.get('layers')!=15 or 'hexagonal' not in c.get('lattices',[])
            or c.get('optimization_order')!=order or c.get('optimization_grid')!=args.training_grid):
        raise ValueError('Configuration does not match the requested OFF metal experiment.')
    logits=cc.read(run/'hexagonal/report.json').get('best',{}).get('logits')
    if not isinstance(logits,list) or len(logits)!=c['layers']+1:
        raise ValueError('No completed best shape with the expected layer count.')
    if not (run/'manifest.json').is_file():
        raise ValueError('Saved experiment manifest is missing.')
    return dict(material=material,case_dir=str(case),config=str(config),run_root=str(run))


def launch(command,env):
    print('COMMAND: '+subprocess.list2cmdline(command),flush=True)
    process=subprocess.Popen(command,cwd=str(ROOT),env=env)
    try:
        code=process.wait()
    except KeyboardInterrupt:
        process.terminate()
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:process.kill();process.wait()
        raise
    if code<0 or code>=128:raise KeyboardInterrupt()
    return code


def save(output,summary):
    cc.write(output/'summary.json',summary)
    fields=['material','status','convergence_status','spectrum_status','mean_R','mean_R_percent',
            'wall_seconds','convergence_report','spectrum_report','error']
    temporary=output/'summary.csv.tmp'
    with temporary.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for material,record in summary['materials'].items():
            writer.writerow({k:material if k=='material' else record.get(k) for k in fields})
    temporary.replace(output/'summary.csv')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--materials',default='Al,Ti,Cr')
    p.add_argument('--orders',default='4:40:2')
    p.add_argument('--grids',default='512,1024')
    p.add_argument('--convergence-wavelengths',default='400,550,700')
    p.add_argument('--spectrum-order',type=int,default=40)
    p.add_argument('--spectrum-grid',type=int,default=512)
    p.add_argument('--spectrum-wavelengths',default='400:700:3')
    p.add_argument('--training-grid',type=int,default=512)
    p.add_argument('--trials',type=int,default=30,help='Select the saved config; does not run training')
    p.add_argument('--run-tag',default='hex_OFF_optuna30_v1')
    p.add_argument('--optimization-order',type=int,help='Override the saved training order for selected materials')
    p.add_argument('--output-root',type=Path)
    p.add_argument('--cases-file',type=Path,help='Optional explicit saved case/config/run paths, as in continue_convergence.py')
    p.add_argument('--prepare-only',action='store_true',help='Check saved results and write the plan without running RCWA')
    args=p.parse_args(argv)
    def interrupted(signum,frame):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupted)
    materials=args.materials.split(',')
    if not materials or len(set(materials))!=len(materials) or any(m not in CASES for m in materials):
        p.error('Select distinct materials from Al,Ti,Cr,Cu,Au')
    try:
        supplied=cc.read(args.cases_file) if args.cases_file else []
        overrides={s['material']:s for s in supplied}
        if args.cases_file and (len(overrides)!=len(supplied) or any(m not in overrides for m in materials)):
            raise ValueError('cases-file must contain one specification for each requested material.')
        orders=cc.numbers(args.orders,True);grids=cc.numbers(args.grids,True)
        conv_waves=cc.numbers(args.convergence_wavelengths)
        spectrum_waves=cc.numbers(args.spectrum_wavelengths)
        if args.spectrum_order not in orders or args.spectrum_grid not in grids:
            raise ValueError('Include the spectrum order/grid in the convergence conditions.')
        if min(args.training_grid,args.trials,args.optimization_order or 1)<=0:
            raise ValueError('Training grid, trials and optimization order must be positive.')
        if any(q<max(32,4*m+4) for m in orders for q in grids):
            raise ValueError('A requested grid undersamples a retained order.')
        if not args.run_tag or Path(args.run_tag).name!=args.run_tag or any(s in args.run_tag for s in '/\\'):
            raise ValueError('run-tag must be one directory name.')
    except ValueError as error:p.error(str(error))
    output=(args.output_root or ROOT/'postprocess_OFF_optuna30_v1').resolve()
    output.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    if env.get('MOTHEYE_DEPENDENCY_PATH') and not env.get('RCWA_OPT_DEPENDENCY_PATH'):
        env['RCWA_OPT_DEPENDENCY_PATH']=env['MOTHEYE_DEPENDENCY_PATH']
    summary=dict(status='running',materials={},device=args.device,orders=orders,grids=grids,
        convergence_wavelengths_nm=conv_waves,spectrum_order=args.spectrum_order,
        spectrum_grid=args.spectrum_grid,spectrum_wavelengths_nm=spectrum_waves)
    active=None;started=time.perf_counter()
    try:
        for material in materials:
            active=material;begin=time.perf_counter()
            record=summary['materials'][material]=dict(status='running')
            try:
                spec=specification(material,args,overrides.get(material))
                folder=output/material;folder.mkdir(parents=True,exist_ok=True)
                cases=folder/'cases.json';cc.write(cases,[spec]);record['case']=spec
                if args.prepare_only:
                    record['status']='prepared'
                    continue
                record['stage']='convergence';save(output,summary)
                print(f'\nCONVERGENCE: {material}',flush=True)
                code=launch([sys.executable,str(ROOT/'continue_convergence.py'),'--root',str(folder),
                    '--cases-file',str(cases),'--device',args.device,'--materials',material,
                    '--lattice','hexagonal','--orders',args.orders,'--grids',args.grids,
                    '--wavelengths',args.convergence_wavelengths,'--oom-policy','stop-material'],env)
                result=cc.read(folder/'convergence_continue_summary.json')['materials'][material]
                record['convergence_status']=result['status']
                if code or not result.get('report'):
                    raise RuntimeError('Convergence setup failed: '+str(result))
                record['convergence_report']=result['report']
                converged=cc.read(result['report']);records=converged['records']
                target=next((x for x in records if x['order']==args.spectrum_order and x['grid']==args.spectrum_grid),None)
                if any(x['status'] in cc.OOM for x in records):
                    record.update(status='completed_with_skips',spectrum_status='skipped_after_oom')
                    print(f'{material}: OOM during convergence; skipping the M{args.spectrum_order} spectrum and moving to the next material.',flush=True)
                    continue
                if not target or target['status']!='completed':
                    record.update(status='completed_with_skips',spectrum_status='skipped_target_convergence_failed')
                    print(f'{material}: spectrum order/grid did not complete the three-wavelength check; moving to the next material.',flush=True)
                    continue
                record['stage']='spectrum';save(output,summary)
                print(f'\nSPECTRUM: {material}, M={args.spectrum_order}, Q={args.spectrum_grid}, {len(spectrum_waves)} wavelengths',flush=True)
                code=launch([sys.executable,str(Path(spec['case_dir'])/'run.py'),'--config',spec['config'],
                    '--device',args.device,'--mode','spectrum','--lattices','hexagonal',
                    '--order',str(args.spectrum_order),'--grid',str(args.spectrum_grid),
                    '--wavelengths',args.spectrum_wavelengths,'--timeout-hours','0',
                    '--output-root',spec['run_root']],env)
                if code:raise RuntimeError(f'Spectrum controller stopped with exit code {code}; inspect the saved worker logs.')
                pointer=cc.read(Path(spec['run_root'])/'hexagonal/latest_spectrum.json')
                spectrum=cc.read(pointer['report']);outcome=spectrum['outcomes'][0]
                if (spectrum['status']!='completed' or spectrum['wavelengths_nm']!=spectrum_waves
                        or outcome['order']!=args.spectrum_order or outcome['grid']!=args.spectrum_grid
                        or len(outcome['rows'])!=len(spectrum_waves)):
                    raise RuntimeError('Saved spectrum does not match the requested conditions.')
                record.update(status='completed',spectrum_status='completed',mean_R=outcome['objective'],
                    mean_R_percent=100*outcome['objective'],spectrum_report=pointer['report'],
                    spectrum_csv=spectrum['spectrum_csv'])
                print(f"{material}: mean R = {record['mean_R']:.12g} ({record['mean_R_percent']:.8g}%)",flush=True)
            except Exception as error:
                record.update(status='error',error=str(error))
                print(f'{material}: {error}; continuing to the next material.',flush=True)
            finally:
                record['wall_seconds']=time.perf_counter()-begin;save(output,summary)
        desired='prepared' if args.prepare_only else 'completed'
        summary['status']=desired if all(x['status']==desired for x in summary['materials'].values()) else 'completed_with_skips'
    except KeyboardInterrupt:
        summary['status']='interrupted'
        if active:summary['materials'][active]['status']='interrupted'
    finally:
        summary['wall_seconds']=time.perf_counter()-started;save(output,summary)
        print('Saved '+str(output/'summary.json'),flush=True)
    if summary['status']=='interrupted':return 130
    return 0 if summary['status'] in ('completed','prepared') and all(
        x['status'] in ('completed','prepared') for x in summary['materials'].values()) else 1


if __name__=='__main__':
    raise SystemExit(main())
