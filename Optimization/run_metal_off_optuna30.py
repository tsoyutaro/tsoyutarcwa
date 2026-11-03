"""Prepare/resume separate OFF Optuna studies through the existing run.py files.

Optuna selects lr, init_top and delta_ratio. Each trial starts a new monotone
linear radius profile, then Adam optimizes all layer radii. The first trial
is the existing queued baseline. Verification remains a separate command.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import uuid

from off_order_controller import POLICY,RETRY_EXIT_CODE

ROOT=Path(__file__).resolve().parent
CASES={
    'Al': ('Al/al_motheye_P200_H500_Nz15',8),
    'Ti': ('Ti/ti_motheye_P200_H500_Nz15',8),
    'Cr': ('cr_motheye',8),
    'Cu': ('Cu/cu_motheye_P200_H500_Nz15',12),
    'Au': ('Au/au_motheye_P200_H500_Nz15',14),
}

def prepare(material, *, grid=512, trials=30, adam_steps=500, run_tag='hex_OFF_optuna30_v1', order=None,
            period_nm=200., height_nm=500.):
    relative,default_order=CASES[material]
    order=default_order if order is None else order
    if order<=0 or grid<max(32,4*order+4):
        raise ValueError('A positive order and a sufficient grid are required.')
    if not all(math.isfinite(v) and v>0 for v in (period_nm,height_nm)):
        raise ValueError('Period and height must be finite and positive.')
    case=ROOT/relative
    c=json.loads((case/'config.json').read_text(encoding='utf-8-sig'))
    c.update(normal_mode='off',layers=15,lattices=['hexagonal'],
             period_nm=period_nm,height_nm=height_nm,optimization_order=order,
             optimization_grid=grid,trials=trials,adam_steps=adam_steps,
             optimization_wavelengths_nm='400:700:10',objective_weighting='arithmetic')
    geometry_changed=(period_nm!=200. or height_nm!=500.)
    condition=case
    if geometry_changed:
        # Keep each geometry's configs, studies and locks separate. The original
        # case still supplies the solver and tabulated dispersion to its workers.
        label=f'P{period_nm:.15g}_H{height_nm:.15g}_Nz15'
        condition=case/'geometry_cases'/label
        c.update(radius_min_nm=.001,radius_max_nm=period_nm/2-.001,
                 baseline_tip_nm=.025*period_nm,baseline_base_nm=.475*period_nm,
                 init_top_min_nm=1.,init_top_max_nm=min(99.,period_nm/2-.011))
        if not (0<c['radius_min_nm']<c['baseline_tip_nm']<c['baseline_base_nm']
                <c['radius_max_nm']<period_nm/2
                and c['radius_min_nm']<c['init_top_min_nm']<c['init_top_max_nm']<c['radius_max_nm']):
            raise ValueError('Period is too small for the radius and initial-profile limits.')
        if 'case_id' in c:
            c['case_id']=f'{material.lower()}_motheye_{label}'
        if material=='Cr':
            # Cr resolves relative CSV paths from the config directory; the
            # independent metal cases resolve them from their original case root.
            material_path=Path(c['material_csv'])
            if not material_path.is_absolute():
                c['material_csv']=(Path('../..')/material_path).as_posix()
    if c.get('verification_orders') and order>max(c['verification_orders']):
        c['verification_orders']=sorted({*c['verification_orders'],order})
        c['verification_grid']=max(c['verification_grid'],32,4*order+4)
    # Initial profiles come from each Optuna trial, never a previous optimum.
    c.pop('initial_logits',None)
    c.pop('initial_profile_provenance',None)
    cfg=condition/f'config_hex_OFF_optuna{trials}_M{order}_Q{grid}.json'
    if cfg.exists():
        if json.loads(cfg.read_text(encoding='utf-8-sig'))!=c:
            raise ValueError(f'Existing config differs; preserve it and choose a separate config: {cfg}')
    else:
        condition.mkdir(parents=True,exist_ok=True)
        cfg.write_text(json.dumps(c,indent=2)+'\n',encoding='utf-8')
    output=condition/'results'/f'{run_tag}_M{order}_Q{grid}'
    return case,cfg,output

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--materials',default='Al,Ti,Cr,Cu,Au')
    p.add_argument('--grid',type=int,default=512)
    p.add_argument('--period-nm',type=float,default=200.,help='One fixed nearest-neighbor period in nm (default: 200)')
    p.add_argument('--height-nm',type=float,default=500.,help='One fixed total moth-eye height in nm (default: 500)')
    p.add_argument('--trials',type=int,default=30)
    p.add_argument('--adam-steps',type=int,default=500)
    p.add_argument('--run-tag',help='Default: hex_OFF_optuna30_auto_v1 with --auto-order, otherwise hex_OFF_optuna30_v1')
    p.add_argument('--auto-order',action='store_true',help='Restart a new study at M+2 after confirmed nonphysical training powers')
    p.add_argument('--start-order',type=int,help='Override the initial order for the selected materials')
    p.add_argument('--max-order',type=int,default=40,help='Inclusive automatic retry ceiling (default: 40)')
    p.add_argument('--prepare-only',action='store_true',help='Write configs and print commands without running optimization')
    p.add_argument('--recover-lock',action='store_true',help='Only after confirming that the previous job has stopped')
    a=p.parse_args(argv)
    if not all(math.isfinite(v) and v>0 for v in (a.period_nm,a.height_nm)):
        p.error('Period and height must be finite and positive')
    if a.period_nm<=2.022:
        p.error('Period is too small for the initial-radius search interval')
    materials=a.materials.split(',')
    if not materials or len(set(materials))!=len(materials) or any(m not in CASES for m in materials):
        p.error('Select distinct materials from Al,Ti,Cr,Cu,Au')
    initial_orders=[a.start_order if a.start_order is not None else CASES[m][1] for m in materials]
    if min(a.grid,a.trials,a.adam_steps,*initial_orders)<=0 or a.grid<max(32,4*max(initial_orders)+4):
        p.error('Positive steps/trials and a sufficient grid are required')
    if a.auto_order and a.max_order<max(initial_orders):
        p.error('max-order must be at least the initial order')
    a.run_tag=a.run_tag or ('hex_OFF_optuna30_auto_v1' if a.auto_order else 'hex_OFF_optuna30_v1')
    if not a.run_tag or Path(a.run_tag).name!=a.run_tag or any(x in a.run_tag for x in '/\\'):
        p.error('run-tag must be one directory name')
    # Children use their original independent packages and source manifests.
    # Translate the optional workspace dependency path for the Cr entry point.
    env=os.environ.copy()
    if env.get('MOTHEYE_DEPENDENCY_PATH') and not env.get('RCWA_OPT_DEPENDENCY_PATH'):
        env['RCWA_OPT_DEPENDENCY_PATH']=env['MOTHEYE_DEPENDENCY_PATH']
    failures=[]
    try:
        for m in materials:
            order=a.start_order if a.start_order is not None else CASES[m][1]
            while True:
                if a.grid<max(32,4*order+4):
                    failures.append({'material':m,'order':order,'reason':'insufficient_grid'})
                    print(f'{m}: grid {a.grid} is insufficient for M={order}; stopping this material.',flush=True)
                    break
                case,cfg,output=prepare(m,grid=a.grid,trials=a.trials,adam_steps=a.adam_steps,run_tag=a.run_tag,order=order,
                                        period_nm=a.period_nm,height_nm=a.height_nm)
                arguments=['--config',str(cfg),'--device',a.device,'--mode','optimize',
                           '--lattices','hexagonal','--timeout-hours','0','--output-root',str(output)]
                if a.recover_lock:arguments.append('--recover-lock')
                attempt=uuid.uuid4().hex
                if a.auto_order:
                    command=[sys.executable,str(ROOT/'off_order_controller.py'),'--case-dir',str(case),
                             '--attempt-id',attempt,'--',*arguments]
                else:command=[sys.executable,str(case/'run.py'),*arguments]
                print(f'\n{m}: OFF P={a.period_nm:g} H={a.height_nm:g} nm, M={order} Q={a.grid}, {a.trials} trials',flush=True)
                print(shlex.join(command),flush=True)
                if a.prepare_only:break
                completed=subprocess.run(command,cwd=str(case),env=env)
                code=completed.returncode
                if code<0 or code>=128:return code if code>=128 else 128-code
                marker_path=output/'order_retry.json'
                marker={}
                if a.auto_order and code==RETRY_EXIT_CODE and marker_path.exists():
                    try:marker=json.loads(marker_path.read_text(encoding='utf-8-sig'))
                    except (OSError,ValueError):pass  # An unconfirmed result never raises M.
                retry=(a.auto_order and code==RETRY_EXIT_CODE and marker.get('policy')==POLICY
                       and marker.get('status')=='retry_higher_order' and marker.get('attempt_id')==attempt
                       and marker.get('order')==order and marker.get('grid')==a.grid
                       and marker.get('normal_mode')=='off')
                if retry and order+2<=a.max_order:
                    print(f'{m}: confirmed nonphysical power at M={order}; restarting at M={order+2} from trial 0 in a separate study.',flush=True)
                    order+=2
                    continue
                if code:
                    reason='maximum_order_reached' if retry else 'controller_stopped'
                    failures.append({'material':m,'order':order,'exit_code':code,'reason':reason})
                    print(f'{m}: stopped at M={order} ({reason}, exit {code}); proceeding to the next material.',flush=True)
                break
    except KeyboardInterrupt:
        return 130
    if failures:print(json.dumps({'status':'some_materials_stopped','materials':failures},indent=2),flush=True)
    return 1 if failures else 0

if __name__=='__main__':
    raise SystemExit(main())
