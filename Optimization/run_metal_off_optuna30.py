"""Prepare/resume separate OFF Optuna studies through the existing run.py files.

Optuna selects lr, init_top and delta_ratio. Each trial starts a new monotone
linear radius profile, then Adam optimizes all layer radii. The first trial
is the existing queued baseline. Verification remains a separate command.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
CASES={
    'Al': ('Al/al_motheye_P200_H500_Nz15',8),
    'Ti': ('Ti/ti_motheye_P200_H500_Nz15',8),
    'Cr': ('cr_motheye',8),
    'Cu': ('Cu/cu_motheye_P200_H500_Nz15',12),
    'Au': ('Au/au_motheye_P200_H500_Nz15',14),
}

def prepare(material, *, grid=512, trials=30, adam_steps=500, run_tag='hex_OFF_optuna30_v1'):
    relative,order=CASES[material]
    case=ROOT/relative
    c=json.loads((case/'config.json').read_text(encoding='utf-8-sig'))
    c.update(normal_mode='off',layers=15,lattices=['hexagonal'],
             period_nm=200.,height_nm=500.,optimization_order=order,
             optimization_grid=grid,trials=trials,adam_steps=adam_steps,
             optimization_wavelengths_nm='400:700:10',objective_weighting='arithmetic')
    # Initial profiles come from each Optuna trial, never a previous optimum.
    c.pop('initial_logits',None)
    c.pop('initial_profile_provenance',None)
    cfg=case/f'config_hex_OFF_optuna{trials}_M{order}_Q{grid}.json'
    if cfg.exists():
        if json.loads(cfg.read_text(encoding='utf-8-sig'))!=c:
            raise ValueError(f'Existing config differs; preserve it and choose a separate config: {cfg}')
    else:
        cfg.write_text(json.dumps(c,indent=2)+'\n',encoding='utf-8')
    output=case/'results'/f'{run_tag}_M{order}_Q{grid}'
    return case,cfg,output

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--materials',default='Al,Ti,Cr,Cu,Au')
    p.add_argument('--grid',type=int,default=512)
    p.add_argument('--trials',type=int,default=30)
    p.add_argument('--adam-steps',type=int,default=500)
    p.add_argument('--run-tag',default='hex_OFF_optuna30_v1')
    p.add_argument('--prepare-only',action='store_true',help='Write configs and print commands without running optimization')
    p.add_argument('--recover-lock',action='store_true',help='Only after confirming that the previous job has stopped')
    a=p.parse_args(argv)
    materials=a.materials.split(',')
    if not materials or len(set(materials))!=len(materials) or any(m not in CASES for m in materials):
        p.error('Select distinct materials from Al,Ti,Cr,Cu,Au')
    if min(a.grid,a.trials,a.adam_steps)<=0 or a.grid<4*max(CASES[m][1] for m in materials)+4:
        p.error('Positive steps/trials and a sufficient grid are required')
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
            case,cfg,output=prepare(m,grid=a.grid,trials=a.trials,adam_steps=a.adam_steps,run_tag=a.run_tag)
            command=[sys.executable,str(case/'run.py'),'--config',str(cfg),'--device',a.device,
                     '--mode','optimize','--lattices','hexagonal','--timeout-hours','0','--output-root',str(output)]
            if a.recover_lock:command.append('--recover-lock')
            print(f'\n{m}: OFF M={CASES[m][1]} Q={a.grid}, {a.trials} trials',flush=True)
            print(shlex.join(command),flush=True)
            if a.prepare_only:continue
            completed=subprocess.run(command,cwd=str(case),env=env)
            code=completed.returncode
            if code<0 or code>=128:return code if code>=128 else 128-code
            if code:
                failures.append({'material':m,'exit_code':code})
                print(f'{m}: stopped (exit {code}); proceeding to the next material.',flush=True)
    except KeyboardInterrupt:
        return 130
    if failures:print(json.dumps({'status':'some_materials_stopped','materials':failures},indent=2),flush=True)
    return 1 if failures else 0

if __name__=='__main__':
    raise SystemExit(main())
