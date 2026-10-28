"""Read the status of the Ti trial without running RCWA or modifying results."""
from pathlib import Path
import argparse
import csv
import json
import math

ROOT=Path(__file__).resolve().parent


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else {}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case-dir',type=Path,default=ROOT/'ti_motheye_P200_H500_Nz15')
    p.add_argument('--run-root',type=Path,default=Path('results/hex_trial1_M8_Q512_v3'))
    p.add_argument('--trial',type=int,default=0)
    args=p.parse_args()
    case=args.case_dir.resolve()
    run=args.run_root if args.run_root.is_absolute() else case/args.run_root
    trial=run/'hexagonal/trials'/f'trial_{args.trial:04d}'
    workflow=read_json(run/'workflow.json')
    outcome=read_json(trial/'outcome.json')
    configuration=read_json(run/'resolved_config.json')
    rows=[]
    csv_path=run/'hexagonal/trials.csv'
    if csv_path.exists():
        with csv_path.open(encoding='utf-8-sig',newline='') as f:
            rows=list(csv.DictReader(f))
    row=next((r for r in rows if int(r['trial_number'])==args.trial),{})
    checkpoint=trial/'adam_checkpoint.pt'
    state={}
    if checkpoint.exists():
        import torch
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    history=state.get('history',[])
    last=history[-1] if history else None
    count=len(configuration.get('optimization_wavelengths_nm',[]))
    complete=(workflow.get('status')=='completed' and outcome.get('status')=='completed'
              and row.get('state')=='COMPLETE' and (trial/'best.json').exists())
    best_loss=state.get('best_loss')
    report=dict(optimization_completed=complete,workflow_status=workflow.get('status'),
        optuna_trial_state=row.get('state'),worker_status=outcome.get('status'),
        adam_updates=state.get('next_step'),successful_gradient_evaluations=len(history),
        best_objective=best_loss if best_loss is not None and math.isfinite(best_loss) else None,
        best_step=state.get('best_step'),last_history=last,
        last_successful_evaluation_seconds_per_wavelength=last['evaluation_seconds']/count if last and count else None,
        error=outcome.get('error'),run_root=str(run))
    print(json.dumps(report,indent=2))
    if complete:return 0
    return 1 if row.get('state')=='FAIL' or outcome.get('status')=='error' else 2


if __name__=='__main__':
    raise SystemExit(main())
