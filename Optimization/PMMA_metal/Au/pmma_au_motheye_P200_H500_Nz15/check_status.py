"""Read dual-radius optimizer status; never rerun RCWA or modify results."""
from pathlib import Path
import argparse,csv,json,math,sys
HERE=Path(__file__).resolve().parent


def read(path):return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else {}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,default=Path('results/hex_trial1_M8_Q512_v1'))
    p.add_argument('--trial',type=int,default=0)
    a=p.parse_args();root=a.run_root if a.run_root.is_absolute() else HERE/a.run_root
    folder=root/'hexagonal/trials'/f'trial_{a.trial:04d}'
    workflow=read(root/'workflow.json');outcome=read(folder/'outcome.json');c=read(root/'resolved_config.json')
    state={};checkpoint=folder/'adam_checkpoint.pt'
    if checkpoint.exists():
        import torch
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    trials=[]
    if (root/'hexagonal/trials.csv').exists():
        with (root/'hexagonal/trials.csv').open(encoding='utf-8-sig',newline='') as f:trials=list(csv.DictReader(f))
    row=next((x for x in trials if int(x['trial_number'])==a.trial),{})
    value=state.get('best_loss');cached=state.get('cached_valid_evaluation');stats=state.get('stats',{})
    last=state.get('history',[]);last=last[-1] if last else None
    profile_ok=None
    if state.get('logits') is not None:
        from optimizer.geometry import radii_values
        pairs=radii_values(state['logits'],c)
        profile_ok=(all(0<a<b<c['period_nm']/2 for a,b in pairs) and
                    all(a[0]<b[0] and a[1]<b[1] for a,b in zip(pairs,pairs[1:])))
    complete=(workflow.get('status')=='completed' and row.get('state')=='COMPLETE' and
              outcome.get('status')=='completed' and state.get('current_state_validated') is True)
    report=dict(optimization_completed=complete,workflow_status=workflow.get('status'),
        optuna_trial_state=row.get('state'),worker_status=outcome.get('status'),
        objective_name='mean_R_plus_T',best_mean_absorptance=1-value if value is not None and math.isfinite(value) else None,
        best_step=state.get('best_step'),accepted_adam_updates=state.get('next_step'),
        valid_gradient_evaluations=stats.get('valid_gradient_evaluations'),rejected_proposals=stats.get('rejected_proposals'),
        current_shape_validated=state.get('current_state_validated'),both_radius_profiles_monotone_and_nested=profile_ok,
        last_history=last,last_evaluation_seconds_per_wavelength=cached['seconds']/len(c['optimization_wavelengths_nm']) if cached else None,
        termination_reason=outcome.get('termination_reason'),error=outcome.get('error'),run_root=str(root))
    print(json.dumps(report,indent=2))
    if complete:return 0
    return 1 if row.get('state')=='FAIL' or outcome.get('status') in ('error','backtracking_exhausted') else 2


if __name__=='__main__':raise SystemExit(main())
