"""Stop a fixed-order study after a confirmed nonphysical training result.

The original controller records FAIL and exports its journal/CSV before this
adapter unwinds its normal lock cleanup. The caller can then start a separate
study at a higher order. Physics, Adam and their checkpoint identities stay
unchanged. Worker processes still use the original controller entry point.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import uuid

RETRY_EXIT_CODE=75
POLICY='restart-study-at-order-plus-two-v1'

def is_nonphysical(outcome):
    if outcome.get('status')!='error' or outcome.get('error_type')!='NonphysicalResult':
        return False
    message=outcome.get('error','')
    return (message=='Invalid shared power flux.'
            or message.startswith('Nonpassive powers:')
            or message.startswith('Invalid lossless powers:'))

def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else None

def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    temporary.replace(path)

class HigherOrderRequired(BaseException):
    def __init__(self,details):
        self.details=details
        super().__init__(details['error'])

def install(driver):
    """Patch orchestration in this process only, and return an undo callback."""
    original_launch=driver.launch
    original_export=driver.export_trials
    original_optimize=driver.optimize_lattice
    pending=None

    def launch(folder,request):
        nonlocal pending
        outcome=original_launch(folder,request)
        if request.get('mode')=='train' and is_nonphysical(outcome):
            pending=dict(error=outcome['error'],error_type=outcome['error_type'],
                         failed_trial=request['trial_number'],
                         failed_trial_folder=str(request['trial_folder']),
                         training_signature=request['signature'],
                         order=request['config']['optimization_order'],
                         grid=request['config']['optimization_grid'],
                         normal_mode=request['config'].get('normal_mode','off'))
        return outcome

    def export_trials(folder,study):
        original_export(folder,study)
        if pending is not None:
            states=[t for t in study.trials if t.number==pending['failed_trial']]
            if (len(states)!=1 or states[0].state.name!='FAIL'
                    or states[0].user_attrs.get('error')!=pending['error']):
                raise RuntimeError('Nonphysical trial was not safely recorded as FAIL.')
            raise HigherOrderRequired(pending)

    def optimize(folder,request_base,deadline):
        # Identity/config/source checks and the controller lock precede this
        # call in the original main(). Do not continue a previously rejected M.
        marker=read_json(Path(folder).parent/'order_retry.json')
        if (marker and marker.get('policy')==POLICY
                and marker.get('status')=='retry_higher_order'
                and marker.get('training_signature')==request_base['signature']
                and marker.get('order')==request_base['config']['optimization_order']
                and marker.get('grid')==request_base['config']['optimization_grid']
                and marker.get('normal_mode')==request_base['config'].get('normal_mode','off')):
            raise HigherOrderRequired(marker)
        return original_optimize(folder,request_base,deadline)

    driver.launch,driver.export_trials,driver.optimize_lattice=launch,export_trials,optimize
    def undo():
        driver.launch,driver.export_trials,driver.optimize_lattice=original_launch,original_export,original_optimize
    return undo

def run(driver,argv,attempt_id):
    # Parse with the real controller so paths/options have identical semantics.
    args=driver.parser().parse_args(argv)
    if args.mode!='optimize' or args.output_root is None:
        raise ValueError('The order adapter requires --mode optimize and --output-root.')
    output=args.output_root.resolve()
    undo=install(driver)
    try:
        return driver.main(argv)
    except HigherOrderRequired as failure:
        details={**failure.details,'status':'retry_higher_order','policy':POLICY,
                 'attempt_id':attempt_id,'next_order':failure.details['order']+2,
                 'utc':datetime.now(timezone.utc).isoformat(),
                 'adapter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        # The original main() has already run its finally block and released
        # controller.lock. Preserve all original checkpoints and trial records.
        if (output/'controller.lock').exists():
            raise RuntimeError('Original controller lock was not released.')
        write_json(output/'order_retry_history'/f'{attempt_id}.json',details)
        write_json(output/'order_retry.json',details)
        workflow=read_json(output/'workflow.json') or {}
        workflow.update(status='nonphysical_order_retry',order_retry=details)
        write_json(output/'workflow.json',workflow)
        print(f"Nonphysical trial {details['failed_trial']} recorded at M={details['order']}; requesting M={details['next_order']} with a new study.",flush=True)
        return RETRY_EXIT_CODE
    finally:
        undo()

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case-dir',required=True,type=Path)
    p.add_argument('--attempt-id',required=True)
    p.add_argument('driver_args',nargs=argparse.REMAINDER)
    a=p.parse_args(argv)
    # Validate before using this ID as a filename.
    if uuid.UUID(a.attempt_id).hex!=a.attempt_id:
        p.error('attempt-id must be a UUID hex string')
    case=a.case_dir.resolve()
    sys.path.insert(0,str(case))
    dependency=os.environ.get('MOTHEYE_DEPENDENCY_PATH') or os.environ.get('RCWA_OPT_DEPENDENCY_PATH')
    if dependency:sys.path.append(dependency)
    if case.name=='cr_motheye':
        sys.path.insert(0,str(case.parent.parent))
        driver=importlib.import_module('Optimization.cr_motheye.run')
    else:
        driver=importlib.import_module('optimizer.driver')
    arguments=a.driver_args[1:] if a.driver_args[:1]==['--'] else a.driver_args
    return run(driver,arguments,a.attempt_id)

if __name__=='__main__':
    raise SystemExit(main())
