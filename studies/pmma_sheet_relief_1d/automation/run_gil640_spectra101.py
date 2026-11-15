"""Run the four both-interface GIL convergence campaigns, then M640 spectra.

Only this wrapper is new. Physics modules and existing point-cache keys are unchanged.
"""
from pathlib import Path
import argparse
import csv
import json
import math
import shlex
import sys

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY))
from automation import converge_until as controller
from model import save_json

GILS = ('0', '0.5', '1', '2')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--solver-root', type=Path)
    p.add_argument('--convergence-dir', type=Path,
                   default=STUDY/'results/fff1d/P3720_w700_Nz960_GIL')
    p.add_argument('--spectrum-root', type=Path, default=STUDY/'results/fff1d')
    p.add_argument('--output-dir', type=Path,
                   default=STUDY/'results/fff1d/workflow_M640_spectra101_both')
    p.add_argument('--dry-run', action='store_true', help='print all five commands; launch no calculations')
    return p


def commands(args):
    common = ['--periods', '3720', '--polarization', 'p', '--slices', '960',
              '--coating', 'gold', '--gold-nm', '50', '--voids', 'both',
              '--device', args.device, '--phase-tol', '0.00005', '--phase-max', '262144']
    if args.solver_root is not None:
        common += ['--solver-root', str(args.solver_root)]
    first = [sys.executable, str(STUDY/'fff1d/converge_gil.py'),
             '--gil-values', ','.join(GILS), '--gil-interface-values', 'both',
             '--start', '16', '--step', '16', '--max-order', '640',
             '--wavelengths', '700', '--tolerance', '0.001', *common,
             '--output-dir', str(args.convergence_dir.resolve())]
    following = []
    for gil in GILS:
        dest = args.spectrum_root.resolve()/f'spectrum_P3720_M640_Nz960_GIL{gil}_both'
        command = [sys.executable, str(STUDY/'fff1d/run_fff.py'),
                   '--gil-nm', gil, '--gil-interfaces', 'both', '--order', '640',
                   '--wavelengths', '400:700:3', *common,
                   '--no-preview', '--output-dir', str(dest)]
        following.append((gil, dest, command))
    return first, following


def check_convergence(path):
    summary = controller.load(path/'gil_summary.json')
    expected = {(float(g), 'none' if float(g) == 0 else 'both') for g in GILS}
    cases = summary.get('cases', [])
    keys = {(r['gil_nm'], r['gil_interfaces']) for r in cases}
    if len(cases) != 4 or keys != expected:
        raise ValueError('Convergence summary does not contain exactly the four requested GIL settings.')
    for r in cases:
        if r['status'] not in ('converged', 'limit_reached') or r['max_order'] != 640:
            raise ValueError(f"Convergence stopped: GIL={r['gil_nm']}, status={r['status']}. Spectra not started.")
        completed = r['last_completed_order']
        if not isinstance(completed, int) or not 16 <= completed <= 640:
            raise ValueError('Invalid completed convergence order.')
        if r['status'] == 'limit_reached' and completed != 640:
            raise ValueError('Maximum-order campaign did not complete M640.')
    return summary


def check_spectrum(path, gil):
    completion = controller.load(path/'completion.json')
    with (path/'spectra.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    expected = {(void, float(400+3*i)) for void in ('air', 'water') for i in range(101)}
    keys = {(r['void'], float(r['wavelength_nm'])) for r in rows}
    if len(rows) != 202 or keys != expected or completion.get('completed_cases') != 202:
        raise ValueError('Incomplete spectrum: expected 101 wavelengths for each of air and water.')
    for r in rows:
        if (int(r['order']) != 640 or int(r['slices']) != 960 or r['polarization'] != 'p'
                or float(r['gil_nm']) != float(gil) or r['gil_interfaces'] != 'both'
                or float(r['period_nm']) != 3720 or float(r['gold_vertical_nm']) != 50
                or r['coating'] != 'gold'):
            raise ValueError('Unexpected spectrum configuration.')
        if r['phase_converged'] != 'True' or any(not math.isfinite(float(r[k])) for k in ('R', 'T', 'A_total')):
            raise ValueError('Nonfinite spectrum or unconverged phase average.')
    if completion.get('phase_converged') is not True:
        raise ValueError('Spectrum phase average did not converge.')
    return dict(gil_nm=float(gil), output_dir=str(path), cases=202,
                passivity_valid=all(r.get('passivity_valid') == 'True' for r in rows),
                point_cache_hits=completion.get('point_cache_hits', 0))


def main(argv=None):
    args = parser().parse_args(argv)
    first, following = commands(args)
    if args.dry_run:
        for command in [first, *[c for _, _, c in following]]:
            print(shlex.join(command))
        return 0
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output/'logs').mkdir(exist_ok=True)
    args.convergence_dir.mkdir(parents=True, exist_ok=True)
    state = dict(status='starting', stage=None, completed_spectra=[],
                 convergence_dir=str(args.convergence_dir.resolve()), device=args.device,
                 order=640, slices=960, wavelengths_nm='400:700:3', points_per_void=101,
                 note='Fixed M640 spectra; convergence and layer accuracy are not certified by workflow completion.')
    def status(value, **extra):
        state.update(status=value, **extra)
        save_json(output/'workflow.json', state)
    acquired = False
    try:
        # Parent-level lock also prevents a second workflow from writing the spectra.
        with controller.campaign_lock(args.convergence_dir):
            acquired = True
            status('running', stage='convergence')
            print('Step 1/5: both-interface convergence, 700 nm, maximum M640', flush=True)
            log = output/'logs/01_convergence.log'
            code = controller.run_child(first, log)
            if code not in (0, 2):
                status('convergence_failed', child_exit_code=code, log=str(log))
                return code if code > 0 else 4
            # Exit 2 can also mean phase failure; inspect every setting before proceeding.
            state['convergence'] = check_convergence(args.convergence_dir)
            for step, (gil, dest, command) in enumerate(following, 2):
                status('running', stage='spectrum', current_gil_nm=float(gil))
                print(f'Step {step}/5: GIL={gil} nm, both interfaces, M640, 101 wavelengths, air/water', flush=True)
                log = output/'logs'/f'{step:02}_spectrum_GIL{gil}.log'
                code = controller.run_child(command, log)
                if code:
                    oom = controller.oom_evidence(controller.log_tail(log))
                    status('oom' if oom else 'spectrum_failed', child_exit_code=code,
                           log=str(log), stop_reason=oom or 'Child calculation did not finish successfully.')
                    return 3 if oom else code if code > 0 else 4
                state['completed_spectra'].append(check_spectrum(dest, gil))
                status('running')
            passive = all(r['passivity_valid'] for r in state['completed_spectra'])
            status('completed' if passive else 'completed_with_nonpassive_points', stage=None,
                   spectrum_cases=808, passivity_valid=passive)
            print(f'Finished all four spectra. Workflow record: {output / "workflow.json"}', flush=True)
            return 0 if passive else 2
    except KeyboardInterrupt:
        if acquired:
            status('interrupted', stop_reason='Rerun the same command to reuse completed points.')
        return 130
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        reason = f'{type(exc).__name__}: {exc}'
        if acquired:
            status('error', stop_reason=reason)
        print(reason, file=sys.stderr, flush=True)
        return 4


if __name__ == '__main__':
    raise SystemExit(main())
