"""Increase M or Nz until two adjacent comparisons pass, or a child runs out of memory.

Kept below automation/ so the existing physics/point-cache signature is unchanged.
Each resolution runs in a separate process; its CUDA context is released on exit.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY))
from converge import compare, read
from model import Materials, geometry_for_args, parse_wavelengths, save_csv, save_json, solver_root
from polarized.run_selected import parser as calculation_parser, source_signature, selected_polarizations

FILES = ('settings.json', 'completion.json', 'spectra.csv',
         'diffraction_orders.csv', 'local_single_pass_orders.csv')
OOM_PATTERNS = (
    r'(?:torch\.)?(?:cuda\.)?OutOfMemoryError', r'CUDA(?: error:)? out of memory',
    r'CUBLAS_STATUS_ALLOC_FAILED', r'CUSOLVER_STATUS_ALLOC_FAILED',
    r'DefaultCPUAllocator[^\n]*(?:allocate memory|not enough memory)',
    r'^\s*(?:numpy\.[\w.]+\.)?_?ArrayMemoryError:',
    r'^\s*MemoryError(?::|\s*$)', r'std::bad_alloc',
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def oom_evidence(text):
    for pattern in OOM_PATTERNS:
        match = re.search(pattern, text, flags=re.I | re.M)
        if match:
            return match.group(0).strip()
    return None


def log_tail(path):
    with Path(path).open('rb') as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 65536))
        return f.read().decode('utf-8', errors='replace')


@contextmanager
def campaign_lock(output):
    # Kernel locks are released after a crash; an existing lock file is harmless.
    f = (output / '.auto.lock').open('a+b')
    acquired = False
    try:
        if f.tell() == 0:
            f.write(b'\0'); f.flush()
        f.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('Another automatic run is using this output directory.') from exc
        acquired = True
        yield
    finally:
        if acquired:
            f.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        f.close()


def run_child(command, log_path):
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
    with log_path.open('w', encoding='utf-8', newline='') as log:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding='utf-8', errors='replace',
                              bufsize=1, env=env) as process:
            try:
                for line in process.stdout:
                    log.write(line); log.flush()
                    print(line, end='', flush=True)
                return process.wait()
            except BaseException:
                # Do not leave a CUDA calculation running after Ctrl+C or parent I/O failure.
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill(); process.wait(timeout=10)
                raise


def summary_key(row):
    return (row['coating'], row['void'], float(row['period_nm']),
            row['polarization'], float(row['wavelength_nm']))


def context(args, extra):
    import numpy as np
    import torch
    child = calculation_parser().parse_args([
        *extra, '--coating', args.coating, '--geometry-mode', args.geometry_mode])
    if any(v == '--' + args.axis or v.startswith('--' + args.axis + '=') for v in extra):
        raise ValueError(f'--{args.axis} is controlled by --start/--step/--factor; remove it.')
    if child.preview_only:
        raise ValueError('--preview-only cannot be used for convergence.')
    if child.order < 1 or child.slices < 1 or child.threads < 1:
        raise ValueError('Positive order, slices, and threads required.')
    if not math.isfinite(child.phase_tol) or child.phase_tol <= 0:
        raise ValueError('Positive finite phase tolerance required.')
    if child.phase_points < 16 or child.phase_points & (child.phase_points - 1):
        raise ValueError('--phase-points must be a power of two >= 16.')
    if child.phase_max < 2 * child.phase_points or child.phase_max & (child.phase_max - 1):
        raise ValueError('--phase-max must be a power of two >= 2*phase-points.')
    root = solver_root(child.solver_root)
    periods = sorted(set(float(p) for p in child.periods.split(',')))
    waves = [float(w) for w in parse_wavelengths(child.wavelengths)]
    coatings = ['uncoated', 'gold'] if args.coating == 'both' else [args.coating]
    voids = ['air', 'water'] if child.voids == 'both' else [child.voids]
    for coating in coatings:
        for period in periods:
            geometry_for_args(child, period, coating)
    keys = {(c, v, p, pol, w) for c in coatings for v in voids
            for p in periods for pol in selected_polarizations(child.polarization) for w in waves}
    if len(periods) != len(child.periods.split(',')) or len(waves) != len(set(waves)):
        raise ValueError('Duplicate periods/wavelengths are not supported.')
    parameters = dict(vars(child))
    if child.polarization == 'both':
        parameters.pop('polarization')  # Keep existing both-polarization campaigns resumable.
    for k in ('output_dir', 'preview_only', 'no_preview', 'no_plot', args.axis):
        parameters.pop(k, None)
    parameters['solver_root'] = str(root)
    for k, v in list(parameters.items()):
        if isinstance(v, Path):
            parameters[k] = str(v.resolve())
    identity = dict(version=1, axis=args.axis, start=args.start, step=args.step,
                    factor=args.factor, tolerance=args.tolerance, consecutive_passes_required=2,
                    calculation=parameters, physics_source_sha256=source_signature(root, child.polarization),
                    materials={k: v['sha256'] for k, v in Materials().metadata().items()},
                    torch=str(torch.__version__), numpy=str(np.__version__))
    identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return dict(child=child, identity=identity, identity_hash=identity_hash, keys=keys)


def validate_result(dest, ctx, axis, value):
    settings, completion = load(dest / 'settings.json'), load(dest / 'completion.json')
    if settings.get('source_sha256') != ctx['identity']['physics_source_sha256']:
        raise ValueError('Calculation source signature differs from the campaign.')
    if settings.get(axis) != value:
        raise ValueError('Child calculation used a different resolution.')
    summaries = read(dest / 'spectra.csv')
    keys = [summary_key(r) for r in summaries]
    if len(keys) != len(set(keys)) or set(keys) != ctx['keys']:
        raise ValueError('Missing, duplicate, or unexpected spectrum conditions.')
    if completion.get('completed_cases') != len(keys):
        raise ValueError('Incomplete child calculation.')
    for row in summaries:
        for name in ('R', 'T', 'A_total', 'R_le_20deg', 'R_gt_20deg', 'phase_error'):
            if not math.isfinite(float(row[name])):
                raise ValueError(f'Nonfinite {name} in spectrum.')
    order = value if axis == 'order' else ctx['child'].order
    orders = read(dest / 'diffraction_orders.csv')
    order_keys = [(summary_key(r), int(r['m'])) for r in orders]
    expected = {(key, m) for key in ctx['keys'] for m in range(-order, order + 1)}
    if len(order_keys) != len(set(order_keys)) or set(order_keys) != expected:
        raise ValueError('Missing, duplicate, or unexpected diffraction orders.')
    if any(not math.isfinite(float(r[k])) for r in orders for k in ('R_m', 'T_m')):
        raise ValueError('Nonfinite order-resolved power.')
    return completion.get('phase_converged') is True and all(r['phase_converged'] == 'True' for r in summaries)


def reusable(dest, ctx, value):
    marker = dest / '.auto_complete.json'
    if not marker.exists():
        return False
    try:
        record = load(marker)
        return (record['identity_sha256'] == ctx['identity_hash'] and record['resolution'] == value
                and set(record['files']) == set(FILES)
                and all((dest / name).is_file() and digest(dest / name) == h
                        for name, h in record['files'].items()))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def checkpoint(dest, ctx, value):
    save_json(dest / '.auto_complete.json', dict(identity_sha256=ctx['identity_hash'],
        resolution=value, files={name: digest(dest / name) for name in FILES}))


def next_resolution(value, args):
    return value + args.step if args.step is not None else max(value + 1, math.ceil(value * args.factor))


def automatic(args, extra, ctx, output):
    campaign = output / 'auto_campaign.json'
    if campaign.exists():
        if load(campaign).get('identity') != ctx['identity']:
            raise ValueError('Different settings/source in this campaign. Use a new --output-dir.')
    else:
        if (output / 'convergence.json').exists() or (output / 'settings.json').exists():
            raise ValueError('Existing manual results: use a new --output-dir for automatic mode.')
        save_json(campaign, dict(identity=ctx['identity'], identity_sha256=ctx['identity_hash']))
    logs = output / 'logs'; logs.mkdir(exist_ok=True)
    checked, comparisons = [], []
    previous = None
    value = args.start
    state = dict(version=1, axis=args.axis, mode='until_converged_or_oom', converged=False,
                 checked_values=checked, tolerance=args.tolerance, consecutive_passes_required=2,
                 criterion='last two adjacent changes <= tolerance, all outgoing R_m/T_m and R/T/A/cones; phase error <= tolerance/4',
                 note='Only tested wavelengths/cases and this axis; independently verify the other axis.')

    def status(name, **fields):
        trailing_passes = 0
        for row in reversed(comparisons):
            if not row['passed']:
                break
            trailing_passes += 1
        state.update(status=name, converged=name == 'converged', attempted_value=value,
                     last_completed_value=checked[-1] if checked else None,
                     consecutive_passes=trailing_passes,
                     updated_utc=datetime.now(timezone.utc).isoformat(), **fields)
        save_json(output / 'convergence.json', state)
        print(f'Automatic convergence: {name}; completed={checked[-1] if checked else None}; attempted={value}', flush=True)

    try:
        while True:
            if args.max_value is not None and value > args.max_value:
                status('limit_reached', stop_reason='Explicit --max-value reached without convergence.')
                return 2
            dest = output / f'{args.axis}_{value}'
            log = logs / f'{args.axis}_{value}.log'
            if reusable(dest, ctx, value):
                status('resuming', log=str(log), next_value=value)
                print(f'Reusing complete {args.axis} {value}', flush=True)
            else:
                marker = dest / '.auto_complete.json'
                if marker.exists():
                    marker.unlink()
                status('running', log=str(log), next_value=value)
                selected = ctx['child'].polarization != 'both' or any(v == '--polarization' or v.startswith('--polarization=') for v in extra)
                runner = STUDY/'polarized/run_selected.py' if selected else STUDY/'run.py'
                command = [sys.executable, str(runner), *extra,
                    '--coating', args.coating, '--geometry-mode', args.geometry_mode,
                    '--' + args.axis, str(value), '--output-dir', str(dest), '--no-plot', '--no-preview']
                print(f'Running {args.axis} {value}', flush=True)
                code = run_child(command, log)
                if code != 0:
                    evidence = oom_evidence(log_tail(log))
                    if evidence:
                        status('oom', child_exit_code=code, oom_evidence=evidence,
                               stop_reason='Child reported an allocation failure; no higher resolution attempted.')
                        return 3
                    if code in (-9, 137):
                        status('killed_possible_oom', child_exit_code=code,
                               stop_reason='Child killed; OOM cannot be confirmed from this exit code alone.')
                        return 137
                    if code == 2 and (dest / 'completion.json').exists() and load(dest / 'completion.json').get('phase_converged') is False:
                        status('phase_unconverged', child_exit_code=code,
                               stop_reason='Phase integration did not converge; increase its budget in a new campaign.')
                        return 5
                    status('child_error', child_exit_code=code,
                           stop_reason='Non-OOM child failure; inspect the saved log.')
                    return 4
            if not validate_result(dest, ctx, args.axis, value):
                status('phase_unconverged', stop_reason='Phase integration did not converge.')
                return 5
            checkpoint(dest, ctx, value)
            checked.append(value)
            if previous is not None:
                change, phase_error, phase_ok = compare(previous, dest)
                if not math.isfinite(change) or not math.isfinite(phase_error):
                    raise ValueError('Nonfinite convergence comparison.')
                row = dict(axis=args.axis, previous_resolution=checked[-2], resolution=value,
                    maximum_change_all_orders_and_totals=change, phase_error_bound_observed=phase_error,
                    phase_converged=phase_ok, tolerance=args.tolerance,
                    passed=bool(change <= args.tolerance and phase_ok and phase_error <= args.tolerance / 4))
                comparisons.append(row)
                save_csv(output / 'adjacent_changes.csv', comparisons)
                state['last_comparison'] = row
                print(f'{checked[-2]} -> {value}: maximum change={change:.6g}; phase={phase_error:.3g}; passed={row["passed"]}', flush=True)
                if len(comparisons) >= 2 and all(r['passed'] for r in comparisons[-2:]):
                    status('converged', next_value=None, stop_reason='Two consecutive adjacent comparisons passed.')
                    return 0
            previous = dest
            following = next_resolution(value, args)
            status('continuing', next_value=following)
            value = following
    except KeyboardInterrupt:
        status('interrupted', stop_reason='Interrupted; rerun the same command to resume.')
        return 130
    except MemoryError:
        status('oom', oom_evidence='MemoryError in convergence controller',
               stop_reason='CPU allocation failed while checking completed results.')
        return 3
    except Exception as exc:
        status('error', stop_reason=f'{type(exc).__name__}: {exc}')
        print(state['stop_reason'], file=sys.stderr, flush=True)
        return 4


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--axis', choices=['order', 'slices'], required=True)
    p.add_argument('--start', type=int, help='default: M96 or Nz960')
    growth = p.add_mutually_exclusive_group()
    growth.add_argument('--step', type=int, help='add this amount; default for order: 16')
    growth.add_argument('--factor', type=float, help='multiply and round up; default for slices: 2')
    p.add_argument('--max-value', type=int, help='optional explicit ceiling; no ceiling by default')
    p.add_argument('--tolerance', type=float, default=1e-3)
    p.add_argument('--coating', choices=['uncoated', 'gold', 'both'], default='gold')
    p.add_argument('--geometry-mode', choices=['scaled', 'legacy'], default='scaled')
    p.add_argument('--output-dir', type=Path)
    args, extra = p.parse_known_args(argv)
    args.start = args.start if args.start is not None else 96 if args.axis == 'order' else 960
    if args.step is None and args.factor is None:
        if args.axis == 'order':
            args.step = 16
        else:
            args.factor = 2.
    if args.start < 1 or (args.step is not None and args.step < 1):
        p.error('Use a positive --start and --step.')
    if args.factor is not None and (not math.isfinite(args.factor) or args.factor <= 1):
        p.error('--factor must be finite and > 1.')
    if args.max_value is not None and args.max_value < args.start:
        p.error('--max-value must be >= --start.')
    if not math.isfinite(args.tolerance) or args.tolerance <= 0:
        p.error('Tolerance must be positive and finite.')
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(errors='replace')
    try:
        ctx = context(args, extra)
        suffix = '' if ctx['child'].polarization == 'both' else '_'+ctx['child'].polarization
        output = (args.output_dir or STUDY / 'results/v2' / args.geometry_mode / args.coating / f'auto_{args.axis}{suffix}').resolve()
        output.mkdir(parents=True, exist_ok=True)
        with campaign_lock(output):
            return automatic(args, extra, ctx, output)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        return 4


if __name__ == '__main__':
    raise SystemExit(main())
