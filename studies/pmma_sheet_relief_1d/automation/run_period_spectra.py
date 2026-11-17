"""Independent per-period/per-void RTA convergence, then spectra at the third-last M."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import sys

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY))
from model import Materials, geometry, parse_wavelengths, save_csv, save_json
from automation import converge_until as controller

FILES = ('settings.json', 'completion.json', 'spectra.csv',
         'diffraction_orders.csv', 'local_single_pass_orders.csv')
METRICS = ('R', 'T', 'A_total')
DEFAULT_PERIODS = '5,10,20,30,40,50,60,70,80,90,100'


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--periods-um', default=DEFAULT_PERIODS, help='comma-separated periods in micrometres')
    p.add_argument('--slices', type=int, default=15000)
    p.add_argument('--start', type=int, default=320)
    p.add_argument('--start-from-previous', action='store_true',
                   help='start each new period at the last converged selected M for the same void')
    p.add_argument('--step', type=int, default=16)
    p.add_argument('--max-order', type=int, default=1600)
    p.add_argument('--tolerance', type=float, default=.001, help='absolute power difference; .001 = 0.1 percentage points')
    p.add_argument('--wavelengths', default='400:700:3')
    p.add_argument('--voids', choices=['air', 'water', 'both'], default='both')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--polarization', choices=['s', 'p'], default='p')
    p.add_argument('--coating', choices=['gold', 'uncoated'], default='gold')
    p.add_argument('--gold-nm', type=float, default=50.)
    p.add_argument('--phase-tol', type=float, default=5e-5)
    p.add_argument('--phase-max', type=int, default=262144)
    p.add_argument('--phase-points', type=int, default=512)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--solver-root', type=Path)
    p.add_argument('--cache-dir', type=Path)
    p.add_argument('--plan-cache-mib', type=int, default=1024)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--dry-run', action='store_true')
    return p


def plan(args):
    periods = [float(x) for x in args.periods_um.split(',')]
    if not periods or len(set(periods)) != len(periods) or any(not math.isfinite(x) or x <= 0 for x in periods):
        raise ValueError('Use distinct positive finite --periods-um.')
    if args.start_from_previous and periods != sorted(periods):
        raise ValueError('--start-from-previous requires periods in increasing order.')
    if args.slices < 1 or args.start < 1 or args.step < 1 or args.max_order < args.start:
        raise ValueError('Require positive slices/start/step and max-order >= start.')
    if (args.max_order-args.start) % args.step:
        raise ValueError('--max-order must lie on the start/step grid.')
    if not math.isfinite(args.tolerance) or args.tolerance <= 0:
        raise ValueError('Require a positive finite tolerance.')
    if not math.isfinite(args.gold_nm) or args.gold_nm < 0 or args.plan_cache_mib < 0:
        raise ValueError('Require nonnegative gold-nm and plan-cache-mib.')
    if args.threads < 1 or not math.isfinite(args.phase_tol) or args.phase_tol <= 0:
        raise ValueError('Require positive threads and phase tolerance.')
    for value in (args.phase_points, args.phase_max):
        if value < 16 or value & (value-1):
            raise ValueError('Phase point budgets must be powers of two >= 16.')
    if args.phase_max < 2*args.phase_points:
        raise ValueError('phase-max must be >= 2*phase-points.')
    waves = [float(w) for w in parse_wavelengths(args.wavelengths)]
    material = Materials()
    voids = ('air', 'water') if args.voids == 'both' else (args.voids,)
    cases = []
    for period in periods:
        nm = period*1000
        n700 = material.pmma(700)[0]
        minimum_700 = math.floor(n700*nm/700)
        minimum_spectrum = max(math.floor(material.pmma(w)[0]*nm/w) for w in waves)
        g = geometry(period=nm, gold=args.gold_nm if args.coating == 'gold' else 0.)
        for void in voids:
            cases.append(dict(period_um=period, period_nm=nm, void=void, geometry=g,
                minimum_700_order=minimum_700, minimum_spectrum_order=minimum_spectrum,
                first_computable_order=args.start+max(0, math.ceil((minimum_700-args.start)/args.step))*args.step))
    return cases, waves


def environment(args):
    from model import solver_root
    root = solver_root(args.solver_root)
    from fff1d.run_fff import source_signature
    import numpy as np
    import torch
    from fast_solver import canonical_device
    return dict(solver_root=str(root), source_sha256=source_signature(root),
        material_sha256={k:v['sha256'] for k,v in Materials().metadata().items()},
        device=str(canonical_device(args.device)), torch=str(torch.__version__), numpy=str(np.__version__))


def identity(args, case, waves, env):
    return dict(version=2, period_nm=case['period_nm'], void=case['void'],
        geometry=case['geometry'], slices=args.slices, polarization=args.polarization,
        coating=args.coating, start=args.start, step=args.step, tolerance=args.tolerance,
        spectrum_wavelengths_nm=waves, convergence_wavelength_nm=700.,
        phase_tol=args.phase_tol, phase_max=args.phase_max, phase_points=args.phase_points,
        gil_nm=0., gil_interfaces='both', symmetry='even', section_sampling='midpoint',
        criterion='max absolute change of total R,T,A only; two consecutive comparisons',
        selection='third-last computed order, provided all spectrum propagating orders are included',
        environment=env)


def point_command(args, case, order, waves, dest):
    command = [sys.executable, str(STUDY/'automation/run_fff_cached.py'),
        '--plan-cache-mib', str(args.plan_cache_mib),
        '--periods', f"{case['period_nm']:g}", '--voids', case['void'],
        '--order', str(order), '--slices', str(args.slices), '--wavelengths', waves,
        '--polarization', args.polarization, '--coating', args.coating,
        '--gold-nm', str(args.gold_nm), '--geometry-mode', 'scaled',
        '--gil-nm', '0', '--gil-interfaces', 'both', '--device', args.device,
        '--phase-tol', str(args.phase_tol), '--phase-max', str(args.phase_max),
        '--phase-points', str(args.phase_points), '--threads', str(args.threads),
        '--no-preview', '--output-dir', str(dest)]
    if waves == '700':
        command.append('--no-plot')
    if args.solver_root:
        command += ['--solver-root', str(args.solver_root)]
    if args.cache_dir:
        command += ['--cache-dir', str(args.cache_dir)]
    return command


def read_result(args, case, order, waves, dest, env):
    settings = controller.load(dest/'settings.json')
    completion = controller.load(dest/'completion.json')
    checks = dict(order=order, slices=args.slices, polarization=args.polarization,
        coating=args.coating, gil_nm=0., gil_interfaces='both', geometry_mode='scaled',
        source_sha256=env['source_sha256'], phase_tol=args.phase_tol,
        phase_max=args.phase_max, phase_points=args.phase_points)
    if any(settings.get(k) != v for k,v in checks.items()):
        raise ValueError('Child settings/source differ from the requested case.')
    if settings.get('geometries', {}).get(f"{args.coating}/P{case['period_nm']:g}") != case['geometry']:
        raise ValueError('Child geometry differs from requested scaled geometry.')
    with (dest/'spectra.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    expected = {(case['void'], w) for w in waves}
    keys = [(r['void'], float(r['wavelength_nm'])) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected or completion.get('completed_cases') != len(rows):
        raise ValueError('Missing, duplicated or unexpected wavelengths/voids.')
    for row in rows:
        if (float(row['period_nm']) != case['period_nm'] or row['polarization'] != args.polarization
                or int(row['order']) != order or int(row['slices']) != args.slices):
            raise ValueError('Unexpected spectrum row settings.')
        if any(not math.isfinite(float(row[k])) for k in (*METRICS, 'phase_error')):
            raise ValueError('Nonfinite power or phase indicator.')
    phase_ok = completion.get('phase_converged') is True and all(r['phase_converged']=='True' for r in rows)
    passive = completion.get('passivity_valid') is True and all(r.get('passivity_valid')=='True' for r in rows)
    return rows, phase_ok, passive


def obtain(args, case, order, waves_text, waves, dest, env, ident_hash):
    marker = dest/'.period_sweep_complete.json'
    complete = False
    if marker.exists():
        try:
            m = controller.load(marker)
            complete = (m['identity_sha256']==ident_hash and m['order']==order and m['waves']==waves
                and set(m['files'])==set(FILES)
                and all((dest/n).is_file() and controller.digest(dest/n)==h for n,h in m['files'].items()))
        except (ValueError, KeyError, OSError, TypeError):
            complete = False
    if complete:
        print(f"Reusing complete P={case['period_um']:g} um {case['void']} M={order} {waves_text} nm", flush=True)
    else:
        dest.mkdir(parents=True, exist_ok=True)
        log = dest/'run.log'
        code = controller.run_child(point_command(args, case, order, waves_text, dest), log)
        if code:
            evidence = controller.oom_evidence(controller.log_tail(log))
            if evidence:
                return None, 'oom', dict(exit_code=code, reason=evidence, log=str(log))
            if code in (-9,137):
                return None, 'killed_possible_oom', dict(exit_code=code, log=str(log))
            if code == 2 and (dest/'completion.json').exists() and not controller.load(dest/'completion.json').get('phase_converged'):
                return None, 'phase_unconverged', dict(exit_code=code, log=str(log))
            return None, 'child_error', dict(exit_code=code, log=str(log))
    rows, phase_ok, passive = read_result(args, case, order, waves, dest, env)
    if not phase_ok:
        return rows, 'phase_unconverged', {}
    save_json(marker, dict(identity_sha256=ident_hash, order=order, waves=waves,
                          files={n:controller.digest(dest/n) for n in FILES}))
    return rows, 'ok', dict(passivity_valid=passive)


def maximum_rta_change(before, after):
    return max(abs(float(after[k])-float(before[k])) for k in METRICS)


def case_start(args, case, dest, seeds):
    effective = argparse.Namespace(**vars(args))
    info = dict(mode='fixed', start_order=args.start)
    if not args.start_from_previous:
        return effective, info
    previous = seeds.get(case['void'])
    manifest = dest/'identity.json'
    if manifest.exists():
        # Preserve a case that was started under either policy. run_case still
        # validates every physical setting, environment and selection rule.
        recorded = controller.load(manifest)['start']
        if type(recorded) is not int or recorded < 1 or (args.max_order-recorded) % args.step:
            raise ValueError(f'Invalid recorded start order in {manifest}.')
        effective.start = recorded
        info = dict(mode='existing_case', start_order=recorded)
    elif previous is not None:
        effective.start = previous['selected_order']
        info = dict(mode='previous_period', start_order=effective.start,
                    source_period_um=previous['period_um'], source_void=case['void'])
    else:
        info = dict(mode='initial_fallback', start_order=args.start)
    return effective, info


def run_case(args, case, waves, env, dest, start_info=None):
    dest.mkdir(parents=True, exist_ok=True)
    ident = identity(args, case, waves, env)
    manifest = dest/'identity.json'
    if manifest.exists() and controller.load(manifest) != ident:
        raise ValueError(f'Different settings/source in {dest}. Use a new output directory.')
    save_json(manifest, ident)
    ident_hash = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()
    state = dict(period_um=case['period_um'], period_nm=case['period_nm'], void=case['void'],
        slices=args.slices, gil_nm=0., status='starting', selected_order=None,
        minimum_700_order=case['minimum_700_order'], minimum_spectrum_order=case['minimum_spectrum_order'],
        criterion=ident['criterion'], selected_order_rule=ident['selection'],
        start_order=args.start, start_strategy=start_info or dict(mode='fixed', start_order=args.start),
        checked_orders=[], skipped_orders=[], comparisons=[])

    def status(value, **extra):
        state.update(status=value, **extra)
        save_json(dest/'case.json', state)

    previous = None
    try:
        for order in range(args.start, args.max_order+1, args.step):
            if order < case['minimum_700_order']:
                state['skipped_orders'].append(dict(order=order, reason='omits propagating PMMA orders at 700 nm'))
                continue
            status('running_convergence', attempted_order=order)
            print(f"P={case['period_um']:g} um / {case['void']}: 700 nm M={order}", flush=True)
            rows, outcome, info = obtain(args, case, order, '700', [700.],
                dest/'convergence'/f'order_{order}', env, ident_hash)
            if outcome != 'ok':
                status(outcome, failure_stage='convergence', **info)
                return state
            row = rows[0]
            state['checked_orders'].append(order)
            if previous:
                old_order, old_row, old_passive = previous
                change = maximum_rta_change(old_row, row)
                phase_ok = max(float(old_row['phase_error']), float(row['phase_error'])) <= args.tolerance/4
                passed = change <= args.tolerance and phase_ok and old_passive and info['passivity_valid']
                comparison = dict(previous_order=old_order, order=order, maximum_RTA_change=change,
                    maximum_RTA_change_pp=100*change, passed=bool(passed),
                    **{f'delta_{k}':abs(float(row[k])-float(old_row[k])) for k in METRICS})
                state['comparisons'].append(comparison)
                save_csv(dest/'adjacent_RTA_changes.csv', state['comparisons'])
                print(f"{old_order} -> {order}: max delta(R,T,A)={change:.7g}; passed={passed}", flush=True)
                two_passes = len(state['comparisons']) >= 2 and all(r['passed'] for r in state['comparisons'][-2:])
                candidate = state['checked_orders'][-3] if two_passes else None
                if two_passes and candidate >= case['minimum_spectrum_order']:
                    state.update(selected_order=candidate, convergence_final_order=order,
                                 convergence_passed=True, convergence_wavelength_nm=700.)
                    break
                if two_passes:
                    print(f"RTA comparisons passed, but candidate M={candidate} omits spectrum propagating orders "
                          f"up to {case['minimum_spectrum_order']}; continuing on the same M grid.", flush=True)
            previous = (order, row, info['passivity_valid'])
            status('continuing_convergence')
        if state['selected_order'] is None:
            status('limit_reached', max_order=args.max_order,
                   reason='No eligible two-consecutive-pass selection by the maximum order; spectrum not started.')
            return state
        selected = state['selected_order']
        status('running_spectrum')
        print(f"Selected M={selected} (third-last); P={case['period_um']:g} um / {case['void']}; spectrum", flush=True)
        rows, outcome, info = obtain(args, case, selected, args.wavelengths, waves,
            dest/'spectrum', env, ident_hash)
        if outcome != 'ok':
            status(outcome, failure_stage='spectrum', **info)
        elif not info['passivity_valid']:
            status('spectrum_nonpassive', completed_points=len(rows),
                   reason='Saved without clipping; some spectrum points fail passivity.')
        else:
            status('completed', completed_points=len(rows),
                   spectrum_order_and_layer_convergence_over_band_verified=False)
        return state
    except KeyboardInterrupt:
        status('interrupted')
        raise
    except MemoryError:
        status('oom', reason='MemoryError in workflow controller')
        return state
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        status('error', reason=f'{type(exc).__name__}: {exc}')
        return state


def save_summary(output, cases):
    save_json(output/'summary.json', dict(cases=cases))
    if cases:
        save_csv(output/'summary.csv', [dict(period_um=c['period_um'], void=c['void'],
            status=c['status'], selected_order=c.get('selected_order'),
            start_order=c.get('start_order'),
            convergence_final_order=c.get('convergence_final_order'), output_dir=c['output_dir']) for c in cases])


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        cases, waves = plan(args)
        if args.dry_run:
            print('Period(um) void  tip=height(um)  first computable M@700  spectrum minimum M')
            for case in cases:
                print(f"{case['period_um']:9g} {case['void']:5s} {case['geometry']['height_nm']/1000:15.6f} "
                      f"{case['first_computable_order']:23d} {case['minimum_spectrum_order']:19d}")
            print(f'Grid M={args.start}:{args.step}:{args.max_order}; wavelengths={args.wavelengths}; '
                  f'{len(cases)} independent cases; {len(waves)} spectral points per case.')
            print('Select third-last M after two passing total-RTA comparisons; no spectrum for unselected cases.')
            if args.start_from_previous:
                print('New periods inherit the selected M from the latest converged smaller period, separately for each void; '
                      'the inherited start is known at runtime. Existing cases keep their recorded start.')
            return 0
        env = environment(args)
        args.device = env['device']
        output = (args.output_dir or STUDY/f'results/fff1d/period_sweep_Nz{args.slices}_GIL0_RTA_start{args.start}_pick3').resolve()
        output.mkdir(parents=True, exist_ok=True)
        results = []
        seeds = {}
        with controller.campaign_lock(output):
            save_json(output/'requested_plan.json', dict(cases=cases, wavelengths_nm=waves,
                max_order=args.max_order, plan_cache_mib=args.plan_cache_mib,
                start_from_previous=args.start_from_previous, initial_start=args.start))
            for case in cases:
                dest = output/f"P{case['period_nm']:g}"/case['void']
                try:
                    effective, start_info = case_start(args, case, dest, seeds)
                    print(f"Start P={case['period_um']:g} um / {case['void']}: M={effective.start} "
                          f"({start_info['mode']})", flush=True)
                    state = run_case(effective, case, waves, env, dest, start_info)
                except KeyboardInterrupt:
                    state = controller.load(dest/'case.json')
                    state['output_dir'] = str(dest); results.append(state)
                    save_summary(output, results)
                    return 130
                state['output_dir'] = str(dest); results.append(state)
                if state.get('convergence_passed') and state.get('selected_order') is not None:
                    seeds[case['void']] = dict(period_um=case['period_um'], selected_order=state['selected_order'])
                save_summary(output, results)
                print(f"Finished P={case['period_um']:g} um {case['void']}: {state['status']}; "
                      f"selected M={state.get('selected_order')}", flush=True)
        if all(r['status']=='completed' for r in results): return 0
        if any(r['status'] in ('oom','killed_possible_oom') for r in results): return 3
        if any(r['status'] in ('error','child_error','phase_unconverged') for r in results): return 4
        return 2
    except (ValueError, OSError, RuntimeError) as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        return 4


if __name__ == '__main__':
    raise SystemExit(main())
