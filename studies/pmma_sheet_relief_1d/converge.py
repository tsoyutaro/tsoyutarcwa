"""Adjacent-resolution checks of every outgoing R_m/T_m and total absorption."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import sys

import numpy as np

from model import HERE, save_csv, save_json


def read(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def compare(left, right):
    def summary_key(r):
        return tuple(r[k] for k in ['coating', 'void', 'period_nm', 'polarization', 'wavelength_nm'])
    before = {summary_key(r): r for r in read(left/'spectra.csv')}
    after = {summary_key(r): r for r in read(right/'spectra.csv')}
    differences = []
    total_keys = ['R', 'T', 'A_total', 'R_le_20deg', 'R_gt_20deg']
    for k, row in after.items():
        differences.extend(abs(float(row[name])-float(before[k][name])) for name in total_keys)
    def order_key(r):
        return summary_key(r)+(r['m'],)
    a = {order_key(r): r for r in read(left/'diffraction_orders.csv')}
    b = {order_key(r): r for r in read(right/'diffraction_orders.csv')}
    for k in a.keys() | b.keys():
        differences.extend(abs(float(a.get(k, {}).get(name, 0))-float(b.get(k, {}).get(name, 0)))
                           for name in ['R_m', 'T_m'])
    phase_error = max(float(r['phase_error']) for r in [*before.values(), *after.values()])
    phase_ok = all(r['phase_converged'] == 'True' for r in [*before.values(), *after.values()])
    return float(max(differences)), phase_error, phase_ok


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--axis', choices=['order', 'slices'], required=True)
    p.add_argument('--values', help='increasing comma-separated integer values')
    p.add_argument('--tolerance', type=float, default=1e-3)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--coating',choices=['uncoated','gold','both'],default='uncoated')
    p.add_argument('--geometry-mode',choices=['scaled','legacy'],default='scaled')
    args, extra = p.parse_known_args()
    if args.tolerance <= 0 or not np.isfinite(args.tolerance):
        p.error('Tolerance must be positive and finite.')
    values = [int(v) for v in (args.values or ('16,24,32,40,48' if args.axis == 'order' else '60,120,240,480')).split(',')]
    if any(a >= b for a, b in zip(values, values[1:])) or min(values) < 1 or len(values) < 3:
        p.error('Use at least three increasing positive resolutions.')
    output = (args.output_dir or HERE/'results/v2'/args.geometry_mode/args.coating/f'convergence_{args.axis}').resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows, previous = [], None
    for value in values:
        dest = output/f'{args.axis}_{value}'
        command = [sys.executable, str(HERE/'run.py'), *extra,'--coating',args.coating,'--geometry-mode',args.geometry_mode,
                   '--'+args.axis, str(value), '--output-dir', str(dest), '--no-plot', '--no-preview']
        print('Running', args.axis, value, flush=True)
        subprocess.run(command, check=True)
        if previous:
            change, phase_error, phase_ok = compare(previous, dest)
            rows.append(dict(axis=args.axis, previous_resolution=int(previous.name.rsplit('_', 1)[1]),
                resolution=value, maximum_change_all_orders_and_totals=change,
                phase_error_bound_observed=phase_error, phase_converged=phase_ok,
                tolerance=args.tolerance, passed=bool(change <= args.tolerance and phase_ok and phase_error <= args.tolerance/4)))
            save_csv(output/'adjacent_changes.csv', rows)
        previous = dest
    passed = len(rows) >= 2 and all(r['passed'] for r in rows[-2:])
    save_json(output/'convergence.json', dict(axis=args.axis, converged=passed,
        checked_values=values, tolerance=args.tolerance,
        criterion='last two adjacent changes <= tolerance, all outgoing R_m/T_m and R/T/A/cones; phase error <= tolerance/4',
        note='Convergence applies only to tested wavelengths/cases and this axis; independently verify the other axis.'))
    print('Converged:', passed, flush=True)
    return 0 if passed else 2


if __name__ == '__main__':
    sys.exit(main())
