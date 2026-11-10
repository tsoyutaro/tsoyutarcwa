"""Run one convergence sweep or spectrum for Au 30 nm / MOSMITE / PET 5 um / silica 1 mm."""

import argparse
import csv
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError('Expected a positive integer.')
    return value


def powers(folder):
    with (folder / 'spectra.csv').open(newline='', encoding='utf-8') as handle:
        return {float(row['wavelength_nm']): tuple(float(row[k]) for k in ('R', 'T', 'A'))
                for row in csv.DictReader(handle)}


def compare(previous, current, tolerance):
    old, new = powers(previous), powers(current)
    if not old or old.keys() != new.keys():
        raise ValueError('Comparison requires completed spectra at the same wavelengths.')
    changes = {wl: max(abs(a - b) for a, b in zip(old[wl], new[wl])) for wl in new}
    worst = max(changes, key=changes.get)
    values = ', '.join(f'{wl:g} nm={100 * change:.4f}' for wl, change in changes.items())
    status = 'within tolerance' if changes[worst] <= tolerance else 'exceeds tolerance'
    print(f'{previous.name} -> {current.name}: max |delta R,T,A| (pp): {values}; '
          f'{status} ({100 * tolerance:g} pp)', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('evaporation', 'sputter'))
    parser.add_argument('--scan', choices=('order', 'slices', 'grid'),
                        help='Omit to calculate the full spectrum.')
    parser.add_argument('--order', type=positive_int, help='Fixed order or start of the order scan.')
    parser.add_argument('--max-order', type=positive_int, help='Optional upper limit for the order scan.')
    parser.add_argument('--slices', type=positive_int, default=150)
    parser.add_argument('--slice-levels', type=positive_int, nargs='+', default=[150, 300, 600])
    parser.add_argument('--grid', type=positive_int, default=512)
    parser.add_argument('--wavelengths', help='Scan default: 400,550,700. Spectrum default: 400:700:5.')
    parser.add_argument('--progress-every', type=positive_int, default=10,
                        help='Refresh cumulative spectra and print progress every N wavelengths.')
    parser.add_argument('--mirror-nk-csv', type=Path,
                        help='Save mirror- and quartz-referenced spectra at each progress update.')
    parser.add_argument('--tolerance', type=float, default=.005,
                        help='Absolute fraction for comparing R/T/A (default: 0.5 percentage point).')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--reuse-geometry', action='store_true',
                        help='Share fixed RCWA geometry across wavelengths for each scan case.')
    parser.add_argument('--native-d6', action='store_true',
                        help='Build P/Q directly on D6 E1 for nonoverlapping circular layers.')
    parser.add_argument('--output-root', type=Path, default=HERE / 'results' / 'Au30_PET5_glass1mm')
    args = parser.parse_args()

    if not 0 < args.tolerance < 1:
        parser.error('--tolerance must be between zero and one.')
    if args.scan != 'order' and args.order is None:
        parser.error('Specify the selected --order.')
    if args.max_order is not None and args.scan != 'order':
        parser.error('--max-order requires --scan order.')

    m = args.order or (14 if args.mode == 'evaporation' else 6)
    nz, grid = args.slices, args.grid
    if args.scan == 'order':
        supported_max = (grid - 2) // 4
        maximum = args.max_order or supported_max
        if not m <= maximum <= supported_max:
            parser.error(f'Require start <= max order <= {supported_max} at grid={grid}.')
        cases = [(order, nz, grid) for order in range(m, maximum + 1, 2)]
    elif args.scan == 'slices':
        cases = [(m, level, grid) for level in args.slice_levels]
    elif args.scan == 'grid':
        cases = [(m, nz, grid), (m, nz, 2 * grid)]
    else:
        cases = [(m, nz, grid)]
    if any(ng < 4 * order + 2 for order, _, ng in cases):
        parser.error('Each grid must satisfy grid >= 4*order+2.')

    wavelengths = args.wavelengths or ('400,550,700' if args.scan else '400:700:5')
    destination = (args.output_root / args.mode / (args.scan or 'spectrum')).resolve()
    command = [
        sys.executable, '-u', str(HERE / f'run_{args.mode}.py'),
        '--device', args.device, '--wavelengths', wavelengths,
        '--progress-every', str(args.progress_every),
        '--pet-nk-csv', str(ROOT / 'studies/mosmite_uncoated/data/PET_Zhang_400_1000nm.csv'),
        '--period-nm', '100', '--height-nm', '200',
        '--base-diameter-nm', '98', '--tip-diameter-nm', '50',
        '--top-gold-nm', '30', '--valley-gold-nm', '30',
        '--film-total-um', '5.2', '--residual-um', '0', '--adhesive-um', '0',
        '--glass-um', '1000', '--layer-storage', 'reduced', '--coherence', 'incoherent',
    ]
    if args.reuse_geometry:
        command.append('--reuse-geometry')
    if args.native_d6:
        command.append('--native-d6')
    if args.mirror_nk_csv:
        command.extend(['--mirror-nk-csv', str(args.mirror_nk_csv.resolve())])
    previous = None
    for order, nz, grid in cases:
        output = destination / f'M{order}_Nz{nz}_G{grid}'
        print(f'{args.mode}: M={order}, Nz={nz}, grid={grid}, wavelengths={wavelengths}', flush=True)
        completed = subprocess.run(command + [
            '--order', str(order), '--slices', str(nz), '--grid', str(grid),
            '--output-dir', str(output),
        ], cwd=ROOT)
        if completed.returncode:
            print(f'Scan stopped at M={order}, Nz={nz}, grid={grid}. See the error above.', file=sys.stderr)
            return completed.returncode
        if previous is not None:
            compare(previous, output, args.tolerance)
        previous = output
    return 0


if __name__ == '__main__':
    sys.exit(main())
