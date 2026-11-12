"""Sequential Au30 sidewall sweep with native D6 and per-case resume.

The default cases are radial sidewall thicknesses 10, 6 and 4 nm.
The cap follows the outer sidewall slope, as in the existing 8 nm case.
"""

import argparse
import math
from pathlib import Path
import subprocess
import sys

from run_au30_glass1mm import positive_int


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def side_nm(text):
    value = float(text)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError('Side thickness must be finite and nonnegative.')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--side-levels', type=side_nm, nargs='+', default=[10., 6., 4.])
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--order', type=positive_int, default=36)
    parser.add_argument('--slices', type=positive_int, default=150)
    parser.add_argument('--grid', type=positive_int, default=512)
    parser.add_argument('--wavelengths', default='400:700:3')
    parser.add_argument('--progress-every', type=positive_int, default=1)
    parser.add_argument('--cap-profile', choices=('current', 'side-continuation'),
                        default='side-continuation')
    parser.add_argument('--mirror-nk-csv', type=Path,
                        default=ROOT / 'studies/cu_ag_al_motheye_nv/data/Rakic (Al).csv')
    parser.add_argument('--output-root', type=Path,
                        default=HERE / 'results/Au30_PET5_glass1mm_side_sweep')
    parser.add_argument('--resume', action='store_true',
                        help='Resume cases with saved spectra; start missing cases normally.')
    parser.add_argument('--preview-only', action='store_true',
                        help='Write geometry/settings to separate preview directories; skip RCWA.')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without running them.')
    args = parser.parse_args()

    if len(set(args.side_levels)) != len(args.side_levels):
        parser.error('--side-levels must be unique.')
    if args.preview_only and args.resume:
        parser.error('Use --preview-only without --resume.')
    if args.grid % 2 or args.grid < max(32, 4 * args.order + 4):
        parser.error('Native D6 requires an even grid >= max(32, 4*order+4).')
    if not args.mirror_nk_csv.is_file():
        parser.error(f'Mirror n,k table does not exist: {args.mirror_nk_csv}')

    planned = []
    for side in args.side_levels:
        case_root = (args.output_root / f'side{side:g}nm').resolve()
        spectrum = case_root / 'evaporation/spectrum' / f'M{args.order}_Nz{args.slices}_G{args.grid}'
        has_spectrum = (spectrum / 'spectra.csv').exists()
        if not args.preview_only and has_spectrum and not args.resume:
            parser.error(f'Saved spectrum exists at {spectrum}; add --resume to continue it.')
        if not args.preview_only and has_spectrum and not (spectrum / 'settings.json').is_file():
            parser.error(f'Saved spectrum has no settings.json: {spectrum}')
        command = [
            sys.executable, '-u', str(HERE / 'run_au30_glass1mm.py'), 'evaporation',
            '--device', args.device, '--order', str(args.order),
            '--slices', str(args.slices), '--grid', str(args.grid),
            '--wavelengths', args.wavelengths, '--progress-every', str(args.progress_every),
            '--side-gold-nm', str(side), '--cap-profile', args.cap_profile,
            '--native-d6', '--mirror-nk-csv', str(args.mirror_nk_csv.resolve()),
            '--output-root', str(case_root),
        ]
        if args.preview_only:
            command.append('--preview-only')
        elif args.resume and has_spectrum:
            command.append('--resume')
        planned.append((side, command))

    # Each case gets its own process; the next case starts after its predecessor.
    for number, (side, command) in enumerate(planned, 1):
        print(f'Case {number}/{len(planned)}: Au side radial={side:g} nm; '
              f'cap={args.cap_profile}; native D6 enabled', flush=True)
        if args.dry_run:
            import shlex
            print(shlex.join(command), flush=True)
            continue
        completed = subprocess.run(command, cwd=ROOT)
        if completed.returncode:
            print(f'Side sweep stopped at side={side:g} nm. See the error above.', file=sys.stderr)
            return completed.returncode
    return 0


if __name__ == '__main__':
    sys.exit(main())
