"""Postprocess absolute RCWA R/T/A and reference-matched R/T/A_comparison."""

import argparse
import csv
from pathlib import Path

import numpy as np

from stack import save_csv, silica_n
from shared.tabulated_nk import load_tabulated_nk


def load_table(path, *columns):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        required = ('wavelength_nm', *columns)
        if not set(required).issubset(reader.fieldnames or ()):
            raise ValueError(f'{path}: expected columns {", ".join(required)}.')
        rows = [[float(row[name]) for name in required] for row in reader]
    if not rows or not np.isfinite(rows).all():
        raise ValueError(f'{path}: expected finite numeric data.')
    data = np.asarray(rows)
    data = data[np.argsort(data[:, 0])]
    wavelengths = data[:, 0]
    if wavelengths[0] <= 0 or np.any(np.diff(wavelengths) <= 0):
        raise ValueError(f'{path}: wavelengths must be positive and unique.')
    return wavelengths, data[:, 1:]


def reference_at(path, column, wavelengths):
    source_wavelengths, data = load_table(path, column)
    percent = data[:, 0]
    if np.any((percent <= 0) | (percent > 100)):
        raise ValueError(f'{path}: {column} must be in (0, 100].')
    if wavelengths[0] < source_wavelengths[0] or wavelengths[-1] > source_wavelengths[-1]:
        raise ValueError(f'{path}: reference must cover all simulation wavelengths.')
    return np.interp(wavelengths, source_wavelengths, percent) / 100


def bare_mirror_r(path, wavelengths):
    optical_constants = load_tabulated_nk(path)
    index = np.sqrt(np.array([optical_constants(wl) for wl in wavelengths]))
    reflectance = np.abs((1 - index) / (1 + index))**2
    if np.any((reflectance <= 0) | (reflectance > 1)):
        raise ValueError(f'{path}: mirror reflectance must be in (0, 1].')
    return reflectance


def bare_quartz_t(wavelengths):
    # Both air/silica interfaces and the incoherent round trips of a lossless plate.
    n = np.array([silica_n(wl).real for wl in wavelengths])
    reflectance = ((n - 1) / (n + 1))**2
    return (1 - reflectance) / (1 + reflectance)


def plot(wavelengths, absolute, referenced, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(7.6, 7.4), sharex=True, constrained_layout=True)
    fig.suptitle('MOSMITE optical spectra')
    marker = 'o' if len(wavelengths) == 1 else None
    labels = (
        ('Reflectance (%)', 'Absolute R', 'Mirror-referenced R'),
        ('Transmittance (%)', 'Absolute T', 'Quartz-referenced T'),
        ('A and comparison (%)', 'Absolute A = 1 - Rabs - Tabs', 'Comparison A = 1 - Rrel - Trel'),
    )
    for ax, abs_values, ref_values, (label, abs_label, ref_label) in zip(axes, absolute, referenced, labels):
        ax.plot(wavelengths, 100 * abs_values, label=abs_label, marker=marker)
        if ref_values is not None:
            ax.plot(wavelengths, 100 * ref_values, '--', label=ref_label, marker=marker)
        ax.set_ylabel(label)
        ax.grid(alpha=.25)
        ax.legend(frameon=False)
    axes[-1].set_xlabel('Wavelength (nm)')
    pending = path.with_suffix('.png.tmp')
    fig.savefig(pending, format='png', dpi=160)
    plt.close(fig)
    pending.replace(path)


def write_referenced_spectrum(output, wavelengths, powers, quartz, mirror):
    r, t = powers.T
    a = 1 - r - t
    if not all(np.all((-2e-6 <= values) & (values <= 1 + 2e-6)) for values in (r, t, a)):
        raise ValueError('Input R and T must be absolute fractions with physical R, T, A.')
    t_rel = t / quartz
    r_rel = r / mirror if mirror is not None else None
    a_compare = 1 - r_rel - t_rel if r_rel is not None else None

    output.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            'wavelength_nm': wl,
            'R_abs_percent': 100 * r[i],
            'T_abs_percent': 100 * t[i],
            'A_abs_percent': 100 * a[i],
            'quartz_abs_T_percent': 100 * quartz[i],
            'T_quartz_reference_percent': 100 * t_rel[i],
            'mirror_abs_R_percent': 100 * mirror[i] if mirror is not None else None,
            'R_mirror_reference_percent': 100 * r_rel[i] if r_rel is not None else None,
            'A_comparison_percent': 100 * a_compare[i] if a_compare is not None else None,
        }
        for i, wl in enumerate(wavelengths)
    ]
    pending = output / 'spectra_referenced.csv.tmp'
    save_csv(pending, rows)
    pending.replace(output / 'spectra_referenced.csv')
    plot(wavelengths, (r, t, a), (r_rel, t_rel, a_compare), output / 'spectra_referenced.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spectrum', type=Path, help='RCWA spectra.csv with wavelength_nm,R,T in fractions.')
    mirror_options = parser.add_mutually_exclusive_group()
    mirror_options.add_argument('--mirror-r-csv', type=Path, help='Calibration data with wavelength_nm,R_percent.')
    mirror_options.add_argument('--mirror-nk-csv', type=Path,
                                help='Material n,k table for a bare, optically thick mirror at normal incidence.')
    parser.add_argument('--quartz-t-csv', type=Path,
                        help='Measured 1 mm quartz blank with wavelength_nm,T_percent. Default: lossless Malitson silica.')
    parser.add_argument('--output-dir', type=Path, help='Default: the directory containing spectra.csv.')
    args = parser.parse_args()

    wavelengths, powers = load_table(args.spectrum, 'R', 'T')
    quartz = (reference_at(args.quartz_t_csv, 'T_percent', wavelengths)
              if args.quartz_t_csv else bare_quartz_t(wavelengths))
    mirror = None
    if args.mirror_r_csv:
        mirror = reference_at(args.mirror_r_csv, 'R_percent', wavelengths)
    elif args.mirror_nk_csv:
        mirror = bare_mirror_r(args.mirror_nk_csv, wavelengths)
    output = args.output_dir or args.spectrum.parent
    write_referenced_spectrum(output, wavelengths, powers, quartz, mirror)
    print(f'Saved {len(wavelengths)} wavelengths: {output.resolve()}')
    if mirror is None:
        print('Mirror-reference and comparison A columns are blank until a mirror CSV is supplied.')
    elif args.mirror_nk_csv:
        print(f'Mirror reference: bare material, normal incidence, {args.mirror_nk_csv.resolve()}')


if __name__ == '__main__':
    main()
