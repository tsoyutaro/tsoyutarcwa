"""Downward PMMA relief, directional Au coating, and local optical tables (nm)."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent


def solver_root(explicit=None):
    candidates = ([Path(explicit)] if explicit else [])
    if os.environ.get('RCWA_ROOT'):
        candidates.append(Path(os.environ['RCWA_ROOT']))
    candidates += list(HERE.parents)
    installation = HERE/'installation.json'
    if installation.exists():
        candidates.append(Path(json.loads(installation.read_text(encoding='utf-8'))['solver_root']))
    for root in candidates:
        if (root/'rcwa_ext').is_dir() and (root/'studies/gold_grating_1d/solver.py').is_file():
            sys.path.insert(0, str(root))
            return root.resolve()
    raise FileNotFoundError('Use --solver-root PATH or RCWA_ROOT; require rcwa_ext and studies/gold_grating_1d/solver.py.')


def supplied_pmma(wavelength):
    x = (float(wavelength)/1000.)**2
    n2 = 1 + .99654*x/(x-.00787) + .18964*x/(x-.02191) + .00411*x/(x-3.85727)
    if not np.isfinite(n2) or n2 <= 0:
        raise ValueError('Invalid wavelength for the supplied PMMA formula.')
    return math.sqrt(n2)


class Materials:
    def __init__(self):
        pmma, au = HERE/'data/Szczurowski.csv', HERE/'data/Au_Johnson_1972.csv'
        self.pmma_path = pmma if pmma.is_file() else None
        self.au_path = au if au.is_file() else None
        self.pmma_table = None
        if self.pmma_path:
            with pmma.open(encoding='utf-8-sig', newline='') as f:
                self.pmma_table = np.array([(1000*float(r['wl']), float(r['n'])) for r in csv.DictReader(f)])
            if np.any(np.diff(self.pmma_table[:, 0]) <= 0):
                raise ValueError('PMMA wavelengths must increase.')
        self.gold_table = None
        if self.au_path:
            with au.open(encoding='utf-8-sig', newline='') as f:
                self.gold_table = np.array([(float(r['wavelength_nm']), float(r['n']), float(r['k'])) for r in csv.DictReader(f)])
            if np.any(np.diff(self.gold_table[:, 0]) <= 0):
                raise ValueError('Au wavelengths must increase.')

    def pmma(self, wavelength):
        w = float(wavelength)
        if self.pmma_table is None:
            if not 400 <= w <= 700:
                raise ValueError('Supplied-formula fallback is enabled only in the requested 400..700 nm band.')
            return supplied_pmma(w), 'supplied_formula_no_csv'
        table = self.pmma_table
        if 400 <= w < table[0, 0] and table[0, 0] <= 405:
            n = np.interp(w, [400., table[0, 0]], [supplied_pmma(400.), table[0, 1]])
            return float(n), 'bridge_formula_at_400_to_csv_at_404.7'
        if not table[0, 0] <= w <= table[-1, 0]:
            raise ValueError(f'{w} nm outside PMMA CSV range; no extrapolation.')
        return float(np.interp(w, table[:, 0], table[:, 1])), 'Szczurowski_csv'

    def gold(self, wavelength):
        if self.gold_table is None:
            raise FileNotFoundError('Gold calculations require data/Au_Johnson_1972.csv.')
        w, table = float(wavelength), self.gold_table
        if not table[0, 0] <= w <= table[-1, 0]:
            raise ValueError('Wavelength outside Au CSV range; no extrapolation.')
        return complex(np.interp(w, table[:, 0], table[:, 1]), np.interp(w, table[:, 0], table[:, 2]))

    def metadata(self):
        return {name: {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                for name, path in [('pmma', self.pmma_path), ('gold', self.au_path)] if path}


def geometry(period=3720., base=None, tip=None, height=None, gold=0.):
    base = period if base is None else base
    tip = period*25/62 if tip is None else tip
    height = period*25/62 if height is None else height
    values = np.array([period, base, tip, height, gold])
    if not np.all(np.isfinite(values)) or not 0 < tip < base <= period or height <= 0 or gold < 0:
        raise ValueError('Require finite 0 < tip < base <= period, height > 0, gold >= 0.')
    run = (base-tip)/2
    return dict(period_nm=float(period), base_width_nm=float(base), tip_width_nm=float(tip),
                height_nm=float(height), gold_vertical_nm=float(gold), gap_nm=float(period-base),
                slope_deg=math.degrees(math.atan2(height, run)),
                gold_normal_nm=float(gold*run/math.hypot(height, run)),
                gold_horizontal_side_offset_nm=float(gold*run/height))


def geometry_for_args(args, period, coating):
    gold = args.gold_nm if coating == 'gold' else 0.
    if args.geometry_mode == 'scaled':
        if any(v is not None for v in (args.base_width, args.tip_width, args.height)):
            raise ValueError('Scaled geometry fixes base=period, tip=height=period*15/37.2. '
                             'Remove width/height flags, or use --geometry-mode legacy for old cases.')
        return geometry(period=period, gold=gold)
    return geometry(period, 3720. if args.base_width is None else args.base_width,
                    1500. if args.tip_width is None else args.tip_width,
                    1500. if args.height is None else args.height, gold)


def width(z, g):
    if z < 0:
        return g['period_nm']
    if z > g['height_nm']:
        return 0.
    return g['base_width_nm']-(g['base_width_nm']-g['tip_width_nm'])*z/g['height_nm']


def layers(g, slices):
    if not isinstance(slices, int) or slices < 1:
        raise ValueError('slices must be a positive integer.')
    h, t = g['height_nm'], g['gold_vertical_nm']
    # Match the valley and tip film boundaries exactly; slices controls maximum dz=h/slices.
    breaks = sorted(set([0., t, h, h+t]))
    result = []
    for left, right in zip(breaks, breaks[1:]):
        count = max(1, int(math.ceil((right-left)/(h/slices)-1e-12)))
        edges = np.linspace(left, right, count+1)
        for a, b in zip(edges, edges[1:]):
            z = (a+b)/2
            core = width(z, g)
            outer = width(z-t, g) if t else core
            result.append(dict(z_top_nm=float(a), z_bottom_nm=float(b),
                               core_width_nm=core, outer_width_nm=outer))
    return result


def save_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def parse_wavelengths(raw):
    if ':' in raw:
        start, stop, step = map(float, raw.split(':'))
        if step <= 0 or stop < start:
            raise ValueError('Use start:stop:positive_step.')
        values = np.arange(start, stop+step*1e-7, step)
    else:
        values = np.array([float(w) for w in raw.split(',')])
    if not len(values) or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError('Wavelengths must be positive and finite.')
    return np.unique(values)
