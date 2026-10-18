"""Normal-incidence optical data and amplitude connection of thick layers."""
from pathlib import Path
import csv
import hashlib
import math
import numpy as np


def load_nk(path):
    path = Path(path).resolve()
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        key = 'wavelength_nm' if 'wavelength_nm' in fields else 'wavelength_um'
        if key not in fields or not {'n', 'k'}.issubset(fields):
            raise ValueError(f'{path}: require wavelength_nm,n,k or wavelength_um,n,k.')
        factor = 1. if key == 'wavelength_nm' else 1000.
        data = np.array([[factor*float(r[key]), float(r['n']), float(r['k'])] for r in reader])
    if (data.ndim != 2 or len(data) < 2 or not np.all(np.isfinite(data))
            or np.any(data[:, 1] <= 0) or np.any(data[:, 2] < 0)):
        raise ValueError(f'Invalid passive optical data: {path}')
    data = data[np.argsort(data[:, 0])]
    if np.any(np.diff(data[:, 0]) <= 0):
        raise ValueError('Duplicate wavelengths in optical data.')
    def evaluate(wl):
        if not data[0, 0]-1e-8 <= wl <= data[-1, 0]+1e-8:
            raise ValueError(f'{wl} nm outside {path}; extrapolation disabled.')
        return complex(np.interp(wl, data[:, 0], data[:, 1]),
                       np.interp(wl, data[:, 0], data[:, 2]))
    return evaluate, {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'range_nm': [float(data[0, 0]), float(data[-1, 0])],
        'interpolation': 'linear n,k separately; no extrapolation'}


def silica_n(wl):
    if not 210 <= wl <= 3710:
        raise ValueError('Malitson fused-silica model requires 210--3710 nm.')
    q = (wl/1000.)**2
    return complex(math.sqrt(1 + .6961663*q/(q-.0684043**2)
        + .4079426*q/(q-.1162414**2) + .8974794*q/(q-9.896161**2)), 0.)


def wavelengths(text):
    if ':' in text:
        start, stop, step = map(float, text.split(':'))
        if stop < start or step <= 0:
            raise ValueError('Invalid wavelength range.')
        values = np.arange(start, stop+step*1e-7, step)
    else:
        values = np.array([float(v) for v in text.split(',')])
    if len(values) == 0 or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError('Wavelengths must be finite and positive.')
    return np.unique(values)


def interface(nl, nr):
    r = (nl-nr)/(nl+nr)
    return 2*nl/(nl+nr), r, -r, 2*nr/(nl+nr)


def tail_amplitudes(port_index, layers, wl, phase_points):
    count = sum(bool(x[2]) for x in layers) if phase_points > 1 else 0
    phases = [x.ravel() for x in np.meshgrid(
        *([np.arange(phase_points)*np.pi/phase_points]*count), indexing='ij')] if count else []
    r = np.zeros(phase_points**count, dtype=np.complex128)
    t = np.ones_like(r)
    nr, slot = 1.+0j, count-1
    for index, thickness, average in reversed(layers):
        tf, rf, rb, tb = interface(index, nr)
        den = 1-rb*r
        r, t = rf+tb*r*tf/den, t*tf/den
        phase = phases[slot] if average and count else 0.
        if average and count:
            slot -= 1
        p = np.exp(2j*np.pi*index*thickness/wl+1j*phase)
        r, t, nr = r*p*p, t*p, index
    tf, rf, rb, tb = interface(port_index, nr)
    den = 1-rb*r
    return rf+tb*r*tf/den, t*tf/den


def finite_sample(surface, port_index, layers, wl, phase_points):
    """Four Jones S blocks, (tf,rf,rb,tb); unpolarized power in external air."""
    tf, rf, rb, tb = [np.asarray(s, dtype=np.complex128) for s in surface]
    r_tail, t_tail = tail_amplitudes(port_index, layers, wl, phase_points)
    den = np.eye(2)[None, :, :] - r_tail[:, None, None]*rb[None, :, :]
    forward = np.linalg.solve(den, np.broadcast_to(tf, den.shape))
    r = rf[None, :, :] + r_tail[:, None, None]*(tb[None, :, :]@forward)
    t = t_tail[:, None, None]*forward
    rx, ry = np.mean(np.sum(np.abs(r)**2, axis=1), axis=0)
    tx, ty = np.mean(np.sum(np.abs(t)**2, axis=1), axis=0)
    result = {'R': float((rx+ry)/2), 'T': float((tx+ty)/2),
              'R_x': float(rx), 'T_x': float(tx), 'R_y': float(ry), 'T_y': float(ty)}
    result['A'] = 1-result['R']-result['T']
    result['cross_R'] = float(np.mean((np.abs(r[:, 0, 1])**2+np.abs(r[:, 1, 0])**2)/2))
    result['cross_T'] = float(np.mean((np.abs(t[:, 0, 1])**2+np.abs(t[:, 1, 0])**2)/2))
    return result


def save_csv(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
