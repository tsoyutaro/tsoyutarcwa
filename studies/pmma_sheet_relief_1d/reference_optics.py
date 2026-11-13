"""Angle-resolved flux and thick-sheet phase averaging, retaining +/-m degeneracy."""
from __future__ import annotations

import math
import numpy as np
from scipy.stats import qmc


def admittance(index, kx, pol):
    q = np.sqrt(complex(index)**2-np.asarray(kx)**2+0j)
    safe = np.where(abs(q) < 1e-12, 1e-12+1e-12j, q)
    return q if pol == 's' else complex(index)**2/safe


def propagating(index, kx):
    return abs(np.asarray(kx)) < float(index)-1e-10


def angles(index, kx):
    values = np.asarray(kx)/index
    return [float(np.degrees(np.arcsin(v))) if abs(v) < 1-1e-10 else None for v in values]


def even_basis(orders, mask):
    selected = np.asarray(orders)[mask]
    groups = sorted(set(abs(selected)))
    u = np.zeros((len(selected), len(groups)))
    for j, m in enumerate(groups):
        u[abs(selected) == m, j] = 1 if m == 0 else 1/math.sqrt(2)
    return selected, np.array(groups), u


def phase_sample(rf, tf, orders, wavelength, period, n_pmma, n_out, pol, phases):
    """Return R_m/T_m for external normal incidence, averaged over supplied sheet phases.

    Tangential electric amplitudes: s=Ey, p=Ex. Each |m| shares a single
    one-way phase between +/-m. Evanescent waves do not cross a thick sheet.
    """
    orders = np.asarray(orders)
    kx = orders*wavelength/period
    mask = propagating(n_pmma, kx)
    selected, groups, u = even_basis(orders, mask)
    if phases.shape[1] != len(groups):
        raise ValueError('One phase variable per propagating |m| group is required.')
    r_even = u.T @ rf[np.ix_(mask, mask)] @ u
    t_even = tf[:, mask] @ u
    group_kx = groups*wavelength/period
    yp = admittance(n_pmma, group_kx, pol)
    ya = admittance(1., group_kx, pol)
    rt = (yp-ya)/(yp+ya)
    tout = 2*yp/(yp+ya)
    yin = float(admittance(1., np.array([0.]), pol)[0].real)
    ypmma0 = float(yp[0].real)
    rin = (yin-ypmma0)/(yin+ypmma0)
    tin = 2*yin/(yin+ypmma0)
    p = np.exp(1j*phases)
    feedback = p*p*rt
    matrix = np.eye(len(groups))[None, :, :]-feedback[:, :, None]*r_even[None, :, :]
    rhs = np.zeros((len(p), len(groups), 1), dtype=complex)
    rhs[:, 0, 0] = p[:, 0]*tin
    down = np.linalg.solve(matrix, rhs)[..., 0]
    up = down @ r_even.T
    external_even = p*up*tout
    external_even[:, 0] += rin
    external = external_even @ u.T
    transmitted = down @ t_even.T
    ya_all = admittance(1., kx[mask], pol)
    yt = admittance(n_out, kx, pol)
    r = np.zeros(len(orders))
    r[mask] = np.mean(abs(external)**2, axis=0)*np.where(propagating(1., kx[mask]), ya_all.real, 0)/yin
    t = np.mean(abs(transmitted)**2, axis=0)*np.where(propagating(n_out, kx), yt.real, 0)/yin
    return r, t


def average_sheet(rf, tf, orders, wavelength, period, n_pmma, n_out, pol,
                  points=512, max_points=16384, tolerance=2e-4, seed=1729):
    if points < 16 or points & (points-1) or max_points < 2*points or max_points & (max_points-1):
        raise ValueError('phase points/max must be powers of two; points >=16, max >=2*points.')
    kx = np.asarray(orders)*wavelength/period
    dimensions = len(set(abs(np.asarray(orders)[propagating(n_pmma, kx)])))
    engines = [qmc.Sobol(dimensions, scramble=True, seed=seed+s) for s in (0, 7919)]
    sums = [np.zeros((2, len(orders))) for _ in engines]
    n, previous, error = 0, None, float('inf')
    target = points
    while target <= max_points:
        for i, engine in enumerate(engines):
            # A single nested Sobol sequence per scramble; bounded solve batches.
            count = target-n
            remaining = count
            while remaining:
                chunk = min(256, remaining)
                phases = engine.random(chunk)*2*np.pi
                result = np.array(phase_sample(rf, tf, orders, wavelength, period, n_pmma, n_out, pol, phases))
                sums[i] += chunk*result
                remaining -= chunk
        means = np.array(sums)/target
        current = means.mean(axis=0)
        difference = means[0]-means[1]
        # Check each order, totals, absorption and the measured reflection cone.
        cone = abs(kx) <= np.sin(np.radians(20))+1e-12
        def metrics(a):
            return np.r_[a.ravel(), a[0].sum(), a[1].sum(), 1-a.sum(), a[0, cone].sum()]
        between = float(np.max(abs(metrics(difference)-metrics(np.zeros_like(difference)))))
        if previous is not None:
            error = max(between, float(np.max(abs(metrics(current)-metrics(previous)))))
            if error <= tolerance:
                return current[0], current[1], dict(points_per_scramble=target, scrambles=2,
                    maximum_change=error, tolerance=tolerance, converged=True,
                    phase_dimensions=dimensions, retained_degeneracy='same phase for +m and -m', seed=seed)
        previous, n, target = current, target, 2*target
    return current[0], current[1], dict(points_per_scramble=n, scrambles=2,
        maximum_change=error, tolerance=tolerance, converged=False,
        phase_dimensions=dimensions, retained_degeneracy='same phase for +m and -m', seed=seed)


def order_rows(r, t, orders, wavelength, period, n_pmma, n_out, **labels):
    kx = np.asarray(orders)*wavelength/period
    ar, at, ap = angles(1., kx), angles(n_out, kx), angles(n_pmma, kx)
    return [dict(labels, wavelength_nm=float(wavelength), m=int(m), R_m=float(rv), T_m=float(tv),
                 theta_R_air_deg=ar[i], theta_T_output_deg=at[i], theta_PMMA_deg=ap[i],
                 propagating_R_air=bool(propagating(1., [kx[i]])[0]),
                 propagating_T_output=bool(propagating(n_out, [kx[i]])[0]),
                 R_within_20deg=bool(ar[i] is not None and abs(ar[i]) <= 20+1e-10))
            for i, (m, rv, tv) in enumerate(zip(orders, r, t))]
