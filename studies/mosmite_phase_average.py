"""Legacy phase quadrature for the uniform MOSMITE layers at normal incidence."""

import numpy as np

from mosmite_optics import _layers, _matrix


PHASE_AVERAGE_THEORY = {
    'method': 'uniform-round-trip-phase-quadrature',
    'implementation': 'Coherent Jones scattering at each combination of equally spaced phases; average reflected and transmitted powers.',
    'assumption': 'Each selected uniform layer has an independently sampled round-trip phase; its material attenuation is fixed.',
}


def _prepend_interface(n_left, n_right, r, t):
    denom = n_left + n_right
    rf = (n_left - n_right) / denom
    tf, tb = 2 * n_left / denom, 2 * n_right / denom
    feedback = 1 + rf * r
    return rf + tb * r * tf / feedback, t * tf / feedback


def _phase_sample(surface, port_index, layers, wl, points):
    count = sum(average for _, _, average in layers) if points > 1 else 0
    phases = iter(reversed(np.meshgrid(
        *([np.arange(points) * np.pi / points] * count), indexing='ij', sparse=True,
    ))) if count else iter(())

    # The tail is isotropic, so calculate its scalar amplitudes before coupling it to RCWA.
    r, t, n_right = 0j, 1 + 0j, 1 + 0j
    for index, depth, average in reversed(layers):
        r, t = _prepend_interface(index, n_right, r, t)
        phase = next(phases) if average and count else 0.
        propagation = np.exp(2j * np.pi * index * depth / wl + 1j * phase)
        r, t, n_right = r * propagation**2, t * propagation, index
    r, t = _prepend_interface(port_index, n_right, r, t)
    r = np.asarray(r).reshape(-1, 1, 1)
    t = np.asarray(t).reshape(-1, 1, 1)

    tf, rf, rb, tb = surface
    feedback = np.eye(2) - r * rb
    forward = np.linalg.solve(feedback, np.broadcast_to(tf, feedback.shape))
    reflected = np.mean(np.abs(rf + r * (tb @ forward))**2, axis=0)
    transmitted = np.mean(np.abs(t * forward)**2, axis=0)
    rx, ry = map(float, reflected.sum(axis=0))
    tx, ty = map(float, transmitted.sum(axis=0))
    R, T = (rx + ry) / 2, (tx + ty) / 2
    return {
        'R': R, 'T': T, 'A': 1 - R - T,
        'R_x': rx, 'T_x': tx, 'R_y': ry, 'T_y': ty,
        'cross_R': float((reflected[0, 1] + reflected[1, 0]) / 2),
        'cross_T': float((transmitted[0, 1] + transmitted[1, 0]) / 2),
    }


def phase_average(surface, port_index, layers, wl, *, points, tolerance):
    """Return legacy powers and their maximum change on doubling the phase grid."""
    if points < 1:
        raise ValueError('Phase points must be positive.')
    layers = _layers(layers, wl)
    port_index = complex(port_index)
    if not np.isfinite(port_index) or port_index.real <= 0 or port_index.imag < 0:
        raise ValueError('Invalid substrate port.')
    surface = tuple(_matrix(value) for value in surface)
    result = _phase_sample(surface, port_index, layers, wl, points)
    error = 0.
    if points > 1 and any(average for _, _, average in layers):
        refined = _phase_sample(surface, port_index, layers, wl, 2 * points)
        error = max(abs(result[key] - refined[key])
                    for key in ('R', 'T', 'A', 'R_x', 'T_x', 'R_y', 'T_y'))
        if not np.isfinite(error) or error > tolerance:
            raise RuntimeError(f'Phase average not converged at {wl:g} nm: {error:.6g}; increase --phase-points.')
    return result, error
