"""Monotone half-widths, following the existing Optimization softmax scheme."""
from __future__ import annotations

import numpy as np


def knots_from_logits(values, nodes, lower, upper):
    """Two independent monotone axes; fix each final logit to remove a gauge.

    There are nodes+1 positive gaps, including the two boundary gaps.
    Values are radii for ellipses and half-widths for rectangles, all in nm.
    """
    values = np.asarray(values, dtype=float).reshape(2, nodes)
    full = np.column_stack([values, np.zeros(2)])
    weights = np.exp(full - full.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    return lower + (upper - lower) * np.cumsum(weights, axis=1)[:, :-1]


def logits_from_knots(knots, lower, upper):
    knots = np.asarray(knots, dtype=float)
    edges = np.column_stack([np.full(2, lower), knots, np.full(2, upper)])
    gaps = np.diff(edges, axis=1)
    if np.any(gaps <= 0):
        raise ValueError("Initial knots must be strictly increasing inside bounds.")
    return np.log(gaps[:, :-1] / gaps[:, -1:]).ravel()


def validate_knots(knots, cfg):
    k = np.asarray(knots, dtype=float)
    if k.ndim != 2 or k.shape[0] != 2 or k.shape[1] < 2:
        raise ValueError("knots_nm must have shape (2, >=2): x axis then y axis.")
    if not np.all(np.isfinite(k)) or np.any(k < cfg.half_min_nm):
        raise ValueError("Half-widths must be finite and above half_min_nm.")
    limits = np.array([cfg.period_x_nm, cfg.period_y_nm])[:, None] / 2
    if np.any(k > cfg.half_max_nm) or np.any(k >= limits):
        raise ValueError("Half-widths exceed bounds or touch neighbouring cells.")
    if np.any(np.diff(k, axis=1) < -1e-12):
        raise ValueError("Both axes must grow monotonically from top to bottom.")
    return k


def slices(knots, cfg):
    knots = validate_knots(knots, cfg)
    z = (np.arange(cfg.layers) + 0.5) / cfg.layers
    nodes_z = np.linspace(0, 1, knots.shape[1])
    return np.array([np.interp(z, nodes_z, axis) for axis in knots]).T


def seeds(cfg):
    """Physical controls and broad, mildly tapered candidate profiles."""
    lo, hi = cfg.half_min_nm, cfg.half_max_nm
    def scaled(xs, ys):
        return lo + (hi - lo) * np.array([xs, ys], dtype=float)
    candidates = [
        scaled([.05, .50, .99], [.05, .25, .45]),
        scaled([.80, .90, .99], [.18, .30, .42]),
        scaled([.50, .75, .99], [.18, .30, .42]),
        scaled([.95, .97, .99], [.28, .30, .32]),
        scaled([.95, .97, .99], [.48, .50, .52]),
        scaled([.05, .50, .99], [.05, .50, .99]),
    ]
    z = np.linspace(0, 1, cfg.nodes)
    return [np.array([np.interp(z, [0, .5, 1], a) for a in k])
            for k in candidates]
