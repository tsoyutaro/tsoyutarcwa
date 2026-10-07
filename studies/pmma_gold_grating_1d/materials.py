"""Interpolate measured n/k before squaring; PMMA is assumed lossless."""
from __future__ import annotations

import bisect
import csv
import math
from pathlib import Path


def check(rows):
    if (len(rows) < 2 or any(not all(math.isfinite(v) for v in row) for row in rows)
            or any(a[0] >= b[0] for a, b in zip(rows, rows[1:]))):
        raise ValueError("Optical tables need at least two finite, increasing wavelength rows.")


def read_pmma(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["wl", "n"]:
            raise ValueError("PMMA CSV requires columns wl,n (wavelength in micrometres).")
        rows = [(1000*float(r["wl"]), float(r["n"])) for r in reader]
    check(rows)
    if any(n <= 0 for _, n in rows):
        raise ValueError("PMMA indices must be positive.")
    return rows


def read_gold(path):
    blocks, current = {}, None
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            row = [s.strip() for s in row]
            if not row or not any(row):
                continue
            if row in (["wl", "n"], ["wl", "k"]):
                current = row[1]
                if current in blocks:
                    raise ValueError("Duplicate gold CSV block.")
                blocks[current] = []
            elif len(row) == 2 and current is not None:
                blocks[current].append((1000*float(row[0]), float(row[1])))
            else:
                raise ValueError("Gold CSV requires separate wl,n and wl,k blocks.")
    if set(blocks) != {"n", "k"}:
        raise ValueError("Gold n and k blocks are required.")
    for block in blocks.values():
        check(block)
    if len(blocks["n"]) != len(blocks["k"]):
        raise ValueError("Gold n/k wavelength grids differ.")
    rows = []
    for (wn, n), (wk, k) in zip(blocks["n"], blocks["k"]):
        if not math.isclose(wn, wk, rel_tol=0., abs_tol=1e-8) or min(n, k) < 0:
            raise ValueError("Invalid gold n/k rows.")
        rows.append((wn, n, k))
    return rows


def interpolate(rows, wave):
    grid = [r[0] for r in rows]
    if not math.isfinite(wave) or not grid[0] <= wave <= grid[-1]:
        raise ValueError(f"{wave:g} nm is outside CSV range {grid[0]:g}..{grid[-1]:g} nm.")
    right = bisect.bisect_left(grid, wave)
    if grid[right] == wave:
        return rows[right][1:]
    left = right-1
    fraction = (wave-grid[left])/(grid[right]-grid[left])
    return tuple(a+fraction*(b-a) for a, b in zip(rows[left][1:], rows[right][1:]))


class Materials:
    def __init__(self, pmma_path, gold_path, extension):
        self.pmma = read_pmma(pmma_path)
        self.gold = read_gold(gold_path)
        self.note = {"pmma_loss_assumption": "k=0", "interpolation": "linear n,k, then epsilon=(n+ik)^2",
                     "original_pmma_range_nm": [self.pmma[0][0], self.pmma[-1][0]],
                     "pmma_shortwave_extension": extension}
        if extension == "linear_to_400nm" and self.pmma[0][0] > 400.:
            (w0, n0), (w1, n1) = self.pmma[:2]
            if not 400 < w0 <= 405:
                raise ValueError("Short extension requires a PMMA table starting between 400 and 405 nm.")
            n400 = n0+(400-w0)*(n1-n0)/(w1-w0)
            if not math.isfinite(n400) or n400 <= 0:
                raise ValueError("The extrapolated PMMA index must be finite and positive.")
            self.note.update(derived_point_nm_n=[400., n400], source_points_nm_n=[[w0, n0], [w1, n1]],
                             method="short linear edge extrapolation; original CSV unchanged")
            self.pmma = [(400., n400), *self.pmma]
        elif extension not in ("none", "linear_to_400nm"):
            raise ValueError("Unknown PMMA shortwave extension policy.")

    def __call__(self, wave):
        n_pmma, = interpolate(self.pmma, wave)
        n_gold, k_gold = interpolate(self.gold, wave)
        return complex(n_pmma*n_pmma), complex(n_gold, k_gold)**2
