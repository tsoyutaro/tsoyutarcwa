"""Load passive optical constants, including refractiveindex.info split CSVs."""

from __future__ import annotations

import bisect
import csv
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TabulatedNK:
    wavelength_nm: tuple[float, ...]
    n: tuple[float, ...]
    k: tuple[float, ...]
    source: str

    def __post_init__(self):
        if len(self.wavelength_nm) < 2 or not (
            len(self.wavelength_nm) == len(self.n) == len(self.k)
        ):
            raise ValueError("At least two matching wavelength/n/k samples are required.")
        if any(not math.isfinite(w) or w <= 0 for w in self.wavelength_nm):
            raise ValueError("Wavelengths must be finite and positive.")
        if any(b <= a for a, b in zip(self.wavelength_nm, self.wavelength_nm[1:])):
            raise ValueError("Wavelengths must be strictly increasing.")
        if any(not math.isfinite(v) or v <= 0 for v in self.n):
            raise ValueError("Refractive indices must be finite and positive.")
        if any(not math.isfinite(v) or v < 0 for v in self.k):
            raise ValueError("Extinction coefficients must be finite and nonnegative.")

    def __call__(self, wavelength_nm: float) -> complex:
        """Interpolate epsilon linearly, as in the existing Au CSV implementation."""
        w = float(wavelength_nm)
        if not math.isfinite(w) or not self.wavelength_nm[0] <= w <= self.wavelength_nm[-1]:
            raise ValueError(
                f"{w} nm is outside the tabulated range "
                f"[{self.wavelength_nm[0]}, {self.wavelength_nm[-1]}] nm."
            )
        right = bisect.bisect_left(self.wavelength_nm, w)
        er = complex(self.n[right], self.k[right]) ** 2
        if self.wavelength_nm[right] == w:
            return er
        left = right - 1
        el = complex(self.n[left], self.k[left]) ** 2
        f = (w - self.wavelength_nm[left]) / (self.wavelength_nm[right] - self.wavelength_nm[left])
        return el + f * (er - el)


def load_tabulated_nk(path: str | Path) -> TabulatedNK:
    """Accept wavelength_nm,n,k or two wl,n / wl,k blocks (wl in micrometres).

    Units follow the headers, rather than being guessed from numerical values.
    Split n and k blocks must have identical wavelength grids.
    """
    source = Path(path)
    blocks: dict[str, dict[float, float]] = {"n": {}, "k": {}}
    header = None
    scale = 1.0
    with source.open(encoding="utf-8-sig", newline="") as handle:
        for line_number, raw in enumerate(csv.reader(handle), 1):
            row = [v.strip() for v in raw]
            if not row or not any(row):
                continue
            names = [v.lower() for v in row]
            if names[0] in {"wl", "wavelength_um", "wavelength_nm"}:
                header = names
                scale = 1000.0 if names[0] in {"wl", "wavelength_um"} else 1.0
                if len(set(names)) != len(names) or set(names[1:]) not in ({"n"}, {"k"}, {"n", "k"}):
                    raise ValueError(f"Invalid n/k CSV header on line {line_number}.")
                continue
            if header is None or len(row) != len(header):
                raise ValueError(f"Missing header or wrong column count on line {line_number}.")
            values = dict(zip(header, row))
            w = float(row[0]) * scale
            for component in header[1:]:
                if w in blocks[component]:
                    raise ValueError(f"Duplicate {component} wavelength on line {line_number}.")
                blocks[component][w] = float(values[component])
    if set(blocks["n"]) != set(blocks["k"]) or len(blocks["n"]) < 2:
        raise ValueError("n and k need identical wavelength grids with at least two samples.")
    wavelengths = tuple(sorted(blocks["n"]))
    return TabulatedNK(wavelengths, tuple(blocks["n"][w] for w in wavelengths),
                       tuple(blocks["k"][w] for w in wavelengths), str(source.resolve()))
