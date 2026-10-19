"""One-dimensional material profiles for Mohamad et al. (2020).

All lengths are in nanometres.  ``sample`` returns relative permittivity,
not refractive index.  The computational propagation direction is downward
for the aluminium example and upward for the substrate-illuminated gold film.
See GEOMETRY_NOTES.md for the conventions needed to interpret the figures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Geometry:
    name: str
    period_nm: float
    depth_nm: float
    n_in: complex
    n_out: complex
    height_nm: float
    film_nm: float = 0.0
    gil_nm: float = 0.0
    n_metal: complex = 0.16 + 5.26j

    def __post_init__(self) -> None:
        if self.name not in {"sinusoidal_aluminium", "triangular_gold"}:
            raise ValueError(f"Unsupported geometry: {self.name!r}")
        if not np.isfinite(self.period_nm) or self.period_nm <= 0:
            raise ValueError("period_nm must be finite and positive")
        for field in ("depth_nm", "height_nm", "film_nm", "gil_nm"):
            value = getattr(self, field)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{field} must be finite and non-negative")
        if self.depth_nm <= 0 or self.height_nm <= 0:
            raise ValueError("depth_nm and height_nm must be positive")
        for field in ("n_in", "n_out", "n_metal"):
            if not np.isfinite(getattr(self, field)):
                raise ValueError(f"{field} must be finite")
        if self.name == "triangular_gold" and self.film_nm <= 0:
            raise ValueError("The gold film must have positive thickness")
        if self.name == "sinusoidal_aluminium" and self.gil_nm:
            raise ValueError("GIL is defined only for the triangular example")

    def surface_nm(self, x: np.ndarray) -> np.ndarray:
        """Return the Al surface depth or the substrate/Au interface height."""
        phase = np.remainder(np.asarray(x, dtype=float), self.period_nm) / self.period_nm
        if self.name == "sinusoidal_aluminium":
            return 0.5 * self.height_nm * (1.0 + np.cos(2.0 * np.pi * phase))
        return self.height_nm * (1.0 - np.abs(2.0 * phase - 1.0))

    def tangent(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the unit tangent (c, s) in the computational (x, z) axes.

        The right derivative is used at the isolated triangular corners.
        The two parallel gold-film interfaces share the same tangent.
        """
        phase = np.remainder(np.asarray(x, dtype=float), self.period_nm) / self.period_nm
        if self.name == "sinusoidal_aluminium":
            slope = -(np.pi * self.height_nm / self.period_nm) * np.sin(2.0 * np.pi * phase)
        else:
            slope = np.where(phase < 0.5, 1.0, -1.0) * (2.0 * self.height_nm / self.period_nm)
        c = 1.0 / np.sqrt(1.0 + slope * slope)
        return c, slope * c

    def sample(self, x: np.ndarray, z_nm: float) -> np.ndarray:
        """Sample relative permittivity at one computational z position.

        For the gold film, a positive GIL width adds a horizontal linear-n
        transition on the dielectric side of each metal/dielectric interface.
        The closest periodic metal boundary is used if transitions overlap.
        """
        if not np.isfinite(z_nm):
            raise ValueError("z_nm must be finite")
        x = np.remainder(np.asarray(x, dtype=float), self.period_nm)
        surface = self.surface_nm(x)
        if self.name == "sinusoidal_aluminium":
            n = np.where(z_nm >= surface, self.n_metal, self.n_in)
            return np.asarray(n * n, dtype=complex)

        below = z_nm < surface
        metal = (~below) & (z_nm <= surface + self.film_nm)
        n = np.where(metal, self.n_metal, np.where(below, self.n_in, self.n_out)).astype(complex)
        if self.gil_nm <= 0:
            return n * n

        # At fixed z the Au set is the inverse image of
        # max(0, z-t) <= f(x) <= min(H, z).  Its four endpoints are
        # available analytically, including intervals meeting at a corner.
        lower = max(0.0, z_nm - self.film_nm)
        upper = min(self.height_nm, z_nm)
        if upper < lower:
            return n * n
        slope = 2.0 * self.height_nm / self.period_nm
        endpoints = (
            lower / slope,
            upper / slope,
            self.period_nm - upper / slope,
            self.period_nm - lower / slope,
        )
        distance = np.full_like(x, np.inf, dtype=float)
        for endpoint in endpoints:
            dx = np.abs(x - endpoint)
            distance = np.minimum(distance, np.minimum(dx, self.period_nm - dx))
        in_gil = (~metal) & (distance < self.gil_nm)
        fraction = np.minimum(distance / self.gil_nm, 1.0)
        dielectric_n = np.where(below, self.n_in, self.n_out)
        graded_n = self.n_metal + fraction * (dielectric_n - self.n_metal)
        n = np.where(in_gil, graded_n, n)
        return np.asarray(n * n, dtype=complex)

    def metadata(self) -> dict:
        """Return JSON-serializable geometry and material assumptions."""
        def parts(value: complex) -> dict[str, float]:
            return {"real": float(np.real(value)), "imag": float(np.imag(value))}

        return {
            "name": self.name,
            "period_nm": self.period_nm,
            "depth_nm": self.depth_nm,
            "height_nm": self.height_nm,
            "film_nm": self.film_nm,
            "gil_nm": self.gil_nm,
            "n_in": parts(self.n_in),
            "n_out": parts(self.n_out),
            "n_metal": parts(self.n_metal),
            "computational_z_direction": "down" if self.name == "sinusoidal_aluminium" else "up",
            "film_thickness_convention": "vertical z offset" if self.film_nm else "semi-infinite metal",
            "gil_convention": "horizontal, dielectric side, linear complex n, nearest periodic boundary",
        }


def sinusoidal_aluminium() -> Geometry:
    """Paper Fig. 4: air/Al sinusoid, period 500 nm, relief 200 nm."""
    metal_n = 1.3 + 7.6j
    return Geometry(
        name="sinusoidal_aluminium",
        period_nm=500.0,
        depth_nm=200.0,
        n_in=1.0 + 0.0j,
        n_out=metal_n,
        height_nm=200.0,
        n_metal=metal_n,
    )


def triangular_gold(
    gil_nm: float = 0.0,
    height_nm: float = 283.2,
    film_nm: float = 50.0,
    n_metal: complex = 0.16 + 5.26j,
) -> Geometry:
    """Paper Fig. 8: triangular corrugated Au film, illuminated from n=1.55."""
    return Geometry(
        name="triangular_gold",
        period_nm=400.0,
        depth_nm=height_nm + film_nm,
        n_in=1.55 + 0.0j,
        n_out=1.0 + 0.0j,
        height_nm=height_nm,
        film_nm=film_nm,
        gil_nm=gil_nm,
        n_metal=n_metal,
    )
