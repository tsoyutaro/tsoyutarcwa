"""Analytic disk material coefficients and periodic three-component normals."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import numpy as np
from scipy.special import j1


@dataclass(frozen=True)
class Geometry:
    period_nm: float = 200.
    height_nm: float = 500.
    tip_radius_nm: float = 5.
    base_radius_nm: float = 95.
    lattice: str = "triangular"
    gil_nm: float = 0.
    gil_rings: int = 12

    def __post_init__(self):
        lengths = (self.period_nm, self.height_nm, self.tip_radius_nm, self.base_radius_nm, self.gil_nm)
        if not all(math.isfinite(x) for x in lengths):
            raise ValueError("Geometry lengths must be finite")
        if self.period_nm <= 0 or self.height_nm <= 0 or not 0 < self.tip_radius_nm <= self.base_radius_nm:
            raise ValueError("Require positive period/height and 0 < tip <= base")
        if self.gil_nm < 0 or self.base_radius_nm + self.gil_nm >= self.period_nm / 2:
            raise ValueError("Require base_radius + GIL < period/2; overlapping cones are not modeled")
        if self.lattice not in ("square", "triangular") or self.gil_rings < 1:
            raise ValueError("Invalid lattice or GIL ring count")

    @property
    def angle(self):
        return math.pi / 3 if self.lattice == "triangular" else math.pi / 2

    @property
    def direct(self):
        return self.period_nm * np.array([[1., math.cos(self.angle)], [0., math.sin(self.angle)]])

    @property
    def reciprocal(self):
        return 2 * np.pi * np.linalg.inv(self.direct).T

    @property
    def area(self):
        return self.period_nm ** 2 * math.sin(self.angle)

    @property
    def slope(self):
        return (self.base_radius_nm - self.tip_radius_nm) / self.height_nm

    def radius(self, z):
        return self.tip_radius_nm + self.slope * float(z)

    def metadata(self):
        return asdict(self)


def harmonics(order: int, lattice="triangular") -> np.ndarray:
    """A D6-closed hexagonal set for triangular cells; square set otherwise."""
    if not isinstance(order, (int, np.integer)) or order < 0:
        raise ValueError("Fourier order must be a nonnegative integer")
    return np.array([(m, n) for m in range(-order, order + 1) for n in range(-order, order + 1)
                     if lattice != "triangular" or max(abs(m), abs(n), abs(m - n)) <= order], dtype=int)


def disk_matrix(radius_nm, differences, geometry):
    """Fourier integral of a centered disk divided by the primitive-cell area."""
    wavevectors = differences @ geometry.reciprocal.T
    argument = np.linalg.norm(wavevectors, axis=-1) * radius_nm
    values = np.ones_like(argument)
    nonzero = abs(argument) > 1e-10
    values[nonzero] = 2 * j1(argument[nonzero]) / argument[nonzero]
    return np.asarray(np.pi * radius_nm ** 2 / geometry.area * values, dtype=complex)


class Operators:
    def __init__(self, geometry: Geometry, orders, grid: int):
        self.geometry, self.orders, self.grid = geometry, orders, grid
        self.difference = orders[:, None, :] - orders[None, :, :]
        maximum_difference = int(abs(self.difference).max())
        if grid < max(16, 2 * maximum_difference + 2):
            raise ValueError(f"Normal grid must be >= {max(16, 2*maximum_difference+2)}")
        self.identity = np.eye(len(orders), dtype=complex)
        u, v = np.meshgrid(np.arange(grid)/grid, np.arange(grid)/grid, indexing="ij")
        x, y = geometry.direct @ np.stack((u.reshape(-1), v.reshape(-1)))
        dx, dy, r2 = [], [], []
        for i in range(-2, 3):
            for j in range(-2, 3):
                center = geometry.direct @ [i, j]
                ax, ay = x - center[0], y - center[1]
                dx.append(ax); dy.append(ay); r2.append(ax*ax + ay*ay)
        r2 = np.array(r2)
        selected = np.argmin(r2, axis=0)
        points = np.arange(grid*grid)
        self.dx = np.array(dx)[selected, points].reshape(grid, grid)
        self.dy = np.array(dy)[selected, points].reshape(grid, grid)
        closest = np.partition(r2, 1, axis=0)[:2]
        self.radius = np.sqrt(closest[0]).reshape(grid, grid)
        self.separation = (np.sqrt(closest[1])-np.sqrt(closest[0])).reshape(grid, grid)

    def material(self, z, epsilon, homogeneous=False):
        if homogeneous:
            return epsilon * self.identity, self.identity / epsilon
        radius = self.geometry.radius(z)
        if self.geometry.gil_nm == 0:
            disk = disk_matrix(radius, self.difference, self.geometry)
            return self.identity + (epsilon-1)*disk, self.identity + (1/epsilon-1)*disk
        # Linear complex n in radial distance, integrated with concentric rings.
        count = self.geometry.gil_rings
        boundaries = radius + np.arange(count+1)*self.geometry.gil_nm/count
        index = np.sqrt(complex(epsilon))
        fractions = (np.arange(count)+.5)/count
        levels = [epsilon, *(((1-fractions)*index+fractions)**2), 1.]
        e, eta = self.identity.copy(), self.identity.copy()
        for j, boundary in enumerate(boundaries):
            disk = disk_matrix(boundary, self.difference, self.geometry)
            e += (levels[j]-levels[j+1])*disk
            eta += (1/levels[j]-1/levels[j+1])*disk
        return e, eta

    def projections(self, z, tilted=True):
        radius = self.geometry.radius(z)
        safe = np.maximum(self.radius, 1e-14)
        slope = self.geometry.slope if tilted else 0.
        normal = np.stack((self.dx/safe, self.dy/safe, np.full_like(safe, -slope)))
        normal /= math.sqrt(1+slope*slope)
        # Smooth dyad continuation away from every physical interface. The
        # mixed dyad at a circle center/Voronoi seam need not have rank one.
        def smooth(value):
            t = np.clip(value, 0, 1)
            return t*t*(3-2*t)
        gap = self.geometry.period_nm-2*(self.geometry.base_radius_nm+self.geometry.gil_nm)
        weight = smooth(self.radius/(radius/2))*smooth(self.separation/gap)
        baseline = (1/3, 1/3, 1/3) if tilted else (.5, .5, 0.)
        result = [[None]*3 for _ in range(3)]
        for a in range(3):
            for b in range(a, 3):
                real = weight*normal[a]*normal[b]
                if a == b:
                    real += (1-weight)*baseline[a]
                coefficients = np.fft.fft2(real)/self.grid**2
                matrix = coefficients[self.difference[...,0] % self.grid,
                                      self.difference[...,1] % self.grid]
                result[a][b] = matrix
                result[b][a] = matrix
        return result
