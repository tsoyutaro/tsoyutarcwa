"""Public problem specifications and scalar/autograd utilities."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
import torcwa

_TWO_PI = 2.0 * math.pi
if not hasattr(torcwa, '_codex_original_rcwa'):
    torcwa._codex_original_rcwa = torcwa.rcwa
_ORIGINAL_TORCWA_RCWA = torcwa._codex_original_rcwa

class UnsupportedCombinationError(ValueError):
    """Raised when an explicitly requested RCWA formulation is ineligible."""


def _finite_positive(name: str, value: object) -> float:
    result = _as_float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive; got {value!r}.")
    return result


def _normalize_method(value: str) -> str:
    normalized = str(value).strip().lower().replace("_", "-")
    aliases = {
        "auto": "auto",
        "standard": "standard",
        "torcwa": "standard",
        "direct": "standard",
        "asr": "asr-fr",
        "asr-fr": "asr-fr",
        "circle-asr": "matched-asr",
        "matched": "matched-asr",
        "matched-asr": "matched-asr",
        "matched-coordinate": "matched-asr",
        "matched-coordinate-asr": "matched-asr",
        "nvm": "nvm",
        "normal-vector": "nvm",
        "normal-vector-method": "nvm",
    }
    if normalized not in aliases:
        raise ValueError(
            "method must be one of 'auto', 'standard', 'asr-fr', "
            "'matched-asr', or 'nvm'."
        )
    return aliases[normalized]


def _normalize_cascade(value: str) -> str:
    normalized = str(value).strip().lower().replace("_", "-")
    if normalized == "alogo2a":
        warnings.warn(
            "'alogo2a' is accepted as a deprecated spelling of 'algo2a'.",
            DeprecationWarning,
            stacklevel=3,
        )
        normalized = "algo2a"
    if normalized in {
        "redheffer",
        "star",
        "torcwa",
        "s-matrix",
        "normal",
        "standard",
        "usual",
    }:
        return "redheffer"
    if normalized in {"algo2a", "algorithm-2a", "li2a", "li-2a"}:
        return "li-2a"
    raise ValueError("cascade must be 'redheffer' or 'li-2a' ('algo2a').")


def _normalize_smatrix_size(value: str) -> str:
    normalized = str(value).strip().lower().replace("_", "-")
    aliases = {
        "full": "full",
        "normal": "full",
        "standard": "full",
        "four": "full",
        "4": "full",
        "half": "half",
        "forward": "half",
        "two": "half",
        "2": "half",
        "quarter": "quarter",
        "reflection": "quarter",
        "one": "quarter",
        "1": "quarter",
    }
    if normalized not in aliases:
        raise ValueError("smatrix_size must be 'full', 'half', or 'quarter'.")
    return aliases[normalized]


def _normalize_group_symmetry(value: str) -> str:
    normalized = str(value).strip().lower().replace("_", "-")
    aliases = {
        "auto": "auto",
        "c2": "c2",
        "inversion": "c2",
        "c2v": "c2v",
        "rectangular": "c2v",
        "d6": "d6",
        "c6v": "d6",
        "hexagonal": "d6",
    }
    if normalized not in aliases:
        raise ValueError(
            "group-theory symmetry must be 'auto', 'c2v', 'c2', or 'd6'."
        )
    return aliases[normalized]


def _normalize_polarization(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("_", "-")
    aliases = {
        "x": "x",
        "ex": "x",
        "linear-x": "x",
        "y": "y",
        "ey": "y",
        "linear-y": "y",
    }
    if normalized not in aliases:
        raise ValueError(
            "polarization reduction currently supports 'x' or 'y' only."
        )
    return aliases[normalized]


@dataclass(frozen=True)
class Lattice:
    """Two-dimensional primitive cell, expressed in a right-handed basis."""

    a1: tuple[float, float]
    a2: tuple[float, float]
    kind: str = "auto"

    def __post_init__(self) -> None:
        a1 = tuple(float(v) for v in self.a1)
        a2 = tuple(float(v) for v in self.a2)
        if len(a1) != 2 or len(a2) != 2:
            raise ValueError("a1 and a2 must each contain two components.")
        if not all(math.isfinite(v) for v in (*a1, *a2)):
            raise ValueError("Lattice vectors must be finite.")
        determinant = a1[0] * a2[1] - a1[1] * a2[0]
        if determinant <= 0.0:
            raise ValueError("Lattice vectors must form a nondegenerate right-handed cell.")
        object.__setattr__(self, "a1", a1)
        object.__setattr__(self, "a2", a2)

        requested = self.kind.strip().lower().replace("_", "-")
        if requested in {"hexagonal", "hex", "hexagonal-2d"}:
            requested = "triangular"
        detected = self._detect_kind()
        if requested == "auto":
            requested = detected
        if requested not in {"square", "rectangular", "triangular", "oblique"}:
            raise ValueError(
                "lattice kind must be square, rectangular, triangular/hexagonal, "
                "oblique, or auto."
            )
        if requested != "oblique" and requested != detected:
            raise ValueError(
                f"Lattice vectors describe {detected!r}, not requested {requested!r}."
            )
        object.__setattr__(self, "kind", requested)

    @classmethod
    def rectangular(cls, lx: float, ly: float) -> "Lattice":
        lx = _finite_positive("lx", lx)
        ly = _finite_positive("ly", ly)
        return cls((lx, 0.0), (0.0, ly), "auto")

    @classmethod
    def square(cls, period: float) -> "Lattice":
        period = _finite_positive("period", period)
        return cls((period, 0.0), (0.0, period), "square")

    @classmethod
    def triangular(cls, period: float) -> "Lattice":
        """2-D triangular Bravais lattice (often called a hexagonal lattice)."""
        period = _finite_positive("period", period)
        return cls(
            (period, 0.0),
            (0.5 * period, 0.5 * math.sqrt(3.0) * period),
            "triangular",
        )

    @classmethod
    def hexagonal(cls, period: float) -> "Lattice":
        """Alias for :meth:`triangular`; this is not a 3-D HCP stack."""
        return cls.triangular(period)

    @classmethod
    def hexagonal_close_packed(cls, period: float) -> "Lattice":
        """2-D close-packed disk lattice (a triangular Bravais lattice).

        This names the in-plane packing explicitly.  It does not create a
        three-dimensional ABAB hexagonal-close-packed crystal.
        """
        return cls.triangular(period)

    @classmethod
    def oblique(
        cls, a1: Sequence[float], a2: Sequence[float]
    ) -> "Lattice":
        return cls(tuple(a1), tuple(a2), "auto")

    @property
    def length1(self) -> float:
        return math.hypot(*self.a1)

    @property
    def length2(self) -> float:
        return math.hypot(*self.a2)

    @property
    def angle_deg(self) -> float:
        dot = self.a1[0] * self.a2[0] + self.a1[1] * self.a2[1]
        cosine = max(-1.0, min(1.0, dot / (self.length1 * self.length2)))
        return math.degrees(math.acos(cosine))

    @property
    def basis_rotation_deg(self) -> float:
        return math.degrees(math.atan2(self.a1[1], self.a1[0]))

    @property
    def is_orthogonal(self) -> bool:
        return abs(self.angle_deg - 90.0) <= 1.0e-7

    @property
    def local_vectors(self) -> tuple[tuple[float, float], tuple[float, float]]:
        angle = math.radians(self.angle_deg)
        return (
            (self.length1, 0.0),
            (self.length2 * math.cos(angle), self.length2 * math.sin(angle)),
        )

    def _detect_kind(self) -> str:
        length_equal = math.isclose(
            self.length1, self.length2, rel_tol=1.0e-7, abs_tol=1.0e-10
        )
        angle = self.angle_deg
        if abs(angle - 90.0) <= 1.0e-7:
            return "square" if length_equal else "rectangular"
        if length_equal and (
            abs(angle - 60.0) <= 1.0e-7 or abs(angle - 120.0) <= 1.0e-7
        ):
            return "triangular"
        return "oblique"


@dataclass(frozen=True)
class Material:
    eps: object
    mu: object = 1.0


@dataclass(frozen=True)
class Rectangle:
    """Axis-aligned physical size with a fractional-cell center by default."""

    size: tuple[float, float]
    center: tuple[float, float] = (0.5, 0.5)
    angle_deg: float = 0.0
    coordinates: str = "fractional"

    def __post_init__(self) -> None:
        size = tuple(float(v) for v in self.size)
        center = tuple(float(v) for v in self.center)
        if len(size) != 2 or len(center) != 2:
            raise ValueError("Rectangle size and center must have two components.")
        if not all(math.isfinite(v) for v in (*size, *center, self.angle_deg)):
            raise ValueError("Rectangle parameters must be finite.")
        if size[0] <= 0.0 or size[1] <= 0.0:
            raise ValueError("Rectangle dimensions must be positive.")
        coordinates = self.coordinates.strip().lower()
        if coordinates not in {"fractional", "cartesian"}:
            raise ValueError("coordinates must be 'fractional' or 'cartesian'.")
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "coordinates", coordinates)


@dataclass(frozen=True)
class Square:
    side: float
    center: tuple[float, float] = (0.5, 0.5)
    angle_deg: float = 0.0
    coordinates: str = "fractional"

    def as_rectangle(self) -> Rectangle:
        side = _finite_positive("side", self.side)
        return Rectangle(
            (side, side), self.center, self.angle_deg, self.coordinates
        )


@dataclass(frozen=True)
class Circle:
    radius: object
    center: tuple[float, float] = (0.5, 0.5)
    coordinates: str = "fractional"

    def __post_init__(self) -> None:
        # Validation deliberately uses a detached scalar, but the value stored
        # on the geometry must remain the caller's Tensor.  Converting it with
        # float(...) here would sever the radius -> Fourier/Bessel/metric graph.
        radius_value = _finite_positive("Circle radius", self.radius)
        radius = self.radius if torch.is_tensor(self.radius) else radius_value
        center = tuple(float(v) for v in self.center)
        if len(center) != 2 or not all(math.isfinite(v) for v in center):
            raise ValueError("Circle center must contain two finite values.")
        coordinates = self.coordinates.strip().lower()
        if coordinates not in {"fractional", "cartesian"}:
            raise ValueError("coordinates must be 'fractional' or 'cartesian'.")
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "coordinates", coordinates)

    @classmethod
    def close_packed(
        cls,
        period: float,
        *,
        gap: float = 0.0,
        center: tuple[float, float] = (0.5, 0.5),
    ) -> "Circle":
        """Circle for a 2-D close-packed triangular lattice.

        Use together with :meth:`Lattice.hexagonal_close_packed`.  ``gap`` is
        the nearest-neighbour edge-to-edge separation; zero gives touching
        disks and a positive value is normally preferable numerically.
        """
        period_value = _finite_positive("period", period)
        gap_value = float(gap)
        if not math.isfinite(gap_value) or not 0.0 <= gap_value < period_value:
            raise ValueError("gap must be finite and satisfy 0 <= gap < period.")
        return cls(0.5 * (period_value - gap_value), center=center)


@dataclass(frozen=True)
class Raster:
    eps: torch.Tensor
    mu: object = 1.0
    shape_hint: str | None = None


@dataclass(frozen=True)
class Homogeneous:
    material: Material


Geometry = Rectangle | Square | Circle | Raster | Homogeneous


@dataclass(frozen=True)
class LayerSpec:
    thickness: object
    geometry: Geometry
    background: Material = field(default_factory=lambda: Material(1.0, 1.0))
    inclusion: Material = field(default_factory=lambda: Material(1.0, 1.0))
    method: str = "auto"
    factorization_rules: bool | None = None
    label: str | None = None


@dataclass(frozen=True)
class ASROptions:
    G: float = 1.0e-3
    circle_G: float = 3.0e-2
    grid: tuple[int, int] = (256, 256)
    factorization_rules: bool = True


@dataclass(frozen=True)
class NVMOptions:
    grid: tuple[int, int] = (256, 256)
    use_lanczos: bool = False
    lanczos_power: int = 2


@dataclass(frozen=True)
class GroupTheoryOptions:
    """Optional symmetry reduction for centered circular layers.

    For a full NVM eigensolve, ``auto`` selects C2v in orthogonal cells and C2
    in triangular/oblique cells.  Source-specific ``polarization="x"|"y"``
    with ``symmetry="auto"`` uses C2v sectors in orthogonal cells and a
    D6-closed reciprocal star with Cs mirror sectors for triangular NVM or
    matched-ASR layers.  Explicit ``symmetry="d6"`` together with x/y solves
    only the corresponding E1 matrix-unit row.  In a general
    oblique NVM cell, Cartesian x/y share one C2 source-accessible sector, so
    either request uses that same reduced sector and retains x/y conversion.
    Explicit ``symmetry="d6"`` with no polarization selects the complete
    native-star A1/A2/B1/B2/E1/E2 decomposition.
    All source-specific reductions require normal incidence and one centered
    circle per cell.
    In non-strict mode an ineligible or inconsistent case falls back to the
    full eigensolve when that fallback is mathematically available.
    """

    enabled: bool = False
    symmetry: str = "auto"
    strict: bool = False
    residual_tolerance: float = 1.0e-8
    polarization: str | None = None


@dataclass(frozen=True)
class OutputSpec:
    """Requested scattering blocks and fields.

    ``smatrix_size`` is ``full``, ``half`` (forward T/R), or ``quarter``
    (forward R).  Fields may be requested with any public S size; the solver
    retains private bidirectional scattering data and modal couplings without
    exposing unrequested public blocks.  ``smatrix=False`` creates a
    fields-only result: all public S blocks are zero placeholders and
    ``S_parameters`` raises, while the private field data remain available.
    """

    smatrix: bool = True
    smatrix_size: str = "full"
    fields: str | bool = "all"

    def normalized_smatrix_size(self) -> str:
        return _normalize_smatrix_size(self.smatrix_size)

    def normalized_fields(self) -> str:
        if isinstance(self.fields, bool):
            return "all" if self.fields else "none"
        normalized = self.fields.strip().lower()
        if normalized not in {"none", "external", "internal", "all"}:
            raise ValueError("fields must be none, external, internal, all, or bool.")
        return normalized


@dataclass
class LayerRecord:
    index: int
    method: str
    shape: str
    lattice: str
    reason: str
    label: str | None = None
    options: dict[str, object] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _as_float(value: object) -> float:
    """Convert a real scalar/tensor to float without accepting an imaginary part."""
    if isinstance(value, bool):
        raise ValueError("Expected a real scalar, got a boolean.")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, complex):
        if abs(value.imag) > 1.0e-12:
            raise ValueError(f"Expected a real value, got {value!r}.")
        return float(value.real)
    tensor = torch.as_tensor(value).detach().cpu()
    if tensor.numel() != 1:
        raise ValueError(f"Expected a scalar, got shape {tuple(tensor.shape)}.")
    if torch.is_complex(tensor):
        if torch.max(torch.abs(tensor.imag)).item() > 1.0e-12:
            raise ValueError(f"Expected a real value, got {value!r}.")
        tensor = tensor.real
    return float(tensor.item())


def _real_parameter_tensor(
    name: str,
    value: object,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
    allow_zero: bool,
) -> tuple[torch.Tensor, float]:
    """Return a real scalar Tensor without detaching its autograd history.

    The accompanying Python value is for validation and discrete control flow
    only.  Every physical formula must use the Tensor returned as the first
    item.  This separation prevents accidental ``float(tensor)`` graph cuts.
    """
    if torch.is_tensor(value):
        source = value.to(device=device)
    else:
        # torch.as_tensor(0.1) follows the global default (normally float32),
        # which permanently rounds Python design values before the requested
        # solver precision is applied below.  Start scalar inputs in double
        # precision and only then cast to the solver's real dtype.
        source = torch.as_tensor(
            value,
            dtype=torch.complex128 if isinstance(value, complex) else torch.float64,
            device=device,
        )
    if source.numel() != 1:
        raise ValueError(f"{name} must be a scalar; got shape {tuple(source.shape)}.")
    if torch.is_complex(source):
        if _as_float(torch.max(torch.abs(source.imag))) > 1.0e-12:
            raise ValueError(f"{name} must be real.")
        source = source.real
    real_dtype = torch.float32 if dtype in {torch.float32, torch.complex64} else torch.float64
    parameter = source.to(dtype=real_dtype, device=device).reshape(())
    scalar = _as_float(parameter)
    valid_sign = scalar >= 0.0 if allow_zero else scalar > 0.0
    qualifier = "nonnegative" if allow_zero else "positive"
    if not math.isfinite(scalar) or not valid_sign:
        raise ValueError(f"{name} must be finite and {qualifier}; got {value!r}.")
    return parameter, scalar
