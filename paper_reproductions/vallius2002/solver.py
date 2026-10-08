"""One-dimensional Fourier modal method and adaptive spatial resolution.

This implements Eqs. (14)--(16), (25)--(27) of Vallius and Honkanen,
Optics Express 10, 24--34 (2002).  Lengths, including wavelength, must use
the same units.  ``harmonics`` is the number of *retained* modes and the
number of physical-space Rayleigh orders.  For ASR, ``oversampling=3``
solves a three-times-larger eigenproblem, as prescribed on page 30.

The scalar field is E_y for TE and H_y for TM.  Its companion boundary
field is gamma E_y for TE and gamma H_y/epsilon for TM.  Every interface
is matched in the same physical x-space Fourier basis, even when each
layer has a different adaptive coordinate map.  Interface and propagation
scattering matrices are joined by the Redheffer star product, so growing
evanescent-wave exponentials are never formed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import warnings

import numpy as np
from scipy import linalg
from scipy.special import roots_legendre

from .devices import resolve_execution


@dataclass(frozen=True)
class LayerSpec:
    """Piecewise-constant permittivity of one finite-thickness layer.

    ``breaks`` contains 0, all internal edges, and ``period``.  The value
    ``epsilon[j]`` applies between ``breaks[j]`` and ``breaks[j+1]``.
    Repeated adjacent materials are merged for the adaptive map, including
    across the periodic seam.  This ensures that equal u-space segments
    are allocated to actual material regions rather than arbitrary cuts.
    """

    thickness: float
    breaks: Sequence[float]
    epsilon: Sequence[complex]
    period: float = 1.0
    name: str = ""

    def __post_init__(self) -> None:
        edges = tuple(float(x) for x in self.breaks)
        eps = tuple(complex(x) for x in self.epsilon)
        period = float(self.period)
        thickness = float(self.thickness)
        if not np.isfinite(period) or period <= 0:
            raise ValueError("period must be positive and finite")
        if not np.isfinite(thickness) or thickness < 0:
            raise ValueError("thickness must be nonnegative and finite")
        if len(edges) != len(eps) + 1 or len(eps) == 0:
            raise ValueError("breaks must have one more entry than epsilon")
        if abs(edges[0]) > 1e-12 * period or abs(edges[-1] - period) > 1e-12 * period:
            raise ValueError("breaks must start at 0 and end at period")
        if not np.all(np.isfinite(edges)) or not np.all(np.diff(edges) > 0):
            raise ValueError("breaks must be finite and strictly increasing")
        if not np.all(np.isfinite(eps)) or any(abs(x) < 1e-15 for x in eps):
            raise ValueError("permittivity must be finite and nonzero")
        object.__setattr__(self, "breaks", edges)
        object.__setattr__(self, "epsilon", eps)
        object.__setattr__(self, "period", period)
        object.__setattr__(self, "thickness", thickness)


def harmonic_orders(count: int) -> np.ndarray:
    """Contiguous orders; an even N uses -N//2, ..., N//2-1."""
    if int(count) != count or count < 1:
        raise ValueError("harmonics must be a positive integer")
    count = int(count)
    return np.arange(-(count // 2), count - count // 2, dtype=int)


def _toeplitz(coefficients: np.ndarray, count: int) -> np.ndarray:
    index = np.subtract.outer(np.arange(count), np.arange(count)) + count - 1
    return np.asarray(coefficients[index], dtype=np.complex128)


def _physical_coefficients(layer: LayerSpec, count: int, reciprocal: bool = False) -> np.ndarray:
    """Exact Fourier coefficients of a physical-space lamellar profile."""
    differences = np.arange(-(count - 1), count)
    coefficients = np.zeros(2 * count - 1, dtype=np.complex128)
    for left, right, epsilon in zip(layer.breaks[:-1], layer.breaks[1:], layer.epsilon):
        width = (right - left) / layer.period
        midpoint = (right + left) / (2 * layer.period)
        value = 1 / epsilon if reciprocal else epsilon
        coefficients += value * width * np.sinc(differences * width) * np.exp(-2j * np.pi * differences * midpoint)
    return coefficients


def _cyclic_regions(layer: LayerSpec) -> tuple[np.ndarray, np.ndarray]:
    """Actual cyclic regions, starting at a material transition.

    The returned physical coordinates may span [origin, origin+period].
    Coordinates need not start at zero because Eq. (25) projects back onto
    the original physical basis, including the resulting translation phase.
    """
    eps = np.asarray(layer.epsilon, dtype=np.complex128)
    edges = np.asarray(layer.breaks, dtype=float)
    transitions = [j for j in range(len(eps)) if eps[j] != eps[j - 1]]
    if not transitions:
        return np.array([0.0, layer.period]), eps[:1]
    start = transitions[0]
    indices = sorted(transitions, key=lambda j: (j - start) % len(eps))
    region_edges = [edges[j] + (layer.period if j < start else 0.0) for j in indices]
    region_edges.append(region_edges[0] + layer.period)
    return np.asarray(region_edges), eps[indices]


def _sqrt_outgoing(eigenvalues: np.ndarray) -> np.ndarray:
    gamma = np.sqrt(np.asarray(eigenvalues, dtype=np.complex128))
    # Eq. (14)'s branch convention: Re(gamma)+Im(gamma)>0.
    gamma[np.real(gamma) + np.imag(gamma) < 0] *= -1
    return gamma


class _PreparedLayer:
    """Wavelength-independent Laurent matrices and projection quadrature."""

    def __init__(self, spec: LayerSpec, method: str, count: int, internal_count: int,
                 quadrature: int | None, G: float, *, prepare_linear_algebra: bool = True):
        self.spec = spec
        self.method = method
        self.count = count
        self.homogeneous = all(e == spec.epsilon[0] for e in spec.epsilon)
        self.internal_count = count if self.homogeneous else internal_count
        internal_count = self.internal_count
        self.orders = harmonic_orders(internal_count)
        self.rayleigh_orders = harmonic_orders(count)
        self.quadrature_points = 0
        self.lossless = all(abs(e.imag) < 1e-14 for e in spec.epsilon)
        self.positive_epsilon = all(e.real > 0 for e in spec.epsilon)

        if self.homogeneous:
            self.epsilon_value = spec.epsilon[0]
            return

        if method == "fmm":
            self.f = np.eye(internal_count, dtype=np.complex128)
            self.a = _toeplitz(_physical_coefficients(spec, internal_count), internal_count)
            self.b = _toeplitz(_physical_coefficients(spec, internal_count, True), internal_count)
            self.inverse_epsilon = self.b
            rayleigh_indices = np.flatnonzero(np.isin(self.orders, self.rayleigh_orders))
            self.K0 = np.eye(internal_count, dtype=np.complex128)[rayleigh_indices]
            self.KQ0 = self.K0 @ self.inverse_epsilon
            if len(rayleigh_indices) != count:
                raise ValueError("internal harmonics must include every Rayleigh order")
        else:
            edges, epsilon = _cyclic_regions(spec)
            regions = len(epsilon)
            u_edges = np.linspace(0, spec.period, regions + 1)
            if quadrature is not None and int(quadrature) < 8:
                raise ValueError("quadrature must be at least 8 points per region")
            # Underintegrating high Fourier differences can make the
            # positive metric f spuriously indefinite.  Treat a requested
            # order as a lower bound and always resolve the retained basis.
            points = max(48, 2 * (internal_count + count), int(quadrature or 0))
            self.quadrature_points = points
            nodes, weights = roots_legendre(points)
            all_u, all_x, all_f, all_weights, all_eps = [], [], [], [], []
            for j, value in enumerate(epsilon):
                du = u_edges[j + 1] - u_edges[j]
                u = u_edges[j] + (nodes + 1) * du / 2
                ratio = (edges[j + 1] - edges[j]) / du
                phase = 2 * np.pi * (u - u_edges[j]) / du
                x = edges[j] + ratio * (u - u_edges[j]) + (G - ratio) * du / (2 * np.pi) * np.sin(phase)
                f = ratio + (G - ratio) * np.cos(phase)
                if np.min(f) <= 0:
                    raise ValueError("G produces a non-monotone adaptive coordinate map")
                all_u.append(u)
                all_x.append(x)
                all_f.append(f)
                all_weights.append(weights * du / (2 * spec.period))
                all_eps.append(np.full(points, value, dtype=np.complex128))
            self.u = np.concatenate(all_u)
            self.x = np.concatenate(all_x)
            self.resolution = np.concatenate(all_f)
            self.weights = np.concatenate(all_weights)
            eps_samples = np.concatenate(all_eps)
            self.epsilon_samples = eps_samples
            differences = np.arange(-(internal_count - 1), internal_count)
            phase_matrix = np.exp(-2j * np.pi * differences[:, None] * self.u[None, :] / spec.period)
            self.f = _toeplitz(phase_matrix @ (self.weights * self.resolution), internal_count)
            self.a = _toeplitz(phase_matrix @ (self.weights * self.resolution * eps_samples), internal_count)
            self.b = _toeplitz(phase_matrix @ (self.weights * self.resolution / eps_samples), internal_count)
            # Q^u = [1/epsilon(u)] H^u must be formed before projection.
            self.inverse_epsilon = _toeplitz(phase_matrix @ (self.weights / eps_samples), internal_count)
            self.K0 = self._projection(0.0)
            self.KQ0 = self._projection(0.0, reciprocal=True)

        if not prepare_linear_algebra:
            return
        identity = np.eye(internal_count, dtype=np.complex128)
        self.inverse_f = linalg.solve(self.f, identity, assume_a="her")
        self.inverse_a = linalg.solve(self.a, identity, assume_a="her" if self.lossless else "gen")
        if self.method == "fmm":
            # TM's wavelength-independent metric is factored once.  Its
            # LU factors permit the standard problem b^-1 A without
            # explicitly constructing b^-1 or repeating a QZ reduction.
            self.b_factors = linalg.lu_factor(self.b, check_finite=False)

    def _projection(self, alpha0: complex, reciprocal: bool = False) -> np.ndarray:
        alpha_p = alpha0 + 2 * np.pi * self.rayleigh_orders / self.spec.period
        alpha_m = alpha0 + 2 * np.pi * self.orders / self.spec.period
        physical = np.exp(-1j * alpha_p[:, None] * self.x[None, :])
        adaptive = np.exp(1j * self.u[:, None] * alpha_m[None, :])
        weights = self.weights * self.resolution
        if reciprocal:
            weights = weights / self.epsilon_samples
        return (physical * weights[None, :]) @ adaptive

    def modes(self, k: float, alpha0: complex, polarization: str,
              retention: str = "smallest_abs", q_projection: str = "direct") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        count = self.count
        if self.homogeneous:
            alpha = alpha0 + 2 * np.pi * harmonic_orders(count) / self.spec.period
            gamma = _sqrt_outgoing(k * k * self.epsilon_value - alpha * alpha)
            W = np.eye(count, dtype=np.complex128)
            V = np.diag(gamma if polarization == "TE" else gamma / self.epsilon_value)
            return W, V, gamma

        alpha = alpha0 + 2 * np.pi * self.orders / self.spec.period
        hermitian = self.lossless and abs(complex(alpha0).imag) < 1e-14
        if polarization == "TE":
            operator = k * k * self.a - alpha[:, None] * self.inverse_f * alpha[None, :]
            metric = self.f
        else:
            operator = k * k * self.f - alpha[:, None] * self.inverse_a * alpha[None, :]
            metric = self.b
            hermitian = hermitian and self.positive_epsilon
        if self.method == "fmm" and polarization == "TE":
            # Here f=I exactly.  A generalized QZ reduction of (A,I)
            # computes the same problem but is substantially slower for
            # the paper's 240-order reference calculation.
            if hermitian:
                operator = (operator + operator.conj().T) / 2
                eigenvalues, eigenvectors = linalg.eigh(operator, check_finite=False)
            else:
                eigenvalues, eigenvectors = linalg.eig(operator, check_finite=False)
        elif hermitian:
            operator = (operator + operator.conj().T) / 2
            metric = (metric + metric.conj().T) / 2
            eigenvalues, eigenvectors = linalg.eigh(operator, metric, check_finite=False)
        elif self.method == "fmm":
            standard_operator = linalg.lu_solve(self.b_factors, operator, check_finite=False)
            eigenvalues, eigenvectors = linalg.eig(standard_operator, check_finite=False)
        else:
            eigenvalues, eigenvectors = linalg.eig(operator, metric, check_finite=False)
        if self.internal_count > count:
            if retention == "smallest_abs":
                keep = np.argsort(np.abs(eigenvalues), kind="stable")[:count]
            else:
                roots = _sqrt_outgoing(eigenvalues)
                propagating = (eigenvalues.real > 0) & (np.abs(eigenvalues.imag) < 1e-8 * np.maximum(1.0, np.abs(eigenvalues)))
                sort_value = np.where(propagating, -eigenvalues.real, np.abs(roots.imag))
                keep = np.lexsort((sort_value, ~propagating))[:count]
            eigenvalues = eigenvalues[keep]
            eigenvectors = eigenvectors[:, keep]
        gamma = _sqrt_outgoing(eigenvalues)
        K = self.K0 if alpha0 == 0 or self.method == "fmm" else self._projection(alpha0)
        W = K @ eigenvectors
        if self.method == "asr" and polarization == "TM" and q_projection == "galerkin":
            # The reduced ASR equation is B_r c'' + O_r c = 0, with
            # B_r=H^H b H and O_r=H^H O H. Its flux is Re(c^H B_r c'/i).
            # Independently truncating H and H/epsilon onto x harmonics
            # does not preserve that flux and can turn a passive layer
            # into an active discrete system. Use the dual boundary trace:
            # W^H V = H^H b H Gamma. This is a Galerkin traction, not a
            # pointwise x-space Laurent product or a power correction.
            scales = np.maximum(linalg.norm(W, axis=0), 1e-30)
            H = eigenvectors / scales[None, :]
            W = W / scales[None, :]
            reduced_metric = H.conj().T @ self.b @ H
            with warnings.catch_warnings():
                warnings.simplefilter("error", linalg.LinAlgWarning)
                try:
                    V = linalg.solve(W.conj().T, reduced_metric * gamma[None, :], check_finite=False)
                except linalg.LinAlgWarning as exc:
                    raise np.linalg.LinAlgError("ASR modal projection is rank deficient; increase retained harmonics") from exc
            return W, V, gamma
        if polarization == "TE":
            Q = W
        elif q_projection == "laurent":
            Q = K @ (self.inverse_epsilon @ eigenvectors)
        else:
            # Directly integrate Q=H/epsilon in Eq. (26).  This is the
            # infinite-Q^u-series limit of K Q^u and avoids truncating a
            # second Fourier product before the physical projection.
            KQ = self.KQ0 if alpha0 == 0 or self.method == "fmm" else self._projection(alpha0, reciprocal=True)
            Q = KQ @ eigenvectors
        # Column scaling changes no field solution and improves interface solves.
        scales = np.maximum(linalg.norm(W, axis=0), 1e-30)
        W = W / scales[None, :]
        V = (Q / scales[None, :]) * gamma[None, :]
        return W, V, gamma


class _Diagnostics:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.max_condition = 1.0
        self.solves = 0

    def solve(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        self.solves += 1
        with warnings.catch_warnings():
            warnings.simplefilter("error", linalg.LinAlgWarning)
            try:
                factors, pivots = linalg.lu_factor(A, check_finite=False)
                result = linalg.lu_solve((factors, pivots), B, check_finite=False)
            except linalg.LinAlgWarning as exc:
                raise np.linalg.LinAlgError("singular interface; increase retained harmonics or avoid an exact modal cutoff") from exc
        if self.enabled:
            estimate, info = linalg.get_lapack_funcs("gecon", (factors,))(factors, linalg.norm(A, 1))
            condition = float(1 / estimate) if estimate > 0 and info == 0 else float("inf")
            self.max_condition = max(self.max_condition, condition)
        return result


def _interface(Wl: np.ndarray, Vl: np.ndarray, Wr: np.ndarray, Vr: np.ndarray,
               diagnostics: _Diagnostics) -> tuple[np.ndarray, ...]:
    """Map incoming (left+, right-) to outgoing (left-, right+)."""
    count = len(Wl)
    A = np.block([[Wl, -Wr], [-Vl, -Vr]])
    B = np.block([[-Wl, Wr], [-Vl, -Vr]])
    scattering = diagnostics.solve(A, B)
    return (scattering[:count, :count], scattering[:count, count:],
            scattering[count:, :count], scattering[count:, count:])


def _star(A: tuple[np.ndarray, ...], B: tuple[np.ndarray, ...],
          diagnostics: _Diagnostics) -> tuple[np.ndarray, ...]:
    """Join A followed by B without a growing evanescent exponential."""
    A11, A12, A21, A22 = A
    B11, B12, B21, B22 = B
    count = len(A11)
    D = np.eye(count, dtype=np.complex128) - B11 @ A22
    joined = diagnostics.solve(D, np.concatenate((B11 @ A21, B12), axis=1))
    left, right = joined[:, :count], joined[:, count:]
    return (A11 + A12 @ left, A12 @ right,
            B21 @ (A21 + A22 @ left), B22 + B21 @ A22 @ right)


def _propagate(scattering: tuple[np.ndarray, ...], phase: np.ndarray) -> tuple[np.ndarray, ...]:
    """Exact star product with (0,X;X,0), using diagonal X directly."""
    S11, S12, S21, S22 = scattering
    return (S11, S12 * phase[None, :], phase[:, None] * S21,
            phase[:, None] * S22 * phase[None, :])


class PreparedStack:
    """Prepare one periodic multilayer geometry and solve wavelengths.

    Args:
        layers: Ordered from incident side to transmission side (+z).
        polarization: ``"TE"`` or ``"TM"``.
        method: ``"fmm"`` or ``"asr"`` (also ``"parametric"``).
        harmonics: Actual retained count N, not the maximum order M.
        oversampling: Internal dimension / N; defaults to 1 for FMM and
            3 for ASR.  Larger FMM values permit an explicit research
            comparison with the same eigenmode-retention rule.
        quadrature: Minimum Gauss--Legendre points per actual material
            region; the actual order is at least
            max(48, 2*(internal_count+harmonics)) to prevent aliasing.
        G: Resolution f=dx/du at each material transition, Eq. (19).
        angle: Incidence angle in radians.
        epsilon_in, epsilon_out: Homogeneous external permittivities.
        diagnostics: Estimate interface/star-product condition numbers.
        device: ``auto`` chooses available CUDA, otherwise CPU; ``cpu``,
            ``cuda`` or ``cuda:0`` overrides it. CUDA requires PyTorch.
        backend: ``auto`` uses SciPy on CPU and PyTorch on CUDA. Explicit
            ``torch`` also supports CPU for tensor-kernel validation.
        retention: ``"smallest_abs"`` is the paper's stated selection.
            ``"physical"`` is a diagnostic alternative selecting all real
            propagating modes from highest gamma squared, then evanescent
            modes from smallest decay.  It is not the paper's prescription.
        q_projection: ``"direct"`` integrates the continuous product
            Q=H/epsilon in Eq. (26), avoiding a second Fourier truncation.
            ``"laurent"`` forms a finite Q^u Laurent vector first.  The
            paper does not specify the finite truncation of Q^u separately.
            ``"galerkin"`` uses a dual TM boundary trace preserving the
            reduced ASR power metric H^H b H. It prevents artificial gain
            from independent field projections; TE and FMM are unchanged.
            Direct remains the default for the original paper reproduction.

    At an exact external Rayleigh cutoff the wavelength is shifted upward
    by ``cutoff_shift`` (default relative 1e-12) and the evaluation value
    is reported.  This computes a one-sided limit of the degenerate
    zero-flux Rayleigh basis; it does not change the requested spectrum grid.
    """

    def __init__(self, layers: Iterable[LayerSpec], polarization: str = "TE",
                 method: str = "fmm", harmonics: int = 15,
                 oversampling: float | None = None, quadrature: int | None = None,
                 G: float = 0.001, epsilon_in: complex = 1.0,
                 epsilon_out: complex = 1.0, angle: float = 0.0,
                 diagnostics: bool = True, retention: str = "smallest_abs",
                 q_projection: str = "direct", device: str = "auto",
                 backend: str = "auto"):
        self.layers = tuple(layers)
        if not self.layers:
            raise ValueError("at least one LayerSpec is required")
        self.polarization = polarization.upper()
        if self.polarization not in {"TE", "TM"}:
            raise ValueError("polarization must be TE or TM")
        aliases = {"rcwa": "fmm", "parametric": "asr"}
        self.method = aliases.get(method.lower(), method.lower())
        if self.method not in {"fmm", "asr"}:
            raise ValueError("method must be fmm or asr")
        self.orders = harmonic_orders(harmonics)
        self.harmonics = int(harmonics)
        self.period = self.layers[0].period
        if any(abs(layer.period - self.period) > 1e-12 * self.period for layer in self.layers):
            raise ValueError("all layers must share one physical period")
        self.epsilon_in = complex(epsilon_in)
        self.epsilon_out = complex(epsilon_out)
        self.angle = float(angle)
        self.diagnostics = bool(diagnostics)
        if retention not in {"smallest_abs", "physical"}:
            raise ValueError("retention must be smallest_abs or physical")
        self.retention = retention
        if q_projection not in {"direct", "laurent", "galerkin"}:
            raise ValueError("q_projection must be direct, laurent, or galerkin")
        self.q_projection = q_projection
        self.G = float(G)
        if not 0 < self.G:
            raise ValueError("G must be positive")
        if oversampling is None:
            oversampling = 3.0 if self.method == "asr" else 1.0
        if not np.isfinite(oversampling) or oversampling < 1:
            raise ValueError("oversampling must be finite and at least 1")
        self.oversampling = float(oversampling)
        self.internal_count = max(self.harmonics, int(round(self.oversampling * self.harmonics)))
        self.execution = resolve_execution(device, backend)
        self.device = self.execution["device"]
        self.backend = self.execution["backend"]
        layer_factory = _PreparedLayer
        if self.backend == "torch":
            from .torch_backend import TorchPreparedLayer
            layer_factory = lambda *args: TorchPreparedLayer(*args, device=self.device)
        # Equal adjacent layers share prepared Fourier matrices.  This is
        # especially useful when a staircase representation repeats slices.
        cache: dict[tuple, _PreparedLayer] = {}
        self.prepared_layers = []
        for layer in self.layers:
            key = (layer.breaks, layer.epsilon, layer.period)
            if key not in cache:
                cache[key] = layer_factory(layer, self.method, self.harmonics,
                                           self.internal_count, quadrature, self.G)
            self.prepared_layers.append(cache[key])

    def _external_modes(self, k: float, alpha0: complex, epsilon: complex) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        alpha = alpha0 + 2 * np.pi * self.orders / self.period
        gamma = _sqrt_outgoing(k * k * epsilon - alpha * alpha)
        W = np.eye(self.harmonics, dtype=np.complex128)
        V = np.diag(gamma if self.polarization == "TE" else gamma / epsilon)
        return W, V, gamma

    def _wavelength_parameters(self, wavelength: float, cutoff_shift: float) -> tuple:
        """Shared host-side validation and identical Rayleigh limit on both backends."""
        wavelength = float(wavelength)
        if not np.isfinite(wavelength) or wavelength <= 0:
            raise ValueError("wavelength must be positive and finite")
        requested_wavelength = wavelength
        k = 2 * np.pi / wavelength
        alpha0 = k * np.sqrt(self.epsilon_in) * np.sin(self.angle)
        alpha = alpha0 + 2 * np.pi * self.orders / self.period
        scale = k * k * max(1.0, abs(self.epsilon_in), abs(self.epsilon_out))
        at_cutoff = any(np.any(np.abs(k * k * epsilon - alpha * alpha) < 1e-14 * scale)
                        for epsilon in (self.epsilon_in, self.epsilon_out))
        if at_cutoff:
            if cutoff_shift == 0:
                raise ValueError("exact external Rayleigh cutoff: provide a nonzero cutoff_shift to evaluate its limit")
            wavelength *= 1 + float(cutoff_shift)
            k = 2 * np.pi / wavelength
            alpha0 = k * np.sqrt(self.epsilon_in) * np.sin(self.angle)
        return requested_wavelength, wavelength, k, alpha0, at_cutoff

    def solve(self, wavelength: float, *, cutoff_shift: float = 1e-12) -> dict:
        if self.backend == "torch":
            from .torch_backend import solve_stack
            return solve_stack(self, wavelength, cutoff_shift=cutoff_shift)
        requested_wavelength, wavelength, k, alpha0, at_cutoff = self._wavelength_parameters(wavelength, cutoff_shift)

        Wleft, Vleft, gamma_in = self._external_modes(k, alpha0, self.epsilon_in)
        Wout, Vout, gamma_out = self._external_modes(k, alpha0, self.epsilon_out)
        count = self.harmonics
        zero = np.zeros((count, count), dtype=np.complex128)
        identity = np.eye(count, dtype=np.complex128)
        scattering = (zero.copy(), identity.copy(), identity.copy(), zero.copy())
        diagnostics = _Diagnostics(self.diagnostics)
        cached_modes: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for layer_index, (spec, prepared) in enumerate(zip(self.layers, self.prepared_layers)):
            mode_key = id(prepared)
            if mode_key not in cached_modes:
                cached_modes[mode_key] = prepared.modes(k, alpha0, self.polarization,
                                                       self.retention, self.q_projection)
            W, V, gamma = cached_modes[mode_key]
            interface = _interface(Wleft, Vleft, W, V, diagnostics)
            # The initial through-scattering matrix is the star identity.
            scattering = interface if layer_index == 0 else _star(scattering, interface, diagnostics)
            propagation = np.exp(1j * gamma * spec.thickness)
            scattering = _propagate(scattering, propagation)
            Wleft, Vleft = W, V
        scattering = _star(scattering, _interface(Wleft, Vleft, Wout, Vout, diagnostics), diagnostics)

        incident = np.zeros(count, dtype=np.complex128)
        zeroth = int(np.flatnonzero(self.orders == 0)[0])
        incident[zeroth] = 1
        reflected = scattering[0] @ incident
        transmitted = scattering[2] @ incident
        admittance_in = gamma_in if self.polarization == "TE" else gamma_in / self.epsilon_in
        admittance_out = gamma_out if self.polarization == "TE" else gamma_out / self.epsilon_out
        incident_flux = float(np.real(admittance_in[zeroth]))
        if incident_flux <= 0:
            raise ValueError("the incident zeroth order must carry positive forward flux")
        R_orders = np.real(admittance_in) / incident_flux * np.abs(reflected) ** 2
        T_orders = np.real(admittance_out) / incident_flux * np.abs(transmitted) ** 2
        R = float(np.sum(R_orders))
        T = float(np.sum(T_orders))
        T0 = float(T_orders[zeroth])
        return {
            "wavelength": requested_wavelength,
            "evaluation_wavelength": wavelength,
            "cutoff_regularized": bool(at_cutoff),
            "T0": T0, "eta0": T0, "T": T, "R": R, "A": 1 - R - T,
            "T_orders": T_orders, "R_orders": R_orders,
            "t": transmitted, "r": reflected, "orders": self.orders.copy(),
            "polarization": self.polarization, "method": self.method,
            "retention": self.retention,
            "q_projection": self.q_projection,
            "retained_modes": count,
            "eigen_dimensions": np.asarray([layer.internal_count for layer in self.prepared_layers]),
            "max_boundary_condition": diagnostics.max_condition,
            "linear_solves": diagnostics.solves,
            "quadrature_points_per_region": max(layer.quadrature_points for layer in self.prepared_layers),
            "device": self.device, "backend": self.backend, "execution": self.execution.copy(),
        }

    def spectrum(self, wavelengths: Iterable[float], **solve_options) -> dict:
        """Evaluate a wavelength grid; arrays retain its input order."""
        values = [self.solve(value, **solve_options) for value in wavelengths]
        if not values:
            raise ValueError("wavelengths must not be empty")
        array_keys = ("wavelength", "evaluation_wavelength", "cutoff_regularized",
                      "T0", "eta0", "T", "R", "A", "T_orders", "R_orders", "t", "r",
                      "max_boundary_condition")
        result = {key: np.asarray([value[key] for value in values]) for key in array_keys}
        result.update({key: values[0][key] for key in
                       ("orders", "polarization", "method", "retention", "q_projection", "retained_modes", "eigen_dimensions",
                        "linear_solves", "quadrature_points_per_region", "device", "backend", "execution")})
        result["wavelengths"] = result["wavelength"]
        return result


def compute_wavelength(layers: Iterable[LayerSpec], wavelength: float, **stack_options) -> dict:
    """Convenience wrapper; reuse PreparedStack for a wavelength sweep."""
    return PreparedStack(layers, **stack_options).solve(wavelength)


def compute_spectrum(layers: Iterable[LayerSpec], wavelengths: Iterable[float], **stack_options) -> dict:
    return PreparedStack(layers, **stack_options).spectrum(wavelengths)
