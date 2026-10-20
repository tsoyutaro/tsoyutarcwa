"""One-dimensional TM RCWA / FFF / differential scattering solver.

Mohamad et al., Optical and Quantum Electronics 52, 127 (2020), Eqs. 4-11.
Time convention exp(-i omega t); computational z increases from incident to exit.
Lengths are in nm and fields are [Ex, Z0 Hy]. N retains orders -N,...,+N.
Only thin-section transfer matrices are formed; the stack uses scattering blocks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import warnings
import numpy as np
from scipy.linalg import expm, solve as linear_solve, LinAlgWarning


METHODS = ("rcwa", "rcwa-fff", "dm-fff")
DM_INTEGRATORS = ("cf4-adaptive", "cf4", "rk4")
SOLVER_REVISION = "2026-10-20-adaptive-dm-centered-gil"


def _solve(a, b):
    # Ill conditioning is a scientific result, not a reason to return a clipped curve.
    with warnings.catch_warnings():
        warnings.simplefilter("error", LinAlgWarning)
        return linear_solve(a, b, check_finite=False)


def convolution(values: np.ndarray, max_order: int) -> np.ndarray:
    """Toeplitz matrix with element (m,n) = FFT(values)[m-n]."""
    values = np.asarray(values, dtype=np.complex128)
    if values.ndim != 1 or len(values) <= 4 * max_order:
        raise ValueError("nfft must exceed 4*N (Toeplitz differences reach +/-2N)")
    orders = np.arange(-max_order, max_order + 1)
    coeff = np.fft.fft(values) / len(values)
    return coeff[(orders[:, None] - orders[None, :]) % len(values)]


def outgoing_q(epsilon: complex, alpha: np.ndarray) -> np.ndarray:
    """Outgoing/decaying square-root branch for exp(+i k0 q z)."""
    q = np.sqrt(complex(epsilon) - alpha ** 2 + 0j)
    flip = (q.imag < 0) | ((np.abs(q.imag) < 1e-14) & (q.real < 0))
    return np.where(flip, -q, q)


@dataclass
class Scattering:
    """[b_left,a_right] = S [a_left,b_right]."""
    s11: np.ndarray
    s12: np.ndarray
    s21: np.ndarray
    s22: np.ndarray


def transparent(size: int) -> Scattering:
    eye = np.eye(size, dtype=complex)
    zero = np.zeros_like(eye)
    return Scattering(zero.copy(), eye.copy(), eye.copy(), zero.copy())


def cascade(left: Scattering, right: Scattering) -> Scattering:
    """Redheffer star product, in spatial left-to-right order."""
    size = left.s11.shape[0]
    joined = _solve(np.eye(size) - left.s22 @ right.s11,
                    np.concatenate((left.s21, left.s22 @ right.s12), axis=1))
    u, v = joined[:, :size], joined[:, size:]
    return Scattering(left.s11 + left.s12 @ right.s11 @ u,
                      left.s12 @ (right.s12 + right.s11 @ v),
                      right.s21 @ u,
                      right.s22 + right.s21 @ v)


def transfer_to_scattering(transfer: np.ndarray) -> Scattering:
    size = transfer.shape[0] // 2
    aa, ab = transfer[:size, :size], transfer[:size, size:]
    ba, bb = transfer[size:, :size], transfer[size:, size:]
    joined = _solve(bb, np.concatenate((ba, np.eye(size)), axis=1))
    x, y = joined[:, :size], joined[:, size:]
    return Scattering(-x, y, aa - ab @ x, ab @ y)


def _reference_transfer(propagator: np.ndarray) -> np.ndarray:
    """W^-1 P W for W=[[I,-I],[I,I]], without a dense basis solve."""
    size = propagator.shape[0] // 2
    ee, eh = propagator[:size, :size], propagator[:size, size:]
    he, hh = propagator[size:, :size], propagator[size:, size:]
    return .5 * np.block([[ee + eh + he + hh, -ee + eh - he + hh],
                         [-ee - eh + he + hh, ee - eh - he + hh]])


def _interface(v_left: np.ndarray, v_right: np.ndarray) -> Scattering:
    """TM interface in magnetic amplitudes, v=q/epsilon."""
    den = v_left + v_right
    if np.any(np.abs(den) < 1e-14):
        raise ValueError("Singular interface at a modal cutoff or surface resonance")
    r = (v_left - v_right) / den
    return Scattering(np.diag(r), np.diag(2 * v_right / den),
                      np.diag(2 * v_left / den), np.diag(-r))


class PropagationMatrix:
    """Normalized Eq.7 generator K, with dF/dz=i*k0*K*F."""
    def __init__(self, geometry, wavelength_nm, angle_deg, max_order, nfft, fff):
        self.geometry = geometry
        self.max_order = max_order
        self.x = np.arange(nfft) * geometry.period_nm / nfft
        self.orders = np.arange(-max_order, max_order + 1)
        self.alpha = (complex(geometry.n_in).real * np.sin(np.deg2rad(angle_deg))
                      + self.orders * wavelength_nm / geometry.period_nm)
        self.identity = np.eye(len(self.orders), dtype=complex)
        self.fff = fff
        boundaries = getattr(geometry, "constant_slice_boundaries", None)
        self.constant_slices = boundaries(self.x) if boundaries is not None else None
        planes = getattr(geometry, "integration_planes", None)
        self.integration_planes = planes() if planes is not None else np.array([])
        if fff:
            c, s = geometry.tangent(self.x)
            c, s = np.broadcast_arrays(c, s, self.x)[:2]
            if not np.allclose(c * c + s * s, 1, atol=1e-10):
                raise ValueError("Tangent must satisfy c^2+s^2=1")
            self.c2 = convolution(c * c, max_order)
            self.cs = convolution(c * s, max_order)

    def __call__(self, z_nm):
        eps = np.broadcast_to(np.asarray(self.geometry.sample(self.x, z_nm), complex),
                              self.x.shape)
        if not np.isfinite(eps).all() or np.any(np.abs(eps) < 1e-14):
            raise ValueError("Permittivity must be finite and nonzero")
        e = convolution(eps, self.max_order)
        inv_eta = _solve(convolution(1 / eps, self.max_order), self.identity)
        if not self.fff:
            # Eq.11, including Li inverse rule for the x-normal interfaces.
            c_inv_alpha = _solve(e, np.diag(self.alpha))
            zero = np.zeros_like(e)
            return np.block([[zero, self.identity - self.alpha[:, None] * c_inv_alpha],
                             [inv_eta, zero]])
        delta = e - inv_eta
        q1 = delta @ self.c2
        b = delta @ self.cs
        a = inv_eta + q1
        c = e - q1
        joined = _solve(c, np.concatenate((b, np.diag(self.alpha)), axis=1))
        ci_b, ci_alpha = np.split(joined, 2, axis=1)
        return np.block([[-self.alpha[:, None] * ci_b,
                          self.identity - self.alpha[:, None] * ci_alpha],
                         [a - b @ ci_b, -b @ ci_alpha]])


def _rk4_propagator(generator, z0, dz, k0, substeps):
    """Integrate fundamental solution over one S section with RK4 substeps."""
    size = 2 * len(generator.orders)
    p = np.eye(size, dtype=complex)
    step = dz / substeps
    for j in range(substeps):
        z = z0 + j * step
        m0 = 1j * k0 * generator(z)
        mh = 1j * k0 * generator(z + .5 * step)
        m1 = 1j * k0 * generator(z + step)
        k1 = m0 @ p
        k2 = mh @ (p + .5 * step * k1)
        k3 = mh @ (p + .5 * step * k2)
        k4 = m1 @ (p + step * k3)
        p = p + step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return p


def _cf4_propagator(generator, z0, dz, k0):
    """Fourth-order commutator-free Magnus step, exact for constant K.

    The right exponential acts first. See Alvermann et al. (2012),
    doi:10.1088/1367-2630/14/10/105008, Eqs. (11)-(12).
    """
    c = np.sqrt(3.) / 6
    a, b = (3 - 2 * np.sqrt(3.)) / 12, (3 + 2 * np.sqrt(3.)) / 12
    early = 1j * k0 * generator(z0 + (.5 - c) * dz)
    late = 1j * k0 * generator(z0 + (.5 + c) * dz)
    return expm(dz * (a * early + b * late)) @ expm(dz * (b * early + a * late))


def _propagator_scattering(propagator):
    if not np.isfinite(propagator).all():
        raise FloatingPointError("Non-finite local propagation; reduce integration step")
    return transfer_to_scattering(_reference_transfer(propagator))


def _scattering_error(coarse, fine, rtol, atol):
    # Scattering blocks stay bounded for decaying modes, whereas full transfer
    # matrices contain both exp(+kappa*h) and exp(-kappa*h). Test all blocks,
    # including evanescent couplings; no incident-mode-only acceptance test.
    return max(float(np.max(np.abs(a - b))) /
               (atol + rtol * max(float(np.max(np.abs(a))), float(np.max(np.abs(b)))))
               for a, b in zip(vars(coarse).values(), vars(fine).values())) / 15.


def _dm_section(generator, z0, dz, k0, substeps, integrator, rtol, atol,
                min_step_nm, diagnostics):
    """Compose local scattering blocks, never a section-wide growing transfer."""
    size = len(generator.orders)
    total = transparent(size)
    end, z = z0 + dz, z0
    max_step = dz / substeps
    planes = generator.integration_planes
    planes = planes[(planes > z0 + 1e-10) & (planes < end - 1e-10)]
    if integrator == "cf4-adaptive" and len(planes):
        cuts = np.concatenate(([z0], np.sort(planes), [end]))
        for start, stop in zip(cuts[:-1], cuts[1:]):
            internal_count = max(1, int(np.ceil((stop - start) / max_step)))
            block = _dm_section(generator, start, stop - start, k0, internal_count,
                                integrator, rtol, atol, min_step_nm, diagnostics)
            total = cascade(total, block)
        return total
    if integrator == "cf4-adaptive" and generator.constant_slices is not None:
        # Exact exponential on each constant slice of the *sampled* material.
        # This resolves every FFT-grid crossing, including ones that an error
        # estimator sampling only the interior of a trial step could miss.
        edges = generator.constant_slices
        cuts = np.concatenate(([z0], edges[(edges > z0 + 1e-10) & (edges < end - 1e-10)], [end]))
        for start, stop in zip(cuts[:-1], cuts[1:]):
            parts = max(1, int(np.ceil((stop - start) / max_step)))
            matrix = 1j * k0 * generator(.5 * (start + stop))
            while True:
                h = (stop - start) / parts
                try:
                    block = _propagator_scattering(expm(h * matrix))
                    break
                except (np.linalg.LinAlgError, LinAlgWarning, ArithmeticError, RuntimeWarning):
                    diagnostics["rejected_steps"] += 1
                    if h <= min_step_nm:
                        raise FloatingPointError(f"Ill-conditioned constant slice at z={start:.12g} nm")
                    parts *= 2
            for _ in range(parts):
                total = cascade(total, block)
            diagnostics["accepted_steps"] += parts
            diagnostics["constant_slice_steps"] += parts
            diagnostics["min_accepted_step_nm"] = min(diagnostics["min_accepted_step_nm"], h)
            diagnostics["max_accepted_step_nm"] = max(diagnostics["max_accepted_step_nm"], h)
        return total
    h = max_step
    while z < end:
        h = min(h, end - z)
        if h <= 8 * np.finfo(float).eps * max(1., abs(end)):
            break
        if integrator == "rk4":
            block = _propagator_scattering(_rk4_propagator(generator, z, h, k0, 1))
            error = 0.
        elif integrator == "cf4":
            block = _propagator_scattering(_cf4_propagator(generator, z, h, k0))
            error = 0.
        else:
            try:
                coarse = _propagator_scattering(_cf4_propagator(generator, z, h, k0))
                left = _propagator_scattering(_cf4_propagator(generator, z, .5 * h, k0))
                right = _propagator_scattering(_cf4_propagator(generator, z + .5 * h, .5 * h, k0))
                block = cascade(left, right)
                error = _scattering_error(coarse, block, rtol, atol)
            except (np.linalg.LinAlgError, LinAlgWarning, ArithmeticError, RuntimeWarning):
                error = np.inf
            if not np.isfinite(error) or error > 1.:
                diagnostics["rejected_steps"] += 1
                if h <= min_step_nm * (1 + 1e-10):
                    raise FloatingPointError(
                        f"DM local error control failed at z={z:.12g} nm, h={h:.6g} nm; "
                        "minimum step reached. Refine nfft / investigate the generator.")
                factor = max(.1, .8 * error ** (-.2)) if np.isfinite(error) else .25
                h = max(min_step_nm, h * min(.5, factor))
                continue
            diagnostics["max_accepted_error_ratio"] = max(diagnostics["max_accepted_error_ratio"], error)
        total = cascade(total, block)
        diagnostics["accepted_steps"] += 1
        diagnostics["min_accepted_step_nm"] = min(diagnostics["min_accepted_step_nm"], h)
        diagnostics["max_accepted_step_nm"] = max(diagnostics["max_accepted_step_nm"], h)
        z = min(end, z + h)
        if integrator == "cf4-adaptive":
            h = min(max_step, h * min(2., max(.5, .9 * max(error, 1e-16) ** (-.2))))
    return total


@dataclass
class Result:
    orders: np.ndarray
    r: np.ndarray
    t: np.ndarray
    R_orders: np.ndarray
    T_orders: np.ndarray
    R: float
    T: float
    A: float
    metadata: dict = field(default_factory=dict)
    fields: dict | None = None

    def reflection(self, order: int) -> float:
        return float(self.R_orders[self._index(order)])

    def transmission(self, order: int) -> float:
        return float(self.T_orders[self._index(order)])

    def _index(self, order):
        found = np.flatnonzero(self.orders == order)
        if not len(found):
            raise ValueError(f"Order {order} was not retained")
        return int(found[0])


def solve(geometry, wavelength_nm: float, angle_deg: float, max_order: int,
          method: str = "dm-fff", section_nm: float = 1., nfft: int = 4096,
          dm_substeps: int = 4, sampling: str = "entrance",
          store_fields: bool = False, dm_integrator: str = "cf4-adaptive",
          dm_rtol: float = 1e-6, dm_atol: float = 1e-9,
          dm_min_step_nm: float = 1e-5) -> Result:
    """Solve a TM grating with illumination from computational z=0.

    Geometry needs period_nm, depth_nm, n_in, n_out, sample(x,z)->epsilon,
    tangent(x)->(c,s). A zero-depth geometry represents a single interface.
    T is flux into the exit medium *at the exit plane*. In a lossy half-space
    it includes eventual substrate absorption. A refers to the finite stack.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if dm_integrator not in DM_INTEGRATORS:
        raise ValueError(f"dm_integrator must be one of {DM_INTEGRATORS}")
    if any(not np.isfinite(v) or v <= 0 for v in (dm_rtol, dm_atol, dm_min_step_nm)):
        raise ValueError("DM tolerances and minimum step must be finite and positive")
    if sampling not in ("entrance", "midpoint"):
        raise ValueError("sampling must be entrance or midpoint")
    if not isinstance(max_order, (int, np.integer)) or max_order < 0:
        raise ValueError("max_order must be a nonnegative integer")
    if wavelength_nm <= 0 or section_nm <= 0 or geometry.period_nm <= 0 or geometry.depth_nm < 0:
        raise ValueError("Invalid length or wavelength")
    if not isinstance(nfft, (int, np.integer)) or nfft <= 4 * max_order:
        raise ValueError("nfft must be an integer greater than 4*N")
    if not isinstance(dm_substeps, (int, np.integer)) or dm_substeps < 1:
        raise ValueError("dm_substeps must be a positive integer")
    if not np.isfinite(angle_deg) or abs(angle_deg) >= 90:
        raise ValueError("Incidence angle must lie strictly between -90 and 90 degrees")
    if abs(complex(geometry.n_in).imag) > 1e-12 or complex(geometry.n_in).real <= 0:
        raise ValueError("The incident half-space must have a positive real refractive index")
    generator = PropagationMatrix(geometry, wavelength_nm, angle_deg, max_order,
                                  nfft, method != "rcwa")
    size = len(generator.orders)
    k0 = 2 * np.pi / wavelength_nm
    vi = outgoing_q(geometry.n_in ** 2, generator.alpha) / geometry.n_in ** 2
    vo = outgoing_q(geometry.n_out ** 2, generator.alpha) / geometry.n_out ** 2
    unit = np.ones(size, dtype=complex)
    blocks = [_interface(vi, unit)]
    total = blocks[0]
    depth = float(geometry.depth_nm)
    count = int(np.ceil(depth / section_nm)) if depth else 0
    # Keep full-size sections, with only the last shortened when needed.
    boundaries = np.minimum(np.arange(count + 1) * section_nm, depth)
    diagnostics = {"accepted_steps": 0, "rejected_steps": 0, "constant_slice_steps": 0,
                   "min_accepted_step_nm": np.inf, "max_accepted_step_nm": 0.,
                   "max_accepted_error_ratio": 0.}
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        dz = b - a
        if method == "dm-fff":
            block = _dm_section(generator, a, dz, k0, dm_substeps, dm_integrator,
                                dm_rtol, dm_atol, dm_min_step_nm, diagnostics)
        else:
            sample = a if sampling == "entrance" else .5 * (a + b)
            p = expm(1j * k0 * dz * generator(sample))
            block = _propagator_scattering(p)
        total = cascade(total, block)
        if store_fields:
            blocks.append(block)
    exit_block = _interface(unit, vo)
    total = cascade(total, exit_block)
    incident = np.zeros(size, dtype=complex)
    incident[max_order] = 1.
    r, t = total.s11 @ incident, total.s21 @ incident
    incoming_flux = float(vi[max_order].real)
    if incoming_flux <= 0:
        raise ValueError("Incident zero order must carry positive flux")
    reflectance = np.abs(r) ** 2 * vi.real / incoming_flux
    transmittance = np.abs(t) ** 2 * vo.real / incoming_flux
    if not np.isfinite(reflectance).all() or not np.isfinite(transmittance).all():
        raise FloatingPointError("Non-finite power efficiency")
    refl, tran = float(reflectance.sum()), float(transmittance.sum())
    metadata = {"solver_revision": SOLVER_REVISION, "method": method, "wavelength_nm": wavelength_nm,
                "angle_deg": angle_deg, "N": int(max_order), "retained_orders": size,
                "nfft": int(nfft), "section_nm": section_nm, "sections": count,
                "dm_substeps": int(dm_substeps), "sampling": sampling,
                "dm_integrator": dm_integrator if method == "dm-fff" else None,
                "dm_rtol": dm_rtol if method == "dm-fff" else None,
                "dm_atol": dm_atol if method == "dm-fff" else None,
                "dm_min_step_nm": dm_min_step_nm if method == "dm-fff" else None,
                "integration_step_nm": section_nm / dm_substeps if method == "dm-fff" else section_nm,
                "precision": "complex128", "time_convention": "exp(-i omega t)",
                "power_warning": bool(refl < -1e-8 or tran < -1e-8 or refl + tran > 1 + 1e-5)}
    if method == "dm-fff":
        metadata["integration_step_note"] = "maximum trial step; exact exponentials between sampled sharp interfaces, otherwise adaptive CF4 accepts two half steps" if dm_integrator == "cf4-adaptive" else "fixed internal step"
        metadata["integration_diagnostics"] = diagnostics
    fields = None
    if store_fields:
        blocks.append(exit_block)
        # Suffix at each cut, built without propagating a growing global transfer.
        suffixes = [None] * len(blocks)
        suffix = transparent(size)
        for j in range(len(blocks) - 1, -1, -1):
            suffixes[j] = suffix
            suffix = cascade(blocks[j], suffix)
        prefix = transparent(size)
        ex, hy = [], []
        for j, block in enumerate(blocks[:-1]):
            prefix = cascade(prefix, block)
            right = suffixes[j]
            forward = _solve(np.eye(size) - prefix.s22 @ right.s11,
                             prefix.s21 @ incident)
            backward = right.s11 @ forward
            ex.append(forward - backward)
            hy.append(forward + backward)
        fields = {"z_nm": boundaries, "Ex_coeff": np.asarray(ex), "h_coeff": np.asarray(hy)}
    return Result(generator.orders, r, t, reflectance, transmittance,
                  refl, tran, 1 - refl - tran, metadata, fields)
