"""Complex128 tensor kernels for CUDA and for CPU parity validation.

Geometry/quadrature coefficients are prepared once on the host, using the
same definitions as SciPy. Matrix inverses, eigenproblems, projections and
interface/Redheffer products use the selected torch device. No large matrix
is moved back to CPU during a wavelength solve. Only final output vectors
and scalar diagnostics are transferred. PyTorch is an optional dependency.
"""

from __future__ import annotations

import numpy as np
import torch

from .solver import _PreparedLayer, _propagate


def _sqrt_outgoing(values):
    gamma = torch.sqrt(values.to(torch.complex128))
    return torch.where(gamma.real + gamma.imag < 0, -gamma, gamma)


class TorchPreparedLayer:
    def __init__(self, spec, method, count, internal_count, quadrature, G, *, device):
        host = _PreparedLayer(spec, method, count, internal_count, quadrature, G,
                              prepare_linear_algebra=False)
        self.device = torch.device(device)
        for name in ("spec", "method", "count", "internal_count", "homogeneous",
                     "quadrature_points", "lossless", "positive_epsilon"):
            setattr(self, name, getattr(host, name))
        self.orders = torch.as_tensor(host.orders, device=device, dtype=torch.float64)
        self.rayleigh_orders = torch.as_tensor(host.rayleigh_orders, device=device, dtype=torch.float64)
        self.identity = torch.eye(self.internal_count, dtype=torch.complex128, device=device)
        if self.homogeneous:
            self.epsilon_value = host.epsilon_value
            return
        for name in ("f", "a", "b", "inverse_epsilon", "K0", "KQ0"):
            setattr(self, name, torch.as_tensor(getattr(host, name), device=device, dtype=torch.complex128))
        if method == "asr":
            for name in ("u", "x", "resolution", "weights", "epsilon_samples"):
                dtype = torch.complex128 if name == "epsilon_samples" else torch.float64
                setattr(self, name, torch.as_tensor(getattr(host, name), device=device, dtype=dtype))
        self.inverse_f = torch.linalg.solve(self.f, self.identity)
        self.inverse_a = torch.linalg.solve(self.a, self.identity)
        self.metric_factors = {"f": torch.linalg.lu_factor(self.f),
                               "b": torch.linalg.lu_factor(self.b)}
        self.cholesky = {}
        if self.lossless:
            self.cholesky["f"] = torch.linalg.cholesky((self.f + self.f.mH) / 2)
            if self.positive_epsilon:
                self.cholesky["b"] = torch.linalg.cholesky((self.b + self.b.mH) / 2)

    def _projection(self, alpha0, reciprocal=False):
        alpha_p = alpha0 + 2 * np.pi * self.rayleigh_orders / self.spec.period
        alpha_m = alpha0 + 2 * np.pi * self.orders / self.spec.period
        physical = torch.exp(-1j * alpha_p[:, None] * self.x[None, :])
        adaptive = torch.exp(1j * self.u[:, None] * alpha_m[None, :])
        weights = self.weights * self.resolution
        if reciprocal:
            weights = weights / self.epsilon_samples
        return (physical * weights[None, :]) @ adaptive

    def modes(self, k, alpha0, polarization, retention="smallest_abs", q_projection="direct"):
        alpha = alpha0 + 2 * np.pi * self.orders / self.spec.period
        if self.homogeneous:
            gamma = _sqrt_outgoing(k * k * self.epsilon_value - alpha * alpha)
            return self.identity, torch.diag(gamma if polarization == "TE" else gamma / self.epsilon_value), gamma
        hermitian = self.lossless and abs(complex(alpha0).imag) < 1e-14
        if polarization == "TE":
            operator = k * k * self.a - alpha[:, None] * self.inverse_f * alpha[None, :]
            metric_name = "f"
        else:
            operator = k * k * self.f - alpha[:, None] * self.inverse_a * alpha[None, :]
            metric_name = "b"
            hermitian = hermitian and self.positive_epsilon
        if self.method == "fmm" and polarization == "TE":
            eigenvalues, eigenvectors = (torch.linalg.eigh((operator + operator.mH) / 2)
                                        if hermitian else torch.linalg.eig(operator))
        elif hermitian:
            # A v = lambda M v; M=L L^H. Solve the Hermitian problem
            # L^-1 A L^-H y=lambda y, then v=L^-H y. Never use eig(M^-1 A)
            # for a lossless metric: it would discard its Hermitian structure.
            L = self.cholesky[metric_name]
            operator = (operator + operator.mH) / 2
            left = torch.linalg.solve_triangular(L, operator, upper=False)
            transformed = torch.linalg.solve_triangular(L, left.mH, upper=False).mH
            eigenvalues, vectors = torch.linalg.eigh((transformed + transformed.mH) / 2)
            eigenvectors = torch.linalg.solve_triangular(L.mH, vectors, upper=True)
        else:
            standard = torch.linalg.lu_solve(*self.metric_factors[metric_name], operator)
            eigenvalues, eigenvectors = torch.linalg.eig(standard)
        if self.internal_count > self.count:
            if retention == "smallest_abs":
                keep = torch.argsort(torch.abs(eigenvalues), stable=True)[:self.count]
            else:
                roots = _sqrt_outgoing(eigenvalues)
                values = eigenvalues.to(torch.complex128)
                propagating = (values.real > 0) & (torch.abs(values.imag) < 1e-8 * torch.clamp(torch.abs(values), min=1))
                sort_value = torch.where(propagating, -values.real, torch.abs(roots.imag))
                keep = torch.argsort(sort_value, stable=True)
                keep = keep[torch.argsort((~propagating[keep]).to(torch.int64), stable=True)][:self.count]
            eigenvalues, eigenvectors = eigenvalues[keep], eigenvectors[:, keep]
        gamma = _sqrt_outgoing(eigenvalues)
        K = self.K0 if alpha0 == 0 or self.method == "fmm" else self._projection(alpha0)
        W = K @ eigenvectors
        if polarization == "TE":
            Q = W
        elif q_projection == "laurent":
            Q = K @ (self.inverse_epsilon @ eigenvectors)
        else:
            KQ = self.KQ0 if alpha0 == 0 or self.method == "fmm" else self._projection(alpha0, reciprocal=True)
            Q = KQ @ eigenvectors
        scales = torch.clamp(torch.linalg.vector_norm(W, dim=0), min=1e-30)
        return W / scales[None, :], Q / scales[None, :] * gamma[None, :], gamma


class _Diagnostics:
    def __init__(self, enabled, identity):
        self.enabled, self.identity = enabled, identity
        self.max_condition = torch.ones((), dtype=torch.float64, device=identity.device)
        self.solves = 0

    def solve(self, A, B):
        self.solves += 1
        factors, pivots, info = torch.linalg.lu_factor_ex(A, check_errors=False)
        if info.item() != 0:
            raise np.linalg.LinAlgError("singular interface; increase retained harmonics or avoid an exact modal cutoff")
        result = torch.linalg.lu_solve(factors, pivots, B)
        if self.enabled:
            # Reuse LU. The exact 1-norm need not equal SciPy's inexpensive
            # LAPACK estimate; both diagnostics are labeled in metadata.
            identity = self.identity if A.shape == self.identity.shape else torch.eye(
                A.shape[0], dtype=A.dtype, device=A.device)
            inverse = torch.linalg.lu_solve(factors, pivots, identity)
            condition = torch.abs(A).sum(dim=0).amax() * torch.abs(inverse).sum(dim=0).amax()
            self.max_condition = torch.maximum(self.max_condition, condition)
        return result


def _interface(Wl, Vl, Wr, Vr, diagnostics):
    count = len(Wl)
    A = torch.cat((torch.cat((Wl, -Wr), dim=1), torch.cat((-Vl, -Vr), dim=1)), dim=0)
    B = torch.cat((torch.cat((-Wl, Wr), dim=1), torch.cat((-Vl, -Vr), dim=1)), dim=0)
    S = diagnostics.solve(A, B)
    return S[:count, :count], S[:count, count:], S[count:, :count], S[count:, count:]


def _star(A, B, diagnostics):
    A11, A12, A21, A22 = A
    B11, B12, B21, B22 = B
    count = len(A11)
    joined = diagnostics.solve(diagnostics.identity - B11 @ A22, torch.cat((B11 @ A21, B12), dim=1))
    left, right = joined[:, :count], joined[:, count:]
    return (A11 + A12 @ left, A12 @ right,
            B21 @ (A21 + A22 @ left), B22 + B21 @ A22 @ right)


@torch.no_grad()
def solve_stack(stack, wavelength, *, cutoff_shift=1e-12):
    requested, wavelength, k, alpha0, at_cutoff = stack._wavelength_parameters(wavelength, cutoff_shift)
    count, device = stack.harmonics, stack.device
    orders = stack.prepared_layers[0].rayleigh_orders
    identity = torch.eye(count, dtype=torch.complex128, device=device)

    def external(epsilon):
        alpha = alpha0 + 2 * np.pi * orders / stack.period
        gamma = _sqrt_outgoing(k * k * epsilon - alpha * alpha)
        return identity, torch.diag(gamma if stack.polarization == "TE" else gamma / epsilon), gamma

    Wleft, Vleft, gamma_in = external(stack.epsilon_in)
    Wout, Vout, gamma_out = external(stack.epsilon_out)
    diagnostics = _Diagnostics(stack.diagnostics, identity)
    cached_modes, scattering = {}, None
    for spec, prepared in zip(stack.layers, stack.prepared_layers):
        if id(prepared) not in cached_modes:
            cached_modes[id(prepared)] = prepared.modes(k, alpha0, stack.polarization,
                                                       stack.retention, stack.q_projection)
        W, V, gamma = cached_modes[id(prepared)]
        interface = _interface(Wleft, Vleft, W, V, diagnostics)
        scattering = interface if scattering is None else _star(scattering, interface, diagnostics)
        scattering = _propagate(scattering, torch.exp(1j * gamma * spec.thickness))
        Wleft, Vleft = W, V
    scattering = _star(scattering, _interface(Wleft, Vleft, Wout, Vout, diagnostics), diagnostics)
    zeroth = int(np.flatnonzero(stack.orders == 0)[0])
    reflected, transmitted = scattering[0][:, zeroth], scattering[2][:, zeroth]
    admittance_in = gamma_in if stack.polarization == "TE" else gamma_in / stack.epsilon_in
    admittance_out = gamma_out if stack.polarization == "TE" else gamma_out / stack.epsilon_out
    incident_flux = admittance_in[zeroth].real
    if incident_flux.item() <= 0:
        raise ValueError("the incident zeroth order must carry positive forward flux")
    R_orders = admittance_in.real / incident_flux * torch.abs(reflected)**2
    T_orders = admittance_out.real / incident_flux * torch.abs(transmitted)**2
    R, T, T0 = R_orders.sum().item(), T_orders.sum().item(), T_orders[zeroth].item()
    return {
        "wavelength": requested, "evaluation_wavelength": wavelength, "cutoff_regularized": bool(at_cutoff),
        "T0": T0, "eta0": T0, "T": T, "R": R, "A": 1 - R - T,
        "T_orders": T_orders.cpu().numpy(), "R_orders": R_orders.cpu().numpy(),
        "t": transmitted.cpu().numpy(), "r": reflected.cpu().numpy(), "orders": stack.orders.copy(),
        "polarization": stack.polarization, "method": stack.method,
        "retention": stack.retention, "q_projection": stack.q_projection,
        "retained_modes": count,
        "eigen_dimensions": np.asarray([layer.internal_count for layer in stack.prepared_layers]),
        "max_boundary_condition": diagnostics.max_condition.item(), "linear_solves": diagnostics.solves,
        "quadrature_points_per_region": max(layer.quadrature_points for layer in stack.prepared_layers),
        "device": device, "backend": stack.backend, "execution": stack.execution.copy(),
    }
