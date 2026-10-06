"""Cartesian Li RCWA for a true x-periodic, y-invariant metal grating.

The fixed-geometry study uses the existing eigensolver, port definitions and
Redheffer cascade. Fourier coefficients are either sampled at cell midpoints
or evaluated analytically for each rectangular slice. No coordinate transform
is required, so there is no inversion of an ASR conversion matrix.
"""
from __future__ import annotations

import time
import torch
from rcwa_ext.auto import AutoRCWA
from rcwa_ext.config import GroupTheoryOptions, Lattice, OutputSpec


class Grating1D(AutoRCWA):
    def _kvectors(self):
        # This adapter has an orthogonal cell. math.cos(pi/2) leaves ~6e-17,
        # which the oblique-cell formula amplifies by |m|*wavelength/period.
        # Set the exact geometry BEFORE building wavevectors and port matrices.
        if self.zeta_deg != 90.0:
            raise ValueError("This 1D study requires an exactly orthogonal cell.")
        self.cos_zeta, self.sin_zeta = 0.0, 1.0
        return super()._kvectors()

    def ridge_convolutions(self, fill, epsilon, grid, *, analytic=False):
        if not 0 < fill < 1 or len(self.order_y) != 1:
            raise ValueError("Require 0 < width < period, M_y=0, and y-invariant material.")
        delta = (self.order_x[:, None]-self.order_x[None, :]).to(torch.float64)
        if analytic:
            # Integral over the centered interval [(1-fill)/2,(1+fill)/2].
            indicator = fill*torch.sinc(delta*fill)*torch.exp(-1j*torch.pi*delta)
            sampled_fill = None
        else:
            self._validate_grid(int(self.order[0]), grid, "x")
            x = (torch.arange(grid, dtype=torch.float64, device=self._device)+0.5)/grid
            samples = (torch.abs(x-0.5) < fill/2).to(torch.float64)
            coefficients = torch.fft.fft(samples)/grid
            # FFT assumes samples at j/N; our quadrature points are (j+1/2)/N.
            indices = delta.to(torch.long)
            indicator = coefficients[indices]*torch.exp(-1j*torch.pi*delta/grid)
            sampled_fill = float(samples.mean().item())
        identity = self._eye(self.order_N)
        direct = identity+(epsilon-1)*indicator
        reciprocal = identity+(1/epsilon-1)*indicator
        return direct, reciprocal, sampled_fill

    def add_ridge(self, thickness, fill, epsilon, grid, *, audit=False, analytic=False):
        self._require_kvectors()
        direct, reciprocal, sampled_fill = self.ridge_convolutions(
            fill, epsilon, grid, analytic=analytic)
        identity = self._eye(self.order_N)
        inverse_direct = self._solve(direct, identity)
        # Li inverse rule for the normal displacement; direct rule for E_y/E_z.
        normal = self._solve(reciprocal, identity)
        kx = torch.diagonal(self.Kx_norm)
        if bool(torch.any(torch.abs(self.Ky_norm) > 1e-14)):
            raise ValueError("This study implements normal incidence and K_y=0 only.")
        zeros = torch.zeros_like(identity)
        p12 = identity-kx[:, None]*inverse_direct*kx[None, :]
        q12 = torch.diag(kx*kx)-direct
        p = torch.cat((torch.cat((zeros, p12), 1),
                       torch.cat((-identity, zeros), 1)), 0)
        q = torch.cat((torch.cat((zeros, q12), 1),
                       torch.cat((normal, zeros), 1)), 0)
        eigenvalues, electric = self._eig(p @ q)
        kz = self._positive_kz(eigenvalues)
        magnetic = self._magnetic_eigenvectors(p, q, electric, kz)
        self.layer_N += 1
        self.thickness.append(torch.as_tensor(thickness, dtype=torch.float64,
                                             device=self._device))
        self.eps_conv.append(direct)
        self.mu_conv.append(identity)
        self.P.append(p)
        self.Q.append(q)
        self.kz_norm.append(kz)
        self.E_eigvec.append(electric)
        self.H_eigvec.append(magnetic)
        self._append_smatrix_from_cartesian_modes(electric, magnetic)
        details = {"sampled_fill_fraction": sampled_fill,
                   "fill_fraction_error": None if analytic else sampled_fill-fill}
        if audit:
            def condition(matrix):
                singular = torch.linalg.svdvals(matrix)
                if not bool(torch.all(torch.isfinite(singular))) or float(singular[-1]) <= 0:
                    raise RuntimeError("Singular or nonfinite 1D operator/modal matrix.")
                return float((singular[0]/singular[-1]).item())
            product = p @ q @ electric
            residual = torch.linalg.norm(product-electric*eigenvalues[None, :])/torch.linalg.norm(product)
            details.update(normal_epsilon_reciprocal_condition=condition(reciprocal),
                           electric_eigenvector_condition=condition(electric),
                           magnetic_eigenvector_condition=condition(magnetic),
                           relative_eigen_residual=float(residual.item()))
        return details


def _simulation(order, wavelength, period, epsilon, device):
    sim = Grating1D(freq=period/wavelength, order=[order, 0],
                    lattice=Lattice.square(1.0), cascade="redheffer",
                    outputs=OutputSpec(smatrix_size="half", fields="none"),
                    group_theory=GroupTheoryOptions(enabled=False),
                    dtype=torch.complex128, device=device)
    sim.add_input_layer(eps=1.0, mu=1.0)
    sim.add_output_layer(eps=epsilon, mu=1.0)
    sim.set_incident_angle(0.0, 0.0)
    return sim


def port_results(sim):
    sim.solve_global_smatrix()
    if "Tf" not in sim.computed_smatrix_blocks:
        raise RuntimeError("The forward transmission block is required for substrate flux.")
    n = sim.order_N
    zero_order = int(torch.nonzero(sim.order_x == 0)[0].item())
    def flux(e, v, direction):
        h = direction*(v @ e)
        return float((0.5*torch.real(torch.sum(
            e[:n]*h[n:].conj()-e[n:]*h[:n].conj()))).item())
    result = {}
    for pol, offset in (("TE", n), ("TM", 0)):
        source = torch.zeros(2*n, dtype=sim._dtype, device=sim._device)
        source[zero_order+offset] = 1
        incident = flux(source, sim.Vi, 1)
        if incident <= 0:
            raise RuntimeError("Nonpositive incident flux.")
        r = -flux(sim.S[1] @ source, sim.Vi, -1)/incident
        p = flux(sim.S[0] @ source, sim.Vo, 1)/incident
        result[pol] = {"reflectance": r, "power_into_substrate": p,
                       "relief_absorptance": 1-r-p, "absorptance_total": 1-r}
    return result


def simulate(config, numbers, wavelength, epsilon, requested_device):
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu explicitly.")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    geometry = config["geometry"]
    period, nz = geometry["period_nm"], numbers["slices"]
    analytic = config["solver"]["fourier_coefficients"] == "analytic"
    audits, max_fill_error = [], 0.0
    with torch.no_grad():
        sim = _simulation(numbers["order"], wavelength, period, epsilon, device)
        selected = {0, (nz-1)//2, nz-1}
        for layer in range(nz):
            s = (layer+0.5)/nz
            width = geometry["top_width_nm"]+(
                geometry["bottom_width_nm"]-geometry["top_width_nm"])*s**geometry["profile_power"]
            details = sim.add_ridge(geometry["height_nm"]/nz/period, width/period,
                                    epsilon, numbers["grid"], audit=layer in selected,
                                    analytic=analytic)
            if not analytic:
                max_fill_error = max(max_fill_error, abs(details["fill_fraction_error"]))
            if layer in selected:
                audits.append(dict(details, layer=layer+1, width_nm=width))
        observables = port_results(sim)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = dict(numbers, wavelength_nm=wavelength,
                  coefficient_method="analytic" if analytic else "sampled-midpoints",
                  grid_used=None if analytic else numbers["grid"],
                  epsilon_gold_real=epsilon.real, epsilon_gold_imag=epsilon.imag,
                  polarizations=observables, runtime_seconds=time.perf_counter()-started,
                  maximum_fill_fraction_error_all_layers=None if analytic else max_fill_error,
                  selected_layer_diagnostics=audits,
                  fourier_orders_x=2*numbers["order"]+1, fourier_orders_y=1,
                  environment={"torch": torch.__version__, "device": str(device),
                               "cuda": torch.version.cuda})
    if device.type == "cuda":
        result.update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device))
        result["environment"]["gpu"] = torch.cuda.get_device_name(device)
    result["transmittance_far"] = 0.0
    return result
