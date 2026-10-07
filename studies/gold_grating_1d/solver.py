"""Cartesian Li RCWA for a true x-periodic, y-invariant metal grating.

Scattering-only Redheffer calculations separate TE and TM throughout the
eigensolve, layer scattering, cascade, and port flux. Each polarization uses
N=2*M+1 orders; layers are cascaded immediately instead of retaining their
operators and modes. The original Cartesian path remains available for field
recovery and explicit parity checks with polarization_separated=False.
"""
from __future__ import annotations

import time
import torch
from rcwa_ext.auto import AutoRCWA
from rcwa_ext.config import GroupTheoryOptions, Lattice, OutputSpec


class Grating1D(AutoRCWA):
    def __init__(self, *args, polarization_separated=True, **kwargs):
        self._separated_stack_s = None
        self._separated_global_s = None
        self._separated_public_s = None
        self._separated_last_operators = None
        self._legacy_s = None
        super().__init__(*args, **kwargs)
        self.polarization_separated = bool(polarization_separated)
        self._separated_enabled = (
            self.polarization_separated and not self.store_mode_couplings
            and self.smatrix_algorithm == "redheffer"
            and not self.use_group_theory and len(self.order_y) == 1)

    @property
    def S(self):
        """Materialize Cartesian public blocks only when a caller needs them."""
        if self._separated_global_s is not None:
            if self._separated_public_s is None:
                # Internal batch order is TE, TM; Cartesian order is Ex, Ey.
                self._separated_public_s = [
                    torch.block_diag(block[1], block[0])
                    for block in self._separated_global_s]
            return self._separated_public_s
        if self._legacy_s is None:
            raise AttributeError("Solve the global scattering matrix first.")
        return self._legacy_s

    @S.setter
    def S(self, blocks):
        self._legacy_s = blocks
        self._separated_global_s = self._separated_public_s = None

    def _polarized_admittance(self, port):
        n = self.order_N
        # Both scalar magnetic variables carry positive forward power:
        # TE uses -Hx, TM uses Hy.
        return torch.stack((-torch.diagonal(port[:n, n:]),
                             torch.diagonal(port[n:, :n])))

    def _polarized_interface(self, blocks):
        n = self.order_N
        return [torch.stack((block[n:, n:], block[:n, :n]))
                for block in blocks]

    def _polarized_port_interface(self, name):
        blocks = getattr(self, name, None)
        if blocks is not None:
            return self._polarized_interface(blocks)
        identity = self._eye(self.order_N).expand(2, -1, -1)
        zero = torch.zeros_like(identity)
        return [identity, zero, zero, identity]

    def _polarized_connect(self, left, right, *, forward_only=False):
        """Redheffer star product on two independent N-by-N batches."""
        tf_l, rf_l, rb_l, tb_l = left
        tf_r, rf_r, rb_r, tb_r = right
        identity = self._eye(self.order_N)
        denominator_lr = identity-rb_l @ rf_r
        denominator_rl = identity-rf_r @ rb_l
        rhs_lr, rhs_rl = tf_l, rf_r @ tf_l
        if not forward_only:
            rhs_lr = torch.cat((rhs_lr, rb_l @ tb_r), -1)
            rhs_rl = torch.cat((rhs_rl, tb_r), -1)
        solved_lr = self._solve(denominator_lr, rhs_lr)
        solved_rl = self._solve(denominator_rl, rhs_rl)
        n = self.order_N
        transmission = tf_r @ solved_lr[..., :n]
        reflection = rf_l+tb_l @ solved_rl[..., :n]
        if forward_only:
            zero = torch.zeros_like(reflection)
            return [transmission, reflection, zero, zero]
        return [transmission, reflection,
                rb_r+tf_r @ solved_lr[..., n:],
                tb_l @ solved_rl[..., n:]]

    def _polarized_layer_smatrix(self, electric, magnetic, kz, thickness):
        identity = self._eye(self.order_N)
        reference = torch.diag_embed(self._polarized_admittance(self.Vf))
        inverse_electric = self._solve(electric, identity)
        inverse_magnetic_reference = self._solve(magnetic, reference)
        a = inverse_electric+inverse_magnetic_reference
        b = inverse_electric-inverse_magnetic_reference
        phase = torch.exp(1j*self.omega*kz*thickness)[..., :, None]
        xb, xa = phase*b, phase*a
        # Factor each coefficient matrix once for all its right-hand sides.
        ab, axb, axa = self._solve(a, torch.cat((b, xb, xa), -1)).chunk(3, -1)
        core = a-xb @ axb
        rhs_t = phase*(a-b @ ab)
        rhs_r = xb @ axa-b
        transmission, reflection = self._solve(
            core, torch.cat((rhs_t, rhs_r), -1)).chunk(2, -1)
        return [transmission, reflection, reflection, transmission]

    def _append_polarized_layer(self, blocks, thickness):
        if self._separated_stack_s is None:
            self._separated_stack_s = self._polarized_port_interface("Sin")
        self._separated_stack_s = self._polarized_connect(
            self._separated_stack_s, blocks)
        self.layer_N += 1
        self.thickness.append(thickness)
        self._separated_global_s = self._separated_public_s = None

    def _add_ridge_separated(self, thickness, direct, reciprocal, *, audit):
        identity = self._eye(self.order_N)
        inverse_direct, normal = self._solve(
            torch.stack((direct, reciprocal)), identity).unbind(0)
        kx = torch.diagonal(self.Kx_norm)
        p_tm = identity-kx[:, None]*inverse_direct*kx[None, :]
        q_te = direct-torch.diag(kx*kx)
        operators = torch.stack((q_te, p_tm @ normal))
        if getattr(self, "stable_eig_grad", False):
            # Custom torcwa eig-grad implementations can require 2D inputs.
            pairs = [self._eig(operator) for operator in operators]
            eigenvalues = torch.stack([pair[0] for pair in pairs])
            electric = torch.stack([pair[1] for pair in pairs])
        else:
            eigenvalues, electric = self._eig(operators)
        kz = self._positive_kz(eigenvalues)
        magnetic_te = electric[0]*kz[0][None, :]
        magnetic_tm = self._magnetic_eigenvectors(
            p_tm, normal, electric[1], kz[1])
        magnetic = torch.stack((magnetic_te, magnetic_tm))
        thickness = torch.as_tensor(thickness, dtype=torch.float64,
                                    device=self._device)
        self._append_polarized_layer(self._polarized_layer_smatrix(
            electric, magnetic, kz, thickness), thickness)
        # Only the last operators are retained for explicit diagnostics.
        self._separated_last_operators = (p_tm, -q_te, normal)
        details = {"polarization_separated": True,
                   "eigenproblem_dimension_per_polarization": self.order_N}
        if audit:
            singular_e = torch.linalg.svdvals(electric)
            singular_h = torch.linalg.svdvals(magnetic)
            singular_b = torch.linalg.svdvals(reciprocal)
            for singular in (singular_e, singular_h, singular_b):
                if (not bool(torch.all(torch.isfinite(singular)))
                        or float(torch.min(singular)) <= 0):
                    raise RuntimeError("Singular or nonfinite 1D operator/modal matrix.")
            product = operators @ electric
            difference = product-electric*eigenvalues[:, None, :]
            residuals = (torch.linalg.matrix_norm(difference)
                         / torch.linalg.matrix_norm(product))
            details.update(
                normal_epsilon_reciprocal_condition=float(
                    (singular_b.max()/singular_b.min()).item()),
                electric_eigenvector_condition=float(
                    (singular_e.max()/singular_e.min()).item()),
                magnetic_eigenvector_condition=float(
                    (singular_h.max()/singular_h.min()).item()),
                relative_eigen_residual=float(
                    (torch.linalg.norm(difference)/torch.linalg.norm(product)).item()),
                polarization_diagnostics={pol: {
                    "electric_eigenvector_condition": float(
                        (singular_e[index, 0]/singular_e[index, -1]).item()),
                    "magnetic_eigenvector_condition": float(
                        (singular_h[index, 0]/singular_h[index, -1]).item()),
                    "relative_eigen_residual": float(residuals[index].item())}
                    for index, pol in enumerate(("TE", "TM"))})
        return details

    def cartesian_operators(self):
        """Reconstruct the last P/Q only for an explicit operator check."""
        if not self._separated_enabled:
            return self.P[-1], self.Q[-1]
        if self._separated_last_operators is None:
            raise RuntimeError("No rectangular layer operators are available.")
        p_tm, q12, normal = self._separated_last_operators
        identity = self._eye(self.order_N)
        zero = torch.zeros_like(identity)
        return (torch.cat((torch.cat((zero, p_tm), 1),
                           torch.cat((-identity, zero), 1)), 0),
                torch.cat((torch.cat((zero, q12), 1),
                           torch.cat((normal, zero), 1)), 0))

    def add_layer(self, thickness, eps=1.0, mu=1.0):
        if not self._separated_enabled:
            return super().add_layer(thickness, eps=eps, mu=mu)
        self._require_kvectors()
        if bool(torch.any(torch.abs(self.Ky_norm) > 1e-14)):
            raise ValueError("Separated 1D layers require K_y=0.")
        eps = torch.as_tensor(eps, dtype=self._dtype, device=self._device)
        mu = torch.as_tensor(mu, dtype=self._dtype, device=self._device)
        if eps.numel() != 1 or mu.numel() != 1:
            if self.layer_N:
                raise ValueError("For raster/mixed layers, construct with polarization_separated=False.")
            self._separated_enabled = False
            self._separated_global_s = self._separated_public_s = None
            return super().add_layer(thickness, eps=eps, mu=mu)
        eps, mu = eps.reshape(()), mu.reshape(())
        thickness = torch.as_tensor(thickness, dtype=torch.float64,
                                    device=self._device)
        if (not bool(torch.isfinite(eps)) or not bool(torch.isfinite(mu))
                or not bool(torch.isfinite(thickness)) or float(thickness) < 0):
            raise ValueError("Require finite material parameters and nonnegative thickness.")
        kz = self._positive_kz(eps*mu-self.Kx_norm_dn**2)
        safe_kz = torch.where(torch.abs(kz) < 1e-12,
                              kz+(1e-10+1e-10j), kz)
        admittance = torch.stack((safe_kz/mu,
                                  (safe_kz+self.Kx_norm_dn**2/safe_kz)/mu))
        reference = self._polarized_admittance(self.Vf)
        reflection = (reference-admittance)/(reference+admittance)
        phase = torch.exp(1j*self.omega*kz*thickness)
        denominator = 1-reflection**2*phase**2
        tf = torch.diag_embed((1-reflection**2)*phase/denominator)
        rf = torch.diag_embed(reflection*(1-phase**2)/denominator)
        self._append_polarized_layer([tf, rf, rf, tf], thickness)
        self._separated_last_operators = None

    def solve_polarized_smatrix(self):
        if not self._separated_enabled:
            raise RuntimeError("This simulation uses the Cartesian scattering path.")
        self._require_kvectors()
        if bool(torch.any(torch.abs(self.Ky_norm) > 1e-14)):
            raise ValueError("Separated 1D scattering requires K_y=0.")
        if self.store_mode_couplings != self._initial_store_mode_couplings:
            raise RuntimeError("Field-storage mode is fixed at construction time.")
        if self._separated_global_s is None:
            stack = self._separated_stack_s
            if stack is None:
                stack = self._polarized_port_interface("Sin")
            self._separated_global_s = self._polarized_connect(
                stack, self._polarized_port_interface("Sout"),
                forward_only=self.smatrix_size != "full")
            if self.smatrix_size == "quarter":
                self._separated_global_s[0] = torch.zeros_like(self._separated_global_s[1])
            self._separated_public_s = None
            self.C = [[], []]
            self.cascade_diagnostics = {
                "algorithm": "redheffer", "size": self.smatrix_size,
                "computed_blocks": self.computed_smatrix_blocks,
                "polarization_separated": True, "streaming_layers": True,
                "matrix_dimension_per_polarization": self.order_N}
        return self._separated_global_s

    def solve_global_smatrix(self):
        if self._separated_enabled:
            self.solve_polarized_smatrix()
        else:
            super().solve_global_smatrix()

    def _kvectors(self):
        # This adapter has an orthogonal cell. math.cos(pi/2) leaves ~6e-17,
        # which the oblique-cell formula amplifies by |m|*wavelength/period.
        # Set the exact geometry BEFORE building wavevectors and port matrices.
        if self.zeta_deg != 90.0:
            raise ValueError("This 1D study requires an exactly orthogonal cell.")
        self.cos_zeta, self.sin_zeta = 0.0, 1.0
        result = super()._kvectors()
        self._separated_stack_s = None
        self._separated_global_s = self._separated_public_s = None
        self._legacy_s = None
        return result

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
        if bool(torch.any(torch.abs(self.Ky_norm) > 1e-14)):
            raise ValueError("This study implements normal incidence and K_y=0 only.")
        direct, reciprocal, sampled_fill = self.ridge_convolutions(
            fill, epsilon, grid, analytic=analytic)
        if self._separated_enabled:
            thickness_value = torch.as_tensor(thickness, dtype=torch.float64,
                                              device=self._device)
            if not bool(torch.isfinite(thickness_value)) or float(thickness_value) < 0:
                raise ValueError("Require finite nonnegative thickness.")
            details = self._add_ridge_separated(
                thickness_value, direct, reciprocal, audit=audit)
            details.update(sampled_fill_fraction=sampled_fill,
                           fill_fraction_error=None if analytic else sampled_fill-fill)
            return details
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


def _simulation(order, wavelength, period, epsilon, device, *, polarization_separated=True):
    sim = Grating1D(freq=period/wavelength, order=[order, 0],
                    lattice=Lattice.square(1.0), cascade="redheffer",
                    outputs=OutputSpec(smatrix_size="half", fields="none"),
                    group_theory=GroupTheoryOptions(enabled=False),
                    polarization_separated=polarization_separated,
                    dtype=torch.complex128, device=device)
    sim.add_input_layer(eps=1.0, mu=1.0)
    sim.add_output_layer(eps=epsilon, mu=1.0)
    sim.set_incident_angle(0.0, 0.0)
    return sim


def port_results(sim):
    if getattr(sim, "_separated_enabled", False):
        blocks = sim.solve_polarized_smatrix()
        if "Tf" not in sim.computed_smatrix_blocks:
            raise RuntimeError("The forward transmission block is required for substrate flux.")
        zero_order = int(torch.nonzero(sim.order_x == 0)[0].item())
        input_y = sim._polarized_admittance(getattr(sim, "Vi", sim.Vf))
        output_y = sim._polarized_admittance(getattr(sim, "Vo", sim.Vf))
        incident = 0.5*input_y[:, zero_order].real
        if bool(torch.any(incident <= 0)):
            raise RuntimeError("Nonpositive incident flux.")
        def flux(electric, admittance):
            return 0.5*torch.real(torch.sum(
                electric*(admittance*electric).conj(), -1))
        reflectance = flux(blocks[1][:, :, zero_order], input_y)/incident
        substrate = flux(blocks[0][:, :, zero_order], output_y)/incident
        return {pol: {"reflectance": float(reflectance[index].item()),
                      "power_into_substrate": float(substrate[index].item()),
                      "relief_absorptance": float((1-reflectance[index]-substrate[index]).item()),
                      "absorptance_total": float((1-reflectance[index]).item())}
                for index, pol in enumerate(("TE", "TM"))}
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
        sim = _simulation(numbers["order"], wavelength, period, epsilon, device,
                          polarization_separated=config["solver"].get("polarization_separated", True))
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
                  polarization_separated=sim._separated_enabled,
                  eigenproblem_dimension_per_polarization=(
                      sim.order_N if sim._separated_enabled else None),
                  environment={"torch": torch.__version__, "device": str(device),
                               "cuda": torch.version.cuda})
    if device.type == "cuda":
        result.update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device))
        result["environment"]["gpu"] = torch.cuda.get_device_name(device)
    result["transmittance_far"] = 0.0
    return result
