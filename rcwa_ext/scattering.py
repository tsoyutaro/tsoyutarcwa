"""Shared modal algebra, stable solves, cascades, and field recovery."""

from __future__ import annotations

from typing import Sequence

import torch
import torcwa

from .config import (
    LayerRecord, UnsupportedCombinationError, _ORIGINAL_TORCWA_RCWA,
    _TWO_PI, _as_float, _finite_positive, _normalize_cascade,
    _normalize_smatrix_size, _real_parameter_tensor,
)

from .fields import _FieldRecoveryMixin

class _StableLinearAlgebraMixin(_FieldRecoveryMixin):
    """Small helpers that retain torcwa's output dtype but solve in complex128."""

    def __init__(self, *args, **kwargs):
        if len(args) < 3:
            raise TypeError("RCWA initialization requires freq, order, and L.")
        _finite_positive("freq", args[0])
        order_value = tuple(args[1])
        if len(order_value) != 2 or any(
            isinstance(v, bool) or int(v) != v or int(v) < 0 for v in order_value
        ):
            raise ValueError("order must contain two nonnegative integers.")
        lattice_lengths = tuple(args[2])
        if len(lattice_lengths) != 2:
            raise ValueError("L must contain two lattice lengths.")
        _finite_positive("L[0]", lattice_lengths[0])
        _finite_positive("L[1]", lattice_lengths[1])

        smatrix_size = kwargs.pop("smatrix_size", None)
        smatrix_output = kwargs.pop("smatrix_output", None)
        if smatrix_size is not None and smatrix_output is not None:
            if _normalize_smatrix_size(smatrix_size) != _normalize_smatrix_size(
                smatrix_output
            ):
                raise ValueError("smatrix_size and smatrix_output disagree.")
        self.smatrix_size = _normalize_smatrix_size(
            smatrix_size
            if smatrix_size is not None
            else (smatrix_output if smatrix_output is not None else "full")
        )
        self.expose_smatrix = bool(kwargs.pop("expose_smatrix", True))
        enable_fields = kwargs.pop("enable_fields", None)
        store_requested = kwargs.pop("store_mode_couplings", None)
        if enable_fields is not None and store_requested is not None:
            if bool(enable_fields) != bool(store_requested):
                raise ValueError(
                    "enable_fields and store_mode_couplings disagree."
                )
        self.store_mode_couplings = bool(
            enable_fields
            if enable_fields is not None
            else (
                self.smatrix_size == "full"
                if store_requested is None
                else store_requested
            )
        )

        cascade = kwargs.pop("cascade", None)
        smatrix_algorithm = kwargs.pop("smatrix_algorithm", None)
        if cascade is not None and smatrix_algorithm is not None:
            if _normalize_cascade(cascade) != _normalize_cascade(smatrix_algorithm):
                raise ValueError("cascade and smatrix_algorithm disagree.")
        self.smatrix_algorithm = _normalize_cascade(
            smatrix_algorithm if smatrix_algorithm is not None else (
                cascade if cascade is not None else "redheffer"
            )
        )
        self.verify_cascade = bool(kwargs.pop("verify_cascade", True))
        self.compute_condition_numbers = bool(
            kwargs.pop("compute_condition_numbers", False)
        )
        super().__init__(*args, **kwargs)
        # torcwa 0.1.4.2 defines pi as 3.141592652589793.  Recompute omega
        # with Python's correctly rounded constant; otherwise even an identity
        # ASR map leaks about 3e-10 into off-diagonal Fourier orders.
        self.omega = _TWO_PI * self.freq
        self._initial_store_mode_couplings = self.store_mode_couplings
        self.layer_records: list[LayerRecord] = []
        self._asr_slot_by_layer: dict[int, int] = {}
        self._asr_field_context_by_layer: dict[int, dict[str, torch.Tensor]] = {}
        self._reduced_longitudinal_context_by_layer: dict[
            int, dict[str, torch.Tensor]
        ] = {}
        self._physical_material_by_layer: dict[
            int, tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._nvm_eps_tensor_by_layer: dict[int, torch.Tensor] = {}
        self.cascade_diagnostics: dict[str, object] = {}

    @property
    def computed_smatrix_blocks(self) -> tuple[str, ...]:
        """Names of the public S blocks requested by ``smatrix_size``."""
        if not self.expose_smatrix:
            return ()
        if self.smatrix_size == "quarter":
            return ("Rf",)
        if self.smatrix_size == "half":
            return ("Tf", "Rf")
        return ("Tf", "Rf", "Rb", "Tb")

    def _cartesian_e_to_h(self, kz: torch.Tensor, *, mu=1.0) -> torch.Tensor:
        """Cartesian transverse E-to-H modal map for an isotropic medium."""
        safe = torch.where(
            torch.abs(kz) < 1.0e-12,
            kz
            + torch.as_tensor(
                1.0e-10 + 1.0e-10j,
                dtype=self._dtype,
                device=self._device,
            ),
            kz,
        )
        left = torch.cat(
            (
                torch.diag(-self.Ky_norm_dn * self.Kx_norm_dn / safe),
                torch.diag(safe + self.Kx_norm_dn**2 / safe),
            ),
            dim=0,
        )
        right = torch.cat(
            (
                torch.diag(-safe - self.Ky_norm_dn**2 / safe),
                torch.diag(self.Kx_norm_dn * self.Ky_norm_dn / safe),
            ),
            dim=0,
        )
        mu_t = torch.as_tensor(mu, dtype=self._dtype, device=self._device)
        return torch.cat((left, right), dim=1) / mu_t

    def _interface_s(
        self, medium_v: torch.Tensor, *, input_side: bool
    ) -> list[torch.Tensor]:
        inverse_sum = self._solve(
            self.Vf + medium_v, self._eye(2 * self.order_N)
        )
        difference = self.Vf - medium_v
        if input_side:
            return [
                2.0 * torch.matmul(inverse_sum, medium_v),
                -torch.matmul(inverse_sum, difference),
                torch.matmul(inverse_sum, difference),
                2.0 * torch.matmul(inverse_sum, self.Vf),
            ]
        return [
            2.0 * torch.matmul(inverse_sum, self.Vf),
            torch.matmul(inverse_sum, difference),
            -torch.matmul(inverse_sum, difference),
            2.0 * torch.matmul(inverse_sum, medium_v),
        ]

    def _kvectors(self) -> None:
        """Rectangular reciprocal vectors with magnetic half-space support."""
        refractive_scale = torch.real(
            torch.sqrt(
                self.eps_in * self.mu_in
                if self.angle_layer == "input"
                else self.eps_out * self.mu_out
            )
        )
        self.kx0_norm = refractive_scale * torch.sin(self.inc_ang) * torch.cos(
            self.azi_ang
        )
        self.ky0_norm = refractive_scale * torch.sin(self.inc_ang) * torch.sin(
            self.azi_ang
        )
        self.kx_norm = self.kx0_norm + self.order_x * self.Gx_norm
        self.ky_norm = self.ky0_norm + self.order_y * self.Gy_norm
        kx_grid, ky_grid = torch.meshgrid(
            self.kx_norm, self.ky_norm, indexing="ij"
        )
        self.Kx_norm_dn = kx_grid.reshape(-1)
        self.Ky_norm_dn = ky_grid.reshape(-1)
        self.Kx_norm = torch.diag(self.Kx_norm_dn)
        self.Ky_norm = torch.diag(self.Ky_norm_dn)

        kz_free = self._positive_kz(
            1.0 - self.Kx_norm_dn**2 - self.Ky_norm_dn**2
        )
        self.Vf = self._cartesian_e_to_h(kz_free, mu=1.0)
        if hasattr(self, "Sin"):
            kz_input = self._positive_kz(
                self.eps_in * self.mu_in
                - self.Kx_norm_dn**2
                - self.Ky_norm_dn**2
            )
            self.Vi = self._cartesian_e_to_h(kz_input, mu=self.mu_in)
            self.Sin.clear()
            self.Sin.extend(self._interface_s(self.Vi, input_side=True))
        if hasattr(self, "Sout"):
            kz_output = self._positive_kz(
                self.eps_out * self.mu_out
                - self.Kx_norm_dn**2
                - self.Ky_norm_dn**2
            )
            self.Vo = self._cartesian_e_to_h(kz_output, mu=self.mu_out)
            self.Sout.clear()
            self.Sout.extend(self._interface_s(self.Vo, input_side=False))

    def add_input_layer(self, eps=1.0, mu=1.0):
        if hasattr(self, "Kx_norm"):
            raise RuntimeError(
                "Add input/output layers before set_incident_angle()."
            )
        return _ORIGINAL_TORCWA_RCWA.add_input_layer(self, eps=eps, mu=mu)

    def add_output_layer(self, eps=1.0, mu=1.0):
        if hasattr(self, "Kx_norm"):
            raise RuntimeError(
                "Add input/output layers before set_incident_angle()."
            )
        return _ORIGINAL_TORCWA_RCWA.add_output_layer(self, eps=eps, mu=mu)

    def set_incident_angle(self, inc_ang, azi_ang, angle_layer="input"):
        if self.layer_N:
            raise RuntimeError(
                "Incident angle cannot be changed after layers were added; "
                "create a new simulation so every modal basis is rebuilt."
            )
        # Upstream torcwa appends to Sin/Sout on repeated calls.  Clearing here
        # prevents stale interface blocks from remaining at indices 0..3.
        if hasattr(self, "Sin"):
            self.Sin.clear()
        if hasattr(self, "Sout"):
            self.Sout.clear()
        return _ORIGINAL_TORCWA_RCWA.set_incident_angle(
            self, inc_ang, azi_ang, angle_layer=angle_layer
        )

    def add_layer(self, thickness, eps=1.0, mu=1.0):
        if getattr(self, "polarization_reduction", None) is not None:
            raise UnsupportedCombinationError(
                "Polarization-reduced mode currently accepts NVM circle layers "
                "only; homogeneous or raster internal layers need a reduced-basis "
                "implementation before they can be mixed into this stack."
            )
        if not hasattr(self, "Kx_norm"):
            raise RuntimeError("Call set_incident_angle() before add_layer().")
        layer_index = self.layer_N
        eps_value = torch.as_tensor(
            eps, dtype=self._dtype, device=self._device
        )
        mu_value = torch.as_tensor(
            mu, dtype=self._dtype, device=self._device
        )
        thickness_tensor, _ = _real_parameter_tensor(
            "thickness",
            thickness,
            dtype=self._dtype,
            device=self._device,
            allow_zero=True,
        )
        if not torch.all(torch.isfinite(eps_value)):
            raise ValueError("eps contains NaN or infinity.")
        if not torch.all(torch.isfinite(mu_value)):
            raise ValueError("mu contains NaN or infinity.")

        eps_homogeneous = eps_value.ndim == 0 or (
            eps_value.ndim == 1 and eps_value.numel() == 1
        )
        mu_homogeneous = mu_value.ndim == 0 or (
            mu_value.ndim == 1 and mu_value.numel() == 1
        )
        self.eps_conv.append(
            eps_value * self._eye(self.order_N)
            if eps_homogeneous
            else self._material_conv(eps_value)
        )
        self.mu_conv.append(
            mu_value * self._eye(self.order_N)
            if mu_homogeneous
            else self._material_conv(mu_value)
        )
        self.layer_N += 1
        self.thickness.append(thickness_tensor)
        if eps_homogeneous and mu_homogeneous:
            _ORIGINAL_TORCWA_RCWA._eigen_decomposition_homogenous(
                self, eps_value, mu_value
            )
        else:
            _ORIGINAL_TORCWA_RCWA._eigen_decomposition(self)
        kz = self.kz_norm[-1]
        flip = (kz.imag < 0.0) | (
            (torch.abs(kz.imag) <= 1.0e-14) & (kz.real < 0.0)
        )
        self.kz_norm[-1] = torch.where(flip, -kz, kz)
        magnetic_modes = self._magnetic_eigenvectors(
            self.P[-1], self.Q[-1], self.E_eigvec[-1], self.kz_norm[-1]
        )
        self.H_eigvec.append(magnetic_modes)
        self._append_smatrix_from_cartesian_modes(
            self.E_eigvec[-1], magnetic_modes
        )
        self.layer_records.append(
            LayerRecord(
                index=layer_index,
                method="standard",
                shape="homogeneous" if eps_value.numel() == 1 else "raster",
                lattice=getattr(self, "lattice_kind", "rectangular"),
                reason="LEGACY_STANDARD_LAYER",
            )
        )
        if eps_value.ndim == 2:
            eps_grid = eps_value.to(dtype=self._dtype, device=self._device)
            if mu_value.ndim == 2:
                mu_grid = mu_value.to(dtype=self._dtype, device=self._device)
            else:
                mu_grid = torch.ones_like(eps_grid) * mu_value
            self._physical_material_by_layer[layer_index] = (eps_grid, mu_grid)
        return None

    def _eye(self, size: int) -> torch.Tensor:
        return torch.eye(size, dtype=self._dtype, device=self._device)

    def _solve(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.linalg.solve(
            a.to(torch.complex128), b.to(torch.complex128)
        ).to(self._dtype)

    def _right_solve(self, b: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Return ``b @ inv(a)`` without forming an inverse (plain transpose)."""
        return self._solve(a.mT, b.mT).mT

    def _eig(self, matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if getattr(self, "stable_eig_grad", False) and hasattr(torcwa, "Eig"):
            values, vectors = torcwa.Eig.apply(matrix)
        else:
            values, vectors = torch.linalg.eig(matrix)
        return values, vectors

    def _magnetic_eigenvectors(
        self,
        p: torch.Tensor,
        q: torch.Tensor,
        electric_modes: torch.Tensor,
        kz: torch.Tensor,
    ) -> torch.Tensor:
        """Build H modes using the stable Maxwell relation ``V=Q W Γ^-1``.

        ``P^-1 W Γ`` is algebraically equivalent because ``P Q W=W Γ²``.
        It is not numerically equivalent when ``P`` is ill-conditioned, which
        is precisely the situation of the non-factorized metal-patch ASR
        problem near order 20.  The paper writes the Q form explicitly.  Only
        an exactly/nearly zero propagation constant falls back to the P form.
        """
        real_dtype = (
            torch.float32 if self._dtype == torch.complex64 else torch.float64
        )
        epsilon = torch.finfo(real_dtype).eps
        scale = torch.maximum(
            torch.max(torch.abs(kz)),
            torch.ones((), dtype=real_dtype, device=self._device),
        )
        threshold = 64.0 * epsilon * scale
        regular = torch.abs(kz) > threshold
        safe_kz = torch.where(regular, kz, torch.ones_like(kz))
        magnetic_from_q = torch.matmul(
            q,
            electric_modes * (1.0 / safe_kz)[None, :],
        )
        if bool(torch.all(regular)):
            return magnetic_from_q

        magnetic_from_p = self._solve(
            p, electric_modes * kz[None, :]
        )
        return torch.where(
            regular[None, :], magnetic_from_q, magnetic_from_p
        )

    @staticmethod
    def _positive_kz(kz_squared: torch.Tensor) -> torch.Tensor:
        kz = torch.sqrt(kz_squared)
        # torcwa uses exp(+j*kz*z); Im(kz) >= 0 is therefore the decaying branch.
        flip = (kz.imag < 0.0) | (
            (torch.abs(kz.imag) <= 1.0e-14) & (kz.real < 0.0)
        )
        return torch.where(flip, -kz, kz)

    def _append_smatrix_from_cartesian_modes(
        self, e_modes: torch.Tensor, h_modes: torch.Tensor
    ) -> None:
        """
        Append torcwa-compatible [Tf, Rf, Rb, Tb] blocks and Cf/Cb.

        This is algebraically equivalent to the paper's Eq. (29), after
        translating its [R, T; T, R] block convention to torcwa's convention.
        """
        if not getattr(self, "store_mode_couplings", True):
            self._append_smatrix_from_equation_29(e_modes, h_modes)
            return

        size = 2 * self.order_N
        identity = self._eye(size)
        zero = torch.zeros_like(identity)
        phase = torch.diag(
            torch.exp(
                1.0j
                * self.omega
                * self.kz_norm[-1]
                * self.thickness[-1]
            )
        )

        vf_inverse_h = self._solve(self.Vf, h_modes)
        a = e_modes + vf_inverse_h
        b_phase = torch.matmul(e_modes - vf_inverse_h, phase)
        coupling = torch.cat(
            (
                torch.cat((a, b_phase), dim=1),
                torch.cat((b_phase, a), dim=1),
            ),
            dim=0,
        )

        cf = self._solve(coupling, torch.cat((2.0 * identity, zero), dim=0))
        cb = self._solve(coupling, torch.cat((zero, 2.0 * identity), dim=0))
        self.Cf.append(cf)
        self.Cb.append(cb)

        n2 = size
        cf_plus, cf_minus = cf[:n2], cf[n2:]
        cb_plus, cb_minus = cb[:n2], cb[n2:]
        e_phase = torch.matmul(e_modes, phase)

        # torcwa convention: [Tf, Rf, Rb, Tb].
        self.layer_S11.append(
            torch.matmul(e_phase, cf_plus) + torch.matmul(e_modes, cf_minus)
        )
        self.layer_S21.append(
            torch.matmul(e_modes, cf_plus)
            + torch.matmul(e_phase, cf_minus)
            - identity
        )
        self.layer_S12.append(
            torch.matmul(e_phase, cb_plus)
            + torch.matmul(e_modes, cb_minus)
            - identity
        )
        self.layer_S22.append(
            torch.matmul(e_modes, cb_plus) + torch.matmul(e_phase, cb_minus)
        )

    def _append_smatrix_from_equation_29(
        self, e_modes: torch.Tensor, h_modes: torch.Tensor
    ) -> None:
        """
        Paper Eq. (29), mapped to torcwa's [Tf, Rf, Rb, Tb] ordering.

        This scattering-only path avoids the 4N-by-4N coupling matrix and the
        Cf/Cb field-reconstruction matrices. It is mathematically identical to
        _append_smatrix_from_cartesian_modes, but is much less memory intensive
        for the paper's ASR N=M=20 run.
        """
        size = 2 * self.order_N
        identity = self._eye(size)
        phase = torch.diag(
            torch.exp(
                1.0j
                * self.omega
                * self.kz_norm[-1]
                * self.thickness[-1]
            )
        )

        # A=(TW)^-1 W0+(TV)^-1 V0 and B=(TW)^-1 W0-(TV)^-1 V0.
        # In torcwa's Cartesian reference medium, W0=I and V0=Vf.
        a = self._solve(e_modes, identity) + self._solve(h_modes, self.Vf)
        b = self._solve(e_modes, identity) - self._solve(h_modes, self.Vf)
        a_inverse_b = self._solve(a, b)
        a_inverse_xb = self._solve(a, torch.matmul(phase, b))
        a_inverse_xa = self._solve(a, torch.matmul(phase, a))

        core = a - torch.matmul(
            torch.matmul(phase, b), a_inverse_xb
        )
        reflection_rhs = torch.matmul(
            torch.matmul(phase, b), a_inverse_xa
        ) - b
        transmission_rhs = torch.matmul(
            phase, a - torch.matmul(b, a_inverse_b)
        )
        reflection = self._solve(core, reflection_rhs)
        transmission = self._solve(core, transmission_rhs)

        self.layer_S11.append(transmission)
        self.layer_S21.append(reflection)
        self.layer_S12.append(reflection)
        self.layer_S22.append(transmission)

    def solve_global_smatrix_s_only(self) -> None:
        """Redheffer star product without field-reconstruction coefficients."""
        size = 2 * self.order_N
        identity = self._eye(size)
        zero = torch.zeros_like(identity)

        def connect(sm, sn):
            inverse_1 = self._solve(
                identity - torch.matmul(sm[2], sn[1]), identity
            )
            inverse_2 = self._solve(
                identity - torch.matmul(sn[1], sm[2]), identity
            )
            return [
                torch.matmul(sn[0], torch.matmul(inverse_1, sm[0])),
                sm[1]
                + torch.matmul(
                    sm[3],
                    torch.matmul(inverse_2, torch.matmul(sn[1], sm[0])),
                ),
                sn[2]
                + torch.matmul(
                    sn[0],
                    torch.matmul(inverse_1, torch.matmul(sm[2], sn[3])),
                ),
                torch.matmul(sm[3], torch.matmul(inverse_2, sn[3])),
            ]

        if self.layer_N > 0:
            scattering = [
                self.layer_S11[0],
                self.layer_S21[0],
                self.layer_S12[0],
                self.layer_S22[0],
            ]
            for layer in range(1, self.layer_N):
                scattering = connect(
                    scattering,
                    [
                        self.layer_S11[layer],
                        self.layer_S21[layer],
                        self.layer_S12[layer],
                        self.layer_S22[layer],
                    ],
                )
        else:
            scattering = [identity, zero, zero, identity]

        if hasattr(self, "Sin"):
            scattering = connect(self.Sin, scattering)
        if hasattr(self, "Sout"):
            scattering = connect(scattering, self.Sout)
        self.S = scattering
        self.C = [[], []]

    def _mask_smatrix_blocks(
        self, scattering: Sequence[torch.Tensor]
    ) -> list[torch.Tensor]:
        """Replace unrequested public blocks with shape-compatible zeros."""
        if not self.expose_smatrix:
            zero = torch.zeros_like(scattering[0])
            return [zero, zero.clone(), zero.clone(), zero.clone()]
        if self.smatrix_size == "full":
            return list(scattering)
        zero = torch.zeros_like(scattering[0])
        if self.smatrix_size == "half":
            return [scattering[0], scattering[1], zero, zero]
        return [zero, scattering[1], zero, zero]

    def _solve_global_smatrix_redheffer_partial(self) -> None:
        """Backward Redheffer recursion for forward-only S-matrix blocks."""
        if self.smatrix_size == "full":
            self.solve_global_smatrix_s_only()
            return
        if self.layer_N == 0:
            self.solve_global_smatrix_s_only()
            self.S = self._mask_smatrix_blocks(self.S)
            return

        size = 2 * self.order_N
        identity = self._eye(size)
        zero = torch.zeros_like(identity)
        if hasattr(self, "Sout"):
            transmission = self.Sout[0] if self.smatrix_size == "half" else None
            reflection = self.Sout[1]
        else:
            transmission = identity if self.smatrix_size == "half" else None
            reflection = zero

        def prepend(left: Sequence[torch.Tensor]) -> None:
            nonlocal transmission, reflection
            tf_left, rf_left, rb_left, tb_left = left
            denominator_r = identity - torch.matmul(reflection, rb_left)
            reflected_rhs = torch.matmul(reflection, tf_left)
            reflection_new = rf_left + torch.matmul(
                tb_left, self._solve(denominator_r, reflected_rhs)
            )
            if transmission is not None:
                denominator_t = identity - torch.matmul(rb_left, reflection)
                transmission = torch.matmul(
                    transmission, self._solve(denominator_t, tf_left)
                )
            reflection = reflection_new

        for layer in range(self.layer_N - 1, -1, -1):
            prepend(
                (
                    self.layer_S11[layer],
                    self.layer_S21[layer],
                    self.layer_S12[layer],
                    self.layer_S22[layer],
                )
            )
        if hasattr(self, "Sin"):
            prepend(self.Sin)

        self.S = (
            [transmission, reflection, zero, zero]
            if transmission is not None
            else [zero, reflection, zero, zero]
        )
        self.C = [[], []]

    @staticmethod
    def _modal_basis_matrix(
        e_modes: torch.Tensor, h_modes: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat(
            (
                torch.cat((e_modes, e_modes), dim=1),
                torch.cat((h_modes, -h_modes), dim=1),
            ),
            dim=0,
        )

    def _solve_global_smatrix_li2a(self) -> None:
        """Lifeng Li's stable interface-t recursion (algorithm 2a, Eq. 19a)."""
        if not hasattr(self, "Kx_norm"):
            raise RuntimeError("Call set_incident_angle() before solving.")

        size = 2 * self.order_N
        identity = self._eye(size)
        zero = torch.zeros_like(identity)
        input_v = getattr(self, "Vi", self.Vf)
        output_v = getattr(self, "Vo", self.Vf)
        bases = [
            self._modal_basis_matrix(identity, input_v),
            *[
                self._modal_basis_matrix(e_modes, h_modes)
                for e_modes, h_modes in zip(self.E_eigvec, self.H_eigvec)
            ],
            self._modal_basis_matrix(identity, output_v),
        ]

        # Interface p is preceded by propagation through medium p.  The input
        # half-space has zero numerical thickness; every internal medium uses
        # the stable decaying phase X=exp(+j*omega*kz*d).
        phases = [identity]
        phases.extend(
            torch.diag(
                torch.exp(1.0j * self.omega * kz * thickness)
            )
            for kz, thickness in zip(self.kz_norm, self.thickness)
        )

        tf, rf, rb, tb = identity, zero, zero, identity
        interface_condition_numbers: list[torch.Tensor] = []
        for interface, (left_basis, right_basis) in enumerate(
            zip(bases[:-1], bases[1:])
        ):
            transfer = self._solve(right_basis, left_basis)
            if self.compute_condition_numbers:
                interface_condition_numbers.append(
                    torch.linalg.cond(right_basis.to(torch.complex128))
                )
            t11 = transfer[:size, :size]
            t12 = transfer[:size, size:]
            t21 = transfer[size:, :size]
            t22 = transfer[size:, size:]
            phase = phases[interface]

            # Li Eq. (19a') and (19a), translated to torcwa's block order
            # [Tf, Rf, Rb, Tb].  f+ and inv(f-) are the same stable phase for
            # the +/-kz modal pair used by torcwa.
            v_matrix = torch.matmul(phase, torch.matmul(rb, phase))
            denominator = t22 + torch.matmul(t21, v_matrix)
            rb_new = self._right_solve(
                t12 + torch.matmul(t11, v_matrix), denominator
            )
            tb_new = self._right_solve(torch.matmul(tb, phase), denominator)
            tf_new = torch.matmul(
                torch.matmul(t11 - torch.matmul(rb_new, t21), phase), tf
            )
            rf_new = rf - torch.matmul(
                torch.matmul(torch.matmul(tb_new, t21), phase), tf
            )
            tf, rf, rb, tb = tf_new, rf_new, rb_new, tb_new

        self.S = [tf, rf, rb, tb]
        self.C = [[], []]
        self.cascade_diagnostics = {
            "algorithm": "li-2a",
        }
        if self.compute_condition_numbers:
            self.cascade_diagnostics[
                "interface_condition_numbers"
            ] = interface_condition_numbers

    def _li2a_interface_pairs_right_to_left(
        self,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Return Li t+/t- interface blocks ordered from output to input."""
        if self.layer_N == 0:
            return []
        identity = self._eye(2 * self.order_N)
        output_v = getattr(self, "Vo", self.Vf)
        input_v = getattr(self, "Vi", self.Vf)

        def pair(
            left_e: torch.Tensor,
            left_h: torch.Tensor,
            right_e: torch.Tensor,
            right_h: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            electric = self._solve(left_e, right_e)
            magnetic = self._solve(left_h, right_h)
            return 0.5 * (electric + magnetic), 0.5 * (electric - magnetic)

        pairs = [
            pair(
                self.E_eigvec[-1],
                self.H_eigvec[-1],
                identity,
                output_v,
            )
        ]
        for right_layer in range(self.layer_N - 1, 0, -1):
            left_layer = right_layer - 1
            pairs.append(
                pair(
                    self.E_eigvec[left_layer],
                    self.H_eigvec[left_layer],
                    self.E_eigvec[right_layer],
                    self.H_eigvec[right_layer],
                )
            )
        pairs.append(
            pair(
                identity,
                input_v,
                self.E_eigvec[0],
                self.H_eigvec[0],
            )
        )
        return pairs

    def _solve_global_smatrix_li2a_partial(self) -> None:
        """Li algorithm 2a computing Rf alone or the forward Tf/Rf pair."""
        if self.smatrix_size == "full":
            self._solve_global_smatrix_li2a()
            return
        if self.layer_N == 0:
            self.solve_global_smatrix_s_only()
            self.S = self._mask_smatrix_blocks(self.S)
            self.cascade_diagnostics = {
                "algorithm": "li-2a",
                "size": self.smatrix_size,
                "empty_stack_via": "redheffer",
            }
            return

        pairs = self._li2a_interface_pairs_right_to_left()
        t_plus, t_minus = pairs[0]
        identity = self._eye(2 * self.order_N)
        zero = torch.zeros_like(identity)
        reflection = self._right_solve(t_minus, t_plus)
        transmission = (
            self._solve(t_plus, identity)
            if self.smatrix_size == "half"
            else None
        )

        for offset, layer in enumerate(range(self.layer_N - 1, -1, -1), start=1):
            phase = torch.exp(
                1.0j * self.omega * self.kz_norm[layer] * self.thickness[layer]
            )
            omega = phase[:, None] * reflection * phase[None, :]
            t_plus, t_minus = pairs[offset]
            denominator = t_plus + torch.matmul(t_minus, omega)
            numerator = t_minus + torch.matmul(t_plus, omega)
            reflection = self._right_solve(numerator, denominator)
            if transmission is not None:
                transmission = self._right_solve(
                    transmission * phase[None, :], denominator
                )

        self.S = (
            [transmission, reflection, zero, zero]
            if transmission is not None
            else [zero, reflection, zero, zero]
        )
        self.C = [[], []]
        self.cascade_diagnostics = {
            "algorithm": "li-2a",
            "size": self.smatrix_size,
        }

    def solve_global_smatrix(self) -> None:
        if self.store_mode_couplings != self._initial_store_mode_couplings:
            raise RuntimeError(
                "Field-storage mode is fixed at construction time; create a new "
                "simulation instead of changing store_mode_couplings."
            )
        if self.smatrix_algorithm == "redheffer":
            # Upstream torcwa creates one-dimensional zero-reflection blocks for
            # an empty stack.  The local implementation keeps every S block a
            # square matrix, which is also what S_parameters() expects.
            if self.store_mode_couplings and self.layer_N > 0:
                _ORIGINAL_TORCWA_RCWA.solve_global_smatrix(self)
                self._field_smatrix = list(self.S)
                self.S = self._mask_smatrix_blocks(self.S)
            elif self.store_mode_couplings:
                self.solve_global_smatrix_s_only()
                self._field_smatrix = list(self.S)
                self.S = self._mask_smatrix_blocks(self.S)
            elif self.smatrix_size == "full":
                self.solve_global_smatrix_s_only()
            else:
                self._solve_global_smatrix_redheffer_partial()
            self.S = self._mask_smatrix_blocks(self.S)
            self.cascade_diagnostics = {
                "algorithm": "redheffer",
                "size": self.smatrix_size,
                "computed_blocks": self.computed_smatrix_blocks,
            }
            return

        if self.smatrix_size == "full":
            self._solve_global_smatrix_li2a()
        else:
            self._solve_global_smatrix_li2a_partial()
        li_scattering = [block.clone() for block in self.S]
        if self.store_mode_couplings or self.verify_cascade:
            # Algorithm 2a deliberately bypasses per-layer s matrices.  The
            # Redheffer pass supplies the same stable internal coupling matrices
            # needed by field reconstruction and is also an optional parity
            # check.  The public S remains the Li-2a result.
            if self.store_mode_couplings and self.layer_N > 0:
                _ORIGINAL_TORCWA_RCWA.solve_global_smatrix(self)
                self._field_smatrix = list(self.S)
                self.S = self._mask_smatrix_blocks(self.S)
            elif self.store_mode_couplings:
                self.solve_global_smatrix_s_only()
                self._field_smatrix = list(self.S)
                self.S = self._mask_smatrix_blocks(self.S)
            elif self.smatrix_size == "full":
                self.solve_global_smatrix_s_only()
            else:
                self._solve_global_smatrix_redheffer_partial()
            redheffer_scattering = (
                self._field_smatrix if self.store_mode_couplings else self.S
            )
            compared_indices = (
                (1,)
                if self.smatrix_size == "quarter"
                else ((0, 1) if self.smatrix_size == "half" else (0, 1, 2, 3))
            )
            absolute_errors = [
                torch.max(torch.abs(li_scattering[index] - redheffer_scattering[index])).detach()
                for index in compared_indices
            ]
            maximum_error = torch.max(torch.stack(absolute_errors))
            diagnostics = dict(self.cascade_diagnostics)
            diagnostics.update(
                {
                    "algorithm": "li-2a",
                    "size": self.smatrix_size,
                    "computed_blocks": self.computed_smatrix_blocks,
                    "redheffer_max_abs_error": maximum_error,
                }
            )
            self.cascade_diagnostics = diagnostics
            tolerance = 2.0e-4 if self._dtype == torch.complex64 else 2.0e-9
            if self.verify_cascade and _as_float(maximum_error) > tolerance:
                raise RuntimeError(
                    "Li-2a/Redheffer parity check failed: maximum absolute "
                    f"S-block error is {_as_float(maximum_error):.3e}."
                )
        self.S = self._mask_smatrix_blocks(li_scattering)
        self.cascade_diagnostics.setdefault("size", self.smatrix_size)
        self.cascade_diagnostics.setdefault(
            "computed_blocks", self.computed_smatrix_blocks
        )

    def S_parameters(self, *args, **kwargs):
        """Return public S parameters, rejecting a fields-only request."""
        if not self.expose_smatrix:
            raise RuntimeError(
                "S-matrix output was disabled by OutputSpec(smatrix=False)."
            )
        return _ORIGINAL_TORCWA_RCWA.S_parameters(self, *args, **kwargs)
