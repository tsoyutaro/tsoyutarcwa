"""Oblique-cell NVM operators and analytic circular-layer assembly."""

from __future__ import annotations

import math
from typing import Sequence

import torch

from .config import (
    LayerRecord, UnsupportedCombinationError, _ORIGINAL_TORCWA_RCWA,
    _TWO_PI, _as_float, _normalize_group_symmetry,
    _normalize_polarization, _real_parameter_tensor,
)
from .reduced import _ReducedScatteringMixin
from .scattering import _StableLinearAlgebraMixin
from .symmetry import _SymmetryReductionMixin

class _Jinc(torch.autograd.Function):
    """Autograd-safe ``2 J_1(x) / x`` used by analytic disk coefficients.

    Some supported PyTorch builds expose ``torch.special.bessel_j1`` without a
    backward rule.  The exact recurrence

        d[2 J1(x)/x]/dx = 2 J0(x)/x - 4 J1(x)/x^2

    supplies the missing first derivative.  A local series avoids cancellation
    at the origin and has the same truncation as the former forward expression.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        small = torch.abs(x) < 1.0e-4
        safe = torch.where(small, torch.ones_like(x), x)
        j0 = torch.special.bessel_j0(safe)
        j1 = torch.special.bessel_j1(safe)
        direct = 2.0 * j1 / safe
        series = 1.0 - x**2 / 8.0 + x**4 / 192.0
        ctx.save_for_backward(x, safe, j0, j1, small)
        return torch.where(small, series, direct)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor]:
        x, safe, j0, j1, small = ctx.saved_tensors
        direct_derivative = 2.0 * j0 / safe - 4.0 * j1 / safe**2
        series_derivative = -x / 4.0 + x**3 / 48.0
        derivative = torch.where(small, series_derivative, direct_derivative)
        return (grad_output * derivative,)


def _jinc(x: torch.Tensor) -> torch.Tensor:
    return _Jinc.apply(x)


def _smoothstep(x: torch.Tensor, edge0: object, edge1: object) -> torch.Tensor:
    if not _as_float(edge1) > _as_float(edge0):
        raise ValueError("smoothstep requires edge1 > edge0.")
    t = torch.clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

class CustomRCWA_NVM(_ReducedScatteringMixin, _SymmetryReductionMixin, _StableLinearAlgebraMixin, _ORIGINAL_TORCWA_RCWA):
    """
    NVM solver for circular inclusions in rectangular or oblique cells.

    The oblique primitive vectors are
        a1=(Lx, 0), a2=(Ly*cos(zeta), Ly*sin(zeta)).
    zeta=60 degrees and Lx=Ly is a triangular Bravais lattice.

    At normal incidence, a single circular inclusion can optionally use a C2v
    block decomposition in an orthogonal cell, a D6-closed-star/Cs
    source-specific x/y sector in a triangular cell, a complete-D6 E1
    matrix-unit source row, or the common x/y
    source-accessible C2 sector in a general oblique cell.  Every projected
    block is checked for invariance;
    an ineligible decomposition falls back to the full eigensolve unless strict
    mode was requested.  No modes are skipped or replaced by dummy modes.
    """

    def __init__(self, freq, order, L, **kwargs):
        self.zeta_deg = float(kwargs.pop("zeta_deg", 90.0))
        self.use_group_theory = bool(kwargs.pop("use_group_theory", False))
        self.polarization_reduction = _normalize_polarization(
            kwargs.pop("polarization_reduction", None)
        )
        self.group_theory_symmetry = _normalize_group_symmetry(
            kwargs.pop("group_theory_symmetry", "auto")
        )
        self.group_theory_strict = bool(kwargs.pop("group_theory_strict", False))
        self.group_theory_tolerance = float(
            kwargs.pop("group_theory_tolerance", 1.0e-8)
        )
        if (
            not math.isfinite(self.group_theory_tolerance)
            or self.group_theory_tolerance <= 0.0
        ):
            raise ValueError("group_theory_tolerance must be finite and positive.")
        if self.polarization_reduction is not None and not self.use_group_theory:
            raise ValueError(
                "polarization_reduction requires use_group_theory=True."
            )
        unsupported = [
            name
            for name in ("force_symmetrize", "skip_blocks", "incident_jones_xy")
            if name in kwargs
        ]
        if unsupported:
            raise UnsupportedCombinationError(
                "Unsupported legacy NVM option(s): " + ", ".join(unsupported)
            )
        if not math.isfinite(self.zeta_deg) or not 0.0 < self.zeta_deg < 180.0:
            raise ValueError("zeta_deg must be finite and in (0, 180).")
        if abs(self.zeta_deg - 90.0) <= 1.0e-7:
            self.lattice_kind = "rectangular"
        elif (
            abs(self.zeta_deg - 60.0) <= 1.0e-7
            and math.isclose(_as_float(L[0]), _as_float(L[1]), rel_tol=1.0e-7)
        ):
            self.lattice_kind = "triangular"
        else:
            self.lattice_kind = "oblique"
        super().__init__(freq, order, L, **kwargs)
        self.zeta = math.radians(self.zeta_deg)
        self.sin_zeta = math.sin(self.zeta)
        self.cos_zeta = math.cos(self.zeta)
        if abs(self.sin_zeta) < 1.0e-6:
            raise ValueError("zeta_deg is too close to a degenerate cell.")
        self._nvm_eps_tensor: torch.Tensor | None = None
        self.group_theory_diagnostics: list[dict[str, object]] = []
        self._polarized_layers: list[dict[str, torch.Tensor]] = []
        self._polarization_bases: tuple[torch.Tensor, torch.Tensor] | None = None
        self._native_d6_active = False

    def _kvectors(self) -> None:
        refractive_scale = torch.real(
            torch.sqrt(
                self.eps_in * self.mu_in
                if self.angle_layer == "input"
                else self.eps_out * self.mu_out
            )
        )
        kx0 = refractive_scale * torch.sin(self.inc_ang) * torch.cos(
            self.azi_ang
        )
        ky0 = refractive_scale * torch.sin(self.inc_ang) * torch.sin(
            self.azi_ang
        )

        # Covariant components k.a1/|a1| and k.a2/|a2|.
        k1 = kx0 + self.order_x * self.Gx_norm
        k2 = (
            kx0 * self.cos_zeta
            + ky0 * self.sin_zeta
            + self.order_y * self.Gy_norm
        )
        k1_grid, k2_grid = torch.meshgrid(k1, k2, indexing="ij")
        self.K1_norm_dn = k1_grid.reshape(-1)
        self.K2_norm_dn = k2_grid.reshape(-1)
        self.K1_norm = torch.diag(self.K1_norm_dn)
        self.K2_norm = torch.diag(self.K2_norm_dn)

        self.Kx_norm_dn = self.K1_norm_dn
        self.Ky_norm_dn = (
            self.K2_norm_dn - self.cos_zeta * self.K1_norm_dn
        ) / self.sin_zeta
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

    def _cartesian_e_to_h(self, kz: torch.Tensor, *, mu=1.0) -> torch.Tensor:
        safe = torch.where(
            torch.abs(kz) < 1.0e-12,
            kz + torch.as_tensor(
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

    def _circle_toeplitz(
        self,
        radius: float,
        eps_bg,
        eps_cyl,
        centers: Sequence[tuple[float, float]],
        *,
        use_lanczos: bool,
        lanczos_power: int,
    ) -> torch.Tensor:
        lx, ly = _as_float(self.L[0]), _as_float(self.L[1])
        ox = self.order_x.to(torch.float64)
        oy = self.order_y.to(torch.float64)
        mx, my = len(ox), len(oy)
        count = mx * my

        gx = (
            ox * (_TWO_PI / lx)
        ).repeat_interleave(my).reshape(count, 1)
        gy_lattice = (
            oy * (_TWO_PI / ly)
        ).repeat(mx).reshape(count, 1)
        gy = (
            -gx * self.cos_zeta + gy_lattice
        ) / self.sin_zeta
        dgx = gx - gx.mT
        dgy = gy - gy.mT
        reciprocal_norm = torch.sqrt(dgx**2 + dgy**2)

        structure = torch.zeros(
            (count, count), dtype=self._dtype, device=self._device
        )
        for center_x, center_y in centers:
            structure = structure + torch.exp(
                -1.0j * (dgx * center_x + dgy * center_y)
            )

        cell_area = lx * ly * self.sin_zeta
        fill = math.pi * radius**2 / cell_area
        gr = reciprocal_norm * radius
        disk_transform = _jinc(gr)

        contrast = torch.as_tensor(
            eps_cyl - eps_bg, dtype=self._dtype, device=self._device
        )
        coefficients = contrast * fill * disk_transform * structure
        if use_lanczos:
            maximum = torch.max(reciprocal_norm)
            buffer = min(_TWO_PI / lx, _TWO_PI / ly)
            z = 3.831705970 * reciprocal_norm / (maximum + buffer)
            sigma = _jinc(z)
            coefficients = coefficients * sigma**lanczos_power

        background = torch.as_tensor(
            eps_bg, dtype=self._dtype, device=self._device
        )
        return (coefficients + background * self._eye(count)).to(self._dtype)

    def _projection_matrix(
        self,
        radius: float,
        centers: Sequence[tuple[float, float]],
        nx: int,
        ny: int,
    ) -> torch.Tensor:
        """
        Build a periodic Cartesian normal-vector projection.

        Periodic images are generated with the oblique direct-lattice vectors.
        This fixes the draft's independent Cartesian x/y wrapping, which is not
        periodic for zeta != 90 degrees.
        """
        minimum_x = max(32, 4 * int(self.order[0]) + 4)
        minimum_y = max(32, 4 * int(self.order[1]) + 4)
        if nx < minimum_x or ny < minimum_y:
            raise ValueError(
                "NVM projection grid is too small: "
                f"got ({nx}, {ny}), require at least "
                f"({minimum_x}, {minimum_y})."
            )
        lx, ly = _as_float(self.L[0]), _as_float(self.L[1])
        xi = (
            torch.arange(nx, dtype=torch.float64, device=self._device) / nx
        ) * lx
        eta = (
            torch.arange(ny, dtype=torch.float64, device=self._device) / ny
        ) * ly
        xi_grid, eta_grid = torch.meshgrid(xi, eta, indexing="ij")
        x_grid = xi_grid + eta_grid * self.cos_zeta
        y_grid = eta_grid * self.sin_zeta

        a1 = torch.tensor(
            [lx, 0.0], dtype=torch.float64, device=self._device
        )
        a2 = torch.tensor(
            [ly * self.cos_zeta, ly * self.sin_zeta],
            dtype=torch.float64,
            device=self._device,
        )
        candidate_dx: list[torch.Tensor] = []
        candidate_dy: list[torch.Tensor] = []
        for center_x, center_y in centers:
            center = torch.tensor(
                [center_x, center_y],
                dtype=torch.float64,
                device=self._device,
            )
            for i in range(-2, 3):
                for j in range(-2, 3):
                    image = center + i * a1 + j * a2
                    candidate_dx.append(x_grid - image[0])
                    candidate_dy.append(y_grid - image[1])
        dx_all = torch.stack(candidate_dx)
        dy_all = torch.stack(candidate_dy)
        r2_all = dx_all**2 + dy_all**2
        two_smallest, indices = torch.topk(
            r2_all, k=2, dim=0, largest=False
        )
        nearest = indices[0]
        dx = torch.gather(dx_all, 0, nearest[None])[0]
        dy = torch.gather(dy_all, 0, nearest[None])[0]
        radius_nearest = torch.sqrt(torch.clamp(two_smallest[0], min=0.0))
        radius_second = torch.sqrt(torch.clamp(two_smallest[1], min=0.0))
        safe_radius = torch.clamp(radius_nearest, min=1.0e-14)
        nx_field = dx / safe_radius
        ny_field = dy / safe_radius

        resolution = max(lx / nx, ly / ny)
        taper_radius = torch.maximum(
            radius / 2.0,
            radius.new_tensor(resolution),
        )
        center_weight = _smoothstep(radius_nearest, 0.0, taper_radius)
        boundary_weight = _smoothstep(
            radius_second - radius_nearest, 0.0, 3.0 * resolution
        )
        weight = center_weight * boundary_weight
        pxx_real = 0.5 + (nx_field**2 - 0.5) * weight
        pyy_real = 0.5 + (ny_field**2 - 0.5) * weight
        pxy_real = nx_field * ny_field * weight

        def convolution(real_field: torch.Tensor) -> torch.Tensor:
            fft = torch.fft.fft2(real_field.to(self._dtype)) / (nx * ny)
            mx, my = len(self.order_x), len(self.order_y)
            p_delta = self.order_x[:, None] - self.order_x[None, :]
            q_delta = self.order_y[:, None] - self.order_y[None, :]
            result = torch.zeros(
                (mx * my, mx * my),
                dtype=self._dtype,
                device=self._device,
            )
            for p in range(mx):
                rows = slice(p * my, (p + 1) * my)
                for pp in range(mx):
                    cols = slice(pp * my, (pp + 1) * my)
                    result[rows, cols] = fft[p_delta[p, pp], q_delta]
            # The sampled projection is real, so its convolution is Hermitian.
            return 0.5 * (result + result.mH)

        pxx = convolution(pxx_real)
        pyy = convolution(pyy_real)
        pxy = convolution(pxy_real)
        pyx = pxy.mH
        return torch.cat(
            (
                torch.cat((pxx, pxy), dim=1),
                torch.cat((pyx, pyy), dim=1),
            ),
            dim=0,
        )

    def _build_triangular_nvm_star_pq(
        self,
        eps_zz: torch.Tensor,
        inverse_eps_rule: torch.Tensor,
        projection: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build NVM ``P,Q`` directly on a D6-closed triangular star.

        ``inverse_eps_rule`` is the Fourier convolution matrix of ``1/eps``.
        Both inverse-rule matrices are inverted only *after* restriction to
        the star.  Restricting a rectangular-box inverse would retain virtual
        coupling through corner harmonics that are absent from the D6-closed
        basis and would spoil exact mirror-sector invariance.
        """
        vector_embedding, _, _, _, _ = self._triangular_star_operators()
        star_count = vector_embedding.shape[1] // 2
        scalar_embedding = vector_embedding[: self.order_N, :star_count]

        def restrict_scalar(matrix: torch.Tensor) -> torch.Tensor:
            return torch.matmul(
                scalar_embedding.mH,
                torch.matmul(matrix, scalar_embedding),
            )

        eps_zz_star = restrict_scalar(eps_zz)
        inverse_eps_rule_star = restrict_scalar(inverse_eps_rule)
        identity = self._eye(star_count)
        effective_inverse_rule = self._solve(
            inverse_eps_rule_star, identity
        )
        inverse_eps_zz_star = self._solve(eps_zz_star, identity)
        projection_star = torch.matmul(
            vector_embedding.mH,
            torch.matmul(projection, vector_embedding),
        )

        delta = effective_inverse_rule - eps_zz_star
        zero = torch.zeros_like(delta)
        delta_2s = torch.cat(
            (
                torch.cat((delta, zero), dim=1),
                torch.cat((zero, delta), dim=1),
            ),
            dim=0,
        )
        eps_2s = torch.cat(
            (
                torch.cat((eps_zz_star, zero), dim=1),
                torch.cat((zero, eps_zz_star), dim=1),
            ),
            dim=0,
        )
        eps_tensor = eps_2s + torch.matmul(delta_2s, projection_star)
        eps_xx = eps_tensor[:star_count, :star_count]
        eps_xy = eps_tensor[:star_count, star_count:]
        eps_yx = eps_tensor[star_count:, :star_count]
        eps_yy = eps_tensor[star_count:, star_count:]

        k1 = restrict_scalar(self.K1_norm)
        k2 = restrict_scalar(self.K2_norm)
        s, c = self.sin_zeta, self.cos_zeta

        p11 = (torch.matmul(k1, torch.matmul(inverse_eps_zz_star, k2)) - c * identity) / s
        p12 = (identity - torch.matmul(k1, torch.matmul(inverse_eps_zz_star, k1))) / s
        p21 = (torch.matmul(k2, torch.matmul(inverse_eps_zz_star, k2)) - identity) / s
        p22 = (c * identity - torch.matmul(k2, torch.matmul(inverse_eps_zz_star, k1))) / s
        p_star = torch.cat(
            (torch.cat((p11, p12), dim=1), torch.cat((p21, p22), dim=1)),
            dim=0,
        )

        q11 = (-torch.matmul(k1, k2) - s * eps_yx + c * eps_yy) / s
        q12 = (torch.matmul(k1, k1) - eps_yy) / s
        eps11 = s**2 * eps_xx - s * c * (eps_xy + eps_yx) + c**2 * eps_yy
        q21 = (eps11 - torch.matmul(k2, k2)) / s
        q22 = (torch.matmul(k2, k1) + s * eps_xy - c * eps_yy) / s
        q_star = torch.cat(
            (torch.cat((q11, q12), dim=1), torch.cat((q21, q22), dim=1)),
            dim=0,
        )
        return p_star, q_star

    def add_layer_circle_nvm(
        self,
        thickness,
        radius: float,
        eps_bg,
        eps_cyl,
        centers: Sequence[tuple[float, float]] = ((0.0, 0.0),),
        *,
        core_radius=None,
        eps_core=None,
        use_lanczos: bool = False,
        lanczos_power: int = 2,
        nx: int = 256,
        ny: int = 256,
    ) -> None:
        if not hasattr(self, "Kx_norm"):
            raise RuntimeError(
                "Call set_incident_angle() before add_layer_circle_nvm()."
            )
        if not centers:
            raise ValueError("At least one cylinder center is required.")
        radius, radius_value = _real_parameter_tensor(
            "radius",
            radius,
            dtype=torch.float64,
            device=self._device,
            allow_zero=False,
        )
        core_shell = core_radius is not None or eps_core is not None
        if (core_radius is None) != (eps_core is None):
            raise ValueError(
                "core_radius and eps_core must be supplied together for a "
                "core-shell circle."
            )
        core_radius_value = None
        if core_shell:
            core_radius, core_radius_value = _real_parameter_tensor(
                "core_radius",
                core_radius,
                dtype=torch.float64,
                device=self._device,
                allow_zero=False,
            )
            if core_radius_value >= radius_value:
                raise ValueError("core_radius must be smaller than radius.")
        if not isinstance(lanczos_power, int) or lanczos_power < 0:
            raise ValueError("lanczos_power must be a nonnegative integer.")
        thickness_tensor, _ = _real_parameter_tensor(
            "thickness",
            thickness,
            dtype=self._dtype,
            device=self._device,
            allow_zero=True,
        )
        eps_bg_t = torch.as_tensor(
            eps_bg, dtype=self._dtype, device=self._device
        )
        eps_cyl_t = torch.as_tensor(
            eps_cyl, dtype=self._dtype, device=self._device
        )
        eps_core_t = (
            torch.as_tensor(eps_core, dtype=self._dtype, device=self._device)
            if core_shell
            else None
        )
        materials = (eps_bg_t, eps_cyl_t) + (
            (eps_core_t,) if eps_core_t is not None else ()
        )
        if not all(bool(torch.all(torch.isfinite(value))) for value in materials):
            raise ValueError("NVM permittivities must be finite.")
        if any(bool(torch.any(torch.abs(value) == 0.0)) for value in materials):
            raise ValueError("NVM permittivities must be nonzero.")
        centers = tuple((float(x), float(y)) for x, y in centers)
        if not all(math.isfinite(v) for center in centers for v in center):
            raise ValueError("All circle centers must be finite.")
        if self.group_theory_symmetry == "d6" and self.lattice_kind != "triangular":
            raise UnsupportedCombinationError(
                "symmetry='d6' requires a 60-degree equal-length triangular cell."
            )

        lx, ly = _as_float(self.L[0]), _as_float(self.L[1])
        a1 = (lx, 0.0)
        a2 = (ly * self.cos_zeta, ly * self.sin_zeta)
        tolerance = 1.0e-10 * max(lx, ly)
        for first, center_a in enumerate(centers):
            for second, center_b in enumerate(centers):
                for shift_i in range(-1, 2):
                    for shift_j in range(-1, 2):
                        if first == second and shift_i == 0 and shift_j == 0:
                            continue
                        dx = (
                            center_a[0]
                            - center_b[0]
                            - shift_i * a1[0]
                            - shift_j * a2[0]
                        )
                        dy = (
                            center_a[1]
                            - center_b[1]
                            - shift_i * a1[1]
                            - shift_j * a2[1]
                        )
                        if math.hypot(dx, dy) < 2.0 * radius_value - tolerance:
                            raise ValueError(
                                "Periodic circles overlap; reduce radius or separate centers."
                            )
        layer_index = self.layer_N

        eps_zz = self._circle_toeplitz(
            radius,
            eps_bg,
            eps_cyl,
            centers,
            use_lanczos=use_lanczos,
            lanczos_power=lanczos_power,
        )
        if core_shell:
            # For concentric interfaces the two boundary normals are collinear.
            # The scalar convolution is therefore the outer disk plus an exact
            # analytic correction over the core disk; no rasterized ring is
            # introduced.
            eps_zz = eps_zz + self._circle_toeplitz(
                core_radius,
                0.0,
                eps_core_t - eps_cyl_t,
                centers,
                use_lanczos=use_lanczos,
                lanczos_power=lanczos_power,
            )
        inv_eps = self._circle_toeplitz(
            radius,
            1.0 / eps_bg,
            1.0 / eps_cyl,
            centers,
            use_lanczos=use_lanczos,
            lanczos_power=lanczos_power,
        )
        if core_shell:
            inv_eps = inv_eps + self._circle_toeplitz(
                core_radius,
                0.0,
                1.0 / eps_core_t - 1.0 / eps_cyl_t,
                centers,
                use_lanczos=use_lanczos,
                lanczos_power=lanczos_power,
            )
        inverse_inverse_eps = self._solve(
            inv_eps, self._eye(self.order_N)
        )
        projection = self._projection_matrix(
            core_radius if core_shell else radius,
            centers,
            nx=nx,
            ny=ny,
        )
        delta = inverse_inverse_eps - eps_zz
        zero = torch.zeros_like(delta)
        delta_2n = torch.cat(
            (
                torch.cat((delta, zero), dim=1),
                torch.cat((zero, delta), dim=1),
            ),
            dim=0,
        )
        eps_2n = torch.cat(
            (
                torch.cat((eps_zz, zero), dim=1),
                torch.cat((zero, eps_zz), dim=1),
            ),
            dim=0,
        )
        self._nvm_eps_tensor = eps_2n + torch.matmul(
            delta_2n, projection
        )

        mu_zz = self._eye(self.order_N)
        inv_eps_zz = self._solve(eps_zz, self._eye(self.order_N))
        inv_mu_zz = mu_zz
        n = self.order_N
        eps_xx = self._nvm_eps_tensor[:n, :n]
        eps_xy = self._nvm_eps_tensor[:n, n:]
        eps_yx = self._nvm_eps_tensor[n:, :n]
        eps_yy = self._nvm_eps_tensor[n:, n:]
        s, c = self.sin_zeta, self.cos_zeta
        k1, k2 = self.K1_norm, self.K2_norm

        p11 = (torch.matmul(k1, torch.matmul(inv_eps_zz, k2)) - c * mu_zz) / s
        p12 = (mu_zz - torch.matmul(k1, torch.matmul(inv_eps_zz, k1))) / s
        p21 = (torch.matmul(k2, torch.matmul(inv_eps_zz, k2)) - mu_zz) / s
        p22 = (c * mu_zz - torch.matmul(k2, torch.matmul(inv_eps_zz, k1))) / s
        p = torch.cat(
            (torch.cat((p11, p12), dim=1), torch.cat((p21, p22), dim=1)),
            dim=0,
        )

        q11 = (
            -torch.matmul(k1, torch.matmul(inv_mu_zz, k2))
            - s * eps_yx
            + c * eps_yy
        ) / s
        q12 = (
            torch.matmul(k1, torch.matmul(inv_mu_zz, k1)) - eps_yy
        ) / s
        eps11 = s**2 * eps_xx - s * c * (eps_xy + eps_yx) + c**2 * eps_yy
        q21 = (
            eps11 - torch.matmul(k2, torch.matmul(inv_mu_zz, k2))
        ) / s
        q22 = (
            torch.matmul(k2, torch.matmul(inv_mu_zz, k1))
            + s * eps_xy
            - c * eps_yy
        ) / s
        q = torch.cat(
            (torch.cat((q11, q12), dim=1), torch.cat((q21, q22), dim=1)),
            dim=0,
        )

        identity = self._eye(n)
        zero_n = torch.zeros_like(identity)
        covariant_to_cartesian = torch.cat(
            (
                torch.cat((identity, zero_n), dim=1),
                torch.cat(
                    (
                        -(c / s) * identity,
                        (1.0 / s) * identity,
                    ),
                    dim=1,
                ),
            ),
            dim=0,
        )

        cartesian_modes_ready = False
        complete_d6 = (
            self.use_group_theory
            and self.group_theory_symmetry == "d6"
            and self.polarization_reduction is None
        )
        d6_source = (
            self.use_group_theory
            and self.group_theory_symmetry == "d6"
            and self.polarization_reduction is not None
        )
        triangular_source_reduction = (
            self.lattice_kind == "triangular"
            and self.polarization_reduction is not None
        )
        if complete_d6 or triangular_source_reduction:
            cell_center = (
                0.5 * (a1[0] + a2[0]),
                0.5 * (a1[1] + a2[1]),
            )
            center_error = (
                math.inf
                if len(centers) != 1
                else math.hypot(
                    centers[0][0] - cell_center[0],
                    centers[0][1] - cell_center[1],
                )
            )
            if center_error > tolerance:
                raise UnsupportedCombinationError(
                    "Triangular D6/Cs reduction requires one circle at the "
                    "primitive-cell center."
                )
        if complete_d6:
            triangular_star_pq = self._build_triangular_nvm_star_pq(
                eps_zz, inv_eps, projection
            )
            vector_embedding, _, _, _, _ = self._triangular_star_operators()
            transform_star = (
                vector_embedding.mH
                @ covariant_to_cartesian
                @ vector_embedding
            )
            (
                kz,
                w_cartesian_star,
                h_cartesian_star,
                vector_embedding,
                transform_inverse_star,
            ) = self._complete_d6_eigendecomposition(
                *triangular_star_pq,
                transform_star,
                layer_index=layer_index,
                backend="NVM",
            )
            w_covariant = vector_embedding @ (
                transform_inverse_star @ w_cartesian_star
            )
            h_covariant = vector_embedding @ (
                transform_inverse_star @ h_cartesian_star
            )
            w_cartesian = vector_embedding @ w_cartesian_star
            h_cartesian = vector_embedding @ h_cartesian_star
            self._register_complete_d6_layer(
                vector_embedding,
                w_cartesian_star,
                h_cartesian_star,
                kz,
            )
            cartesian_modes_ready = True
        elif d6_source:
            triangular_star_pq = self._build_triangular_nvm_star_pq(
                eps_zz, inv_eps, projection
            )
            vector_embedding, _, _, _, _ = self._triangular_star_operators()
            transform_star = (
                vector_embedding.mH
                @ covariant_to_cartesian
                @ vector_embedding
            )
            (
                kz,
                w_cartesian_star,
                h_cartesian_star,
                vector_embedding,
                transform_inverse_star,
            ) = self._d6_source_eigendecomposition(
                *triangular_star_pq,
                transform_star,
                layer_index=layer_index,
                backend="NVM",
            )
            w_covariant = vector_embedding @ (
                transform_inverse_star @ w_cartesian_star
            )
            h_covariant = vector_embedding @ (
                transform_inverse_star @ h_cartesian_star
            )
            w_cartesian = vector_embedding @ w_cartesian_star
            h_cartesian = vector_embedding @ h_cartesian_star
            cartesian_modes_ready = True
        elif self.polarization_reduction is not None:
            if self.lattice_kind == "triangular":
                triangular_star_pq = self._build_triangular_nvm_star_pq(
                    eps_zz, inv_eps, projection
                )
                (
                    kz,
                    w_covariant,
                    h_covariant,
                    w_cartesian,
                    h_cartesian,
                ) = self._matched_polarization_eigendecomposition(
                    p,
                    q,
                    covariant_to_cartesian,
                    layer_index=layer_index,
                    triangular_star_pq=triangular_star_pq,
                    backend="NVM",
                )
                cartesian_modes_ready = True
            else:
                kz, w_covariant, h_covariant = self._polarization_eigendecomposition(
                    p, q, centers, layer_index
                )
        else:
            grouped = (
                self._group_theory_eigendecomposition(
                    p, q, centers, layer_index
                )
                if self.use_group_theory
                else None
            )
            if grouped is None:
                kz_squared, w_covariant = self._eig(torch.matmul(p, q))
                kz = self._positive_kz(kz_squared)
                if not self.use_group_theory:
                    self.group_theory_diagnostics.append(
                        {
                            "layer": layer_index,
                            "requested": False,
                            "applied": False,
                            "symmetry": None,
                            "reason": "disabled",
                        }
                    )
            else:
                kz, w_covariant = grouped
            h_covariant = self._magnetic_eigenvectors(
                p, q, w_covariant, kz
            )
        if not cartesian_modes_ready:
            w_cartesian = torch.matmul(
                covariant_to_cartesian, w_covariant
            )
            h_cartesian = torch.matmul(
                covariant_to_cartesian, h_covariant
            )
        if self.polarization_reduction is not None:
            assert self._polarization_bases is not None
            electric_basis, magnetic_basis = self._polarization_bases
            self._polarized_layers[-1]["electric"] = torch.matmul(
                electric_basis.mH, w_cartesian
            )
            self._polarized_layers[-1]["magnetic"] = torch.matmul(
                magnetic_basis.mH, h_cartesian
            )

        self.layer_N += 1
        self.thickness.append(thickness_tensor)
        self.eps_conv.append(eps_zz)
        self.mu_conv.append(mu_zz)
        self.P.append(p)
        self.Q.append(q)
        self.kz_norm.append(kz)
        self.E_eigvec.append(w_cartesian)
        self.H_eigvec.append(h_cartesian)
        self._nvm_eps_tensor_by_layer[layer_index] = self._nvm_eps_tensor
        if (
            self.store_mode_couplings
            and self.lattice_kind == "triangular"
            and (complete_d6 or self.polarization_reduction is not None)
        ):
            vector_embedding, _, _, _, _ = self._triangular_star_operators()
            star_count = vector_embedding.shape[1] // 2
            scalar_embedding = vector_embedding[: self.order_N, :star_count]
            self._reduced_longitudinal_context_by_layer[layer_index] = {
                "scalar_embedding": scalar_embedding,
                "eps_conv_reduced": scalar_embedding.mH @ eps_zz @ scalar_embedding,
                "mu_conv_reduced": scalar_embedding.mH @ mu_zz @ scalar_embedding,
            }

        xi = (
            torch.arange(nx, dtype=torch.float64, device=self._device) / nx
        ) * lx
        eta = (
            torch.arange(ny, dtype=torch.float64, device=self._device) / ny
        ) * ly
        xi_grid, eta_grid = torch.meshgrid(xi, eta, indexing="ij")
        x_grid = xi_grid + eta_grid * self.cos_zeta
        y_grid = eta_grid * self.sin_zeta
        distances = []
        for center_x, center_y in centers:
            for shift_i in range(-2, 3):
                for shift_j in range(-2, 3):
                    image_x = center_x + shift_i * a1[0] + shift_j * a2[0]
                    image_y = center_y + shift_i * a1[1] + shift_j * a2[1]
                    distances.append(
                        (x_grid - image_x) ** 2 + (y_grid - image_y) ** 2
                    )
        inside = torch.min(torch.stack(distances), dim=0).values <= radius**2
        eps_physical = torch.where(
            inside,
            torch.as_tensor(eps_cyl, dtype=self._dtype, device=self._device),
            torch.as_tensor(eps_bg, dtype=self._dtype, device=self._device),
        )
        if core_shell:
            core_inside = (
                torch.min(torch.stack(distances), dim=0).values <= core_radius**2
            )
            eps_physical = torch.where(core_inside, eps_core_t, eps_physical)
        self._physical_material_by_layer[layer_index] = (
            eps_physical,
            torch.ones_like(eps_physical),
        )
        self.layer_records.append(
            LayerRecord(
                index=layer_index,
                method="nvm",
                shape=(
                    "core-shell-circle"
                    if core_shell and len(centers) == 1
                    else "core-shell-circle-array"
                    if core_shell
                    else "circle"
                    if len(centers) == 1
                    else "circle-array"
                ),
                lattice=getattr(self, "lattice_kind", "rectangular"),
                reason=(
                    "CURVED_ANALYTIC_CONCENTRIC_BOUNDARIES"
                    if core_shell
                    else "CURVED_ANALYTIC_BOUNDARY"
                ),
                options={
                    "radius": radius_value,
                    "core_radius": core_radius_value,
                    "centers": centers,
                    "grid": (nx, ny),
                    "use_lanczos": use_lanczos,
                    "lanczos_power": lanczos_power,
                    "factorization_scheme": (
                        "concentric-normal-vector"
                        if core_shell
                        else "normal-vector"
                    ),
                    "group_theory": dict(self.group_theory_diagnostics[-1]),
                },
            )
        )
        if self.polarization_reduction is None and not complete_d6:
            self._append_smatrix_from_cartesian_modes(
                w_cartesian, h_cartesian
            )
        self._nvm_eps_tensor = None

    def add_layer_circle_shell_nvm(
        self,
        thickness,
        core_radius,
        outer_radius,
        eps_bg,
        eps_shell,
        eps_core,
        centers: Sequence[tuple[float, float]] = ((0.0, 0.0),),
        *,
        use_lanczos: bool = False,
        lanczos_power: int = 2,
        nx: int = 256,
        ny: int = 256,
    ) -> None:
        """Add an analytic concentric core-shell layer using one NV field.

        Both circular interfaces have the same radial normal direction.  A
        single periodic normal-vector projector is therefore sufficient, while
        the Fourier coefficients of ``epsilon`` and ``1/epsilon`` include both
        radii analytically.  This avoids a hard-raster annulus and provides an
        independent, non-ASR convergence path for coaxial structures.
        """

        self.add_layer_circle_nvm(
            thickness,
            outer_radius,
            eps_bg,
            eps_shell,
            centers,
            core_radius=core_radius,
            eps_core=eps_core,
            use_lanczos=use_lanczos,
            lanczos_power=lanczos_power,
            nx=nx,
            ny=ny,
        )

    def solve_global_smatrix(self) -> None:
        if self.polarization_reduction is not None or self._native_d6_active:
            self._solve_polarization_reduced_smatrix()
            return
        super().solve_global_smatrix()
