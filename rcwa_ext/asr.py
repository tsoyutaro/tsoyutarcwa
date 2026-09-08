"""ASR tensor factorization, modal operators, and layer assembly."""

from __future__ import annotations

import math

import torch

from .asr_maps import ASRMapping, CircleASRMapping, _ASRMappingMixin
from .config import (
    LayerRecord, UnsupportedCombinationError, _ORIGINAL_TORCWA_RCWA,
    _TWO_PI, _as_float, _real_parameter_tensor,
)
from .scattering import _StableLinearAlgebraMixin

class CustomRCWA_ASR_FR(_ASRMappingMixin, _StableLinearAlgebraMixin, _ORIGINAL_TORCWA_RCWA):
    """
    Paper-faithful 2D ASR-FR solver for centered rectangular patches.

    Required call order is the same as torcwa:

        sim = CustomRCWA_ASR_FR(freq, order, L, ...)
        sim.add_input_layer(...)       # optional
        sim.add_output_layer(...)      # optional
        sim.set_incident_angle(theta, phi)
        sim.add_layer_metal_patch_asr(...)
        sim.solve_global_smatrix()
    """

    def __init__(self, freq, order, L, **kwargs):
        self.asr_G = float(kwargs.pop("asr_G", 1.0e-3))
        quadrature_grid = kwargs.pop("asr_quadrature_grid", None)
        if quadrature_grid is not None:
            if (
                isinstance(quadrature_grid, bool)
                or int(quadrature_grid) != quadrature_grid
                or int(quadrature_grid) < 32
            ):
                raise ValueError("asr_quadrature_grid must be an integer >= 32.")
            quadrature_grid = int(quadrature_grid)
        self.asr_quadrature_grid = quadrature_grid
        self.matched_asr_G = float(kwargs.pop("matched_asr_G", 3.0e-2))
        profile_aliases = {
            "equalized": "equalized",
            "balanced": "equalized",
            "default": "equalized",
            "weiss2009": "weiss2009",
            "weiss-2009": "weiss2009",
            "paper": "weiss2009",
            "identity": "identity",
            "none": "identity",
            "no-asr": "identity",
        }
        raw_profile = str(
            kwargs.pop("matched_asr_profile", "equalized")
        ).strip().lower().replace("_", "-")
        self.matched_asr_profile = profile_aliases.get(raw_profile)
        if self.matched_asr_profile is None:
            raise ValueError(
                "matched_asr_profile must be 'equalized', 'weiss2009', "
                "or 'identity'."
            )
        self.matched_asr_min_jacobian = float(
            kwargs.pop("matched_asr_min_jacobian", 1.0e-12)
        )
        if not 0.0 < self.asr_G < 1.0:
            raise ValueError("asr_G must be in (0, 1); the paper uses 0.001.")
        if not 0.0 < self.matched_asr_G < 1.0:
            raise ValueError(
                "matched_asr_G must be in (0,1); 0.03 corresponds to "
                "Weiss et al. eta=0.97."
            )
        if (
            not math.isfinite(self.matched_asr_min_jacobian)
            or self.matched_asr_min_jacobian <= 0.0
        ):
            raise ValueError(
                "matched_asr_min_jacobian must be finite and positive."
            )
        self.lattice_kind = "rectangular"
        super().__init__(freq, order, L, **kwargs)
        self.asr_mappings: list[ASRMapping | CircleASRMapping] = []
        self.asr_T_matrices: list[torch.Tensor] = []
        self.asr_Tz_matrices: list[torch.Tensor | None] = []
        self.asr_condition_numbers: list[torch.Tensor | None] = []
        self.asr_material_tensors: list[dict[str, torch.Tensor]] = []
        self.E_eigvec_uv: list[torch.Tensor] = []
        self.H_eigvec_uv: list[torch.Tensor] = []
        self._last_asr_transform_condition: torch.Tensor | None = None

    def solve_global_smatrix(self) -> None:
        """Dispatch to Redheffer or Li-2a while preserving field couplings."""
        super().solve_global_smatrix()

    def _axis_toeplitz_samples(
        self, field_uv: torch.Tensor, axis: str
    ) -> torch.Tensor:
        """Finite 1-D Toeplitz operators at every sample of the other axis."""
        field_uv = field_uv.to(self._dtype)
        if axis == "v":
            coefficients = torch.fft.fft(field_uv, dim=1) / field_uv.shape[1]
            delta = self.order_y[:, None] - self.order_y[None, :]
            return coefficients[:, delta]
        if axis == "u":
            coefficients = torch.fft.fft(field_uv, dim=0) / field_uv.shape[0]
            delta = self.order_x[:, None] - self.order_x[None, :]
            return coefficients[delta, :].permute(2, 0, 1)
        raise ValueError("axis must be 'u' or 'v'.")

    def _assemble_matrix_valued_toeplitz(
        self, samples: torch.Tensor, outer_axis: str
    ) -> torch.Tensor:
        """Fourier-expand a matrix-valued function along its outer axis."""
        coefficients = torch.fft.fft(samples, dim=0) / samples.shape[0]
        mx, my = len(self.order_x), len(self.order_y)
        result = torch.zeros(
            (mx * my, mx * my), dtype=self._dtype, device=self._device
        )
        if outer_axis == "u":
            delta = self.order_x[:, None] - self.order_x[None, :]
            for p in range(mx):
                rows = slice(p * my, (p + 1) * my)
                for pp in range(mx):
                    cols = slice(pp * my, (pp + 1) * my)
                    result[rows, cols] = coefficients[delta[p, pp]]
            return result
        if outer_axis == "v":
            delta = self.order_y[:, None] - self.order_y[None, :]
            for p in range(mx):
                rows = slice(p * my, (p + 1) * my)
                for pp in range(mx):
                    cols = slice(pp * my, (pp + 1) * my)
                    result[rows, cols] = coefficients[delta, p, pp]
            return result
        raise ValueError("outer_axis must be 'u' or 'v'.")

    def _symmetric_factorized_transverse_tensor(
        self,
        tensor11: torch.Tensor,
        tensor12: torch.Tensor,
        tensor21: torch.Tensor,
        tensor22: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Weiss et al. symmetric factorization, Eqs. (29)-(36).

        The 2->1 construction supplies (11,12); the symmetry-related 1->2
        construction supplies (21,22).  This avoids the polarization asymmetry
        caused by choosing only one directional factorization order.
        """
        determinant = tensor11 * tensor22 - tensor12 * tensor21
        my = len(self.order_y)
        identity_v = torch.eye(
            my, dtype=self._dtype, device=self._device
        ).expand(tensor11.shape[0], my, my)

        inverse_22_v = self._axis_toeplitz_samples(1.0 / tensor22, "v")
        ratio_12_v = self._axis_toeplitz_samples(tensor12 / tensor22, "v")
        ratio_21_v = self._axis_toeplitz_samples(tensor21 / tensor22, "v")
        schur_11_v = self._axis_toeplitz_samples(
            determinant / tensor22, "v"
        )
        effective_22_v = self._solve(inverse_22_v, identity_v)
        effective_21_v = self._solve(inverse_22_v, ratio_21_v)
        effective_12_v = torch.matmul(ratio_12_v, effective_22_v)
        effective_11_v = schur_11_v + torch.matmul(
            ratio_12_v, effective_21_v
        )

        effective_11_v_inverse = self._solve(
            effective_11_v, identity_v
        )
        outer_11_inverse = self._assemble_matrix_valued_toeplitz(
            effective_11_v_inverse, "u"
        )
        result11 = self._solve(
            outer_11_inverse, self._eye(self.order_N)
        )
        result12 = torch.matmul(
            result11,
            self._assemble_matrix_valued_toeplitz(
                torch.matmul(effective_11_v_inverse, effective_12_v), "u"
            ),
        )

        mx = len(self.order_x)
        identity_u = torch.eye(
            mx, dtype=self._dtype, device=self._device
        ).expand(tensor11.shape[1], mx, mx)
        inverse_11_u = self._axis_toeplitz_samples(1.0 / tensor11, "u")
        ratio_12_u = self._axis_toeplitz_samples(tensor12 / tensor11, "u")
        ratio_21_u = self._axis_toeplitz_samples(tensor21 / tensor11, "u")
        schur_22_u = self._axis_toeplitz_samples(
            determinant / tensor11, "u"
        )
        effective_11_u = self._solve(inverse_11_u, identity_u)
        effective_12_u = self._solve(inverse_11_u, ratio_12_u)
        effective_21_u = torch.matmul(ratio_21_u, effective_11_u)
        effective_22_u = schur_22_u + torch.matmul(
            ratio_21_u, effective_12_u
        )

        effective_22_u_inverse = self._solve(
            effective_22_u, identity_u
        )
        outer_22_inverse = self._assemble_matrix_valued_toeplitz(
            effective_22_u_inverse, "v"
        )
        result22 = self._solve(
            outer_22_inverse, self._eye(self.order_N)
        )
        result21 = torch.matmul(
            result22,
            self._assemble_matrix_valued_toeplitz(
                torch.matmul(effective_22_u_inverse, effective_21_u), "v"
            ),
        )
        return result11, result12, result21, result22

    def _generalized_li_factorized_transverse_tensor(
        self,
        tensor11: torch.Tensor,
        tensor12: torch.Tensor,
        tensor21: torch.Tensor,
        tensor22: torch.Tensor,
        normal_u: torch.Tensor,
        normal_v: torch.Tensor,
        normal_weight: torch.Tensor | None = None,
        *,
        convolution=None,
        harmonic_count: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply Li's inverse rule to the local normal/tangential traces.

        ``A`` maps covariant electric components to contravariant electric
        displacement.  With the undeformed computational metric ``g``, set
        ``C=g A`` and ``B=Pt+Pn C``.  The auxiliary field ``G=B E`` consists
        of the continuous tangential E and normal D traces.  Consequently,

            C_eff = [C B^-1] [B^-1]^-1,
            A_eff = g^-1 C_eff.

        This is the generalized Li factorization needed by a non-separable
        double-matched map.  For scalar epsilon and a u-normal interface it
        reduces exactly to ``diag([1/epsilon]^-1, [epsilon])``.
        """
        conv = self._material_conv if convolution is None else convolution
        count = self.order_N if harmonic_count is None else int(harmonic_count)
        if count <= 0:
            raise ValueError("harmonic_count must be positive.")
        if normal_u.shape != tensor11.shape or normal_v.shape != tensor11.shape:
            raise ValueError(
                "Generalized Li normals must match the tensor sampling shape."
            )

        real_dtype = tensor11.real.dtype
        dtype = tensor11.dtype
        cosine_real = torch.as_tensor(
            _as_float(getattr(self, "cos_zeta", 0.0)),
            dtype=real_dtype,
            device=self._device,
        )
        sine_squared_real = 1.0 - cosine_real**2
        if _as_float(sine_squared_real) <= 1.0e-12:
            raise UnsupportedCombinationError(
                "Generalized Li factorization requires a non-degenerate lattice metric."
            )
        cosine = cosine_real.to(dtype)
        sine_squared = sine_squared_real.to(dtype)

        # n_flat=d(rho), n_sharp=g^-1 n_flat, and
        # Pn=n_flat outer n_sharp/(n_flat.n_sharp).
        n1 = normal_u.to(real_dtype)
        n2 = normal_v.to(real_dtype)
        n_sharp_1 = (n1 - cosine_real * n2) / sine_squared_real
        n_sharp_2 = (n2 - cosine_real * n1) / sine_squared_real
        normal_squared = n1 * n_sharp_1 + n2 * n_sharp_2
        normal_tolerance = 64.0 * torch.finfo(real_dtype).eps
        inverse_normal_squared = torch.where(
            normal_squared > normal_tolerance,
            1.0 / torch.clamp(normal_squared, min=normal_tolerance),
            torch.zeros_like(normal_squared),
        )
        p11 = (n1 * n_sharp_1 * inverse_normal_squared).to(dtype)
        p12 = (n1 * n_sharp_2 * inverse_normal_squared).to(dtype)
        p21 = (n2 * n_sharp_1 * inverse_normal_squared).to(dtype)
        p22 = (n2 * n_sharp_2 * inverse_normal_squared).to(dtype)
        if normal_weight is not None:
            if normal_weight.shape != tensor11.shape:
                raise ValueError(
                    "Generalized Li normal weight must match the tensor sampling shape."
                )
            weight = torch.clamp(normal_weight.to(real_dtype), 0.0, 1.0).to(dtype)
            # The normal extension is arbitrary away from material interfaces.
            # Blend to the isotropic trace-one projector at the circle centre
            # and periodic Voronoi seams.  The physical boundaries retain
            # weight=1 and are therefore unchanged.
            half = torch.as_tensor(0.5, dtype=dtype, device=self._device)
            p11 = half + weight * (p11 - half)
            p12 = weight * p12
            p21 = weight * p21
            p22 = half + weight * (p22 - half)

        # C=g A for g=[[1,cos(zeta)],[cos(zeta),1]].
        c11 = tensor11 + cosine * tensor21
        c12 = tensor12 + cosine * tensor22
        c21 = cosine * tensor11 + tensor21
        c22 = cosine * tensor12 + tensor22

        one = torch.ones_like(c11)
        pc11 = p11 * c11 + p12 * c21
        pc12 = p11 * c12 + p12 * c22
        pc21 = p21 * c11 + p22 * c21
        pc22 = p21 * c12 + p22 * c22
        b11 = one - p11 + pc11
        b12 = -p12 + pc12
        b21 = -p21 + pc21
        b22 = one - p22 + pc22
        determinant_b = b11 * b22 - b12 * b21
        determinant_scale = torch.maximum(
            torch.maximum(torch.abs(b11 * b22), torch.abs(b12 * b21)),
            torch.ones_like(torch.abs(determinant_b)),
        )
        if bool(
            torch.any(
                torch.abs(determinant_b)
                <= 128.0 * torch.finfo(real_dtype).eps * determinant_scale
            )
        ):
            raise RuntimeError(
                "The generalized Li continuous-field transform is singular; "
                "check material parameters and the matched map."
            )
        inverse_b11 = b22 / determinant_b
        inverse_b12 = -b12 / determinant_b
        inverse_b21 = -b21 / determinant_b
        inverse_b22 = b11 / determinant_b

        u11 = c11 * inverse_b11 + c12 * inverse_b21
        u12 = c11 * inverse_b12 + c12 * inverse_b22
        u21 = c21 * inverse_b11 + c22 * inverse_b21
        u22 = c21 * inverse_b12 + c22 * inverse_b22

        def block(values: tuple[torch.Tensor, ...]) -> torch.Tensor:
            a11, a12, a21, a22 = (conv(value) for value in values)
            return torch.cat(
                (
                    torch.cat((a11, a12), dim=1),
                    torch.cat((a21, a22), dim=1),
                ),
                dim=0,
            )

        inverse_b_conv = block(
            (inverse_b11, inverse_b12, inverse_b21, inverse_b22)
        )
        u_conv = block((u11, u12, u21, u22))
        if inverse_b_conv.shape != (2 * count, 2 * count):
            raise RuntimeError(
                "Generalized Li convolution returned an inconsistent harmonic dimension."
            )
        # Right solve: C_eff=U_conv @ inverse(inverse_B_conv).
        c_effective = self._solve(inverse_b_conv.mT, u_conv.mT).mT
        ce11 = c_effective[:count, :count]
        ce12 = c_effective[:count, count:]
        ce21 = c_effective[count:, :count]
        ce22 = c_effective[count:, count:]

        inverse_metric = 1.0 / sine_squared
        a11 = inverse_metric * (ce11 - cosine * ce21)
        a12 = inverse_metric * (ce12 - cosine * ce22)
        a21 = inverse_metric * (ce21 - cosine * ce11)
        a22 = inverse_metric * (ce22 - cosine * ce12)
        return a11, a12, a21, a22

    def _pulled_back_circle_normal(
        self,
        mapping: CircleASRMapping,
        taper_radius,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the physical radial normal as a computational covector.

        If ``F(u,v)=(x,y)`` is the matched-coordinate map and
        ``phi=(x-xc)^2+(y-yc)^2``, the material-interface covector in the
        computational coordinates is

            d(phi o F) = (phi_x x_u + phi_y y_u) du
                       + (phi_x x_v + phi_y y_v) dv.

        Both interfaces of a concentric core-shell circle are level sets of
        the same ``phi`` and therefore share this normal field.  The nearest
        periodic image is selected so that the extension is periodic.  Its
        scale is immaterial because the generalized Li projector normalizes
        the covector pointwise.
        """

        lx, ly = _as_float(self.L[0]), _as_float(self.L[1])
        cosine = float(getattr(self, "cos_zeta", 0.0))
        sine = float(getattr(self, "sin_zeta", 1.0))
        center_x = 0.5 * (lx + cosine * ly)
        center_y = 0.5 * sine * ly
        dx_candidates: list[torch.Tensor] = []
        dy_candidates: list[torch.Tensor] = []
        distance_candidates: list[torch.Tensor] = []
        for shift_i in (-1, 0, 1):
            for shift_j in (-1, 0, 1):
                image_x = center_x + shift_i * lx + shift_j * cosine * ly
                image_y = center_y + shift_j * sine * ly
                dx = mapping.x - image_x
                dy = mapping.y - image_y
                dx_candidates.append(dx)
                dy_candidates.append(dy)
                distance_candidates.append(dx**2 + dy**2)
        two_smallest, indices = torch.topk(
            torch.stack(distance_candidates, dim=0),
            k=2,
            dim=0,
            largest=False,
        )
        nearest = indices[:1]
        dx = torch.gather(torch.stack(dx_candidates, dim=0), 0, nearest)[0]
        dy = torch.gather(torch.stack(dy_candidates, dim=0), 0, nearest)[0]
        normal_u = dx * mapping.x_u + dy * mapping.y_u
        normal_v = dx * mapping.x_v + dy * mapping.y_v
        radius_nearest = torch.sqrt(torch.clamp(two_smallest[0], min=0.0))
        radius_second = torch.sqrt(torch.clamp(two_smallest[1], min=0.0))
        real_dtype = mapping.x.real.dtype
        taper = torch.as_tensor(
            taper_radius, dtype=real_dtype, device=self._device
        )
        resolution = torch.as_tensor(
            max(lx / mapping.x.shape[0], ly / mapping.x.shape[1]),
            dtype=real_dtype,
            device=self._device,
        )

        def smoothstep(value: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
            parameter = torch.clamp(value / upper, 0.0, 1.0)
            return parameter**2 * (3.0 - 2.0 * parameter)

        center_width = torch.maximum(0.5 * taper, resolution)
        center_weight = smoothstep(radius_nearest, center_width)
        seam_weight = smoothstep(radius_second - radius_nearest, 3.0 * resolution)
        return normal_u, normal_v, center_weight * seam_weight

    def _build_circle_conversion_matrices(
        self, mapping: CircleASRMapping
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """General 2-D modal conversion for a non-separable circle map.

        If A = det(J) J^{-T}, covariant transverse fields obey

            [E_x,E_y]^T dxdy = A [E_u,E_v]^T dudv.

        The four blocks below are therefore the non-separable extension of
        Wang et al. Eqs. (19)-(22).  FFTs evaluate all transformed orders for
        one Cartesian row at once.  ``Tz`` uses the area weight det(J).
        """
        mx, my = len(self.order_x), len(self.order_y)
        nx, ny = mapping.x.shape
        k1_grid = getattr(self, "K1_norm_dn", self.Kx_norm_dn).reshape(mx, my)
        k2_grid = getattr(self, "K2_norm_dn", self.Ky_norm_dn).reshape(mx, my)
        if not torch.allclose(k1_grid, k1_grid[:, :1].expand_as(k1_grid)):
            raise RuntimeError("The first covariant wave number must depend on order_x only.")
        if not torch.allclose(k2_grid, k2_grid[:1, :].expand_as(k2_grid)):
            raise RuntimeError("The second covariant wave number must depend on order_y only.")

        zero_p = int(torch.nonzero(self.order_x == 0, as_tuple=False)[0])
        zero_q = int(torch.nonzero(self.order_y == 0, as_tuple=False)[0])
        k1_incident = k1_grid[zero_p, zero_q]
        k2_incident = k2_grid[zero_p, zero_q]
        omega = torch.as_tensor(
            self.omega, dtype=self._dtype, device=self._device
        )
        u_grid = mapping.u[:, None].to(self._dtype)
        v_grid = mapping.v[None, :].to(self._dtype)
        x_grid = mapping.x.to(self._dtype)
        y_grid = mapping.y.to(self._dtype)
        computational_bloch = torch.exp(
            1.0j
            * omega
            * (k1_incident * u_grid + k2_incident * v_grid)
        )

        # det(J) J^{-T}/sin(zeta).  Division by the primitive-cell area
        # ratio makes an undeformed oblique map reduce exactly to J^{-T}.
        area_ratio = float(getattr(self, "sin_zeta", 1.0))
        weights = torch.stack(
            (
                mapping.y_v,
                -mapping.y_u,
                -mapping.x_v,
                mapping.x_u,
                mapping.det_j,
            ),
            dim=0,
        ).to(self._dtype) / area_ratio
        blocks = torch.zeros(
            (5, self.order_N, self.order_N),
            dtype=self._dtype,
            device=self._device,
        )
        column_x = torch.remainder(self.order_x, nx).to(torch.int64)
        column_y = torch.remainder(self.order_y, ny).to(torch.int64)
        kx_rows = self.Kx_norm_dn.reshape(-1)
        ky_rows = self.Ky_norm_dn.reshape(-1)
        for row in range(self.order_N):
            physical_phase = torch.exp(
                -1.0j
                * omega
                * (kx_rows[row] * x_grid + ky_rows[row] * y_grid)
            )
            spectra = torch.fft.ifft2(
                weights * (computational_bloch * physical_phase)[None, :, :],
                dim=(-2, -1),
            )
            blocks[:, row, :] = spectra[
                :, column_x[:, None], column_y[None, :]
            ].reshape(5, self.order_N)

        transform_xy = torch.cat(
            (
                torch.cat((blocks[0], blocks[1]), dim=1),
                torch.cat((blocks[2], blocks[3]), dim=1),
            ),
            dim=0,
        )
        return transform_xy, blocks[4]

    def _build_circle_asr_pq(
        self,
        eps11: torch.Tensor,
        eps12: torch.Tensor,
        eps21: torch.Tensor,
        eps22: torch.Tensor,
        eps33: torch.Tensor,
        mu11: torch.Tensor,
        mu12: torch.Tensor,
        mu21: torch.Tensor,
        mu22: torch.Tensor,
        mu33: torch.Tensor,
        *,
        factorization_rules: bool,
        normal_factorize_mu: bool = False,
        factorization_normals: (
            tuple[torch.Tensor, torch.Tensor]
            | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            | None
        ) = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        eps33_conv = self._material_conv(eps33)
        mu33_conv = self._material_conv(mu33)
        inv_eps33 = self._solve(eps33_conv, self._eye(self.order_N))
        inv_mu33 = self._solve(mu33_conv, self._eye(self.order_N))
        if factorization_rules and factorization_normals is not None:
            eps11_m, eps12_m, eps21_m, eps22_m = (
                self._generalized_li_factorized_transverse_tensor(
                    eps11,
                    eps12,
                    eps21,
                    eps22,
                    *factorization_normals,
                )
            )
            if normal_factorize_mu:
                mu11_m, mu12_m, mu21_m, mu22_m = (
                    self._generalized_li_factorized_transverse_tensor(
                        mu11,
                        mu12,
                        mu21,
                        mu22,
                        *factorization_normals,
                    )
                )
            else:
                # The NV correction belongs to each discontinuous constitutive
                # product.  For the usual nonmagnetic structures mu has no
                # material jump (Peng et al., Eqs. 8--10 only modify epsilon);
                # applying the nonlinear normal/tangential transform to the
                # coordinate metric as well needlessly squares the conditioning
                # of a strongly compressed ASR map.  Retain symmetric ASR
                # factorization for that smooth transformed mu tensor.
                mu11_m, mu12_m, mu21_m, mu22_m = (
                    self._symmetric_factorized_transverse_tensor(
                        mu11, mu12, mu21, mu22
                    )
                )
        elif factorization_rules:
            eps11_m, eps12_m, eps21_m, eps22_m = (
                self._symmetric_factorized_transverse_tensor(
                    eps11, eps12, eps21, eps22
                )
            )
            mu11_m, mu12_m, mu21_m, mu22_m = (
                self._symmetric_factorized_transverse_tensor(
                    mu11, mu12, mu21, mu22
                )
            )
        else:
            eps11_m, eps12_m, eps21_m, eps22_m = (
                self._material_conv(eps11),
                self._material_conv(eps12),
                self._material_conv(eps21),
                self._material_conv(eps22),
            )
            mu11_m, mu12_m, mu21_m, mu22_m = (
                self._material_conv(mu11),
                self._material_conv(mu12),
                self._material_conv(mu21),
                self._material_conv(mu22),
            )

        ku = getattr(self, "K1_norm", self.Kx_norm)
        kv = getattr(self, "K2_norm", self.Ky_norm)
        p11 = mu21_m + torch.matmul(ku, torch.matmul(inv_eps33, kv))
        p12 = mu22_m - torch.matmul(ku, torch.matmul(inv_eps33, ku))
        p21 = torch.matmul(kv, torch.matmul(inv_eps33, kv)) - mu11_m
        p22 = -mu12_m - torch.matmul(kv, torch.matmul(inv_eps33, ku))
        p = torch.cat(
            (torch.cat((p11, p12), dim=1), torch.cat((p21, p22), dim=1)),
            dim=0,
        )
        q11 = -eps21_m - torch.matmul(ku, torch.matmul(inv_mu33, kv))
        q12 = torch.matmul(ku, torch.matmul(inv_mu33, ku)) - eps22_m
        q21 = eps11_m - torch.matmul(kv, torch.matmul(inv_mu33, kv))
        q22 = eps12_m + torch.matmul(kv, torch.matmul(inv_mu33, ku))
        q = torch.cat(
            (torch.cat((q11, q12), dim=1), torch.cat((q21, q22), dim=1)),
            dim=0,
        )
        return p, q, eps33_conv, mu33_conv

    def _piecewise_rectangular_conv(
        self,
        values: torch.Tensor,
        u_breaks: torch.Tensor,
        v_breaks: torch.Tensor,
    ) -> torch.Tensor:
        """Exact BTTB convolution for a rectangular piecewise-constant grid.

        Peng--Zhang first staircase the circular boundary and then place every
        material jump at an adaptive-coordinate breakpoint.  Each resulting
        uv rectangle is constant, so its Fourier coefficient can be integrated
        analytically instead of being obtained from an FFT raster.  This is
        both closer to their Laurent-rule statement and important for the very
        high Ag/air contrast.
        """
        if values.ndim != 2:
            raise ValueError("Piecewise rectangular values must be two-dimensional.")
        if values.shape != (u_breaks.numel() - 1, v_breaks.numel() - 1):
            raise ValueError("Piecewise values do not match the breakpoint grid.")
        if bool(torch.any(u_breaks[1:] <= u_breaks[:-1])) or bool(
            torch.any(v_breaks[1:] <= v_breaks[:-1])
        ):
            raise ValueError("Piecewise breakpoints must be strictly increasing.")

        mx, my = len(self.order_x), len(self.order_y)
        dx = self.order_x[:, None] - self.order_x[None, :]
        dy = self.order_y[:, None] - self.order_y[None, :]
        minimum_dx = int(torch.min(dx).detach().cpu().item())
        maximum_dx = int(torch.max(dx).detach().cpu().item())
        minimum_dy = int(torch.min(dy).detach().cpu().item())
        maximum_dy = int(torch.max(dy).detach().cpu().item())
        unique_x = torch.arange(
            minimum_dx,
            maximum_dx + 1,
            dtype=torch.float64,
            device=self._device,
        )
        unique_y = torch.arange(
            minimum_dy,
            maximum_dy + 1,
            dtype=torch.float64,
            device=self._device,
        )

        def interval_integrals(
            harmonics: torch.Tensor, breaks: torch.Tensor
        ) -> torch.Tensor:
            length = breaks[-1] - breaks[0]
            scaled0 = (breaks[:-1] - breaks[0]) / length
            scaled1 = (breaks[1:] - breaks[0]) / length
            harmonic_grid = harmonics[:, None]
            phase0 = torch.exp(-1.0j * _TWO_PI * harmonic_grid * scaled0[None, :])
            phase1 = torch.exp(-1.0j * _TWO_PI * harmonic_grid * scaled1[None, :])
            safe_harmonic = torch.where(
                harmonic_grid == 0.0,
                torch.ones_like(harmonic_grid),
                harmonic_grid,
            )
            nonzero = (phase1 - phase0) / (-1.0j * _TWO_PI * safe_harmonic)
            zero = (scaled1 - scaled0)[None, :].expand_as(nonzero)
            return torch.where(harmonic_grid == 0.0, zero, nonzero).to(self._dtype)

        integral_x = interval_integrals(unique_x, u_breaks)
        integral_y = interval_integrals(unique_y, v_breaks)
        coefficient_grid = torch.matmul(
            integral_x,
            torch.matmul(values.to(self._dtype), integral_y.mT),
        )
        index_x = (dx - minimum_dx).to(torch.int64)
        index_y = (dy - minimum_dy).to(torch.int64)
        return coefficient_grid[
            index_x[:, None, :, None], index_y[None, :, None, :]
        ].reshape(mx * my, mx * my)

    def _peng_eq8_transverse_epsilon(
        self,
        epsilon_uv: torch.Tensor,
        normal_u: torch.Tensor,
        normal_v: torch.Tensor,
        *,
        epsilon_matrix: torch.Tensor | None = None,
        inverse_epsilon_matrix: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Implement Peng--Zhang Eqs. (8)-(9) with their published sign.

        The paper defines ``Delta = E - [[1/epsilon]]^-1`` and subtracts
        ``Delta N_ij``.  ``correction`` below is therefore ``-Delta``.  The
        returned first four matrices map ``[E_u,E_v]`` to ``[D_u,D_v]``;
        the final two are ``E`` and the inverse-rule matrix, retained for
        diagnostics and regression tests.
        """
        if epsilon_uv.shape != normal_u.shape or epsilon_uv.shape != normal_v.shape:
            raise ValueError("Peng NV epsilon and normal fields must share a grid.")
        if bool(torch.any(torch.abs(epsilon_uv) == 0.0)):
            raise ValueError("Peng NV requires nonzero permittivity samples.")
        epsilon = (
            self._material_conv(epsilon_uv)
            if epsilon_matrix is None
            else epsilon_matrix
        )
        inverse_rule = self._solve(
            self._material_conv(1.0 / epsilon_uv)
            if inverse_epsilon_matrix is None
            else inverse_epsilon_matrix,
            self._eye(self.order_N),
        )
        correction = inverse_rule - epsilon
        n_uu = self._material_conv(normal_u * normal_u)
        n_uv = self._material_conv(normal_u * normal_v)
        n_vv = self._material_conv(normal_v * normal_v)
        epsilon_uu = epsilon + torch.matmul(correction, n_uu)
        epsilon_uv_matrix = torch.matmul(correction, n_uv)
        epsilon_vu_matrix = epsilon_uv_matrix
        epsilon_vv = epsilon + torch.matmul(correction, n_vv)
        return (
            epsilon_uu,
            epsilon_uv_matrix,
            epsilon_vu_matrix,
            epsilon_vv,
            epsilon,
            inverse_rule,
        )

    def _peng_idw_normal_field(
        self,
        mapping: ASRMapping,
        core_radius,
        outer_radius,
        *,
        boundary_samples: int,
        neighbors: int,
        power: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate the paper's boundary-gradient/IDW normal-vector field.

        Boundary normals are analytic Cartesian radial gradients on both
        circles and are placed at their inverse-mapped ``uv`` boundary
        coordinates.  The remaining periodic ``uv`` cell is filled by
        k-nearest inverse-distance weighting, an explicit accelerated-IDW
        realization of the method described after Eq. (10).
        """
        if (
            isinstance(boundary_samples, bool)
            or int(boundary_samples) != boundary_samples
            or boundary_samples < 16
        ):
            raise ValueError("boundary_samples must be an integer >= 16.")
        if (
            isinstance(neighbors, bool)
            or int(neighbors) != neighbors
            or neighbors < 1
        ):
            raise ValueError("neighbors must be a positive integer.")
        if not math.isfinite(float(power)) or float(power) <= 0.0:
            raise ValueError("IDW power must be finite and positive.")
        boundary_samples = int(boundary_samples)
        neighbors = int(neighbors)
        lx, ly = _as_float(self.L[0]), _as_float(self.L[1])
        real_dtype = mapping.x.dtype
        radii = torch.stack(
            tuple(
                torch.as_tensor(value, dtype=real_dtype, device=self._device)
                for value in (core_radius, outer_radius)
            )
        )
        angles = (
            torch.arange(
                boundary_samples, dtype=real_dtype, device=self._device
            )
            + 0.5
        ) * (_TWO_PI / boundary_samples)
        cosine, sine = torch.cos(angles), torch.sin(angles)
        sample_x = 0.5 * lx + (radii[:, None] * cosine[None, :]).reshape(-1)
        sample_y = 0.5 * ly + (radii[:, None] * sine[None, :]).reshape(-1)
        sample_nx = cosine.repeat(2)
        sample_ny = sine.repeat(2)

        def inverse_axis(
            physical: torch.Tensor,
            physical_breaks: torch.Tensor,
            transformed_breaks: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Invert Eq. (7) and return the local derivative at each point."""
            interval = torch.bucketize(
                physical, physical_breaks[1:-1], right=False
            )
            x0 = physical_breaks[interval]
            x1 = physical_breaks[interval + 1]
            u0 = transformed_breaks[interval]
            u1 = transformed_breaks[interval + 1]
            dx, du = x1 - x0, u1 - u0
            a1 = (u1 * x0 - u0 * x1) / du
            a2 = dx / du
            a3 = self.asr_G * du - dx
            coordinate = u0 + (physical - x0) * du / dx
            for _ in range(10):
                phase = _TWO_PI * (coordinate - u0) / du
                mapped = a1 + a2 * coordinate + (a3 / _TWO_PI) * torch.sin(
                    phase
                )
                derivative = a2 + (a3 / du) * torch.cos(phase)
                coordinate = torch.minimum(
                    u1,
                    torch.maximum(
                        u0,
                        coordinate
                        - (mapped - physical) / torch.clamp(derivative, min=1.0e-14),
                    ),
                )
            phase = _TWO_PI * (coordinate - u0) / du
            derivative = a2 + (a3 / du) * torch.cos(phase)
            return coordinate, derivative

        sample_u, sample_f = inverse_axis(
            sample_x, mapping.x_breaks, mapping.u_breaks
        )
        sample_v, sample_g = inverse_axis(
            sample_y, mapping.y_breaks, mapping.v_breaks
        )
        # The boundary locations live in uv after ASR, while the interpolated
        # vector components remain Cartesian.  This permits the published
        # scalar Cartesian NV operator to be formed first and then transformed
        # rigorously with the Piola/Jacobian factors below.
        del sample_f, sample_g
        sample_count = sample_u.numel()
        selected_neighbors = min(neighbors, sample_count)

        uu, vv = torch.meshgrid(mapping.u, mapping.v, indexing="ij")
        flat_u, flat_v = uu.reshape(-1), vv.reshape(-1)
        interpolated_x: list[torch.Tensor] = []
        interpolated_y: list[torch.Tensor] = []
        # Bound the temporary distance matrix independently of the requested
        # Fourier grid.  This matters for the paper-scale 256 x 256 sampling.
        chunk_size = 2048
        distance_floor = (
            min(lx / mapping.x.numel(), ly / mapping.y.numel()) * 1.0e-9
        ) ** 2
        for start in range(0, flat_u.numel(), chunk_size):
            stop = min(start + chunk_size, flat_u.numel())
            du = flat_u[start:stop, None] - sample_u[None, :]
            dv = flat_v[start:stop, None] - sample_v[None, :]
            du = torch.remainder(du + 0.5 * lx, lx) - 0.5 * lx
            dv = torch.remainder(dv + 0.5 * ly, ly) - 0.5 * ly
            squared = du**2 + dv**2
            nearest_squared, indices = torch.topk(
                squared, k=selected_neighbors, dim=1, largest=False
            )
            weights = torch.pow(
                torch.clamp(nearest_squared, min=distance_floor),
                -0.5 * float(power),
            )
            weights = weights / torch.sum(weights, dim=1, keepdim=True)
            interpolated_x.append(
                torch.sum(weights * sample_nx[indices], dim=1)
            )
            interpolated_y.append(
                torch.sum(weights * sample_ny[indices], dim=1)
            )

        normal_x = torch.cat(interpolated_x).reshape(uu.shape)
        normal_y = torch.cat(interpolated_y).reshape(vv.shape)
        norm = torch.sqrt(normal_x**2 + normal_y**2)
        xx, yy = torch.meshgrid(mapping.x, mapping.y, indexing="ij")
        fallback_x = xx - 0.5 * lx
        fallback_y = yy - 0.5 * ly
        fallback_norm = torch.sqrt(fallback_x**2 + fallback_y**2)
        fallback_x = torch.where(
            fallback_norm > 1.0e-14,
            fallback_x / torch.clamp(fallback_norm, min=1.0e-14),
            torch.ones_like(fallback_x),
        )
        fallback_y = torch.where(
            fallback_norm > 1.0e-14,
            fallback_y / torch.clamp(fallback_norm, min=1.0e-14),
            torch.zeros_like(fallback_y),
        )
        normal_x = torch.where(
            norm > 1.0e-12,
            normal_x / torch.clamp(norm, min=1.0e-12),
            fallback_x,
        )
        normal_y = torch.where(
            norm > 1.0e-12,
            normal_y / torch.clamp(norm, min=1.0e-12),
            fallback_y,
        )
        final_norm = torch.sqrt(normal_x**2 + normal_y**2)
        normal_x = normal_x / torch.clamp(final_norm, min=1.0e-14)
        normal_y = normal_y / torch.clamp(final_norm, min=1.0e-14)
        return normal_x, normal_y

    def _build_peng_asr_nv_pq(
        self,
        epsilon_uv: torch.Tensor,
        mu_uv: torch.Tensor,
        mapping: ASRMapping,
        normal_x: torch.Tensor,
        normal_y: torch.Tensor,
        *,
        coordinate_rule: str = "paper-disclosed",
        epsilon_matrix: torch.Tensor | None = None,
        inverse_epsilon_matrix: torch.Tensor | None = None,
        mu_matrix: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply published NV Eqs. (8)-(10) after separable ASR Eq. (7).

        Eq. (8)'s scalar Cartesian NVM operator is built first.  The short
        paper does not publish the combined ASR/NV transformed-tensor equation,
        so the published form and three tensor-ordering sensitivity rules are
        supported explicitly.  ``paper-disclosed`` uses the Eq. (8)-(10)
        matrices directly on the Eq. (7) sampling grid, exactly as documented
        in the article.  ``tensor-metric`` applies the continuous diagonal-Jacobian tensor law
        (diagonal blocks acquire g/f and f/g, while the off-diagonal blocks do
        not); ``asr-correction`` preserves the Eq. (7) ASR factorization and
        adds only the Eq. (8) NV correction; ``metric-left`` is the historical
        row-scaled sensitivity rule; ``piola`` applies both field transforms.
        """
        (
            scalar_uu,
            scalar_uv,
            scalar_vu,
            scalar_vv,
            _epsilon,
            _inverse_rule,
        ) = self._peng_eq8_transverse_epsilon(
            epsilon_uv,
            normal_x,
            normal_y,
            epsilon_matrix=epsilon_matrix,
            inverse_epsilon_matrix=inverse_epsilon_matrix,
        )
        f = mapping.f[:, None].to(self._dtype)
        g = mapping.g[None, :].to(self._dtype)
        ones = torch.ones_like(epsilon_uv)
        normalized_rule = str(coordinate_rule).strip().lower().replace("_", "-")
        if normalized_rule not in {
            "paper-disclosed",
            "tensor-metric",
            "asr-correction",
            "metric-left",
            "piola",
        }:
            raise ValueError(
                "coordinate_rule must be 'paper-disclosed', 'tensor-metric', "
                "'asr-correction', 'metric-left', or 'piola'."
            )
        left_x = self._material_conv(ones * g)
        left_y = self._material_conv(ones * f)
        if normalized_rule == "paper-disclosed":
            eps11_m = scalar_uu
            eps12_m = scalar_uv
            eps21_m = scalar_vu
            eps22_m = scalar_vv
            mu11_m = self._material_conv(mu_uv) if mu_matrix is None else mu_matrix
            mu22_m = mu11_m
            eps33_conv = _epsilon
            mu33_conv = mu11_m
        else:
            eps33_conv = self._material_conv(epsilon_uv * f * g)
            mu33_conv = self._material_conv(mu_uv * f * g)
        if normalized_rule == "paper-disclosed":
            pass
        elif normalized_rule == "piola":
            right_x = self._material_conv(ones / f)
            right_y = self._material_conv(ones / g)
            # D_uv = det(J) J^-1 D_xy and E_xy = J^-T E_uv.
            eps11_m = torch.matmul(left_x, torch.matmul(scalar_uu, right_x))
            eps12_m = torch.matmul(left_x, torch.matmul(scalar_uv, right_y))
            eps21_m = torch.matmul(left_y, torch.matmul(scalar_vu, right_x))
            eps22_m = torch.matmul(left_y, torch.matmul(scalar_vv, right_y))
            mu_scalar = self._material_conv(mu_uv)
            mu11_m = torch.matmul(left_x, torch.matmul(mu_scalar, right_x))
            mu22_m = torch.matmul(left_y, torch.matmul(mu_scalar, right_y))
        elif normalized_rule == "tensor-metric":
            # For J=diag(f,g), det(J) J^-1 eps J^-T has factors
            # g/f, 1, 1, f/g.  The scalar blocks already contain the
            # published inverse-rule/NV convolution, so only the smooth
            # diagonal metric is applied here.
            metric_uu = self._material_conv(ones * g / f)
            metric_vv = self._material_conv(ones * f / g)
            eps11_m = torch.matmul(metric_uu, scalar_uu)
            eps12_m = scalar_uv
            eps21_m = scalar_vu
            eps22_m = torch.matmul(metric_vv, scalar_vv)
            mu11_m = self._material_conv(mu_uv * g / f)
            mu22_m = self._material_conv(mu_uv * f / g)
        elif normalized_rule == "metric-left":
            metric_uu = self._material_conv(ones * g / f)
            metric_vv = self._material_conv(ones * f / g)
            eps11_m = torch.matmul(metric_uu, scalar_uu)
            eps12_m = torch.matmul(metric_uu, scalar_uv)
            eps21_m = torch.matmul(metric_vv, scalar_vu)
            eps22_m = torch.matmul(metric_vv, scalar_vv)
            mu11_m = self._material_conv(mu_uv * g / f)
            mu22_m = self._material_conv(mu_uv * f / g)
        else:
            # Preserve the ASR-only factorization and add precisely the normal
            # correction of Eq. (8).  This avoids multiplying two severely
            # ill-conditioned truncated coordinate matrices when G=0.001.
            epsilon_11 = epsilon_uv * g / f
            epsilon_22 = epsilon_uv * f / g
            eps11_base = self._factorized_bttb(
                1.0 / epsilon_11,
                invert_u_toeplitz=True,
                invert_final_bttb=False,
            )
            eps22_base = self._factorized_bttb(
                epsilon_22,
                invert_u_toeplitz=True,
                invert_final_bttb=True,
            )
            correction_uu = scalar_uu - _epsilon
            correction_uv = scalar_uv
            correction_vu = scalar_vu
            correction_vv = scalar_vv - _epsilon
            metric_uu = self._material_conv(ones * g / f)
            metric_vv = self._material_conv(ones * f / g)
            eps11_m = eps11_base + torch.matmul(metric_uu, correction_uu)
            eps12_m = torch.matmul(metric_uu, correction_uv)
            eps21_m = torch.matmul(metric_vv, correction_vu)
            eps22_m = eps22_base + torch.matmul(metric_vv, correction_vv)
            mu11_m = self._factorized_bttb(
                1.0 / (mu_uv * g / f),
                invert_u_toeplitz=True,
                invert_final_bttb=False,
            )
            mu22_m = self._factorized_bttb(
                mu_uv * f / g,
                invert_u_toeplitz=True,
                invert_final_bttb=True,
            )
        inverse_eps33 = self._solve(eps33_conv, self._eye(self.order_N))
        inverse_mu33 = self._solve(mu33_conv, self._eye(self.order_N))
        ku, kv = self.Kx_norm, self.Ky_norm
        zero = torch.zeros_like(mu11_m)

        p11 = zero + torch.matmul(ku, torch.matmul(inverse_eps33, kv))
        p12 = mu22_m - torch.matmul(ku, torch.matmul(inverse_eps33, ku))
        p21 = torch.matmul(kv, torch.matmul(inverse_eps33, kv)) - mu11_m
        p22 = zero - torch.matmul(kv, torch.matmul(inverse_eps33, ku))
        p = torch.cat(
            (torch.cat((p11, p12), dim=1), torch.cat((p21, p22), dim=1)),
            dim=0,
        )
        q11 = -eps21_m - torch.matmul(ku, torch.matmul(inverse_mu33, kv))
        q12 = torch.matmul(ku, torch.matmul(inverse_mu33, ku)) - eps22_m
        q21 = eps11_m - torch.matmul(kv, torch.matmul(inverse_mu33, kv))
        q22 = eps12_m + torch.matmul(kv, torch.matmul(inverse_mu33, ku))
        q = torch.cat(
            (torch.cat((q11, q12), dim=1), torch.cat((q21, q22), dim=1)),
            dim=0,
        )
        return p, q, eps33_conv, mu33_conv

    def add_layer_circle_asr(
        self,
        thickness,
        radius: float,
        eps_bg,
        eps_cyl,
        *,
        mu_bg=1.0,
        mu_cyl=1.0,
        core_radius=None,
        eps_core=None,
        mu_core=1.0,
        radial_mapping: str = "outer",
        nx: int = 256,
        ny: int = 256,
        factorization_rules: bool = True,
        normal_vector_factorization: bool = False,
    ) -> None:
        """Add a centered circular layer by matched-coordinate ASR-FR.

        This is the rigorous circle-ASR route of Weiss et al.; it is not the
        mathematically unjustified operation of multiplying the Cartesian NVM
        matrix by a one-dimensional ASR matrix after truncation.
        """
        self._require_kvectors()
        thickness_tensor, _ = _real_parameter_tensor(
            "thickness",
            thickness,
            dtype=self._dtype,
            device=self._device,
            allow_zero=True,
        )
        core_shell = core_radius is not None or eps_core is not None
        mapping_aliases = {
            "outer": "outer",
            "single": "outer",
            "double": "double",
            "dual": "double",
            "both": "double",
            "double-matched": "double",
        }
        normalized_mapping = mapping_aliases.get(
            str(radial_mapping).strip().lower().replace("_", "-")
        )
        if normalized_mapping is None:
            raise ValueError(
                "radial_mapping must be 'outer' or 'double' "
                "('dual', 'both', and 'double-matched' are aliases)."
            )
        if not core_shell and normalized_mapping != "outer":
            raise ValueError("radial_mapping='double' requires a core-shell circle.")
        requested_factorization_rules = bool(factorization_rules)
        if normal_vector_factorization and not requested_factorization_rules:
            raise ValueError(
                "normal_vector_factorization=True requires "
                "factorization_rules=True."
            )
        if (core_radius is None) != (eps_core is None):
            raise ValueError(
                "core_radius and eps_core must be supplied together for a "
                "core-shell circle."
            )
        raw_materials = [eps_bg, eps_cyl, mu_bg, mu_cyl]
        if core_shell:
            raw_materials.extend((eps_core, mu_core))
        material_values = [
            torch.as_tensor(value, dtype=self._dtype, device=self._device)
            for value in raw_materials
        ]
        if not all(bool(torch.all(torch.isfinite(v))) for v in material_values):
            raise ValueError("matched-ASR materials must be finite.")
        if any(bool(torch.any(torch.abs(v) == 0.0)) for v in material_values):
            raise ValueError("matched-ASR epsilon and mu values must be nonzero.")
        eps_bg_t, eps_cyl_t, mu_bg_t, mu_cyl_t = material_values[:4]
        eps_core_t = mu_core_t = None
        if core_shell:
            eps_core_t, mu_core_t = material_values[4:]
        magnetic_material_jump = not torch.equal(mu_bg_t, mu_cyl_t)
        if core_shell:
            magnetic_material_jump = magnetic_material_jump or not torch.equal(
                mu_core_t, mu_cyl_t
            )
        radius, radius_value = _real_parameter_tensor(
            "radius",
            radius,
            dtype=torch.float64,
            device=self._device,
            allow_zero=False,
        )
        core_radius_value = None
        if core_shell:
            if (
                normalized_mapping == "outer"
                and torch.is_tensor(core_radius)
                and core_radius.requires_grad
            ):
                raise UnsupportedCombinationError(
                    "The inner core/shell boundary is sampled rather than "
                    "coordinate-matched, so core_radius does not have a "
                    "reliable shape derivative. Keep core_radius fixed or use "
                    "a separately matched single-circle parameterization."
                )
            core_radius, core_radius_value = _real_parameter_tensor(
                "core_radius",
                core_radius,
                dtype=torch.float64,
                device=self._device,
                allow_zero=False,
            )
            if core_radius_value >= radius_value:
                raise ValueError(
                    "core_radius must be smaller than the outer matched radius."
                )
        triangular = getattr(self, "lattice_kind", "rectangular") == "triangular"
        if getattr(self, "group_theory_symmetry", "auto") == "d6" and not triangular:
            raise UnsupportedCombinationError(
                "symmetry='d6' requires a 60-degree equal-length triangular cell."
            )
        if core_shell and normalized_mapping == "double":
            mapping = self.build_double_matched_circle_asr_mapping(
                nx, ny, core_radius, radius
            )
        elif triangular:
            mapping = self.build_triangular_circle_asr_mapping(nx, ny, radius)
        else:
            if hasattr(self, "cos_zeta") and abs(_as_float(self.cos_zeta)) > 1.0e-8:
                raise UnsupportedCombinationError(
                    "matched circle ASR supports orthogonal or 60-degree triangular cells."
                )
            mapping = self.build_circle_asr_mapping(nx, ny, radius)
        lx, ly = _as_float(self.L[0]), _as_float(self.L[1])
        if core_shell and normalized_mapping == "double":
            if mapping.matched_outer_mask is None or mapping.matched_core_mask is None:
                raise RuntimeError("Double-matched map did not provide fixed material masks.")
            inside = mapping.matched_outer_mask
        else:
            inside = self._periodic_circle_mask(mapping.x, mapping.y, radius)
        eps_uv = torch.where(inside, eps_cyl_t, eps_bg_t)
        mu_uv = torch.where(inside, mu_cyl_t, mu_bg_t)
        if core_shell:
            core_inside = (
                mapping.matched_core_mask
                if normalized_mapping == "double"
                else self._periodic_circle_mask(mapping.x, mapping.y, core_radius)
            )
            if core_inside is None:
                raise RuntimeError("Core-shell map did not provide a core mask.")
            eps_uv = torch.where(core_inside, eps_core_t, eps_uv)
            mu_uv = torch.where(core_inside, mu_core_t, mu_uv)

        h = mapping.det_j.to(self._dtype)
        x_u, x_v = mapping.x_u.to(self._dtype), mapping.x_v.to(self._dtype)
        y_u, y_v = mapping.y_u.to(self._dtype), mapping.y_v.to(self._dtype)
        metric11 = (x_v**2 + y_v**2) / h
        metric12 = -(x_u * x_v + y_u * y_v) / h
        metric22 = (x_u**2 + y_u**2) / h
        eps11, eps12, eps21, eps22, eps33 = (
            eps_uv * metric11,
            eps_uv * metric12,
            eps_uv * metric12,
            eps_uv * metric22,
            eps_uv * h,
        )
        mu11, mu12, mu21, mu22, mu33 = (
            mu_uv * metric11,
            mu_uv * metric12,
            mu_uv * metric12,
            mu_uv * metric22,
            mu_uv * h,
        )
        factorization_normals = None
        if (
            normal_vector_factorization
            and core_shell
            and normalized_mapping == "double"
            and factorization_rules
        ):
            if (
                mapping.interface_normal_u is None
                or mapping.interface_normal_v is None
            ):
                raise RuntimeError(
                    "Double-matched map did not provide interface normals for Li factorization."
                )
            factorization_normals = (
                mapping.interface_normal_u,
                mapping.interface_normal_v,
            )
        elif normal_vector_factorization and factorization_rules:
            # The outer-only map aligns only the outer boundary.  Pulling the
            # physical radial normal back through the full 2-D Jacobian gives
            # the correct normal for both the matched outer circle and the
            # sampled concentric inner circle.  This is the ASR-then-NV route;
            # it is not a post-truncation multiplication by a Cartesian NVM.
            factorization_normals = self._pulled_back_circle_normal(
                mapping, core_radius if core_shell else radius
            )
        p, q, eps33_conv, mu33_conv = self._build_circle_asr_pq(
            eps11,
            eps12,
            eps21,
            eps22,
            eps33,
            mu11,
            mu12,
            mu21,
            mu22,
            mu33,
            factorization_rules=factorization_rules,
            normal_factorize_mu=magnetic_material_jump,
            factorization_normals=factorization_normals,
        )
        transform, transform_z_all = self._build_circle_conversion_matrices(mapping)
        layer_index = self.layer_N
        complete_d6 = (
            bool(getattr(self, "use_group_theory", False))
            and getattr(self, "group_theory_symmetry", "auto") == "d6"
            and getattr(self, "polarization_reduction", None) is None
        )
        d6_source = (
            bool(getattr(self, "use_group_theory", False))
            and getattr(self, "group_theory_symmetry", "auto") == "d6"
            and getattr(self, "polarization_reduction", None) is not None
        )
        if complete_d6:
            triangular_star_pq = self._build_triangular_star_pq(
                eps11,
                eps12,
                eps21,
                eps22,
                eps33,
                mu11,
                mu12,
                mu21,
                mu22,
                mu33,
                factorization_rules=factorization_rules,
                normal_factorize_mu=magnetic_material_jump,
                factorization_normals=factorization_normals,
            )
            vector_embedding, _, _, _, _ = self._triangular_star_operators()
            transform_star = vector_embedding.mH @ transform @ vector_embedding
            (
                kz,
                w_cartesian_star,
                v_cartesian_star,
                vector_embedding,
                transform_inverse_star,
            ) = self._complete_d6_eigendecomposition(
                *triangular_star_pq,
                transform_star,
                layer_index=layer_index,
                backend="matched-ASR",
            )
            w_uv = vector_embedding @ (
                transform_inverse_star @ w_cartesian_star
            )
            v_uv = vector_embedding @ (
                transform_inverse_star @ v_cartesian_star
            )
            w_cartesian = vector_embedding @ w_cartesian_star
            v_cartesian = vector_embedding @ v_cartesian_star
            self._register_complete_d6_layer(
                vector_embedding,
                w_cartesian_star,
                v_cartesian_star,
                kz,
            )
        elif d6_source:
            triangular_star_pq = self._build_triangular_star_pq(
                eps11,
                eps12,
                eps21,
                eps22,
                eps33,
                mu11,
                mu12,
                mu21,
                mu22,
                mu33,
                factorization_rules=factorization_rules,
                normal_factorize_mu=magnetic_material_jump,
                factorization_normals=factorization_normals,
            )
            vector_embedding, _, _, _, _ = self._triangular_star_operators()
            transform_star = vector_embedding.mH @ transform @ vector_embedding
            (
                kz,
                w_cartesian_star,
                v_cartesian_star,
                vector_embedding,
                transform_inverse_star,
            ) = self._d6_source_eigendecomposition(
                *triangular_star_pq,
                transform_star,
                layer_index=layer_index,
                backend="matched-ASR",
            )
            w_uv = vector_embedding @ (
                transform_inverse_star @ w_cartesian_star
            )
            v_uv = vector_embedding @ (
                transform_inverse_star @ v_cartesian_star
            )
            w_cartesian = vector_embedding @ w_cartesian_star
            v_cartesian = vector_embedding @ v_cartesian_star
        elif getattr(self, "polarization_reduction", None) is not None:
            triangular_star_pq = (
                self._build_triangular_star_pq(
                    eps11,
                    eps12,
                    eps21,
                    eps22,
                    eps33,
                    mu11,
                    mu12,
                    mu21,
                    mu22,
                    mu33,
                    factorization_rules=factorization_rules,
                    normal_factorize_mu=magnetic_material_jump,
                    factorization_normals=factorization_normals,
                )
                if triangular
                else None
            )
            kz, w_uv, v_uv, w_cartesian, v_cartesian = (
                self._matched_polarization_eigendecomposition(
                    p,
                    q,
                    transform,
                    layer_index=layer_index,
                    triangular_star_pq=triangular_star_pq,
                )
            )
        else:
            kz_squared, w_uv = self._eig(torch.matmul(p, q))
            kz = self._positive_kz(kz_squared)
            v_uv = self._magnetic_eigenvectors(p, q, w_uv, kz)
            w_cartesian = torch.matmul(transform, w_uv)
            v_cartesian = torch.matmul(transform, v_uv)
        transform_z = transform_z_all if self.store_mode_couplings else None

        self.layer_N += 1
        self.thickness.append(thickness_tensor)
        self.eps_conv.append(eps33_conv)
        self.mu_conv.append(mu33_conv)
        self.P.append(p)
        self.Q.append(q)
        self.kz_norm.append(kz)
        self.E_eigvec_uv.append(w_uv)
        self.H_eigvec_uv.append(v_uv)
        self.E_eigvec.append(w_cartesian)
        self.H_eigvec.append(v_cartesian)
        self.asr_mappings.append(mapping)
        self.asr_T_matrices.append(transform)
        self.asr_Tz_matrices.append(transform_z)
        self.asr_condition_numbers.append(
            torch.linalg.cond(transform.to(torch.complex128))
            if self.compute_condition_numbers
            else None
        )
        self.asr_material_tensors.append(
            {
                "eps_uv": eps_uv,
                "eps11": eps11,
                "eps12": eps12,
                "eps21": eps21,
                "eps22": eps22,
                "eps33": eps33,
                "mu_uv": mu_uv,
                "mu11": mu11,
                "mu12": mu12,
                "mu21": mu21,
                "mu22": mu22,
                "mu33": mu33,
            }
        )
        slot = len(self.asr_mappings) - 1
        self._asr_slot_by_layer[layer_index] = slot
        if transform_z is not None:
            field_context = {
                "electric_modes_uv": w_uv,
                "magnetic_modes_uv": v_uv,
                "electric_modes_cartesian": w_cartesian,
                "magnetic_modes_cartesian": v_cartesian,
                "transform_xy": transform,
                "transform_z": transform_z,
                "eps33_conv": eps33_conv,
                "mu33_conv": mu33_conv,
            }
            if triangular and (
                complete_d6
                or getattr(self, "polarization_reduction", None) is not None
            ):
                vector_embedding, _, _, _, _ = self._triangular_star_operators()
                star_count = vector_embedding.shape[1] // 2
                scalar_embedding = vector_embedding[: self.order_N, :star_count]
                field_context.update(
                    {
                        "scalar_embedding": scalar_embedding,
                        "eps33_conv_reduced": (
                            scalar_embedding.mH @ eps33_conv @ scalar_embedding
                        ),
                        "mu33_conv_reduced": (
                            scalar_embedding.mH @ mu33_conv @ scalar_embedding
                        ),
                        "transform_z_reduced": (
                            scalar_embedding.mH @ transform_z @ scalar_embedding
                        ),
                    }
                )
            self._asr_field_context_by_layer[layer_index] = field_context

        x_uniform_axis = (
            torch.arange(nx, dtype=torch.float64, device=self._device) * lx / nx
        )
        y_uniform_axis = (
            torch.arange(ny, dtype=torch.float64, device=self._device) * ly / ny
        )
        coordinate_1, coordinate_2 = torch.meshgrid(
            x_uniform_axis, y_uniform_axis, indexing="ij"
        )
        cosine = float(getattr(self, "cos_zeta", 0.0))
        sine = float(getattr(self, "sin_zeta", 1.0))
        physical_x = coordinate_1 + cosine * coordinate_2
        physical_y = sine * coordinate_2
        physical_inside = self._periodic_circle_mask(
            physical_x, physical_y, radius
        )
        physical_eps = torch.where(physical_inside, eps_cyl_t, eps_bg_t)
        physical_mu = torch.where(physical_inside, mu_cyl_t, mu_bg_t)
        if core_shell:
            physical_core = self._periodic_circle_mask(
                physical_x, physical_y, core_radius
            )
            physical_eps = torch.where(physical_core, eps_core_t, physical_eps)
            physical_mu = torch.where(physical_core, mu_core_t, physical_mu)
        self._physical_material_by_layer[layer_index] = (
            physical_eps,
            physical_mu,
        )
        self.layer_records.append(
            LayerRecord(
                index=layer_index,
                method="matched-asr",
                shape="core-shell-circle" if core_shell else "circle",
                lattice=getattr(self, "lattice_kind", "rectangular"),
                reason=(
                    "DOUBLE_MATCHED_CORE_SHELL_COORDINATES"
                    if core_shell and normalized_mapping == "double"
                    else "OUTER_MATCHED_CORE_SHELL_COORDINATES"
                    if core_shell
                    else "MATCHED_CIRCLE_COORDINATES"
                ),
                options={
                    "radius": radius_value,
                    "core_radius": core_radius_value,
                    "grid": (nx, ny),
                    "asr_G": self.matched_asr_G,
                    "factorization_rules_requested": requested_factorization_rules,
                    "factorization_rules": factorization_rules,
                    "normal_vector_factorization_requested": bool(
                        normal_vector_factorization
                    ),
                    "magnetic_normal_factorization": bool(
                        factorization_normals is not None
                        and magnetic_material_jump
                    ),
                    "factorization_scheme": (
                        "generalized-li-epsilon+generalized-li-mu"
                        if factorization_normals is not None
                        and magnetic_material_jump
                        else "generalized-li-epsilon+weiss-symmetric-mu"
                        if factorization_normals is not None
                        else "weiss-symmetric-29-36"
                        if factorization_rules
                        else "direct-laurent"
                    ),
                    "radial_mapping": normalized_mapping,
                    "matched_boundaries": (
                        "inner+outer"
                        if core_shell and normalized_mapping == "double"
                        else "outer"
                    ),
                    "double_match_coordinate_fractions": (
                        (1.0 / 3.0, 2.0 / 3.0)
                        if core_shell and normalized_mapping == "double"
                        else None
                    ),
                    "radial_monotonicity_guaranteed": bool(
                        core_shell and normalized_mapping == "double"
                    ),
                    "effective_radial_slope": (
                        _as_float(mapping.effective_radial_slope)
                        if mapping.effective_radial_slope is not None
                        else None
                    ),
                    "central_radial_slope": (
                        _as_float(mapping.central_radial_slope)
                        if mapping.central_radial_slope is not None
                        else None
                    ),
                    "boundary_radial_slope": (
                        _as_float(mapping.boundary_radial_slope)
                        if mapping.boundary_radial_slope is not None
                        else None
                    ),
                    "minimum_radial_secant": (
                        _as_float(mapping.minimum_radial_secant)
                        if mapping.minimum_radial_secant is not None
                        else None
                    ),
                    "minimum_mapping_jacobian": (
                        _as_float(torch.min(mapping.det_j))
                    ),
                    "map": (
                        "monotone C2 double-matched radial quintic Hermite map"
                        if core_shell and normalized_mapping == "double"
                        else "D6 hex-to-circle periodic Hermite map"
                        if triangular
                        else "Weiss-37-38 + separable ASR"
                    ),
                    "conversion": "general-2d-T",
                    "polarization": getattr(self, "polarization_reduction", None),
                    "group_theory": (
                        dict(self.group_theory_diagnostics[-1])
                        if complete_d6 or d6_source
                        else None
                    ),
                },
            )
        )
        if (
            getattr(self, "polarization_reduction", None) is None
            and not complete_d6
        ):
            self._append_smatrix_from_cartesian_modes(w_cartesian, v_cartesian)

    def add_layer_circle_shell_asr(
        self,
        thickness,
        core_radius,
        outer_radius,
        eps_bg,
        eps_shell,
        eps_core,
        *,
        mu_bg=1.0,
        mu_shell=1.0,
        mu_core=1.0,
        radial_mapping: str = "outer",
        nx: int = 256,
        ny: int = 256,
        factorization_rules: bool = True,
        normal_vector_factorization: bool = False,
    ) -> None:
        """Add a concentric core-shell circle with selectable radial matching.

        ``radial_mapping='outer'`` retains the legacy outer-boundary map.  The
        core/shell boundary is sampled and a trainable ``core_radius`` is
        rejected.  ``radial_mapping='double'`` uses a C2 radial map whose two
        computational support curves map exactly to the core/shell and
        shell/background circles.  Both radii then remain differentiable
        through the map and its Jacobian.  With ``factorization_rules=True``,
        the double map uses generalized Li normal-D/tangential-E tensor
        factorization; it does not multiply by a Cartesian NVM projector.
        With the outer-only map, ``normal_vector_factorization=True`` pulls
        the physical radial normal back through the matched map and applies
        the same generalized Li factorization to both concentric interfaces.
        Fourier-order and grid convergence are required for either choice.
        """
        self.add_layer_circle_asr(
            thickness,
            outer_radius,
            eps_bg,
            eps_shell,
            mu_bg=mu_bg,
            mu_cyl=mu_shell,
            core_radius=core_radius,
            eps_core=eps_core,
            mu_core=mu_core,
            radial_mapping=radial_mapping,
            nx=nx,
            ny=ny,
            factorization_rules=factorization_rules,
            normal_vector_factorization=normal_vector_factorization,
        )

    def add_layer_circle_shell_peng_asr(
        self,
        thickness,
        core_radius,
        outer_radius,
        eps_bg,
        eps_shell,
        eps_core,
        *,
        mu_bg=1.0,
        mu_shell=1.0,
        mu_core=1.0,
        nx: int = 256,
        ny: int = 256,
        staircase_grid: int = 64,
        uv_interval_allocation: str = "cube-root",
        normal_vector_factorization: bool = False,
        nv_boundary_samples: int = 256,
        nv_neighbors: int = 16,
        nv_power: float = 2.0,
        nv_coordinate_rule: str = "paper-disclosed",
    ) -> None:
        """Add Peng--Zhang stepped ASR or ASR-NV concentric-circle layer.

        This is the dedicated reproduction route for Eq. (7) and, when
        requested, Eqs. (8)-(10).  It must not be confused with the smooth
        Weiss matched-circle implementation exposed by
        :meth:`add_layer_circle_shell_asr`.

        The article does not disclose the staircase resolution, the placement
        of transformed jump points, or the accelerated-IDW parameters.  All
        three are therefore explicit arguments and are copied to layer
        metadata so that a run is reproducible without presenting those
        choices as values reported by the authors.
        """
        self._require_kvectors()
        if getattr(self, "lattice_kind", "rectangular") not in {
            "rectangular",
            "square",
        }:
            raise UnsupportedCombinationError(
                "Peng stepped ASR is implemented for the orthogonal Fig. 2 cell."
            )
        thickness_tensor, _ = _real_parameter_tensor(
            "thickness",
            thickness,
            dtype=self._dtype,
            device=self._device,
            allow_zero=True,
        )
        core, core_value = _real_parameter_tensor(
            "core_radius",
            core_radius,
            dtype=torch.float64,
            device=self._device,
            allow_zero=False,
        )
        outer, outer_value = _real_parameter_tensor(
            "outer_radius",
            outer_radius,
            dtype=torch.float64,
            device=self._device,
            allow_zero=False,
        )
        if core_value >= outer_value:
            raise ValueError("core_radius must be smaller than outer_radius.")
        material_values = tuple(
            torch.as_tensor(value, dtype=self._dtype, device=self._device)
            for value in (eps_bg, eps_shell, eps_core, mu_bg, mu_shell, mu_core)
        )
        if not all(bool(torch.all(torch.isfinite(value))) for value in material_values):
            raise ValueError("Peng ASR material values must be finite.")
        if any(bool(torch.any(torch.abs(value) == 0.0)) for value in material_values):
            raise ValueError("Peng ASR material values must be nonzero.")
        eps_bg_t, eps_shell_t, eps_core_t, mu_bg_t, mu_shell_t, mu_core_t = (
            material_values
        )
        mapping = self.build_stepped_circle_asr_mapping(
            nx,
            ny,
            core,
            outer,
            staircase_grid=staircase_grid,
            interval_allocation=uv_interval_allocation,
        )
        if mapping.region_grid is None or mapping.staircase_grid is None:
            raise RuntimeError("Stepped circle map did not retain its region raster.")
        lx, ly = _as_float(self.L[0]), _as_float(self.L[1])

        def sampled_regions(x_values: torch.Tensor, y_values: torch.Tensor) -> torch.Tensor:
            x_index = torch.floor(
                torch.remainder(x_values, lx) * mapping.staircase_grid / lx
            ).to(torch.int64)
            y_index = torch.floor(
                torch.remainder(y_values, ly) * mapping.staircase_grid / ly
            ).to(torch.int64)
            x_index = torch.clamp(x_index, 0, mapping.staircase_grid - 1)
            y_index = torch.clamp(y_index, 0, mapping.staircase_grid - 1)
            return mapping.region_grid[x_index[:, None], y_index[None, :]]

        region_uv = sampled_regions(mapping.x, mapping.y)
        epsilon_uv = torch.where(
            region_uv == 1,
            eps_shell_t,
            torch.where(region_uv == 2, eps_core_t, eps_bg_t),
        )
        mu_uv = torch.where(
            region_uv == 1,
            mu_shell_t,
            torch.where(region_uv == 2, mu_core_t, mu_bg_t),
        )
        physical_x_mid = 0.5 * (mapping.x_breaks[:-1] + mapping.x_breaks[1:])
        physical_y_mid = 0.5 * (mapping.y_breaks[:-1] + mapping.y_breaks[1:])
        rectangular_regions = sampled_regions(physical_x_mid, physical_y_mid)
        rectangular_epsilon = torch.where(
            rectangular_regions == 1,
            eps_shell_t,
            torch.where(rectangular_regions == 2, eps_core_t, eps_bg_t),
        )
        rectangular_mu = torch.where(
            rectangular_regions == 1,
            mu_shell_t,
            torch.where(rectangular_regions == 2, mu_core_t, mu_bg_t),
        )
        exact_epsilon_conv = self._piecewise_rectangular_conv(
            rectangular_epsilon, mapping.u_breaks, mapping.v_breaks
        )
        exact_inverse_epsilon_conv = self._piecewise_rectangular_conv(
            1.0 / rectangular_epsilon, mapping.u_breaks, mapping.v_breaks
        )
        exact_mu_conv = self._piecewise_rectangular_conv(
            rectangular_mu, mapping.u_breaks, mapping.v_breaks
        )
        f = mapping.f[:, None].to(self._dtype)
        g = mapping.g[None, :].to(self._dtype)
        eps11 = epsilon_uv * g / f
        eps22 = epsilon_uv * f / g
        eps33 = epsilon_uv * f * g
        mu11 = mu_uv * g / f
        mu22 = mu_uv * f / g
        mu33 = mu_uv * f * g

        normal_x = normal_y = None
        normalized_nv_rule = (
            str(nv_coordinate_rule).strip().lower().replace("_", "-")
            if normal_vector_factorization
            else None
        )
        if normal_vector_factorization:
            normal_x, normal_y = self._peng_idw_normal_field(
                mapping,
                core,
                outer,
                boundary_samples=nv_boundary_samples,
                neighbors=nv_neighbors,
                power=nv_power,
            )
            p, q, eps33_conv, mu33_conv = self._build_peng_asr_nv_pq(
                epsilon_uv,
                mu_uv,
                mapping,
                normal_x,
                normal_y,
                coordinate_rule=normalized_nv_rule,
                epsilon_matrix=exact_epsilon_conv,
                inverse_epsilon_matrix=exact_inverse_epsilon_conv,
                mu_matrix=exact_mu_conv,
            )
        else:
            p, q, eps33_conv, mu33_conv = self._build_asr_pq(
                epsilon_uv,
                epsilon_uv,
                epsilon_uv,
                mu_uv,
                mu_uv,
                mu_uv,
                factorization_rules=False,
                direct_convolutions={
                    "eps11": exact_epsilon_conv,
                    "eps22": exact_epsilon_conv,
                    "eps33": exact_epsilon_conv,
                    "mu11": exact_mu_conv,
                    "mu22": exact_mu_conv,
                    "mu33": exact_mu_conv,
                },
            )

        kz_squared, electric_uv = self._eig(torch.matmul(p, q))
        kz = self._positive_kz(kz_squared)
        magnetic_uv = self._magnetic_eigenvectors(p, q, electric_uv, kz)
        # The article writes the Fourier amplitudes directly in the adaptive
        # coordinates and does not publish an interface/Jacobian conversion.
        # Its disclosed discretization therefore uses identity matching.  The
        # alternative tensor rules retain the full coordinate conversion.
        disclosed_coordinate_form = (
            not normal_vector_factorization
            or normalized_nv_rule == "paper-disclosed"
        )
        if disclosed_coordinate_form:
            transform = self._eye(2 * self.order_N)
            transform_z = (
                self._eye(self.order_N) if self.store_mode_couplings else None
            )
            self._last_asr_transform_condition = 1.0
        else:
            transform = self._build_conversion_matrix_T(mapping)
            transform_z = (
                self._build_conversion_matrix_Tz(mapping)
                if self.store_mode_couplings
                else None
            )
        electric_cartesian = torch.matmul(transform, electric_uv)
        magnetic_cartesian = torch.matmul(transform, magnetic_uv)
        layer_index = self.layer_N

        self.layer_N += 1
        self.thickness.append(thickness_tensor)
        self.eps_conv.append(eps33_conv)
        self.mu_conv.append(mu33_conv)
        self.P.append(p)
        self.Q.append(q)
        self.kz_norm.append(kz)
        self.E_eigvec_uv.append(electric_uv)
        self.H_eigvec_uv.append(magnetic_uv)
        self.E_eigvec.append(electric_cartesian)
        self.H_eigvec.append(magnetic_cartesian)
        self.asr_mappings.append(mapping)
        self.asr_T_matrices.append(transform)
        self.asr_Tz_matrices.append(transform_z)
        self.asr_condition_numbers.append(
            self._last_asr_transform_condition
            if self.compute_condition_numbers
            else None
        )
        material_record = {
            "eps_uv": epsilon_uv,
            "eps11": eps11,
            "eps22": eps22,
            "eps33": eps33,
            "mu_uv": mu_uv,
            "mu11": mu11,
            "mu22": mu22,
            "mu33": mu33,
        }
        if normal_x is not None and normal_y is not None:
            material_record["normal_x"] = normal_x
            material_record["normal_y"] = normal_y
        self.asr_material_tensors.append(material_record)
        slot = len(self.asr_mappings) - 1
        self._asr_slot_by_layer[layer_index] = slot
        if transform_z is not None:
            self._asr_field_context_by_layer[layer_index] = {
                "electric_modes_uv": electric_uv,
                "magnetic_modes_uv": magnetic_uv,
                "electric_modes_cartesian": electric_cartesian,
                "magnetic_modes_cartesian": magnetic_cartesian,
                "transform_xy": transform,
                "transform_z": transform_z,
                "eps33_conv": eps33_conv,
                "mu33_conv": mu33_conv,
            }

        physical_x = (
            torch.arange(nx, dtype=torch.float64, device=self._device) * lx / nx
        )
        physical_y = (
            torch.arange(ny, dtype=torch.float64, device=self._device) * ly / ny
        )
        physical_regions = sampled_regions(physical_x, physical_y)
        physical_epsilon = torch.where(
            physical_regions == 1,
            eps_shell_t,
            torch.where(physical_regions == 2, eps_core_t, eps_bg_t),
        )
        physical_mu = torch.where(
            physical_regions == 1,
            mu_shell_t,
            torch.where(physical_regions == 2, mu_core_t, mu_bg_t),
        )
        self._physical_material_by_layer[layer_index] = (
            physical_epsilon,
            physical_mu,
        )
        minimum_jacobian = float(
            (torch.min(mapping.f) * torch.min(mapping.g)).detach().cpu()
        )
        factorization_scheme = (
            f"peng-eq7-asr+eq8-10-nv-idw-{normalized_nv_rule}"
            if normal_vector_factorization
            else "peng-eq7-asr-exact-rectangular-laurent"
        )
        self.layer_records.append(
            LayerRecord(
                index=layer_index,
                method="peng-asr-nv" if normal_vector_factorization else "peng-asr",
                shape="stepped-core-shell-circle",
                lattice=getattr(self, "lattice_kind", "rectangular"),
                reason="PENG_ZHANG_2025_EQ7_EQ8_10_REPRODUCTION",
                options={
                    "core_radius": core_value,
                    "outer_radius": outer_value,
                    "grid": (nx, ny),
                    "staircase_grid": mapping.staircase_grid,
                    "x_jump_count": int(mapping.x_breaks.numel() - 2),
                    "y_jump_count": int(mapping.y_breaks.numel() - 2),
                    "asr_G": self.asr_G,
                    "asr_equation": "Peng-Zhang Eq. (7)",
                    "uv_break_allocation": mapping.interval_allocation,
                    "material_fourier_integration": "exact-piecewise-rectangular",
                    "normal_vector_factorization": normal_vector_factorization,
                    "normal_field": (
                        "analytic-boundary-gradient + periodic-kNN-IDW"
                        if normal_vector_factorization
                        else None
                    ),
                    "nv_boundary_samples_per_circle": (
                        int(nv_boundary_samples)
                        if normal_vector_factorization
                        else None
                    ),
                    "nv_neighbors": (
                        int(nv_neighbors) if normal_vector_factorization else None
                    ),
                    "nv_power": (
                        float(nv_power) if normal_vector_factorization else None
                    ),
                    "nv_equations": (
                        "Peng-Zhang Eqs. (8)-(10) followed by the recorded ASR coordinate rule"
                        if normal_vector_factorization
                        else None
                    ),
                    "nv_coordinate_rule": normalized_nv_rule,
                    "factorization_scheme": factorization_scheme,
                    "minimum_mapping_jacobian": minimum_jacobian,
                    "unstated_reproduction_assumptions": [
                        f"{mapping.interval_allocation} u_l/v_l interval allocation",
                        "user-selected staircase resolution",
                    ]
                    + (
                        [
                            "periodic k-nearest accelerated IDW",
                            f"{normalized_nv_rule} ordering combines Cartesian NV with ASR coordinates",
                        ]
                        if normal_vector_factorization
                        else []
                    ),
                },
            )
        )
        self._append_smatrix_from_cartesian_modes(
            electric_cartesian, magnetic_cartesian
        )

    def _factorized_bttb(
        self,
        material_uv: torch.Tensor,
        *,
        invert_u_toeplitz: bool,
        invert_final_bttb: bool,
    ) -> torch.Tensor:
        """
        Build the mixed u/v factorization used in Eqs. (13)-(15).

        For every v sample a u-Toeplitz matrix is formed and optionally
        inverted.  Its entries are then Fourier transformed in v and assembled
        into a BTTB matrix.  The final BTTB matrix is optionally inverted.
        """
        material_uv = material_uv.to(self._dtype)
        nu, nv = material_uv.shape
        mx, my = len(self.order_x), len(self.order_y)
        p_delta = self.order_x[:, None] - self.order_x[None, :]
        q_delta = self.order_y[:, None] - self.order_y[None, :]

        coefficients_u = torch.fft.fft(material_uv, dim=0) / nu
        # Shape: [v sample, p, p'].
        u_toeplitz = coefficients_u[p_delta, :].permute(2, 0, 1)
        if invert_u_toeplitz:
            identity_u = torch.eye(
                mx, dtype=torch.complex128, device=self._device
            ).expand(nv, mx, mx)
            u_toeplitz = torch.linalg.solve(
                u_toeplitz.to(torch.complex128), identity_u
            ).to(self._dtype)

        coefficients_v = torch.fft.fft(u_toeplitz, dim=0) / nv
        result = torch.zeros(
            (mx * my, mx * my), dtype=self._dtype, device=self._device
        )
        for p in range(mx):
            rows = slice(p * my, (p + 1) * my)
            for pp in range(mx):
                cols = slice(pp * my, (pp + 1) * my)
                result[rows, cols] = coefficients_v[q_delta, p, pp]

        if invert_final_bttb:
            result = self._solve(result, self._eye(mx * my))
        return result

    def _axis_toeplitz_1d(
        self, samples: torch.Tensor, order: torch.Tensor
    ) -> torch.Tensor:
        """Toeplitz matrix of a periodic 1-D function sampled uniformly."""
        values = samples.to(self._dtype)
        coefficients = torch.fft.fft(values) / values.numel()
        delta = order[:, None] - order[None, :]
        return coefficients[delta]

    def _rect_separable_convolutions(
        self,
        mapping: ASRMapping,
        eps_bg: torch.Tensor,
        eps_rect: torch.Tensor,
        mu_bg: torch.Tensor,
        mu_rect: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """High-accuracy direct-rule tensors without a huge 2-D raster.

        A centered transformed rectangle is separable.  For example,
        ``eps11=(eps_bg+Delta*Ix*Iy)*(1/f)*g`` is a sum of two outer
        products, so its BTTB matrix is the corresponding sum of Kronecker
        products of 1-D Toeplitz matrices.  This permits thousands of
        quadrature samples per axis for the ill-conditioned N=20 ASR reference
        while keeping memory independent of the square of that sample count.
        """
        inside_x = (
            (mapping.x >= mapping.x_breaks[1])
            & (mapping.x < mapping.x_breaks[2])
        ).to(self._dtype)
        inside_y = (
            (mapping.y >= mapping.y_breaks[1])
            & (mapping.y < mapping.y_breaks[2])
        ).to(self._dtype)
        f = mapping.f.to(self._dtype)
        g = mapping.g.to(self._dtype)

        def separable(x_values: torch.Tensor, y_values: torch.Tensor) -> torch.Tensor:
            return torch.kron(
                self._axis_toeplitz_1d(x_values, self.order_x),
                self._axis_toeplitz_1d(y_values, self.order_y),
            )

        def tensor_set(
            background: torch.Tensor, inclusion: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            contrast = inclusion - background
            component_11 = (
                background * separable(1.0 / f, g)
                + contrast * separable(inside_x / f, inside_y * g)
            )
            component_22 = (
                background * separable(f, 1.0 / g)
                + contrast * separable(inside_x * f, inside_y / g)
            )
            component_33 = (
                background * separable(f, g)
                + contrast * separable(inside_x * f, inside_y * g)
            )
            return component_11, component_22, component_33

        eps11, eps22, eps33 = tensor_set(eps_bg, eps_rect)
        mu11, mu22, mu33 = tensor_set(mu_bg, mu_rect)
        return {
            "eps11": eps11,
            "eps22": eps22,
            "eps33": eps33,
            "mu11": mu11,
            "mu22": mu22,
            "mu33": mu33,
        }

    def _centered_interval_toeplitz(
        self, fill_factor: float, order: torch.Tensor
    ) -> torch.Tensor:
        """Exact Toeplitz matrix of a centered periodic interval.

        The unit-cell interval is ``[(1-f)/2, (1+f)/2)``.  Its Fourier
        coefficient is ``f sinc(k f) exp(-i pi k)`` in PyTorch's normalized
        sinc convention.  Keeping this analytic avoids confusing Cartesian
        Li-factorization convergence with raster sampling convergence.
        """
        if not 0.0 < fill_factor < 1.0:
            raise ValueError("fill_factor must be strictly between zero and one.")
        delta = order[:, None] - order[None, :]
        delta_real = delta.to(torch.float64)
        phase = torch.exp(-1.0j * math.pi * delta_real).to(self._dtype)
        return (
            fill_factor * torch.sinc(delta_real * fill_factor).to(self._dtype)
            * phase
        )

    def _rect_cartesian_li_convolutions(
        self,
        eps_bg: torch.Tensor,
        eps_rect: torch.Tensor,
        mu_bg: torch.Tensor,
        mu_rect: torch.Tensor,
        fill_factor_x: float,
        fill_factor_y: float,
    ) -> dict[str, torch.Tensor]:
        """Exact separable Cartesian Li inverse/direct-rule operators.

        For an axis-aligned rectangle, the interface normal is everywhere
        parallel to x or y (apart from measure-zero corners).  The crossed-
        grating factorization can therefore be formed without a sampled normal
        vector field: first construct/invert the x Toeplitz matrices on the two
        y regions, then Fourier-expand that matrix-valued function in y.  This
        is the Cartesian, identity-coordinate limit of Eqs. (13)--(15).
        """
        tx = self._centered_interval_toeplitz(fill_factor_x, self.order_x)
        ty = self._centered_interval_toeplitz(fill_factor_y, self.order_y)
        ix = self._eye(len(self.order_x))
        iy = self._eye(len(self.order_y))

        def kron(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            # torch.linalg.solve may return a non-contiguous strided view.
            # Older supported PyTorch releases implement kron through view(),
            # which rejects that layout even though the matrix is valid.
            return torch.kron(left.contiguous(), right.contiguous())

        indicator = kron(tx, ty)

        def material_set(
            background: torch.Tensor, inclusion: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            if bool(torch.any(torch.abs(background) == 0.0)) or bool(
                torch.any(torch.abs(inclusion) == 0.0)
            ):
                raise ValueError(
                    "Cartesian Li factorization requires nonzero material values."
                )
            direct = background * self._eye(self.order_N) + (
                inclusion - background
            ) * indicator

            x_direct_inside = background * ix + (inclusion - background) * tx
            x_direct_outside_inverse = ix / background
            x_direct_inside_inverse = self._solve(x_direct_inside, ix)
            inverse_x_then_y = kron(x_direct_outside_inverse, iy) + kron(
                x_direct_inside_inverse - x_direct_outside_inverse, ty
            )
            component_22 = self._solve(
                inverse_x_then_y, self._eye(self.order_N)
            )

            reciprocal_bg = 1.0 / background
            reciprocal_rect = 1.0 / inclusion
            x_reciprocal_inside = reciprocal_bg * ix + (
                reciprocal_rect - reciprocal_bg
            ) * tx
            x_reciprocal_outside_inverse = ix / reciprocal_bg
            x_reciprocal_inside_inverse = self._solve(
                x_reciprocal_inside, ix
            )
            component_11 = kron(
                x_reciprocal_outside_inverse, iy
            ) + kron(
                x_reciprocal_inside_inverse
                - x_reciprocal_outside_inverse,
                ty,
            )
            return component_11, component_22, direct

        eps11, eps22, eps33 = material_set(eps_bg, eps_rect)
        mu11, mu22, mu33 = material_set(mu_bg, mu_rect)
        return {
            "eps11": eps11,
            "eps22": eps22,
            "eps33": eps33,
            "mu11": mu11,
            "mu22": mu22,
            "mu33": mu33,
        }

    def _build_conversion_matrix_T(self, mapping: ASRMapping) -> torch.Tensor:
        """
        Discretize Eqs. (21)-(22) in torcwa's +j spatial-phase convention.

        The opposite sign from the printed equations is intentional: the paper
        expands fields with -j spatial phase, whereas torcwa reconstructs fields
        with +j spatial phase.
        """
        mx, my = len(self.order_x), len(self.order_y)
        kx_grid = self.Kx_norm_dn.reshape(mx, my)
        ky_grid = self.Ky_norm_dn.reshape(mx, my)
        if not torch.allclose(kx_grid, kx_grid[:, :1].expand_as(kx_grid)):
            raise RuntimeError("ASR requires a rectangular reciprocal lattice.")
        if not torch.allclose(ky_grid, ky_grid[:1, :].expand_as(ky_grid)):
            raise RuntimeError("ASR requires a rectangular reciprocal lattice.")

        kx = kx_grid[:, 0]
        ky = ky_grid[0, :]
        omega = torch.as_tensor(
            self.omega, dtype=self._dtype, device=self._device
        )
        u = mapping.u.to(self._dtype)
        v = mapping.v.to(self._dtype)
        x = mapping.x.to(self._dtype)
        y = mapping.y.to(self._dtype)
        f = mapping.f.to(self._dtype)
        g = mapping.g.to(self._dtype)

        # Matrix layout must follow Eq. (19): s_xy = T @ s_uv.
        # Therefore rows are Cartesian orders (m, n) and columns are
        # transformed-coordinate orders (p, q).  The previous implementation
        # evaluated the same integral with axes [p, m] / [q, n], effectively
        # using T.T at every interface and breaking power conservation.
        phase_x = torch.exp(
            1.0j
            * omega
            * (
                kx[None, :, None] * u[None, None, :]
                - kx[:, None, None] * x[None, None, :]
            )
        )
        phase_y = torch.exp(
            1.0j
            * omega
            * (
                ky[None, :, None] * v[None, None, :]
                - ky[:, None, None] * y[None, None, :]
            )
        )
        tx_u = torch.mean(phase_x, dim=-1)
        tx_v = torch.mean(g[None, None, :] * phase_y, dim=-1)
        ty_u = torch.mean(f[None, None, :] * phase_x, dim=-1)
        ty_v = torch.mean(phase_y, dim=-1)
        self._last_asr_transform_condition = None
        if self.compute_condition_numbers:
            singular_values = [
                torch.linalg.svdvals(matrix.to(torch.complex128))
                for matrix in (tx_u, tx_v, ty_u, ty_v)
            ]
            tx_max = singular_values[0][0] * singular_values[1][0]
            tx_min = singular_values[0][-1] * singular_values[1][-1]
            ty_max = singular_values[2][0] * singular_values[3][0]
            ty_min = singular_values[2][-1] * singular_values[3][-1]
            self._last_asr_transform_condition = torch.maximum(
                tx_max, ty_max
            ) / torch.minimum(tx_min, ty_min)
        tx = torch.kron(tx_u, tx_v)
        ty = torch.kron(ty_u, ty_v)

        zero = torch.zeros_like(tx)
        return torch.cat(
            (torch.cat((tx, zero), dim=1), torch.cat((zero, ty), dim=1)),
            dim=0,
        )

    def _build_conversion_matrix_Tz(self, mapping: ASRMapping) -> torch.Tensor:
        """Cartesian projection for Ez/Hz, including dx*dy=f*g*du*dv."""
        mx, my = len(self.order_x), len(self.order_y)
        kx_grid = self.Kx_norm_dn.reshape(mx, my)
        ky_grid = self.Ky_norm_dn.reshape(mx, my)
        if not torch.allclose(kx_grid, kx_grid[:, :1].expand_as(kx_grid)):
            raise RuntimeError("ASR requires a rectangular reciprocal lattice.")
        if not torch.allclose(ky_grid, ky_grid[:1, :].expand_as(ky_grid)):
            raise RuntimeError("ASR requires a rectangular reciprocal lattice.")
        kx, ky = kx_grid[:, 0], ky_grid[0, :]
        omega = torch.as_tensor(
            self.omega, dtype=self._dtype, device=self._device
        )
        u, v = mapping.u.to(self._dtype), mapping.v.to(self._dtype)
        x, y = mapping.x.to(self._dtype), mapping.y.to(self._dtype)
        f, g = mapping.f.to(self._dtype), mapping.g.to(self._dtype)
        phase_x = torch.exp(
            1.0j
            * omega
            * (
                kx[None, :, None] * u[None, None, :]
                - kx[:, None, None] * x[None, None, :]
            )
        )
        phase_y = torch.exp(
            1.0j
            * omega
            * (
                ky[None, :, None] * v[None, None, :]
                - ky[:, None, None] * y[None, None, :]
            )
        )
        weighted_x = torch.mean(f[None, None, :] * phase_x, dim=-1)
        weighted_y = torch.mean(g[None, None, :] * phase_y, dim=-1)
        return torch.kron(weighted_x, weighted_y)

    def _build_asr_pq(
        self,
        eps11: torch.Tensor,
        eps22: torch.Tensor,
        eps33: torch.Tensor,
        mu11: torch.Tensor,
        mu22: torch.Tensor,
        mu33: torch.Tensor,
        *,
        factorization_rules: bool,
        direct_convolutions: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if direct_convolutions is None:
            eps33_conv = self._material_conv(eps33)
            mu33_conv = self._material_conv(mu33)
        else:
            eps33_conv = direct_convolutions["eps33"]
            mu33_conv = direct_convolutions["mu33"]
        inv_eps33 = self._solve(eps33_conv, self._eye(self.order_N))
        inv_mu33 = self._solve(mu33_conv, self._eye(self.order_N))

        if factorization_rules:
            mu22_effective = self._factorized_bttb(
                mu22, invert_u_toeplitz=True, invert_final_bttb=True
            )
            mu11_effective = self._factorized_bttb(
                1.0 / mu11,
                invert_u_toeplitz=True,
                invert_final_bttb=False,
            )
            eps22_effective = self._factorized_bttb(
                eps22, invert_u_toeplitz=True, invert_final_bttb=True
            )
            eps11_effective = self._factorized_bttb(
                1.0 / eps11,
                invert_u_toeplitz=True,
                invert_final_bttb=False,
            )
        else:
            if direct_convolutions is None:
                mu22_effective = self._material_conv(mu22)
                mu11_effective = self._material_conv(mu11)
                eps22_effective = self._material_conv(eps22)
                eps11_effective = self._material_conv(eps11)
            else:
                mu22_effective = direct_convolutions["mu22"]
                mu11_effective = direct_convolutions["mu11"]
                eps22_effective = direct_convolutions["eps22"]
                eps11_effective = direct_convolutions["eps11"]

        ku, kv = self.Kx_norm, self.Ky_norm
        p11 = torch.matmul(ku, torch.matmul(inv_eps33, kv))
        p12 = mu22_effective - torch.matmul(
            ku, torch.matmul(inv_eps33, ku)
        )
        p21 = torch.matmul(kv, torch.matmul(inv_eps33, kv)) - mu11_effective
        p22 = -torch.matmul(kv, torch.matmul(inv_eps33, ku))
        p = torch.cat(
            (
                torch.cat((p11, p12), dim=1),
                torch.cat((p21, p22), dim=1),
            ),
            dim=0,
        )

        q11 = -torch.matmul(ku, torch.matmul(inv_mu33, kv))
        q12 = torch.matmul(ku, torch.matmul(inv_mu33, ku)) - eps22_effective
        q21 = eps11_effective - torch.matmul(
            kv, torch.matmul(inv_mu33, kv)
        )
        q22 = torch.matmul(kv, torch.matmul(inv_mu33, ku))
        q = torch.cat(
            (
                torch.cat((q11, q12), dim=1),
                torch.cat((q21, q22), dim=1),
            ),
            dim=0,
        )
        return p, q, eps33_conv, mu33_conv

    def add_layer_rect_asr(
        self,
        thickness,
        eps_bg,
        eps_rect,
        fill_factor_x: float,
        fill_factor_y: float,
        *,
        mu_bg=1.0,
        mu_rect=1.0,
        nx: int = 256,
        ny: int = 256,
        factorization_rules: bool = True,
    ) -> None:
        """Add a centered, axis-aligned rectangular ASR-FR layer."""
        self._require_kvectors()
        thickness_tensor, _ = _real_parameter_tensor(
            "thickness",
            thickness,
            dtype=self._dtype,
            device=self._device,
            allow_zero=True,
        )
        layer_index = self.layer_N
        mapping = self.build_asr_mapping(
            nx, ny, fill_factor_x, fill_factor_y
        )
        x_inside = (mapping.x >= mapping.x_breaks[1]) & (
            mapping.x < mapping.x_breaks[2]
        )
        y_inside = (mapping.y >= mapping.y_breaks[1]) & (
            mapping.y < mapping.y_breaks[2]
        )
        inside = x_inside[:, None] & y_inside[None, :]

        eps_bg_t = torch.as_tensor(
            eps_bg, dtype=self._dtype, device=self._device
        )
        eps_rect_t = torch.as_tensor(
            eps_rect, dtype=self._dtype, device=self._device
        )
        mu_bg_t = torch.as_tensor(
            mu_bg, dtype=self._dtype, device=self._device
        )
        mu_rect_t = torch.as_tensor(
            mu_rect, dtype=self._dtype, device=self._device
        )
        if not all(
            bool(torch.all(torch.isfinite(value)))
            for value in (eps_bg_t, eps_rect_t, mu_bg_t, mu_rect_t)
        ):
            raise ValueError("ASR materials must be finite.")
        eps_uv = torch.where(inside, eps_rect_t, eps_bg_t)
        mu_uv = torch.where(inside, mu_rect_t, mu_bg_t)

        f = mapping.f[:, None].to(self._dtype)
        g = mapping.g[None, :].to(self._dtype)
        eps11 = eps_uv * g / f
        eps22 = eps_uv * f / g
        eps33 = eps_uv * f * g
        mu11 = mu_uv * g / f
        mu22 = mu_uv * f / g
        mu33 = mu_uv * f * g

        quadrature_mapping = mapping
        if self.asr_quadrature_grid is not None:
            quadrature_nx = max(nx, self.asr_quadrature_grid)
            quadrature_ny = max(ny, self.asr_quadrature_grid)
            if quadrature_nx != nx or quadrature_ny != ny:
                quadrature_mapping = self.build_asr_mapping(
                    quadrature_nx,
                    quadrature_ny,
                    fill_factor_x,
                    fill_factor_y,
                )
        direct_convolutions = (
            None
            if factorization_rules
            else self._rect_separable_convolutions(
                quadrature_mapping,
                eps_bg_t,
                eps_rect_t,
                mu_bg_t,
                mu_rect_t,
            )
        )

        p, q, eps33_conv, mu33_conv = self._build_asr_pq(
            eps11,
            eps22,
            eps33,
            mu11,
            mu22,
            mu33,
            factorization_rules=factorization_rules,
            direct_convolutions=direct_convolutions,
        )
        kz_squared, w_uv = self._eig(torch.matmul(p, q))
        kz = self._positive_kz(kz_squared)
        v_uv = self._magnetic_eigenvectors(p, q, w_uv, kz)

        transform = self._build_conversion_matrix_T(quadrature_mapping)
        transform_z = (
            self._build_conversion_matrix_Tz(quadrature_mapping)
            if self.store_mode_couplings
            else None
        )
        w_cartesian = torch.matmul(transform, w_uv)
        v_cartesian = torch.matmul(transform, v_uv)

        self.layer_N += 1
        self.thickness.append(thickness_tensor)
        self.eps_conv.append(eps33_conv)
        self.mu_conv.append(mu33_conv)
        self.P.append(p)
        self.Q.append(q)
        self.kz_norm.append(kz)
        self.E_eigvec_uv.append(w_uv)
        self.H_eigvec_uv.append(v_uv)
        self.E_eigvec.append(w_cartesian)
        self.H_eigvec.append(v_cartesian)
        self.asr_mappings.append(mapping)
        self.asr_T_matrices.append(transform)
        self.asr_Tz_matrices.append(transform_z)
        self.asr_condition_numbers.append(
            self._last_asr_transform_condition
            if self.compute_condition_numbers else None
        )
        self.asr_material_tensors.append(
            {
                "eps_uv": eps_uv,
                "eps11": eps11,
                "eps22": eps22,
                "eps33": eps33,
                "mu_uv": mu_uv,
                "mu11": mu11,
                "mu22": mu22,
                "mu33": mu33,
            }
        )
        slot = len(self.asr_mappings) - 1
        self._asr_slot_by_layer[layer_index] = slot
        if transform_z is not None:
            self._asr_field_context_by_layer[layer_index] = {
                "electric_modes_uv": w_uv,
                "magnetic_modes_uv": v_uv,
                "electric_modes_cartesian": w_cartesian,
                "magnetic_modes_cartesian": v_cartesian,
                "transform_xy": transform,
                "transform_z": transform_z,
                "eps33_conv": eps33_conv,
                "mu33_conv": mu33_conv,
            }

        lx, ly = _as_float(self.L[0]), _as_float(self.L[1])
        x_uniform = (
            torch.arange(nx, dtype=torch.float64, device=self._device) * lx / nx
        )
        y_uniform = (
            torch.arange(ny, dtype=torch.float64, device=self._device) * ly / ny
        )
        physical_inside = (
            (x_uniform[:, None] >= mapping.x_breaks[1])
            & (x_uniform[:, None] < mapping.x_breaks[2])
            & (y_uniform[None, :] >= mapping.y_breaks[1])
            & (y_uniform[None, :] < mapping.y_breaks[2])
        )
        eps_physical = torch.where(physical_inside, eps_rect_t, eps_bg_t)
        mu_physical = torch.where(physical_inside, mu_rect_t, mu_bg_t)
        self._physical_material_by_layer[layer_index] = (
            eps_physical,
            mu_physical,
        )
        width_x, width_y = fill_factor_x * lx, fill_factor_y * ly
        shape = "square" if math.isclose(
            width_x, width_y, rel_tol=1.0e-8, abs_tol=1.0e-12
        ) else "rectangle"
        self.layer_records.append(
            LayerRecord(
                index=layer_index,
                method="asr-fr" if factorization_rules else "asr",
                shape=shape,
                lattice=getattr(self, "lattice_kind", "rectangular"),
                reason="RECTILINEAR_SEPARABLE_BOUNDARY",
                options={
                    "asr_G": self.asr_G,
                    "grid": (nx, ny),
                    "quadrature_grid": (
                        len(quadrature_mapping.u), len(quadrature_mapping.v)
                    ),
                    "factorization_rules": factorization_rules,
                    "fill_factor": (fill_factor_x, fill_factor_y),
                },
            )
        )
        self._append_smatrix_from_cartesian_modes(
            w_cartesian, v_cartesian
        )

    def add_layer_rect_li(
        self,
        thickness,
        eps_bg,
        eps_rect,
        fill_factor_x: float,
        fill_factor_y: float,
        *,
        mu_bg=1.0,
        mu_rect=1.0,
        physical_grid: int = 256,
    ) -> None:
        """Add an axis-aligned rectangle using Cartesian Li factorization.

        No ASR coordinate map and no normal-vector field are used.  The
        Fourier operators are analytic for the centered rectangle; the grid is
        retained only for optional real-space field/material reconstruction.
        """
        self._require_kvectors()
        if not isinstance(physical_grid, int) or physical_grid < 2:
            raise ValueError("physical_grid must be an integer >= 2.")
        thickness_tensor, _ = _real_parameter_tensor(
            "thickness",
            thickness,
            dtype=self._dtype,
            device=self._device,
            allow_zero=True,
        )
        layer_index = self.layer_N
        materials = tuple(
            torch.as_tensor(value, dtype=self._dtype, device=self._device)
            for value in (eps_bg, eps_rect, mu_bg, mu_rect)
        )
        if not all(bool(torch.all(torch.isfinite(value))) for value in materials):
            raise ValueError("Cartesian Li materials must be finite.")
        eps_bg_t, eps_rect_t, mu_bg_t, mu_rect_t = materials
        convolutions = self._rect_cartesian_li_convolutions(
            eps_bg_t,
            eps_rect_t,
            mu_bg_t,
            mu_rect_t,
            fill_factor_x,
            fill_factor_y,
        )
        p, q, eps_conv, mu_conv = self._build_asr_pq(
            convolutions["eps11"],
            convolutions["eps22"],
            convolutions["eps33"],
            convolutions["mu11"],
            convolutions["mu22"],
            convolutions["mu33"],
            factorization_rules=False,
            direct_convolutions=convolutions,
        )
        kz_squared, electric_modes = self._eig(torch.matmul(p, q))
        kz = self._positive_kz(kz_squared)
        magnetic_modes = self._magnetic_eigenvectors(
            p, q, electric_modes, kz
        )

        self.layer_N += 1
        self.thickness.append(thickness_tensor)
        self.eps_conv.append(eps_conv)
        self.mu_conv.append(mu_conv)
        self.P.append(p)
        self.Q.append(q)
        self.kz_norm.append(kz)
        self.E_eigvec.append(electric_modes)
        self.H_eigvec.append(magnetic_modes)

        lx, ly = _as_float(self.L[0]), _as_float(self.L[1])
        x = (torch.arange(physical_grid, device=self._device) + 0.5) / physical_grid
        y = (torch.arange(physical_grid, device=self._device) + 0.5) / physical_grid
        inside_x = torch.abs(x - 0.5) < fill_factor_x / 2.0
        inside_y = torch.abs(y - 0.5) < fill_factor_y / 2.0
        inside = inside_x[:, None] & inside_y[None, :]
        self._physical_material_by_layer[layer_index] = (
            torch.where(inside, eps_rect_t, eps_bg_t),
            torch.where(inside, mu_rect_t, mu_bg_t),
        )
        width_x, width_y = fill_factor_x * lx, fill_factor_y * ly
        shape = "square" if math.isclose(
            width_x, width_y, rel_tol=1.0e-8, abs_tol=1.0e-12
        ) else "rectangle"
        self.layer_records.append(
            LayerRecord(
                index=layer_index,
                method="cartesian-li",
                shape=shape,
                lattice="rectangular",
                reason="AXIS_ALIGNED_ANALYTIC_LI_FACTORIZATION",
                options={
                    "fill_factor": (fill_factor_x, fill_factor_y),
                    "factorization_scheme": "x-Toeplitz-inverse/y-BTTB-Li",
                    "fourier_coefficients": "analytic",
                    "physical_grid": physical_grid,
                },
            )
        )
        self._append_smatrix_from_cartesian_modes(
            electric_modes, magnetic_modes
        )

    def add_layer_metal_patch_asr(
        self,
        thickness,
        eps_bg,
        eps_metal,
        fill_factor_x: float,
        fill_factor_y: float,
        *,
        nx: int = 256,
        ny: int = 256,
        factorization_rules: bool = True,
    ) -> None:
        """Backward-compatible name for a nonmagnetic rectangular patch."""
        self.add_layer_rect_asr(
            thickness,
            eps_bg,
            eps_metal,
            fill_factor_x,
            fill_factor_y,
            nx=nx,
            ny=ny,
            factorization_rules=factorization_rules,
        )
