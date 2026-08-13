"""ASR tensor factorization, modal operators, and layer assembly."""

from __future__ import annotations

import math

import torch

from .asr_maps import ASRMapping, CircleASRMapping, _ASRMappingMixin
from .config import (
    LayerRecord, UnsupportedCombinationError, _ORIGINAL_TORCWA_RCWA,
    _as_float, _real_parameter_tensor,
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
        self.matched_asr_G = float(kwargs.pop("matched_asr_G", 3.0e-2))
        if not 0.0 < self.asr_G < 1.0:
            raise ValueError("asr_G must be in (0, 1); the paper uses 0.001.")
        if not 0.0 < self.matched_asr_G < 1.0:
            raise ValueError(
                "matched_asr_G must be in (0,1); 0.03 corresponds to "
                "Weiss et al. eta=0.97."
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        eps33_conv = self._material_conv(eps33)
        mu33_conv = self._material_conv(mu33)
        inv_eps33 = self._solve(eps33_conv, self._eye(self.order_N))
        inv_mu33 = self._solve(mu33_conv, self._eye(self.order_N))
        if factorization_rules:
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

    def add_layer_circle_asr(
        self,
        thickness,
        radius: float,
        eps_bg,
        eps_cyl,
        *,
        mu_bg=1.0,
        mu_cyl=1.0,
        nx: int = 256,
        ny: int = 256,
        factorization_rules: bool = True,
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
        material_values = [
            torch.as_tensor(value, dtype=self._dtype, device=self._device)
            for value in (eps_bg, eps_cyl, mu_bg, mu_cyl)
        ]
        if not all(bool(torch.all(torch.isfinite(v))) for v in material_values):
            raise ValueError("matched-ASR materials must be finite.")
        if any(bool(torch.any(torch.abs(v) == 0.0)) for v in material_values):
            raise ValueError("matched-ASR epsilon and mu values must be nonzero.")
        eps_bg_t, eps_cyl_t, mu_bg_t, mu_cyl_t = material_values
        radius, radius_value = _real_parameter_tensor(
            "radius",
            radius,
            dtype=torch.float64,
            device=self._device,
            allow_zero=False,
        )
        triangular = getattr(self, "lattice_kind", "rectangular") == "triangular"
        if getattr(self, "group_theory_symmetry", "auto") == "d6" and not triangular:
            raise UnsupportedCombinationError(
                "symmetry='d6' requires a 60-degree equal-length triangular cell."
            )
        if triangular:
            mapping = self.build_triangular_circle_asr_mapping(nx, ny, radius)
        else:
            if hasattr(self, "cos_zeta") and abs(_as_float(self.cos_zeta)) > 1.0e-8:
                raise UnsupportedCombinationError(
                    "matched circle ASR supports orthogonal or 60-degree triangular cells."
                )
            mapping = self.build_circle_asr_mapping(nx, ny, radius)
        lx, ly = _as_float(self.L[0]), _as_float(self.L[1])
        inside = self._periodic_circle_mask(mapping.x, mapping.y, radius)
        eps_uv = torch.where(inside, eps_cyl_t, eps_bg_t)
        mu_uv = torch.where(inside, mu_cyl_t, mu_bg_t)

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
        )
        transform, transform_z_all = self._build_circle_conversion_matrices(mapping)
        layer_index = self.layer_N
        complete_d6 = (
            bool(getattr(self, "use_group_theory", False))
            and getattr(self, "group_theory_symmetry", "auto") == "d6"
            and getattr(self, "polarization_reduction", None) is None
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
        self._physical_material_by_layer[layer_index] = (
            torch.where(physical_inside, eps_cyl_t, eps_bg_t),
            torch.where(physical_inside, mu_cyl_t, mu_bg_t),
        )
        self.layer_records.append(
            LayerRecord(
                index=layer_index,
                method="matched-asr",
                shape="circle",
                lattice=getattr(self, "lattice_kind", "rectangular"),
                reason="MATCHED_CIRCLE_COORDINATES",
                options={
                    "radius": radius_value,
                    "grid": (nx, ny),
                    "asr_G": self.matched_asr_G,
                    "factorization_rules": factorization_rules,
                    "map": (
                        "D6 hex-to-circle periodic Hermite map"
                        if triangular
                        else "Weiss-37-38 + separable ASR"
                    ),
                    "conversion": "general-2d-T",
                    "polarization": getattr(self, "polarization_reduction", None),
                    "group_theory": (
                        dict(self.group_theory_diagnostics[-1])
                        if complete_d6
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        eps33_conv = self._material_conv(eps33)
        mu33_conv = self._material_conv(mu33)
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
            mu22_effective = self._material_conv(mu22)
            mu11_effective = self._material_conv(mu11)
            eps22_effective = self._material_conv(eps22)
            eps11_effective = self._material_conv(eps11)

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

        p, q, eps33_conv, mu33_conv = self._build_asr_pq(
            eps11,
            eps22,
            eps33,
            mu11,
            mu22,
            mu33,
            factorization_rules=factorization_rules,
        )
        kz_squared, w_uv = self._eig(torch.matmul(p, q))
        kz = self._positive_kz(kz_squared)
        v_uv = self._magnetic_eigenvectors(p, q, w_uv, kz)

        transform = self._build_conversion_matrix_T(mapping)
        transform_z = (
            self._build_conversion_matrix_Tz(mapping)
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
            torch.linalg.cond(transform.to(torch.complex128))
            if self.compute_condition_numbers
            else None
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
                method="asr-fr",
                shape=shape,
                lattice=getattr(self, "lattice_kind", "rectangular"),
                reason="RECTILINEAR_SEPARABLE_BOUNDARY",
                options={
                    "asr_G": self.asr_G,
                    "grid": (nx, ny),
                    "factorization_rules": factorization_rules,
                    "fill_factor": (fill_factor_x, fill_factor_y),
                },
            )
        )
        self._append_smatrix_from_cartesian_modes(
            w_cartesian, v_cartesian
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
