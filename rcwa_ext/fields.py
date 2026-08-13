"""Fourier-field reconstruction and spatial synthesis."""

from __future__ import annotations

from typing import Sequence

import torch

from .config import _as_float

class _FieldRecoveryMixin:
    """Recover internal/external Fourier amplitudes and sampled fields."""

    def _coordinate_tensor(self, value: object) -> torch.Tensor:
        real_dtype = (
            torch.float32 if self._dtype == torch.complex64 else torch.float64
        )
        probe = torch.as_tensor(value, device=self._device)
        if torch.is_complex(probe):
            if torch.any(torch.abs(probe.imag) > 1.0e-12):
                raise ValueError("Field coordinates must be real.")
            return probe.real.to(real_dtype)
        # Supplying dtype at construction avoids first rounding Python lists to
        # float32 and only then widening them in complex128 simulations.
        return torch.as_tensor(
            value, dtype=real_dtype, device=self._device
        )

    def _require_solved_source(self) -> None:
        if not hasattr(self, "S"):
            raise RuntimeError("Call solve_global_smatrix() before requesting fields.")
        if not hasattr(self, "E_i"):
            raise RuntimeError(
                "Call source_planewave() or source_fourier() before requesting fields."
            )
        if getattr(self, "source_direction", None) not in {"forward", "backward"}:
            raise RuntimeError("Field source_direction must be 'forward' or 'backward'.")
        expected = 2 * self.order_N
        if self.E_i.ndim != 2 or tuple(self.E_i.shape) != (expected, 1):
            raise ValueError(
                f"Field reconstruction expects one Fourier source column with "
                f"shape ({expected}, 1)."
            )
        bases = getattr(self, "_polarization_bases", None)
        if bases is not None:
            electric_basis = bases[0]
            projected = electric_basis @ (electric_basis.mH @ self.E_i)
            denominator = torch.maximum(
                torch.linalg.vector_norm(self.E_i),
                torch.as_tensor(
                    torch.finfo(self.E_i.real.dtype).tiny,
                    dtype=self.E_i.real.dtype,
                    device=self._device,
                ),
            )
            residual = torch.linalg.vector_norm(self.E_i - projected) / denominator
            tolerance = 2.0e-4 if self._dtype == torch.complex64 else 2.0e-9
            if _as_float(residual) > tolerance:
                raise ValueError(
                    "The requested source lies outside the solved D6/Cs/C2 "
                    "subspace. Use the matching x/y source, a symmetry-adapted "
                    "Fourier source, or disable source-specific reduction."
                )

    def _require_field_region(self, *, internal: bool) -> None:
        """Enforce ``OutputSpec.fields`` without breaking direct solver use.

        ``field_regions`` is installed by :class:`AutoRCWA`.  Direct
        ``CustomRCWA_*`` construction predates ``OutputSpec``; for that API,
        stored couplings imply all regions and scattering alone permits only
        an external-field reconstruction.
        """
        configured = getattr(
            self,
            "field_regions",
            "all" if self.store_mode_couplings else "external",
        )
        allowed = {"internal", "all"} if internal else {"external", "all"}
        if configured not in allowed:
            region = "internal" if internal else "external"
            raise RuntimeError(
                f"{region.capitalize()} fields were not requested; construct "
                f"the simulation with OutputSpec(fields='{region}') or "
                "OutputSpec(fields='all')."
            )

    def _internal_fourier_fields(
        self, layer_num: int, z_prop: object
    ) -> tuple[torch.Tensor, ...]:
        if not self.store_mode_couplings:
            raise RuntimeError(
                "Internal fields require enable_fields=True (or "
                "store_mode_couplings=True) when the simulation is created."
            )
        direction_index = 0 if self.source_direction == "forward" else 1
        if (
            not hasattr(self, "C")
            or len(self.C[direction_index]) <= layer_num
        ):
            raise RuntimeError(
                "Internal mode-coupling coefficients are unavailable; solve again "
                "with field storage enabled."
            )

        coupling = torch.matmul(
            self.C[direction_index][layer_num], self.E_i
        ).squeeze(-1)
        kz = self.kz_norm[layer_num]
        mode_count = int(kz.numel())
        z_value = torch.as_tensor(
            z_prop, dtype=self._dtype, device=self._device
        )
        thickness = torch.as_tensor(
            self.thickness[layer_num], dtype=self._dtype, device=self._device
        )
        if z_value.numel() != 1:
            raise ValueError("z_prop must be a scalar; use field_xz/field_yz for a z axis.")
        z_real = _as_float(z_value)
        thickness_real = _as_float(thickness)
        tolerance = 1.0e-10 * max(1.0, abs(thickness_real))
        if z_real < -tolerance or z_real > thickness_real + tolerance:
            raise ValueError(
                f"Internal z_prop must lie in [0, {thickness_real:g}]."
            )
        a_plus = (
            torch.exp(1.0j * self.omega * kz * z_value)
            * coupling[:mode_count]
        )
        a_minus = (
            torch.exp(1.0j * self.omega * kz * (thickness - z_value))
            * coupling[mode_count:]
        )
        electric_amplitude = a_plus + a_minus
        magnetic_amplitude = a_plus - a_minus

        if layer_num in self._asr_field_context_by_layer:
            context = self._asr_field_context_by_layer[layer_num]
            electric_uv = torch.matmul(
                context["electric_modes_uv"], electric_amplitude
            )
            magnetic_uv = torch.matmul(
                context["magnetic_modes_uv"], magnetic_amplitude
            )
            eu, ev = electric_uv[: self.order_N], electric_uv[self.order_N :]
            hu, hv = magnetic_uv[: self.order_N], magnetic_uv[self.order_N :]
            electric_rhs = torch.matmul(self.Ky_norm, hu) - torch.matmul(
                self.Kx_norm, hv
            )
            magnetic_rhs = torch.matmul(self.Kx_norm, ev) - torch.matmul(
                self.Ky_norm, eu
            )
            scalar_embedding = context.get("scalar_embedding")
            if scalar_embedding is None:
                ez_uv = self._solve(context["eps33_conv"], electric_rhs)
                hz_uv = self._solve(context["mu33_conv"], magnetic_rhs)
                ez_reduced = None
                hz_reduced = None
            else:
                ez_reduced = self._solve(
                    context["eps33_conv_reduced"],
                    scalar_embedding.mH @ electric_rhs,
                )
                hz_reduced = self._solve(
                    context["mu33_conv_reduced"],
                    scalar_embedding.mH @ magnetic_rhs,
                )
                ez_uv = scalar_embedding @ ez_reduced
                hz_uv = scalar_embedding @ hz_reduced
            electric_modes_cartesian = context.get("electric_modes_cartesian")
            magnetic_modes_cartesian = context.get("magnetic_modes_cartesian")
            if electric_modes_cartesian is None:
                electric_xy = torch.matmul(context["transform_xy"], electric_uv)
                magnetic_xy = torch.matmul(context["transform_xy"], magnetic_uv)
            else:
                electric_xy = electric_modes_cartesian @ electric_amplitude
                magnetic_xy = magnetic_modes_cartesian @ magnetic_amplitude
            transform_z_reduced = context.get("transform_z_reduced")
            if transform_z_reduced is None:
                ez = torch.matmul(context["transform_z"], ez_uv)
                hz = torch.matmul(context["transform_z"], hz_uv)
            else:
                assert ez_reduced is not None and hz_reduced is not None
                ez = scalar_embedding @ (transform_z_reduced @ ez_reduced)
                hz = scalar_embedding @ (transform_z_reduced @ hz_reduced)
        else:
            electric_xy = torch.matmul(
                self.E_eigvec[layer_num], electric_amplitude
            )
            magnetic_xy = torch.matmul(
                self.H_eigvec[layer_num], magnetic_amplitude
            )
            ex0, ey0 = (
                electric_xy[: self.order_N],
                electric_xy[self.order_N :],
            )
            hx0, hy0 = (
                magnetic_xy[: self.order_N],
                magnetic_xy[self.order_N :],
            )
            electric_rhs = torch.matmul(self.Ky_norm, hx0) - torch.matmul(
                self.Kx_norm, hy0
            )
            magnetic_rhs = torch.matmul(self.Kx_norm, ey0) - torch.matmul(
                self.Ky_norm, ex0
            )
            context = self._reduced_longitudinal_context_by_layer.get(layer_num)
            if context is None:
                ez = self._solve(self.eps_conv[layer_num], electric_rhs)
                hz = self._solve(self.mu_conv[layer_num], magnetic_rhs)
            else:
                scalar_embedding = context["scalar_embedding"]
                ez = scalar_embedding @ self._solve(
                    context["eps_conv_reduced"],
                    scalar_embedding.mH @ electric_rhs,
                )
                hz = scalar_embedding @ self._solve(
                    context["mu_conv_reduced"],
                    scalar_embedding.mH @ magnetic_rhs,
                )

        ex, ey = electric_xy[: self.order_N], electric_xy[self.order_N :]
        hx, hy = magnetic_xy[: self.order_N], magnetic_xy[self.order_N :]
        return ex, ey, ez, hx, hy, hz

    def _external_fourier_fields(
        self, layer_num: int, z_prop: object
    ) -> tuple[torch.Tensor, ...]:
        if layer_num not in {-1, self.layer_N}:
            raise ValueError("External layer must be -1 (input) or layer_N (output).")
        input_side = layer_num == -1
        eps = self.eps_in if input_side else self.eps_out
        mu = self.mu_in if input_side else self.mu_out
        v_matrix = (
            getattr(self, "Vi", self.Vf)
            if input_side
            else getattr(self, "Vo", self.Vf)
        )
        kz = torch.sqrt(eps * mu - self.Kx_norm_dn**2 - self.Ky_norm_dn**2)
        if input_side:
            # z<=0: the evanescent branch must decay as the distance from the
            # interface grows, so Im(kz)<=0 for exp(+j*kz*z).
            kz = torch.where(kz.imag > 0.0, torch.conj(kz), kz)
        else:
            # z>=0: the outgoing branch decays for Im(kz)>=0.
            kz = torch.where(kz.imag < 0.0, torch.conj(kz), kz)
        kz2 = torch.cat((kz, kz))
        z_value = torch.as_tensor(
            z_prop, dtype=self._dtype, device=self._device
        )
        if z_value.numel() != 1:
            raise ValueError("z_prop must be a scalar; use field_xz/field_yz for a z axis.")
        z_real = _as_float(z_value)
        if (input_side and z_real > 1.0e-12) or (
            not input_side and z_real < -1.0e-12
        ):
            side = "nonpositive" if input_side else "nonnegative"
            raise ValueError(f"External z_prop must be {side} on this side.")
        phase_forward = torch.exp(1.0j * self.omega * kz2 * z_value)
        # Conjugating the selected outgoing phase gives the counter-propagating
        # branch while retaining decay for evanescent harmonics on either side.
        phase_backward = torch.conj(phase_forward)
        incident = self.E_i.squeeze(-1)
        scattering = getattr(self, "_field_smatrix", self.S)
        zero = torch.zeros_like(incident)

        if input_side and self.source_direction == "forward":
            electric_plus = incident * phase_forward
            electric_minus = torch.matmul(scattering[1], incident) * phase_backward
        elif input_side:
            electric_plus = zero
            electric_minus = torch.matmul(scattering[3], incident) * phase_backward
        elif self.source_direction == "forward":
            electric_plus = torch.matmul(scattering[0], incident) * phase_forward
            electric_minus = zero
        else:
            electric_plus = torch.matmul(scattering[2], incident) * phase_forward
            electric_minus = incident * phase_backward

        magnetic_plus = torch.matmul(v_matrix, electric_plus)
        magnetic_minus = -torch.matmul(v_matrix, electric_minus)
        electric_xy = electric_plus + electric_minus
        magnetic_xy = magnetic_plus + magnetic_minus
        ex, ey = electric_xy[: self.order_N], electric_xy[self.order_N :]
        hx, hy = magnetic_xy[: self.order_N], magnetic_xy[self.order_N :]
        eps_t = torch.as_tensor(eps, dtype=self._dtype, device=self._device)
        mu_t = torch.as_tensor(mu, dtype=self._dtype, device=self._device)
        ez = (self.Ky_norm_dn * hx - self.Kx_norm_dn * hy) / eps_t
        hz = (self.Kx_norm_dn * ey - self.Ky_norm_dn * ex) / mu_t
        return ex, ey, ez, hx, hy, hz

    def _fourier_fields(
        self, layer_num: int, z_prop: object
    ) -> tuple[torch.Tensor, ...]:
        self._require_solved_source()
        if layer_num in {-1, self.layer_N}:
            self._require_field_region(internal=False)
            return self._external_fourier_fields(layer_num, z_prop)
        if not 0 <= layer_num < self.layer_N:
            raise IndexError(
                f"layer_num must be -1, 0..{self.layer_N - 1}, or {self.layer_N}."
            )
        self._require_field_region(internal=True)
        return self._internal_fourier_fields(layer_num, z_prop)

    def _synthesize_xy(
        self,
        coefficients: Sequence[torch.Tensor],
        x_axis: object,
        y_axis: object,
    ) -> tuple[torch.Tensor, ...]:
        x = self._coordinate_tensor(x_axis).reshape(-1, 1, 1)
        y = self._coordinate_tensor(y_axis).reshape(1, -1, 1)
        phase = torch.exp(
            1.0j
            * self.omega
            * (
                self.Kx_norm_dn.reshape(1, 1, -1) * x
                + self.Ky_norm_dn.reshape(1, 1, -1) * y
            )
        )
        return tuple(
            torch.sum(value.reshape(1, 1, -1) * phase, dim=-1)
            for value in coefficients
        )

    def field_xy(self, layer_num, x_axis, y_axis, z_prop=0.0):
        """Return Cartesian [E,H] on an xy plane, including transformed ASR layers."""
        layer_num = int(layer_num)
        values = self._synthesize_xy(
            self._fourier_fields(layer_num, z_prop), x_axis, y_axis
        )
        return list(values[:3]), list(values[3:])

    def _locate_z(self, z_value: float) -> tuple[int, float]:
        if self.layer_N == 0:
            return (-1, z_value) if z_value < 0.0 else (0, z_value)
        if z_value < 0.0:
            return -1, z_value
        lower = 0.0
        for layer, thickness in enumerate(self.thickness):
            upper = lower + _as_float(thickness)
            if z_value <= upper:
                return layer, z_value - lower
            lower = upper
        return self.layer_N, z_value - lower

    def field_xz(self, x_axis, z_axis, y):
        """Return Cartesian [E,H] on an xz plane at physical coordinate y."""
        z_values = self._coordinate_tensor(z_axis).reshape(-1)
        if z_values.numel() == 0:
            raise ValueError("z_axis must contain at least one coordinate.")
        columns: list[list[torch.Tensor]] = [[] for _ in range(6)]
        for z in z_values:
            layer_num, local_z = self._locate_z(_as_float(z))
            electric, magnetic = self.field_xy(
                layer_num, x_axis, [y], z_prop=local_z
            )
            for index, value in enumerate((*electric, *magnetic)):
                columns[index].append(value[:, 0])
        result = [torch.stack(component, dim=1) for component in columns]
        return result[:3], result[3:]

    def field_yz(self, y_axis, z_axis, x):
        """Return Cartesian [E,H] on a yz plane at physical coordinate x."""
        z_values = self._coordinate_tensor(z_axis).reshape(-1)
        if z_values.numel() == 0:
            raise ValueError("z_axis must contain at least one coordinate.")
        columns: list[list[torch.Tensor]] = [[] for _ in range(6)]
        for z in z_values:
            layer_num, local_z = self._locate_z(_as_float(z))
            electric, magnetic = self.field_xy(
                layer_num, [x], y_axis, z_prop=local_z
            )
            for index, value in enumerate((*electric, *magnetic)):
                columns[index].append(value[0, :])
        result = [torch.stack(component, dim=1) for component in columns]
        return result[:3], result[3:]

    def return_layer(self, layer_num, nx=100, ny=100):
        """Return physical material grids; ASR tensors are never exposed as epsilon."""
        layer_num = int(layer_num)
        if layer_num not in self._physical_material_by_layer:
            return _ORIGINAL_TORCWA_RCWA.return_layer(
                self, layer_num, nx=nx, ny=ny
            )
        eps_grid, mu_grid = self._physical_material_by_layer[layer_num]
        ix = (
            torch.arange(nx, device=self._device, dtype=torch.int64)
            * eps_grid.shape[0]
            // nx
        )
        iy = (
            torch.arange(ny, device=self._device, dtype=torch.int64)
            * eps_grid.shape[1]
            // ny
        )
        return eps_grid[ix[:, None], iy[None, :]], mu_grid[
            ix[:, None], iy[None, :]
        ]
