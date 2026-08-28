"""C2/C2v/D6 bases and source-specific polarization eigenspaces."""

from __future__ import annotations

import math
import warnings
from typing import Sequence

import torch

from .config import (
    UnsupportedCombinationError, _TWO_PI, _as_float,
)

class _SymmetryReductionMixin:
    """Build invariant harmonic subspaces and reduced eigensystems."""

    def _one_dimensional_parity_bases(
        self,
        orders: torch.Tensor,
        fundamental: float,
        center: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Phase-aware even/odd Fourier bases about ``center``."""
        order_values = [int(value) for value in orders.detach().cpu().tolist()]
        index = {value: position for position, value in enumerate(order_values)}
        if 0 not in index or any(-value not in index for value in order_values):
            raise UnsupportedCombinationError(
                "Group theory requires a sign-symmetric Fourier-order set."
            )

        even_columns: list[torch.Tensor] = []
        odd_columns: list[torch.Tensor] = []
        zero_column = torch.zeros(
            len(order_values), dtype=self._dtype, device=self._device
        )
        zero_column[index[0]] = 1.0
        even_columns.append(zero_column)
        root_two = math.sqrt(2.0)
        for order_value in sorted(value for value in order_values if value > 0):
            positive = index[order_value]
            negative = index[-order_value]
            phase = torch.exp(
                torch.as_tensor(
                    -1.0j * fundamental * order_value * center,
                    dtype=self._dtype,
                    device=self._device,
                )
            )
            even = torch.zeros_like(zero_column)
            odd = torch.zeros_like(zero_column)
            even[positive] = phase / root_two
            even[negative] = torch.conj(phase) / root_two
            odd[positive] = phase / root_two
            odd[negative] = -torch.conj(phase) / root_two
            even_columns.append(even)
            odd_columns.append(odd)

        even_basis = torch.stack(even_columns, dim=1)
        odd_basis = (
            torch.stack(odd_columns, dim=1)
            if odd_columns
            else torch.empty(
                (len(order_values), 0),
                dtype=self._dtype,
                device=self._device,
            )
        )
        return even_basis, odd_basis

    @staticmethod
    def _kronecker(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ab,cd->acbd", first, second).reshape(
            first.shape[0] * second.shape[0],
            first.shape[1] * second.shape[1],
        )

    def _component_basis(
        self, first: torch.Tensor, second: torch.Tensor
    ) -> torch.Tensor:
        """Place scalar bases in the Ex and Ey blocks, respectively."""
        row_count = self.order_N
        top = torch.cat(
            (
                first,
                torch.zeros(
                    (row_count, second.shape[1]),
                    dtype=self._dtype,
                    device=self._device,
                ),
            ),
            dim=1,
        )
        bottom = torch.cat(
            (
                torch.zeros(
                    (row_count, first.shape[1]),
                    dtype=self._dtype,
                    device=self._device,
                ),
                second,
            ),
            dim=1,
        )
        return torch.cat((top, bottom), dim=0)

    def _c2v_group_blocks(
        self, center: tuple[float, float]
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        lx, ly = _as_float(self.L[0]), _as_float(self.L[1])
        even_x, odd_x = self._one_dimensional_parity_bases(
            self.order_x, _TWO_PI / lx, center[0]
        )
        even_y, odd_y = self._one_dimensional_parity_bases(
            self.order_y, _TWO_PI / ly, center[1]
        )
        ee = self._kronecker(even_x, even_y)
        eo = self._kronecker(even_x, odd_y)
        oe = self._kronecker(odd_x, even_y)
        oo = self._kronecker(odd_x, odd_y)
        w1 = self._component_basis(oe, eo)
        w2 = self._component_basis(eo, oe)
        w3 = self._component_basis(ee, oo)
        w4 = self._component_basis(oo, ee)
        return [(w1, w2), (w2, w1), (w3, w4), (w4, w3)]

    def _c2_scalar_bases(
        self, center: tuple[float, float]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Even/odd scalar bases for inversion about ``center``."""
        order_x = [int(value) for value in self.order_x.detach().cpu().tolist()]
        order_y = [int(value) for value in self.order_y.detach().cpu().tolist()]
        pairs = [(m, n) for m in order_x for n in order_y]
        index = {pair: position for position, pair in enumerate(pairs)}
        if any((-m, -n) not in index for m, n in pairs):
            raise UnsupportedCombinationError(
                "Group theory requires a sign-symmetric Fourier-order set."
            )

        lx, ly = _as_float(self.L[0]), _as_float(self.L[1])
        even_columns: list[torch.Tensor] = []
        odd_columns: list[torch.Tensor] = []
        processed: set[tuple[int, int]] = set()
        root_two = math.sqrt(2.0)
        for m, n in pairs:
            if (m, n) in processed:
                continue
            opposite = (-m, -n)
            column = torch.zeros(
                self.order_N, dtype=self._dtype, device=self._device
            )
            if opposite == (m, n):
                column[index[(m, n)]] = 1.0
                even_columns.append(column)
            else:
                gx = _TWO_PI * m / lx
                gy = (_TWO_PI * n / ly - gx * self.cos_zeta) / self.sin_zeta
                phase = torch.exp(
                    torch.as_tensor(
                        -1.0j * (gx * center[0] + gy * center[1]),
                        dtype=self._dtype,
                        device=self._device,
                    )
                )
                even = column.clone()
                odd = column.clone()
                even[index[(m, n)]] = phase / root_two
                even[index[opposite]] = torch.conj(phase) / root_two
                odd[index[(m, n)]] = phase / root_two
                odd[index[opposite]] = -torch.conj(phase) / root_two
                even_columns.append(even)
                odd_columns.append(odd)
            processed.add((m, n))
            processed.add(opposite)

        odd_basis = (
            torch.stack(odd_columns, dim=1)
            if odd_columns
            else torch.empty(
                (self.order_N, 0),
                dtype=self._dtype,
                device=self._device,
            )
        )
        return torch.stack(even_columns, dim=1), odd_basis

    def _c2_group_blocks(
        self, center: tuple[float, float]
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        even, odd = self._c2_scalar_bases(center)
        even_vector = self._component_basis(even, even)
        odd_vector = self._component_basis(odd, odd)
        return [(even_vector, even_vector), (odd_vector, odd_vector)]

    def _group_theory_failure(
        self, layer_index: int, symmetry: str, reason: str
    ) -> None:
        diagnostic = {
            "layer": layer_index,
            "requested": True,
            "applied": False,
            "symmetry": symmetry,
            "reason": reason,
        }
        self.group_theory_diagnostics.append(diagnostic)
        if self.group_theory_strict:
            raise UnsupportedCombinationError(reason)
        warnings.warn(
            f"Group-theory eigensolve was not used for layer {layer_index}: {reason}",
            RuntimeWarning,
            stacklevel=3,
        )

    def _group_theory_eigendecomposition(
        self,
        p: torch.Tensor,
        q: torch.Tensor,
        centers: Sequence[tuple[float, float]],
        layer_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Block-diagonal eigensolve with an invariance residual check."""
        symmetry = self.group_theory_symmetry
        if symmetry == "auto":
            symmetry = "c2v" if abs(self.zeta_deg - 90.0) <= 1.0e-7 else "c2"
        if symmetry == "d6":
            self._group_theory_failure(
                layer_index,
                symmetry,
                "complete D6 must be assembled on the native triangular star "
                "by the NVM or matched-ASR circle backend",
            )
            return None
        if abs(_as_float(self.inc_ang)) > 1.0e-8:
            self._group_theory_failure(
                layer_index, symmetry, "group theory currently requires normal incidence"
            )
            return None
        if len(centers) != 1:
            self._group_theory_failure(
                layer_index, symmetry, "group theory currently requires one circle per cell"
            )
            return None
        if symmetry == "c2v" and abs(self.zeta_deg - 90.0) > 1.0e-7:
            self._group_theory_failure(
                layer_index, symmetry, "C2v reduction requires an orthogonal cell"
            )
            return None

        center = centers[0]
        try:
            blocks = (
                self._c2v_group_blocks(center)
                if symmetry == "c2v"
                else self._c2_group_blocks(center)
            )
        except UnsupportedCombinationError as error:
            self._group_theory_failure(layer_index, symmetry, str(error))
            return None

        projected: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        residuals: list[torch.Tensor] = []
        for electric_basis, magnetic_basis in blocks:
            if electric_basis.shape[1] == 0:
                continue
            p_image = torch.matmul(p, magnetic_basis)
            q_image = torch.matmul(q, electric_basis)
            p_sub = torch.matmul(electric_basis.mH, p_image)
            q_sub = torch.matmul(magnetic_basis.mH, q_image)
            real_dtype = p_image.real.dtype
            tiny = torch.as_tensor(
                torch.finfo(real_dtype).tiny,
                dtype=real_dtype,
                device=self._device,
            )
            p_residual = torch.linalg.vector_norm(
                p_image - torch.matmul(electric_basis, p_sub)
            ) / torch.maximum(torch.linalg.vector_norm(p_image), tiny)
            q_residual = torch.linalg.vector_norm(
                q_image - torch.matmul(magnetic_basis, q_sub)
            ) / torch.maximum(torch.linalg.vector_norm(q_image), tiny)
            residuals.append(torch.maximum(p_residual.real, q_residual.real))
            projected.append((electric_basis, p_sub, q_sub))

        maximum_residual = torch.max(torch.stack(residuals))
        effective_tolerance = max(
            self.group_theory_tolerance,
            2.0e-5 if self._dtype == torch.complex64 else 1.0e-10,
        )
        if _as_float(maximum_residual) > effective_tolerance:
            self._group_theory_failure(
                layer_index,
                symmetry,
                "projected operators are not invariant "
                f"(relative residual {_as_float(maximum_residual):.3e})",
            )
            return None

        kz_parts: list[torch.Tensor] = []
        electric_parts: list[torch.Tensor] = []
        try:
            for electric_basis, p_sub, q_sub in projected:
                kz_squared, electric_sub = self._eig(torch.matmul(p_sub, q_sub))
                kz_parts.append(self._positive_kz(kz_squared))
                electric_parts.append(torch.matmul(electric_basis, electric_sub))
        except RuntimeError as error:
            self._group_theory_failure(
                layer_index, symmetry, f"block eigensolve failed: {error}"
            )
            return None

        kz = torch.cat(kz_parts)
        electric_modes = torch.cat(electric_parts, dim=1)
        if electric_modes.shape != (2 * self.order_N, 2 * self.order_N):
            self._group_theory_failure(
                layer_index, symmetry, "projected blocks do not form a complete basis"
            )
            return None
        order = torch.argsort(kz.real - 1.0e-6 * kz.imag, descending=True)
        self.group_theory_diagnostics.append(
            {
                "layer": layer_index,
                "requested": True,
                "applied": True,
                "symmetry": symmetry,
                "blocks": tuple(part.shape[0] for part in kz_parts),
                "max_invariance_residual": _as_float(maximum_residual),
                "tolerance": effective_tolerance,
            }
        )
        return kz[order], electric_modes[:, order]

    def _polarization_eigendecomposition(
        self,
        p: torch.Tensor,
        q: torch.Tensor,
        centers: Sequence[tuple[float, float]],
        layer_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Solve the C2v or C2 sector reached by a normal-incidence x/y source."""
        polarization = self.polarization_reduction
        assert polarization in {"x", "y"}
        if abs(_as_float(self.inc_ang)) > 1.0e-8:
            raise UnsupportedCombinationError(
                "Polarization reduction requires normal incidence."
            )
        if len(centers) != 1:
            raise UnsupportedCombinationError(
                "Polarization reduction currently requires one circle per cell."
            )

        orthogonal = abs(self.zeta_deg - 90.0) <= 1.0e-7
        if orthogonal:
            if self.group_theory_symmetry not in {"auto", "c2v"}:
                raise UnsupportedCombinationError(
                    "Orthogonal x/y polarization reduction requires C2v symmetry."
                )
            blocks = self._c2v_group_blocks(centers[0])
            block_index = 2 if polarization == "x" else 3
            symmetry_name = "c2v"
        else:
            if self.group_theory_symmetry not in {"auto", "c2"}:
                raise UnsupportedCombinationError(
                    "General-oblique x/y source reduction requires C2 symmetry."
                )
            blocks = self._c2_group_blocks(centers[0])
            # C2 does not distinguish Cartesian x from y: both zero-order
            # sources occupy the same scalar-even vector sector.  The sector
            # retains their possible cross-polarization coupling.
            block_index = 0
            symmetry_name = "c2-source-sector"
        electric_basis, magnetic_basis = blocks[block_index]
        p_image = torch.matmul(p, magnetic_basis)
        q_image = torch.matmul(q, electric_basis)
        p_sub = torch.matmul(electric_basis.mH, p_image)
        q_sub = torch.matmul(magnetic_basis.mH, q_image)
        real_dtype = p_image.real.dtype
        tiny = torch.as_tensor(
            torch.finfo(real_dtype).tiny,
            dtype=real_dtype,
            device=self._device,
        )
        p_residual = torch.linalg.vector_norm(
            p_image - torch.matmul(electric_basis, p_sub)
        ) / torch.maximum(torch.linalg.vector_norm(p_image), tiny)
        q_residual = torch.linalg.vector_norm(
            q_image - torch.matmul(magnetic_basis, q_sub)
        ) / torch.maximum(torch.linalg.vector_norm(q_image), tiny)
        maximum_residual = torch.maximum(p_residual.real, q_residual.real)
        effective_tolerance = max(
            self.group_theory_tolerance,
            2.0e-5 if self._dtype == torch.complex64 else 1.0e-10,
        )
        if _as_float(maximum_residual) > effective_tolerance:
            raise UnsupportedCombinationError(
                "Polarization sector is not invariant: relative residual "
                f"{_as_float(maximum_residual):.3e}."
            )

        kz_squared, electric_sub = self._eig(torch.matmul(p_sub, q_sub))
        kz = self._positive_kz(kz_squared)
        electric_modes = torch.matmul(electric_basis, electric_sub)
        magnetic_modes = self._magnetic_eigenvectors(
            p, q, electric_modes, kz
        )
        magnetic_sub = torch.matmul(magnetic_basis.mH, magnetic_modes)
        magnetic_residual = torch.linalg.vector_norm(
            magnetic_modes - torch.matmul(magnetic_basis, magnetic_sub)
        ) / torch.maximum(torch.linalg.vector_norm(magnetic_modes), tiny)
        if _as_float(magnetic_residual) > effective_tolerance:
            raise UnsupportedCombinationError(
                "Reduced magnetic modes left the paired symmetry sector: residual "
                f"{_as_float(magnetic_residual):.3e}."
            )

        order = torch.argsort(kz.real - 1.0e-6 * kz.imag, descending=True)
        kz = kz[order]
        electric_modes = electric_modes[:, order]
        magnetic_modes = magnetic_modes[:, order]
        electric_sub = torch.matmul(electric_basis.mH, electric_modes)
        magnetic_sub = torch.matmul(magnetic_basis.mH, magnetic_modes)

        if self._polarization_bases is None:
            self._polarization_bases = (electric_basis, magnetic_basis)
        else:
            old_electric, old_magnetic = self._polarization_bases
            basis_error = torch.maximum(
                torch.max(torch.abs(old_electric - electric_basis)),
                torch.max(torch.abs(old_magnetic - magnetic_basis)),
            )
            if _as_float(basis_error) > effective_tolerance:
                raise UnsupportedCombinationError(
                    "Every polarization-reduced layer must share the same "
                    "symmetry center."
                )

        self._polarized_layers.append(
            {
                "electric": electric_sub,
                "magnetic": magnetic_sub,
                "kz": kz,
            }
        )
        self.group_theory_diagnostics.append(
            {
                "layer": layer_index,
                "requested": True,
                "applied": True,
                "symmetry": symmetry_name,
                "backend": "NVM",
                "polarization": polarization,
                "selected_block": block_index,
                "reduced_dimension": electric_basis.shape[1],
                "full_dimension": 2 * self.order_N,
                "max_invariance_residual": _as_float(maximum_residual),
                "magnetic_residual": _as_float(magnetic_residual),
                "tolerance": effective_tolerance,
            }
        )
        return kz, electric_modes, magnetic_modes

    @staticmethod
    def _component_transform(
        component_matrix: torch.Tensor, scalar_matrix: torch.Tensor
    ) -> torch.Tensor:
        """Kronecker component/spatial action in component-major ordering."""
        return torch.cat(
            tuple(
                torch.cat(
                    tuple(component_matrix[i, j] * scalar_matrix for j in range(2)),
                    dim=1,
                )
                for i in range(2)
            ),
            dim=0,
        )

    def _involution_eigenspace(
        self, operator: torch.Tensor, parity: int
    ) -> torch.Tensor:
        """Orthonormal range basis of (I + parity*R)/2 for R^2=I."""
        projector = 0.5 * (
            self._eye(operator.shape[0]) + float(parity) * operator
        )
        left, singular_values, _ = torch.linalg.svd(projector)
        threshold = (
            2.0e-5 if self._dtype == torch.complex64 else 2.0e-11
        ) * torch.max(singular_values)
        rank = int(torch.count_nonzero(singular_values > threshold))
        return left[:, :rank]

    def _triangular_star_operators(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """D6-star embedding and x-axis mirror operators.

        The retained scalar set max(|m|,|n|,|m-n|)<=M is closed under D6.
        Reflection in the local physical x axis maps (m,n)->(m,m-n).  The
        (-1)^m phase places the mirror through the cell-centred cylinder.
        """
        order_x = [int(value) for value in self.order_x.detach().cpu().tolist()]
        order_y = [int(value) for value in self.order_y.detach().cpu().tolist()]
        order_bound = min(max(abs(value) for value in order_x), max(abs(value) for value in order_y))
        all_pairs = [(m, n) for m in order_x for n in order_y]
        full_index = {pair: position for position, pair in enumerate(all_pairs)}
        star_pairs = [
            pair
            for pair in all_pairs
            if max(abs(pair[0]), abs(pair[1]), abs(pair[0] - pair[1]))
            <= order_bound
        ]
        star_index = {pair: position for position, pair in enumerate(star_pairs)}
        if not star_pairs or any((m, m - n) not in star_index for m, n in star_pairs):
            raise UnsupportedCombinationError(
                "The Fourier orders do not contain a mirror-closed triangular star."
            )
        scalar_embedding = torch.zeros(
            (self.order_N, len(star_pairs)),
            dtype=self._dtype,
            device=self._device,
        )
        scalar_reflection = torch.zeros(
            (len(star_pairs), len(star_pairs)),
            dtype=self._dtype,
            device=self._device,
        )
        for column, (m, n) in enumerate(star_pairs):
            scalar_embedding[full_index[(m, n)], column] = 1.0
            row = star_index[(m, m - n)]
            scalar_reflection[row, column] = -1.0 if m % 2 else 1.0

        vector_embedding = self._component_basis(scalar_embedding, scalar_embedding)
        covariant_polar = torch.tensor(
            [[1.0, 0.0], [1.0, -1.0]],
            dtype=self._dtype,
            device=self._device,
        )
        cartesian_polar = torch.tensor(
            [[1.0, 0.0], [0.0, -1.0]],
            dtype=self._dtype,
            device=self._device,
        )
        uv_electric = self._component_transform(
            covariant_polar, scalar_reflection
        )
        uv_magnetic = self._component_transform(
            -covariant_polar, scalar_reflection
        )
        cartesian_electric = self._component_transform(
            cartesian_polar, scalar_reflection
        )
        cartesian_magnetic = self._component_transform(
            -cartesian_polar, scalar_reflection
        )
        return (
            vector_embedding,
            uv_electric,
            uv_magnetic,
            cartesian_electric,
            cartesian_magnetic,
        )

    def _triangular_star_group_operators(
        self,
    ) -> tuple[torch.Tensor, list[dict[str, object]]]:
        """Return the native star embedding and all twelve Cartesian D6 actions."""
        vector_embedding, _, _, _, _ = self._triangular_star_operators()
        order_x = [int(value) for value in self.order_x.detach().cpu().tolist()]
        order_y = [int(value) for value in self.order_y.detach().cpu().tolist()]
        order_bound = min(
            max(abs(value) for value in order_x),
            max(abs(value) for value in order_y),
        )
        star_pairs = [
            (m, n)
            for m in order_x
            for n in order_y
            if max(abs(m), abs(n), abs(m - n)) <= order_bound
        ]
        star_index = {pair: position for position, pair in enumerate(star_pairs)}

        def rotate(pair: tuple[int, int]) -> tuple[int, int]:
            m, n = pair
            return m - n, m

        def mirror(pair: tuple[int, int]) -> tuple[int, int]:
            m, n = pair
            return m, m - n

        entries: list[dict[str, object]] = []
        for reflected in (False, True):
            for power in range(6):
                scalar = torch.zeros(
                    (len(star_pairs), len(star_pairs)),
                    dtype=self._dtype,
                    device=self._device,
                )
                inverse_index = torch.empty(
                    len(star_pairs), dtype=torch.long, device=self._device
                )
                source_phase = torch.empty(
                    len(star_pairs), dtype=self._dtype, device=self._device
                )
                for column, pair in enumerate(star_pairs):
                    mapped = mirror(pair) if reflected else pair
                    for _ in range(power):
                        mapped = rotate(mapped)
                    if mapped not in star_index:
                        raise UnsupportedCombinationError(
                            "The native triangular star is not closed under D6."
                        )
                    exponent = pair[0] + pair[1] - mapped[0] - mapped[1]
                    phase = -1.0 if exponent % 2 else 1.0
                    mapped_row = star_index[mapped]
                    scalar[mapped_row, column] = phase
                    inverse_index[mapped_row] = column
                    source_phase[column] = phase

                angle = power * math.pi / 3.0
                cosine, sine = math.cos(angle), math.sin(angle)
                rotation = torch.tensor(
                    [[cosine, -sine], [sine, cosine]],
                    dtype=self._dtype,
                    device=self._device,
                )
                if reflected:
                    reflection = torch.tensor(
                        [[1.0, 0.0], [0.0, -1.0]],
                        dtype=self._dtype,
                        device=self._device,
                    )
                    component = torch.matmul(rotation, reflection)
                    determinant = -1.0
                else:
                    component = rotation
                    determinant = 1.0
                entries.append(
                    {
                        "rotation_power": power,
                        "reflected": reflected,
                        "scalar": scalar,
                        "scalar_inverse_index": inverse_index,
                        "scalar_phase_at_inverse": source_phase[inverse_index],
                        "component_electric": component,
                        "component_magnetic": determinant * component,
                        "electric": self._component_transform(component, scalar),
                        "magnetic": self._component_transform(
                            determinant * component, scalar
                        ),
                    }
                )
        return vector_embedding, entries

    def _d6_transform_intertwiner(
        self,
        matrix: torch.Tensor,
        entry: dict[str, object],
        *,
        left: str,
        right: str,
    ) -> torch.Tensor:
        """Apply D_left(g) matrix D_right(g)^H in O(D^2) using permutations."""
        inverse_index = entry["scalar_inverse_index"]
        phase = entry["scalar_phase_at_inverse"]
        left_component = entry[f"component_{left}"]
        right_component = entry[f"component_{right}"]
        assert isinstance(inverse_index, torch.Tensor)
        assert isinstance(phase, torch.Tensor)
        assert isinstance(left_component, torch.Tensor)
        assert isinstance(right_component, torch.Tensor)
        scalar_dimension = inverse_index.shape[0]
        blocks = matrix.reshape(
            2, scalar_dimension, 2, scalar_dimension
        ).permute(0, 2, 1, 3)
        blocks = blocks.index_select(2, inverse_index).index_select(
            3, inverse_index
        )
        blocks = (
            blocks
            * phase[None, None, :, None]
            * torch.conj(phase)[None, None, None, :]
        )
        transformed = torch.einsum(
            "ia,abxy,jb->ijxy",
            left_component,
            blocks,
            torch.conj(right_component),
        )
        return transformed.permute(0, 2, 1, 3).reshape(matrix.shape)

    @staticmethod
    def _d6_character_table() -> dict[str, tuple[int, tuple[complex, ...]]]:
        """Characters ordered as r^0..r^5, s,r s,..,r^5 s."""
        table: dict[str, tuple[int, tuple[complex, ...]]] = {}
        for label, rotation_sign, mirror_sign in (
            ("A1", 1, 1),
            ("A2", 1, -1),
            ("B1", -1, 1),
            ("B2", -1, -1),
        ):
            rotations = tuple(complex(rotation_sign**power) for power in range(6))
            reflections = tuple(
                complex((rotation_sign**power) * mirror_sign)
                for power in range(6)
            )
            table[label] = (1, rotations + reflections)
        for harmonic in (1, 2):
            rotations = tuple(
                complex(2.0 * math.cos(harmonic * power * math.pi / 3.0))
                for power in range(6)
            )
            table[f"E{harmonic}"] = (2, rotations + (0.0j,) * 6)
        return table

    def _d6_irrep_matrices(
        self,
    ) -> dict[str, tuple[torch.Tensor, ...]]:
        """Unitary D6 irrep matrices in the same group order as the actions."""
        matrices: dict[str, tuple[torch.Tensor, ...]] = {}
        for label, rotation_sign, mirror_sign in (
            ("A1", 1, 1),
            ("A2", 1, -1),
            ("B1", -1, 1),
            ("B2", -1, -1),
        ):
            values = [
                [[complex(rotation_sign**power)]] for power in range(6)
            ]
            values.extend(
                [
                    [[complex((rotation_sign**power) * mirror_sign)]]
                    for power in range(6)
                ]
            )
            matrices[label] = tuple(
                torch.tensor(value, dtype=self._dtype, device=self._device)
                for value in values
            )
        reflection = torch.tensor(
            [[1.0, 0.0], [0.0, -1.0]],
            dtype=self._dtype,
            device=self._device,
        )
        for harmonic in (1, 2):
            rotations: list[torch.Tensor] = []
            for power in range(6):
                angle = harmonic * power * math.pi / 3.0
                rotations.append(
                    torch.tensor(
                        [
                            [math.cos(angle), -math.sin(angle)],
                            [math.sin(angle), math.cos(angle)],
                        ],
                        dtype=self._dtype,
                        device=self._device,
                    )
                )
            matrices[f"E{harmonic}"] = tuple(
                rotations + [rotation @ reflection for rotation in rotations]
            )
        return matrices

    def _d6_matrix_unit(
        self,
        operators: list[dict[str, object]],
        irrep_matrices: tuple[torch.Tensor, ...],
        *,
        representation: str,
        row: int,
        column: int,
    ) -> torch.Tensor:
        """Return P^alpha_row,column for an electric or magnetic action."""
        dimension = irrep_matrices[0].shape[0]
        first = operators[0][representation]
        assert isinstance(first, torch.Tensor)
        matrix_unit = torch.zeros_like(first)
        for irrep, entry in zip(irrep_matrices, operators):
            action = entry[representation]
            assert isinstance(action, torch.Tensor)
            coefficient = (
                dimension
                * torch.conj(irrep[row, column])
                / len(operators)
            )
            matrix_unit = matrix_unit + coefficient * action
        return matrix_unit

    def _projector_range(self, projector: torch.Tensor) -> torch.Tensor:
        hermitian = 0.5 * (projector + projector.mH)
        threshold = 5.0e-5 if self._dtype == torch.complex64 else 5.0e-10
        diagonal = torch.real(torch.diagonal(hermitian)).clone()
        columns: list[torch.Tensor] = []
        for _ in range(projector.shape[0]):
            pivot_value, pivot = torch.max(diagonal, dim=0)
            if _as_float(pivot_value) <= threshold:
                break
            vector = hermitian[:, pivot]
            if columns:
                basis = torch.stack(columns, dim=1)
                vector = vector - basis @ (basis.mH @ vector)
                # A second pass controls loss of orthogonality for large stars.
                vector = vector - basis @ (basis.mH @ vector)
            norm = torch.linalg.vector_norm(vector)
            if _as_float(norm) <= threshold:
                diagonal[pivot] = 0.0
                continue
            vector = vector / norm
            columns.append(vector)
            diagonal = torch.clamp(diagonal - torch.abs(vector) ** 2, min=0.0)
        if not columns:
            return torch.empty(
                (projector.shape[0], 0),
                dtype=self._dtype,
                device=self._device,
            )
        basis = torch.stack(columns, dim=1)
        orthogonality = torch.max(
            torch.abs(basis.mH @ basis - self._eye(basis.shape[1]))
        )
        range_residual = torch.linalg.vector_norm(
            hermitian @ basis - basis
        ) / torch.maximum(
            torch.linalg.vector_norm(basis),
            torch.as_tensor(
                torch.finfo(hermitian.real.dtype).tiny,
                dtype=hermitian.real.dtype,
                device=self._device,
            ),
        )
        if _as_float(torch.maximum(orthogonality.real, range_residual.real)) > threshold:
            raise UnsupportedCombinationError(
                "D6 projector range is not orthonormal."
            )
        return basis

    def _d6_symmetrized_cartesian_pq(
        self,
        p_star: torch.Tensor,
        q_star: torch.Tensor,
        transform_star: torch.Tensor,
    ):
        """Return fully Reynolds-averaged Cartesian operators on the D6 star."""
        if getattr(self, "lattice_kind", "rectangular") != "triangular":
            raise UnsupportedCombinationError(
                "D6 decomposition requires a 60-degree equal-length triangular cell."
            )
        if abs(_as_float(self.inc_ang)) > 1.0e-8:
            raise UnsupportedCombinationError(
                "D6 decomposition requires normal incidence."
            )
        vector_embedding, group = self._triangular_star_group_operators()
        star_dimension = vector_embedding.shape[1]
        transform_inverse = self._solve(
            transform_star, self._eye(star_dimension)
        )
        p_cart = transform_star @ p_star @ transform_inverse
        q_cart = transform_star @ q_star @ transform_inverse

        p_symmetric = torch.zeros_like(p_cart)
        q_symmetric = torch.zeros_like(q_cart)
        for entry in group:
            p_symmetric = p_symmetric + self._d6_transform_intertwiner(
                p_cart, entry, left="electric", right="magnetic"
            )
            q_symmetric = q_symmetric + self._d6_transform_intertwiner(
                q_cart, entry, left="magnetic", right="electric"
            )
        p_symmetric = p_symmetric / len(group)
        q_symmetric = q_symmetric / len(group)
        tiny = torch.as_tensor(
            torch.finfo(p_cart.real.dtype).tiny,
            dtype=p_cart.real.dtype,
            device=self._device,
        )
        symmetry_correction = torch.maximum(
            torch.linalg.vector_norm(p_symmetric - p_cart)
            / torch.maximum(torch.linalg.vector_norm(p_cart), tiny),
            torch.linalg.vector_norm(q_symmetric - q_cart)
            / torch.maximum(torch.linalg.vector_norm(q_cart), tiny),
        ).real
        tolerance = max(
            self.group_theory_tolerance,
            5.0e-5 if self._dtype == torch.complex64 else 5.0e-9,
        )
        return (
            vector_embedding,
            group,
            transform_inverse,
            p_symmetric,
            q_symmetric,
            symmetry_correction,
            tiny,
            tolerance,
        )

    def _complete_d6_eigendecomposition(
        self,
        p_star: torch.Tensor,
        q_star: torch.Tensor,
        transform_star: torch.Tensor,
        *,
        layer_index: int,
        backend: str,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Solve all D6 irreps, using matrix-unit rows for E1/E2."""
        (
            vector_embedding,
            group,
            transform_inverse,
            p_symmetric,
            q_symmetric,
            symmetry_correction,
            tiny,
            tolerance,
        ) = self._d6_symmetrized_cartesian_pq(
            p_star, q_star, transform_star
        )
        star_dimension = vector_embedding.shape[1]
        identity = self._eye(star_dimension)

        electric_projector_sum = torch.zeros_like(p_symmetric)
        magnetic_projector_sum = torch.zeros_like(q_symmetric)
        kz_parts: list[torch.Tensor] = []
        electric_parts: list[torch.Tensor] = []
        magnetic_parts: list[torch.Tensor] = []
        block_diagnostics: list[dict[str, object]] = []
        character_table = self._d6_character_table()
        matrix_table = self._d6_irrep_matrices()
        for label in ("A1", "A2", "B1", "B2", "E1", "E2"):
            irrep_dimension, characters = character_table[label]
            electric_projector = torch.zeros_like(p_symmetric)
            magnetic_projector = torch.zeros_like(q_symmetric)
            for character, entry in zip(characters, group):
                electric = entry["electric"]
                magnetic = entry["magnetic"]
                assert isinstance(electric, torch.Tensor)
                assert isinstance(magnetic, torch.Tensor)
                coefficient = irrep_dimension * character.conjugate() / len(group)
                electric_projector = electric_projector + coefficient * electric
                magnetic_projector = magnetic_projector + coefficient * magnetic
            electric_projector = 0.5 * (
                electric_projector + electric_projector.mH
            )
            magnetic_projector = 0.5 * (
                magnetic_projector + magnetic_projector.mH
            )
            electric_projector_sum = electric_projector_sum + electric_projector
            magnetic_projector_sum = magnetic_projector_sum + magnetic_projector
            electric_isotypic_dimension = int(
                round(_as_float(torch.trace(electric_projector).real))
            )
            magnetic_isotypic_dimension = int(
                round(_as_float(torch.trace(magnetic_projector).real))
            )
            if electric_isotypic_dimension != magnetic_isotypic_dimension:
                raise UnsupportedCombinationError(
                    f"D6 {label} electric/magnetic isotypic dimensions disagree."
                )
            if electric_isotypic_dimension == 0:
                block_diagnostics.append(
                    {
                        "irrep": label,
                        "irrep_dimension": irrep_dimension,
                        "isotypic_dimension": 0,
                        "solved_dimension": 0,
                        "multiplicity": 0,
                    }
                )
                continue
            irrep_matrices = matrix_table[label]
            if irrep_dimension == 1:
                electric_solve_projector = electric_projector
                magnetic_solve_projector = magnetic_projector
                electric_transfer = None
                magnetic_transfer = None
            else:
                # Schur's lemma gives I_d tensor A on an isotypic component.
                # Solve one matrix-unit row only, then generate the degenerate
                # partner with P_10.  This avoids passing a symmetry-enforced
                # repeated spectrum to torch.linalg.eig/backward.
                electric_solve_projector = self._d6_matrix_unit(
                    group,
                    irrep_matrices,
                    representation="electric",
                    row=0,
                    column=0,
                )
                magnetic_solve_projector = self._d6_matrix_unit(
                    group,
                    irrep_matrices,
                    representation="magnetic",
                    row=0,
                    column=0,
                )
                electric_transfer = self._d6_matrix_unit(
                    group,
                    irrep_matrices,
                    representation="electric",
                    row=1,
                    column=0,
                )
                magnetic_transfer = self._d6_matrix_unit(
                    group,
                    irrep_matrices,
                    representation="magnetic",
                    row=1,
                    column=0,
                )
            electric_solve_projector = 0.5 * (
                electric_solve_projector + electric_solve_projector.mH
            )
            magnetic_solve_projector = 0.5 * (
                magnetic_solve_projector + magnetic_solve_projector.mH
            )
            electric_basis = self._projector_range(electric_solve_projector)
            magnetic_basis = self._projector_range(magnetic_solve_projector)
            if electric_basis.shape[1] != magnetic_basis.shape[1]:
                raise UnsupportedCombinationError(
                    f"D6 {label} electric/magnetic multiplicities disagree."
                )
            solved_dimension = electric_basis.shape[1]
            if solved_dimension * irrep_dimension != electric_isotypic_dimension:
                raise UnsupportedCombinationError(
                    f"D6 {label} matrix-unit rank does not reproduce its isotypic dimension."
                )
            p_image = p_symmetric @ magnetic_basis
            q_image = q_symmetric @ electric_basis
            p_sub = electric_basis.mH @ p_image
            q_sub = magnetic_basis.mH @ q_image
            p_residual = torch.linalg.vector_norm(
                p_image - electric_basis @ p_sub
            ) / torch.maximum(torch.linalg.vector_norm(p_image), tiny)
            q_residual = torch.linalg.vector_norm(
                q_image - magnetic_basis @ q_sub
            ) / torch.maximum(torch.linalg.vector_norm(q_image), tiny)
            residual = torch.maximum(p_residual.real, q_residual.real)
            if _as_float(residual) > tolerance:
                raise UnsupportedCombinationError(
                    f"D6 {label} block is not invariant: {_as_float(residual):.3e}."
                )
            kz_squared, electric_sub = self._eig(p_sub @ q_sub)
            kz = self._positive_kz(kz_squared)
            electric_modes = electric_basis @ electric_sub
            magnetic_modes = self._magnetic_eigenvectors(
                p_symmetric, q_symmetric, electric_modes, kz
            )
            magnetic_sub = magnetic_basis.mH @ magnetic_modes
            magnetic_residual = torch.linalg.vector_norm(
                magnetic_modes - magnetic_basis @ magnetic_sub
            ) / torch.maximum(torch.linalg.vector_norm(magnetic_modes), tiny)
            if _as_float(magnetic_residual) > tolerance:
                raise UnsupportedCombinationError(
                    f"D6 {label} magnetic modes left their matrix-unit row."
                )
            partner_residual = torch.zeros(
                (), dtype=p_symmetric.real.dtype, device=self._device
            )
            if irrep_dimension == 2:
                assert electric_transfer is not None
                assert magnetic_transfer is not None
                electric_partner = electric_transfer @ electric_modes
                magnetic_partner = magnetic_transfer @ magnetic_modes
                magnetic_partner_from_maxwell = self._magnetic_eigenvectors(
                    p_symmetric, q_symmetric, electric_partner, kz
                )
                partner_residual = torch.linalg.vector_norm(
                    magnetic_partner - magnetic_partner_from_maxwell
                ) / torch.maximum(
                    torch.linalg.vector_norm(magnetic_partner_from_maxwell), tiny
                )
                if _as_float(partner_residual) > tolerance:
                    raise UnsupportedCombinationError(
                        f"D6 {label} matrix-unit partner violates Maxwell intertwining."
                    )
                kz = torch.cat((kz, kz))
                electric_modes = torch.cat(
                    (electric_modes, electric_partner), dim=1
                )
                magnetic_modes = torch.cat(
                    (magnetic_modes, magnetic_partner_from_maxwell), dim=1
                )
            kz_parts.append(kz)
            electric_parts.append(electric_modes)
            magnetic_parts.append(magnetic_modes)
            block_diagnostics.append(
                {
                    "irrep": label,
                    "irrep_dimension": irrep_dimension,
                    "isotypic_dimension": electric_isotypic_dimension,
                    "solved_dimension": solved_dimension,
                    "multiplicity": solved_dimension,
                    "max_invariance_residual": _as_float(residual),
                    "magnetic_residual": _as_float(magnetic_residual),
                    "matrix_unit_partner_residual": _as_float(partner_residual),
                }
            )

        completeness = torch.maximum(
            torch.max(torch.abs(electric_projector_sum - identity)),
            torch.max(torch.abs(magnetic_projector_sum - identity)),
        )
        if _as_float(completeness) > tolerance:
            raise UnsupportedCombinationError(
                "D6 character projectors do not resolve the native star."
            )
        kz = torch.cat(kz_parts)
        electric_modes = torch.cat(electric_parts, dim=1)
        magnetic_modes = torch.cat(magnetic_parts, dim=1)
        if electric_modes.shape != (star_dimension, star_dimension):
            raise UnsupportedCombinationError(
                "D6 isotypic dimensions do not sum to the native-star dimension."
            )
        order = torch.argsort(kz.real - 1.0e-6 * kz.imag, descending=True)
        kz = kz[order]
        electric_modes = electric_modes[:, order]
        magnetic_modes = magnetic_modes[:, order]
        self.group_theory_diagnostics.append(
            {
                "layer": layer_index,
                "requested": True,
                "applied": True,
                "symmetry": "D6-complete-native-star",
                "backend": backend,
                "irreps": tuple(block_diagnostics),
                "irrep_order": ("A1", "A2", "B1", "B2", "E1", "E2"),
                "total_solved_eigen_dimension": sum(
                    int(record["solved_dimension"])
                    for record in block_diagnostics
                ),
                "dense_eigen_cubic_work_ratio": sum(
                    int(record["solved_dimension"]) ** 3
                    for record in block_diagnostics
                )
                / star_dimension**3,
                "star_vector_dimension": star_dimension,
                "full_dimension": 2 * self.order_N,
                "projector_completeness": _as_float(completeness),
                "operator_symmetrization": _as_float(symmetry_correction),
                "tolerance": tolerance,
            }
        )
        return (
            kz,
            electric_modes,
            magnetic_modes,
            vector_embedding,
            transform_inverse,
        )

    def _d6_source_eigendecomposition(
        self,
        p_star: torch.Tensor,
        q_star: torch.Tensor,
        transform_star: torch.Tensor,
        *,
        layer_index: int,
        backend: str,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Solve only the E1 matrix-unit row reached by a normal x/y source."""
        polarization = self.polarization_reduction
        if polarization not in {"x", "y"}:
            raise UnsupportedCombinationError(
                "D6 source-row reduction requires polarization='x' or 'y'."
            )
        (
            vector_embedding,
            group,
            transform_inverse,
            p_symmetric,
            q_symmetric,
            symmetry_correction,
            tiny,
            tolerance,
        ) = self._d6_symmetrized_cartesian_pq(
            p_star, q_star, transform_star
        )
        star_dimension = vector_embedding.shape[1]
        row = 0 if polarization == "x" else 1
        e1_matrices = self._d6_irrep_matrices()["E1"]
        electric_projector = self._d6_matrix_unit(
            group,
            e1_matrices,
            representation="electric",
            row=row,
            column=row,
        )
        magnetic_projector = self._d6_matrix_unit(
            group,
            e1_matrices,
            representation="magnetic",
            row=row,
            column=row,
        )
        electric_projector = 0.5 * (
            electric_projector + electric_projector.mH
        )
        magnetic_projector = 0.5 * (
            magnetic_projector + magnetic_projector.mH
        )
        electric_basis_star = self._projector_range(electric_projector)
        magnetic_basis_star = self._projector_range(magnetic_projector)
        if electric_basis_star.shape[1] != magnetic_basis_star.shape[1]:
            raise UnsupportedCombinationError(
                "D6 E1 electric/magnetic source-row multiplicities disagree."
            )
        reduced_dimension = electric_basis_star.shape[1]
        if reduced_dimension == 0:
            raise UnsupportedCombinationError("The selected D6 E1 source row is empty.")

        p_image = p_symmetric @ magnetic_basis_star
        q_image = q_symmetric @ electric_basis_star
        p_sub = electric_basis_star.mH @ p_image
        q_sub = magnetic_basis_star.mH @ q_image
        p_residual = torch.linalg.vector_norm(
            p_image - electric_basis_star @ p_sub
        ) / torch.maximum(torch.linalg.vector_norm(p_image), tiny)
        q_residual = torch.linalg.vector_norm(
            q_image - magnetic_basis_star @ q_sub
        ) / torch.maximum(torch.linalg.vector_norm(q_image), tiny)
        invariance_residual = torch.maximum(p_residual.real, q_residual.real)
        if _as_float(invariance_residual) > tolerance:
            raise UnsupportedCombinationError(
                "D6 E1 source row is not invariant: relative residual "
                f"{_as_float(invariance_residual):.3e}."
            )

        kz_squared, electric_sub = self._eig(p_sub @ q_sub)
        kz = self._positive_kz(kz_squared)
        electric_modes_star = electric_basis_star @ electric_sub
        magnetic_modes_star = self._magnetic_eigenvectors(
            p_symmetric,
            q_symmetric,
            electric_modes_star,
            kz,
        )
        magnetic_sub = magnetic_basis_star.mH @ magnetic_modes_star
        magnetic_residual = torch.linalg.vector_norm(
            magnetic_modes_star - magnetic_basis_star @ magnetic_sub
        ) / torch.maximum(torch.linalg.vector_norm(magnetic_modes_star), tiny)
        if _as_float(magnetic_residual) > tolerance:
            raise UnsupportedCombinationError(
                "D6 E1 magnetic modes left the selected source row."
            )

        order = torch.argsort(kz.real - 1.0e-6 * kz.imag, descending=True)
        kz = kz[order]
        electric_modes_star = electric_modes_star[:, order]
        magnetic_modes_star = magnetic_modes_star[:, order]
        electric_basis = vector_embedding @ electric_basis_star
        magnetic_basis = vector_embedding @ magnetic_basis_star

        zero_x = int(torch.nonzero(self.order_x == 0, as_tuple=False)[0, 0])
        zero_y = int(torch.nonzero(self.order_y == 0, as_tuple=False)[0, 0])
        harmonic = zero_x * len(self.order_y) + zero_y
        source = torch.zeros(
            2 * self.order_N, dtype=self._dtype, device=self._device
        )
        source[harmonic if polarization == "x" else self.order_N + harmonic] = 1.0
        source_residual = torch.linalg.vector_norm(
            source - electric_basis @ (electric_basis.mH @ source)
        ) / torch.maximum(torch.linalg.vector_norm(source), tiny)
        if _as_float(source_residual) > tolerance:
            raise UnsupportedCombinationError(
                "The requested zero-order source is outside its D6 E1 matrix-unit row."
            )

        order_x = [int(value) for value in self.order_x.detach().cpu().tolist()]
        order_y = [int(value) for value in self.order_y.detach().cpu().tolist()]
        order_bound = min(
            max(abs(value) for value in order_x),
            max(abs(value) for value in order_y),
        )
        expected_dimension = order_bound * (order_bound + 1) + 1
        if reduced_dimension != expected_dimension:
            raise UnsupportedCombinationError(
                "Unexpected D6 E1 source-row dimension: "
                f"{reduced_dimension}, expected {expected_dimension}."
            )
        # Mutate the cascade state only after every source-row invariant has
        # passed.  This keeps a rejected layer from leaving partial state.
        self._install_polarization_basis(
            electric_basis, magnetic_basis, tolerance
        )
        self._polarized_layers.append(
            {
                "electric": electric_basis_star.mH @ electric_modes_star,
                "magnetic": magnetic_basis_star.mH @ magnetic_modes_star,
                "kz": kz,
            }
        )
        full_dimension = 2 * self.order_N
        self.group_theory_diagnostics.append(
            {
                "layer": layer_index,
                "requested": True,
                "applied": True,
                "symmetry": "D6-E1-source-row",
                "backend": backend,
                "polarization": polarization,
                "irrep": "E1",
                "matrix_unit_row": row,
                "reduced_dimension": reduced_dimension,
                "expected_dimension": expected_dimension,
                "star_vector_dimension": star_dimension,
                "full_dimension": full_dimension,
                "max_invariance_residual": _as_float(invariance_residual),
                "magnetic_residual": _as_float(magnetic_residual),
                "source_projection_residual": _as_float(source_residual),
                "operator_symmetrization": _as_float(symmetry_correction),
                "dense_eigen_cubic_work_ratio_vs_star": (
                    reduced_dimension / star_dimension
                ) ** 3,
                "dense_eigen_cubic_work_ratio_vs_rectangular": (
                    reduced_dimension / full_dimension
                ) ** 3,
                "tolerance": tolerance,
            }
        )
        return (
            kz,
            electric_modes_star,
            magnetic_modes_star,
            vector_embedding,
            transform_inverse,
        )

    def _register_complete_d6_layer(
        self,
        vector_embedding: torch.Tensor,
        electric_modes: torch.Tensor,
        magnetic_modes: torch.Tensor,
        kz: torch.Tensor,
    ) -> None:
        """Register one all-irrep native-star layer for reduced-basis cascade."""
        tolerance = max(
            self.group_theory_tolerance,
            5.0e-5 if self._dtype == torch.complex64 else 5.0e-9,
        )
        self._install_polarization_basis(
            vector_embedding, vector_embedding, tolerance
        )
        self._polarized_layers.append(
            {
                "electric": electric_modes,
                "magnetic": magnetic_modes,
                "kz": kz,
            }
        )
        self._native_d6_active = True

    def _build_triangular_star_pq(
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Assemble P,Q directly on a D6-closed reciprocal-lattice star.

        Crucially, longitudinal Toeplitz inverses are taken *after* star
        restriction.  Restricting an inverse computed on a rectangular order
        box would re-introduce the non-D6 corner harmonics and destroy exact
        mirror-sector invariance.  The optional transverse rule is a
        coordinate-covariant 2x2 block inverse rule on the same star.
        """
        vector_embedding, _, _, _, _ = self._triangular_star_operators()
        star_count = vector_embedding.shape[1] // 2
        scalar_embedding = vector_embedding[: self.order_N, :star_count]

        def convolution(field: torch.Tensor) -> torch.Tensor:
            return torch.matmul(
                scalar_embedding.mH,
                torch.matmul(self._material_conv(field), scalar_embedding),
            )

        def transverse(
            value11: torch.Tensor,
            value12: torch.Tensor,
            value21: torch.Tensor,
            value22: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            if not factorization_rules:
                return tuple(
                    convolution(value)
                    for value in (value11, value12, value21, value22)
                )
            determinant = value11 * value22 - value12 * value21
            inverse11 = value22 / determinant
            inverse12 = -value12 / determinant
            inverse21 = -value21 / determinant
            inverse22 = value11 / determinant
            inverse_block = torch.cat(
                (
                    torch.cat((convolution(inverse11), convolution(inverse12)), dim=1),
                    torch.cat((convolution(inverse21), convolution(inverse22)), dim=1),
                ),
                dim=0,
            )
            effective = self._solve(inverse_block, self._eye(2 * star_count))
            return (
                effective[:star_count, :star_count],
                effective[:star_count, star_count:],
                effective[star_count:, :star_count],
                effective[star_count:, star_count:],
            )

        eps11_m, eps12_m, eps21_m, eps22_m = transverse(
            eps11, eps12, eps21, eps22
        )
        mu11_m, mu12_m, mu21_m, mu22_m = transverse(
            mu11, mu12, mu21, mu22
        )
        eps33_m = convolution(eps33)
        mu33_m = convolution(mu33)
        identity = self._eye(star_count)
        inverse_eps33 = self._solve(eps33_m, identity)
        inverse_mu33 = self._solve(mu33_m, identity)
        k1 = torch.matmul(
            scalar_embedding.mH,
            torch.matmul(self.K1_norm, scalar_embedding),
        )
        k2 = torch.matmul(
            scalar_embedding.mH,
            torch.matmul(self.K2_norm, scalar_embedding),
        )
        p11 = mu21_m + torch.matmul(k1, torch.matmul(inverse_eps33, k2))
        p12 = mu22_m - torch.matmul(k1, torch.matmul(inverse_eps33, k1))
        p21 = torch.matmul(k2, torch.matmul(inverse_eps33, k2)) - mu11_m
        p22 = -mu12_m - torch.matmul(k2, torch.matmul(inverse_eps33, k1))
        p_star = torch.cat(
            (torch.cat((p11, p12), dim=1), torch.cat((p21, p22), dim=1)),
            dim=0,
        )
        q11 = -eps21_m - torch.matmul(k1, torch.matmul(inverse_mu33, k2))
        q12 = torch.matmul(k1, torch.matmul(inverse_mu33, k1)) - eps22_m
        q21 = eps11_m - torch.matmul(k2, torch.matmul(inverse_mu33, k2))
        q22 = eps12_m + torch.matmul(k2, torch.matmul(inverse_mu33, k1))
        q_star = torch.cat(
            (torch.cat((q11, q12), dim=1), torch.cat((q21, q22), dim=1)),
            dim=0,
        )
        return p_star, q_star

    def _install_polarization_basis(
        self,
        electric_basis: torch.Tensor,
        magnetic_basis: torch.Tensor,
        tolerance: float,
    ) -> None:
        if self._polarization_bases is None:
            self._polarization_bases = (electric_basis, magnetic_basis)
            return
        old_electric, old_magnetic = self._polarization_bases
        if old_electric.shape != electric_basis.shape or old_magnetic.shape != magnetic_basis.shape:
            raise UnsupportedCombinationError(
                "Every polarization-reduced layer must use the same Fourier sector."
            )
        electric_projector_error = torch.max(
            torch.abs(
                torch.matmul(old_electric, old_electric.mH)
                - torch.matmul(electric_basis, electric_basis.mH)
            )
        )
        magnetic_projector_error = torch.max(
            torch.abs(
                torch.matmul(old_magnetic, old_magnetic.mH)
                - torch.matmul(magnetic_basis, magnetic_basis.mH)
            )
        )
        if _as_float(torch.maximum(electric_projector_error, magnetic_projector_error)) > tolerance:
            raise UnsupportedCombinationError(
                "Every polarization-reduced layer must share the same symmetry centre."
            )

    def _matched_polarization_eigendecomposition(
        self,
        p: torch.Tensor,
        q: torch.Tensor,
        transform: torch.Tensor,
        *,
        layer_index: int,
        triangular_star_pq: tuple[torch.Tensor, torch.Tensor] | None = None,
        backend: str = "matched-ASR",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Solve one x/y mirror sector for matched-ASR or triangular NVM."""
        polarization = self.polarization_reduction
        assert polarization in {"x", "y"}
        if abs(_as_float(self.inc_ang)) > 1.0e-8:
            raise UnsupportedCombinationError(
                f"{backend} polarization reduction requires normal incidence."
            )
        tolerance = max(
            self.group_theory_tolerance,
            5.0e-5 if self._dtype == torch.complex64 else 5.0e-9,
        )
        tiny = torch.as_tensor(
            torch.finfo(p.real.dtype).tiny,
            dtype=p.real.dtype,
            device=self._device,
        )

        if getattr(self, "lattice_kind", "rectangular") == "triangular":
            (
                vector_embedding,
                reflection_e_uv,
                reflection_h_uv,
                reflection_e_cart,
                reflection_h_cart,
            ) = self._triangular_star_operators()
            if triangular_star_pq is None:
                raise RuntimeError(
                    f"Triangular {backend} polarization reduction needs star P,Q."
                )
            p_star, q_star = triangular_star_pq
            # Directional Fourier factorization and finite sampling can leave
            # roundoff-level mirror asymmetry.  The Reynolds average restores
            # the exact Cs subgroup of the D6-symmetric physical problem.
            p_sym = 0.5 * (
                p_star
                + torch.matmul(
                    reflection_e_uv,
                    torch.matmul(p_star, reflection_h_uv),
                )
            )
            q_sym = 0.5 * (
                q_star
                + torch.matmul(
                    reflection_h_uv,
                    torch.matmul(q_star, reflection_e_uv),
                )
            )
            symmetry_correction = torch.maximum(
                torch.linalg.vector_norm(p_sym - p_star)
                / torch.maximum(torch.linalg.vector_norm(p_star), tiny),
                torch.linalg.vector_norm(q_sym - q_star)
                / torch.maximum(torch.linalg.vector_norm(q_star), tiny),
            ).real
            parity = 1 if polarization == "x" else -1
            electric_uv_basis = self._involution_eigenspace(
                reflection_e_uv, parity
            )
            magnetic_uv_basis = self._involution_eigenspace(
                reflection_h_uv, parity
            )
            electric_cart_basis_star = self._involution_eigenspace(
                reflection_e_cart, parity
            )
            magnetic_cart_basis_star = self._involution_eigenspace(
                reflection_h_cart, parity
            )
            if electric_uv_basis.shape[1] != magnetic_uv_basis.shape[1]:
                raise UnsupportedCombinationError(
                    "Electric and magnetic triangular mirror sectors have unequal size."
                )
            p_image = torch.matmul(p_sym, magnetic_uv_basis)
            q_image = torch.matmul(q_sym, electric_uv_basis)
            p_sub = torch.matmul(electric_uv_basis.mH, p_image)
            q_sub = torch.matmul(magnetic_uv_basis.mH, q_image)
            p_residual = torch.linalg.vector_norm(
                p_image - torch.matmul(electric_uv_basis, p_sub)
            ) / torch.maximum(torch.linalg.vector_norm(p_image), tiny)
            q_residual = torch.linalg.vector_norm(
                q_image - torch.matmul(magnetic_uv_basis, q_sub)
            ) / torch.maximum(torch.linalg.vector_norm(q_image), tiny)
            maximum_residual = torch.maximum(p_residual.real, q_residual.real)
            if _as_float(maximum_residual) > tolerance:
                raise UnsupportedCombinationError(
                    f"Triangular {backend} mirror sector is not invariant: "
                    f"relative residual {_as_float(maximum_residual):.3e}."
                )

            kz_squared, electric_sub = self._eig(torch.matmul(p_sub, q_sub))
            kz = self._positive_kz(kz_squared)
            electric_star = torch.matmul(electric_uv_basis, electric_sub)
            magnetic_star = self._magnetic_eigenvectors(
                p_sym, q_sym, electric_star, kz
            )
            magnetic_sub = torch.matmul(magnetic_uv_basis.mH, magnetic_star)
            magnetic_residual = torch.linalg.vector_norm(
                magnetic_star - torch.matmul(magnetic_uv_basis, magnetic_sub)
            ) / torch.maximum(torch.linalg.vector_norm(magnetic_star), tiny)

            transform_star = torch.matmul(
                vector_embedding.mH,
                torch.matmul(transform, vector_embedding),
            )
            transform_sym = 0.5 * (
                transform_star
                + torch.matmul(
                    reflection_e_cart,
                    torch.matmul(transform_star, reflection_e_uv),
                )
            )
            transform_correction = torch.linalg.vector_norm(
                transform_sym - transform_star
            ) / torch.maximum(torch.linalg.vector_norm(transform_star), tiny)
            electric_cart_star = torch.matmul(transform_sym, electric_star)
            magnetic_cart_star = torch.matmul(transform_sym, magnetic_star)
            electric_cart_sub = torch.matmul(
                electric_cart_basis_star.mH, electric_cart_star
            )
            magnetic_cart_sub = torch.matmul(
                magnetic_cart_basis_star.mH, magnetic_cart_star
            )
            electric_cart_residual = torch.linalg.vector_norm(
                electric_cart_star
                - torch.matmul(electric_cart_basis_star, electric_cart_sub)
            ) / torch.maximum(torch.linalg.vector_norm(electric_cart_star), tiny)
            magnetic_cart_residual = torch.linalg.vector_norm(
                magnetic_cart_star
                - torch.matmul(magnetic_cart_basis_star, magnetic_cart_sub)
            ) / torch.maximum(torch.linalg.vector_norm(magnetic_cart_star), tiny)
            conversion_residual = torch.maximum(
                electric_cart_residual.real, magnetic_cart_residual.real
            )
            if _as_float(torch.maximum(magnetic_residual.real, conversion_residual)) > tolerance:
                raise UnsupportedCombinationError(
                    f"Triangular {backend} modes left the selected mirror sector."
                )

            electric_basis = torch.matmul(
                vector_embedding, electric_cart_basis_star
            )
            magnetic_basis = torch.matmul(
                vector_embedding, magnetic_cart_basis_star
            )
            electric_uv_full = torch.matmul(vector_embedding, electric_star)
            magnetic_uv_full = torch.matmul(vector_embedding, magnetic_star)
            electric_cart_full = torch.matmul(electric_basis, electric_cart_sub)
            magnetic_cart_full = torch.matmul(magnetic_basis, magnetic_cart_sub)
            symmetry_name = "D6-star/Cs(x-mirror)"
            star_dimension = vector_embedding.shape[1]
        else:
            blocks = self._c2v_group_blocks(
                (0.5 * _as_float(self.L[0]), 0.5 * _as_float(self.L[1]))
            )
            block_index = 2 if polarization == "x" else 3
            electric_uv_basis, magnetic_uv_basis = blocks[block_index]
            p_image = torch.matmul(p, magnetic_uv_basis)
            q_image = torch.matmul(q, electric_uv_basis)
            p_sub = torch.matmul(electric_uv_basis.mH, p_image)
            q_sub = torch.matmul(magnetic_uv_basis.mH, q_image)
            p_residual = torch.linalg.vector_norm(
                p_image - torch.matmul(electric_uv_basis, p_sub)
            ) / torch.maximum(torch.linalg.vector_norm(p_image), tiny)
            q_residual = torch.linalg.vector_norm(
                q_image - torch.matmul(magnetic_uv_basis, q_sub)
            ) / torch.maximum(torch.linalg.vector_norm(q_image), tiny)
            maximum_residual = torch.maximum(p_residual.real, q_residual.real)
            if _as_float(maximum_residual) > tolerance:
                raise UnsupportedCombinationError(
                    f"{backend} C2v sector is not invariant: relative residual "
                    f"{_as_float(maximum_residual):.3e}."
                )
            kz_squared, electric_sub = self._eig(torch.matmul(p_sub, q_sub))
            kz = self._positive_kz(kz_squared)
            electric_uv_full = torch.matmul(electric_uv_basis, electric_sub)
            magnetic_uv_full = self._magnetic_eigenvectors(
                p, q, electric_uv_full, kz
            )
            magnetic_sub = torch.matmul(magnetic_uv_basis.mH, magnetic_uv_full)
            magnetic_residual = torch.linalg.vector_norm(
                magnetic_uv_full - torch.matmul(magnetic_uv_basis, magnetic_sub)
            ) / torch.maximum(torch.linalg.vector_norm(magnetic_uv_full), tiny)
            electric_basis, magnetic_basis = electric_uv_basis, magnetic_uv_basis
            electric_cart_raw = torch.matmul(transform, electric_uv_full)
            magnetic_cart_raw = torch.matmul(transform, magnetic_uv_full)
            electric_cart_sub = torch.matmul(electric_basis.mH, electric_cart_raw)
            magnetic_cart_sub = torch.matmul(magnetic_basis.mH, magnetic_cart_raw)
            electric_cart_full = torch.matmul(electric_basis, electric_cart_sub)
            magnetic_cart_full = torch.matmul(magnetic_basis, magnetic_cart_sub)
            conversion_residual = torch.maximum(
                torch.linalg.vector_norm(electric_cart_raw - electric_cart_full)
                / torch.maximum(torch.linalg.vector_norm(electric_cart_raw), tiny),
                torch.linalg.vector_norm(magnetic_cart_raw - magnetic_cart_full)
                / torch.maximum(torch.linalg.vector_norm(magnetic_cart_raw), tiny),
            ).real
            if _as_float(torch.maximum(magnetic_residual.real, conversion_residual)) > tolerance:
                raise UnsupportedCombinationError(
                    f"{backend} conversion left the selected C2v sector."
                )
            symmetry_correction = torch.zeros((), dtype=p.real.dtype, device=self._device)
            transform_correction = torch.zeros((), dtype=p.real.dtype, device=self._device)
            symmetry_name = "C2v"
            star_dimension = 2 * self.order_N

        order = torch.argsort(kz.real - 1.0e-6 * kz.imag, descending=True)
        kz = kz[order]
        electric_uv_full = electric_uv_full[:, order]
        magnetic_uv_full = magnetic_uv_full[:, order]
        electric_cart_full = electric_cart_full[:, order]
        magnetic_cart_full = magnetic_cart_full[:, order]
        electric_cart_sub = torch.matmul(electric_basis.mH, electric_cart_full)
        magnetic_cart_sub = torch.matmul(magnetic_basis.mH, magnetic_cart_full)
        self._install_polarization_basis(electric_basis, magnetic_basis, tolerance)
        self._polarized_layers.append(
            {
                "electric": electric_cart_sub,
                "magnetic": magnetic_cart_sub,
                "kz": kz,
            }
        )
        self.group_theory_diagnostics.append(
            {
                "layer": layer_index,
                "requested": True,
                "applied": True,
                "symmetry": symmetry_name,
                "backend": backend,
                "polarization": polarization,
                "reduced_dimension": electric_basis.shape[1],
                "star_vector_dimension": star_dimension,
                "full_dimension": 2 * self.order_N,
                "max_invariance_residual": _as_float(maximum_residual),
                "magnetic_residual": _as_float(magnetic_residual),
                "conversion_residual": _as_float(conversion_residual),
                "operator_symmetrization": _as_float(symmetry_correction),
                "transform_symmetrization": _as_float(transform_correction),
                "tolerance": tolerance,
            }
        )
        return (
            kz,
            electric_uv_full,
            magnetic_uv_full,
            electric_cart_full,
            magnetic_cart_full,
        )
