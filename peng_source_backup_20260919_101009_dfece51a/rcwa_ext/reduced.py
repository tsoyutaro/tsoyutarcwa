"""Polarization-reduced interfaces and Redheffer/Li-2a cascades."""

from __future__ import annotations

from typing import Sequence

import torch

from .config import UnsupportedCombinationError, _as_float

class _ReducedScatteringMixin:
    """Cascade only the source-accessible symmetry sector."""

    def _reduced_interface_s(
        self,
        reference_v: torch.Tensor,
        medium_v: torch.Tensor,
        *,
        input_side: bool,
    ) -> list[torch.Tensor]:
        identity = self._eye(reference_v.shape[0])
        inverse_sum = self._solve(reference_v + medium_v, identity)
        difference = reference_v - medium_v
        if input_side:
            return [
                2.0 * torch.matmul(inverse_sum, medium_v),
                -torch.matmul(inverse_sum, difference),
                torch.matmul(inverse_sum, difference),
                2.0 * torch.matmul(inverse_sum, reference_v),
            ]
        return [
            2.0 * torch.matmul(inverse_sum, reference_v),
            torch.matmul(inverse_sum, difference),
            -torch.matmul(inverse_sum, difference),
            2.0 * torch.matmul(inverse_sum, medium_v),
        ]

    def _reduced_layer_smatrix(
        self,
        electric: torch.Tensor,
        magnetic: torch.Tensor,
        kz: torch.Tensor,
        thickness: torch.Tensor,
        reference_v: torch.Tensor,
    ) -> list[torch.Tensor]:
        size = electric.shape[0]
        identity = self._eye(size)
        phase = torch.diag(torch.exp(1.0j * self.omega * kz * thickness))
        a = self._solve(electric, identity) + self._solve(magnetic, reference_v)
        b = self._solve(electric, identity) - self._solve(magnetic, reference_v)
        a_inverse_b = self._solve(a, b)
        a_inverse_xb = self._solve(a, torch.matmul(phase, b))
        a_inverse_xa = self._solve(a, torch.matmul(phase, a))
        core = a - torch.matmul(torch.matmul(phase, b), a_inverse_xb)
        reflection = self._solve(
            core,
            torch.matmul(torch.matmul(phase, b), a_inverse_xa) - b,
        )
        transmission = self._solve(
            core,
            torch.matmul(phase, a - torch.matmul(b, a_inverse_b)),
        )
        return [transmission, reflection, reflection, transmission]

    def _polarized_redheffer_scattering(
        self,
        layers: Sequence[dict[str, torch.Tensor]],
        input_interface: Sequence[torch.Tensor],
        output_interface: Sequence[torch.Tensor],
        *,
        force_full: bool = False,
    ) -> list[torch.Tensor]:
        """Redheffer recursion within a source sector or complete native star."""
        size = layers[0]["electric"].shape[0]
        identity = self._eye(size)
        zero = torch.zeros_like(identity)
        layer_scattering = [
            self._reduced_layer_smatrix(
                layer["electric"],
                layer["magnetic"],
                layer["kz"],
                thickness,
                self._polarization_reference_v,
            )
            for layer, thickness in zip(layers, self.thickness)
        ]
        effective_size = "full" if force_full else self.smatrix_size
        if effective_size == "full":
            def connect(
                left: Sequence[torch.Tensor], right: Sequence[torch.Tensor]
            ) -> list[torch.Tensor]:
                tf_l, rf_l, rb_l, tb_l = left
                tf_r, rf_r, rb_r, tb_r = right
                inverse_lr = self._solve(
                    identity - torch.matmul(rb_l, rf_r), identity
                )
                inverse_rl = self._solve(
                    identity - torch.matmul(rf_r, rb_l), identity
                )
                return [
                    torch.matmul(tf_r, torch.matmul(inverse_lr, tf_l)),
                    rf_l
                    + torch.matmul(
                        tb_l,
                        torch.matmul(
                            inverse_rl, torch.matmul(rf_r, tf_l)
                        ),
                    ),
                    rb_r
                    + torch.matmul(
                        tf_r,
                        torch.matmul(
                            inverse_lr, torch.matmul(rb_l, tb_r)
                        ),
                    ),
                    torch.matmul(tb_l, torch.matmul(inverse_rl, tb_r)),
                ]

            scattering = list(input_interface)
            for layer in layer_scattering:
                scattering = connect(scattering, layer)
            return connect(scattering, output_interface)

        transmission = output_interface[0] if effective_size == "half" else None
        reflection = output_interface[1]

        def prepend(left: Sequence[torch.Tensor]) -> None:
            nonlocal transmission, reflection
            tf_left, rf_left, rb_left, tb_left = left
            reflection_new = rf_left + torch.matmul(
                tb_left,
                self._solve(
                    identity - torch.matmul(reflection, rb_left),
                    torch.matmul(reflection, tf_left),
                ),
            )
            if transmission is not None:
                transmission = torch.matmul(
                    transmission,
                    self._solve(
                        identity - torch.matmul(rb_left, reflection),
                        tf_left,
                    ),
                )
            reflection = reflection_new

        for layer in reversed(layer_scattering):
            prepend(layer)
        prepend(input_interface)
        return (
            [transmission, reflection, zero, zero]
            if transmission is not None
            else [zero, reflection, zero, zero]
        )

    def _polarized_li2a_scattering(
        self,
        layers: Sequence[dict[str, torch.Tensor]],
        input_v: torch.Tensor,
        output_v: torch.Tensor,
        *,
        force_full: bool = False,
    ) -> list[torch.Tensor]:
        """Li algorithm 2a within a source sector or complete native star."""
        size = layers[0]["electric"].shape[0]
        identity = self._eye(size)
        zero = torch.zeros_like(identity)

        effective_size = "full" if force_full else self.smatrix_size
        if effective_size == "full":
            def modal_basis(
                electric: torch.Tensor, magnetic: torch.Tensor
            ) -> torch.Tensor:
                return torch.cat(
                    (
                        torch.cat((electric, electric), dim=1),
                        torch.cat((magnetic, -magnetic), dim=1),
                    ),
                    dim=0,
                )

            bases = [
                modal_basis(identity, input_v),
                *[
                    modal_basis(layer["electric"], layer["magnetic"])
                    for layer in layers
                ],
                modal_basis(identity, output_v),
            ]
            phases = [identity]
            phases.extend(
                torch.diag(
                    torch.exp(
                        1.0j * self.omega * layer["kz"] * thickness
                    )
                )
                for layer, thickness in zip(layers, self.thickness)
            )
            tf, rf, rb, tb = identity, zero, zero, identity
            for interface, (left_basis, right_basis) in enumerate(
                zip(bases[:-1], bases[1:])
            ):
                transfer = self._solve(right_basis, left_basis)
                t11 = transfer[:size, :size]
                t12 = transfer[:size, size:]
                t21 = transfer[size:, :size]
                t22 = transfer[size:, size:]
                phase = phases[interface]
                v_matrix = torch.matmul(phase, torch.matmul(rb, phase))
                denominator = t22 + torch.matmul(t21, v_matrix)
                rb_new = self._right_solve(
                    t12 + torch.matmul(t11, v_matrix), denominator
                )
                tb_new = self._right_solve(
                    torch.matmul(tb, phase), denominator
                )
                tf_new = torch.matmul(
                    torch.matmul(t11 - torch.matmul(rb_new, t21), phase),
                    tf,
                )
                rf_new = rf - torch.matmul(
                    torch.matmul(torch.matmul(tb_new, t21), phase), tf
                )
                tf, rf, rb, tb = tf_new, rf_new, rb_new, tb_new
            return [tf, rf, rb, tb]

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
                layers[-1]["electric"],
                layers[-1]["magnetic"],
                identity,
                output_v,
            )
        ]
        for right_layer in range(len(layers) - 1, 0, -1):
            left_layer = right_layer - 1
            pairs.append(
                pair(
                    layers[left_layer]["electric"],
                    layers[left_layer]["magnetic"],
                    layers[right_layer]["electric"],
                    layers[right_layer]["magnetic"],
                )
            )
        pairs.append(
            pair(
                identity,
                input_v,
                layers[0]["electric"],
                layers[0]["magnetic"],
            )
        )

        t_plus, t_minus = pairs[0]
        reflection = self._right_solve(t_minus, t_plus)
        transmission = (
            self._solve(t_plus, identity)
            if effective_size == "half"
            else None
        )
        for offset, layer_index in enumerate(
            range(len(layers) - 1, -1, -1), start=1
        ):
            layer = layers[layer_index]
            phase = torch.exp(
                1.0j
                * self.omega
                * layer["kz"]
                * self.thickness[layer_index]
            )
            omega = phase[:, None] * reflection * phase[None, :]
            t_plus, t_minus = pairs[offset]
            denominator = t_plus + torch.matmul(t_minus, omega)
            reflection = self._right_solve(
                t_minus + torch.matmul(t_plus, omega), denominator
            )
            if transmission is not None:
                transmission = self._right_solve(
                    transmission * phase[None, :], denominator
                )
        return (
            [transmission, reflection, zero, zero]
            if transmission is not None
            else [zero, reflection, zero, zero]
        )

    def _reduced_internal_couplings(
        self,
        layers: Sequence[dict[str, torch.Tensor]],
        input_v: torch.Tensor,
        output_v: torch.Tensor,
        full_scattering: Sequence[torch.Tensor],
        electric_embedding: torch.Tensor,
    ) -> list[list[torch.Tensor]]:
        """Map external Fourier sources to each layer's two modal amplitudes.

        Each stored block contains ``[a_plus(z=0), a_minus(z=d)]``.  The
        boundary states are propagated by continuity rather than by forming a
        global multilayer matrix, so the reconstruction uses the same stable
        modal bases and decaying phases as the reduced cascade.
        """
        size = layers[0]["electric"].shape[0]
        identity = self._eye(size)

        def modal_basis(electric: torch.Tensor, magnetic: torch.Tensor) -> torch.Tensor:
            return torch.cat(
                (
                    torch.cat((electric, electric), dim=1),
                    torch.cat((magnetic, -magnetic), dim=1),
                ),
                dim=0,
            )

        left_boundaries: list[torch.Tensor] = []
        right_boundaries: list[torch.Tensor] = []
        for layer, thickness in zip(layers, self.thickness):
            phase = torch.diag(
                torch.exp(1.0j * self.omega * layer["kz"] * thickness)
            )
            basis = modal_basis(layer["electric"], layer["magnetic"])
            zero = torch.zeros_like(identity)
            left_propagation = torch.cat(
                (
                    torch.cat((identity, zero), dim=1),
                    torch.cat((zero, phase), dim=1),
                ),
                dim=0,
            )
            right_propagation = torch.cat(
                (
                    torch.cat((phase, zero), dim=1),
                    torch.cat((zero, identity), dim=1),
                ),
                dim=0,
            )
            left_boundaries.append(basis @ left_propagation)
            right_boundaries.append(basis @ right_propagation)

        input_basis = modal_basis(identity, input_v)
        output_basis = modal_basis(identity, output_v)
        tf, rf, rb, tb = full_scattering
        forward_state = input_basis @ torch.cat((identity, rf), dim=0)
        backward_state = output_basis @ torch.cat((rb, identity), dim=0)
        forward_reduced: list[torch.Tensor] = []
        for layer_index in range(len(layers)):
            amplitudes = self._solve(
                left_boundaries[layer_index], forward_state
            )
            forward_reduced.append(amplitudes)
            forward_state = right_boundaries[layer_index] @ amplitudes
        backward_reduced: list[torch.Tensor] = [
            torch.empty(0, dtype=self._dtype, device=self._device)
            for _ in layers
        ]
        for layer_index in range(len(layers) - 1, -1, -1):
            amplitudes = self._solve(
                right_boundaries[layer_index], backward_state
            )
            backward_reduced[layer_index] = amplitudes
            backward_state = left_boundaries[layer_index] @ amplitudes

        forward_target = output_basis @ torch.cat((tf, torch.zeros_like(tf)), dim=0)
        backward_target = input_basis @ torch.cat((torch.zeros_like(tb), tb), dim=0)
        tiny = torch.as_tensor(
            torch.finfo(forward_state.real.dtype).tiny,
            dtype=forward_state.real.dtype,
            device=self._device,
        )
        boundary_residual = torch.maximum(
            torch.linalg.vector_norm(forward_state - forward_target)
            / torch.maximum(torch.linalg.vector_norm(forward_target), tiny),
            torch.linalg.vector_norm(backward_state - backward_target)
            / torch.maximum(torch.linalg.vector_norm(backward_target), tiny),
        )
        tolerance = 2.0e-4 if self._dtype == torch.complex64 else 2.0e-9
        if _as_float(boundary_residual) > tolerance:
            raise RuntimeError(
                "Reduced internal-field boundary reconstruction failed: "
                f"{_as_float(boundary_residual):.3e}."
            )
        source_projection = electric_embedding.mH
        self._reduced_field_boundary_residual = boundary_residual.detach()
        return [
            [amplitudes @ source_projection for amplitudes in forward_reduced],
            [amplitudes @ source_projection for amplitudes in backward_reduced],
        ]

    def _solve_polarization_reduced_smatrix(self) -> None:
        if not self._polarized_layers or self._polarization_bases is None:
            raise RuntimeError(
                "Add at least one eligible symmetry-reduced circle layer before solving."
            )
        if len(self._polarized_layers) != self.layer_N:
            raise UnsupportedCombinationError(
                "Every internal layer must participate in polarization reduction."
            )
        electric_basis, magnetic_basis = self._polarization_bases

        def reduce_v(matrix: torch.Tensor) -> torch.Tensor:
            return torch.matmul(
                magnetic_basis.mH, torch.matmul(matrix, electric_basis)
            )

        reference_v = reduce_v(self.Vf)
        input_v = reduce_v(getattr(self, "Vi", self.Vf))
        output_v = reduce_v(getattr(self, "Vo", self.Vf))
        self._polarization_reference_v = reference_v
        input_interface = self._reduced_interface_s(
            reference_v, input_v, input_side=True
        )
        output_interface = self._reduced_interface_s(
            reference_v, output_v, input_side=False
        )
        need_field_data = bool(self.store_mode_couplings)
        redheffer = self._polarized_redheffer_scattering(
            self._polarized_layers,
            input_interface,
            output_interface,
            force_full=need_field_data,
        )
        if self.smatrix_algorithm == "redheffer":
            reduced_scattering = redheffer
            parity_error = None
        else:
            reduced_scattering = self._polarized_li2a_scattering(
                self._polarized_layers,
                input_v,
                output_v,
                force_full=need_field_data,
            )
            indices = (
                (0, 1, 2, 3)
                if self.smatrix_size == "full" or need_field_data
                else ((1,) if self.smatrix_size == "quarter" else (0, 1))
            )
            parity_error = torch.max(
                torch.stack(
                    [
                        torch.max(
                            torch.abs(
                                reduced_scattering[index] - redheffer[index]
                            )
                        )
                        for index in indices
                    ]
                )
            )
            tolerance = 2.0e-4 if self._dtype == torch.complex64 else 2.0e-9
            if self.verify_cascade and _as_float(parity_error) > tolerance:
                raise RuntimeError(
                    "Polarization-reduced Li-2a/Redheffer parity check failed: "
                    f"{_as_float(parity_error):.3e}."
                )

        full_size = 2 * self.order_N
        zero = torch.zeros(
            (full_size, full_size),
            dtype=self._dtype,
            device=self._device,
        )

        def expand(block: torch.Tensor) -> torch.Tensor:
            return torch.matmul(
                electric_basis, torch.matmul(block, electric_basis.mH)
            )

        if need_field_data:
            self._field_smatrix = [expand(block) for block in redheffer]
            self.C = self._reduced_internal_couplings(
                self._polarized_layers,
                input_v,
                output_v,
                redheffer,
                electric_basis,
            )
        else:
            self.C = [[], []]

        if self.smatrix_size == "full":
            self.S = [expand(block) for block in reduced_scattering]
        elif self.smatrix_size == "half":
            self.S = [
                expand(reduced_scattering[0]),
                expand(reduced_scattering[1]),
                zero,
                zero,
            ]
        else:
            self.S = [zero, expand(reduced_scattering[1]), zero, zero]
        self.S = self._mask_smatrix_blocks(self.S)
        self.cascade_diagnostics = {
            "algorithm": self.smatrix_algorithm,
            "size": self.smatrix_size,
            "computed_blocks": self.computed_smatrix_blocks,
            "polarization": self.polarization_reduction,
            "reduced_dimension": electric_basis.shape[1],
            "full_dimension": full_size,
        }
        if need_field_data:
            self.cascade_diagnostics["field_boundary_residual"] = (
                self._reduced_field_boundary_residual
            )
        if parity_error is not None:
            self.cascade_diagnostics[
                "redheffer_max_abs_error"
            ] = parity_error.detach()
