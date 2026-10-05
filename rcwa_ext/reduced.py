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
        # Recover couplings by Redheffer composition. Inverting a boundary
        # matrix with diag(exp(i*kz*d)) divides by exponentially small phases
        # and turns evanescent roundoff into a growing wave in thick/metals.
        reference_v = self._polarization_reference_v
        zero = torch.zeros_like(identity)
        scattering = self._reduced_interface_s(reference_v, input_v, input_side=True)
        forward_reduced, backward_reduced = [], []

        def coupling_solve(matrix, rhs):
            # Mode norms can span many decades after strong ASR compression.
            # Row/column equilibration preserves the linear system and avoids
            # treating a very small mode column as floating-point noise.
            a,b = matrix.to(torch.complex128),rhs.to(torch.complex128)
            floor = torch.finfo(a.real.dtype).tiny
            rows = a.abs().amax(1).clamp_min(floor)
            row_scaled = a/rows[:,None]
            columns = row_scaled.abs().amax(0).clamp_min(floor)
            scaled = row_scaled/columns[None,:]
            def solve(value):
                return torch.linalg.solve(scaled,value/rows[:,None])/columns[:,None]
            value = solve(b)
            for _ in range(2):
                value = value+solve(b-a@value)
            return value.to(self._dtype)

        def connect(right, cf=None, cb=None):
            nonlocal scattering, forward_reduced, backward_reduced
            tf_l, rf_l, rb_l, tb_l = scattering
            tf_r, rf_r, rb_r, tb_r = right
            lr = self._solve(identity-rb_l@rf_r, identity)
            rl = self._solve(identity-rf_r@rb_l, identity)
            new_f = [a+b@rl@rf_r@tf_l for a,b in zip(forward_reduced,backward_reduced)]
            new_b = [b@rl@tb_r for b in backward_reduced]
            if cf is not None:
                new_f.append(cf@lr@tf_l)
                new_b.append(cb+cf@lr@rb_l@tb_r)
            forward_reduced, backward_reduced = new_f, new_b
            scattering = [tf_r@lr@tf_l, rf_l+tb_l@rl@rf_r@tf_l,
                          rb_r+tf_r@lr@rb_l@tb_r, tb_l@rl@tb_r]

        for layer, thickness in zip(layers, self.thickness):
            e,h = layer["electric"],layer["magnetic"]
            phase = torch.diag(torch.exp(1j*self.omega*layer["kz"]*thickness))
            vh = self._solve(reference_v,h)
            plus,minus = e+vh,e-vh
            boundary = torch.cat((torch.cat((plus,minus@phase),1),
                                  torch.cat((minus@phase,plus),1)),0)
            all_c = coupling_solve(boundary,2*self._eye(2*size))
            cf,cb = all_c[:,:size],all_c[:,size:]
            ep = e@phase
            right = [ep@cf[:size]+e@cf[size:], e@cf[:size]+ep@cf[size:]-identity,
                     ep@cb[:size]+e@cb[size:]-identity, e@cb[:size]+ep@cb[size:]]
            connect(right,cf,cb)
        connect(self._reduced_interface_s(reference_v,output_v,input_side=False))

        # Check BOTH sides and every layer boundary. This check never uses
        # propagation by an inverse decaying phase.
        # Use the scattering blocks from this SAME coupling cascade. Two
        # algebraically equivalent layer formulas can differ substantially
        # for ill-conditioned, evanescent high-order ports. Mixing them makes
        # the field boundary check compare two different numerical solutions.
        self._reduced_coupling_smatrix = scattering
        tf,rf,rb,tb = scattering
        forward_state = input_basis@torch.cat((identity,rf),0)
        backward_state = output_basis@torch.cat((rb,identity),0)
        forward_target = output_basis@torch.cat((tf,zero),0)
        backward_target = input_basis@torch.cat((zero,tb),0)
        comparisons = [(left_boundaries[0]@forward_reduced[0],forward_state),
                       (right_boundaries[-1]@forward_reduced[-1],forward_target),
                       (right_boundaries[-1]@backward_reduced[-1],backward_state),
                       (left_boundaries[0]@backward_reduced[0],backward_target)]
        for i in range(len(layers)-1):
            comparisons.extend((
                (right_boundaries[i]@forward_reduced[i],left_boundaries[i+1]@forward_reduced[i+1]),
                (right_boundaries[i]@backward_reduced[i],left_boundaries[i+1]@backward_reduced[i+1])))
        residuals = [torch.linalg.vector_norm(a-b)/torch.maximum(
                    torch.maximum(torch.linalg.vector_norm(a),torch.linalg.vector_norm(b)),
                    torch.as_tensor(torch.finfo(a.real.dtype).tiny,dtype=a.real.dtype,device=a.device))
                    for a,b in comparisons]
        boundary_residual = torch.stack(residuals).max()
        tolerance = 2e-4 if self._dtype == torch.complex64 else 2e-9
        if _as_float(boundary_residual) > tolerance:
            raise RuntimeError("Reduced internal-field boundary reconstruction failed: "
                               f"{_as_float(boundary_residual):.3e}.")
        source_projection = electric_embedding.mH
        self._reduced_field_boundary_residual = boundary_residual.detach()
        return [
            [amplitudes @ source_projection for amplitudes in forward_reduced],
            [amplitudes @ source_projection for amplitudes in backward_reduced],
        ]

    def solve_polarization_source(self, source: torch.Tensor, *, release_operators=False):
        """Return (transmitted, reflected) E vectors without expanding S.

        Requires a source in the configured symmetry sector, fields disabled,
        and half/full scattering. Public S is empty for this vector-only solve.

        release_operators=True discards the stored full-space P/Q operators
        before cascading. Their list slots become None; subsequent use must
        stay on the reduced, fields-disabled path (rebuild for other uses).
        Reduced modes, ports and autograd graphs are preserved. Autograd or
        external references can retain storage, so byte counts are not a
        measurement of released device memory. Default preserves operators.
        """
        if self.store_mode_couplings or self.smatrix_size == "quarter":
            raise UnsupportedCombinationError("Source response requires fields disabled and half/full scattering.")
        source = torch.as_tensor(source, dtype=self._dtype, device=self._device)
        if source.shape != (2 * self.order_N,):
            raise ValueError("Source must be one transverse Fourier vector of length 2*order_N.")
        if not bool(torch.isfinite(source).all()):
            raise ValueError("Source must be finite.")
        if self._polarization_bases is None:
            raise UnsupportedCombinationError("Source response requires a configured symmetry-reduced layer.")
        basis = self._polarization_bases[0]
        residual = torch.linalg.vector_norm(source - basis @ (basis.mH @ source))
        scale = torch.linalg.vector_norm(source).clamp_min(torch.finfo(source.real.dtype).tiny)
        tolerance = 2e-5 if self._dtype == torch.complex64 else 1e-9
        if _as_float(residual / scale) > tolerance:
            raise UnsupportedCombinationError("Source lies outside the selected symmetry sector.")
        if not self._polarized_layers or len(self._polarized_layers) != self.layer_N:
            raise UnsupportedCombinationError("Every internal layer must participate in polarization reduction.")
        discarded_bytes = 0
        if release_operators:
            # Only P/Q are discarded: reduced cascades use reduced E/H/kz,
            # and power observables still need the original input/output ports.
            # Replace list slots rather than clearing lists to preserve indices.
            for name in ("P", "Q"):
                operators = getattr(self, name)
                for index in range(len(operators)):
                    if operators[index] is not None:
                        discarded_bytes += operators[index].numel() * operators[index].element_size()
                        operators[index] = None
        response = self._solve_polarization_reduced_smatrix(incident_source=source)
        self.cascade_diagnostics["release_operators"] = bool(release_operators)
        self.cascade_diagnostics["discarded_operator_bytes"] = discarded_bytes
        return response

    def _solve_polarization_reduced_smatrix(self, *, incident_source=None):
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
        need_field_data = bool(self.store_mode_couplings)
        need_redheffer = self.smatrix_algorithm == "redheffer" or self.verify_cascade or need_field_data
        redheffer = None
        parity_error = None
        if need_redheffer:
            input_interface = self._reduced_interface_s(reference_v, input_v, input_side=True)
            output_interface = self._reduced_interface_s(reference_v, output_v, input_side=False)
            redheffer = self._polarized_redheffer_scattering(
                self._polarized_layers, input_interface, output_interface,
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
            if self.verify_cascade:
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
        if incident_source is not None:
            projected_source = electric_basis.mH @ incident_source
            transmitted = electric_basis @ (reduced_scattering[0] @ projected_source)
            reflected = electric_basis @ (reduced_scattering[1] @ projected_source)
            self.S = []
            self.C = [[], []]
            self.cascade_diagnostics = {
                "algorithm": self.smatrix_algorithm, "size": self.smatrix_size,
                "computed_blocks": (), "output": "incident-source-vectors",
                "redheffer_computed": need_redheffer,
                "polarization": self.polarization_reduction,
                "reduced_dimension": electric_basis.shape[1], "full_dimension": full_size,
            }
            if parity_error is not None:
                self.cascade_diagnostics["redheffer_max_abs_error"] = parity_error.detach()
            return transmitted, reflected
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
            self.C = self._reduced_internal_couplings(
                self._polarized_layers,
                input_v,
                output_v,
                redheffer,
                electric_basis,
            )
            redheffer = self._reduced_coupling_smatrix
            self._field_smatrix = [expand(block) for block in redheffer]
            if self.smatrix_algorithm == 'redheffer':
                reduced_scattering = redheffer
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
