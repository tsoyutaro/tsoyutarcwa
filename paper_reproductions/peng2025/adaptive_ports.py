"""Shared adaptive homogeneous ports for the Peng total-power study.

At finite truncation these are generalized homogeneous modes, not ordinary
Cartesian Fourier amplitudes. Keep this backend private to the power-only
reproducer; field recovery and arbitrary mixed-coordinate stacks are excluded.
"""
from __future__ import annotations

import torch

from rcwa_solver_auto import AutoRCWA


class _AdaptivePortRCWA(AutoRCWA):
    def add_layer_circle_shell_peng_asr(self, *args, **kwargs):
        if self.layer_N or self.store_mode_couplings:
            raise ValueError("Shared adaptive ports require one patterned layer and fields='none'.")
        if abs(float(self.inc_ang)) > 1e-10:
            raise ValueError("The shared adaptive power study supports normal incidence only.")
        if kwargs.get('normal_vector_factorization', False):
            raise ValueError("Shared adaptive ports currently require the exact strip-Li ASR operator.")
        super().add_layer_circle_shell_peng_asr(*args, **kwargs)
        self.layer_records[-1].options.update({
            'interface_rule': 'shared-adaptive',
            'port_basis': 'generalized-adaptive-homogeneous-modes',
        })

    def add_layer(self, thickness, eps=1.0, mu=1.0):
        if torch.as_tensor(eps).numel() != 1 or torch.as_tensor(mu).numel() != 1:
            raise ValueError("Shared adaptive ports allow only homogeneous layers after the pattern.")
        return super().add_layer(thickness, eps=eps, mu=mu)

    def _peng_weighted_convolutions(self, *args, **kwargs):
        result = super()._peng_weighted_convolutions(*args, **kwargs)
        self._port_jacobians = result['jacobian_x'], result['jacobian_y']
        return result

    def _peng_conversion_matrices(self, mapping):
        def axis(f, k):
            # K Phi = F Phi kappa, Phi^H F Phi = I. Both matrices are
            # Hermitian; using F^(-1/2) retains a well-conditioned eigenbasis.
            eigenvalues, vectors = torch.linalg.eigh((f + f.mH) / 2)
            if bool((eigenvalues <= 0).any()):
                raise ValueError("The adaptive-port Jacobian must be positive definite.")
            inverse_root = (vectors * eigenvalues.rsqrt()[None, :]) @ vectors.mH
            operator = inverse_root @ torch.diag(k) @ inverse_root
            kappa, phi = torch.linalg.eigh((operator + operator.mH) / 2)
            phi = inverse_root @ phi
            diagonal = torch.diagonal(phi)
            phase = diagonal / diagonal.abs().clamp_min(1e-14)
            return kappa.to(self._dtype), phi * phase.conj()[None, :]

        f, g = self._port_jacobians
        mx, my = len(self.order_x), len(self.order_y)
        kx, phi = axis(f, self.Kx_norm_dn.reshape(mx, my)[:, 0])
        ky, psi = axis(g, self.Ky_norm_dn.reshape(mx, my)[0, :])
        # The covariant port basis W has blocks (F Phi)xPsi and
        # Phix(G Psi). W^H C W=C, C=[[0,I],[-I,0]], so the same W^-1
        # converts E and H without an ill-conditioned Cartesian projection.
        transform = torch.block_diag(
            torch.kron(self._solve(f @ phi, self._eye(mx)), self._solve(psi, self._eye(my))),
            torch.kron(self._solve(phi, self._eye(mx)), self._solve(g @ psi, self._eye(my))),
        )
        self.Kx_norm_dn = kx[:, None].expand(mx, my).reshape(-1)
        self.Ky_norm_dn = ky[None, :].expand(mx, my).reshape(-1)
        self.Kx_norm, self.Ky_norm = torch.diag(self.Kx_norm_dn), torch.diag(self.Ky_norm_dn)
        self.K1_norm, self.K2_norm = self.Kx_norm, self.Ky_norm
        self.K1_norm_dn, self.K2_norm_dn = self.Kx_norm_dn, self.Ky_norm_dn
        transverse = self.Kx_norm_dn**2 + self.Ky_norm_dn**2
        self.Vf = self._cartesian_e_to_h(self._positive_kz(1 - transverse))
        self.Vi = self._cartesian_e_to_h(self._positive_kz(self.eps_in * self.mu_in - transverse), mu=self.mu_in)
        self.Vo = self._cartesian_e_to_h(self._positive_kz(self.eps_out * self.mu_out - transverse), mu=self.mu_out)
        self.Sin = self._interface_s(self.Vi, input_side=True)
        self.Sout = self._interface_s(self.Vo, input_side=False)
        self._last_asr_transform_condition = (
            torch.linalg.cond(transform) if self.compute_condition_numbers else None
        )
        return transform, transform, None
