"""Analytic ellipse/rectangle NV layers for orthogonal cells.

Uses the existing symmetric NV factorization and stable modal/S algebra.
An equal-axis ellipse follows exactly the same ellipse code as any ellipse;
there is deliberately no dispatch to add_layer_circle_nvm().

Only centered, axis-aligned, non-overlapping shapes with mu=1 are supported.
Field reconstruction and symmetry reduction are intentionally unavailable.
"""
from __future__ import annotations

import math

import torch
from rcwa_ext import AutoRCWA
from rcwa_ext.config import LayerRecord, _as_float, _real_parameter_tensor
from rcwa_ext.nvm import _jinc, _smoothstep


class ShapeNVRCWA(AutoRCWA):
    def _check_nv_shape_mode(self):
        if abs(self.zeta_deg - 90.) > 1e-12:
            raise ValueError("Ellipse/rectangle NV currently requires an orthogonal cell.")
        if self.use_group_theory or self.polarization_reduction is not None:
            raise ValueError("Ellipse/rectangle NV uses the full x/y basis; disable group theory.")
        if self.store_mode_couplings:
            raise ValueError("Ellipse/rectangle NV currently supports fields='none' only.")
        if not hasattr(self, "Kx_norm"):
            raise ValueError("Call set_incident_angle() before adding NV layers.")

    def shape_convolution(self, shape, half_x, half_y, background, inclusion):
        """Exact Fourier integrals of epsilon or 1/epsilon, without a raster."""
        lx, ly = _as_float(self.L[0]), _as_float(self.L[1])
        mx, my = len(self.order_x), len(self.order_y)
        gx = (2 * math.pi / lx * self.order_x.to(torch.float64)).repeat_interleave(my)
        gy = (2 * math.pi / ly * self.order_y.to(torch.float64)).repeat(mx)
        dgx, dgy = gx[:, None] - gx[None, :], gy[:, None] - gy[None, :]
        if shape == "ellipse":
            argument = torch.sqrt((half_x * dgx)**2 + (half_y * dgy)**2)
            transform = math.pi * half_x * half_y / (lx * ly) * _jinc(argument)
        elif shape == "rectangle":
            transform = (4 * half_x * half_y / (lx * ly)
                         * torch.sinc(half_x * dgx / math.pi)
                         * torch.sinc(half_y * dgy / math.pi))
        else:
            raise ValueError("shape must be ellipse or rectangle.")
        translation = torch.exp(-1j * (dgx * lx / 2 + dgy * ly / 2))
        bg = torch.as_tensor(background, dtype=self._dtype, device=self._device)
        inc = torch.as_tensor(inclusion, dtype=self._dtype, device=self._device)
        return (bg * self._eye(self.order_N) + (inc - bg) * transform * translation).to(self._dtype)

    @staticmethod
    def normal_direction(shape, dx, dy, half_x, half_y):
        """Boundary-normal extension, excluding the rectangle's isolated corners.

        Ellipse: gradient of x^2/a^2+y^2/b^2.
        Rectangle: gradient of (1-x^2/a^2)(1-y^2/b^2), with global sign removed.
        On x=+/-a the second component vanishes, and on y=+/-b the first
        component vanishes. Both are zero at a corner, whose normal is undefined.
        """
        if shape == "ellipse":
            vx, vy = dx / half_x**2, dy / half_y**2
        elif shape == "rectangle":
            vx = dx / half_x**2 * (1 - (dy / half_y)**2)
            vy = dy / half_y**2 * (1 - (dx / half_x)**2)
        else:
            raise ValueError("shape must be ellipse or rectangle.")
        length = torch.sqrt(vx**2 + vy**2)
        safe = length.clamp_min(1e-14)
        return vx / safe, vy / safe, length

    def shape_projection(self, shape, half_x, half_y, nx, ny):
        """FFT is used only for the periodic normal projector, not material eps.

        Isotropic blending near the center and Voronoi edges matches the
        existing circle NV extension in the equal-axis limit. The blending
        must be saturated to one at each material boundary; reject a grid
        whose three-pixel edge transition could reach a material interface.
        """
        minimum = max(32, 4 * max(int(v) for v in self.order) + 4)
        if min(nx, ny) < minimum:
            raise ValueError(f"NV normal grid requires Q >= {minimum}.")
        lx, ly = _as_float(self.L[0]), _as_float(self.L[1])
        resolution = max(lx / nx, ly / ny)
        # Use the same regularized Voronoi extension as the existing circle
        # implementation. The check exposes under-resolved tiny gaps.
        edge_gap = min(lx - 2 * _as_float(half_x), ly - 2 * _as_float(half_y))
        if edge_gap < 3 * resolution:
            raise ValueError("NV grid is too coarse for the inter-pillar gap; increase --grids.")
        x = torch.arange(nx, dtype=torch.float64, device=self._device) * (lx / nx) - lx / 2
        y = torch.arange(ny, dtype=torch.float64, device=self._device) * (ly / ny) - ly / 2
        dx, dy = torch.meshgrid(x, y, indexing="ij")
        nearest = torch.sqrt(dx**2 + dy**2)
        second = torch.sqrt(torch.minimum((lx - dx.abs())**2 + dy**2,
                                          dx**2 + (ly - dy.abs())**2))
        vx, vy, magnitude = self.normal_direction(shape, dx, dy, half_x, half_y)
        center_radius = torch.maximum(torch.minimum(half_x, half_y) / 2,
                                      half_x.new_tensor(resolution))
        weight = (_smoothstep(nearest, 0., center_radius)
                  * _smoothstep(second - nearest, 0., 3 * resolution))
        # At an isolated rectangle corner use an isotropic value; the corner
        # has measure zero and no unique interface normal.
        weight = torch.where(magnitude > 1e-14, weight, torch.zeros_like(weight))
        pxx = .5 + (vx**2 - .5) * weight
        pyy = .5 + (vy**2 - .5) * weight
        pxy = vx * vy * weight
        count_y = len(self.order_y)
        px = self.order_x.repeat_interleave(count_y)
        py = self.order_y.repeat(len(self.order_x))
        ix, iy = px[:, None] - px[None, :], py[:, None] - py[None, :]
        def conv(field):
            fft = torch.fft.fft2(field.to(self._dtype)) / (nx * ny)
            result = fft[ix, iy]
            return .5 * (result + result.mH)
        xx, yy, xy = conv(pxx), conv(pyy), conv(pxy)
        return torch.cat((torch.cat((xx, xy), 1), torch.cat((xy.mH, yy), 1)), 0)

    def add_layer_shape_nv(self, thickness, shape, half_x, half_y,
                           eps_bg=1., eps_inclusion=6.25, *, nx=256, ny=256):
        self._check_nv_shape_mode()
        a, av = _real_parameter_tensor("half_x", half_x, dtype=torch.float64,
                                      device=self._device, allow_zero=False)
        b, bv = _real_parameter_tensor("half_y", half_y, dtype=torch.float64,
                                      device=self._device, allow_zero=False)
        dz, _ = _real_parameter_tensor("thickness", thickness, dtype=self._dtype,
                                      device=self._device, allow_zero=True)
        lx, ly = _as_float(self.L[0]), _as_float(self.L[1])
        if 2 * av >= lx or 2 * bv >= ly:
            raise ValueError("Shape must leave a positive gap in both cell directions.")
        for value in (eps_bg, eps_inclusion):
            material = torch.as_tensor(value, dtype=self._dtype, device=self._device)
            if material.numel() != 1 or not bool(torch.isfinite(material).all()) or bool((material.abs() == 0).any()):
                raise ValueError("NV materials must be finite, nonzero scalar permittivities.")
        eps = self.shape_convolution(shape, a, b, eps_bg, eps_inclusion)
        inverse_rule = self.shape_convolution(shape, a, b, 1 / eps_bg, 1 / eps_inclusion)
        identity = self._eye(self.order_N)
        projector = self.shape_projection(shape, a, b, int(nx), int(ny))
        tensor = self._symmetric_nvm_tensor(eps, self._solve(inverse_rule, identity) - eps, projector)
        n = self.order_N
        exx, exy, eyx, eyy = tensor[:n, :n], tensor[:n, n:], tensor[n:, :n], tensor[n:, n:]
        inv_eps = self._solve(eps, identity)
        k1, k2, s, c = self.K1_norm, self.K2_norm, self.sin_zeta, self.cos_zeta
        p = torch.cat((
            torch.cat(((k1 @ inv_eps @ k2 - c * identity) / s,
                       (identity - k1 @ inv_eps @ k1) / s), 1),
            torch.cat(((k2 @ inv_eps @ k2 - identity) / s,
                       (c * identity - k2 @ inv_eps @ k1) / s), 1)), 0)
        eps11 = s**2 * exx - s * c * (exy + eyx) + c**2 * eyy
        q = torch.cat((
            torch.cat(((-k1 @ k2 - s * eyx + c * eyy) / s,
                       (k1 @ k1 - eyy) / s), 1),
            torch.cat(((eps11 - k2 @ k2) / s,
                       (k2 @ k1 + s * exy - c * eyy) / s), 1)), 0)
        kz2, electric = self._eig(p @ q)
        kz = self._positive_kz(kz2)
        magnetic = self._magnetic_eigenvectors(p, q, electric, kz)
        layer = self.layer_N
        self.layer_N += 1
        self.thickness.append(dz)
        self.eps_conv.append(eps)
        self.mu_conv.append(identity)
        self.P.append(p)
        self.Q.append(q)
        self.kz_norm.append(kz)
        self.E_eigvec.append(electric)
        self.H_eigvec.append(magnetic)
        self._append_smatrix_from_cartesian_modes(electric, magnetic)
        self.layer_records.append(LayerRecord(
            index=layer, method="nvm", shape=shape, lattice=self.lattice_kind,
            reason="ANALYTIC_SHAPE_AND_NORMAL_VECTOR_FACTORIZATION",
            options={"half_x": av, "half_y": bv, "grid": (nx, ny),
                     "material_fourier": "analytic", "use_lanczos": False,
                     "factorization_rule": "symmetric-nv-anticommutator"}))

