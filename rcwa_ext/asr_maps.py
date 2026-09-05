"""Separable, circular, and D6-equivariant adaptive coordinate maps."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .config import (
    UnsupportedCombinationError, _TWO_PI, _as_float,
    _real_parameter_tensor,
)

@dataclass(frozen=True)
class ASRMapping:
    u: torch.Tensor
    v: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor
    f: torch.Tensor
    g: torch.Tensor
    x_breaks: torch.Tensor
    y_breaks: torch.Tensor
    u_breaks: torch.Tensor
    v_breaks: torch.Tensor


@dataclass(frozen=True)
class CircleASRMapping:
    """Periodic matched coordinates for a circular interface.

    For an orthogonal cell, ``tu,tv`` are the intermediate Weiss coordinates
    after a separable ASR stretch.  For a triangular cell they equal ``u,v``;
    the D6-equivariant two-dimensional map itself supplies the adaptation.
    ``x,y`` are physical Cartesian coordinates and the remaining arrays are
    the entries and determinant of J = d(x,y)/d(u,v).
    A double-matched core-shell map also stores the covector normal
    ``d(rho)`` of its two fixed computational support curves for generalized
    Li normal-D/tangential-E factorization.
    """

    u: torch.Tensor
    v: torch.Tensor
    tu: torch.Tensor
    tv: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor
    x_u: torch.Tensor
    x_v: torch.Tensor
    y_u: torch.Tensor
    y_v: torch.Tensor
    det_j: torch.Tensor
    tu_breaks: torch.Tensor
    tv_breaks: torch.Tensor
    u_breaks: torch.Tensor
    v_breaks: torch.Tensor
    matched_outer_mask: torch.Tensor | None = None
    matched_core_mask: torch.Tensor | None = None
    interface_normal_u: torch.Tensor | None = None
    interface_normal_v: torch.Tensor | None = None
    effective_radial_slope: torch.Tensor | None = None
    central_radial_slope: torch.Tensor | None = None
    boundary_radial_slope: torch.Tensor | None = None
    minimum_radial_secant: torch.Tensor | None = None

class _ASRMappingMixin:
    """Construct coordinate maps and their sampled Jacobians."""

    def _require_kvectors(self) -> None:
        if not hasattr(self, "Kx_norm"):
            raise RuntimeError(
                "Call set_incident_angle() after add_input_layer()/"
                "add_output_layer() and before adding an ASR layer."
            )

    @staticmethod
    def _validate_grid(order: int, samples: int, axis: str) -> None:
        minimum = max(32, 4 * order + 4)
        if samples < minimum:
            raise ValueError(
                f"{axis} sampling grid is too small: got {samples}, "
                f"require at least {minimum} for Fourier order {order}."
            )

    def _piecewise_asr_map(
        self,
        length: float,
        physical_breaks: torch.Tensor,
        samples: int,
        *,
        minimum_slope: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Implement Eqs. (1)-(2) on [0, length).

        The interval allocation is normalized so that sum(Delta u_l)=length:
        Delta u_l is proportional to cbrt(Delta x_l).  This is the periodic,
        dimensionally consistent form of Eq. (2).
        """
        dx = physical_breaks[1:] - physical_breaks[:-1]
        if torch.any(dx <= 0.0):
            raise ValueError("ASR physical breakpoints must be strictly increasing.")

        weights = torch.pow(dx, 1.0 / 3.0)
        du = length * weights / torch.sum(weights)
        # Keep the periodic endpoint exact without overwriting an autograd
        # output in place.  Interior breakpoints retain their design gradient.
        transformed_breaks = torch.cat(
            (
                torch.zeros(1, dtype=torch.float64, device=self._device),
                torch.cumsum(du, dim=0)[:-1],
                torch.full((1,), length, dtype=torch.float64, device=self._device),
            )
        )

        coordinate = (
            torch.arange(samples, dtype=torch.float64, device=self._device)
            * (length / samples)
        )
        interval = torch.bucketize(
            coordinate, transformed_breaks[1:-1], right=False
        )
        u0 = transformed_breaks[interval]
        u1 = transformed_breaks[interval + 1]
        x0 = physical_breaks[interval]
        x1 = physical_breaks[interval + 1]
        local_du = u1 - u0
        local_dx = x1 - x0

        a1 = (u1 * x0 - u0 * x1) / local_du
        a2 = local_dx / local_du
        selected_slope = self.asr_G if minimum_slope is None else float(minimum_slope)
        if not 0.0 < selected_slope < 1.0:
            raise ValueError("minimum_slope must be in (0,1).")
        a3 = selected_slope * local_du - local_dx
        phase = _TWO_PI * (coordinate - u0) / local_du

        mapped = a1 + a2 * coordinate + (a3 / _TWO_PI) * torch.sin(phase)
        jacobian = a2 + (a3 / local_du) * torch.cos(phase)
        if torch.any(jacobian <= 0.0):
            raise RuntimeError("The ASR mapping is not monotone.")
        return coordinate, mapped, jacobian, transformed_breaks

    def build_asr_mapping(
        self,
        nx: int,
        ny: int,
        fill_factor_x: float,
        fill_factor_y: float,
    ) -> ASRMapping:
        """Build the separable x(u), y(v), f(u), g(v) mapping for a centered patch."""
        if not 0.0 < fill_factor_x < 1.0:
            raise ValueError("fill_factor_x must be in (0, 1).")
        if not 0.0 < fill_factor_y < 1.0:
            raise ValueError("fill_factor_y must be in (0, 1).")
        self._validate_grid(int(self.order[0]), nx, "x")
        self._validate_grid(int(self.order[1]), ny, "y")

        lx, ly = _as_float(self.L[0]), _as_float(self.L[1])
        x_breaks = torch.tensor(
            [0.0, 0.5 * lx * (1.0 - fill_factor_x),
             0.5 * lx * (1.0 + fill_factor_x), lx],
            dtype=torch.float64,
            device=self._device,
        )
        y_breaks = torch.tensor(
            [0.0, 0.5 * ly * (1.0 - fill_factor_y),
             0.5 * ly * (1.0 + fill_factor_y), ly],
            dtype=torch.float64,
            device=self._device,
        )
        u, x, f, u_breaks = self._piecewise_asr_map(lx, x_breaks, nx)
        v, y, g, v_breaks = self._piecewise_asr_map(ly, y_breaks, ny)
        return ASRMapping(
            u=u,
            v=v,
            x=x,
            y=y,
            f=f,
            g=g,
            x_breaks=x_breaks,
            y_breaks=y_breaks,
            u_breaks=u_breaks,
            v_breaks=v_breaks,
        )

    @staticmethod
    def _matched_circle_axis(
        main: torch.Tensor,
        cross: torch.Tensor,
        *,
        main_length: float,
        cross_center: float,
        circle_center: float,
        radius: float,
        main_minus: float,
        main_plus: float,
        cross_minus: float,
        cross_plus: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Eq. (37)-(38) of Weiss et al. for one Cartesian component.

        The returned derivatives are with respect to the two *matched*
        coordinates, before the final one-dimensional ASR stretches.
        """
        main_grid = main[:, None]
        cross_grid = cross[None, :]
        central = (cross_grid >= cross_minus) & (cross_grid <= cross_plus)
        offset = cross_grid - cross_center
        root = torch.sqrt(
            torch.clamp(radius**2 - offset**2, min=1.0e-30)
        )
        curve_minus = torch.where(
            central, circle_center - root, main_minus
        )
        curve_plus = torch.where(
            central, circle_center + root, main_plus
        )
        derivative_minus = torch.where(central, offset / root, 0.0)
        derivative_plus = torch.where(central, -offset / root, 0.0)

        lower = main_grid < main_minus
        middle = (main_grid >= main_minus) & (main_grid <= main_plus)
        middle_width = main_plus - main_minus
        upper_width = main_length - main_plus

        lower_weight = main_grid / main_minus
        middle_plus_weight = (main_grid - main_minus) / middle_width
        middle_minus_weight = 1.0 - middle_plus_weight
        upper_curve_weight = (main_length - main_grid) / upper_width
        upper_end_weight = (main_grid - main_plus) / upper_width

        mapped = torch.where(
            lower,
            lower_weight * curve_minus,
            torch.where(
                middle,
                middle_minus_weight * curve_minus
                + middle_plus_weight * curve_plus,
                upper_curve_weight * curve_plus
                + upper_end_weight * main_length,
            ),
        )
        derivative_main = torch.where(
            lower,
            curve_minus / main_minus,
            torch.where(
                middle,
                (curve_plus - curve_minus) / middle_width,
                (main_length - curve_plus) / upper_width,
            ),
        )
        derivative_cross = torch.where(
            lower,
            lower_weight * derivative_minus,
            torch.where(
                middle,
                middle_minus_weight * derivative_minus
                + middle_plus_weight * derivative_plus,
                upper_curve_weight * derivative_plus,
            ),
        )
        return mapped, derivative_main, derivative_cross

    def build_circle_asr_mapping(
        self,
        nx: int,
        ny: int,
        radius: float,
    ) -> CircleASRMapping:
        """Matched-coordinate circle map plus ASR for an orthogonal cell.

        This is the rectangular-period generalization of Weiss et al.
        Eqs. (37)-(38).  The circle is centered at the cell center.  The four
        interface coordinates are displaced by R/sqrt(2), so each quadrant of
        the circular interface is a coordinate-surface segment.
        """
        self._validate_grid(int(self.order[0]), nx, "x")
        self._validate_grid(int(self.order[1]), ny, "y")
        lx, ly = _as_float(self.L[0]), _as_float(self.L[1])
        radius, radius_value = _real_parameter_tensor(
            "radius",
            radius,
            dtype=torch.float64,
            device=self._device,
            allow_zero=False,
        )
        if 2.0 * radius_value >= min(lx, ly):
            raise ValueError(
                "matched-ASR requires a non-touching circle with "
                "2*radius < min(Lx,Ly)."
            )

        displacement = radius / math.sqrt(2.0)
        tx_minus, tx_plus = 0.5 * lx - displacement, 0.5 * lx + displacement
        ty_minus, ty_plus = 0.5 * ly - displacement, 0.5 * ly + displacement
        tx_breaks = torch.stack(
            (radius.new_tensor(0.0), tx_minus, tx_plus, radius.new_tensor(lx))
        )
        ty_breaks = torch.stack(
            (radius.new_tensor(0.0), ty_minus, ty_plus, radius.new_tensor(ly))
        )
        u, tu, dtu_du, u_breaks = self._piecewise_asr_map(
            lx, tx_breaks, nx, minimum_slope=self.matched_asr_G
        )
        v, tv, dtv_dv, v_breaks = self._piecewise_asr_map(
            ly, ty_breaks, ny, minimum_slope=self.matched_asr_G
        )

        x, x_tu, x_tv = self._matched_circle_axis(
            tu,
            tv,
            main_length=lx,
            cross_center=0.5 * ly,
            circle_center=0.5 * lx,
            radius=radius,
            main_minus=tx_minus,
            main_plus=tx_plus,
            cross_minus=ty_minus,
            cross_plus=ty_plus,
        )
        y_transposed, y_tv_transposed, y_tu_transposed = (
            self._matched_circle_axis(
                tv,
                tu,
                main_length=ly,
                cross_center=0.5 * lx,
                circle_center=0.5 * ly,
                radius=radius,
                main_minus=ty_minus,
                main_plus=ty_plus,
                cross_minus=tx_minus,
                cross_plus=tx_plus,
            )
        )
        y = y_transposed.mT
        y_tv = y_tv_transposed.mT
        y_tu = y_tu_transposed.mT

        x_u = x_tu * dtu_du[:, None]
        x_v = x_tv * dtv_dv[None, :]
        y_u = y_tu * dtu_du[:, None]
        y_v = y_tv * dtv_dv[None, :]
        det_j = x_u * y_v - x_v * y_u
        scale = max(lx, ly)
        minimum_jacobian = 1.0e-12 * scale * scale / (lx * ly)
        if not bool(torch.all(torch.isfinite(det_j))):
            raise RuntimeError("The matched circle map produced a non-finite Jacobian.")
        if _as_float(torch.min(det_j)) <= minimum_jacobian:
            raise RuntimeError(
                "The matched circle map is not orientation-preserving; "
                "reduce the circle radius or ASR strength."
            )
        return CircleASRMapping(
            u=u,
            v=v,
            tu=tu,
            tv=tv,
            x=x,
            y=y,
            x_u=x_u,
            x_v=x_v,
            y_u=y_u,
            y_v=y_v,
            det_j=det_j,
            tu_breaks=tx_breaks,
            tv_breaks=ty_breaks,
            u_breaks=u_breaks,
            v_breaks=v_breaks,
        )

    @staticmethod
    def _cubic_hermite(
        t: torch.Tensor,
        y0: torch.Tensor | float,
        slope0: torch.Tensor | float,
        y1: torch.Tensor | float,
        slope1: torch.Tensor | float,
        span: float,
    ) -> torch.Tensor:
        """Value of a two-endpoint cubic Hermite interpolant."""
        h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
        h10 = t**3 - 2.0 * t**2 + t
        h01 = -2.0 * t**3 + 3.0 * t**2
        h11 = t**3 - t**2
        return h00 * y0 + h10 * span * slope0 + h01 * y1 + h11 * span * slope1

    @staticmethod
    def _quintic_hermite_zero_curvature(
        t: torch.Tensor,
        y0: torch.Tensor | float,
        slope0: torch.Tensor | float,
        y1: torch.Tensor | float,
        slope1: torch.Tensor | float,
        span: torch.Tensor | float,
    ) -> torch.Tensor:
        """C2 Hermite segment with zero curvature at both endpoints.

        ``slope0`` and ``slope1`` are derivatives with respect to the
        unnormalised radial coordinate.  Setting the endpoint curvatures to
        zero lets adjacent segments share value, slope, and curvature without
        introducing additional user parameters.
        """

        h00 = 1.0 - 10.0 * t**3 + 15.0 * t**4 - 6.0 * t**5
        h10 = t - 6.0 * t**3 + 8.0 * t**4 - 3.0 * t**5
        h01 = 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5
        h11 = -4.0 * t**3 + 7.0 * t**4 - 3.0 * t**5
        return h00 * y0 + h10 * span * slope0 + h01 * y1 + h11 * span * slope1

    def _double_matched_radial_profile(
        self,
        rho: torch.Tensor,
        *,
        core_ratio: torch.Tensor,
        core_target: torch.Tensor,
        circle_scale: torch.Tensor,
        rho_outer: torch.Tensor,
        matched_slope: torch.Tensor,
        central_slope: torch.Tensor,
        boundary_slope: torch.Tensor,
    ) -> torch.Tensor:
        """Monotone C2 radial profile matching both material interfaces.

        The computational interfaces are ``rho=core_ratio`` and ``rho=1``.
        They are mapped to concentric physical circles through the target
        values ``core_target`` and ``circle_scale``.  The profile returns to
        the identity value at the periodic cell boundary, with the same
        positive slopes at the centre, both interfaces, and periodic boundary.
        The small ASR slope is used only at the two material interfaces.  The
        centre and cell-boundary slopes are separately enlarged while retaining
        a monotonicity certificate.  For endpoint slopes ``m0,m1`` and secant
        ``delta``, the Bernstein coefficients of the radial derivative are

            {m0, m0, 5 delta - 2 m0 - 2 m1, m1, m1}.

        Thus choosing both endpoint slopes in ``(0,delta]`` guarantees strict
        radial monotonicity.  It also avoids the former ``G^2`` compression at
        the coordinate origin and periodic seam.
        """

        zero = torch.zeros_like(circle_scale)

        t_core = torch.clamp(rho / core_ratio, 0.0, 1.0)
        radial_core = self._quintic_hermite_zero_curvature(
            t_core,
            zero,
            central_slope,
            core_target,
            matched_slope,
            core_ratio,
        )

        shell_span = 1.0 - core_ratio
        t_shell = torch.clamp((rho - core_ratio) / shell_span, 0.0, 1.0)
        radial_shell = self._quintic_hermite_zero_curvature(
            t_shell,
            core_target,
            matched_slope,
            circle_scale,
            matched_slope,
            shell_span,
        )

        exterior_span = rho_outer - 1.0
        t_exterior = torch.clamp((rho - 1.0) / exterior_span, 0.0, 1.0)
        radial_exterior = self._quintic_hermite_zero_curvature(
            t_exterior,
            circle_scale,
            matched_slope,
            torch.ones_like(circle_scale) * rho_outer,
            boundary_slope,
            exterior_span,
        )
        return torch.where(
            rho <= core_ratio,
            radial_core,
            torch.where(rho <= 1.0, radial_shell, radial_exterior),
        )

    def build_double_matched_circle_asr_mapping(
        self,
        nx: int,
        ny: int,
        core_radius: float,
        outer_radius: float,
    ) -> CircleASRMapping:
        """Build a C2 radial map matched to both concentric interfaces.

        Homothetic rectangles (orthogonal cells) or regular hexagons
        (triangular cells) in computational space are mapped to the inner and
        outer physical circles.  Three quintic Hermite pieces join the centre,
        both interfaces, and the Wigner--Seitz boundary.  The map is C2 at the
        interfaces and returns to the identity value with matching slope and
        zero curvature at the periodic boundary.
        """

        self._validate_grid(int(self.order[0]), nx, "u")
        self._validate_grid(int(self.order[1]), ny, "v")
        lx, ly = _as_float(self.L[0]), _as_float(self.L[1])
        core_radius, core_value = _real_parameter_tensor(
            "core_radius",
            core_radius,
            dtype=torch.float64,
            device=self._device,
            allow_zero=False,
        )
        outer_radius, outer_value = _real_parameter_tensor(
            "outer_radius",
            outer_radius,
            dtype=torch.float64,
            device=self._device,
            allow_zero=False,
        )
        if core_value >= outer_value:
            raise ValueError("core_radius must be smaller than outer_radius.")

        triangular = getattr(self, "lattice_kind", "rectangular") == "triangular"
        cosine = 0.5 if triangular else 0.0
        sine = math.sqrt(3.0) / 2.0 if triangular else 1.0
        if triangular:
            if not math.isclose(lx, ly, rel_tol=1.0e-8, abs_tol=1.0e-12):
                raise UnsupportedCombinationError(
                    "Triangular double-matched ASR requires equal primitive-vector lengths."
                )
            zeta = float(getattr(self, "zeta_deg", 90.0))
            if abs(zeta - 60.0) > 1.0e-7:
                raise UnsupportedCombinationError(
                    "Triangular double-matched ASR requires a 60-degree cell."
                )
            if nx != ny:
                raise UnsupportedCombinationError(
                    "Triangular double-matched ASR requires an equal u/v grid."
                )
            support_boundary = 0.5 * lx
        else:
            if hasattr(self, "cos_zeta") and abs(_as_float(self.cos_zeta)) > 1.0e-8:
                raise UnsupportedCombinationError(
                    "Double-matched circle ASR supports orthogonal or 60-degree triangular cells."
                )
            support_boundary = 0.5 * min(lx, ly)
        if outer_value >= support_boundary:
            raise ValueError(
                "double-matched ASR requires a non-touching outer circle: "
                "2*outer_radius < min(Lx,Ly)."
            )

        u_axis = torch.arange(nx, dtype=torch.float64, device=self._device) * lx / nx
        v_axis = torch.arange(ny, dtype=torch.float64, device=self._device) * ly / ny
        u_grid, v_grid = torch.meshgrid(u_axis, v_axis, indexing="ij")
        u_leaf = u_grid.detach().requires_grad_(True)
        v_leaf = v_grid.detach().requires_grad_(True)

        if triangular:
            q1_candidates: list[torch.Tensor] = []
            q2_candidates: list[torch.Tensor] = []
            r2_candidates: list[torch.Tensor] = []
            shifts: list[tuple[int, int]] = []
            for shift_i_candidate in (-1, 0, 1):
                for shift_j_candidate in (-1, 0, 1):
                    q1_candidate = u_leaf - 0.5 * lx - shift_i_candidate * lx
                    q2_candidate = v_leaf - 0.5 * ly - shift_j_candidate * ly
                    q1_candidates.append(q1_candidate)
                    q2_candidates.append(q2_candidate)
                    r2_candidates.append(
                        q1_candidate**2
                        + q2_candidate**2
                        + 2.0 * cosine * q1_candidate * q2_candidate
                    )
                    shifts.append((shift_i_candidate, shift_j_candidate))
            nearest = torch.argmin(
                torch.stack(r2_candidates, dim=0), dim=0, keepdim=True
            )
            q1 = torch.gather(torch.stack(q1_candidates, dim=0), 0, nearest)[0]
            q2 = torch.gather(torch.stack(q2_candidates, dim=0), 0, nearest)[0]
            shift_i_values = torch.tensor(
                [item[0] for item in shifts], dtype=torch.float64, device=self._device
            )[:, None, None]
            shift_j_values = torch.tensor(
                [item[1] for item in shifts], dtype=torch.float64, device=self._device
            )[:, None, None]
            shift_i = torch.gather(
                shift_i_values.expand(-1, nx, ny), 0, nearest
            )[0]
            shift_j = torch.gather(
                shift_j_values.expand(-1, nx, ny), 0, nearest
            )[0]
            support_radius = torch.maximum(
                torch.maximum(
                    torch.abs(q1 + 0.5 * q2),
                    torch.abs(0.5 * q1 + q2),
                ),
                torch.abs(0.5 * (q1 - q2)),
            )
        else:
            q1 = u_leaf - 0.5 * lx
            q2 = v_leaf - 0.5 * ly
            shift_i = torch.zeros_like(q1)
            shift_j = torch.zeros_like(q2)
            minimum_length = min(lx, ly)
            support_radius = torch.maximum(
                torch.abs(q1) * minimum_length / lx,
                torch.abs(q2) * minimum_length / ly,
            )

        # Keep both discontinuities on fixed computational curves.  This is
        # essential for a shape derivative: the Boolean material masks then do
        # not move across quadrature samples, and both physical radii enter only
        # through the smooth map/Jacobian.  Fractions 1/3 and 2/3 leave equal
        # computational widths for core, shell, and exterior regions.
        outer_coordinate_radius = outer_radius.new_tensor(
            (2.0 / 3.0) * support_boundary
        )
        core_coordinate_radius = outer_radius.new_tensor(
            (1.0 / 3.0) * support_boundary
        )
        core_ratio = core_coordinate_radius / outer_coordinate_radius
        rho_outer = outer_radius.new_tensor(support_boundary) / outer_coordinate_radius

        # The map preserves each ray, so a positive radial derivative is
        # sufficient for an orientation-preserving map inside every square or
        # hexagonal sector.  Bound the common Hermite endpoint slope by the
        # smallest secant over all three radial segments and all ray angles.
        # The closest point of either support polygon has norm r_u.  Its
        # farthest point is a rectangle corner or a regular-hexagon vertex.
        minimum_interface_norm = outer_coordinate_radius
        if triangular:
            maximum_interface_norm = (
                2.0 / math.sqrt(3.0)
            ) * outer_coordinate_radius
        else:
            aspect_x = lx / minimum_length
            aspect_y = ly / minimum_length
            maximum_interface_norm = math.hypot(
                aspect_x, aspect_y
            ) * outer_coordinate_radius
        core_secant = (
            core_radius / maximum_interface_norm
        ) / core_ratio
        shell_secant = (
            (outer_radius - core_radius) / maximum_interface_norm
        ) / (1.0 - core_ratio)
        exterior_secant = (
            rho_outer - outer_radius / minimum_interface_norm
        ) / (rho_outer - 1.0)
        minimum_radial_secant = torch.minimum(
            torch.minimum(core_secant, shell_secant), exterior_secant
        )
        if _as_float(minimum_radial_secant) <= 0.0:
            raise RuntimeError(
                "The double-matched radial targets are not strictly ordered."
            )
        requested_slope = outer_radius.new_tensor(self.matched_asr_G)
        monotonicity_margin = outer_radius.new_tensor(0.95)
        effective_radial_slope = torch.minimum(
            requested_slope,
            monotonicity_margin * minimum_radial_secant,
        )
        # G is an interface-resolution parameter, not a desirable slope at the
        # coordinate origin or periodic seam.  Keeping G there made det(J)
        # scale as G^2 and was badly conditioned for multilayer metals.  The
        # larger endpoint slopes below remain no greater than their segment's
        # minimum secant, so the Bernstein certificate in the profile applies.
        identity_slope = outer_radius.new_tensor(1.0)
        central_radial_slope = torch.minimum(
            identity_slope,
            monotonicity_margin * core_secant,
        )
        boundary_radial_slope = torch.minimum(
            identity_slope,
            monotonicity_margin * exterior_secant,
        )
        rho = support_radius / outer_coordinate_radius
        normal_weight = torch.ones_like(rho)
        interface_normal_u = torch.autograd.grad(
            rho,
            u_leaf,
            normal_weight,
            retain_graph=True,
            create_graph=False,
        )[0]
        interface_normal_v = torch.autograd.grad(
            rho,
            v_leaf,
            normal_weight,
            retain_graph=True,
            create_graph=False,
        )[0]
        epsilon = 64.0 * torch.finfo(torch.float64).eps
        active = rho > epsilon
        safe_rho = torch.clamp(rho, min=epsilon)
        q1_interface = q1 / safe_rho
        q2_interface = q2 / safe_rho
        interface_norm = torch.sqrt(
            torch.clamp(
                q1_interface**2
                + q2_interface**2
                + 2.0 * cosine * q1_interface * q2_interface,
                min=epsilon**2,
            )
        )
        circle_scale = torch.where(
            active,
            outer_radius / torch.clamp(interface_norm, min=epsilon),
            torch.ones_like(interface_norm),
        )
        core_target = torch.where(
            active,
            core_radius / torch.clamp(interface_norm, min=epsilon),
            torch.ones_like(interface_norm) * core_ratio,
        )
        radial = self._double_matched_radial_profile(
            rho,
            core_ratio=core_ratio,
            core_target=core_target,
            circle_scale=circle_scale,
            rho_outer=rho_outer,
            matched_slope=effective_radial_slope,
            central_slope=central_radial_slope,
            boundary_slope=boundary_radial_slope,
        )
        mapped_q1 = torch.where(active, radial * q1_interface, q1)
        mapped_q2 = torch.where(active, radial * q2_interface, q2)

        absolute_1 = 0.5 * lx + shift_i * lx + mapped_q1
        absolute_2 = 0.5 * ly + shift_j * ly + mapped_q2
        x = absolute_1 + cosine * absolute_2
        y = sine * absolute_2
        ones = torch.ones_like(x)
        differentiable_geometry = bool(
            core_radius.requires_grad or outer_radius.requires_grad
        )
        x_u = torch.autograd.grad(
            x,
            u_leaf,
            ones,
            retain_graph=True,
            create_graph=differentiable_geometry,
        )[0]
        x_v = torch.autograd.grad(
            x,
            v_leaf,
            ones,
            retain_graph=True,
            create_graph=differentiable_geometry,
        )[0]
        y_u = torch.autograd.grad(
            y,
            u_leaf,
            ones,
            retain_graph=True,
            create_graph=differentiable_geometry,
        )[0]
        y_v = torch.autograd.grad(
            y,
            v_leaf,
            ones,
            retain_graph=differentiable_geometry,
            create_graph=differentiable_geometry,
        )[0]
        det_j = x_u * y_v - x_v * y_u
        arrays = (x, y, x_u, x_v, y_u, y_v, det_j)
        if not all(bool(torch.all(torch.isfinite(value))) for value in arrays):
            raise RuntimeError("The double-matched circle map produced non-finite values.")
        minimum_jacobian = 1.0e-11 * sine
        if _as_float(torch.min(det_j)) <= minimum_jacobian:
            raise RuntimeError(
                "The double-matched circle map is not orientation-preserving; "
                "the monotone radial bound passed but a sector Jacobian did not. "
                "Use radial_mapping='outer' and report this geometry."
            )
        if not differentiable_geometry:
            x, y, x_u, x_v, y_u, y_v, det_j = (
                value.detach() for value in (x, y, x_u, x_v, y_u, y_v, det_j)
            )

        identity_u_breaks = torch.tensor(
            [0.0, lx], dtype=torch.float64, device=self._device
        )
        identity_v_breaks = torch.tensor(
            [0.0, ly], dtype=torch.float64, device=self._device
        )
        radial_breaks = torch.stack(
            (
                outer_radius.new_tensor(0.0),
                core_coordinate_radius,
                outer_coordinate_radius,
                outer_radius.new_tensor(support_boundary),
            )
        )
        return CircleASRMapping(
            u=u_axis.detach(),
            v=v_axis.detach(),
            tu=u_axis.detach(),
            tv=v_axis.detach(),
            x=x,
            y=y,
            x_u=x_u,
            x_v=x_v,
            y_u=y_u,
            y_v=y_v,
            det_j=det_j,
            tu_breaks=radial_breaks,
            tv_breaks=radial_breaks.clone(),
            u_breaks=identity_u_breaks,
            v_breaks=identity_v_breaks,
            matched_outer_mask=rho <= 1.0 + epsilon,
            matched_core_mask=rho <= core_ratio + epsilon,
            interface_normal_u=interface_normal_u.detach(),
            interface_normal_v=interface_normal_v.detach(),
            effective_radial_slope=effective_radial_slope.detach(),
            central_radial_slope=central_radial_slope.detach(),
            boundary_radial_slope=boundary_radial_slope.detach(),
            minimum_radial_secant=minimum_radial_secant.detach(),
        )

    def build_triangular_circle_asr_mapping(
        self,
        nx: int,
        ny: int,
        radius: float,
    ) -> CircleASRMapping:
        """D6-equivariant matched map for a triangular Bravais lattice.

        The computational interface is a regular hexagon defined by the
        support function of the Wigner--Seitz cell.  A direction-dependent
        radial Hermite map sends that hexagon exactly to the physical circle,
        has radial slope ``matched_asr_G`` at the interface, and becomes the
        identity (value and first derivative) at the Wigner--Seitz boundary.
        Consequently the nearest-image pieces join periodically.
        """
        self._validate_grid(int(self.order[0]), nx, "u")
        self._validate_grid(int(self.order[1]), ny, "v")
        lx, ly = _as_float(self.L[0]), _as_float(self.L[1])
        if not math.isclose(lx, ly, rel_tol=1.0e-8, abs_tol=1.0e-12):
            raise UnsupportedCombinationError(
                "Triangular matched-ASR requires equal primitive-vector lengths."
            )
        zeta = float(getattr(self, "zeta_deg", 90.0))
        if abs(zeta - 60.0) > 1.0e-7:
            raise UnsupportedCombinationError(
                "The D6 matched map requires a 60-degree triangular primitive cell."
            )
        radius, radius_value = _real_parameter_tensor(
            "radius",
            radius,
            dtype=torch.float64,
            device=self._device,
            allow_zero=False,
        )
        if 2.0 * radius_value >= lx:
            raise ValueError(
                "triangular matched-ASR requires non-touching circles: 2*radius < L."
            )
        if nx != ny:
            raise UnsupportedCombinationError(
                "The D6 matched map requires an equal u/v sampling grid."
            )

        u_axis = torch.arange(nx, dtype=torch.float64, device=self._device) * lx / nx
        v_axis = torch.arange(ny, dtype=torch.float64, device=self._device) * ly / ny
        u_grid, v_grid = torch.meshgrid(u_axis, v_axis, indexing="ij")
        u_leaf = u_grid.detach().requires_grad_(True)
        v_leaf = v_grid.detach().requires_grad_(True)

        # Choose the nearest periodic image of the cell-centred cylinder.
        q1_candidates: list[torch.Tensor] = []
        q2_candidates: list[torch.Tensor] = []
        r2_candidates: list[torch.Tensor] = []
        shifts: list[tuple[int, int]] = []
        cosine = 0.5
        sine = math.sqrt(3.0) / 2.0
        for shift_i in (-1, 0, 1):
            for shift_j in (-1, 0, 1):
                q1 = u_leaf - 0.5 * lx - shift_i * lx
                q2 = v_leaf - 0.5 * ly - shift_j * ly
                q1_candidates.append(q1)
                q2_candidates.append(q2)
                r2_candidates.append(q1**2 + q2**2 + 2.0 * cosine * q1 * q2)
                shifts.append((shift_i, shift_j))
        r2_stack = torch.stack(r2_candidates, dim=0)
        nearest = torch.argmin(r2_stack, dim=0, keepdim=True)
        q1 = torch.gather(torch.stack(q1_candidates, dim=0), 0, nearest)[0]
        q2 = torch.gather(torch.stack(q2_candidates, dim=0), 0, nearest)[0]
        shift_i_values = torch.tensor(
            [item[0] for item in shifts], dtype=torch.float64, device=self._device
        )[:, None, None]
        shift_j_values = torch.tensor(
            [item[1] for item in shifts], dtype=torch.float64, device=self._device
        )[:, None, None]
        shift_i = torch.gather(shift_i_values.expand(-1, nx, ny), 0, nearest)[0]
        shift_j = torch.gather(shift_j_values.expand(-1, nx, ny), 0, nearest)[0]

        # h(q)=L/2 is exactly the Wigner--Seitz boundary.  It is invariant
        # under all six rotations and six reflections of D6.
        hex_radius = torch.maximum(
            torch.maximum(torch.abs(q1 + 0.5 * q2), torch.abs(0.5 * q1 + q2)),
            torch.abs(0.5 * (q1 - q2)),
        )
        rho = hex_radius / radius
        active = rho > 64.0 * torch.finfo(torch.float64).eps
        safe_rho = torch.clamp(rho, min=64.0 * torch.finfo(torch.float64).eps)
        q1_interface = q1 / safe_rho
        q2_interface = q2 / safe_rho
        norm_floor = 64.0 * torch.finfo(torch.float64).eps
        interface_norm = torch.sqrt(
            torch.clamp(
                q1_interface**2
                + q2_interface**2
                + q1_interface * q2_interface,
                min=norm_floor**2,
            )
        )
        safe_interface_norm = torch.clamp(interface_norm, min=norm_floor)
        circle_scale = torch.where(active, radius / safe_interface_norm, 1.0)
        rho_outer = 0.5 * lx / radius

        t_inner = torch.clamp(rho, 0.0, 1.0)
        radial_inner = self._cubic_hermite(
            t_inner, 0.0, 1.0, circle_scale, self.matched_asr_G, 1.0
        )
        outer_span = rho_outer - 1.0
        t_outer = torch.clamp((rho - 1.0) / outer_span, 0.0, 1.0)
        radial_outer = self._cubic_hermite(
            t_outer,
            circle_scale,
            self.matched_asr_G,
            rho_outer,
            1.0,
            outer_span,
        )
        radial = torch.where(
            rho <= 1.0,
            radial_inner,
            torch.where(rho < rho_outer, radial_outer, rho),
        )
        mapped_q1 = torch.where(active, radial * q1_interface, q1)
        mapped_q2 = torch.where(active, radial * q2_interface, q2)

        absolute_1 = 0.5 * lx + shift_i * lx + mapped_q1
        absolute_2 = 0.5 * ly + shift_j * ly + mapped_q2
        x = absolute_1 + cosine * absolute_2
        y = sine * absolute_2
        ones = torch.ones_like(x)
        # Radius optimization needs the mixed derivatives d/dR(dx/du), etc.
        # create_graph=False would make the sampled Jacobian constant in R.
        differentiable_geometry = bool(radius.requires_grad)
        x_u = torch.autograd.grad(
            x,
            u_leaf,
            ones,
            retain_graph=True,
            create_graph=differentiable_geometry,
        )[0]
        x_v = torch.autograd.grad(
            x,
            v_leaf,
            ones,
            retain_graph=True,
            create_graph=differentiable_geometry,
        )[0]
        y_u = torch.autograd.grad(
            y,
            u_leaf,
            ones,
            retain_graph=True,
            create_graph=differentiable_geometry,
        )[0]
        y_v = torch.autograd.grad(
            y,
            v_leaf,
            ones,
            retain_graph=differentiable_geometry,
            create_graph=differentiable_geometry,
        )[0]
        det_j = x_u * y_v - x_v * y_u
        arrays = (x, y, x_u, x_v, y_u, y_v, det_j)
        if not all(bool(torch.all(torch.isfinite(value))) for value in arrays):
            raise RuntimeError("The triangular matched map produced non-finite values.")
        minimum_jacobian = 1.0e-11 * sine
        if _as_float(torch.min(det_j)) <= minimum_jacobian:
            raise RuntimeError(
                "The triangular matched map is not orientation-preserving; "
                "increase circle_G or reduce the circle radius."
            )
        if not differentiable_geometry:
            # u_leaf/v_leaf exist only to obtain the spatial Jacobian.  Do not
            # keep an otherwise useless graph when radius is a fixed constant.
            x, y, x_u, x_v, y_u, y_v, det_j = (
                value.detach() for value in (x, y, x_u, x_v, y_u, y_v, det_j)
            )

        identity_breaks = torch.tensor(
            [0.0, lx], dtype=torch.float64, device=self._device
        )
        return CircleASRMapping(
            u=u_axis.detach(),
            v=v_axis.detach(),
            tu=u_axis.detach(),
            tv=v_axis.detach(),
            x=x,
            y=y,
            x_u=x_u,
            x_v=x_v,
            y_u=y_u,
            y_v=y_v,
            det_j=det_j,
            tu_breaks=torch.stack(
                (radius.new_tensor(0.0), radius, radius.new_tensor(0.5 * lx))
            ),
            tv_breaks=torch.stack(
                (radius.new_tensor(0.0), radius, radius.new_tensor(0.5 * ly))
            ),
            u_breaks=identity_breaks,
            v_breaks=identity_breaks.clone(),
        )

    def _periodic_circle_mask(
        self, x: torch.Tensor, y: torch.Tensor, radius: float
    ) -> torch.Tensor:
        """Circle mask using the nearest image of the primitive-cell centre."""
        lx, ly = _as_float(self.L[0]), _as_float(self.L[1])
        cosine = float(getattr(self, "cos_zeta", 0.0))
        sine = float(getattr(self, "sin_zeta", 1.0))
        center_x = 0.5 * (lx + cosine * ly)
        center_y = 0.5 * sine * ly
        distances: list[torch.Tensor] = []
        for shift_i in (-1, 0, 1):
            for shift_j in (-1, 0, 1):
                image_x = center_x + shift_i * lx + shift_j * cosine * ly
                image_y = center_y + shift_j * sine * ly
                distances.append((x - image_x) ** 2 + (y - image_y) ** 2)
        return torch.min(torch.stack(distances), dim=0).values <= radius**2
