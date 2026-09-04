"""Layer-aware formulation selection and the public AutoRCWA facade."""

from __future__ import annotations

import math

import torch
import torcwa

from .asr import CustomRCWA_ASR_FR
from .asr_maps import ASRMapping, CircleASRMapping
from .config import (
    ASROptions, Circle, Geometry, GroupTheoryOptions, Homogeneous,
    Lattice, LayerSpec, Material, NVMOptions, OutputSpec, Raster,
    Rectangle, Square, UnsupportedCombinationError,
    _ORIGINAL_TORCWA_RCWA, _as_float, _finite_positive,
    _normalize_method, _normalize_polarization, _normalize_smatrix_size,
)
from .nvm import CustomRCWA_NVM

class AutoRCWA(CustomRCWA_NVM):
    """
    Layer-aware RCWA front end with deterministic formulation selection.

    Auto selection is an eligibility heuristic, not a claim of globally optimal
    convergence. It never infers a differentiable geometry from raster values:
    pass Rectangle/Square/Circle metadata when formulation selection is wanted.
    """

    _require_kvectors = CustomRCWA_ASR_FR._require_kvectors
    _validate_grid = staticmethod(CustomRCWA_ASR_FR._validate_grid)
    _piecewise_asr_map = CustomRCWA_ASR_FR._piecewise_asr_map
    build_asr_mapping = CustomRCWA_ASR_FR.build_asr_mapping
    _matched_circle_axis = staticmethod(CustomRCWA_ASR_FR._matched_circle_axis)
    build_circle_asr_mapping = CustomRCWA_ASR_FR.build_circle_asr_mapping
    _cubic_hermite = staticmethod(CustomRCWA_ASR_FR._cubic_hermite)
    _quintic_hermite_zero_curvature = staticmethod(
        CustomRCWA_ASR_FR._quintic_hermite_zero_curvature
    )
    _double_matched_radial_profile = (
        CustomRCWA_ASR_FR._double_matched_radial_profile
    )
    build_double_matched_circle_asr_mapping = (
        CustomRCWA_ASR_FR.build_double_matched_circle_asr_mapping
    )
    build_triangular_circle_asr_mapping = (
        CustomRCWA_ASR_FR.build_triangular_circle_asr_mapping
    )
    _periodic_circle_mask = CustomRCWA_ASR_FR._periodic_circle_mask
    _axis_toeplitz_samples = CustomRCWA_ASR_FR._axis_toeplitz_samples
    _assemble_matrix_valued_toeplitz = (
        CustomRCWA_ASR_FR._assemble_matrix_valued_toeplitz
    )
    _symmetric_factorized_transverse_tensor = (
        CustomRCWA_ASR_FR._symmetric_factorized_transverse_tensor
    )
    _generalized_li_factorized_transverse_tensor = (
        CustomRCWA_ASR_FR._generalized_li_factorized_transverse_tensor
    )
    _build_circle_conversion_matrices = (
        CustomRCWA_ASR_FR._build_circle_conversion_matrices
    )
    _build_circle_asr_pq = CustomRCWA_ASR_FR._build_circle_asr_pq
    add_layer_circle_asr = CustomRCWA_ASR_FR.add_layer_circle_asr
    add_layer_circle_shell_asr = CustomRCWA_ASR_FR.add_layer_circle_shell_asr
    _factorized_bttb = CustomRCWA_ASR_FR._factorized_bttb
    _build_conversion_matrix_T = CustomRCWA_ASR_FR._build_conversion_matrix_T
    _build_conversion_matrix_Tz = CustomRCWA_ASR_FR._build_conversion_matrix_Tz
    _build_asr_pq = CustomRCWA_ASR_FR._build_asr_pq
    add_layer_rect_asr = CustomRCWA_ASR_FR.add_layer_rect_asr
    add_layer_metal_patch_asr = CustomRCWA_ASR_FR.add_layer_metal_patch_asr

    def __init__(
        self,
        freq,
        order,
        L=None,
        *,
        lattice: Lattice | None = None,
        method: str = "auto",
        outputs: OutputSpec | None = None,
        asr: ASROptions | None = None,
        nvm: NVMOptions | None = None,
        cascade: str = "redheffer",
        smatrix_size: str | None = None,
        group_theory: GroupTheoryOptions | bool | None = None,
        **kwargs,
    ):
        if lattice is None:
            if L is None or len(L) != 2:
                raise ValueError("Pass L=[Lx,Ly] or a Lattice object.")
            lattice = Lattice.rectangular(_as_float(L[0]), _as_float(L[1]))
        elif L is not None:
            lengths_match = math.isclose(
                _as_float(L[0]), lattice.length1, rel_tol=1.0e-8
            ) and math.isclose(
                _as_float(L[1]), lattice.length2, rel_tol=1.0e-8
            )
            if not lengths_match:
                raise ValueError("L disagrees with the supplied primitive vectors.")

        self.lattice_spec = lattice
        self.default_method = _normalize_method(method)
        outputs_supplied = outputs is not None
        if outputs is None:
            requested_size = _normalize_smatrix_size(smatrix_size or "full")
            self.output_spec = OutputSpec(
                smatrix_size=requested_size,
                fields="none" if requested_size != "full" else "all",
            )
        else:
            self.output_spec = outputs
            requested_size = self.output_spec.normalized_smatrix_size()
            if (
                smatrix_size is not None
                and _normalize_smatrix_size(smatrix_size) != requested_size
            ):
                raise ValueError("smatrix_size disagrees with OutputSpec.smatrix_size.")
        if not isinstance(self.output_spec.smatrix, bool):
            raise TypeError("OutputSpec.smatrix must be bool.")
        legacy_field_values = [
            bool(kwargs[name])
            for name in ("enable_fields", "store_mode_couplings")
            if name in kwargs
        ]
        if legacy_field_values and any(
            value != legacy_field_values[0] for value in legacy_field_values[1:]
        ):
            raise ValueError("enable_fields and store_mode_couplings disagree.")
        legacy_fields = legacy_field_values[0] if legacy_field_values else None
        self.field_regions = self.output_spec.normalized_fields()
        if not outputs_supplied and legacy_fields is not None:
            # Preserve the pre-OutputSpec API.  In particular,
            # smatrix_size='quarter', enable_fields=True must enable every
            # field region instead of inheriting the partial-S default 'none'.
            self.field_regions = "all" if legacy_fields else "none"
            self.output_spec = OutputSpec(
                smatrix=self.output_spec.smatrix,
                smatrix_size=requested_size,
                fields=self.field_regions,
            )
        elif legacy_fields is not None and legacy_fields != (
            self.field_regions != "none"
        ):
            raise ValueError(
                "enable_fields/store_mode_couplings disagrees with "
                "OutputSpec.fields."
            )
        if not self.output_spec.smatrix and self.field_regions == "none":
            raise ValueError(
                "OutputSpec requests neither an S matrix nor electromagnetic fields."
            )
        self.asr_options = asr or ASROptions()
        self.nvm_options = nvm or NVMOptions()
        if not 0.0 < float(self.asr_options.G) < 1.0:
            raise ValueError("ASROptions.G must be in (0,1).")
        if not 0.0 < float(self.asr_options.circle_G) < 1.0:
            raise ValueError("ASROptions.circle_G must be in (0,1).")
        if (
            len(self.asr_options.grid) != 2
            or len(self.nvm_options.grid) != 2
            or any(int(v) <= 0 for v in (*self.asr_options.grid, *self.nvm_options.grid))
        ):
            raise ValueError("ASR/NVM grids must contain two positive integers.")
        if (
            not isinstance(self.nvm_options.lanczos_power, int)
            or self.nvm_options.lanczos_power < 0
        ):
            raise ValueError("NVM lanczos_power must be a nonnegative integer.")

        if "enable_fields" not in kwargs and "store_mode_couplings" not in kwargs:
            # A partial public S matrix can still be combined with field
            # reconstruction.  The solver retains private bidirectional data
            # while exposing only the requested public S blocks.
            kwargs["store_mode_couplings"] = self.field_regions != "none"
        if group_theory is not None:
            group_options = (
                GroupTheoryOptions(enabled=group_theory)
                if isinstance(group_theory, bool)
                else group_theory
            )
            if not isinstance(group_options, GroupTheoryOptions):
                raise TypeError(
                    "group_theory must be bool, GroupTheoryOptions, or None."
                )
            requested_group_values = {
                "use_group_theory": bool(group_options.enabled),
                "group_theory_symmetry": group_options.symmetry,
                "group_theory_strict": bool(group_options.strict),
                "group_theory_tolerance": float(
                    group_options.residual_tolerance
                ),
                "polarization_reduction": _normalize_polarization(
                    group_options.polarization
                ),
            }
            for name, value in requested_group_values.items():
                if name in kwargs and kwargs[name] != value:
                    raise ValueError(
                        f"{name} disagrees with the group_theory options object."
                    )
                kwargs[name] = value
        kwargs["zeta_deg"] = lattice.angle_deg
        kwargs["cascade"] = cascade
        kwargs["smatrix_size"] = requested_size
        kwargs["expose_smatrix"] = bool(self.output_spec.smatrix)
        super().__init__(
            freq,
            order,
            [lattice.length1, lattice.length2],
            **kwargs,
        )
        self.lattice_kind = lattice.kind
        self.asr_G = float(self.asr_options.G)
        self.matched_asr_G = float(self.asr_options.circle_G)
        # CustomRCWA_ASR_FR initializes this legacy override in its own
        # constructor, but AutoRCWA composes the ASR methods without calling
        # that constructor directly.
        self.asr_quadrature_grid = None
        self.asr_mappings: list[ASRMapping | CircleASRMapping] = []
        self.asr_T_matrices: list[torch.Tensor] = []
        self.asr_Tz_matrices: list[torch.Tensor | None] = []
        self.asr_condition_numbers: list[torch.Tensor | None] = []
        self.asr_material_tensors: list[dict[str, torch.Tensor]] = []
        self.E_eigvec_uv: list[torch.Tensor] = []
        self.H_eigvec_uv: list[torch.Tensor] = []
        self.selection_notes: list[str] = []
        if abs(lattice.basis_rotation_deg) > 1.0e-8:
            self.selection_notes.append(
                "Incidence azimuth and Cartesian field coordinates are expressed "
                "in the local frame where a1 points along +x."
            )

    def _geometry_rectangle(self, geometry: Geometry) -> Rectangle | None:
        if isinstance(geometry, Square):
            return geometry.as_rectangle()
        return geometry if isinstance(geometry, Rectangle) else None

    def _center_local(
        self, center: tuple[float, float], coordinates: str
    ) -> tuple[float, float]:
        if coordinates == "cartesian":
            return center
        a1, a2 = self.lattice_spec.local_vectors
        # All backends use the same canonical primitive cell
        # {xi*a1 + eta*a2 | 0 <= xi, eta < 1}.  Keeping this convention here
        # prevents a half-period translation when an automatic stack mixes ASR,
        # NVM and ordinary raster layers.
        xi = center[0] % 1.0
        eta = center[1] % 1.0
        return (
            xi * a1[0] + eta * a2[0],
            xi * a1[1] + eta * a2[1],
        )

    def _is_centered(self, geometry: Rectangle) -> bool:
        if geometry.coordinates == "fractional":
            return (
                abs((geometry.center[0] % 1.0) - 0.5) <= 1.0e-10
                and abs((geometry.center[1] % 1.0) - 0.5) <= 1.0e-10
            )
        a1, a2 = self.lattice_spec.local_vectors
        cell_center = (
            0.5 * (a1[0] + a2[0]),
            0.5 * (a1[1] + a2[1]),
        )
        return (
            abs(geometry.center[0] - cell_center[0]) <= 1.0e-10
            and abs(geometry.center[1] - cell_center[1]) <= 1.0e-10
        )

    @staticmethod
    def _is_axis_aligned(geometry: Rectangle) -> bool:
        # The current ASR size tuple is (physical x width, physical y width),
        # so a 90-degree rotation is not silently reinterpreted.
        return abs(math.remainder(float(geometry.angle_deg), 180.0)) <= 1.0e-10

    @staticmethod
    def _unity_mu(material: Material) -> bool:
        try:
            return math.isclose(_as_float(material.mu), 1.0, abs_tol=1.0e-12)
        except (TypeError, ValueError, RuntimeError):
            return False

    def select_method(self, spec: LayerSpec) -> tuple[str, str]:
        requested = _normalize_method(spec.method)
        if requested == "auto":
            requested = self.default_method
        geometry = spec.geometry
        rectangle = self._geometry_rectangle(geometry)

        if requested == "auto":
            if isinstance(geometry, Homogeneous):
                return "standard", "HOMOGENEOUS_LAYER"
            if rectangle is not None:
                eligible = (
                    self.lattice_spec.is_orthogonal
                    and self._is_centered(rectangle)
                    and self._is_axis_aligned(rectangle)
                    and rectangle.size[0] < self.lattice_spec.length1
                    and rectangle.size[1] < self.lattice_spec.length2
                )
                if eligible:
                    return "asr-fr", "RECTILINEAR_SEPARABLE_BOUNDARY"
                return "standard", "RECTANGLE_NOT_ASR_ELIGIBLE"
            if isinstance(geometry, Circle):
                nvm_material = self._unity_mu(spec.background) and self._unity_mu(
                    spec.inclusion
                )
                if nvm_material and self.lattice_spec.kind in {
                    "square",
                    "rectangular",
                    "triangular",
                }:
                    return "nvm", "CURVED_ANALYTIC_BOUNDARY"
                return "standard", "CIRCLE_NVM_INELIGIBLE"
            return "standard", "RASTER_HAS_NO_ANALYTIC_SHAPE_METADATA"

        if requested == "asr-fr":
            if rectangle is None:
                raise UnsupportedCombinationError(
                    "ASR-FR requires a Rectangle or Square geometry."
                )
            if not self.lattice_spec.is_orthogonal:
                raise UnsupportedCombinationError(
                    "ASR-FR uses a separable orthogonal map and cannot be used "
                    "with a triangular/oblique primitive cell."
                )
            if not self._is_centered(rectangle) or not self._is_axis_aligned(
                rectangle
            ):
                raise UnsupportedCombinationError(
                    "This ASR implementation requires a centered axis-aligned rectangle."
                )
            return "asr-fr", "EXPLICIT_ASR"

        if requested == "matched-asr":
            if not isinstance(geometry, Circle):
                raise UnsupportedCombinationError(
                    "matched-ASR requires Circle geometry."
                )
            if not (
                self.lattice_spec.is_orthogonal
                or self.lattice_spec.kind == "triangular"
            ):
                raise UnsupportedCombinationError(
                    "matched-ASR requires an orthogonal cell or a 60-degree "
                    "equal-length triangular primitive cell."
                )
            if not self._is_centered(geometry):
                raise UnsupportedCombinationError(
                    "matched-ASR currently requires a circle at the cell center."
                )
            if 2.0 * _finite_positive("radius", geometry.radius) >= min(
                self.lattice_spec.length1, self.lattice_spec.length2
            ):
                raise UnsupportedCombinationError(
                    "matched-ASR requires a non-touching circle inside the cell."
                )
            return "matched-asr", "EXPLICIT_MATCHED_CIRCLE_ASR"

        if requested == "nvm":
            if not isinstance(geometry, Circle):
                raise UnsupportedCombinationError(
                    "The analytic NVM backend currently requires Circle geometry."
                )
            if not (
                self._unity_mu(spec.background) and self._unity_mu(spec.inclusion)
            ):
                raise UnsupportedCombinationError(
                    "The current circle NVM formulation is nonmagnetic (mu=1)."
                )
            return "nvm", "EXPLICIT_NVM"

        return "standard", "EXPLICIT_STANDARD"

    def _raster_grid(self) -> tuple[int, int]:
        return (
            max(
                32,
                int(self.asr_options.grid[0]),
                4 * int(self.order[0]) + 4,
            ),
            max(
                32,
                int(self.asr_options.grid[1]),
                4 * int(self.order[1]) + 4,
            ),
        )

    def _rasterize(
        self, geometry: Rectangle | Circle, background: Material, inclusion: Material
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nx, ny = self._raster_grid()
        l1, l2 = self.lattice_spec.length1, self.lattice_spec.length2
        angle = math.radians(self.lattice_spec.angle_deg)
        xi = (
            torch.arange(nx, dtype=torch.float64, device=self._device) / nx
        ) * l1
        eta = (
            torch.arange(ny, dtype=torch.float64, device=self._device) / ny
        ) * l2
        xi_grid, eta_grid = torch.meshgrid(xi, eta, indexing="ij")
        x_grid = xi_grid + eta_grid * math.cos(angle)
        y_grid = eta_grid * math.sin(angle)
        a1, a2 = self.lattice_spec.local_vectors
        center = self._center_local(geometry.center, geometry.coordinates)
        candidates: list[torch.Tensor] = []
        for shift_i in range(-2, 3):
            for shift_j in range(-2, 3):
                cx = center[0] + shift_i * a1[0] + shift_j * a2[0]
                cy = center[1] + shift_i * a1[1] + shift_j * a2[1]
                dx, dy = x_grid - cx, y_grid - cy
                if isinstance(geometry, Circle):
                    candidates.append(dx**2 + dy**2 <= geometry.radius**2)
                else:
                    theta = math.radians(geometry.angle_deg)
                    local_x = math.cos(theta) * dx + math.sin(theta) * dy
                    local_y = -math.sin(theta) * dx + math.cos(theta) * dy
                    candidates.append(
                        (torch.abs(local_x) <= 0.5 * geometry.size[0])
                        & (torch.abs(local_y) <= 0.5 * geometry.size[1])
                    )
        inside = torch.any(torch.stack(candidates), dim=0)
        eps_grid = torch.where(
            inside,
            torch.as_tensor(inclusion.eps, dtype=self._dtype, device=self._device),
            torch.as_tensor(background.eps, dtype=self._dtype, device=self._device),
        )
        mu_grid = torch.where(
            inside,
            torch.as_tensor(inclusion.mu, dtype=self._dtype, device=self._device),
            torch.as_tensor(background.mu, dtype=self._dtype, device=self._device),
        )
        return eps_grid, mu_grid

    def add_structured_layer(self, spec: LayerSpec) -> str:
        if not hasattr(self, "Kx_norm"):
            raise RuntimeError(
                "Call set_incident_angle() before add_structured_layer()."
            )
        if (
            self.polarization_reduction is not None
            and not isinstance(spec.geometry, Circle)
        ):
            raise UnsupportedCombinationError(
                "Polarization reduction currently accepts analytic Circle geometry only."
            )
        selected, reason = self.select_method(spec)
        geometry = spec.geometry
        rectangle = self._geometry_rectangle(geometry)
        if self.use_group_theory and self.group_theory_symmetry == "d6":
            if not isinstance(geometry, Circle) or selected not in {
                "nvm",
                "matched-asr",
            }:
                raise UnsupportedCombinationError(
                    "D6 reduction currently requires a centered analytic Circle "
                    "with method='nvm' or method='matched-asr'."
                )
            if not self._is_centered(geometry):
                raise UnsupportedCombinationError(
                    "D6 reduction requires the circle at the primitive-cell center."
                )
            if self.lattice_kind != "triangular":
                raise UnsupportedCombinationError(
                    "D6 reduction requires a 60-degree equal-length triangular cell."
                )
        if spec.factorization_rules is not None and selected not in {
            "asr-fr",
            "matched-asr",
        }:
            raise UnsupportedCombinationError(
                "factorization_rules is an ASR-FR option and cannot be applied "
                f"to the selected {selected!r} backend."
            )

        if selected == "asr-fr":
            assert rectangle is not None
            fill_x = rectangle.size[0] / self.lattice_spec.length1
            fill_y = rectangle.size[1] / self.lattice_spec.length2
            if not 0.0 < fill_x < 1.0 or not 0.0 < fill_y < 1.0:
                raise ValueError("ASR rectangle must fit strictly inside the cell.")
            factorization = (
                self.asr_options.factorization_rules
                if spec.factorization_rules is None
                else bool(spec.factorization_rules)
            )
            nx, ny = (int(v) for v in self.asr_options.grid)
            self.add_layer_rect_asr(
                spec.thickness,
                spec.background.eps,
                spec.inclusion.eps,
                fill_x,
                fill_y,
                mu_bg=spec.background.mu,
                mu_rect=spec.inclusion.mu,
                nx=nx,
                ny=ny,
                factorization_rules=factorization,
            )
        elif selected == "matched-asr":
            assert isinstance(geometry, Circle)
            factorization = (
                self.asr_options.factorization_rules
                if spec.factorization_rules is None
                else bool(spec.factorization_rules)
            )
            nx, ny = (int(v) for v in self.asr_options.grid)
            self.add_layer_circle_asr(
                spec.thickness,
                geometry.radius,
                spec.background.eps,
                spec.inclusion.eps,
                mu_bg=spec.background.mu,
                mu_cyl=spec.inclusion.mu,
                nx=nx,
                ny=ny,
                factorization_rules=factorization,
            )
        elif selected == "nvm":
            assert isinstance(geometry, Circle)
            center = self._center_local(geometry.center, geometry.coordinates)
            nx, ny = (int(v) for v in self.nvm_options.grid)
            self.add_layer_circle_nvm(
                spec.thickness,
                geometry.radius,
                spec.background.eps,
                spec.inclusion.eps,
                centers=(center,),
                use_lanczos=self.nvm_options.use_lanczos,
                lanczos_power=self.nvm_options.lanczos_power,
                nx=nx,
                ny=ny,
            )
        else:
            if (
                isinstance(geometry, Circle)
                and torch.is_tensor(geometry.radius)
                and geometry.radius.requires_grad
            ):
                raise UnsupportedCombinationError(
                    "A hard rasterized circle has a discontinuous occupancy mask "
                    "and therefore no useful radius gradient. Select method='nvm' "
                    "or method='matched-asr' for differentiable radius optimization."
                )
            if isinstance(geometry, Homogeneous):
                self.add_layer(
                    spec.thickness,
                    eps=geometry.material.eps,
                    mu=geometry.material.mu,
                )
            elif isinstance(geometry, Raster):
                eps = geometry.eps.to(dtype=self._dtype, device=self._device)
                if eps.ndim != 2:
                    raise ValueError("Raster.eps must be a two-dimensional tensor.")
                minimum_x = max(2 * int(self.order[0]) + 1, 4)
                minimum_y = max(2 * int(self.order[1]) + 1, 4)
                if eps.shape[0] < minimum_x or eps.shape[1] < minimum_y:
                    raise ValueError(
                        f"Raster grid {tuple(eps.shape)} is too small for order "
                        f"{tuple(self.order)}."
                    )
                self.add_layer(spec.thickness, eps=eps, mu=geometry.mu)
            else:
                assert rectangle is not None or isinstance(geometry, Circle)
                analytic = rectangle if rectangle is not None else geometry
                eps, mu = self._rasterize(
                    analytic, spec.background, spec.inclusion
                )
                self.add_layer(spec.thickness, eps=eps, mu=mu)

        record = self.layer_records[-1]
        record.reason = reason
        record.label = spec.label
        rectangle_is_square = rectangle is not None and math.isclose(
            rectangle.size[0],
            rectangle.size[1],
            rel_tol=1.0e-8,
            abs_tol=1.0e-12,
        )
        record.shape = (
            "square"
            if isinstance(spec.geometry, Square) or rectangle_is_square
            else spec.geometry.__class__.__name__.lower()
        )
        record.options["selected_by"] = "auto" if spec.method == "auto" else "explicit"
        return selected

    def explain_plan(self) -> list[dict[str, object]]:
        return [
            {
                "index": record.index,
                "label": record.label,
                "method": record.method,
                "shape": record.shape,
                "lattice": record.lattice,
                "reason": record.reason,
                "options": dict(record.options),
                "warnings": list(record.warnings),
            }
            for record in self.layer_records
        ]


# Short aliases for existing import styles.
ASRRCWA = CustomRCWA_ASR_FR
NVMRCWA = CustomRCWA_NVM
rcwa = AutoRCWA


def install_as_torcwa_rcwa(kind: str = "auto"):
    """
    Explicitly replace torcwa.rcwa for legacy programs.

    New code should import the desired class directly instead of monkey-patching.
    """
    normalized = kind.strip().lower()
    if normalized in {"asr", "asr-fr", "asr_fr"}:
        selected = CustomRCWA_ASR_FR
    elif normalized in {"nvm", "normal-vector", "normal_vector"}:
        selected = CustomRCWA_NVM
    elif normalized in {"auto", "adaptive"}:
        selected = AutoRCWA
    elif normalized in {"standard", "torcwa", "original"}:
        selected = _ORIGINAL_TORCWA_RCWA
    else:
        raise ValueError("kind must be auto, standard, asr, or nvm.")
    torcwa.rcwa = selected
    return selected
