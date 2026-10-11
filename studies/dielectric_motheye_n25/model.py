"""Identical physical stack and power normalization for the two formulations."""
from __future__ import annotations

import gc
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from studies.shared.convergence_io import write_json

METHODS = {"nv": "nvm", "asr": "matched-asr"}
POWERS = ("reflectance", "transmittance", "absorptance")


@dataclass(frozen=True)
class Geometry:
    index: float = 2.5
    period_nm: float = 200.
    height_nm: float = 500.
    tip_radius_nm: float = 5.
    base_radius_nm: float = 95.
    profile_power: float = 1.
    asr_circle_g: float = .03

    def validate(self):
        if any(not math.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("All geometry/index parameters must be finite and positive.")
        if self.tip_radius_nm > self.base_radius_nm:
            raise ValueError("Tip radius must not exceed the base radius.")
        if 2 * self.base_radius_nm >= self.period_nm:
            raise ValueError("Require 2*base_radius < period (a positive inter-pillar gap).")
        if not 0 < self.asr_circle_g < 1:
            raise ValueError("ASR G must lie in (0,1).")

    def radius(self, j: int, slices: int):
        return self.tip_radius_nm + (self.base_radius_nm - self.tip_radius_nm) * ((j + .5) / slices) ** self.profile_power

    def fill_fraction(self, lattice: str):
        area = self.period_nm ** 2 * (math.sqrt(3) / 2 if lattice == "hexagonal" else 1)
        return math.pi * self.base_radius_nm ** 2 / area


@dataclass(frozen=True)
class Numerical:
    order: int
    slices: int
    grid: int


def finite_powers(row: dict) -> dict:
    values = {name: float(row[name]) for name in POWERS}
    if not all(math.isfinite(v) for v in values.values()):
        raise ValueError("Nonfinite power in a solver result.")
    return values


def physical(row: dict, energy_tolerance: float) -> bool:
    p = finite_powers(row)
    return (all(-1e-7 <= p[key] <= 1 + 1e-7 for key in ("reflectance", "transmittance"))
            and abs(p["reflectance"] + p["transmittance"] - 1) <= energy_tolerance)


def adjacent_error(coarse, fine):
    if not coarse or len(coarse) != len(fine):
        raise ValueError("Spectra must be nonempty with matching lengths.")
    errors = dict.fromkeys(POWERS, 0.)
    for a, b in zip(coarse, fine):
        if a["wavelength_nm"] != b["wavelength_nm"]:
            raise ValueError("Spectra use different wavelengths.")
        av, bv = finite_powers(a), finite_powers(b)
        for name in POWERS:
            errors[name] = max(errors[name], abs(av[name] - bv[name]))
    return max(errors.values()), errors


def choose_candidate(candidates, spectra, tolerance, energy_tolerance):
    comparisons = []
    for coarse, fine in zip(candidates, candidates[1:]):
        maximum, per_metric = adjacent_error(spectra[coarse], spectra[fine])
        passive = all(physical(row, energy_tolerance) for row in spectra[coarse] + spectra[fine])
        comparisons.append({"coarse": coarse, "fine": fine, "max_abs_change": maximum,
                            "per_metric": per_metric, "physical": passive,
                            "passed": passive and maximum <= tolerance})
    passed = len(comparisons) >= 2 and all(c["passed"] for c in comparisons[-2:])
    return candidates[-2] if passed else candidates[-1], passed, comparisons


def trapezoid_weights(wavelengths):
    if len(wavelengths) == 1:
        return [1.]
    span = wavelengths[-1] - wavelengths[0]
    weights = [0.] * len(wavelengths)
    for i, (a, b) in enumerate(zip(wavelengths, wavelengths[1:])):
        weights[i] += (b - a) / (2 * span)
        weights[i + 1] += (b - a) / (2 * span)
    return weights


def mean_reflectance(rows):
    return sum(weight * row["reflectance"] for weight, row in
               zip(trapezoid_weights([r["wavelength_nm"] for r in rows]), rows))


def synchronize(device):
    if device.type == "cuda":
        import torch
        torch.cuda.synchronize(device)


def release(device):
    gc.collect()
    if device.type == "cuda":
        import torch
        torch.cuda.empty_cache()


def solve(wavelength, numerical: Numerical, geometry: Geometry, *, method: str,
          lattice: str, device, use_symmetry=True, parameters=None,
          square_asr_profile="radial", checkpoint_layers=True):
    """Return scalar diagnostics and differentiable R. No global no_grad: ASR
    needs autograd for its spatial Jacobian; fixed-geometry maps detach it.
    """
    import torch
    from rcwa_solver_auto import (ASROptions, AutoRCWA, Circle, GroupTheoryOptions,
                                  Lattice, LayerSpec, Material, NVMOptions, OutputSpec)
    from studies.pmma_gold_motheye.common import _discard_flux_only_auxiliary, _zero_order_x_source

    if method not in METHODS or lattice not in {"hexagonal", "square"}:
        raise ValueError("Unknown method or lattice.")
    if min(numerical.order, numerical.slices, numerical.grid) <= 0:
        raise ValueError("Numerical parameters must be positive.")
    g = geometry
    parameters = parameters or {}
    height = torch.as_tensor(parameters.get("height_nm", g.height_nm), dtype=torch.float64, device=device)
    base = torch.as_tensor(parameters.get("base_radius_nm", g.base_radius_nm), dtype=torch.float64, device=device)
    synchronize(device)
    started = time.perf_counter()
    def make_sim():
        sim = AutoRCWA(
            freq=g.period_nm / wavelength, order=[numerical.order] * 2,
            lattice=Lattice.triangular(1.) if lattice == "hexagonal" else Lattice.square(1.),
            outputs=OutputSpec(smatrix_size="half", fields="none"), cascade="redheffer",
            asr=ASROptions(circle_G=g.asr_circle_g, grid=(numerical.grid,) * 2,
                          circle_profile=square_asr_profile if lattice == "square" else "equalized",
                          factorization_rules=True),
            nvm=NVMOptions(grid=(numerical.grid,) * 2, use_lanczos=False),
            group_theory=GroupTheoryOptions(enabled=use_symmetry,
                symmetry="d6" if lattice == "hexagonal" else "c2v",
                polarization="x" if use_symmetry else None, strict=use_symmetry),
            dtype=torch.complex128, device=device, verify_cascade=False)
        sim.add_input_layer(eps=1.)
        sim.add_output_layer(eps=g.index ** 2)
        sim.set_incident_angle(0., 0.)
        return sim

    sim = make_sim()
    checkpointed = bool(checkpoint_layers and method == "asr" and use_symmetry and base.requires_grad)

    def single_layer_modes(radius):
        # A fresh simulator makes recomputation pure: backward must never
        # append layers to the simulator used by the forward cascade.
        local = make_sim()
        selected = local.add_structured_layer(LayerSpec(
            thickness=1., geometry=Circle(radius / g.period_nm),
            background=Material(1.), inclusion=Material(g.index ** 2),
            method=METHODS[method], factorization_rules=True))
        if selected != METHODS[method] or len(local._polarized_layers) != 1:
            raise RuntimeError("Layer checkpoint requires one invariant source sector.")
        modes = local._polarized_layers[0]
        return (modes["electric"], modes["magnetic"], modes["kz"],
                *local._polarization_bases)

    for j in range(numerical.slices):
        radius = g.tip_radius_nm + (base - g.tip_radius_nm) * ((j + .5) / numerical.slices) ** g.profile_power
        if checkpointed:
            from torch.utils.checkpoint import checkpoint
            electric, magnetic, kz, eb, hb = checkpoint(
                single_layer_modes, radius, use_reentrant=False, preserve_rng_state=False)
            sim._install_polarization_basis(eb.detach(), hb.detach(), 1e-8)
            sim._polarized_layers.append({"electric": electric, "magnetic": magnetic, "kz": kz})
            sim.thickness.append(height / (numerical.slices * g.period_nm))
            sim.layer_N += 1
            continue
        selected = sim.add_structured_layer(LayerSpec(
            thickness=height / (numerical.slices * g.period_nm),
            geometry=Circle(radius / g.period_nm), background=Material(1.),
            inclusion=Material(g.index ** 2), method=METHODS[method],
            factorization_rules=True if method == "asr" else None,
            label=f"moth-eye-{j}"))
        if selected != METHODS[method]:
            raise RuntimeError("Solver silently selected a different formulation.")
        _discard_flux_only_auxiliary(sim)
    if checkpointed:
        sim._solve_polarization_reduced_smatrix()
    else:
        sim.solve_global_smatrix()
    incident = _zero_order_x_source(sim)

    def flux(electric, medium_v, direction):
        magnetic = direction * (medium_v @ electric)
        n = electric.numel() // 2
        return .5 * torch.real(torch.sum(electric[:n] * magnetic[n:].conj()
                                        - electric[n:] * magnetic[:n].conj()))

    incoming = flux(incident, sim.Vi, 1)
    reflected = -flux(sim.S[1] @ incident, sim.Vi, -1) / incoming
    transmitted = flux(sim.S[0] @ incident, sim.Vo, 1) / incoming
    r, t = float(reflected.detach().cpu()), float(transmitted.detach().cpu())
    synchronize(device)
    diagnostics = sim.cascade_diagnostics
    harmonic_count = (3 * numerical.order * (numerical.order + 1) + 1
                      if lattice == "hexagonal" and use_symmetry else (2 * numerical.order + 1) ** 2)
    row = {"lattice": lattice, "method": method, "wavelength_nm": wavelength,
           **asdict(numerical), "retained_scalar_harmonics": harmonic_count,
           "eigen_dimension": diagnostics.get("reduced_dimension"),
           "full_dimension": diagnostics.get("full_dimension", 2 * sim.order_N),
           "reflectance": r, "transmittance": t, "absorptance": 1 - r - t,
           "energy_residual": r + t - 1, "runtime_seconds": time.perf_counter() - started,
           "device": str(device), "base_fill_fraction": g.fill_fraction(lattice),
           "asr_profile": ("d6-radial" if lattice == "hexagonal" else square_asr_profile) if method == "asr" else None,
           "gradient_layer_checkpointing": checkpointed}
    finite_powers(row)
    return row, reflected


class Study:
    def __init__(self, *, geometry, method, lattice, device, use_symmetry, signature, checkpoint,
                 square_asr_profile="radial", checkpoint_layers=True):
        self.geometry, self.method, self.lattice = geometry, method, lattice
        self.device, self.use_symmetry = device, use_symmetry
        self.square_asr_profile, self.checkpoint_layers = square_asr_profile, checkpoint_layers
        self.signature, self.checkpoint = signature, checkpoint
        self.cases = {}
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
            if saved["signature"] != signature:
                raise ValueError("Checkpoint physical/solver signature differs; use a new --output-root.")
            self.cases = saved["cases"]
            for key, row in self.cases.items():
                finite_powers(row)
                if key != self.key(Numerical(row["order"], row["slices"], row["grid"]), row["wavelength_nm"]):
                    raise ValueError("Corrupt checkpoint case key.")
            print(f"{lattice}/{method}: resuming {len(self.cases)} cases.", flush=True)

    @staticmethod
    def key(n, w):
        return f"M{n.order}|Nz{n.slices}|Q{n.grid}|lambda{w:.17g}"

    def evaluate(self, n, w):
        key = self.key(n, w)
        if key in self.cases:
            return self.cases[key]
        print(f"{self.lattice}/{self.method}: M={n.order} Nz={n.slices} Q={n.grid} lambda={w:g}", flush=True)
        row, tensor = solve(w, n, self.geometry, method=self.method, lattice=self.lattice,
                            device=self.device, use_symmetry=self.use_symmetry,
                            square_asr_profile=self.square_asr_profile, checkpoint_layers=self.checkpoint_layers)
        del tensor
        release(self.device)
        self.cases[key] = row
        write_json(self.checkpoint, {"signature": self.signature, "cases": self.cases})
        print(f"  R={row['reflectance']:.8f} T={row['transmittance']:.8f} "
              f"R+T-1={row['energy_residual']:.2e}; {row['runtime_seconds']:.2f}s", flush=True)
        return row

    def spectrum(self, n, wavelengths):
        return [self.evaluate(n, w) for w in wavelengths]
