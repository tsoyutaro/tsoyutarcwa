"""Material preparation, geometry and resumable PMMA/Cr power calculations.

Only the actual solve imports Torch. Dry runs and plotting work without CUDA.
The three-region RCWA engine is shared with the measured PMMA/Au study.
"""
from __future__ import annotations

import bisect
import csv
import gc
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from studies.shared.tabulated_nk import load_tabulated_nk
from studies.pmma_gold_motheye.geometry_preview import build_profile_layers

ROOT = Path(__file__).resolve().parents[2]
POWERS = ("reflectance", "transmittance", "absorptance")


def content_hash(path: Path) -> str:
    # Identical content on Linux and Windows must have the same identity.
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def signature_for(assumptions: dict) -> str:
    return hashlib.sha256(json.dumps(assumptions, sort_keys=True,
                                    allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row)) if rows else ["wavelength_nm", *POWERS]
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list))
                             else value for key, value in row.items()})
    temporary.replace(path)


@dataclass(frozen=True)
class Geometry:
    period_nm: float = 200.0
    height_nm: float = 500.0
    tip_radius_nm: float = 5.0
    base_radius_nm: float = 65.0
    cr_side_thickness_nm: float = 30.0
    cr_top_thickness_nm: float = 30.0
    cr_valley_thickness_nm: float = 30.0
    profile_power: float = 1.0
    asr_circle_g: float = 0.03

    def validate(self) -> None:
        for key, value in asdict(self).items():
            if not math.isfinite(value):
                raise ValueError(f"{key} must be finite.")
            if key in {"cr_top_thickness_nm", "cr_valley_thickness_nm"}:
                if value < 0:
                    raise ValueError(f"{key} must be nonnegative.")
            elif value <= 0:
                raise ValueError(f"{key} must be positive.")
        if self.tip_radius_nm > self.base_radius_nm:
            raise ValueError("Tip radius must not exceed base radius.")
        if 2 * (self.base_radius_nm + self.cr_side_thickness_nm) >= self.period_nm:
            raise ValueError("Coated bases overlap: require 2*(base+side) < period.")
        if self.cr_valley_thickness_nm > self.height_nm:
            raise ValueError("Valley thickness must not exceed PMMA height.")
        if not 0 < self.asr_circle_g < 1:
            raise ValueError("ASR G must lie in (0,1).")

    def profile(self, slices: int):
        return build_profile_layers(
            height_nm=self.height_nm, tip_radius_nm=self.tip_radius_nm,
            base_radius_nm=self.base_radius_nm,
            gold_thickness_nm=self.cr_side_thickness_nm,
            profile_power=self.profile_power, slices=slices,
            valley_gold_thickness_nm=self.cr_valley_thickness_nm)


@dataclass(frozen=True)
class Numerical:
    order: int
    slices: int
    grid: int


class PMMAIndex:
    """Linear n interpolation, k=0; optionally extend the first segment to 400 nm."""

    def __init__(self, path: Path, policy: str):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
        if not rows or [v.strip().lower() for v in rows[0]] != ["wl", "n"]:
            raise ValueError("PMMA CSV requires the header wl,n (wl in micrometres).")
        pairs = [(1000 * float(row[0]), float(row[1])) for row in rows[1:]
                 if row and any(v.strip() for v in row)]
        if len(pairs) < 2 or any(not math.isfinite(w) or not math.isfinite(n)
                                 or w <= 0 or n <= 0 for w, n in pairs):
            raise ValueError("PMMA CSV needs at least two finite positive samples.")
        if any(b[0] <= a[0] for a, b in zip(pairs, pairs[1:])):
            raise ValueError("PMMA wavelengths must be strictly increasing.")
        self.wavelengths, self.indices = tuple(zip(*pairs))
        self.policy = policy
        if policy not in {"linear-to-400", "error"}:
            raise ValueError("Unknown PMMA shortwave policy.")
        if policy == "linear-to-400" and self.wavelengths[0] > 405:
            raise ValueError("400 nm extension is only allowed from data starting <=405 nm.")

    def __call__(self, wavelength_nm: float) -> float:
        w = float(wavelength_nm)
        if not math.isfinite(w) or w > self.wavelengths[-1]:
            raise ValueError(f"PMMA wavelength {w} nm is outside its data range.")
        if w < self.wavelengths[0]:
            if self.policy != "linear-to-400" or w < 400:
                raise ValueError(f"PMMA wavelength {w} nm is below its data range; "
                                 "use wavelengths >=404.7 or --pmma-shortwave-policy linear-to-400.")
            left, right = 0, 1
        else:
            right = bisect.bisect_left(self.wavelengths, w)
            if self.wavelengths[right] == w:
                return self.indices[right]
            left = right - 1
        fraction = ((w - self.wavelengths[left]) /
                    (self.wavelengths[right] - self.wavelengths[left]))
        return self.indices[left] + fraction * (self.indices[right] - self.indices[left])

    def provenance(self, path: Path) -> dict:
        return {"csv": path.name, "sha256_lf": content_hash(path), "k": 0.0,
                "tabulated_range_nm": [self.wavelengths[0], self.wavelengths[-1]],
                "interpolation": "linear n in wavelength_nm",
                "shortwave_policy": self.policy,
                "extension_source_points": list(zip(self.wavelengths[:2], self.indices[:2])),
                "n_400nm": self(400) if self.policy == "linear-to-400"
                and self.wavelengths[0] > 400 else None}


def finite_powers(row: dict) -> dict[str, float]:
    values = {name: float(row[name]) for name in POWERS}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("R/T/A contains NaN or infinity.")
    return values


def physical(row: dict) -> bool:
    values = finite_powers(row)
    return (not row.get("passivity_warning", False)
            and all(-1e-6 <= value <= 1 + 1e-6 for value in values.values())
            and abs(sum(values.values()) - 1) <= 1e-8)


def adjacent_error(coarse: list[dict], fine: list[dict]) -> tuple[float, dict]:
    if not coarse or len(coarse) != len(fine):
        raise ValueError("Convergence spectra must be nonempty and equally sized.")
    errors = dict.fromkeys(POWERS, 0.0)
    for a, b in zip(coarse, fine):
        if a["wavelength_nm"] != b["wavelength_nm"]:
            raise ValueError("Convergence spectra use different wavelengths.")
        av, bv = finite_powers(a), finite_powers(b)
        for name in POWERS:
            errors[name] = max(errors[name], abs(av[name] - bv[name]))
    return max(errors.values()), errors


def choose_candidate(candidates: list[int], spectra: dict, tolerance: float):
    comparisons = []
    for coarse, fine in zip(candidates, candidates[1:]):
        maximum, per_metric = adjacent_error(spectra[coarse], spectra[fine])
        passive = all(physical(row) for row in spectra[coarse] + spectra[fine])
        comparisons.append({"coarse": coarse, "fine": fine,
                            "max_abs_change": maximum, "per_metric": per_metric,
                            "nonpassive_candidate": not passive,
                            "passed": passive and maximum <= tolerance})
    # Require the final two refinements; an early flat region cannot pass.
    passed = len(comparisons) >= 2 and all(row["passed"] for row in comparisons[-2:])
    return candidates[-2] if passed else candidates[-1], passed, comparisons


class Study:
    def __init__(self, *, geometry: Geometry, cr_csv: Path, pmma_csv: Path,
                 pmma_policy: str, device, radial_mapping: str, cascade: str,
                 use_symmetry: bool, signature: str, checkpoint: Path):
        self.geometry = geometry
        self.cr = load_tabulated_nk(cr_csv)
        self.pmma = PMMAIndex(pmma_csv, pmma_policy)
        self.device = device
        self.radial_mapping = radial_mapping
        self.cascade = cascade
        self.use_symmetry = use_symmetry
        self.signature, self.checkpoint = signature, checkpoint
        self.cases = {}
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
            if saved.get("signature") != signature:
                raise ValueError("Checkpoint geometry/material/solver signature differs. "
                                 "Use a different --output-root; no cached cases were reused.")
            self.cases = saved["cases"]
            for key, row in self.cases.items():
                finite_powers(row)
                expected = self.key(Numerical(row["order"], row["profile_slices"], row["grid"]),
                                    row["wavelength_nm"])
                if key != expected:
                    raise ValueError("Checkpoint case key disagrees with its numerical conditions.")
            print(f"Resuming {len(self.cases)} completed cases.", flush=True)

    @staticmethod
    def key(numerical: Numerical, wavelength: float) -> str:
        return f"M{numerical.order}|Nz{numerical.slices}|Q{numerical.grid}|{wavelength:.17g}"

    def evaluate(self, numerical: Numerical, wavelength: float) -> dict:
        key = self.key(numerical, wavelength)
        if key in self.cases:
            return self.cases[key]
        import torch
        from studies.pmma_gold_motheye.common import GeometryConfig, NumericalConfig, simulate_case
        g = self.geometry
        n = self.pmma(wavelength)
        geometry = GeometryConfig(period_nm=g.period_nm, height_nm=g.height_nm,
                                  tip_radius_nm=g.tip_radius_nm, base_radius_nm=g.base_radius_nm,
                                  gold_thickness_nm=g.cr_side_thickness_nm,
                                  profile_power=g.profile_power, pmma_index=n,
                                  lattice="triangular", include_top_cap=g.cr_top_thickness_nm > 0,
                                  asr_circle_g=g.asr_circle_g)
        print(f"PMMA/Cr: M={numerical.order} Nz={numerical.slices} "
              f"grid={numerical.grid} lambda={wavelength:g} nm device={self.device}", flush=True)
        try:
            # ASR uses autograd to obtain the spatial mapping Jacobian even for
            # fixed geometry. Those builders detach their outputs before the
            # eigenproblem; a global no_grad context would break that Jacobian.
            with torch.enable_grad():
                row = simulate_case(
                    wavelength, NumericalConfig(**asdict(numerical), radial_mapping=self.radial_mapping),
                    geometry, self.cr, cascade=self.cascade, use_symmetry=self.use_symmetry,
                    symmetry_reduction="d6-source", factorization_rules=True, device=self.device,
                    valley_gold_thickness_nm=g.cr_valley_thickness_nm,
                    top_cap_thickness_nm=g.cr_top_thickness_nm, discard_auxiliary=True)
            for old, new in (("epsilon_gold_real", "epsilon_cr_real"),
                             ("epsilon_gold_imag", "epsilon_cr_imag"),
                             ("valley_gold_thickness_nm", "cr_valley_thickness_nm")):
                row[new] = row.pop(old)
            row.update(material="Cr", pmma_index=n, device=str(self.device),
                       torch_version=torch.__version__, cr_side_thickness_nm=g.cr_side_thickness_nm,
                       pmma_shortwave_extrapolated=wavelength < self.pmma.wavelengths[0])
            finite_powers(row)
            row["passivity_warning"] = not physical(row)
            self.cases[key] = row
            write_json(self.checkpoint, {"signature": self.signature, "cases": self.cases})
            print(f"  R={row['reflectance']:.8f} T={row['transmittance']:.8f} "
                  f"A={row['absorptance']:.8f} ({row['runtime_seconds']:.2f} s)"
                  + (" PASSIVITY WARNING" if row["passivity_warning"] else ""), flush=True)
            return row
        finally:
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def spectrum(self, numerical: Numerical, wavelengths: tuple[float, ...]) -> list[dict]:
        return [self.evaluate(numerical, w) for w in wavelengths]
