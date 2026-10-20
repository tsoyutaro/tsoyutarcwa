"""Run the explicitly specified studies of Mohamad et al., OQE 52:127 (2020)."""
from __future__ import annotations

import argparse
import cmath
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import warnings

# Small and medium dense matrices are often slower with many BLAS threads.
# Explicit user settings take precedence; this applies before importing NumPy/SciPy.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import LinAlgWarning

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from paper_reproductions.mohamad2020.geometries import sinusoidal_aluminium, triangular_gold
    from paper_reproductions.mohamad2020.solver import DM_INTEGRATORS, SOLVER_REVISION, solve
else:
    from .geometries import sinusoidal_aluminium, triangular_gold
    from .solver import DM_INTEGRATORS, SOLVER_REVISION, solve

from studies.shared import gold_dispersion


REFERENCE_R_MINUS_ONE = 0.8785309971543
METHODS = ("rcwa", "rcwa-fff", "dm-fff")
STUDIES = ("smoke", "fig5", "fig6", "fig9a", "fig9b", "fig10a", "fig10b", "all")
CSV_COLUMNS = (
    "study", "curve", "method", "N", "nfft", "section_nm", "integration_step_nm",
    "dm_substeps", "dm_integrator", "sampling", "wavelength_nm", "angle_deg", "gil_nm", "gil_side", "R_minus_1",
    "R0", "T0", "R", "T", "A", "relative_error", "elapsed_s", "status", "error",
)
PAPER_PDF = Path(r"C:\Users\tsoyu\Downloads\Fast Fourier factorization for differential method and RCWA a powerful tool for the modeling of non-lamellar metallic diffraction gratings.pdf")


def number_list(value: str, integer: bool = False) -> list:
    """Comma-separated values or inclusive start:stop[:step] ranges."""
    answer = []
    for item in value.split(","):
        parts = item.strip().split(":")
        if len(parts) == 1:
            answer.append(float(parts[0]))
        elif len(parts) in (2, 3):
            start, stop = map(float, parts[:2])
            step = float(parts[2]) if len(parts) == 3 else 1.0
            if step <= 0 or stop < start:
                raise argparse.ArgumentTypeError("Ranges require stop >= start and step > 0")
            count = int(np.floor((stop - start) / step + 1e-9)) + 1
            if count > 10000:
                raise argparse.ArgumentTypeError("Range has more than 10000 values")
            answer.extend(start + step * i for i in range(count))
        else:
            raise argparse.ArgumentTypeError("Use values such as 2,6,10 or 2:100:2")
    if not answer or any(not np.isfinite(x) for x in answer):
        raise argparse.ArgumentTypeError("Values must be finite")
    if integer:
        if any(x < 1 or int(x) != x for x in answer):
            raise argparse.ArgumentTypeError("Orders N must be positive integers")
        return sorted(set(int(x) for x in answer))
    return sorted(set(answer))


def serializable(value):
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return serializable(value.tolist())
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, np.generic):
        return serializable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Runner:
    def __init__(self, args):
        self.args = args
        self.gold_model = None
        if args.study in ("fig10b", "all"):
            self.gold_model = gold_dispersion.load_gold_csv(args.gold_data)
            # Fail before running other figures if the requested spectrum is
            # outside the shared table. The shared loader never extrapolates.
            for wavelength in args.wavelengths:
                self.gold_model(wavelength)
        self.out = args.output_dir.resolve()
        if self.out.exists() and any(self.out.iterdir()) and not args.overwrite:
            raise ValueError(f"Output folder is not empty: {self.out}. Use a new --output-dir; --overwrite explicitly replaces generated files.")
        self.out.mkdir(parents=True, exist_ok=True)
        self.failed = 0
        self.records = []
        self.meta = {
            "solver_revision": SOLVER_REVISION,
            "paper": {
                "title": "Fast Fourier factorization for differential method and RCWA: a powerful tool for the modeling of non-lamellar metallic diffraction gratings",
                "authors": "Habib Mohamad et al.",
                "doi": "10.1007/s11082-020-2240-y", "year": 2020,
                "source_path": str(PAPER_PDF),
                "source_sha256": sha256(PAPER_PDF) if PAPER_PDF.is_file() else None,
            },
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "created_japan": datetime.now(timezone(timedelta(hours=9))).isoformat(),
            "python": sys.version, "platform": platform.platform(),
            "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "matplotlib")},
            "command": sys.argv,
            "blas_thread_environment": {name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
            "arguments": serializable(vars(args)),
            "profile_note": "NOT paper settings: reduced quick demonstration" if args.profile == "quick" else "Paper-oriented numerical settings; assess convergence and geometry assumptions separately",
            "definitions": {
                "N": "maximum Fourier order; field has 2*N+1 harmonics",
                "section_nm": "thickness of each cascaded S-matrix section",
                "integration_step_nm": "maximum trial step section_nm/dm_substeps for adaptive DM; fixed step for other integrators; section_nm for RCWA",
                "dm_integrator": "cf4-adaptive: fourth-order Magnus with local scattering step-doubling error control; rk4: legacy fixed explicit integration for diagnosis",
                "relative_error": "abs(R_minus_1 - 0.8785309971543)/0.8785309971543 (Fig.5)",
            },
            "limitations": [
                "C-Method / commercial MC-Grating comparison is not implemented.",
                "Fig.8 labels 283.2 nm; it is interpreted as peak-to-valley relief, and gold thickness 50 nm is modeled vertically.",
                "GIL has total horizontal width delta, centered on each original interface (Fig.7b,c), followed by epsilon=n**2; one-sided options are diagnostic.",
                "Fig.10b uses the project's existing Au CSV with the shared linear-epsilon interpolation; the paper does not identify its full dispersion table.",
                "Field plots reconstruct truncated Fourier coefficients directly; interface Gibbs effects remain.",
                "The default adaptive Magnus integrator solves the paper's ODE, but differs from the original fixed RK4 discretization. Local error estimates do not certify Fourier/geometry convergence.",
            ],
            "source_hashes": {path.name: sha256(path) for path in Path(__file__).parent.glob("*.py")},
            "computations": [],
        }
        if self.gold_model is not None:
            self.meta["gold_data"] = {
                "path": str(args.gold_data.resolve()), "sha256": sha256(args.gold_data),
                "loader": "studies.shared.gold_dispersion.load_gold_csv",
                "interpolation": "linear epsilon, then passive sqrt for n+i*k",
                "range_nm": [self.gold_model.wavelength_nm[0], self.gold_model.wavelength_nm[-1]],
                "shared_default": args.gold_data.resolve() == gold_dispersion.DEFAULT_GOLD_CSV.resolve(),
                "used_by": ["fig10b"],
                "note": "Fig.9 and Fig.10a retain the paper's explicit n=0.16+5.26j at 810 nm.",
            }
            self.meta["shared_source_hashes"] = {
                "studies/shared/gold_dispersion.py": sha256(Path(gold_dispersion.__file__)),
            }
        self.save_metadata()

    def save_metadata(self):
        self.meta["updated_utc"] = datetime.now(timezone.utc).isoformat()
        self.meta["failed_computations"] = self.failed
        self.meta["unphysical_computations"] = sum(row["status"] == "UNPHYSICAL" for row in self.records)
        (self.out / "settings.json").write_text(json.dumps(serializable(self.meta), ensure_ascii=False, indent=2), encoding="utf-8")

    def compute(self, study, curve, geom, wavelength, angle, order, method, *, section=None, nfft=None, gil=0.0, fields=False):
        section = self.args.section_nm if section is None or self.args.section_override else section
        nfft = self.args.nfft if nfft is None else nfft
        row = dict.fromkeys(CSV_COLUMNS, np.nan)
        row.update(study=study, curve=curve, method=method, N=order, nfft=nfft,
                   section_nm=section, dm_substeps=self.args.dm_substeps, sampling=self.args.sampling,
                   dm_integrator=self.args.dm_integrator if method == "dm-fff" else "",
                   gil_side=self.args.gil_side if geom.name == "triangular_gold" else "",
                   integration_step_nm=section / self.args.dm_substeps if method.startswith("dm") else section,
                   wavelength_nm=wavelength, angle_deg=angle, gil_nm=gil, status="OK", error="")
        result = None
        start = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                warnings.simplefilter("error", LinAlgWarning)
                result = solve(geom, wavelength, angle, order, method=method, section_nm=section,
                               nfft=nfft, dm_substeps=self.args.dm_substeps,
                               dm_integrator=self.args.dm_integrator, dm_rtol=self.args.dm_rtol,
                               dm_atol=self.args.dm_atol, dm_min_step_nm=self.args.dm_min_step_nm,
                               sampling=self.args.sampling, store_fields=fields)
            values = [result.reflection(-1), result.reflection(0), result.transmission(0), result.R, result.T, result.A]
            if not np.isfinite(values).all():
                raise ArithmeticError("non-finite diffraction efficiency")
            row.update(zip(("R_minus_1", "R0", "T0", "R", "T", "A"), map(float, values)))
            if min(result.R, result.T) < -1e-7 or result.R + result.T > 1.00001:
                row.update(status="UNPHYSICAL", error=f"passive energy balance R={result.R:.8g}, T={result.T:.8g}, A={result.A:.8g}; finite values preserved")
            if study == "fig5":
                row["relative_error"] = abs(row["R_minus_1"] - REFERENCE_R_MINUS_ONE) / REFERENCE_R_MINUS_ONE
        except (np.linalg.LinAlgError, ArithmeticError, RuntimeWarning, LinAlgWarning, ValueError) as error:
            row.update(status="FAILED", error=f"{type(error).__name__}: {error}")
            result = None
            self.failed += 1
        row["elapsed_s"] = time.perf_counter() - start
        detail = {"row": row, "geometry": geom.metadata()}
        if result is not None:
            detail["solver"] = result.metadata
        self.meta["computations"].append(detail)
        self.records.append(row)
        print(f"{study}: {curve}, N={order}, lambda={wavelength:g} nm: {row['status']}, R0={row['R0']:.8g}, R-1={row['R_minus_1']:.8g}, {row['elapsed_s']:.2f} s", flush=True)
        if row["error"]:
            print("  " + row["error"], flush=True)
        # Preserve completed work when a long convergence sweep is interrupted.
        self.write_csv(study)
        self.save_metadata()
        return row, result

    def write_csv(self, study):
        with (self.out / f"{study}.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(row for row in self.records if row["study"] == study)

    def plot(self, study, ylabel, key, *, xlabel="Maximum Fourier order N", xkey="N", log=False):
        rows = [row for row in self.records if row["study"] == study]
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        for curve in dict.fromkeys(row["curve"] for row in rows):
            values = [row for row in rows if row["curve"] == curve]
            ax.plot([row[xkey] for row in values], [row[key] for row in values], "o-", ms=3, label=curve)
        if log:
            ax.set_yscale("log")
        ax.set(xlabel=xlabel, ylabel=ylabel, title=f"{study}: {self.args.profile} settings")
        ax.grid(True, which="both", alpha=.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(self.out / f"{study}.png", dpi=180)
        plt.close(fig)

    def smoke(self):
        for method in METHODS:
            self.compute("smoke", method, sinusoidal_aluminium(), 632.8, 40, 2, method,
                         section=max(self.args.section_nm, 20.0))

    def fig5(self):
        for method in METHODS:
            for nfft in ([self.args.nfft, 1024] if self.args.include_nfft1024 and self.args.nfft != 1024 and method == "dm-fff" else [self.args.nfft]):
                curve = f"{method}, NFFT={nfft}"
                for order in self.args.orders:
                    self.compute("fig5", curve, sinusoidal_aluminium(), 632.8, 40, order, method, nfft=nfft)
        self.plot("fig5", "Relative error of R(-1)", "relative_error", log=True)

    def fig6(self):
        field_maps = []
        order = self.args.field_order
        for method in METHODS:
            _, result = self.compute("fig6", method, sinusoidal_aluminium(), 632.8, 40, order, method, fields=True)
            if result is None:
                continue
            data = result.fields
            if data is None or "Ex_coeff" not in data:
                raise ValueError("store_fields=True did not return Ex_coeff")
            x = np.linspace(0, 500.0, self.args.field_nx, endpoint=False)
            phase = np.exp(2j * np.pi * x[:, None] / 500.0 * result.orders[None, :])
            ex = np.abs(np.asarray(data["Ex_coeff"]) @ phase.T)
            z = np.asarray(data["z_nm"])
            np.savez_compressed(self.out / f"fig6_{method}.npz", x_nm=x, z_nm=z, Ex_abs=ex,
                                Ex_coeff=data["Ex_coeff"], h_coeff=data["h_coeff"], orders=result.orders)
            field_maps.append((method, x, z, ex))
        if not field_maps:
            return
        fig, axes = plt.subplots(1, len(field_maps), figsize=(5 * len(field_maps), 4.4), squeeze=False)
        vmax = max(float(np.nanpercentile(item[3], 99)) for item in field_maps)
        for ax, (method, x, z, ex) in zip(axes[0], field_maps):
            image = ax.pcolormesh(x, 200.0 - z, ex, shading="auto", cmap="turbo", vmin=0, vmax=vmax)
            ax.plot(x, 100.0 * (1 - np.cos(2 * np.pi * x / 500.0)), "w--", lw=.7)
            ax.set(xlabel="x (nm)", ylabel="Physical height (nm)", title=f"{method}, N={order}",
                   xlim=(0, 500), ylim=(0, 200))
            fig.colorbar(image, ax=ax, label="|Ex| / |Z0 H_inc|")
        fig.suptitle(f"Fig.6 comparison: {self.args.profile}; color limit = largest 99th percentile")
        fig.tight_layout()
        fig.savefig(self.out / "fig6.png", dpi=180)
        plt.close(fig)

    def fig9a(self):
        steps = (self.args.section_nm / self.args.dm_substeps,) if self.args.section_override else (1.0, .5, .25)
        for dz in steps:
            # Dz is a maximum step for adaptive DM, fixed for the legacy RK4.
            for order in self.args.orders:
                self.compute("fig9a", f"DM-FFF, Dz={dz:g} nm", triangular_gold(gil_side=self.args.gil_side), 810, 11,
                             order, "dm-fff", section=dz * self.args.dm_substeps)
        self.plot("fig9a", "Zero-order reflectance R0", "R0")

    def fig9b(self):
        section = float(self.args.dm_substeps) if self.args.profile == "paper" else self.args.section_nm
        for gil in self.args.gil_thicknesses:
            for order in self.args.orders:
                self.compute("fig9b", f"DM-FFF, GIL={gil:g} nm", triangular_gold(gil_nm=gil, gil_side=self.args.gil_side),
                             810, 11, order, "dm-fff", gil=gil, section=section)
        self.plot("fig9b", "Zero-order reflectance R0", "R0")

    def fig10a(self):
        curves = (("dm-fff", 0), ("dm-fff", 1), ("rcwa-fff", 1), ("rcwa", 0))
        for method, gil in curves:
            section = (float(self.args.dm_substeps) if method == "dm-fff" else 1.0) if self.args.profile == "paper" else self.args.section_nm
            for order in self.args.orders:
                self.compute("fig10a", f"{method}, GIL={gil} nm", triangular_gold(gil_nm=gil, gil_side=self.args.gil_side),
                             810, 15, order, method, gil=gil, section=section)
        self.plot("fig10a", "Zero-order reflectance R0", "R0")

    def fig10b(self):
        order = max(self.args.orders)
        section = float(self.args.dm_substeps) if self.args.profile == "paper" else self.args.section_nm
        for gil in (0, .3):
            for wavelength in self.args.wavelengths:
                # This is exactly the epsilon interpolator used by the other
                # Au studies. GIL needs n, so recover the passive square root.
                metal = cmath.sqrt(self.gold_model(wavelength))
                self.compute("fig10b", f"DM-FFF, GIL={gil} nm, N={order}",
                             triangular_gold(gil_nm=gil, n_metal=metal, gil_side=self.args.gil_side), wavelength, 15, order, "dm-fff", gil=gil, section=section)
        self.plot("fig10b", "Zero-order transmittance T0", "T0", xlabel="Wavelength (nm)", xkey="wavelength_nm")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--study", choices=STUDIES, default="smoke")
    parser.add_argument("--profile", choices=("quick", "paper"), default="quick")
    parser.add_argument("--orders", type=lambda value: number_list(value, integer=True), help="Maximum orders N: 2,6,10 or inclusive 2:100:2")
    parser.add_argument("--section-nm", type=float, help="Override all section thicknesses; quick default 5 nm; paper Fig.5/6 1 nm, Fig.9/10 DM step 1 nm")
    parser.add_argument("--dm-substeps", type=int, default=4)
    parser.add_argument("--dm-integrator", choices=DM_INTEGRATORS, default="cf4-adaptive",
                        help="Adaptive Magnus is the corrected default; rk4 retains the unsafe fixed-step method for comparisons")
    parser.add_argument("--dm-rtol", type=float, default=1e-6, help="Relative local error in each scattering block's maximum-entry norm")
    parser.add_argument("--dm-atol", type=float, default=1e-9, help="Local scattering step-doubling absolute tolerance")
    parser.add_argument("--dm-min-step-nm", type=float, default=1e-5)
    parser.add_argument("--nfft", type=int, default=4096)
    parser.add_argument("--sampling", choices=("entrance", "midpoint"), default="entrance")
    parser.add_argument("--gil-side", choices=("centered", "metal", "dielectric"), default="centered",
                        help="Paper Fig.7 centers total width delta on the interface; one-sided options are diagnostic")
    parser.add_argument("--gil-thicknesses", type=number_list, default=[0., 1., 5., 7., 10.],
                        help="Fig.9b only: GIL thicknesses in nm; restrict to 5 for the N=60 regression")
    parser.add_argument("--include-nfft1024", action="store_true", help="Add DM-FFF NFFT=1024 to Fig.5")
    parser.add_argument("--field-order", type=int, help="Fig.6 N (quick 8; paper 50)")
    parser.add_argument("--field-nx", type=int, default=256)
    parser.add_argument("--gold-data", "--gold-csv", dest="gold_data", type=Path,
                        default=gold_dispersion.DEFAULT_GOLD_CSV,
                        help="Au CSV for fig10b (shared existing table by default); n,k or epsilon_real,epsilon_imag")
    parser.add_argument("--wavelengths", type=number_list, help="Fig.10b wavelengths in nm; values or inclusive start:stop:step")
    parser.add_argument("--output-dir", type=Path, help="Output folder (default results/<profile>_<integrator>; old results/paper is preserved)")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing generated files in a nonempty output folder")
    args = parser.parse_args(argv)
    args.section_override = args.section_nm is not None
    args.orders = args.orders or (sorted([*range(2, 101, 2), 95]) if args.profile == "paper" else [2, 6, 10])
    args.section_nm = args.section_nm if args.section_nm is not None else (1.0 if args.profile == "paper" else 5.0)
    args.field_order = args.field_order if args.field_order is not None else (50 if args.profile == "paper" else 8)
    args.wavelengths = args.wavelengths or number_list("400:850:5" if args.profile == "paper" else "400:850:50")
    args.output_dir = args.output_dir or Path(__file__).parent / "results" / f"{args.profile}_{args.dm_integrator}"
    if args.section_nm <= 0 or not np.isfinite(args.section_nm) or args.dm_substeps < 1 or args.field_order < 1 or args.field_nx < 2:
        parser.error("section-nm, dm-substeps and field-order must be positive; field-nx must be >= 2")
    if args.nfft <= 4 * max(max(args.orders), args.field_order):
        parser.error("nfft must exceed four times the largest N to retain epsilon Fourier coefficients")
    if any(value <= 0 for value in args.wavelengths):
        parser.error("wavelengths must be positive")
    if any(value < 0 for value in args.gil_thicknesses):
        parser.error("GIL thicknesses must be nonnegative")
    if any(not np.isfinite(v) or v <= 0 for v in (args.dm_rtol, args.dm_atol, args.dm_min_step_nm)):
        parser.error("DM tolerances and minimum step must be finite and positive")
    if args.study in ("fig10b", "all") and not args.gold_data.is_file():
        parser.error(f"Gold CSV does not exist: {args.gold_data}")
    if args.profile == "quick":
        print("QUICK DEMONSTRATION: reduced N and section settings; not the paper's numerical settings.", flush=True)
    if args.dm_integrator == "rk4":
        print("LEGACY FIXED RK4: high-order metallic gratings can produce artificial gain. Check step refinement; use cf4-adaptive for reliable integration.", flush=True)
    elif args.dm_integrator == "cf4-adaptive":
        print("DM uses adaptive CF4 Magnus; Dz is a maximum trial step. This corrects the fixed-RK4 integration, not the paper's Fourier truncation errors.", flush=True)
    try:
        runner = Runner(args)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    requested = ("fig5", "fig6", "fig9a", "fig9b", "fig10a", "fig10b") if args.study == "all" else (args.study,)
    for study in requested:
        getattr(runner, study)()
    runner.save_metadata()
    unphysical = sum(row["status"] == "UNPHYSICAL" for row in runner.records)
    print(f"Saved: {runner.out}; {len(runner.records)} computations; {runner.failed} FAILED; {unphysical} UNPHYSICAL.", flush=True)
    return 1 if runner.failed else (3 if unphysical else 0)


if __name__ == "__main__":
    raise SystemExit(main())
