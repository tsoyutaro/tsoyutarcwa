"""Calculate Figs. 6, 7, 9 and 11; preserve spectra and numerical provenance."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

from .geometries import FIGURES, PAPER_DOI, geometry_for_figure, geometry_metadata
from .solver import PreparedStack


ROOT = Path(__file__).resolve().parent


def json_safe(value):
    if isinstance(value, (complex, np.complexfloating)):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    return value


def harmonic_count(label, convention):
    return int(label) if convention == "count" else 2 * int(label) + 1


def plot_spectra(figure, rows, destination, *, overlay_reference=False, paper_sampling=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator

    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    fig, axes = plt.subplots(2, 1, figsize=(4.3, 5.2), sharex=True)
    setting = FIGURES[figure]
    for index, (ax, method) in enumerate(zip(axes, ("fmm", "asr"))):
        plot_modes = sorted({r["modes"] for r in rows if r["method"] == method})
        for j, modes in enumerate(plot_modes):
            subset = [r for r in rows if r["method"] == method and r["modes"] == modes]
            if paper_sampling:
                panel = "a" if method == "fmm" else "b"
                style = "solid" if j == 0 else "diamond"
                path = ROOT / "reference" / f"fig{figure}_{panel}_{style}.csv"
                if path.exists():
                    ref = np.genfromtxt(path, delimiter=",", names=True)
                    subset = [min(subset, key=lambda r: abs(r["wavelength"] - x)) for x in ref["wavelength"]]
            ax.plot([r["wavelength"] for r in subset], [r["T0"] for r in subset],
                    color="black", lw=1.0 if j == 0 else 0.65,
                    marker=None if j == 0 else "d", markersize=3.2,
                    markerfacecolor="none", markeredgewidth=0.65,
                    markevery=1 if paper_sampling else max(1, len(subset) // 45), label=f"{modes} modes")
        high = [r for r in rows if r["method"] == "fmm_reference"]
        if high:
            ax.plot([r["wavelength"] for r in high], [r["T0"] for r in high],
                    color="black", ls="--", lw=0.9, label=f"{high[0]['modes']} modes FMM")
        if overlay_reference:
            refpath = ROOT / "reference" / f"fig{figure}_{'a' if index == 0 else 'b'}_solid.csv"
            if refpath.exists():
                ref = np.genfromtxt(refpath, delimiter=",", names=True)
                ax.plot(ref[ref.dtype.names[0]], ref[ref.dtype.names[1]],
                        color="#b64e34", ls=":", lw=1.2, label="PDF solid curve")
        ax.set_title(f"({chr(97 + index)}) {'FMM' if method == 'fmm' else 'Parametric representation'}", loc="left")
        ax.set_ylabel(r"$\eta_0$")
        ax.set_xlim(*setting["range"])
        if figure == 6:
            ax.set_ylim(0.185, 0.265)
        elif figure == 7:
            ax.set_ylim(0, 0.56)
        elif figure == 9:
            ax.set_ylim(0, 1.04)
        else:
            ax.set_ylim(0, 0.45)
        values = [r["T0"] for r in rows if r["method"] in (method, "fmm_reference")]
        if values:
            lower, upper = ax.get_ylim()
            ax.set_ylim(min(lower, min(values) * 0.98), max(upper, max(values) * 1.03))
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.legend(fontsize=7, loc="best", frameon=False)
        ax.set_xlabel(r"$\lambda / d$")
    fig.suptitle(f"Vallius & Honkanen (2002), Fig. {figure}, {setting['polarization']}" +
                 ("\nCalculated at the published sample wavelengths" if paper_sampling else ""), fontsize=10)
    fig.tight_layout()
    fig.savefig(destination, dpi=200)
    plt.close(fig)


def reference_metrics(figure, rows):
    result = {}
    comparisons = [("a", "a", "fmm", FIGURES[figure]["modes"][0], "solid"),
                   ("b", "b", "asr", FIGURES[figure]["modes"][0], "solid"),
                   ("a_high", "a", "fmm", FIGURES[figure]["modes"][1], "diamond"),
                   ("b_high", "b", "asr", FIGURES[figure]["modes"][1], "diamond")]
    if figure in (6, 7):
        comparisons.append(("fmm_reference", "a", "fmm_reference", 240, "dashed"))
    for key, panel, method, modes, style in comparisons:
        path = ROOT / "reference" / f"fig{figure}_{panel}_{style}.csv"
        if not path.exists():
            continue
        ref = np.genfromtxt(path, delimiter=",", names=True)
        x, y = ref[ref.dtype.names[0]], ref[ref.dtype.names[1]]
        subset = [r for r in rows if r["method"] == method and r["modes"] == modes]
        if not subset:
            continue
        sx = np.array([r["wavelength"] for r in subset])
        sy = np.array([r["T0"] for r in subset])
        keep = (x >= sx.min()) & (x <= sx.max())
        error = np.interp(x[keep], sx, sy) - y[keep]
        result[key] = {"reference": path.name, "modes": modes, "samples": int(keep.sum()),
                         "mae": float(np.mean(np.abs(error))),
                         "rmse": float(np.sqrt(np.mean(error ** 2))),
                         "maximum_absolute_error": float(np.max(np.abs(error))),
                         "assessment": "diagnostic only; no claim of exact reproduction"}
    return result


def run_figure(figure, args):
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.glob("*.py"))}
    setting = FIGURES[figure]
    layers = geometry_for_figure(figure, slices=args.slices, metal_widths=args.metal_widths)
    count = args.points or (401 if figure == 9 else 241)
    if args.study == "smoke":
        count = args.points or 9
    wavelengths = np.linspace(*setting["range"], count)
    if args.wavelengths:
        wavelengths = np.array(sorted({float(x) for x in args.wavelengths.split(",")}))
        if not len(wavelengths) or np.any(wavelengths <= 0):
            raise ValueError("Wavelengths must be positive.")
    if args.include_reference_grid:
        reference_grids = []
        for panel in ("a", "b"):
            for style in ("solid", "diamond", "dashed"):
                path = ROOT / "reference" / f"fig{figure}_{panel}_{style}.csv"
                if path.exists():
                    data = np.genfromtxt(path, delimiter=",", names=True)
                    reference_grids.extend(data["wavelength"].tolist())
        wavelengths = np.unique(np.concatenate((wavelengths, reference_grids)))
    modes = args.modes or setting["modes"]
    series = [(method, m) for method in ("fmm", "asr") for m in modes]
    reference_modes = args.reference_modes if args.reference_modes is not None else (0 if args.study == "smoke" else 240)
    if figure in (6, 7) and reference_modes:
        series.append(("fmm_reference", reference_modes))
    destination = args.output_dir
    destination.mkdir(parents=True, exist_ok=True)
    csvpath = destination / f"fig{figure}.csv"
    fields = ["figure", "method", "modes", "harmonics", "eigen_dimension", "quadrature_actual", "wavelength",
              "evaluation_wavelength", "T0", "T", "R", "A", "max_boundary_condition"]
    started = time.perf_counter()
    rows = []
    with csvpath.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method, m in series:
            actual_method = "fmm" if method == "fmm_reference" else method
            harmonics = harmonic_count(m, args.mode_convention)
            stack = PreparedStack(layers, setting["polarization"], actual_method,
                                  harmonics=harmonics, quadrature=args.quadrature, G=args.G,
                                  oversampling=args.oversampling if actual_method == "asr" else 1,
                                  retention=args.retention, q_projection=args.q_projection)
            series_wavelengths = wavelengths
            if method == "fmm_reference":
                series_wavelengths = np.linspace(*setting["range"], args.reference_points)
                refpath = ROOT / "reference" / f"fig{figure}_a_dashed.csv"
                if args.include_reference_grid and refpath.exists():
                    ref = np.genfromtxt(refpath, delimiter=",", names=True)
                    series_wavelengths = np.unique(np.concatenate((series_wavelengths, ref["wavelength"])))
            print(f"Fig {figure}: {method}, label={m}, N={harmonics}, points={len(series_wavelengths)}", flush=True)
            for i, wavelength in enumerate(series_wavelengths):
                solution = stack.solve(float(wavelength))
                row = {"figure": figure, "method": method, "modes": int(m), "harmonics": harmonics,
                       "eigen_dimension": max(solution["eigen_dimensions"]),
                       "quadrature_actual": int(solution.get("quadrature_points_per_region", 0)),
                       "wavelength": float(wavelength),
                       "evaluation_wavelength": solution.get("evaluation_wavelength", float(wavelength)),
                       **{key: float(solution[key]) for key in ("T0", "T", "R", "A", "max_boundary_condition")}}
                if not all(np.isfinite(row[key]) for key in ("T0", "T", "R", "A")):
                    raise FloatingPointError(f"Nonfinite spectrum at Fig {figure}, {method}, {m}, {wavelength}")
                rows.append(row)
                writer.writerow(row)
                if i % max(1, len(series_wavelengths) // 5) == 0:
                    handle.flush()
                    print(f"  {i + 1}/{len(series_wavelengths)}, lambda={wavelength:.7g}, T0={row['T0']:.7g}", flush=True)
    plot_spectra(figure, rows, destination / f"fig{figure}.png")
    if args.overlay_reference:
        plot_spectra(figure, rows, destination / f"fig{figure}_comparison.png", overlay_reference=True)
    if args.include_reference_grid:
        plot_spectra(figure, rows, destination / f"fig{figure}_paper_sampling.png",
                     overlay_reference=args.overlay_reference, paper_sampling=True)
    metadata = {
        "paper_doi": PAPER_DOI, "figure": figure, "polarization": setting["polarization"],
        "normal_incidence": True, "epsilon_in": 1, "epsilon_out": 1,
        "mode_convention": args.mode_convention, "harmonic_rule": "N=M" if args.mode_convention == "count" else "N=2M+1",
        "asr_eigen_oversampling": args.oversampling, "G": args.G, "quadrature_requested_minimum": args.quadrature,
        "quadrature_actual_maximum": max(r["quadrature_actual"] for r in rows),
        "retention": args.retention, "q_projection": args.q_projection,
        "include_reference_grid": args.include_reference_grid, "reference_modes": reference_modes,
        "reference_points_requested": args.reference_points,
        "cylinder_slices": args.slices, "geometry": geometry_metadata(layers),
        "metal_dimensions": "inferred from Fig. 5 vector drawing; not specified numerically in the paper" if figure in (6, 7) else None,
        "wavelengths": wavelengths.tolist(), "seconds": time.perf_counter() - started,
        "versions": {"python": platform.python_version(), "numpy": np.__version__},
        "source_sha256": source_hashes,
        "reference_comparison": reference_metrics(figure, rows),
        "max_abs_energy_residual": max(abs(r["A"]) for r in rows) if figure in (9, 11) else None,
        "maximum_T0": max(r["T0"] for r in rows), "command": sys.argv,
        "completed": True,
    }
    (destination / f"fig{figure}_metadata.json").write_text(json.dumps(json_safe(metadata), indent=2), encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=("all", "paper", "smoke", "fig6", "fig7", "fig9", "fig11"), default="all")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "paper")
    parser.add_argument("--points", type=int, help="Wavelength points per figure (default 241; Fig. 9: 401).")
    parser.add_argument("--wavelengths", help="Explicit comma-separated wavelengths.")
    parser.add_argument("--modes", type=lambda value: tuple(int(x) for x in value.split(",")))
    parser.add_argument("--mode-convention", choices=("count", "order"), default="order",
                        help="count: caption M is N retained modes; order: caption M gives 2M+1 modes.")
    parser.add_argument("--reference-modes", type=int, help="Fig. 6/7 FMM reference: default 240, smoke 0; 0 disables.")
    parser.add_argument("--reference-points", type=int, default=51, help="Independent wavelength grid for the expensive reference.")
    parser.add_argument("--quadrature", type=int, default=192)
    parser.add_argument("--oversampling", type=int, default=3, help="ASR eigensystem size / retained size (paper: 3).")
    parser.add_argument("--G", type=float, default=0.001)
    parser.add_argument("--slices", type=int, default=120)
    parser.add_argument("--metal-widths", type=lambda value: tuple(float(x) for x in value.split(",")))
    parser.add_argument("--overlay-reference", action="store_true")
    parser.add_argument("--include-reference-grid", action="store_true", help="Also solve at the original PDF vertices and markers.")
    parser.add_argument("--retention", choices=("smallest_abs", "physical"), default="smallest_abs")
    parser.add_argument("--q-projection", choices=("direct", "laurent"), default="direct")
    args = parser.parse_args()
    if args.points is not None and args.points < 2:
        parser.error("--points must be >= 2")
    if args.modes is not None and (len(args.modes) != 2 or min(args.modes) < 1):
        parser.error("--modes requires two positive integers")
    if (args.reference_modes is not None and args.reference_modes < 0) or args.oversampling < 1:
        parser.error("mode settings must be nonnegative (oversampling >= 1)")
    if args.reference_points < 2:
        parser.error("--reference-points must be >= 2")
    figures = list(FIGURES) if args.study in ("all", "paper", "smoke") else [int(args.study[3:])]
    for figure in figures:
        run_figure(figure, args)


if __name__ == "__main__":
    main()
