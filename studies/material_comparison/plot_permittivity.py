"""Plot the Au/Cr relative permittivity used by the moth-eye studies.

Run from the project root:
    python studies/material_comparison/plot_permittivity.py

Only material interpolation and plotting are performed; no RCWA run is needed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from studies.shared.tabulated_nk import load_tabulated_nk

STEM = "au_cr_permittivity_400_700nm"
COLUMNS = ("wavelength_nm", "au_epsilon_real", "au_epsilon_imag",
           "cr_epsilon_real", "cr_epsilon_imag")


def export_data(au_csv: Path, cr_csv: Path, output_root: Path) -> list[dict]:
    au = load_tabulated_nk(au_csv)
    cr = load_tabulated_nk(cr_csv)
    rows = []
    for wavelength_nm in range(400, 701):
        ea, ec = au(wavelength_nm), cr(wavelength_nm)
        if not all(math.isfinite(v) for v in (ea.real, ea.imag, ec.real, ec.imag)):
            raise ValueError(f"Non-finite permittivity at {wavelength_nm} nm.")
        rows.append(dict(zip(COLUMNS, (wavelength_nm, ea.real, ea.imag,
                                      ec.real, ec.imag))))
    with (output_root / f"{STEM}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "quantity": "complex relative permittivity (dimensionless)",
        "time_convention": "exp(-i omega t); passive Im(epsilon) >= 0",
        "conversion": "epsilon = (n + i*k)^2",
        "interpolation": "linear in epsilon versus wavelength, as in the solver",
        "range_nm": [400, 700],
        "step_nm": 1,
        "sources": {},
        "anchors": [r for r in rows if r["wavelength_nm"] in (400, 550, 700)],
    }
    for name, path, model in (("Au", au_csv, au), ("Cr", cr_csv, cr)):
        metadata["sources"][name] = {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "sample_count": len(model.wavelength_nm),
            "samples_400_700nm": [
                {"wavelength_nm": w, "epsilon_real": (complex(n, k)**2).real,
                 "epsilon_imag": (complex(n, k)**2).imag}
                for w, n, k in zip(model.wavelength_nm, model.n, model.k)
                if 400 <= w <= 700
            ],
        }
    (output_root / f"{STEM}_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return rows


def plot_data(rows: list[dict], output_root: Path) -> None:
    # Import lazily so material data can be exported without plotting packages.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator, MultipleLocator

    metadata = json.loads((output_root / f"{STEM}_metadata.json").read_text(encoding="utf-8"))
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.labelsize": 12, "axes.titlesize": 13,
        "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "none", "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    wavelengths = [float(row["wavelength_nm"]) for row in rows]
    colors = {"Au": "#AD7600", "Cr": "#1F77B4"}
    for ax, component, title, label in (
        (axes[0], "real", "(a) Real part", r"$\mathrm{Re}(\varepsilon_r)$"),
        (axes[1], "imag", "(b) Imaginary part", r"$\mathrm{Im}(\varepsilon_r)$"),
    ):
        for material in ("Au", "Cr"):
            ax.plot(wavelengths, [float(row[f"{material.lower()}_epsilon_{component}"])
                                 for row in rows], color=colors[material], linewidth=2.4,
                    label=f"{material} (CSV)")
            samples = metadata["sources"][material]["samples_400_700nm"]
            ax.scatter([sample["wavelength_nm"] for sample in samples],
                       [sample[f"epsilon_{component}"] for sample in samples],
                       color=colors[material], s=22, edgecolors="white", linewidths=0.6,
                       zorder=3)
        ax.axhline(0, color="#606060", linewidth=0.8, linestyle="--", zorder=0)
        ax.set(xlim=(400, 700), xlabel="Wavelength (nm)", ylabel=label, title=title)
        ax.xaxis.set_major_locator(MultipleLocator(50))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.grid(axis="both", color="#E1E4E8", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, loc="lower left" if component == "real" else "upper left")
    axes[1].set_ylim(bottom=0)
    fig.suptitle("Relative permittivity of Au and Cr: 400–700 nm", fontsize=16, y=0.97)
    fig.text(0.5, 0.035,
             r"Lines: linear interpolation in $\varepsilon_r$; dots: CSV samples. "
             r"$\varepsilon_r=(n+ik)^2$; convention: $e^{-i\omega t}$.",
             ha="center", fontsize=9, color="#505050")
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.19, top=0.80, wspace=0.28)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(output_root / f"{STEM}.{extension}", dpi=220, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--au-csv", type=Path,
                        default=PROJECT_ROOT / "studies/gold_motheye3/data/au_measured_nk.csv")
    parser.add_argument("--cr-csv", type=Path,
                        default=PROJECT_ROOT / "studies/cr_motheye/data/Johnson.csv")
    parser.add_argument("--output-root", type=Path,
                        default=Path(__file__).resolve().parent / "results")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--data-only", action="store_true", help="Export data without Matplotlib.")
    mode.add_argument("--plot-only", action="store_true", help="Redraw exported data and metadata.")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        with (output_root / f"{STEM}.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    else:
        rows = export_data(args.au_csv.resolve(), args.cr_csv.resolve(), output_root)
    if not args.data_only:
        plot_data(rows, output_root)
    print(f"Output: {output_root}")
    for row in rows:
        if float(row["wavelength_nm"]) in (400, 550, 700):
            print(json.dumps(row))


if __name__ == "__main__":
    main()
