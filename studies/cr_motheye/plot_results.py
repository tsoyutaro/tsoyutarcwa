"""Plot the Cr convergence deltas and absolute powers against each numerical axis."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studies.gold_motheye.plot_results import plot_convergence, plot_spectrum, read_spectrum


def make_figures(report_path, output_dir=None):
    report_path = Path(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    suffix = "_convergence.json"
    if not report_path.name.endswith(suffix):
        raise ValueError("Expected a *_convergence.json report.")
    prefix = report_path.name[:-len(suffix)]
    output = Path(output_dir) if output_dir else report_path.parent / (prefix + "_figures")
    output.mkdir(parents=True, exist_ok=True)
    anchor = report_path.parent / (prefix + "_anchor_spectrum.csv")
    spectrum = report_path.parent / (prefix + "_spectrum.csv")
    chosen = anchor
    if report["status"] == "converged" and spectrum.exists():
        dense = read_spectrum(spectrum)
        if all(int(row[k]) == int(report["recommendation"][k])
               for row in dense for k in ("order", "slices", "grid")):
            chosen = spectrum
    plot_convergence(report, output / "convergence.svg", False)
    plot_spectrum(read_spectrum(chosen), report, chosen, output / "spectrum.svg", False)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"Saved SVG convergence/spectrum in {output}; Matplotlib is needed for absolute axis plots.")
        return output
    cases = json.loads((report_path.parent / (prefix + "_checkpoint.json")).read_text(encoding="utf-8"))
    if cases["signature"] != report["signature"]:
        raise ValueError("Checkpoint and convergence report signatures disagree.")
    rows = list(cases["cases"].values())
    wavelengths = report["assumptions"]["anchor_wavelengths"]
    axis_labels = {"order": "Fourier order M", "slices": "Height slices Nz", "grid": "ASR integration grid"}
    for axis in axis_labels:
        record = next(r for r in reversed(report["history"]) if r["axis"] == axis)
        fixed = record["fixed"]
        selected = [row for row in rows if int(row[axis]) in record["candidates"]
                    and all(int(row[k]) == int(fixed[k]) for k in ("order", "slices", "grid") if k != axis)]
        fig, panels = plt.subplots(len(wavelengths), 2, figsize=(10, 2.8 * len(wavelengths)), squeeze=False)
        for i, w in enumerate(wavelengths):
            group = sorted((row for row in selected if float(row["wavelength_nm"]) == float(w)), key=lambda r: r[axis])
            if len(group) != len(record["candidates"]):
                raise ValueError(f"Incomplete {axis} scan at {w} nm.")
            x = [row[axis] for row in group]
            keys = (("reflectance", "R"), ("absorptance_total", "A total"))
            split = (("power_into_substrate", "P substrate"), ("motheye_absorptance", "A moth-eye"))
            if report["assumptions"]["geometry"]["substrate_mode"] == "finite":
                split = (("transmittance_far", "T far"),)
            for panel, series in zip(panels[i], (keys, split)):
                for key, label in series:
                    panel.plot(x, [row[key] for row in group], "o-", label=label, markersize=4)
                panel.set(xlabel=axis_labels[axis], ylabel="Power / incident power", title=f"{w:g} nm")
                panel.grid(alpha=0.25)
                panel.legend()
        fixed_text = ", ".join(f"{k}={fixed[k]}" for k in ("order", "slices", "grid") if k != axis)
        fig.suptitle(f"Chromium moth-eye | cycle {record['cycle']} | {fixed_text}\n{report['status']}")
        fig.tight_layout()
        for extension in ("png", "svg"):
            fig.savefig(output / (f"absolute_{axis}." + extension), dpi=170)
        plt.close(fig)
    print(f"Figures: {output.resolve()}")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=PACKAGE / "results" / "cr_motheye_convergence.json")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    make_figures(args.report, args.output_dir)


if __name__ == "__main__":
    main()
