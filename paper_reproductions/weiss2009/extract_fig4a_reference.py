"""Extract the actual vector polylines in Weiss 2009 Fig. 4(a), not solver data.

Requires pdfplumber and numpy only. This extractor intentionally targets the
supplied publisher PDF layout and fails rather than guessing for another PDF.
The generated CSV and provenance JSON are bundled so simulation needs no PDF.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def canonical_text_sha(path: Path) -> str:
    text = path.read_text(encoding="utf-8-sig")
    canonical = "\n".join(text.splitlines()) + "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def extract(pdf: Path, output: Path) -> None:
    import pdfplumber

    with pdfplumber.open(pdf) as document:
        page = document.pages[10]
        if "Spectralbehavior" not in "".join(page.extract_text().split()):
            raise ValueError("Expected Fig. 4 on PDF page 11.")
        # Five vertical labelled grid lines: 250, 300, 350, 400, 450 THz.
        ticks = sorted({line["x0"] for line in page.lines
                        if 190 < line["x0"] < 290 and line["width"] < 1e-6
                        and abs(line["top"] - 82.7196) < .01
                        and abs(line["bottom"] - 195.6696) < .01})
        if len(ticks) != 5:
            raise ValueError("Fig. 4(a) frequency-axis calibration not recognized.")
        slope, intercept = np.polyfit(ticks, [250, 300, 350, 400, 450], 1)
        top, bottom = 82.7196, 195.6696
        curves = [c for c in page.curves if c["width"] > 110
                  and 187 < c["x0"] < 188 and c["x1"] < 301
                  and c["top"] >= top and c["bottom"] <= bottom
                  and len(c["pts"]) >= 90]
        if len(curves) != 3:
            raise ValueError("Expected exactly three Fig. 4(a) vector curves.")
        named = {}
        raw = {}
        for curve in curves:
            dash = curve.get("dash")
            name = "T" if not dash else "R" if len(dash[0]) == 1 else "A"
            if name in named:
                raise ValueError("Ambiguous curve line styles.")
            # No fitted physical model, smoothing, normalization, or clipping.
            xy = np.asarray(curve["pts"], dtype=float)
            if not np.all(np.diff(xy[:, 0]) > 0):
                raise ValueError("Reference curve is not single-valued in frequency.")
            named[name] = np.column_stack((slope * xy[:, 0] + intercept,
                                          (bottom - xy[:, 1]) / (bottom - top)))
            raw[name] = {"pdf_points_top_origin": xy.tolist(), "dash": dash,
                         "linewidth_pdf_points": curve["linewidth"]}
        if set(named) != {"T", "R", "A"}:
            raise ValueError("Missing T/R/A reference curve.")
        frequencies = named["T"][:, 0]
        if not np.allclose(named["R"][:, 0], frequencies, atol=1e-8):
            raise ValueError("T and R frequency knots differ unexpectedly.")
        for data in named.values():
            if data[0, 0] > frequencies[0] or data[-1, 0] < frequencies[-1]:
                raise ValueError("Reference interpolation would extrapolate.")
        output.mkdir(parents=True, exist_ok=True)
        target = output / "fig4a_reference.csv"
        with target.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["frequency_THz", "T", "R", "A"])
            for f in frequencies:
                writer.writerow([f"{f:.10f}"] + [
                    f"{np.interp(f, named[c][:, 0], named[c][:, 1]):.10f}"
                    for c in ("T", "R", "A")])
        metadata = {
            "source_kind": "publisher_figure_vector_digitization",
            "paper_doi": "10.1364/OE.17.008051", "figure": "4(a)",
            "pdf_page_1based": 11, "source_filename": pdf.name,
            "source_pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
            "reference_csv_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "reference_csv_canonical_sha256": canonical_text_sha(target),
            "frequency_ticks_pdf_x": ticks, "frequency_ticks_THz": [250, 300, 350, 400, 450],
            "frequency_per_pdf_x": float(slope), "frequency_intercept": float(intercept),
            "tick_fit_max_residual_THz": float(np.max(np.abs(
                slope * np.asarray(ticks) + intercept - [250, 300, 350, 400, 450]))),
            "power_axis_top_pdf": top, "power_axis_bottom_pdf": bottom,
            "frequency_range_THz": [float(frequencies[0]), float(frequencies[-1])],
            "rows": len(frequencies), "curves": raw,
            "note": "Plotted polylines, not authors' raw numerical data. A is linearly "
                    "interpolated at the T/R knots; collinear A vertices were omitted in PDF. "
                    "No R+T+A normalization. Axis and line-width uncertainty remains. "
                    "Comparison tolerances are user criteria, not error bars from the paper.",
        }
        (output / "fig4a_reference_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Extracted {len(frequencies)} points: {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "reference")
    args = parser.parse_args()
    extract(args.pdf, args.output_dir)
