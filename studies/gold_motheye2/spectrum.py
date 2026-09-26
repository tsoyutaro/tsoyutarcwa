"""Calculate and plot a fixed-order Au moth-eye reflectance spectrum.

The numerical setup matches gold_motheye2/converge.py: grid=256, Nz=100,
measured Au n,k, half S matrix, and the D6 source reduction. The default
101 points cover 400..700 nm at exactly 3 nm spacing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GRID = 256
SLICES = 100
DEFAULT_GOLD_CSV = HERE / "data" / "au_measured_nk.csv"
CSV_COLUMNS = (
    "wavelength_nm", "reflectance", "transmittance_far", "absorptance_total",
    "motheye_absorptance", "substrate_absorptance", "power_into_substrate",
    "runtime_seconds", "order", "slices", "grid", "passivity_warning",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, payload: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def save_csv(path: Path, cases: dict[str, dict[str, object]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(cases.values(), key=lambda row: float(row["wavelength_nm"])))
    temporary.replace(path)


def load_csv(path: Path) -> list[tuple[float, float]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        raw = list(csv.DictReader(handle))
    if not raw:
        raise ValueError(f"No completed wavelength cases in {path}")
    points = [(float(row["wavelength_nm"]), float(row["reflectance"]))
              for row in raw]
    if (any(not math.isfinite(w) or not math.isfinite(r) or not 0 <= r <= 1
            for w, r in points) or len({w for w, _ in points}) != len(points)):
        raise ValueError("The spectrum CSV has invalid or duplicate points")
    return sorted(points)


def plot(csv_path: Path, metadata_path: Path, destination: Path) -> None:
    points = load_csv(csv_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = [float(value) for value in metadata["wavelengths_nm"]]
    if any(w not in expected for w, _ in points):
        raise ValueError("The CSV contains wavelengths outside this run")
    width, height = 1080, 680
    left, right, top, bottom = 100, 1010, 130, 570
    start, stop = expected[0], expected[-1]
    if stop <= start:
        raise ValueError("The wavelength range must have distinct endpoints")
    y_max = max(10.0, min(100.0, math.ceil(max(r for _, r in points) * 100 / 10) * 10))
    x_for = lambda w: left + (w - start) / (stop - start) * (right - left)
    y_for = lambda r: bottom - 100 * r / y_max * (bottom - top)
    parts = [
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="49" font-size="29" font-weight="bold">Au moth-eye reflectance spectrum</text>',
        f'<text x="{left}" y="80" font-size="17">M={int(metadata["order"])}, '
        f'Nz={int(metadata["slices"])}, grid={int(metadata["grid"])}; '
        f'{len(points)}/{len(expected)} calculated wavelengths</text>',
        f'<text x="{left}" y="105" font-size="14" fill="#475569">'
        'Measured Au n,k; curves connect computed points.</text>',
    ]
    for tick in range(6):
        value = tick * y_max / 5
        y = bottom - tick * (bottom - top) / 5
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" '
                     'stroke="#e2e8f0"/>')
        parts.append(f'<text x="{left-14}" y="{y+5:.2f}" text-anchor="end" '
                     f'font-size="14" fill="#475569">{value:g}%</text>')
    for tick in range(7):
        value = start + tick * (stop - start) / 6
        x = x_for(value)
        parts.append(f'<line x1="{x:.2f}" y1="{bottom}" x2="{x:.2f}" '
                     f'y2="{bottom+7}" stroke="#64748b"/>')
        parts.append(f'<text x="{x:.2f}" y="{bottom+29}" text-anchor="middle" '
                     f'font-size="14" fill="#475569">{value:g}</text>')
    parts.extend((
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#334155"/>',
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#334155"/>',
        f'<text x="{(left+right)/2:.1f}" y="{bottom+68}" text-anchor="middle" '
        'font-size="17">Wavelength (nm)</text>',
        f'<text x="24" y="{(top+bottom)/2:.1f}" font-size="17" '
        'transform="rotate(-90 24 350)" text-anchor="middle">Reflectance (%)</text>',
    ))
    coordinates = " ".join(f"{x_for(w):.2f},{y_for(r):.2f}" for w, r in points)
    if len(points) > 1:
        parts.append(f'<polyline points="{coordinates}" fill="none" '
                     'stroke="#1d4ed8" stroke-width="2.7"/>')
    parts.extend(f'<circle cx="{x_for(w):.2f}" cy="{y_for(r):.2f}" r="2.7" '
                 'fill="#1d4ed8"/>' for w, r in points)
    parts.append(f'<text x="{left}" y="655" font-size="13" fill="#475569">'
                 + html.escape("Fixed-order spectrum; check numerical convergence at the final design.")
                 + '</text>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
           f'width="{width}" height="{height}" role="img">\n'
           '<g font-family="Arial, Helvetica, sans-serif" fill="#182235">\n'
           + "\n".join(parts) + "\n</g>\n</svg>\n")
    destination.write_text(svg, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", type=int, default=16)
    parser.add_argument("--start-nm", type=float, default=400.0)
    parser.add_argument("--stop-nm", type=float, default=700.0)
    parser.add_argument("--points", type=int, default=101)
    parser.add_argument("--gold-csv", type=Path, default=DEFAULT_GOLD_CSV)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--cascade", choices=("redheffer", "algo2a"),
                        default="redheffer")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.order < 1 or args.points < 2:
        parser.error("--order must be positive and --points must be at least 2")
    if (not math.isfinite(args.start_nm) or not math.isfinite(args.stop_nm)
            or args.start_nm <= 0 or args.stop_nm <= args.start_nm):
        parser.error("Require 0 < --start-nm < --stop-nm")
    output_dir = (args.output_dir or HERE / "results" / f"spectrum_M{args.order}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "reflectance_spectrum.csv"
    meta_path = output_dir / "spectrum_metadata.json"
    checkpoint_path = output_dir / "spectrum_checkpoint.json"
    svg_path = output_dir / "reflectance_spectrum.svg"
    if args.plot_only:
        plot(csv_path, meta_path, svg_path)
        print(svg_path)
        return 0

    import torch
    from studies.gold_motheye.converge import GeometryConfig, NumericalConfig, simulate_case
    from studies.shared.gold_dispersion import build_gold_model

    wavelengths = tuple(args.start_nm + (args.stop_nm - args.start_nm) * i / (args.points - 1)
                        for i in range(args.points))
    gold_csv = args.gold_csv.resolve()
    if not gold_csv.is_file():
        raise FileNotFoundError(f"Measured gold CSV not found: {gold_csv}")
    gold_model = build_gold_model("csv", gold_csv)
    for wavelength in wavelengths:
        gold_model(wavelength)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot see a GPU")
    geometry = GeometryConfig()
    configuration = {
        "study_version": "fixed_order_spectrum_half_smatrix_v1",
        "geometry": asdict(geometry), "order": args.order,
        "slices": SLICES, "grid": GRID, "wavelengths_nm": wavelengths,
        "gold_csv_sha256": sha256(gold_csv),
        "solver_source_sha256": sha256(HERE.parent / "gold_motheye" / "converge.py"),
        "cascade": args.cascade, "dtype": "complex128",
        "smatrix_size": "half", "symmetry_reduction": "d6-source",
    }
    signature = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    cases: dict[str, dict[str, object]] = {}
    if checkpoint_path.exists():
        saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if saved.get("signature") != signature:
            raise RuntimeError("Spectrum checkpoint settings differ; choose a new --output-dir")
        cases = saved.get("cases", {})
    numerical = NumericalConfig(order=args.order, slices=SLICES, grid=GRID)
    for index, wavelength in enumerate(wavelengths, 1):
        key = f"{wavelength:.12g}"
        if key in cases:
            continue
        print(f"spectrum {index}/{len(wavelengths)}: M={args.order}, "
              f"wavelength={wavelength:g} nm", flush=True)
        result = simulate_case(wavelength, numerical, geometry, gold_model,
                               cascade=args.cascade, use_symmetry=True,
                               symmetry_reduction="d6-source", device=device)
        if result["passivity_warning"]:
            raise RuntimeError(f"Passivity warning at wavelength={wavelength:g} nm")
        reflectance = float(result["reflectance"])
        if not math.isfinite(reflectance) or not 0 <= reflectance <= 1:
            raise RuntimeError(f"Nonphysical reflectance at wavelength={wavelength:g} nm")
        cases[key] = result
        save_json(checkpoint_path, {"signature": signature, "cases": cases})
        save_csv(csv_path, cases)
        save_json(meta_path, {**configuration, "completed_points": len(cases),
                              "expected_points": len(wavelengths),
                              "gold_csv": str(gold_csv)})
        plot(csv_path, meta_path, svg_path)
    # Recover cleanly if an earlier job ended between checkpoint and plot writes.
    save_csv(csv_path, cases)
    save_json(meta_path, {**configuration, "completed_points": len(cases),
                          "expected_points": len(wavelengths),
                          "gold_csv": str(gold_csv)})
    plot(csv_path, meta_path, svg_path)
    print(f"spectrum: {csv_path}")
    print(f"figure: {svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
