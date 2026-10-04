"""Run the 100-slice, M=20 Au/PMMA spectrum at 400:3:700 nm.

The measured PMMA CSV begins at 404.7 nm. This launcher writes a separate
CSV with one linearly extrapolated 400 nm point, then uses the existing
checkpointed RCWA runner. The original measured CSV is never modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "measured_30nm_Nz100_M20_400_700_101pt"
WAVELENGTHS = "400:700:3"  # inclusive: 101 points


def prepare_pmma_csv(output: Path) -> Path:
    source = HERE / "data" / "Szczurowski.csv"
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 3 or rows[0] != ["wl", "n"] or any(len(row) != 2 for row in rows[1:]):
        raise ValueError(f"Unexpected PMMA CSV format: {source}")
    first_um, first_n = map(float, rows[1])
    second_um, second_n = map(float, rows[2])
    first_nm, second_nm = first_um * 1000.0, second_um * 1000.0
    if not 400.0 < first_nm <= 405.0 or second_nm <= first_nm:
        raise ValueError("Expected the measured PMMA table to begin just above 400 nm.")
    n_400 = first_n + (400.0 - first_nm) * (second_n - first_n) / (second_nm - first_nm)
    derived = output / "Szczurowski_linear_to_400nm.csv"
    with derived.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(rows[0])
        writer.writerow(["0.4000", f"{n_400:.15g}"])
        writer.writerows(rows[1:])
    note = {
        "original_csv": str(source.resolve()),
        "original_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "derived_csv": str(derived.resolve()),
        "derived_sha256": hashlib.sha256(derived.read_bytes()).hexdigest(),
        "method": "linear extrapolation using the first two measured PMMA points",
        "source_points_nm_n": [[first_nm, first_n], [second_nm, second_n]],
        "derived_point_nm_n": [400.0, n_400],
        "affected_spectrum_wavelengths_nm": [400.0, 403.0],
    }
    (output / "pmma_400nm_extension.json").write_text(
        json.dumps(note, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return derived


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prepare-only", action="store_true",
                        help="Write material/geometry previews without an RCWA calculation")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pmma_csv = prepare_pmma_csv(output)
    print("M=20, Nz=100, 400-700 nm in 3 nm steps (101 wavelengths)", flush=True)
    print("PMMA at 400 and 403 nm uses a short linear edge extrapolation; see "
          f"{output / 'pmma_400nm_extension.json'}", flush=True)
    command = [
        sys.executable, str(HERE / "run_pmma_gold_30nm.py"),
        "--pmma-csv", str(pmma_csv),
        "--gold-csv", str(HERE / "data" / "au_measured_nk.csv"),
        "--solver-root", str(HERE.parents[1]),
        "--output-dir", str(output),
        "--wavelengths", WAVELENGTHS,
        "--order", "20",
        "--slices", "100",
        "--grid", "256",
        "--mapping", "outer",
        "--valley-gold-nm", "30",
        "--device", args.device,
    ]
    if args.prepare_only:
        command.append("--preview-only")
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
