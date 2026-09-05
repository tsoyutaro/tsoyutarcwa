"""Calculate R/T/A spectrum of the nominal 30 nm Au-coated MOSMITE film."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_OUTPUTS_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_ROOT = Path(__file__).resolve().parent
if str(_OUTPUTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_OUTPUTS_ROOT))

from studies.mosmite_gold_30nm.common import (
    NumericalConfig,
    add_arguments,
    configs_from_args,
    parse_wavelengths,
    simulate_case,
    write_csv,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--wavelengths", default="400:700:10")
    parser.add_argument("--order", type=int, default=8)
    parser.add_argument("--slices", type=int, default=48)
    parser.add_argument("--grid", type=int, default=384)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=_PACKAGE_ROOT / "results" / "mosmite_au30",
    )
    args = parser.parse_args()
    geometry, gold, device = configs_from_args(args)
    numerical = NumericalConfig(args.order, args.slices, args.grid)
    wavelengths = parse_wavelengths(args.wavelengths)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for wavelength in wavelengths:
        print(f"solve: lambda={wavelength:g} nm", flush=True)
        result = simulate_case(
            wavelength,
            numerical,
            geometry,
            gold,
            device=device,
            cascade=args.cascade,
        )
        rows.append(result)
        write_json(
            args.output_prefix.with_name(args.output_prefix.name + "_checkpoint.json"),
            {"geometry": asdict(geometry), "numerics": asdict(numerical), "results": rows},
        )

    write_csv(args.output_prefix.with_suffix(".csv"), rows)
    write_json(
        args.output_prefix.with_suffix(".json"),
        {
            "model": "hybrid monotone-double-matched-ASR / periodic-union raster",
            "geometry": asdict(geometry),
            "numerics": asdict(numerical),
            "device": str(device),
            "results": rows,
        },
    )

    figure, axis = plt.subplots(figsize=(7.2, 4.8), dpi=240)
    axis.plot(wavelengths, [row["reflectance"] for row in rows], label="R")
    axis.plot(wavelengths, [row["transmittance"] for row in rows], label="T")
    axis.plot(wavelengths, [row["absorptance"] for row in rows], label="A")
    axis.set(xlabel="Wavelength (nm)", ylabel="Power fraction", ylim=(-0.02, 1.02))
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output_prefix.with_suffix(".png"), bbox_inches="tight")
    plt.close(figure)
    print(f"results: {args.output_prefix.resolve()}.[csv|json|png]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

