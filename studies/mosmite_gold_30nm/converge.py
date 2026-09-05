"""Convergence scan for the 30 nm Au-coated MOSMITE hybrid model."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

_OUTPUTS_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_ROOT = Path(__file__).resolve().parent
if str(_OUTPUTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_OUTPUTS_ROOT))

from studies.mosmite_gold_30nm.common import (
    NumericalConfig,
    add_arguments,
    configs_from_args,
    parse_int_list,
    parse_wavelengths,
    simulate_case,
    write_csv,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--axis", choices=("order", "slices", "grid"), required=True)
    parser.add_argument("--candidates", required=True, help="comma-separated positive integers")
    parser.add_argument("--wavelengths", default="400,550,700")
    parser.add_argument("--order", type=int, default=8)
    parser.add_argument("--slices", type=int, default=48)
    parser.add_argument("--grid", type=int, default=384)
    parser.add_argument("--tolerance", type=float, default=5.0e-3)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=_PACKAGE_ROOT / "results" / "convergence" / "mosmite_au30",
    )
    args = parser.parse_args()
    geometry, gold, device = configs_from_args(args)
    base = NumericalConfig(args.order, args.slices, args.grid)
    candidates = parse_int_list(args.candidates, "candidates")
    wavelengths = parse_wavelengths(args.wavelengths)
    if len(candidates) < 3:
        raise ValueError("at least three candidates are required")
    if args.tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = args.output_prefix.with_name(args.output_prefix.name + "_checkpoint.json")
    signature = {
        "axis": args.axis,
        "candidates": candidates,
        "wavelengths": wavelengths,
        "geometry": asdict(geometry),
        "base": asdict(base),
        "cascade": args.cascade,
    }
    cases: dict[str, dict[str, object]] = {}
    if checkpoint.exists():
        stored = json.loads(checkpoint.read_text(encoding="utf-8"))
        if stored.get("signature") != json.loads(json.dumps(signature)):
            raise RuntimeError("checkpoint settings differ; choose another --output-prefix")
        cases = stored.get("cases", {})

    spectra: dict[int, list[dict[str, object]]] = {}
    for candidate in candidates:
        numerical = replace(base, **{args.axis: candidate})
        spectra[candidate] = []
        for wavelength in wavelengths:
            key = f"{candidate}:{wavelength:.12g}"
            if key not in cases:
                print(
                    f"solve: {args.axis}={candidate}, lambda={wavelength:g} nm",
                    flush=True,
                )
                cases[key] = simulate_case(
                    wavelength,
                    numerical,
                    geometry,
                    gold,
                    device=device,
                    cascade=args.cascade,
                )
                write_json(checkpoint, {"signature": signature, "cases": cases})
            spectra[candidate].append(cases[key])

    comparisons = []
    for coarse, fine in zip(candidates, candidates[1:]):
        per_metric = {}
        for name in ("reflectance", "transmittance", "absorptance"):
            per_metric[name] = max(
                abs(float(left[name]) - float(right[name]))
                for left, right in zip(spectra[coarse], spectra[fine])
            )
        maximum = max(per_metric.values())
        comparisons.append(
            {
                "coarse": coarse,
                "fine": fine,
                "maximum_absolute_change": maximum,
                "per_metric": per_metric,
                "passed": maximum <= args.tolerance,
            }
        )
    selected = candidates[-1]
    converged = False
    for first, second in zip(comparisons, comparisons[1:]):
        if first["passed"] and second["passed"]:
            selected = first["fine"]
            converged = True
            break
    report = {
        "status": "converged" if converged else "candidate_range_insufficient",
        "model": "hybrid monotone-double-matched-ASR / periodic-union raster",
        "signature": signature,
        "tolerance": args.tolerance,
        "criterion": "two consecutive adjacent refinements below tolerance",
        "comparisons": comparisons,
        "selected": selected,
        "selected_spectrum": spectra[selected],
        "device": str(device),
    }
    write_json(args.output_prefix.with_suffix(".json"), report)
    write_csv(args.output_prefix.with_suffix(".csv"), cases.values())
    print(json.dumps({"status": report["status"], "selected": selected}, indent=2))
    print(f"report: {args.output_prefix.with_suffix('.json').resolve()}")
    return 0 if converged else 2


if __name__ == "__main__":
    raise SystemExit(main())

