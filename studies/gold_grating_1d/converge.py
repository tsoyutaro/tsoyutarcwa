"""Run or extend one discretization sweep; other settings stay fixed."""
from __future__ import annotations

import math
import sys
from common import (HERE, AXES, Study, export_cases, geometry_svg, integers,
                    load_config, parser, validate_numbers, write_json)


def main():
    p = parser(__doc__)
    p.add_argument("--axis", choices=AXES, required=True)
    p.add_argument("--values", type=integers, help="Positive, increasing comma-separated integers.")
    for axis in AXES:
        p.add_argument("--"+axis, type=int)
    p.add_argument("--wavelengths", help="Override wavelengths, e.g. 400,550,700.")
    args = p.parse_args()
    if args.prepare_only and args.report_only:
        p.error("Choose only one of --prepare-only and --report-only.")
    config, model = load_config(args.config)
    if args.axis == "grid" and config["solver"]["fourier_coefficients"] == "analytic":
        p.error("Analytic Fourier coefficients are independent of grid; use sampled coefficients for a grid sweep.")
    if args.wavelengths:
        waves = [float(w) for w in args.wavelengths.split(",")]
        if (not waves or any(not math.isfinite(w) or w <= 0 for w in waves) or
            any(a >= b for a, b in zip(waves, waves[1:]))):
            p.error("Wavelengths must be finite, positive and strictly increasing.")
        for wave in waves:
            if model(wave).imag <= 0:
                p.error("An absorbing Au substrate is required.")
        config["wavelengths_nm"] = waves
    fixed = dict(config["fixed_numerics"])
    for axis in AXES:
        if getattr(args, axis) is not None:
            fixed[axis] = getattr(args, axis)
    values = args.values or config["search_values"][args.axis]
    for value in values:
        validate_numbers(dict(fixed, **{args.axis: value}),
                         sampled=config["solver"]["fourier_coefficients"] == "sampled")
    output = args.output_dir or HERE/"results"/f"{args.axis}_M{fixed['order']}_Nz{fixed['slices']}_grid{fixed['grid']}"
    study = Study(config, model, output, args.device)
    plan = {"axis": args.axis, "values": values, "fixed_numerics": fixed,
            "config": config, "signature": study.signature}
    write_json(study.output/"plan.json", plan)
    geometry_svg(config, study.output/"geometry.svg", fixed["slices"])
    if args.prepare_only:
        print(f"Prepared; no optical calculations. Plan: {study.output/'plan.json'}")
        print(f"Structure: {study.output/'geometry.svg'}")
        return 0
    try:
        report = study.sweep(args.axis, values, fixed, calculate=not args.report_only)
    except Exception:
        write_json(study.output/"report.json", study.sweep_report(args.axis, values, fixed))
        export_cases(study.checkpoint, study.output)
        raise
    write_json(study.output/"report.json", report)
    from plot import plot_sweep
    plot_sweep(study.checkpoint, report, study.output)
    export_cases(study.checkpoint, study.output)
    print(f"status: {report['status']}; checkpoint: {study.path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
