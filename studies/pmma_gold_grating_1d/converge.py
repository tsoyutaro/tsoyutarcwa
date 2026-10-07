"""Sweep order or PMMA profile slices at fixed other settings, with resumption."""
from __future__ import annotations

import math
import sys
if __package__:
    from .common import HERE, AXES, Study, export_cases, geometry_svg, integers, load_config, parser, write_json
else:
    from common import HERE, AXES, Study, export_cases, geometry_svg, integers, load_config, parser, write_json


def main():
    p = parser(__doc__)
    p.add_argument("--axis", choices=AXES, required=True)
    p.add_argument("--values", type=integers)
    p.add_argument("--order", type=int)
    p.add_argument("--slices", type=int)
    p.add_argument("--wavelengths")
    args = p.parse_args()
    if args.prepare_only and args.report_only:
        p.error("Choose --prepare-only or --report-only.")
    config, model = load_config(args.config)
    if args.wavelengths:
        waves = [float(w) for w in args.wavelengths.split(",")]
        if (not waves or any(not math.isfinite(w) or w <= 0 for w in waves)
                or any(a >= b for a, b in zip(waves, waves[1:]))):
            p.error("Wavelengths must be finite, positive and increasing.")
        for wave in waves:
            model(wave)
        config["wavelengths_nm"] = waves
    fixed = dict(config["fixed_numerics"])
    for axis in AXES:
        if getattr(args, axis) is not None:
            fixed[axis] = getattr(args, axis)
    values = args.values or config["search_values"][args.axis]
    if len(values) < 3:
        p.error("At least three values are needed for two comparisons.")
    study = Study(config, model, args.output_dir or HERE/"results"/f"{args.axis}_M{fixed['order']}_Nz{fixed['slices']}", args.device)
    write_json(study.output/"plan.json", {"axis": args.axis, "values": values, "fixed_numerics": fixed,
                                         "config": config, "signature": study.signature, "grid_used": None})
    geometry_svg(config, study.output/"geometry.svg", fixed["slices"])
    if args.prepare_only:
        print(f"Prepared; no optical calculation: {study.output/'geometry.svg'}")
        return 0
    try:
        report = study.sweep(args.axis, values, fixed, calculate=not args.report_only)
    except Exception:
        report = study.sweep_report(args.axis, values, fixed)
        write_json(study.output/"report.json", report)
        from studies.pmma_gold_grating_1d.plot import plot_sweep
        plot_sweep(study.checkpoint, report, study.output)
        export_cases(study.checkpoint, study.output)
        raise
    write_json(study.output/"report.json", report)
    from studies.pmma_gold_grating_1d.plot import plot_sweep
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
