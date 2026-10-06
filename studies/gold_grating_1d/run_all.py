"""Bounded search for M, Nz and grid, including checks at the selected tuple."""
from __future__ import annotations

import sys
from common import (HERE, AXES, Study, compare, export_cases, geometry_svg,
                    load_config, parser, validate_numbers, write_json)


def stable_candidate(report):
    if report["status"] != "converged_within_tested_values":
        return None
    # Choose the first point supported by two passing steps AND a stable
    # remaining tail. A single accidental small difference is insufficient.
    for index in range(2, len(report["values"])):
        if all(all(pair["passes_tolerance"] for pair in pairs[index-2:])
               for pairs in report["adjacent_changes"].values()):
            return report["values"][index]
    return None


def run_search(study, values, *, calculate, max_rounds):
    reference = {axis: values[axis][-1] for axis in AXES}
    summary = {"signature": study.signature, "reference_numerics": reference,
               "reference_coefficient_method": "analytic; independent of grid",
               "search_values": values, "wavelengths_nm": study.config["wavelengths_nm"],
               "tolerance": study.config["tolerance"], "initial_sweeps": {},
               "verification_rounds": [],
               "criterion": "Every metric for TE/TM at every sampled wavelength passes both final steps of each axis at the selected tuple, and selected vs analytic-coefficient all-high reference differs by at most tolerance.",
               "scope": "Convergence within these tested discretizations and sampled wavelengths; no bound on absolute error, other wavelengths, material uncertainty or fabricated geometry."}
    candidates = {}
    # At first use the largest other discretizations to reduce their influence
    # on each axis. Later recheck all three axes at the affordable chosen tuple.
    for axis in AXES:
        report = study.sweep(axis, values[axis], reference, calculate=calculate)
        summary["initial_sweeps"][axis] = report
        candidates[axis] = stable_candidate(report)
    if any(value is None for value in candidates.values()):
        summary["status"] = ("incomplete" if any(r["status"] == "incomplete"
                             for r in summary["initial_sweeps"].values()) else "not_converged")
        summary["axes_requiring_extension"] = [a for a, r in summary["initial_sweeps"].items()
                                                if r["status"] == "not_converged"]
        summary["axes_with_missing_cases"] = [a for a, r in summary["initial_sweeps"].items()
                                             if r["status"] == "incomplete"]
        return summary
    selected = dict(candidates)
    for iteration in range(max_rounds):
        verification = {"round": iteration+1, "numerics": dict(selected), "axis_checks": {},
                        "reference_comparison": {}}
        failed = []
        incomplete = False
        for axis in AXES:
            index = values[axis].index(selected[axis])
            triple = values[axis][index-2:index+1]
            report = study.sweep(axis, triple, selected, calculate=calculate)
            verification["axis_checks"][axis] = report
            if report["status"] != "converged_within_tested_values":
                failed.append(axis)
                incomplete |= report["status"] == "incomplete"
        reference_failed = False
        for wave in study.config["wavelengths_nm"]:
            low = study.get(selected, wave, calculate=calculate)
            high = study.get(reference, wave, calculate=calculate, analytic=True)
            if low is None or high is None:
                delta = {"passes_tolerance": False, "missing": True}
                incomplete = True
            else:
                delta = compare(low, high, study.config)
            verification["reference_comparison"][f"{wave:g}"] = delta
            reference_failed |= not delta["passes_tolerance"]
        summary["verification_rounds"].append(verification)
        summary["selected_numerics"] = dict(selected)
        if not failed and not reference_failed:
            summary["status"] = "converged_within_tested_values"
            summary["recommended_numerics"] = dict(selected)
            return summary
        if incomplete:
            summary["status"] = "incomplete"
            return summary
        promote = set(failed) | (set(AXES) if reference_failed else set())
        changed = False
        for axis in AXES:
            index = values[axis].index(selected[axis])
            if axis in promote and index+1 < len(values[axis]):
                selected[axis] = values[axis][index+1]
                changed = True
        if not changed:
            summary["status"] = "not_converged"
            summary["axes_requiring_extension"] = failed or list(AXES)
            return summary
        print(f"Recheck at raised discretizations: {selected}", flush=True)
    summary["status"] = "verification_round_limit"
    return summary


def main():
    p = parser(__doc__)
    p.add_argument("--max-rounds", type=int, default=6)
    args = p.parse_args()
    if args.prepare_only and args.report_only:
        p.error("Choose only one of --prepare-only and --report-only.")
    if args.max_rounds <= 0:
        p.error("--max-rounds must be positive.")
    config, model = load_config(args.config)
    if config["solver"]["fourier_coefficients"] != "sampled":
        p.error("Grid search needs sampled coefficients. For analytic coefficients use converge.py --axis order or slices.")
    values = config["search_values"]
    if any(len(v) < 3 for v in values.values()):
        p.error("run_all requires at least three values per axis.")
    reference = {axis: values[axis][-1] for axis in AXES}
    for axis in AXES:
        for value in values[axis]:
            validate_numbers(dict(reference, **{axis: value}))
    study = Study(config, model, args.output_dir or HERE/"results"/"all_search", args.device)
    write_json(study.output/"plan.json", {"config": config, "reference_numerics": reference,
               "signature": study.signature, "max_verification_rounds": args.max_rounds})
    geometry_svg(config, study.output/"geometry.svg", config["fixed_numerics"]["slices"])
    count = (sum(len(v) for v in values.values())-2)*len(config["wavelengths_nm"])
    print(f"Initial three sweeps: at most {count} distinct wavelength solves; each gives TE and TM.", flush=True)
    print(f"High reference: {reference}; tolerance {100*config['tolerance']:g} percentage points.", flush=True)
    if args.prepare_only:
        print(f"Prepared; no optical calculations. Structure: {study.output/'geometry.svg'}")
        return 0
    try:
        report = run_search(study, values, calculate=not args.report_only, max_rounds=args.max_rounds)
    except Exception:
        write_json(study.output/"report.json", {"status": "interrupted", "signature": study.signature,
                   "completed_cases": len(study.checkpoint["cases"]), "errors": study.checkpoint["errors"]})
        export_cases(study.checkpoint, study.output)
        raise
    write_json(study.output/"report.json", report)
    export_cases(study.checkpoint, study.output)
    if "selected_numerics" in report:
        geometry_svg(config, study.output/"selected_geometry.svg", report["selected_numerics"]["slices"])
    if "recommended_numerics" in report:
        print(f"Recommended within tested settings: {report['recommended_numerics']}")
    if report.get("axes_requiring_extension"):
        print(f"Extend search_values in config.json for: {report['axes_requiring_extension']}")
    if report.get("axes_with_missing_cases"):
        print(f"Missing saved cases for: {report['axes_with_missing_cases']}; rerun without --report-only to calculate.")
    print(f"status: {report['status']}; report: {study.output/'report.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
