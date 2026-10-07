"""Search Fourier order and PMMA slices for a 1D Au-coated PMMA grating."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from studies.pmma_gold_grating_1d.common import (HERE, Study, compare, export_cases, geometry_svg, integers,
                    load_config, parser, read_json, validate_numbers, write_json)

SEARCH_AXES = ("order", "slices")


def stable_candidate(report):
    if report["status"] != "converged_within_tested_values":
        return None
    for index in range(2, len(report["values"])):
        if all(all(pair["passes_tolerance"] for pair in pairs[index-2:])
               for pairs in report["adjacent_changes"].values()):
            return report["values"][index]
    return None


def run_search(study, values, *, calculate, max_rounds):
    if study.config["solver"]["fourier_coefficients"] != "analytic":
        raise ValueError("This follow-up requires analytic rectangular Fourier coefficients.")
    reference = dict(study.config["fixed_numerics"])
    reference.update({axis: values[axis][-1] for axis in SEARCH_AXES})
    summary = {"signature": study.signature, "formulation": "1D Cartesian Li",
               "coefficient_method": "analytic", "grid_used": None,
               "reference_numerics": reference, "search_values": values,
               "wavelengths_nm": study.config["wavelengths_nm"],
               "material_assumptions": study.config["material"]["metadata"],
               "tolerance": study.config["tolerance"], "initial_sweeps": {},
               "verification_rounds": [],
               "criterion": "Every R/T/A metric for TE and TM at every sampled wavelength passes both final adjacent steps of order and slices at the selected tuple, and selected vs all-high analytic reference differs by at most tolerance.",
               "scope": "Tested M and Nz, sampled wavelengths, analytic coefficients; no absolute-error or continuous-spectrum guarantee."}
    candidates = {}
    for axis in SEARCH_AXES:
        report = study.sweep(axis, values[axis], reference, calculate=calculate)
        summary["initial_sweeps"][axis] = report
        candidates[axis] = stable_candidate(report)
    if any(value is None for value in candidates.values()):
        summary["axes_requiring_extension"] = [axis for axis, report in summary["initial_sweeps"].items()
                                               if report["status"] == "not_converged"]
        summary["axes_with_missing_cases"] = [axis for axis, report in summary["initial_sweeps"].items()
                                             if report["status"] == "incomplete"]
        summary["status"] = "incomplete" if summary["axes_with_missing_cases"] else "not_converged"
        return summary
    selected = dict(reference, **candidates)
    for iteration in range(max_rounds):
        verification = {"round": iteration+1,
                        "numerics": {axis: selected[axis] for axis in SEARCH_AXES},
                        "grid_used": None, "axis_checks": {}, "reference_comparison": {}}
        failed, incomplete = [], False
        for axis in SEARCH_AXES:
            index = values[axis].index(selected[axis])
            report = study.sweep(axis, values[axis][index-2:index+1], selected, calculate=calculate)
            verification["axis_checks"][axis] = report
            if report["status"] != "converged_within_tested_values":
                failed.append(axis)
                incomplete |= report["status"] == "incomplete"
        reference_failed = False
        for wave in study.config["wavelengths_nm"]:
            low = study.get(selected, wave, calculate=calculate)
            high = study.get(reference, wave, calculate=calculate)
            if low is None or high is None:
                delta = {"passes_tolerance": False, "missing": True}
                incomplete = True
            else:
                delta = compare(low, high, study.config)
            verification["reference_comparison"][f"{wave:g}"] = delta
            reference_failed |= not delta["passes_tolerance"]
        summary["verification_rounds"].append(verification)
        summary["selected_numerics"] = {axis: selected[axis] for axis in SEARCH_AXES}
        if not failed and not reference_failed:
            summary["status"] = "converged_within_tested_values"
            summary["recommended_numerics"] = {axis: selected[axis] for axis in SEARCH_AXES}
            return summary
        if incomplete:
            summary["status"] = "incomplete"
            return summary
        promote = set(failed) | (set(SEARCH_AXES) if reference_failed else set())
        changed = False
        for axis in SEARCH_AXES:
            index = values[axis].index(selected[axis])
            if axis in promote and index+1 < len(values[axis]):
                selected[axis] = values[axis][index+1]
                changed = True
        if not changed:
            summary["status"] = "not_converged"
            summary["axes_requiring_extension"] = failed or list(SEARCH_AXES)
            return summary
        print(f"Recheck analytic M={selected['order']}, Nz={selected['slices']}", flush=True)
    summary["status"] = "verification_round_limit"
    return summary


def main():
    p = parser(__doc__)
    p.add_argument("--orders", type=integers, help="Default: config order search; saved plan reused when resuming.")
    p.add_argument("--slice-values", type=integers, help="Default: config slices search; saved plan reused when resuming.")
    p.add_argument("--max-rounds", type=int, default=6)
    args = p.parse_args()
    if args.prepare_only and args.report_only:
        p.error("Choose only one of --prepare-only and --report-only.")
    if args.max_rounds <= 0:
        p.error("--max-rounds must be positive.")
    config, model = load_config(args.config)
    config["solver"]["fourier_coefficients"] = "analytic"
    output = args.output_dir or HERE/"results"/"analytic_search"
    study = Study(config, model, output, args.device)
    old_plan_path = study.output/"plan.json"
    old_plan = read_json(old_plan_path) if old_plan_path.exists() else {}
    saved_values = old_plan.get("search_values", {}) if old_plan.get("driver") == "analytic-two-axis-PMMA-Au" else {}
    values = {"order": args.orders or saved_values.get("order") or config["search_values"]["order"],
              "slices": args.slice_values or saved_values.get("slices") or config["search_values"]["slices"]}
    if any(len(v) < 3 for v in values.values()):
        p.error("At least three values per axis are needed for two adjacent steps.")
    reference = dict(config["fixed_numerics"], order=values["order"][-1], slices=values["slices"][-1])
    validate_numbers(reference)
    driver_sha = hashlib.sha256((HERE/"run_all.py").read_bytes()).hexdigest()
    write_json(study.output/"plan.json", {"driver": "analytic-two-axis-PMMA-Au", "config": config,
               "search_values": values, "reference_numerics": reference, "grid_used": None,
               "signature": study.signature, "driver_sha256": driver_sha,
               "max_verification_rounds": args.max_rounds})
    geometry_svg(config, study.output/"geometry.svg", reference["slices"])
    count = (sum(len(v) for v in values.values())-1)*len(config["wavelengths_nm"])
    print("Fourier coefficients: analytic Au/PMMA/air intervals; grid_used=null.", flush=True)
    print(f"M plan: {values['order']}; Nz plan: {values['slices']}.", flush=True)
    print(f"Initial two sweeps: at most {count} distinct wavelength solves; both TE and TM.", flush=True)
    print(f"Tolerance: {100*config['tolerance']:g} percentage points; output: {study.output}", flush=True)
    note = config["material"]["metadata"]
    if "derived_point_nm_n" in note and min(config["wavelengths_nm"]) < note["original_pmma_range_nm"][0]:
        print(f"PMMA below {note['original_pmma_range_nm'][0]:g} nm uses a short linear edge extrapolation to 400 nm; original CSV unchanged.", flush=True)
    if args.prepare_only:
        print(f"Prepared; no optical calculations. Structure: {study.output/'geometry.svg'}")
        return 0
    try:
        report = run_search(study, values, calculate=not args.report_only, max_rounds=args.max_rounds)
    except Exception:
        write_json(study.output/"report.json", {"status": "interrupted", "signature": study.signature,
                   "coefficient_method": "analytic", "grid_used": None,
                   "completed_cases": len(study.checkpoint["cases"]), "errors": study.checkpoint["errors"]})
        export_cases(study.checkpoint, study.output)
        raise
    report["driver_sha256"] = driver_sha
    write_json(study.output/"report.json", report)
    export_cases(study.checkpoint, study.output)
    if "selected_numerics" in report:
        geometry_svg(config, study.output/"selected_geometry.svg", report["selected_numerics"]["slices"])
    if "recommended_numerics" in report:
        print(f"Recommended analytic settings: {report['recommended_numerics']}")
    if report.get("axes_requiring_extension"):
        print(f"Extend --orders / --slice-values for: {report['axes_requiring_extension']}")
    if report.get("axes_with_missing_cases"):
        print(f"Missing saved cases for: {report['axes_with_missing_cases']}; rerun without --report-only.")
    print(f"status: {report['status']}; report: {study.output/'report.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
