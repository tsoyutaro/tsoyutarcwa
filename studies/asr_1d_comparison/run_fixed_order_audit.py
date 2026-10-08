"""Run controlled fixed-order ASR comparisons and sampled CPU projection audits.

Run from the project root:
    python -m studies.asr_1d_comparison.run_fixed_order_audit --device cuda
The same file can be copied to the project root and run directly.
Only the Python standard library is needed by this launcher; child modules use
the existing scientific dependencies and the same Python executable.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path


STUDIES = ("gold_grating_1d", "pmma_gold_grating_1d")


def increasing_list(raw, converter):
    try:
        values = [converter(item.strip()) for item in raw.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if (not values or any(not math.isfinite(value) or value <= 0 for value in values)
            or any(left >= right for left, right in zip(values, values[1:]))):
        raise argparse.ArgumentTypeError("Use positive, strictly increasing values")
    return values


def find_project_root(explicit):
    candidates = ([Path(explicit)] if explicit else
                  [Path.cwd(), *Path(__file__).resolve().parents])
    for candidate in candidates:
        candidate = candidate.resolve()
        directory = candidate / "studies/asr_1d_comparison"
        if all((directory / name).is_file() for name in
               ("compare.py", "diagnose_passivity.py")):
            return candidate
    raise ValueError("Project root not found. Run inside the project or pass --project-root PATH.")


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_jobs(args, output):
    """Always build the full design; stage selection does not change its settings."""
    selected = STUDIES if args.study == "all" else (args.study,)
    base = [sys.executable, "-u", "-m", "studies.asr_1d_comparison.compare",
            "--study", args.study, "--device", args.device, "--backend", args.backend,
            "--q-projection", "galerkin", "--orders", str(args.order),
            "--wavelengths", ",".join(f"{wave:g}" for wave in args.wavelengths),
            "--gold-slices", str(args.gold_slices), "--pmma-slices", str(args.pmma_slices),
            "--reference-order", str(args.reference_order),
            "--reference-check-order", str(args.reference_check_order),
            "--threads", str(args.threads), "--diagnostics"]
    variants = []
    for index, quad in enumerate(args.quadratures):
        variants.append(dict(name=f"internal_q{quad}",
                             stage="internal" if index == 0 else "quadrature",
                             ratios=args.ratios, quadrature=quad, G=args.baseline_G))
    for g in args.g_values:
        if g != args.baseline_G:
            variants.append(dict(name=f"G{g:g}_r{args.baseline_ratio}_q{args.quadratures[-1]}",
                                 stage="g", ratios=[args.baseline_ratio],
                                 quadrature=args.quadratures[-1], G=g))
    jobs = []
    for variant in variants:
        target = output / variant["name"]
        command = [*base, "--asr-ratios", ",".join(map(str, variant["ratios"])),
                   "--quadrature", str(variant["quadrature"]), "--G", f"{variant['G']:g}",
                   "--output-root", str(target)]
        jobs.append(dict(kind="comparison", output=str(target), command=command, **variant))
    for variant in variants:
        for study in selected:
            slices = args.gold_slices if study == STUDIES[0] else args.pmma_slices
            for wave in args.wavelengths:
                for ratio in variant["ratios"]:
                    name = f"{study}_{wave:g}nm_r{ratio}_q{variant['quadrature']}_G{variant['G']:g}"
                    target = output / "projection" / f"{name}.json"
                    command = [sys.executable, "-u", "-m",
                               "studies.asr_1d_comparison.diagnose_passivity",
                               "--study", study, "--order", str(args.order),
                               "--wavelength", f"{wave:g}", "--slices", str(slices),
                               "--ratio", str(ratio), "--quadrature", str(variant["quadrature"]),
                               "--G", f"{variant['G']:g}", "--output", str(target)]
                    jobs.append(dict(kind="projection", stage="projection", name=name,
                                     output=str(target), command=command))
    return jobs


def verify_comparison(job, args):
    """compare.py records per-case failures while still returning exit status 0."""
    selected = STUDIES if args.study == "all" else (args.study,)
    errors = []
    for study in selected:
        slices = args.gold_slices if study == STUDIES[0] else args.pmma_slices
        path = Path(job["output"]) / study / "checkpoint.json"
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
        requested = [("li", order) for order in
                     sorted({args.order, args.reference_check_order, args.reference_order})]
        requested += [(f"asr_r{ratio}", args.order) for ratio in job["ratios"]]
        for series, order in requested:
            for wave in args.wavelengths:
                key = f"{series}|{slices}|{order}|{wave:.12g}"
                case = checkpoint["cases"].get(key)
                if not case or case.get("error"):
                    errors.append(f"{study}/{key}: {case.get('error') if case else 'missing case'}")
                    continue
                for pol in ("TE", "TM"):
                    if not case.get("polarizations", {}).get(pol):
                        errors.append(f"{study}/{key}: missing {pol} result")
                    if series != "li":
                        condition = case.get("diagnostics", {}).get(pol, {}).get("max_boundary_condition")
                        if condition is None or not math.isfinite(condition):
                            errors.append(f"{study}/{key}: missing/nonfinite {pol} boundary condition; "
                                          "use a fresh --output-root if cached without --diagnostics")
    if errors:
        raise RuntimeError("\n".join(errors))


def export_summaries(jobs, output):
    comparison_rows, projection_rows = [], []
    for job in jobs:
        target = Path(job["output"])
        if job["kind"] == "comparison":
            for study in STUDIES:
                table, checkpoint = target / study / "cases.csv", target / study / "checkpoint.json"
                if not table.is_file() or not checkpoint.is_file():
                    continue
                cases = json.loads(checkpoint.read_text(encoding="utf-8"))["cases"]
                with table.open(encoding="utf-8-sig", newline="") as handle:
                    for row in csv.DictReader(handle):
                        key = (f"{row['series']}|{row['slices']}|{row['order']}|"
                               f"{float(row['wavelength_nm']):.12g}")
                        boundary = cases.get(key, {}).get("diagnostics", {}).get(
                            row["polarization"], {}).get("max_boundary_condition")
                        comparison_rows.append(dict(campaign=job["name"], campaign_stage=job["stage"],
                                                    campaign_G=job["G"],
                                                    quadrature_minimum=job["quadrature"],
                                                    **row, max_boundary_condition=boundary))
        elif target.is_file():
            data = json.loads(target.read_text(encoding="utf-8"))
            for layer in data["selected_layers"]:
                galerkin = layer["projections"]["galerkin"]
                projection_rows.append(dict(
                    study=data["study"], wavelength_nm=data["wavelength_nm"],
                    order=data["order"], harmonics=data["harmonics"],
                    eigen_dimension=data["internal_dimension"], G=data["G"],
                    slices=data["slices"], quadrature_minimum=data["quadrature_minimum"],
                    quadrature_actual=layer["quadrature_actual"],
                    backend=data["backend"], device=data["device"],
                    layer_index_zero_based=layer["layer_index_zero_based"], layer_name=layer["name"],
                    projected_field_condition_2norm=layer["projected_field_condition"],
                    generalized_eigen_residual=layer["generalized_eigen_residual"],
                    min_imag_gamma=layer["min_imag_gamma"],
                    galerkin_power_metric_relative_error=galerkin["power_metric_relative_error"],
                    field_generator_loss_relative_minimum=galerkin["field_generator_loss"]["relative_minimum"],
                    companion_generator_loss_relative_minimum=galerkin["companion_generator_loss"]["relative_minimum"]))
    write_csv(output / "comparison_summary.csv", comparison_rows)
    write_csv(output / "projection_summary.csv", projection_rows)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", default="cuda", help="Comparison device: cuda, cuda:0, cpu, or auto")
    parser.add_argument("--backend", choices=("auto", "scipy", "torch"), default="auto")
    parser.add_argument("--stage", choices=("all", "compare", "internal", "quadrature", "g", "projection"), default="all")
    parser.add_argument("--study", choices=("all", *STUDIES), default="all")
    parser.add_argument("--order", type=int, default=48)
    parser.add_argument("--ratios", type=lambda raw: increasing_list(raw, int), default=[3, 4, 5])
    parser.add_argument("--quadratures", type=lambda raw: increasing_list(raw, int), default=[2048, 4096])
    parser.add_argument("--g-values", type=lambda raw: increasing_list(raw, float), default=[.0003, .001, .003])
    parser.add_argument("--baseline-G", type=float, default=.001)
    parser.add_argument("--baseline-ratio", type=int, default=4)
    parser.add_argument("--wavelengths", type=lambda raw: increasing_list(raw, float), default=[650., 700.])
    parser.add_argument("--gold-slices", type=int, default=420)
    parser.add_argument("--pmma-slices", type=int, default=300)
    parser.add_argument("--reference-order", type=int, default=96)
    parser.add_argument("--reference-check-order", type=int, default=88)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Print commands only; create no result files")
    args = parser.parse_args(argv)
    if min(args.order, args.gold_slices, args.pmma_slices, args.threads) < 1:
        parser.error("Order, slices, and threads must be positive")
    if not args.order < args.reference_order or not 0 < args.reference_check_order < args.reference_order:
        parser.error("Require reference-order > order and 0 < reference-check-order < reference-order")
    if len(args.quadratures) < 2:
        parser.error("Use at least two quadrature settings")
    if args.baseline_ratio not in args.ratios or args.baseline_G not in args.g_values:
        parser.error("The baseline ratio and G must be included in their sweep lists")
    minimum = max(48, 2 * (max(args.ratios) + 1) * (2 * args.order + 1))
    if min(args.quadratures) < minimum:
        parser.error(f"Use quadratures >= {minimum} so all ratios use identical actual integration points")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        project = find_project_root(args.project_root)
        raw_output = args.output_root or Path(f"studies/asr_1d_comparison/results/fixed_order_audit_M{args.order}")
        output = (raw_output if raw_output.is_absolute() else project / raw_output).resolve()
        jobs = build_jobs(args, output)
        selected = [job for job in jobs if args.stage == "all" or job["stage"] == args.stage
                    or args.stage == "compare" and job["kind"] == "comparison"]
        n = 2 * args.order + 1
        print(f"Project: {project}\nOutput: {output}\nM={args.order}, N={n}; "
              f"internal dimensions={[ratio*n for ratio in args.ratios]}", flush=True)
        print(f"Comparison: device={args.device}, backend={args.backend}, Galerkin, TE/TM.\n"
              "Projection audit: CPU/SciPy, TM, sampled layers/interfaces (not a GPU/all-layer condition audit).",
              flush=True)
        if args.dry_run:
            for index, job in enumerate(selected, 1):
                print(f"[{index}/{len(selected)}] {job['name']}\n{shlex.join(job['command'])}")
            return 0
        settings = {key: value for key, value in vars(args).items()
                    if key not in ("project_root", "output_root", "stage", "dry_run")}
        manifest = output / "audit_plan.json"
        if manifest.is_file():
            previous = json.loads(manifest.read_text(encoding="utf-8"))
            if previous["settings"] != settings:
                raise ValueError("Audit settings changed. Use a new --output-root to keep conditions separate.")
        save_json(manifest, dict(settings=settings, project_root=str(project), jobs=jobs,
                                 note="Fixed retained order; finite Li reference is not an exact solution. "
                                      "Projection audits are sampled CPU/SciPy runs."))
        try:
            for index, job in enumerate(selected, 1):
                print(f"\n[{index}/{len(selected)}] {job['kind']}: {job['name']}", flush=True)
                subprocess.run(job["command"], cwd=project, check=True)
                if job["kind"] == "comparison":
                    verify_comparison(job, args)
        finally:
            export_summaries(jobs, output)
        print(f"\nFinished: {output}\n"
              "Summary files: comparison_summary.csv / projection_summary.csv (when available).\n"
              "To resume comparisons, rerun with the same settings. CPU projection audits are recomputed.",
              flush=True)
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted. Completed comparison cases remain in checkpoints.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Audit stopped: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
