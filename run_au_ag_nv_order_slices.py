"""Compare NV Fourier orders and height slices for Au:700 nm and Ag:550 nm.

Reuse the existing memory-safe, per-case subprocess solvers unchanged.
After failure, skip the upper-right M/Nz quadrant for that material/structure/
wavelength, while continuing smaller conditions and independent jobs.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import io
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import run_gold_motheye_order_nv as gold
import run_cu_ag_al_motheye_order_nv as silver

nv = gold.nv
BACKENDS = {"Au": gold, "Ag": silver}
FAILURE_STATUSES = {"oom", "calculation_failed", "job_setup_failed", "fixed_grid_limit"}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def integer_list(value):
    values = [int(part.strip()) for part in value.split(",")]
    if not values or min(values) < 1 or len(set(values)) != len(values):
        raise ValueError("Use distinct positive integers.")
    return sorted(values)


def build_plan(args):
    orders, slices = integer_list(args.orders), integer_list(args.slices)
    step = orders[1]-orders[0] if len(orders) > 1 else 2
    if any(b-a != step for a,b in zip(orders, orders[1:])):
        raise ValueError("--orders must be an evenly spaced list.")
    if args.grid < max(32, 4*max(orders)+4):
        raise ValueError("--grid must be >= max(32, 4*max_order+4).")
    if args.cpu_threads < 1 or not math.isfinite(args.tolerance) or args.tolerance <= 0:
        raise ValueError("Use positive cpu-threads and tolerance (power fraction).")
    if args.device not in ("cpu", "cuda") and not (args.device.startswith("cuda:") and args.device[5:].isdigit()):
        raise ValueError("Use cpu, cuda or cuda:<index>.")
    structures = ["metal", "pmma"] if args.structures == "both" else [args.structures]
    cases = []
    for part in args.cases.split(","):
        material, wave = part.strip().split(":")
        material = {"au":"Au", "ag":"Ag"}[material.lower()]
        wave = float(wave)
        if not math.isfinite(wave) or wave <= 0:
            raise ValueError("Use positive finite wavelengths.")
        if (material, wave) not in cases:
            cases.append((material, wave))
    jobs = []
    for material, wave in cases:
        backend = BACKENDS[material]
        for structure in structures:
            for nz in slices:
                fixed = backend.build_parser().parse_args([])
                fixed.device, fixed.cpu_threads = args.device, args.cpu_threads
                fixed.slices, fixed.grid = nz, args.grid
                fixed.start_order, fixed.step, fixed.max_order = min(orders), step, max(orders)
                fixed.tolerance = args.tolerance
                identity = backend.build_identity(fixed, material, structure, [wave])
                key = f"{material}|{structure}|wl={wave:g}|Nz={nz}"
                relative = f"{structure}/{material.lower()}/wl{wave:g}/Nz{nz}"
                jobs.append(dict(key=key, material=material, structure=structure, wavelength_nm=wave,
                                 slices=nz, relative_output=relative,
                                 plan=dict(identity=identity, max_order=max(orders))))
    return dict(version=1, method="NV", orders=orders, slices=slices, grid=args.grid,
                tolerance=args.tolerance, structures=structures, cases=cases, jobs=jobs,
                driver_sha256=nv.digest(Path(__file__)),
                pruning_rule="same material/structure/wavelength, M>=failed_M AND Nz>=failed_Nz",
                rcwa="Unchanged existing NV workers; complex128, D6 x source, Lanczos off, memory-safe storage")


def blockers(job, order, failures):
    return [f for f in failures if f["material"] == job["material"]
            and f["structure"] == job["structure"] and f["wavelength_nm"] == job["wavelength_nm"]
            and order >= f["order"] and job["slices"] >= f["slices"]]


def remember_failure(saved, job, status, order, detail):
    event = dict(material=job["material"], structure=job["structure"], wavelength_nm=job["wavelength_nm"],
                 slices=job["slices"], order=int(order), status=status, detail=detail)
    identity = tuple(event[k] for k in ("material", "structure", "wavelength_nm", "slices", "order", "status"))
    if not any(tuple(f[k] for k in ("material", "structure", "wavelength_nm", "slices", "order", "status")) == identity
               for f in saved["failures"]):
        saved["failures"].append(event)


def read_job_cases(job, output):
    checkpoint = output/job["relative_output"]/"checkpoint.json"
    if not checkpoint.exists():
        return {}, None
    cached = load_json(checkpoint)
    identity = job["plan"]["identity"]
    if cached["signature"] != nv.identity_signature(identity):
        raise ValueError(f"Saved job conditions changed: {checkpoint}. Use a new output directory.")
    backend = BACKENDS[job["material"]]
    for key, row in cached["cases"].items():
        backend.validate_case(row, identity)
        if key != nv.case_key(int(row["order"]), float(row["wavelength_nm"])):
            raise ValueError(f"Invalid saved case key: {checkpoint}")
    return cached["cases"], cached


def write_csv(path, rows, columns):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    nv.atomic_text(path, stream.getvalue())


def export_results(saved, output):
    plan = saved["plan"]
    cases, statuses, lookup = [], [], {}
    for job in plan["jobs"]:
        cached, _ = read_job_cases(job, output)
        for order in plan["orders"]:
            key = nv.case_key(order, job["wavelength_nm"])
            row = cached.get(key)
            blocking = blockers(job, order, saved["failures"])
            exact = [f for f in blocking if f["order"] == order and f["slices"] == job["slices"]]
            status = "ok" if row else exact[-1]["status"] if exact else "skipped_dominated" if blocking else "pending"
            statuses.append(dict(material=job["material"], structure=job["structure"], wavelength_nm=job["wavelength_nm"],
                                 order=order, slices=job["slices"], grid=plan["grid"], status=status,
                                 blocked_by=json.dumps([dict(order=f["order"], slices=f["slices"], status=f["status"]) for f in blocking]),
                                 result_directory=str(output/job["relative_output"])))
            if row:
                cases.append(row)
                lookup[(job["material"], job["structure"], job["wavelength_nm"], order, job["slices"])] = row
    write_csv(output/"all_cases.csv", cases, gold.COLUMNS)
    write_csv(output/"case_status.csv", statuses, list(statuses[0]))
    comparisons = []
    for material, wave in plan["cases"]:
        for structure in plan["structures"]:
            for axis, values, fixed_values in (("order", plan["orders"], plan["slices"]),
                                                ("slices", plan["slices"], plan["orders"])):
                for fixed in fixed_values:
                    for low, high in zip(values, values[1:]):
                        keys = [(material, structure, wave, v, fixed) if axis == "order" else
                                (material, structure, wave, fixed, v) for v in (low, high)]
                        pair = [lookup.get(k) for k in keys]
                        if any(r is None for r in pair):
                            continue
                        changes = {name:abs(pair[1][name]-pair[0][name]) for name in ("reflectance", "absorptance", "transmittance")}
                        comparisons.append(dict(material=material, structure=structure, wavelength_nm=wave,
                                                 axis=axis, fixed_slices=fixed if axis=="order" else "",
                                                 fixed_order=fixed if axis=="slices" else "", low=low, high=high,
                                                 delta_R_pp=100*changes["reflectance"], delta_A_pp=100*changes["absorptance"],
                                                 delta_T_pp=100*changes["transmittance"],
                                                 passes_R_A=max(changes[n] for n in ("reflectance","absorptance")) <= plan["tolerance"]))
    write_csv(output/"comparison_changes.csv", comparisons,
              ("material","structure","wavelength_nm","axis","fixed_slices","fixed_order","low","high","delta_R_pp","delta_A_pp","delta_T_pp","passes_R_A"))
    counts = {name:sum(s["status"]==name for s in statuses) for name in
              ("ok","oom","calculation_failed","job_setup_failed","fixed_grid_limit","skipped_dominated","pending")}
    summary = dict(status=saved["status"], method="NV", orders=plan["orders"], slices=plan["slices"],
                   grid=plan["grid"], tolerance_pp=100*plan["tolerance"], cases=plan["cases"],
                   structures=plan["structures"], planned_cases=len(statuses), counts=counts,
                   failures=saved["failures"], pruning_rule=plan["pruning_rule"],
                   interpretation="R/A adjacent differences at fixed other axis; incomplete comparisons are omitted, never treated as converged.")
    nv.write_json(output/"batch_summary.json", summary)
    return cases, summary


def plot_results(plan, cases, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for material, wave in plan["cases"]:
        for structure in plan["structures"]:
            selected = [r for r in cases if r["material"]==material and r["structure"]==structure and r["wavelength_nm"]==wave]
            if not selected:
                continue
            fig, axes = plt.subplots(2,2,figsize=(12,8), layout="constrained")
            fig.suptitle(f"{material} / {structure} | NV | {wave:g} nm\nP=200 nm, h=500 nm, Q={plan['grid']}; normal x polarization", fontsize=14)
            for i, (axis, fixed_name, fixed_values) in enumerate((("order","slices",plan["slices"]), ("slices","order",plan["orders"]))):
                for j, (field,label) in enumerate((("reflectance","Reflectance R (%)"),("absorptance","Total absorptance A (%)"))):
                    ax = axes[i,j]
                    for fixed in fixed_values:
                        pts = sorted((r for r in selected if r[fixed_name]==fixed), key=lambda r:r[axis])
                        if pts:
                            ax.plot([r[axis] for r in pts], [100*r[field] for r in pts], "o-",
                                    label=f"{'Nz' if fixed_name=='slices' else 'M'}={fixed}")
                    ax.set_xlabel("Fourier order M" if axis=="order" else "Profile slices Nz")
                    ax.set_ylabel(label)
                    ax.set_xticks(plan["orders"] if axis=="order" else plan["slices"])
                    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
                    ax.grid(alpha=.25)
                    ax.legend(fontsize=9)
            path = output/structure/material.lower()/f"NV_order_slices_wl{wave:g}"
            path.parent.mkdir(parents=True,exist_ok=True)
            fig.savefig(path.parent/(path.name+".png"),dpi=170)
            fig.savefig(path.parent/(path.name+".svg"))
            plt.close(fig)


def execute(plan, output, *, retry_failures=False):
    output = Path(output).resolve()
    checkpoint = output/"batch_checkpoint.json"
    signature = nv.identity_signature(plan)
    if checkpoint.exists():
        saved = load_json(checkpoint)
        if saved["signature"] != signature:
            raise ValueError("Batch settings/source changed; use a new --output-root.")
    else:
        saved = dict(signature=signature, plan=plan, status="running", failures=[])
    # Recover a failure if the parent was interrupted after the child persisted it.
    for job in plan["jobs"]:
        cached, state = read_job_cases(job, output)
        if state and state.get("status") in FAILURE_STATUSES:
            missing = [m for m in plan["orders"] if nv.case_key(m,job["wavelength_nm"]) not in cached]
            detail = state.get("failure") or {}
            if missing:
                remember_failure(saved,job,state["status"],detail.get("order", min(missing)),detail)
    if retry_failures:
        saved["failures"] = []
    saved["status"] = "running"
    nv.write_json(output/"batch_plan.json",plan)
    nv.write_json(checkpoint,saved)
    try:
        for job in plan["jobs"]:
            allowed = [m for m in plan["orders"] if not blockers(job,m,saved["failures"])]
            if not allowed:
                print(f"SKIP {job['key']}: all requested orders dominated by a recorded failure.",flush=True)
                continue
            effective = deepcopy(job["plan"])
            effective["max_order"] = max(allowed)
            backend = BACKENDS[job["material"]]
            path = output/job["relative_output"]
            print(f"\n=== {job['key']}: M={allowed}; Q={plan['grid']} ===",flush=True)
            try:
                report = backend.run_job(effective,path)
            except (ValueError,OSError,KeyError,RuntimeError) as error:
                report = dict(status="job_setup_failed",failure=dict(error=f"{type(error).__name__}: {error}"))
            if report["status"] in FAILURE_STATUSES:
                cached,_ = read_job_cases(job,output)
                missing = [m for m in allowed if nv.case_key(m,job["wavelength_nm"]) not in cached]
                detail = report.get("failure") or {}
                if missing:
                    failed_order = detail.get("order",min(missing))
                    remember_failure(saved,job,report["status"],failed_order,detail)
                    print(f"CONTINUE after {report['status']}: prune M>={failed_order}, Nz>={job['slices']} for this material/structure/wavelength only.",flush=True)
            nv.write_json(checkpoint,saved)
            export_results(saved,output)
    except KeyboardInterrupt:
        saved["status"] = "interrupted"
        nv.write_json(checkpoint,saved)
        export_results(saved,output)
        raise
    saved["status"] = ("completed_with_failures" if any(f["status"]!="oom" for f in saved["failures"]) else
                       "completed_with_oom_skips" if saved["failures"] else "requested_grid_complete")
    nv.write_json(checkpoint,saved)
    cases,summary = export_results(saved,output)
    print(f"\nBatch status: {saved['status']}\nCounts: {summary['counts']}\nResults: {output}",flush=True)
    return cases,summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--cases",default="Au:700,Ag:550",help="Material:wavelength_nm pairs")
    parser.add_argument("--structures",choices=("both","metal","pmma"),default="both")
    parser.add_argument("--orders",default="30,32,34,36")
    parser.add_argument("--slices",default="60,90,120",help="Profile slice counts; PMMA adds one top cap")
    parser.add_argument("--grid",type=int,default=512)
    parser.add_argument("--tolerance",type=float,default=.005,help="Absolute power fraction; .005 = .5 percentage point")
    parser.add_argument("--cpu-threads",type=int,default=4)
    parser.add_argument("--output-root",type=Path,default=ROOT/"studies/au_ag_nv_order_slices/results/M30-36_Nz60-120_Q512")
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--plot-only",action="store_true")
    parser.add_argument("--no-plots",action="store_true")
    parser.add_argument("--retry-failures",action="store_true",help="Retry previously failed conditions; successful cases remain cached")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    output = args.output_root.resolve()
    try:
        if args.plot_only:
            saved = load_json(output/"batch_checkpoint.json")
            cases,_ = export_results(saved,output)
            plot_results(saved["plan"],cases,output)
            return 0
        plan = build_plan(args)
        print(f"NV M={plan['orders']}, Nz={plan['slices']}, Q={plan['grid']}; {len(plan['jobs'])*len(plan['orders'])} requested cases.",flush=True)
        if args.dry_run:
            nv.write_json(output/"batch_plan_preview.json",plan)
            print(f"Preflight only; no RCWA calculation. Plan: {output/'batch_plan_preview.json'}")
            return 0
        cases,summary = execute(plan,output,retry_failures=args.retry_failures)
        if not args.no_plots:
            try:
                plot_results(plan,cases,output)
            except Exception as error:
                nv.atomic_text(output/"plot_error.txt",f"{type(error).__name__}: {error}\n")
                print(f"Plotting failed; calculation results are saved: {error}",file=sys.stderr)
        # OOM is an expected resource boundary. Non-OOM failures are reported
        # only after every independent/remaining feasible condition was tried.
        return 1 if summary["status"]=="completed_with_failures" else 0
    except KeyboardInterrupt:
        return 130
    except (ValueError,OSError,KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
