"""Check NV slices then sweep orders, or use --orders-only at fixed Nz.

Uses the existing Au/Ag workers unchanged. Checkpoints, never pasted/rounded
CSV values, can be reused after validating solver/data/fixed-condition identity.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import run_au_ag_nv_order_slices as grid

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "studies/au_ag_nv_order_slices/results"
SCAN_KEYS = {"start_order", "step", "tolerance"}


def fixed_signature(identity):
    # Retain geometry, Nz/Q, wavelength, precision, device, source and data hashes.
    # JSON canonicalization also equates tuples in live PMMA provenance with
    # the lists necessarily stored in JSON checkpoints.
    return grid.nv.identity_signature({k: v for k, v in identity.items() if k not in SCAN_KEYS})


def group_key(job):
    return f"{job['material']}|{job['structure']}|wl={job['wavelength_nm']:g}"


def build_plan(args):
    if args.start_order < 1 or args.step < 1 or args.max_order < args.start_order:
        raise ValueError("Use positive orders/step and max-order >= start-order.")
    orders = list(range(args.start_order, args.max_order + 1, args.step))
    if orders[-1] != args.max_order:
        raise ValueError("max-order must be reached exactly by start-order + n*step.")
    slices = grid.integer_list(args.slices)
    if args.orders_only and (len(slices) != 1 or min(slices) < 2):
        raise ValueError("--orders-only requires exactly one slice count >= 2, e.g. --slices 210.")
    if not args.orders_only and (len(slices) < 3 or min(slices) < 2):
        raise ValueError("Use at least three distinct slice counts >= 2 (default 180,210,240).")
    base = grid.build_parser().parse_args([])
    for name in ("device", "cases", "structures", "grid", "tolerance", "cpu_threads"):
        setattr(base, name, getattr(args, name))
    base.orders, base.slices = str(args.max_order), ",".join(map(str, slices))
    layers = grid.build_plan(base)
    if args.orders_only:
        layers["jobs"] = []
    base.orders, base.slices = ",".join(map(str, orders)), str(slices[-1])
    sweep = grid.build_plan(base)
    return dict(version=1, method="NV", orders_only=args.orders_only, controller_sha256=grid.nv.digest(Path(__file__)),
                layer_plan=layers, order_plan=sweep,
                reuse_roots=[str(p.resolve()) for p in (args.reuse_root or [RESULTS])],
                gate="Bypassed: fixed-Nz order sweep; layer convergence is not evaluated." if args.orders_only else
                     "Every adjacent Nz pair must have both abs(delta R) and abs(delta A) <= tolerance.",
                interpretation="Order convergence at the specified Nz, wavelengths and Q only." if args.orders_only else
                               "Gate at the stated wavelength, M and Q only; not a full-spectrum/grid error bound.")


class ReuseCatalog:
    def __init__(self, roots, output):
        self.entries, self.notes, self.imports = {}, [], []
        output = Path(output).resolve()
        for root in roots:
            root = Path(root)
            if not root.exists():
                continue
            paths = [root] if root.is_file() else root.rglob("checkpoint.json")
            for path in sorted(paths):
                # Current workflow caches are validated when their jobs run.
                if not path.resolve().is_relative_to(output):
                    self.add(path)

    def add(self, path):
        path = Path(path).resolve()
        try:
            saved = grid.load_json(path)
            identity = saved["plan"]["identity"]
            backend = grid.BACKENDS[identity["material"]]
            if saved["signature"] != grid.nv.identity_signature(identity):
                raise ValueError("checkpoint identity signature mismatch")
            for key, row in saved["cases"].items():
                backend.validate_case(row, identity)
                if key != grid.nv.case_key(int(row["order"]), float(row["wavelength_nm"])):
                    raise ValueError("checkpoint case key mismatch")
            self.entries[str(path)] = saved
        except (ValueError, OSError, KeyError, TypeError) as error:
            self.notes.append(dict(path=str(path), reason=f"{type(error).__name__}: {error}"))

    def seed(self, job, path):
        backend, identity = grid.BACKENDS[job["material"]], job["plan"]["identity"]
        checkpoint = path / "checkpoint.json"
        if checkpoint.exists():
            saved = grid.load_json(checkpoint)
            if saved["signature"] != grid.nv.identity_signature(identity):
                raise ValueError(f"Saved job conditions changed: {checkpoint}")
            for key, row in saved["cases"].items():
                backend.validate_case(row, identity)
                if key != grid.nv.case_key(int(row["order"]), float(row["wavelength_nm"])):
                    raise ValueError("Saved case key changed.")
        else:
            saved = dict(signature=grid.nv.identity_signature(identity), cases={},
                         plan=job["plan"], status="seeded", failure=None)
        candidates = {}
        for source, cached in self.entries.items():
            if fixed_signature(cached["plan"]["identity"]) != fixed_signature(identity):
                continue
            for key, row in cached["cases"].items():
                m = int(row["order"])
                if (identity["start_order"] <= m <= job["plan"]["max_order"]
                        and (m - identity["start_order"]) % identity["step"] == 0):
                    backend.validate_case(row, identity)
                    candidates.setdefault(key, []).append((source, cached, row))
        for key, options in candidates.items():
            if key in saved["cases"]:
                continue
            first = options[0][2]
            conflict = any(abs(float(first[name]) - float(row[name])) > 1e-10
                           for _, _, row in options[1:] for name in backend.POWER_NAMES
                           if first.get(name) is not None and row.get(name) is not None)
            if conflict:
                self.notes.append(dict(path=";".join(o[0] for o in options),
                                       reason=f"Conflicting cached powers for {key}; recompute."))
                continue
            for source, cached, row in options:
                environment = cached.get("environment")
                if saved.get("environment") is not None and environment != saved["environment"]:
                    continue
                saved["cases"][key] = deepcopy(row)
                if environment is not None:
                    saved["environment"] = environment
                self.imports.append(dict(material=job["material"], structure=job["structure"],
                                         wavelength_nm=job["wavelength_nm"], slices=job["slices"],
                                         order=int(row["order"]), source=source, destination=str(checkpoint)))
                print(f"REUSE {group_key(job)} Nz={job['slices']}, M={row['order']}: {source}", flush=True)
                break
        saved["plan"] = job["plan"]
        if saved["cases"]:
            backend.persist(path, saved)
        return saved


def run_job(job, path, catalog, retry_failures):
    failure_path = path / "job_setup_failure.json"
    try:
        if failure_path.exists() and not retry_failures:
            report = grid.load_json(failure_path)
            if report["signature"] != grid.nv.identity_signature(job["plan"]):
                raise ValueError(f"Saved setup failure conditions changed: {failure_path}")
            return dict(cases={}), report
        saved = catalog.seed(job, path)
        if saved.get("status") in grid.FAILURE_STATUSES and not retry_failures:
            print(f"KEEP recorded {saved['status']}: {group_key(job)} Nz={job['slices']}", flush=True)
            return saved, dict(status=saved["status"], failure=saved.get("failure"))
        report = grid.BACKENDS[job["material"]].run_job(job["plan"], path)
        saved = grid.load_json(path / "checkpoint.json")
        catalog.add(path / "checkpoint.json")
        failure_path.unlink(missing_ok=True)
        return saved, report
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        report = dict(status="job_setup_failed", signature=grid.nv.identity_signature(job["plan"]),
                      failure=dict(error=f"{type(error).__name__}: {error}"))
        grid.nv.write_json(failure_path, report)
        return dict(cases={}), report


def layer_gate(rows, slices, order, wave, tolerance):
    comparisons = []
    for low, high in zip(slices, slices[1:]):
        pair = [rows.get((nz, order, wave)) for nz in (low, high)]
        change = {name: abs(pair[1][name] - pair[0][name]) if all(pair) else None
                  for name in ("reflectance", "absorptance", "transmittance")}
        passed = all(pair) and all(change[n] <= tolerance for n in ("reflectance", "absorptance"))
        comparisons.append(dict(low=low, high=high, passed=bool(passed),
                                **{f"delta_{label}_pp": 100 * change[name] if change[name] is not None else None
                                   for label, name in (("R", "reflectance"), ("A", "absorptance"), ("T", "transmittance"))}))
    return all(c["passed"] for c in comparisons), comparisons


def export(output, saved, catalog):
    layers, sweeps, gates, changes, groups = [], [], [], [], []
    for state in saved["groups"].values():
        job = state["job"]
        prefix = {n: job[n] for n in ("material", "structure", "wavelength_nm")}
        layers.extend(state["layer_rows"])
        sweeps.extend(state["order_rows"])
        gates.append(dict(**prefix, layer_status=state["layer_status"],
                          layer_gate_passed=state["layer_gate_passed"], order_status=state["order_status"]))
        for comparison in state["comparisons"]:
            changes.append(dict(**prefix, axis="slices", fixed_order=saved["plan"]["order_plan"]["orders"][-1],
                                fixed_slices="", **comparison))
        lookup = {int(r["order"]): r for r in state["order_rows"]}
        orders = saved["plan"]["order_plan"]["orders"]
        tail = []
        for low, high in zip(orders, orders[1:]):
            pair = [lookup.get(m) for m in (low, high)]
            if not all(pair):
                continue
            delta = {label: 100 * abs(pair[1][name] - pair[0][name]) for label, name in
                     (("R", "reflectance"), ("A", "absorptance"), ("T", "transmittance"))}
            passed = max(delta["R"], delta["A"]) <= 100 * saved["plan"]["order_plan"]["tolerance"]
            change = dict(**prefix, axis="order", fixed_order="", fixed_slices=job["slices"],
                          low=low, high=high, passed=passed, **{f"delta_{k}_pp": v for k, v in delta.items()})
            changes.append(change)
            tail.append(change)
        complete = len(lookup) == len(orders)
        tail_pass = complete and len(tail) >= 2 and all(c["passed"] for c in tail[-2:])
        groups.append(dict(**gates[-1], order_range_complete=complete, order_tail_R_A_passed=tail_pass,
                           layer_failure=state.get("layer_failure"), order_failure=state.get("order_failure")))
    grid.write_csv(output / "layer_cases.csv", layers, grid.gold.COLUMNS)
    grid.write_csv(output / "order_cases.csv", sweeps, grid.gold.COLUMNS)
    unique = {(r["material"], r["structure"], r["wavelength_nm"], r["slices"], r["order"]): r for r in layers + sweeps}
    grid.write_csv(output / "all_cases.csv", unique.values(), grid.gold.COLUMNS)
    grid.write_csv(output / "layer_gate.csv", gates,
                   ("material", "structure", "wavelength_nm", "layer_status", "layer_gate_passed", "order_status"))
    grid.write_csv(output / "comparison_changes.csv", changes,
                   ("material", "structure", "wavelength_nm", "axis", "fixed_order", "fixed_slices", "low", "high",
                    "delta_R_pp", "delta_A_pp", "delta_T_pp", "passed"))
    # Keep the reuse audit across interrupted/resumed controller calls.
    for event in catalog.imports:
        if event not in saved["reuse_audit"]:
            saved["reuse_audit"].append(event)
    summary = dict(status=saved["status"], method="NV", orders_only=saved["plan"].get("orders_only", False),
                   groups=groups, unique_saved_cases=len(unique),
                   layer_gate_passed=sum(g["layer_gate_passed"] for g in groups),
                   reused_cases=len(saved["reuse_audit"]), reused_sources=saved["reuse_audit"],
                   cache_notes=catalog.notes, gate=saved["plan"]["gate"],
                   interpretation=saved["plan"]["interpretation"])
    grid.nv.write_json(output / "workflow_summary.json", summary)
    grid.nv.write_json(output / "workflow_checkpoint.json", saved)
    return summary


def plot_results(output, saved):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for state in saved["groups"].values():
        job = state["job"]
        series = [(state["order_rows"], "order", saved["plan"]["order_plan"]["orders"])]
        if not saved["plan"].get("orders_only", False):
            series.insert(0, (state["layer_rows"], "slices", saved["plan"]["layer_plan"]["slices"]))
        fig, axes = plt.subplots(len(series), 3, figsize=(14, 4 * len(series)),
                                 squeeze=False, layout="constrained")
        fig.suptitle(f"{job['material']} / {job['structure']} | NV | {job['wavelength_nm']:g} nm\n"
                     f"P=200 nm, h=500 nm, Q={saved['plan']['order_plan']['grid']}; normal x polarization")
        fields = (("reflectance", "Reflectance R"), ("transmittance", "Transmittance T"),
                  ("absorptance", "Total absorptance A")) if saved["plan"].get("orders_only", False) else (
                  ("reflectance", "Reflectance R"), ("absorptance", "Total absorptance A"),
                  ("transmittance", "Transmittance T"))
        for i, (rows, axis, values) in enumerate(series):
            lookup = {int(r[axis]): r for r in rows}
            for j, (name, label) in enumerate(fields):
                ax = axes[i, j]
                y = [100 * lookup[v][name] if v in lookup else float("nan") for v in values]
                ax.plot(values, y, "o-")  # NaN breaks lines at missing/failed conditions.
                ax.set(xlabel="Profile slices Nz" if axis == "slices" else "Fourier order M",
                       ylabel=label + " (%)", xticks=values)
                ax.set_title(f"M={saved['plan']['order_plan']['orders'][-1]}" if axis == "slices" else
                             f"Nz={job['slices']}; {state['order_status']}")
                ax.ticklabel_format(axis="y", style="plain", useOffset=False)
                ax.grid(alpha=.25)
        target = output / "figures" / f"{job['material']}_{job['structure']}_wl{job['wavelength_nm']:g}_NV"
        target.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".png", ".svg"):
            fig.savefig(target.parent / (target.name + suffix), dpi=170)
        plt.close(fig)


def execute(plan, output, *, retry_failures=False, no_plots=False):
    output = Path(output).resolve()
    checkpoint = output / "workflow_checkpoint.json"
    signature = grid.nv.identity_signature(plan)
    if checkpoint.exists():
        saved = grid.load_json(checkpoint)
        if saved["signature"] != signature:
            raise ValueError("Workflow settings/source changed; use a new --output-root.")
    else:
        saved = dict(signature=signature, plan=plan, status="running", groups={}, reuse_audit=[])
    saved["status"] = "running"
    catalog = ReuseCatalog(plan["reuse_roots"], output)
    # Include validated caches from earlier calls of this workflow as reuse candidates.
    if output.exists():
        for path in sorted(output.rglob("checkpoint.json")):
            catalog.add(path)
    output.mkdir(parents=True, exist_ok=True)
    grid.nv.write_json(output / "workflow_plan.json", plan)
    try:
        for order_job in plan["order_plan"]["jobs"]:
            key = group_key(order_job)
            state = dict(job=order_job, layer_rows=[], order_rows=[], comparisons=[],
                         layer_status="running", layer_gate_passed=False, order_status="pending")
            saved["groups"][key] = state
            orders_only = plan.get("orders_only", False)
            print(f"\n=== {'FIXED NZ ORDER SWEEP' if orders_only else 'LAYER CHECK'} {key} ===", flush=True)
            for job in plan["layer_plan"]["jobs"]:
                if group_key(job) != key:
                    continue
                cached, report = run_job(job, output / "layer_check" / job["relative_output"], catalog, retry_failures)
                state["layer_rows"].extend(cached["cases"].values())
                if report["status"] in grid.FAILURE_STATUSES:
                    state.update(layer_status=report["status"], layer_failure=report.get("failure"))
                    print(f"CONTINUE other groups after {report['status']}; skip higher Nz and order sweep for {key}.", flush=True)
                    break
            if orders_only:
                passed = False
                state["layer_status"] = "not_checked_orders_only"
            else:
                lookup = {(int(r["slices"]), int(r["order"]), float(r["wavelength_nm"])): r for r in state["layer_rows"]}
                passed, comparisons = layer_gate(lookup, plan["layer_plan"]["slices"],
                                                  plan["order_plan"]["orders"][-1], order_job["wavelength_nm"],
                                                  plan["order_plan"]["tolerance"])
                passed = passed and state["layer_status"] not in grid.FAILURE_STATUSES
                state.update(layer_gate_passed=passed, comparisons=comparisons)
                if state["layer_status"] == "running":
                    state["layer_status"] = "passed" if passed else "not_converged"
            if not passed and not orders_only:
                state["order_status"] = "skipped_layer_gate"
                print(f"SKIP order sweep: {key}; layer gate={state['layer_status']}.", flush=True)
            else:
                print(f"{'ORDERS ONLY' if orders_only else 'PASS layer gate'}: {key}; "
                      f"sweep M={plan['order_plan']['orders']} at Nz={order_job['slices']}.", flush=True)
                cached, report = run_job(order_job, output / "order_sweep" / order_job["relative_output"], catalog, retry_failures)
                state.update(order_rows=list(cached["cases"].values()), order_status=report["status"],
                             order_failure=report.get("failure"))
            export(output, saved, catalog)
    except KeyboardInterrupt:
        saved["status"] = "interrupted"
        export(output, saved, catalog)
        raise
    saved["status"] = "complete"
    summary = export(output, saved, catalog)
    if not no_plots:
        try:
            plot_results(output, saved)
        except Exception as error:
            grid.nv.atomic_text(output / "plot_error.txt", f"{type(error).__name__}: {error}\n")
            print(f"Plot error (results saved): {error}", flush=True)
    for group in summary["groups"]:
        print(f"{group['material']} / {group['structure']}: layer={group['layer_status']}; order={group['order_status']}", flush=True)
    print(f"Results: {output}", flush=True)
    return summary


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cases", default="Au:700,Ag:550")
    p.add_argument("--structures", choices=("both", "metal", "pmma"), default="both")
    p.add_argument("--slices", default="180,210,240")
    p.add_argument("--orders-only", action="store_true", help="Sweep orders at one fixed Nz, bypassing layer checks.")
    p.add_argument("--start-order", type=int, default=4)
    p.add_argument("--step", type=int, default=2)
    p.add_argument("--max-order", type=int, default=36, help="Layer checks also use this order.")
    p.add_argument("--grid", type=int, default=512)
    p.add_argument("--tolerance", type=float, default=.005, help="Absolute power fraction; .005 = .5 percentage point.")
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--reuse-root", type=Path, action="append", help="Repeatable checkpoint search root; default: previous Au/Ag results.")
    p.add_argument("--output-root", type=Path, default=RESULTS / "auto_Nz180-240_M4-36_Q512")
    p.add_argument("--retry-failures", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        plan = build_plan(args)
        if args.dry_run:
            grid.nv.write_json(args.output_root.resolve() / "workflow_plan_preview.json", plan)
            print(f"Preflight: {len(plan['order_plan']['jobs'])} groups; Nz={plan['layer_plan']['slices']}; "
                  f"orders-only={args.orders_only}; orders={plan['order_plan']['orders']}. No RCWA calculation.")
            return 0
        summary = execute(plan, args.output_root, retry_failures=args.retry_failures, no_plots=args.no_plots)
        errors = {"calculation_failed", "job_setup_failed"}
        return int(any(g["layer_status"] in errors or g["order_status"] in errors for g in summary["groups"]))
    except KeyboardInterrupt:
        return 130
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
