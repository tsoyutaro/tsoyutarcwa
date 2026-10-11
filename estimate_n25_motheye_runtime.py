"""Estimate remaining H100 runtime from saved scalar cases; no optical solves."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GROUPS = list(product(("hexagonal", "square"), ("nv", "asr")))


def load_cases(folder):
    path = folder / "checkpoint.json"
    return list(json.loads(path.read_text(encoding="utf-8"))["cases"].values()) if path.exists() else []


def planned_cases(base, axes, anchors, spectrum):
    needed = set()
    current = [values[-2] for values in axes.values()]
    for _ in range(2):
        start = current.copy()
        for axis in ("slices", "order", "grid"):
            index = list(axes).index(axis)
            for value in axes[axis]:
                numerical = current.copy()
                numerical[index] = value
                needed.update((tuple(numerical), w) for w in anchors)
            current[index] = base[index]
        if current == start:
            break
    needed.update((tuple(base), w) for w in spectrum)
    refinements = []
    for index, values in enumerate(axes.values()):
        numerical = list(base)
        numerical[index] = values[-1]
        refinements.append(tuple(numerical))
        needed.update((tuple(numerical), w) for w in spectrum)
    needed.update((tuple(values[-1] for values in axes.values()), w) for w in spectrum)
    return needed, refinements


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "studies/dielectric_motheye_n25/results/repaired_v2")
    parser.add_argument("--previous-root", type=Path, default=ROOT / "studies/dielectric_motheye_n25/results")
    parser.add_argument("--anchors", default="400,550,700")
    parser.add_argument("--audit-points", type=int, default=11)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args(argv)
    plan = json.loads((args.output_root / "plan.json").read_text(encoding="utf-8"))
    axes = {axis: plan["candidate_axes"][axis] for axis in ("order", "slices", "grid")}
    if axes != {"order": [4, 6, 8, 10, 12, 14, 16], "slices": [40, 60, 80, 100], "grid": [256, 384, 512]}:
        parser.error("This timing model covers the M4..16/Nz40..100/Q256..512 study only.")
    anchors = tuple(sorted(set(float(v) for v in args.anchors.split(","))))
    if not anchors or args.audit_points <= 0:
        parser.error("Positive audit count and nonempty wavelengths are required.")
    current = {(l, m): load_cases(args.output_root / l / m) for l, m in GROUPS}
    previous = {(l, m): load_cases(args.previous_root / l / m) for l, m in GROUPS}
    try:
        nv_per_layer = {order: statistics.median(r["runtime_seconds"] / r["slices"]
            for r in current["hexagonal", "nv"] if r["order"] == order and r["grid"] == 384)
            for order in axes["order"]}
        square_curves = {method: {order: statistics.median(r["runtime_seconds"] / r["slices"]
            for r in previous["square", method] if r["order"] == order and r["grid"] >= 256)
            for order in (4, 6, 8, 10, 12)} for method in ("nv", "asr")}
    except statistics.StatisticsError:
        parser.error("Insufficient saved GPU timings; this model needs the existing repaired_v2 NV and previous GPU cases.")

    def timing(lattice, method, numerical):
        order, slices, _ = numerical
        if lattice == "hexagonal":
            per_layer = nv_per_layer[order] * (1.56 if method == "asr" else 1.)
        else:
            curve = square_curves[method]
            exponent = math.log(curve[12] / curve[8]) / math.log(12 / 8)
            per_layer = (curve[order] if order <= 12 else curve[12] * (order / 12) ** exponent) * 1.5
        return per_layer * slices

    scenarios = []
    for label, bases, asr_gradient_multiplier in (
        ("low_order", [(6, 60, 384), (6, 60, 384), (4, 60, 384), (6, 60, 384)], 3.),
        ("moderate_order", [(6, 60, 384), (10, 60, 384), (6, 60, 384), (10, 60, 384)], 3.5),
        ("high_order", [(6, 60, 384), (14, 80, 384), (6, 60, 384), (14, 80, 384)], 4.)):
        groups = []
        for (lattice, method), base in zip(GROUPS, bases):
            needed, refinements = planned_cases(base, axes, anchors, plan["spectrum_wavelengths_nm"])
            completed = {(tuple(r[k] for k in axes), r["wavelength_nm"]) for r in current[lattice, method]}
            needed -= completed
            optical_seconds = sum(timing(lattice, method, numerical) for numerical, _ in needed)
            multiplier = asr_gradient_multiplier if method == "asr" else 1.6
            # Three measured repeats plus warmup, eight FD perturbations,
            # and one gradient at each of three refined numerical settings.
            tasks = {**{f"forward_{i}": timing(lattice, method, base) for i in range(3)},
                "forward_warmup": timing(lattice, method, base),
                **{f"gradient_{i}": multiplier * timing(lattice, method, base) for i in range(3)},
                "gradient_warmup": multiplier * timing(lattice, method, base)}
            for parameter in ("height_nm", "base_radius_nm"):
                for scale in ("1.0", "0.5"):
                    for sign in ("plus", "minus"):
                        tasks[f"fd_{parameter}_{scale}_{sign}"] = timing(lattice, method, base)
            tasks.update({f"refined_gradient_{axis}": multiplier * timing(lattice, method, numerical)
                          for axis, numerical in zip(axes, refinements)})
            # Audit tasks have their own identity. Remove them only when a
            # finished audit confirms both the mesh and wavelength count.
            audit_path = args.output_root / lattice / method / "audit.json"
            if audit_path.exists():
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
                settings = audit["settings"]
                if tuple(settings["numerical"][k] for k in axes) == base and len(settings["wavelengths"]) == args.audit_points:
                    progress_path = audit_path.with_name("audit_checkpoint.json")
                    if progress_path.exists():
                        done = json.loads(progress_path.read_text(encoding="utf-8"))["tasks"]
                        tasks = {k: v for k, v in tasks.items() if k not in done}
            audit_seconds = args.audit_points * sum(tasks.values())
            groups.append({"group": lattice + "/" + method, "assumed_numerical": base,
                "remaining_optical_cases": len(needed), "optical_hours": optical_seconds / 3600,
                "audit_hours": audit_seconds / 3600, "total_hours": (optical_seconds + audit_seconds) / 3600})
        scenarios.append({"scenario": label, "groups": groups, "total_hours": sum(g["total_hours"] for g in groups)})
    result = {"initial_anchors_nm": anchors, "audit_wavelength_count": args.audit_points,
        "spectrum_wavelength_count": len(plan["spectrum_wavelengths_nm"]), "scenarios": scenarios,
        "completed_case_counts": {l + "/" + m: len(rows) for (l, m), rows in current.items()},
        "assumptions": ["H100 timings; no new optical calculations", "Two stable axis-scan cycles",
            "No extra adaptive anchor rounds; those may increase time", "All final mesh checks and gradient audits pass",
            "ASR gradients estimated as 3-4 times forward; new GPU gradients not yet measured",
            "Square M14/16 extrapolated from old GPU cases and scaled by 1.5",
            "Forward Q512 cost approximated by Q384 cost; GPU load and host overhead excluded"]}
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for scenario in scenarios:
        print(f"{scenario['scenario']}: remaining {scenario['total_hours']:.2f} hours")
        for group in scenario["groups"]:
            print(f"  {group['group']}: optical {group['optical_hours']:.2f} h; "
                  f"audit {group['audit_hours']:.2f} h; total {group['total_hours']:.2f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
