"""Print saved Au moth-eye values and adjacent changes without a new solve.

Uses only the Python standard library. The default checkpoint is the
700 nm order sweep at Nz=100, grid=576; another checkpoint may be supplied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = HERE / "results" / "order_700_Nz100_grid576" / "checkpoint.json"
METRICS = ("reflectance", "power_into_substrate", "motheye_absorptance")
LABELS = {"order": "M", "slices": "Nz", "grid": "grid"}


def key(value, wavelength):
    return f"{value}|{wavelength:.12g}"


def read_checkpoint(path):
    saved = json.loads(path.read_text(encoding="utf-8"))
    plan, cases = saved.get("plan"), saved.get("cases")
    if not isinstance(plan, dict) or not isinstance(cases, dict):
        raise ValueError("Use checkpoint.json, which contains both plan and cases.")
    signature = hashlib.sha256(json.dumps(plan, sort_keys=True,
                                         allow_nan=False).encode("utf-8")).hexdigest()
    if saved.get("signature") != signature:
        raise ValueError("The checkpoint signature does not match its plan.")
    if plan["axis"] not in LABELS:
        raise ValueError("Expected an order, slices or grid sweep.")
    values = plan["values"]
    wavelengths = plan["wavelengths_nm"]
    if (not values or values != sorted(set(values))
            or any(not isinstance(value, int) or value <= 0 for value in values)
            or not wavelengths or len(set(wavelengths)) != len(wavelengths)
            or any(not math.isfinite(float(w)) or float(w) <= 0 for w in wavelengths)
            or not math.isfinite(float(plan["tolerance"])) or plan["tolerance"] <= 0):
        raise ValueError("Invalid sweep values, wavelengths or tolerance.")
    for saved_key, row in cases.items():
        value, wavelength = row["value"], float(row["wavelength_nm"])
        expected = dict(plan["fixed_numerics"])
        expected[plan["axis"]] = value
        if (saved_key != key(value, wavelength) or row["axis"] != plan["axis"]
                or value not in values or wavelength not in wavelengths
                or any(row[name] != number for name, number in expected.items())
                or any(not math.isfinite(float(row[name])) for name in METRICS)):
            raise ValueError(f"Case data do not match the saved plan: {saved_key}")
    return saved


def print_results(saved):
    plan, cases = saved["plan"], saved["cases"]
    values, wavelengths = plan["values"], plan["wavelengths_nm"]
    label = LABELS[plan["axis"]]
    fixed = plan["fixed_numerics"]
    fixed_text = ", ".join(f"{LABELS[name]}={number}" for name, number in fixed.items()
                           if name != plan["axis"])
    expected_count = len(values) * len(wavelengths)
    complete = all(key(v, w) in cases for v in values for w in wavelengths)
    print(f"Saved {plan['axis']} sweep: {len(cases)}/{expected_count} cases; {fixed_text}")
    print(f"Tolerance: {100*plan['tolerance']:g} percentage points for each R/P_sub/A_pillar.")
    print("P_sub is power entering the Au substrate; A_pillar = 1 - R - P_sub.")
    all_pass = True
    for wavelength in wavelengths:
        print(f"\n{wavelength:g} nm: values in percent; runtime in seconds")
        print(f"{label:>5}        R(%)    P_sub(%) A_pillar(%)    runtime(s)")
        for value in values:
            row = cases.get(key(value, wavelength))
            if row is None:
                print(f"{value:5d}  not calculated")
                continue
            runtime = row.get("runtime_seconds")
            runtime_text = "-" if runtime is None else f"{float(runtime):.2f}"
            print(f"{value:5d} " + " ".join(f"{100*float(row[name]):11.6f}" for name in METRICS)
                  + f" {runtime_text:>13}")
        print("\nAdjacent signed changes in percentage points (+ increase, - decrease):")
        print(f"{label+' pair':>11}          dR       dP_sub    dA_pillar     max_abs   pass")
        comparisons = []
        for low, high in zip(values, values[1:]):
            first, second = cases.get(key(low, wavelength)), cases.get(key(high, wavelength))
            if first is None or second is None:
                print(f"{str(low)+'-'+str(high):>11}  not calculated")
                comparisons.append(False)
                continue
            deltas = [100*(float(second[name])-float(first[name])) for name in METRICS]
            maximum = max(abs(delta) for delta in deltas)
            passed = maximum <= 100*plan["tolerance"]
            comparisons.append(passed)
            print(f"{str(low)+'-'+str(high):>11} "
                  + " ".join(f"{delta:11.6g}" for delta in deltas)
                  + f" {maximum:11.6g}  {passed}")
        wavelength_pass = bool(complete and len(comparisons) >= 2 and all(comparisons[-2:]))
        all_pass = all_pass and wavelength_pass
        print(f"{wavelength:g} nm: final two adjacent steps pass = {wavelength_pass}")
    status = ("incomplete" if not complete else
              "converged_within_tested_values" if all_pass else "not_converged")
    print(f"\nstatus: {status}")
    print("Criterion: each metric must pass at every wavelength across both final adjacent steps.")
    print("This reports the saved sweep; other fixed discretizations are not assessed.")
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, nargs="?", default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    path = args.checkpoint.resolve()
    try:
        saved = read_checkpoint(path)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))
    print(f"checkpoint: {path}")
    print_results(saved)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
