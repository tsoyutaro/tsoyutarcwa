"""Show completed RCWA order results after a later order runs out of memory.

Usage: python3 show_checkpoint_results.py [checkpoint.json ...] [--max-order 20]
Only the Python standard library is required; no optical solve is started.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parent
    / "results"
    / "measured_30nm_Nz100_order"
    / "checkpoint.json"
)
METRICS = ("reflectance", "transmittance", "absorptance")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoints", nargs="*", type=Path,
        help="One or more checkpoints from matching runs (default: Nz100 output beside this script)",
    )
    parser.add_argument("--max-order", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=0.005)
    args = parser.parse_args()
    if args.max_order <= 0 or not math.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error("--max-order and --tolerance must be positive")
    checkpoints = args.checkpoints or [DEFAULT_CHECKPOINT]
    results: dict[tuple[int, float], dict] = {}
    expected_orders: set[int] = set()
    expected_wavelengths = None
    common_settings = None
    for checkpoint_path in checkpoints:
        try:
            payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            cases = payload["cases"]
            if not isinstance(cases, dict):
                raise TypeError("checkpoint 'cases' must be a JSON object")
            settings_path = checkpoint_path.with_name("settings.json")
            if settings_path.is_file():
                settings = json.loads(settings_path.read_text(encoding="utf-8"))
                expected_orders.update(int(x) for x in settings["numerics"]["orders"])
                wavelengths_here = sorted(
                    float(item["wavelength_nm"])
                    for item in settings["materials_at_wavelengths"]
                )
                if expected_wavelengths is None:
                    expected_wavelengths = wavelengths_here
                elif wavelengths_here != expected_wavelengths:
                    raise ValueError("checkpoints have different wavelength lists")
                comparable = json.loads(json.dumps(settings))
                comparable["numerics"].pop("orders")
                if common_settings is None:
                    common_settings = comparable
                elif comparable != common_settings:
                    raise ValueError("checkpoints have different settings besides Fourier orders")
            elif len(checkpoints) > 1:
                raise ValueError(f"missing companion settings.json for {checkpoint_path}")
            for key, case in cases.items():
                order = int(case["order"])
                if order > args.max_order:
                    continue
                wavelength = float(case["wavelength_nm"])
                values = {name: float(case[name]) for name in METRICS}
                if not math.isfinite(wavelength) or not all(math.isfinite(v) for v in values.values()):
                    raise ValueError(f"non-finite result in saved case {key}")
                identity = (order, wavelength)
                if identity in results:
                    if any(results[identity][name] != value for name, value in values.items()):
                        raise ValueError(f"conflicting result for M={order}, wavelength={wavelength:g} nm")
                    continue
                results[identity] = {
                    **case, **values, "order": order, "wavelength_nm": wavelength
                }
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.error(f"cannot read checkpoint {checkpoint_path}: {exc}")
    if not results:
        parser.error(f"no completed cases through M={args.max_order}")

    wavelengths = expected_wavelengths or sorted({wavelength for _, wavelength in results})
    orders = sorted(x for x in (expected_orders or {order for order, _ in results}) if x <= args.max_order)
    print("Checkpoints: " + ", ".join(str(path) for path in checkpoints))
    print(f"Saved results through M={args.max_order}; values are fractions, not percent.")
    print(
        f"{'wavelength_nm':>13} {'M':>3} {'R':>12} {'T':>12} {'A':>12} "
        f"{'previous_M':>10} {'delta_R':>12} {'delta_T':>12}"
    )
    for wavelength in wavelengths:
        previous = None
        for order in orders:
            case = results.get((order, wavelength))
            if case is None:
                continue
            if previous is None:
                prior_order, delta_r, delta_t = "-", "-", "-"
            else:
                prior_order = str(previous["order"])
                delta_r = f"{abs(case['reflectance'] - previous['reflectance']):.6g}"
                delta_t = f"{abs(case['transmittance'] - previous['transmittance']):.6g}"
            print(
                f"{wavelength:13g} {order:3d} "
                f"{case['reflectance']:12.7g} {case['transmittance']:12.7g} "
                f"{case['absorptance']:12.7g} {prior_order:>10} "
                f"{delta_r:>12} {delta_t:>12}"
            )
            previous = case

    complete = {order for order in orders if all((order, w) in results for w in wavelengths)}
    incomplete = sorted(set(orders) - complete)
    if incomplete:
        print(f"Incomplete saved orders (excluded from convergence checks): {incomplete}")
    print("\nMaximum absolute change across saved wavelengths:")
    print(f"{'M pair':>9} {'delta_R':>12} {'delta_T':>12} {'delta_A':>12} {'pass':>6}")
    comparisons: list[tuple[int, int, bool]] = []
    for coarse, fine in zip(orders, orders[1:]):
        if coarse not in complete or fine not in complete:
            continue
        deltas = {
            name: max(abs(results[(fine, w)][name] - results[(coarse, w)][name]) for w in wavelengths)
            for name in METRICS
        }
        nonpassive = any(
            results[(order, w)].get("passivity_warning", False)
            for order in (coarse, fine) for w in wavelengths
        )
        passed = max(deltas.values()) <= args.tolerance and not nonpassive
        comparisons.append((coarse, fine, passed))
        print(
            f"{f'{coarse}-{fine}':>9} {deltas['reflectance']:12.6g} "
            f"{deltas['transmittance']:12.6g} {deltas['absorptance']:12.6g} "
            f"{str(passed):>6}"
        )
    # An early low-order plateau does not establish convergence if later orders fail.
    passing_tail = []
    for pair in reversed(comparisons):
        if not pair[2] or (passing_tail and pair[1] != passing_tail[-1][0]):
            break
        passing_tail.append(pair)
    candidate = passing_tail[-1][1] if len(passing_tail) >= 2 else None
    if candidate is None:
        print("Convergence: not confirmed by two consecutive passing steps at the high-order end.")
    else:
        print(f"Convergence candidate: M={candidate} (high-order passing sequence).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
