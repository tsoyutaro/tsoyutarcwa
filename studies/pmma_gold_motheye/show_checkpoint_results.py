"""Show completed RCWA order results after a later order runs out of memory.

Usage: python3 show_checkpoint_results.py [checkpoint.json] [--max-order 20]
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
        "checkpoint", nargs="?", type=Path, default=DEFAULT_CHECKPOINT,
        help="Checkpoint from run_pmma_gold_30nm.py (default: Nz100 output beside this script)",
    )
    parser.add_argument("--max-order", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=0.005)
    args = parser.parse_args()
    if args.max_order <= 0 or not math.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error("--max-order and --tolerance must be positive")
    try:
        payload = json.loads(args.checkpoint.read_text(encoding="utf-8"))
        cases = payload["cases"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(f"cannot read checkpoint {args.checkpoint}: {exc}")
    if not isinstance(cases, dict):
        parser.error("checkpoint 'cases' must be a JSON object")

    results: dict[tuple[int, float], dict] = {}
    for key, case in cases.items():
        try:
            order = int(case["order"])
            wavelength = float(case["wavelength_nm"])
            values = {name: float(case[name]) for name in METRICS}
        except (KeyError, TypeError, ValueError) as exc:
            parser.error(f"invalid saved case {key}: {exc}")
        if order > args.max_order:
            continue
        if not math.isfinite(wavelength) or not all(math.isfinite(v) for v in values.values()):
            parser.error(f"non-finite result in saved case {key}")
        identity = (order, wavelength)
        if identity in results:
            parser.error(f"duplicate result for M={order}, wavelength={wavelength:g} nm")
        results[identity] = {
            **case, **values, "order": order, "wavelength_nm": wavelength
        }
    if not results:
        parser.error(f"no completed cases through M={args.max_order}")

    wavelengths = sorted({wavelength for _, wavelength in results})
    orders = sorted({order for order, _ in results})
    settings_path = args.checkpoint.with_name("settings.json")
    if settings_path.is_file():
        try:
            settings = json.loads(settings_path.read_text(encoding="utf-8"))
            wavelengths = sorted(
                float(item["wavelength_nm"])
                for item in settings["materials_at_wavelengths"]
            )
            orders = sorted(
                int(order) for order in settings["numerics"]["orders"]
                if int(order) <= args.max_order
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.error(f"cannot read companion settings {settings_path}: {exc}")
    print(f"Checkpoint: {args.checkpoint}")
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
    candidate = next(
        (
            first[1] for first, second in zip(comparisons, comparisons[1:])
            if first[2] and second[2] and first[1] == second[0]
        ),
        None,
    )
    if candidate is None:
        print("Convergence: not confirmed by two consecutive passing steps.")
    else:
        print(f"Convergence candidate: M={candidate} (two consecutive steps passed).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
