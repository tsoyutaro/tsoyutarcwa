"""Audit previously saved 100-slice Au/PMMA R/T/A cases without PyTorch."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
STUDY_RESULTS = HERE.parents[1] / "studies" / "pmma_gold_motheye" / "results"
DEFAULT_CHECKPOINTS = (
    STUDY_RESULTS / "measured_30nm_Nz100_order_4_6_8" / "checkpoint.json",
    STUDY_RESULTS / "measured_30nm_Nz100_order" / "checkpoint.json",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path, nargs="*",
                        default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--output", type=Path,
                        default=HERE / "results" / "saved_Nz100_audit.json")
    args = parser.parse_args()
    checks = []
    for checkpoint_path in args.checkpoints:
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        for case in payload["cases"].values():
            r, t, a = (float(case[name]) for name in
                       ("reflectance", "transmittance", "absorptance"))
            finite = all(math.isfinite(value) for value in (r, t, a))
            residual = abs(r + t + a - 1.0) if finite else None
            passed = (finite and min(r, t, a) >= -1e-6 and
                      max(r, t, a) <= 1 + 1e-6 and residual <= 1e-10 and
                      not case.get("passivity_warning", False))
            checks.append({"order": case["order"], "wavelength_nm": case["wavelength_nm"],
                           "reflectance": r, "transmittance": t, "absorptance": a,
                           "energy_sum_residual": residual, "passed": passed,
                           "source": str(checkpoint_path.resolve())})
    if not checks:
        raise ValueError("No cases found in the supplied checkpoints.")
    report = {"passed": all(case["passed"] for case in checks),
              "case_count": len(checks),
              "orders": sorted({case["order"] for case in checks}),
              "wavelengths_nm": sorted({case["wavelength_nm"] for case in checks}),
              "largest_energy_sum_residual": max(case["energy_sum_residual"]
                                                  for case in checks if case["energy_sum_residual"] is not None),
              "scope": "Stored-result passivity audit only; A was recorded as 1-R-T, so the energy sum is not an independent conservation test.",
              "cases": checks}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                     allow_nan=False) + "\n", encoding="utf-8")
    print(f"{sum(case['passed'] for case in checks)}/{len(checks)} stored cases pass; "
          f"report: {args.output.resolve()}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
