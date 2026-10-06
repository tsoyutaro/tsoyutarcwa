"""Resume the specific pre-orthogonal-fix checkpoint after strict parity checks.

Only the known solver.py revision may differ. All geometry/material/core hashes
must match. Keep an exact backup and provenance for the original saved values.
No general signature bypass is provided.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
from common import (METRICS, POLS, case_key, digest, healthy, read_json,
                    signature_inputs, write_json)

OLD_SOLVER_SHA256 = "bc357d61745e4e6b1092444b311d24b82895da8d22bf3f848cc50569c9a6d38e"
# Same known revision with Windows CRLF line endings; no code differences.
OLD_SOLVER_HASHES = {OLD_SOLVER_SHA256,
                     "ca88d5d5958a84296fdd48f889ac6983570b2816ddb3eb56efa8d2fd39a83eb1"}
SOLVER_PATH = "studies/gold_grating_1d/solver.py"
PARITY_TOLERANCE = 1e-8


def resume_orthogonal_checkpoint(config, model, output, device, *, calculate=True):
    """Return a migration record, or None when no migration is needed."""
    output = Path(output).resolve()
    path = output/"checkpoint.json"
    if not path.exists():
        return None
    original_bytes = path.read_bytes()
    checkpoint = read_json(path)
    inputs = signature_inputs(config)
    new_signature = digest(inputs)
    if checkpoint.get("signature") == new_signature:
        return checkpoint.get("orthogonal_checkpoint_resume")
    old_inputs = checkpoint.get("inputs", {})
    old_solver_hash = old_inputs.get("source_sha256", {}).get(SOLVER_PATH)
    permitted_old_inputs = copy.deepcopy(inputs)
    permitted_old_inputs["source_sha256"][SOLVER_PATH] = old_solver_hash
    if (old_solver_hash not in OLD_SOLVER_HASHES or old_inputs != permitted_old_inputs or
            checkpoint.get("signature") != digest(old_inputs)):
        raise ValueError("Saved inputs differ beyond the known orthogonal-cell fix. Use a new --output-dir.")
    if not calculate:
        raise ValueError("Saved results predate the orthogonal-cell fix. First rerun without --prepare-only/--report-only to verify parity and resume.")
    cases = checkpoint.get("cases", {})
    for key, row in cases.items():
        if (case_key(row, row["wavelength_nm"]) != key or
                row.get("coefficient_method") != "analytic" or
                row.get("grid_used") is not None or
                not healthy(row, config["passivity_tolerance"])):
            raise ValueError("Saved analytic case metadata/values are invalid; use a new --output-dir.")
    # Up to three largest saved orders; choose the largest Nz and wavelength
    # at each order. These cover the most amplified roundoff in the saved sweep.
    orders = sorted({row["order"] for row in cases.values()})[-3:]
    references = [max((item for item in cases.items() if item[1]["order"] == order),
                      key=lambda item: (item[1]["slices"], item[1]["wavelength_nm"]))
                  for order in orders]
    parity = []
    if references:
        from solver import simulate
        for key, old in references:
            numbers = {axis: old[axis] for axis in ("order", "slices", "grid")}
            wave = old["wavelength_nm"]
            print(f"verify orthogonal-fix parity: M={numbers['order']}, Nz={numbers['slices']}, wavelength={wave:g} nm", flush=True)
            new = simulate(config, numbers, wave, model(wave), device)
            errors = {pol: {metric: abs(new["polarizations"][pol][metric]-old["polarizations"][pol][metric])
                            for metric in METRICS} for pol in POLS}
            maximum = max(error for metrics in errors.values() for error in metrics.values())
            if (not math.isfinite(maximum) or maximum > PARITY_TOLERANCE or
                    not healthy(new, config["passivity_tolerance"])):
                raise ValueError(f"Orthogonal-fix parity failed for {key}: {maximum:g}. Original checkpoint preserved; use a new --output-dir.")
            parity.append({"key": key, "absolute_errors": errors,
                           "max_absolute_error": maximum,
                           "recomputed_runtime_seconds": new["runtime_seconds"],
                           "environment": new.get("environment", {}), "passed": True})
            print(f"PASS orthogonal-fix parity: max absolute difference={maximum:.3g}", flush=True)
    backup = output/f"checkpoint_before_orthogonal_fix_{checkpoint['signature'][:12]}.json"
    if backup.exists():
        if backup.read_bytes() != original_bytes:
            raise ValueError("An orthogonal-fix backup already exists with different contents; original checkpoint preserved.")
    else:
        backup.write_bytes(original_bytes)
    record = {"policy": "exact-orthogonal-cell-v1", "original_signature": checkpoint["signature"],
              "old_solver_sha256": old_solver_hash,
              "new_solver_sha256": inputs["source_sha256"][SOLVER_PATH],
              "reused_cases": len(cases), "backup": str(backup),
              "parity_absolute_tolerance": PARITY_TOLERANCE,
              "selected_reference_checks": parity,
              "scope": "Known orthogonal roundoff fix, unchanged other inputs, selected-case R/P_sub/A_relief parity; not a convergence claim."}
    for row in cases.values():
        row["orthogonal_fix_reused_from"] = {"signature": checkpoint["signature"],
                                            "solver_sha256": old_solver_hash}
    checkpoint["signature"], checkpoint["inputs"] = new_signature, inputs
    checkpoint["orthogonal_checkpoint_resume"] = record
    write_json(path, checkpoint)
    print(f"Reusing {len(cases)} saved cases after orthogonal-fix parity; backup: {backup}", flush=True)
    return record
