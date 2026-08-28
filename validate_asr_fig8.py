"""Validate the Wang et al. Fig. 8 reproduction setup and implementation.

The default checks are dependency-light and do not run a large RCWA problem.
``--integration`` additionally runs order-one ASR/ASR-FR smoke spectra and
compares Redheffer with Li algorithm 2a when torch and torcwa are available.
"""

from __future__ import annotations

import argparse
import ast
import csv
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path


C_MM_PER_NS = 299.792458
PERIOD_MM = 30.0


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


def record(checks: list[Check], name: str, passed: bool, detail: str) -> None:
    checks.append(Check(name, bool(passed), detail))


def _assignment_literals(tree: ast.Module) -> dict[str, object]:
    values: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            value = node.value
        else:
            continue
        if not isinstance(target, ast.Name) or value is None:
            continue
        try:
            values[target.id] = ast.literal_eval(value)
        except (ValueError, TypeError):
            pass
    return values


def _static_checks(root: Path) -> list[Check]:
    checks: list[Check] = []
    reproduction = root / "reproduce_asr_fig8.py"
    asr_source_path = root / "rcwa_ext" / "asr.py"
    maps_source_path = root / "rcwa_ext" / "asr_maps.py"
    scattering_source_path = root / "rcwa_ext" / "scattering.py"
    source = reproduction.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(reproduction))
    constants = _assignment_literals(tree)

    expected = {
        "PERIOD_MM": 30.0,
        "PATCH_FILL_X": 0.5,
        "PATCH_FILL_Y": 0.5,
        "PATCH_THICKNESS_MM": 0.01,
        "ASR_G": 1.0e-3,
        "FIG8_FREQUENCY_MIN_GHZ": 2.0,
        "FIG8_FREQUENCY_MAX_GHZ": 18.0,
        "FIG8_ASR_FR_ORDER": 8,
        "FIG8_ASR_ORDER": 20,
    }
    mismatches = {
        key: (constants.get(key), value)
        for key, value in expected.items()
        if constants.get(key) != value
    }
    record(
        checks,
        "paper Fig. 8 constants",
        not mismatches,
        "exact match" if not mismatches else repr(mismatches),
    )
    record(
        checks,
        "passive metal sign in exp(-iwt) convention",
        "METAL_EPS_TORCWA = 1.0 + 1.0e6j" in source,
        "epsilon=1+1e6j",
    )
    record(
        checks,
        "public modular import",
        "from rcwa_ext import CustomRCWA_ASR_FR" in source
        and "rcwa_solver_2" not in source,
        "no dependency on absent rcwa_solver_2.py",
    )
    record(
        checks,
        "half S matrix for forward x source",
        'smatrix_size="half"' in source
        and 'polarization="pp"' in source
        and 'polarization="sp"' in source,
        "Tf/Rf only; co- and cross-polarized output powers are summed",
    )
    record(
        checks,
        "Figure 8(b) observables",
        '"delta_R"' in source and '"delta_T"' in source,
        "delta_R=R_total-R00 and delta_T=T_total-T00",
    )
    record(
        checks,
        "HFSS provenance guard",
        "No HFSS values are synthesized" in source and "--hfss-csv" in source,
        "external samples are optional and never fabricated",
    )

    cutoff = C_MM_PER_NS / PERIOD_MM
    record(
        checks,
        "first Rayleigh cutoff",
        math.isclose(cutoff, 9.993081933333333, rel_tol=0.0, abs_tol=1e-12),
        f"fc=c/Lambda={cutoff:.12f} GHz",
    )

    # Eq. (2): transformed interval widths are proportional to cube roots.
    dx = (7.5, 15.0, 7.5)
    weights = tuple(value ** (1.0 / 3.0) for value in dx)
    du = tuple(PERIOD_MM * value / sum(weights) for value in weights)
    record(
        checks,
        "ASR transformed intervals",
        math.isclose(sum(du), PERIOD_MM, abs_tol=1e-13)
        and math.isclose(du[0], du[2], abs_tol=1e-13)
        and du[1] > du[0],
        f"Delta-u={du}",
    )

    # Eq. (1): a3=G*Delta-u-Delta-x gives dx/du=G at an interface.
    minimum_slopes = []
    mid_slopes = []
    for physical, transformed in zip(dx, du):
        a2 = physical / transformed
        a3 = 1.0e-3 * transformed - physical
        minimum_slopes.append(a2 + a3 / transformed)
        mid_slopes.append(a2 - a3 / transformed)
    record(
        checks,
        "ASR mapping monotonicity and matched slope",
        max(abs(value - 1.0e-3) for value in minimum_slopes) < 2e-16
        and min(mid_slopes) > 0.0,
        f"interface slopes={minimum_slopes}, opposite-side slopes={mid_slopes}",
    )

    asr_source = asr_source_path.read_text(encoding="utf-8")
    maps_source = maps_source_path.read_text(encoding="utf-8")
    scattering_source = scattering_source_path.read_text(encoding="utf-8")
    factorization_tokens = (
        "1.0 / eps11",
        "invert_u_toeplitz=True",
        "eps22",
        "invert_final_bttb=True",
    )
    record(
        checks,
        "mixed inverse/direct Fourier factorization",
        all(token in asr_source for token in factorization_tokens),
        "epsilon_11 uses inverse rule; epsilon_22 uses the complementary rule",
    )
    mapping_tokens = (
        "torch.pow(dx, 1.0 / 3.0)",
        "a3 = selected_slope * local_du - local_dx",
        "torch.sin(phase)",
        "torch.cos(phase)",
    )
    record(
        checks,
        "paper Eq. (1)-(2) mapping implementation",
        all(token in maps_source for token in mapping_tokens),
        "cube-root allocation and sinusoidal matched-coordinate map found",
    )
    conversion_tokens = (
        "_build_conversion_matrix_T(mapping)",
        "w_cartesian = torch.matmul(transform, w_uv)",
        "v_cartesian = torch.matmul(transform, v_uv)",
    )
    record(
        checks,
        "layer-to-Cartesian modal conversion T",
        all(token in asr_source for token in conversion_tokens),
        "ASR modal bases are converted before S-matrix assembly",
    )
    stable_mode_tokens = (
        "V=Q W Γ^-1",
        "magnetic_from_q = torch.matmul(",
        "electric_modes * (1.0 / safe_kz)[None, :]",
    )
    record(
        checks,
        "stable magnetic modal relation",
        all(token in scattering_source for token in stable_mode_tokens),
        "uses paper V=QW/Gamma instead of an ill-conditioned P inverse",
    )
    separable_tokens = (
        "_rect_separable_convolutions",
        "background * separable(1.0 / f, g)",
        "contrast * separable(inside_x / f, inside_y * g)",
        "asr_quadrature_grid",
    )
    record(
        checks,
        "high-resolution separable ASR quadrature",
        all(token in asr_source for token in separable_tokens),
        "N=20 direct-rule tensors avoid a quadrature_grid^2 raster",
    )
    record(
        checks,
        "non-passive output guard",
        "Non-passive numerical result" in source
        and "passivity_tolerance" in source,
        "R+T>1+tolerance is rejected before CSV/plot output",
    )
    return checks


def _read_numeric_csv(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    numeric = (
        "frequency_GHz",
        "R00",
        "T00",
        "R_total",
        "T_total",
        "delta_R",
        "delta_T",
        "A_total",
    )
    for row in rows:
        for key in numeric:
            row[key] = float(row[key])
    return rows


def _run_smoke(root: Path, output: Path, cascade: str, methods: str) -> list[dict[str, object]]:
    command = [
        sys.executable,
        str(root / "reproduce_asr_fig8.py"),
        "--study",
        "smoke",
        "--methods",
        methods,
        "--device",
        "cpu",
        "--dtype",
        "complex128",
        "--grid",
        "32",
        "--quadrature-grid",
        "512",
        "--cascade",
        cascade,
        "--output-dir",
        str(output),
        "--no-plot",
    ]
    completed = subprocess.run(
        command,
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if completed.returncode:
        raise RuntimeError(
            f"smoke command failed ({completed.returncode})\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return _read_numeric_csv(output / "smoke_powers.csv")


def _integration_checks(root: Path) -> list[Check]:
    checks: list[Check] = []
    missing = [
        module
        for module in ("torch", "torcwa")
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        record(
            checks,
            "RCWA integration availability",
            False,
            "unavailable modules: " + ", ".join(missing),
        )
        return checks

    with tempfile.TemporaryDirectory(prefix="asr_fig8_validation_") as temp:
        temporary = Path(temp)
        redheffer = _run_smoke(root, temporary / "redheffer", "redheffer", "both")
        algo2a = _run_smoke(root, temporary / "algo2a", "algo2a", "asr-fr")

    record(
        checks,
        "smoke row count",
        len(redheffer) == 4 and len(algo2a) == 2,
        f"Redheffer={len(redheffer)}, Li-2a={len(algo2a)}",
    )
    finite_and_passive = True
    identities = True
    below_cutoff = True
    higher_order_nonnegative = True
    for row in redheffer:
        values = [
            float(row[key])
            for key in ("R00", "T00", "R_total", "T_total", "A_total")
        ]
        finite_and_passive &= all(math.isfinite(value) for value in values)
        finite_and_passive &= float(row["A_total"]) >= -1e-6
        identities &= math.isclose(
            float(row["delta_R"]),
            float(row["R_total"]) - float(row["R00"]),
            abs_tol=2e-10,
        )
        identities &= math.isclose(
            float(row["delta_T"]),
            float(row["T_total"]) - float(row["T00"]),
            abs_tol=2e-10,
        )
        higher_order_nonnegative &= float(row["delta_R"]) >= -2e-10
        higher_order_nonnegative &= float(row["delta_T"]) >= -2e-10
        if math.isclose(float(row["frequency_GHz"]), 6.0):
            below_cutoff &= abs(float(row["delta_R"])) < 2e-10
            below_cutoff &= abs(float(row["delta_T"])) < 2e-10
    record(checks, "finite passive smoke powers", finite_and_passive, "A>=-1e-6")
    record(checks, "power difference identities", identities, "total-zero identities")
    record(
        checks,
        "only zeroth order below cutoff",
        below_cutoff,
        "at 6 GHz, total and zero-order powers coincide",
    )
    record(
        checks,
        "higher-order powers are nonnegative",
        higher_order_nonnegative,
        "delta_R,delta_T >= numerical tolerance",
    )

    redheffer_asr_fr = {
        float(row["frequency_GHz"]): row
        for row in redheffer
        if row["method"] == "ASR-FR"
    }
    maximum = 0.0
    for row in algo2a:
        reference = redheffer_asr_fr[float(row["frequency_GHz"])]
        maximum = max(
            maximum,
            *(abs(float(row[key]) - float(reference[key])) for key in (
                "R00", "T00", "R_total", "T_total"
            )),
        )
    record(
        checks,
        "Redheffer/Li-2a ASR-FR parity",
        maximum < 2e-9,
        f"maximum power difference={maximum:.3e}",
    )
    return checks


def _paper_stress_checks(root: Path) -> list[Check]:
    checks: list[Check] = []
    missing = [
        module
        for module in ("torch", "torcwa")
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        record(
            checks,
            "paper-order stress availability",
            False,
            "unavailable modules: " + ", ".join(missing),
        )
        return checks
    with tempfile.TemporaryDirectory(prefix="asr_fig8_paper_stress_") as temp:
        output = Path(temp) / "stress"
        command = [
            sys.executable,
            str(root / "reproduce_asr_fig8.py"),
            "--study",
            "fig8",
            "--methods",
            "asr",
            "--frequencies",
            "8,11,14",
            "--asr-order",
            "20",
            "--grid",
            "256",
            "--quadrature-grid",
            "4096",
            "--device",
            "cuda",
            "--dtype",
            "complex128",
            "--output-dir",
            str(output),
            "--no-plot",
        ]
        completed = subprocess.run(
            command,
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if completed.returncode:
            record(
                checks,
                "paper-order ASR N=20 execution",
                False,
                completed.stderr[-1000:] or completed.stdout[-1000:],
            )
            return checks
        rows = _read_numeric_csv(output / "fig8_powers.csv")
    maximum_balance = max(float(row["R_total"]) + float(row["T_total"]) for row in rows)
    minimum_absorption = min(float(row["A_total"]) for row in rows)
    finite = all(
        math.isfinite(float(row[key]))
        for row in rows
        for key in ("R00", "T00", "R_total", "T_total", "A_total")
    )
    record(
        checks,
        "paper-order ASR N=20 passivity stress",
        len(rows) == 3 and finite and minimum_absorption >= -5e-5,
        f"rows={len(rows)}, max(R+T)={maximum_balance:.8g}, min(A)={minimum_absorption:.3e}",
    )
    return checks


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integration", action="store_true")
    parser.add_argument(
        "--paper-stress",
        action="store_true",
        help="Also run ASR N=M=20 at 8, 11, and 14 GHz on CUDA.",
    )
    parser.add_argument("--json", type=Path)
    return parser


def main(args: argparse.Namespace) -> dict[str, object]:
    root = Path(__file__).resolve().parent
    checks = _static_checks(root)
    if args.integration:
        checks.extend(_integration_checks(root))
    if args.paper_stress:
        checks.extend(_paper_stress_checks(root))
    failed = [check for check in checks if not check.passed]
    report = {
        "passed": not failed,
        "integration_requested": args.integration,
        "paper_stress_requested": args.paper_stress,
        "paper_doi": "10.1364/OE.459110",
        "checks": [asdict(check) for check in checks],
    }
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"[{status}] {check.name}: {check.detail}")
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    if failed:
        raise SystemExit(f"{len(failed)} validation check(s) failed")
    print(f"All {len(checks)} checks passed.")
    return report


if __name__ == "__main__":
    main(_parser().parse_args())
