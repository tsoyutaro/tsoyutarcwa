"""Numerical validation for rcwa_solver_auto.py.

Run this file in an environment containing PyTorch and torcwa 0.1.4.2::

    python validation/validate_rcwa_solver_auto.py

The default order/grid are deliberately small.  Increase ``--order`` and
``--grid`` for a convergence study after the implementation checks pass.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
import torch

_OUTPUTS_ROOT = Path(__file__).resolve().parent.parent
if str(_OUTPUTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_OUTPUTS_ROOT))

from rcwa_solver_auto import (
    ASROptions,
    AutoRCWA,
    Circle,
    GroupTheoryOptions,
    Lattice,
    LayerSpec,
    Material,
    NVMOptions,
    OutputSpec,
    Rectangle,
)


@dataclass
class Check:
    name: str
    error: float
    limit: float
    detail: str = ""

    @property
    def passed(self) -> bool:
        return math.isfinite(self.error) and self.error <= self.limit


def _maximum_error(first: torch.Tensor, second: torch.Tensor) -> float:
    return float(torch.max(torch.abs(first - second)).detach().cpu())


def _all_finite(blocks: list[torch.Tensor]) -> bool:
    return all(bool(torch.all(torch.isfinite(block)).detach().cpu()) for block in blocks)


def _zero_order_source(simulation: AutoRCWA, polarization: str) -> torch.Tensor:
    x_index = int(torch.nonzero(simulation.order_x == 0, as_tuple=False)[0, 0])
    y_index = int(torch.nonzero(simulation.order_y == 0, as_tuple=False)[0, 0])
    harmonic = x_index * len(simulation.order_y) + y_index
    source = torch.zeros(
        2 * simulation.order_N,
        dtype=simulation._dtype,
        device=simulation._device,
    )
    source[harmonic if polarization == "x" else simulation.order_N + harmonic] = 1.0
    return source


def _make_circle_simulation(
    *,
    lattice: Lattice,
    geometry: Circle,
    algorithm: str,
    size: str,
    use_group_theory: bool,
    polarization: str | None = None,
    order: int,
    grid: int,
    device: torch.device,
) -> AutoRCWA:
    simulation = AutoRCWA(
        freq=1.0 / 1.55,
        order=[order, order],
        lattice=lattice,
        cascade=algorithm,
        outputs=OutputSpec(smatrix_size=size, fields="none"),
        nvm=NVMOptions(grid=(grid, grid)),
        group_theory=GroupTheoryOptions(
            enabled=use_group_theory or polarization is not None,
            symmetry="auto",
            strict=use_group_theory or polarization is not None,
            residual_tolerance=2.0e-8,
            polarization=polarization,
        ),
        verify_cascade=False,
        dtype=torch.complex128,
        device=device,
    )
    simulation.add_input_layer(eps=1.0)
    simulation.add_output_layer(eps=2.25)
    simulation.set_incident_angle(
        inc_ang=torch.tensor(0.0, dtype=torch.float64, device=device),
        azi_ang=torch.tensor(0.0, dtype=torch.float64, device=device),
    )
    selected = simulation.add_structured_layer(
        LayerSpec(
            thickness=0.22,
            geometry=geometry,
            background=Material(1.0),
            inclusion=Material(4.0),
            method="nvm",
            label="circle-validation",
        )
    )
    if selected != "nvm":
        raise AssertionError(f"circle layer selected {selected!r}, expected 'nvm'")
    simulation.solve_global_smatrix()
    if not _all_finite(simulation.S):
        raise AssertionError("circle simulation produced NaN or infinity")
    return simulation


def _make_asr_smoke_simulation(
    *, order: int, grid: int, device: torch.device
) -> AutoRCWA:
    simulation = AutoRCWA(
        freq=1.0 / 1.55,
        order=[order, order],
        lattice=Lattice.rectangular(1.0, 1.2),
        cascade="redheffer",
        outputs=OutputSpec(smatrix_size="full", fields="none"),
        asr=ASROptions(grid=(grid, grid)),
        dtype=torch.complex128,
        device=device,
    )
    simulation.set_incident_angle(
        inc_ang=torch.tensor(0.0, dtype=torch.float64, device=device),
        azi_ang=torch.tensor(0.0, dtype=torch.float64, device=device),
    )
    selected = simulation.add_structured_layer(
        LayerSpec(
            thickness=0.18,
            geometry=Rectangle((0.45, 0.55)),
            background=Material(1.0),
            inclusion=Material(3.2),
            method="asr-fr",
            label="asr-regression",
        )
    )
    if selected != "asr-fr":
        raise AssertionError(f"rectangle selected {selected!r}, expected 'asr-fr'")
    simulation.solve_global_smatrix()
    if not _all_finite(simulation.S):
        raise AssertionError("ASR smoke simulation produced NaN or infinity")
    return simulation


def _record_comparison(
    checks: list[Check],
    name: str,
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> None:
    scale = max(
        1.0,
        float(torch.max(torch.abs(first)).detach().cpu()),
        float(torch.max(torch.abs(second)).detach().cpu()),
    )
    checks.append(Check(name, _maximum_error(first, second), atol + rtol * scale))


def run_validation(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    period = 1.0
    cases: dict[str, tuple[Lattice, Circle, str]] = {
        "rectangular-circle": (
            Lattice.rectangular(period, 1.15 * period),
            Circle(0.28 * period),
            "c2v",
        ),
        "hexagonal-close-packed": (
            Lattice.hexagonal_close_packed(period),
            Circle.close_packed(period, gap=0.08 * period),
            "c2",
        ),
        "general-oblique": (
            Lattice.oblique((period, 0.0), (0.37 * period, 1.07 * period)),
            Circle(0.24 * period),
            "c2",
        ),
    }
    checks: list[Check] = []
    diagnostics: dict[str, object] = {}

    for case_name, (lattice, circle, expected_symmetry) in cases.items():
        full: dict[str, AutoRCWA] = {}
        for algorithm in ("redheffer", "algo2a"):
            full[algorithm] = _make_circle_simulation(
                lattice=lattice,
                geometry=circle,
                algorithm=algorithm,
                size="full",
                use_group_theory=False,
                order=args.order,
                grid=args.grid,
                device=device,
            )

        for index, block_name in enumerate(("Tf", "Rf", "Rb", "Tb")):
            _record_comparison(
                checks,
                f"{case_name}: algo2a/full vs redheffer/full {block_name}",
                full["algo2a"].S[index],
                full["redheffer"].S[index],
                atol=args.atol,
                rtol=args.rtol,
            )

        for algorithm in ("redheffer", "algo2a"):
            half = _make_circle_simulation(
                lattice=lattice,
                geometry=circle,
                algorithm=algorithm,
                size="half",
                use_group_theory=False,
                order=args.order,
                grid=args.grid,
                device=device,
            )
            quarter = _make_circle_simulation(
                lattice=lattice,
                geometry=circle,
                algorithm=algorithm,
                size="quarter",
                use_group_theory=False,
                order=args.order,
                grid=args.grid,
                device=device,
            )
            _record_comparison(
                checks,
                f"{case_name}: {algorithm} half Tf",
                half.S[0],
                full[algorithm].S[0],
                atol=args.atol,
                rtol=args.rtol,
            )
            _record_comparison(
                checks,
                f"{case_name}: {algorithm} half Rf",
                half.S[1],
                full[algorithm].S[1],
                atol=args.atol,
                rtol=args.rtol,
            )
            _record_comparison(
                checks,
                f"{case_name}: {algorithm} quarter Rf",
                quarter.S[1],
                full[algorithm].S[1],
                atol=args.atol,
                rtol=args.rtol,
            )
            zero_error = max(
                float(torch.max(torch.abs(quarter.S[index])).detach().cpu())
                for index in (0, 2, 3)
            )
            checks.append(
                Check(
                    f"{case_name}: {algorithm} quarter unused blocks",
                    zero_error,
                    0.0,
                )
            )

        grouped = _make_circle_simulation(
            lattice=lattice,
            geometry=circle,
            algorithm="redheffer",
            size="full",
            use_group_theory=True,
            order=args.order,
            grid=args.grid,
            device=device,
        )
        group_diagnostic = grouped.group_theory_diagnostics[-1]
        diagnostics[case_name] = {
            key: (
                float(value.detach().cpu())
                if isinstance(value, torch.Tensor) and value.numel() == 1
                else value
            )
            for key, value in group_diagnostic.items()
        }
        if not bool(group_diagnostic.get("applied")):
            raise AssertionError(f"{case_name}: group theory was not applied")
        if group_diagnostic.get("symmetry") != expected_symmetry:
            raise AssertionError(
                f"{case_name}: expected {expected_symmetry}, got "
                f"{group_diagnostic.get('symmetry')}"
            )
        for index, block_name in enumerate(("Tf", "Rf", "Rb", "Tb")):
            _record_comparison(
                checks,
                f"{case_name}: grouped vs full eigensolve {block_name}",
                grouped.S[index],
                full["redheffer"].S[index],
                atol=args.atol,
                rtol=args.rtol,
            )

        if case_name == "rectangular-circle":
            for polarization in ("x", "y"):
                source = _zero_order_source(full["redheffer"], polarization)
                polarized: dict[str, AutoRCWA] = {}
                for algorithm in ("redheffer", "algo2a"):
                    polarized[algorithm] = _make_circle_simulation(
                        lattice=lattice,
                        geometry=circle,
                        algorithm=algorithm,
                        size="half",
                        use_group_theory=True,
                        polarization=polarization,
                        order=args.order,
                        grid=args.grid,
                        device=device,
                    )
                    for index, block_name in ((0, "Tf"), (1, "Rf")):
                        _record_comparison(
                            checks,
                            f"{case_name}: {algorithm} {polarization}-only {block_name}",
                            torch.matmul(polarized[algorithm].S[index], source),
                            torch.matmul(full[algorithm].S[index], source),
                            atol=args.atol,
                            rtol=args.rtol,
                        )
                    reduced_dimension = int(
                        polarized[algorithm].cascade_diagnostics[
                            "reduced_dimension"
                        ]
                    )
                    full_dimension = int(
                        polarized[algorithm].cascade_diagnostics[
                            "full_dimension"
                        ]
                    )
                    checks.append(
                        Check(
                            f"{case_name}: {algorithm} {polarization}-only dimension reduction",
                            0.0 if reduced_dimension < full_dimension else 1.0,
                            0.0,
                            f"{reduced_dimension}/{full_dimension}",
                        )
                    )
                for index, block_name in ((0, "Tf"), (1, "Rf")):
                    _record_comparison(
                        checks,
                        f"{case_name}: polarized algo2a vs redheffer {polarization} {block_name}",
                        torch.matmul(polarized["algo2a"].S[index], source),
                        torch.matmul(polarized["redheffer"].S[index], source),
                        atol=args.atol,
                        rtol=args.rtol,
                    )

            quarter = _make_circle_simulation(
                lattice=lattice,
                geometry=circle,
                algorithm="algo2a",
                size="quarter",
                use_group_theory=True,
                polarization="x",
                order=args.order,
                grid=args.grid,
                device=device,
            )
            source = _zero_order_source(full["algo2a"], "x")
            _record_comparison(
                checks,
                f"{case_name}: polarized algo2a x-only quarter Rf",
                torch.matmul(quarter.S[1], source),
                torch.matmul(full["algo2a"].S[1], source),
                atol=args.atol,
                rtol=args.rtol,
            )

        if case_name == "hexagonal-close-packed":
            triangular_polarized: dict[tuple[str, str], AutoRCWA] = {}
            for polarization in ("x", "y"):
                source = _zero_order_source(full["redheffer"], polarization)
                for algorithm in ("redheffer", "algo2a"):
                    triangular_polarized[(algorithm, polarization)] = _make_circle_simulation(
                        lattice=lattice,
                        geometry=circle,
                        algorithm=algorithm,
                        size="half",
                        use_group_theory=True,
                        polarization=polarization,
                        order=args.order,
                        grid=args.grid,
                        device=device,
                    )
                redheffer_sector = triangular_polarized[("redheffer", polarization)]
                algo2a_sector = triangular_polarized[("algo2a", polarization)]
                diagnostic = redheffer_sector.group_theory_diagnostics[-1]
                checks.append(
                    Check(
                        f"hexagonal NVM {polarization}-sector D6-star/Cs applied",
                        0.0
                        if diagnostic.get("symmetry") == "D6-star/Cs(x-mirror)"
                        and diagnostic.get("backend") == "NVM"
                        else 1.0,
                        0.0,
                    )
                )
                checks.append(
                    Check(
                        f"hexagonal NVM {polarization}-sector invariance",
                        float(diagnostic["max_invariance_residual"]),
                        2e-10,
                    )
                )
                reduced_dimension = int(
                    redheffer_sector.cascade_diagnostics["reduced_dimension"]
                )
                full_dimension = int(
                    redheffer_sector.cascade_diagnostics["full_dimension"]
                )
                checks.append(
                    Check(
                        f"hexagonal NVM {polarization}-sector dimension reduction",
                        0.0 if reduced_dimension < full_dimension else 1.0,
                        0.0,
                        f"{reduced_dimension}/{full_dimension}",
                    )
                )
                for index, block_name in ((0, "Tf"), (1, "Rf")):
                    _record_comparison(
                        checks,
                        f"hexagonal NVM {polarization}-sector algo2a/Redheffer {block_name}",
                        torch.matmul(algo2a_sector.S[index], source),
                        torch.matmul(redheffer_sector.S[index], source),
                        atol=args.atol,
                        rtol=args.rtol,
                    )

            quarter = _make_circle_simulation(
                lattice=lattice,
                geometry=circle,
                algorithm="algo2a",
                size="quarter",
                use_group_theory=True,
                polarization="x",
                order=args.order,
                grid=args.grid,
                device=device,
            )
            source = _zero_order_source(full["algo2a"], "x")
            _record_comparison(
                checks,
                "hexagonal NVM x-sector algo2a quarter/half Rf",
                torch.matmul(quarter.S[1], source),
                torch.matmul(
                    triangular_polarized[("algo2a", "x")].S[1], source
                ),
                atol=args.atol,
                rtol=args.rtol,
            )

        if case_name == "general-oblique":
            oblique_polarized: dict[tuple[str, str], AutoRCWA] = {}
            for polarization in ("x", "y"):
                source = _zero_order_source(full["redheffer"], polarization)
                for algorithm in ("redheffer", "algo2a"):
                    sector = _make_circle_simulation(
                        lattice=lattice,
                        geometry=circle,
                        algorithm=algorithm,
                        size="half",
                        use_group_theory=True,
                        polarization=polarization,
                        order=args.order,
                        grid=args.grid,
                        device=device,
                    )
                    oblique_polarized[(algorithm, polarization)] = sector
                    diagnostic = sector.group_theory_diagnostics[-1]
                    checks.append(
                        Check(
                            f"oblique NVM {polarization}-source common C2 sector",
                            0.0
                            if diagnostic.get("symmetry") == "c2-source-sector"
                            else 1.0,
                            0.0,
                        )
                    )
                    for index, block_name in ((0, "Tf"), (1, "Rf")):
                        _record_comparison(
                            checks,
                            f"oblique NVM {algorithm} {polarization}-source/full {block_name}",
                            torch.matmul(sector.S[index], source),
                            torch.matmul(full[algorithm].S[index], source),
                            atol=args.atol,
                            rtol=args.rtol,
                        )
                    checks.append(
                        Check(
                            f"oblique NVM {polarization}-source dimension reduction",
                            0.0
                            if int(sector.cascade_diagnostics["reduced_dimension"])
                            < int(sector.cascade_diagnostics["full_dimension"])
                            else 1.0,
                            0.0,
                        )
                    )
                for index, block_name in ((0, "Tf"), (1, "Rf")):
                    _record_comparison(
                        checks,
                        f"oblique NVM {polarization}-source algo2a/Redheffer {block_name}",
                        torch.matmul(
                            oblique_polarized[("algo2a", polarization)].S[index],
                            source,
                        ),
                        torch.matmul(
                            oblique_polarized[("redheffer", polarization)].S[index],
                            source,
                        ),
                        atol=args.atol,
                        rtol=args.rtol,
                    )

            oblique_x_basis = oblique_polarized[("redheffer", "x")]._polarization_bases[0]
            oblique_y_basis = oblique_polarized[("redheffer", "y")]._polarization_bases[0]
            checks.append(
                Check(
                    "oblique NVM x/y requests share the C2 source sector",
                    _maximum_error(
                        oblique_x_basis @ oblique_x_basis.mH,
                        oblique_y_basis @ oblique_y_basis.mH,
                    ),
                    2e-10,
                )
            )

            quarter = _make_circle_simulation(
                lattice=lattice,
                geometry=circle,
                algorithm="algo2a",
                size="quarter",
                use_group_theory=True,
                polarization="x",
                order=args.order,
                grid=args.grid,
                device=device,
            )
            source = _zero_order_source(full["algo2a"], "x")
            _record_comparison(
                checks,
                "oblique NVM x-source algo2a quarter/half Rf",
                torch.matmul(quarter.S[1], source),
                torch.matmul(
                    oblique_polarized[("algo2a", "x")].S[1], source
                ),
                atol=args.atol,
                rtol=args.rtol,
            )

    _make_asr_smoke_simulation(order=args.order, grid=args.grid, device=device)
    checks.append(Check("rectangular ASR-FR regression smoke test", 0.0, 0.0))

    failed = [check for check in checks if not check.passed]
    report = {
        "passed": not failed,
        "device": str(device),
        "dtype": "complex128",
        "order": args.order,
        "grid": args.grid,
        "checks": [
            {
                "name": check.name,
                "passed": check.passed,
                "error": check.error,
                "limit": check.limit,
                "detail": check.detail,
            }
            for check in checks
        ],
        "group_theory": diagnostics,
    }
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        print(
            f"[{status}] {check.name}: error={check.error:.3e}, "
            f"limit={check.limit:.3e}"
        )
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {args.json}")
    if failed:
        raise SystemExit(f"{len(failed)} validation check(s) failed")
    print(f"All {len(checks)} validation checks passed.")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", type=int, default=1)
    parser.add_argument("--grid", type=int, default=48)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--atol", type=float, default=2.0e-8)
    parser.add_argument("--rtol", type=float, default=2.0e-7)
    parser.add_argument("--json", type=Path)
    return parser


if __name__ == "__main__":
    run_validation(_parser().parse_args())
