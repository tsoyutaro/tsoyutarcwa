"""Diagnose Peng Fig. 2 circle-map conditioning without an RCWA eigensolve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

if __package__:
    from .common import PaperGeometry, select_device
else:
    from common import PaperGeometry, select_device

from rcwa_solver_auto import ASROptions, AutoRCWA, Lattice, OutputSpec


_PACKAGE_ROOT = Path(__file__).resolve().parent


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mapping", choices=("both", "outer", "double"), default="both"
    )
    parser.add_argument("--grid", type=int, default=256)
    parser.add_argument("--asr-g", type=float, default=3.0e-2)
    parser.add_argument("--safe-min-jacobian", type=float, default=1.0e-8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=_PACKAGE_ROOT / "results" / "mapping_diagnostic.json",
    )
    return parser


def _mapping_metrics(mapping, safe_floor: float) -> dict[str, object]:
    jacobian = torch.stack(
        (
            torch.stack((mapping.x_u, mapping.x_v), dim=-1),
            torch.stack((mapping.y_u, mapping.y_v), dim=-1),
        ),
        dim=-2,
    )
    singular_values = torch.linalg.svdvals(jacobian)
    condition = singular_values[..., 0] / singular_values[..., 1]
    minimum = float(torch.min(mapping.det_j).detach().cpu())
    return {
        "minimum_jacobian": minimum,
        "maximum_jacobian": float(torch.max(mapping.det_j).detach().cpu()),
        "minimum_singular_value": float(
            torch.min(singular_values[..., 1]).detach().cpu()
        ),
        "maximum_pointwise_jacobian_condition": float(
            torch.max(condition).detach().cpu()
        ),
        "safe_minimum_jacobian": safe_floor,
        "usable": minimum > safe_floor,
        "effective_radial_slope": (
            None
            if mapping.effective_radial_slope is None
            else float(mapping.effective_radial_slope.detach().cpu())
        ),
    }


def main() -> int:
    args = _parser().parse_args()
    if args.grid < 32:
        raise ValueError("--grid must be at least 32.")
    if not 0.0 < args.asr_g < 1.0:
        raise ValueError("--asr-g must lie in (0,1).")
    if args.safe_min_jacobian <= 0.0:
        raise ValueError("--safe-min-jacobian must be positive.")

    geometry = PaperGeometry()
    device = select_device(args.device)
    simulation = AutoRCWA(
        freq=0.2,
        order=[1, 1],
        lattice=Lattice.square(1.0),
        outputs=OutputSpec(smatrix_size="quarter", fields="none"),
        # Use the orientation floor while measuring.  The stricter requested
        # safety floor is assessed in the report instead of hiding the value
        # behind an exception.
        asr=ASROptions(
            circle_G=args.asr_g,
            minimum_circle_jacobian=1.0e-12,
            grid=(args.grid, args.grid),
        ),
        verify_cascade=False,
        dtype=torch.complex128,
        device=device,
    )
    inner = geometry.inner_radius_um / geometry.period_um
    outer = geometry.outer_radius_um / geometry.period_um
    requested = ("outer", "double") if args.mapping == "both" else (args.mapping,)
    mappings: dict[str, object] = {}
    for name in requested:
        try:
            mapping = (
                simulation.build_circle_asr_mapping(args.grid, args.grid, outer)
                if name == "outer"
                else simulation.build_double_matched_circle_asr_mapping(
                    args.grid, args.grid, inner, outer
                )
            )
            mappings[name] = _mapping_metrics(mapping, args.safe_min_jacobian)
        except RuntimeError as error:
            mappings[name] = {"usable": False, "error": str(error)}

    report = {
        "geometry": {
            "period_um": geometry.period_um,
            "inner_radius_um": geometry.inner_radius_um,
            "outer_radius_um": geometry.outer_radius_um,
        },
        "grid": args.grid,
        "asr_g": args.asr_g,
        "device": str(device),
        "mappings": mappings,
        "recommendation": (
            "For matched-coordinate runs, use radial_mapping='double'."
            if bool(mappings.get("double", {}).get("usable", False))
            else "No tested map passed the requested safety floor."
        ),
        "scope": (
            "This diagnostic checks mapping conditioning only; usable=true "
            "does not establish RCWA passivity or Fourier-order convergence."
        ),
        "next_step": (
            "Run reproduce_square --study convergence and require passive, "
            "stable tail values before generating a spectrum."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.mapping == "both":
        return 0 if bool(mappings.get("double", {}).get("usable", False)) else 2
    return 0 if bool(mappings[args.mapping].get("usable", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
