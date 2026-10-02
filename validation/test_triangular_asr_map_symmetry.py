"""Regression checks for triangular matched-ASR sector derivatives.

Run from the project root:
    python3 validation/test_triangular_asr_map_symmetry.py --device cuda
No saved optimization files or optical eigensolves are needed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rcwa_solver_auto import ASROptions, AutoRCWA, Lattice, OutputSpec
from studies.gold_motheye2.diagnose_asr_grid import analyze, numpy_reference, production_map


def radius_objectives(simulation, radius, n):
    mapping = simulation.build_triangular_circle_asr_mapping(n, n, radius)
    u, v = torch.meshgrid(mapping.u, mapping.v, indexing="ij")
    displacement = torch.stack((mapping.x - u - .5*v,
                                mapping.y - np.sqrt(3.)/2*v))
    primitive = torch.stack((torch.stack((mapping.x_u, mapping.x_v)),
                             torch.stack((mapping.y_u, mapping.y_v))))
    inverse_cell = primitive.new_tensor([[1., -1./np.sqrt(3.)],
                                         [0., 2./np.sqrt(3.)]])
    jacobian = torch.einsum("abij,bc->acij", primitive, inverse_cell)
    return displacement.square().mean(), jacobian.square().mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.device == "cpu":
        torch.set_num_threads(2)
    simulation = AutoRCWA(freq=1., order=[18, 18], lattice=Lattice.triangular(1.),
                          asr=ASROptions(circle_G=.03), outputs=OutputSpec(fields="none"),
                          dtype=torch.complex128, device=torch.device(args.device))
    checks = []

    def record(name, error, tolerance):
        value = float(error)
        row = {"name": name, "error": value, "tolerance": tolerance,
               "passed": bool(np.isfinite(value) and value <= tolerance)}
        checks.append(row)
        print(f"{'PASS' if row['passed'] else 'FAIL'} {name}: {value:.6g}", flush=True)

    for n in (256, 384, 512):
        for radius in (.05, .251, .4862):
            displacement, jacobian = production_map(simulation, n, radius)
            metrics = analyze(displacement, jacobian, 1e-8)
            for symmetry in ("rotation60", "reflection_x"):
                record(f"grid={n} R={radius} {symmetry} Jacobian",
                       metrics[symmetry]["jacobian_relative_max_error"], 1e-8)
            # The independently evaluated geometry must remain unchanged.
            reference, _ = numpy_reference(n, radius, .03)
            record(f"grid={n} R={radius} coordinates",
                   np.max(np.abs(displacement-reference)), 2e-12)
            if metrics["cartesian_jacobian_det_min"] <= 0:
                raise AssertionError("The map must remain orientation-preserving.")

    for value in (.24, .4862):
        radius = torch.tensor(value, dtype=torch.float64, device=args.device,
                              requires_grad=True)
        objectives = radius_objectives(simulation, radius, 384)
        derivatives = [torch.autograd.grad(objective, radius, retain_graph=index==0)[0]
                       for index, objective in enumerate(objectives)]
        step = 1e-7
        plus = radius_objectives(simulation, value+step, 384)
        minus = radius_objectives(simulation, value-step, 384)
        for index, name in enumerate(("coordinates", "Jacobian")):
            numerical = (float(plus[index])-float(minus[index]))/(2*step)
            derivative = float(derivatives[index])
            error = abs(derivative-numerical)/max(abs(derivative), abs(numerical), 1e-10)
            record(f"R={value} {name} radius gradient vs finite difference", error, 2e-5)

    result = {"device": args.device, "torch_version": torch.__version__, "checks": checks,
              "passed": all(row["passed"] for row in checks)}
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
