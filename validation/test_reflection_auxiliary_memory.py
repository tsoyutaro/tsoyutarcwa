"""Check auxiliary-memory release against normal R and radius gradients."""
from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import rcwa_solver_auto
from studies.gold_motheye2 import optimize_adam as shared
from studies.shared.gold_dispersion import build_gold_model


def auxiliary_bytes(simulation):
    storages = {}
    def visit(item):
        if isinstance(item, torch.Tensor):
            storage = item.untyped_storage()
            storages[(str(item.device), storage.data_ptr())] = storage.nbytes()
        elif is_dataclass(item):
            for field in fields(item):
                visit(getattr(item, field.name))
        elif isinstance(item, dict):
            for value in item.values():
                visit(value)
        elif isinstance(item, (tuple, list)):
            for value in item:
                visit(value)
    for name in ("asr_mappings", "asr_material_tensors", "asr_T_matrices",
                 "asr_Tz_matrices", "E_eigvec_uv", "H_eigvec_uv",
                 "_physical_material_by_layer"):
        visit(getattr(simulation, name))
    return sum(storages.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.device == "cpu":
        torch.set_num_threads(2)
    base = json.loads((ROOT/"studies/gold_motheye2/results/adam_fullband_Nz100_M8/config.json").read_text())
    saved = json.loads((ROOT/"studies/gold_motheye2/results/adam_fullband_Nz100_M8/checkpoint.json").read_text())
    gold = build_gold_model("csv", ROOT/"studies/gold_motheye2/data/au_measured_nk.csv")
    last = []
    class ObservedAutoRCWA(rcwa_solver_auto.AutoRCWA):
        def __init__(self, *values, **options):
            super().__init__(*values, **options)
            last[:] = [self]

    cases = []
    with patch.object(rcwa_solver_auto, "AutoRCWA", ObservedAutoRCWA):
        for grid, gradient in ((96, False), (384, True), (768, False)):
            config = {**base, "order": 2, "slices": 3, "grid": grid}
            runs = []
            for discard in (False, True):
                logits = torch.tensor(saved["best"]["logits"], dtype=torch.float64,
                                      device=args.device)
                radii = torch.stack(shared.radius_tensor(logits, 3, config, torch)).detach().requires_grad_(gradient)
                reflected = shared.reflectance_tensor(700, radii, config, gold,
                                                      torch.device(args.device), torch,
                                                      discard_auxiliary=discard)
                derivative = torch.autograd.grad(reflected, radii)[0] if gradient else None
                runs.append((float(reflected.detach()),
                             derivative.detach().cpu() if gradient else None,
                             auxiliary_bytes(last[0])))
                last.clear()
                del reflected, derivative, radii
            r_error = abs(runs[0][0]-runs[1][0])
            g_error = float((runs[0][1]-runs[1][1]).abs().max()) if gradient else None
            assert r_error <= 1e-11, (grid, "reflectance", r_error)
            if gradient:
                assert torch.isfinite(runs[0][1]).all() and torch.isfinite(runs[1][1]).all()
                assert g_error <= 1e-9, (grid, "gradient", g_error)
            assert runs[0][2] > 0 and runs[1][2] == 0
            row = {"grid": grid, "normal_R": runs[0][0], "memory_reduced_R": runs[1][0],
                   "absolute_R_error": r_error, "maximum_radius_gradient_error": g_error,
                   "normal_auxiliary_bytes": runs[0][2], "reduced_auxiliary_bytes": runs[1][2]}
            cases.append(row)
            print(json.dumps(row), flush=True)
    result = {"device": args.device, "torch_version": torch.__version__, "order": 2,
              "slices": 3, "wavelength_nm": 700, "gold": "measured CSV", "passed": True,
              "cases": cases}
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
