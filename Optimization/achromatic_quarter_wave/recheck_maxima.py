"""Re-evaluate the screened maximum-phase candidates at a higher RCWA order.

This rechecks fixed candidates; it does not re-optimize at the higher order.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from Optimization.achromatic_quarter_wave.model import Config, evaluate
from Optimization.achromatic_quarter_wave.run import save_csv, save_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--screen", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--order", type=int, default=5)
    p.add_argument("--grid", type=int, default=256)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = p.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((args.screen / "summary.json").read_text(encoding="utf-8"))
    history = [json.loads(x) for x in (args.screen / "history.jsonl").read_text(encoding="utf-8").splitlines()]
    rows, designs = [], []
    for item in summary["per_wavelength_maxima"]:
        if not item["transmission_constraint"]:
            continue
        n = float(item["case"].split("_")[0][1:].replace("p", "."))
        shape = item["case"].split("_")[1]
        d = next(d for d in history if d["structure_index"] == n
                 and d["shape"] == shape and d["candidate"] == item["candidate"])
        cfg = replace(Config(**d["config"]), order=args.order, grid=args.grid, device=args.device)
        r = evaluate(d["knots_nm"], shape, n, item["wavelength_nm"], cfg)
        rows.append(dict(case=item["case"], candidate=item["candidate"],
                         order=cfg.order, grid=cfg.grid, layers=cfg.layers, **r))
        designs.append(dict(case=item["case"], wavelength_nm=item["wavelength_nm"],
                            knots_nm=d["knots_nm"], config=cfg.to_dict()))
        save_csv(args.output / "maxima_recheck.csv", rows)
        save_json(args.output / "maxima_designs.json", designs)
        print(item["case"], item["wavelength_nm"], r["retardance_deg"], flush=True)


if __name__ == "__main__":
    main()
