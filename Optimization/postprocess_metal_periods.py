"""Evaluate completed Ti/Cr H500 optimizations at P100, P150, P250 and P300."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import continue_convergence as cc
from postprocess_cu_au import VISIBLE, context, convergence, locked, prepare, spectrum

ROOT = Path(__file__).resolve().parent
PERIODS = (100, 150, 250, 300)
CASES = {"Ti": "Ti/ti_motheye_P200_H500_Nz15", "Cr": "cr_motheye"}
RUN_PATTERN = re.compile(r"hex_OFF_optuna30_(auto_)?v1_M(\d+)_Q512")


def specification(material, period):
    case = ROOT / CASES[material]
    geometry = case / "geometry_cases" / f"P{period}_H500_Nz15"
    completed = []
    for run in (geometry / "results").glob("hex_OFF_optuna30*_M*_Q512"):
        match = RUN_PATTERN.fullmatch(run.name)
        workflow = run / "workflow.json"
        if match and workflow.is_file() and cc.read(workflow).get("status") == "completed":
            completed.append((int(match[2]), bool(match[1]), run))
    if not completed:
        raise ValueError(f"No completed optimization: {geometry}")
    order, _, training = max(completed)
    if (training / "controller.lock").exists():
        raise RuntimeError(f"The training controller lock exists: {training}")
    config = geometry / f"config_hex_OFF_optuna30_M{order}_Q512.json"
    c = cc.read(config)
    expected = dict(material=material, period_nm=period, height_nm=500,
                    layers=15, normal_mode="off", lattices=["hexagonal"],
                    optimization_order=order, optimization_grid=512, trials=30)
    if any(c.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Unexpected training conditions: {config}")
    profile = cc.read(training / "hexagonal/report.json")
    logits = profile.get("best", {}).get("logits")
    if not isinstance(logits, list) or len(logits) != c["layers"] + 1:
        raise ValueError(f"No completed best profile: {training}")
    if not (training / "manifest.json").is_file():
        raise ValueError(f"Saved training manifest is missing: {training}")
    evaluation = training.with_name(training.name + "_forward_eval")
    return dict(material=material, case_dir=str(case), config=str(config),
                run_root=str(evaluation), training=str(training), profile=profile)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material", choices=sorted(CASES), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Check saved results and initialize evaluation directories without RCWA")
    parser.add_argument("--recover-lock", action="store_true",
                        help="Recover evaluation locks after all their previous processes have stopped")
    args = parser.parse_args(argv)
    failed = False
    try:
        for period in PERIODS:
            print(f"\n{args.material} P={period} H=500 nm", flush=True)
            try:
                spec = specification(args.material, period)
                print(f"Training: {spec['training']}", flush=True)
                output = prepare(spec, args)
                if args.prepare_only:
                    continue
                convergence(spec, args)
                with locked(output):
                    current = context(spec, args)
                    spectrum(spec, current, f"{args.material} P{period}/H500 400-700", VISIBLE)
            except (OSError, ValueError, RuntimeError, KeyError) as error:
                failed = True
                print(f"{args.material} P{period}: {error}; continuing to the next period.",
                      file=sys.stderr, flush=True)
        return 1 if failed else 0
    except KeyboardInterrupt:
        print("Interrupted. Saved conditions and wavelengths can be reused on the next run.", flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
