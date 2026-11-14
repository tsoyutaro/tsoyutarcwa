"""Evaluate saved P200/H500 metal optima at 703:1000:3 nm, M40/Q512."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import continue_convergence as cc
from postprocess_cu_au import context, locked, module, prepare, spectrum

ROOT = Path(__file__).resolve().parent
WAVELENGTHS = cc.numbers("703:1000:3")
CASES = {
    "Al": ("Al/al_motheye_P200_H500_Nz15", "config_hex_OFF_optuna30_M8_Q512.json",
           "hex_OFF_optuna30_v1_M8_Q512"),
    "Ti": ("Ti/ti_motheye_P200_H500_Nz15", "config_hex_OFF_optuna30_M8_Q512.json",
           "hex_OFF_optuna30_v1_M8_Q512"),
    "Cr": ("cr_motheye", "config_hex_OFF_optuna30_M8_Q512.json",
           "hex_OFF_optuna30_v1_M8_Q512"),
    "Cu": ("Cu/cu_motheye_P200_H500_Nz15_backtracking_v1", "config.json",
           "hex_OFF_optuna30_backtracking_from15_v1_M14_Q512"),
}


def specification(material):
    relative, config_name, run_name = CASES[material]
    case = ROOT / relative
    config = case / config_name
    training = case / "results" / run_name
    if (training / "controller.lock").exists():
        raise RuntimeError(f"The training controller lock exists: {training}")
    if cc.read(training / "workflow.json").get("status") != "completed":
        raise ValueError(f"No completed optimization: {training}")
    if not (training / "manifest.json").is_file():
        raise ValueError(f"Saved training manifest is missing: {training}")
    c = cc.read(config)
    expected = dict(material=material, period_nm=200, height_nm=500, layers=15,
                    normal_mode="off", lattices=["hexagonal"], optimization_grid=512)
    if any(c.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Unexpected training conditions: {config}")
    profile = cc.read(training / "hexagonal/report.json")
    logits = profile.get("best", {}).get("logits")
    if not isinstance(logits, list) or len(logits) != c["layers"] + 1:
        raise ValueError(f"No completed best profile: {training}")
    loader_path = (ROOT.parent / "studies/shared/tabulated_nk.py" if material == "Cr"
                   else case / "optimizer/tabulated_nk.py")
    loader = module(loader_path, "_nir_tabulated_nk")
    csv_root = config.parent if material == "Cr" else case
    model = loader.load_tabulated_nk(csv_root / c["material_csv"])
    for wave in WAVELENGTHS:
        if model(wave).imag <= 0:
            raise ValueError(f"Material must have passive absorption at {wave:g} nm.")
    evaluation = training.with_name(training.name + "_NIR_M40_Q512_100_v1")
    return dict(material=material, case_dir=str(case), config=str(config),
                run_root=str(evaluation), training=str(training), profile=profile)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materials", default="Al,Ti,Cr,Cu")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Check saved shapes and dispersion coverage without RCWA")
    parser.add_argument("--recover-lock", action="store_true",
                        help="Recover evaluation locks after all their previous processes have stopped")
    args = parser.parse_args(argv)
    materials = args.materials.split(",")
    if (not materials or len(materials) != len(set(materials))
            or any(material not in CASES for material in materials)):
        parser.error("Select distinct materials from Al,Ti,Cr,Cu.")
    failed = False
    try:
        for material in materials:
            print(f"\n{material}: P200/H500, M40/Q512, 703-1000 nm, 100 wavelengths", flush=True)
            try:
                spec = specification(material)
                print(f"Training: {spec['training']}", flush=True)
                output = prepare(spec, args)
                if args.prepare_only:
                    continue
                with locked(output):
                    current = context(spec, args)
                    spectrum(spec, current, f"{material} optimized 703-1000", WAVELENGTHS)
            except (OSError, ValueError, RuntimeError, KeyError) as error:
                failed = True
                print(f"{material}: {error}; continuing to the next material.", file=sys.stderr, flush=True)
        return 1 if failed else 0
    except KeyboardInterrupt:
        print("Interrupted. Saved wavelengths will be reused on the next run.", flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
