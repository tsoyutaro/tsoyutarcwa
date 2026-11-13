"""Resume Cu H400 spectra, then evaluate Au H500 and its reference profiles."""
from __future__ import annotations

import argparse
import cmath
from contextlib import contextmanager
import csv
import importlib.util
import math
from pathlib import Path
import subprocess
import sys

import continue_convergence as cc

ROOT = Path(__file__).resolve().parent
VISIBLE = cc.numbers("400:700:3")
NEAR_INFRARED = cc.numbers("700:1000:3")
ORDER, GRID = 40, 512


def command(arguments):
    result = subprocess.run([str(a) for a in arguments], cwd=ROOT)
    if result.returncode < 0 or result.returncode >= 128:
        raise KeyboardInterrupt
    if result.returncode:
        raise RuntimeError(f"Analysis command stopped with exit code {result.returncode}.")


def specification(material):
    case = ROOT / material / f"{material.lower()}_motheye_P200_H500_Nz15_backtracking_v1"
    if material == "Cu":
        geometry = case / "geometry_cases/P200_H400_Nz15"
        config = geometry / "config_hex_OFF_optuna30_backtracking_M14_Q512.json"
        training = geometry / "results/hex_OFF_optuna30_backtracking_v1_M14_Q512"
        height = 400
    else:
        config = case / "config.json"
        training = case / "results/hex_OFF_optuna30_backtracking_from5_v1_M20_Q512"
        height = 500
    if (training / "controller.lock").exists():
        raise RuntimeError(f"The training controller lock exists: {training}")
    if cc.read(training / "workflow.json").get("status") != "completed":
        raise ValueError(f"No completed optimization: {training}")
    c = cc.read(config)
    if (c.get("material") != material or c.get("period_nm") != 200
            or c.get("height_nm") != height or c.get("layers") != 15
            or c.get("normal_mode") != "off" or c.get("lattices") != ["hexagonal"]):
        raise ValueError(f"Unexpected training conditions: {config}")
    profile = cc.read(training / "hexagonal/report.json")
    logits = profile.get("best", {}).get("logits")
    if not isinstance(logits, list) or len(logits) != c["layers"] + 1:
        raise ValueError(f"No completed best profile: {training}")
    evaluation = training.with_name(training.name + "_forward_eval")
    return dict(material=material, case_dir=str(case), config=str(config),
                run_root=str(evaluation), training=str(training), profile=profile)


def prepare(spec, args):
    output = Path(spec["run_root"])
    profile = output / "hexagonal/report.json"
    if profile.exists() and cc.read(profile).get("best", {}).get("logits") != spec["profile"]["best"]["logits"]:
        raise ValueError(f"Saved evaluation uses a different shape: {output}")
    lock = output / "controller.lock"
    if lock.exists():
        if not args.recover_lock:
            raise RuntimeError(f"Another analysis holds {lock}. After it stops, use --recover-lock.")
        (lock / "owner.json").unlink(missing_ok=True)
        lock.rmdir()
    arguments = [sys.executable, Path(spec["case_dir"]) / "run.py",
                 "--config", spec["config"], "--output-root", output,
                 "--device", args.device, "--mode", "convergence", "--prepare-only"]
    if args.recover_lock:
        arguments.append("--recover-lock")
    command(arguments)
    cc.write(profile, spec["profile"])
    cc.write(output / "cases.json", [
        {key: spec[key] for key in ("material", "case_dir", "config", "run_root")}
    ])
    return output


def convergence(spec, args):
    output = Path(spec["run_root"])
    command([sys.executable, ROOT / "continue_convergence.py",
             "--root", output, "--cases-file", output / "cases.json",
             "--device", args.device, "--materials", spec["material"],
             "--lattice", "hexagonal", "--orders", "4:40:2", "--grids", str(GRID),
             "--wavelengths", "400,550,700", "--backend", "matrix",
             "--oom-policy", "stop-material"])
    summary = cc.read(output / "convergence_continue_summary.json")
    report = cc.read(summary["materials"][spec["material"]]["report"])
    target = any(r["order"] == ORDER and r["grid"] == GRID and r["status"] == "completed"
                 for r in report["records"])
    if any(r["status"] in cc.OOM for r in report["records"]) or not target:
        raise RuntimeError("The M40/Q512 convergence check did not complete; spectra were stopped.")
    print(f"{spec['material']} convergence check: {report['numerical_convergence_passed']}", flush=True)


@contextmanager
def locked(output):
    lock = output / "controller.lock"
    try:
        cc.acquire(lock)
    except FileExistsError as error:
        raise RuntimeError(f"Another analysis holds {lock}. After it stops, use --recover-lock.") from error
    try:
        yield
    finally:
        (lock / "owner.json").unlink(missing_ok=True)
        lock.rmdir()


def context(spec, args):
    output = Path(spec["run_root"]) / "hexagonal/postprocess/resumable_spectrum"
    request = {key: spec[key] for key in ("material", "case_dir", "config", "run_root")}
    result = cc.launch(output / "preparation",
                       dict(mode="prepare", **request, device=args.device, lattice="hexagonal",
                            waves=VISIBLE, backend="matrix"))
    if result["status"] != "ready":
        raise RuntimeError(f"Cannot prepare spectral evaluation: {result.get('error', result['status'])}")
    return result["context"]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def previous_rows(spec, current):
    """Import completed classic spectrum workers only with matching provenance."""
    c = {**current["config"], "rcwa_acceleration": "matrix"}
    folders = {Path(spec["run_root"]), Path(spec["training"])}
    if spec["material"] == "Au":
        folders.update((Path(spec["case_dir"]) / "results").glob("*"))
    rows = {}
    for folder in sorted(folders):
        manifest_path = folder / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = cc.read(manifest_path)
        if (manifest.get("active_source_sha256", manifest.get("source_sha256")) != current["source_sha256"]
                or manifest.get("config") != current["config"]):
            continue
        for path in sorted((folder / "hexagonal/postprocess/spectrum").glob("*/M40_Q512/request.json")):
            request = cc.read(path)
            result_path = path.with_name("result.json")
            if (request.get("config") != c or request.get("logits") != current["logits"]
                    or request.get("signature") != manifest.get("signature")
                    or request.get("runtime_environment") != current["runtime_environment"]
                    or not result_path.is_file()):
                continue
            result = cc.read(result_path)
            if result.get("status") == "completed":
                rows.update({row["wavelength_nm"]: row for row in result.get("rows", [])})
    return rows


def save_spectrum(folder, label, current, waves, rows, method):
    folder.mkdir(parents=True, exist_ok=True)
    values = dict(status="completed", label=label, order=ORDER, grid=GRID, method=method,
                  wavelengths_nm=waves, source_context=current, rows=rows,
                  mean_R=sum(r["reflectance"] for r in rows) / len(rows),
                  mean_absorptance_total=sum(r["absorptance_total"] for r in rows) / len(rows),
                  spectrum_csv=str(folder / "spectrum.csv"))
    with (folder / "spectrum.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cc.write(folder / "report.json", values)
    print(f"{label}: mean R = {100 * values['mean_R']:.8g}%; {values['spectrum_csv']}", flush=True)


def spectrum(spec, current, label, waves, flat=False):
    identity = dict(context={**current, "waves": []}, order=ORDER, grid=GRID, flat=flat)
    folder = Path(spec["run_root"]) / "hexagonal/postprocess/resumable_spectrum" / cc.digest(identity)[:16]
    folder.mkdir(parents=True, exist_ok=True)
    saved = {} if flat else previous_rows(spec, current)
    model = None
    if flat:
        loader = module(Path(spec["case_dir"]) / "optimizer/tabulated_nk.py", "_analysis_tabulated_nk")
        model = loader.load_tabulated_nk(current["config"]["_metal_csv_path"])
    rows = []
    for i, wave in enumerate(waves, 1):
        point = folder / f"wavelength_{wave:.12g}nm"
        result_path = point / "result.json"
        cached = cc.read(result_path) if result_path.is_file() else {}
        if cached.get("status") == "completed":
            row = cached["rows"][0]
            action = "cached"
        elif wave in saved:
            row = saved[wave]
            cc.write(result_path, dict(status="completed", rows=[row], reused_classic_spectrum=True))
            action = "reused"
        elif flat:
            index = cmath.sqrt(model(wave))
            reflectance = abs((1 - index) / (1 + index)) ** 2
            row = dict(wavelength_nm=wave, reflectance=reflectance,
                       absorptance_total=1 - reflectance, power_into_substrate=1 - reflectance,
                       motheye_absorptance=0., transmittance_far=0.)
            cc.write(result_path, dict(status="completed", rows=[row]))
            action = "computed"
        else:
            point_context = point / "context.json"
            cc.write(point_context, {**current, "waves": [wave]})
            result = cc.launch(point, dict(mode="evaluate", context=str(point_context),
                                          order=ORDER, grid=GRID, device=current["device"]))
            if result["status"] != "completed":
                raise RuntimeError(f"{label} at {wave:g} nm: {result.get('error', result['status'])}")
            row = result["rows"][0]
            action = "computed"
        if row["wavelength_nm"] != wave or not math.isfinite(row["reflectance"]):
            raise ValueError(f"Invalid saved spectral point: {result_path}")
        rows.append(row)
        print(f"{label} {i}/{len(waves)}: {wave:g} nm {action}", flush=True)
    save_spectrum(folder / ("near_infrared" if waves == NEAR_INFRARED else "visible"),
                  label, current, waves, rows, "Fresnel, order-independent" if flat else "RCWA matrix")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--recover-lock", action="store_true",
                        help="Recover evaluation locks after all their previous processes have stopped")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Validate saved shapes and initialize evaluation directories without RCWA")
    args = parser.parse_args(argv)
    try:
        specs = [specification(material) for material in ("Cu", "Au")]
        for spec in specs:
            output = prepare(spec, args)
            if args.prepare_only:
                continue
            if spec["material"] == "Au":
                convergence(spec, args)
            with locked(output):
                current = context(spec, args)
                spectrum(spec, current, spec["material"] + " optimized 400-700", VISIBLE)
                if spec["material"] == "Au":
                    spectrum(spec, current, "Au optimized 700-1000", NEAR_INFRARED)
                    spectrum(spec, current, "Au flat 400-700", VISIBLE, flat=True)
                    geometry = module(Path(spec["case_dir"]) / "optimizer/geometry.py", "_analysis_geometry")
                    cone = {**current, "logits": geometry.baseline_logits(
                        {**current["config"], "baseline_tip_nm": 5., "baseline_base_nm": 95.})}
                    spectrum(spec, cone, "Au cone r5/r95 400-700", VISIBLE)
        return 0
    except KeyboardInterrupt:
        print("Interrupted. Saved wavelengths will be reused on the next run.", flush=True)
        return 130
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(str(error), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
