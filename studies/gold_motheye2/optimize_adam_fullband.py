"""Full-band Adam optimization of the measured-Au, 100-slice moth-eye.

Each Adam update uses the weighted mean reflectance gradient at every selected
wavelength. Wavelengths are solved and differentiated one at a time, so peak
autograd memory is that of one optical solve. The partial gradient is saved
after each wavelength for safe resume on a cluster.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studies.gold_motheye2 import optimize_adam as shared

VERSION = "au_adam_fullband_variable_endpoints_v1"
DEFAULT_OUTPUT = HERE / "results" / "adam_fullband_Nz100_M8"

# The row-recomputation fix changes storage, not the optical model or gradient.
# Only this known predecessor may migrate across an ASR source change.
_PRE_RECOMPUTE_ASR_HASHES = {
    "8298b6c6d2a21fa88befdb41c1754583f1b46d7c993df536fcf6da9ba395d70f",
    "1c1e4787dbf6bdf8343fee1c082658b263def9c88b69ce567eca37f26b3fc590",
}
_RECOMPUTE_ASR_SHA256_LF = (
    "523b717b182e28b12fd44ef286092b50247b544a5379519c53bb19bc84b1aa4e"
)


def _signature(config: dict) -> str:
    fixed = {key: value for key, value in config.items()
             if key not in {"requested_steps", "verify_order", "gold_csv", "eval_every",
                            "profile_source_sha256"}}
    return hashlib.sha256(json.dumps(fixed, sort_keys=True).encode()).hexdigest()


def _previous_signature(config: dict) -> str:
    """Signature used before fixing the ASR evaluation grad context."""
    fixed = {key: value for key, value in config.items()
             if key not in {"requested_steps", "verify_order", "gold_csv", "eval_every"}}
    return hashlib.sha256(json.dumps(fixed, sort_keys=True).encode()).hexdigest()


def _source_hash_matches(saved_hash: str, path: Path) -> bool:
    """Accept a source file copied between LF and CRLF file systems."""
    raw = path.read_bytes()
    normalized = raw.replace(b"\r\n", b"\n")
    return saved_hash in {
        hashlib.sha256(raw).hexdigest(),
        hashlib.sha256(normalized).hexdigest(),
        hashlib.sha256(normalized.replace(b"\n", b"\r\n")).hexdigest(),
    }


def _can_migrate_checkpoint(checkpoint: dict, saved: dict | None,
                            current: dict) -> bool:
    if saved is None or checkpoint.get("signature") not in {
        _signature(saved), _previous_signature(saved)
    }:
        return False
    source_paths = {
        "core_sha256": ROOT / "studies" / "gold_motheye" / "converge.py",
        "asr_sha256": ROOT / "rcwa_ext" / "asr.py",
        "auto_sha256": ROOT / "rcwa_ext" / "auto.py",
    }
    for key, path in source_paths.items():
        saved_hash = saved.get(key, "")
        storage_only_fix = (key == "asr_sha256" and
                            saved_hash in _PRE_RECOMPUTE_ASR_HASHES and
                            _source_hash_matches(_RECOMPUTE_ASR_SHA256_LF, path))
        if not (_source_hash_matches(saved_hash, path) or storage_only_fix):
            return False
    ignored = {"requested_steps", "verify_order", "gold_csv", "eval_every",
               "profile_source_sha256", *source_paths}
    saved_fixed = {key: value for key, value in saved.items() if key not in ignored}
    current_fixed = {key: value for key, value in current.items() if key not in ignored}
    return (json.dumps(saved_fixed, sort_keys=True) ==
            json.dumps(current_fixed, sort_keys=True))


def _load_initial_profile(run: Path, current: dict) -> dict:
    """Load parameters only; optical values and Adam moments are not reused."""
    run = run.resolve()
    saved = json.loads((run / "config.json").read_text(encoding="utf-8"))
    checkpoint = json.loads((run / "checkpoint.json").read_text(encoding="utf-8"))
    if saved.get("version") != VERSION or saved.get("training_mode") != "full-band-gradient":
        raise ValueError("--initial-run-dir must be a saved full-band Adam run.")
    if checkpoint.get("signature") not in {_signature(saved), _previous_signature(saved)}:
        raise ValueError("The initial run's config and checkpoint do not match.")
    for key in ("geometry", "segments", "diameter_margin_nm", "logit_limit", "gold_csv_sha256_lf"):
        if saved.get(key) != current.get(key):
            raise ValueError(f"Initial profile has incompatible {key}.")
    # Both known versions use the same radius parameterization. Their only
    # difference is the optional release of reflection-only auxiliary data.
    known_profile_sources = {
        "34365c7b55f9691f1d0293af9c1b85ad01e993a4d10ec6c2c350ede533e08367",
        "57544287841e1a229808eb8d884fd65e0407b021ae53ab7820f3d4430ace5831",
    }
    if (not _source_hash_matches(saved.get("profile_source_sha256", ""), HERE / "optimize_adam.py")
            and not (saved.get("profile_source_sha256") in known_profile_sources
                     and shared.csv_data_hash(HERE / "optimize_adam.py") in known_profile_sources)):
        raise ValueError("Initial profile parameterization source differs.")
    best = checkpoint.get("best")
    if best is None:
        raise ValueError("The initial run has no saved best profile.")
    logits = [float(value) for value in best["logits"]]
    if (len(logits) != current["segments"] + 2 or
            any(not math.isfinite(value) or abs(value) > current["logit_limit"] for value in logits)):
        raise ValueError("The initial run has invalid profile parameters.")
    return {"source_run_dir": str(run), "source_training_signature": checkpoint["signature"],
            "source_best_step": best["step"], "logits": logits}


def _initial_checkpoint(signature: str, logits: list[float]) -> dict:
    return {"signature": signature, "step": 0, "logits": list(logits),
            "m": [0.0] * len(logits), "v": [0.0] * len(logits), "best": None,
            "history": [], "evaluations": [], "pending": None, "pending_update": None}


def _finish_evaluation(output: Path, checkpoint: dict, config: dict,
                       gold_model, device, torch) -> None:
    pending = checkpoint.get("pending")
    if pending is not None:
        label, order_text, step_text = pending["key"].split("|")
        if int(step_text.removeprefix("step=")) != checkpoint["step"]:
            raise RuntimeError("Pending evaluation does not match checkpoint step.")
        shared.evaluate(output, checkpoint, label, int(order_text.removeprefix("M=")),
                        pending["logits"], config, gold_model, device, torch)


def _candidate(output: Path, checkpoint: dict, config: dict,
               gold_model, device, torch) -> dict:
    row = shared.find_evaluation(checkpoint, "candidate", config["order"],
                                 checkpoint["logits"], checkpoint["step"])
    if row is None:
        row = shared.evaluate(output, checkpoint, "candidate", config["order"],
                              checkpoint["logits"], config, gold_model, device, torch)
    if row["mean_reflectance"] < checkpoint["best"]["mean_reflectance"]:
        checkpoint["best"] = {"step": checkpoint["step"],
                              "logits": list(checkpoint["logits"]),
                              "mean_reflectance": row["mean_reflectance"]}
        shared.write_json(output / "checkpoint.json", checkpoint)
    shared.save_results(output, checkpoint, config)
    return row


def _full_gradient_step(output: Path, checkpoint: dict, config: dict,
                        gold_model, device, torch) -> tuple[float, float]:
    """Accumulate and checkpoint one weighted gradient per wavelength."""
    step = checkpoint["step"] + 1
    wavelengths = config["wavelengths_nm"]
    started = time.perf_counter()
    partial = checkpoint.get("pending_update")
    if partial is None:
        partial = {"step": step, "logits": list(checkpoint["logits"]),
                   "next_index": 0, "gradient": [0.0]*len(checkpoint["logits"]),
                   "band_mean": 0.0}
        checkpoint["pending_update"] = partial
        shared.write_json(output / "checkpoint.json", checkpoint)
    elif partial["step"] != step or partial["logits"] != checkpoint["logits"]:
        raise RuntimeError("Partial full-band gradient does not match current checkpoint.")

    for index in range(partial["next_index"], len(wavelengths)):
        wavelength = wavelengths[index]
        weight = config["weights"][index]
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        parameters = torch.tensor(partial["logits"], dtype=torch.float64,
                                  device=device, requires_grad=True)
        radii = shared.radius_tensor(parameters, config["slices"], config, torch)
        result = shared.reflectance_tensor(wavelength, radii, config,
                                           gold_model, device, torch)
        (result * weight).backward()
        if (parameters.grad is None or
                not bool(torch.isfinite(parameters.grad).all().detach().cpu())):
            raise RuntimeError(f"Invalid shape gradient at step {step}, {wavelength:g} nm.")
        gradient = parameters.grad.detach().cpu().tolist()
        value = float(result.detach().cpu())
        partial["gradient"] = [old+float(new) for old,new
                               in zip(partial["gradient"], gradient)]
        partial["band_mean"] += weight*value
        partial["next_index"] = index+1
        shared.write_json(output / "checkpoint.json", checkpoint)
        del result, radii, parameters
        gc.collect()
        memory = (f", CUDA peak={torch.cuda.max_memory_allocated(device)/2**30:.2f} GiB"
                  if device.type == "cuda" else "")
        print(f"step {step}/{config['requested_steps']}: {wavelength:g} nm "
              f"R={value:.7f} ({index+1}/{len(wavelengths)}){memory}", flush=True)

    gradient = partial["gradient"]
    norm = math.sqrt(sum(value*value for value in gradient))
    if norm > 1.0:
        gradient = [value/norm for value in gradient]
    m = [0.9*old + 0.1*g for old,g in zip(checkpoint["m"], gradient)]
    v = [0.999*old + 0.001*g*g for old,g in zip(checkpoint["v"], gradient)]
    updated = [value - config["learning_rate"] *
               (mi/(1-0.9**step))/(math.sqrt(vi/(1-0.999**step))+1e-8)
               for value,mi,vi in zip(checkpoint["logits"], m, v)]
    updated = [max(-shared.LOGIT_LIMIT, min(shared.LOGIT_LIMIT, value))
               for value in updated]
    updated_radii = shared.radius_values(updated, config["slices"], config)
    if not all(left < right for left,right in zip(updated_radii, updated_radii[1:])):
        raise RuntimeError("Updated layer radii are not strictly increasing.")
    mean_before_update = partial["band_mean"]
    checkpoint.update({"step": step, "logits": updated, "m": m, "v": v,
                       "pending_update": None})
    checkpoint["history"].append({"step": step, "wavelength_nm": "all",
                                  "reflectance": mean_before_update,
                                  "gradient_norm": norm,
                                  "seconds": time.perf_counter()-started})
    shared.write_json(output / "checkpoint.json", checkpoint)
    print(f"step {step}: full-band mean={mean_before_update:.8f}, "
          f"|grad|={norm:.3g}", flush=True)
    return mean_before_update, norm


def _dense_validation(output: Path, checkpoint: dict, config: dict,
                      wavelengths: tuple[float, ...], order: int,
                      gold_model, device, torch) -> Path:
    """Independent finer-grid comparison, checkpointed after each solve."""
    if order <= 0:
        raise ValueError("Dense validation requires --verify-order greater than zero.")
    best = checkpoint["best"]
    cone = shared.initial_logits(config)
    identity = {"training_signature": checkpoint["signature"],
                "best_logits": best["logits"], "order": order,
                "wavelengths_nm": list(wavelengths)}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    stem = f"dense_validation_M{order}_{len(wavelengths)}pts_{digest}"
    path = output / f"{stem}.json"
    if path.exists():
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("identity") != identity:
            raise RuntimeError("Dense validation file does not match this design.")
    else:
        document = {"identity": identity, "cases": {"cone": {}, "best": {}}}
        shared.write_json(path, document)
    local_config = {**config, "order": order}
    for label, logits in (("cone", cone), ("best", best["logits"])):
        for wavelength in wavelengths:
            key = f"{wavelength:g}"
            if key in document["cases"][label]:
                continue
            if label == "best" and best["logits"] == cone:
                value = document["cases"]["cone"][key]
            else:
                # The ASR map needs autograd for its spatial Jacobian even
                # though these profile parameters are held fixed.
                with torch.enable_grad():
                    parameters = torch.tensor(logits, dtype=torch.float64, device=device)
                    radii = shared.radius_tensor(parameters, config["slices"], config, torch)
                    result = shared.reflectance_tensor(wavelength, radii, local_config,
                                                       gold_model, device, torch)
                    value = float(result.detach().cpu())
                del parameters, radii, result
                gc.collect()
            if not (-1e-6 <= value <= 1+1e-6):
                raise RuntimeError(f"Nonphysical dense-validation R={value:g} at {wavelength:g} nm.")
            document["cases"][label][key] = value
            shared.write_json(path, document)
            print(f"dense {label} M={order} {wavelength:g} nm R={value:.8f}", flush=True)
    weights = shared.trapezoid_weights(wavelengths)
    means = {label: sum(weight*document["cases"][label][f"{wavelength:g}"]
                        for wavelength,weight in zip(wavelengths, weights))
             for label in ("cone", "best")}
    document["band_means"] = means
    document["cone_minus_best"] = means["cone"]-means["best"]
    shared.write_json(path, document)
    rows = []
    for label, logits in (("cone", cone), ("best", best["logits"])):
        rows.append({"label": label, "order": order, "step": checkpoint["step"],
                     "logits": logits, "values": document["cases"][label],
                     "mean_reflectance": means[label]})
    shared.draw_comparison(output / f"{stem}.svg", [("Dense validation", rows[0], rows[1])],
                           {"wavelengths_nm": wavelengths})
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--gold-csv", type=Path, default=shared.DEFAULT_GOLD_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--initial-run-dir", type=Path,
                        help="Start a new run from this full-band run's best shape, with fresh Adam state")
    parser.add_argument("--wavelengths", default="400:700:50")
    parser.add_argument("--slices", type=int, default=100)
    parser.add_argument("--order", type=int, default=8)
    parser.add_argument("--verify-order", type=int, default=16)
    parser.add_argument("--grid", type=int, default=256)
    parser.add_argument("--segments", type=int, default=8)
    parser.add_argument("--diameter-margin-nm", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=28,
                        help="Full-band Adam updates; each uses every selected wavelength")
    parser.add_argument("--eval-every", type=int, default=4,
                        help="Evaluate the full band after this many Adam updates")
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--validation-wavelengths",
                        help="Optional finer, independent M=verify-order check, e.g. 400:700:10")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if (args.slices < 1 or args.order < 1 or args.verify_order < 0 or
            args.grid < 16 or args.segments < 2 or args.segments > args.slices or
            args.steps < 1 or args.eval_every < 1 or
            not math.isfinite(args.learning_rate) or args.learning_rate <= 0 or
            not math.isfinite(args.diameter_margin_nm) or
            not 0 < args.diameter_margin_nm < 10):
        parser.error("Invalid numerical or optimization setting.")
    try:
        wavelengths = shared.wavelengths_from_text(args.wavelengths)
        validation_wavelengths = (shared.wavelengths_from_text(args.validation_wavelengths)
                                  if args.validation_wavelengths else None)
    except ValueError as error:
        parser.error(str(error))
    if validation_wavelengths is not None and args.verify_order == 0:
        parser.error("--validation-wavelengths requires --verify-order > 0.")
    gold_csv = args.gold_csv.resolve()
    if not gold_csv.is_file():
        parser.error(f"Gold CSV not found: {gold_csv}")
    geometry = {
        "period_nm": 200.0, "height_nm": 500.0,
        "tip_radius_nm": 5.0, "base_radius_nm": 95.0,
        "lattice": "triangular", "substrate_mode": "semi-infinite",
        "asr_circle_g": 0.03,
    }
    config = {
        "version": VERSION, "training_mode": "full-band-gradient",
        "geometry": geometry, "slices": args.slices, "order": args.order,
        "verify_order": args.verify_order, "grid": args.grid,
        "segments": args.segments, "diameter_margin_nm": args.diameter_margin_nm,
        "wavelengths_nm": wavelengths,
        "weights": shared.trapezoid_weights(wavelengths),
        "gold_csv": str(gold_csv),
        "gold_csv_sha256_lf": shared.csv_data_hash(gold_csv),
        "core_sha256": shared.file_hash(ROOT / "studies" / "gold_motheye" / "converge.py"),
        "asr_sha256": shared.file_hash(ROOT / "rcwa_ext" / "asr.py"),
        "auto_sha256": shared.file_hash(ROOT / "rcwa_ext" / "auto.py"),
        "profile_source_sha256": shared.file_hash(HERE / "optimize_adam.py"),
        "learning_rate": args.learning_rate, "logit_limit": shared.LOGIT_LIMIT,
        "objective": "normal-incidence x-polarized total reflectance, trapezoidal band mean",
        "requested_steps": args.steps, "eval_every": args.eval_every,
    }
    output = args.output_dir.resolve()
    if args.initial_run_dir is not None:
        if output == args.initial_run_dir.resolve():
            parser.error("Use a separate --output-dir for a run initialized from a saved profile.")
        try:
            config["initial_profile"] = _load_initial_profile(args.initial_run_dir, config)
        except (ValueError, KeyError, OSError) as error:
            parser.error(str(error))
        # New stages fingerprint the coordinate map and all solver modules,
        # in addition to the legacy compatibility fields above.
        source_paths = sorted((ROOT / "rcwa_ext").glob("*.py")) + [
            ROOT / "rcwa_solver_auto.py", HERE / "optimize_adam.py",
            ROOT / "studies/gold_motheye/converge.py",
            ROOT / "studies/shared/gold_dispersion.py"]
        config["solver_sources_sha256_lf"] = {
            str(path.relative_to(ROOT)).replace("\\", "/"): shared.csv_data_hash(path)
            for path in source_paths}
        config["optimizer_source_sha256_lf"] = shared.csv_data_hash(Path(__file__))
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint.json"
    signature = _signature(config)
    cone_logits = shared.initial_logits(config)
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("signature") != signature:
            saved_config_path = output / "config.json"
            saved_config = (json.loads(saved_config_path.read_text(encoding="utf-8"))
                            if saved_config_path.exists() else None)
            if not _can_migrate_checkpoint(checkpoint, saved_config, config):
                raise RuntimeError("Checkpoint settings differ. Use a new --output-dir.")
            checkpoint["signature"] = signature
            shared.write_json(checkpoint_path, checkpoint)
            print("Updated the checkpoint signature for compatible ASR fixes.", flush=True)
        if args.steps < checkpoint["step"] and not args.prepare_only:
            raise RuntimeError("--steps is below the completed update count.")
    else:
        initial_logits = config.get("initial_profile", {}).get("logits", cone_logits)
        checkpoint = _initial_checkpoint(signature, initial_logits)
    shared.write_json(output / "config.json", config)
    shared.save_profile(output, "cone", cone_logits, config)
    if "initial_profile" in config:
        shared.save_profile(output, "initial", config["initial_profile"]["logits"], config)
    shared.write_json(checkpoint_path, checkpoint)
    if args.prepare_only:
        print(f"Prepared full-band run in {output}")
        return 0

    import torch
    from studies.shared.gold_dispersion import build_gold_model
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot see a GPU.")
    gold_model = build_gold_model("csv", gold_csv)
    for wavelength in wavelengths:
        gold_model(wavelength)
    if validation_wavelengths is not None:
        for wavelength in validation_wavelengths:
            gold_model(wavelength)
    shared.write_json(checkpoint_path, checkpoint)
    print(f"Full-band Adam: {len(wavelengths)} wavelengths per update, "
          f"{args.steps} updates, M={args.order}, Nz={args.slices}", flush=True)
    print("Differentiable ASR conversion: recompute FFT rows during backward.", flush=True)

    cone = shared.find_evaluation(checkpoint, "cone", args.order, cone_logits)
    if cone is None:
        cone = shared.evaluate(output, checkpoint, "cone", args.order,
                               cone_logits, config, gold_model, device, torch)
    if checkpoint["best"] is None:
        checkpoint["best"] = {"step": 0, "logits": cone_logits,
                              "mean_reflectance": cone["mean_reflectance"]}
        shared.write_json(checkpoint_path, checkpoint)
    _finish_evaluation(output, checkpoint, config, gold_model, device, torch)
    if "initial_profile" in config and checkpoint["step"] == 0 and checkpoint["logits"] != cone_logits:
        # Re-evaluate the starting shape with this stage's wavelength weights
        # before any update, so it remains eligible as the best candidate.
        _candidate(output, checkpoint, config, gold_model, device, torch)
    if checkpoint["step"] > 0 and (
            checkpoint["step"] % args.eval_every == 0 or checkpoint["step"] == args.steps):
        _candidate(output, checkpoint, config, gold_model, device, torch)
    shared.save_results(output, checkpoint, config)

    for _ in range(checkpoint["step"], args.steps):
        _full_gradient_step(output, checkpoint, config, gold_model, device, torch)
        if checkpoint["step"] % args.eval_every == 0 or checkpoint["step"] == args.steps:
            row = _candidate(output, checkpoint, config, gold_model, device, torch)
            print(f"evaluated mean: cone={cone['mean_reflectance']:.8f}, "
                  f"candidate={row['mean_reflectance']:.8f}, "
                  f"best={checkpoint['best']['mean_reflectance']:.8f}", flush=True)

    best = checkpoint["best"]
    if args.verify_order:
        verified_cone = shared.find_evaluation(checkpoint, "verify_cone",
                                               args.verify_order, cone_logits)
        if verified_cone is None:
            verified_cone = shared.evaluate(output, checkpoint, "verify_cone",
                                            args.verify_order, cone_logits, config,
                                            gold_model, device, torch)
        if shared.find_evaluation(checkpoint, "verify_best", args.verify_order,
                                  best["logits"]) is None:
            if best["logits"] == cone_logits:
                checkpoint["evaluations"].append({**verified_cone,
                                                  "label": "verify_best",
                                                  "logits": best["logits"]})
                shared.write_json(checkpoint_path, checkpoint)
            else:
                shared.evaluate(output, checkpoint, "verify_best",
                                args.verify_order, best["logits"], config,
                                gold_model, device, torch)
    shared.save_results(output, checkpoint, config)
    if validation_wavelengths is not None:
        dense_path = _dense_validation(output, checkpoint, config, validation_wavelengths,
                                       args.verify_order, gold_model, device, torch)
        print(f"Dense validation: {dense_path}")
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
