"""Optimize fifteen-layer ellipse and rectangle tapers with OFF-NVM and Adam."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from .geometry import logits_from_knots
from .model import Config
from .nv_optimize_model import InvalidResponse, evaluate, half_widths, wavelength_loss
from .run import save_csv, save_json


LAYERS = 15
INDICES = (1.5, 2.0, 2.5)
SHAPES = ("ellipse", "rectangle")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "nv_adam_optuna"


def initial_logits(trial, cfg):
    axes = []
    for axis in ("x", "y"):
        tip = trial.suggest_float(f"tip_{axis}_nm", 6.0, 85.0)
        taper = trial.suggest_float(f"taper_fraction_{axis}", 0.05, 0.99)
        base = tip + taper * (cfg.half_max_nm - tip)
        axes.append(np.linspace(tip, base, cfg.layers))
    return logits_from_knots(axes, cfg.half_min_nm, cfg.half_max_nm).reshape(2, cfg.layers)


def spectral_loss(logits, shape, cfg, args, *, backward):
    """Accumulate gradients one wavelength at a time to bound GPU memory."""
    if backward:
        logits.grad = None
    total = 0.0
    with torch.set_grad_enabled(backward):
        for wavelength in args.wavelengths:
            loss = wavelength_loss(half_widths(logits, cfg), shape, wavelength, cfg, args.goal)
            value = loss.detach().item()
            if not math.isfinite(value):
                raise InvalidResponse(f"Nonfinite {args.goal} loss at {wavelength:g} nm")
            total += value
            if backward:
                (loss / len(args.wavelengths)).backward()
    if backward and (logits.grad is None or not torch.isfinite(logits.grad).all()):
        raise InvalidResponse("Nonfinite or missing geometry gradient")
    return total / len(args.wavelengths)


def train(trial, shape, cfg, args, optuna):
    lr = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
    logits = torch.nn.Parameter(torch.tensor(initial_logits(trial, cfg), dtype=torch.float64,
                                             device=cfg.device))
    adam = torch.optim.Adam([logits], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        adam, factor=0.5, patience=40, min_lr=lr * 0.001,
    )
    best_loss, best_step, best_widths = math.inf, 0, None
    for step in range(args.adam_steps + 1):
        loss = spectral_loss(logits, shape, cfg, args, backward=step < args.adam_steps)
        if loss < best_loss:
            best_loss, best_step = loss, step
            best_widths = half_widths(logits.detach(), cfg).cpu().tolist()
        trial.report(best_loss, step)
        if trial.should_prune():
            raise optuna.TrialPruned()
        if step == args.adam_steps or step - best_step >= args.patience:
            break
        scheduler.step(loss)
        adam.step()
        with torch.no_grad():
            logits.clamp_(-20, 20)
    trial.set_user_attr("half_widths_nm", best_widths)
    trial.set_user_attr("best_adam_step", best_step)
    return best_loss


def run_case(args, shape, index, optuna):
    cfg = Config(n_substrate=index, layers=LAYERS, order=args.order, grid=args.grid,
                 device=args.device)
    cfg.validate()
    label = f"{shape}_n{index:.1f}".replace(".", "p")
    folder = args.output / label / args.goal
    settings = dict(shape=shape, material_index=index, substrate_index=index,
                    normal_mode="off", lanczos=False, target_deg=90,
                    goal=args.goal, wavelengths_nm=args.wavelengths, config=cfg.to_dict(),
                    adam_steps=args.adam_steps, patience=args.patience, seed=args.seed)
    path = folder / "run_metadata.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != settings:
        raise ValueError(f"Settings changed for {folder}. Use a different --output.")
    save_json(path, settings)

    study = optuna.create_study(
        study_name=label + "_" + args.goal, direction="minimize",
        storage=f"sqlite:///{(folder / 'study.sqlite3').resolve().as_posix()}",
        load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20),
    )
    previous_trials = study.get_trials(deepcopy=False)
    remaining = args.trials - sum(t.state.is_finished() for t in previous_trials)
    if remaining > 0:
        if previous_trials:
            # Keep the stored observations and advance the random sequence
            # on restart, including while TPE is still drawing startup trials.
            study.sampler = optuna.samplers.TPESampler(seed=args.seed + len(previous_trials))
        study.optimize(lambda trial: train(trial, shape, cfg, args, optuna),
                       n_trials=remaining, catch=(InvalidResponse, torch.linalg.LinAlgError))
    trials = study.get_trials(deepcopy=False)
    save_csv(folder / "trials.csv", [
        dict(trial=t.number, status=t.state.name, objective=t.value,
             best_adam_step=t.user_attrs.get("best_adam_step"), parameters=json.dumps(t.params))
        for t in trials
    ])
    if not any(t.state == optuna.trial.TrialState.COMPLETE for t in trials):
        raise RuntimeError(f"No completed trials in {folder}. See trials.csv and the console errors.")
    best = study.best_trial
    widths = torch.tensor(best.user_attrs["half_widths_nm"], dtype=torch.float64, device=cfg.device)
    rows = evaluate(widths, shape, args.wavelengths, cfg)
    errors = [row["target_error_deg"] for row in rows if row["target_error_deg"] is not None]
    phase_valid = len(errors) == len(rows)
    report = dict(mean_reflectance=float(np.mean([
                      (row["Rx_total"] + row["Ry_total"]) / 2 for row in rows])),
                  phase_rms_deg=float(np.sqrt(np.mean(np.square(errors)))) if phase_valid else None,
                  phase_max_error_deg=max(map(abs, errors)) if phase_valid else None,
                  max_energy_error=max(row["energy_error"] for row in rows))
    best_design = dict(**settings, best_trial=best.number,
                       best_adam_step=best.user_attrs["best_adam_step"],
                       parameters=best.params, objective=best.value,
                       objective_unit="power fraction" if args.goal == "reflectance" else "rad^2",
                       half_widths_nm=widths.cpu().tolist(), metrics=report)
    dz = cfg.height_nm / cfg.layers
    save_csv(folder / "best_profile.csv", [
        dict(layer=i + 1, z_center_nm=(i + 0.5) * dz, thickness_nm=dz,
             half_x_nm=a, half_y_nm=b)
        for i, (a, b) in enumerate(widths.T.cpu().tolist())
    ])
    save_csv(folder / "best_spectrum.csv", rows)
    save_json(folder / "best_design.json", best_design)
    print(f"{label} {args.goal}: {json.dumps(report)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal", required=True, choices=("reflectance", "phase"))
    parser.add_argument("--shapes", nargs="+", choices=SHAPES, default=list(SHAPES))
    parser.add_argument("--indices", nargs="+", type=float, choices=INDICES, default=list(INDICES))
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:<index>")
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--grid", type=int, default=128)
    parser.add_argument("--wavelengths", nargs="+", type=float, default=list(range(400, 701, 10)))
    parser.add_argument("--trials", type=int, default=30, help="Total Optuna trial budget per case")
    parser.add_argument("--adam-steps", type=int, default=500)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if min(args.order, args.grid, args.trials, args.adam_steps, args.patience, args.threads) < 1:
        parser.error("Order, grid, trial budget, Adam steps, patience and threads must be positive.")
    if args.grid < max(32, 4 * args.order + 4):
        parser.error("NVM grid must be at least max(32, 4*order+4).")
    if any(not math.isfinite(w) or w <= 0 for w in args.wavelengths) or len(set(args.wavelengths)) != len(args.wavelengths):
        parser.error("Wavelengths must be positive, finite and unique.")
    args.wavelengths.sort()
    try:
        device = torch.device(args.device)
    except (TypeError, RuntimeError) as exc:
        parser.error(str(exc))
    if device.type not in {"cpu", "cuda"}:
        parser.error("Use --device cpu, cuda, or cuda:<index>.")
    if device.type == "cuda" and (not torch.cuda.is_available()
                                   or (device.index is not None and device.index >= torch.cuda.device_count())):
        parser.error(f"{device} is not available in this PyTorch environment.")
    try:
        import optuna
    except ImportError as exc:
        parser.exit(1, f"Optuna import failed: {exc}. Install the package from requirements.txt.\n")
    torch.set_num_threads(args.threads)
    for shape in dict.fromkeys(args.shapes):
        for index in dict.fromkeys(args.indices):
            run_case(args, shape, index, optuna)


if __name__ == "__main__":
    main()
