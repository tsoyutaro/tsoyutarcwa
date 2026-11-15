"""Checkpointed Adam: stored best shape and loss always refer to the same state."""
from __future__ import annotations

import copy
import math
import time
from pathlib import Path

from .common import torch_save, write_csv, write_json
from .geometry import initial_logits, radii_values, profile_rows
from .physics import objective


def cpu_tree(value):
    import torch
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def train(request, device, trial=None, evaluator=objective):
    import torch
    c, lattice, params = request["config"], request["lattice"], request["params"]
    folder = Path(request["trial_folder"])
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/"adam_checkpoint.pt"
    top = params["init_top"]
    bottom = top+params["delta_ratio"]*(c["radius_max_nm"]-top)
    bottom = min(c["radius_max_nm"]-1e-7, max(top, bottom))
    logits = torch.nn.Parameter(torch.tensor(initial_logits(top, bottom, c), dtype=torch.float64, device=device))
    optimizer = torch.optim.Adam([logits], lr=params["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=c["scheduler_factor"],
                    patience=c["scheduler_patience"], min_lr=params["lr"]*.001)
    history, best_loss, best_logits, best_step, stale, next_step = [], math.inf, None, None, 0, 0
    if path.exists():
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["signature"] != request["signature"] or state["params"] != params:
            raise ValueError("Adam checkpoint identity differs; use a new experiment directory.")
        with torch.no_grad():
            logits.copy_(state["logits"].to(device))
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history, best_loss = state["history"], state["best_loss"]
        best_logits, best_step, stale, next_step = state["best_logits"], state["best_step"], state["stale"], state["next_step"]
    def save():
        torch_save(path, {"signature": request["signature"], "params": params, "logits": logits.detach().cpu(),
            "optimizer": cpu_tree(optimizer.state_dict()), "scheduler": scheduler.state_dict(),
            "history": history, "best_loss": best_loss, "best_logits": best_logits, "best_step": best_step,
            "stale": stale, "next_step": next_step})
        write_csv(folder/"history.csv", history)
    # A first-step OOM still leaves a resumable, identified starting state.
    save()
    for step in range(next_step, c["adam_steps"]):
        if request.get("deadline") is not None and time.time() >= request["deadline"]:
            save()
            return {"status": "paused", "reason": "time_budget", "next_step": next_step}
        optimizer.zero_grad(set_to_none=True)
        result = evaluator(logits, c, lattice, c["optimization_wavelengths_nm"], c["optimization_order"],
                           c["optimization_grid"], device, gradient=True)
        value = result["objective"]
        gradient = result["gradient"]
        # Evaluate and record the current model BEFORE updating it.
        improved = value < best_loss
        substantial = value < best_loss-c["min_improvement"]
        if improved:
            best_loss, best_logits, best_step = value, logits.detach().cpu().tolist(), step
        stale = 0 if substantial else stale+1
        history.append({"step": step, "objective": value, "best_objective": best_loss,
                        "gradient_norm": float(torch.linalg.vector_norm(gradient).cpu()),
                        "learning_rate": optimizer.param_groups[0]["lr"], "evaluation_seconds": result["seconds"]})
        logits.grad = gradient.to(device)
        optimizer.step()
        with torch.no_grad():
            logits.clamp_(-c["logit_limit"], c["logit_limit"])
        scheduler.step(value)
        next_step = step+1
        save()
        if step % 10 == 0:
            print(f"trial {request['trial_number']} step {step}: mean R={value:.9g}; best={best_loss:.9g}", flush=True)
        if trial is not None and step % c["pruner_interval_steps"] == 0:
            trial.report(best_loss, step)
            if trial.should_prune():
                return {"status": "pruned", "best_loss": best_loss, "next_step": next_step}
        if stale >= c["patience"]:
            break
    # Measure the last updated state as well; never assign an old loss to it.
    final = evaluator(logits, c, lattice, c["optimization_wavelengths_nm"], c["optimization_order"],
                      c["optimization_grid"], device, gradient=False)
    if final["objective"] < best_loss:
        best_loss, best_logits, best_step = final["objective"], logits.detach().cpu().tolist(), next_step
    save()
    result = {"status": "completed", "trial_number": request["trial_number"], "params": params,
              "best_loss": best_loss, "best_logits": best_logits, "best_step": best_step,
              "radii_nm": radii_values(best_logits, c), "completed_steps": next_step}
    write_json(folder/"best.json", result)
    write_csv(folder/"best_profile.csv", profile_rows(best_logits, c))
    return result
