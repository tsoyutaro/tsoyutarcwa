"""Checkpointed Adam with rollback after numerically nonphysical proposals."""
from __future__ import annotations

import copy
import math
from pathlib import Path
import time

from .common import torch_save, write_csv, write_json
from .geometry import initial_logits, profile_rows, radii_values
from .physics import NonphysicalResult, objective

POLICY = "validated-adam-backtracking-v1"


def cpu_tree(value):
    import torch
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def train(request, device, trial=None, evaluator=objective):
    import torch

    config = request["config"]
    parameters = request["params"]
    lattice = request["lattice"]
    folder = Path(request["trial_folder"])
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint_path = folder / "adam_checkpoint.pt"
    maximum_retries = config["nonphysical_max_retries"]
    retry_factor = config["nonphysical_lr_factor"]

    top = parameters["init_top"]
    bottom = top + parameters["delta_ratio"] * (config["radius_max_nm"] - top)
    bottom = min(config["radius_max_nm"] - 1e-7, max(top, bottom))
    logits = torch.nn.Parameter(torch.tensor(
        initial_logits(top, bottom, config), dtype=torch.float64, device=device))
    optimizer = torch.optim.Adam([logits], lr=parameters["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=config["scheduler_factor"],
        patience=config["scheduler_patience"], min_lr=parameters["lr"] * .001)
    state = dict(
        policy=POLICY, signature=request["signature"], params=parameters,
        next_step=0, history=[], best_loss=math.inf, best_logits=None,
        best_step=None, stale=0, current_evaluation=None, retry_index=0,
        recovery_events=[], finished_status=None, termination_reason=None,
        last_reported_step=-1)

    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if (state.get("policy") != POLICY or state.get("signature") != request["signature"]
                or state.get("params") != parameters):
            raise ValueError("Backtracking checkpoint identity differs; use a new experiment directory.")
        with torch.no_grad():
            logits.copy_(state["logits"].to(device))
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])

    def save():
        # After the initial evaluation, every saved current point has passed
        # all wavelengths. A rejected proposal never replaces this point.
        state["logits"] = logits.detach().cpu().clone()
        state["optimizer"] = cpu_tree(optimizer.state_dict())
        state["scheduler"] = copy.deepcopy(scheduler.state_dict())
        torch_save(checkpoint_path, state)
        write_csv(folder / "history.csv", state["history"])
        write_json(folder / "recovery.json", dict(
            policy=POLICY, maximum_retries=maximum_retries, lr_factor=retry_factor,
            current_retry_index=state["retry_index"], events=state["recovery_events"]))

    def evaluate():
        result = evaluator(
            logits, config, lattice, config["optimization_wavelengths_nm"],
            config["optimization_order"], config["optimization_grid"], device, gradient=True)
        gradient = result["gradient"]
        if not isinstance(gradient, torch.Tensor) or gradient.shape != logits.shape:
            raise ValueError("Evaluator returned an invalid logits-gradient shape.")
        if not math.isfinite(result["objective"]) or not bool(torch.isfinite(gradient).all()):
            raise NonphysicalResult("Nonfinite objective or logits gradient.")
        return dict(objective=float(result["objective"]),
                    gradient=gradient.detach().cpu().clone(), seconds=float(result["seconds"]))

    def record_evaluation(evaluation, learning_rate, retries):
        value = evaluation["objective"]
        substantial = value < state["best_loss"] - config["min_improvement"]
        if value < state["best_loss"]:
            state["best_loss"] = value
            state["best_logits"] = logits.detach().cpu().tolist()
            state["best_step"] = state["next_step"]
        state["stale"] = 0 if substantial else state["stale"] + 1
        state["current_evaluation"] = evaluation
        state["history"].append(dict(
            step=state["next_step"], objective=value, best_objective=state["best_loss"],
            gradient_norm=float(torch.linalg.vector_norm(evaluation["gradient"])),
            learning_rate=learning_rate, evaluation_seconds=evaluation["seconds"],
            accepted_update_retries=retries))

    def outcome(status):
        return dict(
            status=status, trial_number=request["trial_number"], params=parameters,
            best_loss=state["best_loss"], best_logits=state["best_logits"],
            best_step=state["best_step"],
            radii_nm=radii_values(state["best_logits"], config),
            completed_steps=state["next_step"], next_step=state["next_step"],
            successful_gradient_evaluations=len(state["history"]),
            recovery_attempts=len(state["recovery_events"]),
            numerical_early_stop=state["termination_reason"] == "nonphysical_retry_exhausted",
            termination_reason=state["termination_reason"], recovery_policy=POLICY)

    def finish(status, reason):
        state["finished_status"] = status
        state["termination_reason"] = reason
        save()
        result = outcome(status)
        write_json(folder / "best.json", result)
        write_csv(folder / "best_profile.csv", profile_rows(state["best_logits"], config))
        print(
            f"trial {request['trial_number']} {reason}: accepted updates={state['next_step']}; "
            f"best mean R={state['best_loss']:.9g}", flush=True)
        return result

    def report_and_prune():
        step = state["next_step"]
        if trial is None or step % config["pruner_interval_steps"] or step <= state["last_reported_step"]:
            return False
        trial.report(state["best_loss"], step)
        state["last_reported_step"] = step
        should_prune = trial.should_prune()
        save()
        return should_prune

    def timed_out():
        return request.get("deadline") is not None and time.time() >= request["deadline"]

    # Identifies an initial point even if its first evaluation is interrupted
    # or runs out of memory. current_evaluation=None means it is still pending.
    save()
    if state["finished_status"] is not None:
        return finish(state["finished_status"], state["termination_reason"])
    if state["current_evaluation"] is None:
        if timed_out():
            return dict(status="paused", reason="time_budget", next_step=0)
        try:
            first_evaluation = evaluate()
        except NonphysicalResult as error:
            state["recovery_events"].append(dict(
                stage="initial_evaluation", step=0, error=str(error),
                proposed_logits=logits.detach().cpu().tolist()))
            save()
            raise
        record_evaluation(first_evaluation, optimizer.param_groups[0]["lr"], 0)
        save()
        print(
            f"trial {request['trial_number']} initial state validated: "
            f"mean R={first_evaluation['objective']:.9g}", flush=True)
    if report_and_prune():
        return finish("pruned", "optuna_pruned")

    while state["next_step"] < config["adam_steps"]:
        if timed_out():
            save()
            return dict(status="paused", reason="time_budget", next_step=state["next_step"])

        # Restore the source point together with both Adam moments, the step
        # count and scheduler state when a proposed update is rejected.
        previous_logits = logits.detach().cpu().clone()
        previous_optimizer = cpu_tree(optimizer.state_dict())
        previous_scheduler = copy.deepcopy(scheduler.state_dict())
        learning_rates = [group["lr"] for group in optimizer.param_groups]
        source_value = state["current_evaluation"]["objective"]
        optimizer.zero_grad(set_to_none=True)
        logits.grad = state["current_evaluation"]["gradient"].to(device)
        optimizer.step()
        with torch.no_grad():
            logits.clamp_(-config["logit_limit"], config["logit_limit"])

        try:
            candidate = evaluate()
        except NonphysicalResult as error:
            proposed_logits = logits.detach().cpu().tolist()
            with torch.no_grad():
                logits.copy_(previous_logits.to(device))
            optimizer.load_state_dict(copy.deepcopy(previous_optimizer))
            scheduler.load_state_dict(copy.deepcopy(previous_scheduler))
            state["recovery_events"].append(dict(
                stage="proposal", step=state["next_step"], retry_index=state["retry_index"],
                learning_rates=learning_rates, error=str(error), proposed_logits=proposed_logits,
                restored_objective=source_value))
            if state["retry_index"] == maximum_retries:
                return finish("completed", "nonphysical_retry_exhausted")
            state["retry_index"] += 1
            for group in optimizer.param_groups:
                group["lr"] *= retry_factor
            save()
            print(
                f"trial {request['trial_number']} update {state['next_step'] + 1}: "
                f"nonphysical proposal; restored validated state; retry "
                f"{state['retry_index']}/{maximum_retries}, lr={optimizer.param_groups[0]['lr']:.9g}",
                flush=True)
            continue

        retries = state["retry_index"]
        state["next_step"] += 1
        state["retry_index"] = 0
        scheduler.step(source_value)
        record_evaluation(candidate, learning_rates[0], retries)
        save()
        if state["next_step"] % 10 == 0 or retries:
            print(
                f"trial {request['trial_number']} update {state['next_step']}: "
                f"mean R={candidate['objective']:.9g}; best={state['best_loss']:.9g}; retries={retries}",
                flush=True)
        if report_and_prune():
            return finish("pruned", "optuna_pruned")
        if state["stale"] >= config["patience"]:
            return finish("completed", "patience")

    return finish("completed", "max_steps")
