"""Adam with transactional physical backtracking; checkpoints contain valid states."""
from __future__ import annotations

import copy
import math
import time
from pathlib import Path

from .common import torch_save, write_csv, write_json
from .geometry import initial_logits, radii_values, profile_rows
from .physics import objective, NonphysicalResult

ENGINE_VERSION = 'physical-backtracking-v4'


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
    c, lattice, params = request['config'], request['lattice'], request['params']
    factor = c.get('physical_backtracking_factor', .5)
    retries = c.get('physical_backtracking_max_retries', 12)
    min_lr = c.get('physical_backtracking_min_lr', 1e-6)
    if (not math.isfinite(factor) or not 0 < factor < 1 or
            isinstance(retries, bool) or not isinstance(retries, int) or retries < 0 or
            not math.isfinite(min_lr) or not 0 < min_lr <= params['lr']):
        raise ValueError('Invalid physical backtracking settings.')
    folder = Path(request['trial_folder'])
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/'adam_checkpoint.pt'
    top = params['init_top']
    bottom = top+params['delta_ratio']*(c['radius_max_nm']-top)
    bottom = min(c['radius_max_nm']-1e-7, max(top, bottom))
    logits = torch.nn.Parameter(torch.tensor(initial_logits(top, bottom, c), dtype=torch.float64, device=device))
    optimizer = torch.optim.Adam([logits], lr=params['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=c['scheduler_factor'],
                    patience=c['scheduler_patience'], min_lr=params['lr']*.001)
    history, best_loss, best_logits, best_step, stale, next_step = [], math.inf, None, None, 0, 0
    cached_result = None
    rejections = []
    stats = dict(evaluation_calls=0, valid_gradient_evaluations=0, rejected_proposals=0,
                 total_evaluation_seconds=0.)
    if path.exists():
        state = torch.load(path, map_location='cpu', weights_only=True)
        if (state['signature'] != request['signature'] or state['params'] != params or
                state.get('engine_version') != ENGINE_VERSION):
            raise ValueError('Checkpoint identity/engine differs; use a new experiment directory.')
        with torch.no_grad():
            logits.copy_(state['logits'].to(device))
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        history, best_loss = state['history'], state['best_loss']
        best_logits, best_step, stale, next_step = state['best_logits'], state['best_step'], state['stale'], state['next_step']
        cached_result, rejections, stats = state['cached_valid_evaluation'], state['rejections'], state['stats']
        if next_step != len(history) or (next_step > 0 and cached_result is None):
            raise ValueError('Inconsistent accepted-state checkpoint.')

    def save():
        torch_save(path, dict(engine_version=ENGINE_VERSION, signature=request['signature'], params=params,
            logits=logits.detach().cpu(), optimizer=cpu_tree(optimizer.state_dict()),
            scheduler=cpu_tree(scheduler.state_dict()), history=history, best_loss=best_loss,
            best_logits=best_logits, best_step=best_step, stale=stale, next_step=next_step,
            current_state_validated=cached_result is not None, cached_valid_evaluation=cached_result,
            rejections=rejections, stats=stats))
        write_csv(folder/'history.csv', history)
        write_csv(folder/'backtracking.csv', rejections)

    def evaluate():
        if torch.device(device).type == 'cuda':
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        stats['evaluation_calls'] += 1
        try:
            result = evaluator(logits, c, lattice, c['optimization_wavelengths_nm'], c['optimization_order'],
                               c['optimization_grid'], device, gradient=True)
            gradient = result['gradient']
            if (not math.isfinite(result['objective']) or gradient is None or
                    gradient.shape != logits.shape or not bool(torch.isfinite(gradient).all())):
                raise NonphysicalResult('Nonfinite objective or logits gradient.')
            stats['valid_gradient_evaluations'] += 1
            return cpu_tree(result)
        finally:
            if torch.device(device).type == 'cuda':
                torch.cuda.synchronize(device)
            stats['total_evaluation_seconds'] += time.perf_counter()-started

    def paused():
        return request.get('deadline') is not None and time.time() >= request['deadline']

    def finish(status, reason, error=None):
        save()
        result = dict(status=status, engine_version=ENGINE_VERSION, termination_reason=reason,
            trial_number=request['trial_number'], params=params, best_loss=best_loss,
            best_logits=best_logits, best_step=best_step, radii_nm=radii_values(best_logits, c),
            completed_steps=next_step, rejected_proposals=stats['rejected_proposals'], stats=stats)
        if error is not None:
            result['error'] = error
        write_json(folder/'best.json', result)
        write_csv(folder/'best_profile.csv', profile_rows(best_logits, c))
        return result

    # Before initial validation only, this is a signed starting state, not an accepted candidate.
    save()
    if paused():
        return dict(status='paused', reason='time_budget', next_step=next_step)
    if cached_result is None:
        try:
            cached_result = evaluate()
        except BaseException:
            save()
            raise  # An invalid initial state cannot be repaired by shrinking an Adam update.
        best_loss, best_logits, best_step = cached_result['objective'], logits.detach().cpu().tolist(), 0
        save()

    for step in range(next_step, c['adam_steps']):
        if paused():
            save()
            return dict(status='paused', reason='time_budget', next_step=next_step)
        base_logits = logits.detach().clone()
        base_optimizer = cpu_tree(optimizer.state_dict())
        base_scheduler = cpu_tree(scheduler.state_dict())
        base_result = cached_result
        attempt_lr = float(optimizer.param_groups[0]['lr'])

        def restore():
            with torch.no_grad():
                logits.copy_(base_logits)
            optimizer.load_state_dict(cpu_tree(base_optimizer))
            scheduler.load_state_dict(cpu_tree(base_scheduler))
            optimizer.zero_grad(set_to_none=True)

        rejected_here = sum(e['accepted_steps'] == next_step for e in rejections)
        if rejected_here > retries:
            return finish('backtracking_exhausted', 'no_valid_step_within_retry_limits',
                          'Physical backtracking exhausted; valid best shape retained in trial best.json.')
        for attempt in range(rejected_here, retries+1):
            restore()
            for group in optimizer.param_groups:
                group['lr'] = attempt_lr
            optimizer.zero_grad(set_to_none=True)
            logits.grad = base_result['gradient'].to(device).clone()
            optimizer.step()
            with torch.no_grad():
                logits.clamp_(-c['logit_limit'], c['logit_limit'])
            try:
                if not bool(torch.isfinite(logits).all()):
                    raise NonphysicalResult('Nonfinite proposed logits.')
                candidate = evaluate()
            except NonphysicalResult as error:
                candidate_logits = logits.detach().cpu().tolist() if bool(torch.isfinite(logits).all()) else None
                restore()
                rejected_here += 1
                stats['rejected_proposals'] += 1
                event = dict(accepted_steps=next_step, attempt=attempt+1, attempted_lr=attempt_lr,
                             error=str(error), candidate_logits=candidate_logits)
                rejections.append(event)
                print(f"REJECT update {next_step+1} attempt {attempt+1}: lr={attempt_lr:.9g}; {error}", flush=True)
                smaller = max(min_lr, attempt_lr*factor)
                exhausted = attempt >= retries or smaller >= attempt_lr
                for group in optimizer.param_groups:
                    group['lr'] = attempt_lr if exhausted else smaller
                save()  # Save the valid base and rolled-back Adam state, never the rejected candidate.
                if exhausted:
                    return finish('backtracking_exhausted', 'no_valid_step_within_retry_limits',
                                  'Physical backtracking exhausted; valid best shape retained in trial best.json.')
                attempt_lr = smaller
                if paused():
                    return dict(status='paused', reason='time_budget', next_step=next_step)
                continue
            except BaseException:
                restore()
                for group in optimizer.param_groups:
                    group['lr'] = attempt_lr
                save()
                raise  # OOM, interruption, and coding errors are not disguised as physical rejections.
            break

        # Commit only a physically valid candidate with a finite logits gradient.
        value = candidate['objective']
        substantial = value < best_loss-c['min_improvement']
        if value < best_loss:
            best_loss, best_logits, best_step = value, logits.detach().cpu().tolist(), step+1
        stale = 0 if substantial else stale+1
        scheduler.step(base_result['objective'])
        cached_result = candidate
        next_step = step+1
        history.append(dict(step=next_step, objective=value, best_objective=best_loss,
            objective_before_update=base_result['objective'],
            gradient_norm=float(torch.linalg.vector_norm(candidate['gradient'])),
            learning_rate=attempt_lr, evaluation_seconds=candidate['seconds'],
            backtracking_rejections=rejected_here))
        save()
        if step % 10 == 0 or rejected_here:
            print(f"trial {request['trial_number']} accepted step {next_step}: mean R={value:.9g}; "
                  f"best={best_loss:.9g}; lr={attempt_lr:.9g}; rejected={rejected_here}", flush=True)
        if trial is not None and step % c['pruner_interval_steps'] == 0:
            trial.report(best_loss, step)
            if trial.should_prune():
                return finish('pruned', 'optuna_pruner')
        if stale >= c['patience']:
            return finish('completed', 'patience')
    # The final state has already passed the full gradient-enabled production evaluation.
    return finish('completed', 'adam_steps')
