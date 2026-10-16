"""Traceable CUDA/MIG changes without resetting NV experiment checkpoints.

Only CUDA hardware metadata may vary within the same reported GPU model.
Software, physics sources and configuration remain checked. A known legacy
driver can adopt this orchestration-only update while retaining its original
checkpoint signature; both source versions remain recorded in the manifest.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import uuid

from studies.shared.convergence_io import content_hash, signature_for, write_json

DRIVER = "Optimization/dielectric_motheye_n25/run.py"
POLICY_SOURCE = "nv_runtime_policy.py"
LEGACY_DRIVER_SHA256 = "2bc7ef243312b0afa0943ceea2db1b3169a613e579399f431d435548e8e8329a"
HARDWARE_FIELDS = frozenset(("device", "gpu", "gpu_total_bytes"))
POLICY = "same-gpu-model-cuda-hardware-only-v1"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def is_cuda(value):
    return isinstance(value, str) and re.fullmatch(r"cuda(?::[0-9]+)?", value) is not None


def gpu_model(name):
    return re.sub(r"\s+MIG\s+.*$", "", str(name).strip(), flags=re.IGNORECASE)


def check_runtime(previous, current):
    """Return permitted hardware differences; raise on all other differences."""
    if previous is None:
        return {}
    if not isinstance(previous, dict) or not isinstance(current, dict):
        raise RuntimeError("Missing or invalid runtime metadata.")
    differences = {key: {"previous": previous.get(key), "current": current.get(key)}
                   for key in sorted(set(previous) | set(current))
                   if previous.get(key) != current.get(key)}
    if not differences:
        return {}
    software = set(differences)-HARDWARE_FIELDS
    if software:
        details = "; ".join(f"{key}: {previous.get(key)!r} -> {current.get(key)!r}" for key in sorted(software))
        raise RuntimeError("Runtime software differs; GPU-only changes are allowed. "+details)
    if not (is_cuda(previous.get("device")) and is_cuda(current.get("device"))):
        raise RuntimeError("CPU/CUDA switching is not an allowed GPU configuration change.")
    if not previous.get("gpu") or not current.get("gpu") or gpu_model(previous["gpu"]) != gpu_model(current["gpu"]):
        raise RuntimeError(f"GPU model differs: {previous.get('gpu')!r} -> {current.get('gpu')!r}.")
    if any(not isinstance(env.get("gpu_total_bytes"), int) or env["gpu_total_bytes"] <= 0
           for env in (previous, current)):
        raise RuntimeError("GPU memory metadata must be positive integer byte counts.")
    return differences


def execution_sources(existing):
    result = dict(existing)
    result[POLICY_SOURCE] = content_hash(Path(__file__))
    return result


def _identity(manifest, sources=None, device=None):
    return {"version": manifest["version"], "config": manifest["config"],
            "source_sha256": manifest["source_sha256"] if sources is None else sources,
            "requested_device": manifest["requested_device"] if device is None else device}


def reconcile_manifest(previous, identity):
    """Keep the original checkpoint signature, accepting only a known upgrade."""
    if previous is None:
        return {**copy.deepcopy(identity), "signature": signature_for(identity),
                "runtime_environment": None, "runtime_policy": POLICY}, None
    if signature_for(_identity(previous)) != previous["signature"]:
        raise ValueError("Saved manifest identity/signature is inconsistent.")
    if any(previous[key] != identity[key] for key in ("version", "config")):
        raise ValueError("Configuration differs from the saved experiment; use a new output directory.")
    expected_sources = previous.get("active_source_sha256", previous["source_sha256"])
    expected_device = previous.get("active_requested_device", previous["requested_device"])
    if "active_source_sha256" in previous and previous.get("active_execution_signature") != signature_for(
            _identity(previous, expected_sources, expected_device)):
        raise ValueError("Saved active-source identity is inconsistent.")
    if expected_device != identity["requested_device"] and not (
            is_cuda(expected_device) and is_cuda(identity["requested_device"])):
        raise ValueError("Requested CPU/CUDA device type differs from this experiment.")
    current_sources = identity["source_sha256"]
    changed = {key for key in set(expected_sources) | set(current_sources)
               if expected_sources.get(key) != current_sources.get(key)}
    migration = None
    updated = copy.deepcopy(previous)
    if changed:
        if not (changed == {DRIVER, POLICY_SOURCE} and POLICY_SOURCE not in expected_sources
                and expected_sources.get(DRIVER) == LEGACY_DRIVER_SHA256
                and DRIVER in current_sources and POLICY_SOURCE in current_sources):
            raise ValueError("Source differs beyond the known GPU-policy upgrade: "+", ".join(sorted(changed)))
        migration = {"utc": utc_now(), "kind": "known-gpu-policy-upgrade-v1",
            "previous_source_sha256": expected_sources, "current_source_sha256": current_sources,
            "checkpoint_signature_preserved": previous["signature"]}
        updated.setdefault("source_upgrade_history", []).append(migration)
    if expected_device != identity["requested_device"]:
        updated.setdefault("device_selection_history", []).append({"utc": utc_now(),
            "previous": expected_device, "current": identity["requested_device"]})
    updated.update(active_source_sha256=copy.deepcopy(current_sources),
                   active_requested_device=identity["requested_device"],
                   active_execution_signature=signature_for(identity), runtime_policy=POLICY)
    return updated, migration


def record_runtime(manifest, current):
    previous = manifest.get("runtime_environment")
    differences = check_runtime(previous, current)
    if "initial_runtime_environment" not in manifest:
        manifest["initial_runtime_environment"] = copy.deepcopy(previous if previous is not None else current)
    if previous != current:
        manifest.setdefault("runtime_history", []).append({"utc": utc_now(),
            "kind": "initial-runtime" if previous is None else "gpu-configuration-change",
            "policy": POLICY, "previous": copy.deepcopy(previous), "current": copy.deepcopy(current),
            "differences": differences,
            "note": "Same numerical model; hardware changes can alter timings and floating-point rounding."})
    manifest["runtime_environment"] = copy.deepcopy(current)


def persist_manifest(path, original, updated):
    """Called while holding the controller lock; preserve an exact prior copy."""
    actual = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    if actual != original:
        raise RuntimeError("Manifest changed before the controller lock was acquired; retry.")
    if original is not None and actual != updated:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup = path.with_name(f"manifest.backup.{stamp}.{uuid.uuid4().hex[:8]}.json")
        with backup.open("xb") as handle:
            handle.write(path.read_bytes())
    write_json(path, updated)
