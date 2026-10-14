"""Memory and checkpoint helpers shared by the transition-metal NV runs.

Extracted from the validated Cr NV runner without changing the matrix algebra.
"""
from __future__ import annotations
from dataclasses import fields,is_dataclass
import hashlib,json
from pathlib import Path

LEGACY_LISTS = ("P", "Q", "eps_conv", "mu_conv", "E_eigvec", "H_eigvec")


AUXILIARY_LISTS = ("asr_mappings", "asr_material_tensors", "asr_T_matrices",
                   "asr_Tz_matrices", "asr_condition_numbers", "E_eigvec_uv", "H_eigvec_uv")


AUXILIARY_DICTS = ("_asr_slot_by_layer", "_asr_field_context_by_layer", "_physical_material_by_layer",
                   "_nvm_eps_tensor_by_layer", "_reduced_longitudinal_context_by_layer")


STORAGE_POLICY = "NV-per-layer-D6-release-and-streamed-Redheffer-v1"


def tensor_storage_bytes(items):
    """Unique live tensor storage, excluding allocator cache and autograd graphs."""
    storages = {}
    def visit(value):
        if hasattr(value, "untyped_storage"):
            storage = value.untyped_storage()
            storages[(str(value.device), storage.data_ptr())] = storage.nbytes()
        elif is_dataclass(value) and not isinstance(value, type):
            for field in fields(value):
                visit(getattr(value, field.name))
        elif isinstance(value, dict):
            for entry in value.values():
                visit(entry)
        elif isinstance(value, (tuple, list)):
            for entry in value:
                visit(entry)
    visit(items)
    return sum(storages.values())


def discard_unused_layer_storage(simulation):
    """Keep independent reduced E/H/kz, thickness, ports and diagnostics only.

    Discard full-space operators and NV field-recovery tensors after the NV
    layer has installed independent reduced modes. Preserve all port and
    reduced-cascade data. This is for fixed geometry without fields/gradients.
    """
    if simulation.store_mode_couplings or simulation.field_regions != "none":
        raise RuntimeError("Storage release requires fields disabled.")
    if (getattr(simulation, "group_theory_symmetry", None) != "d6"
            or getattr(simulation, "polarization_reduction", None) != "x"
            or len(simulation._polarized_layers) != simulation.layer_N
            or simulation.group_theory_diagnostics[-1].get("symmetry") != "D6-E1-source-row"
            or simulation.group_theory_diagnostics[-1].get("backend") != "NVM"
            or simulation.layer_records[-1].method != "nvm"):
        raise RuntimeError("Storage release requires every layer to use the NV D6 E1 source row.")
    if any(t.requires_grad for layer in simulation._polarized_layers for t in layer.values()):
        raise RuntimeError("Storage release is only for fixed-geometry, non-gradient calculations.")
    for name in LEGACY_LISTS:
        if len(getattr(simulation, name)) != simulation.layer_N:
            raise RuntimeError(f"Unexpected layer-storage layout: {name}")
    for name in AUXILIARY_LISTS + AUXILIARY_DICTS:
        if not hasattr(simulation, name):
            raise RuntimeError(f"Required auxiliary container is missing: {name}")
    discarded = tensor_storage_bytes([getattr(simulation, name)
                                     for name in LEGACY_LISTS + AUXILIARY_LISTS + AUXILIARY_DICTS])
    for name in LEGACY_LISTS:
        getattr(simulation, name)[-1] = None
    for name in AUXILIARY_LISTS + AUXILIARY_DICTS:
        getattr(simulation, name).clear()
    return discarded


def memory_safe_class(original_class):
    class FluxOnlyRCWA(original_class):
        def add_structured_layer(self, spec):
            selected = super().add_structured_layer(spec)
            released = discard_unused_layer_storage(self)
            self._discarded_auxiliary_storage_bytes = getattr(self, "_discarded_auxiliary_storage_bytes", 0) + released
            retained = tensor_storage_bytes(vars(self))
            self._post_layer_retained_tensor_peak_bytes = max(
                getattr(self, "_post_layer_retained_tensor_peak_bytes", 0), retained)
            return selected

        def _polarized_redheffer_scattering(self, layers, input_interface, output_interface, *, force_full=False):
            # Same reverse-order half-S recurrence as rcwa_ext/reduced.py.
            # Build and consume one layer S instead of retaining all Nz S's.
            if force_full or self.store_mode_couplings or self.smatrix_size != "half":
                raise RuntimeError("Memory-safe NV cascade requires half-S output and no fields.")
            if len(layers) != len(self.thickness):
                raise RuntimeError("Reduced layers and thicknesses disagree.")
            identity = self._eye(layers[0]["electric"].shape[0])
            transmission, reflection = output_interface[:2]
            def prepend(left):
                nonlocal transmission, reflection
                tf_left, rf_left, rb_left, tb_left = left
                reflection_new = rf_left + tb_left @ self._solve(
                    identity - reflection @ rb_left, reflection @ tf_left)
                transmission = transmission @ self._solve(
                    identity - rb_left @ reflection, tf_left)
                reflection = reflection_new
            for index in range(len(layers) - 1, -1, -1):
                layer = layers[index]
                layer_s = self._reduced_layer_smatrix(layer["electric"], layer["magnetic"],
                    layer["kz"], self.thickness[index], self._polarization_reference_v)
                prepend(layer_s)
                del layer_s
            prepend(input_interface)
            zero = 0 * identity
            return [transmission, reflection, zero, zero]
    return FluxOnlyRCWA


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def identity_signature(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()


def case_key(order, wave):
    return f"M={order}|wl={wave:.12g}"


def initialize_device(torch_module, requested):
    device = torch_module.device(requested)
    if device.type == "cuda":
        if not torch_module.cuda.is_available():
            raise ValueError("Requested CUDA device is unavailable.")
        index = device.index if device.index is not None else torch_module.cuda.current_device()
        if not 0 <= index < torch_module.cuda.device_count():
            raise ValueError("Requested CUDA device is unavailable.")
        # CUDA management APIs require an explicit index even when the solver
        # accepts the generic device name "cuda".
        device = torch_module.device("cuda", index)
        torch_module.cuda.set_device(device)
        torch_module.cuda.reset_peak_memory_stats(device)
    return device
