"""Select SciPy/CPU or the optional PyTorch/CUDA backend explicitly."""

from __future__ import annotations

import importlib
import re


def resolve_execution(device: str = "auto", backend: str = "auto") -> dict:
    """Auto selects by CUDA availability, not by a performance benchmark.

    Explicit CUDA never falls back to CPU. ``backend='torch', device='cpu'``
    runs the very same tensor kernels used on CUDA, for parity validation.
    CUDA indices are local to CUDA_VISIBLE_DEVICES.
    """
    device, backend = str(device).lower(), str(backend).lower()
    if device not in {"auto", "cpu", "cuda"} and not re.fullmatch(r"cuda:\d+", device):
        raise ValueError("device must be auto, cpu, cuda, or cuda:<index>")
    if backend not in {"auto", "scipy", "torch"}:
        raise ValueError("backend must be auto, scipy, or torch")
    if backend == "scipy" and device.startswith("cuda"):
        raise ValueError("the scipy backend supports only CPU; use backend='torch' for CUDA")
    result = {"requested_device": device, "requested_backend": backend,
              "device": "cpu", "backend": "scipy", "dtype": "complex128",
              "device_name": "CPU", "torch_version": None, "cuda_version": None,
              "condition_number_method": "LAPACK 1-norm estimate"}
    if backend == "scipy" or (device == "cpu" and backend == "auto"):
        result["selection_reason"] = "explicit CPU/SciPy selection"
        return result
    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:
        if device.startswith("cuda") or backend == "torch":
            raise RuntimeError("PyTorch is required for this backend. Install a CUDA-enabled "
                               "build for GPU execution: https://pytorch.org/get-started/locally/") from exc
        result["selection_reason"] = "PyTorch is not installed; auto selected CPU/SciPy"
        return result
    result.update(torch_version=torch.__version__, cuda_version=torch.version.cuda)
    available = torch.cuda.is_available() if device != "cpu" else False
    if device.startswith("cuda") and not available:
        raise RuntimeError(f"CUDA was requested but is unavailable (PyTorch {torch.__version__}, "
                           f"CUDA build {torch.version.cuda}). Check the CUDA-enabled PyTorch "
                           "installation and GPU/driver visibility, or select --device cpu.")
    if available and device != "cpu":
        index = int(device.split(":")[1]) if ":" in device else torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device index {index} is out of range; "
                               f"visible device count is {torch.cuda.device_count()}")
        selected = f"cuda:{index}"
        # Verify runtime initialization before any output CSV is opened.
        try:
            torch.empty(0, dtype=torch.complex128, device=selected)
            name = torch.cuda.get_device_name(index)
        except RuntimeError as exc:
            raise RuntimeError(f"Cannot initialize {selected}: {exc}") from exc
        result.update(device=selected, backend="torch", device_name=name,
                      selection_reason="CUDA is available" if device == "auto" else "explicit CUDA selection")
    elif backend == "torch":
        result.update(backend="torch", selection_reason="PyTorch tensor backend on CPU")
    else:
        result["selection_reason"] = "CUDA is unavailable; auto selected CPU/SciPy"
    if result["backend"] == "torch":
        result["condition_number_method"] = "exact 1-norm from LU inverse (PyTorch)"
    return result
