"""Vallius and Honkanen (2002): FMM with adaptive spatial resolution."""

from .solver import LayerSpec, PreparedStack, compute_spectrum, compute_wavelength
from .devices import resolve_execution

__all__ = ["LayerSpec", "PreparedStack", "compute_spectrum", "compute_wavelength", "resolve_execution"]
