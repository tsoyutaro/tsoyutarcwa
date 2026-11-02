"""Modular differentiable RCWA extensions for torcwa."""

from .asr import CustomRCWA_ASR_FR
from .asr_maps import ASRMapping, CircleASRMapping
from .auto import AutoRCWA, install_as_torcwa_rcwa
from .config import (
    ASROptions,
    Circle,
    GroupTheoryOptions,
    Homogeneous,
    Lattice,
    LayerRecord,
    LayerSpec,
    Material,
    NVMOptions,
    OutputSpec,
    Raster,
    Rectangle,
    Square,
    UnsupportedCombinationError,
)
from .nvm import CustomRCWA_NVM

ASRRCWA = CustomRCWA_ASR_FR
NVMRCWA = CustomRCWA_NVM
rcwa = AutoRCWA

__all__ = [
    "ASROptions",
    "ASRMapping",
    "ASRRCWA",
    "AutoRCWA",
    "Circle",
    "CircleASRMapping",
    "CustomRCWA_ASR_FR",
    "CustomRCWA_NVM",
    "GroupTheoryOptions",
    "Homogeneous",
    "Lattice",
    "LayerRecord",
    "LayerSpec",
    "Material",
    "NVMOptions",
    "NVMRCWA",
    "OutputSpec",
    "Raster",
    "Rectangle",
    "Square",
    "UnsupportedCombinationError",
    "install_as_torcwa_rcwa",
    "rcwa",
]
