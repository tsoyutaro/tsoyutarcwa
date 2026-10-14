#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# Activate your existing CUDA-enabled PyTorch / torcwa environment first.
python run_cr_nv_optuna_adam.py --device cuda --timeout-hours 0 "$@"
