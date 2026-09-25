#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=06:00:00
#$ -N gold_motheye

# Submit from the repository root with: qsub -g YOUR_TSUBAME_GROUP studies/gold_motheye/tsubame4_gold_motheye.sh
set -eu
module purge
module load cuda

python3 -c 'import torch; print("PyTorch", torch.__version__, "CUDA", torch.cuda.is_available()); assert torch.cuda.is_available(), "CUDA PyTorch is required"'

script=studies/gold_motheye/converge.py
plotter=studies/gold_motheye/plot_results.py
prefix=studies/gold_motheye/results/gold_motheye_extended
report=${prefix}_convergence.json
figures=studies/gold_motheye/results/figures_extended

# Extend all three axes because the existing 0.005 criterion was not met.
args=(
  --device cuda
  --orders 3,4,5,6,7,8,9,10
  --slices 12,16,24,32,48,64,80,96
  --grids 96,128,192,256,384,512
  --tolerance 0.005
  --max-cycles 3
  --output-prefix "$prefix"
)

set +e
python3 "$script" "${args[@]}"
study_status=$?
set -e
if [ "$study_status" -ne 0 ] && [ "$study_status" -ne 2 ]; then
  exit "$study_status"
fi

# Plot anchor data even if the numerical study remains unresolved.
python3 "$plotter" --report "$report" --output-dir "$figures"

if [ "$study_status" -eq 0 ]; then
  # This reruns the axis selection from the checkpoint, then calculates 5 nm samples.
  python3 "$script" "${args[@]}" \
    --run-final-spectrum --spectrum-wavelengths 400:700:5
  python3 "$plotter" --report "$report" --output-dir "$figures"
else
  echo "Numerical convergence is unresolved; dense spectrum skipped." >&2
fi

exit "$study_status"
