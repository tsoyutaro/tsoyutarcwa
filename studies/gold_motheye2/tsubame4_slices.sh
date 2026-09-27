#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=06:00:00
#$ -N gold_nz

# Submit from the project root containing rcwa_solver_auto.py.
# qsub -g YOUR_TSUBAME_GROUP studies/gold_motheye2/tsubame4_slices.sh
# qsub -g YOUR_TSUBAME_GROUP -v SWEEP_ORDER=18 studies/gold_motheye2/tsubame4_slices.sh
set -eu
module purge
module load cuda

python3 -c 'import torch; assert torch.cuda.is_available(), "CUDA PyTorch is required"; print("PyTorch", torch.__version__)'
order=${SWEEP_ORDER:-16}
python3 studies/gold_motheye2/converge_slices.py --device cuda --order "$order"
