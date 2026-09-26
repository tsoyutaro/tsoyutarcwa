#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=12:00:00
#$ -N gold_order

# Submit from the project root (the directory containing rcwa_solver_auto.py):
# qsub -g YOUR_TSUBAME_GROUP studies/gold_motheye2/tsubame4_order.sh
set -eu
module purge
module load cuda

python3 -c 'import torch; assert torch.cuda.is_available(), "CUDA PyTorch is required"; print("PyTorch", torch.__version__)'
python3 studies/gold_motheye2/converge.py --device cuda
