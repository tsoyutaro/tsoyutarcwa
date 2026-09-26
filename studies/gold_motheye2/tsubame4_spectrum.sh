#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=12:00:00
#$ -N gold_spec

# Submit from the project root containing rcwa_solver_auto.py.
# qsub -g YOUR_TSUBAME_GROUP studies/gold_motheye2/tsubame4_spectrum.sh
# qsub -g YOUR_TSUBAME_GROUP -v SPECTRUM_ORDER=20 studies/gold_motheye2/tsubame4_spectrum.sh
set -eu
module purge
module load cuda

python3 -c 'import torch; assert torch.cuda.is_available(), "CUDA PyTorch is required"; print("PyTorch", torch.__version__)'
order=${SPECTRUM_ORDER:-16}
python3 studies/gold_motheye2/spectrum.py --device cuda --order "$order"
