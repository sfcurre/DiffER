#!/bin/bash
#SBATCH --job-name=Retro-Diffusion
#SBATCH --time=0:39:59
#SBATCH --output="out/reports/Retro-Diffusion-Eval-%j.out"
#SBATCH --account=PAS2170
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=BEGIN,END,FAIL

#Code to run main.py

set -x
set -e

module load cuda
source /usr/local/python/3.6-conda5.2/etc/profile.d/conda.sh
conda activate deepchem

cd ~/Retro-Diffusion

python eval.py --data_path data/USPTO_50K_PtoR_aug20 --task forward_prediction --name ForwardDiffusionEval --load out/models/ForwardDiffusionFinetune_34.pkl

