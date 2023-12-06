#!/bin/bash
#SBATCH --job-name=Retro-Diffusion
#SBATCH --time=47:59:59
#SBATCH --output="out/reports/Retro-Diffusion-%j.out"
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

python main.py --data_path data/USPTO_50K_PtoR_aug20 --task forward_prediction --epochs 30 --name ForwardDiffusion_VB+MSE_CosineBeta \
               --lr 0.0001 --loss_terms mse,vb --num_timesteps 2000 --beta_schedule cosine

#python main.py --data_path data/USPTO_50K_PtoR_aug20 --task backward_prediction --epochs 30 --name BackwardDiffusion0001Deeper --lr 0.0001

#python main.py --data_path data/USPTO_50K_PtoR_aug20 --task forward_prediction --epochs 10 --name ForwardDiffusionFT0001 --lr 0.0001 --load out/models/ForwardDiffusion001_9.pkl

#python main.py --data_path data/USPTO_50K_PtoR_aug20 --task forward_prediction --epochs 10 --name ForwardDiffusionFT00001 --lr 0.00001 --load out/models/ForwardDiffusionFT0001_9.pkl

