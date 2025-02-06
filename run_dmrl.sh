#!/bin/bash

#SBATCH -N 1                    # use 1 node
#SBATCH -n 1                    # 1 task
#SBATCH --cpus-per-task 4       # cpu cores to use
#SBATCH -t 0-01:00:00           # 1 days, 0 hours, 0 minutes, 0 seconds
#SBATCH -p gpu_batch            # use the gpu partition
#SBATCH -J RetroDiffusion       # Job name
#SBATCH --mem=32000             # 32000 MB memory (RAM)
#SBATCH --gres=gpu:1            # 1 GPU # can also use --gpus-per-task, -gres=gpu:1, -G etc
#SBATCH -o out/reports/Retro-Diffusion-%A.txt
#SBATCH -e out/reports/Retro-Diffusion-Err-%A.txt

#Code to run main.py

set -x
set -e

# module load cuda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate deepchem

cd ~/Retro-Diffusion

nvidia-smi
# python train.py --name BackwardDiffusion_PadLimit20 --config_path configs/pad_limit_20.yaml
# python train.py --name BackwardDiffusion_PadLimit20_Fine1 --config_path configs/pad_limit_20.yaml --load out/models/BackwardDiffusion_PadLimit20_29.pkl

python train.py --name BackwardDiffusionContinuous_PadLimit20 --config_path configs/pad_limit_20_continuous.yaml
