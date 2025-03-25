#!/bin/bash

#SBATCH -N 1                    # use 1 node
#SBATCH -n 1                    # 1 task
#SBATCH --cpus-per-task 4       # cpu cores to use
#SBATCH -t 0-12:00:00            # 0 days, 6 hours, 0 minutes, 0 seconds
#SBATCH -p gpu_batch            # use the gpu partition
#SBATCH -J RetroDiffusion       # Job name
#SBATCH --mem=32000             # 32000 MB memory (RAM)
#SBATCH --gres=gpu:1            # 1 GPU # can also use --gpus-per-task, -gres=gpu:1, -G etc
#SBATCH -o out/reports/Retro-Diffusion-Eval-%A.txt
#SBATCH -e out/reports/Retro-Diffusion-Eval-Err-%A.txt

#Code to run main.py

#Code to run main.py

set -x
set -e

# module load cuda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate deepchem

cd ~/Retro-Diffusion

PadLimit=40
FineTune=1
FineTuneSub=$((FineTune-1))

# python eval.py --name "BackwardDiffusion_PadLimit$PadLimit-${FineTune}" --config_path "configs/pad_limit_spec.yaml" --load "out/models/BackwardDiffusion_PadLimit${PadLimit}-${FineTune}_29.pkl" --num_samples 1

# python eval.py --name "BackwardDiffusionContinuous_PadLimit${PadLimit}-${FineTune}" --config_path "configs/pad_limit_spec_continuous.yaml" --load "out/models/BackwardDiffusionContinuous_PadLimit${PadLimit}-${FineTune}.pkl" --num_samples 1

# python eval.py --name "BackwardDiffusionContGaussian_PadLimit${PadLimit}-${FineTune}" --config_path "configs/pad_limit_spec_continuous_gaussian.yaml" --load "out/models/BackwardDiffusionContGaussian_PadLimit${PadLimit}-${FineTune}.pkl" --num_samples 1

# python eval.py --name "BackwardDiffusionContinuousAttn_PadLimit${PadLimit}-${FineTune}" --config_path "configs/pad_limit_spec_continuous.yaml" --load "out/models/BackwardDiffusionContinuous_PadLimit${PadLimit}-${FineTune}.pkl" --num_samples 0 --record_attns

python eval.py --name "BackwardDiffusionContinuous_NoPadLimit-${FineTune}" --config_path "configs/no_pad_limit_continuous.yaml" --load "out/models/BackwardDiffusionContinuous_NoPadLimit-${FineTune}.pkl" --num_samples 1