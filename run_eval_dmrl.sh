#!/bin/bash

#SBATCH -N 1                    # use 1 node
#SBATCH -n 1                    # 1 task
#SBATCH --cpus-per-task 4       # cpu cores to use
#SBATCH -t 2-00:00:00            # 0 days, 6 hours, 0 minutes, 0 seconds
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

PadLimit=20
FineTune=3
FineTuneSub=$((FineTune-1))

python eval.py --name "BackwardUnifiedContinuousFull_PadLimit${PadLimit}-${FineTune}_test" --config_path "configs/unified_continuous_full.yaml" --load "out/models/BackwardUnifiedContinuousFull_PadLimit${PadLimit}-${FineTune}.pkl" --pad_limit $PadLimit --num_samples 1 --batch_size 80 --test

# python eval.py --name "BackwardUnifiedDiscreteFull_PadLimit${PadLimit}-${FineTune}_test" --config_path "configs/unified_discrete_full.yaml" --load "out/models/BackwardUnifiedDiscreteFull_PadLimit${PadLimit}-${FineTune}.pkl" --pad_limit $PadLimit --num_samples 1 --batch_size 80 --test

#===============================================================================

# python eval.py --name "BackwardUnifiedContinuous_PadLimit${PadLimit}-${FineTune}_test" --config_path "configs/unified_continuous.yaml" --load "out/models/BackwardUnifiedContinuous_PadLimit${PadLimit}-${FineTune}.pkl" --pad_limit $PadLimit --num_samples 1 --batch_size 80 --test

# python eval_RL.py --name "BackwardUnifiedContinuousRLParticle_PadLimit${PadLimit}-${FineTune}_test" --config_path "configs/unified_continuous.yaml" --load "out/models/BackwardUnifiedContinuous_PadLimit${PadLimit}-${FineTune}.pkl" --pad_limit $PadLimit --num_samples 1 --batch_size 80 --test

# python eval.py --name "BackwardUnifiedContinuous_NoPadLimit-2_test" --config_path "configs/unified_continuous.yaml" --load "out/models/BackwardUnifiedContinuous_NoPadLimit-2.pkl" --pad_limit -1 --num_samples 1 --record_attns 1 --batch_size 80 --test

# python eval_RL.py --name "BackwardUnifiedContinuousRL_NoPadLimit-${FineTune}_test" --config_path "configs/unified_continuous.yaml" --load "out/models/BackwardUnifiedContinuous_NoPadLimit-${FineTune}.pkl" --pad_limit -1 --num_samples 1 --batch_size 80 --test


# python eval_RL.py --name "BackwardUnifiedDiscreteRLParticle_PadLimit${PadLimit}-${FineTune}_test" --config_path "configs/unified_discrete.yaml" --load "out/models/BackwardUnifiedDiscrete_PadLimit${PadLimit}-${FineTune}.pkl" --pad_limit $PadLimit --num_samples 1 --batch_size 80 --test


# python eval.py --name "BackwardUnifiedDiscrete_PadLimit${PadLimit}-${FineTune}_test" --config_path "configs/unified_discrete.yaml" --load "out/models/BackwardUnifiedDiscrete_PadLimit${PadLimit}-${FineTune}.pkl" --pad_limit $PadLimit --num_samples 1 --batch_size 80 --test


