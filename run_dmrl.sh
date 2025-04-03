#!/bin/bash

#SBATCH -N 1                    # use 1 node
#SBATCH -n 1                    # 1 task
#SBATCH --cpus-per-task 4       # cpu cores to use
#SBATCH -t 1-00:00:00           # 1 days, 0 hours, 0 minutes, 0 seconds
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

PadLimit=20
FineTune=0
FineTuneSub=$((FineTune-1))

# if (($FineTune == 0)); then
#     python train.py --name "BackwardDiffusion_PadLimit$PadLimit-${FineTune}" --config_path "configs/pad_limit_spec.yaml" --pad_limit $PadLimit
# else
#     python train.py --name "BackwardDiffusion_PadLimit$PadLimit-${FineTune}" --config_path "configs/pad_limit_spec.yaml" --load "out/models/BackwardDiffusion_PadLimit${PadLimit}-${FineTuneSub}_29.pkl" --pad_limit $PadLimit
# fi

# if (($FineTune == 0)); then
#     python train.py --name "BackwardDiffusionContinuous_PadLimit${PadLimit}-${FineTune}" --config_path "configs/pad_limit_spec_continuous.yaml"  --pad_limit $PadLimit
# else
#     python train.py --name "BackwardDiffusionContinuous_PadLimit${PadLimit}-${FineTune}" --config_path "configs/pad_limit_spec_continuous.yaml" --load "out/models/BackwardDiffusionContinuous_PadLimit${PadLimit}-${FineTuneSub}.pkl" --pad_limit $PadLimit
# fi

# if (($FineTune == 0)); then
#     python train.py --name "BackwardDiffusionUnified_PadLimit${PadLimit}-${FineTune}" --config_path "configs/unified_continuous.yaml"  --pad_limit $PadLimit
# else
#     python train.py --name "BackwardDiffusionUnified_PadLimit${PadLimit}-${FineTune}" --config_path "configs/unified_continuous.yaml" --load "out/models/BackwardDiffusionUnified_PadLimit${PadLimit}-${FineTuneSub}.pkl" --pad_limit $PadLimit
# fi

# if (($FineTune == 0)); then
#     python train.py --name "BackwardDiffusionContGaussian_PadLimit${PadLimit}-${FineTune}" --config_path "configs/pad_limit_spec_continuous_gaussian.yaml"  --pad_limit $PadLimit
# else
#     python train.py --name "BackwardDiffusionContGaussian_PadLimit${PadLimit}-${FineTune}" --config_path "configs/pad_limit_spec_continuous_gaussian.yaml" --load "out/models/BackwardDiffusionContGaussian_PadLimit${PadLimit}-${FineTuneSub}.pkl" --pad_limit $PadLimit
# fi

# if (($FineTune == 0)); then
#     python train.py --name "BackwardDiffusionContinuous_NoPadLimit-${FineTune}" --config_path "configs/no_pad_limit_continuous.yaml"
# else
#     python train.py --name "BackwardDiffusionContinuous_NoPadLimit-${FineTune}" --config_path "configs/no_pad_limit_continuous.yaml" --load "out/models/BackwardDiffusionContinuous_NoPadLimit-${FineTuneSub}.pkl"
# fi

# if (($FineTune == 0)); then
#     python train.py --name "ForwardDiffusionContinuous_NoPadLimit-${FineTune}" --config_path "configs/no_pad_limit_continuous_forward.yaml"
# else
#     python train.py --name "ForwardDiffusionContinuous_NoPadLimit-${FineTune}" --config_path "configs/no_pad_limit_continuous_forward.yaml" --load "out/models/ForwardDiffusionContinuous_NoPadLimit-${FineTuneSub}.pkl"
# fi

# if (($FineTune == 0)); then
#     python train.py --name "BackwardUnifiedContinuous_PadLimit${PadLimit}-${FineTune}" --config_path "configs/unified_continuous.yaml" --pad_limit -1
# else
#     python train.py --name "BackwardUnifiedContinuous_PadLimit${PadLimit}-${FineTune}" --config_path "configs/unified_continuous.yaml" --load "out/models/BackwardUnifiedContinuous_PadLimit${PadLimit}-${FineTuneSub}.pkl" --pad_limit -1
# fi

# if (($FineTune == 0)); then
#     python train.py --name "BackwardUnifiedContinuous_NoPadLimit-${FineTune}" --config_path "configs/unified_continuous.yaml" --pad_limit -1
# else
#     python train.py --name "BackwardUnifiedContinuous_NoPadLimit-${FineTune}" --config_path "configs/unified_continuous.yaml" --load "out/models/BackwardUnifiedContinuous_NoPadLimit-${FineTuneSub}.pkl" --pad_limit -1
# fi

# if (($FineTune == 0)); then
#     python train.py --name "ForwardUnifiedContinuous_NoPadLimit-${FineTune}" --config_path "configs/unified_continuous_forward.yaml" --pad_limit -1
# else
#     python train.py --name "ForwardUnifiedContinuous_NoPadLimit-${FineTune}" --config_path "configs/unified_continuous_forward.yaml" --load "out/models/ForwardUnifiedContinuous_NoPadLimit-${FineTuneSub}.pkl" --pad_limit -1
# fi

# if (($FineTune == 0)); then
#     python train.py --name "BackwardUnifiedDiscrete_NoPadLimit-${FineTune}" --config_path "configs/unified_discrete.yaml" --pad_limit -1
# else
#     python train.py --name "BackwardUnifiedDiscrete_NoPadLimit-${FineTune}" --config_path "configs/unified_discrete.yaml" --load "out/models/BackwardUnifiedDiscrete_NoPadLimit-${FineTuneSub}.pkl" --pad_limit -1
# fi

if (($FineTune == 0)); then
    python train.py --name "BackwardUnifiedContinuousMoE_NoPadLimit-${FineTune}" --config_path "configs/unified_continuous_moe.yaml" --pad_limit -1
else
    python train.py --name "BackwardUnifiedContinuousMoE_NoPadLimit-${FineTune}" --config_path "configs/unified_continuous_moe.yaml" --load "out/models/BackwardUnifiedContinuousMoE_NoPadLimit-${FineTuneSub}.pkl" --pad_limit -1
fi