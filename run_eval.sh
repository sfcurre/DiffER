#!/bin/bash
#SBATCH --job-name=Retro-Diffusion
#SBATCH --time=23:59:59
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

python eval.py --data_path data/USPTO_50K_PtoR_aug20 --task backward_prediction --name BackwardDiffusionEvalT200RsmilesVariantLCE15PrePost --num_timesteps 200 --beta_schedule cosine --batch_size 64 \
               --load out/models/BackwardDiffusion_T200_Rsmiles_VariantLCE15PrePost_Fine2_29.pkl --batch_limit 20 --num_samples 20

