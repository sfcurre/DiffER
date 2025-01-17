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

python main.py --data_path data/USPTO_50K_PtoR_aug20 --task backward_prediction --epochs 30 --name BackwardDiffusion_T200_Rsmiles_VariantLDiff15 --pad_limit 15 --length_loss cross_entropy \
               --lr 0.0001 --aug_prob 0.0 --loss_terms mse,vb --num_timesteps 200 --beta_schedule cosine #--load out/models/BackwardDiffusion_T200_Rsmiles_VariantLCE15PrePost_Fine1_29.pkl

#python main.py --data_path data/USPTO_50K_PtoR_aug20 --task backward_prediction --epochs 30 --name BackwardDiffusion_VB+MSE+LL_CosineBeta_Tsampling_LR0001_T200_Rsmiles_LenInPlus50 \
#               --lr 0.0001 --aug_prob 0.0 --loss_terms mse,vb --num_timesteps 200 --beta_schedule cosine #--load out/models/BackwardDiffusion_VB+MSE+LL_CosineBeta_Tsampling_LR0001_T200_Rsmiles_LenInPlus50_29.pkl

#python main.py --data_path data/USPTO_50K_PtoR_aug20 --task backward_prediction --epochs 30 --name BackwardDiffusion_VB+MSE+LL_CosineBeta_Tsampling_LR0001_T200_Rsmiles_VariantL_Fine1 \
#               --lr 0.0001 --aug_prob 0.0 --loss_terms mse,vb --num_timesteps 200 --beta_schedule cosine --load out/models/BackwardDiffusion_VB+MSE+LL_CosineBeta_Tsampling_LR0001_T200_Rsmiles_VariantL_Fine0_29.pkl

#python main.py --data_path data/USPTO_50K_PtoR_aug20 --task backward_prediction --epochs 30 --name BackwardDiffusion_VB+MSE+LL_CosineBeta_Tsampling_LR0001_T200_Rsmiles_FullPadVariant_Fine0 \
#               --lr 0.0001 --aug_prob 0.0 --loss_terms mse,vb --num_timesteps 200 --beta_schedule cosine --load out/models/BackwardDiffusion_VB+MSE+LL_CosineBeta_Tsampling_LR0001_T200_Rsmiles_FullPadVariant_29.pkl

#python main.py --data_path data/USPTO_50K_PtoR_aug20 --task backward_prediction --epochs 30 --name BackwardDiffusion_VB+MSE+LL_CosineBeta_Tsampling_LR0001_T200_Rsmiles_FullPad_Fine0 \
#               --lr 0.0001 --aug_prob 0.0 --loss_terms mse,vb --num_timesteps 200 --beta_schedule cosine --load out/models/BackwardDiffusion_VB+MSE+LL_CosineBeta_Tsampling_LR0001_T200_Rsmiles_FullPad_20.pkl

#python main.py --data_path data/USPTO_50K_PtoR_aug20 --task backward_prediction --epochs 30 --name BackwardDiffusion_VB+MSE+LL_CosineBeta_Tsampling_LR0001_T200 \
#               --lr 0.0001 --loss_terms mse,vb --length_loss weighted_sum --num_timesteps 200 --beta_schedule cosine --load out/models/BackwardDiffusion_VB+MSE+LLV_CosineBeta_Tsampling_LR0001_T200_29.pkl

#python main.py --data_path data/USPTO_50K_PtoR_aug20 --task backward_prediction --epochs 30 --name BackwardDiffusion_VB+MSE_CosineBeta_Tsampling_LR0001_T200_TrueL \
#               --lr 0.0001 --loss_terms mse,vb --num_timesteps 200 --beta_schedule cosine #--load out/models/BackwardDiffusion_VB+MSE+LLV_CosineBeta_Tsampling_LR0001_T200_TrueL_29.pkl

#python main.py --data_path data/USPTO_50K_PtoR_aug20 --task backward_prediction --epochs 30 --name BackwardDiffusion_VB+MSE_CosineBeta_Tsampling_LR0001_T500_Fine2 \
#               --lr 0.0001 --loss_terms mse,vb --num_timesteps 500 --beta_schedule cosine --load out/models/BackwardDiffusion_VB+MSE_CosineBeta_Tsampling_LR0001_T500_Fine1_29.pkl

####################################

#python main.py --data_path data/USPTO_50K_PtoR_aug20 --task backward_prediction --epochs 50 --name BackwardDiffusion_VB+MSE_CosineBeta_Tsampling_LR0001_T100_Fine0 \
#               --lr 0.0001 --loss_terms mse,vb --num_timesteps 100 --beta_schedule cosine --load out/models/BackwardDiffusion_VB+MSE_CosineBeta_Tsampling_LR0001_T100_29.pkl

#python main.py --data_path data/USPTO_50K_PtoR_aug20 --task backward_prediction --epochs 30 --name BackwardDiffusion_VB+MSE_CosineBeta_Tsampling_LR0001_T1000_Fine2 \
#               --lr 0.0001 --loss_terms mse,vb --num_timesteps 1000 --beta_schedule cosine --load out/models/BackwardDiffusion_VB+MSE_CosineBeta_Tsampling_LR0001_T1000_Fine1_29.pkl

#python main.py --data_path data/USPTO_50K_PtoR_aug20 --task forward_prediction --epochs 30 --name ForwardDiffusion_VB+MSE_CosineBeta_Tsampling_LR0001_T200_Fine2 \
#               --lr 0.0001 --loss_terms mse,vb --num_timesteps 200 --beta_schedule cosine --load out/models/ForwardDiffusion_VB+MSE_CosineBeta_Tsampling_LR0001_T200_Fine1_29.pkl

#python main.py --data_path data/USPTO_50K_PtoR_aug20 --task forward_prediction --epochs 30 --name ForwardDiffusion_VB+MSE_CosineBeta_Tsampling_LR0001_T1000_Fine0 \
#               --lr 0.0001 --loss_terms mse,vb --num_timesteps 1000 --beta_schedule cosine --load out/models/ForwardDiffusion_VB+MSE_CosineBeta_Tsampling_LR0001_T1000_29.pkl

#python main.py --data_path data/USPTO_50K_PtoR_aug20 --task forward_prediction --epochs 30 --name ForwardDiffusion_VB+MSE_CosineBeta_Tsampling_LR0001_T1000_Fine2 \
#               --lr 0.0001 --loss_terms mse,vb --num_timesteps 1000 --beta_schedule cosine --load out/models/ForwardDiffusion_VB+MSE_CosineBeta_Tsampling_LR0001_T1000_Fine1_29.pkl

#python main.py --data_path data/USPTO_50K_PtoR_aug20 --task forward_prediction --epochs 30 --name ForwardDiffusion_VB+MSE_CosineBeta_Tsampling_Fine1 \
#               --lr 0.0001 --loss_terms mse,vb --num_timesteps 2000 --beta_schedule cosine --load out/models/ForwardDiffusion_VB+MSE_CosineBeta_Tsampling_Fine0_29.pkl

#python main.py --data_path data/USPTO_50K_PtoR_aug20 --task forward_prediction --epochs 30 --name ForwardDiffusion_VB+MSE_CosineBeta_Tsampling_Fine4 \
#               --lr 0.0001 --loss_terms mse,vb --num_timesteps 2000 --beta_schedule cosine --load out/models/ForwardDiffusion_VB+MSE_CosineBeta_Tsampling_Fine3_29.pkl

#python main.py --data_path data/USPTO_50K_PtoR_aug20 --task backward_prediction --epochs 30 --name BackwardDiffusion0001Deeper --lr 0.0001

#python main.py --data_path data/USPTO_50K_PtoR_aug20 --task forward_prediction --epochs 10 --name ForwardDiffusionFT0001 --lr 0.0001 --load out/models/ForwardDiffusion001_9.pkl

#python main.py --data_path data/USPTO_50K_PtoR_aug20 --task forward_prediction --epochs 10 --name ForwardDiffusionFT00001 --lr 0.00001 --load out/models/ForwardDiffusionFT0001_9.pkl

