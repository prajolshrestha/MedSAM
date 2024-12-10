#!/bin/bash -l                 
#
#SBATCH --gres=gpu:a100:1 -p a100
#SBATCH --time=10:00:00
                                 
#unset SLURM_EXPORT_ENV             # enable export of environment from this script to srun
            
module load python/3.8-anaconda cuda/11.8.0 cudnn/8.8.0.121-11.8

cd ${HOME}/medsam_file/MedSAM/finetune_src
conda activate medsam

python train_sam_3.py