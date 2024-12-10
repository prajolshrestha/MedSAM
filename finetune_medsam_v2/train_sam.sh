#!/bin/bash -l                 
#
#SBATCH --gres=gpu:a100:1 -p a100
#SBATCH --time=01:30:00
                                 
#unset SLURM_EXPORT_ENV             # enable export of environment from this script to srun
            
module load python/3.8-anaconda cuda/11.8.0 cudnn/8.8.0.121-11.8

cd ${HOME}/medsam_file/MedSAM/finetune_medsam_v2
conda activate medsam

python cell_segmentation.py