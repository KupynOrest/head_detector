#!/bin/bash
#SBATCH --job-name=generate_heads
#SBATCH --time=2-00:00:00
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --exclude=gnodee1,gnodeg3
#SBATCH --constraint=gmem48G
#SBATCH --array=90,92,95,43,49
##SBATCH --mail-user=okupyn@robots.ox.ac.uk
##SBATCH --mail-type=START,END,FAIL,ARRAY_TASKS
pwd; hostname; date
nvidia-smi
source ~/.bashrc
date +"%R activating conda env"
conda activate head_det
date +"%R starting script"
cd /users/okupyn/head_detector/data_generator || exit
export PYTHONPATH=..
python generate_laion.py /scratch/shared/beegfs/shared-datasets/LAION_Metadata /work/okupyn/VGGHead/large

date +"%R slurm job done"
