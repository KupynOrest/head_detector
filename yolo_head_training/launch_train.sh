#!/bin/bash
#SBATCH --job-name=vgghead_train
#SBATCH --time=4-00:00:00
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --exclude=gnodek5
#SBATCH --constraint=gmem48G
##SBATCH --mail-user=okupyn@robots.ox.ac.uk
##SBATCH --mail-type=START,END,FAIL,ARRAY_TASKS
pwd; hostname; date
nvidia-smi
source ~/.bashrc
date +"%R activating conda env"
conda activate head_det
export OMP_NUM_THREADS=16
export NCCL_DEBUG=INFO
date +"%R starting script"
cd /users/okupyn/head_detector/yolo_head_training || exit
torchrun --nproc_per_node=4 train.py --config-name=yolo_heads_m_large_orest num_gpus=4 multi_gpu=DDP experiment_suffix='full_loss'


date +"%R slurm job done"