#!/bin/bash
#SBATCH --job-name=vgghead_train
#SBATCH --time=3-00:00:00
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --constraint=gmem48G
##SBATCH --mail-user=okupyn@robots.ox.ac.uk
##SBATCH --mail-type=START,END,FAIL,ARRAY_TASKS
pwd; hostname; date
nvidia-smi
source ~/.bashrc
date +"%R activating conda env"
conda activate head_det
date +"%R starting script"
cd /users/okupyn/head_detector/yolo_head_training || exit
python train.py --config-name=yolo_heads_m \
    dataset_params.train_dataset_params.data_dir=/work/okupyn/VGGHead/small \
    dataset_params.val_dataset_params.data_dir=/work/okupyn/VGGHead/small \
    num_gpus=4 multi_gpu=DDP


date +"%R slurm job done"